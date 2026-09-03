// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit test for the physical lens-flare visibility gate in
// shaders/Tonemap.slang (the sun-lensflare PR, issue #335).
//
// WHAT BROKE, AND WHY ONLY A UNIT TEST CAN GUARD IT
//
// The physical flare multiplies the whole frame by a sun-visibility gate
// sampled once at the sun's projected screen UV. The gate is a
// higher-order Reinhard centred on sun_lum = 5. It was first written in
// the DIRECT form
//
//     L = (sun_lum / 5)^16 ;  vis = L / (L + 1)
//
// The physical solar disc is sun_lum ~ 1.59e6 (E / Omega, #280), so
// L ~ 1e88 OVERFLOWS fp32 (max 3.4e38) to +Inf, and vis = Inf/(Inf+1)
// = NaN. vis is uniform across the frame (one sample), so 0*NaN NaNs
// every pixel and saturate(NaN) -> 0 blacks the ENTIRE swapchain the
// instant the sun -- or any highlight above sun_lum ~ 1280, where
// (1280/5)^16 = 256^16 = 2^128 first exceeds fp32 max -- sits at
// sun_uv.
//
// The fix rewrites the gate as the overflow-safe algebraic identity
//
//     L/(L+1) == 1 / (1 + (5/sun_lum)^16)
//
// bit-equivalent in the finite range but unable to NaN: a bright sun
// drives (5/sun_lum)^16 -> 0 (vis -> 1); a dark sample drives it -> +Inf,
// giving vis = 1/(1+Inf) = 0.
//
// The flare is SWAPCHAIN-only: the offline capture reads accum_hdr, which
// never carries flare or bloom, so NO golden image can see this gate
// (issue #337). The offline float64 reproduction the PR's renders used
// could not see it either -- 1e88 is finite in double (max 1.8e308). The
// ONLY automated protection against the black-screen regression class is
// this host test, which reproduces the gate in true fp32 and proves the
// shipped (reciprocal) form stays finite where the old (direct) form
// NaNs.
//
// KEEP IN SYNC with shaders/Tonemap.slang: search that file for the
// comment "keep in sync with tests/pt_tonemap_flare_gate_test.cpp".
// TEST_CASE("shader gate is still the overflow-safe form") below re-reads
// the shader and pins the reciprocal construction so the two cannot drift.
//
// FAST MATH IS DELIBERATELY OFF FOR THIS TEST (see tests/CMakeLists.txt).
// Unlike its sibling shader-mirror tests (pt_math_sphere / _altitude /
// _atmosphere) which build with -ffast-math to match Metal's default
// reassociation, this test needs STRICT IEEE fp32: -ffast-math implies
// -ffinite-math-only, under which the compiler may ASSUME no Inf/NaN can
// occur and fold std::isfinite(...) to a constant true -- which would make
// the red half of the red-then-green pass vacuously and defeat the whole
// test. The overflow -> Inf -> NaN chain this test exists to catch is
// exactly what finite-math-only is licensed to erase, so we hold the host
// mirror to real IEEE semantics, which is also what the real Metal GPU
// produced (the black frame is proof the hardware made the NaN).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The SHIPPED gate. Transcribed operation-for-operation from
// shaders/Tonemap.slang's physical-flare block:
//     float inv = 5.0 / max(sun_lum, 1e-8);
//     inv = inv*inv; inv = inv*inv; inv = inv*inv; inv = inv*inv;
//     float vis = 1.0 / (1.0 + inv);
// Every literal carries the `f` suffix and every local is `float`, so the
// arithmetic is true 32-bit -- Slang's `5.0` in a float context is a
// float, and the point of this test is to reproduce fp32, not double.
float vis_gate_new(float sun_lum) {
    float inv = 5.0f / std::max(sun_lum, 1e-8f);
    inv = inv * inv;   // ^2
    inv = inv * inv;   // ^4
    inv = inv * inv;   // ^8
    inv = inv * inv;   // ^16  -> (5/sun_lum)^16
    return 1.0f / (1.0f + inv);
}

// The OLD, pre-fix gate -- the direct L/(L+1) that overflows to +Inf and
// then NaNs. It exists ONLY so the test can prove the regression the fix
// removes; it is not shipped anywhere.
float vis_gate_old(float sun_lum) {
    float L = sun_lum / 5.0f;
    L = L * L;   // ^2
    L = L * L;   // ^4
    L = L * L;   // ^8
    L = L * L;   // ^16  -> (sun_lum/5)^16
    return L / (L + 1.0f);
}

// A sun_lum sweep spanning 0 .. 1e7, built at RUNTIME (so the compiler
// evaluates the gates for real rather than folding the whole test away),
// and including the three load-bearing points the fix turns on:
//   * 0.0        -- a genuinely dark sample (gate must read ~0)
//   * 1280       -- the first luminance at which the OLD form overflows
//                   ( (1280/5)^16 = 256^16 = 2^128 > fp32 max )
//   * 1.59e6     -- the physical solar disc E/Omega (#280); the config
//                   this PR ships turns on, and where the old form NaNed
//                   the whole frame
std::vector<float> make_sun_lum_sweep() {
    std::vector<float> xs;
    // A dense-ish geometric ramp from a night-sky trickle to well past
    // the solar disc, so nothing between the pinned points slips through.
    for (float s = 1e-4f; s <= 1e7f; s *= 2.0f) xs.push_back(s);
    // The load-bearing points, added explicitly so they are present
    // regardless of where the geometric ramp lands.
    xs.push_back(0.0f);
    xs.push_back(1280.0f);
    xs.push_back(1.59e6f);   // physical solar disc
    xs.push_back(1e7f);      // sweep ceiling
    return xs;
}

}  // namespace

TEST_CASE("shipped gate is finite and in [0,1] across the whole sweep") {
    // The invariant the black-screen bug violated: for EVERY sun_lum the
    // frame can present -- from a dark night sample to the 1.59e6 solar
    // disc -- the gate is a finite number in [0,1]. A single non-finite
    // value here is the whole-frame-black regression, because the gate is
    // multiplied uniformly into every pixel.
    for (float s : make_sun_lum_sweep()) {
        const float vis = vis_gate_new(s);
        CAPTURE(s);
        CAPTURE(vis);
        CHECK(std::isfinite(vis));
        CHECK(vis >= 0.0f);
        CHECK(vis <= 1.0f);
    }
}

TEST_CASE("shipped gate passes the bright sun and blocks the dark sample") {
    // Bright: the physical solar disc must pass essentially unattenuated,
    // or the flare it exists to draw would be gated off.
    CHECK(vis_gate_new(1.59e6f) > 0.99f);
    // The old overflow threshold is comfortably above the gate centre too.
    CHECK(vis_gate_new(1280.0f) > 0.99f);
    // Dark: a night-sky sample (~1e-3) and a literal zero must gate off,
    // so the flare vanishes when the sun is below the horizon / occluded.
    CHECK(vis_gate_new(0.0f)   < 0.01f);
    CHECK(vis_gate_new(1e-3f)  < 0.01f);
    // The gate is centred on sun_lum = 5: the transition midpoint reads
    // ~0.5, which pins that the fix kept the SAME curve, not just a safe
    // number. ( 1 / (1 + (5/5)^16) = 1/2. )
    CHECK(vis_gate_new(5.0f) == doctest::Approx(0.5f));
}

TEST_CASE("RED: the old direct form NaNs -- the regression this locks out") {
    // This is the red half of red-then-green. The pre-fix gate is NOT
    // finite at the physical solar disc: (1.59e6/5)^16 overflows fp32 to
    // +Inf and Inf/(Inf+1) = NaN. If this assertion ever fails (the old
    // form comes back finite), the test harness is compiling under
    // finite-math-only and is no longer proving anything -- see the
    // FAST MATH note at the top of the file.
    CHECK_FALSE(std::isfinite(vis_gate_old(1.59e6f)));
    // ...and it already NaNs at the 1280 overflow threshold, which is why
    // even an ocean sun-glint (not just the disc) blacked the frame.
    CHECK_FALSE(std::isfinite(vis_gate_old(1280.0f)));

    // GREEN: the shipped form is finite at exactly those inputs.
    CHECK(std::isfinite(vis_gate_new(1.59e6f)));
    CHECK(std::isfinite(vis_gate_new(1280.0f)));

    // And where BOTH forms stay finite (below the overflow threshold),
    // they AGREE -- proving the reciprocal is the same curve, not a
    // different gate that merely happens not to overflow.
    for (float s : {0.5f, 1.0f, 5.0f, 20.0f, 100.0f, 1000.0f}) {
        CAPTURE(s);
        REQUIRE(std::isfinite(vis_gate_old(s)));
        CHECK(vis_gate_new(s) == doctest::Approx(vis_gate_old(s)).epsilon(1e-5));
    }
}

TEST_CASE("shader gate is still the overflow-safe form") {
    // The transcription above only means something while it matches
    // shaders/Tonemap.slang. Pin the reciprocal construction the shipped
    // gate depends on, whitespace-stripped so reformatting the shader
    // does not fail this. Mirrors the "shader mirror is still faithful"
    // pattern used by pt_math_sphere / _altitude / _atmosphere.
    std::ifstream f(PT_SHADER_TONEMAP_PATH);
    REQUIRE_MESSAGE(f.good(), "cannot open " PT_SHADER_TONEMAP_PATH);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    std::string tight;
    tight.reserve(src.size());
    for (char ch : src) {
        if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
    }

    // The overflow-safe numerator and the reciprocal itself. If the
    // physical-flare gate ever reverts to the direct L/(L+1) on the sun
    // sample, these disappear and the test fails.
    CHECK(tight.find("floatinv=5.0/max(sun_lum,1e-8)") != std::string::npos);
    CHECK(tight.find("floatvis=1.0/(1.0+inv)") != std::string::npos);
    // The keep-in-sync anchor tying the shader to this file must survive.
    CHECK(tight.find("keepinsyncwithtests/pt_tonemap_flare_gate_test.cpp")
          != std::string::npos);
}

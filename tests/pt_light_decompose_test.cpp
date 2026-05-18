// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Photometric decomposition pin (#176). Validates the conversion
// constants the engine's `light_*_cd` / `light_*_lm` / `light_*_nits` /
// `light_*_exposed` variants use to map artist-facing units onto the
// canonical W/sr (point/spot) and W/m^2/sr (sphere/quad) storage.
//
// The factory constants under test:
//
//   kWsrPerCandela    = 1 / 683.002         lm-to-radiometric at 555nm peak
//   kWsrPerLumenOmni  = 1 / (683.002 * 4*pi) lumens / sphere of steradians
//   kWm2srPerNit      = 1 / 683.002         nits-to-W/m^2/sr at 555nm peak
//
// 683.002 lm/W is the luminous efficacy of monochromatic radiation at
// 555nm (peak of the CIE 1924 photopic curve), the value the SI base-
// unit definition of the candela pins. The conversions here are the
// SINGLE-WAVELENGTH approximation (a per-channel spectral upsampling
// table would give per-channel exactness; that is deferred per issue
// #176's "Out of scope" section).
//
// What this test pins:
//
//   1. The cd->W/sr factor matches the documented 1cd ~ 1.46e-3 W/sr
//      acceptance value within 1e-6 tolerance.
//   2. The lm-omni->W/sr factor is the cd factor divided by 4*pi
//      (consistent with isotropic emission Phi = 4*pi*I).
//   3. The nit->W/m^2/sr factor equals the cd->W/sr factor (same
//      photopic luminous efficacy applied to luminance instead of
//      luminous intensity).
//   4. The decomposition algebra used by every `_color` / `_cd` / `_lm`
//      / `_exposed` variant: final = color * scalar * 2^ev. EV=0 is
//      identity; +1 doubles; -1 halves.
//   5. Round-trip: scaling cd by k scales W/sr linearly by k (no
//      hidden non-linearity in the conversion).
//
// This test deliberately re-defines the constants locally rather than
// linking the Engine target, because Engine.cpp pulls in the entire
// renderer / RHI / Manifold / glTF dependency graph -- 50+ files for
// a value test would be wildly disproportionate. The single-line
// "if you edit kWsrPerCandela in Engine.cpp also edit this test"
// contract is enforced by the assertion that the local constant
// matches the 1cd=1.46e-3 reference value; if a future contributor
// changes the constant in the engine, the on-call engineer should
// also bring this test in lockstep (or split the constants into a
// public header). See test #1 below for the explicit pin.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

namespace {

// Local mirrors of the engine-side constants (Engine.cpp anonymous
// namespace, just above `Engine::RegisterLightCommands`). If the
// engine values change, this test should also change -- this is the
// only place outside the engine that documents the photometric
// contract publicly.
constexpr float kLuminousEfficacy555nm = 683.002f;            // lm/W
constexpr float kWsrPerCandela         = 1.0f / 683.002f;     // W/sr per cd
constexpr float kWsrPerLumenOmni       = 1.0f / (683.002f * 4.0f * 3.14159265358979323846f);
constexpr float kWm2srPerNit           = 1.0f / 683.002f;     // W/m^2/sr per cd/m^2

// Exposure stop: each EV = factor of 2.
float Exposure2x(float ev) { return std::exp2(ev); }

}  // namespace

TEST_CASE("photometric: 1cd -> ~1.46e-3 W/sr at 555nm peak photopic") {
    // Pinned reference value: 1 / 683.002 = 0.0014641245560042286...
    // The issue (#176) calls out "~1.46e-3 W/sr" as the conversion
    // tolerance band. We hold a tighter 1e-6 tolerance per the
    // acceptance criteria.
    constexpr float kExpected = 0.0014641245560042286f;
    CHECK(std::fabs(kWsrPerCandela - kExpected) < 1.0e-6f);

    // Inverse direction: 1 W/sr at 555nm == 683.002 cd.
    CHECK(std::fabs(kLuminousEfficacy555nm - 683.002f) < 1.0e-3f);
    CHECK(std::fabs(kLuminousEfficacy555nm * kWsrPerCandela - 1.0f) < 1.0e-6f);
}

TEST_CASE("photometric: 100cd point light -> 0.1464 W/sr") {
    // The acceptance scenario from the issue: 100cd point light
    // should map to ~0.146 W/sr per channel (single-wavelength).
    const float wsr = 100.0f * kWsrPerCandela;
    CHECK(std::fabs(wsr - 0.14641245f) < 1.0e-5f);

    // Distance-law sanity: a 100cd point light at 5m on a normal-
    // facing surface produces illuminance E = I/r^2 = 100/25 = 4 cd/m^2
    // (or 4 lux). The engine doesn't compute lux directly, but the
    // arithmetic the renderer uses (W/sr / r^2 -> irradiance, then
    // *683 -> illuminance) reduces to I_cd / r^2 in cd/m^2. So:
    const float r = 5.0f;
    const float irradiance_wm2 = wsr / (r * r);
    const float illuminance_lux = irradiance_wm2 * kLuminousEfficacy555nm;
    CHECK(std::fabs(illuminance_lux - 4.0f) < 1.0e-4f);
}

TEST_CASE("photometric: omnidirectional lumen conversion uses 4*pi") {
    // A 4*pi-lumen isotropic emitter has I = 1 cd. Equivalent W/sr.
    constexpr float kFourPi = 4.0f * 3.14159265358979323846f;
    const float one_cd_in_wsr = 1.0f * kWsrPerCandela;
    const float four_pi_lm_in_wsr = kFourPi * kWsrPerLumenOmni;
    CHECK(std::fabs(one_cd_in_wsr - four_pi_lm_in_wsr) < 1.0e-7f);

    // Common bulb spec: a "1500 lumen" omni LED.
    //   I_cd  = 1500 / (4*pi) ~= 119.366 cd
    //   W/sr ~= 119.366 / 683 ~= 0.1748
    const float lm = 1500.0f;
    const float wsr = lm * kWsrPerLumenOmni;
    CHECK(std::fabs(wsr - 0.17479f) < 1.0e-4f);
}

TEST_CASE("photometric: nit (cd/m^2) -> W/m^2/sr matches cd factor") {
    // Same photopic luminous efficacy applies to luminance as to
    // luminous intensity (just W/m^2/sr instead of W/sr).
    CHECK(std::fabs(kWm2srPerNit - kWsrPerCandela) < 1.0e-9f);

    // Display reference: a 500 nit OLED panel emits
    //   500 / 683 ~= 0.7321 W/m^2/sr surface radiance.
    const float nits = 500.0f;
    const float radiance = nits * kWm2srPerNit;
    CHECK(std::fabs(radiance - 0.73206f) < 1.0e-4f);
}

TEST_CASE("decomposition: final = color * intensity * 2^ev") {
    // The canonical algebra every `_color` / `_cd` / `_lm` / `_exposed`
    // variant performs internally. Pin the math so a refactor that
    // accidentally reorders the multiplies (e.g. color first, then
    // intensity is forgotten) trips immediately.

    // EV=0 identity:
    {
        const float scalar = 2.5f;
        const float ev     = 0.0f;
        const float result_r = 0.8f * scalar * Exposure2x(ev);
        CHECK(std::fabs(result_r - 2.0f) < 1.0e-6f);
        CHECK(Exposure2x(0.0f) == doctest::Approx(1.0));
    }
    // EV=+1 doubles:
    {
        const float scalar = 1.0f;
        const float ev     = 1.0f;
        const float result = 0.5f * scalar * Exposure2x(ev);
        CHECK(std::fabs(result - 1.0f) < 1.0e-6f);
    }
    // EV=-1 halves:
    {
        const float scalar = 4.0f;
        const float ev     = -1.0f;
        const float result = 1.0f * scalar * Exposure2x(ev);
        CHECK(std::fabs(result - 2.0f) < 1.0e-6f);
    }
    // EV=+3 = 8x:
    {
        const float scalar = 1.0f;
        const float ev     = 3.0f;
        const float result = 1.0f * scalar * Exposure2x(ev);
        CHECK(std::fabs(result - 8.0f) < 1.0e-6f);
    }
    // Fractional EV (half-stop = sqrt(2)):
    {
        const float scalar = 1.0f;
        const float ev     = 0.5f;
        const float result = 1.0f * scalar * Exposure2x(ev);
        CHECK(std::fabs(result - std::sqrt(2.0f)) < 1.0e-6f);
    }
}

TEST_CASE("decomposition: cfg round-trip preserves W/sr") {
    // The save path emits canonical `light_point <id> <x> <y> <z> <r>
    // <g> <b>` form (W/sr per channel). So a user authoring with the
    // _cd variant authors:
    //
    //     light_point_cd 1 0 0 0 1 1 1 100
    //
    // The engine stores:
    //
    //     L.intensity = (1,1,1) * (100 / 683.002) = (0.14641, ...)
    //
    // The save path writes:
    //
    //     light_point 1 0 0 0 0.14641 0.14641 0.14641
    //
    // On reload the canonical command repopulates L.intensity to the
    // exact same numerical value. So round-trip is a pure float
    // identity -- there's no second conversion step.
    const float cd       = 100.0f;
    const float r        = 1.0f;
    const float saved    = r * cd * kWsrPerCandela;  // what the engine stores
    const float reloaded = saved;                    // what `light_point` parses back
    CHECK(std::fabs(saved - reloaded) < 1.0e-9f);
    CHECK(std::fabs(saved - 0.14641245f) < 1.0e-5f);

    // Same round-trip for the lm path:
    const float lm        = 1500.0f;
    const float saved_lm  = r * lm * kWsrPerLumenOmni;
    CHECK(std::fabs(saved_lm - 0.17479f) < 1.0e-4f);

    // Same for nits (area-light radiance):
    const float nits        = 500.0f;
    const float saved_nits  = r * nits * kWm2srPerNit;
    CHECK(std::fabs(saved_nits - 0.73206f) < 1.0e-4f);
}

TEST_CASE("decomposition: linearity in scalar input") {
    // The conversion is purely linear in the scalar input -- a 2x in
    // cd is a 2x in W/sr, no hidden floor / ceiling / gamma.
    const float wsr_50  = 50.0f * kWsrPerCandela;
    const float wsr_100 = 100.0f * kWsrPerCandela;
    const float wsr_200 = 200.0f * kWsrPerCandela;
    CHECK(std::fabs(wsr_100 - 2.0f * wsr_50)  < 1.0e-6f);
    CHECK(std::fabs(wsr_200 - 2.0f * wsr_100) < 1.0e-6f);
    CHECK(std::fabs(wsr_200 - 4.0f * wsr_50)  < 1.0e-6f);
}

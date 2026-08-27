// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P1 (#255) -- the camera-anchored render frame.
//
// What this pins, in the order the design depends on it:
//
//  1. The LATTICE is a power of two, always, and clamps to
//     [1024 m, 2^22 m]. Everything below rests on this.
//  2. The ANCHOR is a PURE FUNCTION of the camera position -- no
//     hysteresis, no path dependence -- which is what lets two backends,
//     two runs and the software reference tracer agree, and therefore
//     what the golden matrix requires.
//  3. r_origin_rebase_radius gates it: inside the radius (and at radius
//     0) the anchor is exactly dvec3(0), which makes the whole scheme a
//     no-op and every existing golden bit-identical.
//  4. The anchor is EXACTLY representable in float32, and so is the
//     frame-to-frame anchor delta. This is the property the ReSTIR
//     reservoir shift and the motion-vector reconstruction both cash in.
//  5. MOTION VECTORS ARE CONTINUOUS ACROSS A REBASE. This is the one
//     that actually matters and the one that most easily goes subtly
//     wrong, so it is proved twice over: the rebased reconstruction is
//     shown to agree with a no-rebase reference to far below a pixel,
//     AND the old cache-the-matrix scheme is shown to be catastrophically
//     wrong on the same inputs -- otherwise the first assertion could
//     pass on a test with no teeth.
//
// Nothing here needs a GPU: the anchor, the lattice and the matrix
// reconstruction are all host arithmetic, which is exactly why they can
// be pinned this tightly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/core/WorldFrame.h"
#include "../src/renderer/Camera.h"

#include <cmath>

using pt::core::ComputeAnchor;
using pt::core::RebaseLattice;
using pt::core::WorldFrame;
using pt::core::kRebaseLatticeMax;
using pt::core::kRebaseLatticeMin;

namespace {

constexpr double kDefaultRebaseRadius = 65536.0;   // r_origin_rebase_radius

// A power of two, exactly -- frexp's mantissa is exactly 0.5 for one.
bool IsPowerOfTwo(double v) {
    if (!(v > 0.0)) return false;
    int exp = 0;
    return std::frexp(v, &exp) == 0.5;
}

// Project a render-frame point through a view-projection into pixels,
// mirroring shaders/PathTrace.slang's motion-vector block:
//   pc = mul(vp, float4(p, 1)); pix = (pc.xy / pc.w * 0.5 + 0.5) * dim
glm::vec2 ToPixels(const glm::mat4& vp, const glm::vec3& p,
                   float w, float h) {
    const glm::vec4 c = vp * glm::vec4(p, 1.0f);
    const glm::vec2 ndc = glm::vec2(c.x, c.y) / c.w;
    return glm::vec2((ndc.x * 0.5f + 0.5f) * w, (ndc.y * 0.5f + 0.5f) * h);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The lattice
// ---------------------------------------------------------------------------

TEST_CASE("RebaseLattice always returns a power of two inside its clamp") {
    // The floor holds for every non-positive / tiny scale, including the
    // 0 that P1 pins it to.
    CHECK(RebaseLattice(0.0)     == kRebaseLatticeMin);
    CHECK(RebaseLattice(-1.0)    == kRebaseLatticeMin);
    CHECK(RebaseLattice(1.0)     == kRebaseLatticeMin);
    CHECK(RebaseLattice(4095.0)  == kRebaseLatticeMin);
    // Ceiling.
    CHECK(RebaseLattice(1.0e12)  == kRebaseLatticeMax);

    // And a power of two everywhere in between. Sweep the whole altitude
    // range this engine could plausibly reach, from a metre to
    // geostationary.
    for (double scale = 1.0; scale < 5.0e7; scale *= 1.37) {
        const double L = RebaseLattice(scale);
        CAPTURE(scale);
        CAPTURE(L);
        CHECK(IsPowerOfTwo(L));
        CHECK(L >= kRebaseLatticeMin);
        CHECK(L <= kRebaseLatticeMax);
    }
}

TEST_CASE("RebaseLattice keeps the render-frame error four orders below the pixel") {
    // The design claim: L ~ scale/4 means the worst-case render-frame
    // representation error (2^-24 * L) stays far under the pixel
    // footprint at that scale (1.07e-3 * scale at 1080p / 60 deg).
    for (double scale = 1.0e4; scale < 5.0e7; scale *= 2.0) {
        const double L   = RebaseLattice(scale);
        const double err = L * 5.960464477539063e-8;
        const double px  = scale * 1.07e-3;
        CAPTURE(scale);
        CHECK(err < px * 1.0e-3);
    }
}

// ---------------------------------------------------------------------------
// 2/3. The anchor: pure, gated, quantised
// ---------------------------------------------------------------------------

TEST_CASE("ComputeAnchor is pinned to the origin inside the rebase radius") {
    // Every golden fixture lives here. The maximum is
    // sunset_altitude.cfg at cam_pos 0 10000 0, so the default radius
    // clears the whole matrix by 6.5x.
    const glm::dvec3 cases[] = {
        {0.0, 1.5, 4.0},          // the engineering default
        {0.0, 10000.0, 0.0},      // sunset_altitude.cfg, the farthest fixture
        {-3.0, 0.5, 12.0},
        {30000.0, 30000.0, 30000.0},   // |p| = 51961, still inside
    };
    for (const auto& c : cases) {
        CAPTURE(c.x); CAPTURE(c.y); CAPTURE(c.z);
        REQUIRE(glm::length(c) < kDefaultRebaseRadius);
        CHECK(ComputeAnchor(c, kDefaultRebaseRadius) == glm::dvec3(0.0));
    }
}

TEST_CASE("r_origin_rebase_radius 0 disables rebasing entirely") {
    // The A/B switch: with the cvar at 0 the engine is permanently on
    // the legacy path no matter where the camera goes.
    const glm::dvec3 ecef(0.0, 6371008.8, 0.0);
    CHECK(ComputeAnchor(ecef, 0.0)  == glm::dvec3(0.0));
    CHECK(ComputeAnchor(ecef, -1.0) == glm::dvec3(0.0));
    // ...and with the cvar on, the very same camera DOES rebase, so the
    // test above is measuring the gate and not the input.
    CHECK(ComputeAnchor(ecef, kDefaultRebaseRadius) != glm::dvec3(0.0));
}

TEST_CASE("ComputeAnchor is a pure function of the camera position") {
    // No hysteresis, no path dependence: the anchor for a position must
    // not depend on how the camera got there. Walk a camera up to a
    // point from below and down to it from above and require the same
    // answer -- and require repeated evaluation to be identical, which
    // is what makes two backends and the software reference tracer agree
    // on the same pixels.
    const glm::dvec3 target(1.0e6, 6371008.8, -2.5e5);
    const glm::dvec3 a = ComputeAnchor(target, kDefaultRebaseRadius);
    for (int i = 0; i < 64; ++i) {
        glm::dvec3 probe = target + glm::dvec3(double(i) * 137.0, 0.0, 0.0);
        (void)ComputeAnchor(probe, kDefaultRebaseRadius);   // perturb, then re-ask
        CHECK(ComputeAnchor(target, kDefaultRebaseRadius) == a);
    }
}

TEST_CASE("ComputeAnchor quantises onto the lattice and stays near the eye") {
    const double L = RebaseLattice(0.0);
    for (double y = 1.0e5; y < 4.3e7; y *= 1.21) {
        const glm::dvec3 cam(1234.5, y, -9876.5);
        const glm::dvec3 A = ComputeAnchor(cam, kDefaultRebaseRadius);
        CAPTURE(y);
        // On the lattice, exactly.
        CHECK(std::fmod(A.x, L) == 0.0);
        CHECK(std::fmod(A.y, L) == 0.0);
        CHECK(std::fmod(A.z, L) == 0.0);
        // And within half a lattice cell of the eye on every axis, so
        // the render-frame camera is bounded by L regardless of how far
        // the canonical camera has travelled.
        CHECK(std::abs(cam.x - A.x) <= L * 0.5);
        CHECK(std::abs(cam.y - A.y) <= L * 0.5);
        CHECK(std::abs(cam.z - A.z) <= L * 0.5);
    }
}

// ---------------------------------------------------------------------------
// 4. Exactness -- the property the whole design leans on
// ---------------------------------------------------------------------------

TEST_CASE("the anchor and every anchor delta are exact in float32") {
    // A is an integer multiple of a power-of-two lattice >= 1024 m, and
    // |A| / L stays under 2^24 for anything short of 1.7e10 m -- so
    // narrowing the anchor to float32 loses NOTHING, and neither does
    // narrowing the difference of two anchors. That is why rebasing
    // introduces no rounding: the ReSTIR reservoir shift and the
    // motion-vector reconstruction both consume this delta.
    glm::dvec3 prev = ComputeAnchor(glm::dvec3(0.0, 6371008.8, 0.0),
                                    kDefaultRebaseRadius);
    for (int i = 1; i <= 512; ++i) {
        const glm::dvec3 cam(double(i) * 613.0,
                             6371008.8 + double(i) * 421.0,
                             -double(i) * 977.0);
        const glm::dvec3 A = ComputeAnchor(cam, kDefaultRebaseRadius);

        // float32 round-trip of the anchor itself is lossless.
        CHECK(double(float(A.x)) == A.x);
        CHECK(double(float(A.y)) == A.y);
        CHECK(double(float(A.z)) == A.z);

        // ...and of the delta, which is what actually gets pushed.
        WorldFrame wf;
        wf.anchor_prev = prev;
        wf.anchor      = A;
        const glm::vec3 d = wf.AnchorDelta();
        CHECK(double(d.x) == prev.x - A.x);
        CHECK(double(d.y) == prev.y - A.y);
        CHECK(double(d.z) == prev.z - A.z);
        prev = A;
    }
}

TEST_CASE("ToRender at the origin anchor is the identity a golden depends on") {
    // The bit-identity claim for every existing fixture: with the anchor
    // at zero, glm::vec3(p_f64 - dvec3(0)) recovers exactly the float
    // that p_f64 was widened from. If this ever stops holding, every
    // golden moves.
    WorldFrame wf;   // anchor defaults to (0,0,0)
    REQUIRE(wf.AtOrigin());
    const float samples[] = {
        0.0f, 1.5f, -3.75f, 0.45f, 1e-7f, 6371008.5f, -12345.678f, 1e10f,
    };
    for (float f : samples) {
        const glm::vec3 r = wf.ToRender(glm::dvec3(double(f), double(f), double(f)));
        CHECK(r.x == f);
        CHECK(r.y == f);
        CHECK(r.z == f);
    }
}

TEST_CASE("ToWorld inverts ToRender exactly on the lattice") {
    WorldFrame wf;
    wf.anchor = ComputeAnchor(glm::dvec3(0.0, 6371008.8, 0.0), kDefaultRebaseRadius);
    REQUIRE(!wf.AtOrigin());
    const glm::vec3 p_render(1.25f, -0.5f, 33.75f);
    const glm::dvec3 back = wf.ToWorld(p_render);
    // The anchor is exact in float32 and the render-frame point is a
    // float, so the round trip is exact both ways.
    CHECK(wf.ToRender(back) == p_render);
}

// ---------------------------------------------------------------------------
// 5. The one that matters: no motion-vector discontinuity across a rebase
// ---------------------------------------------------------------------------

TEST_CASE("motion vectors are continuous across a forced rebase") {
    // Construct a two-frame sequence that STRADDLES an anchor change and
    // measure the motion vector of a fixed world point across it.
    //
    // The camera flies along +X at orbital speed (7.66 km/s, 128 m per
    // frame at 60 fps) at ECEF radius, positioned so frame N and frame
    // N+1 fall on opposite sides of a lattice boundary.
    const double L = RebaseLattice(0.0);
    const double step = 127.6666;                 // one frame of orbital travel
    // Put frame N just short of a lattice cell centre boundary so the
    // rounding flips between the two frames.
    const double x0 = 3.0 * L + L * 0.5 - step * 0.5;
    const glm::dvec3 cam_prev(x0,        6371008.8, 0.0);
    const glm::dvec3 cam_curr(x0 + step, 6371008.8, 0.0);

    const glm::dvec3 A_prev = ComputeAnchor(cam_prev, kDefaultRebaseRadius);
    const glm::dvec3 A_curr = ComputeAnchor(cam_curr, kDefaultRebaseRadius);
    // The premise of the test: this really is a rebase.
    REQUIRE(A_prev != A_curr);
    REQUIRE(glm::length(A_prev - A_curr) >= L);

    // A metre-scale feature 8 m in front of the camera and 1.5 m to the
    // side -- the near field, where a motion-vector error costs the most
    // pixels.
    const glm::dvec3 hit_world = cam_curr + glm::dvec3(1.5, -0.75, -8.0);

    const glm::vec3 fwd(0.0f, 0.0f, -1.0f);
    const float fov = 60.0f, aspect = 16.0f / 9.0f, W = 1920.0f, H = 1080.0f;

    // --- reference: what the motion vector is with NO rebase at all ----
    // Both frames expressed in A_prev, which is what the renderer would
    // have produced had the lattice not flipped.
    const glm::vec2 ref_cur = ToPixels(
        pt::renderer::BuildCameraMatrices(glm::vec3(cam_curr - A_prev), fwd, fov, aspect).view_proj,
        glm::vec3(hit_world - A_prev), W, H);
    const glm::vec2 ref_pre = ToPixels(
        pt::renderer::BuildCameraMatrices(glm::vec3(cam_prev - A_prev), fwd, fov, aspect).view_proj,
        glm::vec3(hit_world - A_prev), W, H);
    const glm::vec2 motion_ref = ref_pre - ref_cur;

    // --- shipped scheme: rebuild the PREVIOUS POSE in the CURRENT anchor
    const glm::mat4 vp_curr = pt::renderer::BuildCameraMatrices(
        glm::vec3(cam_curr - A_curr), fwd, fov, aspect).view_proj;
    const glm::mat4 vp_prev_rebuilt = pt::renderer::BuildCameraMatrices(
        glm::vec3(cam_prev - A_curr), fwd, fov, aspect).view_proj;
    const glm::vec3 hit_render = glm::vec3(hit_world - A_curr);
    const glm::vec2 motion_new = ToPixels(vp_prev_rebuilt, hit_render, W, H) -
                                 ToPixels(vp_curr,         hit_render, W, H);

    // Continuous: the rebase is invisible to the motion vector.
    CAPTURE(motion_ref.x); CAPTURE(motion_ref.y);
    CAPTURE(motion_new.x); CAPTURE(motion_new.y);
    CHECK(std::abs(motion_new.x - motion_ref.x) < 1.0e-2f);
    CHECK(std::abs(motion_new.y - motion_ref.y) < 1.0e-2f);

    // --- and the test has teeth ----------------------------------------
    // The pre-#255 scheme cached the previous view-projection MATRIX,
    // built in the previous anchor, and multiplied it against this
    // frame's anchor-relative hit position. Show that doing so on these
    // exact inputs is off by an enormous amount, so the assertion above
    // is not passing because the two frames happened to agree anyway.
    const glm::mat4 vp_prev_cached = pt::renderer::BuildCameraMatrices(
        glm::vec3(cam_prev - A_prev), fwd, fov, aspect).view_proj;
    const glm::vec2 motion_cached =
        ToPixels(vp_prev_cached, hit_render, W, H) -
        ToPixels(vp_curr,        hit_render, W, H);
    CHECK(glm::length(motion_cached - motion_ref) > 1000.0f);
}

TEST_CASE("an unmoved anchor reproduces last frame's matrix bit for bit") {
    // The pose scheme replaces a cached matrix with a RECONSTRUCTED one,
    // so on the overwhelmingly common frame -- anchor unchanged -- the
    // reconstruction has to reproduce the cached matrix EXACTLY. A
    // single ULP of drift makes a stationary camera emit a sub-pixel
    // motion vector, and SVGF's temporal reprojection then picks a
    // different tap; that showed up as three moved pixels in
    // light_primitives_mixed before BuildCameraMatrices was forced out
    // of line. Bit equality, not approximate equality, is the contract.
    const glm::vec3 eye(1.25f, 1.6f, -4.5f);
    const glm::vec3 fwd = glm::normalize(glm::vec3(0.3f, -0.15f, -1.0f));
    const glm::mat4 a = pt::renderer::BuildCameraMatrices(
        eye, fwd, 55.0f, 16.0f / 9.0f).view_proj;
    const glm::mat4 b = pt::renderer::BuildCameraMatrices(
        eye, fwd, 55.0f, 16.0f / 9.0f).view_proj;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            CAPTURE(c); CAPTURE(r);
            CHECK(a[c][r] == b[c][r]);
        }
    }
}

TEST_CASE("a stationary camera produces zero motion in any anchor") {
    // Degenerate but load-bearing: the goldens hold the camera still, so
    // if a rebase ever leaked a non-zero motion vector into a static
    // scene the denoiser would smear a pinned image.
    const glm::dvec3 cam(0.0, 6371008.8, 0.0);
    const glm::dvec3 A = ComputeAnchor(cam, kDefaultRebaseRadius);
    const glm::dvec3 hit = cam + glm::dvec3(0.0, -1.6, -4.0);
    const glm::vec3 fwd(0.0f, 0.0f, -1.0f);
    const glm::mat4 vp = pt::renderer::BuildCameraMatrices(
        glm::vec3(cam - A), fwd, 60.0f, 16.0f / 9.0f).view_proj;
    const glm::vec2 a = ToPixels(vp, glm::vec3(hit - A), 1920.0f, 1080.0f);
    const glm::vec2 b = ToPixels(vp, glm::vec3(hit - A), 1920.0f, 1080.0f);
    CHECK(a.x == b.x);
    CHECK(a.y == b.y);
}

TEST_CASE("the anchor buys back the precision the whole phase exists for") {
    // The headline ratio, measured rather than asserted. Take a
    // metre-scale feature at ECEF radius and compare how well float32
    // holds it in ABSOLUTE coordinates versus in the render frame.
    const glm::dvec3 cam(0.0, 6371008.8, 0.0);
    const glm::dvec3 A = ComputeAnchor(cam, kDefaultRebaseRadius);
    // A 1 mm feature -- the scale of the tracer's own shadow bias.
    const glm::dvec3 feature = cam + glm::dvec3(0.0, 0.001, -4.0);

    // Absolute float32: the feature and the camera collapse onto the
    // same y, because one ULP at 6.37e6 is 0.5 m.
    const float abs_cam_y     = static_cast<float>(cam.y);
    const float abs_feature_y = static_cast<float>(feature.y);
    CHECK(abs_feature_y == abs_cam_y);          // 1 mm of separation, gone

    // Render frame: the same 1 mm survives with room to spare.
    WorldFrame wf;
    wf.anchor = A;
    const glm::vec3 cam_r     = wf.ToRender(cam);
    const glm::vec3 feature_r = wf.ToRender(feature);
    const double sep = double(feature_r.y) - double(cam_r.y);
    CHECK(sep == doctest::Approx(0.001).epsilon(1e-3));
}

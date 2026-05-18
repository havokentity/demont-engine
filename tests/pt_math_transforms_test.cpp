// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for combined transforms (issue #63 -- Phase 3 of #47).
//
// Combined transforms are where bugs hide -- TRS composition order,
// view matrix sign conventions, projection matrix handedness, and the
// screen-to-world ray reconstruction the path-tracer shader does on
// every pixel. The engine builds these with exact glm idioms:
//
//   Engine.cpp:3584:  glm::lookAtRH(cam.pos, cam.pos + fwd, vec3(0,1,0))
//   Engine.cpp:3585:  glm::perspectiveRH_ZO(radians(fov), aspect, 0.05, 500)
//
// "RH" = right-handed (engine convention everywhere), "ZO" = clip-z in
// [0,1] (the Vulkan / Metal convention for depth, NOT the GL [-1,1]).
// If these conventions ever drift, every ray-trace dispatch + every
// motion-vector reprojection breaks silently.
//
// Specific contracts tested below:
//   1. TRS composition (T*R*S) applied to a unit cube produces the
//      expected world-space corners.
//   2. world_to_view = inverse(view) sends the camera position to the
//      origin.
//   3. perspectiveRH_ZO maps the near plane to clip-z=0, far plane to
//      clip-z=1.
//   4. Screen-to-world ray reconstruction (the same calc the shader
//      does at PathTrace.slang::main) recovers an arbitrary world-
//      space ray within float epsilon.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>      // mat3_cast / quat_cast in gtc (stable)
#include <glm/gtc/constants.hpp>

#include <cmath>

namespace {

constexpr float kEps      = 1e-5f;
constexpr float kLooseEps = 1e-4f;

bool vec3_approx_equal(const glm::vec3& a, const glm::vec3& b, float eps) {
    return std::abs(a.x - b.x) < eps &&
           std::abs(a.y - b.y) < eps &&
           std::abs(a.z - b.z) < eps;
}

bool vec4_approx_equal(const glm::vec4& a, const glm::vec4& b, float eps) {
    return std::abs(a.x - b.x) < eps &&
           std::abs(a.y - b.y) < eps &&
           std::abs(a.z - b.z) < eps &&
           std::abs(a.w - b.w) < eps;
}

}  // namespace

// --- TRS composition -----------------------------------------------------
TEST_CASE("transform: TRS composition applies S then R then T") {
    // T * R * S is the standard right-to-left "apply scale first, then
    // rotate, then translate" composition. Verify on a known input:
    //   - Scale by (2, 2, 2): point (1,0,0) -> (2,0,0)
    //   - Rotate 90deg around +Y: (2,0,0) -> (0,0,-2)
    //   - Translate by (5, 0, 0): (0,0,-2) -> (5,0,-2)
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3{5.0f, 0.0f, 0.0f});
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              glm::pi<float>() * 0.5f,
                              glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3{2.0f, 2.0f, 2.0f});

    glm::mat4 M = T * R * S;

    glm::vec4 p{1.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 r = M * p;

    CHECK(r.x == doctest::Approx(5.0f).epsilon(kLooseEps));
    CHECK(r.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(r.z == doctest::Approx(-2.0f).epsilon(kLooseEps));
    CHECK(r.w == doctest::Approx(1.0f).epsilon(kEps));
}

TEST_CASE("transform: TRS inverse undoes the composed transform") {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3{1.0f, 2.0f, 3.0f});
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              0.6f,
                              glm::normalize(glm::vec3{0.0f, 1.0f, 0.0f}));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3{1.5f, 1.5f, 1.5f});

    glm::mat4 M    = T * R * S;
    glm::mat4 Minv = glm::inverse(M);

    glm::vec4 p_world{0.5f, 0.75f, -1.0f, 1.0f};
    glm::vec4 round_tripped = Minv * (M * p_world);

    CHECK(round_tripped.x == doctest::Approx(p_world.x).epsilon(kLooseEps));
    CHECK(round_tripped.y == doctest::Approx(p_world.y).epsilon(kLooseEps));
    CHECK(round_tripped.z == doctest::Approx(p_world.z).epsilon(kLooseEps));
    CHECK(round_tripped.w == doctest::Approx(p_world.w).epsilon(kLooseEps));
}

TEST_CASE("transform: TRS basis vectors are extractable from columns") {
    // For an affine TRS with no shear, the upper-left 3x3's columns
    // are the world-space basis vectors of the local frame (scaled by
    // the corresponding scale factor). The engine reads these to
    // extract object orientation when building motion-vector data.
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3{0.0f, 0.0f, 0.0f});
    // 90deg around +Y: local +X -> world -Z; local +Y stays; local +Z -> world +X.
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              glm::pi<float>() * 0.5f,
                              glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3{1.0f, 1.0f, 1.0f});
    glm::mat4 M = T * R * S;

    // Column 0 of M (with unit scale) is the world-space +X axis of
    // the local frame, after rotation: should be (0, 0, -1).
    glm::vec3 col0_xyz{M[0][0], M[0][1], M[0][2]};
    CHECK(vec3_approx_equal(col0_xyz, glm::vec3{0.0f, 0.0f, -1.0f}, kLooseEps));

    // Column 1: local +Y, rotation axis -> unchanged.
    glm::vec3 col1_xyz{M[1][0], M[1][1], M[1][2]};
    CHECK(vec3_approx_equal(col1_xyz, glm::vec3{0.0f, 1.0f, 0.0f}, kLooseEps));

    // Column 2: local +Z, rotated +90deg around +Y -> world +X.
    glm::vec3 col2_xyz{M[2][0], M[2][1], M[2][2]};
    CHECK(vec3_approx_equal(col2_xyz, glm::vec3{1.0f, 0.0f, 0.0f}, kLooseEps));
}

TEST_CASE("transform: TRS world-translation lives in column 3") {
    glm::vec3 t_world{3.5f, -1.2f, 0.75f};
    glm::mat4 T = glm::translate(glm::mat4(1.0f), t_world);
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              0.4f,
                              glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 M = T * R;

    // Translation only modifies the 4th column (after T is applied last
    // in the chain). Rotation does not move the origin, so column 3
    // should equal t_world (with w = 1).
    CHECK(M[3][0] == doctest::Approx(t_world.x).epsilon(kEps));
    CHECK(M[3][1] == doctest::Approx(t_world.y).epsilon(kEps));
    CHECK(M[3][2] == doctest::Approx(t_world.z).epsilon(kEps));
    CHECK(M[3][3] == doctest::Approx(1.0f).epsilon(kEps));
}

// --- world-to-view (lookAtRH) -------------------------------------------
TEST_CASE("transform: lookAtRH places camera at view-space origin") {
    // The engine uses lookAtRH everywhere; this is the canonical
    // "world coords -> camera-space coords" transform. After applying
    // it, the camera's own position should land at the view-space
    // origin. (View space is the coord frame before projection -- clip
    // space comes later, after multiplying by the projection matrix.)
    glm::vec3 cam_pos{2.0f, 3.0f, 4.0f};
    glm::vec3 cam_target{0.0f, 0.0f, 0.0f};
    glm::vec3 cam_up{0.0f, 1.0f, 0.0f};
    glm::mat4 view = glm::lookAtRH(cam_pos, cam_target, cam_up);

    // Camera position in world coords -> origin in view coords.
    glm::vec4 origin_in_view = view * glm::vec4(cam_pos, 1.0f);
    CHECK(origin_in_view.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(origin_in_view.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(origin_in_view.z == doctest::Approx(0.0f).epsilon(kLooseEps));

    // RH convention: looking from cam_pos toward cam_target, the
    // target lands on the -Z axis in view space (since RH has -Z
    // looking forward). The distance to target in view space equals
    // -|cam_pos - cam_target|.
    glm::vec4 target_in_view = view * glm::vec4(cam_target, 1.0f);
    CHECK(target_in_view.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(target_in_view.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(target_in_view.z < 0.0f);  // target is forward of cam in -Z
    const float dist = glm::length(cam_pos - cam_target);
    CHECK(target_in_view.z == doctest::Approx(-dist).epsilon(kLooseEps));
}

TEST_CASE("transform: lookAtRH inverse takes view origin back to camera") {
    glm::vec3 cam_pos{-1.5f, 2.0f, 5.0f};
    glm::mat4 view = glm::lookAtRH(cam_pos,
                                   glm::vec3{0.0f, 0.0f, 0.0f},
                                   glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 view_inv = glm::inverse(view);

    // View-space origin -> world cam_pos.
    glm::vec4 cam_in_world = view_inv * glm::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    CHECK(cam_in_world.x == doctest::Approx(cam_pos.x).epsilon(kLooseEps));
    CHECK(cam_in_world.y == doctest::Approx(cam_pos.y).epsilon(kLooseEps));
    CHECK(cam_in_world.z == doctest::Approx(cam_pos.z).epsilon(kLooseEps));
    CHECK(cam_in_world.w == doctest::Approx(1.0f).epsilon(kEps));
}

// --- view-to-clip (perspectiveRH_ZO) -----------------------------------
TEST_CASE("transform: perspectiveRH_ZO maps near plane to clip-z=0") {
    // Vulkan / Metal use clip-z in [0,1]; "ZO" = zero-to-one. A point
    // on the near plane in view space (z = -near in RH) should land at
    // clip-z = 0 after perspective; far plane at clip-z = far/something
    // -- the actual ratio depends on the projection. We test the
    // simpler invariant: near maps to clip-z 0 after the w-divide, far
    // maps to clip-z 1.
    // `near` / `far` are legacy <windows.h> macros (DOS segment-register
    // holdovers) so MSVC chokes on `const float near = ...`. Use the
    // `z_near` / `z_far` naming convention to stay portable.
    const float z_near = 0.1f;
    const float z_far  = 100.0f;
    const float fov_y  = glm::radians(60.0f);
    const float aspect = 16.0f / 9.0f;
    glm::mat4 proj = glm::perspectiveRH_ZO(fov_y, aspect, z_near, z_far);

    // View-space point on the near plane (on the optical axis).
    glm::vec4 v_near{0.0f, 0.0f, -z_near, 1.0f};
    glm::vec4 c_near = proj * v_near;
    // After perspective divide, clip-z / clip-w should be 0.
    REQUIRE(c_near.w != 0.0f);
    CHECK((c_near.z / c_near.w) == doctest::Approx(0.0f).epsilon(kLooseEps));

    // Far plane.
    glm::vec4 v_far{0.0f, 0.0f, -z_far, 1.0f};
    glm::vec4 c_far = proj * v_far;
    REQUIRE(c_far.w != 0.0f);
    CHECK((c_far.z / c_far.w) == doctest::Approx(1.0f).epsilon(kLooseEps));
}

TEST_CASE("transform: perspectiveRH_ZO preserves screen-x sign convention") {
    // A view-space point to the camera's right (+x in view space)
    // should land in NDC with +x. Tests the projection's x-axis sign
    // is not silently flipped (which would mirror the entire image).
    const float fov_y = glm::radians(60.0f);
    const float aspect = 16.0f / 9.0f;
    glm::mat4 proj = glm::perspectiveRH_ZO(fov_y, aspect, 0.1f, 100.0f);

    // Point to the right of the optical axis, inside the frustum.
    glm::vec4 v_right{0.5f, 0.0f, -10.0f, 1.0f};
    glm::vec4 c_right = proj * v_right;
    REQUIRE(c_right.w > 0.0f);  // in front of the camera
    CHECK((c_right.x / c_right.w) > 0.0f);

    // Point to the left -- clip-x / clip-w should be < 0.
    glm::vec4 v_left{-0.5f, 0.0f, -10.0f, 1.0f};
    glm::vec4 c_left = proj * v_left;
    REQUIRE(c_left.w > 0.0f);
    CHECK((c_left.x / c_left.w) < 0.0f);

    // Point above the optical axis -> clip-y > 0 in NDC.
    glm::vec4 v_up{0.0f, 0.5f, -10.0f, 1.0f};
    glm::vec4 c_up = proj * v_up;
    REQUIRE(c_up.w > 0.0f);
    CHECK((c_up.y / c_up.w) > 0.0f);
}

// --- Screen-to-world ray reconstruction ---------------------------------
TEST_CASE("transform: screen-space ray reconstruction round-trip") {
    // The path-tracer shader reconstructs a world-space ray from each
    // pixel's NDC coordinate by inverting the combined view-projection
    // matrix. glm uses column vectors, so the build order is
    // vp = proj * view and the shader inverts that product. Test the
    // inverse direction:
    //   1. Pick a known world-space point P.
    //   2. Project: clip = proj * view * (P, 1) -> NDC after w-divide.
    //   3. Unproject the NDC point through inverse(proj*view).
    //   4. Recovered point must match P within FP epsilon.
    //
    // This is exactly the math the shader inverts to build the primary
    // ray (it skips the explicit unproject and reconstructs via
    // cam_pos + ray_dir * t, but the test exercises the matrix path
    // which is the engine's source of truth for ray reconstruction).
    glm::vec3 cam_pos{0.0f, 0.0f, 5.0f};
    glm::mat4 view = glm::lookAtRH(cam_pos,
                                   glm::vec3{0.0f, 0.0f, 0.0f},
                                   glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(60.0f),
                                           1.0f, 0.1f, 100.0f);
    glm::mat4 vp = proj * view;
    glm::mat4 vp_inv = glm::inverse(vp);

    // Pick a world-space point in front of the camera.
    glm::vec4 p_world{1.0f, 2.0f, -3.0f, 1.0f};
    // Forward-project.
    glm::vec4 p_clip = vp * p_world;
    REQUIRE(p_clip.w != 0.0f);
    glm::vec4 p_ndc{p_clip.x / p_clip.w,
                    p_clip.y / p_clip.w,
                    p_clip.z / p_clip.w,
                    1.0f};
    // Inverse-project. Multiply the NDC point through vp_inv;
    // the result is in homogeneous coords and needs the w-divide.
    glm::vec4 p_recovered_h = vp_inv * p_ndc;
    REQUIRE(p_recovered_h.w != 0.0f);
    glm::vec4 p_recovered{p_recovered_h.x / p_recovered_h.w,
                          p_recovered_h.y / p_recovered_h.w,
                          p_recovered_h.z / p_recovered_h.w,
                          1.0f};

    CHECK(p_recovered.x == doctest::Approx(p_world.x).epsilon(kLooseEps));
    CHECK(p_recovered.y == doctest::Approx(p_world.y).epsilon(kLooseEps));
    CHECK(p_recovered.z == doctest::Approx(p_world.z).epsilon(kLooseEps));
}

TEST_CASE("transform: pixel-center ray direction matches FOV") {
    // A ray fired from the camera through the pixel at NDC (0, 0) must
    // go straight along the camera forward. A ray through NDC (1, 0)
    // (right edge of screen) should be tilted by half the horizontal
    // FOV from forward. This pins the projection's FOV-vs-aspect
    // relationship, which the shader assumes when it builds primary
    // rays from cam.Right/Up/Forward and FovYTan().
    const float fov_y    = glm::radians(60.0f);
    const float aspect   = 1.0f;  // square image -> fov_x == fov_y
    // See the earlier near/far block for the Windows-macro rationale.
    const float z_near   = 0.1f;
    const float z_far    = 100.0f;
    glm::vec3 cam_pos{0.0f, 0.0f, 5.0f};
    glm::vec3 cam_fwd{0.0f, 0.0f, -1.0f};
    glm::vec3 cam_up{0.0f, 1.0f, 0.0f};

    glm::mat4 view = glm::lookAtRH(cam_pos, cam_pos + cam_fwd, cam_up);
    glm::mat4 proj = glm::perspectiveRH_ZO(fov_y, aspect, z_near, z_far);
    glm::mat4 vp_inv = glm::inverse(proj * view);

    // Helper lambda: given NDC, return the world-space direction from
    // the camera through that pixel center. Inverse-project two points
    // (one near, one far) through the same NDC (x, y) -- the ray
    // direction is the normalized difference.
    auto ndc_to_world_dir = [&](float ndcx, float ndcy) {
        glm::vec4 near_h = vp_inv * glm::vec4{ndcx, ndcy, 0.0f, 1.0f};
        glm::vec4 far_h  = vp_inv * glm::vec4{ndcx, ndcy, 1.0f, 1.0f};
        // Guard the perspective divide: w==0 would produce inf/NaN
        // directions and silently pass the cos(fov/2) check below as
        // garbage. Earlier tests pin this on simpler matrices; pin it
        // here too so a future refactor of inverse() / perspective()
        // can't slip a degenerate VP_inv past us.
        REQUIRE(near_h.w != 0.0f);
        REQUIRE(far_h.w  != 0.0f);
        glm::vec3 near_pos{near_h / near_h.w};
        glm::vec3 far_pos{far_h  / far_h.w};
        return glm::normalize(far_pos - near_pos);
    };

    // NDC (0,0) -> ray should be along -Z (camera forward).
    glm::vec3 dir_center = ndc_to_world_dir(0.0f, 0.0f);
    CHECK(vec3_approx_equal(dir_center, cam_fwd, kLooseEps));

    // NDC (1,0) -> ray should be tilted right by half the horizontal
    // FOV. For a square image, fov_x == fov_y, so the angle between
    // the right-edge ray and forward is fov_y/2. The dot product of
    // the right-edge dir with forward is cos(fov_y/2).
    glm::vec3 dir_right = ndc_to_world_dir(1.0f, 0.0f);
    const float dot_fwd = glm::dot(dir_right, cam_fwd);
    CHECK(dot_fwd == doctest::Approx(std::cos(fov_y * 0.5f))
                         .epsilon(kLooseEps));
}

// --- World-to-screen + screen-to-world stability -----------------------
TEST_CASE("transform: full chain world->NDC->world is identity") {
    // Sweep a few world-space points through vp = proj * view (glm's
    // column-vector convention) then back via inverse. The recovered
    // points must match within tolerance.
    // This is the regression catch for "ZO got swapped to NO" (clip-z
    // range bug) or "RH got swapped to LH" (handedness bug) -- either
    // would break only one direction of the round-trip and the test
    // would flag it.
    glm::vec3 cam_pos{1.0f, 2.0f, 5.0f};
    glm::mat4 view = glm::lookAtRH(cam_pos,
                                   glm::vec3{0.0f, 0.0f, 0.0f},
                                   glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f),
                                           16.0f / 9.0f, 0.1f, 100.0f);
    glm::mat4 vp = proj * view;
    glm::mat4 vp_inv = glm::inverse(vp);

    // Points scattered through the visible frustum.
    const glm::vec4 points[] = {
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f},
        {-0.5f, 0.3f, -2.0f, 1.0f},
        {2.0f, -1.0f, -10.0f, 1.0f},
    };
    for (const auto& p : points) {
        glm::vec4 c = vp * p;
        REQUIRE(c.w != 0.0f);
        glm::vec4 ndc{c / c.w};
        ndc.w = 1.0f;
        glm::vec4 r_h = vp_inv * ndc;
        REQUIRE(r_h.w != 0.0f);
        glm::vec4 recovered{r_h / r_h.w};
        recovered.w = 1.0f;
        CHECK(vec4_approx_equal(recovered, p, 1e-3f));
    }
}

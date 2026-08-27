// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace pt::renderer {

// Near / far planes of the engine's projection. Named here because the
// motion-vector reconstruction (#255) has to rebuild LAST frame's
// projection with exactly the same constants this frame's used, or the
// two matrices stop being comparable.
//
// Only .x, .y and .w of the clip-space product are consumed by the
// motion-vector write, and for a right-handed projection w = -z_view,
// which depends on neither plane -- so today these are inert for motion
// vectors. They are not inert for MetalFX, which is handed
// viewToClipMatrix directly.
inline constexpr float kCameraNearZ = 0.05f;
inline constexpr float kCameraFarZ  = 500.0f;

struct CameraMatrices {
    glm::mat4 view;        // world(render frame) -> view
    glm::mat4 proj;        // view -> clip, reverse-Z-off, [0,1] depth
    glm::mat4 view_proj;   // proj * view
};

// noinline, deliberately. See the exactness note on BuildCameraMatrices.
#if defined(_MSC_VER)
#  define PT_CAMERA_NOINLINE __declspec(noinline)
#else
#  define PT_CAMERA_NOINLINE __attribute__((noinline))
#endif

// The engine's render-frame camera matrices, in ONE place so the shipped
// path and its regression test cannot drift apart (#255).
//
// `eye_render` is the camera position IN THE RENDER FRAME -- i.e. already
// anchor-relative. That is the entire fix for the pre-#255 motion-vector
// cancellation: lookAtRH's translation column is -R*eye, and the shader
// multiplies this matrix against anchor-relative hit positions, so an
// eye of magnitude 6.4e6 made view-space position a difference of two
// ~6.4e6 float32 quantities and produced ~0.5 m of noise -- hundreds of
// pixels of motion-vector jitter at close range, every frame, rebase or
// no rebase.
//
// Feeding the PREVIOUS frame's pose through this same function in the
// CURRENT anchor is what makes motion vectors continuous across a
// rebase: both matrices and the hit point then live in one frame, and
// because the anchor lattice is a power of two the previous eye's
// re-expression is exact.
//
// WHY noinline, AND WHY ALL THREE MATRICES COME OUT OF ONE CALL.
// The motion-vector scheme rebuilds the previous frame's view-projection
// from the previous frame's pose and requires the result to equal, BIT
// FOR BIT, the matrix that frame actually rendered with -- otherwise a
// static camera emits a sub-pixel motion vector and SVGF's temporal
// reprojection picks a different tap. Two inline expansions of the same
// arithmetic are not obliged to agree under Release's -ffast-math /
// /fp:fast: the compiler may contract one into an FMA and not the other.
// Forcing a single out-of-line body makes "same inputs, same bits" a
// property of the machine code rather than a hope. Measured: with this
// inline, light_primitives_mixed__metal__svgf_atrous moves 3 pixels by
// one LSB; with it out of line, nothing moves at all.
//
// The three matrices ship together for the same reason -- MetalFX is
// handed worldToView and viewToClip separately while the shader is
// handed the product, and computing the product anywhere other than
// beside its own factors reintroduces the divergence.
PT_CAMERA_NOINLINE inline CameraMatrices
BuildCameraMatrices(const glm::vec3& eye_render, const glm::vec3& fwd,
                    float fov_deg, float aspect) {
    CameraMatrices m;
    m.view = glm::lookAtRH(eye_render, eye_render + fwd, glm::vec3(0, 1, 0));
    m.proj = glm::perspectiveRH_ZO(glm::radians(fov_deg), aspect,
                                   kCameraNearZ, kCameraFarZ);
    m.view_proj = m.proj * m.view;
    return m;
}

struct Camera {
    // Planetary P1 (#255): the camera position is CANONICAL WORLD metres
    // in double precision, and it is the one thing in the engine that
    // integrates absolute position over time (UpdateCamera does
    // `pos_w += dir * speed * dt` every frame). At WGS-84 radius one
    // float32 ULP is 0.5 m, so a float camera cannot even be nudged --
    // and it is the camera that the render frame's anchor is derived
    // from, so its precision floors everything downstream.
    //
    // Renamed from `pos` deliberately: the `_w` suffix marks the
    // canonical frame, and the rename turns every read site into a
    // compile error so none of them can silently keep treating the
    // camera as a render-frame quantity. Convert with
    // pt::core::WorldFrame::ToRender at the GPU pack boundary and
    // NOWHERE else.
    //
    // Forward()/Right()/Up() below stay glm::vec3 -- directions are
    // scale-free and must NOT be widened.
    glm::dvec3 pos_w   { 0.0, 1.5, 4.0 };
    float     yaw      { 0.0f };           // radians around world-Y
    float     pitch    { -0.20f };         // radians; clamped +/- 85 deg
    float     fov_deg  { 60.0f };

    glm::vec3 Forward() const noexcept {
        float cp = std::cos(pitch);
        return { std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp };
    }
    glm::vec3 Right() const noexcept {
        return glm::normalize(glm::cross(Forward(), glm::vec3{0, 1, 0}));
    }
    glm::vec3 Up() const noexcept {
        return glm::normalize(glm::cross(Right(), Forward()));
    }
    float FovYTan() const noexcept {
        return std::tan(glm::radians(fov_deg) * 0.5f);
    }

    void ClampPitch() noexcept {
        constexpr float lim = glm::radians(85.0f);
        if (pitch >  lim) pitch =  lim;
        if (pitch < -lim) pitch = -lim;
    }
};

}  // namespace pt::renderer

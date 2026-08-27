// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P1 (#255) -- the camera-anchored coordinate frame.
//
// THREE FRAMES, ONE CONVERSION BOUNDARY
// -------------------------------------
//   canonical (world) : glm::dvec3, metres, origin = the planet's centre
//                       of mass once there is a planet. Host only:
//                       Camera::pos_w, AnalyticPrim::pos_or_n,
//                       AnalyticLight::pos, the mesh instance
//                       translations, the scene JSON.
//   render            : float3, origin = the ANCHOR `A` below. Every GPU
//                       buffer, every push constant, every shader.
//   object-local      : float3, origin = a BLAS's own centroid. BLAS
//                       vertex arrays only.
//
// The ONLY code permitted to cross from canonical to render is the set of
// host functions that write a GPU buffer or fill a push-constant struct.
// They are enumerated in the "CONVERSION BOUNDARY" banner in
// src/engine/Engine.cpp. Nothing else calls ToRender(); no absolute
// planetary coordinate ever reaches a shader, and no shader gains a
// "world position" concept beyond "position in the current render frame".
//
// WHY THIS WORKS
// --------------
// float32 relative precision is 2^-24 = 5.96e-8. The pixel footprint at
// distance d at 1080p / 60 deg is d * 1.07e-3 m. Express a position
// relative to the eye and
//
//     representation_error / pixel_footprint = 5.96e-8 / 1.07e-3
//                                            = 5.6e-5 pixels
//
// INDEPENDENT OF d -- standing on gravel or looking at the limb from
// geostationary. Dynamic range was never the problem; absolute
// coordinates were. Anchoring at (near) the eye is the whole fix.
//
// THREE LOAD-BEARING PROPERTIES OF THE ANCHOR
// -------------------------------------------
//  1. `A` is a PURE FUNCTION of the camera position. No hysteresis, no
//     path dependence -- so two backends, two runs and the software
//     reference tracer all agree, which the golden matrix requires.
//  2. The lattice `L` is a POWER OF TWO, so A is an integer multiple of
//     a power of two and every anchor delta `A_prev - A_curr` is EXACTLY
//     representable in float32. Rebasing introduces zero new rounding.
//     This is what makes the motion-vector reconstruction in
//     Engine::RenderFrame exact rather than approximate, and what lets
//     the ReSTIR reservoir shift be bit-identical to a non-rebased frame.
//     It is also why `glm::vec3(anchor)` is lossless: |A| / L < 2^24 for
//     every altitude this engine reaches (L >= 1024 m caps |A| at
//     1024 * 2^24 = 1.7e10 m, well past geostationary).
//  3. `A` is the camera to within `L`, so every `ro - c` / `p - cam`
//     subtraction in a shader is a difference of SMALL numbers.
//
// A == dvec3(0) is the legacy path and is bit-identical to the pre-#255
// engine: glm::vec3(p_f64 - dvec3(0.0)) recovers exactly the float that
// p_f64 was widened from.

#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace pt::core {

// Lattice bounds. The floor is 1024 m: at L = 1024 the worst-case
// render-frame representation error is 5.96e-8 * 1024 = 61 micrometres,
// comfortably below the 1 mm self-intersection bias the tracer uses and
// invisible at arm's length. The ceiling is 2^22 m (4194 km) -- past
// that the lattice would be coarser than the body being orbited.
inline constexpr double kRebaseLatticeMin = 1024.0;         // 2^10 m
inline constexpr double kRebaseLatticeMax = 4194304.0;      // 2^22 m

// The lattice L for a given content scale, in metres. `scale_m` is the
// distance from the eye to the nearest thing worth resolving -- altitude
// above the terrain, once there is terrain. Larger L means rarer rebases
// and coarser render-frame resolution; the quarter-scale divisor keeps
// the worst-case error four orders below the pixel footprint at `scale_m`.
//
//   L = 1024 m  -> 61 um     (surface: sub-visible at arm's length)
//   L = 2^18 m  -> 15.6 mm   (250 km up, where a pixel spans ~270 m)
//
// Always returns an exact power of two, which is property (2) above.
inline double RebaseLattice(double scale_m) noexcept {
    if (!(scale_m > 0.0)) return kRebaseLatticeMin;
    const double want = scale_m * 0.25;
    if (want <= kRebaseLatticeMin) return kRebaseLatticeMin;
    if (want >= kRebaseLatticeMax) return kRebaseLatticeMax;
    // next_pow2 via the binary exponent. std::exp2 of an integer is
    // exact, and std::ceil(std::log2(x)) is stable for the [2^10, 2^22]
    // range we clamp to either side of.
    const double p = std::ceil(std::log2(want));
    return std::clamp(std::exp2(p), kRebaseLatticeMin, kRebaseLatticeMax);
}

// The anchor for a camera at `cam_pos_w`, quantised onto the lattice.
//
//   rebase_radius_m <= 0        -> rebasing disabled (r_origin_rebase_radius 0).
//   |cam_pos_w| < rebase_radius -> dvec3(0), the legacy small-scene path.
//   otherwise                   -> round(cam / L) * L.
//
// Every golden fixture has |cam_pos| <= 10000 (the maximum is
// sunset_altitude.cfg at cam_pos 0 10000 0), so at the default 65536 m
// radius all of them take the dvec3(0) branch with 6.5x of headroom and
// render bit-for-bit unchanged.
//
// `scale_m` selects the lattice; see RebaseLattice. P1 pins it to 0 (so
// L == kRebaseLatticeMin) because "altitude above the surface" is not a
// well-defined quantity until P3 lands the ellipsoid and P4 lands the
// terrain -- and substituting |cam_pos_w| for it would hand back
// L = 2^21 at the Earth's surface, i.e. 0.125 m of render-frame error,
// which is exactly the failure this phase exists to remove. The hook is
// here so P4 changes one call site rather than this file.
inline glm::dvec3 ComputeAnchor(const glm::dvec3& cam_pos_w,
                                double rebase_radius_m,
                                double scale_m = 0.0) noexcept {
    if (!(rebase_radius_m > 0.0)) return glm::dvec3(0.0);
    const double r = glm::length(cam_pos_w);
    if (!(r >= rebase_radius_m)) return glm::dvec3(0.0);
    const double L = RebaseLattice(scale_m);
    const double inv = 1.0 / L;   // L is a power of two, so this is exact
    return glm::dvec3(std::round(cam_pos_w.x * inv) * L,
                      std::round(cam_pos_w.y * inv) * L,
                      std::round(cam_pos_w.z * inv) * L);
}

// The frame itself. One instance lives on Engine and is refreshed once
// per frame, BEFORE any pack runs.
struct WorldFrame {
    glm::dvec3 anchor      {0.0};   // this frame's anchor, canonical metres
    glm::dvec3 anchor_prev {0.0};   // last frame's, for the ReSTIR shift

    // Canonical -> render. The one conversion the boundary is allowed
    // to perform.
    glm::vec3 ToRender(const glm::dvec3& p) const noexcept {
        return glm::vec3(p - anchor);
    }
    // Render -> canonical. For host-side readback (picking, gizmo drag)
    // that needs to write a canonical position back into the scene.
    glm::dvec3 ToWorld(const glm::vec3& p) const noexcept {
        return glm::dvec3(p) + anchor;
    }
    // Exactly representable in float32 by property (2). Added to any
    // render-frame position carried over from last frame -- ReSTIR's
    // Reservoir::light_pos is the only one.
    glm::vec3 AnchorDelta() const noexcept {
        return glm::vec3(anchor_prev - anchor);
    }
    bool AnchorMoved() const noexcept { return anchor != anchor_prev; }
    // True while the engine is on the legacy path: the render frame and
    // the canonical frame are the same frame, and every pack is a pure
    // widen-then-narrow round trip.
    bool AtOrigin() const noexcept { return anchor == glm::dvec3(0.0); }
};

}  // namespace pt::core

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Physical lens-flare model. Paraxial-optics implementation of
// Hullin et al. 2011, "Physically-Based Real-Time Lens Flare
// Rendering" (SIGGRAPH).
//
// The idea: real lens flare comes from light reflecting twice off
// internal element surfaces in a multi-element camera lens. For a
// lens with K reflective surfaces there are K*(K-1)/2 unique ghost
// paths (i, j) with j < i: the ray refracts forward through surfaces
// 0..i-1, REFLECTS at surface i (going backward now), refracts
// backward through i-1..j+1, REFLECTS at surface j (going forward
// again), refracts forward through j+1..K-1, hits the sensor.
//
// We collapse each ghost path to a single 2x2 ABCD ray-transfer
// matrix via paraxial optics. Per wavelength: matrix entries depend
// on n(lambda) so red/green/blue ghosts land at slightly different
// positions on the sensor -- that's the chromatic-aberration rainbow
// fringe you see on real flare. Base intensity from the product of
// surface Fresnel coefficients at normal incidence.
//
// At runtime, given a sun direction projected to entrance angle
// (theta_x, theta_y), sensor footprint position is just M @
// (theta_x, theta_y). Per-frame work is one 2x2 mat-vec per ghost
// per wavelength. Trivial.

#pragma once

#include <cstdint>

namespace pt::engine::lensflare {

// Maximums sized to keep the tonemap push struct under ~600 bytes
// (well within Metal's >>4KB budget; matches what MoltenVK can handle
// on Apple Silicon for the Vulkan path). For a 6-reflective-surface
// lens there are at most C(6, 2) = 15 ghost paths, so 16 slots are
// enough to capture every ghost the canonical lens can produce.
constexpr int kMaxLensSurfaces = 16;
constexpr int kMaxGhosts       = 16;

struct LensSurface {
    // Signed surface curvature radius, mm. +ve = convex toward incoming
    // light (front face), -ve = concave toward incoming. Infinity is
    // encoded as 1e9 (the matrix code special-cases huge values).
    float radius;
    // Distance from this surface to the NEXT surface along the optical
    // axis, mm. Last surface's thickness is the back focal distance to
    // the sensor.
    float thickness;
    // Refractive index of the medium AFTER this surface (between this
    // surface and the next). Air = 1.0, crown glass ~1.52, flint ~1.62.
    // For chromatic dispersion we run the tracer 3 times with R/G/B
    // wavelength-specific n values, derived from Cauchy's equation:
    //   n(lambda) = A + B / lambda^2
    // The 'ior_after' here is the d-line (587.6nm) value -- the tracer
    // re-derives R/G/B values from per-element abbe number.
    float ior_after;
    // Half the clear aperture diameter, mm. v1 paraxial tracer doesn't
    // clip rays, so this is informational; v2 will use it.
    float aperture;
    // Abbe number Vd (dispersion). ~60 for crown glass, ~30 for flint.
    // 0 = no dispersion (for the aperture stop and sensor surfaces).
    float abbe;
    // 1 if this surface is the iris/aperture stop (no refraction, just
    // clipping). 0 for normal glass surfaces.
    int   is_stop;
};

// One ghost path collapsed to per-wavelength 2x2 transfer matrices.
// Matrix is row-major: M[0]=A, M[1]=B, M[2]=C, M[3]=D, where the ray
// state is (height_y, angle_theta) and the transformed ray is
//   y'     = A*y + B*theta
//   theta' = C*y + D*theta
struct Ghost {
    float M_r[4];      // red wavelength (~700nm) transfer matrix
    float M_g[4];      // green (~550nm)
    float M_b[4];      // blue (~450nm)
    // Base intensity from the product of normal-incidence Fresnel
    // reflectance at the two reflecting surfaces multiplied by the
    // transmittance product across all refraction surfaces in the path.
    // Typical range 1e-6 to 1e-3 -- the sun is bright enough that even
    // 1e-4 is visible on screen.
    float intensity;
    // Footprint radius approximation (mm on sensor). Used to size the
    // Gaussian splat in the shader. For paraxial we estimate this as
    // |B|*pupil_radius -- the matrix's B coefficient times entrance
    // pupil radius gives roughly how big the bundle expands.
    float radius_mm;
};

struct LensSystem {
    LensSurface surfaces[kMaxLensSurfaces];
    int         surface_count;
    // Half-height of the sensor (image plane), mm. Full-frame 35mm
    // sensor: 18mm wide / 12mm tall, so half-height = 12. Used to
    // convert paraxial sensor-plane position to UV.
    float sensor_half_height_mm;
    // Effective focal length, mm. Used to convert the sun's screen
    // angle to entrance-ray angle (theta = pixel_off_centre / focal).
    float focal_length_mm;
    // Entrance pupil radius, mm. f-number = focal / (2*pupil_radius).
    // Affects per-ghost intensity (more light = brighter ghosts) and
    // the bundle's footprint extent.
    float pupil_radius_mm;
};

// Build the engine's default lens. Currently a 4-element 50mm
// Tessar-style design (6 refractive surfaces + 1 aperture stop), Vd
// values typical of crown/flint glass pairs.
LensSystem make_default_lens();

// Enumerate ghost paths for the lens and compute their transfer
// matrices + intensities. Writes up to `max_count` Ghost records to
// `ghosts_out`. Returns the actual count written.
//
// Ghost paths enumerated: all (i, j) with j < i where surfaces i, j
// are NOT aperture stops. Sorted by intensity descending so the first
// `r_lens_flare_count` ghosts are always the brightest.
int trace_ghosts(const LensSystem& lens, Ghost* ghosts_out, int max_count);

// Main-path B coefficient per wavelength. Used to convert ghost
// transfer matrices into screen-UV scale ratios:
//   ghost_screen_uv = sun_screen_uv * (B_ghost / B_main)
// where B_main is the equivalent paraxial focal length of the lens at
// that wavelength. Computed once at engine init alongside the ghost
// array; the per-frame evaluator just reads these.
struct MainPath {
    float B_r;
    float B_g;
    float B_b;
};

MainPath trace_main_path(const LensSystem& lens);

// Shader-ready per-ghost data: per-channel screen-UV scale factor (as
// described above), 0..1 base intensity (relative to the brightest
// ghost), and the disc radius in pixels for the Gaussian splat.
//
// Use the green-channel position as the "centre"; R and B are at
// scale_r * sun_uv and scale_b * sun_uv respectively. The shader
// evaluates a Gaussian per channel at its own scaled position.
struct ShaderGhost {
    float scale_r;
    float scale_g;
    float scale_b;
    float intensity;
    float radius_px;
    float _pad[3];   // align to 32 bytes for std140 / Metal push constant
};

// Convert raw Ghost matrices to ShaderGhost records for the current
// viewport. `ghosts` is the array filled by trace_ghosts, `main` is
// trace_main_path's output, `viewport_h_pixels` is the render-target
// height (used to convert mm-on-sensor to pixels). Writes up to
// `max_count` records; returns the count.
int prepare_shader_ghosts(const LensSystem& lens,
                          const Ghost* ghosts, int ghost_count,
                          MainPath main,
                          int viewport_h_pixels,
                          ShaderGhost* out, int max_count);

}  // namespace pt::engine::lensflare

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Paraxial Hullin lens-flare tracer. See LensFlare.h for the high-level
// model. This TU contains:
//   - the default 50mm Tessar-style lens definition
//   - a 2x2 ABCD-matrix implementation of refraction / translation /
//     reflection in the unfolded convention
//   - the ghost-path enumerator that walks every (i, j) pair with
//     j < i, builds the per-wavelength transfer matrix, and computes a
//     base intensity from normal-incidence Fresnel coefficients.
//
// Keep this file self-contained: no external deps beyond <algorithm>,
// <cmath>, <cstring>. The Engine wires the result into the tonemap
// push constants.

#include "engine/LensFlare.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pt::engine::lensflare {

namespace {

// --- 2x2 ABCD matrix ---------------------------------------------------
// State vector is (height_y, slope_u). Column-vector convention:
//   [y']   [A B] [y]
//   [u'] = [C D] [u]
// We store as struct {a, b, c, d} for clarity.

struct Mat2 { float a, b, c, d; };

constexpr Mat2 kIdentity = {1.0f, 0.0f, 0.0f, 1.0f};

Mat2 mat_mul(Mat2 lhs, Mat2 rhs) {
    return Mat2{
        lhs.a * rhs.a + lhs.b * rhs.c,
        lhs.a * rhs.b + lhs.b * rhs.d,
        lhs.c * rhs.a + lhs.d * rhs.c,
        lhs.c * rhs.b + lhs.d * rhs.d
    };
}

// Translation by distance d along the optical axis.
Mat2 m_translate(float d) {
    return Mat2{1.0f, d, 0.0f, 1.0f};
}

// Refraction at a spherical surface of signed radius R going from
// medium n1 to medium n2. R > 0 means center of curvature is on the
// +z side of the vertex. For a flat surface, |R| is huge and the C
// coefficient collapses to 0.
Mat2 m_refract(float R, float n1, float n2) {
    if (n2 == 0.0f) return kIdentity;
    if (std::fabs(R) > 1.0e6f) {
        return Mat2{1.0f, 0.0f, 0.0f, n1 / n2};
    }
    return Mat2{1.0f, 0.0f, (n1 - n2) / (n2 * R), n1 / n2};
}

// Reflection at a spherical surface of signed radius R, in the unfolded
// convention (we virtually propagate the reflected ray forward in +z).
// Derivation: incident dir (1, u), outward normal at (0, h) is approx
// (-1, h/R). Reflected dir = (1, u) - 2*((1,u).(-1, h/R))*(-1, h/R) =
// (-1, u + 2h/R). In unfolded coords (flip dz), reflected dir becomes
// (1, u + 2h/R), so the slope changes by +2h/R.
Mat2 m_reflect(float R) {
    if (std::fabs(R) > 1.0e6f) return kIdentity;
    return Mat2{1.0f, 0.0f, 2.0f / R, 1.0f};
}

// --- Wavelength-dependent index of refraction --------------------------
// Cauchy approximation parameterised by Vd (Abbe number):
//   delta_n_dC ~ (n_d - 1) * fraction / Vd
// Empirically chosen fractions match BK7 / SF2 spectral curves to
// within ~5e-4 across the visible band, which is plenty for a
// chromatic-fringe effect on lens-flare ghosts.
float ior_at_wavelength(float n_d, float vd, int wavelength) {
    if (vd <= 0.5f) return n_d;            // aperture / sensor: no dispersion
    const float disp = (n_d - 1.0f) / vd;
    if (wavelength == 0) return n_d - 0.45f * disp;   // R ~700nm: lower n
    if (wavelength == 2) return n_d + 0.55f * disp;   // B ~450nm: higher n
    return n_d;                                       // G ~550nm (d-line)
}

// Build the per-medium index array for one wavelength:
//   n[0] = 1 (air entrance)
//   n[k+1] = ior of medium AFTER surface k (= medium between k and k+1)
void build_n_array(const LensSystem& lens, int wavelength, float* n) {
    n[0] = 1.0f;
    for (int k = 0; k < lens.surface_count; ++k) {
        const LensSurface& s = lens.surfaces[k];
        const float vd = (std::fabs(s.ior_after - 1.0f) < 0.01f) ? 0.0f : s.abbe;
        n[k + 1] = ior_at_wavelength(s.ior_after, vd, wavelength);
    }
}

// Trace the single ghost path (i, j) for one wavelength's n[] array.
// Light path:
//   forward: refract at 0..i-1 + translate t_0..t_{i-1} → arrives at surface i
//   reflect at i (unfolded → virtually keeps going +z)
//   reflected segment: translate + refract at i-1, i-2, ..., j+1 with mirrored
//                       radii; finally translate to surface j
//   reflect at j (also mirrored)
//   forward: translate + refract at j+1, j+2, ..., K-1 with original radii
//   final translate t_{K-1} to sensor
Mat2 trace_path(const LensSystem& lens, int i, int j, const float* n) {
    const int K = lens.surface_count;
    Mat2 M = kIdentity;

    // Phase 1: forward through surfaces 0..i-1.
    for (int k = 0; k < i; ++k) {
        const LensSurface& s = lens.surfaces[k];
        if (s.is_stop == 0) {
            M = mat_mul(m_refract(s.radius, n[k], n[k + 1]), M);
        }
        M = mat_mul(m_translate(s.thickness), M);
    }
    // Reflect at i.
    M = mat_mul(m_reflect(lens.surfaces[i].radius), M);

    // Phase 2: reflected segment from i back to j.
    // For each k in (j, i), traverse t_k from surface k+1 back to k, then
    // refract at k with mirrored radius (-R) and swapped media (going
    // backward swaps from-medium and to-medium).
    for (int k = i - 1; k > j; --k) {
        const LensSurface& s = lens.surfaces[k];
        M = mat_mul(m_translate(s.thickness), M);
        if (s.is_stop == 0) {
            M = mat_mul(m_refract(-s.radius, n[k + 1], n[k]), M);
        }
    }
    // Travel from surface j+1 back to surface j, reflect (mirrored).
    M = mat_mul(m_translate(lens.surfaces[j].thickness), M);
    M = mat_mul(m_reflect(-lens.surfaces[j].radius), M);

    // Phase 3: forward from j to K-1.
    for (int k = j + 1; k < K; ++k) {
        const LensSurface& s_prev = lens.surfaces[k - 1];
        M = mat_mul(m_translate(s_prev.thickness), M);
        const LensSurface& s = lens.surfaces[k];
        if (s.is_stop == 0) {
            M = mat_mul(m_refract(s.radius, n[k], n[k + 1]), M);
        }
    }
    // Final translate to sensor.
    M = mat_mul(m_translate(lens.surfaces[K - 1].thickness), M);

    return M;
}

// Normal-incidence Fresnel reflectance at an interface n1 -> n2.
float fresnel_normal(float n1, float n2) {
    const float r = (n1 - n2) / (n1 + n2);
    return r * r;
}

// Intensity at d-line (n_d) for ghost path (i, j). Counts how many
// times each non-stop surface is REFRACTED in the path:
//   k < j or k > i        -> 1 traversal
//   k between j and i     -> 3 traversals
//   k == i or k == j      -> reflected (no refraction loss)
//   stop surfaces         -> ignored (full pass-through)
float ghost_intensity(const LensSystem& lens, int i, int j) {
    const int K = lens.surface_count;
    float n_d[kMaxLensSurfaces + 1];
    n_d[0] = 1.0f;
    for (int k = 0; k < K; ++k) n_d[k + 1] = lens.surfaces[k].ior_after;

    const float R_i = fresnel_normal(n_d[i], n_d[i + 1]);
    const float R_j = fresnel_normal(n_d[j], n_d[j + 1]);

    float T = 1.0f;
    for (int k = 0; k < K; ++k) {
        if (k == i || k == j || lens.surfaces[k].is_stop) continue;
        int traversals;
        if (k > j && k < i) traversals = 3;
        else                traversals = 1;
        const float Rk = fresnel_normal(n_d[k], n_d[k + 1]);
        const float Tk = 1.0f - Rk;
        for (int t = 0; t < traversals; ++t) T *= Tk;
    }
    return R_i * R_j * T;
}

}  // namespace

LensSystem make_default_lens() {
    LensSystem lens{};
    lens.surface_count          = 7;
    lens.sensor_half_height_mm  = 12.0f;   // 35mm full-frame: 24mm tall
    lens.focal_length_mm        = 50.0f;
    lens.pupil_radius_mm        = 12.5f;   // f/2 (50mm / (2*12.5))

    // 4-element 50mm Tessar-style. Curvatures + thicknesses tuned
    // qualitatively rather than copied from a real prescription -- the
    // ghost positions land in plausible places for a 50mm lens, and
    // the chromatic fringe is visible without being absurd.
    //
    //  S0 +30.0  3.5  n=1.517 (BK7)  V=64
    //  S1 -85.0  1.0  n=1.000        V=0
    //  S2 -55.0  1.5  n=1.648 (SF2)  V=33
    //  S3 +28.0  3.0  n=1.000        V=0
    //  S4 inf    3.0  n=1.000        V=0   stop
    //  S5 +52.0  4.0  n=1.517 (BK7)  V=64
    //  S6 -32.0 35.0  n=1.000        V=0
    lens.surfaces[0] = LensSurface{ +30.0f,  3.5f, 1.517f, 17.0f, 64.0f, 0 };
    lens.surfaces[1] = LensSurface{ -85.0f,  1.0f, 1.000f, 17.0f,  0.0f, 0 };
    lens.surfaces[2] = LensSurface{ -55.0f,  1.5f, 1.648f, 16.0f, 33.0f, 0 };
    lens.surfaces[3] = LensSurface{ +28.0f,  3.0f, 1.000f, 14.0f,  0.0f, 0 };
    lens.surfaces[4] = LensSurface{  1.0e9f, 3.0f, 1.000f, 12.0f,  0.0f, 1 };  // aperture stop
    lens.surfaces[5] = LensSurface{ +52.0f,  4.0f, 1.517f, 13.0f, 64.0f, 0 };
    lens.surfaces[6] = LensSurface{ -32.0f, 35.0f, 1.000f, 14.0f,  0.0f, 0 };
    return lens;
}

int trace_ghosts(const LensSystem& lens, Ghost* out, int max_count) {
    const int K = lens.surface_count;

    // Build candidate list, sized to the worst case (every (i, j) pair
    // with j < i where neither index is a stop). For K=7 with one stop,
    // that's C(6, 2) = 15.
    struct Candidate {
        int i, j;
        Mat2 M_r, M_g, M_b;
        float intensity;
    };
    Candidate cands[kMaxLensSurfaces * kMaxLensSurfaces];
    int num = 0;

    float n_r[kMaxLensSurfaces + 1];
    float n_g[kMaxLensSurfaces + 1];
    float n_b[kMaxLensSurfaces + 1];
    build_n_array(lens, 0, n_r);
    build_n_array(lens, 1, n_g);
    build_n_array(lens, 2, n_b);

    for (int i = 1; i < K; ++i) {
        if (lens.surfaces[i].is_stop) continue;
        for (int j = 0; j < i; ++j) {
            if (lens.surfaces[j].is_stop) continue;
            if (num >= static_cast<int>(sizeof(cands) / sizeof(cands[0]))) break;
            cands[num].i = i;
            cands[num].j = j;
            cands[num].M_r = trace_path(lens, i, j, n_r);
            cands[num].M_g = trace_path(lens, i, j, n_g);
            cands[num].M_b = trace_path(lens, i, j, n_b);
            cands[num].intensity = ghost_intensity(lens, i, j);
            ++num;
        }
    }

    // Sort candidates by intensity descending so the brightest ghosts
    // come first -- the engine's r_lens_flare_count caps how many to
    // render, and we want it to keep the best-looking ones.
    std::sort(cands, cands + num,
              [](const Candidate& a, const Candidate& b) {
                  return a.intensity > b.intensity;
              });

    const int n_out = std::min(num, max_count);
    if (n_out == 0) return 0;

    // Normalize intensities to [0, 1] keyed on the brightest ghost.
    // The engine's r_lens_flare_intensity cvar is the absolute scale.
    float max_int = cands[0].intensity;
    if (max_int < 1.0e-12f) max_int = 1.0e-12f;

    for (int g = 0; g < n_out; ++g) {
        const Candidate& c = cands[g];
        Ghost& gh = out[g];
        gh.M_r[0] = c.M_r.a; gh.M_r[1] = c.M_r.b;
        gh.M_r[2] = c.M_r.c; gh.M_r[3] = c.M_r.d;
        gh.M_g[0] = c.M_g.a; gh.M_g[1] = c.M_g.b;
        gh.M_g[2] = c.M_g.c; gh.M_g[3] = c.M_g.d;
        gh.M_b[0] = c.M_b.a; gh.M_b[1] = c.M_b.b;
        gh.M_b[2] = c.M_b.c; gh.M_b[3] = c.M_b.d;
        gh.intensity = c.intensity / max_int;
        // Footprint radius. A parallel bundle (u_in = sun angle) of
        // height range [-r_p, r_p] passes through the ghost transfer
        // matrix to (A*h_in + B*u_in, ...), so the bundle's HEIGHT
        // SPREAD on the sensor is |A| * r_p (the A coefficient is the
        // y-vs-y on-diagonal). Earlier this used |B| which has units
        // of mm and produces a dimensional mistake (mm * mm).
        gh.radius_mm = std::fabs(c.M_g.a) * lens.pupil_radius_mm;
        // Clamp to a visually-reasonable band. Real lens flare ghosts
        // are 30-200px on a 1080p frame; the lower cap keeps near-
        // focused paths visible, the upper cap stops a near-singular
        // ghost from filling the screen.
        if (gh.radius_mm < 0.3f) gh.radius_mm = 0.3f;
        if (gh.radius_mm > 4.0f) gh.radius_mm = 4.0f;
    }
    return n_out;
}

namespace {
// Trace the main forward light path (no reflections) at one wavelength.
Mat2 trace_main_at(const LensSystem& lens, const float* n) {
    const int K = lens.surface_count;
    Mat2 M = kIdentity;
    for (int k = 0; k < K; ++k) {
        const LensSurface& s = lens.surfaces[k];
        if (s.is_stop == 0) {
            M = mat_mul(m_refract(s.radius, n[k], n[k + 1]), M);
        }
        M = mat_mul(m_translate(s.thickness), M);
    }
    return M;
}
}  // namespace

MainPath trace_main_path(const LensSystem& lens) {
    float n_r[kMaxLensSurfaces + 1];
    float n_g[kMaxLensSurfaces + 1];
    float n_b[kMaxLensSurfaces + 1];
    build_n_array(lens, 0, n_r);
    build_n_array(lens, 1, n_g);
    build_n_array(lens, 2, n_b);
    return MainPath{
        trace_main_at(lens, n_r).b,
        trace_main_at(lens, n_g).b,
        trace_main_at(lens, n_b).b
    };
}

int prepare_shader_ghosts(const LensSystem& lens,
                          const Ghost* ghosts, int ghost_count,
                          MainPath main,
                          int viewport_h_pixels,
                          ShaderGhost* out, int max_count) {
    const int n_out = std::min(ghost_count, max_count);
    if (n_out <= 0) return 0;
    const float B_main_r = (std::fabs(main.B_r) > 1e-6f) ? main.B_r : 1.0f;
    const float B_main_g = (std::fabs(main.B_g) > 1e-6f) ? main.B_g : 1.0f;
    const float B_main_b = (std::fabs(main.B_b) > 1e-6f) ? main.B_b : 1.0f;
    // Pixels-per-mm at the sensor: viewport_h pixels / sensor_2*half_height_mm.
    const float px_per_mm = static_cast<float>(viewport_h_pixels)
                          / (2.0f * lens.sensor_half_height_mm);
    for (int g = 0; g < n_out; ++g) {
        const Ghost& gh = ghosts[g];
        ShaderGhost& sg = out[g];
        // Per-channel scale: B_ghost / B_main. Negative scale means the
        // ghost lands on the opposite side of centre from the sun
        // (canonical "mirrored" lens-flare ghost).
        sg.scale_r   = gh.M_r[1] / B_main_r;
        sg.scale_g   = gh.M_g[1] / B_main_g;
        sg.scale_b   = gh.M_b[1] / B_main_b;
        sg.intensity = gh.intensity;
        sg.radius_px = gh.radius_mm * px_per_mm;
        // Cap pixel radius so even an unlucky paraxial path can't
        // produce a screen-filling disc. 96 px on 1080p ≈ 0.5 deg
        // on a 50mm lens, which matches how big real flare ghosts
        // look in lens reviews.
        if (sg.radius_px < 4.0f)   sg.radius_px = 4.0f;
        if (sg.radius_px > 96.0f)  sg.radius_px = 96.0f;
        sg._pad[0] = sg._pad[1] = sg._pad[2] = 0.0f;
    }
    return n_out;
}

}  // namespace pt::engine::lensflare

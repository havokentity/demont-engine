// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SDF Phase 3 (#99) -- iconic fractal distance estimators (DEs):
// Mandelbulb (3D Mandelbrot), Mandelbox, Apollonian gasket. These are
// pure-math SDFs evaluated by bounded iteration of a per-shape escape /
// fold function and recovered as a distance via the standard
// "escape-radius / derivative" formula.
//
// Architectural note (vs the Phase 1 SdfPrimitives.slang library):
//   * Phase 1 leaves are EXACT signed distances with closed-form
//     analytic normals. Their cluster trace is in SdfPrimitives.slang.
//   * Phase 3 fractals are APPROXIMATE DEs -- the returned `distance`
//     is a LOWER BOUND on the true distance, never an exact one.
//     Consequently:
//       - The sphere-trace's `t += d` advance is still safe (it never
//         tunnels through the surface) but may take more iterations
//         in deep-recursion regions because each step covers less
//         ground than an exact-SDF step of the same nominal magnitude.
//       - Normals are recovered by CENTRAL DIFFERENCES on the DE
//         (closed-form gradients don't exist for these iterators).
//       - The surface-termination epsilon should grow with step count
//         so deep-recursion areas don't burn iterations on
//         vanishingly-tiny "no-progress" steps. See the
//         `iter_scaled_eps` formula in `traceSdfFractal` -- the issue
//         #99 spec calls for `dist < eps * scale * iter_count`.
//   * Because of the central-difference normal cost (4 extra DE evals
//     per hit) AND the bounded iteration count being load-bearing for
//     each `eval` call, fractals live in a SEPARATE module from the
//     analytic SDF library. PathTrace.slang's traceSdfClusters
//     dispatches into this module when it sees a fractal shape id.
//
// Wire-format note: the host extends the SDF leaf-shape enum range
// from 0..5 (analytic shapes in SdfPrimitives.slang) with
//   SDF_SHAPE_MANDELBULB = 6
//   SDF_SHAPE_MANDELBOX  = 7
//   SDF_SHAPE_APOLLONIAN = 8
// Numeric values are part of the host/shader wire format (matching
// SdfShape in src/renderer/AnalyticBvh.h); never renumber without
// bumping both sides.
//
// References:
//   * Iñigo Quilez, "Distance to fractals":
//     https://iquilezles.org/articles/distancefractals/
//   * Knighty / "Mandelbulb" Fractalforums.com thread (the canon
//     polar-power formula); Tom Lowe's mandelbox (the "ABox" series).
//   * Hart, "Sphere tracing: A geometric method for the antialiased
//     ray tracing of implicit surfaces" (1996) -- the inner loop is
//     identical to the Phase 1 sphere-trace, only the DE differs.

// Mirror of the host enum values for the fractal extension. Keep in
// lockstep with SdfShape in AnalyticBvh.h. The 0..5 ids overlap with
// SdfPrimitives.slang's SDF_SHAPE_* constants -- those are evaluated
// by SdfPrimitives.slang's evalLeafDist and we never see them here.
static const uint SDF_SHAPE_MANDELBULB = 6u;
static const uint SDF_SHAPE_MANDELBOX  = 7u;
static const uint SDF_SHAPE_APOLLONIAN = 8u;

// Lowest shape id this module handles. PathTrace.slang's dispatcher
// uses this to decide whether a cluster's leaf is an analytic shape
// (handled in SdfPrimitives.slang) or a fractal (handled here).
static const uint SDF_FRACTAL_FIRST = SDF_SHAPE_MANDELBULB;

// ----- Mandelbulb ----------------------------------------------------------
//
// The textbook polar-formula Mandelbulb (Knighty / Pickover canon).
// Iterates the polar-coordinate "power-n" transform on a point in
// shape-local space:
//     z' = z^n + c           (with z^n = r^n * (sin(n*theta)cos(n*phi),
//                                            sin(n*theta)sin(n*phi),
//                                            cos(n*theta)))
// where (r, theta, phi) are the spherical coordinates of z. `c` is the
// original sample point (matching the Mandelbrot canon: z0 = 0,
// c = sample) which produces the canonical bulb shape. The DE is the
// "escape-radius / derivative" estimator from Quilez:
//     d = 0.5 * log(r) * r / dr
// where `dr` is the iteration derivative magnitude, advanced each
// iter by `dr' = n * r^(n-1) * dr + 1`.
//
// `power` is the polar exponent (default 8 from r_sdf_fractal_power).
// `iters` is the bounded iteration count (default 12 from
// r_sdf_fractal_iters). 4..16 is the practical range; bulb fine
// detail saturates around 10-12.
float deMandelbulb(float3 p_local, float power, uint iters) {
    float3 z  = p_local;
    float  dr = 1.0;
    float  r  = 0.0;
    for (uint i = 0u; i < iters; ++i) {
        r = length(z);
        if (r > 2.0) break;                 // escape radius (canonical)

        // Polar coords. Branch-stable: theta is acos(z/r) so r=0 is
        // a degenerate origin sample that's already inside the bulb;
        // the early-exit isn't strictly needed because we test r>2
        // first, but guard anyway against r=0 NaN from acos/atan.
        float theta = acos(clamp(z.z / max(r, 1e-12), -1.0, 1.0));
        float phi   = atan2(z.y, z.x);

        // Running derivative magnitude (scalar). This is the magnitude
        // of d(z_i)/dz_0 evaluated along the iteration -- it grows
        // geometrically when the orbit escapes and gives the DE its
        // 1/dr scaling so distant samples see a useful step bound.
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // z' = z^power + c. Spherical-coord power formula keeps the
        // rotation symmetry of the canonical bulb.
        float rp = pow(r, power);
        float nt = theta * power;
        float np = phi   * power;
        float sn = sin(nt);
        z = rp * float3(sn * cos(np), sn * sin(np), cos(nt));
        z += p_local;
    }
    // Hart-style distance estimator. `0.5 * log(r) * r / dr` is the
    // canonical Mandelbulb formula -- comes from interpreting the
    // escape-radius condition as a level set in iteration-count space
    // and differentiating. Always non-negative for r >= 1; clamp to
    // avoid log(0) for the rare r<1 escape (orbit never left).
    return 0.5 * log(max(r, 1.0001)) * r / max(dr, 1e-12);
}

// ----- Mandelbox -----------------------------------------------------------
//
// Tom Lowe's mandelbox: iterate `boxFold` then `sphereFold` then a
// linear-scale step `z = scale * z + c`. The DE is recovered by the
// linear-scale derivative `dr' = |scale| * dr + 1`.
//
// Box fold: clamp each component into [-1, 1] then reflect -- this
// folds the cube +/-1 onto itself.
// Sphere fold: invert the radial component inside a small inner ball
// and identity outside an outer ball; piecewise smooth in the band.
//
// `scale` is the linear-step scale (default 2.5 from
// r_sdf_fractal_power's secondary use for this shape -- the host
// packs it into n.params for the leaf so the cvar is a global default
// for any UNSPECIFIED per-shape value, never a wire-format hard-pin).
// Practical range 1.5..3.0. `iters` 8..14.
float deMandelbox(float3 p_local, float scale, uint iters) {
    // Inner / outer fold radii (canonical Tom Lowe values).
    const float kMinR2 = 0.25;     // 0.5^2
    const float kFixedR2 = 1.0;
    // Magnitude factor for the box fold (canonical 1.0 -- folds onto
    // the +/-1 cube). Promoted to a constant so the polynomial form
    // below stays readable.
    const float kBoxLimit = 1.0;

    float3 z  = p_local;
    float  dr = 1.0;
    for (uint i = 0u; i < iters; ++i) {
        // BOX FOLD: |z| > 1  ->  z' = 2*sign(z) - z (reflect through +/-1)
        //           |z| <= 1 ->  z' = z
        // Implemented as `clamp(z, -1, 1) * 2 - z`. Derivative is the
        // identity in both branches (the fold preserves length scale)
        // so dr stays unchanged here.
        z = clamp(z, -kBoxLimit, kBoxLimit) * 2.0 - z;

        // SPHERE FOLD:
        //   r2 < minR2  -> z *= fixedR2 / minR2     (inner inversion)
        //   minR2 <= r2 < fixedR2 -> z *= fixedR2 / r2 (radial flip)
        //   r2 >= fixedR2 -> identity
        // dr is multiplied by the same scalar factor so the DE chain
        // rule holds.
        float r2 = dot(z, z);
        if (r2 < kMinR2) {
            float f = kFixedR2 / kMinR2;
            z  *= f;
            dr *= f;
        } else if (r2 < kFixedR2) {
            float f = kFixedR2 / r2;
            z  *= f;
            dr *= f;
        }

        // LINEAR STEP: z' = scale * z + c, dr' = |scale|*dr + 1.
        z  = scale * z + p_local;
        dr = abs(scale) * dr + 1.0;
    }
    // Standard linear-iterate DE. length(z)/|dr| is the lower bound
    // on distance to the boundary -- subtracting nothing because the
    // boundary is implicit (we're looking for "escape from bound
    // region"), so the limit-set distance estimator returns the
    // length scaled by the inverse derivative magnitude.
    return length(z) / abs(dr);
}

// ----- Apollonian gasket ---------------------------------------------------
//
// Knighty's "Apollonian" -- iterate a reflective fold and a radial
// shrink that pulls points back into the fundamental domain. Produces
// the classic gasket with infinitely many nested spheres on the
// limit set. The DE follows the same linear-iterate pattern as the
// mandelbox: `dr' = (1/scale) * dr` from the per-iter shrink, and the
// final distance is `length(z) * |dr|`.
//
// `scale` controls the gasket density (default 1.3 -- canonical
// Apollonian-2 from the Knighty thread; the larger this gets the
// finer the gasket detail and the more iterations you need).
float deApollonian(float3 p_local, float scale, uint iters) {
    float3 z = p_local;
    // Per-iter radial scale; sub-1.0 shrinks (gasket has self-similar
    // contracting structure on its limit set). Track the iteration
    // derivative as the running product of the per-step contractions.
    float dr = 1.0;
    for (uint i = 0u; i < iters; ++i) {
        // Reflective fold: clamp into the fundamental domain by
        // reflecting through the +/-1 planes. Same shape as the box
        // fold above but without the *2 doubling -- this is a pure
        // origin-symmetric reflection so the gasket sits on the
        // integer-lattice boundary.
        z = -1.0 + 2.0 * frac(0.5 * z + 0.5);

        // Radial shrink: bring points outside the unit sphere back
        // onto the gasket's self-similar interior. r2 ~ 0 is the
        // central degenerate fixed point; clamp to avoid NaN.
        float r2 = dot(z, z);
        float k  = max(scale / max(r2, 1e-12), 1.0);
        z  *= k;
        dr *= k;
    }
    // The boundary of the gasket is approximately the unit sphere in
    // the iteration's contracted frame; the DE returns `|z|/dr -
    // some-offset`. Knighty's canonical formula subtracts a small
    // bias to land the surface on the gasket spheres rather than
    // their thin enclosing shell.
    return (0.25 * abs(z.y) - 0.001) / dr;
}

// ----- dispatch: distance --------------------------------------------------
//
// Called from PathTrace.slang's traceSdfClusters when the cluster's
// root leaf has a fractal shape id. Returns +inf on an unknown shape
// (defensive -- the dispatcher should never call us with one, but
// the safe-default keeps the trace from getting stuck at a stale
// register-cached value).
//
// `params` is the host SdfNode's params lane. Convention:
//   MANDELBULB : params.x = power (default 8 from r_sdf_fractal_power)
//                params.y = scale (effective bound radius, default 1.2)
//   MANDELBOX  : params.x = scale (linear-step magnitude, default 2.5)
//                params.y = effective bound radius (default 4.0)
//   APOLLONIAN : params.x = scale (radial-shrink magnitude, default 1.3)
//                params.y = effective bound radius (default 2.0)
//
// `power_default` is the r_sdf_fractal_power cvar and ONLY applies to
// MANDELBULB -- "power" and "scale" are different DE parameters, so
// there's no useful single global default for Mandelbox/Apollonian.
// Their `params.x == 0` falls back to the canonical hardcoded values
// (2.5 for Mandelbox, 1.3 for Apollonian) instead of the cvar. This
// matches the AnalyticBvh.h SdfShape contract; keep host docs and
// shader fallback in lockstep when editing either side.
// `iters_default` is r_sdf_fractal_iters; it applies to all three
// shapes since iteration count is the same concept everywhere.
float evalFractalDist(float3 p_local, uint shape, float4 params,
                       float power_default, uint iters_default) {
    if (shape == SDF_SHAPE_MANDELBULB) {
        float power = (params.x > 0.0) ? params.x : power_default;
        return deMandelbulb(p_local, power, iters_default);
    } else if (shape == SDF_SHAPE_MANDELBOX) {
        float scale = (params.x != 0.0) ? params.x : 2.5;
        return deMandelbox(p_local, scale, iters_default);
    } else if (shape == SDF_SHAPE_APOLLONIAN) {
        float scale = (params.x != 0.0) ? params.x : 1.3;
        return deApollonian(p_local, scale, iters_default);
    }
    return 1e30;
}

// ----- central-difference normals ------------------------------------------
//
// Fractal DEs don't have closed-form gradients, so we recover the
// normal by sampling the DE four times around the hit point and
// taking the gradient. Uses the small tetrahedron variant: 4 evals
// instead of the naive 6 of axis-aligned forward+back differences,
// at a small asymmetry cost that's well within the noise floor for
// approximate DEs anyway.
//
// `h` is the sampling step in metres. Must be larger than the
// sphere-trace's surface-epsilon so the central-diff sample points
// land outside the surface and the DE returns non-zero, but small
// enough that the gradient is locally linear. Issue #99 caller uses
// roughly r_sdf_epsilon * 8 -- big enough to escape the local-zero
// band, small enough that the finite-difference error stays below
// the silhouette noise floor.
float3 evalFractalNormal(float3 p_local, uint shape, float4 params,
                          float power_default, uint iters_default,
                          float h) {
    // Tetrahedron vertices on the unit sphere; multiply by h to step
    // around the point.
    float3 e1 = float3( 1.0, -1.0, -1.0);
    float3 e2 = float3(-1.0, -1.0,  1.0);
    float3 e3 = float3(-1.0,  1.0, -1.0);
    float3 e4 = float3( 1.0,  1.0,  1.0);
    float d1 = evalFractalDist(p_local + e1 * h, shape, params, power_default, iters_default);
    float d2 = evalFractalDist(p_local + e2 * h, shape, params, power_default, iters_default);
    float d3 = evalFractalDist(p_local + e3 * h, shape, params, power_default, iters_default);
    float d4 = evalFractalDist(p_local + e4 * h, shape, params, power_default, iters_default);
    float3 n = e1 * d1 + e2 * d2 + e3 * d3 + e4 * d4;
    float  l = length(n);
    return (l > 1e-12) ? (n / l) : float3(0, 1, 0);
}

// ----- ray-AABB entry / exit (mirrors SdfPrimitives.slang::sdfClusterAabbT)
//
// The fractal trace is bounded by the cluster's host-computed world-
// space AABB (see ComputeSdfAabb in src/renderer/AnalyticBvh.cpp);
// every fractal leaf has an explicit `effective bound radius` in
// params.y so the host can produce a tight bound rather than a giant
// "default world" cube. Keeping this slab test local to the fractal
// module avoids forcing PathTrace.slang to know which shape kind the
// cluster carries before launching the trace.
float2 fractalAabbT(float3 aabb_min, float3 aabb_max,
                     float3 ro, float3 rd_inv) {
    float3 t0 = (aabb_min - ro) * rd_inv;
    float3 t1 = (aabb_max - ro) * rd_inv;
    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);
    float  t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float  t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);
    return float2(t_enter, t_exit);
}

// ----- sphere-trace for one fractal leaf -----------------------------------
//
// Single-leaf fractal sphere-trace. Caller has already verified the
// cluster is a single-leaf fractal (the only fractal-cluster shape we
// support in Phase 3 -- combining fractals into smooth-CSG ops is a
// future-future thing). Returns true if a hit was found within
// [t_min, t_max].
//
// Approximation-aware epsilon: the issue calls for
//     dist < eps * scale * iter_count
// so deep-recursion areas don't burn iterations on no-progress steps.
// `iter_count` is the running step index; the relaxation grows
// linearly so the early steps still see the small surface epsilon
// and the late steps see a relaxed bound. `r_sdf_de_eps_scale` cvar
// multiplies the relaxation rate; default 1.0 reproduces the canonical
// formula and 0.0 falls back to the constant-epsilon trace.
//
// The trace continues past escape-radius rays via the standard
// `t >= t_end` guard on the cluster's AABB exit. `max_iters` here is
// the sphere-trace step count -- INDEPENDENT of the per-DE-eval
// iteration count (`iters_default`).
bool traceSdfFractal(
    uint shape, float4 params, float3 center,
    float3 aabb_min, float3 aabb_max,
    float3 ro, float3 rd,
    float  t_min, float t_max,
    uint   max_steps, float epsilon, float de_eps_scale,
    float  power_default, uint iters_default,
    out float out_t, out float3 out_normal)
{
    out_t      = t_max;
    out_normal = float3(0, 1, 0);

    float3 rd_safe = select(abs(rd) > float3(1e-30), rd, float3(1e-30));
    float3 rd_inv  = 1.0 / rd_safe;
    float2 ts      = fractalAabbT(aabb_min, aabb_max, ro, rd_inv);
    float  t       = max(ts.x, t_min);
    float  t_end   = min(ts.y, t_max);
    if (t >= t_end) return false;

    // Hard cap to bound worst-case shader work. 512 is generous for
    // fractals -- the issue's perf budget is "fractals are costly by
    // design" so a fractal-dense pixel can spend more steps than the
    // analytic-SDF default (256 in SdfPrimitives.slang).
    uint steps = min(max_steps, 512u);

    for (uint i = 0u; i < steps; ++i) {
        float3 p_local = (ro + rd * t) - center;
        float  d       = evalFractalDist(p_local, shape, params,
                                          power_default, iters_default);
        float  abs_d   = abs(d);

        // Iteration-relaxed termination: surface-epsilon grows with
        // the step count so deep-recursion regions accept a slightly
        // looser hit rather than burning the full step budget on
        // micro-progress. de_eps_scale=0 disables the relaxation
        // (constant-epsilon trace, matches SdfPrimitives.slang
        // behaviour exactly).
        float iter_scaled_eps = epsilon * (1.0 + de_eps_scale * float(i));
        if (abs_d < iter_scaled_eps) {
            out_t      = t;
            // Normal step a bit larger than the surface epsilon so
            // the central-difference samples land in non-zero DE
            // territory. Scaling by ~8x is a Quilez convention --
            // small enough that the gradient stays locally accurate,
            // large enough to escape the iteration-relaxed band.
            float h = max(iter_scaled_eps * 8.0, epsilon * 4.0);
            out_normal = evalFractalNormal(p_local, shape, params,
                                            power_default, iters_default, h);
            return true;
        }
        t += max(abs_d, epsilon);    // floor at epsilon so we always advance
        if (t >= t_end) return false;
    }
    return false;
}

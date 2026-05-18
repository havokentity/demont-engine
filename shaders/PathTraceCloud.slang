// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Procedural cloud field: hash-based 3D value noise + fbm + density
// shaping. Compiled to a Slang IR module so PathTrace.slang's import
// resolves to a pre-compiled blob (parallel ninja build, no front-
// end re-parse on every PathTrace.spv rebuild).
//
// The cloud parameters are passed as float4 args, NOT read from the
// Push cbuffer, so this module is self-contained and target-agnostic
// (same .slang-module file links into both the SPIR-V and Metal
// targets of PathTrace.slang).

// Hash-based 3D value noise + fbm. Cheaper than gradient (Perlin)
// noise -- the smoothstep filter step gives enough visual continuity
// for cloud bodies at this scale, and we don't need C2 continuity for
// volumetric sampling. ~30 ALU per sample x ~4 octaves x 16 march
// samples x 1080p = ~6 GFLOP/frame on the worst march, well under
// budget on M4 Max.

// Integer mixing hash (Wellons' Lowbias32 -- 3 multiplies + 3 xorshifts,
// near-perfect avalanche on every input bit).  The previous
// frac(p * 0.3183099 + 0.1) scaled-frac trick had two structural flaws
// that produced visible patches/seams in the cloud body:
//   1. Output biased toward extremes (the product p.x*p.y*p.z*(sum)
//      collapses near zero whenever any component is small).
//   2. Adjacent integer grid cells produce CORRELATED outputs because
//      the multiplied frac() is similar for similar inputs -- visible
//      as low-frequency "patches" even at small p magnitudes.
// Lowbias32 + uint cast on integer-floored input fixes both.
public uint cloud_hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
public float cloud_hash(float3 p) {
    // Caller (cloud_value_noise) always passes already-floored float
    // values; int cast is exact.  Two's-complement wrap on negatives
    // is well-defined and the hash treats it as just another bit
    // pattern, so world-space coords below origin still produce
    // decorrelated cells.
    int3 ip = int3(p);
    uint h = cloud_hash_u32(uint(ip.x));
    h = cloud_hash_u32(h ^ uint(ip.y));
    h = cloud_hash_u32(h ^ uint(ip.z));
    return float(h & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

public float cloud_value_noise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    // Quintic Hermite interpolation: f^3 (6f^2 - 15f + 10). Zero
    // first AND second derivatives at f=0 and f=1, so the noise has
    // no visible cell boundaries even at heavy fbm summation. Cubic
    // smoothstep (3f^2 - 2f^3) leaks a faint grid through the haze.
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = cloud_hash(i + float3(0, 0, 0));
    float b = cloud_hash(i + float3(1, 0, 0));
    float c = cloud_hash(i + float3(0, 1, 0));
    float d = cloud_hash(i + float3(1, 1, 0));
    float e = cloud_hash(i + float3(0, 0, 1));
    float fx = cloud_hash(i + float3(1, 0, 1));
    float gx = cloud_hash(i + float3(0, 1, 1));
    float hx = cloud_hash(i + float3(1, 1, 1));
    return lerp(lerp(lerp(a,  b,  f.x), lerp(c,  d,  f.x), f.y),
                lerp(lerp(e,  fx, f.x), lerp(gx, hx, f.x), f.y),
                f.z);
}

public float cloud_fbm(float3 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    [unroll(6)]
    for (int i = 0; i < octaves; ++i) {
        v += a * cloud_value_noise(p);
        p *= 2.02;          // non-integer freq mult avoids axis-aligned artefacts
        a *= 0.5;
    }
    return v;
}

// 3D curl noise (Bridson et al. 2007, "Curl-Noise for Procedural Fluid
// Flow"). Curl of a 3D potential field psi(p) = (Px, Py, Pz):
//     C(p) = ( dPz/dy - dPy/dz,
//              dPx/dz - dPz/dx,
//              dPy/dx - dPx/dy )
// C is divergence-free by construction (div(curl) = 0), so when used
// as a displacement on the density-sample position it preserves
// volume on average -- the cloud doesn't grow or shrink globally,
// it gets sheared into eddies. That's the right "turbulent fluid"
// character for cumulus filaments and shear-streamer cirrus.
//
// Three decorrelated potentials are needed; we tap cloud_value_noise
// at three large constant offsets so the components don't share a hash.
// If all three sampled the same scalar field N(q), curl(N,N,N) would
// reduce to pair-wise gradient differences of one noise (each component
// = dN/dα - dN/dβ for some axes α,β), which is divergence-free but
// collapses the displacement onto a locally 2D plane and produces
// visible stripe artefacts along (1,1,1). Offsets are arbitrary but
// must be far enough apart that the hash cells don't visibly correlate
// -- 31.4, 217.7, 853.1 give independent-looking outputs.
//
// Derivatives are estimated via central differences with step `h`
// (one-tenth of the noise period at the curl scale). Cheaper than
// analytic gradients for value-noise and visually indistinguishable
// at the scales we care about.
public float3 cloud_curl_noise(float3 p, float scale) {
    // scale = cycles/metre; convert to noise-space coordinate
    float3 q = p * scale;
    // Central-difference step. The smallest representable feature in
    // value-noise is ~one cell (h=1 in noise space); 0.1 keeps the
    // finite difference well inside the smoothed quintic interpolant
    // where the noise is C2, avoiding stair-step artefacts from
    // sampling across cell boundaries.
    const float h = 0.1;
    const float3 ox = float3(h, 0.0, 0.0);
    const float3 oy = float3(0.0, h, 0.0);
    const float3 oz = float3(0.0, 0.0, h);
    // Three decorrelated potential components. Constant offsets keep
    // each component's hash domain separate -- without these all three
    // potentials would sample the same scalar field N(q), so the curl
    // would degenerate to a trivial differential of one noise:
    //   curl(N,N,N) = (dN/dy - dN/dz, dN/dz - dN/dx, dN/dx - dN/dy)
    // -- still divergence-free, but every component is just a pair-wise
    // gradient difference of the SAME field, so the displacement field
    // collapses onto a 2D plane locally and produces stripe artefacts
    // along the (1,1,1) axis instead of true 3D turbulence. The offsets
    // break that aliasing so each potential has its own independent
    // hash domain (Bridson 2007, sec. 3).
    const float3 offX = float3( 31.4, 217.7, 853.1);
    const float3 offY = float3(853.1,  31.4, 217.7);
    const float3 offZ = float3(217.7, 853.1,  31.4);
    // dPx/d{y,z}
    float Px_yp = cloud_value_noise(q + offX + oy);
    float Px_yn = cloud_value_noise(q + offX - oy);
    float Px_zp = cloud_value_noise(q + offX + oz);
    float Px_zn = cloud_value_noise(q + offX - oz);
    // dPy/d{x,z}
    float Py_xp = cloud_value_noise(q + offY + ox);
    float Py_xn = cloud_value_noise(q + offY - ox);
    float Py_zp = cloud_value_noise(q + offY + oz);
    float Py_zn = cloud_value_noise(q + offY - oz);
    // dPz/d{x,y}
    float Pz_xp = cloud_value_noise(q + offZ + ox);
    float Pz_xn = cloud_value_noise(q + offZ - ox);
    float Pz_yp = cloud_value_noise(q + offZ + oy);
    float Pz_yn = cloud_value_noise(q + offZ - oy);
    float inv2h = 1.0 / (2.0 * h);
    float dPz_dy = (Pz_yp - Pz_yn) * inv2h;
    float dPy_dz = (Py_zp - Py_zn) * inv2h;
    float dPx_dz = (Px_zp - Px_zn) * inv2h;
    float dPz_dx = (Pz_xp - Pz_xn) * inv2h;
    float dPy_dx = (Py_xp - Py_xn) * inv2h;
    float dPx_dy = (Px_yp - Px_yn) * inv2h;
    return float3(dPz_dy - dPy_dz,
                  dPx_dz - dPz_dx,
                  dPy_dx - dPx_dy);
}

// Density at a world-space point. Returns 0 outside the cloud layer
// or where the noise threshold isn't met. The volumetric march
// multiplies this by sigma_t to vary extinction along the ray.
//
// `octaves` parameterises the fbm complexity: the primary cloud march
// uses 6 (full visual quality), shadow rays use 3 (rough density
// estimate for Beer-Lambert is enough; saves ~50% of noise lookups
// on the dominant shadow path).
//
// cp1: (base_y, top_y, coverage, peak_sigma_t)
// cp2: (wind_x, wind_z, freq, time_seconds)
// cp3: (seed_offset_x, seed_offset_z, detail_amount, rayleigh_int_unused_here)
// cp4: (curl_amount, curl_scale_cycles_per_m, erosion_amount, _reserved)
//      curl_amount in [0,1] is the displacement magnitude in metres
//      applied to the sample position before density evaluation.
//      curl_scale is the curl-noise frequency in cycles per metre
//      (independent of cp2.z so coarse cloud bodies can carry fine
//      curl filaments).
//      erosion_amount in [0,1] adds a high-frequency edge-erosion
//      term on top of the existing cp3.z detail noise; eats away
//      soft cloud margins into wisps without hollowing dense cores.
//      0 = identical to cp3.z-only behaviour on main.
public float cloud_density_at(float3 pos,
                              float4 cp1, float4 cp2, float4 cp3,
                              float4 cp4,
                              int octaves) {
    if (cp1.w <= 0.0) return 0.0;
    float base_y = cp1.x;
    float top_y  = cp1.y;
    if (pos.y < base_y || pos.y > top_y) return 0.0;
    float h_norm = (pos.y - base_y) / max(top_y - base_y, 1e-3);
    // Height profile: ramp up from the bottom, fall off before the top.
    // Peaks around h_norm = 0.4 -- biases toward dense lower volumes,
    // wispy upper volumes (cumulus shape).
    float h_profile = smoothstep(0.0, 0.2, h_norm)
                    * (1.0 - smoothstep(0.6, 1.0, h_norm));
    if (h_profile <= 0.0) return 0.0;
    // Wind translation + per-day seed offset.
    float3 sp = pos
              + float3(cp2.x * cp2.w, 0.0, cp2.y * cp2.w)
              + float3(cp3.x, 0.0, cp3.y);
    // Curl-noise displacement (Bridson 2007). Applied in world space
    // BEFORE the noise-frequency scaling so curl_scale stays in the
    // physically intuitive cycles/metre regime regardless of cp2.z.
    // Wind-drift the curl input too so the eddy pattern flows with
    // the cloud body (otherwise turbulent detail looks "pinned" to
    // world space while the cloud itself drifts past it -- visually
    // wrong, breaks the fluid-flow illusion). cp4.x = magnitude (m),
    // cp4.y = curl frequency. 0 = no displacement = identical to the
    // pre-curl code path (regression-safe gate for r_clouds_curl_amount 0).
    if (cp4.x > 0.0) {
        float3 c = cloud_curl_noise(sp, cp4.y);
        sp += c * cp4.x;
    }
    sp *= cp2.z;
    // Cumulus body: caller-controlled octaves of quintic-interpolated
    // value-noise fbm. Higher octaves -> more wispy detail, more cost.
    float low = cloud_fbm(sp, octaves);
    // Linear ramp coverage -- saturate((low - (1-coverage)) / coverage).
    // Tried smoothstep but the steep S-curve near the threshold meant a
    // single hi-freq fbm dip into the [1-coverage, 1-coverage+0.05]
    // band collapses to zero density, punching visible holes through
    // the cloud body. Linear keeps the response monotonic and
    // proportional to the noise value -- holes only appear where the
    // noise actually says "no cloud".
    float coverage = saturate(cp1.z);
    float density  = saturate((low - (1.0 - coverage)) / max(coverage, 0.01));
    if (density <= 0.0) return 0.0;
    // Detail erosion: a higher-frequency fbm erodes EDGES of clouds
    // into wispy fingers, but should NOT punch holes in dense cores.
    // The previous unconditional subtract was hollowing out cumulus
    // bodies (donut-shaped clouds). Solution: scale erosion strength
    // by (1 - density)^2 so density 1.0 is fully protected, density
    // 0.5 erodes at 0.25x, density 0.1 erodes at full strength. Real
    // cumulus shows wispy trails, not Swiss-cheese cores.
    if (cp3.z > 0.0) {
        float hi = cloud_fbm(sp * 4.0, 3);
        float edge = (1.0 - density) * (1.0 - density);
        density = saturate(density - hi * cp3.z * edge);
    }
    // Secondary high-frequency edge erosion. Same wispy-finger
    // principle as cp3.z but tuned to a much higher noise frequency
    // (8x base, vs 4x for the primary detail). Produces the
    // filamentous outer fringe characteristic of real cumulus seen
    // up close: cp3.z gives the bulk wispy edge, cp4.z paints the
    // sub-feature wisps that fade into clear air. Same
    // (1 - density)^2 edge mask so dense cores stay protected.
    // 0 = identical to main behaviour (this whole branch elides).
    if (cp4.z > 0.0) {
        float hi2 = cloud_fbm(sp * 8.0, 3);
        float edge2 = (1.0 - density) * (1.0 - density);
        density = saturate(density - hi2 * cp4.z * edge2);
    }
    // Internal variation: a SOFT secondary fbm gives the cloud volume
    // some texture without holes. Bias toward 1.0 (multiplier 0.75-1.0
    // instead of 0.55-1.0) so we don't dim the cloud body uniformly.
    float internal = 0.75 + 0.25 * cloud_fbm(sp * 1.7, 3);
    density *= internal;
    // cp1.w is peak sigma_t per metre. Real meteorology:
    //   light cumulus    ~0.03-0.05 /m
    //   typical cumulus  ~0.05-0.10
    //   stratus / nimbus ~0.04-0.08
    //   storm / Cb       ~0.10-0.30
    return density * h_profile * cp1.w;
}

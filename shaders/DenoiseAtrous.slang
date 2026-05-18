// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Proper SVGF A-Trous wavelet filter (Schied et al. 2017). Edge-stops
// driven by:
//   - depth weight    w_z = exp(-|z_p - z_q| / (sigma_z * |grad_z . offset| + eps))
//   - normal weight   w_n = pow(max(0, dot(n_p, n_q)), sigma_n)
//   - luminance weight w_l = exp(-|l_p - l_q| / (sigma_l * sqrt(g3x3 * Var(l_p)) + eps))
//
// All three terms have the SVGF denominators wired in -- the depth
// gradient lets fronto-parallel surfaces stay strict while grazing
// slopes are lenient, and the variance term lets converged shading
// regions reject neighbours instead of blurring through them. The
// previous "fixed-weight" version shipped a too-tolerant bilateral
// blur; this implementation is the canonical SVGF filter shape.
//
// Variance also rides the cascade: each pass filters its variance with
// the SQUARED edge-aware weights (Var' = Sum(w^2 * Var_q) / (Sum w)^2)
// and the next pass picks it up as its w_l denominator. So later passes
// progressively tighten the luminance gate as the running variance
// drops -- which is what gives SVGF its "wide spatial radius without
// blurring details" property.
//
// Pass schedule: the host runs N dispatches (N = 1..5, configurable via
// the r_svgf_atrous_passes cvar; each pass uses the same 5x5 kernel but
// doubles its tap stride per pass -- 1, 2, 4, 8, 16. So effective
// footprint per pass count: 5x5, 9x9, 17x17, 33x33, 65x65. N = 5 is
// the canonical Schied 2017 configuration; default N = 1. Variance
// ping-pongs through both `variance_in` and `variance_out` slots each
// pass.

[[vk::binding(0,  0)]] RWTexture2D<float4> color_in;            // input radiance (RGBA16F)
[[vk::binding(1,  0)]] RWTexture2D<float4> color_history_unused;// unread; host binds dummy
[[vk::binding(2,  0)]] RWTexture2D<float>  depth_tex;           // R32F linear depth
[[vk::binding(3,  0)]] RWTexture2D<float2> motion_unused;       // unread; host binds dummy
[[vk::binding(4,  0)]] RWTexture2D<float4> normal_tex;          // RGBA16F world normal
[[vk::binding(5,  0)]] RWTexture2D<float4> color_out;           // filtered radiance (RGBA16F)
[[vk::binding(6,  0)]] RWTexture2D<float>  depth_hist_unused;   // unread; host binds anything R32F
[[vk::binding(7,  0)]] RWTexture2D<float4> normal_hist_unused;  // unread; host binds anything RGBA16F
// Buffer-backed bindings start at 8. RWStructuredBuffer<T> is a storage
// buffer in Vulkan / a device pointer in MSL; it does *not* count toward
// Metal's 8-RW-texture compute limit. Indexing is row-major
// (y*width + x); the host sizes each buffer to width*height elements.
[[vk::binding(8,  0)]] RWStructuredBuffer<float2> moments_in_unused;  // unread; host binds any RG32 buffer
[[vk::binding(9,  0)]] RWStructuredBuffer<float2> moments_out_unused; // unwritten; host binds any RG32 buffer
[[vk::binding(10, 0)]] RWStructuredBuffer<float>  variance_in;         // variance from previous pass
[[vk::binding(11, 0)]] RWStructuredBuffer<float>  variance_out;        // filtered variance for next pass
// Albedo G-buffer (binding 12). Issue #119 -- SVGF albedo demodulation.
// Sample-only (`Texture2D`) so it doesn't count toward Metal's 8 RW
// texture cap. The chain runs on the demodulated lighting signal so
// edge-stops aren't fighting texture detail; on the LAST atrous pass
// the kernel multiplies by albedo to produce the final remodulated
// output. Intermediate passes leave the multiply off so the next
// pass's divide cancels cleanly -- with `final_remod = 0` on
// intermediate passes the kernel only divides input and forwards.
[[vk::binding(12, 0)]] Texture2D<float4> albedo_tex;

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  step_size;       // 1, 2, 4, ...
    uint  _pad0;
    float sigma_depth;     // gradient-scaled depth tolerance
    float sigma_normal;    // pow exponent on normal dot product
    float sigma_color;     // variance-scaled luminance tolerance
    float _pad1;
    // Issue #119 -- albedo demodulation gate. 1 = treat color_in as
    // demodulated lighting (divide per-tap to operate the filter on
    // the lighting signal). 0 = legacy (color_in is full radiance,
    // skip the divide/multiply).
    uint  demod_enabled;
    // Issue #119 -- final-pass-of-chain marker. When BOTH demod_enabled
    // and final_remod are set, multiply `filtered_color` by the centre
    // pixel's albedo before writing `color_out`, restoring the textured
    // radiance for the downstream bloom + tonemap chain. The host sets
    // this only on the LAST A-Trous dispatch when the output target is
    // `output` (not the SVGF feedback target). Intermediate passes and
    // the feedback-target pass leave this 0.
    uint  final_remod;
    uint  _pad2;
    uint  _pad3;
};

// Floor on the per-channel albedo divisor. Kept in lockstep with
// DenoiseTemporal.slang.
static const float kDemodEps = 0.01;

// Sky / no-surface sentinel cutoff. Pixels with albedo channel sum
// below this are treated as emissive (sky/miss) and bypass demod
// entirely so the multiply-back doesn't zero them out.
static const float kAlbedoSkyEps = 1.0 / 65536.0;

bool isSkyAlbedo(float3 a) {
    return (a.r + a.g + a.b) <= kAlbedoSkyEps;
}

// Read color_in[p] in (optionally) demodulated space. Identical
// semantics to DenoiseTemporal.slang's helper -- divides by per-tap
// albedo when demod is on, except for sky pixels (albedo == 0)
// which carry emissive radiance that must NOT be remodulated.
// Bypassing demod for sky also makes the variance estimate sensible:
// dividing sky-luminance by kDemodEps would create a 100x luminance
// spike that the A-Trous gate would refuse to average through (and
// so the sky-pixel chain degenerates to pass-through anyway).
float3 readDemodColor(uint2 p) {
    float3 c = color_in[uint2(p)].rgb;
    if (demod_enabled != 0u) {
        float3 a = albedo_tex[uint2(p)].rgb;
        if (!isSkyAlbedo(a)) {
            c /= max(a, float3(kDemodEps, kDemodEps, kDemodEps));
        }
    }
    return c;
}

// Binomial-derived 1D weights. The 2D kernel is the outer product:
//   [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1] / 256
// Each kKernel1D entry is the row sum / 16 -- i.e. the 1D filter that
// produces the 2D kernel when squared. Sum of 1D = 1, sum of 2D = 1.
static const float kKernel1D[5] = {
    1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0
};

// 3x3 Gaussian for the variance pre-filter (Schied 2017 4.4).
// Same binomial pattern as the A-Trous taps, just at 3x3.
//   [1 2 1; 2 4 2; 1 2 1] / 16
static const float kGauss3[3][3] = {
    { 1.0/16.0, 2.0/16.0, 1.0/16.0 },
    { 2.0/16.0, 4.0/16.0, 2.0/16.0 },
    { 1.0/16.0, 2.0/16.0, 1.0/16.0 },
};

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

uint pixelIndex(uint2 p) { return p.y * width + p.x; }

float clampDepthRead(int2 p) {
    int2 q = clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
    return depth_tex[uint2(q)];
}

float clampVarianceRead(int2 p) {
    int2 q = clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
    return variance_in[pixelIndex(uint2(q))];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    // Read the centre pixel in (possibly) demodulated space. We
    // preserve `center.a` for the sample-count tag that rides the
    // chain alongside the RGB. Issue #119: the entire kernel operates
    // on demodulated lighting when `demod_enabled` is set; the
    // multiply-back happens at the final color_out write below.
    float4 center_full  = color_in[tid];
    float3 center_rgb   = readDemodColor(tid);
    float4 center       = float4(center_rgb, center_full.a);
    float  center_depth = depth_tex[tid];
    float3 center_n     = normal_tex[tid].xyz;
    float  center_lum   = luminance(center_rgb);
    bool   has_normal   = length(center_n) > 0.5;

    // Depth gradient via forward/backward differences; pick the smaller
    // magnitude on each axis so finite-difference taps crossing a
    // silhouette (fg/sky boundary -> huge "gradient") don't corrupt the
    // edge stop. Standard SVGF improvement; NRD does the same.
    float z_l = clampDepthRead(int2(tid) + int2(-1, 0));
    float z_r = clampDepthRead(int2(tid) + int2( 1, 0));
    float z_d = clampDepthRead(int2(tid) + int2(0, -1));
    float z_u = clampDepthRead(int2(tid) + int2(0,  1));
    float dz_fx = z_r - center_depth;
    float dz_bx = center_depth - z_l;
    float dz_fy = z_u - center_depth;
    float dz_by = center_depth - z_d;
    float dz_dx = (abs(dz_fx) < abs(dz_bx)) ? dz_fx : dz_bx;
    float dz_dy = (abs(dz_fy) < abs(dz_by)) ? dz_fy : dz_by;

    // 3x3 Gaussian-prefiltered variance at the center. This is the
    // "stationarity" measurement that drives the luminance gate: low
    // value (converged) -> tight gate; high value (noisy) -> loose
    // gate. We use the prefiltered value rather than variance_in[tid]
    // directly because the per-pixel variance is itself noisy, and
    // jitter in the gate denominator translates to flicker in the
    // filtered output.
    float gauss_var = 0.0;
    [unroll] for (int gy = -1; gy <= 1; ++gy) {
        [unroll] for (int gx = -1; gx <= 1; ++gx) {
            gauss_var += kGauss3[gy + 1][gx + 1] *
                         clampVarianceRead(int2(tid) + int2(gx, gy));
        }
    }
    // sigma_color is multiplied by sqrt(prefiltered_variance); we keep
    // a small additive epsilon so converged regions (variance -> 0)
    // still tolerate a tiny luminance delta -- otherwise quantisation
    // noise rejects every neighbour and the filter degenerates to a
    // pass-through that re-noises later passes.
    float sigma_l_denom = sigma_color * sqrt(max(gauss_var, 0.0)) + 1e-4;

    float3 sum_color  = float3(0.0);
    float  sum_var    = 0.0;
    float  sum_w      = 0.0;
    // Geometric weight sum (depth*normal, NO luminance term). This is
    // the silhouette detector for the edge-aware TAA-preservation
    // blend below. At interior pixels neighbours pass the depth/normal
    // gates -> this sums to ~1.0. At silhouettes the gates collapse
    // toward 0 for off-side neighbours -> sum drops toward the center
    // tap's kw alone (~0.14).
    float  sum_w_geom = 0.0;
    int    s          = int(step_size);

    [unroll] for (int dy = -2; dy <= 2; ++dy) {
        [unroll] for (int dx = -2; dx <= 2; ++dx) {
            int2 q_pix = int2(tid) + int2(dx * s, dy * s);
            if (q_pix.x < 0 || q_pix.y < 0 ||
                q_pix.x >= int(width) || q_pix.y >= int(height)) continue;
            uint2 q = uint2(q_pix);

            float kw = kKernel1D[dx + 2] * kKernel1D[dy + 2];
            // Demodulated tap: divide-by-albedo for the lighting
            // estimate at each neighbour. This matches the centre-tap
            // signal so the luminance edge-stop computes the right
            // delta. When demod is off, readDemodColor falls through
            // to the raw radiance read.
            float3 nc_rgb  = readDemodColor(q);
            float  nd      = depth_tex[q];
            float3 nn      = normal_tex[q].xyz;
            float  nv      = variance_in[pixelIndex(q)];

            // Depth weight: gradient-projected expected delta. Adding
            // a small relative epsilon (1% of center depth) lets
            // fronto-parallel surfaces still average a few neighbours
            // when |grad_z| collapses to 0; without it the filter
            // becomes a no-op on flat geometry.
            float expected_dz = abs(dz_dx * float(dx * s) +
                                    dz_dy * float(dy * s));
            float depth_denom = sigma_depth * expected_dz +
                                1e-2 * max(center_depth, 1e-3);
            float w_depth = exp(-abs(nd - center_depth) / depth_denom);

            // Normal weight: skip the gate when either side is the
            // sky sentinel (zero normal). Sky-sky comparisons keep
            // w_normal = 1; sky-surface boundaries lean on the depth
            // weight (which now sees a huge |z_p - z_q| at the
            // 1000m sky depth + a clean center gradient) to reject.
            float w_normal = 1.0;
            if (has_normal && length(nn) > 0.5) {
                float ndot = max(dot(center_n, nn), 0.0);
                w_normal = pow(ndot, max(sigma_normal, 1e-3));
            }

            // Luminance weight: variance-scaled. The (dx,dy) == (0,0)
            // tap is exactly the center: |l_p - l_q| = 0 -> w_lum = 1.
            float lum_d = abs(luminance(nc_rgb) - center_lum);
            float w_lum = exp(-lum_d / sigma_l_denom);

            float w = kw * w_depth * w_normal * w_lum;
            sum_color += w * nc_rgb;
            // Filtered variance uses w^2 weights (variance of a
            // weighted sum is the sum of squared weights * variances
            // when the inputs are independent -- Schied 2017 (5)).
            sum_var   += w * w * nv;
            sum_w     += w;
            // Geometric weight (kernel * depth * normal, no luminance).
            // The center tap (dx=dy=0) contributes kw_center = 36/256
            // since its depth/normal weights are exactly 1.
            sum_w_geom += kw * w_depth * w_normal;
        }
    }

    float3 filtered_color;
    float  filtered_var;
    if (sum_w > 1e-6) {
        filtered_color = sum_color / sum_w;
        filtered_var   = sum_var / max(sum_w * sum_w, 1e-12);
    } else {
        // Pathological case: all neighbours rejected. Hold the center
        // color + variance so we don't push a 0 into the next pass.
        filtered_color = center.rgb;
        filtered_var   = variance_in[pixelIndex(tid)];
    }

    // Edge-aware TAA preservation. The temporal pass averages
    // jittered 1spp samples over ~32 frames, which sub-pixel-supersamples
    // silhouettes -- center.rgb arrives here as a soft TAA mean.
    // But the A-Trous depth/normal gates asymmetrically reject the
    // off-silhouette side, so the filtered output snaps to whichever
    // side the unjittered G-buffer says this pixel "belongs to" --
    // erasing the TAA mean and producing visible aliasing. Detect
    // that case via the geometric (depth*normal) weight sum: it
    // collapses toward kw_center at silhouettes (only the center tap
    // contributes) and stays near 1.0 in interiors. Lerp the filter
    // output back toward the input center value by that factor so
    // silhouettes preserve the TAA result while interiors get full
    // denoising.
    const float kw_center        = (6.0 / 16.0) * (6.0 / 16.0);
    const float kw_norm_denom    = max(1.0 - kw_center, 1e-6);
    float w_geom_normalized = saturate((sum_w_geom - kw_center) / kw_norm_denom);
    float edge_factor       = 1.0 - w_geom_normalized;
    filtered_color = lerp(filtered_color, center.rgb, edge_factor);
    // Issue #119 remod: on the final pass of the chain (when
    // final_remod is set AND demod is on), multiply the filtered
    // lighting by the centre pixel's albedo so the output is the
    // textured radiance the bloom/tonemap chain expects. Sky pixels
    // (albedo == 0) bypass the multiply because the upstream divide
    // also bypassed them -- their signal carries emissive radiance
    // that should be forwarded unchanged so celestials / sky stay
    // visible in the denoised result.
    // Non-final passes (intermediate atrous, or the pass-1 feedback
    // target) leave the signal demodulated so subsequent passes /
    // the next frame's temporal divide compose correctly.
    if (demod_enabled != 0u && final_remod != 0u) {
        float3 a = albedo_tex[uint2(tid)].rgb;
        if (!isSkyAlbedo(a)) {
            filtered_color *= a;
        }
    }
    // Preserve the sample-count tag from the input so subsequent
    // passes / the next frame's temporal reprojection can still see
    // how converged the source pixel was.
    color_out[tid]                = float4(filtered_color, center.a);
    variance_out[pixelIndex(tid)] = filtered_var;
}

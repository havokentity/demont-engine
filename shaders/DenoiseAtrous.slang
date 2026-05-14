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
// regions reject neighbours instead of blurring through them. Together
// with the 3-pass 5x5 cascade (step=1,2,4 -> effective 17x17) this is
// the canonical SVGF filter shape; the previous "fixed-weight" version
// shipped a too-tolerant bilateral blur.
//
// Variance also rides the cascade: each pass filters its variance with
// the SQUARED edge-aware weights (Var' = Sum(w^2 * Var_q) / (Sum w)^2)
// and the next pass picks it up as its w_l denominator. So later passes
// progressively tighten the luminance gate as the running variance
// drops -- which is what gives SVGF its "wide spatial radius without
// blurring details" property.
//
// A-Trous step sizes: 1 -> 2 -> 4. Run as three host dispatches with
// the variance ping-ponged in lockstep.

[[vk::binding(0,  0)]] RWTexture2D<float4> color_in;            // input radiance (RGBA16F)
[[vk::binding(1,  0)]] RWTexture2D<float4> color_history_unused;// unread; host binds dummy
[[vk::binding(2,  0)]] RWTexture2D<float>  depth_tex;           // R32F linear depth
[[vk::binding(3,  0)]] RWTexture2D<float2> motion_unused;       // unread; host binds dummy
[[vk::binding(4,  0)]] RWTexture2D<float4> normal_tex;          // RGBA16F world normal
[[vk::binding(5,  0)]] RWTexture2D<float4> color_out;           // filtered radiance (RGBA16F)
[[vk::binding(6,  0)]] RWTexture2D<float>  depth_hist_unused;   // unread; host binds anything R32F
[[vk::binding(7,  0)]] RWTexture2D<float4> normal_hist_unused;  // unread; host binds anything RGBA16F
[[vk::binding(8,  0)]] RWTexture2D<float2> moments_in_unused;   // unread; host binds anything RG32F
[[vk::binding(9,  0)]] RWTexture2D<float2> moments_out_unused;  // unwritten; host binds anything RG32F
[[vk::binding(10, 0)]] RWTexture2D<float>  variance_in;         // R32F variance from previous pass
[[vk::binding(11, 0)]] RWTexture2D<float>  variance_out;        // R32F filtered variance for next pass

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
};

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

float clampDepthRead(int2 p) {
    int2 q = clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
    return depth_tex[uint2(q)];
}

float clampVarianceRead(int2 p) {
    int2 q = clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1));
    return variance_in[uint2(q)];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    float4 center       = color_in[tid];
    float  center_depth = depth_tex[tid];
    float3 center_n     = normal_tex[tid].xyz;
    float  center_lum   = luminance(center.rgb);
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

    float3 sum_color = float3(0.0);
    float  sum_var   = 0.0;
    float  sum_w     = 0.0;
    int    s         = int(step_size);

    [unroll] for (int dy = -2; dy <= 2; ++dy) {
        [unroll] for (int dx = -2; dx <= 2; ++dx) {
            int2 q_pix = int2(tid) + int2(dx * s, dy * s);
            if (q_pix.x < 0 || q_pix.y < 0 ||
                q_pix.x >= int(width) || q_pix.y >= int(height)) continue;
            uint2 q = uint2(q_pix);

            float kw = kKernel1D[dx + 2] * kKernel1D[dy + 2];
            float4 nc      = color_in[q];
            float  nd      = depth_tex[q];
            float3 nn      = normal_tex[q].xyz;
            float  nv      = variance_in[q];

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
            float lum_d = abs(luminance(nc.rgb) - center_lum);
            float w_lum = exp(-lum_d / sigma_l_denom);

            float w = kw * w_depth * w_normal * w_lum;
            sum_color += w * nc.rgb;
            // Filtered variance uses w^2 weights (variance of a
            // weighted sum is the sum of squared weights * variances
            // when the inputs are independent -- Schied 2017 (5)).
            sum_var   += w * w * nv;
            sum_w     += w;
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
        filtered_var   = variance_in[tid];
    }
    // Preserve the sample-count tag from the input so subsequent
    // passes / the next frame's temporal reprojection can still see
    // how converged the source pixel was.
    color_out[tid]    = float4(filtered_color, center.a);
    variance_out[tid] = filtered_var;
}

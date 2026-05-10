// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Edge-aware A-trous wavelet filter. Reads the temporally-accumulated
// `color_in` (from DenoiseTemporal) and writes a spatially-filtered
// result into `color_out`. Run this kernel multiple times with
// increasing step_size to widen the effective spatial neighbourhood
// without quadratic cost: step=1 is a 5x5 neighbourhood, step=2 is
// effectively 9x9, step=4 is effectively 17x17, etc. Three passes
// (steps 1, 2, 4) gives a perceptually 17x17 filter at the cost of
// only 3 * 25 taps per pixel.
//
// "Edge-aware" means each tap is weighted by:
//   - depth similarity  (depth_tex)         -> preserves silhouettes
//   - normal similarity (normal_tex)        -> preserves creases
//   - color similarity  (color_in luminance)-> preserves shading detail
//
// Sky pixels (zero normal) skip the normal weight and rely on depth
// + color similarity alone. The kernel matches the reference SVGF
// paper (Schied et al. 2017) but uses fixed weights rather than
// variance-driven ones to keep this pass cheap.

[[vk::binding(0, 0)]] RWTexture2D<float4> color_in;
// Slot 1 unused for the atrous pass; bound to a 1x1 placeholder by
// the host so the pipeline layout matches the temporal pass.
[[vk::binding(1, 0)]] RWTexture2D<float4> color_history_unused;
[[vk::binding(2, 0)]] RWTexture2D<float>  depth_tex;
// Slot 3 unused for the atrous pass.
[[vk::binding(3, 0)]] RWTexture2D<float2> motion_unused;
[[vk::binding(4, 0)]] RWTexture2D<float4> normal_tex;
[[vk::binding(5, 0)]] RWTexture2D<float4> color_out;

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  step_size;       // 1, 2, 4, ...
    uint  _pad0;
    float sigma_depth;     // larger = depth differences more tolerated
    float sigma_normal;    // smaller = normals must agree more strictly
    float sigma_color;     // larger = color differences more tolerated
    float _pad1;
};

// 5x5 a-trous kernel weights (Dammertz / SVGF reference).
//   1/256, 4/256, 6/256, 4/256, 1/256
static const float kKernel1D[5] = {
    1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0
};

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    float4 center      = color_in[tid];
    float  center_depth = depth_tex[tid];
    float3 center_n     = normal_tex[tid].xyz;
    float  center_lum   = luminance(center.rgb);
    bool   has_normal   = length(center_n) > 0.5;

    float3 sum_color  = float3(0.0);
    float  sum_weight = 0.0;
    float  sum_count  = 0.0;
    int    s          = int(step_size);

    [unroll] for (int dy = -2; dy <= 2; ++dy) {
        [unroll] for (int dx = -2; dx <= 2; ++dx) {
            int2 p = int2(tid) + int2(dx * s, dy * s);
            if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height)) continue;
            uint2 up = uint2(p);

            float kw = kKernel1D[dx + 2] * kKernel1D[dy + 2];
            float4 nc      = color_in[up];
            float  nd      = depth_tex[up];
            float3 nn      = normal_tex[up].xyz;

            // Depth weight: relative-depth Gaussian. The depth_tex
            // values are in world units along the camera forward axis,
            // so the absolute scale shifts with scene size; using a
            // relative metric makes sigma_depth scene-independent.
            float depth_d = abs(nd - center_depth) / max(center_depth, 1e-3);
            float w_depth = exp(-depth_d / max(sigma_depth, 1e-3));

            // Normal weight: pow(max(0, dot)). Skip when either side
            // is the sky sentinel.
            float w_normal = 1.0;
            if (has_normal && length(nn) > 0.5) {
                float ndot = max(dot(center_n, nn), 0.0);
                w_normal = pow(ndot, max(sigma_normal, 1e-3));
            }

            // Color (luminance) weight: rejects fireflies that the
            // temporal pass didn't catch.
            float lum_d = abs(luminance(nc.rgb) - center_lum);
            float w_lum = exp(-lum_d / max(sigma_color, 1e-3));

            float w = kw * w_depth * w_normal * w_lum;
            sum_color  += w * nc.rgb;
            sum_weight += w;
            sum_count   = max(sum_count, nc.a);
        }
    }
    float3 filtered = (sum_weight > 1e-6) ? (sum_color / sum_weight) : center.rgb;
    color_out[tid] = float4(filtered, sum_count);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// GPU-side auto-exposure pass.
//
// Replaces the per-frame CPU readback of accum_hdr with a tiny
// compute reduction. A single workgroup of 64 threads samples
// accum_hdr at a regular stride, computes the LOG-AVERAGE
// (geometric mean) luminance via group-shared reduction, then
// writes the smoothed exposure scalar back into the persistent
// ExposureState buffer. The path tracer's tonemap reads
// `exposure_state[0]` directly -- no CPU readback, no GPU-side
// pipeline barrier between the path-trace dispatch and the next
// frame.
//
// Why log-average rather than arithmetic mean? Real-world HDR
// scenes produce extreme outliers -- the sun disc reaches ~1.6e9
// nits while the surrounding sky is ~1e4 and ground objects are
// near 1.0. Arithmetic mean of N samples where one pixel is the
// sun and N-1 are scene gives `mean = (1.6e9 + N*scene)/N` ~=
// 1.6e9/N, which dominates regardless of scene content. Auto-
// exposure then computes target = 0.18 / mean ~= 0, gets clamped
// to exp_min, and the actual scene stays burned out because the
// floor prevents real exposure compensation. Log-average uses the
// geometric mean of luminance: extreme outliers are compressed by
// the log so a single sun pixel barely moves the average. This is
// Reinhard's 2002 standard technique and the de-facto base for
// modern game engine auto-exposure.
//
// Switch to a 256-bin histogram if we ever need percentile
// clipping (cinematic look) or HDR display peak picking.
//
// Bindings:
//   binding 1  : accum_hdr (RWTexture2D<float4>) -- shared with PathTrace
//   binding 15 : exposure_state (RWStructuredBuffer<float>, 1 element)
//
// Push constants (32B, fits within the 112B unified push range):
//   width, height: accum_hdr dimensions
//   stride       : sample step in pixels (16 matches CPU code)
//   key          : middle-grey target (r_exposure_target, e.g. 0.18)
//   exp_min/max  : clamps for current_exposure
//   adapt_speed  : log-space lerp factor; <=0 means "lock to target"

[[vk::binding(1, 0)]]  RWTexture2D<float4> accum_hdr;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> exposure_state;

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  stride;
    uint  pad0;
    float key;
    float exp_min;
    float exp_max;
    float adapt_speed;
};

groupshared float g_lum_sum[64];
groupshared uint  g_lum_count[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupThreadID) {
    uint w_steps = (width  + stride - 1) / stride;
    uint h_steps = (height + stride - 1) / stride;
    uint total   = w_steps * h_steps;

    // Per-thread accumulation in LOG space. The eps floor on lum
    // (1e-6) keeps log() finite for near-black pixels; the if (lum > 0)
    // guard skips truly zero pixels so they don't pull the mean down
    // toward log(eps) ~= -14 and bias dark-scene exposure.
    float sum  = 0.0;
    uint  cnt  = 0;
    for (uint i = gid; i < total; i += 64u) {
        uint x = (i % w_steps) * stride;
        uint y = (i / w_steps) * stride;
        if (x < width && y < height) {
            float3 c = accum_hdr[uint2(x, y)].rgb;
            float lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            if (lum > 0.0) {
                sum += log(max(lum, 1e-6));
                cnt += 1u;
            }
        }
    }
    g_lum_sum[gid]   = sum;
    g_lum_count[gid] = cnt;
    GroupMemoryBarrierWithGroupSync();

    // Single-thread reduction over the 64 partials. The whole job
    // is a few microseconds on a 5090; LDS-tree reduce isn't worth
    // the code complexity here.
    if (gid == 0u) {
        float total_sum   = 0.0;
        uint  total_count = 0u;
        for (uint i = 0u; i < 64u; ++i) {
            total_sum   += g_lum_sum[i];
            total_count += g_lum_count[i];
        }
        if (total_count == 0u) return;

        // Geometric mean: total_sum is the sum of log(lum) values, so
        // exp(total_sum / total_count) is the geometric mean of all
        // sampled luminances. Robust to extreme outliers (sun disc,
        // bright specular highlights) that would dominate an arithmetic
        // mean and force exposure to clamp at exp_min for outdoor
        // scenes regardless of the rest of the frame's luminance.
        float log_avg   = total_sum / float(total_count);
        float avg_lum   = exp(log_avg);
        float target    = key / max(avg_lum, 1e-6);
        target          = clamp(target, exp_min, exp_max);

        float current = exposure_state[0];
        if (current <= 0.0 || adapt_speed <= 0.0) {
            // Either uninitialised, or "locked" preset (DSLR/linear):
            // jump straight to target. Smoothing kicks in only when
            // adapt_speed > 0 AND there's a prior value to fade from.
            current = target;
        } else {
            // Geometric (log-space) lerp: perceptually-uniform fade.
            current = exp(log(current) * (1.0 - adapt_speed) +
                          log(target)  * adapt_speed);
        }
        exposure_state[0] = current;
    }
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Proper SVGF temporal accumulation (Schied et al. 2017). Reprojects the
// previous frame's color + luminance moments using motion vectors,
// validates the reprojection against depth+normal history, blends with
// the current noisy radiance, and emits:
//
//   color_out          -- temporally-blended RGB + sample count in .a
//   moments_out        -- temporally-blended luminance moments (mu1, mu2)
//   variance_out       -- per-pixel luminance variance, ready for the
//                         A-Trous chain to use as its w_l denominator
//
// Two changes from the previous "SVGF-lite":
//   1. Moments + variance: track running mean and mean-of-squares of
//      luminance via a 2nd ping-pong history (moments_history_a/b).
//      Variance = mu2 - mu1^2 gives the A-Trous pass a real measure of
//      how converged the pixel is, so it can keep stationary regions
//      crisp instead of bilateral-blurring them.
//   2. Spatial variance fallback: pixels with sample_count < 4 don't
//      have enough temporal samples for a reliable variance estimate
//      (just-disoccluded surfaces). For those pixels we additionally
//      compute a 7x7 spatial luminance variance and lerp toward it
//      based on sample count -- so newly-revealed surfaces get a fat
//      luminance sigma until temporal moments stabilise.
//
// History invariants:
//   color_history_in.a       -- running sample count (clamped to 32).
//   moments_history_in.r     -- mu1 (running mean luminance).
//   moments_history_in.g     -- mu2 (running mean of luminance^2).
//
// On reprojection failure (off-screen, depth/normal mismatch) all
// three trails reset: color = noisy_now, moments = (l_now, l_now^2),
// variance falls back to the 7x7 spatial estimate.

[[vk::binding(0,  0)]] RWTexture2D<float4> color_in;            // path-tracer noisy radiance, RGBA16F
[[vk::binding(1,  0)]] RWTexture2D<float4> color_history_in;    // prev-frame accumulated color, RGBA16F
[[vk::binding(2,  0)]] RWTexture2D<float>  depth_tex;           // current-frame linear depth, R32F
[[vk::binding(3,  0)]] RWTexture2D<float2> motion_tex;          // pixel-space motion (prev - curr), RG16F
[[vk::binding(4,  0)]] RWTexture2D<float4> normal_tex;          // current-frame world-space normal, RGBA16F
[[vk::binding(5,  0)]] RWTexture2D<float4> color_out;           // accumulated color, RGBA16F (.a = sample count)
[[vk::binding(6,  0)]] RWTexture2D<float>  depth_history_in;    // prev-frame depth, R32F
[[vk::binding(7,  0)]] RWTexture2D<float4> normal_history_in;   // prev-frame normal, RGBA16F
[[vk::binding(8,  0)]] RWTexture2D<float2> moments_history_in;  // prev-frame (mu1, mu2), RG32F
[[vk::binding(9,  0)]] RWTexture2D<float2> moments_history_out; // this-frame (mu1, mu2), RG32F
[[vk::binding(10, 0)]] RWTexture2D<float>  variance_in_unused;  // unread by temporal -- host binds a placeholder
[[vk::binding(11, 0)]] RWTexture2D<float>  variance_out;        // this-frame initial variance, R32F

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  reset_history;
    uint  _pad0;
    float depth_tolerance;
    float normal_tolerance;
    float min_alpha;
    float _pad1;
};

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// 2x2 bilinear reprojection with per-tap depth+normal validation.
// Returns the blended color (rgb) + aggregated sample count (a) AND
// the matching moments at the same reprojected position. Both share
// the same validity test, so when we accept a tap for color we also
// accept its moments -- the two trails never desync.
struct ReprojSample {
    float4 color;
    float2 moments;
    bool   valid;
};

ReprojSample sampleHistory(float2 prev_pos, float curr_depth, float3 curr_normal) {
    ReprojSample r;
    r.color = float4(0.0);
    r.moments = float2(0.0);
    r.valid = false;

    int2 base = int2(floor(prev_pos - 0.5));
    float2 frac = (prev_pos - 0.5) - float2(base);
    float w[4] = {
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x         * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x         * frac.y,
    };
    int2 off[4] = { int2(0,0), int2(1,0), int2(0,1), int2(1,1) };

    float4 acc_color = float4(0.0);
    float2 acc_mom   = float2(0.0);
    float  wsum      = 0.0;
    [unroll] for (int i = 0; i < 4; ++i) {
        int2 p = base + off[i];
        if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height)) continue;
        uint2 up = uint2(p);
        float prev_depth = depth_history_in[up];
        float depth_delta = abs(prev_depth - curr_depth) / max(curr_depth, 1e-3);
        if (depth_delta > depth_tolerance) continue;
        float3 prev_n = normal_history_in[up].xyz;
        if (length(curr_normal) > 0.5 && length(prev_n) > 0.5) {
            float n_dot = dot(curr_normal, prev_n);
            if (n_dot < normal_tolerance) continue;
        }
        acc_color += w[i] * color_history_in[up];
        acc_mom   += w[i] * moments_history_in[up];
        wsum      += w[i];
    }
    if (wsum > 1e-3) {
        r.color   = acc_color / wsum;
        r.moments = acc_mom   / wsum;
        r.valid   = true;
    }
    return r;
}

// 7x7 spatial luminance variance estimate. Used to seed the variance
// channel for pixels with too-few temporal samples to trust mu2 - mu1^2
// (Schied 2017 section 4.3). All taps are clamped in-bounds.
float spatialVariance(uint2 tid) {
    float sum  = 0.0;
    float sum2 = 0.0;
    float n    = 0.0;
    [unroll] for (int dy = -3; dy <= 3; ++dy) {
        [unroll] for (int dx = -3; dx <= 3; ++dx) {
            int2 p = int2(tid) + int2(dx, dy);
            if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height)) continue;
            float l = luminance(color_in[uint2(p)].rgb);
            sum  += l;
            sum2 += l * l;
            n    += 1.0;
        }
    }
    if (n < 1.0) return 0.0;
    float mu  = sum  / n;
    float mu2 = sum2 / n;
    return max(0.0, mu2 - mu * mu);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    float4 noisy      = color_in[tid];
    float  curr_depth = depth_tex[tid];
    float3 curr_n     = normal_tex[tid].xyz;
    float  l_now      = luminance(noisy.rgb);

    // Hard reset path: stamp the noisy color, seed moments at (l_now,
    // l_now^2) so initial variance is 0 (1-sample regions defer to the
    // spatial variance fallback below regardless), and write a fresh
    // variance derived purely from the 7x7 spatial neighborhood.
    if (reset_history != 0u) {
        color_out[tid]           = float4(noisy.rgb, 1.0);
        moments_history_out[tid] = float2(l_now, l_now * l_now);
        variance_out[tid]        = spatialVariance(tid);
        return;
    }

    float2 motion = motion_tex[tid];
    // Pixel-center reprojection: G-buffer writes motion = prev_pix - curr_pix
    // where both are pixel-CENTER coords (see PathTrace.slang).
    float2 prev_pos = float2(tid) + 0.5 + motion;
    bool in_frame = prev_pos.x >= 0.0 && prev_pos.y >= 0.0
                  && prev_pos.x < float(width) && prev_pos.y < float(height);

    ReprojSample h;
    h.color = float4(0.0);
    h.moments = float2(0.0);
    h.valid = false;
    if (in_frame) h = sampleHistory(prev_pos, curr_depth, curr_n);

    if (!h.valid) {
        // Disocclusion: drop history. Spatial variance carries us until
        // the temporal moments stabilise over the next ~4 frames.
        color_out[tid]           = float4(noisy.rgb, 1.0);
        moments_history_out[tid] = float2(l_now, l_now * l_now);
        variance_out[tid]        = spatialVariance(tid);
        return;
    }

    // Adaptive alpha: 1/(N+1) for the first few samples after reset,
    // then min_alpha steady-state. Same blend factor drives color and
    // moments so they stay in lockstep (a desynced mu1/color would
    // corrupt variance).
    float n     = clamp(h.color.a, 1.0, 32.0);
    float alpha = max(min_alpha, 1.0 / (n + 1.0));

    float3 blended_color = lerp(h.color.rgb, noisy.rgb, alpha);
    float2 blended_mom   = lerp(h.moments,
                                float2(l_now, l_now * l_now),
                                alpha);

    // NaN/Inf shielding -- HDR fireflies must not poison the long-running
    // mean / second moment.
    if (any(isnan(blended_color)) || any(isinf(blended_color))) blended_color = noisy.rgb;
    if (any(isnan(blended_mom))   || any(isinf(blended_mom)))   blended_mom   = float2(l_now, l_now * l_now);

    float new_count = min(n + 1.0, 32.0);
    color_out[tid]           = float4(blended_color, new_count);
    moments_history_out[tid] = blended_mom;

    // Temporal variance from the running moments.
    float temporal_var = max(0.0, blended_mom.y - blended_mom.x * blended_mom.x);

    // Sample-count-driven blend with the 7x7 spatial fallback. Below 4
    // samples we trust the spatial estimate more than the unsteady
    // temporal one (Schied 2017 4.3). The lerp is on sample count, not
    // alpha, so a low-min_alpha tweak doesn't push us back into spatial
    // territory once converged.
    float final_var;
    if (new_count < 4.0) {
        float spatial_var = spatialVariance(tid);
        float t = new_count / 4.0;
        final_var = lerp(spatial_var, temporal_var, t);
    } else {
        final_var = temporal_var;
    }
    variance_out[tid] = final_var;
}

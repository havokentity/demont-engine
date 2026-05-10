// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SVGF-style temporal accumulation for the Vulkan path tracer.
// Reads the per-frame noisy radiance the path tracer writes into
// `color_in`, reprojects the previous frame's accumulated history
// (color_history_in) using motion vectors, and blends them into
// color_out -- which next frame becomes color_history_in.
//
// Why this is "SVGF-lite": we skip variance estimation and the
// luminance-weighted reprojection rejection, but we keep the two
// pieces that matter for visual quality at 1 spp:
//   1. Temporal blend with adaptive alpha. Disocclusions (motion
//      reprojects off-screen, or the depth/normal of the historical
//      sample disagrees with the current pixel's) collapse alpha to
//      1.0 so we don't ghost stale color into freshly-revealed
//      surfaces. Stable pixels get alpha ~= 0.10 so they pull most
//      of their value from the long-running mean.
//   2. Bilinear filtering of the reprojected history. Motion vectors
//      land on sub-pixel positions; nearest-tap reprojection looks
//      jittery on slow camera moves. 4-tap bilinear reads + per-tap
//      validity check gives stable temporal sub-pixel reconstruction.
//
// History invariant: color_history_in.a stores the running sample
// count (clamped to 32) so the next frame's blend can use 1/(N+1)
// adaptive alpha for the first few samples after a reset (faster
// convergence than the steady-state 0.10).

[[vk::binding(0, 0)]] RWTexture2D<float4> color_in;          // path-tracer output, RGBA16F
[[vk::binding(1, 0)]] RWTexture2D<float4> color_history_in;  // prev-frame accumulated, RGBA16F (.a = sample count)
[[vk::binding(2, 0)]] RWTexture2D<float>  depth_tex;         // this-frame R32F linear camera-space depth
[[vk::binding(3, 0)]] RWTexture2D<float2> motion_tex;        // RG16F pixel-space motion (prev - curr)
[[vk::binding(4, 0)]] RWTexture2D<float4> normal_tex;        // this-frame RGBA16F world-space normal
[[vk::binding(5, 0)]] RWTexture2D<float4> color_out;         // this-frame accumulated, RGBA16F
[[vk::binding(6, 0)]] RWTexture2D<float>  depth_history_in;  // prev-frame R32F depth history (reproject sample source)
[[vk::binding(7, 0)]] RWTexture2D<float4> normal_history_in; // prev-frame RGBA16F normal history (reproject sample source)

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  reset_history;     // 1 = ignore history this frame
    uint  _pad0;
    float depth_tolerance;   // relative depth difference allowed for reproj
    float normal_tolerance;  // dot-product threshold (1.0 = exact, 0.0 = any)
    float min_alpha;         // steady-state blend factor (e.g. 0.10)
    float _pad1;
};

// Sample a 2x2 bilinear neighbourhood from color_history_in at the
// reprojected position. Each tap is validated against the current
// pixel's depth + normal; failing taps drop out of the average. If
// no taps pass we return alpha=1 to disable temporal reuse.
struct ReprojSample {
    float4 color;       // .rgb = blended color, .a = aggregated sample count
    bool   valid;
};

ReprojSample sampleHistory(float2 prev_pos, float curr_depth, float3 curr_normal) {
    ReprojSample r;
    r.color = float4(0.0);
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

    float4 acc = float4(0.0);
    float  wsum = 0.0;
    // Disocclusion test:
    // Compare this frame's depth/normal at curr_pos against previous
    // frame's depth/normal history sampled at prev_pos (true cross-
    // frame validation, matching SVGF's intended gating).
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
        acc  += w[i] * color_history_in[up];
        wsum += w[i];
    }
    if (wsum > 1e-3) {
        r.color = acc / wsum;
        r.valid = true;
    }
    return r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    float4 noisy = color_in[tid];
    float  curr_depth = depth_tex[tid];
    float3 curr_n = normal_tex[tid].xyz;

    // Reset path: just stamp the current frame and seed the history
    // counter. Avoids a single-frame ghost from undefined memory in
    // color_history_in on first frame after backend init.
    if (reset_history != 0u) {
        color_out[tid] = float4(noisy.rgb, 1.0);
        return;
    }

    float2 motion = motion_tex[tid];
    float2 prev_pos = float2(tid) + motion;
    bool in_frame = prev_pos.x >= 0.0 && prev_pos.y >= 0.0
                  && prev_pos.x < float(width) && prev_pos.y < float(height);

    ReprojSample h;
    h.color = float4(0.0);
    h.valid = false;
    if (in_frame) h = sampleHistory(prev_pos, curr_depth, curr_n);

    if (!h.valid) {
        color_out[tid] = float4(noisy.rgb, 1.0);
        return;
    }

    // Adaptive alpha: 1/(N+1) for the first few samples after a
    // reset (faster convergence), clamped to min_alpha for steady
    // state. Sample count lives in history.a; cap at 32 so a long-
    // stationary scene can still adapt to lighting changes.
    float n = clamp(h.color.a, 1.0, 32.0);
    float alpha = max(min_alpha, 1.0 / (n + 1.0));

    float3 blended = lerp(h.color.rgb, noisy.rgb, alpha);
    // Sanitize: NaNs / Infs from the path tracer's HDR pixels would
    // poison the running mean forever. Drop them.
    if (any(isnan(blended)) || any(isinf(blended))) blended = noisy.rgb;
    color_out[tid] = float4(blended, min(n + 1.0, 32.0));
}

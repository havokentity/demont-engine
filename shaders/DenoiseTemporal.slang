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

// vk::image_format qualifiers below match each binding's engine-side
// VkImage format. See PathTrace.slang's top-level comment for the
// NVIDIA Format=Unknown bug rationale.
[[vk::binding(0,  0), vk::image_format("rgba16f")]] RWTexture2D<float4> color_in;            // path-tracer noisy radiance, RGBA16F
[[vk::binding(1,  0), vk::image_format("rgba16f")]] RWTexture2D<float4> color_history_in;    // prev-frame accumulated color, RGBA16F
[[vk::binding(2,  0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;           // current-frame linear depth, R32F
[[vk::binding(3,  0), vk::image_format("rg16f")]]   RWTexture2D<float2> motion_tex;          // pixel-space motion (prev - curr), RG16F
[[vk::binding(4,  0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_tex;          // current-frame world-space normal, RGBA16F
[[vk::binding(5,  0), vk::image_format("rgba16f")]] RWTexture2D<float4> color_out;           // accumulated color, RGBA16F (.a = sample count)
[[vk::binding(6,  0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_history_in;    // prev-frame depth, R32F
[[vk::binding(7,  0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_history_in;   // prev-frame normal, RGBA16F
// Buffer-backed bindings start at 8. Storage *buffers* don't share Metal's
// 8-texture-with-read_write-access compute limit (we keep 8 RW textures at
// the cap; 12 RW textures crashed slangc's MSL output). Indexing is
// row-major: idx = y*width + x.
[[vk::binding(8,  0)]] RWStructuredBuffer<float2> moments_history_in;  // prev-frame (mu1, mu2)
[[vk::binding(9,  0)]] RWStructuredBuffer<float2> moments_history_out; // this-frame (mu1, mu2)
[[vk::binding(10, 0)]] RWStructuredBuffer<float>  variance_in_unused;  // unread by temporal -- host binds a placeholder
[[vk::binding(11, 0)]] RWStructuredBuffer<float>  variance_out;        // this-frame initial variance
// Albedo G-buffer (binding 12). Issue #119: SVGF albedo demodulation.
// On Vulkan slot 12 is bound as STORAGE_IMAGE (the underlying VkImage
// has STORAGE_BIT only, never SAMPLED_BIT), so SPIR-V wants
// Sampled=2 i.e. `RWTexture2D`. On Metal we keep `Texture2D` so the
// MSL emit is `access::sample` (read-only), which sidesteps Metal's
// 8-RW-texture compute cap -- the temporal pass already pins 8 RW
// textures at slots 0..7 and a 9th `RWTexture2D` would trip slangc's
// MSL output. The host binds this to the engine's albedo_tex (=
// primary-hit surface albedo) whenever r_svgf_albedo_demod is on;
// when off the binding is still wired so the descriptor set stays
// well-formed, but the shader skips every divide / multiply via the
// runtime `demod_enabled` push gate. Sky pixels (albedo == 0)
// bypass the divide entirely and forward emissive radiance.
#ifdef PT_TARGET_SPIRV
[[vk::binding(12, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> albedo_tex;
#else
[[vk::binding(12, 0)]] Texture2D<float4>   albedo_tex;
#endif

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
    // Issue #119 -- albedo demodulation gate. 1 = divide color samples
    // by albedo on input (keeps the temporal pass operating on the
    // "lighting only" signal, which is what SVGF's edge-stops were
    // designed to denoise -- texture detail flows through the chain
    // demodulated and gets restored at the final output stage). 0 =
    // legacy pre-demod behaviour (full radiance through the chain;
    // albedo binding still bound but never read). MVP keeps the
    // demodulated signal demodulated all the way to the OUTPUT write;
    // the engine adds a separate remod kernel after this pass for
    // svgf_basic, and the atrous chain remod's on its final pass.
    uint  demod_enabled;
    uint  _pad2;
    uint  _pad3;
    uint  _pad4;
};

// Floor on the per-channel albedo divisor. Picked so that:
//   - Near-black surfaces (5% reflectance) keep their lighting in a
//     numerically reasonable range without ballooning the variance
//     denominator past what the A-Trous luminance gate expects.
// SVGF papers typically use 1e-2 to 1e-3; we use 1e-2 which trades a
// touch of low-reflectance fidelity for stability on the running
// variance. Note: sky pixels (albedo == 0 sentinel) bypass demod
// entirely; see `isSkyAlbedo` below.
static const float kDemodEps = 0.01;

// Pixels with this much total albedo or less are treated as "no
// surface" (sky / miss). The path tracer writes float3(0) into
// albedo_tex at primary miss, so the cutoff just needs to be larger
// than fp16 zero noise but smaller than any plausible material
// albedo (PBR diffuse materials are clamped to >= 0.04 by physical
// constraint).
static const float kAlbedoSkyEps = 1.0 / 65536.0;

bool isSkyAlbedo(float3 a) {
    return (a.r + a.g + a.b) <= kAlbedoSkyEps;
}

// Read color_in[p] in (optionally) demodulated space. When demod is
// on, divides by max(albedo_tex[p], kDemodEps) per channel so the
// temporal blend operates on incident lighting rather than reflected
// radiance. Sky pixels (albedo == 0) bypass demod -- the path
// tracer's sky / celestials terms are emissive radiance, not
// reflected radiance, so dividing by anything would corrupt the
// signal and the remod multiply (by 0) would zero the sky entirely.
// Returning color_in[p].rgb unchanged for those pixels keeps the
// final remod a no-op for them: the matching multiply-back site
// also bypasses sky pixels.
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

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

uint pixelIndex(uint2 p) { return p.y * width + p.x; }

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
        acc_mom   += w[i] * moments_history_in[pixelIndex(up)];
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
// (Schied 2017 section 4.3). All taps are clamped in-bounds. Reads
// the demodulated signal when `demod_enabled` is set so the variance
// represents lighting variance, not material-color variance (the
// latter would dominate the gate at multi-coloured texture boundaries
// and re-introduce the bleed demod is supposed to eliminate).
float spatialVariance(uint2 tid) {
    float sum  = 0.0;
    float sum2 = 0.0;
    float n    = 0.0;
    [unroll] for (int dy = -3; dy <= 3; ++dy) {
        [unroll] for (int dx = -3; dx <= 3; ++dx) {
            int2 p = int2(tid) + int2(dx, dy);
            if (p.x < 0 || p.y < 0 || p.x >= int(width) || p.y >= int(height)) continue;
            float l = luminance(readDemodColor(uint2(p)));
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

    // Read noisy radiance in (optionally) demodulated space. With
    // demod on, `noisy_rgb` is the divided-by-albedo "incident
    // lighting" estimate; everything downstream (history blend, moment
    // update, variance) operates on this signal. The multiply-back to
    // restore textured radiance happens in a separate remod kernel
    // (svgf_basic) or the final A-Trous pass (svgf_atrous).
    float3 noisy_rgb  = readDemodColor(tid);
    float  curr_depth = depth_tex[tid];
    float3 curr_n     = normal_tex[tid].xyz;
    float  l_now      = luminance(noisy_rgb);

    // Hard reset path: stamp the noisy (demodulated) color, seed
    // moments at (l_now, l_now^2) so initial variance is 0 (1-sample
    // regions defer to the spatial variance fallback below regardless),
    // and write a fresh variance derived purely from the 7x7 spatial
    // neighborhood. The output stays in demod space -- the remod
    // kernel (or final A-Trous pass) multiplies albedo back on the
    // way to `output`.
    if (reset_history != 0u) {
        color_out[tid]                       = float4(noisy_rgb, 1.0);
        moments_history_out[pixelIndex(tid)] = float2(l_now, l_now * l_now);
        variance_out[pixelIndex(tid)]        = spatialVariance(tid);
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
        color_out[tid]                       = float4(noisy_rgb, 1.0);
        moments_history_out[pixelIndex(tid)] = float2(l_now, l_now * l_now);
        variance_out[pixelIndex(tid)]        = spatialVariance(tid);
        return;
    }

    // Adaptive alpha: 1/(N+1) for the first few samples after reset,
    // then min_alpha steady-state. Same blend factor drives color and
    // moments so they stay in lockstep (a desynced mu1/color would
    // corrupt variance). Both history and current frame are in the
    // demodulated signal space when demod is on -- the divide happens
    // at every shader entry into the chain so the cross-frame blend
    // composes correctly.
    float n     = clamp(h.color.a, 1.0, 32.0);
    float alpha = max(min_alpha, 1.0 / (n + 1.0));

    float3 blended_color = lerp(h.color.rgb, noisy_rgb, alpha);
    float2 blended_mom   = lerp(h.moments,
                                float2(l_now, l_now * l_now),
                                alpha);

    // NaN/Inf shielding -- HDR fireflies must not poison the long-running
    // mean / second moment. Demodulation can blow up to ~100x the input
    // on near-black albedo (1/eps = 100 with kDemodEps = 0.01), so this
    // gate is more load-bearing under demod than it was under straight
    // radiance.
    if (any(isnan(blended_color)) || any(isinf(blended_color))) blended_color = noisy_rgb;
    if (any(isnan(blended_mom))   || any(isinf(blended_mom)))   blended_mom   = float2(l_now, l_now * l_now);

    float new_count = min(n + 1.0, 32.0);
    color_out[tid]                       = float4(blended_color, new_count);
    moments_history_out[pixelIndex(tid)] = blended_mom;

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
    variance_out[pixelIndex(tid)] = final_var;
}

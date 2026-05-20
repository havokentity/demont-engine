// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Composite the procedural-raymarched cloud pre-pass output
// (clouds_color_tex) into the path-traced HDR (hdr_inout). Runs only
// when r_clouds_mode == procedural_raymarched.
//
// Math: pre-multiplied alpha blend.
//
//   hdr_inout = clouds_color.rgb + (1.0 - clouds_color.a) * hdr_inout
//
// When clouds.a == 0 (no clouds along this ray) the HDR is unchanged.
// When clouds.a == 1 (fully opaque cloud) the HDR is replaced with
// clouds.rgb. Anything in between is a smooth alpha blend.
//
// Depth gate (sky-only): clouds appear only where the path-traced
// scene didn't hit a foreground object. The G-buffer depth_tex carries
// 1000.0 (sentinel) at sky pixels, the actual primary-hit distance
// otherwise. Without this gate the cloud composite would paint over
// the ground geometry at glancing camera angles where the cloud-slab
// intersect picks up world-space samples below the cloud base.
//
// Sub-resolution upsample (Wave 8 #28): when clouds_color_tex was
// rendered at reduced resolution (engine cvar
// r_clouds_raymarched_resolution_scale < 1.0), the composite reconstructs
// the full-res cloud value with a DEPTH-BILATERAL upsample. It samples the
// 2x2 low-res cloud texels surrounding the full-res pixel's source
// coordinate and blends them with weights = (bilinear spatial weight) x
// (depth-similarity weight). The depth guide is the full-res scene depth
// sampled at each low-res tap's source pixel: where a tap straddles a
// geometry silhouette its depth diverges sharply from the target pixel's
// depth and the tap is suppressed, so cloud edges against foreground
// geometry stay crisp instead of bleeding across the silhouette (the
// classic nearest-depth artefact of a plain bilinear upsample). On a
// pure-sky neighbourhood every tap has the same (sentinel) depth so the
// filter degrades gracefully to plain bilinear. At scale == 1.0 the
// upsample is skipped entirely (the cloud texture is already full-res).

[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_inout;
[[vk::binding(1, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> clouds_color_tex;
[[vk::binding(2, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;

[[vk::push_constant]]
cbuffer Push {
    // .x = composite_active (0 -> early-out, kernel is a no-op).
    // .y = sub-resolution scale used by the raymarch pass (1.0 = full,
    //      0.5 = half). At scale < 1 the composite reconstructs the
    //      full-res cloud value via a depth-bilateral upsample of the
    //      low-res clouds_color_tex (Wave 8 #28); at 1.0 the upsample is
    //      skipped (clouds_color_tex was already full-res).
    // .z reserved
    // .w reserved
    float4 composite_params;
};

// Depth-bilateral upsample of the low-res cloud texture (Wave 8 #28).
// `pix` is the full-res pixel; `scale` the render-resolution scale (<1);
// `dim` the full-res extent; `target_depth` the scene depth at `pix`.
// Returns the reconstructed pre-multiplied cloud value (rgb = in-scatter,
// a = coverage).
//
// Low-res texel (cx,cy) was written by the pre-pass for the full-res
// block centred at full-res pixel ((cx+0.5)/scale). Inverting that, the
// low-res source coordinate for full-res pixel `pix` is
//     src = (pix + 0.5) * scale - 0.5.
// floor(src) + {0,1} give the surrounding 2x2 low-res texels; frac(src)
// is the bilinear blend factor.
float4 bilateralUpsampleClouds(uint2 pix, float scale, uint2 dim,
                               float target_depth) {
    float2 src = (float2(pix) + 0.5) * scale - 0.5;
    float2 base = floor(src);
    float2 fr   = src - base;            // bilinear weights in [0,1]
    // Low-res grid extent (matches the pre-pass dispatch / write region).
    int2 ldim = int2(max(1, int(float(dim.x) * scale + 0.5)),
                     max(1, int(float(dim.y) * scale + 0.5)));
    // Depth-similarity falloff. depth in metres; 1/sigma chosen so a
    // ~10% relative depth gap roughly halves the weight. Scene-scale
    // independent because we normalise by the target depth.
    float4 acc = float4(0.0);
    float  wsum = 0.0;
    [unroll] for (int dy = 0; dy < 2; ++dy) {
        [unroll] for (int dx = 0; dx < 2; ++dx) {
            int2 c = int2(base) + int2(dx, dy);
            c = clamp(c, int2(0, 0), ldim - 1);
            // Spatial bilinear weight for this corner.
            float wx = (dx == 0) ? (1.0 - fr.x) : fr.x;
            float wy = (dy == 0) ? (1.0 - fr.y) : fr.y;
            float w_spatial = wx * wy;
            // Full-res source pixel this low-res texel represents, so we
            // can read the matching scene depth for the bilateral guide.
            int2 sp = int2(float2(c) / scale + 0.5 / scale);
            sp = clamp(sp, int2(0, 0), int2(dim) - 1);
            float d = depth_tex[uint2(sp)];
            // Relative depth difference. Sky (sentinel ~1e10) vs sky -> 0;
            // sky vs foreground -> large -> tap suppressed. Guard the
            // divide against a zero target depth.
            float rel = abs(d - target_depth) / max(target_depth, 1.0);
            float w_depth = exp(-rel * rel * 8.0);
            float w = w_spatial * w_depth;
            acc  += clouds_color_tex[uint2(c)] * w;
            wsum += w;
        }
    }
    // Fallback: if every tap was suppressed (pathological depth field),
    // use the nearest texel so we never divide by ~0.
    if (wsum < 1e-5) {
        int2 c = clamp(int2(src + 0.5), int2(0, 0), ldim - 1);
        return clouds_color_tex[uint2(c)];
    }
    return acc / wsum;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (composite_params.x < 0.5) return;

    float scale = clamp(composite_params.y, 0.05, 1.0);
    float depth = depth_tex[tid.xy];

    float4 clouds;
    if (scale >= 0.999) {
        // Full-res: the cloud texture is 1:1 with the screen, sample
        // directly (the depth-bilateral path is a no-op here anyway).
        clouds = clouds_color_tex[tid.xy];
    } else {
        // Sub-res: depth-bilateral reconstruction so cloud silhouettes
        // against foreground geometry stay sharp (Wave 8 #28).
        clouds = bilateralUpsampleClouds(tid.xy, scale, dim, depth);
    }
    if (clouds.a <= 0.001) return;     // no cloud contribution at this pixel

    // Sky-only composite. The path tracer writes a large sentinel to
    // depth_tex at primary-miss pixels (sky); foreground hits write the
    // actual distance. We want clouds to only blend where the sky is
    // visible (otherwise foreground objects would get translucent cloud
    // rgb tints layered on them). Use a soft threshold so anti-aliased
    // silhouettes still get partial cloud cover -- 500m is the practical
    // limit where the cloud field can plausibly sit behind ANY scene
    // geometry on a typical scene (lights, terrain). `depth` was read at
    // the top of main() for the bilateral guide; reuse it here.
    if (depth < 500.0) return;

    float4 hdr  = hdr_inout[tid.xy];
    float3 out_rgb = clouds.rgb + (1.0 - clouds.a) * hdr.rgb;
    hdr_inout[tid.xy] = float4(out_rgb, 1.0);
}

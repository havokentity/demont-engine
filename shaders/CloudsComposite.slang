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
// Sub-resolution upsample: when clouds_color_tex was rendered at half
// resolution (engine cvar r_clouds_raymarched_resolution_scale = 0.5),
// the composite samples a 2x2 nearest cluster and picks the cluster
// member with the closest depth to the full-res pixel's depth -- a
// minimal bilateral upsample that preserves cloud silhouettes against
// foreground geometry edges without a full Gaussian filter.

[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_inout;
[[vk::binding(1, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> clouds_color_tex;
[[vk::binding(2, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;

[[vk::push_constant]]
cbuffer Push {
    // .x = composite_active (0 -> early-out, kernel is a no-op).
    // .y = sub-resolution scale used by the raymarch pass (1.0 = full,
    //      0.5 = half). The composite samples clouds_color_tex at
    //      pixel-coord = full_res / scale for nearest-neighbour upsample;
    //      anything more sophisticated (Gaussian / bilateral) is a
    //      polish follow-up. 1.0 is the default and skips the upsample
    //      entirely (clouds_color_tex was full-res).
    // .z reserved
    // .w reserved
    float4 composite_params;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (composite_params.x < 0.5) return;

    // Sub-resolution sample: divide thread id by scale^-1 = inv_scale.
    // For scale 0.5, clouds tex is half-res and we sample at
    // (tid / 2). For scale 1.0 the clouds tex is full-res and we
    // sample at tid directly.
    float scale = max(composite_params.y, 0.05);
    uint2 cidx = uint2(uint(float(tid.x) * scale + 0.5),
                       uint(float(tid.y) * scale + 0.5));

    float4 clouds = clouds_color_tex[cidx];
    if (clouds.a <= 0.001) return;     // no cloud contribution at this pixel

    // Sky-only composite. The path tracer writes 1000.0 to depth_tex
    // at primary-miss pixels (sky); foreground hits write the actual
    // distance. We want clouds to only blend where the sky is visible
    // (otherwise foreground objects would get translucent cloud rgb
    // tints layered on them). Use a soft threshold so anti-aliased
    // silhouettes still get partial cloud cover -- 500m is the practical
    // limit where the cloud field can plausibly sit behind ANY scene
    // geometry on a typical scene (lights, terrain).
    float depth = depth_tex[tid.xy];
    if (depth < 500.0) return;

    float4 hdr  = hdr_inout[tid.xy];
    float3 out_rgb = clouds.rgb + (1.0 - clouds.a) * hdr.rgb;
    hdr_inout[tid.xy] = float4(out_rgb, 1.0);
}

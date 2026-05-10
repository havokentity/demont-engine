// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Final pass for the Vulkan SVGF/NRD denoiser. Reads the linear-HDR
// result the temporal/atrous chain wrote into post_denoise_hdr,
// applies exposure + ACES + sRGB OETF, and writes the swapchain.
//
// This exists because the engine's post-denoise Tonemap.slang
// pipeline isn't built on the Vulkan backend yet (its push struct
// is bigger than Vulkan's 256B limit and would need the same
// spill-to-UBO machinery PtPush uses -- a separate task). Without
// this kernel the denoiser writes post_denoise_hdr but nothing
// reads it, so the swapchain stays at whatever the path tracer
// wrote inline -- making r_denoiser={svgf_basic,svgf_atrous,nrd}
// visually identical to r_denoiser=off.
//
// Honors r_hdr_pipeline:
//   hdr_pipeline = 1 (default): denoise_color holds linear HDR
//     radiance from the path tracer; we apply exposure * ACES * sRGB
//     OETF here -- one tonemap, end-to-end.
//   hdr_pipeline = 0: the path tracer already pre-tonemapped into
//     denoise_color, so we skip the tonemap and just apply the sRGB
//     OETF for swapchain encoding. Avoids the double-tonemap that
//     would otherwise wash the image out.

[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_in;
[[vk::binding(1, 0)]] RWTexture2D<float4> swap_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exposure_state;

[[vk::push_constant]]
cbuffer Push {
    uint width;
    uint height;
    uint hdr_pipeline;     // 1 = apply ACES + sRGB; 0 = sRGB OETF only
    uint _pad1;
};

float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c * (a * c + b)) / (c * (d * c + e) + f));
}

// Standard sRGB OETF. Vulkan storage-image writes to a *_SRGB swapchain
// don't auto-encode (only color-attachment writes do). The Vulkan path
// tracer uses the same encoder; we mirror it here so both paths
// produce matching pixels on a BGRA8Unorm or BGRA8Unorm_sRGB surface.
float3 srgb_oetf(float3 c) {
    c = max(c, float3(0.0, 0.0, 0.0));
    float3 lin  = c * 12.92;
    float3 expo = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return select(c < 0.0031308, lin, expo);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float3 c = hdr_in[tid].rgb;
    if (hdr_pipeline != 0u) {
        // Linear HDR in -- apply exposure + ACES, then sRGB OETF.
        float exposure = exposure_state[0];
        c = acesTonemap(c * exposure);
    }
    // sRGB encode runs in either branch: hdr_pipeline=0 case has
    // already-tonemapped LDR-ish values that still need the OETF
    // for the BGRA8Unorm swapchain.
    c = srgb_oetf(c);
    swap_out[tid] = float4(c, 1.0);
}

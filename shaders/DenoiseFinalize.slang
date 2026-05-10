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
// Quality note: this kernel always applies ACES + sRGB. It assumes
// the path tracer fed denoise_color the raw linear HDR radiance
// (i.e. r_hdr_pipeline = 1, the default). With r_hdr_pipeline = 0
// the path tracer pre-applies tonemap to denoise_color and this
// kernel would double-tonemap -- a visibly washed-out image. The
// engine warns on that combo. When the proper Tonemap pipeline
// lands on Vulkan it'll honor r_hdr_pipeline correctly.

[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_in;
[[vk::binding(1, 0)]] RWTexture2D<float4> swap_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exposure_state;

[[vk::push_constant]]
cbuffer Push {
    uint width;
    uint height;
    uint _pad0;
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
    float3 hdr = hdr_in[tid].rgb;
    float exposure = exposure_state[0];
    float3 ldr = acesTonemap(hdr * exposure);
    ldr = srgb_oetf(ldr);
    swap_out[tid] = float4(ldr, 1.0);
}

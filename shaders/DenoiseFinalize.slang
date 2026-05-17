// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Final pass for the Vulkan SVGF/NRD denoiser. Reads the linear-HDR
// result the temporal/atrous chain wrote into post_denoise_hdr,
// optionally composites the bloom layer pre-tonemap, applies
// exposure + ACES + sRGB OETF, and writes the swapchain.
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
// Bloom on Vulkan: Tonemap.slang IS built on the SPIR-V path (see
// VulkanDevice.cpp's pipeline build block) but Engine.cpp gates
// `use_engine_tonemap` to Metal only, so DenoiseFinalize is what
// composites bloom on Vulkan today -- the bloom-pyramid kernels
// (BloomDown / BloomUp) only need 2 storage-image bindings each and
// run cleanly on the SPIR-V path. We sample mip[0] with bilinear and
// add it pre-ACES so it gets the same curve squash as the rest of
// the image, matching Metal's Tonemap.slang behaviour. Disabled-
// bloom contract: VulkanNrdDenoiser binds color_in_view as a
// safe-but-unread fallback at binding 3 and pushes
// bloom_intensity = 0 so the `if (bloom_intensity > 0.0)` gate
// short-circuits the sample. No 1x1 placeholder texture is
// involved on this path.
//
// Honors r_hdr_pipeline:
//   hdr_pipeline = 1 (default): denoise_color holds linear HDR
//     radiance from the path tracer; we apply exposure * ACES * sRGB
//     OETF here -- one tonemap, end-to-end. Bloom is added pre-ACES.
//   hdr_pipeline = 0: the path tracer already pre-tonemapped into
//     denoise_color, so we skip the tonemap and just apply the sRGB
//     OETF for swapchain encoding. Bloom is added pre-OETF.

[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_in;
[[vk::binding(1, 0)]] RWTexture2D<float4> swap_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exposure_state;
// Bloom mip 0 (half-res linear HDR). When Vulkan bloom is disabled / not
// built, the host binds a safe valid fallback view here and pushes
// bloom_intensity = 0, so the shader's bloom path short-circuits before
// any sample becomes observable.
[[vk::binding(3, 0)]] RWTexture2D<float4> bloom_tex;
// Star-split accumulator (issue #46). Same role as Tonemap.slang's
// stars_tex on Metal: PathTrace.slang accumulates the per-pixel star
// contribution here, bypassing the SVGF a-trous kernel that would
// otherwise smudge sub-pixel star energy into surrounding sky texels.
// Added pre-bloom-composite / pre-ACES so the stars land in the
// final HDR pre-tonemap and ACES-squash uniformly with the rest of
// the image. When star-split is off (cvar r_star_split 0 or the
// accumulator isn't allocated) the host binds a 1x1 placeholder
// texture here purely for descriptor-set validity; the shader's
// `stars_present` push gate ensures this slot is never sampled in
// that case, so the placeholder's contents don't matter.
[[vk::binding(4, 0)]] RWTexture2D<float4> stars_tex;

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  hdr_pipeline;     // 1 = apply ACES + sRGB; 0 = sRGB OETF only
    float bloom_intensity;  // 0 = skip bloom add entirely
    // Star-split present flag (issue #46). 1 -> add stars_tex to the
    // HDR colour pre-bloom; 0 -> skip the additive read entirely
    // (the slot is bound to a 1x1 placeholder when stars aren't
    // routed). Mirrors VulkanNrdDenoiser::FinalizePush at the same
    // byte offset.
    uint  stars_present;
    // 12 bytes of explicit pad so the push block size is a multiple
    // of 16, matching the std140 cbuffer rule.
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
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

// Bilinear sample of bloom_tex at the screen UV. Same code shape as
// Tonemap.slang's bloom mix on Metal -- mip[0] is half-res so we map
// (screen_pixel_centre / screen_dim * mip_dim) and 4-tap interpolate.
// Out-of-range UVs clamp to edges.
float3 sampleBloom(uint2 tid, uint2 dim) {
    uint2 bdim;
    bloom_tex.GetDimensions(bdim.x, bdim.y);
    int2  bd = int2(bdim);
    float2 fb = (float2(tid) + 0.5) * float2(bdim) / float2(dim) - 0.5;
    int2  i0  = int2(floor(fb));
    float2 t  = frac(fb);
    int2 p00 = clamp(i0,                  int2(0,0), bd - int2(1,1));
    int2 p10 = clamp(i0 + int2(1,0),      int2(0,0), bd - int2(1,1));
    int2 p01 = clamp(i0 + int2(0,1),      int2(0,0), bd - int2(1,1));
    int2 p11 = clamp(i0 + int2(1,1),      int2(0,0), bd - int2(1,1));
    float3 b00 = bloom_tex[uint2(p00)].rgb;
    float3 b10 = bloom_tex[uint2(p10)].rgb;
    float3 b01 = bloom_tex[uint2(p01)].rgb;
    float3 b11 = bloom_tex[uint2(p11)].rgb;
    float3 b0  = lerp(b00, b10, t.x);
    float3 b1  = lerp(b01, b11, t.x);
    return lerp(b0, b1, t.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float3 c = hdr_in[tid].rgb;

    // Star-split composite (issue #46). Stars stored in stars_tex
    // bypass the SVGF a-trous kernel; add them back here pre-ACES so
    // they land in the final image without the smudge artefacts the
    // wavelet pass would otherwise inflict. Gated on `stars_present`
    // so the read is elided when the engine isn't routing stars
    // (cvar r_star_split = 0 or no accumulator) -- the slot is bound
    // to a 1x1 placeholder in that case, purely for descriptor-set
    // validity.
    if (stars_present != 0u) {
        c += stars_tex[tid].rgb;
    }

    // Bloom composite. Pre-tonemap so highlights get the same ACES
    // squash as the rest of the image. bloom_intensity = 0 short-
    // circuits the bilinear sample.
    if (bloom_intensity > 0.0) {
        c += bloom_intensity * sampleBloom(tid, uint2(width, height));
    }

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

// Post-denoise tonemap.
//
// Reads a per-pixel linear HDR colour (the MetalFX output, RGBA16F)
// and writes the ACES-tonemapped sRGB result to the swapchain. Sits
// between MetalFX and present in the denoiser-on path; the
// denoiser-off path bypasses this kernel and tonemaps inline in
// PathTrace.slang's swapchain write (the tradeoff: a one-pass write
// when not denoising, two passes when denoising in HDR).
//
// Bindings (Slang assigns Metal texture(0..1) in declaration order):
//   slot 0 (texture) : hdr_in -- RGBA16F linear HDR from MetalFX
//   slot 1 (texture) : ldr_out -- swapchain (BGRA8Unorm_sRGB) -- the
//                       linear-to-sRGB encode is implicit on store.

[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_in;
[[vk::binding(1, 0)]] RWTexture2D<float4> ldr_out;
// Half-res linear HDR bloom layer. The engine builds this each frame
// via a downsample/upsample pyramid (BloomDown.slang + BloomUp.slang)
// from the bright pixels of hdr_in. Sampled here with bilinear and
// added pre-tonemap so it gets ACES-curve-compressed alongside the
// rest of the image. Always bound; the engine zero-fills it (or
// passes a 1x1 dummy) when r_bloom = 0 and bloom_intensity = 0.
[[vk::binding(2, 0)]] RWTexture2D<float4> bloom_tex;

[[vk::push_constant]]
cbuffer Push {
    float exposure;
    // 1 = passthrough copy (the path tracer already applied exposure +
    // ACES). 0 = apply exposure + ACES here. The engine drives this
    // off r_hdr_pipeline so users can A/B the two paths.
    uint  passthrough;
    // Linear scale on the bloom layer added pre-tonemap. 0 = no
    // bloom (skip the sample entirely so a dummy 1x1 binding works).
    float bloom_intensity;
    float pad;
};

float3 acesTonemap(float3 c) {
    // Narkowicz fit. Same constants as PathTrace.slang -- duplicated
    // rather than shared so this kernel stays a single self-contained
    // TU. Drift between the two would manifest as a visible diff
    // between denoiser-on and denoiser-off, so keep the magic numbers
    // identical to the path-tracer copy.
    const float a = 2.51, b = 0.03, ck = 2.43, d = 0.59, e = 0.14;
    return saturate((c * (a * c + b)) / (c * (ck * c + d) + e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    ldr_out.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;

    float3 c = hdr_in[tid.xy].rgb;

    // Add bloom (linear, pre-tonemap) so highlights get the same
    // ACES curve squash as the rest of the image. Bilinear sample
    // from the half-res mip-0 of the bloom pyramid; the engine
    // zero-fills the texture when r_bloom is off so this branch
    // costs ~4 texel reads either way -- skip them entirely when
    // intensity is zero to keep the no-bloom path branch-free.
    if (bloom_intensity > 0.0) {
        uint2 bdim;
        bloom_tex.GetDimensions(bdim.x, bdim.y);
        float2 fb = (float2(tid.xy) + 0.5) * float2(bdim) / float2(dim) - 0.5;
        int2  i0  = int2(floor(fb));
        float2 t  = frac(fb);
        int2  bd  = int2(bdim);
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
        c += bloom_intensity * lerp(b0, b1, t.y);
    }

    float3 ldr = (passthrough != 0u) ? c : acesTonemap(c * exposure);
    ldr_out[tid.xy] = float4(ldr, 1.0);
}

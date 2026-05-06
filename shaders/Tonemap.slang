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

[[vk::push_constant]]
cbuffer Push {
    float exposure;
    // 1 = passthrough copy (the path tracer already applied exposure +
    // ACES). 0 = apply exposure + ACES here. The engine drives this
    // off r_hdr_pipeline so users can A/B the two paths.
    uint  passthrough;
    float pad[2];
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
    float3 ldr = (passthrough != 0u) ? c : acesTonemap(c * exposure);
    ldr_out[tid.xy] = float4(ldr, 1.0);
}

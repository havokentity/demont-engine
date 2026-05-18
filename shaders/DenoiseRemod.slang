// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Issue #119 -- SVGF albedo remodulation kernel.
//
// The DenoiseTemporal + DenoiseAtrous chain operates on the
// demodulated lighting signal (radiance / albedo) when
// r_svgf_albedo_demod is on, so SVGF's depth/normal/luminance
// edge-stops don't fight texture detail. The CHAIN'S OUTPUT is still
// demodulated -- the temporal pass writes its color_out as the
// SVGF feedback target (next frame's history), and the chain's last
// touchable image is that feedback target in svgf_basic (atrous is
// skipped) or in svgf_atrous with n_passes == 1 (the single A-Trous
// pass also targets the feedback texture for Schied feedback).
//
// In those cases the kernel below is dispatched between the SVGF
// chain and `output`: it reads the demodulated lighting from the
// feedback texture, multiplies by the centre pixel's primary-hit
// albedo, and writes the textured radiance into `output` for the
// downstream bloom + tonemap chain.
//
// For svgf_atrous with n_passes >= 2 the LAST A-Trous pass already
// handles the multiply-back via its `final_remod` push flag, so the
// host skips this kernel.
//
// Why a fresh kernel rather than overloading DenoiseTemporal:
// temporal's color_out IS the feedback target -- we can't write a
// remodulated value to it without corrupting next frame's reprojection
// (the temporal pass requires feedback in the demod space). A separate
// kernel cleanly separates "history that next frame divides" from
// "radiance the rest of the engine consumes."

// vk::image_format("rgba16f") on both: demod_in and color_out are
// both RGBA16F engine textures. See PathTrace.slang for the NVIDIA
// Format=Unknown rationale.
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> demod_in;     // SVGF chain's last write, demodulated lighting (RGBA16F)
[[vk::binding(1, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> color_out;    // textured radiance for bloom + tonemap (RGBA16F)
// Read-only albedo binding. Vulkan binds slot 12 as STORAGE_IMAGE
// (the texture has STORAGE_BIT only), so SPIR-V wants `RWTexture2D`
// (Sampled=2). On Metal we keep `Texture2D` so the MSL emit is
// `access::sample`, which doesn't count against the 8-RW-texture
// per-kernel cap. Same slot DenoiseTemporal / DenoiseAtrous use, so
// the engine plumbs the same texture handle into all three passes
// without divergent layouts.
#ifdef PT_TARGET_SPIRV
[[vk::binding(12, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> albedo_tex;
#else
[[vk::binding(12, 0)]] Texture2D<float4>   albedo_tex;
#endif

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  _pad0;
    uint  _pad1;
};

// Sky / no-surface cutoff. Pixels with albedo this small or smaller
// are emissive (path tracer wrote float3(0) on miss), so their
// upstream "demodulated" value is in fact the raw radiance --
// multiplying by zero would erase the sky entirely. Bypass multiply
// for those pixels to preserve the original signal.
static const float kAlbedoSkyEps = 1.0 / 65536.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float4 demod = demod_in[tid];
    float3 a     = albedo_tex[uint2(tid)].rgb;
    // Sky pixels: forward the upstream value (which was never
    // demodulated). Surface pixels: lighting * surface_color =
    // textured radiance. Preserve the sample-count tag in .a so
    // downstream consumers still see how converged the source pixel
    // was (matches the temporal / atrous passes upstream).
    float3 rgb = (a.r + a.g + a.b > kAlbedoSkyEps)
                     ? (demod.rgb * a)
                     : demod.rgb;
    color_out[tid] = float4(rgb, demod.a);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Bloom downsample (Jorge Jimenez ACM 2014 13-tap kernel).
//
// 13 bilinear samples in three groups:
//   - 1 centre (E)
//   - 4 inner corners at +/-1 source pixels (J,K,L,M)
//   - 8 outer ring at +/-2 source pixels (4 corners A,C,G,I + 4 edges
//     B,D,F,H)
// Bilinear-sampled => effective 36 source texels per dst texel. The
// weights approximate a Gaussian, so a single hot HDR pixel produces
// a circular soft halo instead of the boxy footprint a uniform box
// filter gives. This is the kernel Call of Duty / Frostbite / Unreal
// use for production bloom.
//
// First-mip-only extras:
//   - luminance threshold (with smoothstep knee) so only HDR
//     highlights enter the pyramid.
//   - Karis-average anti-flicker on the four inner-corner taps:
//     weight each by 1/(1+lum) before averaging so a single rogue
//     firefly pixel doesn't dominate and shimmer between frames.
//
// Bindings:
//   slot 0 (texture) : src -- the previous mip (or post_denoise_hdr
//                       at the top of the chain).
//   slot 1 (texture) : dst -- this mip.

// vk::image_format("rgba16f") matches the bloom_mip texture format
// the engine allocates in Engine.cpp (.format = RGBA16F). See the
// fix-up PR for the NVIDIA Format=Unknown rationale.
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> src;
[[vk::binding(1, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> dst;

// MSL slot padding -- see Tonemap.slang's notice for the full story.
// Dummies push the Push cbuffer to MSL buffer(7), matching the
// engine's universal push_slot=max_bound_slot+1=7 contract that
// PathTrace's bindings (slots 1..6) drag along to every dispatch
// after it. Dead branch below references them so Slang's optimizer
// keeps them.
//
// Gated to Metal only: the dummy types (StructuredBuffer at binding 6
// in particular) would collide with the shared Vulkan layout, which
// has binding 6 as a storage IMAGE (denoise_color in the path tracer).
// SPIR-V doesn't need the slot-alignment trick anyway.
#ifdef PT_TARGET_METAL
[[vk::binding(2, 0)]]  RaytracingAccelerationStructure _slot_tlas;
[[vk::binding(3, 0)]]  StructuredBuffer<float>  _slot_mesh_positions;
[[vk::binding(4, 0)]]  StructuredBuffer<uint>   _slot_mesh_indices;
[[vk::binding(5, 0)]]  StructuredBuffer<float4> _slot_primitives;
[[vk::binding(6, 0)]]  StructuredBuffer<float>  _slot_marginal_cdf;
[[vk::binding(7, 0)]]  StructuredBuffer<float>  _slot_conditional_cdf;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> _slot_exposure_state;
#endif

[[vk::push_constant]]
cbuffer Push {
    float threshold;     // > 0 = first mip (apply threshold + Karis).
    float pad[3];
};

// Software bilinear sample at a fractional source-pixel position. We
// don't have a hardware sampler bound to the storage texture so we
// emulate: read the 4 surrounding texels and lerp by sub-pixel
// fraction. Edge clamping keeps off-image samples from contributing
// dark zeroes that would shift the kernel weight.
float3 sampleBilinear(float2 sxy, int2 sd) {
    float2 fxy = sxy - 0.5;
    int2   i0  = int2(floor(fxy));
    float2 t   = fxy - float2(i0);
    int2 p00 = clamp(i0,             int2(0, 0), sd - int2(1, 1));
    int2 p10 = clamp(i0 + int2(1,0), int2(0, 0), sd - int2(1, 1));
    int2 p01 = clamp(i0 + int2(0,1), int2(0, 0), sd - int2(1, 1));
    int2 p11 = clamp(i0 + int2(1,1), int2(0, 0), sd - int2(1, 1));
    float3 c00 = src[uint2(p00)].rgb;
    float3 c10 = src[uint2(p10)].rgb;
    float3 c01 = src[uint2(p01)].rgb;
    float3 c11 = src[uint2(p11)].rgb;
    float3 a = lerp(c00, c10, t.x);
    float3 b = lerp(c01, c11, t.x);
    return lerp(a, b, t.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dst_dim, src_dim;
    dst.GetDimensions(dst_dim.x, dst_dim.y);
    src.GetDimensions(src_dim.x, src_dim.y);
    if (tid.x >= dst_dim.x || tid.y >= dst_dim.y) return;

    // Force Slang to emit the slot-padding bindings (see notice at
    // top). Branch never taken at runtime; Slang can't statically
    // prove dst_dim.x != 0xDEADBEEFu so it survives optimization.
    // Metal-only -- the dummies are SPIRV-gated out above.
#ifdef PT_TARGET_METAL
    if (dst_dim.x == 0xDEADBEEFu) {
        let p = _slot_mesh_positions[0];
        let ii = _slot_mesh_indices[0];
        let q = _slot_primitives[0];
        let mc = _slot_marginal_cdf[0];
        let cc = _slot_conditional_cdf[0];
        let ex = _slot_exposure_state[0];
        dst[uint2(0, 0)] = float4(p, float(ii), q.x, mc + cc + ex);
    }
#endif

    int2  sd = int2(src_dim);
    // Source position for this dst pixel's CENTRE: dst is half-res of
    // src, so the centre maps to (tid + 0.5) * 2 in source-pixel
    // units. That happens to land between the 4 source pixels that
    // the dst pixel covers; sampleBilinear then averages them.
    float2 sp = (float2(tid.xy) + 0.5) * 2.0;

    // 13 bilinear taps at +/-1 and +/-2 source-pixel offsets.
    float3 e = sampleBilinear(sp,                       sd);     // centre

    float3 a = sampleBilinear(sp + float2(-2.0, -2.0),  sd);     // outer corners
    float3 c = sampleBilinear(sp + float2( 2.0, -2.0),  sd);
    float3 g = sampleBilinear(sp + float2(-2.0,  2.0),  sd);
    float3 i = sampleBilinear(sp + float2( 2.0,  2.0),  sd);

    float3 b = sampleBilinear(sp + float2( 0.0, -2.0),  sd);     // outer edges
    float3 d = sampleBilinear(sp + float2(-2.0,  0.0),  sd);
    float3 f = sampleBilinear(sp + float2( 2.0,  0.0),  sd);
    float3 h = sampleBilinear(sp + float2( 0.0,  2.0),  sd);

    float3 j = sampleBilinear(sp + float2(-1.0, -1.0),  sd);     // inner corners
    float3 k = sampleBilinear(sp + float2( 1.0, -1.0),  sd);
    float3 l = sampleBilinear(sp + float2(-1.0,  1.0),  sd);
    float3 m = sampleBilinear(sp + float2( 1.0,  1.0),  sd);

    float3 result;
    if (threshold > 0.0) {
        // Karis-average anti-flicker on the inner block. Each of the
        // 4 inner taps is weighted by 1/(1+luma) before averaging --
        // suppresses lone bright firefly pixels that the path tracer
        // produces while specular paths converge.
        float wj = 1.0 / (1.0 + dot(j, float3(0.2126, 0.7152, 0.0722)));
        float wk = 1.0 / (1.0 + dot(k, float3(0.2126, 0.7152, 0.0722)));
        float wl = 1.0 / (1.0 + dot(l, float3(0.2126, 0.7152, 0.0722)));
        float wm = 1.0 / (1.0 + dot(m, float3(0.2126, 0.7152, 0.0722)));
        float wsum = wj + wk + wl + wm + 1e-6;
        float3 inner = (j*wj + k*wk + l*wl + m*wm) / wsum;
        // Karis weighting applies to the INNER 4-tap group only
        // (above); the centre + outer-ring taps use the plain COD
        // weights below. That asymmetry is deliberate-enough in
        // practice -- the inner group dominates the energy (0.5
        // weight) so damping fireflies there removes most of the
        // flicker -- but don't come here expecting a full Karis
        // average over all 13 taps.
        result = e         * 0.125
               + inner     * 0.5
               + (a+c+g+i) * (0.125 / 4.0)
               + (b+d+f+h) * (0.25  / 4.0);
        // Soft luminance GATE: smoothstep of the pixel's luminance
        // across [threshold - 0.5, threshold + 0.5] (an ABSOLUTE
        // half-luminance knee, not stops), scaling the FULL colour.
        // A pixel just past the knee contributes its entire radiance
        // -- this is a gate, not a (lum - threshold) soft-knee
        // extract. Keep the r_bloom_threshold cvar help in lockstep
        // with this description.
        float lum  = dot(result, float3(0.2126, 0.7152, 0.0722));
        float knee = 0.5;
        float w    = smoothstep(threshold - knee, threshold + knee, lum);
        result *= w;
    } else {
        // Subsequent mips: no threshold, no Karis. Straight COD
        // weighted sum.
        result = e         * 0.125
               + (j+k+l+m) * 0.125
               + (a+c+g+i) * (0.125 / 4.0)
               + (b+d+f+h) * (0.25  / 4.0);
    }

    dst[tid.xy] = float4(result, 1.0);
}

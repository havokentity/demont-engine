// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Bloom upsample (additive, bilinear).
//
// Reads `src` (the smaller mip) and adds a smoothly-resampled copy
// to `dst` (the larger mip). For each dst pixel we map back to a
// floating-point source position and take 9 BILINEAR samples on a
// 3x3 grid centred there. Each bilinear sample is 4 source texels
// blended by sub-pixel weights, so a sharp single-pixel highlight
// in `src` shows up as a softly-shaded disk in `dst` instead of a
// stepped square.
//
// (Earlier draft used point-sampled int-texel reads -- visibly boxy
// around point-light sources like the sun, which the user spotted.
// Bilinear is the standard dual-filter fix and adds 3x reads per
// tap, which is fine since the upsample runs on already-shrunken
// mips.)

[[vk::binding(0, 0)]] RWTexture2D<float4> src;
[[vk::binding(1, 0)]] RWTexture2D<float4> dst;

// MSL slot padding -- see Tonemap.slang's notice. Aligns Push to
// MSL buffer(7) so the engine's universal push_slot=7 contract
// is satisfied on every dispatch after PathTrace.
[[vk::binding(2, 0)]]  RaytracingAccelerationStructure _slot_tlas;
[[vk::binding(3, 0)]]  StructuredBuffer<float>  _slot_mesh_positions;
[[vk::binding(4, 0)]]  StructuredBuffer<uint>   _slot_mesh_indices;
[[vk::binding(5, 0)]]  StructuredBuffer<float4> _slot_primitives;
[[vk::binding(6, 0)]]  StructuredBuffer<float>  _slot_marginal_cdf;
[[vk::binding(7, 0)]]  StructuredBuffer<float>  _slot_conditional_cdf;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> _slot_exposure_state;

[[vk::push_constant]]
cbuffer Push {
    // Spread multiplier on the 3x3 sample positions. 1.0 = textbook
    // dual-filter blur; > 1 = wider, more diffuse halo; < 1 = tight,
    // sharp core. Engine plumbs r_bloom_radius into here.
    float radius;
    float pad[3];
};

float3 sampleBilinear(float2 sxy, int2 sd) {
    // sxy is the source-pixel-space float position; sample 4
    // surrounding integer texels and lerp by fractional offset.
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

    // Force Slang to emit the slot-padding bindings. See notice at
    // top. Branch never taken at runtime.
    if (dst_dim.x == 0xDEADBEEFu) {
        let p = _slot_mesh_positions[0];
        let ii = _slot_mesh_indices[0];
        let q = _slot_primitives[0];
        let mc = _slot_marginal_cdf[0];
        let cc = _slot_conditional_cdf[0];
        let ex = _slot_exposure_state[0];
        dst[uint2(0, 0)] = float4(p, float(ii), q.x, mc + cc + ex);
    }

    int2  sd  = int2(src_dim);
    // Centre this dst pixel's source position. dst is 2x larger than
    // src, so source pos = (tid + 0.5) / 2 in source-pixel units.
    float2 sp = (float2(tid.xy) + 0.5) * 0.5;
    float  r  = max(radius, 0.0);

    // 3x3 sample grid spaced `radius` source pixels apart, bilinear
    // at each sample so the kernel is effectively a 4x4 Gaussian-ish
    // blur. Tent weights (1,2,1) / 16 give a smoother falloff than a
    // uniform 3x3 box. Increasing radius widens the per-mip spread,
    // which (compounded over the 5 upsamples) gives a noticeably
    // wider, softer halo without changing mip count.
    float3 a = sampleBilinear(sp + float2(-r, -r), sd);
    float3 b = sampleBilinear(sp + float2( 0, -r), sd);
    float3 c = sampleBilinear(sp + float2( r, -r), sd);
    float3 d = sampleBilinear(sp + float2(-r,  0), sd);
    float3 e = sampleBilinear(sp,                  sd);
    float3 f = sampleBilinear(sp + float2( r,  0), sd);
    float3 g = sampleBilinear(sp + float2(-r,  r), sd);
    float3 h = sampleBilinear(sp + float2( 0,  r), sd);
    float3 i = sampleBilinear(sp + float2( r,  r), sd);
    float3 tent = (a + c + g + i) * (1.0 / 16.0)
                + (b + d + f + h) * (2.0 / 16.0)
                +  e              * (4.0 / 16.0);

    float3 prev = dst[tid.xy].rgb;
    dst[tid.xy] = float4(prev + tent, 1.0);
}

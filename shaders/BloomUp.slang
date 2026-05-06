// Bloom upsample (additive).
//
// Reads `src` (the smaller mip), expands it 2x via a 3x3 tent kernel,
// and adds the result to `dst` (the larger mip). The tent gives a
// smoother halo than a box upsample without needing more samples.
// Repeated up the pyramid the result is a wide soft glow with a
// sharp HDR core -- the standard dual-filter bloom.

[[vk::binding(0, 0)]] RWTexture2D<float4> src;
[[vk::binding(1, 0)]] RWTexture2D<float4> dst;

float3 sampleSrc(int2 p, int2 dim) {
    p = clamp(p, int2(0, 0), dim - int2(1, 1));
    return src[uint2(p)].rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dst_dim, src_dim;
    dst.GetDimensions(dst_dim.x, dst_dim.y);
    src.GetDimensions(src_dim.x, src_dim.y);
    if (tid.x >= dst_dim.x || tid.y >= dst_dim.y) return;

    // 3x3 tent in source space. Source pixel for this dst pixel is
    // (tid / 2). Tent is { {1,2,1}, {2,4,2}, {1,2,1} } / 16.
    int2 sp = int2(tid.xy >> 1u);
    int2 sd = int2(src_dim);
    float3 a = sampleSrc(sp + int2(-1, -1), sd);
    float3 b = sampleSrc(sp + int2( 0, -1), sd);
    float3 c = sampleSrc(sp + int2( 1, -1), sd);
    float3 d = sampleSrc(sp + int2(-1,  0), sd);
    float3 e = sampleSrc(sp + int2( 0,  0), sd);
    float3 f = sampleSrc(sp + int2( 1,  0), sd);
    float3 g = sampleSrc(sp + int2(-1,  1), sd);
    float3 h = sampleSrc(sp + int2( 0,  1), sd);
    float3 i = sampleSrc(sp + int2( 1,  1), sd);
    float3 tent = (a + c + g + i) * (1.0 / 16.0)
                + (b + d + f + h) * (2.0 / 16.0)
                +  e              * (4.0 / 16.0);

    float3 prev = dst[tid.xy].rgb;
    dst[tid.xy] = float4(prev + tent, 1.0);
}

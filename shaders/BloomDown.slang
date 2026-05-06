// Bloom downsample.
//
// 4x4 box-filter downsample (16 src texels per dst texel, equal
// weights). The first mip in the chain also applies a luminance
// threshold: any contribution below `threshold` is dropped, so only
// HDR highlights enter the bloom pyramid. Subsequent mips just
// downsample what's already in the chain.
//
// Bindings:
//   slot 0 (texture) : src -- the previous mip (or post_denoise_hdr
//                       at the top of the chain).
//   slot 1 (texture) : dst -- this mip.
//
// Push:
//   threshold > 0  -> first mip, kill below-threshold pixels.
//   threshold == 0 -> later mip, no thresholding.

[[vk::binding(0, 0)]] RWTexture2D<float4> src;
[[vk::binding(1, 0)]] RWTexture2D<float4> dst;

[[vk::push_constant]]
cbuffer Push {
    float threshold;
    float pad[3];
};

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

    int2 sp = int2(tid.xy * 2u);
    int2 sd = int2(src_dim);
    float3 sum = float3(0.0);
    [unroll] for (int dy = -1; dy <= 2; ++dy)
    [unroll] for (int dx = -1; dx <= 2; ++dx) {
        sum += sampleSrc(sp + int2(dx, dy), sd);
    }
    float3 result = sum * (1.0 / 16.0);

    if (threshold > 0.0) {
        // First-mip threshold: only let bright HDR pixels into the
        // pyramid. Soft knee gives a smoother transition than a hard
        // cutoff -- still cheap, just a smoothstep on luminance.
        float lum  = dot(result, float3(0.2126, 0.7152, 0.0722));
        float knee = 0.5;
        float w    = smoothstep(threshold - knee, threshold + knee, lum);
        result *= w;
    }

    dst[tid.xy] = float4(result, 1.0);
}

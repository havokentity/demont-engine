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
    uint  passthrough;
    float bloom_intensity;
    // Lens flare. Samples the bloom mip at mirrored-across-centre
    // screen positions to fake internal lens reflections. 0 = no
    // flare (skip the loop).
    float flare_intensity;
    // Per-channel offset between R/G/B sample scales -- creates the
    // chromatic-aberration rainbow fringe along ghost edges. 0 = no
    // dispersion (achromatic ghosts). Real range 0.0..0.05.
    float flare_dispersion;
    // Number of ghost reflections (1..6). The shader has 6
    // hardcoded ghost scales; this caps how many to use.
    uint  flare_count;
    // Per-ghost luminance gate. Each bloom sample is killed below
    // this threshold so only the brightest sources (sun, hot
    // speculars) trigger flare -- bloom keeps its own lower
    // threshold for halos. With this gate flare reads as a "real
    // lens" effect rather than every-bright-pixel-ghosts.
    float flare_threshold;
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

    // Bloom + lens flare both bilinear-sample the bloom pyramid;
    // factor the bilinear sampler into a helper so the flare loop
    // doesn't repeat the math six times.
    uint2 bdim;
    bloom_tex.GetDimensions(bdim.x, bdim.y);
    int2  bd = int2(bdim);

    // sampleBloomUV(uv): bilinear bloom_tex sample at normalised UV.
    // (out-of-range UVs clamp to edges -- ghost positions can land
    // off-screen and should fade rather than wrap or sample garbage.)
    // No anonymous helpers in Slang/HLSL profile we ship, so keep
    // this inlined per call site.

    // Bloom (linear, pre-tonemap) so highlights get the same
    // ACES curve squash as the rest of the image.
    if (bloom_intensity > 0.0) {
        float2 fb = (float2(tid.xy) + 0.5) * float2(bdim) / float2(dim) - 0.5;
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
        c += bloom_intensity * lerp(b0, b1, t.y);
    }

    // Lens flare. Reflections off internal lens elements appear in a
    // real camera as ghost copies of bright sources placed
    // approximately mirrored across the lens centre, with various
    // sub-/super-magnifications. We fake the effect by sampling the
    // already-extracted bloom layer at N "mirror scale" positions:
    //   sample_uv = -(uv - 0.5) * scale + 0.5
    // and adding the result back, tinted per ghost so the artefact
    // reads as colourful flare rather than a duplicate sun. Per-
    // channel scale offsets give the rainbow chromatic aberration
    // you see on cheap lenses.
    if (flare_intensity > 0.0) {
        // 6 ghost scales + tints. The scales spread inwards (< 1)
        // and outwards (> 1) of the centre, the tints rotate around
        // the colour wheel so a single bright source spawns a
        // varied flare rather than 6 identical copies.
        const float kScales[6] = {0.55, 0.78, 1.20, 1.45, 1.85, 2.40};
        const float3 kTints[6] = {
            float3(1.00, 0.55, 0.55),  // red
            float3(0.55, 1.00, 0.55),  // green
            float3(0.55, 0.55, 1.00),  // blue
            float3(1.00, 0.75, 0.40),  // amber
            float3(0.65, 0.40, 1.00),  // purple
            float3(0.40, 0.95, 0.80),  // teal
        };
        float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
        float2 cv = uv - 0.5;
        uint   n  = min(flare_count, 6u);
        float3 flare = float3(0.0);
        for (uint gi = 0u; gi < n; ++gi) {
            float scale = kScales[gi];
            // Per-channel scale offset for chromatic dispersion.
            // Flipping sign for B vs R puts blue and red on opposite
            // sides of the green ghost -- canonical lens-flare look.
            float disp = flare_dispersion;
            float2 g_uv_r = -cv * (scale + disp) + 0.5;
            float2 g_uv_g = -cv *  scale         + 0.5;
            float2 g_uv_b = -cv * (scale - disp) + 0.5;
            // Bilinear sample on each channel's UV. Ghosts that
            // land off-screen clamp to the edge tint -- visually
            // they fade because the bloom tex is mostly zero there.
            float3 r3, g3, b3;
            {
                float2 fb = g_uv_r * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                r3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            {
                float2 fb = g_uv_g * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                g3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            {
                float2 fb = g_uv_b * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                b3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            float3 ghost = float3(r3.r, g3.g, b3.b);
            // Per-ghost luminance gate. Real lens coatings absorb
            // low-amplitude reflections; only sources bright enough
            // to overwhelm them produce visible ghosts. We emulate
            // by killing bloom samples whose luminance falls below
            // flare_threshold (with a smoothstep knee for a soft
            // transition rather than a hard cliff).
            float lum = dot(ghost, float3(0.2126, 0.7152, 0.0722));
            float gate_knee = max(flare_threshold * 0.25, 0.1);
            float gate = smoothstep(flare_threshold,
                                    flare_threshold + gate_knee, lum);
            // Radial fade: ghosts fade out as they near the screen
            // edge so we don't get a hard rectangle of flare on
            // wide-aperture shots.
            float fade = saturate(1.0 - length(uv - 0.5) * 1.5);
            flare += ghost * kTints[gi] * fade * gate;
        }
        c += flare_intensity * flare;
    }

    float3 ldr = (passthrough != 0u) ? c : acesTonemap(c * exposure);
    ldr_out[tid.xy] = float4(ldr, 1.0);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
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
    // (image mode only) Per-ghost luminance gate against the
    // bloom-mip sample. 0 = no gate.
    float flare_threshold;
    // 1 = 'sun' mode: project the sun's screen UV (passed in
    // sun_uv) and draw discrete soft Gaussian-disc ghosts at
    // its mirrored positions. Clean, no scene-mirror artefacts.
    // 0 = 'image' mode: mirror-sample the bloom mip at every
    // output pixel (the legacy code path).
    uint  flare_mode_sun;
    // (sun mode) base Gaussian disc radius in normalised screen
    // units. Each ghost scales this by its own size factor so
    // mirrors-near-centre are tighter than mirrors-far-out.
    float flare_size;
    // Sun's screen UV (computed engine-side from sun_dir + camera
    // basis). Sentinel (-2, -2) means sun is behind / off-screen
    // so the flare loop is skipped.
    float2 sun_uv;
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
        // 6 ghost scales + tints shared between modes.
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

        // 'sun' mode: explicit sun-position flare. Project sun's
        // screen UV (engine-side); draw discrete soft Gaussian-disc
        // ghosts at its mirrored positions. No bloom sampling --
        // this is what people mean by lens flare.
        if (flare_mode_sun != 0u) {
            // Skip if sun is behind camera or off-screen -- engine
            // sentinels with (-2, -2) so any out-of-[-0.5, 1.5]
            // value means "no flare".
            if (sun_uv.x > -1.0 && sun_uv.x < 2.0 &&
                sun_uv.y > -1.0 && sun_uv.y < 2.0) {
                // Distance in PIXEL space so ghosts read as real
                // circles regardless of aspect ratio. UV is
                // [0,1] x [0,1] but a 16:9 screen has 1.78x more
                // pixels along X than Y -- using raw UV distance
                // gave horizontally-stretched ellipses. Multiplying
                // d_uv by float2(dim) puts both axes in pixels;
                // r is then a pixel radius derived from flare_size
                // (which is fraction-of-screen-height, the more
                // intuitive author-time scale).
                float2 dimf = float2(dim);
                float r_h = flare_size * float(dim.y);   // in pixels
                for (uint gi = 0u; gi < n; ++gi) {
                    float scale = kScales[gi];
                    // Ghost centre = mirror sun across screen centre,
                    // scaled. scale=1 places it exactly opposite the
                    // sun; <1 closer to centre; >1 past centre.
                    float2 ghost_c = 0.5 - (sun_uv - 0.5) * scale;
                    float2 d_px = (uv - ghost_c) * dimf;
                    // Slight per-ghost size variation. Far-side
                    // ghosts a touch larger than near-side.
                    float r = r_h * (0.7 + 0.3 * scale);
                    // Chromatic aberration: per-channel radius
                    // offset gives the rainbow fringe on edges.
                    float r_R = r * (1.0 + flare_dispersion * 8.0);
                    float r_G = r;
                    float r_B = r * (1.0 - flare_dispersion * 8.0);
                    float d2 = dot(d_px, d_px);
                    // Super-Gaussian (exp(-(d/r)^4)) for a clean
                    // disc edge instead of a soft halo bleed.
                    float qR = d2 / max(r_R * r_R, 1e-6);
                    float qG = d2 / max(r_G * r_G, 1e-6);
                    float qB = d2 / max(r_B * r_B, 1e-6);
                    float i_R = exp(-qR * qR);
                    float i_G = exp(-qG * qG);
                    float i_B = exp(-qB * qB);
                    float3 ghost = float3(i_R, i_G, i_B) * kTints[gi] * 4.0;
                    // Radial fade in pixel space too, so the
                    // vignette is circular.
                    float2 cd_px = (uv - 0.5) * dimf;
                    float diag   = 0.5 * length(dimf);
                    float fade   = saturate(1.0 - length(cd_px) / diag * 1.2);
                    flare += ghost * fade;
                }
                c += flare_intensity * flare;
            }
            float3 ldr_sun = (passthrough != 0u) ? c : acesTonemap(c * exposure);
            ldr_out[tid.xy] = float4(ldr_sun, 1.0);
            return;
        }

        // 'image' mode (legacy): mirror-sample the bloom mip at
        // every output pixel. Looks cinematic but reflects the
        // whole scene -- not what a real lens does.
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
            // Optional source-luminance gate. CRUCIAL: we measure
            // luminance against the *original* hdr_in at the ghost's
            // position, not the post-pyramid bloom sample. The
            // bloom-chain upsample stage amplifies values (each mip
            // adds the upper mip's halo into the lower one) so a
            // moderately bright source ends up at higher luminance
            // in bloom mip 0 than its original brightness -- gating
            // against the bloom sample lets things slip through that
            // shouldn't. Sampling hdr_in at the same ghost UV gives
            // the truthful "is this position actually the sun?" gate.
            float gate = 1.0;
            if (flare_threshold > 0.0) {
                int2 idim = int2(dim);
                float2 fb = g_uv_g * float2(dim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), idim - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), idim - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), idim - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), idim - int2(1,1));
                float3 s00 = hdr_in[uint2(p00)].rgb;
                float3 s10 = hdr_in[uint2(p10)].rgb;
                float3 s01 = hdr_in[uint2(p01)].rgb;
                float3 s11 = hdr_in[uint2(p11)].rgb;
                float3 src = lerp(lerp(s00, s10, t.x),
                                  lerp(s01, s11, t.x), t.y);
                float src_lum = dot(src, float3(0.2126, 0.7152, 0.0722));
                // Higher-order Reinhard: lum^16 / (lum^16 + t^16).
                // Steep cliff right at `t`, smooth math (no branch),
                // and -- unlike linear Reinhard -- doesn't dim the
                // *passing* samples. Sun at lum=50 with t=30 ->
                // ratio=1.67, ratio^16 ~= 3600, gate ~= 1.0 (full
                // brightness). Wall at lum=25 with t=30 -> ratio=0.83,
                // ratio^16 ~= 0.05, gate ~= 0.05 (killed). The previous
                // linear gate forced thresholds in the 100s to filter
                // walls and shredded the sun; this one separates them
                // cleanly at threshold values near the actual luminance
                // boundary you want.
                float L = src_lum / max(flare_threshold, 1e-4);
                L = L * L;  // ^2
                L = L * L;  // ^4
                L = L * L;  // ^8
                L = L * L;  // ^16
                gate = L / (L + 1.0);
            }
            // Radial fade: ghosts fade out as they near the screen
            // edge so we don't get a hard rectangle of flare on
            // wide-aperture shots.
            float fade = saturate(1.0 - length(uv - 0.5) * 1.5);
            flare += ghost * kTints[gi] * fade * gate;
        }
        c += flare_intensity * flare;
    }

    // Debug: when r_lens_flare_intensity is exactly negative,
    // overlay a green crosshair at sun_uv so the user can see
    // where the engine projects the sun. (Normal mode never sets
    // negative intensity; engine sets a sentinel of -1.0 when
    // dump_sun_uv_overlay = 1.)
    if (flare_intensity < 0.0 &&
        sun_uv.x > -1.0 && sun_uv.x < 2.0 &&
        sun_uv.y > -1.0 && sun_uv.y < 2.0) {
        float2 uvN = (float2(tid.xy) + 0.5) / float2(dim);
        float2 d = (uvN - sun_uv) * float2(dim) / 2.0;
        // 24-px arms, 2-px thick.
        bool on_h = abs(d.y) < 1.0 && abs(d.x) < 24.0;
        bool on_v = abs(d.x) < 1.0 && abs(d.y) < 24.0;
        if (on_h || on_v) {
            ldr_out[tid.xy] = float4(0.0, 1.0, 0.0, 1.0);
            return;
        }
    }
    float3 ldr = (passthrough != 0u) ? c : acesTonemap(c * exposure);
    ldr_out[tid.xy] = float4(ldr, 1.0);
}

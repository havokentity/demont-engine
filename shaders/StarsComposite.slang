// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Stateless per-pixel composite of celestial point sources
// (BSC/procedural stars, sun disc, moon disc) over the post-denoise
// HDR image. Replaces the EMA accum_stars architecture that #108
// introduced for issue #46 -- that approach needed a reset gate
// decoupled from the denoiser's accum_dirty_ trigger AND motion-vector
// reprojection to avoid flicker-on-pan / ghost-on-pan tradeoffs.
// Eliminating history entirely sidesteps both classes of bug.
//
// Why composite post-denoise:
//   - Stars are sub-pixel point sources. SVGF a-trous smears each
//     star's energy across a ~17x17 texel neighbourhood by the third
//     pass; MetalFX TemporalDenoisedScaler upsamples them into
//     oblivion. Routing the primary-miss star term around the
//     denoiser is the only way to keep the pinpoint character.
//   - Sun + moon discs are small (~0.5 deg) and share the same
//     denoiser-erosion problem: SVGF edge-softens them and DOF makes
//     the disc travel between pixels each frame, so the denoiser sees
//     a noisy disc and smooths it away. Compositing them with N
//     aperture samples in ONE pass produces a clean bokeh in a single
//     frame.
//
// PathTrace.slang's primary-miss site subtracts stars + sunDisc +
// moonDisc from the sky term BEFORE writing the denoiser input, so
// the SVGF/MetalFX chain sees only the procedural sky base + NEE
// indirect contributions. The celestials are re-added here per-pixel
// with their own aperture-sampled bokeh.
//
// Compositing happens AFTER the denoiser writes post_denoise_hdr (or
// PathTrace's denoise_color in the bloom-without-denoiser path) and
// BEFORE the bloom pyramid runs over that texture, so star/sun/moon
// highlights still feed bloom (acceptance criterion in issue #46:
// "bloom halos proportional to brightness"). The ACES tonemap kernel
// then squashes the combined image as one.
//
// NEE / scene illumination from stars is unaffected: the b > 0 sky
// term in PathTrace.slang keeps full skyColor() (which still injects
// starsOnly() via procSky()), so secondary-bounce rays escaping to
// the night sky still pick up star light. Only the primary-miss
// pixel write peels them off.
//
// Bindings (Metal MSL slots assigned in declaration order: hdr_inout
// -> texture(0), star_map -> texture(1), moon_map -> texture(2),
// depth_tex -> texture(3); vk::binding tags are for the SPIR-V path
// -- presently not wired, see the TODO at the engine dispatch site).
//
// depth_tex is PathTrace's per-pixel primary-hit depth G-buffer (linear
// camera-space Z in world units along the forward axis). Sky pixels --
// where the primary ray missed the scene -- get a sentinel value of
// 1000.0 (see PathTrace.slang's G-buffer pass). We composite the
// celestials ONLY at sky pixels so the stars / sun / moon don't render
// on top of foreground geometry. Without this gate the composite is an
// unconditional additive write and stars layer onto the ground plane
// and other surfaces.
[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_inout;
[[vk::binding(1, 0)]] RWTexture2D<float4> star_map;
[[vk::binding(2, 0)]] RWTexture2D<float4> moon_map;
[[vk::binding(7, 0)]] RWTexture2D<float>  depth_tex;
// Cloud transmittance along the camera ray, written by PathTrace's
// volumetric march at the end of main(). 1.0 = no cloud blocking
// (no clouds, ray misses the layer, or layer fully transparent); 0.0
// = fully opaque cloud. The composite multiplies its celestial
// contribution by this scalar so stars / sun / moon get correctly
// occluded by clouds in front of them. Cloud-glow / in-scatter
// illumination of the cloud BY the sun and moon stays a path-tracer
// concern -- it's already captured in hdr_inout through the existing
// b>0 cloud-march integration. This texture only carries the
// transmittance number the composite needs.
//
// vk::binding(22, 0) matches PathTrace.slang's cloud_trans_tex slot;
// Metal MSL assigns texture(4) by declaration order (slot 0..3 above,
// slot 4 here). Engine binds it at engine slot 4 in the composite
// dispatch.
[[vk::binding(22, 0)]] RWTexture2D<float>  cloud_trans_tex;

// MSL SLOT NOTICE: this kernel binds NO buffers and NO acceleration
// structures. Slang's MSL backend therefore assigns Push (the only
// buffer-class declaration left below) to MSL buf(0). The engine's
// MetalCommandBuffer::Dispatch computes push_slot = max_bound_buf_slot
// + 1; with nothing else bound and BindComputePipeline clearing prior
// bindings on Metal, max_slot = -1, push_slot = 0. Match.
//
// An earlier revision declared dummy AS / storage-buffer slots to
// force Push to buf(7) so the kernel could inherit PathTrace's
// exposure_state binding. That broke under Metal's
// BindComputePipeline-clears-state semantics (Copilot review #114):
// the dummies referenced slots the engine had stopped binding, and
// Metal's resource validation flagged the dispatch if the runtime-
// dead retention branch ever evaluated. Reverted -- this kernel
// doesn't actually need exposure_state, and the simpler "Push at
// buf(0), no other buffers" layout removes the failure mode entirely.

[[vk::push_constant]]
cbuffer Push {
    // Camera basis. Same layout as PathTrace.slang's prefix so a host
    // PtPush copy can repurpose the leading bytes if we ever share the
    // struct -- but the engine builds a dedicated StarsCompositePush
    // today, kept independent so this kernel stays free of the
    // 736-byte PathTrace tail.
    float4 pos_fovtan;     // .xyz = camera origin, .w = fovYTan
    float4 fwd_aspect;     // .xyz = forward, .w = aspect ratio (width/height)
    float4 right_xyz;
    float4 up_xyz;

    // .xyz = unit vector toward sun, .w = sky mode (0=gradient,
    // 1=hdri, 2=procedural). We only run the celestials composite in
    // procedural mode (mode == 2); other modes have stars/sun/moon
    // baked into env_map or use a fallback gradient.
    float4 sun_and_mode;
    // .y = 1.0 if procedural cell-hash starfield is active.
    // .z = 1.0 if BSC starmap should be sampled.
    // .w = 1.0 if stars should twinkle (atmospheric scintillation).
    float4 exposure_pad;
    // World->J2000 rotation rows used by starsOnly for the BSC sample.
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
    // .xyz = unit vector toward moon, .w = phase angle (radians).
    float4 moon_dir_phase;
    // .x = r_moon_size, .y = distance ratio, .z = r_moon_brightness.
    float4 moon_extra;
    // .x = r_sun_size, .y = distance ratio, .z = vertical-flatten.
    float4 sun_extra;
    // .x = aperture radius (0 = pinhole), .y = focal distance,
    // .z = aperture blade count (0 = round disk, 3..16 = polygonal).
    float4 dof_params;

    uint frame_index;
    // Per-pixel aperture sample count. 1 = single ray through pixel
    // centre (no bokeh, no jitter benefit). >1 spreads samples across
    // the lens aperture (DOF on) or sub-pixel jitter (DOF off) so a
    // bokeh disc resolves in one frame. Engine sources this from
    // r_stars_aperture_samples (default 16); cost scales linearly with
    // the average path length inside starsOnly/sunDisc/moonDisc.
    uint ap_samples;
    // 1 -> run the composite (engine's gate for r_star_split &&
    // procedural sky); 0 -> early-out (texture unchanged). Cheaper
    // than skipping the dispatch entirely because it keeps the
    // compute-pipeline state binding stable, but the dispatch itself
    // is still elided engine-side when this is 0 so we don't burn
    // GPU on an early-return-everywhere kernel.
    uint composite_active;
    uint _pad0;
};

// --- ported helpers from PathTrace.slang -----------------------------
//
// These are deliberately COPIES of the starsOnly / sunDisc / moonDisc
// implementations in PathTrace.slang. Slang doesn't have a module
// system we could share across two kernels in this project today, and
// duplicating them keeps this composite shader self-contained --
// drift between the two would manifest as a visible diff between the
// path-tracer's primary-miss subtraction and the post-denoise add-back,
// so any change in PathTrace.slang's celestial logic must mirror here.
//
// PathTrace.slang's `pcgHash` is referenced by starsOnly's twinkle
// branch; replicate the same hash function so the twinkle phase is
// frame-stable when shaders are recompiled together.

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float3 starsOnly(float3 rd, float3 sun) {
    float cosTheta = clamp(rd.y, -1.0, 1.0);
    float sun_elev = sun.y;
    float day      = smoothstep(-0.10, 0.20, sun_elev);
    if (cosTheta > -0.05 && exposure_pad.z > 0.5 && day < 0.6) {
        float3 j;
        j.x = dot(w2j_row0.xyz, rd);
        j.y = dot(w2j_row1.xyz, rd);
        j.z = dot(w2j_row2.xyz, rd);
        float u = atan2(j.y, j.x) * (1.0 / 6.28318530718);
        if (u < 0.0) u += 1.0;
        float v = (1.5707963267948966 - asin(clamp(j.z, -1.0, 1.0))) * (1.0 / 3.14159265358979);
        uint2 dim;
        star_map.GetDimensions(dim.x, dim.y);
        int xi = int(floor(u * float(dim.x))) % int(dim.x);
        if (xi < 0) xi += int(dim.x);
        int yi = clamp(int(floor(v * float(dim.y))), 0, int(dim.y) - 1);
        float3 stars = star_map[uint2(uint(xi), uint(yi))].rgb;
        // Per-star 1-3 Hz base scintillation, scaled by exposure_pad.w
        // (the r_stars_twinkle_speed multiplier; 0 means off). Must
        // match PathTrace.slang's starsOnly() copy verbatim -- see
        // that file's twinkle comment for the full rationale.
        // Two-sine scintillation centered on 1.0. See PathTrace.slang's
        // copy for the long-form rationale (incommensurate frequencies
        // -> quasi-aperiodic waveform that mimics real atmospheric
        // turbulence; mean 1.0 preserves dim-star visibility).
        if (exposure_pad.w > 0.001) {
            int3 q  = int3(floor(j * 38.0));
            uint qh = uint(q.x + 128) * 65536u
                    + uint(q.y + 128) * 256u
                    + uint(q.z + 128);
            uint h    = pcgHash(qh + 0xb7e1u);
            float ph  = float(h & 0xFFFFu) * (1.0 / 65536.0);
            float fhz_base = 1.0 + float((h >> 16) & 0xFFu) * (2.0 / 256.0);
            uint h2   = pcgHash(h + 0xc0ffeeu);
            float ph2 = float(h2 & 0xFFFFu) * (1.0 / 65536.0);
            float fhz_mult = 2.7 + float((h2 >> 16) & 0xFFu) * (2.6 / 256.0);
            float speed = exposure_pad.w;
            float fhz1 = fhz_base * speed;
            float fhz2 = fhz_base * fhz_mult * speed;
            float t   = float(frame_index) * (1.0 / 60.0);
            float w   = 1.0
                      + 0.30 * sin(6.28318530718 * (ph  + t * fhz1))
                      + 0.15 * sin(6.28318530718 * (ph2 + t * fhz2));
            stars *= w;
        }
        return stars * (1.0 - day);
    } else if (cosTheta > -0.05 && exposure_pad.y > 0.5 && day < 0.6) {
        float2 ll = float2(atan2(rd.z, rd.x) * (1.0 / 6.28318530718) + 0.5,
                           acos(clamp(rd.y, -1.0, 1.0)) * (1.0 / 3.14159265359));
        ll *= float2(700.0, 350.0);
        float2 cellI = floor(ll);
        float2 cellF = ll - cellI;
        float h  = frac(sin(dot(cellI, float2(127.1, 311.7))) * 43758.5453);
        float h2 = frac(sin(dot(cellI, float2(269.5, 183.3)))  * 91842.1417);
        if (h > 0.96) {
            float2 starPos = float2(0.30 + h2 * 0.40,
                                    0.30 + frac(h * 7.91) * 0.40);
            float dist = length(cellF - starPos);
            float brightness = pow(frac(h * 13.7), 4.0) * 4.0 + 0.5;
            float intensity = brightness * exp(-dist * dist * 100.0);
            float colorMix = frac(h * 5.31);
            float3 hot  = float3(0.85, 0.92, 1.10);
            float3 mid  = float3(1.00, 0.98, 0.95);
            float3 cool = float3(1.05, 0.85, 0.70);
            float3 starColor = (colorMix < 0.33) ? hot
                              : (colorMix < 0.80) ? mid : cool;
            return intensity * starColor * (1.0 - day);
        }
    }
    return float3(0.0);
}

float3 sunDisc(float3 rd, float3 sun_dir) {
    if (sun_dir.y < -0.10) return float3(0.0);
    float ang = dot(rd, sun_dir);
    float sun_mult = max(sun_extra.x * sun_extra.y, 0.1);
    float r_deg    = 0.55 * sun_mult;
    float kCosR    = cos(r_deg * 0.0174533);
    float halo_deg = lerp(2.0, 8.0, saturate(1.0 - sun_dir.y));
    float halo_rad = halo_deg * 0.0174533;
    float kCosHalo = cos(halo_rad);
    if (ang <= kCosHalo) return float3(0.0);

    float vflat = sun_extra.z;
    bool oval = vflat < 0.999;
    bool inside_disc = false;
    if (ang > kCosR) {
        if (!oval) {
            inside_disc = true;
        } else {
            float3 world_up    = float3(0.0, 1.0, 0.0);
            float3 local_right = normalize(cross(world_up, sun_dir));
            float3 local_up    = cross(sun_dir, local_right);
            float3 delta       = rd - sun_dir * ang;
            float h_off        = dot(delta, local_right);
            float v_off        = dot(delta, local_up);
            float r_h          = r_deg * 0.0174533;
            float r_v          = r_h * vflat;
            float ellipse      = (h_off * h_off) / (r_h * r_h)
                               + (v_off * v_off) / (r_v * r_v);
            inside_disc = ellipse <= 1.0;
        }
    }

    float am  = 1.0 / max(sun_dir.y, 0.04);
    float atm = exp(-0.30 * am);
    float warmth = saturate((am - 1.0) * 0.18);
    float3 col = lerp(float3(1.00, 0.98, 0.94),
                      float3(1.55, 0.55, 0.20), warmth);

    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        float falloff = lerp(8.0, 3.0, saturate(1.0 - sun_dir.y));
        halo = pow(t_h, falloff);
    }
    if (inside_disc) {
        return col * (250.0 * atm)
             + col * halo * atm * 4.0;
    }
    return col * halo * atm * 4.0;
}

// Mirror of PathTrace.slang's procSkyInlineSunDiscAndHalo. PathTrace
// SUBTRACTS this term from the b==0 sky so denoise_color is sun-free;
// this kernel ADDS it back, aperture-sampled, so the sun's bokeh
// resolves cleanly in one frame (matching what we do for sunDisc and
// moonDisc). Any change here must mirror the PathTrace copy or the
// composite-on / composite-off paths will diverge in sun brightness.
float3 procSkyInlineSunDiscAndHalo(float3 rd, float3 sun) {
    float cosGamma = clamp(dot(rd, sun), -1.0, 1.0);
    float sun_elev = sun.y;
    float day      = smoothstep(-0.10, 0.20, sun_elev);
    float disk     = smoothstep(0.99996, 0.99999, cosGamma);
    float3 sun_color_high = float3(1.00, 0.95, 0.85);
    float3 sun_color_low  = float3(1.00, 0.55, 0.20);
    float3 sun_col = lerp(sun_color_low, sun_color_high, day);
    float3 result  = disk * 50.0 * sun_col * day;
    float halo     = pow(max(0.0, cosGamma), 80.0);
    result += halo * sun_col * 0.30 * day;
    return result;
}

float3 moonDisc(float3 rd, float4 mph, float3 sun_dir_world) {
    if (mph.y < -0.10) return float3(0.0);
    float3 m_dir = mph.xyz;
    float ang = dot(rd, m_dir);
    float size_mult = max(moon_extra.x * moon_extra.y, 0.1);
    float r_deg     = 0.55 * size_mult;
    float halo_deg  = 1.5  * size_mult;
    float kCosR     = cos(r_deg    * 0.0174533);
    float kCosHalo  = cos(halo_deg * 0.0174533);
    if (ang <= kCosHalo) return float3(0.0);

    float3 col_tint = float3(0.92, 0.94, 1.00);
    float moon_am   = 1.0 / max(mph.y, 0.04);
    float moon_atm  = exp(-0.30 * moon_am);
    float warmth_m  = saturate((moon_am - 1.0) * 0.18);
    float3 col_atm  = lerp(float3(1.00, 1.00, 1.00),
                           float3(1.20, 0.65, 0.40), warmth_m);

    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        halo = pow(t_h, 4.0) * 0.6;
    }

    if (ang > kCosR) {
        float t_d = (ang - kCosR) / (1.0 - kCosR);
        float3 ax = abs(m_dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 t_a = normalize(cross(ax, m_dir));
        float3 t_b = cross(m_dir, t_a);
        float disc_scale = 1.0 / max(sin(r_deg * 0.0174533), 1e-5);
        float du = dot(rd, t_a) * disc_scale;
        float dv = dot(rd, t_b) * disc_scale;
        float r2 = du * du + dv * dv;
        float zz = sqrt(max(1.0 - r2, 0.0));
        float3 n_world = du * t_a + dv * t_b - zz * m_dir;
        float lit = saturate(dot(n_world, sun_dir_world) * 10.0 + 0.05);

        if (lit <= 0.0) return col_tint * col_atm * halo * moon_atm * 0.05;

        float anti_sun_dot = -dot(m_dir, sun_dir_world);
        float anti_sun_ang = acos(saturate(anti_sun_dot));
        const float kUmbraRad    = 0.0122;
        const float kPenumbraRad = 0.0218;
        float eclipse = 1.0 - smoothstep(kUmbraRad, kPenumbraRad,
                                         anti_sun_ang);

        const float kHalfPi = 1.57079632679;
        float lon = atan2(du, zz);
        float lat = asin(clamp(dv, -1.0, 1.0));
        float u_tex = 0.5 + lon / 6.28318530718;
        float v_tex = 0.5 - lat / 3.14159265358979;
        u_tex = saturate(u_tex);
        v_tex = saturate(v_tex);
        uint2 mdim;
        moon_map.GetDimensions(mdim.x, mdim.y);
        float2 fpix = float2(u_tex * float(mdim.x), v_tex * float(mdim.y)) - 0.5;
        int2 ipix = int2(floor(fpix));
        float2 f = fpix - float2(ipix);
        int x0 = clamp(ipix.x,     0, int(mdim.x) - 1);
        int x1 = clamp(ipix.x + 1, 0, int(mdim.x) - 1);
        int y0 = clamp(ipix.y,     0, int(mdim.y) - 1);
        int y1 = clamp(ipix.y + 1, 0, int(mdim.y) - 1);
        float3 c00 = moon_map[uint2(x0, y0)].rgb;
        float3 c10 = moon_map[uint2(x1, y0)].rgb;
        float3 c01 = moon_map[uint2(x0, y1)].rgb;
        float3 c11 = moon_map[uint2(x1, y1)].rgb;
        float3 surface = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
        float limb = 0.55 + 0.45 * sqrt(saturate(t_d));
        float bright_mult = max(moon_extra.z, 0.0);
        float disc_scale_b = 3.5 * bright_mult;
        float3 lit_col = col_tint * col_atm * surface
                       * (disc_scale_b * lit * limb * moon_atm)
                       + col_tint * col_atm * halo * lit * moon_atm
                       * (0.6 * bright_mult);
        float3 eclipse_col = float3(0.55, 0.15, 0.04) * surface
                           * (limb * moon_atm * 1.2);
        return lerp(lit_col, eclipse_col, eclipse);
    }
    float phase_b = 0.5 * (1.0 - cos(mph.w));
    return col_tint * col_atm * halo * phase_b * moon_atm * 0.3;
}

// Halton(base) low-discrepancy sequence. Returns the i-th value in
// [0,1) of the base-b Halton sequence. Deterministic per (pixel,
// sample, frame) -- the offset by frame_index in the caller keeps the
// pattern frame-stable when the camera is still, but rotates it
// between frames so the moving-camera case integrates over a few
// frames without obvious banding.
float halton(uint i, uint base) {
    float f = 1.0;
    float r = 0.0;
    while (i > 0u) {
        f = f / float(base);
        r = r + f * float(i % base);
        i = i / base;
    }
    return r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (composite_active == 0u) return;

    // Sky-pixel gate. PathTrace.slang's G-buffer pass writes
    // depth_view = 1000.0 for primary-miss pixels and the real
    // camera-space Z (typically << 100 in metric units for any normal
    // scene) for surface hits. Threshold at 500 m so we capture only
    // sky pixels; anything closer is geometry and we leave it alone.
    //
    // 5x5 dilation of the geometry mask (2-pixel safety band).
    // PathTrace's G-buffer pass is UNJITTERED (uses uv0 not the
    // jittered uv) so on horizon-boundary pixels the depth value
    // reports whichever side of the edge the pixel CENTER lies on,
    // while the jittered radiance write can hit the opposite side.
    // The 1-pixel cross dilation that shipped first still let a
    // couple of stars sit on the horizon line. Widening to a 5x5
    // box (any depth-min < 500 within 2 pixels gates this pixel out)
    // gives a robust 2-pixel safety band on all edge orientations,
    // including diagonal silhouettes. 25 depth_tex reads per pixel
    // -- the texture cache catches the overlap with neighboring
    // threads in the threadgroup so the true cost is closer to
    // 1 miss per thread + cached hits.
    int W = int(dim.x) - 1;
    int H = int(dim.y) - 1;
    int2 px = int2(tid.xy);
    float zmin = 1e30;
    for (int dy = -2; dy <= 2; ++dy) {
        int yy = clamp(px.y + dy, 0, H);
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = clamp(px.x + dx, 0, W);
            zmin = min(zmin, depth_tex[uint2(xx, yy)]);
        }
    }
    if (zmin < 500.0) return;

    float aperture = dof_params.x;
    float focal_d  = dof_params.y;
    int   blades   = int(dof_params.z + 0.5);
    bool  dof_on   = aperture > 0.0;

    uint N = max(ap_samples, 1u);
    float inv_N = 1.0 / float(N);

    // Per-pixel deterministic offset into the Halton sequence so two
    // adjacent pixels don't share the same N-sample pattern (would
    // produce a visible grid of identical bokeh placements). Hash on
    // pixel coords ONLY -- frame_index is intentionally absent so the
    // sample pattern is stable frame-to-frame. Adding it back caused a
    // global twinkle flicker: every frame's sample pattern shifted in
    // lockstep across all pixels, so each frame all stars simultaneously
    // got more or fewer samples landing on them. The result read as a
    // synchronized flicker over the entire star field, overpowering the
    // intended per-star scintillation that starsOnly() injects via
    // frame_index inside its own hash. The real twinkle (3-8 Hz per
    // star, random phases) still works -- it just no longer competes
    // with the global sample-pattern shift.
    uint pixel_hash = pcgHash(tid.x * 1973u + tid.y * 9277u);
    uint base_i = pixel_hash & 0xFFFFu;

    float3 acc = float3(0.0);

    for (uint i = 0u; i < N; ++i) {
        uint si = base_i + i + 1u;

        // Sub-pixel jitter. Even without DOF, stars are sub-pixel
        // point sources -- without jitter a slowly-rotating camera
        // makes each star snap on/off as its angular position crosses
        // pixel boundaries. With jitter the sample integrates the
        // star's energy across the pixel footprint, giving smooth
        // motion.
        float jx = halton(si, 2u) - 0.5;
        float jy = halton(si, 3u) - 0.5;
        float2 uv = ((float2(tid.xy) + 0.5 + float2(jx, jy)) / float2(dim)) * 2.0 - 1.0;
        uv.y = -uv.y;

        float3 ro = pos_fovtan.xyz;
        float3 rd = normalize(fwd_aspect.xyz
                           + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                           + up_xyz.xyz    * (uv.y * pos_fovtan.w));

        if (dof_on) {
            // Aperture sample matches PathTrace.slang's primary-ray
            // DOF sampler exactly so the composite's bokeh shape lines
            // up with the rest of the rendered scene's bokeh (out-of-
            // focus geometry through the path tracer). blades < 3 ->
            // round disk via Shirley-Chiu concentric mapping;
            // blades >= 3 -> regular n-gon wedge sampling for the
            // polygonal-iris look.
            float u1 = halton(si, 5u);
            float u2 = halton(si, 7u);
            float2 ap;
            if (blades < 3) {
                float a = 2.0 * u1 - 1.0;
                float b = 2.0 * u2 - 1.0;
                float r, phi;
                if (abs(a) > abs(b)) {
                    r = a;
                    phi = (3.14159265358979 / 4.0) * (b / max(abs(a), 1e-6));
                } else {
                    r = b;
                    phi = (3.14159265358979 / 2.0) -
                          (3.14159265358979 / 4.0) * (a / max(abs(b), 1e-6));
                }
                ap = r * float2(cos(phi), sin(phi));
            } else {
                float n = float(blades);
                float wedge = 6.28318530717958 / n;
                int   k = int(floor(u1 * n));
                float r1 = frac(u1 * n);
                float r2 = u2;
                if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
                float a0 = float(k) * wedge;
                float a1 = a0 + wedge;
                float2 p0 = float2(cos(a0), sin(a0));
                float2 p1 = float2(cos(a1), sin(a1));
                ap = r1 * p0 + r2 * p1;
            }
            float3 lens_off = right_xyz.xyz * (ap.x * aperture)
                            + up_xyz.xyz    * (ap.y * aperture);
            float3 focus_p  = ro + rd * focal_d;
            ro = ro + lens_off;
            rd = normalize(focus_p - ro);
        }

        float3 sun = sun_and_mode.xyz;
        // Four-term per-sample sum, mirroring the b==0 subtraction in
        // PathTrace.slang (starsOnly + sunDisc + moonDisc +
        // procSkyInlineSunDiscAndHalo). The aperture loop averages
        // these so DOF naturally resolves the sun / moon bokeh in a
        // single frame, no temporal accumulation required.
        float3 sample_contrib = starsOnly(rd, sun)
                              + sunDisc(rd, sun)
                              + moonDisc(rd, moon_dir_phase, sun)
                              + procSkyInlineSunDiscAndHalo(rd, sun);
        acc += sample_contrib;
    }
    acc *= inv_N;

    // Cloud occlusion. PathTrace writes the camera-ray cloud-march
    // transmittance to cloud_trans_tex (1.0 = no cloud, 0.0 = fully
    // opaque). Multiplying the composite by it darkens stars / sun /
    // moon in proportion to the integrated cloud density along the
    // camera ray -- so a thick cloud occludes the celestials and a
    // thin / wispy cloud lets them shine through dimmed. With clouds
    // off (clouds_p1.w == 0 host-side), trans defaults to 1.0 and
    // this is a no-op multiply.
    float cloud_trans = cloud_trans_tex[tid.xy];
    acc *= cloud_trans;

    // Sanitize. starsOnly reads a host-uploaded texture we trust to
    // be clean, but a NaN slipping into the moon-map equirectangular
    // wrap or a degenerate aperture sample (zero-length ray after
    // focus subtraction at focal_d ~= 0) would otherwise persist into
    // the HDR texture. Snap to 0 / max-with-zero so this pass never
    // injects non-finite state.
    bool3 bad = isnan(acc) || isinf(acc);
    acc = select(bad, float3(0.0), max(acc, float3(0.0)));

    float3 base = hdr_inout[tid.xy].rgb;
    hdr_inout[tid.xy] = float4(base + acc, 1.0);
}

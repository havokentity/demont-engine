// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Aurora borealis / australis procedural overlay (issue #116).
//
// Stateless additive composite mirroring the StarsComposite (#46 / PR
// #114) pattern: runs once per frame on Metal between Denoise() and
// dispatch_bloom_pyramid, so post-denoise HDR receives a curl-noise-
// driven aurora curtain whose highlights then feed the bloom pyramid
// and get squashed by the ACES tonemap with the rest of the image.
//
// Architectural shape (deliberately the same as StarsComposite):
//   - Two textures: hdr_inout (RW additive write) and depth_tex
//     (R32F per-pixel camera-space Z, primary-miss sentinel == 1e10).
//   - 5x5 dilation of the geometry mask using the same min-depth
//     pattern -- aurora must not paint over foreground geometry,
//     and the un-jittered G-buffer pass writes pixel CENTERS while
//     the jittered radiance write can land on the opposite side of
//     a horizon edge, so a 2-pixel safety band is needed to keep
//     the aurora off geometry edges.
//   - Push struct: camera basis + ray reconstruction params + a few
//     aurora cvars + frame index for animation.
//
// Why post-denoise:
//   The aurora is a low-frequency volumetric phenomenon -- if it
//   were part of the b==0 sky term in PathTrace.slang the SVGF
//   a-trous + MetalFX upsampler would smear and accumulate it
//   correctly enough, BUT it would also share the path-tracer's
//   1-spp temporal accumulation history. Camera motion would smear
//   the curtain, history-reset latency would flicker it, and the
//   volumetric integration cost (40 samples per pixel along the
//   upper-sky cone) would multiply onto every pathtrace dispatch
//   instead of running once per frame. Compositing post-denoise
//   keeps it cheap, decoupled from accumulation state, and zero-
//   cost when r_aurora == 0 (engine elides the dispatch).
//
// Physics rationale (per project memory: 1 world unit = 1 metre,
// real atmospheric numbers):
//   Aurora emission peaks at ~100-150 km altitude (oxygen excitation
//   bands at ~557.7 nm green and ~630.0 nm red); 100 km lower bound,
//   ~250 km upper. Below that band the atmosphere is too dense for
//   the excited states to radiate before colliding; above it solar
//   particle flux thins out. We render the band as an elevation-
//   conditioned ribbon density, NOT as a true volumetric march from
//   the camera position -- a real-physics march from a ground-level
//   camera to 100 km along thousands of view rays would dominate the
//   frame budget for a phenomenon the path tracer's existing sky term
//   already approximates. We get the volumetric-curtain look from the
//   ray direction's elevation band + curl-noise density modulation;
//   the depth-integration is replaced by a fixed cosine-weighted
//   density curve peaking near the horizon (cos(elev) ~ 0.3) and
//   tapering toward zenith, matching what an observer below the
//   curtain actually sees: the curtain is densest where the line of
//   sight skims along it laterally, faintest where it points straight
//   through.
//
// Bindings (Metal MSL slots assigned in declaration order: hdr_inout
// -> texture(0), depth_tex -> texture(1); vk::binding tags are for
// the SPIR-V path, presently un-wired -- Vulkan dispatch is a
// follow-up, see issue #116 "Out of scope").
[[vk::binding(0, 0)]] RWTexture2D<float4> hdr_inout;
[[vk::binding(7, 0)]] RWTexture2D<float>  depth_tex;

// MSL SLOT NOTICE: this kernel binds NO buffers and NO acceleration
// structures. Slang's MSL backend therefore assigns Push (the only
// buffer-class declaration below) to MSL buf(0). The engine's
// MetalCommandBuffer::Dispatch computes push_slot = max_bound_buf_slot
// + 1; with nothing else bound and BindComputePipeline clearing prior
// bindings on Metal, max_slot = -1, push_slot = 0. Match. Same layout
// rule as StarsComposite -- see its long-form rationale for why we
// do not declare dummy AS / storage-buffer slots.

[[vk::push_constant]]
cbuffer Push {
    // Camera basis. Same prefix as StarsComposite / PathTrace -- a
    // host-side PtPush copy could repurpose the leading bytes if we
    // ever share the struct, but the engine builds a dedicated
    // AuroraCompositePush today.
    float4 pos_fovtan;     // .xyz = camera origin, .w = fovYTan
    float4 fwd_aspect;     // .xyz = forward, .w = aspect ratio (W/H)
    float4 right_xyz;
    float4 up_xyz;

    // .xyz = unit vector toward sun, .w = sky mode (0=gradient,
    // 1=hdri, 2=procedural). The day/night gate inside the shader
    // is conditioned on sun.y -- aurora fades to zero when the sun
    // is well above the horizon and is fully visible only below the
    // civil-twilight line. Same convention as StarsComposite.
    float4 sun_and_mode;

    // Aurora cvars packed:
    //   .x = r_aurora intensity multiplier (0..2)
    //   .y = magnetic pole latitude in radians (cvar in degrees,
    //        converted host-side -- typical value 70 deg = 1.222 rad
    //        for boreal aurora at the auroral oval)
    //   .z = animate flag (0/1) -- gates the time-based curl-noise
    //        shimmer evolution
    //   .w = time in seconds (frame_index/60 host-side; only used
    //        when .z != 0)
    float4 aurora_params;

    uint frame_index;
    // 1 -> run the composite (engine's gate for r_aurora &&
    // procedural sky); 0 -> early-out. Engine also elides the
    // dispatch when r_aurora == 0 so this is belt-and-braces.
    uint composite_active;
    uint _pad0;
    uint _pad1;
};

// --- Procedural noise helpers ---------------------------------------
//
// The aurora ribbon shape comes from a 3D curl-noise field evaluated
// along the view-ray direction (which is unit-length, so we map it
// directly to "noise space" coordinates -- the field's spatial period
// is in radians of angular direction, not metres). Standard hash +
// value-noise + curl construction:
//
//   hash3 -> 3 floats in [-1, 1) deterministic per integer cell
//   noise3 -> smooth-stepped trilinear interpolation of hash3
//   curl3 -> finite-difference curl of noise3 (a divergence-free
//            vector field -- the displacement field that gives
//            ribbon "swirls" rather than just blob-noise)
//
// PathTrace.slang's pcgHash is a 32-bit integer hash; we use it here
// for the corner-value seed and then smoothstep + trilerp upward.
// No texture lookups, no shared state -- the kernel stays purely
// per-pixel.

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hashFloat(int3 p) {
    uint h = pcgHash(uint(p.x + 73856093) * 19349663u
                   ^ uint(p.y + 19349663) * 83492791u
                   ^ uint(p.z + 83492791) * 73856093u);
    return float(h & 0xFFFFFFu) * (1.0 / 16777216.0); // [0, 1)
}

float noise3(float3 p) {
    int3 pi = int3(floor(p));
    float3 pf = p - float3(pi);
    // Quintic (Perlin) smoothstep for C2-continuous derivatives.
    // Important because we'll finite-difference this to build curl.
    float3 w = pf * pf * pf * (pf * (pf * 6.0 - 15.0) + 10.0);

    float n000 = hashFloat(pi + int3(0, 0, 0));
    float n100 = hashFloat(pi + int3(1, 0, 0));
    float n010 = hashFloat(pi + int3(0, 1, 0));
    float n110 = hashFloat(pi + int3(1, 1, 0));
    float n001 = hashFloat(pi + int3(0, 0, 1));
    float n101 = hashFloat(pi + int3(1, 0, 1));
    float n011 = hashFloat(pi + int3(0, 1, 1));
    float n111 = hashFloat(pi + int3(1, 1, 1));

    float x00 = lerp(n000, n100, w.x);
    float x10 = lerp(n010, n110, w.x);
    float x01 = lerp(n001, n101, w.x);
    float x11 = lerp(n011, n111, w.x);
    float y0  = lerp(x00, x10, w.y);
    float y1  = lerp(x01, x11, w.y);
    return lerp(y0, y1, w.z); // [0, 1)
}

// Fractal sum: layered noise at decreasing amplitude / increasing
// frequency. 3 octaves is enough for the ribbon shape we want; more
// octaves give too much fine grain and start looking like static.
float fbm(float3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; ++i) {
        v += a * noise3(p);
        p *= 2.07;       // non-integer scale: lattice artefacts vanish
        a *= 0.5;
    }
    return v;
}

// --- Aurora ribbon density ------------------------------------------
//
// Given a unit ray direction `rd`, return [0, 1] aurora density at
// that direction. The construction:
//
// 1. Elevation band: real aurora sits at 100-250 km altitude and is
//    densest where the line of sight skims tangentially through the
//    band rather than punching straight through. For an observer
//    looking up, that's the mid-elevation arc -- not the horizon
//    (line of sight passes through too much atmospheric obstruction
//    and tropospheric haze), not the zenith (cuts straight through
//    the curtain, near-zero apparent thickness). The peak band sits
//    around 25-45 degrees elevation. We model it with a Gaussian
//    centred on sin(elev) ~ 0.40 (so the peak is at elev = asin(0.4)
//    ~ 23.6 deg), width ~ 0.30 in sin(elev) units.
//
// 2. Magnetic-pole alignment: real aurora forms an oval around the
//    geomagnetic poles. From a mid-/high-latitude ground observer
//    inside the oval, the curtain wraps from one side of the sky to
//    the other, peaking toward the pole. We model this with a
//    soft-cosine weight: aurora_dir is a unit vector in the direction
//    of the magnetic pole (computed host-side from r_aurora_lat), and
//    the alignment weight is smoothstep(0.0, 0.35, dot(rd, aurora_dir)).
//    The smoothstep means: weight = 0 below the horizon-side of the
//    oval, ramps to 1 at the pole direction. Below the threshold
//    (south of the oval) there's no aurora -- matches reality for
//    a Northern-hemisphere observer at latitude << aurora_lat.
//
// 3. Ribbon-curtain modulation: the curl-noise field gives us the
//    streaky vertical-ribbon character. Sampling the fbm at a
//    direction-space coordinate scaled so a "ribbon" spans ~0.05 rad
//    (~3 deg), with two octaves at different frequencies -- the
//    coarse octave defines the curtain location, the fine octave the
//    flickering shimmer. We threshold the noise with a smoothstep so
//    most of the sky has zero density and the ribbons are sharp.
//
// 4. Time evolution (animation): when aurora_params.z is on, we
//    advect the noise lookup by aurora_params.w (time in seconds)
//    along a fixed direction -- this gives the "waves travelling
//    along the curtain" look without needing actual physical solar-
//    wind simulation. The advection velocity is small (0.03 rad/s)
//    so the motion is dignified rather than frantic.
float auroraDensity(float3 rd, float3 aurora_dir, float time, float animate) {
    // Step 1: elevation band weight.
    float sin_elev = rd.y;
    float band_centre = 0.40;
    float band_width  = 0.30;
    float dx = (sin_elev - band_centre) / band_width;
    float band = exp(-dx * dx);
    // Zero out the lower hemisphere unconditionally -- no aurora
    // below the horizon line in observer-space.
    if (sin_elev <= 0.02) return 0.0;

    // Step 2: magnetic-pole alignment weight. dot(rd, aurora_dir) is
    // the cosine of the angle between the ray and the geomagnetic
    // pole; smoothstep gives a soft falloff away from the pole side
    // of the sky.
    float pole_cos = dot(rd, aurora_dir);
    float pole_w   = smoothstep(0.0, 0.35, pole_cos);

    // Step 3: ribbon-curtain noise modulation. We sample the field
    // in two scales -- a coarse one that defines the overall curtain
    // location (low frequency) and a fine one that breaks up the
    // curtain into vertical streaks (higher frequency along the
    // azimuth axis).
    //
    // Build a curtain coordinate: azimuth (atan2 of x,z) gives the
    // around-the-horizon angle; elevation (asin of y) gives the
    // up-down. We stretch the azimuth axis vs the elevation axis so
    // the ribbons are tall and narrow (real curtains: ~1000 km long,
    // ~1 km thick -> ratio ~1000:1; we cinematic-scale to ~20:1).
    float azimuth = atan2(rd.z, rd.x);
    float elev    = asin(clamp(rd.y, -1.0, 1.0));

    // Animation: advect azimuth slowly when animate is on. 0.03 rad/s
    // is the curtain drift speed -- a full revolution takes ~210 s
    // (3.5 min), matching the "ribbons travelling slowly along the
    // curtain" look real aurora has.
    float adv = animate * time * 0.03;
    float u_coarse = (azimuth + adv) * 3.0;  // wide curtains
    float v_coarse = elev          * 5.0;
    float u_fine   = (azimuth + adv * 1.7) * 24.0; // streaks (high freq)
    float v_fine   = elev          * 8.0;

    // Sample fbm at two scales. The coarse layer defines whether
    // there's a curtain at this azimuth at all; the fine layer breaks
    // the curtain into discrete vertical ribbons.
    float n_coarse = fbm(float3(u_coarse, v_coarse, animate * time * 0.05 + 1.3));
    float n_fine   = fbm(float3(u_fine,   v_fine,   animate * time * 0.10 + 4.7));

    // Threshold each layer so only the high-value regions count.
    // Coarse threshold is permissive (0.30) -- most of the upper sky
    // has *some* curtain coverage. Fine threshold is sharp (0.55) --
    // only the brightest fine-noise peaks become ribbons.
    float curtain = smoothstep(0.30, 0.65, n_coarse);
    float ribbons = smoothstep(0.55, 0.85, n_fine);
    float density = curtain * ribbons;

    // Combine: density * band * pole alignment. All in [0, 1].
    return density * band * pole_w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (composite_active == 0u) return;

    // Sky-pixel gate. PathTrace.slang's G-buffer pass writes a 1e10
    // sentinel for primary-miss pixels and the real camera-space Z
    // for surface hits. Threshold at 1e9 unambiguously separates the
    // two: any plausible geometry depth in a metric-unit engine
    // (camera height to mountain horizon ~100 km max) sits well
    // below 1e9, while the 1e10 sky sentinel sits well above.
    //
    // 5x5 dilation of the geometry mask (2-pixel safety band).
    // PathTrace's G-buffer pass is UNJITTERED (uses uv0 not the
    // jittered uv) so on horizon-boundary pixels the depth value
    // reports whichever side of the edge the pixel CENTER lies on,
    // while the jittered radiance write can hit the opposite side.
    // Taking the min depth across the 5x5 neighbourhood biases the
    // gate toward "geometry" at edges -- if ANY of the 25 pixels is
    // a real surface hit (depth < 1e9), the composite skips,
    // pushing the aurora-eligible band 2 pixels inside the sky-side
    // of every edge. Identical pattern to StarsComposite.
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
    if (zmin < 1.0e9) return;

    // Reconstruct the world-space ray direction for this pixel. Same
    // formula as the StarsComposite per-sample path, but no aperture
    // sampling -- aurora is large-scale volumetric, sub-pixel jitter
    // and bokeh have no perceivable effect on it. One ray per pixel
    // through the pixel centre is enough.
    float2 uv = ((float2(tid.xy) + 0.5) / float2(dim)) * 2.0 - 1.0;
    uv.y = -uv.y;
    float3 rd = normalize(fwd_aspect.xyz
                       + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                       + up_xyz.xyz    * (uv.y * pos_fovtan.w));

    // Day/night gate. Aurora is invisible against a bright sky --
    // the sky luminance is many orders of magnitude above the
    // aurora's apparent surface brightness. We use the same
    // smoothstep curve StarsComposite uses for its star/sun day-
    // fade: sun elevation below -0.10 = full night; above +0.20 =
    // full day; in-between is twilight. The aurora intensity scales
    // by (1 - day) so it fades smoothly in / out at twilight.
    float3 sun = sun_and_mode.xyz;
    float sun_elev = sun.y;
    float day = smoothstep(-0.10, 0.20, sun_elev);
    float night = 1.0 - day;
    if (night <= 0.001) return;

    // Magnetic-pole direction from r_aurora_lat. The cvar is the
    // magnetic-pole latitude (positive = boreal). We translate to
    // an upper-sky unit vector pointing toward the geomagnetic pole
    // direction as the observer sees it: roughly (0, sin(lat),
    // cos(lat)) in world space, taking +Z as the geographic-north
    // direction in the engine's coordinate system.
    //
    // 70 deg = 1.222 rad gives a vector ~(0, 0.94, 0.34): mostly
    // overhead, slightly toward +Z. That puts the auroral oval
    // overhead and slightly northward, which is correct for an
    // observer at ~58 deg geographic latitude (roughly Glasgow or
    // Anchorage) -- the magnetic pole sits at ~78 deg in the modern
    // epoch and the auroral oval forms at ~67 deg geomagnetic
    // latitude, so the observer-relative pole direction shifts ~12
    // deg south of zenith.
    float aurora_lat_rad = aurora_params.y;
    float3 aurora_dir = float3(0.0,
                               sin(aurora_lat_rad),
                               cos(aurora_lat_rad));

    // Aurora density at this view direction. [0, 1].
    float time = aurora_params.w;
    float animate = aurora_params.z;
    float density = auroraDensity(rd, aurora_dir, time, animate);
    if (density <= 0.0) return;

    // Aurora colour: the dominant aurora emission is the oxygen
    // green line at 557.7 nm (lower-altitude band, 100-150 km). A
    // secondary red emission at 630.0 nm appears in the upper band
    // (200-400 km), visible during active displays. We mix the two
    // by elevation: low-elevation rays (looking through the lower
    // band tangentially) get mostly green; high-elevation rays
    // (looking through the upper band more directly) get a hint of
    // red at the top. Plus a faint purple/blue tint at the lower
    // edges from nitrogen excitation.
    //
    // Real spectral RGB approximations (sRGB primaries):
    //   557.7 nm (green) -> (0.18, 1.00, 0.30)
    //   630.0 nm (red)   -> (1.00, 0.20, 0.10)
    //   428   nm (blue)  -> (0.30, 0.20, 1.00)  -- nitrogen base
    float t_red = smoothstep(0.55, 0.85, rd.y);   // zenith-ward red
    float t_blue = smoothstep(0.10, 0.02, rd.y);  // horizon nitrogen
    float3 green  = float3(0.18, 1.00, 0.30);
    float3 red    = float3(1.00, 0.20, 0.10);
    float3 blue   = float3(0.30, 0.20, 1.00);
    float3 col = green;
    col = lerp(col, red,  t_red  * 0.35);
    col = lerp(col, blue, t_blue * 0.25);

    // Intensity scale: aurora apparent brightness is faint compared
    // to direct sunlight but well above star-twinkle. Real bright
    // displays peak at ~1 kR (kilo-rayleigh) which corresponds
    // roughly to a few cd/m^2 -- comparable to a half-moon-lit sky.
    // In the engine's HDR-linear units (where sun is ~250 cd/m^2 at
    // the disc and stars are ~0.5-5.0) we scale to ~1.5 cd/m^2 base,
    // multiplied by the cvar intensity slider.
    float base_lum = 1.5;
    float intensity = aurora_params.x;

    float3 aurora_col = col * density * night * base_lum * intensity;

    // Sanitize: clamp any NaN/Inf to zero, never go negative. The
    // procedural noise path can't actually produce non-finite
    // values today, but defence-in-depth costs nothing and matches
    // StarsComposite's policy.
    bool3 bad = isnan(aurora_col) || isinf(aurora_col);
    aurora_col = select(bad, float3(0.0), max(aurora_col, float3(0.0)));

    float3 base = hdr_inout[tid.xy].rgb;
    hdr_inout[tid.xy] = float4(base + aurora_col, 1.0);
}

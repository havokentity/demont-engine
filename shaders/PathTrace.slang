// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Unified path tracer.
//
// One kernel renders everything: a dynamic list of analytic primitives
// (spheres, planes) AND triangle meshes via TLAS. Each path bounce
// finds the nearest hit across all surface types, then dispatches on
// the surface's material (Lambert / metal / dielectric).
//
// Bindings (Slang/MSL maps these to MTL buffer slots starting at 0 for
// the AS, then sequential storage buffers, with push constants last):
//   slot 0 (texture) : swapchain (BGRA8Unorm) -- tonemapped LDR
//   slot 1 (texture) : accum_hdr (RGBA32F)    -- linear average + sample count in .a
//   slot 2 (accel)   : scene_tlas             -- mesh BLAS instances
//   slot 3 (buffer)  : mesh_positions         -- raw float[] xyz packed
//   slot 4 (buffer)  : mesh_indices           -- uint[] indices, 3 per tri
//   slot 5 (buffer)  : primitives             -- 3 float4 per primitive (see CPU side)
//
// `mesh_positions` is `float`-typed (not `float3`) because Slang emits
// StructuredBuffer<float3> with a 16-byte stride per element on MSL,
// while the buffer we upload is tightly packed 12-byte float3.

[[vk::binding(0, 0)]] RWTexture2D<float4> output;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_hdr;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure scene_tlas;
[[vk::binding(3, 0)]] StructuredBuffer<float>  mesh_positions;
[[vk::binding(4, 0)]] StructuredBuffer<uint>   mesh_indices;
[[vk::binding(5, 0)]] StructuredBuffer<float4> primitives;
// P10 denoiser G-buffer outputs. Only written when denoiser_enabled != 0.
[[vk::binding(6, 0)]] RWTexture2D<float4> denoise_color;   // RGBA16F per-frame HDR (NOT accumulated)
[[vk::binding(7, 0)]] RWTexture2D<float>  depth_tex;       // R32F  linear camera-space depth
[[vk::binding(8, 0)]] RWTexture2D<float2> motion_tex;      // RG16F pixel-space (prev - curr) delta
// P11 environment map. RGBA32F lat-long HDR; sampled by ray direction
// when env_map_present != 0. Used for both skybox visibility and
// indirect lighting in one place (path tracing collapses skybox + IBL).
[[vk::binding(9, 0)]] RWTexture2D<float4> env_map;

// P11 MIS / NEE: precomputed CDFs over env_map luminance. marginal_cdf
// is length H (one entry per row); conditional_cdf is length H*W
// (per-row CDF over columns). Both monotonic, ending at 1.0. Used for
// importance-sampled direct-light samples at Lambert hits.
[[vk::binding(10, 0)]] StructuredBuffer<float> marginal_cdf;
[[vk::binding(11, 0)]] StructuredBuffer<float> conditional_cdf;

// Yale BSC starmap: an RGBA16F equirectangular texture in the J2000
// equatorial frame, rasterised once at startup from BSC5.dat. The host
// pushes a per-frame world->J2000 rotation matrix; the shader samples
// this texture additively when the sun is below the horizon.
[[vk::binding(12, 0)]] RWTexture2D<float4> star_map;

[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    uint   frame_index;
    uint   reset_accum;
    uint   max_bounces;
    uint   tlas_present;     // 1 if a CSG mesh is bound, 0 -> skip RayQuery
    uint   prim_count;       // number of analytic primitives in `primitives`
    uint   spp;              // samples per pixel this dispatch (>=1)
    uint   denoiser_enabled; // 1 -> write G-buffer + denoise_color for MetalFX
    uint   env_map_present;  // 1 -> sample env_map for sky; 0 -> procedural gradient
    float2 halton_jitter;    // per-frame sub-pixel offset in [-0.5,0.5]
    float  env_intensity;    // scalar multiplier on env_map samples
    float  env_total_luminance;  // sum of luminance(u,v)*sin(theta) -- PDF normalizer
    float4x4 curr_view_proj;
    float4x4 prev_view_proj;
    // .xyz = unit vector toward sun, .w = sky mode encoded as a float
    // (0 = gradient, 1 = hdri, 2 = procedural). Packed as float4 so the
    // host's uint sky_mode follows the 16-byte alignment Slang puts
    // after a float3 in MSL.
    float4 sun_and_mode;
    // .x = manual HDR exposure multiplier applied before ACES tonemap.
    // .y = 1.0 if legacy procedural starfield is active.
    // .z = 1.0 if BSC starmap should be sampled.
    // .w = 1.0 if stars should twinkle (atmospheric scintillation fake).
    float4 exposure_pad;
    // World->J2000 rotation, row-major. The 3x3 rotation only uses
    // 9 floats; the .w lanes are repurposed as engine flags:
    //   row0.w = HDR pipeline flag (1 = path tracer writes raw HDR
    //            to denoise_color, post-pass applies exposure+ACES;
    //            0 = legacy pre-tonemap).
    //   row1.w = refractive-shadow-ray budget for caustics
    //            (0 = treat dielectrics as opaque blockers,
    //            positive = max dielectric bounces a NEE shadow ray
    //            may chain through). Capped to 16 by the engine.
    //   row2.w = unused / reserved.
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
    // Thin-lens DOF parameters.
    //   .x = aperture radius (0 = pinhole, no DOF).
    //   .y = focal distance (world units).
    //   .z = aperture blade count (0 = round disk; 3..16 = polygonal
    //        iris -> polygonal bokeh).
    //   .w reserved.
    float4 dof_params;
    // Volumetric single-scatter parameters.
    //   .x = extinction coefficient (per world unit). 0 disables.
    //   .y = Henyey-Greenstein anisotropy g in [-1, 1]. +0.7 gives
    //        forward-scattering atmosphere; 0 = isotropic (fog);
    //        -0.5 = back-scattering (rare materials).
    //   .z = intensity multiplier on the final scatter contribution.
    //   .w = march sample count (cast to int; 4..64 typical).
    float4 vol_params;
    // Volumetric cloud parameters. When clouds_p1.w > 0 the volumetric
    // march multiplies sigma_t by a position-dependent cloud density
    // sampled from a procedural fbm noise field.
    //   clouds_p1: (base_y, top_y, coverage[0..1], peak_density)
    //              base_y / top_y are world-space heights bracketing
    //              the cloud layer; pos.y outside this range = 0.
    //              coverage = sky-fill fraction; peak_density is the
    //              extinction at the densest interior point.
    //   clouds_p2: (wind_x, wind_z, freq, time_seconds)
    //              wind translates the noise input by wind*time so
    //              the cloud field drifts. freq is cycles per world
    //              unit (~0.005 = ~200m features). time advances per
    //              frame to animate the field.
    //   clouds_p3: (seed_offset_x, seed_offset_z, detail_amount, _)
    //              seed shifts the noise hash so different seeds
    //              produce visually distinct cloud patterns at the
    //              same preset (each "day" can have its own seed).
    //              detail_amount in [0,1] blends in a higher-frequency
    //              octave for wispy edges.
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
};

// --- Procedural cloud field --------------------------------------------
//
// Hash-based 3D value noise + fbm. Cheaper than gradient (Perlin)
// noise -- the smoothstep filter step gives enough visual continuity
// for cloud bodies at this scale, and we don't need C2 continuity for
// volumetric sampling. ~30 ALU per sample × ~4 octaves × 16 march
// samples × 1080p = ~6 GFLOP/frame on the worst march, well under
// budget on M4 Max.

float cloud_hash(float3 p) {
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float cloud_value_noise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    // Quintic Hermite interpolation: f^3 (6f^2 - 15f + 10). Zero
    // first AND second derivatives at f=0 and f=1, so the noise has
    // no visible cell boundaries even at heavy fbm summation. Cubic
    // smoothstep (3f^2 - 2f^3) leaks a faint grid through the haze.
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = cloud_hash(i + float3(0, 0, 0));
    float b = cloud_hash(i + float3(1, 0, 0));
    float c = cloud_hash(i + float3(0, 1, 0));
    float d = cloud_hash(i + float3(1, 1, 0));
    float e = cloud_hash(i + float3(0, 0, 1));
    float fx = cloud_hash(i + float3(1, 0, 1));
    float gx = cloud_hash(i + float3(0, 1, 1));
    float hx = cloud_hash(i + float3(1, 1, 1));
    return lerp(lerp(lerp(a,  b,  f.x), lerp(c,  d,  f.x), f.y),
                lerp(lerp(e,  fx, f.x), lerp(gx, hx, f.x), f.y),
                f.z);
}

float cloud_fbm(float3 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    [unroll(6)]
    for (int i = 0; i < octaves; ++i) {
        v += a * cloud_value_noise(p);
        p *= 2.02;          // non-integer freq mult avoids axis-aligned artefacts
        a *= 0.5;
    }
    return v;
}

// Density at a world-space point. Returns 0 outside the cloud layer
// or where the noise threshold isn't met. The volumetric march
// multiplies this by sigma_t to vary extinction along the ray.
//
// `octaves` parameterises the fbm complexity: the primary cloud march
// uses 6 (full visual quality), shadow rays use 3 (rough density
// estimate for Beer-Lambert is enough; saves ~50% of noise lookups
// on the dominant shadow path).
float cloud_density_at(float3 pos,
                       float4 cp1, float4 cp2, float4 cp3,
                       int octaves) {
    if (cp1.w <= 0.0) return 0.0;
    float base_y = cp1.x;
    float top_y  = cp1.y;
    if (pos.y < base_y || pos.y > top_y) return 0.0;
    float h_norm = (pos.y - base_y) / max(top_y - base_y, 1e-3);
    // Height profile: ramp up from the bottom, fall off before the top.
    // Peaks around h_norm = 0.4 -- biases toward dense lower volumes,
    // wispy upper volumes (cumulus shape).
    float h_profile = smoothstep(0.0, 0.2, h_norm)
                    * (1.0 - smoothstep(0.6, 1.0, h_norm));
    if (h_profile <= 0.0) return 0.0;
    // Wind translation + per-day seed offset.
    float3 sp = pos
              + float3(cp2.x * cp2.w, 0.0, cp2.y * cp2.w)
              + float3(cp3.x, 0.0, cp3.y);
    sp *= cp2.z;
    // Cumulus body: caller-controlled octaves of quintic-interpolated
    // value-noise fbm. Higher octaves -> more wispy detail, more cost.
    float low = cloud_fbm(sp, octaves);
    // Smooth coverage transition. smoothstep(1-cov, 1, low) gives a
    // soft S-curve from 0 (no cloud) to 1 (max cloud) over the
    // noise-value band, instead of the hard linear ramp the previous
    // saturate((low - (1-cov)) / cov) produced. Soft edges = light
    // partially passes through wispy outer regions.
    float coverage = saturate(cp1.z);
    float density  = smoothstep(1.0 - coverage, 1.0, low);
    if (density <= 0.0) return 0.0;
    // Detail subtract: a higher-frequency fbm erodes edges so the
    // cloud silhouette has wispy fingers rather than smooth blobs.
    if (cp3.z > 0.0) {
        float hi = cloud_fbm(sp * 4.0, 3);
        density = saturate(density - hi * cp3.z);
    }
    // Internal variation: secondary fbm modulates density inside the
    // cloud so a thick body isn't uniform white. Real cumulus has
    // brighter rims and darker cores; this gives the volume some
    // texture without being noisy.
    float internal = 0.55 + 0.45 * cloud_fbm(sp * 1.7, 3);
    density *= internal;
    // cp1.w is peak sigma_t per metre. Real meteorology:
    //   light cumulus    ~0.03-0.05 /m
    //   typical cumulus  ~0.05-0.10
    //   stratus / nimbus ~0.04-0.08
    //   storm / Cb       ~0.10-0.30
    return density * h_profile * cp1.w;
}

static const uint MAT_LAMBERT    = 0u;
static const uint MAT_METAL      = 1u;
static const uint MAT_DIELECTRIC = 2u;

static const uint PRIM_SPHERE = 0u;
static const uint PRIM_PLANE  = 1u;

static const float3 LIGHT_DIR    = float3(0.4, 1.0, 0.3);
static const float3 MESH_ALBEDO  = float3(1.00, 0.85, 0.45);   // gold metal default

uint pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}
float randf(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed & 0x00FFFFFFu) / 16777216.0;
}

float3 sampleCosineHemisphere(float3 n, inout uint seed) {
    float u1 = randf(seed);
    float u2 = randf(seed);
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1));
    float3 ax = abs(n.x) > 0.9 ? float3(0,1,0) : float3(1,0,0);
    float3 t  = normalize(cross(n, ax));
    float3 b  = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

// The sun is a disc, not a point. Half-angular-radius from Earth is
// 0.00465 rad (~0.266 deg). Shadow rays should sample a random
// direction inside that disc instead of the centre direction; over
// the temporal accumulator this gives soft penumbrae naturally
// (clouds, foliage, hard objects all gain real-physics-soft edges).
//
// This is *not* a softening hack -- the sun really is a disc and a
// point sample is the wrong physics. We're just doing the integral
// the path tracer was missing.
static const float kSunAngularRadius = 0.00465;

float3 sampleSunDisc(float3 sun_dir, inout uint seed) {
    float u1  = randf(seed);
    float u2  = randf(seed);
    // Uniform disc sample, scaled by sun's angular radius (small angle
    // so this approximates a uniform solid-angle sample on the disc).
    float r   = sqrt(u1) * kSunAngularRadius;
    float phi = 6.28318530718 * u2;
    float2 d  = float2(cos(phi), sin(phi)) * r;
    // Build an orthonormal basis around sun_dir and offset.
    float3 ax = abs(sun_dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t  = normalize(cross(ax, sun_dir));
    float3 b  = cross(sun_dir, t);
    return normalize(sun_dir + t * d.x + b * d.y);
}

float fresnelSchlick(float cos_theta, float r0) {
    float m = 1.0 - cos_theta;
    return r0 + (1.0 - r0) * m * m * m * m * m;
}

bool intersectSphere(float3 ro, float3 rd, float3 c, float rad, out float t) {
    float3 oc = ro - c;
    float b   = dot(oc, rd);
    float k   = dot(oc, oc) - rad * rad;
    float h   = b * b - k;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > 1e-3) ? t0 : t1;
    return t > 1e-3;
}
bool intersectPlane(float3 ro, float3 rd, float3 n, float d, out float t) {
    float denom = dot(rd, n);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, n) + d) / denom;
    return t > 1e-3;
}

struct HitInfo {
    bool   hit;
    float  t;
    float3 normal;
    uint   mat;
    float3 albedo;
    float  roughness;
    float  ior;
};

HitInfo traceScene(float3 ro, float3 rd) {
    HitInfo h;
    h.hit = false; h.t = 1e30;
    h.normal = float3(0,1,0); h.mat = MAT_LAMBERT;
    h.albedo = float3(0.5); h.roughness = 0.0; h.ior = 1.0;

    // Analytic primitives. Each occupies 3 consecutive float4s in
    // `primitives`:
    //   v0 = (xyz: center / normal,  w: radius / d)
    //   v1 = (rgb: albedo,           a: roughness)
    //   v2 = (x: type,  y: material, z: ior, w: pad)
    for (uint i = 0u; i < prim_count; ++i) {
        float4 v0 = primitives[i * 3u + 0u];
        float4 v1 = primitives[i * 3u + 1u];
        float4 v2 = primitives[i * 3u + 2u];
        uint type = uint(v2.x);
        float t;
        if (type == PRIM_SPHERE) {
            if (intersectSphere(ro, rd, v0.xyz, v0.w, t) && t < h.t) {
                h.hit = true; h.t = t;
                h.normal = normalize(ro + rd * t - v0.xyz);
                h.mat = uint(v2.y);
                h.albedo = v1.rgb;
                h.roughness = v1.a;
                h.ior = v2.z;
            }
        } else if (type == PRIM_PLANE) {
            if (intersectPlane(ro, rd, v0.xyz, v0.w, t) && t < h.t) {
                h.hit = true; h.t = t;
                h.normal = v0.xyz;
                h.mat = uint(v2.y);
                h.albedo = v1.rgb;
                h.roughness = v1.a;
                h.ior = v2.z;
            }
        }
    }

    // Triangle meshes via TLAS. Skipped when the host signals there's
    // no mesh bound (tlas_present == 0) -- accessing scene_tlas in
    // that case would be undefined.
    if (tlas_present != 0u) {
        RayDesc r;
        r.Origin    = ro;
        r.Direction = rd;
        r.TMin      = 1e-3;
        r.TMax      = h.t;     // limit search to current closest

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, r);
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float t_mesh = q.CommittedRayT();
            if (t_mesh < h.t) {
                uint prim = q.CommittedPrimitiveIndex();
                uint i0 = mesh_indices[prim * 3 + 0];
                uint i1 = mesh_indices[prim * 3 + 1];
                uint i2 = mesh_indices[prim * 3 + 2];
                float3 v0 = float3(mesh_positions[i0 * 3 + 0],
                                   mesh_positions[i0 * 3 + 1],
                                   mesh_positions[i0 * 3 + 2]);
                float3 v1 = float3(mesh_positions[i1 * 3 + 0],
                                   mesh_positions[i1 * 3 + 1],
                                   mesh_positions[i1 * 3 + 2]);
                float3 v2 = float3(mesh_positions[i2 * 3 + 0],
                                   mesh_positions[i2 * 3 + 1],
                                   mesh_positions[i2 * 3 + 2]);
                float3 obj_n = normalize(cross(v1 - v0, v2 - v0));
                float3x3 wfm = (float3x3)q.CommittedObjectToWorld4x3();
                h.hit = true; h.t = t_mesh;
                h.normal = normalize(mul(wfm, obj_n));
                h.mat = MAT_METAL;
                h.albedo = MESH_ALBEDO;
                h.roughness = 0.05;
                h.ior = 1.0;
            }
        }
    }
    return h;
}

// Hand-tuned analytic sky inspired by Preetham/Hosek-Wilkie but ~30
// lines instead of ~150. Drives day/sunset/night transitions from
// `sun_dir.y` (sun's vertical component above horizon). Result is HDR
// linear; the sun disk is around 50x normal sky brightness so it
// punches through ACES tonemap and reads as "the sun" rather than a
// bright dot.
float3 procSky(float3 rd, float3 sun) {
    float cosTheta = clamp(rd.y, -1.0, 1.0);    // 1=zenith, 0=horizon, -1=nadir
    float cosGamma = clamp(dot(rd, sun), -1.0, 1.0);
    float sun_elev = sun.y;                     // sin(elevation)

    // Day vs night blend: smoothstep on sun elevation across the
    // civil-twilight band.
    float day = smoothstep(-0.10, 0.20, sun_elev);

    // Sunset glow: peaks when sun is just at the horizon.
    float sunset = smoothstep(0.30, -0.05, sun_elev) *
                   smoothstep(-0.20, 0.05, sun_elev);

    // Day sky palette.
    float3 zenith_day  = float3(0.32, 0.55, 0.92);
    float3 horizon_day = float3(0.78, 0.88, 1.00);
    // Sunset palette pulls toward orange/red on the side of the sun.
    float side_factor  = smoothstep(0.0, 0.7, dot(normalize(float3(rd.x, 0, rd.z)),
                                                  normalize(float3(sun.x, 0, sun.z))));
    float3 horizon_sunset = lerp(float3(0.95, 0.45, 0.20),
                                 float3(1.00, 0.65, 0.30), side_factor);
    float3 horizon_col    = lerp(horizon_day, horizon_sunset, sunset);
    float3 zenith_col     = lerp(zenith_day,  float3(0.18, 0.18, 0.30), sunset * 0.6);

    // Vertical gradient from horizon to zenith.
    float t = smoothstep(0.0, 0.40, cosTheta);
    float3 sky = lerp(horizon_col, zenith_col, t);

    // Apply day/night brightness curve. Night is a deep navy with
    // a faint horizon line, no sun.
    float3 night = float3(0.005, 0.010, 0.030) +
                   smoothstep(0.0, 0.05, abs(cosTheta - 0.0)) * 0.0;
    sky = lerp(night, sky, day);

    // BSC starmap (real Yale Bright Star Catalog, J2000 frame). Sampled
    // additively whenever the host has uploaded the texture and we're
    // looking above the horizon. Falls back to the procedural cell-hash
    // scatter below when the starmap isn't available -- the host gates
    // exposure_pad.z (BSC) and .y (procedural) to be mutually exclusive.
    if (cosTheta > -0.05 && exposure_pad.z > 0.5 && day < 0.6) {
        // World ray -> J2000 unit vector via the per-frame rotation.
        float3 j;
        j.x = dot(w2j_row0.xyz, rd);
        j.y = dot(w2j_row1.xyz, rd);
        j.z = dot(w2j_row2.xyz, rd);
        // Equirectangular: u from atan2(y, x) (RA), v from asin(z) (dec).
        // Match the host rasteriser: u = ra/360 wraps, v = (90-dec)/180.
        float u = atan2(j.y, j.x) * (1.0 / 6.28318530718);
        if (u < 0.0) u += 1.0;
        float v = (1.5707963267948966 - asin(clamp(j.z, -1.0, 1.0))) * (1.0 / 3.14159265358979);
        uint2 dim;
        star_map.GetDimensions(dim.x, dim.y);
        int xi = int(floor(u * float(dim.x))) % int(dim.x);
        if (xi < 0) xi += int(dim.x);
        int yi = clamp(int(floor(v * float(dim.y))), 0, int(dim.y) - 1);
        float3 stars = star_map[uint2(uint(xi), uint(yi))].rgb;
        // Twinkle. Hash on a coarse-quantised J2000 direction (~1.5
        // deg bins) so every texel covering a single star agrees on
        // phase + frequency, and the whole star modulates as one
        // unit. (Hashing on raw texel coords made adjacent texels of
        // a multi-texel bright star twinkle out-of-phase, which
        // averaged flat to the eye.) +/-45%% amplitude over a 3-8 Hz
        // sinusoid; reads as broadband atmospheric scintillation.
        if (exposure_pad.w > 0.5) {
            int3 q  = int3(floor(j * 38.0));   // ~1.5 deg bins
            uint qh = uint(q.x + 128) * 65536u
                    + uint(q.y + 128) * 256u
                    + uint(q.z + 128);
            uint h    = pcgHash(qh + 0xb7e1u);
            float ph  = float(h & 0xFFFFu) * (1.0 / 65536.0);
            float fhz = 3.0 + float((h >> 16) & 0xFFu) * (5.0 / 256.0);
            float t   = float(frame_index) * (1.0 / 60.0);
            float w   = 0.55 + 0.45 * sin(6.28318530718 * (ph + t * fhz));
            stars *= w;
        }
        // Fade with day so stars vanish at sunrise.
        sky += stars * (1.0 - day);
    } else if (cosTheta > -0.05 && exposure_pad.y > 0.5 && day < 0.6) {
        // Cells sized so each one is ~3-5 pixels wide on a 1080p frame
        // -- smaller cells make stars sub-pixel and they vanish.
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
            sky += intensity * starColor * (1.0 - day);
        }
    }

    // Sun disk: very tight angular radius (~0.25 deg ~= cos > 0.99996).
    float disk = smoothstep(0.99996, 0.99999, cosGamma);
    float3 sun_color_high = float3(1.00, 0.95, 0.85);
    float3 sun_color_low  = float3(1.00, 0.55, 0.20);
    float3 sun_col = lerp(sun_color_low, sun_color_high, day);
    sky += disk * 50.0 * sun_col * day;

    // Soft halo around the sun (Mie-ish, falls off rapidly).
    float halo = pow(max(0.0, cosGamma), 80.0);
    sky += halo * sun_col * 0.30 * day;
    // Wider, dimmer secondary scatter (Rayleigh-tinted) extends the
    // sunset orange across more of the sky.
    float wide = pow(max(0.0, cosGamma), 6.0);
    sky += wide * sun_col * sunset * 0.50;

    return max(sky, float3(0.0));
}

float3 skyColor(float3 rd) {
    uint mode = uint(sun_and_mode.w);
    if (mode == 2u) {
        return procSky(rd, sun_and_mode.xyz);
    }
    if (env_map_present != 0u) {
        // Lat-long lookup: u from atan2 of horizontal plane, v from
        // arccos of vertical component. (u, v) in [0,1].
        uint2 dim;
        env_map.GetDimensions(dim.x, dim.y);
        float u = atan2(rd.z, rd.x) * (1.0 / 6.28318530718) + 0.5;
        float v = acos(clamp(rd.y, -1.0, 1.0)) * (1.0 / 3.14159265359);
        // Bilinear by hand: nearest-neighbour gives visible quantisation
        // around the sun where the env-map is brightest.
        float2 fpix = float2(u * float(dim.x), v * float(dim.y)) - 0.5;
        int2  ipix = int2(floor(fpix));
        float2 f   = fpix - float2(ipix);
        int x0 = (ipix.x % int(dim.x) + int(dim.x)) % int(dim.x);
        int x1 = (x0 + 1) % int(dim.x);
        int y0 = clamp(ipix.y, 0, int(dim.y) - 1);
        int y1 = clamp(ipix.y + 1, 0, int(dim.y) - 1);
        float3 c00 = env_map[uint2(x0, y0)].rgb;
        float3 c10 = env_map[uint2(x1, y0)].rgb;
        float3 c01 = env_map[uint2(x0, y1)].rgb;
        float3 c11 = env_map[uint2(x1, y1)].rgb;
        float3 sky = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
        return sky * env_intensity;
    }
    float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
    return lerp(float3(0.50, 0.55, 0.62), float3(1.0, 1.0, 1.0), t);
}

float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c*(a*c+b))/(c*(d*c+e)+f));
}

// ---------- env-map importance sampling (NEE for Lambert) ----------------
struct EnvSample {
    float3 dir;
    float3 color;     // env_map sample value times env_intensity
    float  pdf;       // solid-angle PDF for this direction
};

// Binary search a monotonically-non-decreasing CDF in [base, base+count) for
// the smallest index i such that cdf[base+i] >= u. CDF expected to end at 1.0.
uint cdfSearch(uint base, uint count, float u) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (conditional_cdf[base + mid] < u) lo = mid + 1;
        else                                  hi = mid;
    }
    return min(lo, count - 1);
}
uint cdfSearchMarginal(uint count, float u) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (marginal_cdf[mid] < u) lo = mid + 1;
        else                        hi = mid;
    }
    return min(lo, count - 1);
}

EnvSample sampleEnvMap(float2 xi) {
    EnvSample s;
    s.dir = float3(0,1,0); s.color = float3(0); s.pdf = 0.0;
    if (env_map_present == 0u || env_total_luminance <= 0.0) return s;

    uint W, H;
    env_map.GetDimensions(W, H);

    uint v_idx = cdfSearchMarginal(H, xi.y);
    uint u_idx = cdfSearch(v_idx * W, W, xi.x);

    float u = (float(u_idx) + 0.5) / float(W);
    float v = (float(v_idx) + 0.5) / float(H);

    float theta = v * 3.14159265358979;
    float phi   = u * 6.28318530717958 - 3.14159265358979;
    float sin_theta = sin(theta);

    s.dir = float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));

    float3 raw  = env_map[uint2(u_idx, v_idx)].rgb;
    s.color     = raw * env_intensity;

    float lum   = 0.2126 * raw.r + 0.7152 * raw.g + 0.0722 * raw.b;
    // p(u,v) in pixel-space = lum*sin / total. Solid-angle Jacobian for
    // lat-long: dω = (2π² / (W*H)) * sin(θ) du dv. So p(ω) = p(u,v) /
    // jacobian = (lum * W * H) / (2π² * total).
    if (sin_theta > 1e-6) {
        s.pdf = (float(lum) * float(W * H)) /
                (2.0 * 3.14159265358979 * 3.14159265358979 * env_total_luminance);
    }
    return s;
}

// RGB transmittance along a shadow ray. 1.0 = unoccluded, 0.0 = blocked
// by an opaque hit. Dielectric hits don't block: the ray refracts
// through (in -> out), continuing toward the light, multiplied by the
// dielectric's tint (cheap stand-in for Beer's-law absorption). This
// is what turns "shadow under a glass diamond" into actual caustic
// patterns -- the focal lensing emerges from the geometry the
// refracted shadow ray traces, not from any fancy sampling.
//
// Optical depth through the cloud layer along [ro, ro + rd*t_max].
// Returns 0 when clouds are off or the ray misses the layer; otherwise
// integrates cloud_density_at across the ray-slab intersection. Used
// by transmittance() to give clouds genuine shadow-casting behaviour:
// any NEE shadow ray that grazes a dense cloud loses light.
float cloud_optical_depth(float3 ro, float3 rd, float t_max) {
    if (clouds_p1.w <= 0.0) return 0.0;
    float base_y = clouds_p1.x;
    float top_y  = clouds_p1.y;
    float ry = rd.y;
    float t_in, t_out;
    if (abs(ry) > 1e-6) {
        float t_b = (base_y - ro.y) / ry;
        float t_t = (top_y  - ro.y) / ry;
        t_in  = max(0.0, min(t_b, t_t));
        t_out = max(t_b, t_t);
    } else {
        if (ro.y >= base_y && ro.y <= top_y) {
            t_in  = 0.0;
            t_out = t_max;
        } else {
            return 0.0;
        }
    }
    t_out = min(t_out, t_max);
    // Cap shadow march distance: a near-horizon ray crossing the cloud
    // layer at very shallow elevation can span hundreds of kilometres,
    // far more than the cloud field's coherence length. 20km is plenty
    // to capture the shadow of any reasonable cloud bank in front of
    // the sun, and limits the sample budget.
    if (t_out - t_in > 20000.0) t_in = t_out - 20000.0;
    if (t_out <= t_in) return 0.0;
    // 4 coarse samples + 3 fbm octaves for the shadow ray. Beer-Lambert
    // exp(-OD) only needs a rough density estimate to look right.
    const int Ns = 4;
    float dt = (t_out - t_in) / float(Ns);
    float od = 0.0;
    [unroll] for (int i = 0; i < Ns; ++i) {
        float t = t_in + (float(i) + 0.5) * dt;
        float3 sp = ro + rd * t;
        od += cloud_density_at(sp, clouds_p1, clouds_p2, clouds_p3, 3) * dt;
    }
    return od;
}

// Up to kMaxRefract dielectric bounces; beyond that the ray gives up
// (returns 0) so a pathological all-glass scene can't loop forever.
// TIR also returns 0 (light can't pass that interface).
float3 transmittance(float3 ro, float3 rd, float t_max) {
    // Cloud optical depth along the original (un-refracted) ray. Real
    // clouds live in a layer high above the scene -- approximating the
    // shadow as the straight-line cloud path is fine because no glass
    // sits inside the cloud layer for the refracted segment to matter.
    float3 cloud_t = float3(exp(-cloud_optical_depth(ro, rd, t_max)));
    // Engine packs the per-frame refraction budget into w2j_row1.w
    // (0 = caustics disabled, treat dielectrics as opaque shadow
    // blockers; >0 = max dielectric bounces a single shadow ray may
    // chain through). Capped to 16 in the engine for safety.
    int kMaxRefract = int(w2j_row1.w);
    float3 t = float3(1.0);
    float3 o = ro;
    float3 d = rd;
    for (int i = 0; i <= kMaxRefract; ++i) {
        HitInfo h = traceScene(o, d);
        if (!h.hit || h.t >= t_max) return t * cloud_t;     // unoccluded
        if (h.mat != MAT_DIELECTRIC) return float3(0.0);    // opaque blocker
        if (i == kMaxRefract)        return float3(0.0);    // budget hit

        // Refract through the dielectric face. Same Snell + Fresnel
        // logic as the path-tracer's dielectric BRDF, but we keep
        // only the transmitted lobe (the reflected lobe doesn't reach
        // the light along this path). Fresnel is folded into the
        // transmittance so glancing angles correctly attenuate.
        float ndi   = dot(h.normal, d);
        float3 nf   = (ndi < 0.0) ? h.normal : -h.normal;
        float ior_a = (ndi < 0.0) ? 1.0   : h.ior;
        float ior_b = (ndi < 0.0) ? h.ior : 1.0;
        float eta   = ior_a / ior_b;
        float cos_i = abs(dot(nf, -d));
        float r0    = (ior_a - ior_b) / (ior_a + ior_b);
        r0 = r0 * r0;
        float fr    = fresnelSchlick(cos_i, r0);   // reflected fraction
        float3 refracted = refract(d, nf, eta);
        if (length(refracted) < 1e-3) return float3(0.0);   // TIR
        // Transmitted fraction times Beer-ish tint.
        t *= (1.0 - fr) * h.albedo;
        // Step past the interface, continue in the refracted direction.
        float advance = h.t + 1e-3;
        o = o + d * advance;
        d = normalize(refracted);
        t_max = max(t_max - advance, 0.0);
    }
    return float3(0.0);
}

// Convenience predicate around transmittance() for the metal /
// dielectric branches that just want the legacy boolean answer.
bool occluded(float3 ro, float3 rd, float t_max) {
    return all(transmittance(ro, rd, t_max) <= float3(0.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    uint2 dim;
    output.GetDimensions(dim.x, dim.y);
    if (any(tid >= dim)) return;

    uint seed = pcgHash(tid.x * 1973u + tid.y * 9277u + frame_index * 26699u);

    // P10 G-buffer pass (no shading): unjittered primary ray writes
    // depth + motion vectors for MetalFX TemporalDenoisedScaler.
    // Stable per-pixel data is what the denoiser wants; we don't
    // include the pixel jitter here.
    if (denoiser_enabled != 0u) {
        float2 uv0 = ((float2(tid) + 0.5) / float2(dim)) * 2.0 - 1.0;
        uv0.y = -uv0.y;
        float3 ro0 = pos_fovtan.xyz;
        float3 rd0 = normalize(fwd_aspect.xyz
                             + right_xyz.xyz * (uv0.x * fwd_aspect.w * pos_fovtan.w)
                             + up_xyz.xyz    * (uv0.y * pos_fovtan.w));
        HitInfo h0 = traceScene(ro0, rd0);

        float3 hit_world;
        float  depth_view;   // linear camera-space Z (world units along forward)
        if (h0.hit) {
            hit_world = ro0 + rd0 * h0.t;
            depth_view = h0.t * dot(rd0, normalize(fwd_aspect.xyz));
        } else {
            hit_world  = ro0 + rd0 * 1000.0;   // far point so sky still gets motion
            depth_view = 1000.0;
        }

        // Pixel-space motion via reprojection. We still need the
        // perspective matrices for this even though depth itself goes
        // to MetalFX in linear units.
        float4 pc = mul(curr_view_proj, float4(hit_world, 1.0));
        float4 pp = mul(prev_view_proj, float4(hit_world, 1.0));
        float2 cur_pix = (float2( pc.x, -pc.y) / max(pc.w, 1e-6) * 0.5 + 0.5) * float2(dim);
        float2 pre_pix = (float2( pp.x, -pp.y) / max(pp.w, 1e-6) * 0.5 + 0.5) * float2(dim);
        float2 motion  = pre_pix - cur_pix;

        depth_tex[tid]  = depth_view;
        motion_tex[tid] = motion;
    }

    // Multi-sample loop: average `spp` independent paths into a single
    // radiance estimate. spp>1 buys cleaner frames during motion at a
    // proportional GPU cost; spp=1 is the classic 1-spp interactive mode.
    uint sample_count = (spp == 0u) ? 1u : spp;
    float3 frame_radiance = float3(0.0);
    for (uint s = 0u; s < sample_count; ++s) {
        // When MetalFX is on, all pixels of the frame share the same
        // sub-pixel offset (Halton sequence picked on host) so the
        // denoiser can reconstruct the supersampled result. With the
        // denoiser off, per-pixel random jitter gives free AA via
        // accumulation. spp>1 always gets per-sample random offsets so
        // multi-sampling actually reduces variance.
        float2 jitter;
        if (denoiser_enabled != 0u && s == 0u) {
            jitter = halton_jitter;
        } else {
            jitter = float2(randf(seed), randf(seed)) - 0.5;
        }
        float2 uv = ((float2(tid) + 0.5 + jitter) / float2(dim)) * 2.0 - 1.0;
        uv.y = -uv.y;

        float3 ro = pos_fovtan.xyz;
        float3 rd = normalize(fwd_aspect.xyz
                           + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                           + up_xyz.xyz    * (uv.y * pos_fovtan.w));

        // Thin-lens DOF. Sample a point on the aperture, recompute
        // the ray to pass through the focal-plane intersection of
        // the original pinhole ray. Aperture sampling shape =
        // round disk for circular bokeh, regular polygon for
        // polygonal-iris bokeh (matching real lens aperture blades).
        // dof_params.x = aperture radius (0 = skip, pinhole).
        if (dof_params.x > 0.0) {
            float aperture = dof_params.x;
            float focal_d  = dof_params.y;
            int   blades   = int(dof_params.z + 0.5);
            float u1 = randf(seed), u2 = randf(seed);
            float2 ap;
            if (blades < 3) {
                // Concentric-disk mapping (Shirley-Chiu): square
                // (u1,u2) -> uniformly distributed unit disk.
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
                // Polygon sampling: pick a triangular wedge of the
                // regular n-gon, sample uniformly inside it. Bokeh
                // gets that classic 5-/6-/8-sided look.
                float n = float(blades);
                float wedge = 6.28318530717958 / n;
                int   i = int(floor(u1 * n));
                float r1 = frac(u1 * n);
                float r2 = u2;
                if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
                float a0 = float(i) * wedge;
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

        float3 throughput = float3(1.0);
        float3 radiance   = float3(0.0);
        // After a Lambert bounce we did NEE for direct env-map lighting,
        // so the next BRDF-bounce ray's miss contribution is already
        // counted -- skip on miss to avoid double counting.
        bool   skip_sky_on_miss = false;

        // Save primary-ray params for the volumetric march (single
        // scatter along the camera ray, evaluated after the bounce
        // loop). primary_t is the depth at which the camera ray hits
        // something; sky-miss defaults to 1000m so we still get
        // atmospheric haze along the full visible distance.
        float3 primary_ro = ro;
        float3 primary_rd = rd;
        // Sky-pixel default extends to ~500km so cirrus near the
        // horizon (8000-9500m altitude, needs ~500km horizontal travel
        // for 1 deg elevation rays) is reachable. The cloud march
        // separately caps its span at 30km, so this just unlocks the
        // ray-slab intersection to find the layer in the first place;
        // it doesn't multiply the actual march cost.
        float  primary_t  = 500000.0;
        bool   primary_t_recorded = false;
        bool   primary_hit = false;

        for (uint b = 0u; b < max_bounces; ++b) {
        HitInfo h = traceScene(ro, rd);
        if (b == 0u) {
            // Sky pixels (no hit) keep the 50km default so cloud
            // marches reach the layer; object hits override.
            primary_t = h.hit ? h.t : primary_t;
            primary_hit = h.hit;
            primary_t_recorded = true;
        }
        if (!h.hit) {
            if (!skip_sky_on_miss) {
                radiance += throughput * skyColor(rd);
            }
            break;
        }

        float3 hit_pt = ro + rd * h.t;
        float  ndi = dot(h.normal, rd);
        float3 nf  = (ndi < 0.0) ? h.normal : -h.normal;

        if (h.mat == MAT_LAMBERT) {
            if (env_map_present != 0u) {
                // NEE: importance-sampled env-map direct light.
                float2 xi = float2(randf(seed), randf(seed));
                EnvSample env = sampleEnvMap(xi);
                float n_dot_l = dot(nf, env.dir);
                if (env.pdf > 0.0 && n_dot_l > 0.0) {
                    // RGB transmittance through any dielectrics
                    // between this hit and the light -- 1.0 free path,
                    // 0.0 fully blocked, in-between for refraction
                    // through glass / diamond meshes (caustics).
                    float3 trans = transmittance(hit_pt + nf * 1e-3, env.dir, 1e30);
                    if (any(trans > float3(0.0))) {
                        float3 brdf = h.albedo / 3.14159265358979;
                        radiance += throughput * env.color * trans * brdf * n_dot_l / env.pdf;
                    }
                }
                skip_sky_on_miss = true;
            } else {
                // Procedural sun NEE + skylight ambient. Drives the
                // procedural / gradient sky modes -- the scene's
                // directional light follows the actual sun position
                // (so it goes dark at night) and a small diffuse term
                // emulates skylight so unlit faces aren't pure black.
                float3 sun_dir  = sun_and_mode.xyz;
                float  sun_elev = sun_dir.y;            // sin(elevation)

                if (sun_elev > 0.0) {
                    float n_dot_l = max(0.0, dot(nf, sun_dir));
                    float3 sun_jit = sampleSunDisc(sun_dir, seed);
                    float3 trans  = (n_dot_l > 0.0)
                        ? transmittance(hit_pt + nf * 1e-3, sun_jit, 1e30)
                        : float3(0.0);
                    if (n_dot_l > 0.0 && any(trans > float3(0.0))) {
                        // Atmospheric attenuation: optical depth times
                        // air mass. Air mass = 1/sin(elev) (thin-slab
                        // approximation, accurate down to ~5 deg). The
                        // exp(-tau*am) curve matches real sunset
                        // behaviour: brightness drops off a cliff once
                        // elev < 5 deg, but stays bright through midday.
                        float am  = 1.0 / max(sun_elev, 0.04);
                        float tau = 0.30;             // tunable haze
                        float trans_atm = exp(-tau * am);
                        // Wavelength-dependent attenuation: blue
                        // scatters more, so colour reddens with air
                        // mass. Wavelength weights mimic Rayleigh
                        // (~1/lambda^4) without doing a full spectral
                        // walk -- enough to read as "warm at sunset".
                        float warmth = saturate((am - 1.0) * 0.18);
                        float3 sun_col = lerp(float3(1.00, 0.97, 0.92),
                                              float3(1.55, 0.55, 0.20),
                                              warmth);
                        // I0 in tonemap-relative units. 80 is tuned so
                        // a noon sun on a Lambert white wall reads
                        // ~1.0 after exposure*ACES at typical exposure.
                        float strength = 80.0 * trans_atm;
                        float3 sun_rad = sun_col * strength;
                        float3 brdf    = h.albedo / 3.14159265358979;
                        // Multiply by per-channel transmittance through
                        // any dielectrics on the shadow ray, so glass
                        // tint / diamond focal lensing modulate the
                        // direct light correctly.
                        radiance += throughput * sun_rad * trans * brdf * n_dot_l;
                    }
                }
                // Diffuse skylight ambient. Tracks the day/night blend
                // procSky uses for its background gradient -- not
                // physically derived but visually consistent with the
                // sky the camera sees behind the scene.
                float day = smoothstep(-0.10, 0.20, sun_elev);
                float3 sky_amb = lerp(float3(0.005, 0.010, 0.025),
                                      float3(0.30, 0.40, 0.60), day);
                radiance += throughput * h.albedo * sky_amb;
                skip_sky_on_miss = true;
            }
            throughput *= h.albedo;
            ro = hit_pt + nf * 1e-3;
            rd = sampleCosineHemisphere(nf, seed);
        }
        else if (h.mat == MAT_METAL) {
            float3 r = reflect(rd, nf);
            r = normalize(r + h.roughness * (sampleCosineHemisphere(nf, seed) - nf));
            throughput *= h.albedo;
            ro = hit_pt + nf * 1e-3;
            rd = r;
            skip_sky_on_miss = false;   // metal didn't NEE; sky on miss is real
        }
        else {  // dielectric
            float ior_a = (ndi < 0.0) ? 1.0   : h.ior;
            float ior_b = (ndi < 0.0) ? h.ior : 1.0;
            float eta = ior_a / ior_b;
            float cos_i = abs(dot(nf, -rd));
            float r0 = (ior_a - ior_b) / (ior_a + ior_b);
            r0 = r0 * r0;
            float fr = fresnelSchlick(cos_i, r0);
            float3 reflected = reflect(rd, nf);
            float3 refracted = refract(rd, nf, eta);
            bool tir = (length(refracted) < 1e-3);
            if (tir || randf(seed) < fr) {
                rd = reflected;
                ro = hit_pt + nf * 1e-3;
            } else {
                rd = normalize(refracted);
                ro = hit_pt - nf * 1e-3;
            }
            skip_sky_on_miss = false;   // dielectric didn't NEE either
        }

            if (b >= 2u) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                p = clamp(p, 0.05, 1.0);
                if (randf(seed) > p) break;
                throughput /= p;
            }
        }
        // Volumetric single-scatter along the primary ray. Two
        // independent marches because the natural sampling rates
        // differ:
        //   - Haze: dense near the camera, attenuates exp(-density*t)
        //     so beyond ~5km it contributes nothing. Cap the march
        //     range at 5km to keep dt small.
        //   - Clouds: live in a thin altitude band (e.g. 200-500m).
        //     Clip the march to the actual ray-slab intersection so
        //     all N samples land inside the cloud, not in empty air.
        // The two contributions add into `radiance`. Both share the
        // sun radiance and HG phase setup below.
        if ((vol_params.x > 0.0 || clouds_p1.w > 0.0) && primary_t_recorded) {
            float  g         = vol_params.y;
            float  vol_int   = vol_params.z;
            int    N         = max(1, int(vol_params.w));
            float3 sun_dir   = sun_and_mode.xyz;
            float  sun_elev  = sun_dir.y;
            // Sun radiance at this elevation (matches Lambert sun
            // colour so beams + lit surfaces agree on tint).
            float3 sun_rad = float3(0.0);
            if (sun_elev > 0.0) {
                float am  = 1.0 / max(sun_elev, 0.04);
                float tau = 0.30;
                float t_atm = exp(-tau * am);
                float warmth = saturate((am - 1.0) * 0.18);
                sun_rad = lerp(float3(1.00, 0.97, 0.92),
                               float3(1.55, 0.55, 0.20), warmth)
                        * (80.0 * t_atm);
            }
            float cos_theta = dot(primary_rd, -sun_dir);
            float g2 = g * g;
            float phase = (1.0 - g2)
                        / (4.0 * 3.14159265358979
                                * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));

            // --- Haze march -------------------------------------------------
            // Bounded at 5km because at typical density 0.005-0.05 the haze
            // is fully attenuated by then; marching further wastes samples.
            if (vol_params.x > 0.0) {
                float sigma_t = vol_params.x;
                float t_far   = min(primary_t, 5000.0);
                float dt      = t_far / float(N);
                float jitter  = randf(seed) * dt;
                float trans_eye = 1.0;
                float3 v_color = float3(0.0);
                for (int vi = 0; vi < N; ++vi) {
                    float t = (float(vi) + 0.5) * dt + jitter;
                    if (t >= t_far) break;
                    float3 sp = primary_ro + primary_rd * t;
                    if (sun_elev > 0.0) {
                        float3 sun_jit = sampleSunDisc(sun_dir, seed);
                        float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                        if (any(trans_sun > float3(0.0))) {
                            v_color += sun_rad * trans_sun * phase
                                     * sigma_t * dt * trans_eye;
                        }
                    }
                    trans_eye *= exp(-sigma_t * dt);
                }
                radiance += v_color * vol_int;
            }

            // --- Cloud march ------------------------------------------------
            // Layer-clip [t_in, t_out] to the ray-slab intersection of the
            // cloud altitude band. For object-hit primary rays we also
            // clip at primary_t so clouds don't shine through walls. For
            // sky pixels primary_t = 50km, so the layer is fully visible.
            if (clouds_p1.w > 0.0) {
                float base_y = clouds_p1.x;
                float top_y  = clouds_p1.y;
                float ry = primary_rd.y;
                float t_in, t_out;
                if (abs(ry) > 1e-6) {
                    float t_b = (base_y - primary_ro.y) / ry;
                    float t_t = (top_y  - primary_ro.y) / ry;
                    t_in  = max(0.0, min(t_b, t_t));
                    t_out = max(t_b, t_t);
                } else if (primary_ro.y >= base_y && primary_ro.y <= top_y) {
                    t_in = 0.0; t_out = primary_t;
                } else {
                    t_in = 1.0; t_out = 0.0;   // miss
                }
                t_out = min(t_out, primary_t);
                // Cap the cloud march span at 30km. For low-elevation
                // rays the layer can span hundreds of km; uniform
                // sampling that range produces samples spaced wider
                // than cloud features and looks worse than just
                // marching the closer 30km. Beyond 30km the sky
                // gradient takes over visually.
                if (t_out - t_in > 30000.0) t_in = t_out - 30000.0;
                if (t_out > t_in) {
                    int   Nc = max(N, 12);    // user slider can ask for more
                    float dtc = (t_out - t_in) / float(Nc);
                    float jitc = randf(seed) * dtc;
                    float trans_eye_c = 1.0;
                    float3 v_color_c  = float3(0.0);
                    for (int vi = 0; vi < Nc; ++vi) {
                        float t = t_in + (float(vi) + 0.5) * dtc + jitc;
                        if (t > t_out) break;
                        float3 sp = primary_ro + primary_rd * t;
                        float dens = cloud_density_at(sp, clouds_p1,
                                                      clouds_p2, clouds_p3, 6);
                        if (dens <= 0.0) continue;
                        if (sun_elev > 0.0) {
                            float3 sun_jit = sampleSunDisc(sun_dir, seed);
                            float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                            if (any(trans_sun > float3(0.0))) {
                                v_color_c += sun_rad * trans_sun * phase
                                           * dens * dtc * trans_eye_c;
                            }
                        }
                        trans_eye_c *= exp(-dens * dtc);
                    }
                    radiance += v_color_c * vol_int;
                }
            }
        }

        frame_radiance += radiance;
    }
    frame_radiance /= float(sample_count);

    // Online running mean. reset_accum=1 starts a fresh average. The
    // `n_new` we record is the count of samples averaged into this
    // pixel -- with spp>1 we could store sample_count there, but the
    // shader only reads it as a weight, so 1-per-frame is fine.
    float4 prev = (reset_accum != 0u) ? float4(0,0,0,0) : accum_hdr[tid];
    float n_new = prev.a + 1.0;
    float3 avg  = (prev.rgb * prev.a + frame_radiance) / n_new;
    accum_hdr[tid] = float4(avg, n_new);

    // Per-frame radiance feeds the denoiser. With the HDR pipeline
    // (w2j_row0.w = 1) we hand MetalFX raw linear HDR and the
    // post-pass `tonemap` kernel applies exposure + ACES at the very
    // end -- this preserves highlight detail through temporal reuse.
    // With the legacy LDR pipeline (= 0) we pre-tonemap here and the
    // post-pass kernel runs as a passthrough copy. The toggle exists
    // so r_hdr_pipeline can be flipped for A/B comparison without a
    // rebuild.
    if (denoiser_enabled != 0u) {
        if (w2j_row0.w > 0.5) {
            denoise_color[tid] = float4(frame_radiance, 1.0);
        } else {
            float3 mapped = acesTonemap(frame_radiance * exposure_pad.x);
            denoise_color[tid] = float4(mapped, 1.0);
        }
    }

    output[tid] = float4(acesTonemap(avg * exposure_pad.x), 1.0);
}

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

// Procedural moon surface texture, equirectangular RGBA16F generated
// once at engine init by pt::moon::generateMoonTexture. Sampled in
// moonDisc() to give the disc real mare + crater detail.
[[vk::binding(13, 0)]] RWTexture2D<float4> moon_map;

// GPU-resident exposure scalar. Updated each frame by AutoExposure.slang
// (when r_auto_exposure = 1) or written once by the engine via
// WriteBuffer (when r_auto_exposure = 0, holding the manual r_exposure
// value). Replaces the per-frame CPU readback of accum_hdr that
// stalled the GPU on dGPU paths -- now read directly here in the
// final tonemap, no host round-trip.
[[vk::binding(15, 0)]] RWStructuredBuffer<float> exposure_state;

// The full PtPush is 448 bytes. Native Vulkan / NVIDIA caps push
// constants at 256B, so on SPIR-V we keep only the small per-dispatch
// prefix (camera basis + flags + jitter, 112B) in [[vk::push_constant]]
// and spill the rest into a UBO at binding(14, 0). On Metal,
// [[vk::push_constant]] lowers to a regular setBytes buffer slot
// (no hw size cap), so we keep the full struct there for simplicity.
// VulkanCommandBuffer::Dispatch byte-splits the contiguous host
// PtPush struct: bytes [0..112] -> vkCmdPushConstants, bytes
// [112..448] -> the per-frame Frame UBO upload.
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
#ifndef PT_TARGET_SPIRV
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
    // Volumetric cloud parameters and atmospheric Rayleigh intensity
    // (clouds_p3.w). When clouds_p1.w > 0 the volumetric march
    // samples a position-dependent cloud density.
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
    //   clouds_p3: (seed_offset_x, seed_offset_z, detail_amount,
    //               rayleigh_intensity)
    //              seed shifts the noise hash so different seeds
    //              produce visually distinct cloud patterns at the
    //              same preset.
    //              detail_amount in [0,1] blends in a higher-frequency
    //              octave for wispy edges.
    //              rayleigh_intensity is a uniform scalar on the real
    //              per-channel Rayleigh sigma (5.8/13.5/33.1 ×1e-6 m^-1
    //              for R/G/B at sea level). 1.0 = real atmosphere; >1
    //              boosts the blue tint, 0 disables Rayleigh.
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    // Moon. .xyz = unit vector toward the moon (engine computes via
    // Meeus simplified moon-position formulas + observer lat/lon).
    // .w = phase angle in radians (0 = new, π = full); used for both
    // the moon disc's brightness in skyColor and for moonlight NEE
    // contribution amplitude on cloud / surface BRDF rays.
    float4 moon_dir_phase;
    // Moon extras. .x = r_moon_size, .y = astronomical distance ratio
    // (mean/current; perigee ~1.06, apogee ~0.95), .z = r_moon_brightness
    // (disc-multiplier scale, 1.0 = post-ACES-tuned baseline). .w reserved.
    float4 moon_extra;
    // Sun extras. .x = r_sun_size, .y = Earth-Sun distance ratio
    // (mean/current; perihelion ~1.017, aphelion ~0.984).
    float4 sun_extra;
    // P12 HDRI-extracted sun. .xyz = unit direction toward the bright
    // pixel cluster centroid in the env map (computed at HDRI load).
    // .w = 1.0 if a sun was extracted, 0.0 otherwise. The path tracer
    // casts a directional NEE shadow ray toward this direction in HDRI
    // mode, producing crisp directional shadows independent of how
    // soft/sharp the HDRI sun pixels actually are.
    float4 hdri_sun_dir;
    // P12 HDRI-extracted sun radiant flux. .rgb = ∫_cluster L dΩ over
    // the cluster, in raw env_map units. Shader applies env_intensity
    // for parity with env-map NEE's color scaling. .w reserved.
    float4 hdri_sun_color;
#endif  // !PT_TARGET_SPIRV
};

#ifdef PT_TARGET_SPIRV
// Spilled tail of PtPush — see the comments in the (Metal) Push cbuffer
// above for per-field semantics. Layout is byte-equivalent to bytes
// [112..448] of the host PtPush struct in Engine.cpp. Bound by
// VulkanCommandBuffer::Dispatch from a per-frame ring-buffer UBO.
[[vk::binding(14, 0)]]
cbuffer Frame {
    float4x4 curr_view_proj;
    float4x4 prev_view_proj;
    float4 sun_and_mode;
    float4 exposure_pad;
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
    float4 dof_params;
    float4 vol_params;
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 moon_dir_phase;
    float4 moon_extra;
    float4 sun_extra;
    float4 hdri_sun_dir;
    float4 hdri_sun_color;
};
#endif  // PT_TARGET_SPIRV

// --- Procedural cloud field --------------------------------------------
//
// Hash-based 3D value noise + fbm. Cheaper than gradient (Perlin)
// noise -- the smoothstep filter step gives enough visual continuity
// for cloud bodies at this scale, and we don't need C2 continuity for
// volumetric sampling. ~30 ALU per sample × ~4 octaves × 16 march
// samples × 1080p = ~6 GFLOP/frame on the worst march, well under
// budget on M4 Max.

// Integer mixing hash (Wellons' Lowbias32 -- 3 multiplies + 3 xorshifts,
// near-perfect avalanche on every input bit).  The previous
// frac(p * 0.3183099 + 0.1) scaled-frac trick had two structural flaws
// that produced visible patches/seams in the cloud body:
//   1. Output biased toward extremes (the product p.x*p.y*p.z*(sum)
//      collapses near zero whenever any component is small).
//   2. Adjacent integer grid cells produce CORRELATED outputs because
//      the multiplied frac() is similar for similar inputs -- visible
//      as low-frequency "patches" even at small p magnitudes.
// Lowbias32 + uint cast on integer-floored input fixes both.
uint cloud_hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
float cloud_hash(float3 p) {
    // Caller (cloud_value_noise) always passes already-floored float
    // values; int cast is exact.  Two's-complement wrap on negatives
    // is well-defined and the hash treats it as just another bit
    // pattern, so world-space coords below origin still produce
    // decorrelated cells.
    int3 ip = int3(p);
    uint h = cloud_hash_u32(uint(ip.x));
    h = cloud_hash_u32(h ^ uint(ip.y));
    h = cloud_hash_u32(h ^ uint(ip.z));
    return float(h & 0x00FFFFFFu) * (1.0 / 16777216.0);
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
    // Linear ramp coverage -- saturate((low - (1-coverage)) / coverage).
    // Tried smoothstep but the steep S-curve near the threshold meant a
    // single hi-freq fbm dip into the [1-coverage, 1-coverage+0.05]
    // band collapses to zero density, punching visible holes through
    // the cloud body. Linear keeps the response monotonic and
    // proportional to the noise value -- holes only appear where the
    // noise actually says "no cloud".
    float coverage = saturate(cp1.z);
    float density  = saturate((low - (1.0 - coverage)) / max(coverage, 0.01));
    if (density <= 0.0) return 0.0;
    // Detail erosion: a higher-frequency fbm erodes EDGES of clouds
    // into wispy fingers, but should NOT punch holes in dense cores.
    // The previous unconditional subtract was hollowing out cumulus
    // bodies (donut-shaped clouds). Solution: scale erosion strength
    // by (1 - density)^2 so density 1.0 is fully protected, density
    // 0.5 erodes at 0.25x, density 0.1 erodes at full strength. Real
    // cumulus shows wispy trails, not Swiss-cheese cores.
    if (cp3.z > 0.0) {
        float hi = cloud_fbm(sp * 4.0, 3);
        float edge = (1.0 - density) * (1.0 - density);
        density = saturate(density - hi * cp3.z * edge);
    }
    // Internal variation: a SOFT secondary fbm gives the cloud volume
    // some texture without holes. Bias toward 1.0 (multiplier 0.75-1.0
    // instead of 0.55-1.0) so we don't dim the cloud body uniformly.
    float internal = 0.75 + 0.25 * cloud_fbm(sp * 1.7, 3);
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

// Sun disc + atmospheric halo. Real solar angular diameter is 0.53deg
// (almost identical to the moon -- the reason solar eclipses work).
// Render with an actual disc + a halo whose extent grows at low sun
// elevation, matching the bright bloom you see around the sun at
// sunset/sunrise (more atmospheric thickness + dust = more scatter).
//
// The disc itself stays the same physical 0.55deg half-angle; what
// changes between noon and sunset is the colour (warm/red at low
// elev) and the halo size (much bigger at low elev, fading toward
// a tight halo at zenith). This matches real photos: sunset sun
// SEEMS bigger because the bright halo reads as part of the disc.
float3 sunDisc(float3 rd, float3 sun_dir) {
    if (sun_dir.y < -0.10) return float3(0.0);
    float ang = dot(rd, sun_dir);
    // r_sun_size * astronomical-distance ratio. perihelion ~1.017
    // (sun looks bigger), aphelion ~0.984. Half-angle scales linearly.
    float sun_mult = max(sun_extra.x * sun_extra.y, 0.1);
    float r_deg    = 0.55 * sun_mult;
    float kCosR    = cos(r_deg * 0.0174533);
    // Halo extent grows from cos(2deg) at zenith to cos(8deg) near
    // horizon. The horizon halo is what you see as the "bigger" sun
    // at sunset -- it's atmospheric scattering, not the disc itself.
    float halo_deg   = lerp(2.0, 8.0, saturate(1.0 - sun_dir.y));
    float halo_rad   = halo_deg * 0.0174533;
    float kCosHalo   = cos(halo_rad);
    if (ang <= kCosHalo) return float3(0.0);

    // Atmospheric attenuation + warmth, matching sun_rad in NEE.
    float am  = 1.0 / max(sun_dir.y, 0.04);
    float atm = exp(-0.30 * am);
    float warmth = saturate((am - 1.0) * 0.18);
    float3 col = lerp(float3(1.00, 0.98, 0.94),
                      float3(1.55, 0.55, 0.20), warmth);

    // Halo: soft bloom around the disc.
    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        // Steeper falloff at zenith (tight halo), wider at horizon.
        float falloff = lerp(8.0, 3.0, saturate(1.0 - sun_dir.y));
        halo = pow(t_h, falloff);
    }

    if (ang > kCosR) {
        // Bright disc -- sun is enormously brighter than anything
        // else; ACES will clip the centre to white but the limb +
        // halo carry the warmth.
        return col * (250.0 * atm)
             + col * halo * atm * 4.0;
    }
    return col * halo * atm * 4.0;
}

// Moon disc + halo. Real lunar angular radius is 0.00465 rad (~0.27deg
// half-angle = 0.54deg full).
//   - terminator: per-pixel lit/unlit check via dot(surface_normal,
//     sun_dir). Replaces the previous flat phase-brightness scalar so
//     half-moon shows a curved terminator splitting the disc instead
//     of dimming the whole thing uniformly. New moon -> all unlit.
//   - moon_map texture: real surface (mare + crater) sampled in the
//     moon's tangent frame. Tidally locked, so tangent UV maps
//     directly to near-side surface coordinates.
//   - limb darkening: edge of disc dimmer than centre.
//   - atmospheric attenuation: same model as sun_rad (dims with airmass
//     when moon is near horizon, warms toward red).
//   - subtle halo: soft ring extending to ~1.5deg, simulating
//     atmospheric scattering around the moon.
float3 moonDisc(float3 rd, float4 mph, float3 sun_dir_world) {
    if (mph.y < -0.10) return float3(0.0);
    float3 m_dir = mph.xyz;
    float ang = dot(rd, m_dir);
    // Real lunar angular radius is 0.27 deg; we render at 0.55 deg
    // (~2x real) for visibility at typical 60-FOV 1080p, then scale
    // by r_moon_size (moon_extra.x) AND astronomical distance ratio
    // (moon_extra.y; perigee ~1.06, apogee ~0.95) so supermoons
    // really do render bigger than micro-moons.
    float size_mult = max(moon_extra.x * moon_extra.y, 0.1);
    float r_deg     = 0.55 * size_mult;
    float halo_deg  = 1.5  * size_mult;
    float kCosR     = cos(r_deg    * 0.0174533);
    float kCosHalo  = cos(halo_deg * 0.0174533);
    if (ang <= kCosHalo) return float3(0.0);

    float3 col_tint = float3(0.92, 0.94, 1.00);

    // Atmospheric attenuation matching sun_rad.
    float moon_am   = 1.0 / max(mph.y, 0.04);
    float moon_atm  = exp(-0.30 * moon_am);
    float warmth_m  = saturate((moon_am - 1.0) * 0.18);
    float3 col_atm  = lerp(float3(1.00, 1.00, 1.00),
                           float3(1.20, 0.65, 0.40), warmth_m);

    // Halo ring extends beyond the disc.
    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        halo = pow(t_h, 4.0) * 0.6;
    }

    if (ang > kCosR) {
        float t_d = (ang - kCosR) / (1.0 - kCosR);
        // Tangent frame around the moon. (du, dv) project the view ray
        // onto the disc plane in moon-local coords.
        float3 ax = abs(m_dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 t_a = normalize(cross(ax, m_dir));
        float3 t_b = cross(m_dir, t_a);
        // Scale factor 1/sin(disc half-angle) to map disc-tangent
        // projection to [-1, 1] across the visible diameter. Includes
        // the moon-size multiplier so "big moon" debug mode still
        // keeps the texture mapped onto the disc.
        float disc_scale = 1.0 / max(sin(r_deg * 0.0174533), 1e-5);
        float du = dot(rd, t_a) * disc_scale;
        float dv = dot(rd, t_b) * disc_scale;
        float r2 = du * du + dv * dv;
        float zz = sqrt(max(1.0 - r2, 0.0));

        // Outward surface normal in WORLD frame for this visible point.
        // The visible hemisphere of a sphere at distance D in direction
        // m_dir, viewed from origin, has surface normals pointing back
        // TOWARD the observer (i.e. the -m_dir half), with tangential
        // offsets parameterised by (du, dv). At the disc centre the
        // normal is -m_dir (closest point); at the limb it lies in the
        // tangent plane. Previous + sign on the m_dir term picked the
        // far hemisphere -- caused the terminator to invert and full
        // moon to read as fully unlit.
        float3 n_world = du * t_a + dv * t_b - zz * m_dir;

        // Terminator: lit-fraction at this surface point. Smooth
        // transition over a few percent for an aliasing-free edge.
        float lit = saturate(dot(n_world, sun_dir_world) * 10.0 + 0.05);
        if (lit <= 0.0) return col_tint * col_atm * halo * moon_atm * 0.05;

        // Lunar eclipse / blood moon. Earth's umbra at the moon's
        // distance is ~0.7deg radius (penumbra ~1.25deg). When the
        // moon's centre falls within this cone -- i.e. within ~0.7deg
        // of the antisolar point (-sun_dir) -- direct sunlight is
        // blocked. Atmospheric refraction through Earth's limb
        // smears red light into the umbra (the same Rayleigh
        // attenuation that reddens our own sunsets), so the moon
        // glows deep red instead of going pitch dark. Smooth
        // penumbra transition so partial eclipses blend.
        float anti_sun_dot = -dot(m_dir, sun_dir_world);
        float anti_sun_ang = acos(saturate(anti_sun_dot));
        const float kUmbraRad    = 0.0122;     // ~0.7 deg
        const float kPenumbraRad = 0.0218;     // ~1.25 deg
        float eclipse = 1.0 - smoothstep(kUmbraRad, kPenumbraRad,
                                         anti_sun_ang);

        // Sample the procedural moon map. Equirectangular UV from
        // local (du, dv, zz). Latitude from dv (north/south of moon
        // equator); longitude from du / zz (east/west of moon center).
        const float kHalfPi = 1.57079632679;
        float lon = atan2(du, zz);                              // [-pi/2, pi/2]
        float lat = asin(clamp(dv, -1.0, 1.0));                 // [-pi/2, pi/2]
        // Moon texture is full sphere (2pi wide, pi tall); near side
        // maps to roughly the centre half. u in [0.25, 0.75] keeps
        // texture variation across the visible disc.
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

        // Limb darkening (real lunar limb is darker due to roughness).
        float limb = 0.55 + 0.45 * sqrt(saturate(t_d));

        // Disc multiplier 3.5 baseline scaled by user-controlled
        // r_moon_brightness (moon_extra.z). 1.0 = baseline; 0.7
        // (default) reads as a softer night moon where surface texture
        // stays well below the ACES knee. Halo scales with brightness
        // so the surrounding bloom stays proportionate.
        float bright_mult = max(moon_extra.z, 0.0);
        float disc_scale_b = 3.5 * bright_mult;
        float3 lit_col = col_tint * col_atm * surface
                       * (disc_scale_b * lit * limb * moon_atm)
                       + col_tint * col_atm * halo * lit * moon_atm
                       * (0.6 * bright_mult);

        // Blood-moon colour during eclipse: deep red-orange (the
        // refracted-through-Earth-atmosphere light tone). Surface
        // texture still shows through, just heavily red-tinted.
        // Brightness drops a lot -- a typical totality has the moon
        // ~1/10000 of normal full-moon brightness.
        float3 eclipse_col = float3(0.55, 0.15, 0.04) * surface
                           * (limb * moon_atm * 1.2);
        return lerp(lit_col, eclipse_col, eclipse);
    }
    // Halo only -- no surface here. Halo dims toward new-moon by the
    // average disc lit fraction, approximated by phase brightness.
    float phase_b = 0.5 * (1.0 - cos(mph.w));
    return col_tint * col_atm * halo * phase_b * moon_atm * 0.3;
}

// Per-contribution firefly clamp. Bounds a single indirect light
// contribution (NEE off the env map, ambient skylight, bounce-to-sky
// from b>0) so a single-sample spike from a tiny PDF or BSDF-sampled
// bounce hitting an HDRI sun pixel doesn't blow the running mean.
// Camera-direct sky paths (b == 0 sky-on-miss) bypass this so the
// sun still appears at full intensity in the rendered sky.
//
// Hue-preserving: scale by max-channel ratio rather than per-channel
// min(). A naive per-channel clamp would turn an orange contribution
// (5, 3, 1) into pure grey (1, 1, 1) at ceiling=1 -- the sunset's
// warm bounce light loses its color. Max-channel scaling maps the
// same input to (1, 0.6, 0.2), preserving hue and just bounding
// magnitude. This lets a ceiling of ~10 actually remain on without
// neutralising the indirect colour cast.
//
// Engine plumbs the ceiling via w2j_row2.w; 0 disables.
float3 clampIndirect(float3 c) {
    if (w2j_row2.w <= 0.0) return c;
    float m = max(c.r, max(c.g, c.b));
    return (m > w2j_row2.w) ? c * (w2j_row2.w / m) : c;
}

float3 skyColor(float3 rd) {
    float3 base;
    uint mode = uint(sun_and_mode.w);
    if (mode == 2u) {
        base = procSky(rd, sun_and_mode.xyz);
    } else if (env_map_present != 0u) {
        // Original env_map path follows.
        base = float3(0.0);   // filled below
    } else {
        float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
        base = lerp(float3(0.50, 0.55, 0.62), float3(1.0, 1.0, 1.0), t);
    }
    if (mode == 2u) {
        return base + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDisc(rd, sun_and_mode.xyz);
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
        // P12: HDRI mode is the canonical "all info from the HDRI"
        // path. The HDRI texture itself contains the sun, moon, stars
        // and any other celestial bodies baked in. We deliberately do
        // NOT add a procedural sunDisc / moonDisc overlay -- the user's
        // expectation is that the rendered sky matches the HDRI source.
        // Crisp directional shadows come from the directional NEE
        // toward the extracted sun cluster (see hdri_sun_dir block in
        // the path tracer's Lambert hit handling).
        return sky * env_intensity;
    }
    return base + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDisc(rd, sun_and_mode.xyz);
}

float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c*(a*c+b))/(c*(d*c+e)+f));
}

#if defined(PT_TARGET_SPIRV)
// Standard sRGB OETF (linear -> nonlinear, IEC 61966-2-1: linear seg
// near zero + gamma ~2.4 above). Vulkan storage-image writes do NOT
// auto-encode for sRGB swapchain formats per spec (only color-
// attachment writes do). The Mac/Metal path uses Apple silicon's
// implicit storage-write encoding for BGRA8Unorm_sRGB and skips this.
float3 srgb_oetf(float3 c) {
    c = max(c, float3(0.0, 0.0, 0.0));
    float3 lin  = c * 12.92;
    float3 expo = 1.055 * pow(c, float3(1.0/2.4)) - 0.055;
    return select(c < 0.0031308, lin, expo);
}
#endif

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
                // b == 0 is the camera ray hitting the sky directly --
                // unbounded so the HDRI sun renders at full intensity.
                // b > 0 is a bounce ray escaping; clamp those to kill
                // fireflies from BSDF-sampled bounces hitting the sun.
                if (b == 0u) {
                    radiance += throughput * skyColor(rd);
                } else {
                    radiance += clampIndirect(throughput * skyColor(rd));
                }
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
                        radiance += clampIndirect(throughput * env.color * trans * brdf * n_dot_l / env.pdf);
                    }
                }
                // P12 HDRI extracted sun NEE: env-map NEE samples the
                // sun-MASKED CDF (the bright cluster pixels were zeroed
                // at HDRI load), so the sun's contribution is missing
                // from that estimator. Add it back as a directional
                // NEE toward the cluster centroid: shoots one shadow
                // ray, all Lambert pixels point in the same direction
                // -> sharp directional shadows independent of HDRI sun
                // softness. hdri_sun_color is the precomputed
                // ∫_cluster L dΩ; multiplying by env_intensity matches
                // env-map NEE's intensity treatment so toggling the
                // cvar scales both equivalently.
                if (hdri_sun_dir.w > 0.0) {
                    float n_dot_sun = dot(nf, hdri_sun_dir.xyz);
                    if (n_dot_sun > 0.0) {
                        float3 t_s = transmittance(hit_pt + nf * 1e-3, hdri_sun_dir.xyz, 1e30);
                        if (any(t_s > float3(0.0))) {
                            float3 brdf = h.albedo / 3.14159265358979;
                            radiance += clampIndirect(throughput * hdri_sun_color.xyz
                                                     * env_intensity * t_s * brdf * n_dot_sun);
                        }
                    }
                }
                // No procedural moon NEE in HDRI mode: the HDRI is the
                // canonical sky source, and adding moon_dir_phase-driven
                // light here would re-introduce time-of-day dependence
                // that contradicts the HDRI's own moonlight (or lack of
                // it). If the chosen HDRI doesn't include a moon at
                // night, the scene reads dim -- that's the HDRI being
                // incomplete, not a missing engine feature. Pick an
                // HDRI with the moon baked in or switch r_sky_mode to
                // procedural.
                // Diffuse skylight ambient via cosine-weighted hemisphere
                // sample. Previous version sampled skyColor() at the
                // surface normal exactly -- a single-direction shortcut
                // that causes severe per-pixel fireflies on HDRI scenes:
                // every pixel whose surface normal happens to align with
                // the HDRI sun direction (a 1e6-nit pixel) gets a huge
                // single-sample contribution, producing the classic
                // sphere-speckle pattern visible on Lambert surfaces
                // under sunset/noon HDRIs. Cosine-weighted hemisphere
                // sampling integrates properly: the per-frame value is
                // a random hemisphere tap, and spp + temporal averaging
                // converges to the true diffuse irradiance. Bypasses
                // cloud occlusion intentionally (skyColor reads env_map
                // directly), so cloud-shadowed surfaces still receive
                // ambient light -- the same approximation the original
                // hack made, just without the firefly artifact.
                {
                    float xi_a = randf(seed), xi_b = randf(seed);
                    float r    = sqrt(xi_a);
                    float phi  = 6.28318530717958 * xi_b;
                    float3 t1  = abs(nf.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
                    float3 tx  = normalize(cross(t1, nf));
                    float3 ty  = cross(nf, tx);
                    float3 amb_dir = tx * (r * cos(phi))
                                   + ty * (r * sin(phi))
                                   + nf * sqrt(max(0.0, 1.0 - xi_a));
                    radiance += clampIndirect(throughput * h.albedo * skyColor(amb_dir));
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
                // Moon NEE for the procedural-sky path (see env_map
                // branch above for the rationale).
                if (moon_dir_phase.y > 0.0) {
                    float phase_b = 0.5 * (1.0 - cos(moon_dir_phase.w));
                    if (phase_b > 0.01) {
                        float3 moon_jit = sampleSunDisc(moon_dir_phase.xyz, seed);
                        float ndl = max(0.0, dot(nf, moon_jit));
                        if (ndl > 0.0) {
                            float3 t_m = transmittance(hit_pt + nf * 1e-3, moon_jit, 1e30);
                            if (any(t_m > float3(0.0))) {
                                // Atmospheric attenuation + warmth at low
                                // moon altitude, mirroring sun radiance.
                                float m_elev = max(moon_dir_phase.y, 0.04);
                                float m_am = 1.0 / m_elev;
                                float m_atm = exp(-0.30 * m_am);
                                float m_warmth = saturate((m_am - 1.0) * 0.18);
                                float3 m_col = lerp(float3(0.92, 0.94, 1.00),
                                                    float3(1.20, 0.65, 0.40),
                                                    m_warmth);
                                // 0.3x of sun (was 0.8) -- night feels
                                // genuinely dim. Auto-exposure still
                                // amplifies for visibility, but the
                                // ground / clouds stay readably night
                                // (~1/3 of daylight luminance) instead
                                // of looking like a slightly cool day.
                                float3 moon_rad = m_col * (0.3 * phase_b * m_atm);
                                float3 brdf = h.albedo / 3.14159265358979;
                                radiance += throughput * moon_rad * t_m * brdf * ndl;
                            }
                        }
                    }
                }
                // Calculated diffuse skylight ambient (matches the
                // env_map branch above). Sample the procedural sky
                // model at the surface normal -- single-direction
                // Lambert ambient approximation, real and derived
                // from the sky model rather than hardcoded RGB.
                radiance += throughput * h.albedo * skyColor(nf);
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
        if ((vol_params.x > 0.0 || clouds_p1.w > 0.0 || clouds_p3.w > 0.0)
            && primary_t_recorded) {
            float  g         = vol_params.y;
            float  vol_int   = vol_params.z;
            int    N         = max(1, int(vol_params.w));
            float3 sun_dir   = sun_and_mode.xyz;
            float  sun_elev  = sun_dir.y;
            // Sun radiance at this elevation (matches Lambert sun
            // colour so beams + lit surfaces agree on tint). Warmth
            // saturates more aggressively (0.30 instead of 0.18) and
            // the warm endpoint is a deeper red, so sunset clouds
            // actually read as "blood orange" rather than "tinted
            // beige". am stored separately so the cloud march can
            // re-scale per-sample altitude (high clouds catch less-
            // attenuated, brighter / less-red sunlight than ground).
            float3 sun_rad = float3(0.0);
            float  sun_am  = 0.0;
            if (sun_elev > 0.0) {
                sun_am = 1.0 / max(sun_elev, 0.04);
                float tau = 0.30;
                float t_atm = exp(-tau * sun_am);
                float warmth = saturate((sun_am - 1.0) * 0.30);
                sun_rad = lerp(float3(1.00, 0.97, 0.92),
                               float3(1.55, 0.40, 0.12), warmth)
                        * (80.0 * t_atm);
            }
            float cos_theta = dot(primary_rd, -sun_dir);
            float g2 = g * g;
            float phase = (1.0 - g2)
                        / (4.0 * 3.14159265358979
                                * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));

            // --- Atmospheric (Mie + Rayleigh) march -------------------------
            // Real atmosphere has two scattering populations:
            //   Mie (haze, dust, water droplets): grey, forward-peaked HG
            //     phase, scale height ~1.5km, sigma controlled by
            //     vol_params.x. Gives god rays and sunset warmth.
            //   Rayleigh (gas molecules): wavelength-dependent, blue is
            //     scattered ~6x more than red. Phase is 3/(16π)*(1+cos²θ).
            //     Scale height ~8km. Real per-channel sea-level sigma:
            //       R 5.8e-6  G 13.5e-6  B 33.1e-6  m^-1.
            //     Rayleigh is what makes the daytime sky blue and the
            //     sunset red (long path attenuates blue out, leaving
            //     red); without it, the sky bleaches white at noon.
            //
            // Per channel transmittance through the column is the product
            // of Mie and Rayleigh exp(-sigma*dt), so trans_eye is a
            // float3 (Mie alone is grey; Rayleigh tints the column).
            //
            // Bounded at 60km because Rayleigh's 8km scale height means
            // most of its contribution lives in the bottom ~25km, and
            // 60km is plenty of headroom while keeping dt manageable.
            const float3 kSigmaR = float3(5.8e-6, 13.5e-6, 33.1e-6);
            const float kHmie = 1500.0;
            const float kHrayleigh = 8000.0;
            const float kPi = 3.14159265358979;
            float rayleigh_int = clouds_p3.w;   // 0 disables, 1 = real
            // Atmospheric Mie + Rayleigh march is disabled in HDRI mode:
            // the HDRI has the atmosphere baked into its sky pixels, so
            // adding procedural in-scatter on top double-counts (the
            // sky's blue tint comes from the HDRI directly) and would
            // follow the engine sun_dir rather than the HDRI's actual
            // sun. God rays / sunset haze in HDRI mode are a future
            // feature that would need to anchor on the extracted-sun
            // direction.
            bool march_mie = (vol_params.x > 0.0) && (env_map_present == 0u);
            bool march_ray = (rayleigh_int > 0.0) && (env_map_present == 0u);
            if (march_mie || march_ray) {
                float  sigma_0 = vol_params.x;
                float  t_far   = min(primary_t, 60000.0);
                float  dt      = t_far / float(N);
                float  jitter  = randf(seed) * dt;
                float3 trans_eye = float3(1.0);
                float3 v_color = float3(0.0);
                float phase_mie_v = phase;   // HG, computed earlier
                float phase_ray_v = (3.0 / (16.0 * kPi)) * (1.0 + cos_theta * cos_theta);
                for (int vi = 0; vi < N; ++vi) {
                    float t = (float(vi) + 0.5) * dt + jitter;
                    if (t >= t_far) break;
                    float3 sp = primary_ro + primary_rd * t;
                    float h = max(sp.y, 0.0);
                    float  sigma_mie = march_mie ? sigma_0 * exp(-h / kHmie) : 0.0;
                    float3 sigma_ray = march_ray
                        ? kSigmaR * rayleigh_int * exp(-h / kHrayleigh)
                        : float3(0.0);
                    float3 sigma_total = sigma_ray + float3(sigma_mie);
                    if (sun_elev > 0.0) {
                        float3 sun_jit = sampleSunDisc(sun_dir, seed);
                        float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                        if (any(trans_sun > float3(0.0))) {
                            float3 in_scatter =
                                  float3(sigma_mie) * phase_mie_v
                                + sigma_ray         * phase_ray_v;
                            v_color += sun_rad * trans_sun * in_scatter
                                     * dt * trans_eye;
                        }
                    }
                    trans_eye *= exp(-sigma_total * dt);
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
                    // Skylight ambient term: integrating the sky hemisphere
                    // around a cloud sample gives ~15% of sun irradiance at
                    // the surface (real-Earth ratio of skylight/sun
                    // irradiance is ~0.15-0.20). The sky is also slightly
                    // blue from Rayleigh, so the ambient tint is a touch
                    // bluer than sun_rad. Without this term cloud cores go
                    // pitch-black under sun self-shadow -- correct
                    // single-scatter direct-sun-only physics, but real
                    // clouds get filled in by the surrounding sky.
                    // 1/π factor folds the hemisphere integral in (the same
                    // factor that turns Lambert irradiance into radiance).
                    float3 sky_ambient = float3(0.0);
                    if (sun_elev > 0.0) {
                        sky_ambient = sun_rad * float3(0.7, 0.85, 1.0) * 0.15
                                    * (1.0 / 3.14159265358979);
                    } else if (moon_dir_phase.y > 0.0) {
                        // Moonlight skylight on the cloud at night.
                        // Without this term, clouds at night go pitch
                        // black -- they're correctly receiving zero
                        // sun in shadow but missing the diffuse moon-
                        // hemisphere contribution that real moonlit
                        // clouds actually receive.
                        float phb = 0.5 * (1.0 - cos(moon_dir_phase.w));
                        float ma  = 1.0 / max(moon_dir_phase.y, 0.04);
                        float mat = exp(-0.30 * ma);
                        sky_ambient = float3(0.55, 0.70, 1.00) * (0.04 * phb * mat);
                    }
                    for (int vi = 0; vi < Nc; ++vi) {
                        float t = t_in + (float(vi) + 0.5) * dtc + jitc;
                        if (t > t_out) break;
                        float3 sp = primary_ro + primary_rd * t;
                        float dens = cloud_density_at(sp, clouds_p1,
                                                      clouds_p2, clouds_p3, 6);
                        if (dens <= 0.0) continue;
                        // Direct sun (single-scatter NEE). Scale
                        // sun_rad by altitude-dependent atmospheric
                        // attenuation: high clouds (cirrus at 8km)
                        // sit ABOVE most of the absorbing atmosphere
                        // and catch much brighter / less-reddened
                        // light than the ground does. Approximation:
                        // boost = exp(tau * sun_am * (h / H_atm))
                        // where H_atm ~ 8km is the atmospheric scale
                        // height. For ground-level cumulus the factor
                        // is ~1.05; for cirrus at 8km it's ~5x.
                        if (sun_elev > 0.0) {
                            float h_alt   = max(sp.y, 0.0);
                            float h_norm  = saturate(h_alt / 8000.0);
                            float boost   = exp(0.30 * sun_am * h_norm);
                            float3 sun_at = sun_rad * boost;
                            float3 sun_jit = sampleSunDisc(sun_dir, seed);
                            float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                            if (any(trans_sun > float3(0.0))) {
                                v_color_c += sun_at * trans_sun * phase
                                           * dens * dtc * trans_eye_c;
                            }
                        }
                        // Direct moon (single-scatter NEE). Moon is the
                        // night sky's key light; without this clouds go
                        // pitch-black after sunset. Phase shading via
                        // 0.5 * (1 - cos(phase)) so new moon contributes
                        // nothing, full moon is full-strength.
                        if (moon_dir_phase.y > 0.0) {
                            float phase_b = 0.5 * (1.0 - cos(moon_dir_phase.w));
                            if (phase_b > 0.01) {
                                float3 moon_jit = sampleSunDisc(moon_dir_phase.xyz, seed);
                                float3 trans_moon = transmittance(sp, moon_jit, 1e30);
                                if (any(trans_moon > float3(0.0))) {
                                    float c_t_m = dot(primary_rd, -moon_dir_phase.xyz);
                                    float g2 = g * g;
                                    float phase_m = (1.0 - g2) / (4.0 * 3.14159265358979
                                        * pow(max(1.0 + g2 - 2.0 * g * c_t_m, 1e-4), 1.5));
                                    // Real moon irradiance is ~1/400000 of
                                    // sun's, but we boost to 1/100 (sun_rad
                                    // 80 -> moon ~0.8) so a moonlit scene
                                    // is visible after typical auto-exposure
                                    // without the moon shadowing through
                                    // bright daylight. Atmospheric attenuation
                                    // (air mass 1/sin(elev), tau 0.30) makes
                                    // the moon dim and redden near the
                                    // horizon, matching the sun's behaviour.
                                    float m_elev   = max(moon_dir_phase.y, 0.04);
                                    float m_am     = 1.0 / m_elev;
                                    float m_atm    = exp(-0.30 * m_am);
                                    float m_warmth = saturate((m_am - 1.0) * 0.18);
                                    float3 m_col   = lerp(float3(0.92, 0.94, 1.00),
                                                          float3(1.20, 0.65, 0.40),
                                                          m_warmth);
                                    // 0.3x of sun (was 0.8) -- night feels
                                // genuinely dim. Auto-exposure still
                                // amplifies for visibility, but the
                                // ground / clouds stay readably night
                                // (~1/3 of daylight luminance) instead
                                // of looking like a slightly cool day.
                                float3 moon_rad = m_col * (0.3 * phase_b * m_atm);
                                    v_color_c += moon_rad * trans_moon * phase_m
                                               * dens * dtc * trans_eye_c;
                                }
                            }
                        }
                        // Diffuse skylight (hemisphere integral). No phase
                        // function -- ambient is by definition omnidirectional.
                        v_color_c += sky_ambient * dens * dtc * trans_eye_c;
                        trans_eye_c *= exp(-dens * dtc);
                    }
                    radiance += v_color_c * vol_int;
                }
            }
        }

        frame_radiance += radiance;
    }
    frame_radiance /= float(sample_count);

    // Sanitize: replace NaN / Inf / negative channels with 0 before the
    // running-mean update below. Without this, a single rare bad sample
    // (e.g. transmittance edge case, env.pdf underflowing to ~0 between
    // the > 0 check and the divide, sqrt of slightly-negative numerical
    // residues) injects NaN into accum_hdr[tid]. The running-mean update
    // is `avg = (prev * n_old + new) / n_new`, so NaN propagates through
    // prev forever -- that pixel becomes a permanent black speck. Over
    // many minutes of static-camera accumulation, the contaminated pixel
    // set grows monotonically. Replacing non-finite samples with 0 lets
    // the running mean simply skip the bad sample and keep accumulating
    // the good ones for that pixel.
    {
        bool3 bad = isnan(frame_radiance) || isinf(frame_radiance);
        frame_radiance = select(bad, float3(0.0), max(frame_radiance, float3(0.0)));
    }

    // Online running mean. reset_accum=1 starts a fresh average. The
    // `n_new` we record is the count of samples averaged into this
    // pixel -- with spp>1 we could store sample_count there, but the
    // shader only reads it as a weight, so 1-per-frame is fine.
    float4 prev = (reset_accum != 0u) ? float4(0,0,0,0) : accum_hdr[tid];
    // Sanitize prev: if the previous-frame sample was already NaN/Inf
    // (e.g. from a build before sanitization landed, or imported state),
    // restart that pixel rather than re-averaging in the contamination.
    {
        bool3 prev_bad = isnan(prev.rgb) || isinf(prev.rgb);
        if (any(prev_bad) || isnan(prev.a) || isinf(prev.a)) {
            prev = float4(0, 0, 0, 0);
        }
    }
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
    // Pull the live exposure scalar from the GPU-side state buffer.
    // AutoExposure.slang updates this each frame (auto mode); the
    // engine WriteBuffer's it once when manual mode toggles. Reading
    // here avoids a CPU readback round-trip that on dGPU stalls
    // the queue once per N frames.
    float exposure = exposure_state[0];

    if (denoiser_enabled != 0u) {
        // denoise_color is an RGBA16F linear-HDR intermediate read by
        // the post-pass Tonemap kernel later -- stays linear, no OETF.
        if (w2j_row0.w > 0.5) {
            denoise_color[tid] = float4(frame_radiance, 1.0);
        } else {
            float3 mapped = acesTonemap(frame_radiance * exposure);
            denoise_color[tid] = float4(mapped, 1.0);
        }
    }

    // Swapchain write. Apply the OETF on the SPIR-V path so the OS's
    // EOTF-on-display gives back the original linear values. On Metal
    // the swapchain's BGRA8Unorm_sRGB pixel format encodes implicitly
    // on Apple silicon storage writes -- nothing for the shader to do.
    float3 ldr = acesTonemap(avg * exposure);
#if defined(PT_TARGET_SPIRV)
    ldr = srgb_oetf(ldr);
#endif
    output[tid] = float4(ldr, 1.0);
}

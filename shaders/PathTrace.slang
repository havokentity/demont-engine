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
    // World->J2000 rotation, row-major. row1.w / row2.w are unused; the
    // matrix only needs the 3x3 block. row0.w is repurposed as the
    // HDR-pipeline flag: 1.0 -> path tracer writes raw HDR to
    // denoise_color (post-MetalFX tonemap kernel applies exposure +
    // ACES); 0.0 -> path tracer pre-applies exposure + ACES (legacy).
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
};

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

// Cheap "is this direction occluded?" test. Returns true if any of the
// active scene primitives or the mesh TLAS is hit before TMax.
bool occluded(float3 ro, float3 rd, float t_max) {
    // Analytic primitives.
    for (uint i = 0u; i < prim_count; ++i) {
        float4 v0 = primitives[i * 3u + 0u];
        float4 v2 = primitives[i * 3u + 2u];
        uint type = uint(v2.x);
        float t;
        if (type == PRIM_SPHERE) {
            if (intersectSphere(ro, rd, v0.xyz, v0.w, t) && t < t_max) return true;
        } else if (type == PRIM_PLANE) {
            if (intersectPlane(ro, rd, v0.xyz, v0.w, t) && t < t_max) return true;
        }
    }
    // Mesh TLAS.
    if (tlas_present != 0u) {
        RayDesc r;
        r.Origin = ro; r.Direction = rd; r.TMin = 1e-3; r.TMax = t_max;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, r);
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) return true;
    }
    return false;
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

        float3 throughput = float3(1.0);
        float3 radiance   = float3(0.0);
        // After a Lambert bounce we did NEE for direct env-map lighting,
        // so the next BRDF-bounce ray's miss contribution is already
        // counted -- skip on miss to avoid double counting.
        bool   skip_sky_on_miss = false;

        for (uint b = 0u; b < max_bounces; ++b) {
        HitInfo h = traceScene(ro, rd);
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
                    if (!occluded(hit_pt + nf * 1e-3, env.dir, 1e30)) {
                        float3 brdf = h.albedo / 3.14159265358979;
                        radiance += throughput * env.color * brdf * n_dot_l / env.pdf;
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
                    if (n_dot_l > 0.0 &&
                        !occluded(hit_pt + nf * 1e-3, sun_dir, 1e30)) {
                        // Atmospheric attenuation: optical depth times
                        // air mass. Air mass = 1/sin(elev) (thin-slab
                        // approximation, accurate down to ~5 deg). The
                        // exp(-tau*am) curve matches real sunset
                        // behaviour: brightness drops off a cliff once
                        // elev < 5 deg, but stays bright through midday.
                        float am  = 1.0 / max(sun_elev, 0.04);
                        float tau = 0.30;             // tunable haze
                        float trans = exp(-tau * am);
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
                        float strength = 80.0 * trans;
                        float3 sun_rad = sun_col * strength;
                        float3 brdf    = h.albedo / 3.14159265358979;
                        radiance += throughput * sun_rad * brdf * n_dot_l;
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

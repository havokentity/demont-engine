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
    uint   pad2;
    float2 halton_jitter;    // per-frame sub-pixel offset in [-0.5,0.5]
    float2 pad3;
    float4x4 curr_view_proj;
    float4x4 prev_view_proj;
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

float3 skyColor(float3 rd) {
    float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
    return lerp(float3(0.50, 0.55, 0.62), float3(1.0, 1.0, 1.0), t);
}

float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c*(a*c+b))/(c*(d*c+e)+f));
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

        for (uint b = 0u; b < max_bounces; ++b) {
        HitInfo h = traceScene(ro, rd);
        if (!h.hit) {
            radiance += throughput * skyColor(rd);
            break;
        }

        float3 hit_pt = ro + rd * h.t;
        float  ndi = dot(h.normal, rd);
        float3 nf  = (ndi < 0.0) ? h.normal : -h.normal;

        if (h.mat == MAT_LAMBERT) {
            float3 L  = normalize(LIGHT_DIR);
            float  nl = max(0.0, dot(nf, L));
            radiance += throughput * h.albedo * nl * 0.6;
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

    // Per-frame HDR linear color goes to the denoiser (no tonemap, no
    // accumulation -- MetalFX does its own temporal reuse).
    if (denoiser_enabled != 0u) {
        denoise_color[tid] = float4(frame_radiance, 1.0);
    }

    output[tid] = float4(acesTonemap(avg), 1.0);
}

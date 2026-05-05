// Path tracer that intersects triangle meshes through a TLAS, alongside
// the existing analytic ground plane.  Same Lambert / metal /
// dielectric materials as PathTrace.slang; uses ray query (RayQuery
// inline) so it works on Metal (intersector) and on Vulkan with
// KHR_ray_query when available.
//
// Bindings:
//   slot 0 (texture) : swapchain  (BGRA8)
//   slot 1 (texture) : accum HDR  (RGBA32F)
//   slot 2 (accel)   : scene TLAS
//
// Push: same 80-byte layout as PathTrace.slang.  Material per
// instance is selected via the instance custom index (we currently
// only render one mesh, so this is hardcoded to "metal" for the
// instance and Lambert grey for the analytic ground).

[[vk::binding(0, 0)]] RWTexture2D<float4> output;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_hdr;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure scene_tlas;

[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    uint   frame_index;
    uint   reset_accum;
    uint   max_bounces;
    uint   pad0;
};

static const float3 LIGHT_DIR = float3(0.4, 1.0, 0.3);
static const float3 GROUND_N  = float3(0, 1, 0);
static const float  GROUND_D  = 0.0;
static const float3 GROUND_ALBEDO = float3(0.55, 0.55, 0.55);

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
    float3 t = normalize(cross(n, ax));
    float3 b = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

bool intersectPlane(float3 ro, float3 rd, out float t, out float3 n) {
    float denom = dot(rd, GROUND_N);
    n = GROUND_N;
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, GROUND_N) + GROUND_D) / denom;
    return t > 1e-3;
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

    float2 jitter = float2(randf(seed), randf(seed)) - 0.5;
    float2 uv = ((float2(tid) + 0.5 + jitter) / float2(dim)) * 2.0 - 1.0;
    uv.y = -uv.y;

    float3 ro = pos_fovtan.xyz;
    float3 rd = normalize(fwd_aspect.xyz
                       + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                       + up_xyz.xyz    * (uv.y * pos_fovtan.w));

    float3 throughput = float3(1.0);
    float3 radiance   = float3(0.0);

    for (uint b = 0u; b < max_bounces; ++b) {
        // Intersect the TLAS.
        RayDesc r;
        r.Origin    = ro;
        r.Direction = rd;
        r.TMin      = 1e-3;
        r.TMax      = 1e30;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, r);
        while (q.Proceed()) { /* no non-opaque -- nothing to do */ }

        bool   tlas_hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
        float  t_mesh   = tlas_hit ? q.CommittedRayT() : 1e30;

        // Intersect the ground plane analytically.
        float t_plane;
        float3 plane_n;
        bool plane_hit = intersectPlane(ro, rd, t_plane, plane_n);
        if (plane_hit && t_plane >= t_mesh) plane_hit = false;

        if (!tlas_hit && !plane_hit) {
            radiance += throughput * skyColor(rd);
            break;
        }

        float  t_hit;
        float3 hit_pt;
        float3 nf;
        float3 albedo;
        bool   metal_mat;

        if (tlas_hit && (!plane_hit || t_mesh < t_plane)) {
            t_hit = t_mesh;
            hit_pt = ro + rd * t_hit;
            // RayQuery doesn't expose a per-triangle normal directly --
            // we'd need the vertex buffer bound to compute a real
            // geometric normal. For P8 use the negated ray direction
            // as a stand-in (flat camera-facing). P9+ binds the vertex
            // buffer + does proper geometric normal from primitive
            // index + barycentrics.
            nf = -rd;
            albedo    = float3(1.00, 0.85, 0.45);   // gold metal
            metal_mat = true;
        } else {
            t_hit  = t_plane;
            hit_pt = ro + rd * t_hit;
            nf     = (dot(plane_n, rd) < 0.0) ? plane_n : -plane_n;
            albedo = GROUND_ALBEDO;
            metal_mat = false;
        }

        if (metal_mat) {
            float3 rdir = reflect(rd, nf);
            // Slight roughness so the reflection isn't a perfect mirror.
            rdir = normalize(rdir + 0.05 * (sampleCosineHemisphere(nf, seed) - nf));
            throughput *= albedo;
            ro = hit_pt + nf * 1e-3;
            rd = rdir;
        } else {
            // Lambert with direct light estimate.
            float3 L  = normalize(LIGHT_DIR);
            float  nl = max(0.0, dot(nf, L));
            radiance += throughput * albedo * nl * 0.6;
            throughput *= albedo;
            ro = hit_pt + nf * 1e-3;
            rd = sampleCosineHemisphere(nf, seed);
        }

        if (b >= 2u) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            p = clamp(p, 0.05, 1.0);
            if (randf(seed) > p) break;
            throughput /= p;
        }
    }

    float4 prev = (reset_accum != 0u) ? float4(0,0,0,0) : accum_hdr[tid];
    float n_new = prev.a + 1.0;
    float3 avg  = (prev.rgb * prev.a + radiance) / n_new;
    accum_hdr[tid] = float4(avg, n_new);

    output[tid] = float4(acesTonemap(avg), 1.0);
}

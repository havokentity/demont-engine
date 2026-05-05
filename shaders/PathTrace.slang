// Monte Carlo path tracer with materials + HDR accumulation + ACES tonemap.
//
//   slot 0 (texture): swapchain (BGRA8Unorm) -- tonemapped LDR
//   slot 1 (texture): accum_hdr (RGBA32F)    -- linear average + sample count in .a
//   buffer 0:        push constants (96 bytes)
//
// Each dispatch traces ONE new sample per pixel, blends into accum, then
// tonemaps into the swapchain. Camera movement resets accumulation via
// the reset_accum push constant.
//
// Materials hardcoded for now (3 spheres + ground plane).  Phase 8 will
// pull them from a scene buffer.

[[vk::binding(0, 0)]] RWTexture2D<float4> output;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_hdr;

[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;     // xyz pos, w tan(fov_y/2)
    float4 fwd_aspect;     // xyz forward, w aspect
    float4 right_xyz;      // xyz right
    float4 up_xyz;         // xyz up
    uint   frame_index;
    uint   reset_accum;    // 1 -> discard previous accum
    uint   max_bounces;
    uint   pad0;
};

static const uint MAT_LAMBERT    = 0u;
static const uint MAT_METAL      = 1u;
static const uint MAT_DIELECTRIC = 2u;

struct Sphere {
    float3 c; float  r;
    uint   mat;
    float3 albedo;
    float  roughness;
    float  ior;
};
struct Plane {
    float3 n; float  d;
    uint   mat;
    float3 albedo;
};

static const Sphere SPHERES[3] = {
    { float3(-1.2, 0.5,  0.2), 0.5, MAT_LAMBERT,    float3(0.92, 0.30, 0.30), 0.0, 1.0 },
    { float3( 0.0, 0.5,  0.0), 0.5, MAT_METAL,      float3(1.00, 0.85, 0.45), 0.05, 1.0 },
    { float3( 1.2, 0.5, -0.2), 0.5, MAT_DIELECTRIC, float3(1.00, 1.00, 1.00), 0.0, 1.5 },
};
static const Plane GROUND = { float3(0,1,0), 0.0, MAT_LAMBERT, float3(0.55, 0.55, 0.55) };
static const float3 LIGHT_DIR = float3(0.4, 1.0, 0.3);

// PCG hash
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

bool intersectSphere(float3 ro, float3 rd, Sphere s, out float t) {
    float3 oc = ro - s.c;
    float b   = dot(oc, rd);
    float c   = dot(oc, oc) - s.r * s.r;
    float h   = b * b - c;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > 1e-3) ? t0 : t1;
    return t > 1e-3;
}
bool intersectPlane(float3 ro, float3 rd, Plane p, out float t) {
    float denom = dot(rd, p.n);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, p.n) + p.d) / denom;
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

    [unroll] for (int i = 0; i < 3; ++i) {
        float t;
        if (intersectSphere(ro, rd, SPHERES[i], t) && t < h.t) {
            h.hit = true; h.t = t;
            h.normal = normalize(ro + rd * t - SPHERES[i].c);
            h.mat = SPHERES[i].mat;
            h.albedo = SPHERES[i].albedo;
            h.roughness = SPHERES[i].roughness;
            h.ior = SPHERES[i].ior;
        }
    }
    {
        float t;
        if (intersectPlane(ro, rd, GROUND, t) && t < h.t) {
            h.hit = true; h.t = t;
            h.normal = GROUND.n;
            h.mat = GROUND.mat;
            h.albedo = GROUND.albedo;
            h.roughness = 0.0;
            h.ior = 1.0;
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

    // Pixel jitter -- gives free AA after a few accum samples.
    float2 jitter = float2(randf(seed), randf(seed)) - 0.5;
    float2 uv = ((float2(tid) + 0.5 + jitter) / float2(dim)) * 2.0 - 1.0;
    uv.y = -uv.y;

    float3 ro = pos_fovtan.xyz;
    float3 rd = normalize(fwd_aspect.xyz
                       + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                       + up_xyz.xyz    * (uv.y * pos_fovtan.w));

    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 radiance   = float3(0.0, 0.0, 0.0);

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
            // Direct light estimate (no shadow ray for now -- will add
            // proper visibility in P11 alongside MIS).
            float3 L = normalize(LIGHT_DIR);
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
            // No throughput attenuation for clear glass.
        }

        // Russian roulette beyond bounce 2.
        if (b >= 2u) {
            float p = max(throughput.r, max(throughput.g, throughput.b));
            p = clamp(p, 0.05, 1.0);
            if (randf(seed) > p) break;
            throughput /= p;
        }
    }

    // Accum (running mean).  reset_accum=1 starts a fresh average.
    float4 prev = (reset_accum != 0u) ? float4(0,0,0,0) : accum_hdr[tid];
    float n_new = prev.a + 1.0;
    float3 avg  = (prev.rgb * prev.a + radiance) / n_new;
    accum_hdr[tid] = float4(avg, n_new);

    output[tid] = float4(acesTonemap(avg), 1.0);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// P5-early: analytic intersection of a small hardcoded scene.
//   * 3 spheres (red / green / blue) sitting on a grey ground plane
//   * single directional light + ambient
//   * sky uses the user's r_clear_color through `bg`
//
// All bindings + push-constant layout match the rest of the path:
// storage texture at slot 0, 64 bytes of push constants at buffer 0
// (4x float4 == camera basis + fov/aspect).

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;     // xyz = camera position, w = tan(fov_y * 0.5)
    float4 fwd_aspect;     // xyz = forward,         w = aspect (W/H)
    float4 right_xyz;      // xyz = right
    float4 up_bg_pack;     // xyz = up,              w = packed sky tint (rgb8 in low bits)
};

struct Sphere {
    float3 center;
    float  radius;
    float3 albedo;
};

struct Plane {
    float3 normal;
    float  d;
    float3 albedo;
};

static const float3 LIGHT_DIR = float3(0.4, 1.0, 0.3);

static const Sphere SPHERES[3] = {
    { float3( 0.0, 0.5,  0.0), 0.5, float3(0.90, 0.20, 0.20) },
    { float3(-1.2, 0.5,  0.2), 0.5, float3(0.20, 0.85, 0.30) },
    { float3( 1.2, 0.5, -0.2), 0.5, float3(0.20, 0.40, 0.95) },
};
static const Plane GROUND = { float3(0, 1, 0), 0.0, float3(0.55, 0.55, 0.55) };

bool intersectSphere(float3 ro, float3 rd, Sphere s, out float t) {
    float3 oc = ro - s.center;
    float b  = dot(oc, rd);
    float c  = dot(oc, oc) - s.radius * s.radius;
    float h  = b * b - c;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;
    t = (t0 > 1e-3) ? t0 : t1;
    return t > 1e-3;
}

bool intersectPlane(float3 ro, float3 rd, Plane p, out float t) {
    float denom = dot(rd, p.normal);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, p.normal) + p.d) / denom;
    return t > 1e-3;
}

float3 unpackBg(float packed) {
    // Reinterpret a float as 4 bytes; first 3 = rgb in 0..255.
    uint bits = asuint(packed);
    float r = float((bits >>  0) & 0xff) / 255.0;
    float g = float((bits >>  8) & 0xff) / 255.0;
    float b = float((bits >> 16) & 0xff) / 255.0;
    return float3(r, g, b);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    uint2 dim;
    output.GetDimensions(dim.x, dim.y);
    if (any(tid >= dim)) return;

    // NDC in [-1, 1], y flipped (top-left origin).
    float2 uv = (float2(tid) + 0.5) / float2(dim) * 2.0 - 1.0;
    uv.y = -uv.y;

    float3 cam_pos   = pos_fovtan.xyz;
    float  fov_tan   = pos_fovtan.w;
    float3 cam_fwd   = fwd_aspect.xyz;
    float  aspect    = fwd_aspect.w;
    float3 cam_right = right_xyz.xyz;
    float3 cam_up    = up_bg_pack.xyz;
    float3 bg        = unpackBg(up_bg_pack.w);

    float3 ro = cam_pos;
    float3 rd = normalize(cam_fwd
                       + cam_right * (uv.x * aspect * fov_tan)
                       + cam_up    * (uv.y * fov_tan));

    float  t_near  = 1e30;
    float3 albedo  = float3(0, 0, 0);
    float3 normal  = float3(0, 1, 0);
    bool   hit_any = false;

    [unroll] for (int i = 0; i < 3; ++i) {
        float t;
        if (intersectSphere(ro, rd, SPHERES[i], t) && t < t_near) {
            t_near  = t;
            normal  = normalize(ro + rd * t - SPHERES[i].center);
            albedo  = SPHERES[i].albedo;
            hit_any = true;
        }
    }
    {
        float t;
        if (intersectPlane(ro, rd, GROUND, t) && t < t_near) {
            t_near  = t;
            normal  = GROUND.normal;
            albedo  = GROUND.albedo;
            hit_any = true;
        }
    }

    float3 color;
    if (hit_any) {
        float3 L  = normalize(LIGHT_DIR);
        float  nl = max(0.0, dot(normal, L));
        // Lambert + tiny ambient so shadowed sides still show colour.
        color = albedo * (nl * 0.85 + 0.15);
    } else {
        // Sky gradient: horizon = bg, zenith brighter.
        float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
        color   = lerp(bg, float3(1.0, 1.0, 1.0), t * 0.4);
    }

    output[tid] = float4(color, 1.0);
}

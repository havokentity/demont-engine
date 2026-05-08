// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-math helpers used by the path tracer.  Compiled to a Slang IR
// module (.slang-module) so PathTrace.slang's `import` resolves to a
// pre-compiled blob and ninja can build this module in parallel with
// the entry-point shader, plus skip the front-end re-parse on every
// PathTrace.spv rebuild.  Nothing in here references resource
// bindings or push constants -- it's deliberately self-contained so
// the same .slang-module file links into both the SPIR-V and Metal
// targets unmodified.

// PCG hash + uniform float in [0, 1).  Tiny state (one uint per pixel
// thread) -- the caller mutates the seed in place so successive draws
// from the same thread are decorrelated.
public uint pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}
public float randf(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed & 0x00FFFFFFu) / 16777216.0;
}

// Cosine-weighted hemisphere sample around `n`.  Used by the diffuse
// ambient term in the Lambert HDRI block (cosine weighting matches
// the BRDF, so the pdf cancels out and the estimator collapses to
// just sampling the sky in `amb_dir`).
public float3 sampleCosineHemisphere(float3 n, inout uint seed) {
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
// 0.00465 rad (~0.266 deg). Shadow rays sample a random direction
// inside that disc instead of the centre direction; over the temporal
// accumulator this gives soft penumbrae naturally (clouds, foliage,
// hard objects all gain real-physics-soft edges).
//
// This is *not* a softening hack -- the sun really is a disc and a
// point sample is the wrong physics. We're just doing the integral
// the path tracer was missing.
public static const float kSunAngularRadius = 0.00465;

public float3 sampleSunDisc(float3 sun_dir, inout uint seed) {
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

// Schlick approximation of Fresnel reflectance.  r0 is the base
// reflectance at normal incidence; for dielectrics this is
// ((n1-n2)/(n1+n2))^2.
public float fresnelSchlick(float cos_theta, float r0) {
    float m = 1.0 - cos_theta;
    return r0 + (1.0 - r0) * m * m * m * m * m;
}

// Ray-sphere intersection. Returns the nearer non-self-intersection t
// (>1e-3 epsilon to avoid stuck rays grazing their own surface).
public bool intersectSphere(float3 ro, float3 rd, float3 c, float rad, out float t) {
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

// Ray-plane intersection.  `n` is the plane normal, `d` the signed
// distance from origin (plane: n . p + d = 0).  Returns false for
// nearly-parallel rays (no near-zero division).
public bool intersectPlane(float3 ro, float3 rd, float3 n, float d, out float t) {
    float denom = dot(rd, n);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, n) + d) / denom;
    return t > 1e-3;
}

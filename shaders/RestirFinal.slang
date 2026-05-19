// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// ReSTIR DI Phase A (issue #78) — Final shading pass.
//
// Reads the spatial-pass output reservoir, shoots ONE shadow ray to
// the chosen survivor sample (the whole point of ReSTIR — replace
// O(K + N) shadow rays with a single high-quality one against the
// best resampled candidate), evaluates the Lambert BRDF, and ADDITIVELY
// blends the contribution into `denoise_color` (the same RGBA16F
// intermediate PathTrace.slang writes during the candidate-generation
// pass).
//
// Scope gates:
//   - Only fires when r_restir = 1 AND the denoiser is active. The
//     denoiser-off path keeps PathTrace's existing single-light NEE
//     because (a) the accumulator handles many-light variance on its
//     own at high spp and (b) routing ReSTIR's contribution into the
//     dual accum_hdr + output swapchain write would double the
//     intercept points for a feature that's primarily a denoiser-input
//     quality lever. See engine dispatch gate in Engine.cpp.
//
// Reads:
//   reservoir_in       — spatial pass output (final survivor + W).
//   depth_tex          — reconstruct hit_world from camera basis + z.
//   normal_tex         — Lambert N.
//   albedo_tex         — Lambert albedo for the cos-weighted BRDF.
//   denoise_color      — read+write; the resampled contribution is
//                        added in place.
//   light_prims        — re-trace the survivor's Le at this pixel.
//   primitives / mesh_indices / mesh_positions / bvh_nodes /
//   tri_bvh_nodes / tri_bvh_permuted_ids / sdf_clusters
//                      — shadow-ray scene traversal. Same set of
//                        buffers PathTrace.slang reads for the
//                        primary trace; we re-bind the subset needed
//                        for a SHADOW ray (no material dispatch, just
//                        nearest-hit visibility).
//
// MVP scope: the shadow-ray traversal here is a simplified
// transmittance() — it tests analytic primitives (linear scan, no
// BVH since the analytic prim count is small) plus, when the engine
// has a TLAS bound, a HW ray-query for triangles. SDF clusters are
// scope-out for Phase A's shadow-trace path; their inclusion is a
// straightforward follow-up but adds a chunk of duplicated code that
// only matters for fractal-heavy scenes.

// vk::image_format: depth_tex R32F, normal_tex / albedo_tex /
// denoise_color all RGBA16F. See PathTrace.slang for the NVIDIA
// Format=Unknown rationale.
[[vk::binding(0, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;
[[vk::binding(1, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_tex;
[[vk::binding(2, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> albedo_tex;
[[vk::binding(3, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> denoise_color;
[[vk::binding(4, 0)]]   StructuredBuffer<float4>  reservoir_in;
[[vk::binding(5, 0)]]   StructuredBuffer<float4>  light_prims;
[[vk::binding(6, 0)]]   StructuredBuffer<float4>  primitives;
#if !defined(PT_TARGET_SPIRV) || !defined(PT_SPIRV_NO_RAYQUERY)
[[vk::binding(7, 0)]] RaytracingAccelerationStructure scene_tlas;
#endif

import RestirReservoir;

[[vk::push_constant]]
cbuffer Push {
    uint width;
    uint height;
    uint frame_index;
    uint light_count;
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    uint   restir_enabled;
    uint   prim_count;
    uint   tlas_present;
    uint   linear_prim_count;     // analytic-prim "always-linear" prefix
};

// Match PathTrace's analytic primitive layout: float4 * 3 per prim,
// PRIM_SPHERE = 0, PRIM_PLANE = 1.
static const uint PRIM_SPHERE = 0u;
static const uint PRIM_PLANE  = 1u;

// Sphere intersection. Returns true on a hit + the entry t. Equivalent
// to PathTrace's intersectSphere helper.
bool intersectSphere(float3 ro, float3 rd, float3 c, float r, out float t) {
    float3 oc = ro - c;
    float b = dot(oc, rd);
    float d = b*b - dot(oc, oc) + r*r;
    if (d < 0.0) { t = 0.0; return false; }
    float s = sqrt(d);
    float t0 = -b - s;
    float t1 = -b + s;
    t = (t0 > 1e-4) ? t0 : t1;
    return t > 1e-4;
}

// Shadow visibility: 1.0 = unoccluded, 0.0 = blocked. Conservative
// MVP — analytic-prim linear scan + (when bound) HW ray-query on the
// TLAS. Dielectric prim hits are treated as opaque here (Phase A
// scope-out the caustic-refractive-shadow path; PathTrace.slang's
// shadow ray already supports it, but porting that code triples this
// kernel's size for marginal Phase A win).
float shadowVis(float3 ro, float3 rd, float t_max) {
    // Analytic primitives: linear scan (the host has a BVH for prim
    // counts > r_analytic_bvh_threshold, but the linear loop is fine
    // for shadow rays at MVP scale and avoids the BVH traversal code
    // duplication).
    for (uint i = 0u; i < prim_count; ++i) {
        // Stride 5 float4s (was 3 pre-#85, 4 after motion-blur landed
        // in PR #198, 5 after the #181 emissive followup added v4).
        // Shadow-cast only needs v0/v2 -- v1/v3/v4 unread here -- but
        // the stride must match the host's kFloatsPerPrim=20 layout.
        float4 v0 = primitives[i * 5u + 0u];
        float4 v1 = primitives[i * 5u + 1u];
        float4 v2 = primitives[i * 5u + 2u];
        uint type = uint(v2.x);
        float t;
        if (type == PRIM_SPHERE) {
            if (intersectSphere(ro, rd, v0.xyz, v0.w, t) && t < t_max) {
                return 0.0;
            }
        }
        // PRIM_PLANE: planes don't cast NEE-relevant shadows in the
        // scenes we test (they're floors/walls below the shade point),
        // and including them would require the back-face cull
        // gymnastics PathTrace's `transmittance` does. Skip for MVP.
    }
#if !defined(PT_TARGET_SPIRV) || !defined(PT_SPIRV_NO_RAYQUERY)
    if (tlas_present != 0u) {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        RayDesc desc;
        desc.Origin    = ro;
        desc.Direction = rd;
        desc.TMin      = 1e-3;
        desc.TMax      = t_max;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, desc);
        q.Proceed();
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) return 0.0;
    }
#endif
    return 1.0;
}

float3 reconstructHit(uint2 tid, float z) {
    if (z >= 1.0e9) return float3(0.0);
    float2 dim = float2(float(width), float(height));
    float2 uv  = (float2(tid) + 0.5) / dim;
    float2 uv0 = uv * 2.0 - 1.0;
    uv0.x *= fwd_aspect.w;
    float3 rd = normalize(fwd_aspect.xyz
                          + right_xyz.xyz * (uv0.x * pos_fovtan.w)
                          + up_xyz.xyz    * (uv0.y * pos_fovtan.w));
    float fwd_dot = dot(rd, normalize(fwd_aspect.xyz));
    float t = z / max(fwd_dot, 1e-6);
    return pos_fovtan.xyz + rd * t;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    if (restir_enabled == 0u) return;
    if (light_count == 0u)    return;

    float z_c = depth_tex[tid.xy];
    if (z_c >= 1.0e9) return;                       // sky pixel

    uint pixel_idx = tid.y * width + tid.x;
    Reservoir r = loadReservoir(reservoir_in, pixel_idx);

    if (r.light_idx == kRestirInvalidLight) return;
    if (r.W <= 0.0)                          return;
    if (r.M == 0u)                           return;

    float3 n_c = normal_tex[tid.xy].xyz;
    float nlen = length(n_c);
    if (nlen < 1e-5) return;
    n_c = n_c / nlen;

    float3 albedo = albedo_tex[tid.xy].rgb;
    float3 hit_pt = reconstructHit(tid.xy, z_c);

    // Direction + cosine at the receiver.
    float3 to_light = r.light_pos - hit_pt;
    float  dist2    = max(dot(to_light, to_light), 1e-12);
    float  dist     = sqrt(dist2);
    float3 wi       = to_light / dist;
    float  n_dot_l  = dot(n_c, wi);
    if (n_dot_l <= 0.0) return;

    // Shadow ray. Origin offset by the normal so we don't self-occlude.
    float3 sro = hit_pt + n_c * 1e-3;
    // Stop t_max just shy of the sample point so the light's own
    // surface doesn't block its visibility (mirrors PathTrace's NEE).
    float  t_max = max(dist - 1e-3, 0.0);
    float vis = shadowVis(sro, wi, t_max);
    if (vis <= 0.0) return;

    // Estimator F = throughput * Le_pre * brdf * cos * W * vis.
    // PathTrace's existing NEE uses brdf = albedo / PI for Lambert; we
    // mirror it. r.Le is the PRE-distance-corrected Le that the WRS
    // candidate-generation cached (already includes spot falloff +
    // 1/dist^2 for point/spot). The receiver-side multiplier here is
    // simply n_dot_l * brdf.
    float3 brdf = albedo / 3.14159265358979;
    float3 contrib = r.Le * brdf * n_dot_l * r.W * vis;

    // Additive blend into denoise_color. The PathTrace pass that runs
    // BEFORE this one writes the "everything except analytic-light
    // NEE" radiance to this same texture (when r_restir = 1 the
    // sampleAnalyticLight branch is replaced with the reservoir-write
    // path; see PathTrace.slang's matching gate).
    float4 prev = denoise_color[tid.xy];
    float3 hdr  = prev.rgb + contrib;
    // Sanitize: a degenerate light record could produce NaN/Inf; clamp
    // to 1e4 in line with PathTrace's clampIndirect to avoid firefly
    // pixels.
    bool3 bad = isnan(hdr) || isinf(hdr);
    hdr = select(bad, prev.rgb, max(hdr, float3(0.0)));
    hdr = min(hdr, float3(1e4));
    denoise_color[tid.xy] = float4(hdr, prev.a);
}

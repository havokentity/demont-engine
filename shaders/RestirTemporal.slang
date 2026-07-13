// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// ReSTIR DI Phase A (issue #78) — Temporal reuse pass.
//
// Reads:
//   reservoir_curr_in   — this frame's WRS-initialised reservoir (from
//                          the PathTrace.slang candidate-generation block)
//   reservoir_prev_in   — last frame's FINAL reservoir (output of the
//                          previous frame's RestirSpatial pass; first
//                          frame is zero-initialised so all reservoirs
//                          read as M=0 and the temporal merge is a no-op)
//   motion_tex          — per-pixel reprojection delta (prev_pix - curr_pix)
//                          in pixel units; PathTrace.slang writes this
//                          as part of the existing G-buffer block.
//   depth_tex           — camera-space Z, for disocclusion rejection
//                          (sky pixel sentinel = 1e10, geometry < 1e9)
//   normal_tex          — world-space normal (RGBA16F; .a unused)
//   light_prims         — analytic light list (#73). Read for the
//                          re-evaluation of the prev-frame survivor's
//                          target pdf at this pixel.
//
// Writes:
//   reservoir_curr_out  — the temporally-merged reservoir; consumed by
//                          the spatial pass next.
//
// Algorithm: Bitterli §5 Algorithm 4. For each pixel:
//   1. Load current-frame reservoir (already initialised by PathTrace).
//   2. Reproject via motion_tex; if the prev pixel is in-bounds AND
//      the depth / normal at the prev pixel agrees with this pixel's
//      surface, load the prev reservoir.
//   3. Cap prev reservoir's M at kRestirTemporalMaxM to bound the
//      smear under fast motion.
//   4. combineReservoir(curr, prev, p_hat_prev_at_this_receiver, u).
//   5. Store the merged reservoir.
//
// Disocclusion criteria are the standard SVGF-flavoured pair:
//   * |z_curr - z_prev| / z_curr < kDepthDisoccTol  (5%)
//   * dot(n_curr, n_prev) > kNormalDisoccTol        (0.9 = ~25 deg)
// Pixels failing either fall back to "no prev reservoir" — same
// disocclusion gate the SVGF temporal pass uses, so the visual
// boundary between ReSTIR-reused and ReSTIR-fresh aligns with
// denoiser temporal boundaries.
//
// Bindings (Slang's MSL backend allocates buffer/texture slots in
// declaration order; the engine's BindStorageTexture / BindBuffer
// match these positions):
//   texture(0)  depth_tex
//   texture(1)  motion_tex
//   texture(2)  normal_tex
//   buffer (0)  reservoir_curr_in   (already written by PathTrace)
//   buffer (1)  reservoir_prev_in
//   buffer (2)  reservoir_curr_out
//   buffer (3)  light_prims         (re-evaluate prev survivor's p_hat)
//
// Push: width/height + camera basis (to reconstruct hit_world from
// depth + thread id) + light_count.

// vk::image_format: depth_tex R32F, motion_tex RG16F, normal_tex
// RGBA16F. See PathTrace.slang for the NVIDIA Format=Unknown
// rationale.
[[vk::binding(0, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;
[[vk::binding(1, 0), vk::image_format("rg16f")]]   RWTexture2D<float2> motion_tex;
[[vk::binding(2, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_tex;
[[vk::binding(3, 0)]]   StructuredBuffer<float4>  reservoir_curr_in;
[[vk::binding(4, 0)]]   StructuredBuffer<float4>  reservoir_prev_in;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4>  reservoir_curr_out;
[[vk::binding(6, 0)]]   StructuredBuffer<float4>  light_prims;

import RestirReservoir;

[[vk::push_constant]]
cbuffer Push {
    uint width;
    uint height;
    uint frame_index;       // RNG seed mixer; same convention PathTrace uses
    uint light_count;
    // Camera basis (mirrors PathTrace.slang) — required to reconstruct
    // the receiver world position from camera-space depth + thread id.
    float4 pos_fovtan;      // .xyz = camera origin, .w = fov vertical tan
    float4 fwd_aspect;      // .xyz = camera forward, .w = aspect
    float4 right_xyz;       // .xyz = camera right
    float4 up_xyz;          // .xyz = camera up
    uint   bias_mode;       // 0 = biased (default), 1 = unbiased
    uint   restir_enabled;  // master ON/OFF (mirrors r_restir)
    uint   temporal_enabled;
    uint   _pad0;
};

// Disocclusion tolerances. Matches the SVGF temporal pass for visual
// boundary coherence (a pixel that disoccludes here ALSO disoccludes
// in SVGF, so the user doesn't see a transition where one filter
// rejects and the other accepts).
static const float kDepthDisoccTol  = 0.05;
static const float kNormalDisoccTol = 0.9;

// Tiny PCG-step RNG matching PathTrace.slang's randf(). Imported via
// the standalone math helpers wouldn't work (PathTraceMath has many
// other resource refs); a local copy is cheaper and stays inline.
uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

float randf(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

// Reconstruct world-space hit position from a thread (x, y) + depth.
// Uses the camera basis the host pushed; matches PathTrace.slang's
// primary-ray formula (so the reprojected sample point is consistent
// with what PathTrace traced).
float3 reconstructHit(uint2 tid, float z) {
    if (z >= 1.0e9) return float3(0.0);   // sky pixel
    float2 dim = float2(float(width), float(height));
    float2 uv  = (float2(tid) + 0.5) / dim;
    float2 uv0 = uv * 2.0 - 1.0;
    uv0.x *= fwd_aspect.w;
    // Match PathTrace's G-buffer primary ray, which flips NDC y
    // (uv0.y = -uv0.y) before building rd. Without this the reconstructed
    // hit is mirrored vertically about the view axis, so temporal/spatial
    // reuse evaluates the survivor's target pdf at the wrong world point.
    uv0.y = -uv0.y;
    float3 rd = normalize(fwd_aspect.xyz
                          + right_xyz.xyz * (uv0.x * pos_fovtan.w)
                          + up_xyz.xyz    * (uv0.y * pos_fovtan.w));
    // depth_view = h.t * dot(rd, normalize(fwd_aspect.xyz)) per PathTrace's
    // G-buffer write. Solve for h.t and reconstruct ro + rd * h.t.
    float fwd_dot = dot(rd, normalize(fwd_aspect.xyz));
    float t = z / max(fwd_dot, 1e-6);
    return pos_fovtan.xyz + rd * t;
}

// Load a light record at the given index. Matches PathTrace.slang's
// kLightStrideFloat4 = 5 float4s layout (light_prims #73 + rotation
// gizmo follow-up). v4 = orientation quaternion (xyzw); identity is
// (0,0,0,1) and produces no shape change at sample time.
struct LightRec {
    float3 pos;
    float  radius;
    float3 intensity;
    uint   type;
    float3 dir;
    float  cos_outer;
    float3 u_or_inner;
    float  v_half;
    float4 orient;     // rotation gizmo (xyzw); identity = (0,0,0,1)
};

static const uint kLightStride = 5u;
static const uint LIGHT_POINT  = 0u;
static const uint LIGHT_SPOT   = 1u;
static const uint LIGHT_SPHERE = 2u;
static const uint LIGHT_QUAD   = 3u;

// Local copy of PathTrace.slang's quatRotate; the ReSTIR modules don't
// import PathTrace so the helper lives here. Identity (0,0,0,1) yields
// v unchanged: cross(zero, anything) = 0.
float3 lightQuatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

LightRec loadLight(uint idx) {
    uint base = idx * kLightStride;
    float4 v0 = light_prims[base + 0u];
    float4 v1 = light_prims[base + 1u];
    float4 v2 = light_prims[base + 2u];
    float4 v3 = light_prims[base + 3u];
    float4 v4 = light_prims[base + 4u];
    LightRec L;
    L.pos        = v0.xyz;
    L.radius     = v0.w;
    L.intensity  = v1.rgb;
    L.type       = uint(v1.w + 0.5);
    L.dir        = v2.xyz;
    L.cos_outer  = v2.w;
    L.u_or_inner = v3.xyz;
    L.v_half     = v3.w;
    L.orient     = v4;
    return L;
}

// Re-evaluate the prev-frame survivor's target pdf at THIS pixel's
// receiver. Reservoir.light_pos is the sample point; just recompute
// dist, cos_light_at_sample (for area lights), Le * cos / dist^2,
// then luminance. Mirrors PathTrace's per-type Le computation in
// sampleAnalyticLight().
float prevSurvivorTargetPdf(Reservoir prev, float3 hit_pt, float3 hit_n) {
    if (prev.light_idx == kRestirInvalidLight) return 0.0;
    if (prev.light_idx >= light_count)         return 0.0;
    LightRec L = loadLight(prev.light_idx);
    float3 to_light = prev.light_pos - hit_pt;
    float  dist2    = max(dot(to_light, to_light), 1e-12);
    float  dist     = sqrt(dist2);
    float3 wi       = to_light / dist;
    float  n_dot_l  = dot(hit_n, wi);
    if (n_dot_l <= 0.0) return 0.0;

    float3 Le_pre;
    if (L.type == LIGHT_POINT) {
        Le_pre = L.intensity / dist2;
    } else if (L.type == LIGHT_SPOT) {
        // Rotate the spot axis through the orientation quaternion so the
        // target-pdf evaluation matches PathTrace's NEE math (identity
        // quat is a no-op).
        float3 axis   = normalize(lightQuatRotate(L.orient, L.dir));
        float cos_dir = dot(-wi, axis);
        float cos_in  = L.u_or_inner.x;
        float cos_out = L.cos_outer;
        float falloff = 0.0;
        if (cos_dir >= cos_in) {
            falloff = 1.0;
        } else if (cos_dir > cos_out) {
            falloff = (cos_dir - cos_out) / max(cos_in - cos_out, 1e-6);
        }
        Le_pre = L.intensity * (falloff / dist2);
    } else if (L.type == LIGHT_SPHERE) {
        // The chosen sample point's outward normal is prev.light_n
        // (preserved through resampling). cos_light must be positive
        // for the sample to contribute.
        float cos_light = max(0.0, dot(prev.light_n, -wi));
        if (cos_light <= 0.0) return 0.0;
        Le_pre = L.intensity;
    } else {
        // QUAD: similar to sphere. prev.light_n holds the quad normal.
        float cos_light = dot(prev.light_n, -wi);
        if (cos_light <= 0.0) return 0.0;
        Le_pre = L.intensity;
    }
    float3 contrib = Le_pre * n_dot_l;
    return max(dot(contrib, float3(0.2126, 0.7152, 0.0722)), 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;

    // Master gate: if ReSTIR is off entirely, leave the buffer untouched
    // (PathTrace's initial reservoir flows straight through to the spatial
    // pass, which also no-ops, and the final shading pass reads from it).
    if (restir_enabled == 0u) return;

    uint pixel_idx = tid.y * width + tid.x;
    Reservoir curr = loadReservoir(reservoir_curr_in, pixel_idx);

    // Sky pixels: skip temporal reuse entirely (no surface to reproject).
    // PathTrace's WRS pass writes an empty reservoir at sky pixels; we
    // just copy it through.
    float z_c = depth_tex[tid.xy];
    if (z_c >= 1.0e9) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }

    // If temporal reuse is off, copy curr through unmerged.
    if (temporal_enabled == 0u) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }

    // Reproject. PathTrace writes motion = (prev_pix - curr_pix), so
    // prev_pix = (curr_pix + motion).
    float2 motion = motion_tex[tid.xy];
    int2 prev_px  = int2(int(tid.x) + int(round(motion.x)),
                         int(tid.y) + int(round(motion.y)));
    if (prev_px.x < 0 || prev_px.x >= int(width) ||
        prev_px.y < 0 || prev_px.y >= int(height)) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }

    // Disocclusion gate: depth + normal must agree with the prev pixel.
    // The 1.0e9 sky-sentinel reject keeps sky-to-surface transitions
    // from feeding garbage reservoirs into geometry.
    float z_p = depth_tex[uint2(uint(prev_px.x), uint(prev_px.y))];
    if (z_p >= 1.0e9) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }
    float dz_rel = abs(z_c - z_p) / max(z_c, 1e-3);
    if (dz_rel > kDepthDisoccTol) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }

    float3 n_c = normal_tex[tid.xy].xyz;
    float3 n_p = normal_tex[uint2(uint(prev_px.x), uint(prev_px.y))].xyz;
    float nlen_c = length(n_c);
    float nlen_p = length(n_p);
    if (nlen_c < 1e-5 || nlen_p < 1e-5) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }
    n_c = n_c / nlen_c;
    n_p = n_p / nlen_p;
    if (dot(n_c, n_p) < kNormalDisoccTol) {
        storeReservoir(reservoir_curr_out, pixel_idx, curr);
        return;
    }

    // Load and cap prev reservoir.
    uint prev_idx = uint(prev_px.y) * width + uint(prev_px.x);
    Reservoir prev = loadReservoir(reservoir_prev_in, prev_idx);
    if (prev.M > kRestirTemporalMaxM) {
        // Scale w_sum to preserve W: W = w_sum / (M * p_hat); cap M
        // and scale w_sum together so W stays unchanged.
        float scale = float(kRestirTemporalMaxM) / float(prev.M);
        prev.w_sum *= scale;
        prev.M      = kRestirTemporalMaxM;
    }

    // Re-evaluate prev survivor's target pdf at THIS pixel. Without
    // this re-eval the merge would adopt a stale p_hat from the
    // neighbour and bias the estimator on grazing surfaces.
    float3 hit_pt = reconstructHit(tid.xy, z_c);
    float p_hat_other = prevSurvivorTargetPdf(prev, hit_pt, n_c);

    // RNG: same per-pixel + per-frame mix as PathTrace's seed setup.
    uint seed = pcgHash(tid.x ^ (tid.y * 1973u) ^ (frame_index * 9277u));
    float u = randf(seed);

    // Combine. bias_mode is a stub here for Phase A — biased mode is
    // straight WRS (above); unbiased adds per-pair MIS weighting in
    // the spatial pass primarily. Temporal MIS weighting is a
    // follow-up; for Phase A both modes use the same temporal merge.
    combineReservoir(curr, prev, p_hat_other, u);
    finalizeReservoir(curr);

    storeReservoir(reservoir_curr_out, pixel_idx, curr);
}

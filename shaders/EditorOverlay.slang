// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// 3D transform gizmo overlay (editor 3D gizmos).
//
// Renders a small set of world-space line segments over the rendered
// frame so the editor's selected primitive carries an X/Y/Z arrow
// triad (translate mode), three rotation rings (rotate mode), or
// arrows-with-cube-tips (scale mode). Each segment carries its own
// RGB colour and on-screen pixel half-thickness; the kernel projects
// the endpoints to screen space using the camera matrices in the
// push constants and then runs distance-to-segment per swapchain
// pixel for cheap anti-aliased lines.
//
// Architecture choice: this is NOT a graphics pipeline. The RHI is
// compute-only so the kernel runs over the swapchain image directly,
// writing display-space RGBA (already gamma-encoded for the sRGB
// surface). The trade-off is that the gizmo doesn't go through the
// ACES tonemap curve -- which is what editor users actually want
// anyway: handles need to be readable at any exposure, not artistically
// blended into the scene. The spec mentions inserting "before tonemap"
// to inherit the curve; in practice that's Metal-only, and skipping
// the curve gives us uniform behaviour across Metal / Vulkan /
// Software with no per-backend dispatch ordering.
//
// V1 scope: always-on-top (xray-style, like Blender's default gizmo).
// Depth-test against the scene's primary-hit depth buffer is deferred
// to a follow-up (the engine's depth_tex is only allocated when the
// denoiser is active, so wiring it through unconditionally would
// require additional lifecycle plumbing).
//
// Bindings (Vulkan shared layout: see VulkanDevice's kSlotToTexBinding
// / kSlotToBufBinding):
//   Engine texture slot 0 -> vk::binding(0) = swapchain RW
//   Engine buffer  slot 1 -> vk::binding(3) = segments (RW for layout match)
//
// MSL note: Slang assigns MSL buffer slots in declaration order on
// Metal, so segments lands at MSL buffer(0) unless we shift it. The
// engine's BindBuffer(slot=1) targets MSL buffer(1); to keep that
// alignment we declare a small _slot_dummy0 buffer at engine slot 0
// (just like PerfOverlay does). Hit by a never-taken branch so the
// optimizer keeps it in the slot table.

// vk::image_format("rgba8") -- swapchain is BGRA8_UNORM on both
// Win/Mac; matches PerfOverlay.slang's reasoning. Without the
// format hint NVIDIA's compiler can silently drop the store.
[[vk::binding(0, 0), vk::image_format("rgba8")]]
RWTexture2D<float4> out_image;

// Dummy at vk::binding(4) (mesh_indices in the shared PathTrace
// layout, also STORAGE_BUFFER so the type matches). Forced-emitted
// in a never-taken branch below; on Vulkan the read gets whatever
// the prior dispatch left there but never executes. Same trick as
// PerfOverlay.slang -- shifts segments to MSL buffer(1) in
// declaration order.
[[vk::binding(4, 0)]] StructuredBuffer<uint>     _slot_dummy0;

// Segment list. Each record occupies 3 float4s (48 bytes):
//   s0.xyz = world-space endpoint A
//   s0.w   = pixel half-thickness (e.g. 1.5 for a 3-pixel line);
//            negative means "filled triangle"
//   s1.xyz = world-space endpoint B
//   s1.w   = unused (reserved for future depth-bias)
//   s2.xyz = line colour OR triangle vertex C
//   s2.w   = triangle colour code
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> segments;

[[vk::push_constant]]
cbuffer Push {
    // Camera basis -- same layout as the engine's StarsComposite
    // push so we can repurpose the existing per-frame camera math.
    float4 pos_fovtan;   // .xyz = camera origin, .w = fovYTan
    float4 fwd_aspect;   // .xyz = forward (-Z in world), .w = aspect ratio (W/H)
    float4 right_xyz;    // .xyz = camera right
    float4 up_xyz;       // .xyz = camera up

    // Number of segments in the list (NOT float4 entries -- raw segment
    // count). The dispatch grid is sized to swapchain pixels and each
    // thread iterates over the segment list.
    uint num_segments;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Pixel-space distance from (px, py) to the line segment (x0,y0)-(x1,y1).
float seg_dist(float px, float py,
               float x0, float y0,
               float x1, float y1)
{
    float vx = x1 - x0;
    float vy = y1 - y0;
    float wx = px - x0;
    float wy = py - y0;
    float len2 = vx * vx + vy * vy;
    if (len2 < 1e-6) {
        float dx = px - x0, dy = py - y0;
        return sqrt(dx * dx + dy * dy);
    }
    float t = clamp((wx * vx + wy * vy) / len2, 0.0, 1.0);
    float qx = x0 + t * vx;
    float qy = y0 + t * vy;
    float dx = px - qx, dy = py - qy;
    return sqrt(dx * dx + dy * dy);
}

float edge2(float2 a, float2 b, float2 p) {
    float2 e = b - a;
    float2 w = p - a;
    return e.x * w.y - e.y * w.x;
}

bool point_in_tri(float2 p, float2 a, float2 b, float2 c) {
    float e0 = edge2(a, b, p);
    float e1 = edge2(b, c, p);
    float e2 = edge2(c, a, p);
    bool has_neg = (e0 < 0.0) || (e1 < 0.0) || (e2 < 0.0);
    bool has_pos = (e0 > 0.0) || (e1 > 0.0) || (e2 > 0.0);
    return !(has_neg && has_pos);
}

float3 fill_color_from_code(float code) {
    if (code > 9.5) return float3(1.0, 0.95, 0.10);
    if (code > 2.5) return float3(0.20, 0.40, 1.00);
    if (code > 1.5) return float3(0.20, 0.85, 0.20);
    if (code > 0.5) return float3(0.95, 0.15, 0.18);
    return float3(0.85, 0.85, 0.85);
}

// Project a world-space point into screen-space pixels via the
// standard pinhole camera. Sets is_behind = true if the point is
// behind the camera near plane (we skip drawing such segments).
float2 project(float3 world_p, uint2 dim, out bool is_behind)
{
    float3 to_p   = world_p - pos_fovtan.xyz;
    float3 fwd    = fwd_aspect.xyz;
    float3 right  = right_xyz.xyz;
    float3 up     = up_xyz.xyz;

    float z_view = dot(to_p, fwd);
    is_behind = (z_view <= 0.001);
    if (is_behind) {
        return float2(0.0, 0.0);
    }
    float xs = dot(to_p, right) / z_view;
    float ys = dot(to_p, up)    / z_view;

    float ft = pos_fovtan.w;
    float aspect = fwd_aspect.w;
    // Match StarsComposite's projection:
    //   uv = ((px + 0.5) / dim) * 2 - 1; uv.y = -uv.y;
    //   rd = normalize(fwd + right * uv.x * fovYTan * aspect
    //                      + up * uv.y * fovYTan);
    // Inverse: u_ndc = (xs / (ft * aspect)), v_ndc = (ys / ft);
    //          y in screen pixels is (1 - (v_ndc + 1) / 2) * H
    float u_ndc = xs / (ft * aspect);
    float v_ndc = ys / ft;
    float px = (u_ndc * 0.5 + 0.5) * float(dim.x);
    float py = (1.0 - (v_ndc * 0.5 + 0.5)) * float(dim.y);
    return float2(px, py);
}

// sRGB OETF -- matches Tonemap.slang. We write display-space bytes
// directly to the swapchain (which is BGRA8_UNORM, not sRGB-format),
// so the engine's other gamma-encoded paths and the gizmo end up at
// the same brightness on screen.
float3 srgb_encode(float3 linear_in) {
    float3 c = max(linear_in, float3(0.0));
    float3 lo = 12.92 * c;
    float3 hi = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return lerp(lo, hi, step(0.0031308, c));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Force the dummy storage buffer into Slang's MSL slot table.
    // Same shape as PerfOverlay: a runtime sentinel value the
    // optimizer can't fold away. Never executes on real dispatches.
    if (num_segments == 0xDEADBEEFu) {
        let v = _slot_dummy0[0];
        out_image[uint2(0, 0)] = float4(float(v), 0, 0, 0);
    }

    uint2 dim;
    out_image.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (num_segments == 0u) return;

    float fx = float(tid.x) + 0.5;
    float fy = float(tid.y) + 0.5;

    float4 dst = out_image[uint2(tid.x, tid.y)];

    // Walk every segment. Each contributes a thin anti-aliased line
    // tinted by its own colour. We accumulate via alpha blending so
    // overlapping arrows produce a reasonable composite.
    for (uint i = 0u; i < num_segments; ++i) {
        uint base = i * 3u;
        float4 s0 = segments[base + 0u];
        float4 s1 = segments[base + 1u];
        float4 s2 = segments[base + 2u];

        float3 wa  = s0.xyz;
        float half_thickness = s0.w;
        float3 wb  = s1.xyz;
        float3 col = s2.xyz;

        bool behind_a = false, behind_b = false;
        float2 pa = project(wa, dim, behind_a);
        float2 pb = project(wb, dim, behind_b);
        if (half_thickness < 0.0) {
            bool behind_c = false;
            float2 pc = project(s2.xyz, dim, behind_c);
            if (behind_a || behind_b || behind_c) continue;
            float2 p = float2(fx, fy);
            if (!point_in_tri(p, pa, pb, pc)) continue;
            float d0 = seg_dist(fx, fy, pa.x, pa.y, pb.x, pb.y);
            float d1 = seg_dist(fx, fy, pb.x, pb.y, pc.x, pc.y);
            float d2 = seg_dist(fx, fy, pc.x, pc.y, pa.x, pa.y);
            float edge_d = min(d0, min(d1, d2));
            float aa = 0.42 * saturate(edge_d + 0.5);
            float3 src_dpy = srgb_encode(fill_color_from_code(s2.w));
            dst.rgb = lerp(dst.rgb, src_dpy, aa);
            continue;
        }
        half_thickness = max(half_thickness, 0.5);
        // If either endpoint is behind the camera, skip the segment
        // entirely. Cheap and avoids near-plane clipping math; the
        // gizmo is small relative to camera near plane in typical use.
        if (behind_a || behind_b) continue;

        float d = seg_dist(fx, fy, pa.x, pa.y, pb.x, pb.y);
        if (d > half_thickness + 1.0) continue;

        // Anti-aliased coverage: full inside the half-thickness, fades
        // linearly to zero over the next pixel.
        float aa = saturate((half_thickness + 1.0 - d));
        if (aa <= 0.0) continue;

        // Apply sRGB encode so display brightness matches.
        float3 src_dpy = srgb_encode(col);

        // Alpha-over composite onto the existing pixel.
        dst.rgb = lerp(dst.rgb, src_dpy, aa);
    }

    out_image[uint2(tid.x, tid.y)] = dst;
}

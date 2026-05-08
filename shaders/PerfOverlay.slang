// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// RHI-rendered performance overlay -- final compute pass that
// composites a small HUD onto the post-tonemap swapchain.  Avoids the
// WS_EX_LAYERED-over-Vulkan-swapchain compositing artifacts that the
// native (GDI / NSPanel) overlay hits, and the result shows up in
// `screenshot` PPMs because it lives inside the rendered frame.
//
// v1 scope: panel background + sparkline only.  No text.  A bitmap
// font for digit / label rendering can land in a follow-up; the
// existing native overlay covers the textual readout in the
// meantime.  Cvar `r_perf_overlay_mode native|rhi` selects which
// overlay the engine drives.
//
// Bindings (compatible with the shared 16-slot Vulkan pipeline
// layout, slot 0 = swapchain RW, slot 3 = storage buffer):
//   binding 0  : swapchain RWTexture2D<float4> (read + write).
//   binding 3  : draw list RWStructuredBuffer<DrawCmd>.
//
// Push constants (fits in the engine's 112B unified push range):
//   panel_x, panel_y, panel_w, panel_h : panel rect in pixels.
//   num_cmds                           : number of valid entries in the buffer.
//   panel_color_rgba                   : packed 8-bit RGBA, pre-blend.
//   accent_color_rgba                  : packed 8-bit RGBA for the top hairline.
//   graph_color_rgba                   : packed 8-bit RGBA for the sparkline.

[[vk::binding(0, 0)]] RWTexture2D<float4>          out_image;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4>    draw_list;

[[vk::push_constant]]
cbuffer Push {
    uint panel_x;
    uint panel_y;
    uint panel_w;
    uint panel_h;
    uint num_cmds;
    uint panel_color_rgba;
    uint accent_color_rgba;
    uint graph_color_rgba;
};

// DrawCmd packed into uint4:
//   .x = kind (0 = filled rect, 1 = sparkline segment, 2 = horizontal rule)
//   .y = x0  (or rect.x)
//   .z = y0  (or rect.y) -- for segments, low 16 = y0, high 16 = y1
//   .w = payload:
//          rect    : low 16 = w, high 16 = h
//          segment : low 16 = x1, high 16 = (unused; y1 already in z high)
//          rule    : low 16 = w, high 16 = unused
// Color is a global per-cmd kind (panel/accent/graph) -- avoids
// burning a uint per cmd; mode is implied by `kind`.
//
// Limited geometry vocabulary by design: anything more complex
// (text, gauges, etc.) goes into a richer follow-up.  Today's
// overlay is panel + accent line + sparkline polyline.

float4 unpack_rgba(uint c) {
    float4 r;
    r.x = float((c >>  0) & 0xFFu) / 255.0;
    r.y = float((c >>  8) & 0xFFu) / 255.0;
    r.z = float((c >> 16) & 0xFFu) / 255.0;
    r.w = float((c >> 24) & 0xFFu) / 255.0;
    return r;
}

// Distance from pixel (px, py) to the line segment (x0,y0)-(x1,y1)
// in pixel units.  Used for anti-aliased sparkline rasterization
// (clamp at 1 px = full coverage, fade to 0 over the next pixel).
float seg_dist(float px, float py, float x0, float y0, float x1, float y1) {
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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Dispatch is sized to cover the panel rect; each thread maps to
    // one swapchain pixel inside (or just past) the panel.  Skip
    // pixels outside the panel.
    uint px = panel_x + tid.x;
    uint py = panel_y + tid.y;
    if (tid.x >= panel_w || tid.y >= panel_h) return;

    float4 src = out_image[uint2(px, py)];
    float4 dst = src;

    float4 panel_col  = unpack_rgba(panel_color_rgba);
    float4 accent_col = unpack_rgba(accent_color_rgba);
    float4 graph_col  = unpack_rgba(graph_color_rgba);

    // Panel background (alpha-over the rendered scene).
    dst.rgb = lerp(dst.rgb, panel_col.rgb, panel_col.a);

    // Top accent hairline at row 0 of the panel.
    if (tid.y == 0u) {
        dst.rgb = lerp(dst.rgb, accent_col.rgb, accent_col.a);
    }

    // Walk the draw list.  Cap at num_cmds; the buffer is sized for
    // tier-3 worst case (sparkline points + chrome).
    float fx = float(tid.x);
    float fy = float(tid.y);
    for (uint i = 0u; i < num_cmds; ++i) {
        uint4 cmd  = draw_list[i];
        uint  kind = cmd.x;
        if (kind == 0u) {
            // Filled rect, panel-local coordinates.
            uint rx = cmd.y;
            uint ry = cmd.z;
            uint rw = cmd.w & 0xFFFFu;
            uint rh = (cmd.w >> 16) & 0xFFFFu;
            if (tid.x >= rx && tid.x < rx + rw &&
                tid.y >= ry && tid.y < ry + rh) {
                dst.rgb = lerp(dst.rgb, panel_col.rgb, 0.65);
            }
        } else if (kind == 1u) {
            // Sparkline segment from (x0, y0) -> (x1, y1) in panel-local
            // coords.  z packs y0 in low 16, y1 in high 16 (signed-ish
            // but we treat as unsigned and stay within [0, panel_h)).
            // y axis is panel-local with y=0 at the top, increasing down.
            float x0 = float(cmd.y);
            float y0 = float(cmd.z & 0xFFFFu);
            float x1 = float(cmd.w & 0xFFFFu);
            float y1 = float((cmd.z >> 16) & 0xFFFFu);
            float d = seg_dist(fx, fy, x0, y0, x1, y1);
            float aa = saturate(1.5 - d);   // 1 px hard, 0.5 px feather
            if (aa > 0.0) {
                dst.rgb = lerp(dst.rgb, graph_col.rgb, aa * graph_col.a);
            }
        } else if (kind == 2u) {
            // Horizontal rule (1 px tall, dashed-style filled rect),
            // panel-local.  Used for the 60 fps reference at 16.67 ms.
            uint rx = cmd.y;
            uint ry = cmd.z & 0xFFFFu;
            uint rw = cmd.w & 0xFFFFu;
            if (tid.x >= rx && tid.x < rx + rw && tid.y == ry) {
                // Dashed: 2 px on, 2 px off.
                if ((tid.x & 3u) < 2u) {
                    dst.rgb = lerp(dst.rgb, accent_col.rgb, 0.35);
                }
            }
        }
    }

    out_image[uint2(px, py)] = dst;
}

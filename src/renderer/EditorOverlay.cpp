// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "EditorOverlay.h"

#include <algorithm>
#include <cmath>

namespace pt::renderer {

// ============================================================================
// Free-function helpers
// ============================================================================

glm::vec3 ClosestPointOnLineToRay(const glm::vec3& line_origin,
                                  const glm::vec3& line_dir,
                                  const glm::vec3& ray_origin,
                                  const glm::vec3& ray_dir)
{
    // Classic two-line closest-point: minimise || (line_origin + t*line_dir)
    //                                      - (ray_origin  + s*ray_dir) ||^2
    // over (t, s). Setting partials to zero gives a 2x2 linear system.
    const glm::vec3 w0 = line_origin - ray_origin;
    const float a = glm::dot(line_dir, line_dir);
    const float b = glm::dot(line_dir, ray_dir);
    const float c = glm::dot(ray_dir, ray_dir);
    const float d = glm::dot(line_dir, w0);
    const float e = glm::dot(ray_dir, w0);
    const float denom = a * c - b * b;
    if (std::abs(denom) < 1e-6f) {
        // Lines are parallel; return line_origin (no movement).
        return line_origin;
    }
    const float t = (b * e - c * d) / denom;
    return line_origin + line_dir * t;
}

bool ProjectToScreen(const glm::vec3& world_p,
                     const Camera& cam, float aspect,
                     int fb_w, int fb_h,
                     glm::vec2& out_px)
{
    const glm::vec3 to_p   = world_p - cam.pos;
    const glm::vec3 fwd    = cam.Forward();
    const glm::vec3 right  = cam.Right();
    const glm::vec3 up     = cam.Up();
    const float z_view = glm::dot(to_p, fwd);
    if (z_view <= 0.001f) return false;
    const float xs = glm::dot(to_p, right) / z_view;
    const float ys = glm::dot(to_p, up)    / z_view;
    const float ft = cam.FovYTan();
    if (ft <= 0.0f || aspect <= 0.0f) return false;
    const float u_ndc = xs / (ft * aspect);
    const float v_ndc = ys / ft;
    out_px.x = (u_ndc * 0.5f + 0.5f) * static_cast<float>(fb_w);
    out_px.y = (1.0f - (v_ndc * 0.5f + 0.5f)) * static_cast<float>(fb_h);
    return true;
}

bool ScreenToWorldRay(const Camera& cam, float aspect,
                      int fb_w, int fb_h,
                      double mouse_x, double mouse_y,
                      glm::vec3& out_origin,
                      glm::vec3& out_dir)
{
    if (fb_w <= 0 || fb_h <= 0) return false;
    const float ft = cam.FovYTan();
    if (ft <= 0.0f || aspect <= 0.0f) return false;
    // Pixel -> NDC (matching shader pipeline).
    const float u = (static_cast<float>(mouse_x) / float(fb_w)) * 2.0f - 1.0f;
    const float v = (1.0f - static_cast<float>(mouse_y) / float(fb_h)) * 2.0f - 1.0f;
    const glm::vec3 fwd   = cam.Forward();
    const glm::vec3 right = cam.Right();
    const glm::vec3 up    = cam.Up();
    glm::vec3 d = fwd + right * (u * ft * aspect) + up * (v * ft);
    const float L = glm::length(d);
    if (L < 1e-6f) return false;
    out_origin = cam.pos;
    out_dir    = d / L;
    return true;
}

// Distance from screen-space mouse to the projected line segment
// (pa->pb). Returns FLT_MAX if either endpoint is behind the camera.
static float ScreenSegmentDistance(const glm::vec3& wa, const glm::vec3& wb,
                                   const Camera& cam, float aspect,
                                   int fb_w, int fb_h,
                                   double mouse_x, double mouse_y)
{
    glm::vec2 pa, pb;
    if (!ProjectToScreen(wa, cam, aspect, fb_w, fb_h, pa)) return 1e30f;
    if (!ProjectToScreen(wb, cam, aspect, fb_w, fb_h, pb)) return 1e30f;
    const glm::vec2 m{static_cast<float>(mouse_x),
                      static_cast<float>(mouse_y)};
    const glm::vec2 v = pb - pa;
    const glm::vec2 w = m - pa;
    const float len2 = glm::dot(v, v);
    if (len2 < 1e-6f) {
        return glm::length(m - pa);
    }
    const float t = std::clamp(glm::dot(w, v) / len2, 0.0f, 1.0f);
    const glm::vec2 q = pa + t * v;
    return glm::length(m - q);
}

// ============================================================================
// EditorOverlay
// ============================================================================

EditorOverlay::EditorOverlay() {
    segs_.reserve(128);
}

void EditorOverlay::ClearSegments() {
    segs_.clear();
}

glm::vec3 EditorOverlay::ColorFor(Axis a, Axis hl) const {
    if (a == hl) return glm::vec3(1.0f, 0.95f, 0.10f);   // yellow highlight
    switch (a) {
        case Axis::X: return glm::vec3(0.95f, 0.15f, 0.18f);
        case Axis::Y: return glm::vec3(0.20f, 0.85f, 0.20f);
        case Axis::Z: return glm::vec3(0.20f, 0.40f, 1.00f);
        default:      return glm::vec3(0.85f, 0.85f, 0.85f);
    }
}

void EditorOverlay::EmitAxisSegment(const glm::vec3& a, const glm::vec3& b,
                                    Axis axis, Axis hovered_or_dragged,
                                    float thickness)
{
    Segment s;
    s.a = a;
    s.half_thickness = thickness;
    s.b = b;
    s.depth_bias = 0.0f;
    s.color = ColorFor(axis, hovered_or_dragged);
    segs_.push_back(s);
}

void EditorOverlay::AppendGizmo(const glm::vec3& origin, float size,
                                Axis hovered_axis)
{
    switch (mode_) {
        case Mode::Translate: AppendTranslateGizmo(origin, size, hovered_axis); break;
        case Mode::Rotate:    AppendRotateGizmo   (origin, size, hovered_axis); break;
        case Mode::Scale:     AppendScaleGizmo    (origin, size, hovered_axis); break;
    }
}

void EditorOverlay::AppendTranslateGizmo(const glm::vec3& O, float L,
                                         Axis hl)
{
    // Three axis arms. Cone tip rendered as a small fan of segments
    // around the arrow head so the tip reads as a triangle silhouette
    // from any angle.
    const float shaft_h = L * 0.85f;
    const float tip_len = L * 0.15f;
    const float tip_r   = L * 0.08f;

    auto draw_arm = [&](const glm::vec3& axis_unit, Axis a) {
        const glm::vec3 shaft_end = O + axis_unit * shaft_h;
        const glm::vec3 tip_end   = O + axis_unit * L;
        EmitAxisSegment(O, shaft_end, a, hl, 2.0f);

        // Cone tip: 8 triangular fan segments around the axis. We
        // approximate the cone surface by drawing 8 base-perimeter
        // points connected to the apex.
        glm::vec3 ortho1{0.0f}, ortho2{0.0f};
        if (std::abs(axis_unit.x) < 0.9f) {
            ortho1 = glm::normalize(glm::cross(axis_unit, glm::vec3(1, 0, 0)));
        } else {
            ortho1 = glm::normalize(glm::cross(axis_unit, glm::vec3(0, 1, 0)));
        }
        ortho2 = glm::normalize(glm::cross(axis_unit, ortho1));
        constexpr int N = 8;
        for (int i = 0; i < N; ++i) {
            const float t0 = (i       * 6.28318530718f) / N;
            const float t1 = ((i + 1) * 6.28318530718f) / N;
            const glm::vec3 p0 = shaft_end + ortho1 * (std::cos(t0) * tip_r)
                                           + ortho2 * (std::sin(t0) * tip_r);
            const glm::vec3 p1 = shaft_end + ortho1 * (std::cos(t1) * tip_r)
                                           + ortho2 * (std::sin(t1) * tip_r);
            // perimeter -> apex (two ribs per fan slice)
            EmitAxisSegment(p0, tip_end, a, hl, 1.0f);
            // perimeter ring segment
            EmitAxisSegment(p0, p1, a, hl, 1.0f);
        }
        // Stub from origin to shaft start so the arm doesn't visually
        // detach when the origin is very close to camera (no-op when
        // shaft_h > 0).
        (void)tip_len;
    };

    draw_arm(glm::vec3(1, 0, 0), Axis::X);
    draw_arm(glm::vec3(0, 1, 0), Axis::Y);
    draw_arm(glm::vec3(0, 0, 1), Axis::Z);
}

void EditorOverlay::AppendRotateGizmo(const glm::vec3& O, float L, Axis hl)
{
    // Three rings (one per axis plane) approximated as a polyline of
    // 32 segments. The ring radius matches the arm length so
    // translate-mode and rotate-mode read at similar visual scale.
    constexpr int kSeg = 32;
    auto draw_ring = [&](const glm::vec3& axis_unit, Axis a) {
        glm::vec3 t1{0.0f}, t2{0.0f};
        if (std::abs(axis_unit.x) < 0.9f) {
            t1 = glm::normalize(glm::cross(axis_unit, glm::vec3(1, 0, 0)));
        } else {
            t1 = glm::normalize(glm::cross(axis_unit, glm::vec3(0, 1, 0)));
        }
        t2 = glm::normalize(glm::cross(axis_unit, t1));
        glm::vec3 prev = O + t1 * L;
        for (int i = 1; i <= kSeg; ++i) {
            const float ang = (i * 6.28318530718f) / kSeg;
            const glm::vec3 cur = O + t1 * (std::cos(ang) * L)
                                    + t2 * (std::sin(ang) * L);
            EmitAxisSegment(prev, cur, a, hl, 1.5f);
            prev = cur;
        }
    };
    draw_ring(glm::vec3(1, 0, 0), Axis::X);
    draw_ring(glm::vec3(0, 1, 0), Axis::Y);
    draw_ring(glm::vec3(0, 0, 1), Axis::Z);
}

void EditorOverlay::AppendScaleGizmo(const glm::vec3& O, float L, Axis hl)
{
    // Translate gizmo with cube tips in place of cones. We reuse the
    // translate arms then add 12 box edges per tip.
    const float shaft_h = L * 0.85f;
    const float box_h   = L * 0.12f;

    auto draw_arm_with_cube = [&](const glm::vec3& axis_unit, Axis a) {
        const glm::vec3 shaft_end = O + axis_unit * shaft_h;
        const glm::vec3 box_center = O + axis_unit * (L - box_h * 0.5f);
        EmitAxisSegment(O, shaft_end, a, hl, 2.0f);

        glm::vec3 t1{0.0f}, t2{0.0f};
        if (std::abs(axis_unit.x) < 0.9f) {
            t1 = glm::normalize(glm::cross(axis_unit, glm::vec3(1, 0, 0)));
        } else {
            t1 = glm::normalize(glm::cross(axis_unit, glm::vec3(0, 1, 0)));
        }
        t2 = glm::normalize(glm::cross(axis_unit, t1));
        const float hs = box_h * 0.5f;
        // 8 cube corners.
        glm::vec3 c[8];
        for (int i = 0; i < 8; ++i) {
            const float ax = (i & 1) ? 1.0f : -1.0f;
            const float ay = (i & 2) ? 1.0f : -1.0f;
            const float az = (i & 4) ? 1.0f : -1.0f;
            c[i] = box_center + axis_unit * (ax * hs)
                              + t1        * (ay * hs)
                              + t2        * (az * hs);
        }
        // 12 cube edges (4 along each of three axes).
        const int edges[12][2] = {
            {0,1},{2,3},{4,5},{6,7},
            {0,2},{1,3},{4,6},{5,7},
            {0,4},{1,5},{2,6},{3,7},
        };
        for (int e = 0; e < 12; ++e) {
            EmitAxisSegment(c[edges[e][0]], c[edges[e][1]], a, hl, 1.5f);
        }
    };

    draw_arm_with_cube(glm::vec3(1, 0, 0), Axis::X);
    draw_arm_with_cube(glm::vec3(0, 1, 0), Axis::Y);
    draw_arm_with_cube(glm::vec3(0, 0, 1), Axis::Z);
}

// --- Wave 9 light-gizmo ---
void EditorOverlay::AppendLightIcon(const glm::vec3& O, float size,
                                    const glm::vec3& color, bool highlighted)
{
    if (size <= 0.0f) return;

    // Highlighted lights read brighter + warmer so the selected one
    // stands out among many. We blend the tint toward the gizmo's
    // yellow highlight rather than discarding the chromaticity entirely
    // so a selected red light still hints red.
    glm::vec3 c = color;
    // Guard against an all-zero (black) light: render the marker in a
    // neutral grey so it's still visible / pickable.
    if (c.x + c.y + c.z < 1e-4f) c = glm::vec3(0.7f);
    float thickness = 1.5f;
    if (highlighted) {
        const glm::vec3 hl{1.0f, 0.95f, 0.10f};
        c = glm::mix(c, hl, 0.6f);
        // Lift overall brightness a touch so the selected glyph pops.
        c = glm::min(c * 1.4f + glm::vec3(0.15f), glm::vec3(1.0f));
        thickness = 2.0f;
    }

    auto emit = [&](const glm::vec3& a, const glm::vec3& b) {
        Segment s;
        s.a = a;
        s.half_thickness = thickness;
        s.b = b;
        s.depth_bias = 0.0f;
        s.color = c;
        s._pad = 0.0f;
        segs_.push_back(s);
    };

    // Sun-burst: short rays from the centre. The 6 cardinal arms make
    // the "+" read at any view angle; the 8 cube-corner diagonals fill
    // it out to a star so it never collapses to a single line when the
    // camera looks straight down an axis. A small inner gap keeps the
    // centre open (Blender's light empty / Unity's light gizmo vibe).
    const float inner = size * 0.18f;   // gap radius
    const float outer = size;           // ray tip radius
    const float diag  = size * 0.72f;   // diagonal rays a bit shorter

    const glm::vec3 axes[6] = {
        {+1, 0, 0}, {-1, 0, 0},
        {0, +1, 0}, {0, -1, 0},
        {0, 0, +1}, {0, 0, -1},
    };
    for (const auto& d : axes) {
        emit(O + d * inner, O + d * outer);
    }

    const float s3 = 0.57735027f;   // 1/sqrt(3): unit diagonal component
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 d{
            (i & 1) ? s3 : -s3,
            (i & 2) ? s3 : -s3,
            (i & 4) ? s3 : -s3,
        };
        emit(O + d * inner, O + d * diag);
    }
}
// --- end Wave 9 light-gizmo ---

// Screen-space distance from the mouse cursor to a polygonal ring in
// world space. The ring lies in the plane through `O` with normal
// `n` (unit vector), radius `L`. We sample the ring at `kRingSamples`
// points, project each pair to screen, and return the minimum
// projected-segment distance. Mirrors the segment-distance code path
// used for translate/scale axis arms but operates on the ring
// polyline that AppendRotateGizmo also draws (32 segments).
static float ScreenRingDistance(const glm::vec3& O, const glm::vec3& n,
                                float L,
                                const Camera& cam, float aspect,
                                int fb_w, int fb_h,
                                double mouse_x, double mouse_y)
{
    glm::vec3 t1{0.0f}, t2{0.0f};
    if (std::abs(n.x) < 0.9f) {
        t1 = glm::normalize(glm::cross(n, glm::vec3(1, 0, 0)));
    } else {
        t1 = glm::normalize(glm::cross(n, glm::vec3(0, 1, 0)));
    }
    t2 = glm::normalize(glm::cross(n, t1));
    // Use the same segment count as AppendRotateGizmo so the hit-test
    // geometry matches the visual ring exactly. Each segment's screen
    // distance to the mouse is evaluated; we keep the minimum.
    constexpr int kRingSamples = 32;
    float best = 1e30f;
    glm::vec3 prev = O + t1 * L;
    for (int i = 1; i <= kRingSamples; ++i) {
        const float ang = (i * 6.28318530718f) / kRingSamples;
        const glm::vec3 cur = O + t1 * (std::cos(ang) * L)
                                + t2 * (std::sin(ang) * L);
        const float d = ScreenSegmentDistance(prev, cur, cam, aspect,
                                              fb_w, fb_h, mouse_x, mouse_y);
        if (d < best) best = d;
        prev = cur;
    }
    return best;
}

EditorOverlay::Axis EditorOverlay::HitTest(const glm::vec3& O, float L,
                                           const Camera& cam, float aspect,
                                           int fb_w, int fb_h,
                                           double mx, double my,
                                           float hit_radius_px) const
{
    if (mode_ == Mode::Rotate) {
        // Rotate mode: test the three rings instead of the three arms.
        // The picked axis is the ring's NORMAL (so picking the X ring
        // returns Axis::X -- rotation around the world X axis).
        const float dX = ScreenRingDistance(O, glm::vec3(1, 0, 0), L,
                                            cam, aspect, fb_w, fb_h, mx, my);
        const float dY = ScreenRingDistance(O, glm::vec3(0, 1, 0), L,
                                            cam, aspect, fb_w, fb_h, mx, my);
        const float dZ = ScreenRingDistance(O, glm::vec3(0, 0, 1), L,
                                            cam, aspect, fb_w, fb_h, mx, my);
        float best = hit_radius_px;
        Axis  pick = Axis::None;
        if (dX < best) { best = dX; pick = Axis::X; }
        if (dY < best) { best = dY; pick = Axis::Y; }
        if (dZ < best) { best = dZ; pick = Axis::Z; }
        return pick;
    }

    // Translate / scale: test the three axis arms.
    const glm::vec3 ax_X = O + glm::vec3(L, 0, 0);
    const glm::vec3 ax_Y = O + glm::vec3(0, L, 0);
    const glm::vec3 ax_Z = O + glm::vec3(0, 0, L);
    const float dX = ScreenSegmentDistance(O, ax_X, cam, aspect, fb_w, fb_h, mx, my);
    const float dY = ScreenSegmentDistance(O, ax_Y, cam, aspect, fb_w, fb_h, mx, my);
    const float dZ = ScreenSegmentDistance(O, ax_Z, cam, aspect, fb_w, fb_h, mx, my);
    float best = hit_radius_px;
    Axis  pick = Axis::None;
    if (dX < best) { best = dX; pick = Axis::X; }
    if (dY < best) { best = dY; pick = Axis::Y; }
    if (dZ < best) { best = dZ; pick = Axis::Z; }
    return pick;
}

// Intersect a ray with an infinite plane through `plane_origin` with
// unit normal `plane_normal`. Returns the world-space hit point; the
// `out_hit` ref is unchanged if the ray is parallel (denom == 0) so
// the caller can fall back to a sensible default (the ring's "+X"
// reference tangent).
static bool RayPlaneIntersect(const glm::vec3& ro, const glm::vec3& rd,
                              const glm::vec3& plane_origin,
                              const glm::vec3& plane_normal,
                              glm::vec3& out_hit)
{
    const float denom = glm::dot(rd, plane_normal);
    if (std::abs(denom) < 1e-6f) return false;
    const float t = glm::dot(plane_origin - ro, plane_normal) / denom;
    if (t < 0.0f) return false;
    out_hit = ro + rd * t;
    return true;
}

void EditorOverlay::BeginDrag(Axis a, const glm::vec3& origin,
                              const Camera& cam, float aspect,
                              int fb_w, int fb_h,
                              double mx, double my)
{
    drag_axis_      = a;
    drag_mode_      = mode_;
    drag_start_pos_ = origin;
    switch (a) {
        case Axis::X: drag_axis_dir_ = glm::vec3(1, 0, 0); break;
        case Axis::Y: drag_axis_dir_ = glm::vec3(0, 1, 0); break;
        case Axis::Z: drag_axis_dir_ = glm::vec3(0, 0, 1); break;
        default:      drag_axis_     = Axis::None; return;
    }
    glm::vec3 ro{0.0f}, rd{0.0f};
    if (!ScreenToWorldRay(cam, aspect, fb_w, fb_h, mx, my, ro, rd)) {
        drag_anchor_world_ = origin;
        return;
    }
    if (drag_mode_ == Mode::Rotate) {
        // Rotate mode: anchor is the mouse ray's intersection with the
        // ring's plane, snapped to the ring's circle. drag_ring_radius_
        // matches the visual ring radius (size passed to AppendGizmo).
        // We treat `size` as the ring radius -- this isn't passed to
        // BeginDrag; we recover it from the world-space picked tangent
        // by snapping to the same plane the AppendRotateGizmo uses.
        // For Phase 1 we use the BeginDrag's `origin -> picked-point`
        // tangent regardless of magnitude, then store its world length
        // so UpdateRotateDrag normalises consistently.
        glm::vec3 ring_pick{};
        if (RayPlaneIntersect(ro, rd, origin, drag_axis_dir_, ring_pick)) {
            // Vector from origin into the ring plane.
            const glm::vec3 v = ring_pick - origin;
            // Strip any residual axis-direction component (numerical
            // noise -- RayPlaneIntersect already removes most of it).
            const glm::vec3 t = v - drag_axis_dir_ * glm::dot(v, drag_axis_dir_);
            const float r = glm::length(t);
            if (r > 1e-6f) {
                drag_anchor_world_ = t / r;   // unit tangent at angle 0
                drag_ring_radius_  = r;
            } else {
                drag_anchor_world_ = glm::vec3(0, 0, 0);
                drag_ring_radius_  = 1.0f;
            }
        } else {
            drag_anchor_world_ = glm::vec3(0, 0, 0);
            drag_ring_radius_  = 1.0f;
        }
    } else {
        // Translate / scale mode: closest point on the axis line.
        drag_anchor_world_ =
            ClosestPointOnLineToRay(origin, drag_axis_dir_, ro, rd);
    }
}

glm::vec3 EditorOverlay::UpdateDrag(const Camera& cam, float aspect,
                                    int fb_w, int fb_h,
                                    double mx, double my) const
{
    if (drag_axis_ == Axis::None) return drag_start_pos_;
    glm::vec3 ro{0.0f}, rd{0.0f};
    if (!ScreenToWorldRay(cam, aspect, fb_w, fb_h, mx, my, ro, rd)) {
        return drag_start_pos_;
    }
    const glm::vec3 cur = ClosestPointOnLineToRay(drag_start_pos_,
                                                  drag_axis_dir_, ro, rd);
    const glm::vec3 delta_world = cur - drag_anchor_world_;
    // Project the world delta onto the axis so the result is strictly
    // axis-aligned (closest-point-on-line is on the axis line by
    // construction, but be defensive against floating-point drift).
    const float on_axis = glm::dot(delta_world, drag_axis_dir_);
    return drag_start_pos_ + drag_axis_dir_ * on_axis;
}

float EditorOverlay::UpdateRotateDrag(const Camera& cam, float aspect,
                                      int fb_w, int fb_h,
                                      double mx, double my) const
{
    if (drag_axis_ == Axis::None) return 0.0f;
    glm::vec3 ro{0.0f}, rd{0.0f};
    if (!ScreenToWorldRay(cam, aspect, fb_w, fb_h, mx, my, ro, rd)) {
        return 0.0f;
    }
    // Intersect the mouse ray with the ring's plane (origin =
    // drag_start_pos_, normal = drag_axis_dir_). Project the resulting
    // tangent vector onto the ring plane (strip residual axis
    // component) and measure the signed angle from drag_anchor_world_
    // (the unit tangent stored at BeginDrag time).
    glm::vec3 cur_world{};
    if (!RayPlaneIntersect(ro, rd, drag_start_pos_, drag_axis_dir_, cur_world)) {
        return 0.0f;
    }
    glm::vec3 t = cur_world - drag_start_pos_;
    t = t - drag_axis_dir_ * glm::dot(t, drag_axis_dir_);
    const float len = glm::length(t);
    if (len < 1e-6f) return 0.0f;
    const glm::vec3 cur_tangent = t / len;
    // Signed angle around drag_axis_dir_ from the anchor tangent.
    // sign = sign(cross(anchor, cur) . axis). cosine = dot, sine =
    // length(cross). atan2 of (sin, cos) gives the unambiguous
    // (-pi, pi] angle.
    const glm::vec3 c = glm::cross(drag_anchor_world_, cur_tangent);
    const float sine_signed = glm::dot(c, drag_axis_dir_);
    const float cos_v       = glm::dot(drag_anchor_world_, cur_tangent);
    return std::atan2(sine_signed, cos_v);
}

void EditorOverlay::EndDrag() {
    drag_axis_ = Axis::None;
}

}  // namespace pt::renderer

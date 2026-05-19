// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Camera.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace pt::engine {
class Engine;
}

namespace pt::renderer {

// 3D transform-gizmo overlay (issue: editor 3D gizmos).
//
// Builds the line-segment geometry for a Blender-style XYZ axis gizmo
// at a world-space origin (the centre of the selected analytic
// primitive) and provides hit-testing + drag math for click-and-drag
// manipulation. Engine.cpp owns one instance, calls UpdateGeometry()
// every frame after the selection state is known, then forwards the
// segment list to the GPU via the editor_overlay compute pipeline.
//
// The gizmo class is render-API-agnostic; it produces a flat float
// buffer (the GPU upload format defined by shaders/EditorOverlay.slang:
// three float4s per segment) so the engine can WriteBuffer + dispatch
// without coupling the gizmo logic to any specific RHI.
class EditorOverlay {
public:
    enum class Mode : std::uint8_t { Translate = 0, Rotate = 1, Scale = 2 };

    // Per-segment GPU upload record. Layout matches
    // shaders/EditorOverlay.slang verbatim (three float4s, 48 bytes).
    struct Segment {
        glm::vec3 a;
        float     half_thickness = 1.5f;
        glm::vec3 b;
        float     depth_bias     = 0.0f;     // reserved (v1 not z-tested)
        glm::vec3 color          {1, 1, 0};  // linear RGB (kernel applies sRGB)
        float     _pad           = 0.0f;
    };
    static_assert(sizeof(Segment) == 48, "Segment must match Slang layout");

    // Per-axis IDs returned by HitTest / passed into BeginDrag. Z is
    // the path-tracer's world-Z which is "into the screen" in the
    // default camera orientation, but the gizmo arrows are world-axis-
    // aligned regardless of camera, so the labels stay X/Y/Z.
    enum class Axis : std::uint8_t { None = 0, X = 1, Y = 2, Z = 3 };

    EditorOverlay();

    // Reset the segment list. Called at the start of every frame
    // before geometry is built (or skipped if nothing is selected).
    void ClearSegments();

    // Append the gizmo for `origin` in the current `mode_`. `size`
    // is the world-units length of each axis arm; the engine derives
    // this from the selected primitive's radius_or_d so handles scale
    // with the prim. Cylinder shaft / cone tip / cube tip / ring
    // tessellation is internal -- callers see a flat segment list.
    //
    // `hovered_axis` highlights the axis the mouse is currently over
    // (yellow tint) so the user has visual feedback before clicking.
    // Pass Axis::None when no hover.
    void AppendGizmo(const glm::vec3& origin, float size, Axis hovered_axis);

    // GPU upload accessors.
    const std::vector<Segment>& Segments() const noexcept { return segs_; }
    std::uint32_t                SegmentCount() const noexcept {
        return static_cast<std::uint32_t>(segs_.size());
    }
    std::size_t                  SegmentBytes() const noexcept {
        return segs_.size() * sizeof(Segment);
    }

    // Mode bookkeeping (driven from the gizmo_mode cvar by Engine).
    void SetMode(Mode m) { mode_ = m; }
    Mode GetMode() const noexcept { return mode_; }

    // Hit-test: given a screen-space mouse position (pixels, top-left
    // origin) and the camera + framebuffer dimensions, return the
    // axis whose projected arm is closest to the mouse within
    // `hit_radius_px`. Returns Axis::None if no axis is within
    // tolerance. The origin + size match the same values passed to
    // AppendGizmo.
    Axis HitTest(const glm::vec3& origin, float size,
                 const Camera& cam, float aspect,
                 int fb_w, int fb_h,
                 double mouse_x, double mouse_y,
                 float hit_radius_px = 10.0f) const;

    // Drag math. Engine state machine calls these as the mouse moves.
    //
    // BeginDrag captures the world-space gizmo origin AND a reference
    // world point on the axis line closest to the current mouse ray.
    // Subsequent UpdateDrag calls compute the world-space delta along
    // the axis as the mouse-ray-axis closest-point moves.
    void  BeginDrag(Axis a, const glm::vec3& origin,
                    const Camera& cam, float aspect,
                    int fb_w, int fb_h,
                    double mouse_x, double mouse_y);
    bool  IsDragging() const noexcept { return drag_axis_ != Axis::None; }
    Axis  DragAxis() const noexcept { return drag_axis_; }
    // Returns the new world-space origin given the current mouse
    // position. The caller compares against the stored drag_start_pos_
    // to compute the delta.
    glm::vec3 UpdateDrag(const Camera& cam, float aspect,
                         int fb_w, int fb_h,
                         double mouse_x, double mouse_y) const;
    void  EndDrag();
    // Cancel the in-flight drag without applying any further updates.
    // Engine uses this on Esc to revert to drag_start_pos_.
    glm::vec3 DragStartPos() const noexcept { return drag_start_pos_; }

private:
    // Build the per-mode geometry.
    void AppendTranslateGizmo(const glm::vec3& origin, float size, Axis hovered);
    void AppendRotateGizmo  (const glm::vec3& origin, float size, Axis hovered);
    void AppendScaleGizmo   (const glm::vec3& origin, float size, Axis hovered);

    // Append a single segment with auto-coloured highlight if hovered.
    void EmitAxisSegment(const glm::vec3& a, const glm::vec3& b,
                         Axis axis, Axis hovered_or_dragged,
                         float thickness = 1.5f);
    // Per-axis colours: X = red, Y = green, Z = blue. Yellow when
    // hovered / dragged (Blender convention).
    glm::vec3 ColorFor(Axis a, Axis hovered_or_dragged) const;

    std::vector<Segment> segs_;
    Mode  mode_ = Mode::Translate;

    // Drag state.
    Axis      drag_axis_      = Axis::None;
    glm::vec3 drag_start_pos_ { 0.0f };
    // World-space closest point along the axis line at drag start.
    // Subtracting subsequent closest-points gives the delta to apply.
    glm::vec3 drag_anchor_world_ { 0.0f };
    // World-space axis direction (unit vector). Stored at BeginDrag
    // so UpdateDrag doesn't need to re-derive it.
    glm::vec3 drag_axis_dir_  { 1.0f, 0.0f, 0.0f };
};

// Compute the closest point on a 3D line (origin + t * dir) to a
// 3D ray (ray_origin + s * ray_dir). Returns the world-space point
// on the LINE; the caller uses this as the dragged-axis intersection.
// Exposed for unit tests and the hit-test path.
glm::vec3 ClosestPointOnLineToRay(const glm::vec3& line_origin,
                                  const glm::vec3& line_dir,
                                  const glm::vec3& ray_origin,
                                  const glm::vec3& ray_dir);

// Project a world-space point to screen-space pixels. Returns false
// if the point is behind the camera; otherwise fills `out_px` with
// the (x, y) pixel coordinates (top-left origin, matching the
// shader and the GLFW cursor convention).
bool ProjectToScreen(const glm::vec3& world_p,
                     const Camera& cam, float aspect,
                     int fb_w, int fb_h,
                     glm::vec2& out_px);

// Compute the world-space ray emitted from screen-space pixel
// (mouse_x, mouse_y) through the camera. Returns false if the
// camera has degenerate dimensions; otherwise sets `out_origin`
// (always == cam.pos) and `out_dir` (unit vector).
bool ScreenToWorldRay(const Camera& cam, float aspect,
                      int fb_w, int fb_h,
                      double mouse_x, double mouse_y,
                      glm::vec3& out_origin,
                      glm::vec3& out_dir);

}  // namespace pt::renderer

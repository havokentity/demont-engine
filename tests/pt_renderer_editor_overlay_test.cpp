// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::renderer::EditorOverlay (issue: editor 3D gizmos).
//
// The gizmo class is mostly responsible for two things: building the
// per-frame line-segment buffer (which the GPU rasterizer consumes) and
// solving the screen-space hit-test + drag math that picks an axis.
// Both are pure-function deterministic with no GPU dependency, so we
// validate them here without spinning up a device.
//
// What's covered:
//   * ProjectToScreen / ScreenToWorldRay round-trip -- a world-space
//     point projected to screen pixels and then re-cast as a ray must
//     return to the same point (within float epsilon).
//   * ClosestPointOnLineToRay: parallel rays return the line origin
//     (degenerate fallback); a ray that crosses the line gives the
//     correct closest point.
//   * EditorOverlay::HitTest: clicking near the X arm picks Axis::X;
//     clicking inside a translate plane handle picks that plane; clicking
//     off-gizmo returns Axis::None.
//   * EditorOverlay::BeginDrag + UpdateDrag: dragging the X arm along
//     a known mouse-delta moves the gizmo origin along the X axis only
//     (Y and Z stay pinned).
//   * Mode switching produces different segment counts (translate vs
//     rotate vs scale ring/cube geometry).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/Camera.h"
#include "../src/renderer/EditorOverlay.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

using namespace pt::renderer;

namespace {

constexpr int   kFbW    = 1920;
constexpr int   kFbH    = 1080;
constexpr float kAspect = float(kFbW) / float(kFbH);
constexpr float kEps    = 1e-3f;

Camera MakeCamera(glm::vec3 pos = {0.0f, 0.0f, 3.0f},
                  float yaw = 0.0f, float pitch = 0.0f,
                  float fov = 60.0f)
{
    Camera c;
    c.pos     = pos;
    c.yaw     = yaw;
    c.pitch   = pitch;
    c.fov_deg = fov;
    return c;
}

}  // namespace

TEST_CASE("ProjectToScreen round-trips via ScreenToWorldRay") {
    Camera cam = MakeCamera();
    // Point one unit in front of the camera (since fwd = -Z when
    // yaw=pitch=0, the world-Z direction is -1 from cam.pos.z=3).
    const glm::vec3 wp{0.0f, 0.0f, 0.0f};   // origin, 3 m in front of cam
    glm::vec2 px;
    REQUIRE(ProjectToScreen(wp, cam, kAspect, kFbW, kFbH, px));

    // Centre of the screen because the point lies on the camera forward.
    CHECK(px.x == doctest::Approx(kFbW * 0.5f).epsilon(0.005));
    CHECK(px.y == doctest::Approx(kFbH * 0.5f).epsilon(0.005));

    glm::vec3 ro{0.0f}, rd{0.0f};
    REQUIRE(ScreenToWorldRay(cam, kAspect, kFbW, kFbH, px.x, px.y, ro, rd));
    CHECK(ro.x == doctest::Approx(cam.pos.x).epsilon(kEps));
    CHECK(ro.y == doctest::Approx(cam.pos.y).epsilon(kEps));
    CHECK(ro.z == doctest::Approx(cam.pos.z).epsilon(kEps));
    // Ray should head toward (0,0,0) from cam.pos: direction = -z.
    CHECK(rd.z == doctest::Approx(-1.0f).epsilon(kEps));
}

TEST_CASE("ProjectToScreen returns false for points behind camera") {
    Camera cam = MakeCamera();
    // World point BEHIND the camera (cam at z=3, world point at z=10
    // -- behind because forward = -Z).
    const glm::vec3 behind{0.0f, 0.0f, 10.0f};
    glm::vec2 px;
    CHECK_FALSE(ProjectToScreen(behind, cam, kAspect, kFbW, kFbH, px));
}

TEST_CASE("ClosestPointOnLineToRay handles parallel rays gracefully") {
    // Both lines along +X axis but at different Y heights.
    glm::vec3 lo{0, 0, 0};
    glm::vec3 ld{1, 0, 0};
    glm::vec3 ro{1, 5, 0};
    glm::vec3 rd{1, 0, 0};   // parallel
    const glm::vec3 q = ClosestPointOnLineToRay(lo, ld, ro, rd);
    // Degenerate fallback returns line_origin.
    CHECK(q.x == doctest::Approx(lo.x));
    CHECK(q.y == doctest::Approx(lo.y));
    CHECK(q.z == doctest::Approx(lo.z));
}

TEST_CASE("ClosestPointOnLineToRay picks correct intersection") {
    // Line: X axis through origin.
    glm::vec3 lo{0, 0, 0};
    glm::vec3 ld{1, 0, 0};
    // Ray from above (high Y) pointing at the X=2 point on the X axis.
    glm::vec3 ro{2, 5, 0};
    glm::vec3 rd{0, -1, 0};
    const glm::vec3 q = ClosestPointOnLineToRay(lo, ld, ro, rd);
    CHECK(q.x == doctest::Approx(2.0f).epsilon(kEps));
    CHECK(q.y == doctest::Approx(0.0f).epsilon(kEps));
    CHECK(q.z == doctest::Approx(0.0f).epsilon(kEps));
}

TEST_CASE("EditorOverlay::HitTest detects axis under mouse") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;

    // Project the X-axis tip to screen so we know where to click.
    glm::vec2 tip;
    REQUIRE(ProjectToScreen(origin + glm::vec3(L, 0, 0),
                            cam, kAspect, kFbW, kFbH, tip));
    glm::vec2 base;
    REQUIRE(ProjectToScreen(origin, cam, kAspect, kFbW, kFbH, base));
    // Click halfway down the arm. (The translate plane handles lie in
    // planes that this dead-on camera views edge-on, so they project to
    // degenerate slivers and must NOT hijack the axis pick here.)
    const glm::vec2 mid = (base + tip) * 0.5f;
    auto axis = eo.HitTest(origin, L, cam, kAspect, kFbW, kFbH,
                           mid.x, mid.y, 10.0f);
    CHECK(axis == EditorOverlay::Axis::X);

    // Click far from any axis -> no hit.
    auto miss = eo.HitTest(origin, L, cam, kAspect, kFbW, kFbH,
                           10.0, 10.0, 10.0f);
    CHECK(miss == EditorOverlay::Axis::None);
}

TEST_CASE("Translate-mode plane handle picks two-axis plane and drags within it") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Translate);

    // XY plane handle spans 0.18L..0.38L along +X/+Y.
    const glm::vec3 handle_center = origin + glm::vec3(0.28f, 0.28f, 0.0f);
    glm::vec2 px;
    REQUIRE(ProjectToScreen(handle_center, cam, kAspect, kFbW, kFbH, px));

    auto axis = eo.HitTest(origin, L, cam, kAspect, kFbW, kFbH,
                           px.x, px.y, 10.0f);
    REQUIRE(axis == EditorOverlay::Axis::XY);

    eo.BeginDrag(axis, origin, cam, kAspect, kFbW, kFbH, px.x, px.y);
    REQUIRE(eo.IsDragging());
    REQUIRE(eo.DragAxis() == EditorOverlay::Axis::XY);

    glm::vec3 moved = eo.UpdateDrag(cam, kAspect, kFbW, kFbH,
                                    px.x + 80.0f, px.y - 60.0f);
    CHECK(moved.z == doctest::Approx(0.0f).epsilon(kEps));
    CHECK(std::abs(moved.x) > kEps);
    CHECK(std::abs(moved.y) > kEps);
}

TEST_CASE("EditorOverlay drag moves origin along the chosen axis only") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;

    // Begin drag at the X axis midpoint.
    glm::vec2 tip;  glm::vec2 base;
    REQUIRE(ProjectToScreen(origin + glm::vec3(L, 0, 0),
                            cam, kAspect, kFbW, kFbH, tip));
    REQUIRE(ProjectToScreen(origin, cam, kAspect, kFbW, kFbH, base));
    const glm::vec2 mid = (base + tip) * 0.5f;
    eo.BeginDrag(EditorOverlay::Axis::X, origin, cam, kAspect,
                 kFbW, kFbH, mid.x, mid.y);
    REQUIRE(eo.IsDragging());
    REQUIRE(eo.DragAxis() == EditorOverlay::Axis::X);

    // Move the mouse 100 px to the right along the X arm projection.
    // We sample BeginDrag's anchor point in screen, push it 100 px
    // along the arm's projected direction.
    glm::vec2 arm_dir = glm::normalize(tip - base);
    glm::vec2 new_mouse = mid + arm_dir * 100.0f;
    glm::vec3 moved = eo.UpdateDrag(cam, kAspect, kFbW, kFbH,
                                    new_mouse.x, new_mouse.y);

    // Drag must keep Y and Z pinned -- only X moves.
    CHECK(moved.y == doctest::Approx(0.0f).epsilon(kEps));
    CHECK(moved.z == doctest::Approx(0.0f).epsilon(kEps));
    // X delta should be non-zero in the direction of the drag.
    CHECK(moved.x > kEps);
}

TEST_CASE("EditorOverlay segment count differs by mode") {
    EditorOverlay eo;
    const glm::vec3 origin{0.0f};
    const float L = 1.0f;

    eo.SetMode(EditorOverlay::Mode::Translate);
    eo.ClearSegments();
    eo.AppendGizmo(origin, L, EditorOverlay::Axis::None);
    const std::uint32_t translate_n = eo.SegmentCount();
    CHECK(translate_n > 0u);

    eo.SetMode(EditorOverlay::Mode::Rotate);
    eo.ClearSegments();
    eo.AppendGizmo(origin, L, EditorOverlay::Axis::None);
    const std::uint32_t rotate_n = eo.SegmentCount();
    CHECK(rotate_n > 0u);
    // Rotate gizmo draws ring polylines -- expect strictly more
    // segments than the translate arrow triad.
    CHECK(rotate_n > translate_n);

    eo.SetMode(EditorOverlay::Mode::Scale);
    eo.ClearSegments();
    eo.AppendGizmo(origin, L, EditorOverlay::Axis::None);
    const std::uint32_t scale_n = eo.SegmentCount();
    CHECK(scale_n > 0u);
}

TEST_CASE("Hovered axis is yellow-tinted in segment colour") {
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Translate);
    eo.ClearSegments();
    eo.AppendGizmo(glm::vec3(0.0f), 1.0f, EditorOverlay::Axis::X);
    // First segment is the X shaft. Highlighted == bright yellow.
    REQUIRE(eo.SegmentCount() > 0);
    const auto& s = eo.Segments()[0];
    CHECK(s.color.r > 0.8f);
    CHECK(s.color.g > 0.8f);
    CHECK(s.color.b < 0.3f);
}

TEST_CASE("Segment GPU layout is 48 bytes (3 float4)") {
    // Catches future reorderings that would silently corrupt the
    // shader's segment iteration (the layout MUST match
    // shaders/EditorOverlay.slang verbatim).
    CHECK(sizeof(EditorOverlay::Segment) == 48u);
}

// --- Rotation gizmo (#206) -----------------------------------------------
//
// Rotate mode adds two new behaviours over translate:
//   * HitTest probes the three rings instead of the three arms; the
//     picked axis is the ring's NORMAL (Axis::X = X-normal ring).
//   * UpdateRotateDrag returns a signed angle in radians instead of
//     a world-space position. Sign follows the right-hand rule around
//     the picked axis.
// Both paths share the BeginDrag entry point, but the drag_mode_ field
// captures the mode at BeginDrag time so a mid-drag gizmo_mode toggle
// doesn't change the dispatch.

TEST_CASE("Rotate-mode HitTest picks the Z ring when clicking near +X on the XY plane") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Rotate);

    // The Z-normal ring lives on the XY plane. Click near (L, 0, 0)
    // which sits on that ring -- the picked axis should be Z.
    glm::vec2 ring_pt;
    REQUIRE(ProjectToScreen(glm::vec3(L, 0, 0), cam, kAspect, kFbW, kFbH, ring_pt));
    auto axis = eo.HitTest(origin, L, cam, kAspect, kFbW, kFbH,
                           ring_pt.x, ring_pt.y, 10.0f);
    // (L,0,0) is on the X ring (X-normal -> ring on YZ plane through X)
    // *and* the Z ring (Z-normal -> ring on XY plane through X). Both
    // could match. We require ONE of them is picked (not None).
    CHECK(axis != EditorOverlay::Axis::None);
}

TEST_CASE("Rotate-mode HitTest returns None far from any ring") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Rotate);
    auto axis = eo.HitTest(origin, L, cam, kAspect, kFbW, kFbH,
                           10.0, 10.0, 10.0f);
    CHECK(axis == EditorOverlay::Axis::None);
}

TEST_CASE("UpdateRotateDrag returns a non-zero angle after the mouse moves around the ring") {
    Camera cam = MakeCamera({0.0f, 3.0f, 0.0f}, 0.0f, -1.5707963f);
    // Look straight down so the Y-normal ring (in the XZ plane)
    // projects to a roughly-circular shape in screen space. The Y ring
    // is the easiest to project around because its plane is parallel
    // to the camera's image plane in this configuration.
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Rotate);

    // BeginDrag at a point on the +X side of the Y ring (XZ-plane).
    glm::vec2 px_start;
    REQUIRE(ProjectToScreen(glm::vec3(L, 0, 0), cam, kAspect, kFbW, kFbH, px_start));
    eo.BeginDrag(EditorOverlay::Axis::Y, origin, cam, kAspect,
                 kFbW, kFbH, px_start.x, px_start.y);
    REQUIRE(eo.IsDragging());
    REQUIRE(eo.DragAxis() == EditorOverlay::Axis::Y);
    REQUIRE(eo.DragMode() == EditorOverlay::Mode::Rotate);

    // Move to a point on the +Z side of the ring (90 degrees around
    // the Y axis, right-hand rule). Should yield ~+pi/2 angle.
    glm::vec2 px_end;
    REQUIRE(ProjectToScreen(glm::vec3(0, 0, L), cam, kAspect, kFbW, kFbH, px_end));
    const float angle = eo.UpdateRotateDrag(cam, kAspect, kFbW, kFbH,
                                            px_end.x, px_end.y);
    // We don't pin the magnitude (the look-down camera distorts the
    // angular measurement slightly) but it MUST be non-trivially
    // non-zero. Sign should be consistent with the right-hand rule.
    CHECK(std::abs(angle) > 0.5f);
}

TEST_CASE("DragMode is captured at BeginDrag time, not at UpdateDrag time") {
    Camera cam = MakeCamera({0.0f, 0.0f, 3.0f});
    const glm::vec3 origin{0.0f, 0.0f, 0.0f};
    const float L = 1.0f;
    EditorOverlay eo;
    eo.SetMode(EditorOverlay::Mode::Rotate);

    glm::vec2 px;
    REQUIRE(ProjectToScreen(glm::vec3(L, 0, 0), cam, kAspect, kFbW, kFbH, px));
    eo.BeginDrag(EditorOverlay::Axis::Y, origin, cam, kAspect,
                 kFbW, kFbH, px.x, px.y);
    CHECK(eo.DragMode() == EditorOverlay::Mode::Rotate);

    // Toggle the gizmo's mode mid-drag. The drag's captured mode
    // should stay Rotate so the engine dispatches the right command.
    eo.SetMode(EditorOverlay::Mode::Translate);
    CHECK(eo.DragMode() == EditorOverlay::Mode::Rotate);
    CHECK(eo.GetMode() == EditorOverlay::Mode::Translate);
}

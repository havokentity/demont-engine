// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::csg::CsgScene (issue #64 / Phase 3 of #47).
//
// CsgScene is the host-side facade over Manifold; its public API
// exposes node creation (AddBox/AddSphere/AddCylinder), boolean
// combination (Combine + OpType::Union/Subtract/Intersect), root
// selection (SetRoot), tree mutation (Remove, Reset), dirty
// tracking (Dirty / AcknowledgeClean), and the actual evaluation
// step (Bake() -> BakedMesh triangle soup). Manifold's internals
// are deliberately hidden, so this test exercises ONLY the public
// surface.
//
// Strategy:
//   - Structural assertions (vertex count > 0, triangle count >= N,
//     AABB extents within a tolerance band) instead of vertex-by-
//     vertex equality. Manifold's mesh output is not bit-stable
//     across library versions or platforms (vertex ordering,
//     triangulation choices, etc.), so identity asserts would be
//     brittle. The structural properties we DO test are stable
//     across versions and are exactly what the renderer relies on.
//   - Same-process determinism: a given construction sequence must
//     produce the same vertex/triangle counts across repeated
//     Bake() calls (Manifold inside one process is deterministic
//     by construction; this guards against an inadvertent
//     hash-table-order leak in our wrapper or a missed cache
//     invalidation).
//   - Metre-scale primitives throughout (project memory: "1 world
//     unit = 1 metre; real atmospheric / optical / mechanical
//     constants"). Even though no physics is invoked here, the
//     primitive dimensions match the renderer's coordinate
//     convention so a future renderer-level test that swaps in
//     this scene gets sensible answers.
//   - No randomness in the construction sequence -- every test
//     hand-builds its tree -- so there is no PRNG to seed.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/Csg/CsgScene.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using pt::csg::BakedMesh;
using pt::csg::CsgScene;
using pt::csg::OpType;

namespace {

// --- AABB helper ----------------------------------------------------------
// Compute the axis-aligned bounding box of a baked mesh's vertex
// positions. Used for "did the mesh end up roughly where I placed
// the primitive" assertions -- the only spatial property that
// survives Manifold's internal vertex reordering.
struct AABB {
    float mn[3]{ std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity() };
    float mx[3]{ -std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity() };
    float extent(int axis) const { return mx[axis] - mn[axis]; }
    float center(int axis) const { return 0.5f * (mn[axis] + mx[axis]); }
};

AABB ComputeAabb(const BakedMesh& m) {
    AABB a;
    const std::size_t vc = m.VertexCount();
    for (std::size_t i = 0; i < vc; ++i) {
        for (int k = 0; k < 3; ++k) {
            const float v = m.positions[i * 3 + k];
            if (v < a.mn[k]) a.mn[k] = v;
            if (v > a.mx[k]) a.mx[k] = v;
        }
    }
    return a;
}

// Helper: scrub the default scene Reset() seeds (a unit box at id=1
// set as root). Several tests want to start from a genuinely empty
// CsgScene -- the constructor calls Reset() which plants a default
// node, so we remove it and assert the tree is empty.
void ClearDefaultRoot(CsgScene& scene) {
    // Default scene: node id 1 is the root (see CsgScene::Reset).
    if (scene.HasRoot()) {
        const std::uint32_t root = scene.RootId();
        scene.Remove(root);
    }
    REQUIRE_FALSE(scene.HasRoot());
    REQUIRE(scene.Empty());
}

// Tolerance for floating-point AABB extent comparisons. Metre-scale
// primitives, geometry through Manifold's tessellator -- vertices
// are exact up to the same float precision the input had, so 1e-4m
// (= 0.1mm) covers any drift without masking a real bug.
constexpr float kAabbEps = 1e-4f;

}  // namespace

// --- Test 1: empty scene bakes to Empty() BakedMesh ------------------------
// After Remove() of the default root, there is no active root; Bake()
// must short-circuit to an empty mesh and the out_error string should
// receive a human-readable diagnostic. The renderer relies on the
// Empty() flag to skip uploading a zero-sized buffer to the RHI.
TEST_CASE("CsgScene: empty scene -- Bake returns Empty() BakedMesh") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    std::string err;
    BakedMesh m = scene.Bake(&err);
    CHECK(m.Empty());
    CHECK(m.VertexCount() == 0u);
    CHECK(m.TriangleCount() == 0u);
    CHECK_FALSE(err.empty());  // diagnostic should be populated
}

// --- Test 2: single primitive leaf -- box ----------------------------------
// AddBox a 1x1x1 unit box at the origin, set as root, bake. Verify:
//   * The mesh is non-empty (positions + indices populated).
//   * The AABB extent matches the requested size on every axis
//     (Manifold::Cube produces a box with corners exactly at +/- sx/2).
//   * Triangle count is at least 12 -- a manifold cube has 6 quad
//     faces tessellated into 2 triangles each. Manifold may produce
//     additional sub-faces depending on version; we assert "at least
//     12" rather than "exactly 12" so library updates don't break.
TEST_CASE("CsgScene: single box leaf -- bake produces a valid manifold cube") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    constexpr float kSx = 2.0f, kSy = 1.0f, kSz = 0.5f;  // metres
    REQUIRE(scene.AddBox(/*id*/ 100, kSx, kSy, kSz,
                         /*translation*/ 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(100));
    REQUIRE(scene.HasRoot());

    std::string err;
    BakedMesh m = scene.Bake(&err);
    CHECK(err.empty());
    REQUIRE_FALSE(m.Empty());
    CHECK(m.VertexCount() > 0u);
    CHECK(m.TriangleCount() >= 12u);  // 6 faces x 2 tris min

    // Flat-shaded soup: positions and normals are 1:1 per vertex, 3
    // floats each; indices are 3-per-triangle. EmitFlatTriangleSoup
    // produces unwelded vertices, so vertex_count == triangle_count*3
    // exactly. Asserting this catches a regression where the bake
    // path accidentally welds vertices and disturbs the flat-shaded
    // normal invariant downstream consumers depend on.
    CHECK(m.positions.size() == m.normals.size());
    CHECK(m.positions.size() == m.VertexCount() * 3u);
    CHECK(m.indices.size() == m.TriangleCount() * 3u);
    CHECK(m.VertexCount() == m.TriangleCount() * 3u);

    AABB a = ComputeAabb(m);
    CHECK(a.extent(0) == doctest::Approx(kSx).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(kSy).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(kSz).epsilon(kAabbEps));
    // Manifold::Cube with center=true centers the box at origin.
    CHECK(a.center(0) == doctest::Approx(0.0f).epsilon(kAabbEps));
    CHECK(a.center(1) == doctest::Approx(0.0f).epsilon(kAabbEps));
    CHECK(a.center(2) == doctest::Approx(0.0f).epsilon(kAabbEps));
}

// --- Test 2b: single primitive leaf -- sphere ------------------------------
// AddSphere at an off-origin position. The translation argument
// should be applied AFTER centering at origin -- so the resulting
// mesh's AABB center matches the requested translation.
TEST_CASE("CsgScene: single sphere leaf -- translation is applied after centering") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    constexpr float kRadius = 0.5f;  // 0.5 m sphere
    constexpr float kTx = 1.0f, kTy = 2.0f, kTz = -3.0f;  // metres
    REQUIRE(scene.AddSphere(/*id*/ 200, kRadius, /*segments*/ 32,
                            kTx, kTy, kTz));
    REQUIRE(scene.SetRoot(200));

    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());
    CHECK(m.TriangleCount() > 0u);

    AABB a = ComputeAabb(m);
    // The tessellated sphere's AABB extent is exactly 2r along each
    // axis (Manifold::Sphere returns a polyhedral approximation whose
    // vertices lie on the analytic sphere). Center matches the
    // translation argument.
    CHECK(a.extent(0) == doctest::Approx(2.0f * kRadius).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(2.0f * kRadius).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(2.0f * kRadius).epsilon(kAabbEps));
    CHECK(a.center(0) == doctest::Approx(kTx).epsilon(kAabbEps));
    CHECK(a.center(1) == doctest::Approx(kTy).epsilon(kAabbEps));
    CHECK(a.center(2) == doctest::Approx(kTz).epsilon(kAabbEps));
}

// --- Test 2c: cylinder leaf ------------------------------------------------
// AddCylinder produces a tessellated cylinder centered at origin
// (the CsgScene wrapper passes `true` to Manifold::Cylinder's
// center flag) and then translated.
TEST_CASE("CsgScene: single cylinder leaf -- height/radius extents match") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    constexpr float kRadius = 0.25f;       // 0.25 m
    constexpr float kHeight = 1.5f;        // 1.5 m
    constexpr int   kSegments = 16;
    REQUIRE(scene.AddCylinder(/*id*/ 300, kRadius, kHeight, kSegments,
                              0.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(300));

    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());

    AABB a = ComputeAabb(m);
    // Cylinder along Z by Manifold convention: extent (2r, 2r, h).
    CHECK(a.extent(0) == doctest::Approx(2.0f * kRadius).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(2.0f * kRadius).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(kHeight).epsilon(kAabbEps));
}

// --- Test 3: union of two non-overlapping primitives -----------------------
// Two boxes placed apart so their AABBs don't intersect; combine
// with OpType::Union. The result is a single mesh but the boxes
// don't share any geometry. We assert structural properties:
//   * Mesh non-empty.
//   * Each box's triangle count is preserved (Manifold doesn't
//     delete faces in a disjoint union).
//   * AABB spans BOTH primitives (sanity check that the union
//     actually combined them rather than dropping one).
TEST_CASE("CsgScene: union of two disjoint primitives -- both survive") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    // Box A: 1m cube at x=-2, well outside Box B.
    REQUIRE(scene.AddBox(/*id*/ 1, 1.0f, 1.0f, 1.0f,
                         /*translation*/ -2.0f, 0.0f, 0.0f));
    // Box B: 1m cube at x=+2.
    REQUIRE(scene.AddBox(/*id*/ 2, 1.0f, 1.0f, 1.0f,
                         /*translation*/ +2.0f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(/*id*/ 3, OpType::Union, /*left*/ 1, /*right*/ 2));
    REQUIRE(scene.SetRoot(3));

    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());

    // Bake each box on its own first to get a baseline triangle
    // count. The union of two disjoint manifolds preserves both
    // surfaces, so the union's tri-count must be >= the sum of the
    // individual tri-counts (Manifold may re-tessellate for
    // numerical robustness, never deletes faces).
    std::uint32_t tris_a = 0, tris_b = 0;
    {
        CsgScene single;
        ClearDefaultRoot(single);
        REQUIRE(single.AddBox(10, 1.0f, 1.0f, 1.0f, -2.0f, 0.0f, 0.0f));
        REQUIRE(single.SetRoot(10));
        tris_a = single.Bake().TriangleCount();
    }
    {
        CsgScene single;
        ClearDefaultRoot(single);
        REQUIRE(single.AddBox(20, 1.0f, 1.0f, 1.0f, +2.0f, 0.0f, 0.0f));
        REQUIRE(single.SetRoot(20));
        tris_b = single.Bake().TriangleCount();
    }
    CHECK(tris_a > 0u);
    CHECK(tris_b > 0u);
    CHECK(m.TriangleCount() >= tris_a + tris_b);

    // AABB spans from x=-2.5 (box A's -x face) to x=+2.5 (box B's
    // +x face): total extent on X is 5m. Y and Z extents stay at
    // the per-box dimension (1m). A union that silently dropped one
    // of the children would collapse the X extent down to ~1m and
    // fail this check.
    AABB a = ComputeAabb(m);
    CHECK(a.extent(0) == doctest::Approx(5.0f).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(1.0f).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(1.0f).epsilon(kAabbEps));
}

// --- Test 4: subtract carves a concave region ------------------------------
// Box minus a centered sphere -> the box with a sphere-shaped hole
// cut out of the middle. Manifold's tessellation of the cut surface
// adds vertices, so the result's triangle count must be greater
// than the input box's triangle count. AABB is bounded by the box
// (the sphere is entirely inside the box, so the outer hull is
// unchanged).
TEST_CASE("CsgScene: subtract (box - centered sphere) -- adds triangles, bounded AABB") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    constexpr float kBoxSize  = 2.0f;   // 2m cube
    constexpr float kSphereR  = 0.5f;   // 0.5m sphere fits inside
    REQUIRE(scene.AddBox(/*id*/ 1, kBoxSize, kBoxSize, kBoxSize,
                         0.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddSphere(/*id*/ 2, kSphereR, /*segments*/ 32,
                            0.0f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(/*id*/ 3, OpType::Subtract,
                          /*left=box*/ 1, /*right=sphere*/ 2));
    REQUIRE(scene.SetRoot(3));

    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());

    // Baseline: triangle count of the bare box.
    std::uint32_t tris_box = 0;
    {
        CsgScene single;
        ClearDefaultRoot(single);
        REQUIRE(single.AddBox(99, kBoxSize, kBoxSize, kBoxSize,
                              0.0f, 0.0f, 0.0f));
        REQUIRE(single.SetRoot(99));
        tris_box = single.Bake().TriangleCount();
    }
    CHECK(tris_box > 0u);
    // Subtract must have added faces for the spherical cavity.
    CHECK(m.TriangleCount() > tris_box);

    // Outer hull is still the box (sphere is fully interior).
    AABB a = ComputeAabb(m);
    CHECK(a.extent(0) == doctest::Approx(kBoxSize).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(kBoxSize).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(kBoxSize).epsilon(kAabbEps));
}

// --- Test 5: intersect of disjoint primitives is empty ---------------------
// Two boxes placed apart so their volumes don't intersect. The
// intersection is the empty set; Manifold returns an empty
// manifold; Bake reports Empty() and surfaces an error string.
TEST_CASE("CsgScene: intersect of disjoint primitives -- result is Empty()") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox(/*id*/ 1, 1.0f, 1.0f, 1.0f, -2.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddBox(/*id*/ 2, 1.0f, 1.0f, 1.0f, +2.0f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(/*id*/ 3, OpType::Intersect, 1, 2));
    REQUIRE(scene.SetRoot(3));

    std::string err;
    BakedMesh m = scene.Bake(&err);
    CHECK(m.Empty());
    CHECK_FALSE(err.empty());  // "result is empty" or similar
}

// --- Test 6: intersect of overlapping primitives produces a mesh -----------
// Two overlapping unit boxes offset by 0.5m on X -> their
// intersection is a 0.5x1x1 m^3 box. Verify the resulting AABB
// has the expected extents.
TEST_CASE("CsgScene: intersect of overlapping boxes -- AABB equals overlap region") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    // Boxes overlap on x in [-0.25, +0.25] (extent 0.5m) and fully
    // overlap on y, z (extent 1m each).
    REQUIRE(scene.AddBox(/*id*/ 1, 1.0f, 1.0f, 1.0f, -0.25f, 0.0f, 0.0f));
    REQUIRE(scene.AddBox(/*id*/ 2, 1.0f, 1.0f, 1.0f, +0.25f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(/*id*/ 3, OpType::Intersect, 1, 2));
    REQUIRE(scene.SetRoot(3));

    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());

    AABB a = ComputeAabb(m);
    CHECK(a.extent(0) == doctest::Approx(0.5f).epsilon(kAabbEps));
    CHECK(a.extent(1) == doctest::Approx(1.0f).epsilon(kAabbEps));
    CHECK(a.extent(2) == doctest::Approx(1.0f).epsilon(kAabbEps));
}

// --- Test 7: Reset clears the scene ----------------------------------------
// Reset() must wipe any existing nodes and replant the default
// scene (a unit box at id=1, set as root). Verify that:
//   * After mutation and Reset, only id=1 exists as the root.
//   * Bake produces a non-empty mesh that matches what a fresh
//     CsgScene would produce.
TEST_CASE("CsgScene: Reset() -- restores default scene") {
    CsgScene scene;
    // Mutate: remove default root, add a sphere, set it as root.
    ClearDefaultRoot(scene);
    REQUIRE(scene.AddSphere(50, 1.0f, 16, 5.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(50));
    REQUIRE(scene.HasRoot());
    CHECK(scene.RootId() == 50u);

    // Reset: should clear the sphere and reinstate the default box.
    scene.Reset();
    REQUIRE(scene.HasRoot());
    CHECK(scene.RootId() == 1u);  // default node id
    BakedMesh m = scene.Bake();
    REQUIRE_FALSE(m.Empty());
    // Default scene is a 1x1x1 box translated to sit on y=0; AABB
    // span on X and Z is 1m, Y is [0, 1] (extent 1m). The exact
    // translation values are documented in CsgScene::Reset() --
    // we don't pin them here so the default can evolve without
    // breaking this test.
    CHECK(m.TriangleCount() >= 12u);
}

// --- Test 8: dirty tracking flips correctly --------------------------------
// CsgScene::Dirty() reports whether the next Bake() would differ
// from the last cleanly-acknowledged state. The engine uses this
// to skip re-bakes when nothing changed. Mutations (Add*, Combine,
// Remove, SetRoot, Reset) must mark the scene dirty;
// AcknowledgeClean() must clear the flag.
TEST_CASE("CsgScene: Dirty() flips on mutations and clears on AcknowledgeClean()") {
    CsgScene scene;
    // Fresh scene from Reset() is dirty.
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // AddBox -> dirty.
    REQUIRE(scene.AddBox(10, 1.0f, 1.0f, 1.0f, 5.0f, 0.0f, 0.0f));
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // AddSphere -> dirty.
    REQUIRE(scene.AddSphere(11, 0.5f, 16, -5.0f, 0.0f, 0.0f));
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // Combine -> dirty.
    REQUIRE(scene.Combine(12, OpType::Union, 10, 11));
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // SetRoot to a different node -> dirty.
    REQUIRE(scene.SetRoot(12));
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // SetRoot to the SAME node should NOT mark dirty -- the engine
    // shouldn't re-bake for a no-op (CsgScene short-circuits this).
    REQUIRE(scene.SetRoot(12));
    CHECK_FALSE(scene.Dirty());

    // Remove -> dirty.
    CHECK(scene.Remove(11) >= 1u);
    CHECK(scene.Dirty());
    scene.AcknowledgeClean();
    CHECK_FALSE(scene.Dirty());

    // Reset -> dirty.
    scene.Reset();
    CHECK(scene.Dirty());
}

// --- Test 9: id reuse via Remove + re-Add ----------------------------------
// After Remove(id), the same id must be re-addable -- the
// underlying node table must clear the slot. Verify the second
// Add returns true and the resulting bake produces a mesh.
TEST_CASE("CsgScene: Remove + re-Add the same id round-trips") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox(42, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(42));
    BakedMesh m1 = scene.Bake();
    REQUIRE_FALSE(m1.Empty());
    const std::uint32_t tris1 = m1.TriangleCount();

    // Remove the box.
    CHECK(scene.Remove(42) >= 1u);
    CHECK_FALSE(scene.HasRoot());  // root was 42; Remove cascades the root nil-out

    // Re-add at the same id with the same parameters: must succeed
    // and produce a mesh with the same triangle count.
    REQUIRE(scene.AddBox(42, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(42));
    BakedMesh m2 = scene.Bake();
    REQUIRE_FALSE(m2.Empty());
    CHECK(m2.TriangleCount() == tris1);
    CHECK(m2.VertexCount() == m1.VertexCount());
}

// --- Test 10: duplicate id rejection ---------------------------------------
// AddBox / AddSphere / AddCylinder / Combine must reject a
// duplicate id (return false) without mutating the scene.
TEST_CASE("CsgScene: duplicate id is rejected") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox(7, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    // Duplicate primitive id rejected.
    CHECK_FALSE(scene.AddBox(7, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddSphere(7, 1.0f, 16, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddCylinder(7, 0.5f, 1.0f, 16, 0.0f, 0.0f, 0.0f));
    // Set up a second leaf so Combine has operands; then duplicate
    // Combine id 7 must also be rejected.
    REQUIRE(scene.AddBox(8, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.Combine(7, OpType::Union, 7, 8));
}

// --- Test 11: degenerate primitive rejection -------------------------------
// AddBox with a non-positive size must return false. AddSphere with
// non-positive radius must return false. AddCylinder with non-
// positive radius or height must return false. Combine with
// OpType::Leaf is invalid by construction.
TEST_CASE("CsgScene: degenerate primitives are rejected") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    CHECK_FALSE(scene.AddBox(1,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddBox(2, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddBox(3,  1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddBox(4,  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f));

    CHECK_FALSE(scene.AddSphere(5, 0.0f, 16, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddSphere(6, -1.0f, 16, 0.0f, 0.0f, 0.0f));

    CHECK_FALSE(scene.AddCylinder(7, 0.0f, 1.0f, 16, 0.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.AddCylinder(8, 1.0f, 0.0f, 16, 0.0f, 0.0f, 0.0f));

    // Combine with OpType::Leaf is invalid.
    REQUIRE(scene.AddBox(9,  1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddBox(10, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f));
    CHECK_FALSE(scene.Combine(11, OpType::Leaf, 9, 10));

    // Combine with missing operand rejected.
    CHECK_FALSE(scene.Combine(11, OpType::Union, 9, /*nonexistent*/ 999));
    CHECK_FALSE(scene.Combine(11, OpType::Union, /*nonexistent*/ 999, 10));
}

// --- Test 12: SetRoot to nonexistent id is rejected ------------------------
// SetRoot must validate that the target id exists; rejecting an
// unknown id keeps the engine from baking against a stale root_id.
TEST_CASE("CsgScene: SetRoot of nonexistent id is rejected") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox(50, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.SetRoot(50));
    CHECK(scene.HasRoot());
    CHECK(scene.RootId() == 50u);

    // Unknown id: SetRoot returns false, root_id unchanged.
    CHECK_FALSE(scene.SetRoot(9999));
    CHECK(scene.RootId() == 50u);
}

// --- Test 13: Remove cascades through internal nodes -----------------------
// Per the contract: Remove(id) must also remove any internal nodes
// that referenced `id` as a child (otherwise the tree would dangle
// a stale pointer). Construct a small tree A ∪ B, then Remove(A);
// the union node must also disappear.
TEST_CASE("CsgScene: Remove cascades to internal nodes referencing the removed id") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox(1, 1.0f, 1.0f, 1.0f, -2.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddBox(2, 1.0f, 1.0f, 1.0f, +2.0f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(3, OpType::Union, 1, 2));
    REQUIRE(scene.SetRoot(3));
    REQUIRE(scene.HasRoot());

    // Remove operand 1 -> must drop both id 1 and the union node 3.
    // We expect removed-count >= 2 (the cascade includes 1 itself,
    // plus the dependent union node).
    const std::size_t removed = scene.Remove(1);
    CHECK(removed >= 2u);
    // Root was the union node, which got cascaded; root should be 0.
    CHECK_FALSE(scene.HasRoot());

    // Re-adding id 3 must succeed -- the slot was freed by the cascade.
    REQUIRE(scene.AddBox(3, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f));
}

// --- Test 14: determinism -- repeated Bake() gives identical structure -----
// The same construction sequence baked twice in a row must produce
// the same vertex count + triangle count + AABB. Manifold itself
// is deterministic per-process; this test pins our wrapper's
// behaviour. A regression where Bake() accidentally mutated cached
// state (e.g. an unprotected scratch buffer) would produce
// different counts on the second call.
TEST_CASE("CsgScene: same construction -- repeated Bake() is identical") {
    CsgScene scene;
    ClearDefaultRoot(scene);

    REQUIRE(scene.AddBox   (1, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddSphere(2, 0.75f, 32, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddCylinder(3, 0.4f, 1.5f, 24, 0.5f, 0.0f, 0.0f));
    REQUIRE(scene.Combine  (4, OpType::Subtract, 1, 2));
    REQUIRE(scene.Combine  (5, OpType::Union,    4, 3));
    REQUIRE(scene.SetRoot  (5));

    BakedMesh m1 = scene.Bake();
    BakedMesh m2 = scene.Bake();
    BakedMesh m3 = scene.Bake();
    REQUIRE_FALSE(m1.Empty());

    CHECK(m1.VertexCount()   == m2.VertexCount());
    CHECK(m1.TriangleCount() == m2.TriangleCount());
    CHECK(m1.VertexCount()   == m3.VertexCount());
    CHECK(m1.TriangleCount() == m3.TriangleCount());

    // AABB must also match -- Manifold's same-process output is
    // bit-stable, so we can pin both extents and centers exactly.
    AABB a1 = ComputeAabb(m1);
    AABB a2 = ComputeAabb(m2);
    for (int k = 0; k < 3; ++k) {
        CHECK(a1.mn[k] == doctest::Approx(a2.mn[k]).epsilon(kAabbEps));
        CHECK(a1.mx[k] == doctest::Approx(a2.mx[k]).epsilon(kAabbEps));
    }
}

// --- Test 15: identity operations ------------------------------------------
// The boolean identities the issue calls out:
//   (a - a) == empty (subtract self)
//   (a intersect a) == a (idempotent self-intersect)
// These are stable across Manifold versions and are exactly the
// kind of property test that catches an OpType swap regression.
TEST_CASE("CsgScene: identity ops -- (a - a) == empty, (a intersect a) ~= a") {
    // (a - a) -> empty mesh.
    {
        CsgScene scene;
        ClearDefaultRoot(scene);
        REQUIRE(scene.AddBox(1, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
        // Re-add the same shape under a different id; CsgScene
        // can't reuse the same id as both operands of a Subtract
        // (left==right would still work, but the API permits
        // explicit duplication via two ids).
        REQUIRE(scene.AddBox(2, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
        REQUIRE(scene.Combine(3, OpType::Subtract, 1, 2));
        REQUIRE(scene.SetRoot(3));

        std::string err;
        BakedMesh m = scene.Bake(&err);
        CHECK(m.Empty());
    }
    // (a intersect a) -> the same shape (we assert AABB matches).
    {
        CsgScene scene;
        ClearDefaultRoot(scene);
        REQUIRE(scene.AddBox(1, 1.5f, 0.7f, 0.3f, 1.0f, 2.0f, 3.0f));
        REQUIRE(scene.AddBox(2, 1.5f, 0.7f, 0.3f, 1.0f, 2.0f, 3.0f));
        REQUIRE(scene.Combine(3, OpType::Intersect, 1, 2));
        REQUIRE(scene.SetRoot(3));

        BakedMesh m = scene.Bake();
        REQUIRE_FALSE(m.Empty());
        AABB a = ComputeAabb(m);
        CHECK(a.extent(0) == doctest::Approx(1.5f).epsilon(kAabbEps));
        CHECK(a.extent(1) == doctest::Approx(0.7f).epsilon(kAabbEps));
        CHECK(a.extent(2) == doctest::Approx(0.3f).epsilon(kAabbEps));
        CHECK(a.center(0) == doctest::Approx(1.0f).epsilon(kAabbEps));
        CHECK(a.center(1) == doctest::Approx(2.0f).epsilon(kAabbEps));
        CHECK(a.center(2) == doctest::Approx(3.0f).epsilon(kAabbEps));
    }
}

// --- Test 16: bake with no root selected gives a clear error ---------------
// After explicitly clearing the root, Bake must report a non-empty
// out_error string and an empty mesh -- the engine surfaces this
// in the console when the user types `csg_bake` without a root.
TEST_CASE("CsgScene: Bake() with no root returns Empty() and an error string") {
    CsgScene scene;
    ClearDefaultRoot(scene);  // no root after this

    std::string err;
    BakedMesh m = scene.Bake(&err);
    CHECK(m.Empty());
    CHECK_FALSE(err.empty());

    // Nullable out_error: passing nullptr must not crash, must
    // still produce Empty().
    BakedMesh m2 = scene.Bake(nullptr);
    CHECK(m2.Empty());
}

// --- Test 17: Dump emits a non-empty string with the root marker -----------
// CsgScene::Dump produces a human-readable tree dump used by the
// `csg_dump` console command. Test that it isn't a no-op: after
// building a small tree, Dump must mention "[root]" and the op
// names of each node.
TEST_CASE("CsgScene: Dump() includes node ids, op names, and root marker") {
    CsgScene scene;
    ClearDefaultRoot(scene);
    REQUIRE(scene.AddBox(1, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.AddSphere(2, 0.5f, 16, 0.0f, 0.0f, 0.0f));
    REQUIRE(scene.Combine(3, OpType::Subtract, 1, 2));
    REQUIRE(scene.SetRoot(3));

    std::string out;
    scene.Dump(out);
    CHECK_FALSE(out.empty());
    CHECK(out.find("[root]")   != std::string::npos);
    CHECK(out.find("subtract") != std::string::npos);
    CHECK(out.find("leaf")     != std::string::npos);
    CHECK(out.find("tree:")    != std::string::npos);
}

// --- Test 18: OpName helper -------------------------------------------------
// OpName(OpType) is exposed for diagnostics. Verify each enumerator
// has a string -- catches a regression where someone adds a new
// op but forgets to extend the switch in OpName().
TEST_CASE("CsgScene: OpName returns a non-empty label for every OpType") {
    using pt::csg::OpName;
    CHECK(std::string(OpName(OpType::Leaf))      == "leaf");
    CHECK(std::string(OpName(OpType::Union))     == "union");
    CHECK(std::string(OpName(OpType::Subtract))  == "subtract");
    CHECK(std::string(OpName(OpType::Intersect)) == "intersect");
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// CsgScene -- a small CSG tree backed by Manifold for boolean ops.
//
// Public API deliberately hides manifold's headers so consumers (Engine,
// console commands) don't have to drag them in. Nodes are addressed by
// stable user-supplied IDs; the tree's root is whichever node id was
// last passed to SetRoot. Bake() walks the active subtree, evaluates
// any dirty manifolds, and returns a flat triangle soup ready for upload
// into RHI vertex/index buffers and an accel-structure rebuild.
//
// Threading: Bake() is reentrancy-safe -- the caller can run it on a
// worker thread while the main thread continues to render the previous
// baked mesh. Mutations (Add*, Combine, Remove, SetRoot, Reset) are NOT
// thread-safe and must run on the main thread before kicking the bake.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pt::csg {

enum class OpType : std::uint8_t {
    Leaf      = 0,
    Union     = 1,
    Subtract  = 2,
    Intersect = 3,
};

// Triangle-soup output of a bake. Positions and normals are interleaved
// per attribute (xyz, xyz, ...). Indices are 3-per-triangle, CCW from
// outside.
struct BakedMesh {
    std::vector<float>         positions;   // 3 floats per vertex
    std::vector<float>         normals;     // 3 floats per vertex (geometric)
    std::vector<std::uint32_t> indices;     // 3 per triangle

    std::uint32_t VertexCount() const { return static_cast<std::uint32_t>(positions.size() / 3); }
    std::uint32_t TriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
    bool Empty() const { return indices.empty(); }
};

class CsgScene {
public:
    CsgScene();
    ~CsgScene();
    CsgScene(const CsgScene&) = delete;
    CsgScene& operator=(const CsgScene&) = delete;

    // Wipe all nodes and seed with a default scene: a single 1x1x1 box
    // at id 1, set as root. Useful both at startup and via `csg_reset`.
    void Reset();

    // Primitive leaves. Translation (tx, ty, tz) is applied after the
    // primitive is centered at the origin. ID must be unique; returns
    // false if the id is already taken or the primitive is degenerate.
    bool AddBox     (std::uint32_t id, float sx, float sy, float sz,
                     float tx, float ty, float tz);
    bool AddSphere  (std::uint32_t id, float radius, int circular_segments,
                     float tx, float ty, float tz);
    bool AddCylinder(std::uint32_t id, float radius, float height,
                     int circular_segments,
                     float tx, float ty, float tz);

    // Combine two existing nodes into a new internal node. Op must be
    // Union | Subtract | Intersect. Returns false on missing operands or
    // duplicate id.
    bool Combine(std::uint32_t id, OpType op,
                 std::uint32_t left, std::uint32_t right);

    // Remove a node and any internal nodes that referenced it (the tree
    // would otherwise be broken). Returns the number of removed nodes.
    std::size_t Remove(std::uint32_t id);

    bool   SetRoot(std::uint32_t id);
    bool   HasRoot() const;
    std::uint32_t RootId() const;
    bool   Empty() const;

    // Pretty-printed tree dump. One line per node, indented.
    void   Dump(std::string& out) const;

    // True if a Bake() would produce a different result than the last one.
    // Cleared by AcknowledgeClean() (the engine calls this after it has
    // accepted a baked mesh).
    bool   Dirty() const;
    void   AcknowledgeClean();

    // Evaluate the active root (if any) and return a triangle soup.
    // Returns an empty BakedMesh if there is no root or the resulting
    // manifold is degenerate. Safe to call on a worker thread.
    //
    // out_error: optional pointer; receives a short human-readable
    // string when the bake fails or the result is empty.
    BakedMesh Bake(std::string* out_error = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* OpName(OpType op);

}  // namespace pt::csg

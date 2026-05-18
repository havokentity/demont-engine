// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "CsgScene.h"

#include "../../core/Log.h"
#include "../../core/Tracy.h"

#include <manifold/manifold.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace pt::csg {

namespace {

struct Node {
    OpType        op   = OpType::Leaf;
    std::uint32_t left = 0;
    std::uint32_t right = 0;

    // Leaf-only: the manifold itself. Internal nodes leave this empty
    // (the bake recursively combines child manifolds).
    std::shared_ptr<manifold::Manifold> leaf;
};

// Compute geometric normals for each triangle face and emit unwelded
// vertices so the path tracer sees flat-shaded faces. CSG output looks
// noticeably worse with shared-vertex smooth normals at sharp edges.
void EmitFlatTriangleSoup(const manifold::MeshGL& gl, BakedMesh& out) {
    const std::size_t tri_count = gl.triVerts.size() / 3;
    out.positions.clear();
    out.normals.clear();
    out.indices.clear();
    out.positions.reserve(tri_count * 9);
    out.normals.reserve(tri_count * 9);
    out.indices.reserve(tri_count * 3);

    const std::size_t stride = gl.numProp;
    for (std::size_t t = 0; t < tri_count; ++t) {
        const std::uint32_t i0 = gl.triVerts[t * 3 + 0];
        const std::uint32_t i1 = gl.triVerts[t * 3 + 1];
        const std::uint32_t i2 = gl.triVerts[t * 3 + 2];
        const float* p0 = &gl.vertProperties[i0 * stride];
        const float* p1 = &gl.vertProperties[i1 * stride];
        const float* p2 = &gl.vertProperties[i2 * stride];

        const float ex = p1[0] - p0[0], ey = p1[1] - p0[1], ez = p1[2] - p0[2];
        const float fx = p2[0] - p0[0], fy = p2[1] - p0[1], fz = p2[2] - p0[2];
        float nx = ey * fz - ez * fy;
        float ny = ez * fx - ex * fz;
        float nz = ex * fy - ey * fx;
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-20f) { nx /= len; ny /= len; nz /= len; }

        const std::uint32_t base = static_cast<std::uint32_t>(out.positions.size() / 3);
        for (const float* p : {p0, p1, p2}) {
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.normals.push_back(nx);
            out.normals.push_back(ny);
            out.normals.push_back(nz);
        }
        out.indices.push_back(base + 0);
        out.indices.push_back(base + 1);
        out.indices.push_back(base + 2);
    }
}

}  // namespace

const char* OpName(OpType op) {
    switch (op) {
        case OpType::Leaf:      return "leaf";
        case OpType::Union:     return "union";
        case OpType::Subtract:  return "subtract";
        case OpType::Intersect: return "intersect";
    }
    return "?";
}

struct CsgScene::Impl {
    std::unordered_map<std::uint32_t, Node> nodes;
    std::uint32_t                           root_id = 0;
    bool                                    dirty   = true;

    // Mutex guards Bake's read of the tree against concurrent reads from
    // other threads. Mutations still belong on the main thread.
    mutable std::mutex                      bake_mu;

    std::shared_ptr<manifold::Manifold> EvaluateLocked(std::uint32_t id) const {
        auto it = nodes.find(id);
        if (it == nodes.end()) return nullptr;
        const Node& n = it->second;
        if (n.op == OpType::Leaf) return n.leaf;
        auto l = EvaluateLocked(n.left);
        auto r = EvaluateLocked(n.right);
        if (!l || !r) return nullptr;
        manifold::OpType op = manifold::OpType::Add;
        if      (n.op == OpType::Subtract)  op = manifold::OpType::Subtract;
        else if (n.op == OpType::Intersect) op = manifold::OpType::Intersect;
        return std::make_shared<manifold::Manifold>(l->Boolean(*r, op));
    }

    void DumpRecursive(std::uint32_t id, int depth, std::string& out) const {
        auto it = nodes.find(id);
        out.append(static_cast<std::size_t>(depth) * 2, ' ');
        if (it == nodes.end()) {
            out.append(std::to_string(id));
            out.append(" <missing>\n");
            return;
        }
        const Node& n = it->second;
        out.append(std::to_string(id));
        out.append(" ");
        out.append(OpName(n.op));
        if (n.op != OpType::Leaf) {
            out.append("(");
            out.append(std::to_string(n.left));
            out.append(", ");
            out.append(std::to_string(n.right));
            out.append(")");
        }
        out.append("\n");
        if (n.op != OpType::Leaf) {
            DumpRecursive(n.left, depth + 1, out);
            DumpRecursive(n.right, depth + 1, out);
        }
    }
};

CsgScene::CsgScene() : impl_(std::make_unique<Impl>()) { Reset(); }
CsgScene::~CsgScene() = default;

void CsgScene::Reset() {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    impl_->nodes.clear();
    impl_->root_id = 0;
    impl_->dirty = true;

    // Default scene: a single unit box at id 1, sitting on y=0.
    Node n;
    n.op   = OpType::Leaf;
    n.leaf = std::make_shared<manifold::Manifold>(
        manifold::Manifold::Cube({1.0, 1.0, 1.0}, true).Translate({0.0, 0.5, 0.0}));
    impl_->nodes.emplace(1u, std::move(n));
    impl_->root_id = 1;
}

bool CsgScene::AddBox(std::uint32_t id, float sx, float sy, float sz,
                      float tx, float ty, float tz) {
    if (sx <= 0.0f || sy <= 0.0f || sz <= 0.0f) return false;
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (impl_->nodes.count(id)) return false;
    Node n;
    n.op   = OpType::Leaf;
    n.leaf = std::make_shared<manifold::Manifold>(
        manifold::Manifold::Cube({sx, sy, sz}, true).Translate({tx, ty, tz}));
    impl_->nodes.emplace(id, std::move(n));
    impl_->dirty = true;
    return true;
}

bool CsgScene::AddSphere(std::uint32_t id, float radius, int circular_segments,
                         float tx, float ty, float tz) {
    if (radius <= 0.0f) return false;
    if (circular_segments < 3) circular_segments = 32;
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (impl_->nodes.count(id)) return false;
    Node n;
    n.op   = OpType::Leaf;
    n.leaf = std::make_shared<manifold::Manifold>(
        manifold::Manifold::Sphere(radius, circular_segments).Translate({tx, ty, tz}));
    impl_->nodes.emplace(id, std::move(n));
    impl_->dirty = true;
    return true;
}

bool CsgScene::AddCylinder(std::uint32_t id, float radius, float height,
                           int circular_segments,
                           float tx, float ty, float tz) {
    if (radius <= 0.0f || height <= 0.0f) return false;
    if (circular_segments < 3) circular_segments = 32;
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (impl_->nodes.count(id)) return false;
    Node n;
    n.op   = OpType::Leaf;
    // Cylinder is centered at the origin (true), then translated.
    n.leaf = std::make_shared<manifold::Manifold>(
        manifold::Manifold::Cylinder(height, radius, radius, circular_segments, true)
            .Translate({tx, ty, tz}));
    impl_->nodes.emplace(id, std::move(n));
    impl_->dirty = true;
    return true;
}

bool CsgScene::Combine(std::uint32_t id, OpType op,
                       std::uint32_t left, std::uint32_t right) {
    if (op == OpType::Leaf) return false;
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (impl_->nodes.count(id)) return false;
    if (!impl_->nodes.count(left) || !impl_->nodes.count(right)) return false;
    Node n;
    n.op = op; n.left = left; n.right = right;
    impl_->nodes.emplace(id, std::move(n));
    impl_->dirty = true;
    return true;
}

std::size_t CsgScene::Remove(std::uint32_t id) {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (!impl_->nodes.count(id)) return 0;

    // Cascade: any internal node referencing `id` is also dropped. Repeat
    // until no more references survive.
    std::vector<std::uint32_t> dead{id};
    std::size_t removed = 0;
    while (!dead.empty()) {
        const std::uint32_t d = dead.back();
        dead.pop_back();
        impl_->nodes.erase(d);
        ++removed;
        if (impl_->root_id == d) impl_->root_id = 0;
        for (auto it = impl_->nodes.begin(); it != impl_->nodes.end(); ++it) {
            const Node& n = it->second;
            if (n.op != OpType::Leaf && (n.left == d || n.right == d)) {
                dead.push_back(it->first);
            }
        }
    }
    impl_->dirty = true;
    return removed;
}

bool CsgScene::SetRoot(std::uint32_t id) {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    if (!impl_->nodes.count(id)) return false;
    if (impl_->root_id == id) return true;
    impl_->root_id = id;
    impl_->dirty = true;
    return true;
}

bool CsgScene::HasRoot() const {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    return impl_->root_id != 0 && impl_->nodes.count(impl_->root_id) != 0;
}

std::uint32_t CsgScene::RootId() const {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    return impl_->root_id;
}

bool CsgScene::Empty() const {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    return impl_->nodes.empty();
}

void CsgScene::Dump(std::string& out) const {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    out.clear();
    if (impl_->nodes.empty()) {
        out.append("(empty scene)\n");
        return;
    }
    for (const auto& [id, node] : impl_->nodes) {
        out.append(std::to_string(id));
        out.append(": ");
        out.append(OpName(node.op));
        if (node.op != OpType::Leaf) {
            out.append("(");
            out.append(std::to_string(node.left));
            out.append(", ");
            out.append(std::to_string(node.right));
            out.append(")");
        }
        if (id == impl_->root_id) out.append("  [root]");
        out.append("\n");
    }
    if (impl_->root_id != 0) {
        out.append("\ntree:\n");
        impl_->DumpRecursive(impl_->root_id, 0, out);
    }
}

bool CsgScene::Dirty() const {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    return impl_->dirty;
}

void CsgScene::AcknowledgeClean() {
    std::lock_guard<std::mutex> lk(impl_->bake_mu);
    impl_->dirty = false;
}

BakedMesh CsgScene::Bake(std::string* out_error) const {
    PT_ZONE_SCOPED_N("CsgScene::Bake");
    BakedMesh result;
    std::shared_ptr<manifold::Manifold> root;
    {
        std::lock_guard<std::mutex> lk(impl_->bake_mu);
        if (impl_->root_id == 0) {
            if (out_error) *out_error = "no root selected";
            return result;
        }
        root = impl_->EvaluateLocked(impl_->root_id);
    }
    if (!root) {
        if (out_error) *out_error = "evaluation failed (missing operand)";
        return result;
    }
    if (root->Status() != manifold::Manifold::Error::NoError) {
        if (out_error) *out_error = "manifold reports invalid result";
        return result;
    }
    if (root->IsEmpty()) {
        if (out_error) *out_error = "result is empty";
        return result;
    }
    auto gl = root->GetMeshGL();
    EmitFlatTriangleSoup(gl, result);
    if (out_error) out_error->clear();
    return result;
}

}  // namespace pt::csg

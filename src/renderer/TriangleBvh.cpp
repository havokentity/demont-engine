// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "TriangleBvh.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pt::renderer {

namespace {

inline void AabbReset(float (&mn)[3], float (&mx)[3]) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    mn[0] = mn[1] = mn[2] =  inf;
    mx[0] = mx[1] = mx[2] = -inf;
}

// Grow [mn, mx] to include the AABB stored on `wp`.
inline void AabbIncludeTri(float (&mn)[3], float (&mx)[3],
                           const float (&t_mn)[3], const float (&t_mx)[3]) {
    for (int i = 0; i < 3; ++i) {
        if (t_mn[i] < mn[i]) mn[i] = t_mn[i];
        if (t_mx[i] > mx[i]) mx[i] = t_mx[i];
    }
}

}  // namespace

void TriangleBvh::Build(std::span<const float>         positions,
                        std::span<const std::uint32_t> indices) {
    nodes_.clear();
    permuted_.clear();
    working_.clear();
    if (indices.empty() || (indices.size() % 3u) != 0u) return;

    const std::uint32_t pos_count = static_cast<std::uint32_t>(positions.size() / 3u);
    const std::uint32_t tri_count = static_cast<std::uint32_t>(indices.size() / 3u);
    working_.reserve(tri_count);

    // Build per-triangle scratch. Skip triangles whose indices fall
    // outside the positions buffer -- a corrupt index would otherwise
    // produce NaN AABBs that poison the entire tree. The BakedMesh
    // CSG output never produces that, but a future caller might; the
    // class contract is "ignore malformed input, never crash".
    for (std::uint32_t t = 0; t < tri_count; ++t) {
        const std::uint32_t i0 = indices[3u * t + 0u];
        const std::uint32_t i1 = indices[3u * t + 1u];
        const std::uint32_t i2 = indices[3u * t + 2u];
        if (i0 >= pos_count || i1 >= pos_count || i2 >= pos_count) continue;
        const float* v0 = &positions[3u * i0];
        const float* v1 = &positions[3u * i1];
        const float* v2 = &positions[3u * i2];

        WorkTri wt{};
        for (int a = 0; a < 3; ++a) {
            float mn = std::min(v0[a], std::min(v1[a], v2[a]));
            float mx = std::max(v0[a], std::max(v1[a], v2[a]));
            wt.aabb_min[a] = mn;
            wt.aabb_max[a] = mx;
            // Centroid of vertex centroids (= AABB mid-point for axis-
            // aligned median). Either is fine for centroid-median split;
            // vertex centroid is what AnalyticBvh's centre-of-prim does
            // semantically, so use the analog here.
            wt.centroid[a] = (v0[a] + v1[a] + v2[a]) * (1.0f / 3.0f);
        }
        wt.prim_id = t;
        working_.push_back(wt);
    }

    if (working_.empty()) return;

    // Upper bound on total nodes is 2N-1 (binary tree with N leaves at
    // leaf size 1). With kMaxLeafSize > 1 it's smaller; reserve
    // generously so push_back doesn't realloc mid-build (which would
    // invalidate any reference we hold across recursion).
    nodes_.reserve(2u * working_.size());
    nodes_.push_back({});       // root slot
    BuildRecursive(0u, 0u, static_cast<std::uint32_t>(working_.size()));

    // Emit the permuted triangle-id sequence from the now-reordered
    // working_ list. The shader reads triangles via this indirection so
    // each leaf's [left_first, left_first+count) range is a contiguous
    // run of triangle indices into the original `indices` buffer.
    permuted_.reserve(working_.size());
    for (const auto& wt : working_) permuted_.push_back(wt.prim_id);
}

std::uint32_t TriangleBvh::BuildRecursive(std::uint32_t node_idx,
                                          std::uint32_t first,
                                          std::uint32_t count) {
    // IMPORTANT: never bind a reference into nodes_ across a
    // nodes_.push_back() call. vector reallocation can move the
    // backing storage out from under us. AnalyticBvh hit exactly this
    // bug during initial bring-up (see AnalyticBvh.cpp:60-69 comment);
    // mirror the same fix here.
    float bn[3], bx[3];
    AabbReset(bn, bx);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& wt = working_[first + i];
        AabbIncludeTri(bn, bx, wt.aabb_min, wt.aabb_max);
    }
    for (int a = 0; a < 3; ++a) {
        nodes_[node_idx].aabb_min[a] = bn[a];
        nodes_[node_idx].aabb_max[a] = bx[a];
    }

    if (count <= kMaxLeafSize) {
        nodes_[node_idx].left_first = first;
        nodes_[node_idx].count      = count;
        return 1u;
    }

    // Choose split axis = axis with widest spread of centroids. Same
    // policy as AnalyticBvh::BuildRecursive.
    float centroid_mn[3];
    float centroid_mx[3];
    AabbReset(centroid_mn, centroid_mx);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& c = working_[first + i].centroid;
        for (int a = 0; a < 3; ++a) {
            if (c[a] < centroid_mn[a]) centroid_mn[a] = c[a];
            if (c[a] > centroid_mx[a]) centroid_mx[a] = c[a];
        }
    }
    int   axis   = 0;
    float widest = centroid_mx[0] - centroid_mn[0];
    for (int a = 1; a < 3; ++a) {
        float w = centroid_mx[a] - centroid_mn[a];
        if (w > widest) { widest = w; axis = a; }
    }

    if (widest <= 0.0f) {
        // Degenerate: all centroids coincide on every axis (e.g. a
        // mesh authored entirely at the origin). Fall through to the
        // index-median split below; nth_element on a no-spread
        // centroid still produces a stable in-place partition.
    } else {
        auto* begin = working_.data() + first;
        auto* mid   = begin + count / 2u;
        auto* end   = begin + count;
        std::nth_element(begin, mid, end,
            [axis](const WorkTri& a, const WorkTri& b) {
                return a.centroid[axis] < b.centroid[axis];
            });
    }

    const std::uint32_t left_count  = count / 2u;
    const std::uint32_t right_count = count - left_count;

    // Reserve child slots in `nodes_` BEFORE recursing. Write the
    // internal node's fields first so subsequent push_backs (which
    // may reallocate) don't matter -- the value is already in place.
    const std::uint32_t left_idx  = static_cast<std::uint32_t>(nodes_.size());
    nodes_[node_idx].left_first = left_idx;
    nodes_[node_idx].count      = 0u;   // 0 marks internal
    nodes_.push_back({});
    const std::uint32_t right_idx = left_idx + 1u;
    nodes_.push_back({});

    std::uint32_t emitted = 1u;
    emitted += BuildRecursive(left_idx,  first,              left_count);
    emitted += BuildRecursive(right_idx, first + left_count, right_count);
    return emitted;
}

}  // namespace pt::renderer

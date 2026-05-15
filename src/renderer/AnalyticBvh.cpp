// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "AnalyticBvh.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pt::renderer {

namespace {

inline void AabbInclude(float (&mn)[3], float (&mx)[3],
                        const float (&p)[3], float r) {
    for (int i = 0; i < 3; ++i) {
        if (p[i] - r < mn[i]) mn[i] = p[i] - r;
        if (p[i] + r > mx[i]) mx[i] = p[i] + r;
    }
}

inline void AabbReset(float (&mn)[3], float (&mx)[3]) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    mn[0] = mn[1] = mn[2] =  inf;
    mx[0] = mx[1] = mx[2] = -inf;
}

}  // namespace

void AnalyticBvh::Build(std::span<const BvhPrim> prims) {
    nodes_.clear();
    permuted_.clear();
    working_.clear();
    if (prims.empty()) return;

    working_.reserve(prims.size());
    for (const auto& p : prims) {
        WorkPrim wp{};
        wp.prim       = p;
        wp.centroid[0] = p.center[0];
        wp.centroid[1] = p.center[1];
        wp.centroid[2] = p.center[2];
        working_.push_back(wp);
    }

    // Upper bound on internal+leaf nodes is 2N-1 for N leaf primitives
    // when leaf size is 1. With kMaxLeafSize the count is smaller, but
    // we reserve generously and shrink later.
    nodes_.reserve(2 * working_.size());
    nodes_.push_back({});       // root slot
    BuildRecursive(0, 0, static_cast<std::uint32_t>(working_.size()));

    // Emit the permuted prim-id sequence from the (now reordered)
    // working_ list. The caller permutes its GPU primitive buffer to
    // match so leaf.left_first directly indexes the buffer.
    permuted_.reserve(working_.size());
    for (const auto& wp : working_) permuted_.push_back(wp.prim.prim_id);
}

std::uint32_t AnalyticBvh::BuildRecursive(std::uint32_t node_idx,
                                          std::uint32_t first,
                                          std::uint32_t count) {
    // IMPORTANT: never bind a reference into nodes_ across a
    // nodes_.push_back() call. vector reallocation can move the
    // backing storage, leaving the reference dangling -- internal
    // node writes after the child push_backs were corrupting memory
    // (rendered as empty BVH on first attempt). Compute the AABB
    // into locals and copy to nodes_[node_idx] each time we touch it.
    float bn[3], bx[3];
    AabbReset(bn, bx);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& wp = working_[first + i];
        AabbInclude(bn, bx, wp.prim.center, wp.prim.radius);
    }
    for (int a = 0; a < 3; ++a) {
        nodes_[node_idx].aabb_min[a] = bn[a];
        nodes_[node_idx].aabb_max[a] = bx[a];
    }

    if (count <= kMaxLeafSize) {
        nodes_[node_idx].left_first = first;
        nodes_[node_idx].count      = count;
        return 1;
    }

    // Choose split axis = axis with widest spread of centroids.
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
    int   axis  = 0;
    float widest = centroid_mx[0] - centroid_mn[0];
    for (int a = 1; a < 3; ++a) {
        float w = centroid_mx[a] - centroid_mn[a];
        if (w > widest) { widest = w; axis = a; }
    }

    // Degenerate: all centroids coincide (e.g., concentric spheres at
    // the same origin). Centroid-based split would recurse forever.
    // Fall back to a simple index-median split so leaf sizes still
    // stay bounded by ~kMaxLeafSize log-depth-down. The shader-side
    // per-leaf loop relies on this invariant -- a leaf with hundreds
    // of prims would defeat the whole point of the BVH.
    if (widest <= 0.0f) {
        // No partitioning needed; the range is already arbitrary since
        // every prim has the same centroid. Just slice it down the
        // middle by index.
    } else {
        // Median split: partition around the median centroid on `axis`.
        // nth_element is O(N) and good enough; SAH would build a slightly
        // tighter tree but the build cost on edits would rise nontrivially
        // for a marginal trace-time win on the prim counts we expect.
        auto* begin = working_.data() + first;
        auto* mid   = begin + count / 2;
        auto* end   = begin + count;
        std::nth_element(begin, mid, end,
            [axis](const WorkPrim& a, const WorkPrim& b) {
                return a.centroid[axis] < b.centroid[axis];
            });
    }

    const std::uint32_t left_count  = count / 2;
    const std::uint32_t right_count = count - left_count;

    // Reserve two child slots; left child first, right child immediately
    // after. left_first on the internal node records the left child's
    // index. Write to the internal node BEFORE the push_backs so any
    // reallocation after this point is harmless to it.
    const std::uint32_t left_idx  = static_cast<std::uint32_t>(nodes_.size());
    nodes_[node_idx].left_first = left_idx;
    nodes_[node_idx].count      = 0;   // 0 marks internal
    nodes_.push_back({});
    const std::uint32_t right_idx = left_idx + 1u;
    nodes_.push_back({});

    std::uint32_t emitted = 1;
    emitted += BuildRecursive(left_idx,  first,              left_count);
    emitted += BuildRecursive(right_idx, first + left_count, right_count);
    return emitted;
}

}  // namespace pt::renderer

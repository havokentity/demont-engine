// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace pt::renderer {

// Flat-linear binary BVH over triangle primitives. Follow-up to PR #106:
// the SW Möller-Trumbore linear-scan path that #106 added to
// PathTrace.slang is O(N) per ray, which crushed framerate on even
// modest CSG meshes (~1 FPS at 1080p on the 1032-tri cornell_csg
// scene on an M4 Max).
// This class mirrors AnalyticBvh's structure so the renderer can build
// it on the same async bake thread that produces the BakedMesh and
// upload it as two SSBOs alongside the existing vbuf / ibuf.
//
// Node layout (32 bytes, deliberately identical to renderer::BvhNode):
//
//   float3 aabb_min;     // 12B  -> lo.xyz on GPU
//   uint   left_first;   // 4B   -> asuint(lo.w)
//                        //   internal: index of left child node
//                        //   leaf:     first triangle slot in
//                        //             PermutedPrimIds()
//   float3 aabb_max;     // 12B  -> hi.xyz on GPU
//   uint   count;        // 4B   -> asuint(hi.w)
//                        //   0  -> internal (right child = left_first + 1)
//                        //   >0 -> leaf with this many consecutive
//                        //          triangle indices in PermutedPrimIds()
//
// The GPU side reads this as a StructuredBuffer<float4> of pairs and
// recovers the uint fields with asuint() -- same trick AnalyticBvh
// uses, for the same reason: Slang's MSL emit pads nested float3 inside
// a struct to 16 bytes, which would otherwise produce a 64B GPU stride
// vs the 32B host stride. Keep this layout in sync with the analytic
// BVH if you ever change either.
//
// Build strategy mirrors AnalyticBvh exactly:
//   * top-down median-split along the widest centroid-spread axis
//   * recurse until leaf has <= kMaxLeafSize triangles
//   * reorders an internal triangle-index list in place so leaves point
//     to contiguous ranges in PermutedPrimIds()
//
// SAH would build a slightly tighter tree at higher build cost; the
// project memory says don't pull future work forward -- centroid-median
// is the documented choice for this PR. SAH is a follow-up.
//
// Determinism: Build() is purely deterministic given identical input
// triangles (no std::random / no time-based seeds). std::nth_element
// is implementation-defined in *which* partition it produces among
// valid ones, but for a single build on a single host the ordering
// is stable -- and the shader / test only relies on the resulting
// leaf-contiguous range being the same set of triangles as the host
// reference, not on the specific permutation.
struct TriBvhNode {
    float        aabb_min[3];
    std::uint32_t left_first;
    float        aabb_max[3];
    std::uint32_t count;
};
static_assert(sizeof(TriBvhNode) == 32, "TriBvhNode must be 32 bytes for std430");

class TriangleBvh {
public:
    // Match AnalyticBvh::kMaxLeafSize so the shader-side stack-depth
    // budget (depth ~64) is comfortably above log2(N / leaf_cap) for
    // any reasonable triangle count we'll see in this engine.
    // Empirically tuned later if needed -- 4 is the documented starting
    // point per the PR scope.
    static constexpr std::uint32_t kMaxLeafSize = 4u;

    // Build over a flat triangle source. positions is xyz-packed (3
    // floats per vertex), indices is 3 per triangle (so triangle k
    // reads indices[3k+0..3k+2] which then index into positions/3).
    // This is exactly the shape pt::csg::BakedMesh produces, so the
    // engine can hand it to us untouched.
    //
    // No-op (clears state, leaves Empty() == true) if indices is empty
    // or its size is not a multiple of 3.
    void Build(std::span<const float>         positions,
               std::span<const std::uint32_t> indices);

    bool        Empty()     const { return nodes_.empty(); }
    std::size_t NodeCount() const { return nodes_.size(); }

    // Flat node array, root at index 0. Uploaded directly to a
    // GPU SSBO; the shader reads it as a StructuredBuffer<float4> of
    // (lo, hi) pairs. See the layout comment at the top of this file.
    const std::vector<TriBvhNode>& Nodes() const { return nodes_; }

    // Permuted-triangle-id array. permuted[leaf_first + k] gives the
    // original triangle index (i.e. the k-th triangle's first index in
    // `indices` is at indices[3 * permuted[leaf_first + k]]) for the
    // k-th triangle in a leaf at offset leaf_first. The shader uses
    // this exactly the same way the analytic BVH walker uses its
    // permuted-prim-id list.
    const std::vector<std::uint32_t>& PermutedPrimIds() const { return permuted_; }

private:
    // Per-triangle scratch: AABB + centroid. Built once at Build()
    // entry, reordered during the recursive split. The original
    // triangle index lives in WorkTri::prim_id; emit it to permuted_
    // at the end (the .indices buffer stays untouched -- the shader
    // indirects through permuted_ to find each leaf's triangles).
    struct WorkTri {
        float        aabb_min[3];
        float        aabb_max[3];
        float        centroid[3];
        std::uint32_t prim_id;          // index in indices/3 (the triangle id)
    };

    std::uint32_t BuildRecursive(std::uint32_t node_idx,
                                  std::uint32_t first,
                                  std::uint32_t count);

    std::vector<WorkTri>            working_;
    std::vector<TriBvhNode>         nodes_;
    std::vector<std::uint32_t>      permuted_;
};

}  // namespace pt::renderer

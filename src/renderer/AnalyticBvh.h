// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace pt::renderer {

// Flat-linear binary BVH over analytic primitives that have a finite
// bounding box (spheres today; future: box, disk, cylinder). Planes are
// excluded -- they have infinite extent and stay on the linear-scan
// path at the front of the primitive buffer.
//
// Node layout, 32 bytes total. The GPU-side encoding is intentionally a
// pair of float4s in StructuredBuffer<float4> (lo / hi), with the uint
// fields stored in the .w lanes and recovered via asuint() in the
// shader. The host struct below uses the natural C++ layout (also 32B);
// the GPU side avoids a StructuredBuffer<BvhNode> because Slang's MSL
// emit pads nested float3 inside a struct to 16 bytes, yielding a 64B
// stride on Metal vs the 32B host stride. Keep this in sync if you
// add new fields.
//
//   float3 aabb_min;     // 12B  -> lo.xyz on GPU
//   uint   left_first;   // 4B   -> asuint(lo.w)
//                        //   internal: index of left child node
//                        //   leaf:     first primitive index
//   float3 aabb_max;     // 12B  -> hi.xyz on GPU
//   uint   count;        // 4B   -> asuint(hi.w)
//                        //   0  -> internal (right child = left_first + 1)
//                        //   >0 -> leaf with this many consecutive prims
//
// Build strategy: top-down median-split along the axis of widest spread
// of leaf-centroid distribution. Recurse until the leaf has <=
// kMaxLeafSize primitives. Reorders the input primitive index list in-
// place so leaves point to contiguous ranges.
struct BvhNode {
    float        aabb_min[3];
    std::uint32_t left_first;
    float        aabb_max[3];
    std::uint32_t count;
};
static_assert(sizeof(BvhNode) == 32, "BvhNode must be 32 bytes for std430");

struct BvhPrim {
    float center[3];
    float radius;          // bounding radius; AABB = center +/- radius
    std::uint32_t prim_id; // index into the host primitive list this BVH
                           // came from (used by the caller to reorder
                           // the GPU primitive buffer to match leaf
                           // order).
};

class AnalyticBvh {
public:
    static constexpr std::uint32_t kMaxLeafSize = 4;

    // Build over an input list of finite primitives. Returns the
    // reordered prim_id sequence so the caller can permute the GPU
    // primitive buffer to match. nodes_ then holds the flat BVH; the
    // root is at index 0.
    void Build(std::span<const BvhPrim> prims);

    bool Empty() const { return nodes_.empty(); }
    std::size_t NodeCount() const { return nodes_.size(); }

    // Flat node array, root at index 0. The host struct is 32B; the
    // GPU side reads this as a StructuredBuffer<float4> of pairs and
    // recovers the uint fields with asuint(). See the node-layout
    // comment at the top of this file for why.
    const std::vector<BvhNode>& Nodes() const { return nodes_; }

    // Permuted prim_id sequence. permuted[leaf_first_prim + k] gives
    // the prim_id for the k-th prim in a leaf at offset leaf_first_prim.
    const std::vector<std::uint32_t>& PermutedPrimIds() const { return permuted_; }

private:
    // Recursive build: writes a node at nodes_[node_idx] describing the
    // range working_[first..first+count). Returns the count of nodes
    // emitted in the subtree rooted here (including node_idx).
    std::uint32_t BuildRecursive(std::uint32_t node_idx,
                                  std::uint32_t first,
                                  std::uint32_t count);

    // Working buffer of (prim, centroid) pairs; reordered during build.
    struct WorkPrim {
        BvhPrim prim;
        float   centroid[3];
    };
    std::vector<WorkPrim>           working_;
    std::vector<BvhNode>            nodes_;
    std::vector<std::uint32_t>      permuted_;
};

}  // namespace pt::renderer

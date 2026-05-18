// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace pt::renderer {

// --- SDF Phase 1 (#97) -----------------------------------------------------
// Signed-distance-field primitives. Each `SdfPrim` is one self-contained
// SDF expression -- either a single primitive (sphere / box / torus /
// capsule / rounded box / plane) or a flat sequence of those leaves
// combined via smooth-CSG ops (smin / smax / sdiff / displace). A
// cluster has a tight world-space AABB used as the BVH bound; the shader
// only sphere-traces *inside* that AABB after the BVH says the ray
// entered it.
//
// On-GPU layout per cluster: a fixed-size "header" block (material +
// AABB) followed by a small fixed-size array of node records that the
// shader walks linearly to evaluate the distance. Keeping the per-node
// stride identical on host and shader is load-bearing for the same
// std430 / MSL-padding reasons documented on BvhNode above.
//
// IMPORTANT: SDFs live in a SEPARATE primitive buffer from AnalyticPrim.
// The existing analytic-primitive path is untouched so its goldens stay
// bit-exact across this change.

// SDF leaf-primitive kinds. Numeric values are part of the host/shader
// wire format -- never renumber without bumping the shader too.
//
// Ids 0..5 are analytic SDFs evaluated in shaders/SdfPrimitives.slang
// (closed-form distance + analytic gradient). Ids 6..8 are fractal
// distance estimators (Phase 3 of #96, see issue #99) evaluated in
// shaders/SdfFractals.slang with bounded iteration and central-
// difference normals -- they live in their own shader module because
// the trace path is shape-specific (no smooth-CSG composition with
// analytic shapes in Phase 3) and the normal recovery costs ~4 extra
// DE evaluations per hit.
enum SdfShape : std::uint32_t {
    SDF_SHAPE_SPHERE       = 0,   // params: radius
    SDF_SHAPE_BOX          = 1,   // params: half-extents (3)
    SDF_SHAPE_ROUNDED_BOX  = 2,   // params: half-extents (3), corner radius
    SDF_SHAPE_TORUS        = 3,   // params: major radius R, minor radius r
    SDF_SHAPE_CAPSULE      = 4,   // params: half-length on +Y axis, radius
    SDF_SHAPE_PLANE        = 5,   // params: unit normal (3), distance d (n . p + d = 0)
    // --- SDF Phase 3 (#99) fractal DEs --------------------------------------
    // For all fractal shapes:
    //   params[0] = per-shape "scale" (power for MANDELBULB; linear
    //               step scale for MANDELBOX; radial-shrink scale for
    //               APOLLONIAN). 0 means "use the r_sdf_fractal_power
    //               cvar default".
    //   params[1] = effective bound radius (metres). The host uses
    //               this to size the cluster AABB; rays only enter the
    //               sphere-trace inside this radius. Defaults vary
    //               per shape (Mandelbulb ~1.2, Mandelbox ~4.0,
    //               Apollonian ~2.0); fixture authors override via
    //               the console-command argument.
    //   params[2..3] reserved.
    SDF_SHAPE_MANDELBULB   = 6,
    SDF_SHAPE_MANDELBOX    = 7,
    SDF_SHAPE_APOLLONIAN   = 8,
};

// True when `shape` is a fractal DE (handled by shaders/SdfFractals.slang).
// Used by the host AABB computation and by the upload path's leaf
// validation to keep the analytic and fractal code paths cleanly
// separated.
inline bool IsSdfFractalShape(std::uint32_t shape) {
    return shape == SDF_SHAPE_MANDELBULB ||
           shape == SDF_SHAPE_MANDELBOX  ||
           shape == SDF_SHAPE_APOLLONIAN;
}

// SDF CSG ops. opSmoothUnion / opSmoothSubtract / opSmoothIntersect (k
// is the smoothing radius in metres); opDisplace adds a scalar bump
// magnitude to its child. See SdfPrimitives.slang for the formulas.
enum SdfOp : std::uint32_t {
    SDF_OP_LEAF             = 0,   // params: see SdfShape above
    SDF_OP_SMOOTH_UNION     = 1,   // children: a, b; param k
    SDF_OP_SMOOTH_SUBTRACT  = 2,   // children: a, b; param k (b carved from a)
    SDF_OP_SMOOTH_INTERSECT = 3,   // children: a, b; param k
    SDF_OP_DISPLACE         = 4,   // children: a; param amp (analytic bump along normal)
};

// One node in a cluster's flat op-tree. Children are *node indices*
// within the same cluster. The host emits nodes in POST-ORDER (children
// before parents), so the root of any cluster is always the LAST node:
// index `node_count - 1`. The shader walks the array in ascending order
// so that by the time we reach a node, its children's evaluated
// distances are already in a small register stack. We keep
// nodes_per_cluster fixed and small (Phase 1: <= 8 per cluster) so the
// shader can keep the stack in registers without spilling.
// See ComputeSdfAabb in AnalyticBvh.cpp and sdfClusterDist /
// sdfClusterNormal in SdfPrimitives.slang -- they all read the root as
// `node_count - 1`.
struct SdfNode {
    std::uint32_t op       = SDF_OP_LEAF;        // SdfOp
    std::uint32_t shape    = SDF_SHAPE_SPHERE;   // SdfShape (only meaningful when op == LEAF)
    std::uint32_t child_a  = 0;                  // index of left child  (ops only)
    std::uint32_t child_b  = 0;                  // index of right child (binary ops only)
    // Per-node parameters. Layout:
    //   LEAF SPHERE        : params[0]=radius
    //   LEAF BOX           : params[0..2]=half-extents
    //   LEAF ROUNDED_BOX   : params[0..2]=half-extents, params[3]=corner radius
    //   LEAF TORUS         : params[0]=R, params[1]=r
    //   LEAF CAPSULE       : params[0]=half-height, params[1]=radius
    //   LEAF PLANE         : params[0..2]=normal (unit), params[3]=d
    //   SMOOTH_*           : params[0]=k (smoothing radius, metres)
    //   DISPLACE           : params[0]=amplitude (metres)
    float         params[4] {0.0f, 0.0f, 0.0f, 0.0f};
    // Local-frame origin/offset. The shape SDF is evaluated against
    // (p - center). For ops this lane is reserved (zeroed).
    float         center[3] {0.0f, 0.0f, 0.0f};
    std::uint32_t _pad      = 0;     // pad to 48B for clean std430 stride
};
static_assert(sizeof(SdfNode) == 48, "SdfNode must be 48 bytes for std430 stride");

// One SDF cluster. A cluster maps 1:1 to a BVH leaf primitive. Up to
// kSdfMaxNodes nodes per cluster (Phase 1 -- expression size budget;
// row of `sdf_smin(a, b, k)` only needs 3 nodes each so the typical
// scene stays well under). AABB is computed by the host from the leaf
// shapes plus the smoothing radius.
//
// Material follows the AnalyticPrim convention (0=Lambert,1=Metal,
// 2=Dielectric) so hits feed the same BSDF path as analytic prims.
struct SdfPrim {
    static constexpr std::uint32_t kMaxNodes = 8;
    std::array<SdfNode, kMaxNodes> nodes {};
    std::uint32_t node_count   = 0;
    std::uint32_t material     = 0;     // 0 Lambert, 1 Metal, 2 Dielectric
    float         aabb_min[3]  {0.0f, 0.0f, 0.0f};
    float         aabb_max[3]  {0.0f, 0.0f, 0.0f};
    float         albedo[3]    {0.7f, 0.7f, 0.7f};
    float         roughness    = 0.0f;
    float         ior          = 1.5f;
    std::uint32_t _pad[3]      {0, 0, 0}; // pad to clean std430 boundary
};
// kMaxNodes * 48 = 384 nodes block. + 4 (node_count) + 4 (material) +
// 12 (aabb_min) + 12 (aabb_max) + 12 (albedo) + 4 (roughness) + 4 (ior)
// + 12 (pad) = 448. Bump if you add fields.
static_assert(sizeof(SdfPrim) == 48 * 8 + 4 + 4 + 12 + 12 + 12 + 4 + 4 + 12,
              "SdfPrim layout drift");
// --- end SDF Phase 1 -------------------------------------------------------

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

// --- SDF Phase 1 (#97) -----------------------------------------------------
// Walk the cluster's op-tree and fill aabb_min / aabb_max in-place with
// a tight world-space AABB that bounds the root expression's zero-isosurface.
//
// Leaf shapes contribute their analytic AABB; ops widen by their k /
// amplitude (smoothing radius / displacement magnitude). This is the
// envelope the path tracer's BVH leaf will use to bound the sphere-
// tracing loop, so the bound MUST be a true superset of the actual
// isosurface (false negatives = missed hits, false positives = wasted
// step budget but rendering stays correct). Returns false if the prim
// is malformed (e.g. node_count == 0 or a child index out of range).
bool ComputeSdfAabb(SdfPrim& prim);
// --- end SDF Phase 1 -------------------------------------------------------

}  // namespace pt::renderer

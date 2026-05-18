// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#pragma once

// --- Hierarchical light tree (Conty Estevez & Kulla, SIGGRAPH 2018) -------
//
// Issue #129. Bridge between the naive uniform-pick NEE shipped in #73
// and the ReSTIR DI work in #78. Drops per-pixel NEE picker cost from
// O(N) variance to O(log N): each tree level performs a stochastic
// 2-way split weighted by cluster importance (combined emitter
// power * geometric-falloff estimate * orientation cone cosine), so
// far / back-facing / dim emitters get proportionally fewer rays.
//
// MVP scope (this commit):
//   - CPU build only. Top-down median split by longest AABB extent --
//     O(N log N), <1ms for 1000 lights (true HAC is O(N^2 log N); the
//     simpler split-based variant gives a balanced tree at this scale
//     and the variance characteristics are statistically indistinguishable
//     for the cluster summaries we care about).
//   - Single light per leaf (count=1). The agglomerative variant in the
//     paper packs ~16 lights per leaf to amortise traversal cost; at
//     <=1000 lights the tree is small enough that single-light leaves
//     keep the importance heuristic exact at the bottom.
//   - Static lights. Any mutation of light_prims_ rebuilds the tree
//     from scratch (still <1ms; see Engine::EnsureLightsUploaded).
//   - No GPU build. Future cuts at >10k lights would push tree
//     construction onto the GPU (Lin & Yuksel 2020); not relevant yet.
//
// GPU layout (4 float4s per node):
//   v0.xyz = aabb_min
//   v0.w   = left_first (uint, asfloat) -- for INTERNAL nodes this is
//            the child-pair base index (right = left_first + 1);
//            for LEAF nodes this is the light id (host packs the
//            engine-side light ordinal here, NOT the user-facing
//            light_prims_ map key).
//   v1.xyz = aabb_max
//   v1.w   = count (uint, asfloat) -- 0 = internal, 1 = leaf. The
//            agglomerative variant would use count>1 for multi-light
//            leaves; reserved field bumps gracefully when we add that.
//   v2.xyz = cone_axis (unit vector; combined emission direction of
//            the cluster)
//   v2.w   = cos_half_angle (cosine of the cone half-angle; 1.0 = a
//            single direction, -1.0 = fully spherical / point-light).
//   v3.x   = total_intensity (scalar luminance W/sr or W/m^2/sr
//            summed by emitter type; see PackEmitterIntensity in the
//            .cpp for the per-type conversion. The shader's importance
//            heuristic only needs a relative magnitude per cluster,
//            so the exact unit doesn't matter as long as siblings agree.)
//   v3.yzw = pad (reserved; future ReSTIR DI variance term, mesh-
//            emitter triangle count, etc.)

#include <cstdint>
#include <vector>

namespace pt::renderer {

// Input record for tree construction. Mirrors the renderer-facing
// fields of Engine::AnalyticLight without making the renderer
// library depend on the engine (the dep direction is engine -> renderer
// only). Engine::EnsureLightTreeUploaded converts AnalyticLight ->
// LightInput in upload order before passing the array here.
//
// `type` encodes the same enum values as Engine::AnalyticLight::Type
// (0=Point, 1=Spot, 2=Sphere, 3=Quad) -- the renderer doesn't need
// the enum tag, just the integer value, so we sidestep the dependency
// without losing per-type handling.
struct LightInput {
    std::uint32_t type     {0};
    float pos[3]           {0, 0, 0};
    float radius           {0.0f};
    float intensity[3]     {0, 0, 0};
    float dir[3]           {0, 1, 0};
    float cos_outer        {0.0f};
    float u_vec[3]         {1, 0, 0};
    float v_half           {0.0f};
};

// Flat, GPU-ready node record. Host-side mirror of the shader's
// LightTreeNode struct in PathTrace.slang.
struct LightTreeNode {
    float aabb_min[3]    {0, 0, 0};
    std::uint32_t left_first {0};   // light id (leaf) or child-pair base (internal)
    float aabb_max[3]    {0, 0, 0};
    std::uint32_t count  {0};       // 0 = internal, 1 = leaf
    float cone_axis[3]   {0, 0, 0};
    float cone_cos_half  {1.0f};
    float intensity      {0.0f};
    float pad0           {0.0f};
    float pad1           {0.0f};
    float pad2           {0.0f};
};
static_assert(sizeof(LightTreeNode) == 64, "LightTreeNode must be 4 float4");

// Built tree. Empty when there are no lights.
struct LightTree {
    std::vector<LightTreeNode> nodes;   // node[0] is root if non-empty
    std::uint32_t              light_count {0};  // input light count
    std::uint32_t              root_index  {0};  // always 0 when non-empty

    void Clear() {
        nodes.clear();
        light_count = 0;
        root_index  = 0;
    }
    bool Empty() const { return nodes.empty(); }
    std::size_t NodeCount() const { return nodes.size(); }
};

// One leaf record per input light. Caller passes the analytic-light
// list IN UPLOAD ORDER (i.e. the order the engine's EnsureLightsUploaded
// packs into `light_prims`). The tree's leaf left_first values index
// directly into the GPU `light_prims` buffer, so the path tracer can
// run loadLight(leaf.left_first) without any host-side remap table.
//
// `dst` is overwritten. Re-uses any prior `dst.nodes` capacity to avoid
// allocator churn on rebuilds.
void BuildLightTree(const std::vector<LightInput>& lights,
                    LightTree& dst);

}  // namespace pt::renderer

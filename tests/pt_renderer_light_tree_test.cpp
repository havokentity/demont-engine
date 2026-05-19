// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::renderer::BuildLightTree (issue #129 + #177).
//
// The light tree (Conty Estevez & Kulla 2018) produces a flat array of
// LightTreeNode records that the path-trace shader walks via
// pickLightFromTree (shaders/PathTrace.slang).  The shader's traversal
// hard-codes the invariant that an internal node's two children live at
// CONSECUTIVE indices: right = left_first + 1.  Issue #177: at N=4 the
// builder violated that invariant -- only one subtree's leaves were
// reachable, halving the effective light count -- and the bug
// manifested as "lights show fewer lights" in the user's smoke matrix.
//
// Tests below pin:
//   - The contiguous-children layout invariant (right = left + 1)
//     across N=1..256 + larger samples (200/500/1000).  Pre-fix the
//     test FAILED for N >= 4.
//   - Every input light gets exactly ONE reachable leaf node, and the
//     leaf's `left_first` rejects no light id from [0, N).
//   - Tree shape: N leaves + (N-1) internal = 2N - 1 nodes for N >= 1.
//   - All nodes have a finite AABB and a finite cone (no NaN).
//
// Pure-CPU + deterministic.  Links pt_renderer.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "renderer/LightTree.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

using pt::renderer::BuildLightTree;
using pt::renderer::LightInput;
using pt::renderer::LightTree;
using pt::renderer::LightTreeNode;

namespace {

// Make a deterministic spread of lights across all four supported types.
// Returns a vector of LightInput suitable for BuildLightTree.
std::vector<LightInput> MakeLights(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_d(-10.0f, 10.0f);

    std::vector<LightInput> lights;
    lights.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        LightInput L;
        L.type    = static_cast<std::uint32_t>(i % 4);   // point/spot/sphere/quad mix
        L.pos[0]  = pos_d(rng);
        L.pos[1]  = pos_d(rng);
        L.pos[2]  = pos_d(rng);
        L.radius  = 0.5f;
        L.intensity[0] = 1.0f;
        L.intensity[1] = 1.0f;
        L.intensity[2] = 1.0f;
        L.dir[0] = 0.0f; L.dir[1] = 1.0f; L.dir[2] = 0.0f;
        L.cos_outer = 0.5f;
        L.u_vec[0] = 1.0f; L.u_vec[1] = 0.0f; L.u_vec[2] = 0.0f;
        L.v_half  = 1.0f;
        lights.push_back(L);
    }
    return lights;
}

// Walk the tree as the GPU shader would (every internal node descends
// to {left_first, left_first + 1}).  Collects the set of UNIQUE leaf
// light ids reachable from the root.  This is the post-#177 reachability
// invariant.
std::unordered_set<std::uint32_t>
ShaderReachableLightIds(const LightTree& tree) {
    std::unordered_set<std::uint32_t> out;
    if (tree.nodes.empty()) return out;

    std::vector<std::uint32_t> stack;
    stack.push_back(tree.root_index);
    std::vector<bool> visited(tree.nodes.size(), false);

    while (!stack.empty()) {
        const std::uint32_t idx = stack.back();
        stack.pop_back();
        if (idx >= tree.nodes.size()) {
            // Out-of-range child -> bug; surface via test side-effect.
            FAIL("traversal hit out-of-range node index ", idx);
            return out;
        }
        if (visited[idx]) continue;
        visited[idx] = true;

        const LightTreeNode& n = tree.nodes[idx];
        if (n.count == 1u) {
            out.insert(n.left_first);
        } else {
            stack.push_back(n.left_first);
            stack.push_back(n.left_first + 1u);
        }
    }
    return out;
}

// Verify the tree-structural invariants the GPU walk relies on.
//   1. nodes.size() == 2 * N - 1 for N >= 1 (full binary tree, leaves +
//      internals).
//   2. Every internal node has count == 0 and both children in-range.
//   3. Every leaf has count == 1 and a light_id in [0, light_count).
//   4. Every AABB / cone is finite (no NaN / Inf).
//   5. Light ids on the leaves are a permutation of [0, N).
void CheckStructuralInvariants(const LightTree& tree, std::uint32_t N) {
    REQUIRE(tree.light_count == N);
    REQUIRE(tree.nodes.size() == 2u * N - 1u);

    std::unordered_set<std::uint32_t> leaf_ids;
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        const LightTreeNode& n = tree.nodes[i];

        // Finite AABB / cone (the union math can blow up on NaN inputs;
        // here all inputs are finite so the output must be too).
        for (int k = 0; k < 3; ++k) {
            REQUIRE(std::isfinite(n.aabb_min[k]));
            REQUIRE(std::isfinite(n.aabb_max[k]));
            REQUIRE(std::isfinite(n.cone_axis[k]));
            REQUIRE(n.aabb_min[k] <= n.aabb_max[k]);
        }
        REQUIRE(std::isfinite(n.cone_cos_half));
        REQUIRE(n.cone_cos_half >= -1.0001f);
        REQUIRE(n.cone_cos_half <=  1.0001f);
        REQUIRE(std::isfinite(n.intensity));
        REQUIRE(n.intensity >= 0.0f);

        if (n.count == 0u) {
            // Internal: left + right both addressable.
            REQUIRE(n.left_first + 1u < tree.nodes.size());
        } else if (n.count == 1u) {
            // Leaf: light id valid.
            REQUIRE(n.left_first < N);
            const bool inserted = leaf_ids.insert(n.left_first).second;
            REQUIRE_MESSAGE(inserted,
                "duplicate leaf light_id ", n.left_first,
                " -- light tree must place each input light at exactly one leaf");
        } else {
            FAIL("count must be 0 (internal) or 1 (leaf), got ", n.count);
        }
    }
    REQUIRE(leaf_ids.size() == N);
}

}  // namespace

TEST_CASE("LightTree: empty input -> empty tree") {
    std::vector<LightInput> lights;
    LightTree tree;
    BuildLightTree(lights, tree);
    CHECK(tree.Empty());
    CHECK(tree.light_count == 0u);
    CHECK(tree.NodeCount() == 0u);
}

TEST_CASE("LightTree: single-light input -> single leaf") {
    auto lights = MakeLights(1, /*seed=*/1u);
    LightTree tree;
    BuildLightTree(lights, tree);

    REQUIRE(tree.NodeCount() == 1u);
    CHECK(tree.nodes[0].count == 1u);
    CHECK(tree.nodes[0].left_first == 0u);
    CHECK(tree.light_count == 1u);
}

// PR #177 regression: at N=4 the shader's right = left + 1 walk only
// reached 2 of the 4 input lights.  This explicit small-N cell catches
// the exact size that exposed the bug.
TEST_CASE("LightTree: N=4 -- all leaves reachable via shader walk (#177)") {
    auto lights = MakeLights(4, /*seed=*/4u);
    LightTree tree;
    BuildLightTree(lights, tree);

    CheckStructuralInvariants(tree, /*N=*/4u);

    const auto reachable = ShaderReachableLightIds(tree);
    CHECK_MESSAGE(reachable.size() == 4u,
        "shader walk reached ", reachable.size(),
        "/4 lights -- regression of #177 (right = left + 1 vs builder layout)");
}

// Cover the broader range so the fix doesn't silently regress on
// any other small-or-medium N.  Pre-fix the assertion failed for
// every N >= 4 in this loop.
TEST_CASE("LightTree: all leaves reachable across N=1..256 (#177)") {
    for (int n = 1; n <= 256; ++n) {
        auto lights = MakeLights(n, /*seed=*/static_cast<std::uint32_t>(n));
        LightTree tree;
        BuildLightTree(lights, tree);

        CheckStructuralInvariants(tree, static_cast<std::uint32_t>(n));

        const auto reachable = ShaderReachableLightIds(tree);
        REQUIRE_MESSAGE(reachable.size() == static_cast<std::size_t>(n),
            "N=", n, ": shader walk only reached ", reachable.size(),
            " of ", n, " leaves");
    }
}

// MVP scale: the same fixture the golden test uses (#129 -- 200 lights),
// plus a couple of larger sizes inside the "<=1000 light" MVP envelope.
TEST_CASE("LightTree: all leaves reachable at MVP scale (200/500/1000)") {
    for (int n : {200, 500, 1000}) {
        auto lights = MakeLights(n, /*seed=*/static_cast<std::uint32_t>(n));
        LightTree tree;
        BuildLightTree(lights, tree);

        CheckStructuralInvariants(tree, static_cast<std::uint32_t>(n));

        const auto reachable = ShaderReachableLightIds(tree);
        REQUIRE_MESSAGE(reachable.size() == static_cast<std::size_t>(n),
            "N=", n, ": shader walk only reached ", reachable.size(),
            " of ", n, " leaves");
    }
}

// All lights coincident: the builder falls back to an index-halving
// split.  Make sure the (right = left + 1) invariant still holds in
// that branch.
TEST_CASE("LightTree: coincident lights still produce a reachable tree") {
    const int n = 16;
    std::vector<LightInput> lights;
    lights.reserve(n);
    for (int i = 0; i < n; ++i) {
        LightInput L;
        L.type = 0;  // Point
        L.pos[0] = 0.0f;  L.pos[1] = 0.0f;  L.pos[2] = 0.0f;
        L.intensity[0] = 1.0f; L.intensity[1] = 1.0f; L.intensity[2] = 1.0f;
        L.dir[0] = 0.0f; L.dir[1] = 1.0f; L.dir[2] = 0.0f;
        lights.push_back(L);
    }
    LightTree tree;
    BuildLightTree(lights, tree);
    CheckStructuralInvariants(tree, static_cast<std::uint32_t>(n));
    const auto reachable = ShaderReachableLightIds(tree);
    CHECK(reachable.size() == static_cast<std::size_t>(n));
}

// Confirms the parent-AABB-contains-children invariant.  The shader's
// importance heuristic relies on cluster AABBs that BOUND every leaf
// under them; a builder that fails this would compute the wrong
// per-cluster distance² estimate and bias selection.
TEST_CASE("LightTree: parent AABB encloses both children's AABBs") {
    auto lights = MakeLights(64, /*seed=*/64u);
    LightTree tree;
    BuildLightTree(lights, tree);

    REQUIRE(!tree.Empty());
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
        const LightTreeNode& p = tree.nodes[i];
        if (p.count != 0u) continue;
        const std::uint32_t l = p.left_first;
        const std::uint32_t r = l + 1u;
        REQUIRE(r < tree.nodes.size());
        const LightTreeNode& cl = tree.nodes[l];
        const LightTreeNode& cr = tree.nodes[r];
        for (int k = 0; k < 3; ++k) {
            CHECK(p.aabb_min[k] <= cl.aabb_min[k] + 1e-5f);
            CHECK(p.aabb_min[k] <= cr.aabb_min[k] + 1e-5f);
            CHECK(p.aabb_max[k] >= cl.aabb_max[k] - 1e-5f);
            CHECK(p.aabb_max[k] >= cr.aabb_max[k] - 1e-5f);
        }
        // Intensity sums.
        CHECK(p.intensity == doctest::Approx(cl.intensity + cr.intensity)
                                .epsilon(1e-4f));
    }
}

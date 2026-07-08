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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
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

// ---------------------------------------------------------------------------
// Selection-PMF correctness (the NEE estimator divides by pick_pdf).
//
// The reachability tests above prove every leaf CAN be reached. They say
// nothing about the PROBABILITY of reaching it -- and PathTrace.slang
// divides each NEE sample by `pick_pdf` (sampleAnalyticLight), so a bug
// that keeps every leaf reachable while mis-accumulating pick_pdf (wrong
// sibling's probability, a broken importance ratio) silently biases all
// many-light direct lighting while every existing assertion stays green.
//
// `LeafPickProbability` mirrors pickLightFromTree's math exactly: at each
// internal node the child importances give p_left = il / (il + ir), and
// pick_pdf is the product of the conditional probabilities along the
// root-to-leaf path. Because each light has exactly one leaf, and exactly
// one path reaches it, pick_pdf(L) IS the selection PMF -- so the
// probabilities over all leaves must sum to exactly 1. That is a
// deterministic invariant; no Monte Carlo (and no flakiness) required.

// Host mirror of PathTrace.slang's lightTreeImportance().
float MirrorImportance(const pt::renderer::LightTreeNode& node,
                       const float hit_pt[3], const float n_face[3]) {
    const float cx = 0.5f * (node.aabb_min[0] + node.aabb_max[0]);
    const float cy = 0.5f * (node.aabb_min[1] + node.aabb_max[1]);
    const float cz = 0.5f * (node.aabb_min[2] + node.aabb_max[2]);
    float to[3] = {cx - hit_pt[0], cy - hit_pt[1], cz - hit_pt[2]};
    const float d2 = std::max(to[0]*to[0] + to[1]*to[1] + to[2]*to[2], 1e-12f);
    const float inv_d = 1.0f / std::sqrt(d2);
    const float dir_to[3] = {to[0]*inv_d, to[1]*inv_d, to[2]*inv_d};

    const float ex = 0.5f * (node.aabb_max[0] - node.aabb_min[0]);
    const float ey = 0.5f * (node.aabb_max[1] - node.aabb_min[1]);
    const float ez = 0.5f * (node.aabb_max[2] - node.aabb_min[2]);
    const float rad2 = std::max(ex*ex + ey*ey + ez*ez, 1e-12f);

    const float falloff = 1.0f / std::max(d2, rad2);

    float orient = 1.0f;
    if (node.cone_cos_half > -0.999f) {
        const float cos_emit = -(node.cone_axis[0]*dir_to[0] +
                                 node.cone_axis[1]*dir_to[1] +
                                 node.cone_axis[2]*dir_to[2]);
        const float t = (cos_emit - (node.cone_cos_half - 0.2f)) / 0.2f;
        orient = std::clamp(t, 0.0f, 1.0f);
    }
    const float recv = std::max(0.0f, n_face[0]*dir_to[0] +
                                      n_face[1]*dir_to[1] +
                                      n_face[2]*dir_to[2]);
    return std::max(node.intensity * falloff * orient * recv, 1e-8f);
}

// Accumulate each leaf's selection probability by walking every
// root-to-leaf path, exactly as pickLightFromTree would sample it.
void CollectLeafPickProbabilities(
        const LightTree& tree, std::uint32_t idx, double p_path,
        const float hit_pt[3], const float n_face[3],
        std::unordered_map<std::uint32_t, double>& out) {
    REQUIRE(idx < tree.nodes.size());
    const auto& n = tree.nodes[idx];
    if (n.count == 1u) {          // leaf
        out[n.left_first] += p_path;
        return;
    }
    const std::uint32_t left  = n.left_first;
    const std::uint32_t right = left + 1u;
    REQUIRE(right < tree.nodes.size());
    const float il = MirrorImportance(tree.nodes[left],  hit_pt, n_face);
    const float ir = MirrorImportance(tree.nodes[right], hit_pt, n_face);
    const float total = il + ir;
    const double p_l = (total > 1e-30f) ? double(il) / double(total) : 0.5;
    CollectLeafPickProbabilities(tree, left,  p_path * p_l,        hit_pt, n_face, out);
    CollectLeafPickProbabilities(tree, right, p_path * (1.0 - p_l), hit_pt, n_face, out);
}

TEST_CASE("LightTree: selection PMF sums to 1 and reaches every leaf") {
    // A shade point off to one side with a tilted normal, so the
    // importance terms (falloff, orientation, receiver cosine) all
    // actually vary between clusters rather than collapsing to a
    // uniform split.
    const float hit_pt[3] = {2.5f, 1.0f, -3.0f};
    float n_face[3] = {0.3f, 0.9f, 0.2f};
    const float inv = 1.0f / std::sqrt(n_face[0]*n_face[0] +
                                       n_face[1]*n_face[1] +
                                       n_face[2]*n_face[2]);
    n_face[0] *= inv; n_face[1] *= inv; n_face[2] *= inv;

    for (int n : {1, 2, 3, 4, 7, 16, 64, 200}) {
        auto lights = MakeLights(n, static_cast<std::uint32_t>(n) * 7u + 1u);
        LightTree tree;
        BuildLightTree(lights, tree);

        std::unordered_map<std::uint32_t, double> pmf;
        CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                     hit_pt, n_face, pmf);

        // Every light is reachable with NON-ZERO probability. (The
        // epsilon floor in lightTreeImportance exists precisely so no
        // branch becomes deselectable; a regression that lets an
        // importance hit exactly 0 would strand a whole subtree.)
        CHECK_MESSAGE(pmf.size() == static_cast<std::size_t>(n),
            "N=", n, ": PMF covers ", pmf.size(), " of ", n, " lights");
        for (const auto& [id, p] : pmf) {
            CHECK_MESSAGE(p > 0.0, "N=", n, ": light ", id,
                          " has zero selection probability");
        }

        // The estimator divides by pick_pdf, so the PMF must normalise.
        double sum = 0.0;
        for (const auto& [id, p] : pmf) sum += p;
        CHECK_MESSAGE(sum == doctest::Approx(1.0).epsilon(1e-9),
            "N=", n, ": selection PMF sums to ", sum, ", not 1.0 -- NEE "
            "samples divided by pick_pdf would be biased");
    }
}

TEST_CASE("LightTree: brighter cluster is picked more often (importance is used)") {
    // Two lights equidistant from the shade point, one 100x brighter.
    // A picker that ignored importance (or inverted the ratio) would
    // split 50/50 or favour the dim light -- both caught here, while
    // the reachability tests would stay green.
    std::vector<LightInput> lights(2);
    lights[0].type = 0u;                                  // point
    lights[0].pos[0] = -4.0f; lights[0].pos[1] = 3.0f;
    lights[0].intensity[0] = lights[0].intensity[1] = lights[0].intensity[2] = 1.0f;
    lights[1].type = 0u;
    lights[1].pos[0] =  4.0f; lights[1].pos[1] = 3.0f;
    lights[1].intensity[0] = lights[1].intensity[1] = lights[1].intensity[2] = 100.0f;

    LightTree tree;
    BuildLightTree(lights, tree);

    const float hit_pt[3] = {0.0f, 0.0f, 0.0f};
    const float n_face[3] = {0.0f, 1.0f, 0.0f};   // facing both equally
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0, hit_pt, n_face, pmf);

    REQUIRE(pmf.count(0u) == 1u);
    REQUIRE(pmf.count(1u) == 1u);
    CHECK_MESSAGE(pmf[1u] > pmf[0u],
        "the 100x brighter light (id 1, p=", pmf[1u], ") must be picked more "
        "often than the dim one (id 0, p=", pmf[0u], ") -- importance is "
        "either ignored or inverted");
    // Symmetric geometry => the split should track the intensity ratio.
    CHECK(pmf[1u] / (pmf[0u] + pmf[1u]) > 0.9);
    CHECK(pmf[0u] + pmf[1u] == doctest::Approx(1.0).epsilon(1e-9));
}

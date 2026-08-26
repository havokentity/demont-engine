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
// Issue #250: the layout fix above left a SECOND way for a subtree to
// go dark.  lightTreeImportance evaluated the receiver cosine and the
// emission-cone test on the cluster-CENTRE direction, then floored the
// product at an absolute 1e-8.  A cluster straddling the receiver's
// tangent plane (or its own cone boundary) scored that floor even while
// holding the brightest emitter in the scene, and float32 rounds
// il / (il + ir) to exactly 1.0 once the sibling is 2^24 times larger --
// so `1.0 - p_l` was exactly 0 and the subtree became UNREACHABLE.
// Reachability of the leaf in the tree says nothing about that: the
// walk still visits it, with probability zero.
//
// Tests below pin:
//   - The contiguous-children layout invariant (right = left + 1)
//     across N=1..256 + larger samples (200/500/1000).  Pre-fix the
//     test FAILED for N >= 4.
//   - Every input light gets exactly ONE reachable leaf node, and the
//     leaf's `left_first` rejects no light id from [0, N).
//   - Tree shape: N leaves + (N-1) internal = 2N - 1 nodes for N >= 1.
//   - All nodes have a finite AABB and a finite cone (no NaN).
//   - The selection PMF normalises, AND every light that can actually
//     reach the shade point has strictly positive probability (#250) --
//     across straddling receivers, huge importance ratios, and spot
//     clusters straddling their emission cone.
//
// Pure-CPU + deterministic.  Links pt_renderer.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "renderer/LightTree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
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
// `MirrorImportance` / `CollectLeafPickProbabilities` mirror
// pickLightFromTree's math exactly, IN FLOAT32, because that is where
// #250 lived: with an absolute 1e-8 importance floor a child whose true
// importance was zero still scored 1e-8, and float32 rounds
// il / (il + ir) to exactly 1.0 as soon as the sibling is more than 2^24
// times larger -- so `1.0 - p_l` was exactly 0 and that subtree's lights
// became unsamplable. Computing the conditionals in double would hide
// the collapse the shader actually suffers.
//
// pick_pdf is the product of the conditional probabilities along the
// root-to-leaf path. Because each light has exactly one leaf, and exactly
// one path reaches it, pick_pdf(L) IS the selection PMF -- so the
// probabilities over all leaves must sum to exactly 1. That is a
// deterministic invariant; no Monte Carlo (and no flakiness) required.

// randf()'s lattice (PathTraceMath.slang: seed & 0x00FFFFFF over 2^24).
// pickLightFromTree quantises each split onto it so pick_pdf equals the
// probability the sampler can actually realise.
constexpr float kPickLattice = 16777216.0f;          // 2^24
constexpr float kPickStep    = 1.0f / 16777216.0f;   // 2^-24 (exact)

// Host mirror of PathTrace.slang's lightTreeImportance().
float MirrorImportance(const pt::renderer::LightTreeNode& node,
                       const float hit_pt[3], const float n_face[3]) {
    const float cx = 0.5f * (node.aabb_min[0] + node.aabb_max[0]);
    const float cy = 0.5f * (node.aabb_min[1] + node.aabb_max[1]);
    const float cz = 0.5f * (node.aabb_min[2] + node.aabb_max[2]);
    float to[3] = {cx - hit_pt[0], cy - hit_pt[1], cz - hit_pt[2]};
    const float d2 = std::max(to[0]*to[0] + to[1]*to[1] + to[2]*to[2], 1e-12f);
    const float d  = std::sqrt(d2);
    const float dir_to[3] = {to[0]/d, to[1]/d, to[2]/d};

    const float ex = 0.5f * (node.aabb_max[0] - node.aabb_min[0]);
    const float ey = 0.5f * (node.aabb_max[1] - node.aabb_min[1]);
    const float ez = 0.5f * (node.aabb_max[2] - node.aabb_min[2]);
    const float rad2 = std::max(ex*ex + ey*ey + ez*ez, 1e-12f);

    const float falloff = 1.0f / std::max(d2, rad2);

    // theta_u: half-angle the cluster's bounding sphere subtends.
    const float sin_u = std::min(1.0f, std::sqrt(rad2) / d);
    const float cos_u = std::sqrt(std::max(0.0f, 1.0f - sin_u * sin_u));

    // Orientation, widened by theta_u (alpha = theta_o + theta_u).
    const float cos_emit = -(node.cone_axis[0]*dir_to[0] +
                             node.cone_axis[1]*dir_to[1] +
                             node.cone_axis[2]*dir_to[2]);
    const float cos_o = node.cone_cos_half;
    const float sin_o = std::sqrt(std::max(0.0f, 1.0f - cos_o * cos_o));
    const float cos_a = cos_o * cos_u - sin_o * sin_u;
    const float sin_a = sin_o * cos_u + cos_o * sin_u;
    float orient;
    if (sin_a < 0.0f || cos_emit >= cos_a) {
        orient = 1.0f;
    } else {
        const float sin_emit =
            std::sqrt(std::max(0.0f, 1.0f - cos_emit * cos_emit));
        orient = std::max(0.0f, cos_emit * cos_a + sin_emit * sin_a);
    }

    // Receiver Lambert bound, widened the same way.
    const float cos_i = n_face[0]*dir_to[0] + n_face[1]*dir_to[1] +
                        n_face[2]*dir_to[2];
    float recv;
    if (cos_i >= cos_u) {
        recv = 1.0f;
    } else {
        const float sin_i = std::sqrt(std::max(0.0f, 1.0f - cos_i * cos_i));
        recv = std::max(0.0f, cos_i * cos_u + sin_i * sin_u);
    }

    return node.intensity * falloff * orient * recv;
}

// Host mirror of pickLightFromTree's per-level split probability.
float MirrorSplit(float il, float ir) {
    if (il > 0.0f && ir > 0.0f) {
        return std::clamp(std::ceil((il / (il + ir)) * kPickLattice),
                          1.0f, kPickLattice - 1.0f) * kPickStep;
    }
    if (il > 0.0f) return 1.0f;   // right subtree provably contributes nothing
    if (ir > 0.0f) return 0.0f;   // left subtree provably contributes nothing
    return 0.5f;                  // neither side can reach the shade point
}

// Accumulate each leaf's selection probability by walking every
// root-to-leaf path, exactly as pickLightFromTree would sample it. The
// conditionals are float32 (see above); the path product is carried in
// double so the sum-to-1 assertion isn't testing float accumulation order.
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
    const float p_l = MirrorSplit(il, ir);
    CollectLeafPickProbabilities(tree, left,  p_path * double(p_l),
                                 hit_pt, n_face, out);
    CollectLeafPickProbabilities(tree, right, p_path * double(1.0f - p_l),
                                 hit_pt, n_face, out);
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

        // Every leaf is VISITED by the walk. (Whether it may carry
        // probability zero is a question about contribution, not
        // reachability -- MakeLights spreads emitters all round the
        // origin, so at this hit point some are genuinely behind the
        // tangent plane or aimed away, and zero is the correct answer
        // for those. The "every contributing light is samplable" cases
        // below pin that distinction.)
        CHECK_MESSAGE(pmf.size() == static_cast<std::size_t>(n),
            "N=", n, ": PMF covers ", pmf.size(), " of ", n, " lights");

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

// ---------------------------------------------------------------------------
// #250: support condition of the estimator -- p(L) > 0 wherever the
// integrand is non-zero.
//
// A PMF that sums to 1 and covers every leaf is NOT enough: the sum stays
// 1 while an entire subtree's share migrates to its sibling.
// `radiance += Le * brdf * n_dot_l / ls.pdf` does not down-weight a light
// it never picks, it DELETES it. Every case below is a shade point where
// the pre-fix picker handed a light with a genuinely non-zero contribution
// a probability of exactly 0.
//
// Note these cases must ALSO be honest in the other direction: a light
// that truly cannot contribute (behind the tangent plane, outside a spot's
// outer cone) is *supposed* to get zero, and the fix must not paper that
// over with a floor. So each helper compares against the exact support of
// the corresponding branch of sampleAnalyticLight.

LightInput MakePointLight(float x, float y, float z, float intensity) {
    LightInput L;
    L.type = 0u;
    L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
    L.intensity[0] = L.intensity[1] = L.intensity[2] = intensity;
    return L;
}

LightInput MakeSpotLight(float x, float y, float z,
                         float dx, float dy, float dz,
                         float outer_deg, float intensity) {
    LightInput L;
    L.type = 1u;
    L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
    const float len = std::sqrt(dx*dx + dy*dy + dz*dz);
    L.dir[0] = dx/len; L.dir[1] = dy/len; L.dir[2] = dz/len;
    L.cos_outer = std::cos(outer_deg * 3.14159265358979f / 180.0f);
    L.intensity[0] = L.intensity[1] = L.intensity[2] = intensity;
    return L;
}

// Unshadowed contribution of one light at (hit_pt, n_face) -- the exact
// quantity sampleAnalyticLight estimates, with the exact same support.
//   POINT: I * n_dot_l / d^2, zero at or behind the tangent plane.
//   SPOT : additionally zero at or outside the outer cone (the shader's
//          LIGHT_SPOT falloff is 0 for cos_dir <= cos_outer).
double LightContribution(const LightInput& L,
                         const float hit_pt[3], const float n_face[3]) {
    const double to[3] = {double(L.pos[0]) - hit_pt[0],
                          double(L.pos[1]) - hit_pt[1],
                          double(L.pos[2]) - hit_pt[2]};
    const double d2 = to[0]*to[0] + to[1]*to[1] + to[2]*to[2];
    if (d2 <= 0.0) return 0.0;
    const double d = std::sqrt(d2);
    const double wi[3] = {to[0]/d, to[1]/d, to[2]/d};
    const double n_dot_l = n_face[0]*wi[0] + n_face[1]*wi[1] + n_face[2]*wi[2];
    if (n_dot_l <= 0.0) return 0.0;

    double falloff = 1.0;
    if (L.type == 1u) {                       // spot
        const double cos_dir = -(L.dir[0]*wi[0] + L.dir[1]*wi[1] +
                                 L.dir[2]*wi[2]);
        if (cos_dir <= double(L.cos_outer)) return 0.0;
        const double cos_in = std::cos(std::acos(double(L.cos_outer)) * 0.5);
        falloff = (cos_dir >= cos_in)
                    ? 1.0
                    : (cos_dir - L.cos_outer) /
                      std::max(cos_in - double(L.cos_outer), 1e-6);
    }
    return double(L.intensity[1]) * falloff * n_dot_l / d2;
}

// Fraction of the true direct-lighting energy at (hit_pt, n_face) that the
// selection PMF assigns probability ZERO -- energy the estimator can never
// draw a sample for, at any spp.
double UnsamplableEnergyFraction(const std::vector<LightInput>& lights,
                                 const float hit_pt[3], const float n_face[3],
                                 int* out_lost_lights = nullptr) {
    LightTree tree;
    BuildLightTree(lights, tree);
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                 hit_pt, n_face, pmf);
    double total = 0.0, lost = 0.0;
    int lost_lights = 0;
    for (std::uint32_t i = 0; i < lights.size(); ++i) {
        const double c = LightContribution(lights[i], hit_pt, n_face);
        total += c;
        const auto it = pmf.find(i);
        const double p = (it == pmf.end()) ? 0.0 : it->second;
        if (c > 0.0 && p <= 0.0) { lost += c; ++lost_lights; }
    }
    if (out_lost_lights) *out_lost_lights = lost_lights;
    return (total > 0.0) ? (lost / total) : 0.0;
}

double PmfSum(const std::vector<LightInput>& lights,
              const float hit_pt[3], const float n_face[3]) {
    LightTree tree;
    BuildLightTree(lights, tree);
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                 hit_pt, n_face, pmf);
    double sum = 0.0;
    for (const auto& [id, p] : pmf) sum += p;
    return sum;
}

TEST_CASE("LightTree: every light reaching the shade point is samplable (#250)") {
    // The four-corner light rig from #177, read at receivers the old
    // single hard-coded shade point never exercised. On the vertical and
    // downward-facing receivers the median split puts visible AND hidden
    // lights inside the SAME cluster, so the cluster centre lands on or
    // past the tangent plane -- which used to zero the whole subtree.
    const std::vector<LightInput> corners = {
        MakePointLight(-5.0f, 2.5f, -5.0f, 40.0f),
        MakePointLight( 5.0f, 2.5f, -5.0f, 40.0f),
        MakePointLight(-5.0f, 2.5f,  5.0f, 40.0f),
        MakePointLight( 5.0f, 2.5f,  5.0f, 40.0f),
    };

    struct Receiver { const char* name; float p[3]; float n[3]; };
    const Receiver receivers[] = {
        {"floor centre",     {0.0f, 0.0f,  0.0f}, { 0.0f, 1.0f,  0.0f}},
        {"floor corner",     {4.0f, 0.0f,  4.0f}, { 0.0f, 1.0f,  0.0f}},
        {"wall facing -Z",   {0.0f, 1.0f,  0.0f}, { 0.0f, 0.0f, -1.0f}},
        {"wall facing +X",   {0.0f, 1.0f,  0.0f}, { 1.0f, 0.0f,  0.0f}},
        {"tilted 45 deg",    {1.0f, 0.5f, -1.0f}, { 0.7071f, 0.7071f, 0.0f}},
        {"downward (-Y)",    {0.0f, 1.0f,  0.0f}, { 0.0f,-1.0f,  0.0f}},
    };

    for (const auto& r : receivers) {
        const std::string name(r.name);   // doctest prints a bare const char*
                                          // member as a pointer; bind it.
        int lost_lights = 0;
        const double lost =
            UnsamplableEnergyFraction(corners, r.p, r.n, &lost_lights);
        CHECK_MESSAGE(lost_lights == 0,
            "4-corner rig, receiver '", name, "': ", lost_lights,
            " light(s) with a non-zero contribution have selection "
            "probability ZERO (", 100.0 * lost, "% of the direct lighting "
            "at this point is unsamplable at ANY spp) -- #250");
        CHECK_MESSAGE(PmfSum(corners, r.p, r.n) ==
                          doctest::Approx(1.0).epsilon(1e-9),
            "receiver '", name, "': PMF does not normalise");
    }
}

TEST_CASE("LightTree: lights on both sides of the receiver plane (#250)") {
    // The minimal reproduction from #250. Two lights above the floor,
    // two below it; the median split hands each cluster one of each, so
    // both cluster centres sit near the tangent plane.
    //
    // Pre-fix node dump at this shade point:
    //   node 0: il = 3.567864e-01 (centre.y = +0.927, recv = 0.4368)
    //           ir = 1.000000e-08 (centre.y = -0.167, recv = 0.0000)
    //           il + ir == il exactly  ->  p_l = 1.0,  1 - p_l = 0.0
    // The zeroed subtree holds light 1 -- ABOVE the floor, unshadowed,
    // the largest importance in the tree, ~98% of the direct lighting
    // here. On GPU (Metal, 1024 spp) that cost 76.1% of the whole
    // frame's direct light.
    const std::vector<LightInput> straddle = {
        MakePointLight(-1.554f,  2.865f,  -5.255f, 0.574f),   // above
        MakePointLight(-2.255f,  0.786f,  -1.897f, 2.312f),   // above, dominant
        MakePointLight(-7.926f, -1.119f,  11.406f, 1.044f),   // below
        MakePointLight( 0.184f, -1.010f,  -2.413f, 4.765f),   // below
    };
    const float hit_pt[3] = {-2.136f, 0.0f, -2.592f};
    const float n_face[3] = { 0.0f, 1.0f,  0.0f};

    LightTree tree;
    BuildLightTree(straddle, tree);
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                 hit_pt, n_face, pmf);

    CHECK_MESSAGE(pmf[1u] > 0.0,
        "the dominant light (id 1, at y=+0.786, carrying ~98% of the direct "
        "lighting at this point) has selection probability ", pmf[1u],
        " -- its cluster was written off because the cluster CENTRE sits "
        "below the tangent plane");
    CHECK(pmf[0u] > 0.0);
    // The two below-floor lights genuinely cannot contribute; zero is the
    // right answer for them and the fix must not floor it away.
    CHECK(LightContribution(straddle[2], hit_pt, n_face) == 0.0);
    CHECK(LightContribution(straddle[3], hit_pt, n_face) == 0.0);

    int lost = 0;
    CHECK(UnsamplableEnergyFraction(straddle, hit_pt, n_face, &lost) == 0.0);
    CHECK(lost == 0);
    CHECK(PmfSum(straddle, hit_pt, n_face) ==
              doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("LightTree: spot cluster straddling its emission cone (#250)") {
    // The orientation term had the identical centre-point shape as the
    // receiver cosine, so it fails the same way: a cluster of spots whose
    // COMBINED cone misses the shade point from the cluster centre, while
    // an individual member aims straight at it.
    //
    // Pre-fix PMF at this receiver: [1.0, 0.0, 0.0, 0.0, 1.9e-07] against
    // true contributions [0, 4.08e-02, 0, 0, 0] -- the only spot that lights
    // the point is the one that can never be picked, and all the
    // probability sits on a spot whose cone the point is outside of.
    // 100% of the direct lighting at that receiver, gone.
    const std::vector<LightInput> spots = {
        MakeSpotLight( -8.305f, 5.192f,  2.873f,
                        0.1749f, -0.9028f, -0.3928f, 21.0f, 6.190f),
        MakeSpotLight( -6.158f, 6.147f,  0.689f,
                        0.0759f, -0.9181f, -0.3889f, 42.5f, 5.641f),
        MakeSpotLight(  5.563f, 3.426f,  1.977f,
                        0.4235f, -0.8860f, -0.1889f, 14.7f, 6.078f),
        MakeSpotLight(  7.650f, 5.935f, -5.186f,
                       -0.1138f, -0.9899f, -0.0845f, 14.0f, 2.615f),
        MakeSpotLight(-11.909f, 2.906f, -6.717f,
                        0.3878f, -0.9055f,  0.1723f, 12.2f, 5.967f),
    };
    const float hit_pt[3] = {-9.802f, 0.0f, -2.206f};
    const float n_face[3] = { 0.0f, 1.0f,  0.0f};

    // Spot 1 is the only one whose outer cone actually covers this point.
    REQUIRE(LightContribution(spots[1], hit_pt, n_face) > 0.0);
    for (int i : {0, 2, 3, 4}) {
        REQUIRE(LightContribution(spots[static_cast<std::size_t>(i)],
                                  hit_pt, n_face) == 0.0);
    }

    LightTree tree;
    BuildLightTree(spots, tree);
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                 hit_pt, n_face, pmf);
    CHECK_MESSAGE(pmf[1u] > 0.0,
        "the only spot lighting this point has selection probability ",
        pmf[1u], " -- the emission-cone term was evaluated on the cluster "
        "centre instead of being widened by the angle the cluster subtends");

    int lost = 0;
    CHECK(UnsamplableEnergyFraction(spots, hit_pt, n_face, &lost) == 0.0);
    CHECK(lost == 0);
    CHECK(PmfSum(spots, hit_pt, n_face) == doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("LightTree: a huge importance ratio never starves the dim subtree") {
    // One light almost touching the shade point next to two 200 m out.
    // The falloff ratio is ~1e8, past float32's 2^24, so il / (il + ir)
    // rounds to exactly 1.0 and the far subtree's probability used to
    // become exactly 0. Those lights contribute little, but "little" and
    // "nothing" are different images -- and this collapse needs no
    // straddling geometry at all, just dynamic range.
    const std::vector<LightInput> spread = {
        MakePointLight(   0.0f, 0.05f, 0.0f, 1.0f),
        MakePointLight(   0.0f, 2.50f, 0.0f, 1.0f),
        MakePointLight( 200.0f, 2.50f, 0.0f, 1.0f),
        MakePointLight(-200.0f, 2.50f, 0.0f, 1.0f),
    };
    const float hit_pt[3] = {0.0f, 0.0f, 0.0f};
    const float n_face[3] = {0.0f, 1.0f, 0.0f};   // every light is visible

    LightTree tree;
    BuildLightTree(spread, tree);
    std::unordered_map<std::uint32_t, double> pmf;
    CollectLeafPickProbabilities(tree, tree.root_index, 1.0,
                                 hit_pt, n_face, pmf);
    for (std::uint32_t i = 0; i < spread.size(); ++i) {
        REQUIRE(LightContribution(spread[i], hit_pt, n_face) > 0.0);
        CHECK_MESSAGE(pmf[i] > 0.0,
            "light ", i, " is visible (n_dot_l > 0) but has selection "
            "probability 0 -- float32 collapsed the split");
    }
    CHECK(PmfSum(spread, hit_pt, n_face) == doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("LightTree: no direct-lighting energy is unsamplable (random sweep)") {
    // Randomised sweep over straddling geometry: point lights above AND
    // below the receiver plane, read at floor / wall / tilted normals, plus
    // a spot-light pass for the emission-cone term. Pre-fix these found
    // shade points losing up to 100% of their direct lighting.
    std::mt19937 rng(20260826u);
    std::uniform_real_distribution<float> pos_d(-12.0f, 12.0f);
    std::uniform_real_distribution<float> hi_d(-3.0f, 3.0f);
    std::uniform_real_distribution<float> up_d(2.0f, 7.0f);
    std::uniform_real_distribution<float> int_d(0.5f, 6.0f);
    std::uniform_real_distribution<float> ax_d(-0.6f, 0.6f);
    std::uniform_real_distribution<float> out_d(12.0f, 45.0f);

    const float normals[4][3] = {
        {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
        {0.57735f, 0.57735f, 0.57735f},
    };

    double worst_pt = 0.0;
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 2 + (trial % 40);
        std::vector<LightInput> lights;
        lights.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            lights.push_back(MakePointLight(pos_d(rng), hi_d(rng),
                                            pos_d(rng), int_d(rng)));
        }
        for (const auto& nf : normals) {
            const float hit_pt[3] = {pos_d(rng) * 0.5f, 0.0f,
                                     pos_d(rng) * 0.5f};
            worst_pt = std::max(worst_pt,
                                UnsamplableEnergyFraction(lights, hit_pt, nf));
        }
    }
    CHECK_MESSAGE(worst_pt == 0.0,
        "point-light sweep: worst case leaves ", 100.0 * worst_pt,
        "% of a shade point's direct lighting with zero selection "
        "probability -- the tree integrates a strict subset of the lights");

    double worst_spot = 0.0;
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 4 + (trial % 5);
        std::vector<LightInput> lights;
        lights.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            lights.push_back(MakeSpotLight(pos_d(rng), up_d(rng), pos_d(rng),
                                           ax_d(rng), -1.0f, ax_d(rng),
                                           out_d(rng), int_d(rng)));
        }
        const float nf[3] = {0.0f, 1.0f, 0.0f};
        for (int s = 0; s < 3; ++s) {
            const float hit_pt[3] = {pos_d(rng), 0.0f, pos_d(rng)};
            worst_spot = std::max(worst_spot,
                                  UnsamplableEnergyFraction(lights, hit_pt, nf));
        }
    }
    CHECK_MESSAGE(worst_spot == 0.0,
        "spot-light sweep: worst case leaves ", 100.0 * worst_spot,
        "% of a shade point's direct lighting with zero selection "
        "probability -- the emission-cone term is not a bound");
}

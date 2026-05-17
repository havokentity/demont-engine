// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::renderer::AnalyticBvh (issue #67 / Phase 3 of #47).
//
// The AnalyticBvh class only exposes Build() + the flat node array + the
// permuted prim-id sequence; the actual traversal lives in the path-trace
// shader (shaders/PathTrace.slang:497-525). For host-side testing we
// mirror that traversal exactly here so the BVH path is exercisable
// without a GPU.
//
// Test strategy (Tier 2.1 of `Raytracer Plan/TESTING_STRATEGY.md`):
//
//   1. Build the BVH over a known set of analytic sphere primitives.
//   2. Build a naive linear-scan reference over the same primitives in
//      their original (unpermuted) order.
//   3. Fire K=1000 deterministic random rays. The BVH-traced closest hit
//      and the naive-scan closest hit must match in (prim_id, t,
//      normal) within float epsilon -- any deviation indicates either
//      a builder bug (dropped prim, bad AABB) or a traversal bug (miss
//      a child, AABB-cull bug).
//
// Determinism note: the random rays come from an integer-domain xorshift
// PRNG that maps to floats via a fixed multiplier. No std::random, no
// time-based seeding, no trig chains -- the same seed produces the same
// rays across Mac/Win/Linux + clang/gcc/msvc + Debug/Release. The hit
// equivalence assertion uses an absolute epsilon (`kHitEps = 1e-4f`)
// scaled to metre-scale primitives; both the BVH and the naive paths
// run the same sphere-intersect routine here so any deviation is
// strictly traversal-coverage-not-builder-coverage.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/AnalyticBvh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

using pt::renderer::AnalyticBvh;
using pt::renderer::BvhNode;
using pt::renderer::BvhPrim;

namespace {

// --- Closest-hit record shared by BVH and naive paths --------------------
struct Hit {
    float         t        = std::numeric_limits<float>::infinity();
    float         normal[3]{0.f, 0.f, 0.f};
    std::uint32_t prim_id  = std::numeric_limits<std::uint32_t>::max();
    bool          hit      = false;
};

// --- Deterministic xorshift32 PRNG ----------------------------------------
// Self-contained and platform-independent. No std::random (whose
// implementations differ across libc++/libstdc++/MSVC STL), no trig.
struct XorShift32 {
    std::uint32_t state;
    explicit XorShift32(std::uint32_t seed) : state(seed ? seed : 0xdeadbeefu) {}
    std::uint32_t next() {
        std::uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    // [0, 1)
    float unitf() {
        // 24 bits of mantissa precision; matches std::generate_canonical
        // shape but is implementation-independent.
        return float(next() >> 8) * (1.0f / float(1u << 24));
    }
    // [lo, hi)
    float rangef(float lo, float hi) { return lo + (hi - lo) * unitf(); }
};

// --- Ray-sphere intersect (closest positive root) -------------------------
// Used identically by both the BVH path and the naive scan, so the test
// is strictly comparing *which* prims got tested, not how a single
// intersect was computed. tmin guards against self-intersection / origin
// inside the sphere.
bool IntersectSphere(const float ro[3], const float rd[3],
                     const BvhPrim& p, float tmin,
                     float& out_t, float (&out_n)[3]) {
    float oc[3] = {ro[0] - p.center[0],
                   ro[1] - p.center[1],
                   ro[2] - p.center[2]};
    const float b = oc[0]*rd[0] + oc[1]*rd[1] + oc[2]*rd[2];
    const float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2]
                  - p.radius * p.radius;
    const float disc = b*b - c;
    if (disc < 0.0f) return false;
    const float sd = std::sqrt(disc);
    float t = -b - sd;
    if (t < tmin) t = -b + sd;
    if (t < tmin) return false;
    out_t = t;
    const float hx = ro[0] + rd[0]*t - p.center[0];
    const float hy = ro[1] + rd[1]*t - p.center[1];
    const float hz = ro[2] + rd[2]*t - p.center[2];
    const float invr = (p.radius > 0.f) ? (1.0f / p.radius) : 1.0f;
    out_n[0] = hx * invr;
    out_n[1] = hy * invr;
    out_n[2] = hz * invr;
    return true;
}

// --- Naive linear scan over all prims -------------------------------------
// Reference truth: visit every prim, take the closest positive-t hit.
Hit NaiveTrace(const std::vector<BvhPrim>& prims,
               const float ro[3], const float rd[3], float tmin) {
    Hit h;
    for (std::size_t i = 0; i < prims.size(); ++i) {
        float t; float n[3];
        if (IntersectSphere(ro, rd, prims[i], tmin, t, n) && t < h.t) {
            h.t       = t;
            h.normal[0] = n[0]; h.normal[1] = n[1]; h.normal[2] = n[2];
            h.prim_id = prims[i].prim_id;
            h.hit     = true;
        }
    }
    return h;
}

// --- Ray vs AABB slab intersect -------------------------------------------
// Returns true if the ray intersects the slab box on [tmin, tmax]; tmax
// is the current closest-hit distance, so a box farther than `tmax`
// gets culled. Mirrors `intersectAabb` in shaders/PathTrace.slang.
bool IntersectAabb(const float ro[3], const float rd_inv[3],
                   const float bmn[3], const float bmx[3], float tmax) {
    float t0 = 0.0f;
    float t1 = tmax;
    for (int i = 0; i < 3; ++i) {
        float lo = (bmn[i] - ro[i]) * rd_inv[i];
        float hi = (bmx[i] - ro[i]) * rd_inv[i];
        if (lo > hi) std::swap(lo, hi);
        if (lo > t0) t0 = lo;
        if (hi < t1) t1 = hi;
        if (t0 > t1) return false;
    }
    return true;
}

// --- Host-side BVH traversal ----------------------------------------------
// Mirrors shaders/PathTrace.slang:497-525. Iterative stack-based;
// stack depth caps at 64 (well above any reasonable BVH for our prim
// counts -- log2(N_test) for N_test < 1000 is < 10). The shader-side
// host permutes its GPU primitive buffer to match leaf order, so
// `primitives[leaf_first + k]` is the right entry on the GPU side.
// Here we instead carry the original array through unchanged and
// indirect into it via `permuted_prim_ids[leaf_first + k]`, which the
// BVH populated with each prim's `BvhPrim::prim_id` value. For the
// test convention `prim_id == array_index_in_original_prims`, that
// gives the right entry too. Consumers that use non-index prim_ids
// would need to re-permute their array on upload (which is what the
// renderer does); the test scenes deliberately use prim_id==index so
// the trace round-trip is exercised end-to-end against the naive
// reference.
Hit BvhTrace(const std::vector<BvhNode>&        nodes,
             const std::vector<std::uint32_t>&  permuted,
             const std::vector<BvhPrim>&        original_prims,
             const float ro[3], const float rd[3], float tmin) {
    Hit h;
    if (nodes.empty()) return h;

    // Pre-reciprocal rd; guard zero components like the shader.
    float rd_inv[3];
    for (int i = 0; i < 3; ++i) {
        float d = rd[i];
        if (std::fabs(d) < 1e-30f) d = (d < 0.f) ? -1e-30f : 1e-30f;
        rd_inv[i] = 1.0f / d;
    }

    std::uint32_t stack[64];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        const std::uint32_t idx = stack[--sp];
        const BvhNode& n = nodes[idx];
        if (!IntersectAabb(ro, rd_inv, n.aabb_min, n.aabb_max, h.t)) continue;
        if (n.count > 0u) {
            // Leaf: test the contiguous prim range.
            for (std::uint32_t k = 0; k < n.count; ++k) {
                const std::uint32_t prim_id = permuted[n.left_first + k];
                const BvhPrim& p = original_prims[prim_id];
                float t; float nrm[3];
                if (IntersectSphere(ro, rd, p, tmin, t, nrm) && t < h.t) {
                    h.t = t;
                    h.normal[0] = nrm[0]; h.normal[1] = nrm[1]; h.normal[2] = nrm[2];
                    h.prim_id = p.prim_id;
                    h.hit = true;
                }
            }
        } else {
            if (sp + 2 <= 64) {
                stack[sp++] = n.left_first;
                stack[sp++] = n.left_first + 1u;
            }
        }
    }
    return h;
}

// --- Scene generators -----------------------------------------------------
// Build a deterministic metre-scale scene of spheres scattered through a
// ~20m cube. Sphere radii are 0.2..1.0 m. Center distribution is
// uniform over [-10, 10] m so the BVH has a substantive build job.
// The prim_id is just the index into the returned vector -- that's the
// "naive-scan original ordering" the BVH must round-trip through
// permuted_[].
std::vector<BvhPrim> MakeRandomSphereScene(std::uint32_t n,
                                            std::uint32_t seed) {
    XorShift32 rng(seed);
    std::vector<BvhPrim> prims;
    prims.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        BvhPrim p{};
        p.center[0] = rng.rangef(-10.0f, 10.0f);
        p.center[1] = rng.rangef(-10.0f, 10.0f);
        p.center[2] = rng.rangef(-10.0f, 10.0f);
        p.radius    = rng.rangef(0.2f, 1.0f);
        p.prim_id   = i;
        prims.push_back(p);
    }
    return prims;
}

// Hit-equivalence epsilon. Metre-scale geometry, ray distances bounded
// by ~30m, so 1e-4m == 0.1mm of tolerance per component. The BVH and
// naive paths share IntersectSphere() so a hit on the same prim should
// agree to within bit-exact reproducibility on the same compile, but
// across compilers/optimisation levels a couple of float ULP can
// creep in -- 1e-4 absorbs that.
constexpr float kHitEps = 1e-4f;

// Compare two Hit records for equivalence within float epsilon.
// Reported as separate CHECKs so doctest tells you exactly which
// component disagreed (t / normal / prim_id / hit-flag) instead of a
// black-box "they differ".
void CheckHitsEqual(const Hit& bvh, const Hit& naive, std::uint32_t ray_idx) {
    CAPTURE(ray_idx);
    CHECK(bvh.hit == naive.hit);
    if (!bvh.hit || !naive.hit) return;
    CHECK(bvh.prim_id == naive.prim_id);
    CHECK(bvh.t == doctest::Approx(naive.t).epsilon(kHitEps));
    CHECK(bvh.normal[0] == doctest::Approx(naive.normal[0]).epsilon(kHitEps));
    CHECK(bvh.normal[1] == doctest::Approx(naive.normal[1]).epsilon(kHitEps));
    CHECK(bvh.normal[2] == doctest::Approx(naive.normal[2]).epsilon(kHitEps));
}

// Generate a deterministic ray for fuzz testing. Origins are biased
// outside the scene (radius ~18m sphere) so most rays enter from
// outside and traverse a meaningful path through the BVH. Directions
// point from the origin toward a random target inside the scene (a
// random point in a ~14m cube), so the rays actually intersect the
// occupied volume instead of being uniformly random over the unit
// sphere (which would mostly miss). The direction is computed by
// subtract + normalise -- platform-stable, no trig.
void MakeRay(XorShift32& rng, float ro[3], float rd[3]) {
    // Origin: pick on a "shell" around the scene by sampling [-20,20]
    // then pushing outward to >=15m magnitude. Mirrors how a typical
    // camera shoots in from outside-the-scene during a test render.
    do {
        ro[0] = rng.rangef(-25.0f, 25.0f);
        ro[1] = rng.rangef(-25.0f, 25.0f);
        ro[2] = rng.rangef(-25.0f, 25.0f);
    } while (ro[0]*ro[0] + ro[1]*ro[1] + ro[2]*ro[2] < 15.0f * 15.0f);

    // Aim at a random target inside the scene.
    const float tx = rng.rangef(-14.0f, 14.0f);
    const float ty = rng.rangef(-14.0f, 14.0f);
    const float tz = rng.rangef(-14.0f, 14.0f);
    float dx = tx - ro[0];
    float dy = ty - ro[1];
    float dz = tz - ro[2];
    const float n2 = dx*dx + dy*dy + dz*dz;
    const float inv = 1.0f / std::sqrt(n2);
    rd[0] = dx * inv;
    rd[1] = dy * inv;
    rd[2] = dz * inv;
}

}  // namespace

// --- Test 1: empty BVH -----------------------------------------------------
// Edge case: Build() with zero prims must leave nodes_ empty, and the
// host traverser must short-circuit to "no hit". A bug where the
// builder pushes a sentinel root node would corrupt traversal here.
TEST_CASE("AnalyticBvh: empty BVH returns no hit for any ray") {
    AnalyticBvh bvh;
    bvh.Build({});
    CHECK(bvh.Empty());
    CHECK(bvh.NodeCount() == 0u);
    CHECK(bvh.PermutedPrimIds().empty());

    float ro[3] = {0.f, 0.f, 0.f};
    float rd[3] = {1.f, 0.f, 0.f};
    std::vector<BvhPrim> empty;
    Hit h = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), empty, ro, rd, 0.0f);
    CHECK_FALSE(h.hit);
}

// --- Test 2: single-primitive BVH -----------------------------------------
// N=1 case is meant to be "linear-scan equivalent" per the issue. The
// builder must emit one leaf node spanning the single prim. Two rays:
// one guaranteed to hit, one guaranteed to miss. prim_id matches the
// array index (the test convention used everywhere in this file --
// see the BvhTrace() doc-comment for why).
TEST_CASE("AnalyticBvh: single primitive -- hit + miss") {
    std::vector<BvhPrim> prims;
    BvhPrim p{};
    p.center[0] = 0.f; p.center[1] = 0.f; p.center[2] = 5.f;  // 5m ahead on +Z
    p.radius    = 1.0f;                                       // 1m sphere
    p.prim_id   = 0;                                          // == array index
    prims.push_back(p);

    AnalyticBvh bvh;
    bvh.Build(prims);
    REQUIRE(bvh.NodeCount() == 1u);                           // single leaf
    REQUIRE(bvh.Nodes()[0].count == 1u);                      // count==1 means leaf
    REQUIRE(bvh.PermutedPrimIds().size() == 1u);
    CHECK(bvh.PermutedPrimIds()[0] == 0u);

    // Ray firing straight down +Z from origin must hit the sphere.
    {
        float ro[3] = {0.f, 0.f, 0.f};
        float rd[3] = {0.f, 0.f, 1.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CHECK(b.hit);
        CHECK(n.hit);
        CheckHitsEqual(b, n, 0);
        CHECK(b.prim_id == 0u);
        // Sphere is at z=5 with r=1 -- closest surface is t=4m.
        CHECK(b.t == doctest::Approx(4.0f).epsilon(kHitEps));
    }

    // Ray firing +X from origin must miss the sphere (which is on +Z).
    {
        float ro[3] = {0.f, 0.f, 0.f};
        float rd[3] = {1.f, 0.f, 0.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CHECK_FALSE(b.hit);
        CHECK_FALSE(n.hit);
        CheckHitsEqual(b, n, 1);
    }
}

// --- Test 2b: prim_id round-trip --------------------------------------------
// The builder must carry each prim's user-supplied prim_id through to
// `permuted_` without rewriting it -- consumers rely on this to map
// from leaf entries back to their own data. Use distinctive values
// (not equal to array index) so a builder that accidentally rewrote
// prim_id to the array index would fail here.
TEST_CASE("AnalyticBvh: prim_id field is preserved through Build()") {
    std::vector<BvhPrim> prims;
    constexpr std::uint32_t kIds[] = {1000u, 2000u, 3000u, 4000u, 5000u};
    XorShift32 rng(0xBEEFCAFEu);
    for (auto id : kIds) {
        BvhPrim p{};
        p.center[0] = rng.rangef(-5.0f, 5.0f);
        p.center[1] = rng.rangef(-5.0f, 5.0f);
        p.center[2] = rng.rangef(-5.0f, 5.0f);
        p.radius    = 0.5f;
        p.prim_id   = id;
        prims.push_back(p);
    }
    AnalyticBvh bvh;
    bvh.Build(prims);

    // Multiset equality: every kIds value must appear exactly once in
    // the permuted output (order is allowed to change -- the BVH
    // reorders prims for leaf-locality).
    auto perm = bvh.PermutedPrimIds();
    REQUIRE(perm.size() == std::size(kIds));
    std::vector<std::uint32_t> expected(std::begin(kIds), std::end(kIds));
    std::sort(perm.begin(), perm.end());
    std::sort(expected.begin(), expected.end());
    CHECK(perm == expected);
}

// --- Test 3: random-ray fuzz (main coverage path) -------------------------
// The headline test: ~50 spheres at metre scale, 1000 deterministic
// random rays, BVH closest-hit vs naive-scan closest-hit must agree
// for every ray. A builder that drops prims, a traversal that skips
// a child, or an AABB-cull bug that culls a box still in front of
// the current best-t -- all of those make this test fail with the
// ray_idx CAPTURE pointing at exactly which ray disagreed.
TEST_CASE("AnalyticBvh: BVH closest-hit matches naive scan over 1000 random rays") {
    constexpr std::uint32_t kNumPrims = 50;
    constexpr std::uint32_t kNumRays  = 1000;
    auto prims = MakeRandomSphereScene(kNumPrims, 0xABCDEF01u);
    REQUIRE(prims.size() == kNumPrims);

    AnalyticBvh bvh;
    bvh.Build(prims);
    REQUIRE_FALSE(bvh.Empty());
    REQUIRE(bvh.PermutedPrimIds().size() == kNumPrims);

    // Builder invariant: every original prim_id must appear exactly
    // once in the permuted sequence (no prims dropped, no duplicates).
    // This is the "builder didn't drop anything" check independent of
    // the traversal -- a builder that dropped every other prim would
    // fail here before we even fire a ray.
    {
        std::vector<int> seen(kNumPrims, 0);
        for (auto id : bvh.PermutedPrimIds()) {
            REQUIRE(id < kNumPrims);
            seen[id]++;
        }
        for (std::uint32_t i = 0; i < kNumPrims; ++i) {
            CHECK(seen[i] == 1);
        }
    }

    // Fuzz rays.
    XorShift32 rng(0x12345678u);
    std::uint32_t hit_count_bvh = 0;
    std::uint32_t hit_count_naive = 0;
    for (std::uint32_t r = 0; r < kNumRays; ++r) {
        float ro[3], rd[3];
        MakeRay(rng, ro, rd);
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CheckHitsEqual(b, n, r);
        if (b.hit) ++hit_count_bvh;
        if (n.hit) ++hit_count_naive;
    }
    // Sanity: with 50 spheres scattered over a 20m cube and rays shot
    // from outside the scene toward random in-scene targets, a
    // meaningful fraction should hit. Without this, an unrelated bug
    // (say MakeRay() being broken) could silently pass the test by
    // both BVH and naive paths returning "no hit" for every ray --
    // technically agreeing, but proving nothing. Threshold is loose
    // (~5%) -- the seed produces ~90 hits, well above the floor.
    CHECK(hit_count_bvh == hit_count_naive);
    CHECK(hit_count_bvh > kNumRays / 20);                     // > 5% hits
}

// --- Test 4: many primitives sharing a split-plane coordinate -------------
// Stresses the median-split partition: if the builder's nth_element
// partition mishandles ties (e.g. all centroids equal on the split
// axis), prims could end up dropped or wrongly grouped.
//
// AnalyticBvh picks the *widest* centroid-spread axis before calling
// nth_element (src/renderer/AnalyticBvh.cpp:86-125). To actually exercise
// tie handling we have to (a) force the chosen axis to be X and
// (b) load X with ties. Two outliers at +/-100 m make X by far the
// widest centroid axis; the remaining 62 prims share x=0 exactly, so
// nth_element runs on a 64-element list with 62 ties at the median
// value.
TEST_CASE("AnalyticBvh: many primitives on same splitter plane -- all reachable") {
    std::vector<BvhPrim> prims;
    constexpr std::uint32_t kN = 64;
    XorShift32 rng(0xCAFEF00Du);
    for (std::uint32_t i = 0; i < kN; ++i) {
        BvhPrim p{};
        if (i == 0)      p.center[0] =  100.0f;               // outlier (+X) -- makes X the widest centroid axis
        else if (i == 1) p.center[0] = -100.0f;               // outlier (-X)
        else             p.center[0] =    0.0f;               // 62 ties on the chosen split axis
        p.center[1] = rng.rangef(-5.0f, 5.0f);
        p.center[2] = rng.rangef(-5.0f, 5.0f);
        p.radius    = 0.3f;
        p.prim_id   = i;
        prims.push_back(p);
    }

    AnalyticBvh bvh;
    bvh.Build(prims);
    REQUIRE(bvh.PermutedPrimIds().size() == kN);

    // No prim dropped.
    std::vector<int> seen(kN, 0);
    for (auto id : bvh.PermutedPrimIds()) {
        REQUIRE(id < kN);
        seen[id]++;
    }
    for (std::uint32_t i = 0; i < kN; ++i) CHECK(seen[i] == 1);

    // Fire a ray straight along +X at each sphere's centre and confirm
    // the BVH finds it. This is the "all reachable via ray query"
    // assertion -- naive scan over the same prims will trivially hit;
    // if any prim went missing during partition the BVH path won't.
    // Ray origin at x=-200 clears even the x=-100 outlier.
    for (std::uint32_t i = 0; i < kN; ++i) {
        const float ro[3] = {-200.0f, prims[i].center[1], prims[i].center[2]};
        const float rd[3] = {1.0f, 0.0f, 0.0f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CAPTURE(i);
        CHECK(b.hit);
        CHECK(n.hit);
        CHECK(b.hit == n.hit);
        // Closest-hit prim_id must match -- a missed prim would show up
        // as either no hit or "naive found this prim but BVH found a
        // different (further) one because the closer one was dropped
        // from a leaf".
        CHECK(b.prim_id == n.prim_id);
    }
}

// --- Test 5: AABB-degenerate primitive -- zero-radius "sphere" ------------
// Per issue: "prim AABB degenerate (sphere radius=0)". Such a prim has
// a zero-extent AABB on all three axes. The BVH must still place it in
// a leaf and the AABB intersect test must still allow a ray that
// passes exactly through the centre to descend into the leaf. The
// reference truth is that IntersectSphere with r=0 will hit only along
// the exact ray hitting (center - ro) direction, but for a degenerate
// case we settle for the *structural* invariant: BVH and naive must
// agree (both miss in practice, with floating-point r=0).
//
// Practical sub-case: one degenerate sphere AND one real one. The real
// one must still be findable via ray query through the BVH; the
// degenerate one must not corrupt the AABB or leaf layout.
TEST_CASE("AnalyticBvh: AABB-degenerate (r=0) primitive does not break traversal") {
    std::vector<BvhPrim> prims;
    // The degenerate one, sitting at z=3.
    {
        BvhPrim p{};
        p.center[0] = 0.f; p.center[1] = 0.f; p.center[2] = 3.0f;
        p.radius    = 0.0f;
        p.prim_id   = 0;
        prims.push_back(p);
    }
    // A nearby real one at z=6 with r=1 -- the ray-through-origin will
    // pass through the degenerate prim's AABB before reaching this one.
    {
        BvhPrim p{};
        p.center[0] = 0.f; p.center[1] = 0.f; p.center[2] = 6.0f;
        p.radius    = 1.0f;
        p.prim_id   = 1;
        prims.push_back(p);
    }
    // A few off-axis prims so the BVH actually splits.
    {
        XorShift32 rng(0xFACEC0DEu);
        for (std::uint32_t i = 2; i < 10; ++i) {
            BvhPrim p{};
            p.center[0] = rng.rangef(-3.0f, 3.0f);
            p.center[1] = rng.rangef(-3.0f, 3.0f);
            p.center[2] = rng.rangef(8.0f, 12.0f);
            p.radius    = 0.4f;
            p.prim_id   = i;
            prims.push_back(p);
        }
    }

    AnalyticBvh bvh;
    bvh.Build(prims);
    REQUIRE(bvh.PermutedPrimIds().size() == prims.size());

    // All prim_ids accounted for -- including the degenerate one.
    std::vector<int> seen(prims.size(), 0);
    for (auto id : bvh.PermutedPrimIds()) {
        REQUIRE(id < prims.size());
        seen[id]++;
    }
    for (std::size_t i = 0; i < prims.size(); ++i) CHECK(seen[i] == 1);

    // Ray from origin straight ahead must produce identical BVH-vs-
    // naive results (whatever the closest hit is). The point of this
    // case is the BVH must not break in the presence of a degenerate
    // (zero-extent) AABB -- whether the closest hit ends up being the
    // degenerate point-sphere at z=3 (tangentially intersected by a
    // ray that passes through its centre) or the real sphere at z=6
    // is a separate question handled by the BVH-vs-naive equivalence.
    {
        const float ro[3] = {0.f, 0.f, 0.f};
        const float rd[3] = {0.f, 0.f, 1.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CheckHitsEqual(b, n, 0);
    }

    // Off-axis ray that misses the degenerate prim's zero-extent AABB
    // entirely but should still find the real sphere at z=6. This is
    // the "real prim is reachable" assertion that guards against the
    // degenerate prim's AABB poisoning the parent node's AABB.
    {
        const float ro[3] = {0.f, 0.5f, 0.f};                 // 0.5m above XZ
        const float rd[3] = {0.f, 0.f, 1.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), prims, ro, rd, 0.0f);
        Hit n = NaiveTrace(prims, ro, rd, 0.0f);
        CheckHitsEqual(b, n, 1);
        CHECK(b.hit);
        CHECK(b.prim_id == 1u);                               // the real sphere
    }
}

// --- Test 6: AABB tightness invariant --------------------------------------
// Each BVH node's AABB must contain the AABBs of every leaf-prim
// reachable from that node. A builder that miscomputes a parent AABB
// can cull rays that actually do hit a leaf, leading to false-miss
// regressions that are otherwise only visible when staring at images.
// This walks the tree and verifies the invariant directly. Recursive
// so the test code stays compact; depth is bounded by log2(50) ~ 6
// in the fuzz scene which we reuse here.
TEST_CASE("AnalyticBvh: every node's AABB contains all reachable leaf-prim AABBs") {
    auto prims = MakeRandomSphereScene(50, 0xABCDEF01u);
    AnalyticBvh bvh;
    bvh.Build(prims);
    REQUIRE_FALSE(bvh.Empty());

    const auto& nodes    = bvh.Nodes();
    const auto& permuted = bvh.PermutedPrimIds();

    // Pre-decode every node's AABB into a small struct for clarity.
    struct AABB { float mn[3], mx[3]; };
    auto contains = [](const AABB& outer, const AABB& inner) -> bool {
        // Inflate by an epsilon to absorb float-add rounding when the
        // builder bumps a parent's max down to fit a child whose own
        // AABB was constructed with the same coords -- the parent's
        // bounds should be tight, so equality on a boundary is fine.
        constexpr float eps = 1e-5f;
        for (int i = 0; i < 3; ++i) {
            if (inner.mn[i] < outer.mn[i] - eps) return false;
            if (inner.mx[i] > outer.mx[i] + eps) return false;
        }
        return true;
    };

    // Walk all nodes; for each, gather the leaf-prim AABBs reachable
    // and check containment.
    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        AABB node_aabb;
        for (int a = 0; a < 3; ++a) {
            node_aabb.mn[a] = nodes[i].aabb_min[a];
            node_aabb.mx[a] = nodes[i].aabb_max[a];
        }
        // BFS/DFS from this node, collecting leaf prims.
        std::vector<std::uint32_t> stack{i};
        while (!stack.empty()) {
            std::uint32_t idx = stack.back();
            stack.pop_back();
            const BvhNode& n = nodes[idx];
            if (n.count > 0u) {
                for (std::uint32_t k = 0; k < n.count; ++k) {
                    const std::uint32_t prim_id = permuted[n.left_first + k];
                    REQUIRE(prim_id < prims.size());
                    const BvhPrim& p = prims[prim_id];
                    AABB prim_aabb;
                    for (int a = 0; a < 3; ++a) {
                        prim_aabb.mn[a] = p.center[a] - p.radius;
                        prim_aabb.mx[a] = p.center[a] + p.radius;
                    }
                    CAPTURE(i);
                    CAPTURE(prim_id);
                    CHECK(contains(node_aabb, prim_aabb));
                }
            } else {
                // Internal node: bounds-check child indices before
                // pushing so a builder regression that corrupts
                // left_first surfaces here as a clear REQUIRE failure
                // instead of an OOB read on the next iteration.
                const std::uint32_t lc = n.left_first;
                const std::uint32_t rc = n.left_first + 1u;
                CAPTURE(idx);
                CAPTURE(lc);
                CAPTURE(rc);
                REQUIRE(lc < nodes.size());
                REQUIRE(rc < nodes.size());
                stack.push_back(lc);
                stack.push_back(rc);
            }
        }
    }
}

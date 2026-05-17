// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::renderer::TriangleBvh -- the host-built triangle
// BVH that replaces the O(N) Möller-Trumbore linear scan PR #106
// shipped for Mac-Vulkan SW mesh fallback (issue #44).
//
// Strategy mirrors tests/pt_core_analytic_bvh_test.cpp: the BVH class
// only exposes Build() + the flat node array + the permuted triangle-id
// array; the actual traversal lives in shaders/PathTrace.slang. We
// mirror that traversal here in pure host C++ so the data structure is
// testable without a GPU.
//
//   1. Build the BVH over a known set of triangles.
//   2. Build a naive linear-scan reference over the same triangles.
//   3. Fire K=1000 deterministic random rays. The BVH-traced closest
//      hit and the naive-scan closest hit must match in (t, prim_id,
//      hit-flag) within float epsilon -- any deviation indicates either
//      a builder bug (dropped tri, bad AABB) or a traversal bug
//      (missed a child, AABB-cull bug).
//
// Determinism note: identical to the AnalyticBvh test. Integer-domain
// xorshift32 PRNG -> float via fixed shift+multiply. No std::random,
// no time-based seeds. The same seed produces the same rays + the same
// scene across Mac / Win / Linux + clang / gcc / msvc + Debug / Release.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/TriangleBvh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using pt::renderer::TriangleBvh;
using pt::renderer::TriBvhNode;

namespace {

// --- Hit record shared by BVH + naive paths -------------------------------
struct Hit {
    float         t       = std::numeric_limits<float>::infinity();
    std::uint32_t prim_id = std::numeric_limits<std::uint32_t>::max();
    bool          hit     = false;
};

// --- Deterministic xorshift32 PRNG ----------------------------------------
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
    float unitf() {                                      // [0, 1)
        return float(next() >> 8) * (1.0f / float(1u << 24));
    }
    float rangef(float lo, float hi) {                   // [lo, hi)
        return lo + (hi - lo) * unitf();
    }
};

// --- Möller-Trumbore intersect (closest positive root) --------------------
// Mirrors the shader's intersect routine in shaders/PathTrace.slang
// (the SW path PR #106 added). Two-sided (no winding rejection): the
// CSG bake doesn't guarantee consistent winding relative to incident
// rays, so the shader treats every triangle as two-sided. Reject only
// the degenerate near-zero determinant case.
bool IntersectTriangle(const float ro[3], const float rd[3],
                       const float v0[3], const float v1[3], const float v2[3],
                       float tmin, float& out_t) {
    const float e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    const float e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    // pv = rd cross e2
    const float pv[3] = {
        rd[1]*e2[2] - rd[2]*e2[1],
        rd[2]*e2[0] - rd[0]*e2[2],
        rd[0]*e2[1] - rd[1]*e2[0],
    };
    const float det = e1[0]*pv[0] + e1[1]*pv[1] + e1[2]*pv[2];
    if (std::fabs(det) < 1e-8f) return false;
    const float inv_det = 1.0f / det;

    const float tv[3] = {ro[0]-v0[0], ro[1]-v0[1], ro[2]-v0[2]};
    const float u = (tv[0]*pv[0] + tv[1]*pv[1] + tv[2]*pv[2]) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    // qv = tv cross e1
    const float qv[3] = {
        tv[1]*e1[2] - tv[2]*e1[1],
        tv[2]*e1[0] - tv[0]*e1[2],
        tv[0]*e1[1] - tv[1]*e1[0],
    };
    const float v = (rd[0]*qv[0] + rd[1]*qv[1] + rd[2]*qv[2]) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = (e2[0]*qv[0] + e2[1]*qv[1] + e2[2]*qv[2]) * inv_det;
    if (t < tmin) return false;
    out_t = t;
    return true;
}

// Pull the three vertex positions for triangle `tri` from the flat
// positions / indices arrays. Returns false if any index is OOB --
// the BVH builder skips such triangles, so the naive reference must
// agree.
bool LoadTri(const std::vector<float>&         positions,
             const std::vector<std::uint32_t>& indices,
             std::uint32_t tri,
             float v0[3], float v1[3], float v2[3]) {
    const std::uint32_t pcount = static_cast<std::uint32_t>(positions.size() / 3u);
    const std::uint32_t i0 = indices[3u * tri + 0u];
    const std::uint32_t i1 = indices[3u * tri + 1u];
    const std::uint32_t i2 = indices[3u * tri + 2u];
    if (i0 >= pcount || i1 >= pcount || i2 >= pcount) return false;
    for (int a = 0; a < 3; ++a) {
        v0[a] = positions[3u * i0 + a];
        v1[a] = positions[3u * i1 + a];
        v2[a] = positions[3u * i2 + a];
    }
    return true;
}

// --- Naive linear scan over all triangles ---------------------------------
Hit NaiveTrace(const std::vector<float>&         positions,
               const std::vector<std::uint32_t>& indices,
               const float ro[3], const float rd[3], float tmin) {
    Hit h;
    const std::uint32_t tcount = static_cast<std::uint32_t>(indices.size() / 3u);
    for (std::uint32_t t = 0; t < tcount; ++t) {
        float v0[3], v1[3], v2[3];
        if (!LoadTri(positions, indices, t, v0, v1, v2)) continue;
        float t_hit;
        if (IntersectTriangle(ro, rd, v0, v1, v2, tmin, t_hit) && t_hit < h.t) {
            h.t       = t_hit;
            h.prim_id = t;
            h.hit     = true;
        }
    }
    return h;
}

// --- Ray vs AABB slab intersect (mirrors shaders/PathTrace.slang) ---------
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
// Iterative stack-based walker, mirroring the shader (PathTrace.slang)
// stack walker that this PR adds. Stack depth ~64 covers log2(2^64)
// leaves; vastly more than any reasonable triangle count fits.
Hit BvhTrace(const std::vector<TriBvhNode>&    nodes,
             const std::vector<std::uint32_t>& permuted,
             const std::vector<float>&         positions,
             const std::vector<std::uint32_t>& indices,
             const float ro[3], const float rd[3], float tmin) {
    Hit h;
    if (nodes.empty()) return h;

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
        REQUIRE(idx < nodes.size());
        const TriBvhNode& n = nodes[idx];
        if (!IntersectAabb(ro, rd_inv, n.aabb_min, n.aabb_max, h.t)) continue;
        if (n.count > 0u) {
            CAPTURE(idx);
            CAPTURE(n.left_first);
            CAPTURE(n.count);
            REQUIRE(static_cast<std::size_t>(n.left_first) + n.count
                    <= permuted.size());
            for (std::uint32_t k = 0; k < n.count; ++k) {
                const std::uint32_t tri = permuted[n.left_first + k];
                float v0[3], v1[3], v2[3];
                if (!LoadTri(positions, indices, tri, v0, v1, v2)) continue;
                float t_hit;
                if (IntersectTriangle(ro, rd, v0, v1, v2, tmin, t_hit)
                    && t_hit < h.t) {
                    h.t       = t_hit;
                    h.prim_id = tri;
                    h.hit     = true;
                }
            }
        } else {
            const std::uint32_t lc = n.left_first;
            const std::uint32_t rc = n.left_first + 1u;
            CAPTURE(idx);
            CAPTURE(lc);
            CAPTURE(rc);
            REQUIRE(lc < nodes.size());
            REQUIRE(rc < nodes.size());
            if (sp + 2 <= 64) {
                stack[sp++] = lc;
                stack[sp++] = rc;
            }
        }
    }
    return h;
}

// --- Scene generator ------------------------------------------------------
// Build a deterministic metre-scale triangle soup. Each triangle has
// three independently-jittered vertices in a ~20m cube. Triangle areas
// vary by an order of magnitude; centroids spread across the volume,
// which gives the BVH builder a substantive job (a uniform-vertex
// scene where every triangle has the same centroid would degenerate
// into the all-coincident-centroid fallback path).
//
// Returns positions (3N floats) + indices (3T uints, with T = N/3 since
// each triangle gets its own three vertices, no shared verts).
void MakeRandomTriangleScene(std::uint32_t n_tris, std::uint32_t seed,
                             std::vector<float>&         positions,
                             std::vector<std::uint32_t>& indices) {
    XorShift32 rng(seed);
    positions.clear();
    indices.clear();
    positions.reserve(static_cast<std::size_t>(n_tris) * 9u);
    indices.reserve(static_cast<std::size_t>(n_tris) * 3u);
    for (std::uint32_t t = 0; t < n_tris; ++t) {
        // Anchor vertex anywhere in [-10m, 10m] per axis. Other two
        // vertices are within ~1m of the anchor so each triangle is
        // metre-scale but still has a finite, non-degenerate area.
        const float ax = rng.rangef(-10.0f, 10.0f);
        const float ay = rng.rangef(-10.0f, 10.0f);
        const float az = rng.rangef(-10.0f, 10.0f);
        const float v0[3] = {ax, ay, az};
        const float v1[3] = {ax + rng.rangef(-1.0f, 1.0f),
                             ay + rng.rangef(-1.0f, 1.0f),
                             az + rng.rangef(-1.0f, 1.0f)};
        const float v2[3] = {ax + rng.rangef(-1.0f, 1.0f),
                             ay + rng.rangef(-1.0f, 1.0f),
                             az + rng.rangef(-1.0f, 1.0f)};
        const std::uint32_t base = static_cast<std::uint32_t>(positions.size() / 3u);
        for (int a = 0; a < 3; ++a) positions.push_back(v0[a]);
        for (int a = 0; a < 3; ++a) positions.push_back(v1[a]);
        for (int a = 0; a < 3; ++a) positions.push_back(v2[a]);
        indices.push_back(base + 0u);
        indices.push_back(base + 1u);
        indices.push_back(base + 2u);
    }
}

constexpr float kHitEps = 1e-4f;

void CheckHitsEqual(const Hit& bvh, const Hit& naive, std::uint32_t ray_idx) {
    CAPTURE(ray_idx);
    CHECK(bvh.hit == naive.hit);
    if (!bvh.hit || !naive.hit) return;
    CHECK(bvh.prim_id == naive.prim_id);
    CHECK(bvh.t == doctest::Approx(naive.t).epsilon(kHitEps));
}

// Generate a deterministic random ray. Origin biased outside the scene
// (radius >= 15m from origin), direction aimed at a random target
// inside the scene so most rays meaningfully traverse the BVH.
void MakeRay(XorShift32& rng, float ro[3], float rd[3]) {
    do {
        ro[0] = rng.rangef(-25.0f, 25.0f);
        ro[1] = rng.rangef(-25.0f, 25.0f);
        ro[2] = rng.rangef(-25.0f, 25.0f);
    } while (ro[0]*ro[0] + ro[1]*ro[1] + ro[2]*ro[2] < 15.0f * 15.0f);
    const float tx = rng.rangef(-14.0f, 14.0f);
    const float ty = rng.rangef(-14.0f, 14.0f);
    const float tz = rng.rangef(-14.0f, 14.0f);
    float dx = tx - ro[0];
    float dy = ty - ro[1];
    float dz = tz - ro[2];
    const float inv = 1.0f / std::sqrt(dx*dx + dy*dy + dz*dz);
    rd[0] = dx * inv;
    rd[1] = dy * inv;
    rd[2] = dz * inv;
}

}  // namespace

// --- Test 1: empty BVH -----------------------------------------------------
TEST_CASE("TriangleBvh: empty BVH returns no hit for any ray") {
    TriangleBvh bvh;
    bvh.Build({}, {});
    CHECK(bvh.Empty());
    CHECK(bvh.NodeCount() == 0u);
    CHECK(bvh.PermutedPrimIds().empty());

    std::vector<float> empty_pos;
    std::vector<std::uint32_t> empty_idx;
    float ro[3] = {0.f, 0.f, 0.f};
    float rd[3] = {1.f, 0.f, 0.f};
    Hit h = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), empty_pos, empty_idx,
                     ro, rd, 0.0f);
    CHECK_FALSE(h.hit);
}

// --- Test 1b: malformed indices --------------------------------------------
// Indices size not a multiple of 3 -> Build() must no-op cleanly. The
// CSG bake never produces this, but the class contract is "never crash
// on malformed input".
TEST_CASE("TriangleBvh: indices size not a multiple of 3 -> empty") {
    std::vector<float> positions = {0,0,0, 1,0,0, 0,1,0};
    std::vector<std::uint32_t> bad_indices = {0, 1};  // size 2
    TriangleBvh bvh;
    bvh.Build(positions, bad_indices);
    CHECK(bvh.Empty());
}

// --- Test 2: single triangle -----------------------------------------------
TEST_CASE("TriangleBvh: single triangle -- hit + miss") {
    // A 1m^2 triangle in the XY plane at z=5m, facing the origin.
    std::vector<float> positions = {
       -0.5f, -0.5f, 5.0f,    // v0
        0.5f, -0.5f, 5.0f,    // v1
        0.0f,  0.5f, 5.0f,    // v2
    };
    std::vector<std::uint32_t> indices = {0u, 1u, 2u};

    TriangleBvh bvh;
    bvh.Build(positions, indices);
    REQUIRE(bvh.NodeCount() == 1u);
    REQUIRE(bvh.Nodes()[0].count == 1u);
    REQUIRE(bvh.PermutedPrimIds().size() == 1u);
    CHECK(bvh.PermutedPrimIds()[0] == 0u);

    // Ray firing +Z from origin must hit the triangle (centred on the Z axis).
    {
        float ro[3] = {0.f, 0.f, 0.f};
        float rd[3] = {0.f, 0.f, 1.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), positions, indices,
                         ro, rd, 0.0f);
        Hit n = NaiveTrace(positions, indices, ro, rd, 0.0f);
        CHECK(b.hit);
        CHECK(n.hit);
        CheckHitsEqual(b, n, 0);
        CHECK(b.t == doctest::Approx(5.0f).epsilon(kHitEps));
    }

    // Ray firing +X from origin must miss (triangle is on +Z).
    {
        float ro[3] = {0.f, 0.f, 0.f};
        float rd[3] = {1.f, 0.f, 0.f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), positions, indices,
                         ro, rd, 0.0f);
        Hit n = NaiveTrace(positions, indices, ro, rd, 0.0f);
        CHECK_FALSE(b.hit);
        CHECK_FALSE(n.hit);
        CheckHitsEqual(b, n, 1);
    }
}

// --- Test 3: degenerate (collapsed-to-line) triangle -----------------------
// A triangle whose three vertices are collinear has zero area; the
// Möller-Trumbore intersect's |det| > 1e-8 guard rejects it. The
// builder must still place it in a leaf, the AABB must still be
// finite (a thin line, not NaN), and traversal of nearby real
// triangles must not break.
TEST_CASE("TriangleBvh: degenerate (collinear) triangle does not corrupt traversal") {
    // Triangle 0: degenerate (3 collinear vertices).
    // Triangle 1: a real triangle off to one side.
    std::vector<float> positions = {
        0.0f, 0.0f, 3.0f,   // v0 (tri0)
        1.0f, 0.0f, 3.0f,   // v1 (tri0)
        2.0f, 0.0f, 3.0f,   // v2 (tri0, collinear with v0/v1)
        // tri1: a 1m triangle in the YZ plane at x=2.
       -0.5f, 0.5f, 6.0f,   // v0
        0.5f, 0.5f, 6.0f,   // v1
        0.0f, -0.5f, 6.0f,  // v2
    };
    std::vector<std::uint32_t> indices = {
        0u, 1u, 2u,           // tri 0 (degenerate)
        3u, 4u, 5u,           // tri 1 (real)
    };

    TriangleBvh bvh;
    bvh.Build(positions, indices);
    REQUIRE(bvh.PermutedPrimIds().size() == 2u);

    // No tri dropped from the permuted set.
    std::vector<int> seen(2, 0);
    for (auto id : bvh.PermutedPrimIds()) {
        REQUIRE(id < 2u);
        seen[id]++;
    }
    CHECK(seen[0] == 1);
    CHECK(seen[1] == 1);

    // Every node's AABB must be finite (NaN check). A collinear
    // triangle has a thin but valid 3D AABB.
    for (const auto& n : bvh.Nodes()) {
        for (int a = 0; a < 3; ++a) {
            CHECK(std::isfinite(n.aabb_min[a]));
            CHECK(std::isfinite(n.aabb_max[a]));
            CHECK(n.aabb_min[a] <= n.aabb_max[a]);
        }
    }

    // Ray from origin straight along +Z at y=0.5: must hit the real tri,
    // and BVH and naive must agree.
    float ro[3] = {0.f, 0.f, 0.f};
    float rd[3] = {0.f, 0.083205f, 0.996546f}; // ~atan(0.5/6)
    // Normalise just to be safe.
    {
        float n = std::sqrt(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);
        rd[0] /= n; rd[1] /= n; rd[2] /= n;
    }
    Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), positions, indices,
                     ro, rd, 0.0f);
    Hit nh = NaiveTrace(positions, indices, ro, rd, 0.0f);
    CheckHitsEqual(b, nh, 0);
}

// --- Test 4: many triangles sharing a split-plane coordinate ---------------
// Stresses median-split partition: 1 outlier triangle on each X
// extreme, 62 triangles with centroids exactly at x=0. Builder picks X
// as the widest centroid axis, then nth_element runs on 64 entries
// with 62 ties at the median.
TEST_CASE("TriangleBvh: many triangles on same splitter plane -- all reachable") {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
    XorShift32 rng(0xCAFEF00Du);
    constexpr std::uint32_t kN = 64u;

    auto add_tri_at = [&](float cx, float cy, float cz) {
        // Tiny triangle in the YZ plane so its centroid is at (cx, cy, cz).
        // Three vertices straddle (cx, cy, cz) on the YZ plane only.
        std::uint32_t base = static_cast<std::uint32_t>(positions.size() / 3u);
        positions.insert(positions.end(),
            {cx, cy - 0.1f, cz - 0.1f,
             cx, cy + 0.2f, cz - 0.1f,
             cx, cy - 0.1f, cz + 0.2f});
        indices.push_back(base + 0u);
        indices.push_back(base + 1u);
        indices.push_back(base + 2u);
    };

    for (std::uint32_t i = 0; i < kN; ++i) {
        float cx;
        if (i == 0)      cx =  100.0f;             // outlier (+X)
        else if (i == 1) cx = -100.0f;             // outlier (-X)
        else             cx =    0.0f;             // ties on X
        const float cy = rng.rangef(-5.0f, 5.0f);
        const float cz = rng.rangef(-5.0f, 5.0f);
        add_tri_at(cx, cy, cz);
    }

    TriangleBvh bvh;
    bvh.Build(positions, indices);
    REQUIRE(bvh.PermutedPrimIds().size() == kN);

    // No triangle dropped.
    std::vector<int> seen(kN, 0);
    for (auto id : bvh.PermutedPrimIds()) {
        REQUIRE(id < kN);
        seen[id]++;
    }
    for (std::uint32_t i = 0; i < kN; ++i) CHECK(seen[i] == 1);

    // Fire a ray along +X aimed at each triangle's plane. Each triangle
    // is a tiny YZ-plane patch, so an axis-aligned +X ray at the
    // triangle's (cy, cz) must hit it. Ray origin at x=-200 clears even
    // the x=-100 outlier.
    for (std::uint32_t i = 0; i < kN; ++i) {
        // Reconstruct the centroid from positions: each triangle owns
        // 3 vertices at base = 3*i; vertex 0 has (cx, cy-0.1, cz-0.1).
        const std::uint32_t base = 3u * i;
        const float cx = positions[3u * base + 0];
        const float cy_v0 = positions[3u * base + 1];
        const float cz_v0 = positions[3u * base + 2];
        // Aim a hair "inside" the triangle.
        const float ro[3] = {-200.0f, cy_v0 + 0.05f, cz_v0 + 0.05f};
        const float rd[3] = {1.0f, 0.0f, 0.0f};
        Hit b = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), positions, indices,
                         ro, rd, 0.0f);
        Hit nh = NaiveTrace(positions, indices, ro, rd, 0.0f);
        CAPTURE(i);
        CAPTURE(cx);
        CHECK(b.hit == nh.hit);
        if (b.hit && nh.hit) {
            CHECK(b.prim_id == nh.prim_id);
            CHECK(b.t == doctest::Approx(nh.t).epsilon(kHitEps));
        }
    }
}

// --- Test 5: random-ray fuzz (main coverage) -------------------------------
TEST_CASE("TriangleBvh: BVH closest-hit matches naive scan over 1000 random rays") {
    constexpr std::uint32_t kNumTris = 30u;
    constexpr std::uint32_t kNumRays = 1000u;
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
    MakeRandomTriangleScene(kNumTris, 0xABCDEF01u, positions, indices);
    REQUIRE(indices.size() == 3u * kNumTris);

    TriangleBvh bvh;
    bvh.Build(positions, indices);
    REQUIRE_FALSE(bvh.Empty());
    REQUIRE(bvh.PermutedPrimIds().size() == kNumTris);

    // Builder invariant: every original triangle id appears exactly once.
    {
        std::vector<int> seen(kNumTris, 0);
        for (auto id : bvh.PermutedPrimIds()) {
            REQUIRE(id < kNumTris);
            seen[id]++;
        }
        for (std::uint32_t i = 0; i < kNumTris; ++i) {
            CHECK(seen[i] == 1);
        }
    }

    // Fuzz rays.
    XorShift32 rng(0x12345678u);
    std::uint32_t hit_count_bvh   = 0;
    std::uint32_t hit_count_naive = 0;
    for (std::uint32_t r = 0; r < kNumRays; ++r) {
        float ro[3], rd[3];
        MakeRay(rng, ro, rd);
        Hit b  = BvhTrace(bvh.Nodes(), bvh.PermutedPrimIds(), positions, indices,
                          ro, rd, 0.0f);
        Hit nh = NaiveTrace(positions, indices, ro, rd, 0.0f);
        CheckHitsEqual(b, nh, r);
        if (b.hit)  ++hit_count_bvh;
        if (nh.hit) ++hit_count_naive;
    }
    // Sanity: like the AnalyticBvh test, ensure the seeds + scene
    // actually produce a meaningful number of hits. ~30 random
    // triangles in a 20m cube + rays from outside aimed at random
    // in-scene targets won't have a *huge* hit rate (small targets)
    // but it must be positive. A regression that made every ray miss
    // would silently pass the equivalence check otherwise.
    CHECK(hit_count_bvh == hit_count_naive);
    CHECK(hit_count_bvh > 0u);
}

// --- Test 6: AABB tightness invariant --------------------------------------
// Every internal node's AABB must contain the AABBs of all leaf-tri
// AABBs reachable from it. A builder that miscomputes a parent AABB
// can AABB-cull rays that genuinely hit a leaf.
TEST_CASE("TriangleBvh: every node's AABB contains all reachable leaf-tri AABBs") {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
    MakeRandomTriangleScene(50u, 0xABCDEF01u, positions, indices);
    TriangleBvh bvh;
    bvh.Build(positions, indices);
    REQUIRE_FALSE(bvh.Empty());

    const auto& nodes    = bvh.Nodes();
    const auto& permuted = bvh.PermutedPrimIds();

    struct AABB { float mn[3], mx[3]; };
    auto contains = [](const AABB& outer, const AABB& inner) -> bool {
        constexpr float eps = 1e-5f;
        for (int i = 0; i < 3; ++i) {
            if (inner.mn[i] < outer.mn[i] - eps) return false;
            if (inner.mx[i] > outer.mx[i] + eps) return false;
        }
        return true;
    };

    auto tri_aabb = [&](std::uint32_t tri) -> AABB {
        AABB ab;
        for (int a = 0; a < 3; ++a) { ab.mn[a] =  1e30f; ab.mx[a] = -1e30f; }
        float v0[3], v1[3], v2[3];
        REQUIRE(LoadTri(positions, indices, tri, v0, v1, v2));
        for (int a = 0; a < 3; ++a) {
            ab.mn[a] = std::min(v0[a], std::min(v1[a], v2[a]));
            ab.mx[a] = std::max(v0[a], std::max(v1[a], v2[a]));
        }
        return ab;
    };

    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        AABB node_aabb;
        for (int a = 0; a < 3; ++a) {
            node_aabb.mn[a] = nodes[i].aabb_min[a];
            node_aabb.mx[a] = nodes[i].aabb_max[a];
        }
        std::vector<std::uint32_t> stack{i};
        while (!stack.empty()) {
            std::uint32_t idx = stack.back();
            stack.pop_back();
            const TriBvhNode& n = nodes[idx];
            if (n.count > 0u) {
                CAPTURE(idx);
                CAPTURE(n.left_first);
                CAPTURE(n.count);
                REQUIRE(static_cast<std::size_t>(n.left_first) + n.count
                        <= permuted.size());
                for (std::uint32_t k = 0; k < n.count; ++k) {
                    const std::uint32_t tri = permuted[n.left_first + k];
                    AABB pab = tri_aabb(tri);
                    CAPTURE(i);
                    CAPTURE(tri);
                    CHECK(contains(node_aabb, pab));
                }
            } else {
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

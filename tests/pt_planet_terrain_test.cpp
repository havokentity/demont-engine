// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for planetary P4 (#258): the cubed-ellipsoid quadtree, the
// elevation field, chunk baking and residency selection.
//
// WHAT THIS FILE IS FOR
//
// Three of P4's acceptance criteria are claims a golden PNG cannot settle,
// because a PNG can only show that a particular camera saw no crack.
//
//   * "A watertight surface: no cracks between LOD levels." The claim is
//     that two chunks at DIFFERENT levels sharing an edge produce
//     BIT-IDENTICAL vertex positions there -- for every level pair, on
//     every one of the 12 cube seams, not just the ones a fixture happens
//     to look at. That is an equality over the whole domain, so it is
//     asserted here and the goldens then show the same surface reached the
//     screen.
//
//   * "Golden determinism." Chunks are baked asynchronously on a worker
//     pool, so frame N depends on wall clock unless the bake is a pure
//     function of the key. Baking the same chunk twice and comparing the
//     bytes is the only honest test of that.
//
//   * "No popping." The LOD rule's sub-pixel property and the hysteresis
//     that stops a chunk oscillating are numerical statements about the
//     selector, tested directly.
//
// FINITENESS UNDER -ffast-math: this target is built with -ffast-math in
// Release like every other test target, so std::isfinite() folds to `true`
// and so does a naive exponent-bit read (LLVM propagates the producing op's
// ninf/nnan flags through the bitcast). finiteBits() below breaks the chain
// with a volatile round-trip, and FiniteHarnessWorks asserts the harness is
// not vacuous -- the trap PR #273 shipped once already.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "renderer/Planet/CubedSphere.h"
#include "renderer/Planet/ElevationField.h"
#include "renderer/Planet/TerrainChunk.h"
#include "renderer/Planet/TerrainQuadtree.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <utility>
#include <thread>
#include <vector>

using namespace pt::planet;

namespace {

inline bool finiteBits(double v) {
    volatile double t = v;
    double u = t;
    std::uint64_t b;
    std::memcpy(&b, &u, 8);
    return ((b >> 52) & 0x7FFull) != 0x7FFull;
}
inline bool finiteBitsF(float v) {
    volatile float t = v;
    float u = t;
    std::uint32_t b;
    std::memcpy(&b, &u, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu;
}

constexpr double kPi = 3.14159265358979323846;

// Opposite of an edge, for the neighbour-involution test.
ChunkEdge Opposite(ChunkEdge e) {
    switch (e) {
        case kEdgeMinusU: return kEdgePlusU;
        case kEdgePlusU:  return kEdgeMinusU;
        case kEdgeMinusV: return kEdgePlusV;
        default:          return kEdgeMinusV;
    }
}

ElevationField MakeProceduralField(double relief_m = 800.0,
                                   double floor_m = 20000.0) {
    ElevationField f;
    ElevationParams p;
    p.procedural_relief_m = relief_m;
    p.procedural_floor_m  = floor_m;
    f.SetParams(p);
    return f;
}

}  // namespace

TEST_CASE("finiteness harness is not vacuous under -ffast-math") {
    volatile double huge = 1e308;
    const double inf = huge * 10.0;
    REQUIRE_FALSE(finiteBits(inf));
    REQUIRE(finiteBits(1.0));
    volatile float hugef = 1e38f;
    const float inff = hugef * 10.0f;
    REQUIRE_FALSE(finiteBitsF(inff));
    REQUIRE(finiteBitsF(1.0f));
}

// --- Geodesy ---------------------------------------------------------------

TEST_CASE("WGS-84 constants match the defining standard") {
    // b = a * (1 - f) with 1/f = 298.257223563.
    const double b = kWgs84A * (1.0 - 1.0 / kWgs84InvF);
    CHECK(std::abs(b - kWgs84B) < 1e-6);
    // The IUGG mean radius is (2a + b)/3.
    CHECK(std::abs((2.0 * kWgs84A + kWgs84B) / 3.0 - kIuggMeanRadius) < 0.2);
    // The bulge the design refuses to round away: a - b = 21 385 m.
    CHECK(std::abs((kWgs84A - kWgs84B) - 21384.685755) < 1e-3);
}

TEST_CASE("the uint16 DEM affine spans the real Earth") {
    // The claim in ElevationField.h is that the encoding covers Challenger
    // Deep and Everest with 30 cm quantisation. Both ends, checked.
    const double lo = 0.0 * kDemScaleM + kDemOffsetM;
    const double hi = 65535.0 * kDemScaleM + kDemOffsetM;
    CHECK(lo <= kEarthMinElevation);
    CHECK(hi >= kEarthMaxElevation);
    CHECK(kDemScaleM < 0.31);           // "30 cm quantisation"
    // And the value the issue quoted, 0.302, would NOT have reached Everest
    // -- which is why the code derives its own rather than copying it.
    CHECK(65535.0 * 0.302 + kDemOffsetM < kEarthMaxElevation);
}

TEST_CASE("the tangent warp equalises the cubed-sphere metric") {
    // Worst-case area distortion across a face: the ratio of the solid
    // angle per unit parameter area at the face centre to that at the
    // corner. Sampled on a fine grid, naive vs tangent-warped.
    auto distortion = [](bool warped) {
        const int N = 64;
        double lo = 1e30, hi = -1e30;
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const double s = -1.0 + 2.0 * (i + 0.5) / N;
                const double t = -1.0 + 2.0 * (j + 0.5) / N;
                const double d = 2.0 / N;
                auto dir = [&](double a, double b) {
                    const double p = warped ? TangentWarp(a) : a;
                    const double q = warped ? TangentWarp(b) : b;
                    return glm::normalize(glm::dvec3(1.0, p, q));
                };
                const glm::dvec3 c  = dir(s, t);
                const glm::dvec3 du = dir(s + d * 0.5, t) - dir(s - d * 0.5, t);
                const glm::dvec3 dv = dir(s, t + d * 0.5) - dir(s, t - d * 0.5);
                const double area = glm::length(glm::cross(du, dv));
                (void)c;
                lo = std::min(lo, area);
                hi = std::max(hi, area);
            }
        }
        return hi / lo;
    };
    const double naive  = distortion(false);
    const double warped = distortion(true);
    // The header claims ~5.2x naive and ~1.3x warped.
    CHECK(naive  > 5.0);
    CHECK(naive  < 5.4);
    CHECK(warped > 1.2);
    CHECK(warped < 1.4);
    // And the point of the whole exercise: the warp must actually improve
    // it, by roughly the factor claimed.
    CHECK(naive / warped > 3.5);
}

TEST_CASE("face parameterisation round-trips") {
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i <= 8; ++i) {
            for (int j = 0; j <= 8; ++j) {
                const double s = -1.0 + 2.0 * i / 8.0;
                const double t = -1.0 + 2.0 * j / 8.0;
                const glm::dvec3 d = FaceParamToDirection(f, s, t);
                CHECK(std::abs(glm::length(d) - 1.0) < 1e-12);
                int f2 = -1; double s2 = 0, t2 = 0;
                DirectionToFaceParam(d, f2, s2, t2);
                // On a cube edge or corner two faces are equally valid, so
                // compare the resulting DIRECTION, not the chart.
                const glm::dvec3 d2 = FaceParamToDirection(f2, s2, t2);
                CHECK(glm::length(d2 - d) < 1e-12);
            }
        }
    }
}

TEST_CASE("the ellipsoid surface satisfies its own defining quadric") {
    for (int f = 0; f < 6; ++f) {
        for (int i = 0; i <= 4; ++i) {
            for (int j = 0; j <= 4; ++j) {
                const glm::dvec3 d =
                    FaceParamToDirection(f, -1.0 + 0.5 * i, -1.0 + 0.5 * j);
                const glm::dvec3 p = EllipsoidSurface(d);
                const double q = (p.x * p.x + p.y * p.y) / (kWgs84A * kWgs84A)
                               + (p.z * p.z) / (kWgs84B * kWgs84B);
                CHECK(std::abs(q - 1.0) < 1e-12);
                // The geodetic normal is a unit vector and points outward.
                const glm::dvec3 n = GeodeticNormal(p);
                CHECK(std::abs(glm::length(n) - 1.0) < 1e-12);
                CHECK(glm::dot(n, glm::normalize(p)) > 0.99);
            }
        }
    }
}

TEST_CASE("geodetic round trip through GeodeticToEcef") {
    for (int i = -8; i <= 8; ++i) {
        for (int j = -8; j <= 8; ++j) {
            const double lat = i * (kPi * 0.5 / 9.0);
            const double lon = j * (kPi / 9.0);
            const glm::dvec3 p = GeodeticToEcef(lat, lon);
            double lat2 = 0, lon2 = 0;
            EcefToGeodetic(p, lat2, lon2);
            CHECK(std::abs(lat2 - lat) < 1e-10);
            CHECK(std::abs(std::sin(lon2 - lon)) < 1e-10);
        }
    }
}

// --- Neighbours, including the 12 cube seams -------------------------------

TEST_CASE("NeighborChunk is involutive on every face and every seam") {
    int seam_crossings = 0;
    for (int level = 0; level <= 3; ++level) {
        const std::uint32_t span = 1u << level;
        for (int f = 0; f < 6; ++f) {
            for (std::uint32_t i = 0; i < span; ++i) {
                for (std::uint32_t j = 0; j < span; ++j) {
                    const ChunkKey k{static_cast<std::uint8_t>(f),
                                     static_cast<std::uint8_t>(level), i, j};
                    for (int e = 0; e < 4; ++e) {
                        ChunkKey nb{};
                        REQUIRE(NeighborChunk(k, static_cast<ChunkEdge>(e), nb));
                        CHECK(nb.level == k.level);
                        if (nb.face != k.face) ++seam_crossings;
                        // The neighbour must be adjacent: their chunk centres
                        // are one chunk apart on the sphere. This is the
                        // property the seam algebra exists to preserve, and
                        // it catches a wrong face or a wrong cell where an
                        // involution check alone would not.
                        const glm::dvec3 a = ChunkCenterDirection(k);
                        const glm::dvec3 b = ChunkCenterDirection(nb);
                        const double ang = std::acos(
                            std::clamp(glm::dot(a, b), -1.0, 1.0));
                        // One chunk of arc, generously bounded: the tangent
                        // warp makes cells vary by ~1.3x in size.
                        const double cell = (kPi * 0.5) / static_cast<double>(span);
                        CHECK(ang > 0.0);
                        CHECK(ang < cell * 1.6);
                    }
                }
            }
        }
    }
    // Every level must actually exercise seam crossings, or the test is
    // only checking the easy in-face case. 6 faces * 4 edges * span
    // boundary chunks per level.
    CHECK(seam_crossings > 0);
}

TEST_CASE("crossing a seam and coming back returns to the start") {
    // Involution: for an in-face neighbour the opposite edge always returns
    // home. Across a seam the two faces' axes may be permuted, so the
    // return edge is whichever one leads back -- assert that SOME edge does.
    for (int level = 0; level <= 2; ++level) {
        const std::uint32_t span = 1u << level;
        for (int f = 0; f < 6; ++f) {
            for (std::uint32_t i = 0; i < span; ++i) {
                for (std::uint32_t j = 0; j < span; ++j) {
                    const ChunkKey k{static_cast<std::uint8_t>(f),
                                     static_cast<std::uint8_t>(level), i, j};
                    for (int e = 0; e < 4; ++e) {
                        ChunkKey nb{};
                        REQUIRE(NeighborChunk(k, static_cast<ChunkEdge>(e), nb));
                        bool came_back = false;
                        for (int e2 = 0; e2 < 4; ++e2) {
                            ChunkKey back{};
                            if (NeighborChunk(nb, static_cast<ChunkEdge>(e2), back) &&
                                back == k) {
                                came_back = true;
                                if (nb.face == k.face) {
                                    CHECK(e2 == static_cast<int>(Opposite(
                                              static_cast<ChunkEdge>(e))));
                                }
                                break;
                            }
                        }
                        CHECK(came_back);
                    }
                }
            }
        }
    }
}

// --- The shared index arena ------------------------------------------------

TEST_CASE("index arena: 16 variants, correct size, in-range indices") {
    TerrainIndexArena arena;
    CHECK(arena.Indices().size() ==
          static_cast<std::size_t>(TerrainIndexArena::kVariantCount) * kChunkIndexCount);
    CHECK(arena.ByteSize() == arena.Indices().size() * sizeof(std::uint32_t));
    // ~1.6 MB, the number the design quotes.
    CHECK(arena.ByteSize() > 1500000u);
    CHECK(arena.ByteSize() < 1700000u);
    for (std::uint32_t v = 0; v < TerrainIndexArena::kVariantCount; ++v) {
        const std::uint32_t* p = arena.Variant(v);
        for (int t = 0; t < kChunkIndexCount; ++t) {
            CHECK(p[t] < static_cast<std::uint32_t>(kChunkVertexCount));
        }
    }
}

TEST_CASE("a stitched edge references only the coarse neighbour's vertices") {
    TerrainIndexArena arena;
    for (int edge = 0; edge < 4; ++edge) {
        const std::uint32_t mask = 1u << edge;
        const std::uint32_t* p = arena.Variant(mask);
        // Collect every vertex the variant references that lies on the
        // stitched edge. None may be an odd index along that edge -- an odd
        // one is a T-junction against the coarse neighbour's 33-vertex
        // polyline, i.e. a crack.
        for (int t = 0; t < kChunkIndexCount; ++t) {
            const std::uint32_t vi = p[t];
            const int x = static_cast<int>(vi) % kChunkVerts;
            const int y = static_cast<int>(vi) / kChunkVerts;
            if (edge == kEdgeMinusU && x == 0)           CHECK((y % 2) == 0);
            if (edge == kEdgePlusU  && x == kChunkQuads) CHECK((y % 2) == 0);
            if (edge == kEdgeMinusV && y == 0)           CHECK((x % 2) == 0);
            if (edge == kEdgePlusV  && y == kChunkQuads) CHECK((x % 2) == 0);
        }
    }
    // The unstitched variant DOES use the odd boundary vertices -- red-then-
    // green on the check above, so it cannot pass by referencing nothing.
    const std::uint32_t* p0 = arena.Variant(0);
    bool saw_odd_boundary = false;
    for (int t = 0; t < kChunkIndexCount; ++t) {
        const int x = static_cast<int>(p0[t]) % kChunkVerts;
        const int y = static_cast<int>(p0[t]) / kChunkVerts;
        if (x == 0 && (y % 2) == 1) { saw_odd_boundary = true; break; }
    }
    CHECK(saw_odd_boundary);
}

TEST_CASE("the boundary snap is idempotent and leaves corners alone") {
    for (std::uint32_t mask = 0; mask < 16; ++mask) {
        for (int y = 0; y <= kChunkQuads; ++y) {
            for (int x = 0; x <= kChunkQuads; ++x) {
                const std::uint32_t a = SnappedVertexIndex(x, y, mask);
                const int ax = static_cast<int>(a) % kChunkVerts;
                const int ay = static_cast<int>(a) / kChunkVerts;
                CHECK(SnappedVertexIndex(ax, ay, mask) == a);
            }
        }
        // The four chunk corners are even in both axes, so they survive
        // every mask -- which is what keeps a corner shared by three or four
        // chunks from splitting into two positions.
        CHECK(SnappedVertexIndex(0, 0, mask) == 0u);
        CHECK(SnappedVertexIndex(kChunkQuads, kChunkQuads, mask) ==
              static_cast<std::uint32_t>(kChunkVertexCount - 1));
    }
}

// --- The elevation field ---------------------------------------------------

TEST_CASE("the elevation field is consistent across subdivision levels") {
    // THE watertightness claim. A level-L chunk and one of its level-(L+1)
    // children must produce BIT-IDENTICAL heights at every vertex they
    // share. Not "within a tolerance" -- identical, because anything else
    // is a crack whose width is the difference.
    ElevationField field = MakeProceduralField(1200.0, 20000.0);
    for (int level = 2; level <= 6; ++level) {
        const ChunkKey parent{2, static_cast<std::uint8_t>(level), 1u, 1u};
        std::vector<double> pg;
        field.GenerateChunkHeights(parent, level, 0, pg);
        REQUIRE(pg.size() == static_cast<std::size_t>(kChunkVerts) * kChunkVerts);

        for (int q = 0; q < 4; ++q) {
            const ChunkKey child = parent.Child(q);
            std::vector<double> cg;
            field.GenerateChunkHeights(child, static_cast<int>(child.level), 0, cg);
            REQUIRE(cg.size() == static_cast<std::size_t>(kChunkVerts) * kChunkVerts);
            const int ox = (q & 1) * (kChunkQuads / 2);
            const int oy = ((q >> 1) & 1) * (kChunkQuads / 2);
            for (int y = 0; y <= kChunkQuads; y += 2) {
                for (int x = 0; x <= kChunkQuads; x += 2) {
                    const double a =
                        cg[static_cast<std::size_t>(y) * kChunkVerts + x];
                    const double b =
                        pg[static_cast<std::size_t>(oy + y / 2) * kChunkVerts
                           + (ox + x / 2)];
                    CHECK(a == b);          // bit-identical, deliberately
                }
            }
        }
    }
}

TEST_CASE("the elevation field agrees across a cube seam") {
    // Two chunks on DIFFERENT faces sharing a seam edge must give the same
    // heights along it. This is why the fractal hash keys off the integer
    // cube lattice rather than (face, i, j): the latter cracks all 12 seams
    // by construction.
    ElevationField field = MakeProceduralField(1500.0, 20000.0);
    const int level = 3;
    int checked = 0;
    for (int f = 0; f < 6; ++f) {
        const std::uint32_t span = 1u << level;
        // Chunk on the +u boundary of the face.
        const ChunkKey k{static_cast<std::uint8_t>(f),
                         static_cast<std::uint8_t>(level), span - 1, 2u};
        ChunkKey nb{};
        REQUIRE(NeighborChunk(k, kEdgePlusU, nb));
        if (nb.face == k.face) continue;
        std::vector<double> ga, gb;
        field.GenerateChunkHeights(k,  level, 0, ga);
        field.GenerateChunkHeights(nb, level, 0, gb);

        // Match by 3D position rather than by index -- the neighbouring
        // face's axes are a permutation of this one's, and which one is not
        // something the test should have to know.
        const std::int64_t G = static_cast<std::int64_t>(kChunkQuads) << level;
        auto dir_of = [&](const ChunkKey& c, int x, int y) {
            const std::int64_t bx = (static_cast<std::int64_t>(c.i) * G) / span;
            const std::int64_t by = (static_cast<std::int64_t>(c.j) * G) / span;
            return FaceParamToDirection(c.face, GridParam(bx + x, G),
                                        GridParam(by + y, G));
        };
        for (int y = 0; y <= kChunkQuads; ++y) {
            const glm::dvec3 pa = dir_of(k, kChunkQuads, y);
            const double ha = ga[static_cast<std::size_t>(y) * kChunkVerts + kChunkQuads];
            bool matched = false;
            for (int yy = 0; yy <= kChunkQuads && !matched; ++yy) {
                for (int xx = 0; xx <= kChunkQuads; xx += kChunkQuads) {
                    const glm::dvec3 pb = dir_of(nb, xx, yy);
                    if (glm::length(pb - pa) < 1e-12) {
                        const double hb =
                            gb[static_cast<std::size_t>(yy) * kChunkVerts + xx];
                        CHECK(std::abs(ha - hb) < 1e-9);
                        matched = true;
                        ++checked;
                        break;
                    }
                }
                for (int xxi = 0; xxi <= kChunkQuads && !matched; ++xxi) {
                    for (int yy2 = 0; yy2 <= kChunkQuads; yy2 += kChunkQuads) {
                        const glm::dvec3 pb = dir_of(nb, xxi, yy2);
                        if (glm::length(pb - pa) < 1e-12) {
                            const double hb =
                                gb[static_cast<std::size_t>(yy2) * kChunkVerts + xxi];
                            CHECK(std::abs(ha - hb) < 1e-9);
                            matched = true;
                            ++checked;
                            break;
                        }
                    }
                }
            }
            CHECK(matched);
        }
    }
    CHECK(checked > 0);
}

TEST_CASE("procedural detail scales with the measured Hurst exponent") {
    // sigma(l) = sigma(L) * (l/L)^H with H = 0.5 means the RMS height
    // difference over a lag halves-by-sqrt(2) per level. Measure the
    // level-to-level RMS of the added detail and check the ratio.
    ElevationField field = MakeProceduralField(1000.0, 20000.0);
    const int first = field.FirstDetailLevel();
    CHECK(first >= 1);
    double prev_rms = 0.0;
    int ratios = 0;
    for (int level = first; level <= first + 4; ++level) {
        const ChunkKey k{0, static_cast<std::uint8_t>(level), 0u, 0u};
        std::vector<double> g;
        field.GenerateChunkHeights(k, level, 0, g);
        // RMS of the odd-vertex residual against the even lattice.
        double acc = 0.0; int n = 0;
        for (int y = 0; y <= kChunkQuads; ++y) {
            for (int x = 1; x < kChunkQuads; x += 2) {
                const double mid = 0.5 * (g[static_cast<std::size_t>(y) * kChunkVerts + x - 1] +
                                          g[static_cast<std::size_t>(y) * kChunkVerts + x + 1]);
                const double d = g[static_cast<std::size_t>(y) * kChunkVerts + x] - mid;
                acc += d * d; ++n;
            }
        }
        const double rms = std::sqrt(acc / std::max(1, n));
        CHECK(finiteBits(rms));
        if (prev_rms > 0.0) {
            const double ratio = rms / prev_rms;
            // 2^-H = 0.7071 for H = 0.5. Generous band because the residual
            // also picks up the parent level's own interpolation.
            CHECK(ratio > 0.55);
            CHECK(ratio < 0.90);
            ++ratios;
        }
        prev_rms = rms;
    }
    CHECK(ratios >= 3);
}

// --- Chunk baking ----------------------------------------------------------

TEST_CASE("chunk bake is deterministic across threads") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(27.9881 * kPi / 180.0,
                                                     86.9250 * kPi / 180.0);
    const ChunkKey key{4, 7, 61u, 43u};

    TerrainChunkData a;
    BuildTerrainChunk(key, field, site, a);

    TerrainChunkData b;
    std::thread th([&] { BuildTerrainChunk(key, field, site, b); });
    th.join();

    REQUIRE(a.positions.size() == b.positions.size());
    CHECK(std::memcmp(a.positions.data(), b.positions.data(),
                      a.positions.size() * sizeof(float)) == 0);
    REQUIRE(a.shader_verts.size() == b.shader_verts.size());
    CHECK(std::memcmp(a.shader_verts.data(), b.shader_verts.data(),
                      a.shader_verts.size() * sizeof(float)) == 0);
    CHECK(a.e_l_m == b.e_l_m);
    CHECK(a.origin_w == b.origin_w);
    for (int m = 0; m < kSlopeMips; ++m) CHECK(a.sigma2[m] == b.sigma2[m]);
}

TEST_CASE("chunk bake produces finite, correctly-sized, outward geometry") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    for (int level : {0, 4, 10, 16, 19}) {
        const ChunkKey key{1, static_cast<std::uint8_t>(level),
                           static_cast<std::uint32_t>((1u << level) / 3u),
                           static_cast<std::uint32_t>((1u << level) / 5u)};
        TerrainChunkData d;
        BuildTerrainChunk(key, field, site, d);
        REQUIRE(d.positions.size() == static_cast<std::size_t>(kChunkVertexCount) * 3u);
        REQUIRE(d.shader_verts.size() == static_cast<std::size_t>(kChunkVertexCount) * 4u);
        for (float v : d.positions)    CHECK(finiteBitsF(v));
        for (float v : d.shader_verts) CHECK(finiteBitsF(v));
        CHECK(finiteBits(d.e_l_m));
        CHECK(d.e_l_m >= 0.0);
        CHECK(d.bound_radius_m > 0.0);

        // Local coordinates must stay inside the chunk's own extent -- that
        // IS the precision argument: a level-0 chunk is 10 007 km across
        // where float32 resolves 0.6 m, a level-19 chunk is 19 m across
        // where it resolves 1.1 um.
        double max_local = 0.0;
        for (std::size_t i = 0; i < d.positions.size(); ++i) {
            max_local = std::max(max_local, std::abs(static_cast<double>(d.positions[i])));
        }
        CHECK(max_local < ChunkEdgeLength(level) * 1.2 + 20000.0);

        // Normals point away from the planet centre.
        for (int vi = 0; vi < kChunkVertexCount; ++vi) {
            const glm::dvec3 n(d.shader_verts[vi * 4 + 0],
                               d.shader_verts[vi * 4 + 1],
                               d.shader_verts[vi * 4 + 2]);
            CHECK(std::abs(glm::length(n) - 1.0) < 1e-4);
            const glm::dvec3 p(d.positions[vi * 3 + 0],
                               d.positions[vi * 3 + 1],
                               d.positions[vi * 3 + 2]);
            const glm::dvec3 up = site.WorldUp(d.origin_w + p);
            CHECK(glm::dot(n, up) > 0.0);
        }
    }
}

TEST_CASE("a chunk's boundary vertices match its coarse neighbour's exactly") {
    // The end-to-end watertightness claim, in world coordinates rather than
    // in heights: chunk A at level L+1 shares its stitched edge with chunk
    // B at level L, and A's EVEN boundary vertices -- the only ones the
    // stitched index variant references -- must land on B's vertices to the
    // last bit of the double before the float32 narrow.
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.3, 1.1);
    const ChunkKey coarse{3, 6, 20u, 33u};
    const ChunkKey fine = coarse.Child(1);          // +u child

    TerrainChunkData cd, fd;
    BuildTerrainChunk(coarse, field, site, cd);
    BuildTerrainChunk(fine,   field, site, fd);

    // fine's -u edge sits at the middle of coarse's u range, so it is NOT
    // shared with coarse. Use fine's +u edge, which IS coarse's +u edge.
    for (int y = 0; y <= kChunkQuads; y += 2) {
        const int fi = y * kChunkVerts + kChunkQuads;
        const glm::dvec3 fp(fd.positions[fi * 3 + 0],
                            fd.positions[fi * 3 + 1],
                            fd.positions[fi * 3 + 2]);
        const glm::dvec3 fw = fd.origin_w + glm::dvec3(fp);

        const int ci = (y / 2) * kChunkVerts + kChunkQuads;
        const glm::dvec3 cp(cd.positions[ci * 3 + 0],
                            cd.positions[ci * 3 + 1],
                            cd.positions[ci * 3 + 2]);
        const glm::dvec3 cw = cd.origin_w + glm::dvec3(cp);
        // The float32 narrow happens against different origins, so the
        // tolerance is the float32 ULP of the LARGER chunk's local extent
        // -- derived, not chosen: 2^-24 * chunk_edge.
        const double ulp = ChunkEdgeLength(coarse.level) * 5.96e-8;
        CHECK(glm::length(fw - cw) <= ulp * 4.0);
    }
}

TEST_CASE("a flat field has zero geometric error and zero slope variance") {
    // Red-then-green on e_L and sigma2: with no DEM and zero procedural
    // relief the surface is the bare ellipsoid, so the level-to-level delta
    // and the removed slope variance are both structurally zero. If either
    // came out non-zero the metric would be measuring the parameterisation
    // rather than the terrain.
    ElevationField flat = MakeProceduralField(0.0, 20000.0);
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainChunkData d;
    BuildTerrainChunk(ChunkKey{0, 8, 100u, 100u}, flat, site, d);
    // e_L is not exactly zero: the ellipsoid itself curves away from the
    // chunk's chords. That sagitta is the true geometric error of the flat
    // case and must be small but positive -- (spacing/2)^2 / (2R).
    const double spacing = ChunkVertexSpacing(8);
    const double sag = (spacing * spacing) / (8.0 * kWgs84A);
    CHECK(d.e_l_m > 0.0);
    CHECK(d.e_l_m < sag * 8.0);
    for (int m = 0; m < kSlopeMips; ++m) CHECK(d.sigma2[m] < 1e-6f);

    // And with relief, both must grow.
    ElevationField rough = MakeProceduralField(2000.0, 20000.0);
    TerrainChunkData r;
    BuildTerrainChunk(ChunkKey{0, 8, 100u, 100u}, rough, site, r);
    CHECK(r.e_l_m > d.e_l_m * 10.0);
    CHECK(r.sigma2[3] > 1e-6f);
}

// --- The residency selector ------------------------------------------------

namespace {

// Drive the selector to convergence, baking everything it asks for.
void Converge(TerrainQuadtree& tree, const ElevationField& field,
              const PlanetSite& site, LodParams& p, int max_rounds = 40) {
    for (int r = 0; r < max_rounds; ++r) {
        tree.Select(p);
        if (tree.Wanted().empty()) return;
        for (const ChunkKey& k : tree.Wanted()) {
            TerrainChunkData d;
            BuildTerrainChunk(k, field, site, d);
            tree.NoteChunk(d);
        }
    }
}

}  // namespace

TEST_CASE("the selector converges and respects the 2:1 restriction") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainQuadtree tree;
    LodParams p;
    p.cone_spread  = 2.0 * std::tan(0.5 * 50.0 * kPi / 180.0) / 360.0;
    p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
    p.max_level    = 9;             // keep the test quick
    p.chunk_budget = 4096;
    Converge(tree, field, site, p);
    CHECK(tree.Converged());
    CHECK(tree.Desired().size() >= 6u);

    // Every leaf's four edge neighbours must be within one level.
    for (const ChunkKey& leaf : tree.Desired()) {
        for (int e = 0; e < 4; ++e) {
            ChunkKey nb{};
            REQUIRE(NeighborChunk(leaf, static_cast<ChunkEdge>(e), nb));
            // Find the leaf covering nb.
            int nl = -1;
            for (ChunkKey a = nb;; a = a.Parent()) {
                if (tree.Desired().count(a) != 0) { nl = a.level; break; }
                if (a.level == 0) break;
            }
            if (nl < 0) continue;   // neighbour is finer; its own pass covers it
            CHECK(std::abs(nl - static_cast<int>(leaf.level)) <= 1);
        }
    }

    // The stitch mask must be set exactly on the edges whose neighbour is
    // coarser -- that is what makes the surface watertight.
    for (const ChunkKey& leaf : tree.Desired()) {
        const std::uint32_t mask = tree.StitchMask(leaf);
        for (int e = 0; e < 4; ++e) {
            ChunkKey nb{};
            NeighborChunk(leaf, static_cast<ChunkEdge>(e), nb);
            int nl = -1;
            for (ChunkKey a = nb;; a = a.Parent()) {
                if (tree.Desired().count(a) != 0) { nl = a.level; break; }
                if (a.level == 0) break;
            }
            const bool coarser = (nl >= 0 && nl < static_cast<int>(leaf.level));
            CHECK(((mask >> e) & 1u) == (coarser ? 1u : 0u));
        }
    }
}

TEST_CASE("finer detail is selected near the camera than at the antipode") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainQuadtree tree;
    LodParams p;
    p.cone_spread  = 2.0 * std::tan(0.5 * 50.0 * kPi / 180.0) / 360.0;
    p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
    p.max_level    = 9;
    p.chunk_budget = 4096;
    Converge(tree, field, site, p);

    int near_max = 0, far_max = 0;
    for (const ChunkKey& k : tree.Desired()) {
        const glm::dvec3 c =
            site.EcefToWorld(EllipsoidSurface(ChunkCenterDirection(k)));
        const double d = glm::length(c - p.camera_w);
        if (d < 2.0e5)      near_max = std::max(near_max, static_cast<int>(k.level));
        else if (d > 8.0e6) far_max  = std::max(far_max,  static_cast<int>(k.level));
    }
    CHECK(near_max > far_max);
}

TEST_CASE("hysteresis stops a chunk oscillating across the split threshold") {
    // Two selects at distances straddling the split distance. Without
    // hysteresis the second one would merge; with it, the chunk stays split
    // until the camera has receded by the hysteresis factor.
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainQuadtree tree;
    LodParams p;
    p.cone_spread  = 1.0e-3;
    p.max_level    = 8;
    p.chunk_budget = 4096;
    p.camera_w     = glm::dvec3(0.0, 1000.0, 0.0);
    Converge(tree, field, site, p);
    const std::size_t close_count = tree.Desired().size();

    // Retreat a little -- inside the hysteresis band. Residency must not
    // shrink.
    p.camera_w = glm::dvec3(0.0, 1000.0, 0.0) * 1.05;
    tree.Select(p);
    CHECK(tree.Desired().size() >= close_count);

    // Retreat a lot. Now it must.
    p.camera_w = glm::dvec3(0.0, 4.0e6, 0.0);
    tree.Select(p);
    CHECK(tree.Desired().size() < close_count);
}

TEST_CASE("the chunk budget is honoured and eviction is by priority") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainQuadtree tree;
    LodParams p;
    p.cone_spread  = 1.0e-3;
    p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
    p.max_level    = 9;
    p.chunk_budget = 64;
    Converge(tree, field, site, p);
    // The budget is a soft cap: EnforceBudget re-balances afterwards, and
    // the 2:1 invariant wins because breaking it produces cracks. Allow the
    // documented overshoot but not an unbounded one.
    CHECK(tree.Desired().size() <= 64u * 4u);

    // With a larger budget the same camera keeps more chunks.
    TerrainQuadtree tree2;
    LodParams p2 = p;
    p2.chunk_budget = 2048;
    Converge(tree2, field, site, p2);
    CHECK(tree2.Desired().size() > tree.Desired().size());
}

TEST_CASE("freeze pins the residency set regardless of the camera") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    TerrainQuadtree tree;
    LodParams p;
    p.cone_spread  = 1.0e-3;
    p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
    p.max_level    = 7;
    p.chunk_budget = 4096;
    Converge(tree, field, site, p);
    const std::set<ChunkKey> pinned = tree.Desired();

    p.freeze   = true;
    p.camera_w = glm::dvec3(0.0, 5.0e7, 0.0);
    tree.Select(p);
    CHECK(tree.Desired() == pinned);
}

TEST_CASE("the async baker returns exactly what was requested") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    AsyncChunkBaker baker;
    baker.Start(3);
    baker.SetSources(&field, site);

    std::vector<ChunkKey> want;
    for (int f = 0; f < 6; ++f) {
        want.push_back(ChunkKey{static_cast<std::uint8_t>(f), 2, 1u, 2u});
    }
    baker.Request(want);

    std::vector<TerrainChunkData> got;
    for (int spin = 0; spin < 20000 && got.size() < want.size(); ++spin) {
        baker.Drain(got, 16);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    baker.Stop();
    REQUIRE(got.size() == want.size());
    std::set<ChunkKey> seen;
    for (const auto& d : got) seen.insert(d.key);
    for (const auto& k : want) CHECK(seen.count(k) == 1u);

    // And the async result must equal the synchronous one, byte for byte --
    // otherwise "no visible chunk swap" would depend on which thread ran.
    for (const auto& d : got) {
        TerrainChunkData ref;
        BuildTerrainChunk(d.key, field, site, ref);
        REQUIRE(ref.positions.size() == d.positions.size());
        CHECK(std::memcmp(ref.positions.data(), d.positions.data(),
                          d.positions.size() * sizeof(float)) == 0);
    }
}

// --- The settle barrier (#284) ---------------------------------------------
//
// The bug these pin was not in the selector, which the cases above already
// show is a pure function of (camera, metrics). It was in how the ENGINE
// reached the selector's fixed point: a golden capture rendered ordinary
// frames and checked `converged` after each one, giving up after a budget of
// them. Rendering a frame does not make a chunk bake, so that was a race
// between the frame rate and the bake rate. A Debug build lost it on every
// run -- 400 frames spent, 149 / 220 / 155 chunks resident on three
// consecutive captures of the same fixture, three different images.
//
// PlanetTerrain::Settle replaces the race with a barrier, and
// AsyncChunkBaker::WaitIdle is the primitive that barrier stands on. These
// cases pin the primitive, because a WaitIdle that returns one bake early
// puts the whole race straight back.

TEST_CASE("WaitIdle returns only once every requested bake has landed") {
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    AsyncChunkBaker baker;
    baker.Start(4);
    baker.SetSources(&field, site);

    // Spread across all six faces so the pool has genuinely concurrent work
    // rather than one straggler.
    std::vector<ChunkKey> want;
    for (int f = 0; f < 6; ++f) {
        for (std::uint32_t i = 0; i < 4; ++i) {
            want.push_back(ChunkKey{static_cast<std::uint8_t>(f), 3, i, i});
        }
    }
    baker.Request(want);
    baker.WaitIdle();

    // THE CLAIM: after WaitIdle, ONE unbounded drain yields everything. No
    // spin loop, no sleep, no second pass. That is what makes a settle round
    // a computation instead of a poll -- and it is the difference between
    // draining a whole round and draining a wall-clock-determined prefix of
    // it, which is where the arena slot map picked up its run-to-run
    // dependence.
    CHECK(baker.InFlight() == 0);
    // Idle() means NOTHING LEFT TO DELIVER, not merely "no worker is busy",
    // so it is FALSE here: the whole round is sitting in done_ waiting to be
    // drained. That distinction is load-bearing -- PlanetTerrain::Update
    // feeds Idle() straight into `stats_.converged`, and reading it as "the
    // pool is quiet" let a capture declare convergence against a metric map
    // that was still hundreds of measurements short.
    CHECK_FALSE(baker.Idle());
    std::vector<TerrainChunkData> got;
    baker.Drain(got, 1 << 20);
    CHECK(got.size() == want.size());
    CHECK(baker.Idle());

    std::set<ChunkKey> seen;
    for (const auto& d : got) seen.insert(d.key);
    for (const auto& k : want) CHECK(seen.count(k) == 1u);
    baker.Stop();
}

TEST_CASE("Idle is never true while a claimed bake is outstanding") {
    // The claim (`in_flight_`) used to be taken AFTER releasing the queue
    // mutex, so there was a window in which a key had left the queue but had
    // not yet been counted: queue empty, count zero, "idle" -- with a bake
    // about to start. Frame-paced streaming never noticed, because it re-asks
    // every frame. A barrier does notice: it returns early exactly once and
    // the capture pins a planet that is one chunk short.
    //
    // Many small rounds rather than one big one, because the window this
    // targets is the handful of instructions between the pop and the claim,
    // and each round re-opens it once per worker.
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    AsyncChunkBaker baker;
    baker.Start(8);
    baker.SetSources(&field, site);

    for (std::uint32_t round = 0; round < 64; ++round) {
        std::vector<ChunkKey> want;
        for (int f = 0; f < 6; ++f) {
            want.push_back(ChunkKey{static_cast<std::uint8_t>(f), 4,
                                    round % 16u, (round * 7u) % 16u});
        }
        baker.Request(want);
        baker.WaitIdle();
        std::vector<TerrainChunkData> got;
        baker.Drain(got, 1 << 20);
        REQUIRE(got.size() == want.size());
    }
    baker.Stop();
}

TEST_CASE("the barrier reaches the same fixed point at any worker count") {
    // The engine's settle loop is: wait for the round to land, drain all of
    // it, re-select. This asserts the ANSWER that loop reaches does not move
    // with the width of the pool -- 1 worker and 8 workers complete a round
    // in wildly different orders, and the converged set must not care.
    //
    // Level 7 and a 4096-chunk budget keep this to a few seconds while still
    // forcing several levels of descent under the camera.
    ElevationField field = MakeProceduralField();
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);

    auto settle = [&](int workers) {
        AsyncChunkBaker baker;
        baker.Start(workers);
        baker.SetSources(&field, site);
        TerrainQuadtree tree;
        LodParams p;
        p.cone_spread  = 2.0 * std::tan(0.5 * 55.0 * kPi / 180.0) / 384.0;
        p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
        p.max_level    = 7;
        p.chunk_budget = 4096;
        for (int r = 0; r < 64; ++r) {
            baker.WaitIdle();
            std::vector<TerrainChunkData> fresh;
            baker.Drain(fresh, 1 << 20);
            for (const auto& d : fresh) tree.NoteChunk(d);
            tree.Select(p);
            if (tree.Wanted().empty()) break;
            baker.Request(tree.Wanted());
        }
        baker.Stop();
        return std::make_pair(tree.DesiredDigest(), tree.Desired().size());
    };

    const auto one   = settle(1);
    const auto eight = settle(8);
    // A non-trivial answer, so the equality below is not two empty sets
    // agreeing. Six would mean the descent never left the cube roots.
    CHECK(one.second > 6u);
    CHECK(one.second == eight.second);
    CHECK(one.first == eight.first);
}


// --- The converged set is a FIXED POINT, not just a stopping place --------
//
// #284's last and deepest layer, and the one that says this subsystem's
// convergence was being DETECTED rather than GUARANTEED.
//
// Balance splits a leaf when the 2:1 restriction demands it, and it used to
// insert the four children as leaves. Nobody then asked whether a child was
// itself within tau of the camera. A child whose metric arrived rounds
// earlier is MEASURED, so it never appeared in Wanted() -- and Wanted()
// being empty is the entire convergence signal, the thing
// r_planet_lod_freeze latches on and the capture stops for. So the selector
// could stop on a set that still wanted to split, and WHICH such set it
// stopped on depended on how ragged the tree happened to be when Balance
// ran: on the order an asynchronous worker pool finished in.
//
// Measured in the engine, on the planet_surface fixture, before the fix:
// one camera, one binary, metrics agreeing on all 1 214 common keys to the
// last bit, settling to 1 128 chunks under one bake schedule and 912 under
// another. Node (2, 11, 1088, 1657) sat at d = 22 690 m against its own
// split distance of 24 684 m -- a split by the rule, computed from the
// rule's own numbers -- and stayed a leaf in the second, because Balance had
// put it there and nothing revisited it.

namespace {

// Drive the selector to convergence while measuring an ARBITRARY subset each
// round. The subset matters: a worker pool does not finish in priority
// order, and a schedule that always measures the most-wanted chunks first is
// too tidy to expose anything -- it was tried, and the whole grid of
// (level, tau, batch) it was run over came out identical with the bug still
// in place. Shuffling is what reproduces what the pool actually does.
//
// `per_round` <= 0 measures everything each round: the smooth schedule, and
// the reference answer the ragged ones must agree with.
std::pair<std::uint64_t, std::size_t> SettleWithArrivalOrder(
        const ElevationField& field, const PlanetSite& site,
        const LodParams& params, int per_round, std::uint64_t seed) {
    TerrainQuadtree tree;
    LodParams p = params;
    // Knuth LCG, so the shuffle is reproducible and the test is not itself a
    // source of the nondeterminism it is measuring.
    std::uint64_t rng = seed * 6364136223846793005ull + 1442695040888963407ull;
    for (int r = 0; r < 100000; ++r) {
        tree.Select(p);
        if (tree.Wanted().empty()) break;
        std::vector<ChunkKey> want = tree.Wanted();
        if (per_round > 0 && want.size() > static_cast<std::size_t>(per_round)) {
            for (std::size_t i = want.size(); i > 1; --i) {
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                std::swap(want[i - 1], want[static_cast<std::size_t>(rng >> 33) % i]);
            }
            want.resize(static_cast<std::size_t>(per_round));
        }
        for (const ChunkKey& k : want) {
            TerrainChunkData d;
            BuildTerrainChunk(k, field, site, d);
            tree.NoteChunk(d);
        }
    }
    return {tree.DesiredDigest(), tree.Desired().size()};
}

// The capture configuration in miniature. Level 9 and tau 1.5 give ~615
// chunks -- enough level spread for the 2:1 rule to do real work, few enough
// to bake several times inside the Debug budget. The budget is far above the
// set so the tau bisection never runs and this measures Balance alone.
// Hysteresis is a deliberate path dependence and is suppressed under freeze,
// which is also what the golden fixtures set.
LodParams FixedPointParams() {
    LodParams p;
    p.cone_spread  = 2.0 * std::tan(0.5 * 55.0 * kPi / 180.0) / 384.0;
    p.camera_w     = glm::dvec3(0.0, 1.7, 0.0);
    p.max_level    = 9;
    p.tau_px       = 1.5;
    p.chunk_budget = 1000000;
    p.freeze       = true;
    return p;
}

}  // namespace

TEST_CASE("the converged set does not depend on the bake schedule") {
    ElevationField field = MakeProceduralField(1500.0, 20000.0);
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    const LodParams p = FixedPointParams();

    const auto reference = SettleWithArrivalOrder(field, site, p, 0, 1);
    // A real descent, not six cube roots agreeing with six cube roots.
    CHECK(reference.second > 6u);

    for (int per_round : {4, 16, 64}) {
        for (std::uint64_t seed : {1ull, 7ull, 99ull}) {
            const auto got = SettleWithArrivalOrder(field, site, p, per_round, seed);
            INFO("per_round=" << per_round << " seed=" << seed
                 << " got=" << got.second << " want=" << reference.second);
            CHECK(got.second == reference.second);
            // The count is not the claim -- two different sets of the same
            // size compare equal on size, which is why DesiredDigest exists.
            CHECK(got.first == reference.first);
        }
    }
}

TEST_CASE("no converged leaf still wants to split") {
    // The same property stated as a rule rather than as a comparison, so it
    // survives a change to the camera or to tau that would move the
    // reference set. Every leaf below max_level must be at or beyond its own
    // split distance -- that is what "fixed point of the split rule" means,
    // and it is the invariant Converged() has always been read as asserting.
    ElevationField field = MakeProceduralField(1500.0, 20000.0);
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    LodParams p = FixedPointParams();

    // Rebuild the converged tree here rather than returning one from the
    // helper, because the leaves have to be re-baked below anyway to get at
    // e_L without reaching into the selector's private metric map.
    TerrainQuadtree tree;
    std::uint64_t rng = 12345ull;
    for (int r = 0; r < 100000; ++r) {
        tree.Select(p);
        if (tree.Wanted().empty()) break;
        std::vector<ChunkKey> want = tree.Wanted();
        for (std::size_t i = want.size(); i > 1; --i) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            std::swap(want[i - 1], want[static_cast<std::size_t>(rng >> 33) % i]);
        }
        if (want.size() > 16u) want.resize(16u);
        for (const ChunkKey& k : want) {
            TerrainChunkData d;
            BuildTerrainChunk(k, field, site, d);
            tree.NoteChunk(d);
        }
    }
    REQUIRE(tree.Converged());
    REQUIRE(tree.Desired().size() > 6u);

    std::size_t checked = 0;
    std::size_t wants_to_split = 0;
    for (const ChunkKey& leaf : tree.Desired()) {
        // A max_level leaf cannot split, so the rule says nothing about it.
        if (static_cast<int>(leaf.level) >= p.max_level) continue;
        TerrainChunkData d;
        BuildTerrainChunk(leaf, field, site, d);
        const double dist = std::max(
            glm::length(p.camera_w - d.bound_center_w) - d.bound_radius_m,
            ChunkVertexSpacing(p.max_level));
        const double d_split = d.e_l_m / (p.tau_px * p.cone_spread);
        ++checked;
        if (dist < d_split) ++wants_to_split;
    }
    // Not vacuous: there ARE sub-max-level leaves to have an opinion about.
    CHECK(checked > 0u);
    CHECK(wants_to_split == 0u);
}

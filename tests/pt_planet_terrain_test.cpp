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
#include "renderer/Planet/SurfaceAlbedo.h"
#include "renderer/Planet/TerrainQuadtree.h"
#include "renderer/Planet/TerrainResidency.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
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
    //
    // The level range deliberately STRADDLES the hillslope scale break
    // (#304). With a 20 km data floor the break at 106 m falls between
    // level 10 (152.7 m spacing) and level 11 (76.4 m), and the amplitude
    // law changes shape there; consistency across the break is exactly the
    // property a two-regime law could plausibly lose, so the loop runs to
    // level 13 (19.1 m) rather than stopping short of it.
    ElevationField field = MakeProceduralField(1200.0, 20000.0);
    for (int level = 2; level <= 13; ++level) {
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

// --- The relief plane: the second moment area-averaging destroys (#318) ----

namespace {

// A deterministic, non-periodic per-cell roughness in metres. Non-periodic
// matters: the relief formula is a fixed-lag central difference, so anything
// periodic at that lag is invisible to it -- this roughness has genuine
// point-to-point difference at the lag, so it is real relief, while its
// footprint MEAN is ~flat, so the area average throws it away. That
// asymmetry (real relief, zero mean) is the whole of #318, distilled.
double FixtureRough(int x, int y) {
    std::uint32_t h = static_cast<std::uint32_t>(x) * 0x9e3779b9u
                    ^ static_cast<std::uint32_t>(y + 1) * 0x85ebca6bu;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    const double u = static_cast<double>(h >> 8) * (1.0 / 16777216.0);  // [0,1)
    return (u - 0.5) * 2.0 * 2000.0;                                    // +-2000 m
}

std::uint16_t QuantAffine(double m, double scale, double offset) {
    const double q = std::round((m - offset) / scale);
    return static_cast<std::uint16_t>(std::clamp(q, 0.0, 65535.0));
}

// Write a .ptdem exactly as fetch_planet_dem.py bakes it: 40-byte header,
// the elevation plane, and (v2 only) the relief plane. The field order here
// is the DemHeader layout, which is padding-free at 40 bytes (static_assert
// in ElevationField.h), so this is byte-identical to the tool's output.
void WritePtdem(const std::string& path, bool v2,
                std::uint32_t w, std::uint32_t h,
                const std::vector<double>& elev_m,
                const std::vector<double>& relief_m) {
    std::ofstream f(path, std::ios::binary);
    f.write(v2 ? kDemMagic : kDemMagicV1, 8);
    f.write(reinterpret_cast<const char*>(&w), 4);
    f.write(reinterpret_cast<const char*>(&h), 4);
    const double scale = kDemScaleM, off = kDemOffsetM;
    f.write(reinterpret_cast<const char*>(&scale), 8);
    f.write(reinterpret_cast<const char*>(&off), 8);
    const std::uint32_t flags = v2 ? kDemFlagReliefPlane : 0u;
    const std::uint32_t reserved = 0u;
    f.write(reinterpret_cast<const char*>(&flags), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    for (double m : elev_m) {
        const std::uint16_t v = QuantAffine(m, kDemScaleM, kDemOffsetM);
        f.write(reinterpret_cast<const char*>(&v), 2);
    }
    if (v2) {
        for (double m : relief_m) {
            const std::uint16_t v = QuantAffine(m, kDemReliefScaleM, kDemReliefOffsetM);
            f.write(reinterpret_cast<const char*>(&v), 2);
        }
    }
}

// Build a full-resolution field with genuine but mean-free sub-texel relief,
// and reproduce both bake pipelines from it: the decimated elevation (area
// average) and the TRUE relief (compute_relief's statistic on full res).
struct Fixture {
    int OW, OH, D, FW, FH;
    std::vector<double> dec;     // decimated elevation, OW*OH
    std::vector<double> truer;   // true full-res relief, OW*OH
};

Fixture MakeReliefFixture(int OW = 16, int OH = 8, int D = 12) {
    Fixture fx{OW, OH, D, OW * D, OH * D, {}, {}};
    const int FW = fx.FW, FH = fx.FH;
    std::vector<double> full(static_cast<std::size_t>(FW) * FH);
    for (int y = 0; y < FH; ++y)
        for (int x = 0; x < FW; ++x)
            full[static_cast<std::size_t>(y) * FW + x] = 1000.0 + FixtureRough(x, y);

    // Decimate by area average -- resample()'s block mean, exactly.
    fx.dec.assign(static_cast<std::size_t>(OW) * OH, 0.0);
    for (int J = 0; J < OH; ++J)
        for (int I = 0; I < OW; ++I) {
            double s = 0.0;
            for (int yy = J * D; yy < (J + 1) * D; ++yy)
                for (int xx = I * D; xx < (I + 1) * D; ++xx)
                    s += full[static_cast<std::size_t>(yy) * FW + xx];
            fx.dec[static_cast<std::size_t>(J) * OW + I] = s / (D * D);
        }

    // True relief on the full-res grid -- compute_relief()'s statistic:
    // central diff at lag D (lon wraps, lat clamps), 0.5*(gE^2+gN^2), RMS
    // over the footprint.
    auto at = [&](int x, int y) {
        const int xi = ((x % FW) + FW) % FW;         // lon wraps
        const int yi = std::clamp(y, 0, FH - 1);     // lat clamps
        return full[static_cast<std::size_t>(yi) * FW + xi];
    };
    fx.truer.assign(static_cast<std::size_t>(OW) * OH, 0.0);
    for (int J = 0; J < OH; ++J)
        for (int I = 0; I < OW; ++I) {
            double acc = 0.0;
            for (int yy = J * D; yy < (J + 1) * D; ++yy)
                for (int xx = I * D; xx < (I + 1) * D; ++xx) {
                    const double gE = 0.5 * (at(xx + D, yy) - at(xx - D, yy));
                    const double gN = 0.5 * (at(xx, yy + D) - at(xx, yy - D));
                    acc += 0.5 * (gE * gE + gN * gN);
                }
            fx.truer[static_cast<std::size_t>(J) * OW + I] = std::sqrt(acc / (D * D));
        }
    return fx;
}

}  // namespace

TEST_CASE("the relief plane restores the second moment area-averaging destroys (#318)") {
    // #318: fetch_planet_dem.py area-averages ETOPO to the output grid. An
    // area average preserves each texel's MEAN (so the heights validate) and
    // low-passes away its inter-texel VARIANCE -- sigma(L_dem), the relief the
    // whole fractal continuation is anchored on. Deriving relief from the
    // decimated grid afterwards (the pre-#318 BuildReliefMap path) cannot
    // recover it. The fix measures relief on the full-resolution grid and
    // carries it in a second plane. This asserts on that second moment
    // directly -- the quantity the issue says nothing measured.
    const Fixture fx = MakeReliefFixture();
    const std::string v1 = "pt_dem318_v1.ptdem";   // legacy: no relief plane
    const std::string v2 = "pt_dem318_v2.ptdem";   // fixed: relief plane
    WritePtdem(v1, /*v2=*/false, fx.OW, fx.OH, fx.dec, {});
    WritePtdem(v2, /*v2=*/true,  fx.OW, fx.OH, fx.dec, fx.truer);

    DigitalElevationModel dem1, dem2;
    std::string err;
    REQUIRE(dem1.Load(v1, err));
    REQUIRE(dem2.Load(v2, err));
    // The two ship the SAME elevation grid; only the relief source differs.
    CHECK_FALSE(dem1.ReliefFromFile());   // derived from the decimated grid
    CHECK(dem2.ReliefFromFile());         // read from the plane

    // Sample relief at an interior texel centre. SampleAt lands exactly on the
    // texel there (tx = ty = 0), so it reads that texel's stored/derived value.
    const int I0 = 8, J0 = 4;
    const double lon = -kPi + (I0 + 0.5) * 2.0 * kPi / fx.OW;
    const double lat =  0.5 * kPi - (J0 + 0.5) * kPi / fx.OH;
    double h1 = 0, r1 = 0, h2 = 0, r2 = 0;
    dem1.SampleAt(lat, lon, h1, r1);
    dem2.SampleAt(lat, lon, h2, r2);
    const double truth = fx.truer[static_cast<std::size_t>(J0) * fx.OW + I0];

    // The heights are byte-for-byte the same data both ways -- the average
    // preserved the mean, which is why height-based validation never caught
    // this. If the fix moved the heights it would move every terrain golden
    // for the wrong reason.
    CHECK(std::abs(h1 - h2) < 1e-9);

    // (1) The plane returns the true full-resolution relief within one
    //     quantisation step. The format round-trips the second moment intact.
    CHECK(std::abs(r2 - truth) <= 1.5 * kDemReliefScaleM);   // ~0.45 m

    // (2) THE BUG, asserted: differencing the decimated grid suppresses the
    //     relief to a small fraction of the truth -- the mean threw the
    //     sub-texel variance away. The margin here is ~20x, so this is not a
    //     tuned edge; it is the low-pass filter, measured.
    CHECK(r1 < 0.25 * truth);

    // (3) THE FIX, asserted, and the red/green line: relief the old pipeline
    //     could produce (r1) fails this; relief the new pipeline carries (r2)
    //     passes it. A test that would FAIL on the old area-averaged pipeline.
    CHECK(r2 > 3.0 * r1);

    std::remove(v1.c_str());
    std::remove(v2.c_str());
}

TEST_CASE("the relief plane preserves level-consistency and bake determinism (#318)") {
    // Relief now enters the field from a file plane rather than from
    // BuildReliefMap, so re-establish -- with a real DEM loaded -- the two
    // invariants the whole scheme rests on: a level-L chunk and its level-
    // (L+1) child are BIT-IDENTICAL at shared vertices, and a bake is a pure
    // function of the key. Relief is a continuous bilinear pure function of
    // direction regardless of its source, so this holds; verified, not
    // assumed. (Seam determinism follows by the same mechanism: at a seam the
    // two faces sample the same direction -> same lat/lon -> same bilinear
    // relief, exactly as the "agrees across a cube seam" case shows.)
    const Fixture fx = MakeReliefFixture();
    const std::string v2 = "pt_dem318_consistency.ptdem";
    WritePtdem(v2, /*v2=*/true, fx.OW, fx.OH, fx.dec, fx.truer);
    DigitalElevationModel dem;
    std::string err;
    REQUIRE(dem.Load(v2, err));
    REQUIRE(dem.ReliefFromFile());

    ElevationField field;
    field.SetDem(&dem);
    ElevationParams p;
    field.SetParams(p);
    REQUIRE(field.HasData());

    for (int level = 3; level <= 8; ++level) {
        const ChunkKey parent{2, static_cast<std::uint8_t>(level), 1u, 1u};
        std::vector<double> pg, pg2;
        field.GenerateChunkHeights(parent, level, 0, pg);
        field.GenerateChunkHeights(parent, level, 0, pg2);
        REQUIRE(pg.size() == static_cast<std::size_t>(kChunkVerts) * kChunkVerts);
        // Determinism: the same key bakes to identical bytes.
        for (std::size_t i = 0; i < pg.size(); ++i) CHECK(pg[i] == pg2[i]);

        for (int q = 0; q < 4; ++q) {
            const ChunkKey child = parent.Child(q);
            std::vector<double> cg;
            field.GenerateChunkHeights(child, static_cast<int>(child.level), 0, cg);
            const int ox = (q & 1) * (kChunkQuads / 2);
            const int oy = ((q >> 1) & 1) * (kChunkQuads / 2);
            for (int y = 0; y <= kChunkQuads; y += 2)
                for (int x = 0; x <= kChunkQuads; x += 2) {
                    const double a = cg[static_cast<std::size_t>(y) * kChunkVerts + x];
                    const double b = pg[static_cast<std::size_t>(oy + y / 2) * kChunkVerts
                                        + (ox + x / 2)];
                    CHECK(a == b);   // bit-identical across levels
                }
        }
    }
    std::remove(v2.c_str());
}

TEST_CASE("procedural detail scales with the measured Hurst exponent") {
    // sigma(l) = sigma(L) * (l/L)^H with H = 0.5 means the RMS height
    // difference over a lag halves-by-sqrt(2) per level. Measure the
    // level-to-level RMS of the added detail and check the ratio.
    //
    // This covers the ABOVE-break regime only: with a 20 km data floor,
    // levels 3..7 span 19.5 km down to 1.2 km, two decades clear of the
    // 106 m hillslope break, so the local exponent is H to within 8%. The
    // below-break regime is the subject of the next test.
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

// --- The hillslope scale break (#304) --------------------------------------

TEST_CASE("the structure function is a broken power law with the documented break") {
    // A direct test of the physics rather than of its consequences. S(l)
    // must (a) be EXACTLY 1 at the data floor, so the continuation joins
    // the DEM without a step at the seam between measurement and model;
    // (b) tend to the fluvial exponent H_coarse well above the break;
    // (c) tend to the diffusive exponent H_fine well below it; and (d)
    // have local log-log slope exactly (H_coarse + H_fine)/2 AT the break
    // -- which is what makes L_b "the break length" by definition instead
    // of by fitting. Perron, Kirchner & Dietrich 2008, JGR 113:F04003.
    const double L  = 19546.0;   // earth_lite's data floor, metres
    const double Lb = 106.0;     // the break, see ElevationParams
    const double Hc = 0.5, Hf = 1.0;

    CHECK(RelativeStructureFunction(L, L, Lb, Hc, Hf) == 1.0);   // exact

    // Centred log-log derivative. The 1e-4 step is well inside the smooth
    // region of a C-infinity function and well outside double's noise.
    auto local_exponent = [&](double l) {
        const double e = 1e-4;
        return (std::log(RelativeStructureFunction(l * (1.0 + e), L, Lb, Hc, Hf)) -
                std::log(RelativeStructureFunction(l * (1.0 - e), L, Lb, Hc, Hf))) /
               std::log((1.0 + e) / (1.0 - e));
    };
    CHECK(local_exponent(Lb) == doctest::Approx(0.5 * (Hc + Hf)).epsilon(1e-6));
    // Two decades either side puts the local exponent within 1% of the
    // asymptote; the 2% band records a 2x margin on that.
    CHECK(local_exponent(Lb * 100.0) == doctest::Approx(Hc).epsilon(0.02));
    CHECK(local_exponent(Lb * 0.01)  == doctest::Approx(Hf).epsilon(0.02));

    // Monotone: refining the hierarchy never raises the RMS increment.
    double prev = RelativeStructureFunction(L, L, Lb, Hc, Hf);
    for (double l = L * 0.5; l > 0.1; l *= 0.5) {
        const double s = RelativeStructureFunction(l, L, Lb, Hc, Hf);
        CHECK(s < prev);
        CHECK(finiteBits(s));
        prev = s;
    }

    // break_m = 0 must reproduce the pre-#304 single power law bit-exactly,
    // so the A/B cvar really is an A/B and not an approximation of one.
    for (double l : {1.0, 10.0, 100.0, 1000.0}) {
        CHECK(RelativeStructureFunction(l, L, 0.0, Hc, Hf) == std::pow(l / L, Hc));
    }
}

TEST_CASE("the threshold-hillslope transfer saturates slope at S_c (#330)") {
    // SaturateHillslopeSlope passes a level's displacement through the steady
    // state of the nonlinear flux law q = K S / (1 - (S/S_c)^2) (Roering,
    // Kirchner & Dietrich 1999, WRR 35:853), so a hillslope's angle saturates
    // at the landsliding threshold instead of rising linearly with relief.
    // Six properties fix it; the physics is in the ElevationField header.
    const double sp = 10.0;         // an arbitrary level spacing, metres
    const double Sc = 0.67451;      // tan(34 deg), the asymptote slope
    const double cap = Sc * sp;     // the displacement asymptote

    // (a) EXACT identity at 0, and identity (disabled) for threshold <= 0 --
    // this is the A/B that reproduces the pre-#330 linear law bit-for-bit.
    CHECK(SaturateHillslopeSlope(0.0, sp, Sc) == 0.0);
    for (double d : {-3.0, 0.5, 42.0}) {
        CHECK(SaturateHillslopeSlope(d, sp, 0.0) == d);
        CHECK(SaturateHillslopeSlope(d, sp, -1.0) == d);
    }

    // (b) Identity to a fraction of a percent for sub-threshold ground -- what
    // keeps Kansas, the Sahara and the abyssal plains bit-unchanged. At a
    // tenth of S_c the deviation is the leading (S_lin/S_c)^2 term, ~1%.
    {
        const double d = 0.1 * cap;
        CHECK(SaturateHillslopeSlope(d, sp, Sc) == doctest::Approx(d).epsilon(0.02));
    }

    // (c) Odd: sign in, sign out, magnitude symmetric.
    for (double d : {0.3, 5.0, 500.0}) {
        CHECK(SaturateHillslopeSlope(-d, sp, Sc) ==
              -SaturateHillslopeSlope(d, sp, Sc));
    }

    // (d) Monotone, (e) concave for d > 0, and bounded strictly below the
    // asymptote. Concavity on uniform steps = forward differences that do not
    // increase; monotone + concave + soft cap is exactly "compress, do not
    // clip", so steeper ground stays ordered-steeper and nothing runs away.
    const double h = 0.5;
    double prev = SaturateHillslopeSlope(0.0, sp, Sc);
    double prev_diff = 1e30;
    for (double d = h; d < 400.0; d += h) {
        const double s = SaturateHillslopeSlope(d, sp, Sc);
        CHECK(s > prev);                         // monotone
        CHECK(s < cap);                          // bounded by the asymptote
        CHECK(finiteBits(s));
        const double diff = s - prev;
        CHECK(diff <= prev_diff + 1e-12);        // concave
        prev = s; prev_diff = diff;
    }

    // (f) Asymptotes AT S_c*spacing -- a huge input lands within a whisker of
    // the cap and never past it (it reaches it only in the sense that the
    // (S_c/S_lin)^n correction underflows the mantissa), so cliffs above the
    // threshold form by ACCUMULATION across levels, not from one increment.
    const double big = SaturateHillslopeSlope(1e6, sp, Sc);
    CHECK(big <= cap);
    CHECK(big > 0.999 * cap);
    // ...and at a large-but-finite input the correction is still resolvable,
    // so it is strictly below the cap: the asymptote is approached, not met.
    CHECK(SaturateHillslopeSlope(50.0 * cap, sp, Sc) < cap);
}

TEST_CASE("the saturation draws steep terrain to the threshold and leaves "
          "gentle terrain alone (#330)") {
    // The acceptance, on procedural fields so it needs no DEM: the transfer
    // must pull an overshooting high-relief reference slope down to the
    // threshold while leaving sub-threshold ground essentially untouched.
    const PlanetSite site = PlanetSite::FromGeodetic(0.3, 1.1);
    const ChunkKey key{3, 13, 4321u, 5678u};
    const double Sc = 0.67451;      // tan(34 deg)

    auto ref_rms_tan = [&](double relief, double threshold) {
        ElevationField f;
        ElevationParams p;
        p.procedural_relief_m = relief;
        p.procedural_floor_m  = 20000.0;
        p.hillslope_threshold_slope = threshold;
        f.SetParams(p);
        std::vector<double> sl;
        ReferenceSlope01(key, f, site, sl);
        double acc = 0.0;
        for (double s01 : sl) {
            const double c = std::clamp(1.0 - s01, 1e-9, 1.0);
            const double t = std::sqrt(std::max(0.0, 1.0 - c * c)) / c;
            acc += t * t;
        }
        return std::sqrt(acc / static_cast<double>(sl.size()));
    };

    // Gentle ground (100 m relief) is far below S_c, so ON and OFF agree to a
    // fraction of a percent -- the low-relief terrain classes do not move.
    const double g_on  = ref_rms_tan(100.0, Sc);
    const double g_off = ref_rms_tan(100.0, 0.0);
    CHECK(g_off > 0.02);                               // not vacuous: real slope
    CHECK(g_on == doctest::Approx(g_off).epsilon(0.03));

    // Steep ground (1500 m relief) overshoots the threshold under the linear
    // law; the saturation draws it DOWN toward S_c -- a one-way reduction that
    // shrinks the excess above the threshold, without collapsing to zero (the
    // ground is genuinely steep, so it stays above S_c). Measured on this
    // chunk: RMS slope 1.76 S_c -> 1.46 S_c, the above-threshold excess
    // 0.76 S_c -> 0.46 S_c.
    const double s_off = ref_rms_tan(1500.0, 0.0);
    const double s_on  = ref_rms_tan(1500.0, Sc);
    CHECK(s_off > 1.2 * Sc);                           // the linear law overshoots
    CHECK(s_on  < 0.85 * s_off);                       // saturation pulls it down
    CHECK(s_on  > Sc);                                 // steep terrain stays steep
    CHECK((s_on - Sc) < 0.75 * (s_off - Sc));          // the excess is drawn toward S_c
}

// The RMS of the displacement this level ADDED, read straight out of a
// baked chunk. At an odd-x, even-y vertex the interpolation stencil is
// exactly the mean of the two even-x neighbours in the same row, and with
// a procedural (zero) base height nothing else contributes -- so the
// residual against that mean IS the displacement, to the last bit.
struct LevelResidual {
    double rms = 0.0;
    double clamped_fraction = 0.0;
};

LevelResidual MeasureLevelResidual(const ElevationField& field, int level) {
    const ChunkKey k{0, static_cast<std::uint8_t>(level), 0u, 0u};
    std::vector<double> g;
    field.GenerateChunkHeights(k, level, 0, g);
    const double cap = field.Params().max_slope * ChunkVertexSpacing(level);
    double acc = 0.0;
    long n = 0, clamped = 0;
    for (int y = 0; y <= kChunkQuads; y += 2) {
        for (int x = 1; x < kChunkQuads; x += 2) {
            const double mid =
                0.5 * (g[static_cast<std::size_t>(y) * kChunkVerts + x - 1] +
                       g[static_cast<std::size_t>(y) * kChunkVerts + x + 1]);
            const double d = g[static_cast<std::size_t>(y) * kChunkVerts + x] - mid;
            acc += d * d;
            ++n;
            // The clamp saturates the residual at exactly +-cap, so an
            // engagement is detectable from the output alone.
            if (std::abs(d) >= cap * (1.0 - 1e-9)) ++clamped;
        }
    }
    LevelResidual r;
    r.rms = std::sqrt(acc / static_cast<double>(std::max<long>(n, 1)));
    r.clamped_fraction = static_cast<double>(clamped) / static_cast<double>(std::max<long>(n, 1));
    return r;
}

TEST_CASE("the continuation's added slope stops growing below the hillslope break") {
    // THE #304 regression. A single self-affine law with H = 0.5 makes the
    // added slope sigma(l)/l grow as l^-0.5, i.e. by sqrt(2) EVERY level,
    // without bound -- 16 octaves of that is why the default build looked
    // like noise. Below the hillslope break the exponent goes to 1 and the
    // added slope becomes scale-invariant instead.
    //
    // Relief is 112 m: earth_lite's own area-weighted mean inter-texel RMS
    // relief, so this is a measurement of what the shipping planet does on
    // typical ground, not of a chosen number.
    ElevationField field = MakeProceduralField(112.0, 20000.0);
    const int first = field.FirstDetailLevel();
    REQUIRE(first >= 1);

    auto added_slope = [&](int level) {
        return MeasureLevelResidual(field, level).rms / ChunkVertexSpacing(level);
    };

    // Above the break (level 10 is 152.7 m, the break is 106 m) the slope
    // still grows by 2^(1-H) = sqrt(2) per level: the fluvial regime is
    // untouched, which is the point of breaking the law rather than
    // replacing it. 12% band covers both the hash's sampling noise over
    // ~1 000 vertices and the crossover's pull on the local exponent.
    for (int level = first + 1; level <= 9; ++level) {
        const double ratio = added_slope(level) / added_slope(level - 1);
        CHECK(ratio > std::sqrt(2.0) * 0.88);
        CHECK(ratio < std::sqrt(2.0) * 1.12);
    }

    // Below it, flat. Level 13 is 19.1 m, level 18 is 0.60 m -- five
    // octaves, over which the unbroken law would multiply the slope by
    // 2^2.5 = 5.66.
    for (int level = 14; level <= 18; ++level) {
        const double ratio = added_slope(level) / added_slope(level - 1);
        CHECK(ratio > 0.90);
        CHECK(ratio < 1.10);
    }
    CHECK(added_slope(18) / added_slope(13) < 1.25);

    // And the absolute number, which is the user-visible claim: on typical
    // continental ground the continuation adds a few degrees at walking
    // scale, not a vertical wall. The unbroken law gives 45 deg here (it
    // is pinned by the clamp); measured after the break, 4.4 deg.
    const double deg = std::atan(added_slope(18)) * 180.0 / kPi;
    CHECK(deg > 1.0);      // not vacuous: there IS still detail
    CHECK(deg < 8.0);      // measured 4.4 deg, so ~1.8x margin
}

TEST_CASE("the max_slope clamp is a backstop, not the thing holding the field up") {
    // Before #304 the clamp was load-bearing: area-weighted over
    // earth_lite it fired on 31.6% of level-19 midpoints, and that
    // fraction GREW with every level because truncating a divergence is
    // all it can do. Those vertices were not fractal, they were pinned at
    // a uniform 45 deg -- the "noise" in the bug report.
    //
    // On typical ground it must now never fire, at any level.
    ElevationField typical = MakeProceduralField(112.0, 20000.0);
    for (int level = typical.FirstDetailLevel(); level <= 18; ++level) {
        const LevelResidual r = MeasureLevelResidual(typical, level);
        CHECK(r.clamped_fraction == 0.0);
    }

    // Not vacuous: at extreme relief the field IS bounded near the
    // threshold, but as of #330 it is the SATURATION that bounds it, not the
    // clamp. 2 600 m of relief per 19.5 km texel is the top of earth_lite's
    // range (max 2 662.7 m) -- the Karakoram front and the trench walls --
    // where the threshold-hillslope transfer draws every increment toward
    // S_c = tan(34 deg). The residual therefore sits just under that
    // asymptote, and the tan(45 deg) clamp above it never fires.
    ElevationField extreme = MakeProceduralField(2600.0, 20000.0);
    const LevelResidual e = MeasureLevelResidual(extreme, 18);
    const double asymptote =
        extreme.Params().hillslope_threshold_slope * ChunkVertexSpacing(18);
    CHECK(e.clamped_fraction == 0.0);   // the clamp is idle even here
    CHECK(e.rms < asymptote);           // the saturation is what bounds it
    CHECK(e.rms > 0.4 * asymptote);     // non-vacuous: the ground really is steep

    // The A/B that proves the transfer, not the clamp, is doing the bounding:
    // disable the saturation and the old hard clamp is back to firing on this
    // very same ground -- so the bound moved from truncation to physics.
    ElevationField extreme_lin = MakeProceduralField(2600.0, 20000.0);
    { ElevationParams p = extreme_lin.Params();
      p.hillslope_threshold_slope = 0.0;
      extreme_lin.SetParams(p); }
    const LevelResidual el = MeasureLevelResidual(extreme_lin, 18);
    CHECK(el.clamped_fraction > 0.1);
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
        REQUIRE(d.shader_verts.size()
                == static_cast<std::size_t>(kChunkVertexCount)
                       * static_cast<std::size_t>(kVertexPayloadFloats));
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

        // Normals point away from the planet centre, and rock01 (#307) is
        // a fraction.
        for (int vi = 0; vi < kChunkVertexCount; ++vi) {
            const auto so = static_cast<std::size_t>(vi)
                          * static_cast<std::size_t>(kVertexPayloadFloats);
            const glm::dvec3 n(d.shader_verts[so + 0],
                               d.shader_verts[so + 1],
                               d.shader_verts[so + 2]);
            CHECK(std::abs(glm::length(n) - 1.0) < 1e-4);
            const glm::dvec3 p(d.positions[vi * 3 + 0],
                               d.positions[vi * 3 + 1],
                               d.positions[vi * 3 + 2]);
            const glm::dvec3 up = site.WorldUp(d.origin_w + p);
            CHECK(glm::dot(n, up) > 0.0);
            CHECK(d.shader_verts[so + 4] >= 0.0f);
            CHECK(d.shader_verts[so + 4] <= 1.0f);
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

// ===========================================================================
// Chunk residency: the published cover and the retirement rule
// ===========================================================================
//
// The streamer used to retire every chunk the selector stopped wanting in one
// unconditional step while the replacements were paced over tens of frames,
// so the terrain holed out along the LOD boundary on every camera motion.
// pt::planet::ComputeResidencyCover is the fix's decision procedure, and it
// is pure -- two key sets in, a cover and a retirement list out -- precisely
// so that the claims below are equalities over the whole domain rather than
// impressions of a picture.
//
// The end-to-end statement (tick PlanetTerrain::Update at a realistic
// blas_budget_ms and watch coverage hold every frame) lives in
// pt_planet_residency, which needs an RHI device. What is here is the policy
// itself, which needs nothing, plus the paced-stream contrast that proves
// the coverage assertion is capable of failing.

namespace {

// A complete, 2:1-balanced leaf partition of the sphere at one uniform level.
std::set<ChunkKey> UniformLeaves(int level) {
    std::set<ChunkKey> out;
    const std::uint32_t span = 1u << level;
    for (int f = 0; f < 6; ++f) {
        for (std::uint32_t i = 0; i < span; ++i) {
            for (std::uint32_t j = 0; j < span; ++j) {
                out.insert(ChunkKey{static_cast<std::uint8_t>(f),
                                    static_cast<std::uint8_t>(level), i, j});
            }
        }
    }
    return out;
}

// Replace `k` by its four children. The result stays 2:1 balanced when the
// input was uniform, because a one-level island is exactly the step the
// restriction permits.
std::set<ChunkKey> SplitOne(std::set<ChunkKey> leaves, const ChunkKey& k) {
    REQUIRE(leaves.erase(k) == 1u);
    for (int q = 0; q < 4; ++q) leaves.insert(k.Child(q));
    return leaves;
}

bool IsAntichainSet(const std::set<ChunkKey>& s) {
    for (const ChunkKey& k : s) {
        ChunkKey a = k;
        while (a.level > 0) {
            a = a.Parent();
            if (s.find(a) != s.end()) return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("a split holds the parent until all four children are resident") {
    const std::set<ChunkKey> before = UniformLeaves(1);
    const ChunkKey parent{2, 1, 1, 0};
    const std::set<ChunkKey> after = SplitOne(before, parent);
    REQUIRE(IsEdgeBalanced(before));
    REQUIRE(IsEdgeBalanced(after));

    // The selector has split `parent`; residency has not caught up at all.
    std::set<ChunkKey> resident = before;
    for (int landed = 0; landed <= 4; ++landed) {
        for (int q = 0; q < landed; ++q) resident.insert(parent.Child(q));
        const auto cover = ComputeResidencyCover(after, resident);

        CHECK(IsAntichainSet(cover.published));       // no coincident surfaces
        CHECK(IsEdgeBalanced(cover.published));       // no crack
        // Every desired leaf's ground is still drawn, at every stage.
        CHECK(CoversAll(cover.published, after));

        if (landed < 4) {
            // The parent is still standing in, so it is NOT retirable and it
            // IS what is on screen -- the children that have landed are held
            // back rather than drawn through it.
            CHECK(cover.published.find(parent) != cover.published.end());
            CHECK(std::find(cover.retirable.begin(), cover.retirable.end(),
                            parent) == cover.retirable.end());
            for (int q = 0; q < landed; ++q) {
                CHECK(cover.published.find(parent.Child(q)) ==
                      cover.published.end());
            }
        } else {
            // Fourth child in: the children take over and the parent becomes
            // retirable IN THE SAME PASS, so no frame ever publishes both.
            CHECK(cover.published.find(parent) == cover.published.end());
            for (int q = 0; q < 4; ++q) {
                CHECK(cover.published.find(parent.Child(q)) !=
                      cover.published.end());
            }
            CHECK(std::find(cover.retirable.begin(), cover.retirable.end(),
                            parent) != cover.retirable.end());
        }
        resident = before;
    }
}

TEST_CASE("a merge holds the children until the parent is resident") {
    const ChunkKey parent{4, 1, 0, 1};
    const std::set<ChunkKey> before = SplitOne(UniformLeaves(1), parent);
    const std::set<ChunkKey> after  = UniformLeaves(1);

    // Everything the fine set wanted is resident; the selector has merged.
    std::set<ChunkKey> resident = before;
    auto cover = ComputeResidencyCover(after, resident);
    CHECK(IsAntichainSet(cover.published));
    CHECK(IsEdgeBalanced(cover.published));
    CHECK(CoversAll(cover.published, after));
    for (int q = 0; q < 4; ++q) {
        CHECK(cover.published.find(parent.Child(q)) != cover.published.end());
        CHECK(std::find(cover.retirable.begin(), cover.retirable.end(),
                        parent.Child(q)) == cover.retirable.end());
    }

    // Three of four children retired early would be a hole; the rule does not
    // permit it, so check the partial-arrival state instead: the parent
    // lands, and now all four children go at once.
    resident.insert(parent);
    cover = ComputeResidencyCover(after, resident);
    CHECK(cover.published.find(parent) != cover.published.end());
    for (int q = 0; q < 4; ++q) {
        CHECK(cover.published.find(parent.Child(q)) == cover.published.end());
        CHECK(std::find(cover.retirable.begin(), cover.retirable.end(),
                        parent.Child(q)) != cover.retirable.end());
    }
    CHECK(IsAntichainSet(cover.published));
    CHECK(CoversAll(cover.published, after));
}

// --- #319: the whole cut, and what it costs ------------------------------
TEST_CASE("the interior of a cut is exactly (L - 6) / 3 nodes") {
    // The arena is sized from this identity rather than from a margin, so
    // it is worth stating as an arithmetic fact and then checking against a
    // set the SELECTOR actually produced -- the identity holds for a proper
    // quadtree cut and says nothing about an arbitrary key set, and
    // "TerrainQuadtree::BuildSet produces a proper cut" is the part that
    // could regress.
    //
    // Six cube-face roots, every interior node with exactly four children:
    // a forest of I interior nodes has 6 + 4I nodes in all, of which
    // L = 6 + 4I - I = 6 + 3I are leaves. So I = (L - 6) / 3, exactly.
    for (int level = 0; level <= 4; ++level) {
        const std::set<ChunkKey> leaves = UniformLeaves(level);
        const std::set<ChunkKey> anc = CutAncestors(leaves);
        CHECK(anc.size() == (leaves.size() - 6) / 3);
        // Ancestors are interior nodes: never leaves themselves, so the
        // reserve never competes with the frontier for the same key.
        for (const ChunkKey& a : anc) CHECK(leaves.find(a) == leaves.end());
        // And the set is closed upward, which is what makes it a COVER of
        // every intermediate level a merge can ask for rather than a
        // sample of them.
        for (const ChunkKey& a : anc) {
            if (a.level == 0) continue;
            CHECK(anc.find(a.Parent()) != anc.end());
        }
    }

    // A ragged cut, so the identity is not just a statement about uniform
    // grids: two chunks split, on different faces and at different levels.
    const ChunkKey a{0, 1, 0, 0};
    const ChunkKey b{3, 2, 3, 3};
    const std::set<ChunkKey> ragged =
        SplitOne(SplitOne(SplitOne(UniformLeaves(1), a), ChunkKey{3, 1, 1, 1}), b);
    CHECK(CutAncestors(ragged).size() == (ragged.size() - 6) / 3);

    // The selector's own answer, converged. This is the one that matters:
    // it is the set PlanetTerrain derives `retained_` from every frame.
    ElevationField field = MakeProceduralField(1500.0, 20000.0);
    const PlanetSite site = PlanetSite::FromGeodetic(0.0, 0.0);
    LodParams p = FixedPointParams();
    TerrainQuadtree tree;
    for (int r = 0; r < 64; ++r) {
        tree.Select(p);
        if (tree.Wanted().empty()) break;
        for (const ChunkKey& k : tree.Wanted()) {
            TerrainChunkData d;
            BuildTerrainChunk(k, field, site, d);
            tree.NoteChunk(d);
        }
    }
    REQUIRE(tree.Converged());
    REQUIRE(tree.Desired().size() > 100u);
    const std::set<ChunkKey> anc = CutAncestors(tree.Desired());
    CHECK(anc.size() == (tree.Desired().size() - 6) / 3);
    // Which is what the arena is sized to hold.
    CHECK(tree.Desired().size() + anc.size() <=
          WholeCutSlots(tree.Desired().size()));
}

TEST_CASE("the arena holds the whole cut of any set inside its leaf budget") {
    // WholeCutSlots(B) has to bound |D| + |CutAncestors(D)| for EVERY cut D
    // the selector can produce inside a budget of B, because that bound is
    // the entire reason retained ancestors do not have to compete with
    // desired leaves for slots. Checked exhaustively over the admissible
    // leaf counts rather than argued: a cut has L = 6 + 3I leaves, so the
    // worst case at budget B is the largest such L not exceeding B.
    for (std::size_t budget : {std::size_t{8}, std::size_t{64},
                               std::size_t{150}, std::size_t{224},
                               std::size_t{1024}, std::size_t{2048},
                               std::size_t{8192}}) {
        const std::size_t slots = WholeCutSlots(budget);
        CHECK(slots >= budget);
        for (std::size_t leaves = 6; leaves <= budget; leaves += 3) {
            CHECK(leaves + (leaves - 6) / 3 <= slots);
        }
        // Tight to within the budget's own rounding, not merely
        // sufficient. A cut has 6 + 3I leaves, so a budget that is not
        // itself of that form cannot be reached exactly; the slack is
        // exactly how far the budget sits above the largest cut that fits,
        // which is at most two slots and never a proportional margin.
        const std::size_t worst = budget - ((budget - 6) % 3);
        CHECK(budget - worst <= 2u);
        CHECK(slots - (worst + (worst - 6) / 3) == budget - worst);
    }
    // Degenerate floor: six roots and nothing below them need no surcharge.
    CHECK(WholeCutSlots(6) == 6u);
}

TEST_CASE("retiring what the cover leaves out takes nothing off the screen") {
    // The retirement rule is "a chunk may go once its ground is covered
    // without it". Stated as an equality: removing the whole retirement list
    // from residency must produce the SAME published cover. Anything else
    // means the list contained a chunk that was still doing work.
    const ChunkKey a{0, 1, 0, 0};
    const ChunkKey b{3, 1, 1, 1};
    const std::set<ChunkKey> fine = SplitOne(SplitOne(UniformLeaves(1), a), b);
    const std::set<ChunkKey> coarse = UniformLeaves(1);

    // Both split parents held past their children landing.
    std::set<ChunkKey> fine_plus_parents = fine;
    fine_plus_parents.insert(a);
    fine_plus_parents.insert(b);
    // Merged children held past their parent landing.
    std::set<ChunkKey> coarse_plus_children = coarse;
    for (int q = 0; q < 4; ++q) coarse_plus_children.insert(a.Child(q));

    // A pile of awkward mixtures: mid-split, mid-merge, both at once, and
    // the two COMPLETED transitions -- which are the states that actually
    // have something to retire.
    const std::vector<std::pair<std::set<ChunkKey>, std::set<ChunkKey>>> cases = {
        {fine,   coarse},
        {coarse, fine},
        {fine,   fine},
        {coarse, coarse},
        {SplitOne(UniformLeaves(1), a), fine},
        {fine,   SplitOne(UniformLeaves(1), b)},
        {fine,   fine_plus_parents},
        {coarse, coarse_plus_children},
    };
    std::size_t exercised = 0;
    for (const auto& [desired, resident] : cases) {
        const auto cover = ComputeResidencyCover(desired, resident);
        std::set<ChunkKey> kept = resident;
        for (const ChunkKey& r : cover.retirable) kept.erase(r);
        const auto after = ComputeResidencyCover(desired, kept);
        CHECK(after.published == cover.published);
        if (!cover.retirable.empty()) ++exercised;
    }
    // Not vacuous: some of those cases really did have something to retire.
    CHECK(exercised >= 2u);
}

TEST_CASE("the published cover never overlaps itself and never breaks 2:1") {
    // Sweep every single-chunk split against every partially-arrived
    // residency state, on all six faces. The two properties are what make a
    // transition safe in a path tracer: overlapping instances double-shade
    // with no depth buffer to arbitrate, and a two-level step is a crack the
    // 16-variant index arena cannot stitch.
    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next = [&rng]() {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
    };
    const std::set<ChunkKey> base = UniformLeaves(2);
    std::size_t states = 0;
    std::size_t with_substitutions = 0;
    for (int f = 0; f < 6; ++f) {
        for (int rep = 0; rep < 12; ++rep) {
            const ChunkKey k{static_cast<std::uint8_t>(f), 2,
                             static_cast<std::uint32_t>(next() % 4u),
                             static_cast<std::uint32_t>(next() % 4u)};
            const std::set<ChunkKey> desired = SplitOne(base, k);
            REQUIRE(IsEdgeBalanced(desired));
            // Residency: the pre-split set, plus a random subset of the new
            // children, minus a random handful of unrelated chunks (the
            // streamer is always somewhere behind).
            std::set<ChunkKey> resident = base;
            for (int q = 0; q < 4; ++q) {
                if (next() & 1ull) resident.insert(k.Child(q));
            }
            for (int drop = 0; drop < 3; ++drop) {
                auto it = resident.begin();
                std::advance(it, static_cast<long>(next() % resident.size()));
                resident.erase(it);
            }
            const auto cover = ComputeResidencyCover(desired, resident);
            CHECK(IsAntichainSet(cover.published));
            CHECK(IsEdgeBalanced(cover.published));
            for (const ChunkKey& p : cover.published) {
                CHECK(resident.find(p) != resident.end());
            }
            ++states;
            if (cover.substitutions > 0) ++with_substitutions;
        }
    }
    CHECK(states == 72u);
    // Non-vacuous: the sweep really did produce covers that differ from the
    // desired set, which is the only case the two properties can fail in.
    CHECK(with_substitutions > 0u);
}

TEST_CASE("a paced stream keeps its coverage; immediate eviction does not") {
    // THE CONTRAST THAT MAKES THE COVERAGE ASSERTION MEAN SOMETHING.
    //
    // "Coverage never regresses" is trivially true of a stream that never
    // changes, and every planet golden settles to a fixed point before it
    // captures, so nothing in the suite could previously fail on this. Here
    // the same camera flight is run twice over the same real selector, once
    // under the coverage rule and once under the old rule of retiring
    // whatever the selector stopped wanting -- and the second one is required
    // to LOSE ground, which is the user-visible bug reproduced as a number.
    //
    // Both runs admit at most kAdmit chunks per tick, which is what pacing
    // against r_planet_blas_budget_ms does without putting a clock in the
    // test. The end-to-end version, against the real Update() and the real
    // millisecond budget, is pt_planet_residency.
    ElevationParams ep{};
    ElevationField field;
    field.SetParams(ep);
    const PlanetSite site = PlanetSite::FromGeodetic(0.4886921905584123,
                                                     1.5171188644967204);

    LodParams p{};
    p.tau_px       = 0.5;
    p.hysteresis   = 1.4;
    p.min_level    = 0;
    p.max_level    = 6;
    p.chunk_budget = 8192;                 // no budget pressure in this test
    p.cone_spread  = 2.0 * 0.5773502691896257 / 1080.0;

    constexpr int    kTicks = 40;
    constexpr int    kAdmit = 6;
    constexpr double kAltTop_m    = 400000.0;   // low Earth orbit
    constexpr double kAltBottom_m = 40000.0;

    auto altitude = [&](int tick) {
        const double t = static_cast<double>(tick) / (kTicks - 1);
        return kAltTop_m + (kAltBottom_m - kAltTop_m) * t;
    };

    // `coverage_policy = false` is the old rule, verbatim.
    auto fly = [&](bool coverage_policy) {
        TerrainQuadtree tree;
        std::set<ChunkKey> resident;
        std::set<ChunkKey> prev_published;
        int lost = 0;
        std::size_t max_desired = 0;
        int held_ticks = 0;
        for (int tick = 0; tick < kTicks; ++tick) {
            p.camera_w = glm::dvec3(0.0, altitude(tick), 0.0);
            // Measure whatever the selector asks for. The tree descends one
            // level per completed bake round by construction, so this is the
            // same staircase a real bake pool walks, minus the threads.
            for (int round = 0; round < 4; ++round) {
                tree.Select(p);
                if (tree.Wanted().empty()) break;
                for (const ChunkKey& k : tree.Wanted()) {
                    TerrainChunkData d;
                    BuildTerrainChunk(k, field, site, d);
                    tree.NoteChunk(d);
                }
            }
            tree.Select(p);
            const std::set<ChunkKey>& desired = tree.Desired();
            max_desired = std::max(max_desired, desired.size());

            std::set<ChunkKey> published;
            if (coverage_policy) {
                // Add first, then let the cover decide what is redundant.
                int admitted = 0;
                for (const ChunkKey& k : desired) {
                    if (resident.find(k) != resident.end()) continue;
                    resident.insert(k);
                    if (++admitted >= kAdmit) break;
                }
                const auto cover = ComputeResidencyCover(desired, resident);
                for (const ChunkKey& r : cover.retirable) resident.erase(r);
                published = cover.published;
                if (cover.substitutions > 0) ++held_ticks;
            } else {
                // The old rule: evict everything unwanted, then add what fits.
                for (auto it = resident.begin(); it != resident.end();) {
                    if (desired.find(*it) == desired.end()) it = resident.erase(it);
                    else ++it;
                }
                int admitted = 0;
                for (const ChunkKey& k : desired) {
                    if (resident.find(k) != resident.end()) continue;
                    resident.insert(k);
                    if (++admitted >= kAdmit) break;
                }
                published = resident;
            }
            if (!prev_published.empty() && !CoversAll(published, prev_published)) {
                ++lost;
            }
            CHECK(IsAntichainSet(published));
            CHECK(IsEdgeBalanced(published));
            prev_published = published;
        }
        struct R { int lost; std::size_t max_desired; int held_ticks; };
        return R{lost, max_desired, held_ticks};
    };

    const auto with_policy = fly(true);
    const auto without     = fly(false);

    // The camera really did move the LOD boundary, and the stream really did
    // have to hold chunks -- otherwise both runs would be green for nothing.
    CHECK(with_policy.max_desired > 24u);
    CHECK(with_policy.held_ticks > 0);

    // The bug, reproduced.
    CHECK(without.lost > 0);
    // The fix.
    CHECK(with_policy.lost == 0);
}

// ===========================================================================
// #307: the shading slope is a property of the ground, not of the LOD
// ===========================================================================
//
// #300 drove rock exposure off the interpolated MESH normal, whose slope
// baseline is two chunk cells and therefore a function of the resident
// level. The first case below reproduces that defect; the rest assert the
// property that replaced it and the two claims the replacement rests on --
// that the reference baseline sits in the fluvial regime (so the sub-break
// roughness model cannot move the land cover), and that the coarse
// area-mean branch tracks the pointwise one it prefilters.

namespace {

// The rock exposure #300 would have computed for a chunk: the ramp on the
// slope of the interpolated mesh normal. Kept here, in the tests, because
// it is the thing that had to go -- the engine no longer contains it.
std::vector<double> MeshNormalRock(const TerrainChunkData& d,
                                   const PlanetSite& site) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(kChunkVertexCount));
    for (int vi = 0; vi < kChunkVertexCount; ++vi) {
        const auto s = static_cast<std::size_t>(vi)
                     * static_cast<std::size_t>(kVertexPayloadFloats);
        const glm::dvec3 n(d.shader_verts[s + 0], d.shader_verts[s + 1],
                           d.shader_verts[s + 2]);
        const glm::dvec3 p(d.positions[vi * 3 + 0], d.positions[vi * 3 + 1],
                           d.positions[vi * 3 + 2]);
        const glm::dvec3 up = site.WorldUp(p + d.origin_w);
        out.push_back(SlopeRockFraction(
            std::clamp(1.0 - std::abs(glm::dot(n, up)), 0.0, 1.0)));
    }
    return out;
}

double VertRock(const TerrainChunkData& d, int x, int y) {
    const auto s = (static_cast<std::size_t>(y) * kChunkVerts + x)
                 * static_cast<std::size_t>(kVertexPayloadFloats);
    return d.shader_verts[s + 4];
}

double MeanOf(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

}  // namespace

TEST_CASE("rock exposure is bit-identical across LOD levels, and the mesh "
          "normal it replaced was not") {
    // 800 m of relief at a 20 km data floor: ordinary mountain terrain, and
    // enough that the ramp is engaged rather than pinned at either end --
    // which the non-vacuity checks below insist on.
    ElevationField field = MakeProceduralField(800.0, 20000.0);
    const PlanetSite site = PlanetSite::FromGeodetic(0.4, 1.1);

    const ChunkKey parent{2, 12, 1234u, 2345u};
    TerrainChunkData pd;
    BuildTerrainChunk(parent, field, site, pd);

    // 13 and 14 are one and two levels below the parent; 16 is FOUR levels
    // below the reference lattice, i.e. its mesh carries four octaves of
    // continuation that the reference operator never looks at. If any of
    // them leaked in, this is where it would show.
    for (int child_level : {13, 14, 16}) {
        const int up   = child_level - static_cast<int>(parent.level);
        const int step = 1 << up;
        // At level 16 the parent has 256 children; tiling all of them is
        // 256 chunk bakes for no extra claim. Four is enough to put
        // thousands of shared vertices under the equality.
        const int tile = std::min(step, 4);
        std::size_t shared = 0, engaged = 0;
        double lo_seen = 1.0, hi_seen = 0.0;
        double worst = 0.0;
        for (int cj = 0; cj < tile; ++cj) {
            for (int ci = 0; ci < tile; ++ci) {
                const ChunkKey child{parent.face,
                                     static_cast<std::uint8_t>(child_level),
                                     parent.i * step + ci,
                                     parent.j * step + cj};
                TerrainChunkData cd;
                BuildTerrainChunk(child, field, site, cd);
                for (int y = 0; y <= kChunkQuads; ++y) {
                    for (int x = 0; x <= kChunkQuads; ++x) {
                        const int gx = ci * kChunkQuads + x;
                        const int gy = cj * kChunkQuads + y;
                        if ((gx % step) != 0 || (gy % step) != 0) continue;
                        const double a = VertRock(pd, gx / step, gy / step);
                        const double b = VertRock(cd, x, y);
                        // EQUALITY, not a tolerance. The reference lattice is
                        // shared and the fractional coordinate of a vertex is
                        // an exact dyadic rational, so there is nothing here
                        // for a tolerance to absorb.
                        CHECK(a == b);
                        worst = std::max(worst, std::abs(a - b));
                        ++shared;
                        if (a > 0.0 && a < 1.0) ++engaged;
                        lo_seen = std::min(lo_seen, a);
                        hi_seen = std::max(hi_seen, a);
                    }
                }
            }
        }
        CHECK(worst == 0.0);
        // NOT VACUOUS. An equality over a field that is uniformly zero
        // passes without testing anything, which is this project's
        // recurring defect. Insist that the ramp is actually in its
        // transition band over a large minority of the compared vertices
        // and that the compared values span most of [0, 1].
        CHECK(shared > 250u);
        CHECK(engaged > shared / 10u);
        CHECK(lo_seen < 0.05);
        CHECK(hi_seen > 0.95);
    }

    // The defect, reproduced on the same ground: the mesh normal's own
    // slope baseline is 2 * ChunkVertexSpacing, so it reads a different
    // angle at every level and the ramp on top of it moves with the LOD.
    //
    // 250 m of relief rather than 800: at 800 m the mesh-normal ramp is
    // already SATURATED at both levels, which is exactly the state that hid
    // this defect before #304 -- 0.986 at level 13 and 1.000 at level 15 on
    // the pre-#304 continuation. A saturated pair proves nothing either
    // way, so the reproduction runs where the ramp has room to move.
    // #330's threshold-hillslope saturation trims moderate slopes, which
    // would pull this 250 m-relief reproduction's parent ramp below the
    // non-vacuity floor -- but the LOD drift this case exists to catch is a
    // property of the MESH-NORMAL ramp, orthogonal to the saturation. Disable
    // the saturation here so the reproduction tests exactly the #307
    // phenomenon on the same numbers it always did, rather than a number the
    // saturation happens to move.
    ElevationField gentle = MakeProceduralField(250.0, 20000.0);
    { ElevationParams gpar = gentle.Params();
      gpar.hillslope_threshold_slope = 0.0;
      gentle.SetParams(gpar); }
    const ChunkKey gp{2, 12, 1234u, 2345u};
    const ChunkKey gc{gp.face, 14, gp.i * 4, gp.j * 4};
    TerrainChunkData gpd, gcd;
    BuildTerrainChunk(gp, gentle, site, gpd);
    BuildTerrainChunk(gc, gentle, site, gcd);
    const double mesh_parent = MeanOf(MeshNormalRock(gpd, site));
    const double mesh_child  = MeanOf(MeshNormalRock(gcd, site));
    // Both ends have to be off the rails for the ratio to mean anything.
    CHECK(mesh_parent > 0.005);
    CHECK(mesh_child  < 0.95);
    CHECK(mesh_child > mesh_parent * 1.5);

    // And the same two chunks under the reference operator: no drift at all
    // at the shared vertices, which is the whole point.
    for (int y = 0; y <= kChunkQuads; y += 4) {
        for (int x = 0; x <= kChunkQuads; x += 4) {
            if (x > kChunkQuads / 4 || y > kChunkQuads / 4) continue;
            CHECK(VertRock(gpd, x, y) == VertRock(gcd, x * 4, y * 4));
        }
    }
}

TEST_CASE("the reference slope is linear in the field's own amplitude") {
    // A guard against the operator degenerating into something that is not
    // a slope at all -- a clamp, a constant, or a value dominated by the
    // ellipsoid rather than by the terrain. On a self-affine field the RMS
    // slope is proportional to the relief the continuation is anchored on,
    // so doubling `procedural_relief_m` must double the reference slope
    // tangent while it is small enough not to be bent by the arctangent.
    const PlanetSite site = PlanetSite::FromGeodetic(-0.3, 2.2);
    const ChunkKey key{4, 12, 900u, 700u};
    auto rms_tan = [&](double relief) {
        ElevationField f = MakeProceduralField(relief, 20000.0);
        std::vector<double> sl;
        ReferenceSlope01(key, f, site, sl);
        double acc = 0.0;
        for (double s01 : sl) {
            const double c = std::clamp(1.0 - s01, 1e-9, 1.0);
            const double t = std::sqrt(std::max(0.0, 1.0 - c * c)) / c;
            acc += t * t;
        }
        return std::sqrt(acc / static_cast<double>(sl.size()));
    };
    const double a = rms_tan(60.0);
    const double b = rms_tan(120.0);
    // Small enough that tan is still nearly its argument, large enough that
    // the measurement is not floating-point dust.
    CHECK(a > 0.005);
    CHECK(b < 0.30);
    CHECK(b / a == doctest::Approx(2.0).epsilon(0.02));

    // And with the continuation switched off the reference slope collapses
    // to the ONE residual the geometry has: world "up" is the GEOCENTRIC up
    // (TerrainChunk.h's single documented approximation) while a bare
    // ellipsoid surface follows the GEODETIC normal, and the two differ by
    // the deflection of the vertical -- at most 11.5 arcmin. So a flat
    // field reads at most 1 - cos(11.5 arcmin) and nothing more.
    //
    // Asserting the deflection rather than zero is the point: a check that
    // a flat field reads zero would also pass on an implementation that
    // returned zero unconditionally.
    constexpr double kDeflectionCeil = 5.594e-6;      // 1 - cos(11.5 arcmin)
    ElevationField flat = MakeProceduralField(0.0, 20000.0);
    std::vector<double> sl;
    ReferenceSlope01(key, flat, site, sl);
    double worst_flat = 0.0;
    for (double s01 : sl) {
        CHECK(s01 <= kDeflectionCeil);
        worst_flat = std::max(worst_flat, s01);
    }
    // 5.148e-6 on this chunk, i.e. 0.1839 deg of deflection -- present, and
    // just inside the 0.1917 deg ceiling.
    CHECK(worst_flat > 4.0e-6);
    CHECK(worst_flat == doctest::Approx(5.148e-6).epsilon(2e-3));
}

TEST_CASE("the coarse area-mean branch tracks the pointwise branch it "
          "prefilters") {
    // Chunks coarser than kRefSlopeLevel cannot carry a 180 m signal, so
    // they get the ramp's AREA MEAN against the slope distribution instead
    // of a point sample of it. The two are the two halves of one mip chain,
    // so they must agree in the mean where they meet.
    for (double relief : {800.0, 1200.0, 1800.0}) {
        ElevationField field = MakeProceduralField(relief, 20000.0);
        const PlanetSite site = PlanetSite::FromGeodetic(0.1, 0.2);
        const ChunkKey coarse{3, static_cast<std::uint8_t>(kRefSlopeLevel - 1),
                              600u, 700u};
        TerrainChunkData cd;
        BuildTerrainChunk(coarse, field, site, cd);

        double fine_sum = 0.0;
        std::size_t fine_n = 0;
        for (int dj = 0; dj < 2; ++dj) {
            for (int di = 0; di < 2; ++di) {
                const ChunkKey fine{coarse.face,
                                    static_cast<std::uint8_t>(kRefSlopeLevel),
                                    coarse.i * 2 + di, coarse.j * 2 + dj};
                TerrainChunkData fd;
                BuildTerrainChunk(fine, field, site, fd);
                for (int y = 0; y <= kChunkQuads; ++y) {
                    for (int x = 0; x <= kChunkQuads; ++x) {
                        fine_sum += VertRock(fd, x, y);
                        ++fine_n;
                    }
                }
            }
        }
        double coarse_sum = 0.0;
        std::size_t coarse_n = 0;
        for (int y = 0; y <= kChunkQuads; ++y) {
            for (int x = 0; x <= kChunkQuads; ++x) {
                coarse_sum += VertRock(cd, x, y);
                ++coarse_n;
            }
        }
        const double fine_mean   = fine_sum / static_cast<double>(fine_n);
        const double coarse_mean = coarse_sum / static_cast<double>(coarse_n);
        // The band has to be somewhere the ramp is engaged, or agreeing at
        // zero would count as agreeing.
        CHECK(fine_mean > 0.02);
        CHECK(fine_mean < 0.98);
        // 0.09 is the measured envelope, not a round number. It bounds two
        // residuals now. The first is the pre-#330 one: the field's excess
        // kurtosis against the Gaussian the Rayleigh model assumes (see
        // kRefSlopeRmsGain). The second is #330's: the coarse branch applies
        // the threshold-hillslope reduction analytically (saturating the
        // scale slope at the reference lag), while the pointwise branch gets
        // it per-vertex from the baked field, and the two agree only to the
        // extent the Rayleigh model represents the saturated distribution.
        // Re-measured through the shipped saturation, coarse runs below fine
        // by 0.004 / 0.030 / 0.065 at relief 800 / 1200 / 1800 -- the gap
        // widens with relief as the saturated tail departs from Rayleigh, and
        // 1800 m is the top of earth_lite's range, so 0.09 is a ~1.4x margin.
        CHECK(std::abs(coarse_mean - fine_mean) < 0.09);
    }
}

TEST_CASE("the area-fraction integral is a distribution, not a curve fit") {
    // Monotone, pinned at both ends, and equal to an independent
    // rectangle-rule integration of the same Rayleigh integrand. The point
    // is that RockFractionFromRmsSlope is the ramp averaged over a slope
    // distribution and can be re-derived from that statement alone.
    CHECK(RockFractionFromRmsSlope(0.0) == 0.0);
    CHECK(RockFractionFromRmsSlope(-1.0) == 0.0);
    double prev = -1.0;
    for (double s = 0.02; s < 4.0; s += 0.02) {
        const double v = RockFractionFromRmsSlope(s);
        CHECK(v >= prev);
        CHECK(v >= 0.0);
        CHECK(v <= 1.0);
        prev = v;
    }
    CHECK(RockFractionFromRmsSlope(8.0) > 0.97);

    for (double s : {0.2, 0.45, 0.7, 1.0, 1.6}) {
        const int n = 200000;
        const double hi = 20.0 * s;
        const double dt = hi / n;
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            const double t = (static_cast<double>(i) + 0.5) * dt;
            const double slope01 = 1.0 - 1.0 / std::sqrt(1.0 + t * t);
            const double pdf = t / (s * s) * std::exp(-0.5 * t * t / (s * s));
            acc += SlopeRockFraction(slope01) * pdf * dt;
        }
        CHECK(RockFractionFromRmsSlope(s) == doctest::Approx(acc).epsilon(2e-3));
        // Non-vacuous: the reference integral has to land strictly inside
        // (0, 1) for the comparison to be worth making.
        CHECK(acc > 0.001);
        CHECK(acc < 0.999);
    }
}

TEST_CASE("the reference operator is the one the threshold was measured "
          "with") {
    // Gabet, Pratt-Sitaula & Burbank (2004), Geology 32:629: a 3-arcsecond
    // (~90 m) DEM, slope from the uphill and downhill neighbours. Both
    // halves of that are pinned, because a threshold angle compared against
    // a slope measured over some other baseline is a units error and the
    // engine committed exactly that one.
    const double arcsec3_m = 3.0 * (kIuggMeanRadius * 2.0 * 3.14159265358979323846)
                           / (360.0 * 3600.0);
    CHECK(arcsec3_m == doctest::Approx(92.66).epsilon(1e-3));
    CHECK(kRefSlopeHalfLagM == doctest::Approx(90.0));

    // kRefSlopeLevel is the CLOSEST level to that grid, in log2 -- the test
    // is that no other level is closer, not that 11 is written down twice.
    const double target = std::log2(arcsec3_m);
    int best = -1;
    double best_err = 1e30;
    for (int l = 0; l <= kMaxLevel; ++l) {
        const double err = std::abs(std::log2(ChunkVertexSpacing(l)) - target);
        if (err < best_err) { best_err = err; best = l; }
    }
    CHECK(best == kRefSlopeLevel);
    CHECK(ChunkVertexSpacing(kRefSlopeLevel) == doctest::Approx(76.35).epsilon(1e-3));

    // The coarse branch's calibration, re-measured through the shipped
    // constant so a silent edit to it fails here. The number is the
    // least-squares fit of the per-axis RMS reference slope against
    // relief * S(180 m) / 180 m over 216 level-11 chunks spread across all
    // six cube faces of earth_lite.
    ElevationField field = MakeProceduralField(900.0, 20000.0);
    const double S = RelativeStructureFunction(2.0 * kRefSlopeHalfLagM,
                                               20000.0,
                                               field.Params().hillslope_break_m,
                                               field.Params().hurst,
                                               field.Params().hurst_fine);
    CHECK(S > 0.0);
    const double sigma = ReferenceRmsSlopePerAxis(field, glm::dvec3(0.0, 0.0, 1.0));
    const double gain = sigma / (900.0 * S / (2.0 * kRefSlopeHalfLagM));
    CHECK(gain == doctest::Approx(1.4689).epsilon(1e-3));
}

TEST_CASE("the reference-grid memo answers for the field it was filled from") {
    // #307's bake evaluates the reference slope on the chunk's level-11
    // ANCESTOR, which sixteen level-13 chunks share -- so the ancestor grid
    // is memoised per worker thread rather than regenerated sixteen times.
    // That memo is the one piece of state in an otherwise pure bake, and a
    // cache that answers from the wrong field is precisely the defect class
    // this file keeps catching. Three ways it could be wrong, all pinned.
    const PlanetSite site = PlanetSite::FromGeodetic(0.4, 1.1);
    const ChunkKey key{2, 13, 4936u, 9380u};      // a level-13 descendant

    // (1) A MUTATED FIELD MUST NOT BE SERVED THE OLD GRID.
    // The same ElevationField OBJECT, reconfigured in place -- which is what
    // PlanetTerrain::Configure does when a cvar moves. The address is
    // unchanged, so only the generation stamp can distinguish the two.
    ElevationField mutating = MakeProceduralField(800.0, 20000.0);
    TerrainChunkData before;
    BuildTerrainChunk(key, mutating, site, before);

    ElevationParams gentler = mutating.Params();
    gentler.procedural_relief_m = 120.0;
    const std::uint64_t gen_before = mutating.Generation();
    mutating.SetParams(gentler);
    CHECK(mutating.Generation() != gen_before);
    TerrainChunkData after;
    BuildTerrainChunk(key, mutating, site, after);

    // The value a COLD cache produces. The memo is thread_local, so a
    // freshly spawned thread has an empty one by construction -- there is
    // no entry there to be served, stale or otherwise, and the comparison
    // is therefore against the uncached path rather than against another
    // possibly-stale hit. (Computing it on THIS thread would not do: a
    // memo keyed on the chunk alone serves the same wrong grid to both
    // sides and the equality passes while testing nothing. That is not
    // hypothetical -- it is what this check did before the thread was
    // added, and the reproduction is in the commit that added it.)
    ElevationField fresh = MakeProceduralField(120.0, 20000.0);
    TerrainChunkData reference;
    {
        std::thread cold([&] { BuildTerrainChunk(key, fresh, site, reference); });
        cold.join();
    }

    std::size_t compared = 0, moved = 0, engaged_before = 0;
    for (int y = 0; y <= kChunkQuads; ++y) {
        for (int x = 0; x <= kChunkQuads; ++x) {
            const double a = VertRock(before, x, y);
            const double b = VertRock(after, x, y);
            const double c = VertRock(reference, x, y);
            CHECK(b == c);
            if (a != b) ++moved;
            if (a > 0.0 && a < 1.0) ++engaged_before;
            ++compared;
        }
    }
    // NOT VACUOUS. `b == c` over a field where the two params happen to
    // agree proves nothing, so insist the reconfiguration actually moved
    // the answer on a large fraction of the chunk AND that the pre-change
    // value was in the ramp's transition band rather than pinned at an end.
    // Without these, a memo that ignored the generation entirely would
    // still pass every equality above.
    CHECK(compared == static_cast<std::size_t>(kChunkVertexCount));
    CHECK(moved > compared / 2u);
    CHECK(engaged_before > compared / 20u);

    // (1b) A COPY IS A DIFFERENT FIELD, AND MUST STAMP AS ONE.
    // ElevationField is copied by value all over this file --
    // MakeProceduralField returns one -- and the memo keys on the
    // generation stamp ALONE, having dropped the object's address once the
    // stamp was shown to subsume it. An implicit copy constructor would
    // duplicate the stamp and hand two live objects the same identity.
    // Asserted rather than left to the reasoning that a copy would serve
    // the right grid anyway: that reasoning holds today and would fail
    // silently the moment the class gained a mutable member.
    ElevationField original = MakeProceduralField(800.0, 20000.0);
    ElevationField copied(original);
    ElevationField assigned = MakeProceduralField(120.0, 20000.0);
    assigned = original;
    CHECK(copied.Generation() != original.Generation());
    CHECK(assigned.Generation() != original.Generation());
    CHECK(assigned.Generation() != copied.Generation());
    // ...and the copy still describes the same field, so it must bake the
    // same chunk. A stamp that differs is only correct if the CONTENT
    // agrees; without this the check above would pass on a copy
    // constructor that dropped the params on the floor.
    TerrainChunkData od, cd2;
    BuildTerrainChunk(key, original, site, od);
    BuildTerrainChunk(key, copied,   site, cd2);
    std::size_t copy_engaged = 0;
    for (int y = 0; y <= kChunkQuads; ++y) {
        for (int x = 0; x <= kChunkQuads; ++x) {
            CHECK(VertRock(od, x, y) == VertRock(cd2, x, y));
            if (VertRock(od, x, y) > 0.0 && VertRock(od, x, y) < 1.0) ++copy_engaged;
        }
    }
    CHECK(copy_engaged > static_cast<std::size_t>(kChunkVertexCount) / 20u);

    // (2) TWO LIVE FIELDS MUST NOT SHARE AN ENTRY.
    // Interleaved so that each bake finds the other's entry already in the
    // cache; a memo keyed on the chunk alone would serve it.
    ElevationField steep  = MakeProceduralField(800.0, 20000.0);
    ElevationField shallow = MakeProceduralField(120.0, 20000.0);
    TerrainChunkData s1, h1, s2, h2;
    BuildTerrainChunk(key, steep,   site, s1);
    BuildTerrainChunk(key, shallow, site, h1);
    BuildTerrainChunk(key, steep,   site, s2);
    BuildTerrainChunk(key, shallow, site, h2);
    std::size_t differ = 0;
    for (int y = 0; y <= kChunkQuads; ++y) {
        for (int x = 0; x <= kChunkQuads; ++x) {
            CHECK(VertRock(s1, x, y) == VertRock(s2, x, y));
            CHECK(VertRock(h1, x, y) == VertRock(h2, x, y));
            if (VertRock(s1, x, y) != VertRock(h1, x, y)) ++differ;
        }
    }
    CHECK(differ > static_cast<std::size_t>(kChunkVertexCount) / 2u);

    // (3) THE MEMO IS SMALLER THAN THE WORKING SET, SO IT MUST EVICT
    // CORRECTLY. Sixteen distinct level-11 ancestors against a four-entry
    // cache: every bake after the first four is a miss on a full cache, and
    // a wrong eviction would hand back a NEIGHBOUR's grid rather than
    // nothing.
    //
    // Each chunk is compared against the same chunk baked on its OWN fresh
    // thread. The memo is thread_local, so that thread's cache is empty and
    // its bake is provably a miss -- the uncached answer, by construction.
    //
    // THE OBVIOUS CHEAPER TEST DOES NOT WORK, and this file should say so
    // because the trap is subtle. Baking the sixteen forwards and then
    // backwards on ONE thread and comparing the two orders passes even when
    // the ancestor coordinates are dropped from the cache key entirely:
    // the first entry stays valid across both passes, so both orders read
    // the SAME wrong grid and agree with each other. Two runs that are
    // wrong in the same way are not a check. Measured: with the ancestor
    // coordinates removed, order-vs-order gives 0 mismatches out of 16 and
    // 15 of 16 chunks still differ from the first, so even the
    // non-vacuity guard on that form is satisfied. The comparison has to be
    // against a cache that cannot hold the wrong answer, not against
    // another consultation of the one that does.
    ElevationField field = MakeProceduralField(800.0, 20000.0);
    std::size_t distinct_chunks = 0, engaged_total = 0;
    std::vector<double> first_chunk;
    for (int n = 0; n < 16; ++n) {
        // BOTH halves of the ancestor key have to be separable, and the
        // sequence is built so that each one is. The face alternates, so a
        // same-(i, j) different-face pair is live in the cache at once; the
        // coordinates advance every other chunk, so a same-face
        // different-(i, j) pair is live too, and only two apart in a
        // four-entry cache. A sweep that varied only one of the two would
        // leave the other term untestable -- which is how the first draft
        // of this case managed to pass with the coordinates removed from
        // the key entirely.
        const ChunkKey k{static_cast<std::uint8_t>(n % 2), 13,
                         4936u + static_cast<std::uint32_t>(n / 2) * 4u,
                         9380u + static_cast<std::uint32_t>(n / 2) * 4u};
        // Warm: this thread, whose cache is already full of other
        // ancestors from the bakes above and from earlier iterations.
        TerrainChunkData warm;
        BuildTerrainChunk(k, field, site, warm);
        // Cold: a thread that has never baked anything.
        TerrainChunkData cold;
        {
            std::thread t([&] { BuildTerrainChunk(k, field, site, cold); });
            t.join();
        }
        std::vector<double> w, c;
        for (int i = 0; i < kChunkVertexCount; ++i) {
            const auto o = static_cast<std::size_t>(i) * kVertexPayloadFloats + 4;
            w.push_back(warm.shader_verts[o]);
            c.push_back(cold.shader_verts[o]);
        }
        CHECK(w == c);
        if (n == 0) first_chunk = w;
        else if (w != first_chunk) ++distinct_chunks;
        for (double v : w) if (v > 0.0 && v < 1.0) ++engaged_total;
    }
    // Not vacuous: the sixteen chunks must actually differ from each other
    // -- otherwise "the cache returned the right grid" is a statement about
    // one grid -- and the ramp must be in its transition band somewhere, or
    // an all-zero field would satisfy every equality above.
    CHECK(distinct_chunks >= 14u);
    CHECK(engaged_total > 1000u);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary P4 (#258): the cubed-ellipsoid quadtree ---------------------
//
// Six root quads on a cube, tangent-warped and projected onto the WGS-84
// ellipsoid. Each quad subdivides into four, ~20 levels deep. A chunk is a
// fixed 65x65 vertex grid over its quad, displaced along the geodetic
// normal by the elevation field.
//
// WHY A CUBE AND NOT A LAT/LON GRID
// ---------------------------------
// A lat/lon grid degenerates at the poles: cells shrink to zero width and
// the quadtree's "split into four" stops meaning anything. The cube has six
// charts and no singular point; the price is 12 seam edges, which the
// neighbour query below resolves exactly.
//
// THE TANGENT WARP
// ----------------
// The naive mapping `dir = normalize(n + u*r + v*up)` for u,v in [-1,1]
// bunches samples at the face centre: the worst-case area distortion across
// a face is (sqrt(3))^3 = 5.196x. Substituting `p = tan(u * pi/4)` before
// the normalise (Lambers 2019, "Mappings between Sphere, Disc, and Square",
// JCGT 8(2); the same warp Cesium calls "tangent-adjusted") drops that to
// ~1.3x. That number is load-bearing rather than cosmetic: the LOD selector
// derives one level per chunk, so the metric spread WITHIN a chunk is the
// error the single level has to cover. 1.3x is a third of a level; 5.2x is
// 2.4 levels, which would show as visible over- and under-tessellation
// bands radiating from every face centre.
//
// EXACTNESS AT CHUNK BOUNDARIES (the watertightness argument)
// ----------------------------------------------------------
// Every vertex position is a pure function of (face, g, G) where `g` is the
// integer grid coordinate and `G = 64 * 2^level` the grid size. Both are
// powers of two times 64, so `g / G` is EXACT in double and a level-L
// vertex at grid index 2g computes bit-for-bit the same parameter as the
// level-(L-1) vertex at grid index g. Two chunks that share an edge
// therefore evaluate the elevation field at bit-identical inputs and get
// bit-identical outputs. Combined with the index-variant stitching in
// TerrainIndexArena, the surface is watertight in the strict sense -- no
// sliver, no T-junction, no epsilon.
//
// THE LEVEL TABLE (from the IUGG mean radius, 6 371 008.8 m):
//
//   Level | chunk edge  | vertex spacing | note
//   ------+-------------+----------------+---------------------------------
//     0   | 10 007 km   | 156 km         | one per cube face
//     4   |    626 km   |  9.8 km        | continent scale
//     8   |   39.1 km   |   610 m        | ETOPO 30" data floor
//    12   |   2.44 km   |  38.2 m        | SRTM 1"
//    16   |   152.7 m   |  2.39 m        |
//    19   |    19.1 m   |  0.30 m        | walking-scale detail
//
// Level 19 is the ceiling because a level-0 chunk's 156 km vertex spacing
// reaches one pixel at 2.87e8 m -- past that distance geometry of ANY level
// is pointless, which bounds the tree from the other end.

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace pt::planet {

// --- WGS-84 (NIMA TR8350.2, 3rd ed., 1997) --------------------------------
// The defining constants. `b` is derived from `a` and the flattening
// 1/f = 298.257223563 and is quoted to the digit the standard gives.
inline constexpr double kWgs84A = 6378137.0;             // semi-major, m
inline constexpr double kWgs84B = 6356752.314245;        // semi-minor, m
inline constexpr double kWgs84InvF = 298.257223563;
// Standard gravitational parameter GM, WGS-84 value including the
// atmosphere. Used by the radial-gravity integrator.
inline constexpr double kWgs84Mu = 3.986004418e14;       // m^3 / s^2

// IUGG mean (volumetric-equivalent) radius R_1 = (2a + b) / 3. This is the
// sphere the atmosphere shell uses; see Engine's planet_center_radius.
inline constexpr double kIuggMeanRadius = 6371008.8;

// Elevation bounds of the real Earth, used to size the uint16 DEM encoding
// and to place the backstop body. Challenger Deep -10 935 m (Gardner,
// Armstrong & Calder 2014, "So, How Deep Is the Mariana Trench?"); Everest
// +8 848.86 m (2020 China/Nepal joint survey).
inline constexpr double kEarthMinElevation = -10935.0;
inline constexpr double kEarthMaxElevation = 8848.86;

// Radius of the analytic backstop body the megakernel intersects wherever
// terrain streaming has published no chunk (Engine.cpp's planet_ground.w
// when terrain is on). The ellipsoid's semi-minor axis -- the global minimum
// of the WGS-84 surface -- minus Challenger Deep and a further 65 m, so the
// sphere sits inside every terrain vertex at every latitude and a streaming
// gap shows an opaque planet rather than a hole to the sky. Named here, in
// one place, because BOTH the shader upload (Engine.cpp) and the from-orbit
// terrain cull (TerrainQuadtree, #326) measure against it and must not drift.
inline constexpr double kBackstopRadius = kWgs84B + kEarthMinElevation - 65.0;

// --- Chunk grid geometry --------------------------------------------------
inline constexpr int kChunkQuads   = 64;                 // cells per edge
inline constexpr int kChunkVerts   = kChunkQuads + 1;    // 65
inline constexpr int kChunkVertexCount = kChunkVerts * kChunkVerts;      // 4225
inline constexpr int kChunkTriCount    = kChunkQuads * kChunkQuads * 2;  // 8192
inline constexpr int kChunkIndexCount  = kChunkTriCount * 3;             // 24576

// Detail ceiling. 19 gives 0.30 m vertex spacing; see the table above.
inline constexpr int kMaxLevel = 19;

// Number of baked mean-square-slope mip entries per chunk. 8 covers
// footprints from one vertex spacing (mip 0) up to 128 spacings (mip 7),
// which at level 19 is 38 m and at level 0 is 20 000 km -- i.e. the whole
// chunk. Past that the chunk is a single pixel and the LOD selector has
// long since merged it.
inline constexpr int kSlopeMips = 8;

// --- Chunk identity -------------------------------------------------------
//
// A quadtree node. `face` in [0,6), `level` in [0, kMaxLevel], and (i, j)
// in [0, 2^level) index the quad within the face. i runs along the face's
// +u axis, j along +v.
struct ChunkKey {
    std::uint8_t  face  = 0;
    std::uint8_t  level = 0;
    std::uint32_t i     = 0;
    std::uint32_t j     = 0;

    friend bool operator==(const ChunkKey& a, const ChunkKey& b) noexcept {
        return a.face == b.face && a.level == b.level && a.i == b.i && a.j == b.j;
    }
    // Total order so the selector's residency sets iterate deterministically
    // -- a std::map/std::set walk is the same on every host and every run,
    // which the golden matrix needs and an unordered_map walk cannot give.
    friend bool operator<(const ChunkKey& a, const ChunkKey& b) noexcept {
        if (a.face  != b.face)  return a.face  < b.face;
        if (a.level != b.level) return a.level < b.level;
        if (a.i     != b.i)     return a.i     < b.i;
        return a.j < b.j;
    }

    ChunkKey Parent() const noexcept {
        return (level == 0) ? *this
                            : ChunkKey{face, static_cast<std::uint8_t>(level - 1),
                                       i >> 1, j >> 1};
    }
    ChunkKey Child(int quadrant) const noexcept {
        return ChunkKey{face, static_cast<std::uint8_t>(level + 1),
                        (i << 1) | static_cast<std::uint32_t>(quadrant & 1),
                        (j << 1) | static_cast<std::uint32_t>((quadrant >> 1) & 1)};
    }
    // Grid size along one face axis at this level, in CHUNKS.
    std::uint32_t Span() const noexcept { return 1u << level; }
};

// The four edges of a chunk, in the order the stitching mask bits use.
enum ChunkEdge : int {
    kEdgeMinusU = 0,
    kEdgePlusU  = 1,
    kEdgeMinusV = 2,
    kEdgePlusV  = 3,
};

// --- Face bases -----------------------------------------------------------
//
// Face f maps (u, v) in [-1,1]^2 to the cube point
//     P = normal(f) + u * right(f) + v * up(f)
// with the three vectors an orthonormal right-handed triple. The six
// normals are the six signed axes, in the order +X, -X, +Y, -Y, +Z, -Z.
glm::dvec3 FaceNormal(int face) noexcept;
glm::dvec3 FaceRight(int face) noexcept;
glm::dvec3 FaceUp(int face) noexcept;

// --- The tangent warp -----------------------------------------------------
// s in [-1,1] -> tan(s * pi/4) in [-1,1]. Monotone, odd, exact at 0 and +-1.
double TangentWarp(double s) noexcept;
// The inverse: p in [-1,1] -> (4/pi) * atan(p).
double TangentUnwarp(double p) noexcept;

// Face parameter for grid index `g` out of `grid_size` cells. `grid_size`
// must be a power of two so the division is exact; see the watertightness
// note above.
double GridParam(std::int64_t g, std::int64_t grid_size) noexcept;

// (face, s, t) in [-1,1]^2 -> unit direction, tangent warp applied.
glm::dvec3 FaceParamToDirection(int face, double s, double t) noexcept;

// unit direction -> (face, s, t). The inverse of FaceParamToDirection;
// used to locate the chunk containing a point (e.g. the camera).
void DirectionToFaceParam(const glm::dvec3& dir_unit,
                          int& out_face, double& out_s, double& out_t) noexcept;

// --- The ellipsoid --------------------------------------------------------
//
// A unit direction `d` maps to the ellipsoid point `diag(a, a, b) * d`,
// which satisfies (x/a)^2 + (y/a)^2 + (z/b)^2 = 1 exactly. NOTE this is the
// ellipsoid's own frame: the POLAR axis is +Z, matching the ECEF
// convention. The engine's world frame is a rotation of it (see
// PlanetSite).
glm::dvec3 EllipsoidSurface(const glm::dvec3& dir_unit) noexcept;

// The geodetic normal (the "up" that ellipsoidal height is measured along)
// at an ellipsoid-frame point. For a point ON the surface this is the
// gradient of the defining quadric, normalised.
glm::dvec3 GeodeticNormal(const glm::dvec3& p_ecef) noexcept;

// The FIELD direction of an ellipsoid-frame point: the unit `d` for which
// EllipsoidSurface(d) lies on the ray from the centre through `p_ecef`.
//
// This is NOT normalize(p_ecef), and the difference is not small. The
// parameterisation is `p = diag(a, a, b) * d`, so recovering d means
// applying the INVERSE scaling first: d = normalize(p.x/a, p.y/a, p.z/b).
// Feeding a plain normalize() to the elevation field instead lands on a
// different point of the same ellipsoid -- 8.8 km away at 28 deg latitude,
// measured, which is what the first surface-placement run did to the
// camera. Every host query that converts a POSITION into a field direction
// has to go through here.
glm::dvec3 EcefToFieldDirection(const glm::dvec3& p_ecef) noexcept;

// Geodetic latitude / longitude (radians) of an ellipsoid-frame direction.
// Latitude is the GEODETIC one -- the angle of the geodetic normal, which
// differs from the geocentric angle by up to 11.5 arcmin.
void EcefToGeodetic(const glm::dvec3& p_ecef,
                    double& out_lat_rad, double& out_lon_rad) noexcept;

// Geodetic (lat, lon) -> the ellipsoid surface point in the ellipsoid frame.
glm::dvec3 GeodeticToEcef(double lat_rad, double lon_rad) noexcept;

// --- Chunk extents --------------------------------------------------------

// The four corner directions of a chunk, in the order
// (-u,-v), (+u,-v), (-u,+v), (+u,+v).
void ChunkCornerDirections(const ChunkKey& k, glm::dvec3 out[4]) noexcept;

// Centre direction of a chunk (the (s,t) midpoint, warped).
glm::dvec3 ChunkCenterDirection(const ChunkKey& k) noexcept;

// A conservative bounding sphere for the chunk's geometry in the ELLIPSOID
// frame, given the elevation range [h_min, h_max] the chunk actually spans.
// Centre is the midpoint of the corner + centre extremes; the radius is the
// max distance to any of them plus the elevation span. Used by the LOD
// selector's distance term.
void ChunkBoundingSphere(const ChunkKey& k, double h_min, double h_max,
                         glm::dvec3& out_center, double& out_radius) noexcept;

// Arc length of a chunk's edge on the mean sphere, in metres. The LOD
// table above is this function.
double ChunkEdgeLength(int level) noexcept;
// Vertex spacing = edge / 64.
double ChunkVertexSpacing(int level) noexcept;

// --- Neighbours (including across the 12 cube seams) ----------------------
//
// The chunk sharing `edge` with `k`, at the SAME level. Derived
// geometrically rather than from a hand-written 24-entry adjacency table:
// the shared edge's midpoint is a point on the cube, and re-expressing that
// point in the neighbouring face's basis gives (s', t') directly. All the
// arithmetic is on exact dyadic rationals, so the integer (i', j') comes
// out exactly.
//
// Returns false only for a level-0 key, where all six faces are roots and
// "the neighbour at the same level" is another root -- which it does return,
// so in practice this never fails. The bool exists so a future
// non-cube topology can signal "no neighbour".
bool NeighborChunk(const ChunkKey& k, ChunkEdge edge, ChunkKey& out) noexcept;

// --- The shared index arena ----------------------------------------------
//
// Chunk topology is identical for every chunk, so the index buffer is
// shared. Sixteen variants cover the 2^4 combinations of "the neighbour
// across this edge is one level COARSER". On such an edge the chunk drops
// its odd boundary vertices -- their indices are snapped to the preceding
// even vertex -- so its boundary polyline becomes exactly the coarse
// neighbour's 33-vertex polyline, sharing the same vertices bit-for-bit.
//
// This is strictly better than skirts in a ray tracer. Skirts are thin
// vertical ribbons that DO get hit by grazing rays, and a planet spends
// most of its screen area at grazing angles; they produce dark slivers
// exactly at the horizon. Stitching produces a watertight surface with no
// extra geometry.
//
// The snap leaves one degenerate (two-equal-index) triangle per dropped
// vertex: at most 32 per coarse edge, 128 of 8192 = 1.6% worst case. All
// three acceleration-structure builders accept and cull them, and a
// zero-area triangle is unhittable, so the cost is BLAS bytes rather than
// correctness. The alternative -- a bespoke pentagon triangulation per
// boundary strip plus four corner cases -- buys 1.6% of BLAS size for
// roughly 200 lines of code that has to be right in 16 combinations.
class TerrainIndexArena {
public:
    TerrainIndexArena();

    // The whole arena: 16 * kChunkIndexCount uints, variant v starting at
    // v * kChunkIndexCount.
    const std::vector<std::uint32_t>& Indices() const noexcept { return indices_; }

    static constexpr std::uint32_t kVariantCount = 16u;
    static std::uint32_t VariantOffset(std::uint32_t mask) noexcept {
        return (mask & 15u) * static_cast<std::uint32_t>(kChunkIndexCount);
    }

    // Pointer to variant `mask`'s index block.
    const std::uint32_t* Variant(std::uint32_t mask) const noexcept {
        return indices_.data() + VariantOffset(mask);
    }

    std::size_t ByteSize() const noexcept {
        return indices_.size() * sizeof(std::uint32_t);
    }

private:
    std::vector<std::uint32_t> indices_;
};

// Vertex index within a chunk's 65x65 grid, after applying the boundary
// snap for stitch variant `mask`. Exposed for the unit tests, which assert
// the snap is idempotent and that snapped boundaries agree between a chunk
// and its coarse neighbour.
std::uint32_t SnappedVertexIndex(int x, int y, std::uint32_t mask) noexcept;

}  // namespace pt::planet

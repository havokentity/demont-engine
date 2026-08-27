// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "CubedSphere.h"

#include <algorithm>
#include <cmath>

namespace pt::planet {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Face bases. Right-handed triples (right, up, normal) so that
// cross(right, up) == normal, which makes the derived vertex winding
// consistently outward-facing across all six charts.
//
//   0: +X   1: -X   2: +Y   3: -Y   4: +Z   5: -Z
constexpr double kFaceN[6][3] = {
    { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0,  1}, {0, 0, -1},
};
constexpr double kFaceR[6][3] = {
    { 0, 0,-1}, { 0, 0, 1}, {1,  0, 0}, {1,  0, 0}, {1, 0,  0}, {-1, 0, 0},
};
constexpr double kFaceU[6][3] = {
    { 0, 1, 0}, { 0, 1, 0}, {0,  0, 1}, {0,  0,-1}, {0, 1,  0}, { 0, 1, 0},
};

}  // namespace

glm::dvec3 FaceNormal(int face) noexcept {
    const int f = std::clamp(face, 0, 5);
    return glm::dvec3(kFaceN[f][0], kFaceN[f][1], kFaceN[f][2]);
}
glm::dvec3 FaceRight(int face) noexcept {
    const int f = std::clamp(face, 0, 5);
    return glm::dvec3(kFaceR[f][0], kFaceR[f][1], kFaceR[f][2]);
}
glm::dvec3 FaceUp(int face) noexcept {
    const int f = std::clamp(face, 0, 5);
    return glm::dvec3(kFaceU[f][0], kFaceU[f][1], kFaceU[f][2]);
}

double TangentWarp(double s) noexcept {
    // tan(0) == 0 and tan(pi/4) == 1 to within one ULP; pin the endpoints so
    // a face boundary lands exactly on the cube edge and two adjacent faces
    // agree there bit-for-bit.
    if (s ==  1.0) return  1.0;
    if (s == -1.0) return -1.0;
    if (s ==  0.0) return  0.0;
    if (s > -1.0 && s < 1.0) return std::tan(s * (kPi * 0.25));
    // OUTSIDE the face: continue C1 with the slope at the edge,
    // d/ds tan(s*pi/4) = (pi/4) sec^2(pi/4) = pi/2 at s = +-1. Chunk
    // generation asks for a one-cell halo so vertex normals can be central
    // differences everywhere, and at a face boundary that halo lies on the
    // neighbouring face. The linear continuation puts it on the correct
    // face at very nearly the correct place -- good enough for a normal
    // estimate, and never used for a rendered vertex.
    const double sgn = (s > 0.0) ? 1.0 : -1.0;
    return sgn * (1.0 + (std::abs(s) - 1.0) * (kPi * 0.5));
}

double TangentUnwarp(double p) noexcept {
    if (p <= -1.0) return -1.0;
    if (p >=  1.0) return  1.0;
    if (p == 0.0)  return  0.0;
    return std::atan(p) * (4.0 / kPi);
}

double GridParam(std::int64_t g, std::int64_t grid_size) noexcept {
    if (grid_size == 0) return -1.0;
    // grid_size is 64 * 2^level, i.e. a power of two, so g / grid_size is
    // exact in double for every g <= grid_size (both fit in 53 bits at
    // level 19: 64 * 2^19 = 3.36e7). The multiply by 2 is exact. Only the
    // final subtraction can round, and it rounds identically for a parent
    // and a child evaluating the same point -- which is the watertightness
    // guarantee this function exists to provide.
    return -1.0 + 2.0 * (static_cast<double>(g) / static_cast<double>(grid_size));
}

glm::dvec3 FaceParamToDirection(int face, double s, double t) noexcept {
    const double p = TangentWarp(s);
    const double q = TangentWarp(t);
    const glm::dvec3 v = FaceNormal(face) + p * FaceRight(face) + q * FaceUp(face);
    return glm::normalize(v);
}

void DirectionToFaceParam(const glm::dvec3& dir_unit,
                          int& out_face, double& out_s, double& out_t) noexcept {
    const double ax = std::abs(dir_unit.x);
    const double ay = std::abs(dir_unit.y);
    const double az = std::abs(dir_unit.z);
    int face = 0;
    if (ax >= ay && ax >= az)      face = (dir_unit.x >= 0.0) ? 0 : 1;
    else if (ay >= ax && ay >= az) face = (dir_unit.y >= 0.0) ? 2 : 3;
    else                           face = (dir_unit.z >= 0.0) ? 4 : 5;

    const glm::dvec3 n = FaceNormal(face);
    const glm::dvec3 r = FaceRight(face);
    const glm::dvec3 u = FaceUp(face);
    const double dn = glm::dot(dir_unit, n);
    if (!(std::abs(dn) > 0.0)) {           // degenerate input
        out_face = face; out_s = 0.0; out_t = 0.0;
        return;
    }
    const double p = glm::dot(dir_unit, r) / dn;
    const double q = glm::dot(dir_unit, u) / dn;
    out_face = face;
    out_s = TangentUnwarp(std::clamp(p, -1.0, 1.0));
    out_t = TangentUnwarp(std::clamp(q, -1.0, 1.0));
}

glm::dvec3 EllipsoidSurface(const glm::dvec3& dir_unit) noexcept {
    return glm::dvec3(kWgs84A * dir_unit.x,
                      kWgs84A * dir_unit.y,
                      kWgs84B * dir_unit.z);
}

glm::dvec3 GeodeticNormal(const glm::dvec3& p_ecef) noexcept {
    // Gradient of (x/a)^2 + (y/a)^2 + (z/b)^2 - 1, normalised.
    constexpr double inv_a2 = 1.0 / (kWgs84A * kWgs84A);
    constexpr double inv_b2 = 1.0 / (kWgs84B * kWgs84B);
    const glm::dvec3 g(p_ecef.x * inv_a2, p_ecef.y * inv_a2, p_ecef.z * inv_b2);
    const double len = glm::length(g);
    if (!(len > 0.0)) return glm::dvec3(0.0, 0.0, 1.0);
    return g / len;
}

glm::dvec3 EcefToFieldDirection(const glm::dvec3& p_ecef) noexcept {
    const glm::dvec3 d(p_ecef.x / kWgs84A, p_ecef.y / kWgs84A, p_ecef.z / kWgs84B);
    const double len = glm::length(d);
    if (!(len > 0.0)) return glm::dvec3(0.0, 0.0, 1.0);
    return d / len;
}

void EcefToGeodetic(const glm::dvec3& p_ecef,
                    double& out_lat_rad, double& out_lon_rad) noexcept {
    const glm::dvec3 n = GeodeticNormal(p_ecef);
    out_lat_rad = std::asin(std::clamp(n.z, -1.0, 1.0));
    out_lon_rad = std::atan2(n.y, n.x);
}

glm::dvec3 GeodeticToEcef(double lat_rad, double lon_rad) noexcept {
    // Standard geodetic -> ECEF at zero ellipsoidal height.
    //   N = a / sqrt(1 - e^2 sin^2(lat)),  e^2 = 1 - (b/a)^2
    const double f  = 1.0 / kWgs84InvF;
    const double e2 = f * (2.0 - f);
    const double sl = std::sin(lat_rad);
    const double cl = std::cos(lat_rad);
    const double N  = kWgs84A / std::sqrt(1.0 - e2 * sl * sl);
    return glm::dvec3(N * cl * std::cos(lon_rad),
                      N * cl * std::sin(lon_rad),
                      N * (1.0 - e2) * sl);
}

void ChunkCornerDirections(const ChunkKey& k, glm::dvec3 out[4]) noexcept {
    const std::int64_t G  = static_cast<std::int64_t>(k.Span());
    const double s0 = GridParam(k.i,     G);
    const double s1 = GridParam(k.i + 1, G);
    const double t0 = GridParam(k.j,     G);
    const double t1 = GridParam(k.j + 1, G);
    out[0] = FaceParamToDirection(k.face, s0, t0);
    out[1] = FaceParamToDirection(k.face, s1, t0);
    out[2] = FaceParamToDirection(k.face, s0, t1);
    out[3] = FaceParamToDirection(k.face, s1, t1);
}

glm::dvec3 ChunkCenterDirection(const ChunkKey& k) noexcept {
    const std::int64_t G = static_cast<std::int64_t>(k.Span());
    const double s = 0.5 * (GridParam(k.i, G) + GridParam(k.i + 1, G));
    const double t = 0.5 * (GridParam(k.j, G) + GridParam(k.j + 1, G));
    return FaceParamToDirection(k.face, s, t);
}

void ChunkBoundingSphere(const ChunkKey& k, double h_min, double h_max,
                         glm::dvec3& out_center, double& out_radius) noexcept {
    glm::dvec3 dirs[5];
    ChunkCornerDirections(k, dirs);
    dirs[4] = ChunkCenterDirection(k);

    glm::dvec3 lo( 1e30), hi(-1e30);
    for (const auto& d : dirs) {
        const glm::dvec3 p = EllipsoidSurface(d);
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    out_center = 0.5 * (lo + hi);
    double r = 0.0;
    for (const auto& d : dirs) {
        r = std::max(r, glm::length(EllipsoidSurface(d) - out_center));
    }
    // The surface between the sampled points bulges outward from their
    // convex hull; on a level-0 chunk that sagitta is R*(1 - cos(45 deg))
    // = 0.293 R. Rather than sample a dense grid, take the exact bound:
    // every point of the chunk lies between radius (R_min + h_min) and
    // (R_max + h_max) from the ellipsoid centre, so growing the sphere by
    // the chord-to-arc excess of the widest corner pair is conservative.
    const double half_chord = 0.5 * glm::length(EllipsoidSurface(dirs[1]) -
                                                EllipsoidSurface(dirs[0]));
    const double R = kWgs84A;
    const double sagitta = (half_chord >= R) ? R
                                             : R - std::sqrt(std::max(0.0, R * R - half_chord * half_chord));
    out_radius = r + sagitta + std::max(std::abs(h_min), std::abs(h_max));
}

double ChunkEdgeLength(int level) noexcept {
    // Quarter of a great circle per face edge, halved per level.
    const double face_arc = kIuggMeanRadius * (kPi * 0.5);
    return face_arc / static_cast<double>(1u << std::clamp(level, 0, 31));
}

double ChunkVertexSpacing(int level) noexcept {
    return ChunkEdgeLength(level) / static_cast<double>(kChunkQuads);
}

// --- Neighbours -----------------------------------------------------------

bool NeighborChunk(const ChunkKey& k, ChunkEdge edge, ChunkKey& out) noexcept {
    const std::uint32_t span = k.Span();
    const std::int64_t  di = (edge == kEdgeMinusU) ? -1 : (edge == kEdgePlusU) ? 1 : 0;
    const std::int64_t  dj = (edge == kEdgeMinusV) ? -1 : (edge == kEdgePlusV) ? 1 : 0;
    const std::int64_t  ni = static_cast<std::int64_t>(k.i) + di;
    const std::int64_t  nj = static_cast<std::int64_t>(k.j) + dj;

    // Interior of the same face -- the overwhelmingly common case.
    if (ni >= 0 && ni < static_cast<std::int64_t>(span) &&
        nj >= 0 && nj < static_cast<std::int64_t>(span)) {
        out = ChunkKey{k.face, k.level, static_cast<std::uint32_t>(ni),
                       static_cast<std::uint32_t>(nj)};
        return true;
    }

    // A cube seam. Solved on the CUBE, exactly, with no trigonometry and no
    // normalisation.
    //
    // A point on the shared edge is  Pcube = n_f + E + z * T,  where E is
    // the outward cube axis of the edge (+-right_f or +-up_f), T the
    // along-edge axis, and z the WARPED along-edge parameter. The
    // neighbouring face g is the one whose normal is E, and because every
    // basis vector is a signed coordinate axis, dot(Pcube, n_g) = 1
    // identically. So the same point's parameters on face g are just
    //
    //     p' = dot(n_f, right_g) + z * dot(T, right_g)
    //     q' = dot(n_f, up_g)    + z * dot(T, up_g)
    //
    // and every dot product there is 0 or +-1. One of p'/q' comes out as
    // +-1 (the seam itself, i.e. the outermost cell row on face g) and the
    // other as +-z (the position along the seam). Both are exact.
    const std::int64_t G = static_cast<std::int64_t>(span);
    const double s_mid = 0.5 * (GridParam(k.i, G) + GridParam(k.i + 1, G));
    const double t_mid = 0.5 * (GridParam(k.j, G) + GridParam(k.j + 1, G));

    const glm::dvec3 n_f = FaceNormal(k.face);
    const glm::dvec3 r_f = FaceRight(k.face);
    const glm::dvec3 u_f = FaceUp(k.face);

    glm::dvec3 E(0.0), T(0.0);
    double z = 0.0;
    switch (edge) {
        case kEdgePlusU:  E =  r_f; T = u_f; z = TangentWarp(t_mid); break;
        case kEdgeMinusU: E = -r_f; T = u_f; z = TangentWarp(t_mid); break;
        case kEdgePlusV:  E =  u_f; T = r_f; z = TangentWarp(s_mid); break;
        case kEdgeMinusV: E = -u_f; T = r_f; z = TangentWarp(s_mid); break;
        default: return false;
    }

    // The face whose normal is E.
    int nface = -1;
    for (int g = 0; g < 6; ++g) {
        if (glm::dot(FaceNormal(g), E) > 0.5) { nface = g; break; }
    }
    if (nface < 0) return false;

    const glm::dvec3 r_g = FaceRight(nface);
    const glm::dvec3 u_g = FaceUp(nface);
    const double p_warped = glm::dot(n_f, r_g) + z * glm::dot(T, r_g);
    const double q_warped = glm::dot(n_f, u_g) + z * glm::dot(T, u_g);

    // Unwarp back to the uniform face parameter and take the cell. The
    // along-seam value is the midpoint of a cell, half a cell clear of
    // either boundary, so the ~1-ULP round trip through tan/atan cannot
    // move the floor. The perpendicular value is exactly +-1 (TangentUnwarp
    // pins those endpoints), which floors to the outermost row.
    auto to_cell = [span](double x) -> std::uint32_t {
        const double c = (x + 1.0) * 0.5 * static_cast<double>(span);
        const std::int64_t idx = static_cast<std::int64_t>(std::floor(c));
        return static_cast<std::uint32_t>(
            std::clamp<std::int64_t>(idx, 0, static_cast<std::int64_t>(span) - 1));
    };
    out = ChunkKey{static_cast<std::uint8_t>(nface), k.level,
                   to_cell(TangentUnwarp(std::clamp(p_warped, -1.0, 1.0))),
                   to_cell(TangentUnwarp(std::clamp(q_warped, -1.0, 1.0)))};
    return true;
}

// --- Index arena ----------------------------------------------------------

std::uint32_t SnappedVertexIndex(int x, int y, std::uint32_t mask) noexcept {
    int sx = x, sy = y;
    if (x == 0                && (mask & (1u << kEdgeMinusU)) && (sy & 1)) --sy;
    if (x == kChunkQuads      && (mask & (1u << kEdgePlusU))  && (sy & 1)) --sy;
    if (y == 0                && (mask & (1u << kEdgeMinusV)) && (sx & 1)) --sx;
    if (y == kChunkQuads      && (mask & (1u << kEdgePlusV))  && (sx & 1)) --sx;
    return static_cast<std::uint32_t>(sy * kChunkVerts + sx);
}

TerrainIndexArena::TerrainIndexArena() {
    indices_.resize(static_cast<std::size_t>(kVariantCount) * kChunkIndexCount);
    std::size_t w = 0;
    for (std::uint32_t mask = 0; mask < kVariantCount; ++mask) {
        for (int y = 0; y < kChunkQuads; ++y) {
            for (int x = 0; x < kChunkQuads; ++x) {
                const std::uint32_t v00 = SnappedVertexIndex(x,     y,     mask);
                const std::uint32_t v10 = SnappedVertexIndex(x + 1, y,     mask);
                const std::uint32_t v01 = SnappedVertexIndex(x,     y + 1, mask);
                const std::uint32_t v11 = SnappedVertexIndex(x + 1, y + 1, mask);
                // Counter-clockwise seen from outside the planet: the face
                // bases are right-handed with cross(right, up) == normal,
                // and the grid runs +right in x and +up in y, so (v00, v10,
                // v11) winds outward.
                indices_[w++] = v00; indices_[w++] = v10; indices_[w++] = v11;
                indices_[w++] = v00; indices_[w++] = v11; indices_[w++] = v01;
            }
        }
    }
}

}  // namespace pt::planet

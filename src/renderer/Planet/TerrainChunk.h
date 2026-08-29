// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary P4 (#258): chunk baking -------------------------------------
//
// Turns a ChunkKey into everything the renderer needs: 65x65 chunk-LOCAL
// vertices for the BLAS, per-vertex normals and elevations for the shader,
// the Ulrich (2002) geometric error e_L that drives LOD selection, and the
// per-mip mean-square slope that folds sub-footprint geometry into
// roughness.
//
// LOCAL FROM THE START
// --------------------
// P1 (#255) documented why the CSG mesh keeps its placement in the ray-
// origin shift rather than in the TLAS instance transform: CsgScene bakes
// into ABSOLUTE world coordinates, so splitting the placement between the
// instance transform and the shader would make both halves planetary while
// their sum stayed small, and the acceleration structure would do exactly
// the cancellation the phase existed to remove (measured: 2.2% bad pixels
// on the worldframe equivalence cell).
//
// Terrain has no such constraint, so it takes the arrangement P1 named as
// becoming correct at this phase: vertices are chunk-local from the start
// and the TLAS instance transform carries `chunk_origin - anchor`. A
// level-0 chunk is 10 007 km across, where float32 resolves 0.6 m against a
// 156 km vertex spacing; a level-19 chunk is 19 m across, where float32
// resolves 1.1 um. Precision scales with the chunk for free, and an origin
// rebase is one Device::UpdateTLASInstances that touches no BLAS.
//
// THE WORLD FRAME
// ---------------
// The engine's world +Y is up and its scenes are authored around y = 0, so
// the planet is placed as P3 (#257) placed the analytic body: the reference
// SITE sits at the world origin and the planet centre at (0, -R_site, 0).
// PlanetSite carries the rotation from ECEF into that frame.
//
// One documented approximation: world +Y is the GEOCENTRIC up at the site
// (the direction of the site from the ellipsoid centre), not the GEODETIC
// up (the ellipsoid normal). The two differ by the angle of the vertical,
// at most 11.5 arcmin (0.19 deg) at 45 deg latitude, and using the
// geocentric one is what lets the ellipsoid centre, the atmosphere shell's
// centre and the analytic backstop's centre be the single point
// (0, -R_site, 0) rather than three points a few kilometres apart. P3's
// note that the air and the ground must agree on ONE centre or they
// disagree about the horizon by hundreds of kilometres is the reason that
// trade goes this way.

#include "CubedSphere.h"
#include "ElevationField.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace pt::planet {

// The placement of the ellipsoid in the engine's world frame.
struct PlanetSite {
    double     lat_rad     = 0.0;
    double     lon_rad     = 0.0;
    // Rows are (East, GeocentricUp, South) in ECEF. Right-handed:
    // East x Up = South, so world X = East, Y = Up, Z = South and a camera
    // at yaw 0 (looking down -Z, the engine's convention) looks north.
    glm::dmat3 ecef_to_world{1.0};
    // |P_site| -- the geocentric radius of the ellipsoid surface at the
    // site. The planet centre in world coordinates is (0, -site_radius, 0).
    double     site_radius_m = kIuggMeanRadius;

    static PlanetSite FromGeodetic(double lat_rad, double lon_rad) noexcept;

    glm::dvec3 CenterWorld() const noexcept {
        return glm::dvec3(0.0, -site_radius_m, 0.0);
    }
    glm::dvec3 EcefToWorld(const glm::dvec3& p_ecef) const noexcept {
        return ecef_to_world * p_ecef + CenterWorld();
    }
    glm::dvec3 WorldToEcef(const glm::dvec3& p_world) const noexcept {
        return glm::transpose(ecef_to_world) * (p_world - CenterWorld());
    }
    // Unit direction from the planet centre to a world point, expressed in
    // the ELLIPSOID frame -- the input the elevation field wants. Named for
    // the frame it lands in because the two differ by the site rotation,
    // and comparing this against a world-space normal is a silent bug: at
    // the site's antipode the rotation is a signed permutation, so the dot
    // product comes out exactly zero instead of exactly one.
    glm::dvec3 WorldToEcefDirection(const glm::dvec3& p_world) const noexcept {
        const glm::dvec3 e = WorldToEcef(p_world);
        const double len = glm::length(e);
        return (len > 0.0) ? e / len : glm::dvec3(0.0, 0.0, 1.0);
    }
    // The same direction in WORLD space: the local geocentric up, which is
    // what an outward-facing test compares a world-space normal against.
    glm::dvec3 WorldUp(const glm::dvec3& p_world) const noexcept {
        const glm::dvec3 d = p_world - CenterWorld();
        const double len = glm::length(d);
        return (len > 0.0) ? d / len : glm::dvec3(0.0, 1.0, 0.0);
    }
};

// --- The reference scale for slope-driven land cover (#307) ----------------
//
// #300 drives rock exposure (and, through 1 - rock, snow shedding) off
// `slope01 = 1 - |cos(tilt)|` taken from the MESH vertex normal. That
// normal is a central difference over one CHUNK cell either side, so its
// baseline is 2 * ChunkVertexSpacing(level) -- 38.2 m at level 13, 9.5 m at
// level 15. Slope is not scale-free: on a self-affine surface the RMS slope
// over a baseline l goes as l^(H-1), so the SAME GROUND reads a different
// angle at every LOD. Measured on earth_lite at the reference site over a
// 2x2 chunk block, with the shipped ramp:
//
//     level 13 (38.2 m baseline)   median slope 20.46 deg   mean rock 0.0585
//     level 15 ( 9.5 m baseline)   median slope 26.16 deg   mean rock 0.1841
//
// A 3.1x change in rock exposure on ground that did not move: a chunk
// changes COLOUR when it splits. Before #304 both levels were saturated
// (mean rock 0.35 and 0.74 on the unbroken power law) so the defect was
// invisible; it is not invisible now.
//
// THE FIX IS TO NAME THE SCALE, and the scale is not free to choose. The
// ramp's endpoints are threshold-hillslope angles, and a hillslope angle is
// a measurement with a baseline attached to it. Gabet, Pratt-Sitaula &
// Burbank (2004), "Climatic controls on hillslope angle and relief in the
// Himalayas", Geology 32(7):629-632 (doi:10.1130/G20641.1) -- central
// Nepal, the same range and the same author lineage as Burbank et al.
// (1996), Nature 379:505, which is where the 30 deg in
// kThresholdHillslopeDeg comes from -- state theirs explicitly:
//
//     "...MAR values ... were interpolated ... onto a 3 arcsecond (~90 m
//      spacing) digital elevation model ... because the slope angle of
//      each grid cell is calculated from the elevation of its uphill and
//      downhill neighbors, the length scale of the slope angle
//      measurement is ~270 m"
//
// Brocklehurst & Whipple (2007), JGR Earth Surface 112:F02035
// (doi:10.1029/2006JF000667) use the same ~90 m data for the Nanga Parbat
// region Burbank et al. measured. So the published threshold is a
// 3-arcsecond-grid, one-cell-either-side slope, and the engine reproduces
// that operator rather than inventing one:
//
//   * kRefSlopeLevel = 11 is the subdivision level whose vertex spacing
//     (76.35 m) is closest to 3 arcseconds (~92.6 m at the equator); level
//     10 is 152.7 m, i.e. 1.65x too coarse against level 11's 0.82x.
//   * kRefSlopeHalfLagM = 90 m puts the two difference samples exactly one
//     3-arcsecond cell either side of the vertex, so the BASELINE is the
//     published 180 m regardless of which lattice level carries it.
//
// Two consequences worth stating, and one non-consequence.
//
//   1. The value is a pure function of POSITION. The level-11 lattice is
//      shared by every descendant, `GenerateChunkHeights` is interpolatory
//      and level-consistent, and the fractional lattice coordinate of a
//      chunk vertex is an exact dyadic rational -- so a level-13 chunk and
//      a level-15 chunk covering the same ground compute BIT-IDENTICAL
//      rock exposure. tests/pt_planet_terrain_test.cpp asserts the equality
//      rather than a tolerance, at four levels below the lattice.
//   2. Nothing the continuation does BELOW 76 m enters at all. The
//      reference grid stops at kRefSlopeLevel, so levels 12 through 19 --
//      eight of the twelve octaves the hierarchy spans -- are simply not
//      evaluated. That is what makes (1) exact rather than approximate.
//
//   NOT a consequence: independence from `hurst_fine`. 180 m is above the
//   106 m break, but only by 1.7x, and the broken power law's
//   (l/L)^H_fine factor applies at every lag -- so S(180 m) still moves
//   when H_fine does, and so does the reference slope. It has to: a finite
//   difference is not a low-pass filter, and a real 3-arcsecond DEM slope
//   contains 3-arcsecond roughness for exactly the same reason. What #307
//   decouples is the TESSELLATION, not the terrain model.
//
// THE COST, measured on an M4 Max over the real earth_lite grid: a chunk
// bake goes 1.69 -> 2.32 ms at levels 11 and finer (one extra
// GenerateChunkHeights at level 11 over a 69x69 window, which is ~5x
// cheaper than the level+1 fine grid the bake already builds), and
// 1.78 -> 2.66..3.00 ms at levels 10 and coarser, where the area-mean
// integral runs per vertex instead. Bakes happen on the JobSystem workers,
// off the frame.
inline constexpr int    kRefSlopeLevel     = 11;
inline constexpr double kRefSlopeHalfLagM  = 90.0;

// Per-vertex shader payload stride, in floats. The channels are:
//   0,1,2 = the outward unit normal (same in chunk-local and world space,
//           because the instance transform is a pure translation)
//   3     = the vertex's ellipsoidal height in metres, which the shader
//           uses to pick a biome and, from P5 on, to know whether it is
//           standing on seabed.
//   4     = rock01, the threshold-hillslope rock exposure evaluated at the
//           REFERENCE SCALE above and already through
//           SlopeRockFraction() on the host. #307: the ramp moves to the
//           host because its input has to be a property of the ground, and
//           the ground is only available here.
//
// FIVE FLOATS AND NOT EIGHT. The obvious shape is a second float4, which
// is one line of change and wastes 12 bytes per vertex -- 198 MB of
// zeroes at the 2 048-chunk arena the terrain fixtures ask for. A
// five-float stride costs the shader three extra scalar loads per terrain
// hit and 25% more arena instead of 100%. Measured, from the engine's own
// startup line: 66 -> 83 MB at the default 1 024-chunk budget, 132 -> 165
// MB at the fixtures' 2 048.
inline constexpr int kVertexPayloadFloats = 5;

// The baked product of one chunk.
struct TerrainChunkData {
    ChunkKey key{};

    // Chunk-local float3 vertex positions, tightly packed, 3 floats per
    // vertex, kChunkVertexCount vertices. Fed straight to BLASDesc.
    std::vector<float> positions;

    // Per-vertex shader payload, kChunkVertexCount * kVertexPayloadFloats
    // floats. See kVertexPayloadFloats for the channel list.
    std::vector<float> shader_verts;

    // Canonical world position of the chunk's local origin.
    glm::dvec3 origin_w{0.0};

    // Bounding sphere in canonical world coordinates.
    glm::dvec3 bound_center_w{0.0};
    double     bound_radius_m = 0.0;

    // Ulrich (2002) chunked-LOD delta: the maximum radial distance, in
    // metres, between this chunk's rendered surface and the surface one
    // level finer. Measured against the actual TRIANGULATED surface (both
    // triangles of each quad), not a bilinear stand-in, because the
    // triangulated one is what the ray query intersects.
    double e_l_m = 0.0;

    // Mean-square slope of the detail removed at each footprint mip, mip m
    // covering 2^m vertex spacings. sigma2[0] is 0 by construction (a
    // one-spacing footprint removes nothing). Folded into roughness as
    // alpha^2_eff = alpha^2_material + 2 * sigma2[mip].
    std::array<float, kSlopeMips> sigma2{};

    double h_min_m = 0.0;
    double h_max_m = 0.0;

    // Vertex spacing in metres -- the mip-0 footprint the shader compares
    // cone_width against.
    double vertex_spacing_m = 0.0;
};

// slope01 = 1 - |cos(tilt)| at the REFERENCE baseline, for every vertex of
// `key`, in row-major order (kChunkVertexCount entries). `key.level` must
// be >= kRefSlopeLevel; below that the mesh cannot carry the signal and
// BuildTerrainChunk takes the area-mean branch instead.
//
// A pure function of position: the value at a given point is identical --
// bit for bit, not to a tolerance -- for every chunk level that contains
// it. Exposed rather than kept inside BuildTerrainChunk because both the
// level-consistency assertion and the calibration of the coarse branch are
// statements about THIS quantity rather than about the ramp on top of it.
void ReferenceSlope01(const ChunkKey& key,
                      const ElevationField& field,
                      const PlanetSite& site,
                      std::vector<double>& out);

// The per-axis RMS slope tangent the continuation carries at the reference
// baseline, at a unit direction in the ellipsoid frame. The coarse branch's
// input; a smooth, purely positional function of the DEM's own local
// relief. See kRefSlopeRmsGain in TerrainChunk.cpp for the one measured
// number in it and how it was measured.
double ReferenceRmsSlopePerAxis(const ElevationField& field,
                                const glm::dvec3& dir_unit) noexcept;

// Bake one chunk. Pure function of (key, field, site): callable from any
// number of JobSystem workers at once, and byte-identical across runs,
// which is what makes r_planet_lod_freeze produce reproducible goldens.
void BuildTerrainChunk(const ChunkKey& key,
                       const ElevationField& field,
                       const PlanetSite& site,
                       TerrainChunkData& out);

}  // namespace pt::planet

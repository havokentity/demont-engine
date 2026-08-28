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

// The baked product of one chunk.
struct TerrainChunkData {
    ChunkKey key{};

    // Chunk-local float3 vertex positions, tightly packed, 3 floats per
    // vertex, kChunkVertexCount vertices. Fed straight to BLASDesc.
    std::vector<float> positions;

    // Per-vertex shader payload, kChunkVertexCount float4:
    //   xyz = the outward unit normal (same in chunk-local and world space,
    //         because the instance transform is a pure translation)
    //   w   = the vertex's ellipsoidal height in metres, which the shader
    //         uses to pick a biome and, from P5 on, to know whether it is
    //         standing on seabed.
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

// Bake one chunk. Pure function of (key, field, site): callable from any
// number of JobSystem workers at once, and byte-identical across runs,
// which is what makes r_planet_lod_freeze produce reproducible goldens.
void BuildTerrainChunk(const ChunkKey& key,
                       const ElevationField& field,
                       const PlanetSite& site,
                       TerrainChunkData& out);

}  // namespace pt::planet

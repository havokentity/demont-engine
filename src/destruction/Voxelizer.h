// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Voxelizer -- Phase 1 of the voxel destruction roadmap (issue #139).
//
// Converts a source object (analytic primitive, SDF cluster, or CSG
// triangle mesh) into a VoxelGrid by classifying each voxel center as
// "inside" or "outside" the source. Phase 1 is CPU-side; the SDF /
// analytic queries are cheap and the triangle-mesh inside-test scales
// linearly with the bounding-box volume so we keep voxelization off
// the GPU entirely.
//
// API design follows the engine's "no virtuals" preference: a small
// set of free functions, each taking the typed source plus a grid
// reference. Callers either dispatch on their own type tag or know
// which entrypoint to hit. The shared VoxelGrid.Init is called inside
// each entrypoint so the AABB / dims policy stays consistent.

#include <cstdint>
#include <vector>

#include "VoxelGrid.h"

namespace pt::renderer {
struct SdfPrim;       // forward-decl, full def in renderer/AnalyticBvh.h
}

namespace pt::csg {
struct BakedMesh;     // forward-decl, full def in renderer/Csg/CsgScene.h
}

namespace pt::destruction {

// -----------------------------------------------------------------------------
// Analytic primitive: today only a SPHERE (the only finite-extent
// primitive in the AnalyticPrim set; plane has infinite extent and is
// nonsense to voxelize). center + radius are in metres.
// -----------------------------------------------------------------------------
bool VoxelizeSphere(std::uint32_t object_id,
                    const float center[3], float radius,
                    float voxel_size,
                    const VoxelMaterial& material,
                    VoxelGrid& out);

// -----------------------------------------------------------------------------
// SDF cluster. Uses the cluster's own AABB + the host-side SDF
// evaluator (mirrors sdfClusterDist in shaders/SdfPrimitives.slang).
// A voxel is "occupied" iff sdfClusterDist(center) < voxel_size/2.
// -----------------------------------------------------------------------------
bool VoxelizeSdf(std::uint32_t object_id,
                 const pt::renderer::SdfPrim& cluster,
                 float voxel_size,
                 const VoxelMaterial& material,
                 VoxelGrid& out);

// -----------------------------------------------------------------------------
// CSG triangle mesh. The "inside" test is a +X ray-cast parity count
// (Möller-Trumbore against every triangle), which is robust for the
// closed manifold meshes Manifold produces. AABB is computed from the
// vertex positions; voxel size + material come from the caller.
//
// The mesh is consumed as a flat positions + indices buffer rather
// than a BakedMesh reference so callers in tests / tools that don't
// link pt_renderer can still hit this entry point. (BakedMesh
// converts cheaply into (positions span, indices span) at the call
// site.)
// -----------------------------------------------------------------------------
bool VoxelizeMesh(std::uint32_t object_id,
                  const float* positions,        // 3 floats per vertex
                  std::size_t  vertex_count,
                  const std::uint32_t* indices,  // 3 per triangle
                  std::size_t  index_count,
                  float voxel_size,
                  const VoxelMaterial& material,
                  VoxelGrid& out);

// Convenience overload: BakedMesh wrapper. Pulls the positions /
// indices spans and forwards to VoxelizeMesh. Returns false on an
// empty mesh.
bool VoxelizeBakedMesh(std::uint32_t object_id,
                       const pt::csg::BakedMesh& mesh,
                       float voxel_size,
                       const VoxelMaterial& material,
                       VoxelGrid& out);

// -----------------------------------------------------------------------------
// Cache helpers. Compose a deterministic file name from a source
// (object_id, voxel_size, occupancy-affecting key) tuple. Format:
//   captures/voxel_cache_<object_id>_<vsize_micro>_<key>.bin
// where vsize_micro is voxel_size * 1e6 rounded to int (so 0.1 m =
// "100000"). Caller passes the cache dir prefix (typically the
// engine's captures/ for the smoke-test run).
//
// content_key is an opaque hash that callers can use to invalidate
// stale cache entries when the source's vertex / SDF expression
// content changes (Phase 1: passes 0; Phase 2 may pass a positional
// hash).
std::string MakeCachePath(const std::string& dir_prefix,
                          std::uint32_t object_id,
                          float voxel_size,
                          std::uint64_t content_key);

}  // namespace pt::destruction

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// VoxelGrid -- Phase 1 of the voxel destruction roadmap (issue #139).
//
// A VoxelGrid is the discrete-occupancy representation of a single
// source object (CSG mesh, analytic primitive, or SDF cluster). It is
// produced by the Voxelizer (see Voxelizer.h) and consumed by the
// engine to spawn one analytic box primitive per occupied voxel.
//
// Conventions (per the project's "real physics + metric units"
// directive in MEMORY.md):
//   * voxel_size is in METRES. Default is 0.1 m (set at the engine
//     cvar layer). One world unit equals one metre throughout.
//   * AABB is in world space; the grid is anchored at aabb_min and
//     extends in +X / +Y / +Z by (dims.x, dims.y, dims.z) voxels.
//   * Occupancy is stored as a packed bitmap (one bit per voxel) in
//     linear (i, j, k) order with the X axis fastest. This keeps the
//     hot scan path coherent and the cache footprint small even for
//     large grids (~125 KB for 1e6 voxels).
//   * Material is captured ONCE at voxelization time. Per-voxel
//     material variance is a Phase 3+ concern (chunk merging may
//     change the inheritance model entirely).
//
// On-disk format. Ad-hoc binary; see VoxelGrid.cpp for the byte
// layout. The cache key is (source object id, voxel_size) packed
// into a u64 so the file name is deterministic.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pt::destruction {

struct VoxelMaterial {
    // Material kind: 0 Lambert, 1 Metal, 2 Dielectric (matches
    // AnalyticPrim::Material and SdfPrim::material wire format).
    std::uint32_t kind   = 0u;
    std::array<float, 3> albedo {0.7f, 0.7f, 0.7f};
    float roughness      = 0.0f;
    float ior            = 1.5f;
};

// Header that prefixes the packed bitmap on-disk and lives at the
// front of the runtime VoxelGrid. All fields are little-endian.
struct VoxelGridHeader {
    // Magic + version. Bump version when the layout below changes so
    // out-of-date cache files are rejected cleanly.
    static constexpr std::uint32_t kMagic   = 0x564F584C; // 'VOXL'
    static constexpr std::uint32_t kVersion = 1u;

    std::uint32_t magic        = kMagic;
    std::uint32_t version      = kVersion;
    std::uint32_t object_id    = 0u;
    float         voxel_size   = 0.1f;
    float         aabb_min[3]  {0, 0, 0};
    float         aabb_max[3]  {0, 0, 0};
    std::uint32_t dim_x        = 0u;
    std::uint32_t dim_y        = 0u;
    std::uint32_t dim_z        = 0u;
    std::uint32_t occupied     = 0u;     // count of occupied voxels (cached for stats)
    VoxelMaterial material     {};
};

class VoxelGrid {
public:
    VoxelGrid() = default;

    // Initialise an empty grid covering [aabb_min .. aabb_max] at the
    // requested voxel size. dims are rounded UP so the AABB is fully
    // covered; the grid's effective aabb_max is then aabb_min + dims *
    // voxel_size, which can be larger than the input aabb_max by up to
    // (voxel_size - epsilon) per axis. That's intentional: keeping the
    // grid a clean integer multiple of voxel_size means later cache
    // hits don't have to renegotiate the alignment.
    void Init(std::uint32_t object_id,
              const float aabb_min[3], const float aabb_max[3],
              float voxel_size,
              const VoxelMaterial& material);

    // Occupancy access. Out-of-range indices return false / are no-ops
    // so caller doesn't have to guard each voxel.
    bool   Get(std::uint32_t i, std::uint32_t j, std::uint32_t k) const;
    void   Set(std::uint32_t i, std::uint32_t j, std::uint32_t k, bool occupied);

    // Voxel centre in world space for the (i, j, k) cell.
    std::array<float, 3> VoxelCenter(std::uint32_t i, std::uint32_t j, std::uint32_t k) const;

    // Recompute the cached occupied-count from the bitmap. Called by
    // Voxelizer after a fill pass; callers that mutate cells one at a
    // time should call this before reading Occupied().
    void RefreshOccupiedCount();

    // Accessors.
    const VoxelGridHeader&             Header()   const { return header_; }
    const std::vector<std::uint64_t>&  Bitmap()   const { return bitmap_; }
    std::uint32_t                      Occupied() const { return header_.occupied; }
    std::uint32_t                      DimX()     const { return header_.dim_x; }
    std::uint32_t                      DimY()     const { return header_.dim_y; }
    std::uint32_t                      DimZ()     const { return header_.dim_z; }
    std::uint32_t                      ObjectId() const { return header_.object_id; }
    float                              VoxelSize()const { return header_.voxel_size; }
    const VoxelMaterial&               Material() const { return header_.material; }

    // True iff the grid has at least one voxel position (dims > 0 on
    // every axis). Caller uses this as an "is this grid usable" gate.
    bool Valid() const {
        return header_.dim_x > 0u && header_.dim_y > 0u && header_.dim_z > 0u;
    }

    // ----- on-disk cache --------------------------------------------------
    // Writes header + packed bitmap to `path`. Returns true on success.
    // On error the LOG sink is the engine's stderr stream (LOG_ERROR).
    bool SaveToFile(const std::string& path) const;

    // Reads header + packed bitmap from `path`, replacing the current
    // grid contents. Returns false on missing file, magic / version
    // mismatch, or short read. The grid is left in a known-empty state
    // on failure so callers can fall through to a fresh voxelization.
    bool LoadFromFile(const std::string& path);

private:
    VoxelGridHeader            header_{};
    // One bit per voxel, packed 64 voxels per uint64. Linear ordering
    // is (k * dim_y + j) * dim_x + i with X as the fastest axis. The
    // vector is sized to ceil(total / 64) uint64 words.
    std::vector<std::uint64_t> bitmap_;

    std::uint64_t LinearIndex(std::uint32_t i, std::uint32_t j, std::uint32_t k) const {
        return (std::uint64_t(k) * std::uint64_t(header_.dim_y) + std::uint64_t(j))
                * std::uint64_t(header_.dim_x) + std::uint64_t(i);
    }
};

}  // namespace pt::destruction

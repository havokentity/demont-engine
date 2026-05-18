// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "VoxelGrid.h"

#include "../core/Log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace pt::destruction {

namespace {

// Bit-popcount of a 64-bit word. Used by RefreshOccupiedCount. Bare
// __builtin_popcountll where available; falls back to a portable
// software path on toolchains that don't expose it.
inline std::uint32_t Popcount64(std::uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint32_t>(__builtin_popcountll(w));
#else
    w = w - ((w >> 1) & 0x5555555555555555ull);
    w = (w & 0x3333333333333333ull) + ((w >> 2) & 0x3333333333333333ull);
    w = (w + (w >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return static_cast<std::uint32_t>((w * 0x0101010101010101ull) >> 56);
#endif
}

}  // namespace

void VoxelGrid::Init(std::uint32_t object_id,
                     const float aabb_min[3], const float aabb_max[3],
                     float voxel_size,
                     const VoxelMaterial& material) {
    header_              = {};
    header_.magic        = VoxelGridHeader::kMagic;
    header_.version      = VoxelGridHeader::kVersion;
    header_.object_id    = object_id;
    header_.voxel_size   = (voxel_size > 0.0f) ? voxel_size : 0.1f;
    header_.material     = material;

    // Cover the source AABB in integer voxel multiples. ceil() the
    // extent so the +X / +Y / +Z faces are inside the grid (we'd lose
    // surface voxels otherwise). A degenerate axis (max <= min) gives
    // dims = 1 along that axis so the grid still has one row/column to
    // probe -- voxelization will mark no occupancy there but the grid
    // is still consistent for any later toggling.
    for (int a = 0; a < 3; ++a) {
        header_.aabb_min[a] = aabb_min[a];
        float extent = aabb_max[a] - aabb_min[a];
        if (extent <= 0.0f) extent = header_.voxel_size;  // degenerate axis safeguard
        std::uint32_t dim =
            static_cast<std::uint32_t>(std::ceil(extent / header_.voxel_size));
        if (dim == 0u) dim = 1u;
        // Cap at a sane upper bound. 1024 voxels per axis * 0.1 m = 100 m
        // span which is comfortable for the path tracer; anything bigger
        // gets clipped so a runaway cvar can't OOM us.
        constexpr std::uint32_t kMaxDim = 1024u;
        if (dim > kMaxDim) {
            LOG_WARN("[voxel] dim axis {} clipped from {} to {} (voxel_size={}m, extent={}m)",
                     a, dim, kMaxDim, header_.voxel_size, extent);
            dim = kMaxDim;
        }
        switch (a) {
            case 0: header_.dim_x = dim; break;
            case 1: header_.dim_y = dim; break;
            case 2: header_.dim_z = dim; break;
        }
        header_.aabb_max[a] = aabb_min[a] + float(dim) * header_.voxel_size;
    }

    const std::uint64_t total = std::uint64_t(header_.dim_x)
                              * std::uint64_t(header_.dim_y)
                              * std::uint64_t(header_.dim_z);
    const std::uint64_t words = (total + 63u) / 64u;
    bitmap_.assign(std::size_t(words), 0ull);
    header_.occupied = 0u;
}

bool VoxelGrid::Get(std::uint32_t i, std::uint32_t j, std::uint32_t k) const {
    if (i >= header_.dim_x || j >= header_.dim_y || k >= header_.dim_z) return false;
    const std::uint64_t idx = LinearIndex(i, j, k);
    return (bitmap_[std::size_t(idx >> 6)] >> (idx & 63u)) & 1ull;
}

void VoxelGrid::Set(std::uint32_t i, std::uint32_t j, std::uint32_t k, bool occupied) {
    if (i >= header_.dim_x || j >= header_.dim_y || k >= header_.dim_z) return;
    const std::uint64_t idx  = LinearIndex(i, j, k);
    const std::size_t   word = std::size_t(idx >> 6);
    const std::uint64_t mask = std::uint64_t(1) << (idx & 63u);
    if (occupied) bitmap_[word] |= mask;
    else          bitmap_[word] &= ~mask;
}

std::array<float, 3> VoxelGrid::VoxelCenter(std::uint32_t i, std::uint32_t j, std::uint32_t k) const {
    const float h = 0.5f * header_.voxel_size;
    return {
        header_.aabb_min[0] + float(i) * header_.voxel_size + h,
        header_.aabb_min[1] + float(j) * header_.voxel_size + h,
        header_.aabb_min[2] + float(k) * header_.voxel_size + h,
    };
}

void VoxelGrid::RefreshOccupiedCount() {
    std::uint32_t count = 0u;
    for (std::uint64_t w : bitmap_) count += Popcount64(w);
    header_.occupied = count;
}

bool VoxelGrid::SaveToFile(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        LOG_ERROR("[voxel] save: cannot open '{}' for write", path);
        return false;
    }
    if (std::fwrite(&header_, sizeof(VoxelGridHeader), 1, f) != 1) {
        LOG_ERROR("[voxel] save: header write failed for '{}'", path);
        std::fclose(f);
        return false;
    }
    const std::size_t words = bitmap_.size();
    if (words > 0 &&
        std::fwrite(bitmap_.data(), sizeof(std::uint64_t), words, f) != words) {
        LOG_ERROR("[voxel] save: bitmap write failed for '{}'", path);
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    LOG_INFO("[voxel] saved object_id={} dims={}x{}x{} occupied={} -> {}",
             header_.object_id, header_.dim_x, header_.dim_y, header_.dim_z,
             header_.occupied, path);
    return true;
}

bool VoxelGrid::LoadFromFile(const std::string& path) {
    header_  = {};
    bitmap_.clear();

    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        // Don't LOG_ERROR -- "no cache file yet" is the common case.
        return false;
    }
    VoxelGridHeader hdr{};
    if (std::fread(&hdr, sizeof(VoxelGridHeader), 1, f) != 1) {
        LOG_WARN("[voxel] load: header read failed for '{}'", path);
        std::fclose(f);
        return false;
    }
    if (hdr.magic != VoxelGridHeader::kMagic) {
        LOG_WARN("[voxel] load: '{}' magic mismatch ({:#x} != {:#x})",
                 path, hdr.magic, VoxelGridHeader::kMagic);
        std::fclose(f);
        return false;
    }
    if (hdr.version != VoxelGridHeader::kVersion) {
        LOG_WARN("[voxel] load: '{}' version mismatch ({} != {})",
                 path, hdr.version, VoxelGridHeader::kVersion);
        std::fclose(f);
        return false;
    }
    if (hdr.dim_x == 0u || hdr.dim_y == 0u || hdr.dim_z == 0u) {
        LOG_WARN("[voxel] load: '{}' has zero dims", path);
        std::fclose(f);
        return false;
    }
    const std::uint64_t total = std::uint64_t(hdr.dim_x)
                              * std::uint64_t(hdr.dim_y)
                              * std::uint64_t(hdr.dim_z);
    const std::uint64_t words = (total + 63u) / 64u;
    std::vector<std::uint64_t> bits(std::size_t(words), 0ull);
    if (words > 0 &&
        std::fread(bits.data(), sizeof(std::uint64_t), words, f) != words) {
        LOG_WARN("[voxel] load: bitmap short read for '{}'", path);
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    header_ = hdr;
    bitmap_ = std::move(bits);
    LOG_INFO("[voxel] loaded object_id={} dims={}x{}x{} occupied={} from {}",
             header_.object_id, header_.dim_x, header_.dim_y, header_.dim_z,
             header_.occupied, path);
    return true;
}

}  // namespace pt::destruction

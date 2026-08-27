// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary P4 (#258): the streaming half ------------------------------
//
// Owns everything about terrain that touches the RHI: the chunk vertex
// arena, the shared index arena, the per-instance descriptor buffer, the
// per-chunk BLASes, and the TLAS instance array the engine hands to
// Device::UpdateTLASInstances. The pure computation -- quadtree, elevation
// field, chunk baking -- lives in src/renderer/Planet and knows nothing
// about a GPU.
//
// FOUR STAGES, ONLY THE LAST TWO ON THE RENDER THREAD
//
//   selector    per frame, cheap. Walks the quadtree with the e_L/d metric,
//               produces the desired residency set, diffs it.
//   generation  AsyncChunkBaker's worker pool. DEM sample + fractal
//               continuation -> 65x65 positions, normals, per-mip
//               sigma^2_slope and the chunk's own e_L.
//   upload+build  render thread, budgeted by r_planet_blas_budget_ms.
//               WriteBuffer the arena slice, CreateBLAS.
//   publish     render thread. Rewrite the instance array; the engine
//               calls Device::UpdateTLASInstances once.
//
// The async shape follows pt::renderer::AsyncLightTreeBuilder, not the CSG
// bake. The CSG pattern blanks the screen while a bake is in flight
// (Engine::RenderFrame's loading-frame gate) and JobSystem::Wait mandates a
// blocking join per submit; neither survives contact with something that
// runs tens of times a second. RenderFrame's bake gate must NOT fire for
// terrain streaming, and does not: nothing here touches bake_phase_.
//
// WHAT STILL STALLS, AND WHY IT IS BOUNDED
//
// P0 (#254) delivered a drain-free Device::UpdateTLASInstances, and this
// system's per-frame publish uses it: paced updates move
// AccelGpuStallCount by exactly zero. P0 did NOT deliver an encodable,
// fence-based BLAS build -- Device::CreateBLAS still ends in
// waitUntilCompleted / vkQueueWaitIdle by design, one stall per create,
// documented at Device.h's AccelGpuStallCount. So a chunk BUILD still
// blocks the calling thread once.
//
// This system bounds that rather than pretending it away: builds are paced
// against a measured wall-clock budget (r_planet_blas_budget_ms), so a
// frame spends at most that long inside CreateBLAS no matter how far the
// LOD boundary swept. The residual is a real remainder, not a rounding
// error, and the fix is P0's item 2 -- CommandBuffer::BuildAccelStruct plus
// a polled fence -- which #81 needs anyway.
//
// SLOT REUSE AND THE THREE-FRAME RETIRE
//
// Arena slots and BLASes are not freed the moment a chunk leaves the
// desired set. They go into a retire ring `kRetireFrames` deep, so a slot
// is only rewritten once no in-flight frame can still be tracing the TLAS
// that referenced it. That is the deferred-destruction queue the RHI does
// not have, implemented at the one call site that needs it.

#include "../renderer/Planet/CubedSphere.h"
#include "../renderer/Planet/ElevationField.h"
#include "../renderer/Planet/TerrainChunk.h"
#include "../renderer/Planet/TerrainQuadtree.h"
#include "../rhi/Device.h"
#include "../rhi/Resources.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pt::engine {

// Number of float4 per instance descriptor. MUST match kInstDescFloat4s in
// shaders/PathTrace.slang -- the two are a wire format and there is no
// generated header between them.
inline constexpr std::uint32_t kInstDescFloat4s = 6u;
inline constexpr std::uint32_t kInstKindMesh    = 0u;
inline constexpr std::uint32_t kInstKindTerrain = 1u;

// Frames a retired arena slot / BLAS waits before reuse. Three covers the
// deepest frames-in-flight either backend runs.
inline constexpr int kRetireFrames = 3;

// Retirement slot sentinel: this retirement owns a BLAS but no arena slot.
// A stitch-mask rebuild produces exactly that -- the chunk's vertices are
// unchanged and it keeps its slot, only the index set (and therefore the
// acceleration structure) is superseded.
inline constexpr std::uint32_t kNoSlot = 0xFFFFFFFFu;

struct TerrainConfig {
    bool   enabled       = false;
    double site_lat_rad  = 0.0;
    double site_lon_rad  = 0.0;
    int    worker_count  = 4;
    double blas_budget_ms = 2.0;
    std::string dem_path;
    pt::planet::ElevationParams field{};
    pt::planet::LodParams       lod{};
};

// Per-frame diagnostics, surfaced to the console and to the tests.
struct TerrainStats {
    std::size_t   resident      = 0;
    std::size_t   desired       = 0;
    std::size_t   pending_bakes = 0;
    std::uint64_t blas_builds   = 0;   // cumulative
    std::uint64_t tlas_updates  = 0;   // cumulative
    std::uint64_t evictions     = 0;   // cumulative
    double        last_build_ms = 0.0;
    bool          converged     = false;
    // Order-independent digest of the desired leaf set; see
    // TerrainQuadtree::DesiredDigest. Two runs of a frozen capture must
    // agree on this, and a chunk count is not evidence that they do.
    std::uint64_t desired_digest = 0;
};

class PlanetTerrain {
public:
    PlanetTerrain() = default;
    ~PlanetTerrain();
    PlanetTerrain(const PlanetTerrain&)            = delete;
    PlanetTerrain& operator=(const PlanetTerrain&) = delete;

    // Allocates the arenas and starts the bake workers. `budget` is
    // r_planet_chunk_budget and fixes the arena size for this session --
    // changing it needs a Shutdown/Init pair, which the cvar's on-change
    // does.
    bool Init(pt::rhi::Device* device, const TerrainConfig& cfg);
    void Shutdown();
    bool Ready() const noexcept { return device_ != nullptr && cfg_.enabled; }

    // One frame of streaming. `anchor` is the render frame's anchor, so
    // instance transforms come out camera-relative and an origin rebase is
    // a TLAS update that touches no BLAS. Returns true when the instance
    // array or the descriptors changed and the caller must re-publish.
    bool Update(const pt::planet::LodParams& lod, const glm::dvec3& anchor);

    // The TLAS instances for terrain. instance_id is already the
    // descriptor index (1 + ordinal); slot 0 belongs to the CSG mesh.
    const std::vector<pt::rhi::TLASInstance>& Instances() const noexcept {
        return instances_;
    }
    // Descriptor payload for indices 1..N, laid out as
    // kInstDescFloat4s float4 each. Index 0 is the engine's own.
    const std::vector<float>& DescriptorFloats() const noexcept {
        return descriptors_;
    }

    pt::rhi::BufferHandle VertexBuffer() const noexcept { return vert_buf_; }
    pt::rhi::BufferHandle IndexBuffer()  const noexcept { return index_buf_; }

    const TerrainStats& Stats() const noexcept { return stats_; }
    const pt::planet::PlanetSite& Site() const noexcept { return site_; }
    const pt::planet::ElevationField& Field() const noexcept { return field_; }
    bool HasDem() const noexcept { return field_.HasData(); }

    // Geodetic altitude of a canonical world position above the terrain
    // surface, in metres. Used by the camera-speed scaler, by the world
    // frame's lattice selection, and by physics' ground contact. Caches the
    // last chunk grid it built, so a walk of nearby queries is cheap.
    double AltitudeAboveTerrain(const glm::dvec3& p_world) const;
    // The terrain surface height (ellipsoidal, metres) below a world point.
    double SurfaceHeight(const glm::dvec3& p_world, int level) const;

    // The canonical world position `eye_m` above the terrain, directly
    // below `p_world`. This is what "stand on the surface" means once the
    // ground is a height field: the engine's historical y = 0 is only the
    // surface at the reference site, and only when the elevation there
    // happens to be zero -- with a real DEM the site is at Everest's
    // 8 849 m, and with none it is wherever the fractal put it.
    glm::dvec3 SurfacePosition(const glm::dvec3& p_world, double eye_m) const;

    // Total live TLAS instance count including the reserved mesh slot.
    std::uint32_t TlasCapacity() const noexcept { return tlas_capacity_; }

private:
    struct Resident {
        pt::planet::ChunkKey     key{};
        std::uint32_t            slot = 0;
        std::uint32_t            mask = 0;
        pt::rhi::AccelStructHandle blas{};
        glm::dvec3               origin_w{0.0};
        std::vector<float>       positions;   // kept for a stitch-mask rebuild
    };
    struct Retired {
        std::uint32_t              slot = 0;
        pt::rhi::AccelStructHandle blas{};
        int                        frames = 0;
    };

    bool BuildChunkBlas(Resident& r);
    void RetireChunk(Resident& r);
    void FlushRetired(bool force);
    void RebuildInstanceArray(const glm::dvec3& anchor);

    pt::rhi::Device* device_ = nullptr;
    TerrainConfig    cfg_{};

    pt::planet::PlanetSite            site_{};
    pt::planet::DigitalElevationModel dem_{};
    pt::planet::ElevationField        field_{};
    pt::planet::TerrainIndexArena     index_arena_{};
    pt::planet::TerrainQuadtree       tree_{};
    pt::planet::AsyncChunkBaker       baker_{};

    pt::rhi::BufferHandle vert_buf_{};
    pt::rhi::BufferHandle index_buf_{};

    std::uint32_t              chunk_budget_  = 0;
    std::uint32_t              tlas_capacity_ = 0;
    std::vector<std::uint32_t> free_slots_;
    std::vector<Retired>       retired_;

    std::map<pt::planet::ChunkKey, Resident>                  resident_;
    std::map<pt::planet::ChunkKey, pt::planet::TerrainChunkData> baked_;
    std::vector<pt::planet::ChunkKey>                          requested_;

    std::vector<pt::rhi::TLASInstance> instances_;
    std::vector<float>                 descriptors_;
    TerrainStats                       stats_{};
    glm::dvec3                         last_anchor_{0.0};
    // Previous frame's residency digest; convergence requires two
    // consecutive frames to agree. See the note at the convergence test.
    std::uint64_t                      prev_digest_ = 0;
    bool                               first_update_ = true;

    // One-entry cache for AltitudeAboveTerrain. `mutable` because the query
    // is logically const; guarded by nothing, so it is main-thread only --
    // which is where all three of its callers live.
    mutable pt::planet::ChunkKey cached_key_{};
    mutable bool                 cached_valid_ = false;
    mutable std::vector<double>  cached_grid_;
};

}  // namespace pt::engine

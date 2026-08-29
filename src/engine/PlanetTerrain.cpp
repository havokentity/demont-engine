// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "PlanetTerrain.h"

#include "../core/AssetPath.h"
#include "../core/Log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace pt::engine {

using pt::planet::ChunkKey;
using pt::planet::TerrainChunkData;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Per-vertex shader payload stride, in floats. ONE definition, in
// TerrainChunk.h next to the channel list, so the producer
// (BuildTerrainChunk), the arena size here and the shader's
// StructuredBuffer<float> terrain_verts cannot drift apart.
constexpr std::size_t kVertFloats =
    static_cast<std::size_t>(pt::planet::kVertexPayloadFloats);

}  // namespace

PlanetTerrain::~PlanetTerrain() { Shutdown(); }

bool PlanetTerrain::Init(pt::rhi::Device* device, const TerrainConfig& cfg) {
    Shutdown();
    device_ = device;
    cfg_    = cfg;
    if (device_ == nullptr || !cfg_.enabled) return false;

    if (!device_->SupportsHardwareRT()) {
        // The terrain rides the hardware ray query; there is no software
        // fallback that would be anything but a per-ray loop over a
        // thousand chunks. Say so once and stay off.
        LOG_WARN("planet: terrain requires hardware ray tracing on this "
                 "backend; r_planet_terrain has no effect here");
        cfg_.enabled = false;
        return false;
    }

    site_ = pt::planet::PlanetSite::FromGeodetic(cfg_.site_lat_rad, cfg_.site_lon_rad);

    // --- Elevation data ---------------------------------------------------
    if (!cfg_.dem_path.empty()) {
        std::string err;
        const std::string resolved = pt::ResolveAssetPath(cfg_.dem_path.c_str());
        if (dem_.Load(resolved, err)) {
            LOG_INFO("planet: loaded DEM {} ({}x{}, {:.1f} km/texel at the equator)",
                     resolved, dem_.Width(), dem_.Height(),
                     dem_.TexelAngularSize() * pt::planet::kIuggMeanRadius / 1000.0);
            field_.SetDem(&dem_);
        } else {
            // Loudly, once. A planet with no data is a procedural body, not
            // Earth, and the difference is the entire point of shipping real
            // ETOPO in the first place -- so it is stated rather than
            // silently substituted.
            LOG_WARN("planet: no elevation data ({}) -- rendering a PROCEDURAL "
                     "body, not Earth. Fetch the real ETOPO 2022 grid with:  "
                     "python3 tools/fetch_planet_dem.py",
                     err);
        }
    }
    field_.SetParams(cfg_.field);

    // --- Arenas -----------------------------------------------------------
    // r_planet_chunk_budget is the LEAF cap: the frontier the selector may
    // ask for, and the number EnforceBudget raises tau until the balanced
    // set fits inside. The arena has to hold more than the frontier,
    // because residency is the whole cut (#319) -- every desired leaf plus
    // every interior node above it, kept so a merge has something one level
    // coarser to publish instead of a hole.
    //
    // That surcharge is exact rather than estimated. A quadtree cut with L
    // leaves has exactly (L - 6) / 3 interior nodes, so the arena is
    // WholeCutSlots(leaf budget) and the TLAS one more. Sizing it this way
    // rather than lowering the selector target is what keeps the converged
    // desired set -- and therefore every planet golden -- exactly where it
    // was: Select() is handed the same number it was handed before
    // retention existed.
    leaf_budget_  = static_cast<std::uint32_t>(
        std::clamp(cfg_.lod.chunk_budget, 8, 8192));
    chunk_budget_ = static_cast<std::uint32_t>(
        pt::planet::WholeCutSlots(static_cast<std::size_t>(leaf_budget_)));
    const std::size_t vert_bytes =
        static_cast<std::size_t>(chunk_budget_) *
        static_cast<std::size_t>(pt::planet::kChunkVertexCount) *
        kVertFloats * sizeof(float);
    vert_buf_ = device_->CreateBuffer({
        .size = vert_bytes, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "terrain_verts",
    });
    index_buf_ = device_->CreateBuffer({
        .size = index_arena_.ByteSize(), .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "terrain_indices",
    });
    if (vert_buf_.id == 0 || index_buf_.id == 0) {
        LOG_ERROR("planet: terrain arena allocation failed ({} MB vertices + "
                  "{} KB indices)",
                  vert_bytes / (1024 * 1024), index_arena_.ByteSize() / 1024);
        Shutdown();
        return false;
    }
    device_->WriteBuffer(index_buf_, index_arena_.Indices().data(),
                         index_arena_.ByteSize());

    free_slots_.clear();
    free_slots_.reserve(chunk_budget_);
    // Descending so the first pop is slot 0 -- deterministic slot
    // assignment matters: the arena offset lands in a descriptor the
    // shader reads, so a golden depends on it.
    for (std::uint32_t i = chunk_budget_; i-- > 0;) free_slots_.push_back(i);

    tlas_capacity_ = chunk_budget_ + 1u;   // +1 for the reserved mesh slot

    baker_.Start(std::clamp(cfg_.worker_count, 1, 16));
    baker_.SetSources(&field_, site_);

    LOG_INFO("planet: terrain online -- site {:.4f}N {:.4f}E, budget {} leaves "
             "+ {} retained ancestors = {} arena slots "
             "({:.0f} MB vertex arena + {:.1f} MB index arena), levels {}..{}, "
             "tau {:.2f} px",
             cfg_.site_lat_rad * 180.0 / kPi, cfg_.site_lon_rad * 180.0 / kPi,
             leaf_budget_, chunk_budget_ - leaf_budget_,
             chunk_budget_, static_cast<double>(vert_bytes) / (1024.0 * 1024.0),
             static_cast<double>(index_arena_.ByteSize()) / (1024.0 * 1024.0),
             cfg_.lod.min_level, cfg_.lod.max_level, cfg_.lod.tau_px);
    return true;
}

void PlanetTerrain::Shutdown() {
    baker_.Stop();
    if (device_ != nullptr) {
        for (auto& [k, r] : resident_) {
            if (r.blas.id != 0) device_->DestroyAccelStruct(r.blas);
        }
        FlushRetired(true);
        if (vert_buf_.id  != 0) device_->DestroyBuffer(vert_buf_);
        if (index_buf_.id != 0) device_->DestroyBuffer(index_buf_);
    }
    resident_.clear();
    published_.clear();
    retained_.clear();
    baked_.clear();
    requested_.clear();
    retired_.clear();
    free_slots_.clear();
    instances_.clear();
    descriptors_.clear();
    tree_.Clear();
    vert_buf_ = {};
    index_buf_ = {};
    cached_valid_ = false;
    stats_ = TerrainStats{};
    first_update_ = true;
    device_ = nullptr;
}

bool PlanetTerrain::BuildChunkBlas(Resident& r) {
    pt::rhi::BLASDesc d{};
    d.vertex_positions = r.positions.data();
    d.vertex_count     = static_cast<std::uint32_t>(pt::planet::kChunkVertexCount);
    d.indices          = index_arena_.Variant(r.mask);
    d.index_count      = static_cast<std::uint32_t>(pt::planet::kChunkIndexCount);
    // PreferFastBuild, not the engine-wide default of PreferFastTrace: a
    // chunk is built once and traced for a second or two before the LOD
    // boundary sweeps past it, which is exactly the trade the flag names.
    // P0 (#254) added the field for this.
    d.flags            = pt::rhi::AccelBuildFlags::PreferFastBuild;
    d.debug_name       = "terrain_chunk";
    const auto blas = device_->CreateBLAS(d);
    if (blas.id == 0) return false;
    r.blas = blas;
    ++stats_.blas_builds;
    return true;
}

void PlanetTerrain::RetireChunk(Resident& r) {
    Retired rt;
    rt.slot   = r.slot;
    rt.blas   = r.blas;
    rt.frames = kRetireFrames;
    retired_.push_back(rt);
    ++stats_.evictions;
}

void PlanetTerrain::FlushRetired(bool force) {
    for (auto it = retired_.begin(); it != retired_.end();) {
        if (!force && --it->frames > 0) { ++it; continue; }
        if (it->blas.id != 0 && device_ != nullptr) {
            device_->DestroyAccelStruct(it->blas);
        }
        // kNoSlot marks a retirement that owns ONLY a BLAS: a stitch-mask
        // rebuild retires the superseded acceleration structure while the
        // chunk keeps its arena slot and its vertices. Pushing the sentinel
        // onto the free list corrupted the allocator -- it handed out slot
        // 0xFFFFFFFF, whose arena offset is 68 GB into a 33 MB buffer, and
        // inflated the free list so residency ran past the budget and the
        // TLAS update started refusing more instances than its capacity.
        if (it->slot != kNoSlot) free_slots_.push_back(it->slot);
        it = retired_.erase(it);
    }
}

void PlanetTerrain::RebuildInstanceArray(const glm::dvec3& anchor) {
    instances_.clear();
    instances_.reserve(published_.size());
    descriptors_.assign(static_cast<std::size_t>(published_.size()) *
                            kInstDescFloat4s * 4u,
                        0.0f);

    // The PUBLISHED cover, not every resident chunk. A chunk held through a
    // split and the incoming children underneath it are both resident, and
    // putting both in the TLAS would place two surfaces on the same ground:
    // no depth buffer arbitrates that in a path tracer, so the ray takes
    // whichever the BVH reports first and the shading flickers between two
    // legitimate answers. published_ is a disjoint cover by construction --
    // see pt::planet::ComputeResidencyCover. Still walked in KEY order, so
    // the instance and descriptor layout stays a function of the set rather
    // than of the order it was assembled.
    std::uint32_t ordinal = 0;
    for (const auto& key : published_) {
        const auto rit = resident_.find(key);
        if (rit == resident_.end()) continue;   // cover members are resident
        const Resident& r = rit->second;
        // Descriptor index 0 is the engine's mesh, so terrain starts at 1.
        const std::uint32_t desc_index = ordinal + 1u;
        const glm::vec3 t(r.origin_w - anchor);

        pt::rhi::TLASInstance inst{};
        inst.blas = r.blas;
        // Row-major 3x4, TRANSLATION ONLY. The rotation stays identity on
        // purpose: the shader interpolates baked vertex normals and uses
        // them untransformed, which is only correct because this is a pure
        // translation. Giving a chunk a rotation would silently tilt every
        // normal on it.
        inst.transform[0] = 1; inst.transform[1] = 0; inst.transform[2]  = 0; inst.transform[3]  = t.x;
        inst.transform[4] = 0; inst.transform[5] = 1; inst.transform[6]  = 0; inst.transform[7]  = t.y;
        inst.transform[8] = 0; inst.transform[9] = 0; inst.transform[10] = 1; inst.transform[11] = t.z;
        inst.instance_id  = desc_index;
        inst.mask         = 0xFF;
        instances_.push_back(inst);

        const auto it = baked_.find(key);
        float* d = descriptors_.data() +
                   static_cast<std::size_t>(ordinal) * kInstDescFloat4s * 4u;
        auto put_u = [](float* p, std::uint32_t v) {
            std::memcpy(p, &v, sizeof(float));
        };
        const double spacing = pt::planet::ChunkVertexSpacing(key.level);
        // uv_scale is UV units per metre. One UV tile per vertex spacing
        // gives a texel density that tracks the chunk's own resolution,
        // which is what makes pbrConeTexLod pick a sane mip instead of
        // short-circuiting to 0 the way every mesh hit does today.
        const double uv_scale = 1.0 / std::max(spacing, 1e-6);

        put_u(d + 0, kInstKindTerrain);
        put_u(d + 1, r.slot * static_cast<std::uint32_t>(pt::planet::kChunkVertexCount));
        put_u(d + 2, pt::planet::TerrainIndexArena::VariantOffset(r.mask));
        d[3] = static_cast<float>(uv_scale);
        // Albedo and roughness are overwritten per hit from the biome
        // function; these are the fallbacks for a chunk whose descriptor is
        // read before its vertices land (which the retire ring makes
        // impossible, but a wrong-looking grey beats undefined).
        d[4] = 0.30f; d[5] = 0.30f; d[6] = 0.30f;
        // Base roughness of bare ground before the slope-variance fold.
        // 0.85 is a rough-dielectric surface; real soil and rock are
        // near-Lambertian in the specular lobe.
        d[7] = 0.85f;
        put_u(d + 8, 0u /* MAT_LAMBERT */);
        d[9]  = static_cast<float>(spacing);
        d[10] = 1.0f;                       // ior, unused for MAT_LAMBERT
        d[11] = 0.0f;
        const std::uint32_t no_tile = 0xFFFFFFFFu;
        put_u(d + 12, no_tile); put_u(d + 13, no_tile);
        put_u(d + 14, no_tile); put_u(d + 15, no_tile);
        if (it != baked_.end()) {
            for (int m = 0; m < pt::planet::kSlopeMips; ++m) {
                d[16 + m] = it->second.sigma2[static_cast<std::size_t>(m)];
            }
        }
        ++ordinal;
    }
}

std::vector<ChunkKey> PlanetTerrain::ResidentKeys() const {
    std::vector<ChunkKey> out;
    out.reserve(resident_.size());
    for (const auto& [k, r] : resident_) { (void)r; out.push_back(k); }
    return out;
}

void PlanetTerrain::RetireUncovered(const std::set<ChunkKey>& desired) {
    // The pre-coverage rule: everything the selector stopped wanting goes,
    // right now, whether or not anything covers the ground it stood on.
    // Still what the settling barrier uses -- see the note at its call site.
    for (auto it = resident_.begin(); it != resident_.end();) {
        if (desired.find(it->first) == desired.end()) {
            RetireChunk(it->second);
            baked_.erase(it->first);
            it = resident_.erase(it);
        } else {
            ++it;
        }
    }
}

void PlanetTerrain::ChooseCover(const std::set<ChunkKey>& desired,
                                const pt::planet::LodParams& lod) {
    std::set<ChunkKey> res_keys;
    for (const auto& [k, r] : resident_) { (void)r; res_keys.insert(k); }

    auto cover = pt::planet::ComputeResidencyCover(desired, res_keys);

    // --- The arena gets no headroom, so the hold is what gives -----------
    //
    // Holding an outgoing chunk while its replacements stream in costs a
    // slot the steady state does not need, and there is nowhere to take it
    // from. The selector's target IS the arena size and lowering it would
    // change the converged residency set that every planet golden is pinned
    // against -- a fix that moves the goldens is not a fix.
    //
    // So the holds are released ON DEMAND, against the demand that is
    // actually blocked: chunks the selector wants, whose geometry is
    // already baked, that have no arena slot to go into. Exactly that many
    // stand-ins go, lowest error-per-distance first.
    //
    // The measure is (blocked demand - free slots), and both halves matter.
    // A standing reserve of `chunk_budget_ - |desired|` was the first
    // version and it reads a paced stream as having no slack at all -- the
    // selector runs far ahead of the bake pool, measured 1 023 desired
    // against 158 resident on the descent this is tested against -- so it
    // refused every hold while 863 arena slots sat empty. Netting against
    // the FREE LIST instead of against the budget is what makes the
    // difference: the release fires when the arena is genuinely full and
    // stays silent when it is not, and "genuinely full" is a fact rather
    // than a forecast.
    //
    // It also has to be measured here and not in the add pass. Counting the
    // chunks the add loop turned away conflates "no slot" with "no build
    // budget": the loop breaks out on the millisecond budget, so on a slow
    // host it stops looking long before it has seen the demand, and the
    // deficit comes out near zero exactly when the arena is most clogged.
    // Measured on a Debug build: 5 turned away against 121 stand-ins
    // squatting a 128-slot arena, and the release never fired.
    //
    // Releasing on demand makes the starvation deadlock impossible rather
    // than improbable, which is the point: that bug is self-sustaining --
    // the hold occupies the slot its own replacement needs, so the
    // replacement never arrives and the hold is never retired. Here,
    // wanting the slot is what frees it.
    //
    // What the caller sees when this fires is the old behaviour: the chunk
    // goes and the ground holes out until its replacement lands. That is
    // the right way round to fail. A hole is ugly; an arena overrun hands
    // the TLAS more instances than it has capacity for.
    std::vector<ChunkKey> stand_ins;
    for (const ChunkKey& k : cover.published) {
        if (desired.find(k) == desired.end()) stand_ins.push_back(k);
    }
    // The zoom-out reserve (#319): interior nodes of the desired cut that
    // hold a slot without drawing anything. Disjoint from `stand_ins` by
    // construction -- a retained ancestor the cover DID publish is standing
    // in for ground its children cannot cover, so it is already earning its
    // slot and is counted there instead.
    std::vector<ChunkKey> reserve;
    for (const ChunkKey& k : retained_) {
        if (resident_.find(k) == resident_.end()) continue;
        if (cover.published.find(k) != cover.published.end()) continue;
        reserve.push_back(k);
    }
    std::set<ChunkKey> released;
    // Demand the arena is blocking: desired, baked, and homeless. Retained
    // ancestors are deliberately NOT counted here. The deficit is what
    // authorises a release, and releasing a chunk that is drawing ground to
    // make room for one that is not would invert the whole policy: leaves
    // first, always.
    std::size_t blocked = 0;
    for (const ChunkKey& k : desired) {
        if (resident_.find(k) != resident_.end()) continue;
        if (baked_.find(k) != baked_.end()) ++blocked;
    }
    const std::size_t starved =
        (blocked > free_slots_.size()) ? (blocked - free_slots_.size()) : 0u;
    // Releases already in flight. A retired slot does not rejoin the free
    // list for kRetireFrames, so the deficit this pass sees is the same one
    // the last two passes saw; without netting off its own outstanding
    // releases it would spend the deficit three times and punch three times
    // the holes. See Retired::hold_release for why ORDINARY retirements are
    // deliberately not netted off as well.
    std::size_t releases_in_flight = 0;
    for (const Retired& rt : retired_) {
        if (rt.hold_release) ++releases_in_flight;
    }
    std::size_t deficit =
        (starved > releases_in_flight) ? (starved - releases_in_flight) : 0u;

    // --- The reserve yields before the stand-ins -------------------------
    //
    // Both are slots the steady state does not need, so both are candidates
    // when the arena is genuinely full, but they are not worth the same. A
    // stand-in is DRAWING GROUND this frame; dropping it punches a hole
    // now. A retained ancestor is drawing nothing; dropping it spends
    // insurance against a zoom-out that may not come, and if it does come
    // the cover falls back to the fine tiling underneath -- which is where
    // this policy started. So the reserve goes first and in full before any
    // stand-in is touched.
    //
    // Lowest error-per-distance first within the reserve, which for a chain
    // of ancestors means the FINEST goes first. That is the right end to
    // give up: a missing immediate parent is covered from below by its own
    // four children at a one-level step, which the index arena has a stitch
    // variant for, while a missing grandparent is the two-level step that
    // gets refused and becomes a hole.
    if (deficit > 0 && !reserve.empty()) {
        std::sort(reserve.begin(), reserve.end(),
                  [&](const ChunkKey& a, const ChunkKey& b) {
                      const double pa = tree_.Priority(a, lod);
                      const double pb = tree_.Priority(b, lod);
                      if (pa != pb) return pa < pb;
                      return a < b;
                  });
        const std::size_t drop = std::min(deficit, reserve.size());
        for (std::size_t i = 0; i < drop; ++i) {
            cover.retirable.push_back(reserve[i]);
            released.insert(reserve[i]);
        }
        stats_.retained_dropped += static_cast<std::uint64_t>(drop);
        deficit -= drop;
        reserve.erase(reserve.begin(),
                      reserve.begin() + static_cast<std::ptrdiff_t>(drop));
    }

    const std::size_t cap =
        (deficit >= stand_ins.size()) ? 0u : (stand_ins.size() - deficit);
    if (stand_ins.size() > cap) {
        // Lowest error-per-distance goes first: the same ordering the
        // selector evicts by, so the stand-in that survives is the one whose
        // absence would be most visible. Tie-broken on the key, so two runs
        // of the same state drop the same chunks.
        std::sort(stand_ins.begin(), stand_ins.end(),
                  [&](const ChunkKey& a, const ChunkKey& b) {
                      const double pa = tree_.Priority(a, lod);
                      const double pb = tree_.Priority(b, lod);
                      if (pa != pb) return pa < pb;
                      return a < b;
                  });
        const std::size_t drop = stand_ins.size() - cap;
        for (std::size_t i = 0; i < drop; ++i) {
            cover.published.erase(stand_ins[i]);
            cover.retirable.push_back(stand_ins[i]);
            released.insert(stand_ins[i]);
        }
        stats_.holds_dropped += static_cast<std::uint64_t>(drop);
    }

    published_ = std::move(cover.published);
    stats_.holds_refused += static_cast<std::uint64_t>(cover.refused);
    stats_.repair_rounds = cover.repair_rounds;
    stats_.repair_rounds_peak =
        std::max(stats_.repair_rounds_peak, cover.repair_rounds);
    stats_.held = std::min(stand_ins.size(), cap);
    stats_.starved = starved;
    stats_.retained = reserve.size();

    for (const ChunkKey& k : cover.retirable) {
        // A retained ancestor this pass did NOT release keeps its slot.
        // ComputeResidencyCover reports it as retirable because it is
        // neither desired nor published, which was the whole rule before
        // #319 -- and is exactly the rule that threw the coarse chunks away
        // on the way down and left nothing to publish on the way back up.
        if (released.find(k) == released.end() &&
            retained_.find(k) != retained_.end()) {
            continue;
        }
        auto it = resident_.find(k);
        if (it == resident_.end()) continue;
        RetireChunk(it->second);
        if (released.find(k) != released.end()) retired_.back().hold_release = true;
        baked_.erase(k);
        resident_.erase(it);
    }
}

bool PlanetTerrain::Update(const pt::planet::LodParams& lod,
                           const glm::dvec3& anchor) {
    if (!Ready()) return false;
    FlushRetired(false);

    // --- 1. Drain finished bakes FIRST ------------------------------------
    // Before the selector, not after. Selecting first means the desired set
    // is always computed from a metric map one drain out of date -- and on
    // the frame the last bakes land, the convergence test would then pass
    // against a set that was chosen WITHOUT those metrics. Two runs of a
    // frozen capture converged on genuinely different 1128- and
    // 1074-chunk sets that way, which a chunk count nearly hid and the
    // residency digest made obvious.
    std::vector<TerrainChunkData> fresh;
    // 64 per frame paces an interactive session, where a thousand-chunk
    // drain in one frame would be a visible hitch. A settling round drains
    // EVERYTHING instead, and that is a determinism requirement rather than
    // a throughput one: a 64-chunk prefix of the pool's completion order is
    // a wall-clock quantity, and every decision downstream of it -- which
    // chunks become resident this round, and therefore which arena slot
    // each one gets -- inherits that. See Settle() in the header.
    baker_.Drain(fresh, settling_ ? std::numeric_limits<int>::max() : 64);
    for (auto& d : fresh) {
        tree_.NoteChunk(d);
        baked_[d.key] = std::move(d);
    }

    // --- 2. Selector ------------------------------------------------------
    // The selector target is the LEAF budget, not the arena. EnforceBudget
    // raises tau until the balanced leaf set fits inside it, and the arena
    // is sized at WholeCutSlots(leaf budget) so the ancestors that leaf set
    // implies are already paid for. Handing Select() the arena size instead
    // would let the frontier grow into the slots its own ancestors need.
    pt::planet::LodParams sel = lod;
    sel.chunk_budget = static_cast<int>(leaf_budget_);
    tree_.Select(sel);
    const auto& desired = tree_.Desired();

    // The whole cut: the frontier plus every interior node above it. Held
    // resident, unpublished, so a merge has something one level coarser to
    // publish instead of a hole (#319).
    //
    // EMPTY WHILE SETTLING, and for the same reason step 3a keeps the
    // pre-coverage eviction there: a settle is a barrier with no observer,
    // so there is no ascent to insure against, and putting |desired| / 3
    // extra chunks through the free list would renumber the arena and
    // reorder the BLAS builds that the golden matrix is pinned to.
    retained_ = settling_ ? std::set<ChunkKey>{}
                          : pt::planet::CutAncestors(desired);
    stats_.cut = desired.size() + retained_.size();

    // --- 3. Diff residency ------------------------------------------------
    bool dirty = fresh.empty() ? false : true;
    const std::size_t resident_before = resident_.size();

    // 3a. Evict -- ONLY while settling.
    //
    // Immediate, unconditional eviction is what made the terrain hole out
    // on camera motion: the chunks the selector dropped went away in
    // one step while their replacements were paced over tens of frames. The
    // paced path now retires by coverage instead, in 3c, once the add pass
    // has had its chance to land the replacements.
    //
    // Settling keeps the old rule verbatim, and that is a determinism
    // decision rather than an oversight. A settling round is a barrier with
    // no observer -- Settle() runs to a fixed point before anything is
    // captured -- so there is no transient to smooth. What there IS is a
    // golden matrix pinned to the settled scene down to the arena slot each
    // chunk holds (see PublishedSceneDigest and the #284 note in the
    // header), and slots come off a free list whose order is a function of
    // when each chunk was retired. Holding a chunk two rounds longer would
    // renumber the arena and reorder the BLAS builds for a transient
    // nobody sees.
    if (settling_) RetireUncovered(desired);

    // 3b. Add / restitch, paced against the build budget.
    //
    // EVERYTHING THE BUDGET IS NOT RESPONSIBLE FOR HAPPENS BEFORE THE CLOCK
    // STARTS. r_planet_blas_budget_ms exists because Device::CreateBLAS
    // ends in waitUntilCompleted / vkQueueWaitIdle, so what it is bounding
    // is time spent inside the RHI -- and charging the priority sorts to it
    // does not slow the frame down, it just stops the frame doing any work.
    // Both sorts run a comparator that does two map lookups, so on a Debug
    // host a 700-leaf set costs several milliseconds to ORDER: measured
    // there, the first over_budget() check after the sort was already true
    // and the retained-ancestor pass below admitted nothing, ever. The
    // arena held 699 of a 930-chunk cut for the whole run and the zoom-out
    // it is insurance against holed out exactly as it did before the fix.
    // Release never saw it, which is the shape of a threshold bug.
    //
    // Process in the selector's own priority order so a budgeted frame
    // spends its slots where the geometric error is largest.
    std::vector<ChunkKey> order(desired.begin(), desired.end());
    std::sort(order.begin(), order.end(), [&](const ChunkKey& a, const ChunkKey& b) {
        const double pa = tree_.Priority(a, lod);
        const double pb = tree_.Priority(b, lod);
        if (pa != pb) return pa > pb;
        return a < b;
    });

    // Retained ancestors that are baked and homeless, coarsest-needed
    // first, and the leaf demand the arena is blocking right now. Both are
    // measured against residency AS IT STANDS AT THE TOP OF THE FRAME. The
    // add loop below can only make `blocked` smaller and can only add
    // DESIRED chunks, which are disjoint from the retained set, so a
    // start-of-frame reading is conservative in the safe direction: it can
    // under-admit ancestors, never over-admit them.
    std::size_t blocked_leaves = 0;
    std::vector<ChunkKey> anc_order;
    if (!settling_ && !retained_.empty()) {
        for (const ChunkKey& k : desired) {
            if (resident_.find(k) != resident_.end()) continue;
            if (baked_.find(k) != baked_.end()) ++blocked_leaves;
        }
        anc_order.reserve(retained_.size());
        for (const ChunkKey& k : retained_) {
            if (resident_.find(k) != resident_.end()) continue;
            if (baked_.find(k) == baked_.end()) continue;
            anc_order.push_back(k);
        }
        // Priority is e_L / d and e_L grows with the chunk's span, so
        // highest-first admits the ancestors FURTHEST above the frontier
        // first. That is the right end: a missing immediate parent is
        // covered from below by its own four children at a one-level step
        // the index arena has a stitch variant for, while a missing
        // grandparent is the two-level step that gets refused and becomes
        // the hole.
        std::sort(anc_order.begin(), anc_order.end(),
                  [&](const ChunkKey& a, const ChunkKey& b) {
                      const double pa = tree_.Priority(a, lod);
                      const double pb = tree_.Priority(b, lod);
                      if (pa != pb) return pa > pb;
                      return a < b;
                  });
    }

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    // Unpaced while settling. Pacing decides how much of a round's work
    // lands this round by consulting a clock, which would put wall time
    // back into the residency answer through the same door the drain cap
    // used. A settle is not rendering anything anyone is looking at, so
    // there is no frame to protect.
    const double budget_ms = settling_ ? 0.0 : std::max(cfg_.blas_budget_ms, 0.0);
    auto over_budget = [&]() {
        if (budget_ms <= 0.0) return false;
        const double ms = std::chrono::duration<double, std::milli>(
                              clk::now() - t0).count();
        stats_.last_build_ms = ms;
        return ms >= budget_ms;
    };

    for (const ChunkKey& key : order) {
        auto rit = resident_.find(key);
        // Restitching a chunk that is already resident belongs to the cover
        // pass in the paced path -- doing it here as well, against a
        // different leaf set, would have the two passes rebuild the same
        // BLAS in opposite directions every frame. Leave before computing
        // the mask, not after: StitchMask is four neighbour walks, and at
        // steady state this branch is taken for EVERY leaf, so computing a
        // mask nobody reads was several milliseconds of the build budget
        // per frame on a Debug host.
        if (rit != resident_.end() && !settling_) continue;
        // PROVISIONAL stitch mask, from the desired set. It is the right
        // answer whenever the published cover equals the desired set, which
        // is every frame that is not mid-transition; where it is wrong --
        // a chunk added next to a stand-in that is one level coarser than
        // the selector thinks -- 3d corrects it against the cover, at the
        // cost of one extra BLAS build for that chunk.
        const std::uint32_t mask = tree_.StitchMask(key);
        if (rit != resident_.end()) {
            if (rit->second.mask == mask) continue;
            if (over_budget()) break;
            // The stitch variant changed, so the index set changed and the
            // BLAS has to be rebuilt -- but the VERTICES did not, which is
            // why Resident keeps its positions. Re-baking here would double
            // the generation load for no new geometry.
            const auto old_blas = rit->second.blas;
            rit->second.mask = mask;
            if (BuildChunkBlas(rit->second)) {
                Retired rt; rt.slot = kNoSlot; rt.blas = old_blas;
                rt.frames = kRetireFrames;
                retired_.push_back(rt);
                dirty = true;
            } else {
                rit->second.blas = old_blas;   // keep the old geometry
            }
            continue;
        }
        const auto bit = baked_.find(key);
        if (bit == baked_.end()) continue;      // still baking
        if (free_slots_.empty()) continue;   // arena full; 3c frees it
        if (over_budget()) break;

        Resident r;
        r.key      = key;
        r.mask     = mask;
        r.origin_w = bit->second.origin_w;
        r.positions = bit->second.positions;
        r.slot     = free_slots_.back();
        free_slots_.pop_back();

        const std::size_t stride =
            static_cast<std::size_t>(pt::planet::kChunkVertexCount) * kVertFloats;
        device_->WriteBuffer(vert_buf_, bit->second.shader_verts.data(),
                             stride * sizeof(float),
                             static_cast<std::size_t>(r.slot) * stride * sizeof(float));
        if (!BuildChunkBlas(r)) {
            free_slots_.push_back(r.slot);
            continue;
        }
        resident_.emplace(key, std::move(r));
        dirty = true;
    }

    // --- 3b-bis. Admit retained ancestors, strictly behind the leaves ----
    //
    // Refusing to RETIRE an ancestor is enough while the camera is
    // descending, because the ancestor was a leaf a moment ago and is
    // already resident. It is not enough anywhere the descent skipped a
    // level: TerrainQuadtree::Descend splits as soon as a node's e_L is
    // known, and metrics_ outlives residency, so a camera returning to
    // ground it has already measured goes straight from the root to the
    // deep frontier and never makes the levels between it resident. A
    // Settle() barrier leaves exactly the same gap -- it retires everything
    // outside the frontier by design. Those cuts would still hole out on
    // the way back up, so the missing interior nodes are baked and admitted
    // here.
    //
    // WHAT KEEPS THEM FROM COMPETING WITH THE LEAVES. Three things, in
    // order of how much they matter:
    //
    //   * the arena is sized for both (WholeCutSlots), so at steady state
    //     there is no competition to arbitrate;
    //   * this loop runs AFTER the desired loop and shares its millisecond
    //     budget, so a frame that spent its budget on leaves has none left
    //     to spend here;
    //   * and it will not touch the free slots that this frame's blocked
    //     leaf demand needs. `blocked_leaves` is measured, not forecast --
    //     desired, baked, and homeless at the top of the frame -- which is
    //     the same quantity ChooseCover releases against and for the same
    //     reason: #308 recorded two standing reserves that forecast the
    //     arena's slack and both refused every hold while hundreds of slots
    //     sat empty.
    //
    // The candidate list and its ordering were built before the budget
    // clock started; see the note at 3b for what happened on a Debug host
    // when they were not.
    {
        std::size_t spare = (free_slots_.size() > blocked_leaves)
                                ? (free_slots_.size() - blocked_leaves) : 0u;
        for (const ChunkKey& key : anc_order) {
            if (spare == 0 || free_slots_.empty()) break;
            if (over_budget()) break;
            const auto bit = baked_.find(key);
            if (bit == baked_.end()) continue;

            Resident r;
            r.key      = key;
            // Provisional, and almost always 0: an ancestor's neighbours in
            // the DESIRED set are finer than it is, and StitchMaskFor only
            // sets a bit for a COARSER neighbour. 3d recomputes it against
            // the published cover on the frame this chunk is actually
            // published, which is the only frame the mask means anything.
            r.mask     = tree_.StitchMask(key);
            r.origin_w = bit->second.origin_w;
            r.positions = bit->second.positions;
            r.slot     = free_slots_.back();
            free_slots_.pop_back();

            const std::size_t stride =
                static_cast<std::size_t>(pt::planet::kChunkVertexCount) * kVertFloats;
            device_->WriteBuffer(vert_buf_, bit->second.shader_verts.data(),
                                 stride * sizeof(float),
                                 static_cast<std::size_t>(r.slot) * stride * sizeof(float));
            if (!BuildChunkBlas(r)) {
                free_slots_.push_back(r.slot);
                continue;
            }
            resident_.emplace(key, std::move(r));
            --spare;
            // NOT `dirty`. A retained ancestor is not published, so nothing
            // about the instance array or the descriptors changed; 3c's
            // cover digest is what notices if it does get published.
        }
    }

    // --- 3c. Choose the published cover and retire what it makes redundant.
    //
    // The dirty flag is driven off a DIGEST of the cover, not off its size.
    // A transition can swap one published chunk for another at unchanged
    // count -- a stand-in shifting as the selector's frontier moves -- and
    // a size comparison reports that as "nothing happened", leaving the
    // TLAS holding an instance array for a cover that no longer exists.
    auto cover_digest = [this]() {
        std::uint64_t h = 1469598103934665603ull;
        for (const ChunkKey& k : published_) {
            const std::uint64_t v = (static_cast<std::uint64_t>(k.face)  << 45) |
                                    (static_cast<std::uint64_t>(k.level) << 40) |
                                    (static_cast<std::uint64_t>(k.i)     << 20) |
                                    static_cast<std::uint64_t>(k.j);
            for (int b = 0; b < 8; ++b) {
                h ^= (v >> (b * 8)) & 0xFFull;
                h *= 1099511628211ull;
            }
        }
        return h;
    };
    const std::uint64_t published_before = cover_digest();
    if (settling_) {
        // Immediate eviction already ran, so resident_ is a subset of the
        // desired set and the cover is trivially all of it. Stating that
        // rather than computing it keeps the settling path bit-identical to
        // the pre-coverage one: same instances, same descriptors, same digest.
        published_.clear();
        for (const auto& [k, r] : resident_) { (void)r; published_.insert(k); }
        stats_.held          = 0;
        stats_.starved       = 0;
        stats_.retained      = 0;
        stats_.repair_rounds = 0;
    } else {
        ChooseCover(desired, lod);
    }
    if (cover_digest() != published_before) dirty = true;

    // --- 3d. Reconcile stitch masks against the PUBLISHED cover ----------
    // The mask is what makes the surface watertight without skirts, and it
    // is only meaningful against the set the rays actually intersect. A
    // stand-in held through a split is one level coarser than the desired
    // leaves beside it, so its neighbours have to keep the stitch bit they
    // would otherwise have dropped the moment the selector split it --
    // computing masks from `desired` here would crack the surface exactly
    // where this policy is holding it together.
    //
    // Same budget clock as the add pass. A deferred mask is a crack rather
    // than a hole, which is the one place this change can look worse than
    // what it replaced; it is also the pre-existing exposure of the
    // restitch pass above, not a new one.
    if (!settling_) {
        std::vector<ChunkKey> pub_order(published_.begin(), published_.end());
        std::sort(pub_order.begin(), pub_order.end(),
                  [&](const ChunkKey& a, const ChunkKey& b) {
                      const double pa = tree_.Priority(a, lod);
                      const double pb = tree_.Priority(b, lod);
                      if (pa != pb) return pa > pb;
                      return a < b;
                  });
        for (const ChunkKey& key : pub_order) {
            auto rit = resident_.find(key);
            if (rit == resident_.end()) continue;
            const std::uint32_t want = pt::planet::StitchMaskFor(key, published_);
            if (rit->second.mask == want) continue;
            if (over_budget()) break;
            const auto old_blas = rit->second.blas;
            rit->second.mask = want;
            if (BuildChunkBlas(rit->second)) {
                Retired rt; rt.slot = kNoSlot; rt.blas = old_blas;
                rt.frames = kRetireFrames;
                retired_.push_back(rt);
                dirty = true;
            } else {
                rit->second.blas = old_blas;   // keep the old geometry
            }
        }
    }

    if (resident_.size() != resident_before) dirty = true;

    // Drop baked data for chunks that are neither resident nor desired nor
    // retained, so the CPU cache does not grow without bound as the camera
    // moves. Retained ancestors have to survive this: dropping a bake that
    // has not found an arena slot yet would have step 4 request it again
    // next frame, and the pair would cycle for as long as the arena stayed
    // full.
    for (auto it = baked_.begin(); it != baked_.end();) {
        if (desired.find(it->first) == desired.end() &&
            resident_.find(it->first) == resident_.end() &&
            retained_.find(it->first) == retained_.end()) {
            it = baked_.erase(it);
        } else {
            ++it;
        }
    }

    // --- 4. Request the next bake round ----------------------------------
    requested_.clear();
    for (const ChunkKey& k : tree_.Wanted()) {
        if (baked_.find(k) == baked_.end()) requested_.push_back(k);
    }
    // Also queue desired-but-unbaked chunks that the selector already knows
    // the metric for (they were evicted under budget and came back).
    for (const ChunkKey& k : order) {
        if (resident_.find(k) != resident_.end()) continue;
        if (baked_.find(k) != baked_.end()) continue;
        if (std::find(requested_.begin(), requested_.end(), k) != requested_.end()) continue;
        requested_.push_back(k);
    }
    // Retained ancestors LAST. AsyncChunkBaker::Request stores the list
    // reversed and the workers pop the back, so position in this vector is
    // priority -- putting the ancestors at the tail means the pool bakes
    // them only once every leaf it has been asked for is either done or in
    // flight. Same rule as the arena: leaves first.
    if (!retained_.empty()) {
        std::vector<ChunkKey> anc(retained_.begin(), retained_.end());
        std::sort(anc.begin(), anc.end(), [&](const ChunkKey& a, const ChunkKey& b) {
            const double pa = tree_.Priority(a, lod);
            const double pb = tree_.Priority(b, lod);
            if (pa != pb) return pa > pb;
            return a < b;
        });
        for (const ChunkKey& k : anc) {
            if (resident_.find(k) != resident_.end()) continue;
            if (baked_.find(k) != baked_.end()) continue;
            requested_.push_back(k);
        }
    }
    baker_.Request(requested_);

    // --- 5. Publish -------------------------------------------------------
    if (anchor != last_anchor_) { dirty = true; last_anchor_ = anchor; }
    if (dirty || first_update_) {
        RebuildInstanceArray(anchor);
        first_update_ = false;
        ++stats_.tlas_updates;
    }

    stats_.resident      = resident_.size();
    stats_.resident_peak = std::max(stats_.resident_peak, stats_.resident);
    stats_.published     = published_.size();
    stats_.desired       = desired.size();
    // How much of the whole cut is actually in the arena. Members of the
    // cut, not merely a count of residents: a stale chunk from the previous
    // camera is resident too, and it insures nothing.
    std::size_t desired_resident = 0;
    for (const ChunkKey& k : desired) {
        if (resident_.find(k) != resident_.end()) ++desired_resident;
    }
    std::size_t retained_resident = 0;
    for (const ChunkKey& k : retained_) {
        if (resident_.find(k) != resident_.end()) ++retained_resident;
    }
    stats_.cut_resident = desired_resident + retained_resident;
    stats_.pending_bakes = requested_.size() +
                           static_cast<std::size_t>(baker_.InFlight());
    // "Converged" for the capture gate: the selector has measured every
    // leaf it wants, nothing is baking, and every desired chunk is resident.
    const std::uint64_t digest = tree_.DesiredDigest();
    // CONVERGED means "the answer stopped changing", not "the answer
    // happens to reference only measured nodes right now". With the baker
    // idle and the queue empty the metric map cannot grow, so a digest that
    // matches the previous frame's is a genuine fixed point -- and that is
    // the only statement strong enough to pin a golden against.
    // `resident_.size() == desired.size()` was the pre-#319 form and it now
    // says the wrong thing on the paced path, where residency is the whole
    // cut and legitimately exceeds the frontier. The claim the capture gate
    // wants is "every chunk the selector asked for is in the arena", so
    // that is what is asserted. On the settling path the two are the same
    // statement -- RetireUncovered has already made resident_ a subset of
    // desired -- so the barrier stops exactly where it stopped before.
    stats_.converged = tree_.Converged() && baker_.Idle() &&
                       desired_resident == desired.size() &&
                       requested_.empty() && digest == prev_digest_;
    prev_digest_ = digest;
    stats_.desired_digest = digest;
    return dirty;
}

std::uint64_t PlanetTerrain::PublishedSceneDigest() const {
    // FNV-1a over the published scene in KEY order. resident_ is a
    // std::map and instances_ / descriptors_ are built from it in that same
    // walk, so the digest depends on the scene and not on the order it was
    // assembled -- which is exactly the property under test.
    //
    // The raw vertex arena is deliberately NOT hashed. Chunk geometry is a
    // pure function of the key and the elevation field, both of which are
    // fixed for a session, so hashing 4 225 vertices per chunk would cost
    // millions of mixes to restate what the key set already fixes. The slot
    // is hashed instead, because the slot is where the arrival-order
    // dependence actually lived.
    auto mix = [](std::uint64_t h, std::uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (b * 8)) & 0xFFull;
            h *= 1099511628211ull;
        }
        return h;
    };
    auto mix_f = [&mix](std::uint64_t h, float f) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        return mix(h, bits);
    };
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& [key, r] : resident_) {
        h = mix(h, key.face); h = mix(h, key.level);
        h = mix(h, key.i);    h = mix(h, key.j);
        h = mix(h, r.slot);   h = mix(h, r.mask);
    }
    for (const auto& inst : instances_) {
        for (int t = 0; t < 12; ++t) h = mix_f(h, inst.transform[t]);
        h = mix(h, inst.instance_id);
    }
    for (float f : descriptors_) h = mix_f(h, f);
    return h;
}

bool PlanetTerrain::Settle(const pt::planet::LodParams& lod,
                           const glm::dvec3& anchor, int max_rounds) {
    settle_rounds_ = 0;
    if (!Ready()) return true;
    // Re-entrancy would nest two barriers on one bake pool and the inner
    // one would clear `settling_` on the way out, quietly returning the
    // outer rounds to paced, partially-drained behaviour.
    if (settling_) return stats_.converged;

    settling_ = true;
    bool converged = false;
    for (int r = 0; r < max_rounds; ++r) {
        // Wait for the round the LAST Update requested. On the first
        // iteration nothing has been asked for yet and this returns at
        // once, which is correct: an empty request set is a settled one.
        baker_.WaitIdle();
        Update(lod, anchor);
        settle_rounds_ = r + 1;
        if (stats_.converged) { converged = true; break; }
    }
    settling_ = false;
    stats_.scene_digest = PublishedSceneDigest();
    return converged;
}

double PlanetTerrain::SurfaceHeight(const glm::dvec3& p_world, int level) const {
    // The FIELD direction, not the geocentric one -- see
    // EcefToFieldDirection. They differ by up to 8.8 km on the surface.
    const glm::dvec3 dir =
        pt::planet::EcefToFieldDirection(site_.WorldToEcef(p_world));
    const int lvl = std::clamp(level, 0, pt::planet::kMaxLevel);
    int face = 0; double s = 0.0, t = 0.0;
    pt::planet::DirectionToFaceParam(dir, face, s, t);
    const auto span = static_cast<double>(1u << lvl);
    auto cell = [span](double p) -> std::uint32_t {
        const double c = (p + 1.0) * 0.5 * span;
        const auto idx = static_cast<std::int64_t>(std::floor(c));
        return static_cast<std::uint32_t>(
            std::clamp<std::int64_t>(idx, 0, static_cast<std::int64_t>(span) - 1));
    };
    const ChunkKey key{static_cast<std::uint8_t>(face),
                       static_cast<std::uint8_t>(lvl), cell(s), cell(t)};
    if (!cached_valid_ || !(cached_key_ == key)) {
        field_.GenerateChunkHeights(key, lvl, 0, cached_grid_);
        cached_key_   = key;
        cached_valid_ = true;
    }
    const std::int64_t G = static_cast<std::int64_t>(key.Span());
    const double s0 = pt::planet::GridParam(key.i,     G);
    const double s1 = pt::planet::GridParam(key.i + 1, G);
    const double t0 = pt::planet::GridParam(key.j,     G);
    const double t1 = pt::planet::GridParam(key.j + 1, G);
    const double fu = std::clamp((s - s0) / (s1 - s0), 0.0, 1.0) * pt::planet::kChunkQuads;
    const double fv = std::clamp((t - t0) / (t1 - t0), 0.0, 1.0) * pt::planet::kChunkQuads;
    const int xi = std::clamp(static_cast<int>(std::floor(fu)), 0, pt::planet::kChunkQuads - 1);
    const int yi = std::clamp(static_cast<int>(std::floor(fv)), 0, pt::planet::kChunkQuads - 1);
    const double ax = fu - xi;
    const double ay = fv - yi;
    auto at = [&](int x, int y) {
        return cached_grid_[static_cast<std::size_t>(y) * pt::planet::kChunkVerts + x];
    };
    return (at(xi, yi) * (1.0 - ax) + at(xi + 1, yi) * ax) * (1.0 - ay) +
           (at(xi, yi + 1) * (1.0 - ax) + at(xi + 1, yi + 1) * ax) * ay;
}

glm::dvec3 PlanetTerrain::SurfacePosition(const glm::dvec3& p_world,
                                          double eye_m) const {
    const glm::dvec3 dir =
        pt::planet::EcefToFieldDirection(site_.WorldToEcef(p_world));
    // Query at the finest level the session allows: this places a camera or
    // a body, and being a metre out is the difference between standing on
    // the ground and standing inside it.
    const double h = SurfaceHeight(p_world, cfg_.lod.max_level);
    const glm::dvec3 surf = pt::planet::EllipsoidSurface(dir);
    const glm::dvec3 n    = pt::planet::GeodeticNormal(surf);
    return site_.EcefToWorld(surf + (h + eye_m) * n);
}

double PlanetTerrain::AltitudeAboveTerrain(const glm::dvec3& p_world) const {
    const glm::dvec3 e = site_.WorldToEcef(p_world);
    const double r = glm::length(e);
    if (!(r > 0.0)) return 0.0;
    const glm::dvec3 dir = pt::planet::EcefToFieldDirection(e);
    // Ellipsoidal height of the point itself: distance from the ellipsoid
    // surface along the geodetic normal. For an ECEF point outside the body
    // this is |p| - |surface(dir)| to first order, and the correction is
    // second order in (h/R) -- 8e-4 m at 400 km, which is far below the
    // metre this drives (a camera speed and a lattice choice).
    const double surface_r = glm::length(pt::planet::EllipsoidSurface(dir));
    const double h_point   = r - surface_r;
    // Query the terrain at a level whose vertex spacing is comparable to
    // the altitude itself, floored at 12 (38 m spacing). Sampling level 19
    // from orbit would cost a full fine-grid bake for a number that only
    // has to be right to a few percent.
    int level = 19;
    const double want = std::max(std::abs(h_point) * 0.25, 1.0);
    for (int l = 0; l <= pt::planet::kMaxLevel; ++l) {
        if (pt::planet::ChunkVertexSpacing(l) <= want) { level = l; break; }
    }
    level = std::clamp(level, 0, pt::planet::kMaxLevel);
    return h_point - SurfaceHeight(p_world, level);
}

}  // namespace pt::engine

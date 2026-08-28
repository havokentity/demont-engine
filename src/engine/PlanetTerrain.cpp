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

// Per-vertex shader payload stride, in floats. Mirrors
// StructuredBuffer<float4> terrain_verts in PathTrace.slang.
constexpr std::size_t kVertFloats = 4;

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
    // r_planet_chunk_budget is the HARD resource cap -- the arena is sized
    // for exactly this many chunks and the TLAS for one more. The selector
    // is given a lower target (see Update) because its own budget
    // enforcement is soft at the boundary: merging under pressure can
    // violate the 2:1 restriction, and the re-balance that repairs it can
    // push the set back over. The 2:1 invariant has to win that tie --
    // breaking it produces cracks -- so the arena carries the margin
    // instead. Measured overshoot on the planet_surface fixture was 14%;
    // the target leaves 20%.
    chunk_budget_ = static_cast<std::uint32_t>(
        std::clamp(cfg_.lod.chunk_budget, 8, 8192));
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

    LOG_INFO("planet: terrain online -- site {:.4f}N {:.4f}E, budget {} chunks "
             "({:.0f} MB vertex arena + {:.1f} MB index arena), levels {}..{}, "
             "tau {:.2f} px",
             cfg_.site_lat_rad * 180.0 / kPi, cfg_.site_lon_rad * 180.0 / kPi,
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
    instances_.reserve(resident_.size());
    descriptors_.assign(static_cast<std::size_t>(resident_.size()) *
                            kInstDescFloat4s * 4u,
                        0.0f);

    std::uint32_t ordinal = 0;
    for (auto& [key, r] : resident_) {
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
    // The selector target IS the arena size: EnforceBudget raises tau until
    // the balanced set fits, so the cap is hard on both sides.
    pt::planet::LodParams sel = lod;
    sel.chunk_budget = static_cast<int>(chunk_budget_);
    tree_.Select(sel);
    const auto& desired = tree_.Desired();

    // --- 3. Diff residency ------------------------------------------------
    bool dirty = fresh.empty() ? false : true;

    // 3a. Evict.
    for (auto it = resident_.begin(); it != resident_.end();) {
        if (desired.find(it->first) == desired.end()) {
            RetireChunk(it->second);
            baked_.erase(it->first);
            it = resident_.erase(it);
            dirty = true;
        } else {
            ++it;
        }
    }

    // 3b. Add / restitch, paced against the build budget. The clock starts
    // here rather than at the top of Update so the selector's own cost does
    // not eat the AS budget it is not responsible for.
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

    // Process in the selector's own priority order so a budgeted frame
    // spends its slots where the geometric error is largest.
    std::vector<ChunkKey> order(desired.begin(), desired.end());
    std::sort(order.begin(), order.end(), [&](const ChunkKey& a, const ChunkKey& b) {
        const double pa = tree_.Priority(a, lod);
        const double pb = tree_.Priority(b, lod);
        if (pa != pb) return pa > pb;
        return a < b;
    });

    for (const ChunkKey& key : order) {
        const std::uint32_t mask = tree_.StitchMask(key);
        auto rit = resident_.find(key);
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
        if (free_slots_.empty()) continue;      // arena full; budget guards this
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

    // Drop baked data for chunks that are neither resident nor desired, so
    // the CPU cache does not grow without bound as the camera moves.
    for (auto it = baked_.begin(); it != baked_.end();) {
        if (desired.find(it->first) == desired.end() &&
            resident_.find(it->first) == resident_.end()) {
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
    baker_.Request(requested_);

    // --- 5. Publish -------------------------------------------------------
    if (anchor != last_anchor_) { dirty = true; last_anchor_ = anchor; }
    if (dirty || first_update_) {
        RebuildInstanceArray(anchor);
        first_update_ = false;
        ++stats_.tlas_updates;
    }

    stats_.resident      = resident_.size();
    stats_.desired       = desired.size();
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
    stats_.converged = tree_.Converged() && baker_.Idle() &&
                       resident_.size() == desired.size() &&
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

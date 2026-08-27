// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary P4 (#258): residency selection and asynchronous baking ------
//
// THE LOD METRIC, AND WHY A CAMERA-DERIVED ONE IS SAFE IN A PATH TRACER
// ---------------------------------------------------------------------
// A path tracer cannot select LOD per ray -- every ray intersects the same
// TLAS -- so a camera-derived LOD has to be proven conservative for ALL
// rays, not just primary ones. The engine's own ray-cone formulation makes
// that proof one line.
//
// From shaders/PathTrace.slang the cone at bounce b is
//
//     cone_width_b = cone_spread * sum_{k <= b} t_k
//
// i.e. cone_spread times the TOTAL PATH LENGTH from the eye to the hit.
// By the triangle inequality the total path length is at least the straight
// distance |p - camera|, so for every ray in every path
//
//     cone_width at a hit on p  >=  cone_spread * |p - camera|.
//
// The camera metric is therefore a LOWER BOUND on every ray's footprint --
// the right direction of conservatism. A chunk whose geometric error is
// sub-threshold against cone_spread * dist(camera, chunk) is sub-threshold
// for the puddle bounce, for the mirror reflecting the horizon, and for the
// eighth diffuse bounce.
//
// Two bounded caveats, both already true of the engine's texture LOD.
// Refractive or concave magnification can produce a footprint SMALLER than
// cone_spread * distance; cone_width carries no curvature term, so
// pbrConeTexLod already over-blurs through a magnifier and terrain is no
// worse. r_planet_lod_min_level is the floor against it. Rough reflections
// should widen the cone and do not, which makes geometry FINER than needed
// behind a rough surface -- a cost issue, not a correctness one.
//
// SPLIT RULE
// ----------
//     split when   e_L / d  >  tau * cone_spread
// with e_L the baked Ulrich (2002) delta in metres, d the distance from
// the camera to the chunk's bounding sphere, cone_spread = 2*tan(fovY/2) /
// height -- the EXACT expression the shader computes per pixel, evaluated
// once on the host -- and tau the permitted error in pixels.
//
// At tau = 0.5 every LOD transition is provably sub-pixel, which DELETES
// the need for geomorphing rather than paying for it. Geomorphing in a ray
// tracer means rebuilding the BLAS every frame through the morph, which is
// unaffordable at these rates.
//
// HYSTERESIS IS LOAD-BEARING
// --------------------------
// A chunk oscillating across the threshold issues a BLAS build and a BLAS
// destroy every frame. Rearranged as a distance, the rule splits at
// d < e_L / (tau * cone_spread) and merges only at
// d > hysteresis * e_L / (tau * cone_spread) -- so a chunk that just split
// has to recede 40% further before it merges again.
//
// 2:1 RESTRICTION
// ---------------
// Neighbouring leaves differ by at most one level, enforced by iterating
// the balance pass to a fixed point. That gives 2^4 = 16 index-buffer
// variants (see TerrainIndexArena), which is what makes the surface
// watertight without skirts.
//
// EVICTION BY PRIORITY
// --------------------
// Engine::AcquirePbrTile's mark-and-sweep reclaims the first UNREFERENCED
// tile. Chunks are never unreferenced -- the whole planet is always
// potentially visible -- so the selector needs the one thing that scheme
// lacks: eviction by priority, largest e_L/d wins. Over budget, the leaves
// with the smallest error-per-distance are merged into their parents until
// the set fits.

#include "CubedSphere.h"
#include "ElevationField.h"
#include "TerrainChunk.h"

#include <glm/glm.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace pt::planet {

struct LodParams {
    // r_planet_lod_error_px. Permitted geometric error in pixels.
    double tau_px      = 0.5;
    // r_planet_lod_hysteresis. Merge distance is this multiple of the split
    // distance.
    double hysteresis  = 1.4;
    int    min_level   = 0;
    int    max_level   = kMaxLevel;
    int    chunk_budget = 1024;
    // 2 * tan(fovY/2) / height -- the shader's own cone_spread.
    double cone_spread = 1.0e-3;
    // Camera position in canonical world coordinates.
    glm::dvec3 camera_w{0.0};
    // r_planet_lod_freeze: hold the residency set exactly as it is, so a
    // golden capture is not a function of wall clock.
    bool   freeze = false;
};

// What the selector knows about a node it has seen baked.
struct ChunkMetric {
    double     e_l_m = 0.0;
    glm::dvec3 center_w{0.0};
    double     radius_m = 0.0;
    bool       known = false;
};

class TerrainQuadtree {
public:
    // Recompute the desired leaf set. Nodes whose e_L is not yet known
    // cannot be split -- they become leaves and are reported by Wanted(),
    // so the tree descends one level per completed bake round rather than
    // guessing at an error it has not measured.
    void Select(const LodParams& p);

    const std::set<ChunkKey>& Desired() const noexcept { return desired_; }

    // Desired leaves whose metric is not yet known, most useful first.
    const std::vector<ChunkKey>& Wanted() const noexcept { return wanted_; }

    // 4-bit stitch mask for a desired leaf: bit `edge` set when the
    // neighbour across that edge is one level COARSER, in which case this
    // chunk drops its odd vertices along that edge to match.
    std::uint32_t StitchMask(const ChunkKey& k) const;

    // Record a completed bake so the node becomes splittable.
    void NoteChunk(const TerrainChunkData& d);

    // True when nothing in the desired set is unknown -- i.e. the selector
    // has converged for this camera. The capture gate needs this AND an
    // idle baker AND a residency set equal to the desired set.
    bool Converged() const noexcept { return wanted_.empty(); }

    void Clear();

    std::size_t MetricCount() const noexcept { return metrics_.size(); }

    // Priority of a node: e_L / d, the same quantity the split rule uses.
    // Larger means "more badly needed". Exposed for the eviction order.
    double Priority(const ChunkKey& k, const LodParams& p) const;

private:
    void Descend(const ChunkKey& k, const LodParams& p, bool parent_was_split);
    void BuildSet(const LodParams& p);
    void Balance(const LodParams& p);
    void EnforceBudget(const LodParams& p);
    int  LeafLevelAt(const ChunkKey& probe) const;

    std::map<ChunkKey, ChunkMetric> metrics_;
    // Set once every desired leaf has a measured e_L. Gates the freeze:
    // see the note at the top of Select().
    bool                            converged_once_ = false;
    std::set<ChunkKey>              desired_;
    std::set<ChunkKey>              prev_desired_;
    std::vector<ChunkKey>           wanted_;
};

// --- Asynchronous chunk baking --------------------------------------------
//
// Shaped after pt::renderer::AsyncLightTreeBuilder (src/renderer/LightTree.h),
// NOT after the CSG bake. The CSG pattern blanks the screen while a bake is
// in flight (Engine::RenderFrame's loading-frame gate) and
// JobSystem::Wait mandates a blocking join per submit, so neither is usable
// for something that runs tens of times a second. This is a small persistent
// worker pool with a priority-ordered request queue and a non-blocking
// result drain; the render thread never waits on it.
class AsyncChunkBaker {
public:
    AsyncChunkBaker() = default;
    ~AsyncChunkBaker();
    AsyncChunkBaker(const AsyncChunkBaker&)            = delete;
    AsyncChunkBaker& operator=(const AsyncChunkBaker&) = delete;

    void Start(int workers);
    void Stop();
    bool Running() const noexcept { return !workers_.empty(); }

    // The field must outlive the baker. Changing either source bumps a
    // generation counter, so results baked against the old one are dropped
    // rather than published into a scene that no longer matches them.
    void SetSources(const ElevationField* field, const PlanetSite& site);

    // Replace the pending request list. Requests already in flight finish;
    // queued-but-unstarted ones are discarded. Cheap enough to call every
    // frame.
    void Request(const std::vector<ChunkKey>& keys);

    // Move up to `max` finished bakes out. Non-blocking.
    int Drain(std::vector<TerrainChunkData>& out, int max);

    int  InFlight() const noexcept { return in_flight_.load(std::memory_order_acquire); }
    bool Idle() const;

private:
    void WorkerLoop();

    std::vector<std::thread>      workers_;
    std::atomic<bool>             stop_{false};
    std::atomic<int>              in_flight_{0};
    std::atomic<std::uint32_t>    generation_{0};

    mutable std::mutex            mu_;
    std::condition_variable       cv_;
    std::vector<ChunkKey>         queue_;         // back() is next
    const ElevationField*         field_ = nullptr;
    PlanetSite                    site_{};

    std::mutex                    out_mu_;
    std::vector<TerrainChunkData> done_;
};

}  // namespace pt::planet

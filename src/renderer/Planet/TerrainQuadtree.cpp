// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "TerrainQuadtree.h"

#include <algorithm>
#include <cmath>

namespace pt::planet {

namespace {

// Distance from the camera to a chunk's bounding sphere, floored so a
// camera standing INSIDE the sphere does not divide by zero.
//
// The floor is the finest feature the SCENE can represent -- the vertex
// spacing at max_level, 0.30 m at level 19 -- and deliberately not the
// chunk's own spacing. Using the chunk's own was the first version and it
// is wrong in a way that only shows at coarse levels: a level-0 chunk's
// spacing is 156 km, so a camera standing on one had its distance floored
// at 156 km, and the split rule compared that against a split distance of
// 22 km at tau = 8 px and concluded "no split". The result was a
// non-monotonic tau -- RAISING the permitted error made the planet coarser
// at level 0 and therefore removed the very LOD boundaries a high-tau
// fixture exists to look at. One scene-wide floor makes tau monotone
// again, which it has to be for the rule to mean what its cvar says.
double DistanceToChunk(const ChunkMetric& m, const glm::dvec3& cam,
                       double floor_m) noexcept {
    const double d = glm::length(cam - m.center_w) - m.radius_m;
    return std::max(d, floor_m);
}

}  // namespace

// --- TerrainQuadtree ------------------------------------------------------

void TerrainQuadtree::Clear() {
    metrics_.clear();
    desired_.clear();
    prev_desired_.clear();
    wanted_.clear();
    converged_once_ = false;
}

void TerrainQuadtree::NoteChunk(const TerrainChunkData& d) {
    ChunkMetric m;
    m.e_l_m    = d.e_l_m;
    m.center_w = d.bound_center_w;
    m.radius_m = d.bound_radius_m;
    m.known    = true;
    metrics_[d.key] = m;
}

double TerrainQuadtree::Priority(const ChunkKey& k, const LodParams& p) const {
    const auto it = metrics_.find(k);
    if (it == metrics_.end() || !it->second.known) return 1e30;  // unknown = urgent
    const double d = DistanceToChunk(it->second, p.camera_w,
                                     ChunkVertexSpacing(p.max_level));
    return it->second.e_l_m / d;
}

void TerrainQuadtree::Descend(const ChunkKey& k, const LodParams& p,
                              bool parent_was_split) {
    (void)parent_was_split;
    if (k.level >= p.max_level) { desired_.insert(k); return; }

    const auto it = metrics_.find(k);
    if (it == metrics_.end() || !it->second.known) {
        // Never baked, so its e_L is unmeasured. Do not guess -- make it a
        // leaf and ask for it. The tree then descends one level per
        // completed bake round, which is also what makes the freeze gate
        // meaningful: "converged" is a statement about measurements taken,
        // not about a heuristic settling.
        desired_.insert(k);
        wanted_.push_back(k);
        return;
    }

    if (k.level < p.min_level) {
        for (int q = 0; q < 4; ++q) Descend(k.Child(q), p, true);
        return;
    }

    const ChunkMetric& m = it->second;
    const double d = DistanceToChunk(m, p.camera_w,
                                     ChunkVertexSpacing(p.max_level));
    // Split distance: the range at which the chunk's own error reaches tau
    // pixels. Beyond hysteresis * that, merge.
    const double denom = std::max(p.tau_px * p.cone_spread, 1e-12);
    const double d_split = m.e_l_m / denom;
    // "Was split" means k was an INTERIOR node last frame -- neither a leaf
    // itself nor below one. Testing only `not in prev_desired_` would also
    // catch nodes that did not exist last frame (their parent was the leaf),
    // and loosening the threshold for those would let a chunk skip straight
    // past its own split distance on the frame it first appears.
    bool was_split = true;
    for (ChunkKey a = k;; a = a.Parent()) {
        if (prev_desired_.find(a) != prev_desired_.end()) { was_split = false; break; }
        if (a.level == 0) break;
    }
    const double threshold = was_split ? d_split * std::max(p.hysteresis, 1.0)
                                       : d_split;
    if (d < threshold) {
        for (int q = 0; q < 4; ++q) Descend(k.Child(q), p, true);
    } else {
        desired_.insert(k);
    }
}

int TerrainQuadtree::LeafLevelAt(const ChunkKey& probe) const {
    ChunkKey k = probe;
    for (;;) {
        if (desired_.find(k) != desired_.end()) return static_cast<int>(k.level);
        if (k.level == 0) return -1;
        k = k.Parent();
    }
}

void TerrainQuadtree::Balance(const LodParams& p) {
    // Iterate to a fixed point. Each pass splits any leaf that is more than
    // one level coarser than one of its four edge neighbours; splitting can
    // create new violations, hence the loop. Bounded by max_level passes
    // because every pass strictly increases the minimum level of some node.
    for (int pass = 0; pass <= p.max_level + 1; ++pass) {
        std::vector<ChunkKey> to_split;
        for (const ChunkKey& leaf : desired_) {
            for (int e = 0; e < 4; ++e) {
                ChunkKey nb{};
                if (!NeighborChunk(leaf, static_cast<ChunkEdge>(e), nb)) continue;
                // Walk DOWN from the neighbour's level to find the finest
                // descendant present, by checking whether any of the
                // neighbour's own resident ancestors is coarser than
                // leaf.level - 1.
                const int nl = LeafLevelAt(nb);
                if (nl >= 0 && nl < static_cast<int>(leaf.level) - 1) {
                    // The neighbour's resident ancestor is too coarse.
                    ChunkKey anc = nb;
                    while (static_cast<int>(anc.level) > nl) anc = anc.Parent();
                    to_split.push_back(anc);
                }
            }
        }
        if (to_split.empty()) return;
        for (const ChunkKey& k : to_split) {
            if (desired_.erase(k) == 0) continue;
            for (int q = 0; q < 4; ++q) {
                const ChunkKey c = k.Child(q);
                desired_.insert(c);
                const auto it = metrics_.find(c);
                if (it == metrics_.end() || !it->second.known) wanted_.push_back(c);
            }
        }
    }
}

void TerrainQuadtree::BuildSet(const LodParams& p) {
    desired_.clear();
    wanted_.clear();
    for (int f = 0; f < 6; ++f) {
        Descend(ChunkKey{static_cast<std::uint8_t>(f), 0, 0, 0}, p, false);
    }
    Balance(p);
}

void TerrainQuadtree::EnforceBudget(const LodParams& p) {
    const auto budget = static_cast<std::size_t>(std::max(6, p.chunk_budget));
    if (desired_.size() <= budget) return;

    // OVER BUDGET: raise tau until the set fits, and rebuild the WHOLE set
    // at the higher threshold.
    //
    // The obvious implementation -- merge the lowest-priority sibling groups
    // one at a time -- does not work, and the reason is worth recording. A
    // merged parent is one level coarser than its neighbours' children, so
    // the 2:1 restriction is violated and the re-balance splits it straight
    // back. Merging in isolation reaches an equilibrium ABOVE the budget
    // rather than at it: measured 837 leaves against a 512-chunk arena on
    // the planet_surface fixture, which then could not converge at all
    // because the arena physically could not hold the set. The 2:1
    // invariant has to win that tie (breaking it produces cracks), so the
    // budget has to be spent somewhere the invariant does not object to.
    //
    // Raising tau is exactly that: it is the SAME rule everywhere, so the
    // result is balanced by construction, it degrades smoothly (unlike
    // capping the level, which quarters the count in one step), and it says
    // something honest -- "this camera cannot be served at 0.5 px within
    // the budget, here is the error it can be served at". Eviction is still
    // by priority in the sense the design asked for: a uniformly higher tau
    // merges precisely the leaves with the smallest e_L / d first.
    //
    // Geometric bisection on the multiplier, 12 iterations, from a bracket
    // grown until it fits. Deterministic and a pure function of the metrics
    // and the camera, which the golden matrix requires.
    LodParams q = p;
    double lo = 1.0;
    double hi = 2.0;
    for (int grow = 0; grow < 20; ++grow) {
        q.tau_px = p.tau_px * hi;
        BuildSet(q);
        if (desired_.size() <= budget) break;
        lo = hi;
        hi *= 2.0;
    }
    for (int i = 0; i < 12; ++i) {
        const double mid = std::sqrt(lo * hi);
        q.tau_px = p.tau_px * mid;
        BuildSet(q);
        if (desired_.size() <= budget) hi = mid; else lo = mid;
    }
    q.tau_px = p.tau_px * hi;
    BuildSet(q);
}

void TerrainQuadtree::Select(const LodParams& p) {
    // The freeze only engages once the tree has CONVERGED at least once.
    //
    // Freezing before that pins whatever partial set happened to exist --
    // and because the tree descends one level per completed bake round (a
    // node's e_L is not knowable until the node has been baked), the set on
    // the first frame is the six cube roots. A capture fixture that sets
    // r_planet_lod_freeze 1 up front would therefore render a six-triangle-
    // patch planet, which is exactly what the first run of the
    // planet_surface cell produced: "converged after 2 settle frames, 6
    // chunks resident". The freeze is a statement about the CAMERA, not
    // about the measurements.
    if (p.freeze && converged_once_ && !desired_.empty()) {
        wanted_.clear();
        for (const ChunkKey& k : desired_) {
            const auto it = metrics_.find(k);
            if (it == metrics_.end() || !it->second.known) wanted_.push_back(k);
        }
        return;
    }
    prev_desired_ = desired_;
    BuildSet(p);
    EnforceBudget(p);

    // Most-needed first, so a budgeted bake round spends its slots where
    // the error is largest. Deterministic tie-break on the key keeps two
    // runs in step.
    // EnforceBudget can merge away a node Descend or Balance asked for;
    // baking it would cost a worker slot for geometry nothing references.
    wanted_.erase(std::remove_if(wanted_.begin(), wanted_.end(),
                                 [&](const ChunkKey& k) {
                                     return desired_.find(k) == desired_.end();
                                 }),
                  wanted_.end());
    std::sort(wanted_.begin(), wanted_.end(),
              [&](const ChunkKey& a, const ChunkKey& b) {
                  const double pa = Priority(a, p);
                  const double pb = Priority(b, p);
                  if (pa != pb) return pa > pb;
                  return a < b;
              });
    wanted_.erase(std::unique(wanted_.begin(), wanted_.end()), wanted_.end());
    if (wanted_.empty()) converged_once_ = true;
}

std::uint32_t TerrainQuadtree::StitchMask(const ChunkKey& k) const {
    std::uint32_t mask = 0;
    for (int e = 0; e < 4; ++e) {
        ChunkKey nb{};
        if (!NeighborChunk(k, static_cast<ChunkEdge>(e), nb)) continue;
        const int nl = LeafLevelAt(nb);
        // Only a COARSER neighbour makes this chunk stitch: it drops its odd
        // vertices along that edge so its boundary polyline becomes exactly
        // the coarse neighbour's. A finer neighbour does its own stitching
        // in the other direction.
        if (nl >= 0 && nl < static_cast<int>(k.level)) mask |= (1u << e);
    }
    return mask;
}

// --- AsyncChunkBaker ------------------------------------------------------

AsyncChunkBaker::~AsyncChunkBaker() { Stop(); }

void AsyncChunkBaker::Start(int workers) {
    if (!workers_.empty()) return;
    stop_.store(false, std::memory_order_release);
    const int n = std::clamp(workers, 1, 16);
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) workers_.emplace_back([this] { WorkerLoop(); });
}

void AsyncChunkBaker::Stop() {
    if (workers_.empty()) return;
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
    workers_.clear();
    {
        std::lock_guard lk(mu_);
        queue_.clear();
    }
    {
        std::lock_guard lk(out_mu_);
        done_.clear();
    }
    in_flight_.store(0, std::memory_order_release);
}

void AsyncChunkBaker::SetSources(const ElevationField* field, const PlanetSite& site) {
    {
        std::lock_guard lk(mu_);
        field_ = field;
        site_  = site;
        queue_.clear();
    }
    generation_.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard lk(out_mu_);
        done_.clear();
    }
}

void AsyncChunkBaker::Request(const std::vector<ChunkKey>& keys) {
    {
        std::lock_guard lk(mu_);
        // Highest priority is consumed first, and the workers pop from the
        // back, so store reversed.
        queue_.assign(keys.rbegin(), keys.rend());
    }
    cv_.notify_all();
}

int AsyncChunkBaker::Drain(std::vector<TerrainChunkData>& out, int max) {
    std::lock_guard lk(out_mu_);
    int n = 0;
    while (!done_.empty() && n < max) {
        out.push_back(std::move(done_.back()));
        done_.pop_back();
        ++n;
    }
    return n;
}

bool AsyncChunkBaker::Idle() const {
    if (in_flight_.load(std::memory_order_acquire) != 0) return false;
    std::lock_guard lk(mu_);
    return queue_.empty();
}

void AsyncChunkBaker::WorkerLoop() {
    for (;;) {
        ChunkKey key{};
        const ElevationField* field = nullptr;
        PlanetSite site{};
        std::uint32_t gen = 0;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [this] {
                return stop_.load(std::memory_order_acquire) || !queue_.empty();
            });
            if (stop_.load(std::memory_order_acquire)) return;
            key = queue_.back();
            queue_.pop_back();
            field = field_;
            site  = site_;
            gen   = generation_.load(std::memory_order_acquire);
        }
        if (field == nullptr) continue;
        in_flight_.fetch_add(1, std::memory_order_acq_rel);
        TerrainChunkData data;
        BuildTerrainChunk(key, *field, site, data);
        if (generation_.load(std::memory_order_acquire) == gen) {
            std::lock_guard lk(out_mu_);
            done_.push_back(std::move(data));
        }
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace pt::planet

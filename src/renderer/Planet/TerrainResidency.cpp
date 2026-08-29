// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "TerrainResidency.h"

#include <algorithm>

namespace pt::planet {

namespace {

// A ChunkKey packed into one integer, so the "does anything live strictly
// below this node" index can be a sorted vector of uint64 and a binary
// search rather than a tree of 16-byte keys. face < 6, level <= 19 and
// i, j < 2^19 all fit; the layout is order-preserving with respect to
// ChunkKey::operator< but nothing here depends on that.
constexpr std::uint64_t Pack(const ChunkKey& k) noexcept {
    return (static_cast<std::uint64_t>(k.face)  << 45) |
           (static_cast<std::uint64_t>(k.level) << 40) |
           (static_cast<std::uint64_t>(k.i)     << 20) |
           static_cast<std::uint64_t>(k.j);
}

// The set of nodes that have at least one member of `keys` STRICTLY below
// them. Built once per cover computation: every key contributes its whole
// ancestor chain, which is at most kMaxLevel entries, so this is
// O(|keys| * depth) pushes and one sort -- comparable to a single pass of
// TerrainQuadtree::Balance, which runs at least twice per Select().
class AncestorIndex {
public:
    void Build(const std::set<ChunkKey>& keys) {
        packed_.clear();
        packed_.reserve(keys.size() * 4);
        for (const ChunkKey& k : keys) {
            ChunkKey a = k;
            while (a.level > 0) {
                a = a.Parent();
                packed_.push_back(Pack(a));
            }
        }
        std::sort(packed_.begin(), packed_.end());
        packed_.erase(std::unique(packed_.begin(), packed_.end()), packed_.end());
    }
    bool Has(const ChunkKey& k) const noexcept {
        return std::binary_search(packed_.begin(), packed_.end(), Pack(k));
    }

private:
    std::vector<std::uint64_t> packed_;
};

// The walk that turns (desired, resident) into a disjoint published cover.
//
// One instance per ComputeResidencyCover call; Run() may be invoked several
// times as the balance repair refuses substitutions, and each run starts
// from scratch against a larger `refused` set. Refusal only ever grows, so
// the loop terminates.
class CoverBuilder {
public:
    CoverBuilder(const std::set<ChunkKey>& desired,
                 const std::set<ChunkKey>& resident)
        : desired_(desired), resident_(resident) {
        res_anc_.Build(resident);
        des_anc_.Build(desired);
    }

    // Rebuild the cover. Returns false when some desired leaf's area could
    // not be covered at all -- the cold start, and the aftermath of a
    // refusal.
    bool Run() {
        out_.clear();
        bool complete = true;
        for (int f = 0; f < 6; ++f) {
            const ChunkKey root{static_cast<std::uint8_t>(f), 0, 0, 0};
            if (!Emit(root)) complete = false;
        }
        return complete;
    }

    const std::vector<ChunkKey>& Emitted() const noexcept { return out_; }

    // The substitution a published key belongs to, or false when the key is
    // a desired leaf published at its own level (no substitution at all).
    //
    //   coarse stand-in : the key itself -- it is not desired and the
    //                     selector wants something finer inside it.
    //   fine cover      : the desired leaf above it, which is the node whose
    //                     area is being covered from below.
    bool SubstitutionRoot(const ChunkKey& k, ChunkKey& out) const {
        if (desired_.find(k) != desired_.end()) return false;
        if (des_anc_.Has(k)) { out = k; return true; }     // coarse stand-in
        ChunkKey a = k;
        while (a.level > 0) {
            a = a.Parent();
            if (desired_.find(a) != desired_.end()) { out = a; return true; }
        }
        // Neither desired, nor above a desired leaf, nor below one. The
        // selector's set is a partition of the sphere, so this is
        // unreachable for any key the walk can emit; treat it as its own
        // root rather than silently returning "not a substitution".
        out = k;
        return true;
    }

    void Refuse(const ChunkKey& k) { refused_.insert(k); }
    std::size_t RefusedCount() const noexcept { return refused_.size(); }
    bool IsRefused(const ChunkKey& k) const {
        return refused_.find(k) != refused_.end();
    }

private:
    // Cover k's area, which the caller has established nothing above k will
    // publish. Returns true when k's area came out fully covered OR when
    // this subtree has declined an ancestor stand-in; false asks the caller
    // to stand in for k.
    bool Emit(const ChunkKey& k) {
        const bool is_desired = desired_.find(k) != desired_.end();
        if (is_desired) {
            if (resident_.find(k) != resident_.end()) {
                out_.push_back(k);
                return true;
            }
            // Desired but not resident. Finer leftovers may still tile it --
            // this is the merge transition, where the four children have not
            // been retired yet because their parent has not arrived. A
            // PARTIAL tiling is emitted too: three of four children is three
            // quarters of the ground still drawn, and the alternative is to
            // black out the whole quad because one sibling is late.
            if (!IsRefused(k)) return EmitBelow(k);
            return false;    // ask an ancestor to stand in
        }
        if (des_anc_.Has(k)) {
            // Interior node of the desired tree. Try the children first --
            // the selector's own answer is always preferred to a stand-in.
            const std::size_t mark = out_.size();
            bool all = true;
            for (int q = 0; q < 4; ++q) {
                if (!Emit(k.Child(q))) all = false;
            }
            if (all) return true;
            // Something below is uncovered. Stand in with k if we can, and
            // drop everything the children emitted: k spans all of it, and
            // publishing both would put two surfaces on the same ground.
            if (resident_.find(k) != resident_.end() && !IsRefused(k)) {
                out_.resize(mark);
                out_.push_back(k);
                return true;
            }
            // k cannot stand in. Keep whatever the children managed (a
            // partial cover beats none) and let the caller try.
            return false;
        }
        // Strictly below the desired frontier and not reached through a
        // desired leaf: nothing here is wanted.
        return false;
    }

    // Emit the COARSEST resident tiling of k's area from STRICTLY below,
    // and report whether it came out complete. Coarsest rather than finest:
    // fewer instances, fewer BLASes, and a smaller step against whatever is
    // published across k's boundary.
    bool EmitBelow(const ChunkKey& k) {
        if (k.level >= kMaxLevel) return false;
        bool full = true;
        for (int q = 0; q < 4; ++q) {
            const ChunkKey c = k.Child(q);
            if (resident_.find(c) != resident_.end()) { out_.push_back(c); continue; }
            if (!res_anc_.Has(c)) { full = false; continue; }
            if (!EmitBelow(c)) full = false;
        }
        return full;
    }

    const std::set<ChunkKey>& desired_;
    const std::set<ChunkKey>& resident_;
    AncestorIndex             res_anc_;
    AncestorIndex             des_anc_;
    std::set<ChunkKey>        refused_;
    std::vector<ChunkKey>     out_;
};

}  // namespace

int LeafLevelIn(const std::set<ChunkKey>& leaves, const ChunkKey& probe,
                ChunkKey& out_leaf) noexcept {
    ChunkKey k = probe;
    for (;;) {
        if (leaves.find(k) != leaves.end()) {
            out_leaf = k;
            return static_cast<int>(k.level);
        }
        if (k.level == 0) return -1;
        k = k.Parent();
    }
}

int LeafLevelIn(const std::set<ChunkKey>& leaves, const ChunkKey& probe) noexcept {
    ChunkKey unused{};
    return LeafLevelIn(leaves, probe, unused);
}

std::uint32_t StitchMaskFor(const ChunkKey& k,
                            const std::set<ChunkKey>& leaves) noexcept {
    std::uint32_t mask = 0;
    for (int e = 0; e < 4; ++e) {
        ChunkKey nb{};
        if (!NeighborChunk(k, static_cast<ChunkEdge>(e), nb)) continue;
        const int nl = LeafLevelIn(leaves, nb);
        // Only a COARSER neighbour makes this chunk stitch: it drops its odd
        // vertices along that edge so its boundary polyline becomes exactly
        // the coarse neighbour's. A finer neighbour stitches on its own side.
        if (nl >= 0 && nl < static_cast<int>(k.level)) mask |= (1u << e);
    }
    return mask;
}

bool IsEdgeBalanced(const std::set<ChunkKey>& leaves) noexcept {
    for (const ChunkKey& k : leaves) {
        for (int e = 0; e < 4; ++e) {
            ChunkKey nb{};
            if (!NeighborChunk(k, static_cast<ChunkEdge>(e), nb)) continue;
            const int nl = LeafLevelIn(leaves, nb);
            // nl < 0 means the region across is finer than k; that side of
            // the pair reports the same violation when the loop reaches it,
            // so checking one direction is enough.
            if (nl >= 0 && static_cast<int>(k.level) - nl >= 2) return false;
        }
    }
    return true;
}

bool CoversAll(const std::set<ChunkKey>& cover,
               const std::set<ChunkKey>& reference) noexcept {
    // Recursive descent, written iteratively over an explicit stack so a
    // pathological set cannot blow the C stack.
    AncestorIndex below;
    below.Build(cover);
    std::vector<ChunkKey> stack;
    for (const ChunkKey& r : reference) {
        stack.assign(1, r);
        while (!stack.empty()) {
            const ChunkKey k = stack.back();
            stack.pop_back();
            if (LeafLevelIn(cover, k) >= 0) continue;      // an ancestor spans it
            if (k.level >= kMaxLevel) return false;
            if (!below.Has(k)) return false;               // nothing under it
            for (int q = 0; q < 4; ++q) stack.push_back(k.Child(q));
        }
    }
    return true;
}

std::set<ChunkKey> CutAncestors(const std::set<ChunkKey>& leaves) {
    std::set<ChunkKey> out;
    for (const ChunkKey& k : leaves) {
        ChunkKey a = k;
        while (a.level > 0) {
            a = a.Parent();
            // Once a node is in, its whole chain to the root is in --
            // recorded by whichever leaf reached it first. Stopping there
            // turns the naive O(|leaves| * depth) walk into one probe per
            // leaf plus one insert per ancestor, which matters because this
            // runs every frame beside the cover walk.
            if (!out.insert(a).second) break;
        }
    }
    return out;
}

std::size_t WholeCutSlots(std::size_t leaf_budget) noexcept {
    // I = (L - 6) / 3, and L is congruent to 0 mod 3 for any cut (L = 6 +
    // 3I), so integer division is exact at the largest admissible L rather
    // than a floor that loses a slot: the biggest cut inside a budget of B
    // is the largest multiple of 3 not exceeding B, whose interior count is
    // (B - 6) / 3 truncated. Six roots and no interior nodes is the floor.
    if (leaf_budget <= 6) return leaf_budget;
    return leaf_budget + (leaf_budget - 6) / 3;
}

ResidencyCover ComputeResidencyCover(const std::set<ChunkKey>& desired,
                                     const std::set<ChunkKey>& resident) {
    ResidencyCover r{};
    if (resident.empty()) {
        r.complete = desired.empty();
        return r;
    }

    CoverBuilder builder(desired, resident);

    for (int round = 0;; ++round) {
        r.repair_rounds = round;
        r.complete = builder.Run();
        r.published.clear();
        r.published.insert(builder.Emitted().begin(), builder.Emitted().end());

        // Count substitutions. At steady state -- resident == desired --
        // there are none, and the balance scan below is skipped entirely,
        // so an idle camera pays for one walk and nothing else.
        r.substitutions = 0;
        for (const ChunkKey& k : r.published) {
            if (desired.find(k) == desired.end()) ++r.substitutions;
        }
        if (r.substitutions == 0) break;

        // --- 2:1 repair ---------------------------------------------------
        // The index arena only has variants for a ONE-level step, so a
        // published pair two levels apart is a crack no stitch mask can
        // close. Refuse the substitutions responsible; the region they were
        // covering reverts to a hole, which is what it would have been
        // before this policy existed.
        std::vector<ChunkKey> to_refuse;
        for (const ChunkKey& k : r.published) {
            for (int e = 0; e < 4; ++e) {
                ChunkKey nb{};
                if (!NeighborChunk(k, static_cast<ChunkEdge>(e), nb)) continue;
                ChunkKey across{};
                const int nl = LeafLevelIn(r.published, nb, across);
                if (nl < 0) continue;   // finer or empty across; see below
                if (static_cast<int>(k.level) - nl < 2) continue;
                // k is at least two levels finer than the chunk across the
                // edge. At least one of the two is a substitution -- a pair
                // of plain desired leaves cannot be two levels apart,
                // Balance() having already made the desired set 2:1.
                //
                // Refuse the COARSE side by preference. Refusing a coarse
                // stand-in does not black its region out: the walk keeps
                // whatever its children could cover, which is finer, so the
                // step shrinks and the ground stays drawn. Refusing the fine
                // side is the one that costs a hole, so it is the fallback
                // and not the first move.
                ChunkKey root{};
                if (builder.SubstitutionRoot(across, root)) {
                    if (!builder.IsRefused(root)) to_refuse.push_back(root);
                } else if (builder.SubstitutionRoot(k, root)) {
                    if (!builder.IsRefused(root)) to_refuse.push_back(root);
                }
            }
        }
        if (to_refuse.empty()) break;                 // balanced
        if (round + 1 >= kMaxRepairRounds) {
            // Backstop: refuse every substitution. The published set is then
            // a subset of the desired set, and a subset of a balanced leaf
            // set is balanced -- LeafLevelIn returns -1 for the regions that
            // dropped out, which is the "nothing across" case and not a
            // step. Exact, not approximate.
            for (const ChunkKey& k : r.published) {
                ChunkKey root{};
                if (builder.SubstitutionRoot(k, root)) builder.Refuse(root);
            }
            r.repair_rounds = round + 1;
            r.complete = builder.Run();
            r.published.clear();
            r.published.insert(builder.Emitted().begin(), builder.Emitted().end());
            r.substitutions = 0;
            break;
        }
        // std::set iteration order and a sorted refusal set keep this a pure
        // function of the two inputs; nothing here consults arrival order.
        for (const ChunkKey& k : to_refuse) builder.Refuse(k);
    }

    r.refused = builder.RefusedCount();

    // --- The retirement rule --------------------------------------------
    // A resident chunk earns its slot by being wanted by the selector or by
    // appearing in the cover. Anything else is drawing nothing: either its
    // ground is covered without it -- the split parent whose children have
    // all landed, the merged children whose parent has arrived -- or it is a
    // substitution this pass refused. Retiring it is visually free, which is
    // the property the caller needs and the one the old unconditional
    // eviction did not have.
    for (const ChunkKey& k : resident) {
        if (desired.find(k) != desired.end()) continue;
        if (r.published.find(k) != r.published.end()) continue;
        r.retirable.push_back(k);
    }
    return r;
}

}  // namespace pt::planet

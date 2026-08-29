// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Chunk residency: the published cover and the retirement rule ---------
//
// THE BUG THIS EXISTS TO FIX
//
// PlanetTerrain::Update used to evict in one step and add in another:
// every resident chunk the selector no longer wanted was retired
// immediately and unconditionally, while its replacements were paced
// against r_planet_blas_budget_ms and could take tens of frames to bake and
// build. So the instant the camera moved, the terrain HOLED OUT along the
// whole LOD boundary and filled back in a few frames later. The eviction
// was correct about WHAT is no longer wanted and wrong about WHEN it stops
// being needed: a chunk is needed until something else covers the ground it
// stands on.
//
// This unit answers exactly that question and nothing else. It is pure --
// two key sets in, a cover and a retirement list out -- so it can be
// asserted directly instead of inferred from a picture, and so that
// residency stays a function of (camera, metrics, params) with no wall
// clock anywhere in it. That last property is not decoration: see the
// hysteresis note in TerrainQuadtree.cpp for what happened the last time a
// wall-clock quantity reached a residency decision.
//
// WHAT "COVERED" MEANS
//
// Chunks are quadtree nodes, so a node's area is covered by a set S iff
// either an ancestor-or-self is in S (a coarser chunk spans it) or all four
// child subtrees are covered by S (finer chunks tile it). The retirement
// rule is then one sentence:
//
//     a chunk may be retired once its area is covered WITHOUT it.
//
// which the cover below turns into something even easier to check: the
// cover is chosen FIRST, and a resident chunk that neither appears in it
// nor is wanted by the selector is by definition drawing nothing, so
// retiring it cannot take anything off the screen. Specialised to the two
// cases the streamer actually generates:
//
//     split   parent -> 4 children : the parent stays until all four
//                                    children are resident;
//     merge   4 children -> parent : the children stay until the parent is.
//
// OVERLAP IS PREVENTED BY CONSTRUCTION, NOT BY A DEPTH TEST
//
// This engine path-traces. Two coincident surfaces are a real artefact --
// the ray hits whichever the BVH reports first, so a held parent and its
// children both in the TLAS would double-shade and z-fight with no depth
// buffer to arbitrate. So residency and PUBLICATION are separated. A chunk
// may be resident (arena slot + BLAS) without being published (in the TLAS
// instance array), and the published set is built here as a DISJOINT COVER:
// the walk partitions each node's area among its children, so no two
// published chunks can overlap. A parent held through a split is published
// while its children are incomplete; the moment the fourth child lands the
// children are published and the parent is both unpublished and retirable,
// in the same Update, before anything reaches the TLAS.
//
// THE 2:1 RESTRICTION SURVIVES THE TRANSITION
//
// Stitch masks are what make the surface watertight without skirts, and a
// mask is only meaningful against the set that is actually on screen. A
// held parent is one level coarser than the desired leaves beside it, so
// masks are computed against the PUBLISHED set here (StitchMaskFor), not
// against the desired set -- otherwise the neighbours would drop the stitch
// bit the frame the selector split the parent and crack against the parent
// that is still standing in for it.
//
// That leaves the other half of the guarantee: the published set must
// itself be 2:1 balanced, because the index arena only has variants for a
// ONE-level step (TerrainIndexArena's 16 variants). The desired set is
// balanced by TerrainQuadtree::Balance; substituting a coarser stand-in or
// a finer cover into it can open a two-level step. Those are found and
// repaired by refusing the offending substitution -- which leaves a hole
// exactly where a crack would otherwise have been. A hole is the status quo
// ante and a crack is a new defect, so when the two conflict the hole wins.

#include "CubedSphere.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace pt::planet {

// Level of the unique ancestor-or-self of `probe` present in `leaves` --
// i.e. of the leaf whose area contains `probe`'s. Returns -1 when there is
// none, which for a leaf set means the region there is FINER than `probe`,
// or empty.
int LeafLevelIn(const std::set<ChunkKey>& leaves, const ChunkKey& probe) noexcept;

// As LeafLevelIn but also reports which key was found.
int LeafLevelIn(const std::set<ChunkKey>& leaves, const ChunkKey& probe,
                ChunkKey& out_leaf) noexcept;

// 4-bit stitch mask for `k` against an arbitrary leaf set: bit `edge` is set
// when the leaf across that edge is COARSER, in which case `k` drops its odd
// vertices along the edge to match. TerrainQuadtree::StitchMask is this
// function evaluated on the desired set; the streamer evaluates it on the
// PUBLISHED set, which is the set the rays actually see.
std::uint32_t StitchMaskFor(const ChunkKey& k,
                            const std::set<ChunkKey>& leaves) noexcept;

// True when no two leaves sharing an edge differ by more than one level --
// the restriction the 16 index variants are built for.
bool IsEdgeBalanced(const std::set<ChunkKey>& leaves) noexcept;

// How many repair rounds the 2:1 balance loop gets before it gives up and
// refuses every substitution at once. Each round refuses at least one
// substitution and refusals never un-refuse, so the loop is already finite;
// this bounds the WORK rather than the recursion. Eight is far above
// anything measured (transitions repair in zero or one round) and the
// backstop is exact, not approximate: with every substitution refused the
// published set is a subset of the desired set, and a subset of a balanced
// leaf set is balanced.
//
// Exposed so a caller can tell an ordinary repair from the backstop. The
// backstop is not a slow frame -- it publishes nothing but desired leaves
// that happen to be resident, so it holes out the whole transient.
inline constexpr int kMaxRepairRounds = 8;

// True when every leaf of `reference` has its area covered by `cover`.
bool CoversAll(const std::set<ChunkKey>& cover,
               const std::set<ChunkKey>& reference) noexcept;

// --- RETAINING THE WHOLE CUT, NOT JUST ITS LEAVES (#319) ------------------
//
// The retirement rule above protects a chunk that is CURRENTLY RESIDENT.
// It cannot protect one that was thrown away rounds ago, and the selector
// throws ancestors away by construction: TerrainQuadtree::Descend records a
// key only where the descent STOPS, so `desired` is the leaf frontier and
// every interior node of the cut is, to the streamer, an unwanted chunk.
//
// Descending is safe -- the parent is resident when its children are asked
// for, and the rule holds it until they land. ASCENDING is not. On a merge
// the selector asks for a node that was evicted several levels ago, and
// what is resident in its place is the fine tiling underneath it. That
// tiling covers the ground, but it is more than one level below whatever is
// published across the merge boundary, so the 2:1 repair refuses it and
// leaves a hole -- and the hole lasts until the coarse chunk has been baked
// (~2 ms) and built, paced against r_planet_blas_budget_ms. Over a cut of
// several hundred chunks that is hundreds of frames: the seconds of holes a
// zoom-out shows.
//
// The fix is to keep the WHOLE cut resident rather than its frontier: every
// strict ancestor of every desired leaf stays in the arena, unpublished,
// as the standing answer to "what covers this ground one level coarser".
// Ascending then degrades to briefly-coarser detail instead of a hole.
//
// WHAT IT COSTS, EXACTLY
//
// Not "about a third" -- exactly a third, and the identity is what lets the
// arena be sized for it instead of made to compete for it. Every interior
// node of a quadtree cut has exactly four children, so a forest of six
// cube-face roots with I interior nodes has 6 + 4I nodes in total, of which
// L = 6 + 4I - I = 6 + 3I are leaves. Hence
//
//     I = (L - 6) / 3          exactly, for any cut BuildSet can produce
//
// and the arena that must hold a cut of at most `leaf_budget` leaves is
// leaf_budget + (leaf_budget - 6) / 3 slots. WholeCutSlots is that number.
// The selector's target stays the LEAF budget, so the converged desired set
// -- which every planet golden is pinned against -- is untouched.

// Every strict ancestor of every leaf in `leaves`: the interior nodes of
// the cut whose frontier is `leaves`. A pure function of the leaf set, with
// no clock and no arrival order in it, so retaining them keeps residency a
// function of (camera, metrics, params).
std::set<ChunkKey> CutAncestors(const std::set<ChunkKey>& leaves);

// Arena slots the whole cut of a `leaf_budget`-leaf frontier can occupy:
// leaf_budget + (leaf_budget - 6) / 3, from the identity above. Exact
// rather than a fudge factor -- an arena sized to this can never be made to
// choose between a desired leaf and a retained ancestor at steady state.
std::size_t WholeCutSlots(std::size_t leaf_budget) noexcept;

struct ResidencyCover {
    // The disjoint cover to hand the TLAS. Every member is resident.
    std::set<ChunkKey>    published;
    // Resident chunks that are neither wanted by the selector nor needed as
    // a stand-in: safe to retire this frame.
    std::vector<ChunkKey> retirable;
    // Published chunks that are not desired leaves -- coarse stand-ins held
    // through a split, or finer leftovers covering a merge target that has
    // not arrived. Zero at steady state, which is what lets the caller skip
    // the balance scan entirely.
    std::size_t           substitutions = 0;
    // Substitutions refused because publishing them would have opened a
    // two-level step. Diagnostic; a non-zero value means a hole was
    // preferred to a crack.
    std::size_t           refused = 0;
    // Repair rounds consumed. Diagnostic. Reaching kMaxRepairRounds means
    // the backstop fired and every substitution was refused, which is a
    // published set with holes wherever the desired set is not resident.
    int                   repair_rounds = 0;
    // True when every desired leaf's area is covered. False during the cold
    // start (nothing resident yet) and wherever a substitution was refused.
    bool                  complete = true;
};

// Choose what to publish and what may be retired.
//
// `desired` is the selector's answer -- a complete, 2:1-balanced leaf
// partition of the sphere. `resident` is what actually has an arena slot
// and a BLAS. Pure: no clock, no allocator, no device.
ResidencyCover ComputeResidencyCover(const std::set<ChunkKey>& desired,
                                     const std::set<ChunkKey>& resident);

}  // namespace pt::planet

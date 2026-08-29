// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Terrain chunk residency on the PACED streaming path.
//
// WHY THIS FILE EXISTS SEPARATELY FROM pt_planet_terrain
//
// "When I move the camera around, tiles disappear and then load." The
// streamer used to evict every chunk the selector stopped wanting in one
// unconditional step, while the replacements were paced against
// r_planet_blas_budget_ms and took tens of frames to bake and build, so the
// terrain holed out along the whole LOD boundary on every camera motion.
//
// The existing planet suite could not see that, and could not have. Every
// planet fixture goes through PlanetTerrain::Settle(), which runs UNPACED --
// `budget_ms = settling_ ? 0.0 : ...` -- and iterates to a fixed point
// before a single pixel is captured. A settled capture is by construction a
// fully-resident one, so the transient is not merely untested there, it is
// unobservable. This is the project's recurring vacuous-check shape: the
// assertions are true, and blind.
//
// So this file never calls Settle(). It builds a real PlanetTerrain on a
// real RHI device, flies a camera, and ticks Update() with a REALISTIC
// blas_budget_ms -- the paced branch, the one an interactive session takes
// -- asserting the coverage invariant on every single tick while chunks are
// still in flight. A version of this test that settled first and then
// checked for holes would be another vacuous check, and is explicitly not
// what is here.
//
// WHAT IS ASSERTED, EVERY TICK
//
//   1. COVERAGE NEVER REGRESSES. Ground that was being drawn last tick is
//      still being drawn this tick. This is the bug, stated exactly.
//   2. THE PUBLISHED SET DOES NOT OVERLAP ITSELF. A path tracer has no
//      depth buffer; two coincident surfaces is a real artefact, not a
//      z-fight. The cover is an antichain -- no published chunk is an
//      ancestor of another.
//   3. THE PUBLISHED SET IS 2:1 BALANCED. The index arena has variants for
//      a one-level step and nothing else, so a two-level step is a crack
//      that no stitch mask can close.
//   4. EVERY PUBLISHED CHUNK IS RESIDENT, and residency never exceeds the
//      arena. The arena is a hard cap and holding an outgoing chunk
//      alongside its incoming replacement is exactly the pressure that
//      could break it. Note the cap is PlanetTerrain::ArenaSlots() and no
//      longer r_planet_chunk_budget: since #319 the cvar is the LEAF
//      budget and the arena is WholeCutSlots of it, because residency is
//      the whole cut and not its frontier.
//
// AND THE ONE THE COVERAGE GATE COULD NOT SEE (#319)
//
// Coverage-aware retirement protects a chunk that is resident. It cannot
// protect one that was evicted rounds ago, and the selector evicted every
// ancestor of a visible leaf by construction -- `desired` is the leaf
// frontier. Descending was safe (the parent is resident and held until its
// children land); ASCENDING was not, because the coarse chunk the merge
// asks for went away on the way down and has to be re-baked at ~2 ms a
// chunk, paced. That is the seconds of holes a zoom-out shows, and it is
// invisible to a descent-then-climb flight that never caught up in the
// first place: the coarse chunks are still resident there because they were
// never replaced. The zoom-out case below therefore starts from a CAUGHT-UP
// arena -- settle, then run paced until the streamer goes quiescent -- and
// only then climbs.
//
// AND WHAT MAKES IT NON-VACUOUS
//
// All four hold trivially over a stream where nothing ever moves. The run
// therefore also records that the transient the policy exists for actually
// happened -- chunks were held past the selector dropping them, and
// substitutions were actually published -- and fails if it did not. Units
// are metres and seconds throughout; the camera flies a real descent
// profile at real speeds rather than teleporting between two poses.

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "engine/PlanetTerrain.h"
#include "renderer/Planet/TerrainResidency.h"
#include "rhi/Device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace pt::planet;

namespace {

// --- Flight profile (metres, seconds) -------------------------------------
//
// A descent from 60 km -- high enough that the whole visible cap is coarse
// -- to 4 km, which at these LOD parameters is deep enough to have split the
// tree several levels under the camera, followed by the climb back out. The
// descent exercises splits (parent -> four children) and the climb exercises
// merges (four children -> parent), which are the two transitions the
// retirement rule specialises to.
constexpr double kAltStart_m   = 60000.0;
constexpr double kAltBottom_m  = 4000.0;
// 220 m/s ground speed: a jet airliner's cruise, and fast enough that the
// LOD boundary sweeps rather than creeps.
constexpr double kGroundSpeed_ms = 220.0;
// One tick is one rendered frame. 16 ms is 62.5 fps.
constexpr double kFrameDt_s    = 0.016;
constexpr int    kDescentTicks = 90;
constexpr int    kClimbTicks   = 90;

// A 1080p, 60 degree vertical FOV camera: cone_spread = 2*tan(fovY/2)/h,
// the same expression PathTrace.slang evaluates per pixel.
constexpr double kConeSpread = 2.0 * 0.5773502691896257 / 1080.0;

// Small enough to keep a software-backend BLAS build affordable, deep enough
// that the selector has real levels to move between. |desired| stays well
// under the budget at these settings, which is checked below -- a run that
// sat on the arena cap would be testing the pressure fallback instead of the
// policy.
constexpr int kMaxLevel    = 8;
constexpr int kChunkBudget = 1024;
// The real interactive default of r_planet_blas_budget_ms. Not zero: zero
// disables pacing, and an unpaced stream cannot exhibit the bug.
constexpr double kBlasBudget_ms = 2.0;

// No published chunk may be an ancestor of another -- for quadtree nodes
// that is exactly "no two published chunks overlap".
bool IsAntichain(const std::set<ChunkKey>& s) {
    for (const ChunkKey& k : s) {
        ChunkKey a = k;
        while (a.level > 0) {
            a = a.Parent();
            if (s.find(a) != s.end()) return false;
        }
    }
    return true;
}

struct TickRecord {
    std::size_t desired      = 0;
    std::size_t resident     = 0;
    std::size_t published    = 0;
    std::size_t held         = 0;
    std::size_t starved      = 0;
    // The whole cut and how much of it is in the arena (#319).
    std::size_t retained     = 0;
    std::size_t cut          = 0;
    std::size_t cut_resident = 0;
};

// Everything one flight produced, so the assertions and the non-vacuity
// ledger can be read off one run.
struct FlightResult {
    bool ran = false;
    std::string skip_reason;
    std::vector<TickRecord> ticks;
    // Ticks on which ground drawn last tick stopped being drawn with no
    // policy reason for it. THE BUG, and the number that must be zero.
    int  coverage_regressions = 0;
    // Ticks on which coverage receded because the policy DELIBERATELY chose
    // a hole: a substitution refused to avoid a two-level step (a crack the
    // index arena has no variant for), or a stand-in released because the
    // arena had no slot for an incoming chunk. Both are the documented
    // "correctness beats smoothness" fallback, so they are counted and
    // reported rather than asserted to zero -- but they must be RARE, which
    // is asserted.
    int  regressions_under_fallback = 0;
    int  overlap_failures     = 0;
    int  imbalance_failures   = 0;
    int  orphan_failures      = 0;   // published but not resident
    std::size_t resident_peak = 0;
    // PlanetTerrain::ArenaSlots() for the run: WholeCutSlots(leaf budget),
    // which is what resident_peak has to fit inside.
    std::size_t arena_slots   = 0;
    std::uint64_t holds_dropped = 0;
    std::uint64_t holds_refused = 0;
    std::uint64_t retained_dropped = 0;
    // Worst 2:1 repair depth any tick needed. Reaching
    // pt::planet::kMaxRepairRounds means the backstop fired and the cover
    // collapsed to the resident subset of the desired set.
    int         repair_peak   = 0;
    std::size_t max_desired   = 0;
    int  ticks_with_holds     = 0;
    int  ticks_incomplete     = 0;
    int  distinct_desired_digests = 0;
    std::size_t final_held = 0;
    std::size_t final_published = 0;
    std::size_t settled_resident = 0;
    std::size_t settled_desired  = 0;
    std::size_t final_resident = 0;
    std::size_t final_desired  = 0;
    // Did the last tick's cover span every leaf the selector wanted? The
    // no-deadlock statement: after the jump the whole planet is being drawn
    // again, whatever level it has caught up to.
    bool        final_cover_complete = false;
    // Ticks whose cover spanned the whole desired set.
    int         ticks_complete = 0;
    // --- Recovery phase (settle_first + teleport plans only) -------------
    // Did the streamer catch the selector -- resident == published ==
    // desired, nothing held?
    bool        converged        = false;
    int         recovery_ticks   = 0;
    // How the recovery loop ended. Exactly one of these is true when the
    // run did not converge, and which one is the diagnosis: quiescence
    // means "no work left and still behind", i.e. WEDGED; the cap means
    // "still working when we ran out of patience", i.e. slow.
    bool        ended_on_cap     = false;
    bool        ended_quiescent  = false;
    // --- Warm-up phase (warm_until_quiescent plans) ----------------------
    // What the arena held when the streamer stopped having anything left to
    // do, with the camera frozen at the bottom of the descent. With
    // ancestor retention this is the WHOLE CUT; without it, exactly the
    // leaf frontier -- which is the difference the zoom-out then exposes.
    int         warm_ticks       = 0;
    bool        warm_quiescent   = false;
    std::size_t warm_cut         = 0;
    std::size_t warm_cut_resident = 0;
    std::size_t warm_desired     = 0;
    std::size_t warm_resident    = 0;
    // --- Zoom-out phase --------------------------------------------------
    // Regressions counted only over the climb, so the warm-up's own
    // transient cannot mask or manufacture them.
    int         climb_regressions = 0;
    int         climb_regressions_under_fallback = 0;
    int         climb_ticks_run   = 0;
    std::uint64_t climb_refused   = 0;
    std::uint64_t climb_dropped   = 0;
    std::uint64_t climb_retained_dropped = 0;
    std::size_t first_climb_desired = 0;
    std::size_t last_climb_desired  = 0;
    // Climb ticks on which the ground that was being DRAWN when the zoom
    // started is not fully covered any more. This, and not the tick-to-tick
    // delta, is the number the user is reporting: a hole opens once and
    // then sits there for as long as the coarse chunk takes to bake and
    // build, so a difference-of-consecutive-ticks counter scores the whole
    // outage as a single event. Measured against the pre-zoom published
    // cover, which is a fixed set, so this counts DURATION.
    int         climb_ticks_uncovered = 0;
    // Worst case: the largest number of consecutive uncovered climb ticks.
    int         climb_uncovered_run   = 0;
    bool        climb_quiescent       = false;
};

struct FlightPlan {
    int    chunk_budget = kChunkBudget;
    int    max_level    = kMaxLevel;
    // Ticks of smooth descent, then of smooth climb. A teleport is inserted
    // between the two halves when `teleport` is set: an instantaneous jump
    // to the far side of the planet, which is the worst case the streamer
    // can be handed -- every resident chunk becomes unwanted at once and
    // nothing that replaces it is baked yet.
    int    descent_ticks = kDescentTicks;
    int    climb_ticks   = kClimbTicks;
    bool   teleport      = false;
    // r_planet_blas_budget_ms for the run. The shipping interactive default
    // is 2 ms; a pressure run raises it so the arena rather than the pacing
    // is what the streamer runs into.
    double blas_budget_ms = kBlasBudget_ms;
    // Milliseconds of real time between ticks, emulating a rendered frame.
    // Only sets the rate at which bakes land; the residency ANSWER is a pure
    // function of (camera, metrics, params).
    int    frame_sleep_ms = 4;
    // Fill the arena with a Settle() barrier before the flight starts,
    // instead of letting the paced stream fill it. Only for the arena
    // pressure case, and the reason is machine independence: how full the
    // arena gets under pacing is a question about how fast THIS HOST bakes
    // and builds, and a Debug CI runner gets a third of the way there --
    // which would leave the pressure case green and vacuous. Settle() is a
    // barrier, so it converges to the same residency on every host. The
    // claim under test is about what happens AFTER, on the paced path.
    bool   settle_first  = false;
    double tau_px        = 1.5;
    // Run the post-teleport phase until the streamer has CAUGHT UP rather
    // than for a fixed number of ticks, and stop early if it demonstrably
    // stops working. See the loop for why the stopping rule is work
    // remaining and not ticks elapsed.
    bool   recover_until_converged = false;
    // Runaway guard, not a schedule. Only reached by a host that is still
    // making progress and is slower than anything measured.
    int    recovery_cap    = 4000;
    // Consecutive ticks with no bake queued, none in flight and no BLAS
    // built. That is "there is no work left to do", and if the streamer is
    // still behind at that point it is stuck rather than slow. 60 ticks is
    // ~15x the longest gap measured between two builds on a Debug host.
    int    quiescent_limit = 60;
    // Hold the camera still after the settle and tick the PACED path until
    // the streamer has nothing left to do -- no bake queued, none in
    // flight, no BLAS built for `quiescent_limit` consecutive ticks. That
    // is the same clock-free stopping rule the recovery loop uses, and it
    // is used rather than "until the whole cut is resident" on purpose: the
    // whole-cut condition is the FIX, so a build without the fix would stop
    // in the warm-up instead of reaching the climb that is supposed to
    // expose it. Quiescence is a statement both builds can satisfy, each
    // arriving at its own best arena.
    bool   warm_until_quiescent = false;
    int    warm_cap             = 4000;
    // Climb from kAltBottom_m back to kAltStart_m after the warm-up,
    // asserting coverage on every tick. This is #319's case.
    bool   zoom_out             = false;
    // Altitude the zoom-out ends at, in metres. Defaults to the flight's
    // own ceiling; the #319 case goes higher, to low orbit, because how
    // many LEVELS the frontier merges is log2 of the altitude ratio and
    // that is what decides whether the fine leftovers underneath a merged
    // leaf are one level down (which the index arena can stitch) or four
    // (which it cannot, so the substitution is refused and the ground goes
    // dark).
    double zoom_top_m           = kAltStart_m;
};

FlightResult FlyThePacedPath(pt::rhi::BackendType backend,
                             const FlightPlan& plan) {
    FlightResult fr;

    pt::rhi::NativeWindowHandle window{};
    window.opaque = nullptr;
    window.width  = 64;
    window.height = 64;
    auto dev = pt::rhi::Device::Create(backend, window);
    if (!dev) {
        fr.skip_reason = "backend not built into this binary, or device "
                         "creation failed on this host";
        return fr;
    }
    if (!dev->SupportsHardwareRT()) {
        fr.skip_reason = "device reports no hardware ray tracing, so "
                         "PlanetTerrain::Init refuses to come up";
        return fr;
    }

    pt::engine::TerrainConfig cfg{};
    cfg.enabled        = true;
    // No DEM path: a purely procedural body. The elevation field is a pure
    // function of the key either way, which is all the residency policy
    // cares about, and it keeps the test free of a 400 MB asset download.
    cfg.dem_path.clear();
    cfg.site_lat_rad   = 27.9881 * 3.14159265358979323846 / 180.0;   // Everest
    cfg.site_lon_rad   = 86.9250 * 3.14159265358979323846 / 180.0;
    cfg.worker_count   = 4;
    cfg.blas_budget_ms = plan.blas_budget_ms;
    cfg.lod.tau_px       = plan.tau_px;
    cfg.lod.hysteresis   = 1.4;
    cfg.lod.min_level    = 0;
    cfg.lod.max_level    = plan.max_level;
    cfg.lod.chunk_budget = plan.chunk_budget;
    cfg.lod.cone_spread  = kConeSpread;
    cfg.lod.freeze       = false;

    pt::engine::PlanetTerrain terrain;
    if (!terrain.Init(dev.get(), cfg)) {
        fr.skip_reason = "PlanetTerrain::Init failed on this device";
        return fr;
    }

    // Camera positions are GEODETIC and converted through the site's own
    // frame, not offsets in the local tangent plane. The tangent-plane
    // shortcut is fine for the few kilometres the flight covers and wrong
    // by the radius of the Earth for the teleport: a straight 10 000 km
    // along world +X leaves the camera 7 800 km ABOVE the surface rather
    // than a quarter of the way round it, which turns the intended
    // same-altitude jump into a retreat to deep space and collapses the
    // demand the case exists to create. Measured before the fix: the
    // "teleport" converged on a 12-chunk desired set.
    const pt::planet::PlanetSite& site = terrain.Site();
    auto camera_at = [&site](double lat_rad, double lon_rad, double alt_m) {
        const glm::dvec3 surf = pt::planet::GeodeticToEcef(lat_rad, lon_rad);
        const glm::dvec3 n    = pt::planet::GeodeticNormal(surf);
        return site.EcefToWorld(surf + alt_m * n);
    };
    std::set<ChunkKey> prev_published;
    std::set<std::uint64_t> digests;
    double lon_rad = cfg.site_lon_rad;
    const double lat_rad = cfg.site_lat_rad;
    // Ground speed converted to a longitude rate at this latitude.
    const double lon_rate =
        kGroundSpeed_ms * kFrameDt_s /
        (pt::planet::kIuggMeanRadius * std::max(std::cos(lat_rad), 1e-3));
    std::uint64_t prev_refused = 0;
    std::uint64_t prev_dropped = 0;
    std::uint64_t prev_retained_dropped = 0;
    // Set while the climb runs, so its regressions can be counted on their
    // own -- the warm-up before it has a transient of its own and mixing
    // the two would let either hide in the other.
    bool in_climb = false;
    // The published cover the instant the zoom-out began. Ground drawn then
    // must still be drawn at every tick of the climb -- the selector never
    // stops wanting the sphere covered, it only wants it coarser.
    std::set<ChunkKey> pre_climb_published;
    int uncovered_run = 0;

    // Zeroed for the recovery phase; see the loop.
    double lon_step = lon_rate;
    auto tick = [&](double altitude_m) {
        pt::planet::LodParams lod = cfg.lod;
        lod.camera_w = camera_at(lat_rad, lon_rad, altitude_m);
        lon_rad += lon_step;

        terrain.Update(lod, glm::dvec3(0.0));

        const std::set<ChunkKey>& pub = terrain.Published();
        const auto& st = terrain.Stats();

        // 1. Coverage never regresses -- unless the policy deliberately
        //    chose a hole this very tick, which is the only licence it has.
        const bool fallback_fired = (st.holds_refused != prev_refused) ||
                                    (st.holds_dropped != prev_dropped) ||
                                    (st.retained_dropped != prev_retained_dropped);
        if (in_climb) {
            fr.climb_refused += st.holds_refused - prev_refused;
            fr.climb_dropped += st.holds_dropped - prev_dropped;
            fr.climb_retained_dropped +=
                st.retained_dropped - prev_retained_dropped;
        }
        prev_refused = st.holds_refused;
        prev_dropped = st.holds_dropped;
        prev_retained_dropped = st.retained_dropped;
        if (!prev_published.empty() && !CoversAll(pub, prev_published)) {
            if (fallback_fired) {
                ++fr.regressions_under_fallback;
                if (in_climb) ++fr.climb_regressions_under_fallback;
            } else {
                ++fr.coverage_regressions;
                if (in_climb) ++fr.climb_regressions;
            }
        }
        // 2. The cover does not overlap itself.
        if (!IsAntichain(pub)) ++fr.overlap_failures;
        // 3. One-level steps only.
        if (!IsEdgeBalanced(pub)) ++fr.imbalance_failures;
        // 4. Published implies resident, and the arena is never overrun.
        const auto res = terrain.ResidentKeys();
        const std::set<ChunkKey> res_set(res.begin(), res.end());
        for (const ChunkKey& k : pub) {
            if (res_set.find(k) == res_set.end()) { ++fr.orphan_failures; break; }
        }

        TickRecord rec;
        rec.desired      = st.desired;
        rec.resident     = st.resident;
        rec.published    = st.published;
        rec.held         = st.held;
        rec.starved      = st.starved;
        rec.retained     = st.retained;
        rec.cut          = st.cut;
        rec.cut_resident = st.cut_resident;
        fr.ticks.push_back(rec);
        fr.repair_peak = std::max(fr.repair_peak, st.repair_rounds);
        if (in_climb && !pre_climb_published.empty()) {
            if (!CoversAll(pub, pre_climb_published)) {
                ++fr.climb_ticks_uncovered;
                ++uncovered_run;
                fr.climb_uncovered_run =
                    std::max(fr.climb_uncovered_run, uncovered_run);
            } else {
                uncovered_run = 0;
            }
        }
        if (CoversAll(pub, terrain.Desired())) ++fr.ticks_complete;
        if (st.held > 0) ++fr.ticks_with_holds;
        if (st.published < st.desired) ++fr.ticks_incomplete;
        fr.max_desired = std::max(fr.max_desired, st.desired);
        digests.insert(st.desired_digest);

        prev_published = pub;

        // Emulate a rendered frame. The residency ANSWER is a pure function
        // of (camera, metrics, params) -- see the determinism notes in
        // TerrainQuadtree.cpp -- so this only sets the rate at which bakes
        // land, which is the very thing the paced path has to survive.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(plan.frame_sleep_ms));
    };

    if (plan.settle_first) {
        pt::planet::LodParams lod = cfg.lod;
        lod.camera_w = camera_at(lat_rad, lon_rad, kAltBottom_m);
        terrain.Settle(lod, glm::dvec3(0.0), 64);
        fr.settled_resident = terrain.Stats().resident;
        fr.settled_desired  = terrain.Stats().desired;
    }
    for (int i = 0; i < plan.descent_ticks; ++i) {
        const double t = (plan.descent_ticks > 1)
                             ? static_cast<double>(i) / (plan.descent_ticks - 1)
                             : 0.0;
        tick(kAltStart_m + (kAltBottom_m - kAltStart_m) * t);
    }
    if (plan.warm_until_quiescent) {
        // --- LET THE STREAMER CATCH UP, WITHOUT A CLOCK ------------------
        //
        // The camera is frozen and the paced path is ticked until there is
        // nothing left to do: no bake queued, none in flight, and no BLAS
        // built for `quiescent_limit` consecutive ticks. Same stopping rule
        // as the recovery loop, and for the same reason -- how many ticks a
        // host needs is a property of the host, and a tick budget was
        // already the thing that failed on the Windows runner once.
        //
        // Deliberately NOT "until the whole cut is resident". That is the
        // fix's own statement, so a build without the fix would stop here
        // rather than reach the climb the fix exists for. Quiescence is
        // something both builds reach; what differs is the arena they reach
        // it with, which is exactly what the ledger below records.
        lon_step = 0.0;
        std::uint64_t last_builds = terrain.Stats().blas_builds;
        int quiescent = 0;
        for (int i = 0; i < plan.warm_cap; ++i) {
            tick(kAltBottom_m);
            fr.warm_ticks = i + 1;
            const auto& st = terrain.Stats();
            const bool working =
                st.pending_bakes > 0 || st.blas_builds != last_builds;
            last_builds = st.blas_builds;
            quiescent = working ? 0 : (quiescent + 1);
            if (quiescent >= plan.quiescent_limit) {
                fr.warm_quiescent = true;
                break;
            }
        }
        const auto& st = terrain.Stats();
        fr.warm_cut          = st.cut;
        fr.warm_cut_resident = st.cut_resident;
        fr.warm_desired      = st.desired;
        fr.warm_resident     = st.resident;
    }
    if (plan.zoom_out) {
        // --- THE ZOOM-OUT (#319) -----------------------------------------
        //
        // Straight back out to the altitude the descent started from, which
        // merges the frontier several levels under the camera. Every tick
        // asserts the same coverage invariant as everywhere else in this
        // file; the difference is that the arena underneath it has already
        // caught up, so a chunk that is missing here is missing because it
        // was THROWN AWAY on the way down and not because it has not been
        // baked yet.
        //
        // The camera does not translate during the climb. A lateral
        // component would sweep the LOD boundary as well as raise it and
        // mix split transitions into a case that is about merges.
        lon_step = 0.0;
        pre_climb_published = terrain.Published();
        in_climb = true;
        fr.first_climb_desired = terrain.Stats().desired;
        for (int i = 0; i < plan.climb_ticks; ++i) {
            const double t = (plan.climb_ticks > 1)
                                 ? static_cast<double>(i) / (plan.climb_ticks - 1)
                                 : 1.0;
            // EXPONENTIAL in altitude, not linear. The engine scales free
            // camera speed with height above the terrain (see
            // PlanetTerrain::AltitudeAboveTerrain's callers), so a held
            // control produces a constant number of e-foldings per second
            // -- and it is e-foldings that the LOD frontier tracks, the
            // split distance being proportional to the chunk's own error.
            // A linear ramp would spend almost all of its ticks up where
            // nothing merges any more.
            tick(kAltBottom_m *
                 std::pow(plan.zoom_top_m / kAltBottom_m, t));
            fr.climb_ticks_run = i + 1;
        }
        // --- AND THEN HOLD, STILL WATCHING ------------------------------
        //
        // The zoom itself is over in a fraction of a second; the OUTAGE is
        // not. A coarse chunk that was evicted on the way down has to be
        // baked (~2 ms) and built against r_planet_blas_budget_ms, so the
        // ground it covers stays dark for as long as that queue takes --
        // which for a cut of several hundred chunks is hundreds of frames.
        // Stopping at the top of the climb would score that whole outage as
        // however many ticks the ascent happened to take.
        //
        // Bounded by WORK and not by ticks, the same clock-free rule the
        // other phases use: no bake queued, none in flight, no BLAS built
        // for `quiescent_limit` consecutive ticks.
        std::uint64_t last_builds = terrain.Stats().blas_builds;
        int quiescent = 0;
        for (int i = 0; i < plan.recovery_cap; ++i) {
            tick(plan.zoom_top_m);
            fr.climb_ticks_run = plan.climb_ticks + i + 1;
            const auto& st = terrain.Stats();
            const bool working =
                st.pending_bakes > 0 || st.blas_builds != last_builds;
            last_builds = st.blas_builds;
            quiescent = working ? 0 : (quiescent + 1);
            if (quiescent >= plan.quiescent_limit) { fr.climb_quiescent = true; break; }
        }
        in_climb = false;
        fr.last_climb_desired = terrain.Stats().desired;
    }
    if (plan.teleport) {
        // A quarter of the way round the planet in one frame, at the SAME
        // altitude. Same demand, none of it resident: every chunk in the
        // arena becomes a stand-in for ground nobody is looking at, and the
        // incoming set has 128 slots' worth of nowhere to go.
        lon_rad += 3.14159265358979323846 * 0.5;
    }
    if (plan.recover_until_converged) {
        // --- RECOVERY, BOUNDED BY WORK AND NOT BY TIME -------------------
        //
        // The paced path consults r_planet_blas_budget_ms, so how much
        // lands per tick is a property of the host. A fixed tick count
        // therefore asserts "this machine is fast enough", which is not the
        // property under test -- and it is not a threshold that can be
        // tuned safe, because the race just gets a longer fuse. Measured:
        // 150 ticks is ample on an M4 Max Release and leaves a Windows
        // Debug CI runner 51 chunks of 210 into the same recovery.
        //
        // So the loop runs until the streamer has CAUGHT UP, and the
        // failure it is really looking for -- a stand-in wedged onto the
        // slot its own replacement needs -- has a signature that does not
        // involve a clock at all:
        //
        //     no bake queued, none in flight, no BLAS built for
        //     `quiescent_limit` consecutive ticks, and still behind.
        //
        // That is "there is nothing left to do and we are not done", which
        // is what stuck means. A merely slow host always has work
        // outstanding, so it never trips it; it converges, later. Measured
        // on a Debug build with this exact 224-slot arena: converged at
        // tick ~200 (resident == published == desired == 219, pending 0,
        // held 0) and stayed there for the following 1 300 ticks.
        //
        // The camera is frozen for the recovery. The teleport is the
        // disturbance; continuing to fly would keep the target moving and
        // turn "did it catch up" into a question about the ratio of two
        // rates rather than a fixed point.
        lon_step = 0.0;
        std::uint64_t last_builds = terrain.Stats().blas_builds;
        int quiescent = 0;
        for (int i = 0; i < plan.recovery_cap; ++i) {
            tick(kAltBottom_m);
            fr.recovery_ticks = i + 1;
            const auto& st = terrain.Stats();
            // CAUGHT UP means the published cover is the selector's own
            // answer with nothing standing in for anything, and the arena
            // holds the whole cut behind it. `resident == desired` was the
            // pre-#319 form and is now unreachable by construction:
            // residency is the cut, which exceeds the frontier by exactly
            // its interior nodes.
            if (st.held == 0 && st.published == st.desired &&
                st.cut_resident == st.cut) {
                fr.converged = true;
                break;
            }
            const bool working =
                st.pending_bakes > 0 || st.blas_builds != last_builds;
            last_builds = st.blas_builds;
            quiescent = working ? 0 : (quiescent + 1);
            if (quiescent >= plan.quiescent_limit) {
                fr.ended_quiescent = true;
                break;
            }
            if (i + 1 == plan.recovery_cap) fr.ended_on_cap = true;
        }
    } else if (!plan.zoom_out) {
        for (int i = 0; i < plan.climb_ticks; ++i) {
            const double t = static_cast<double>(i) / (plan.climb_ticks - 1);
            tick(kAltBottom_m + (kAltStart_m - kAltBottom_m) * t);
        }
    }

    fr.arena_slots   = terrain.ArenaSlots();
    fr.resident_peak = terrain.Stats().resident_peak;
    fr.final_held      = terrain.Stats().held;
    fr.final_published = terrain.Stats().published;
    fr.final_resident  = terrain.Stats().resident;
    fr.final_desired   = terrain.Stats().desired;
    fr.final_cover_complete = CoversAll(terrain.Published(), terrain.Desired());
    fr.holds_dropped     = terrain.Stats().holds_dropped;
    fr.holds_refused     = terrain.Stats().holds_refused;
    fr.retained_dropped  = terrain.Stats().retained_dropped;
    fr.repair_peak = std::max(fr.repair_peak,
                              terrain.Stats().repair_rounds_peak);
    fr.distinct_desired_digests = static_cast<int>(digests.size());
    fr.ran = true;
    terrain.Shutdown();
    return fr;
}

FlightResult g_flight;

}  // namespace

TEST_CASE("paced residency: coverage never regresses while chunks stream") {
    g_flight = FlyThePacedPath(pt::rhi::BackendType::Software, FlightPlan{});
    if (!g_flight.ran) {
        MESSAGE("SKIPPED: " << g_flight.skip_reason);
        return;
    }

    // --- Non-vacuity first ------------------------------------------------
    // Every assertion below is trivially satisfied by a stream in which
    // nothing ever changes, so the run has to prove it exercised the thing.
    CHECK(g_flight.ticks.size() ==
          static_cast<std::size_t>(kDescentTicks + kClimbTicks));
    // The camera actually moved the LOD boundary: the selector produced
    // more than one distinct desired set over the flight.
    CHECK(g_flight.distinct_desired_digests > 4);
    // Chunks were actually held past the selector dropping them -- i.e. the
    // exact transient the old code holed out on occurred, repeatedly. If
    // this is zero the four invariants below are true of a stream that never
    // had the opportunity to break them, which is worth nothing.
    CHECK(g_flight.ticks_with_holds > 20);
    // And the stream really was mid-flight rather than settled.
    CHECK(g_flight.ticks_incomplete > 20);
    // No stand-in was dropped for want of an arena slot, so what is being
    // tested here is the policy and not the pressure fallback. (The
    // fallback has its own case in pt_planet_terrain, where the arena can
    // be squeezed without a GPU.)
    CHECK(g_flight.holds_dropped == 0u);

    // --- The invariants ---------------------------------------------------
    CHECK(g_flight.coverage_regressions == 0);
    CHECK(g_flight.overlap_failures     == 0);
    CHECK(g_flight.imbalance_failures   == 0);
    CHECK(g_flight.orphan_failures      == 0);

    // The deliberate holes -- crack avoidance and arena pressure -- are
    // allowed but must stay a rounding error against the flight. Measured
    // 1 of 180 ticks on an M4 Max at these parameters; 5% is ~9 ticks,
    // nine times the measurement, which is margin for a slower host
    // admitting chunks later and therefore substituting for longer. A
    // figure that walks up to this bound is a real regression in the policy,
    // not a flaky threshold.
    CHECK(g_flight.regressions_under_fallback <=
          (kDescentTicks + kClimbTicks) / 20);

    // Misclassification cannot hide anything either: the SUM of the two
    // regression counters is bounded by the same figure. Since #319 the
    // walk has coarse candidates it did not have before -- the retained
    // interior nodes -- and declining one of those is recorded as a refused
    // substitution even when it costs no coverage, because the finer cover
    // that was already there stays. Measured over five Release runs: 30-212
    // refusals against 1 before retention, with the regression count
    // unchanged at 0 real and 2-3 deliberate against a bound of 9 -- three
    // times the measurement. That makes `fallback_fired` true on most
    // ticks, so bounding
    // the sum is what keeps the classification from quietly absorbing a
    // real regression.
    CHECK(g_flight.coverage_regressions +
          g_flight.regressions_under_fallback <=
          (kDescentTicks + kClimbTicks) / 20);

    // The arena is a hard cap and the transient must live inside it. The
    // cap is the ARENA and not r_planet_chunk_budget: since #319 the cvar
    // is the leaf budget and the arena is WholeCutSlots of it, because the
    // whole cut is what stays resident.
    CHECK(g_flight.resident_peak <= g_flight.arena_slots);
    CHECK(g_flight.arena_slots ==
          pt::planet::WholeCutSlots(static_cast<std::size_t>(kChunkBudget)));
    // The 2:1 repair never reached its backstop. That matters because the
    // backstop is not a slower repair, it is a different answer: it refuses
    // EVERY substitution at once, so the published set collapses to the
    // resident subset of the desired set and the whole transient holes out.
    CHECK(g_flight.repair_peak < pt::planet::kMaxRepairRounds);
}

// --- #319: ZOOMING OUT FROM A CAUGHT-UP ARENA ----------------------------
TEST_CASE("paced residency: zooming out never uncovers ground") {
    // THE BUG. "Lots of holes in the terrain, tiles are loading whenever I
    // am just zooming out."
    //
    // Coverage-aware retirement (#308) holds a chunk until something else
    // covers its ground, which protects everything that is RESIDENT. The
    // selector's answer is the leaf frontier, though, so every ancestor of
    // a visible leaf was outside it and step 3a evicted the lot. Descending
    // was safe -- the parent is resident when its children are asked for.
    // Ascending was not: the coarse chunk a merge asks for was thrown away
    // several levels ago, and what is resident in its place is the fine
    // tiling underneath, which is more than one level below whatever is
    // published across the merge boundary. The 2:1 repair refuses that, so
    // the ground is genuinely uncovered until the coarse chunk has been
    // baked (~2 ms) and built against r_planet_blas_budget_ms -- hundreds
    // of frames for a coarse cut of several hundred chunks.
    //
    // WHY THE DESCENT-AND-CLIMB FLIGHT ABOVE CANNOT SEE IT. That flight
    // never catches up: it peaks at ~630 desired against ~270 resident, so
    // on the way back out the coarse chunks are still resident -- they were
    // never replaced, so nothing ever retired them. The bug needs an arena
    // that has caught up, which is what this case builds first:
    //
    //   settle at 4 km              -- machine-independent, and it leaves
    //                                  residency EXACTLY the leaf frontier
    //                                  (Settle retires everything outside
    //                                  it by design), which is the state a
    //                                  long descent ends in;
    //   tick the paced path until   -- no bake queued, none in flight, no
    //   the streamer is quiescent      BLAS built for 60 ticks. Clock-free,
    //                                  and reached by a build with the fix
    //                                  and by one without: the difference
    //                                  is WHAT is in the arena when it is
    //                                  reached, which the ledger records;
    //   climb 4 km -> 60 km         -- asserting coverage every tick.
    //
    // Settle() is deliberately not used for the climb itself. It runs
    // unpaced and iterates to a fixed point before anything is observed, so
    // it converges before a transient can exist -- it is structurally
    // incapable of seeing this, which is this project's recurring
    // vacuous-check shape.
    //
    // RED THEN GREEN, on this exact case. With `retained_` forced empty --
    // the one-line revert of the retention policy, everything else
    // unchanged -- two consecutive Release runs:
    //
    //                              reverted        with retention
    //   arena after warm-up        699 + 0 anc     699 + 231 anc
    //   ground uncovered           20 of 90 ticks  0 of 81
    //   longest uncovered run      20 ticks        0
    //   regressions (deliberate)   12-13           0
    //   substitutions refused      27-29           0
    //
    // Twenty consecutive ticks is the shape of the report: the hole opens
    // once and stays open until the coarse chunks have been baked and
    // built, which is why the counter below measures DURATION against the
    // pre-zoom cover rather than differencing consecutive ticks -- the
    // tick-to-tick counter scores that entire outage as 1.
    FlightPlan plan;
    // tau 6 px keeps the SELECTOR off its own budget cap, so the run
    // measures the policy and not EnforceBudget's tau bisection -- same
    // reasoning as the teleport case below.
    //
    // LEVEL 14 IS LOAD-BEARING and was measured, not guessed. What decides
    // whether a merge opens a hole is how many LEVELS the frontier moves,
    // because the fine leftovers underneath a merged leaf have to be within
    // one level of whatever is published across its boundary for the index
    // arena to stitch them. With a ceiling of 10 the frontier under the
    // camera is pinned AT the ceiling at both ends of the climb -- altitude
    // only changes the area of the fine cap, not its depth -- so the merge
    // is one level at a time and #308's fine-cover-from-below handles it
    // unaided. Measured on the reverted build: ceiling 10 gives 306 -> 252
    // leaves and zero uncovered ticks, ceiling 12 gives 501 -> 252 and 4,
    // ceiling 14 gives 699 -> 252 and 9-12. Fourteen is where the frontier
    // becomes error-limited rather than ceiling-limited, which is the
    // regime a real session runs in (the shipping ceiling is 19).
    plan.tau_px         = 6.0;
    plan.max_level      = 14;
    // 1024 leaves is far more arena than 699 leaves and their 231 ancestors
    // need, on purpose: this case is about retention, not about the
    // pressure fallback, and the assertions below require that no fallback
    // fires at all.
    plan.chunk_budget   = 1024;
    plan.settle_first   = true;
    plan.descent_ticks  = 0;
    plan.warm_until_quiescent = true;
    plan.zoom_out       = true;
    // TWENTY FRAMES from 4 km to 60 km, and the rate matters as much as the
    // range. Measured on the reverted build: the outage appears at 30
    // frames or fewer and is gone by 45, because a slow enough climb lets
    // the coarse chunks bake and build ahead of the frontier -- which is
    // the easy half of the problem and not the reported one. Twenty is 1.5x
    // inside the boundary. As a rate it is 2.7 nepers over 20 frames, i.e.
    // 8.5 e-foldings per second at 62.5 fps: a mouse-wheel zoom, which is
    // what "just zooming out" describes.
    plan.climb_ticks    = 20;
    // The shipping interactive default, not the 8 ms the teleport case
    // raises it to. The outage this exists to catch is measured in frames
    // of BLAS budget, so the budget has to be the one a session runs at.
    plan.blas_budget_ms = 2.0;
    plan.frame_sleep_ms = 4;
    const FlightResult f = FlyThePacedPath(pt::rhi::BackendType::Software, plan);
    if (!f.ran) {
        MESSAGE("SKIPPED: " << f.skip_reason);
        return;
    }

    std::printf("\n--- zoom-out from a caught-up arena (#319) ---\n");
    std::printf("  settled                  %zu resident of %zu desired\n",
                f.settled_resident, f.settled_desired);
    std::printf("  warm-up                  %d ticks%s\n", f.warm_ticks,
                f.warm_quiescent ? " (quiescent)" : " (hit the cap)");
    std::printf("  arena after warm-up      %zu of %zu whole-cut chunks "
                "(%zu leaves + %zu ancestors)\n",
                f.warm_cut_resident, f.warm_cut, f.warm_desired,
                f.warm_cut - f.warm_desired);
    std::printf("  climb + hold             %d ticks%s, desired %zu -> %zu\n",
                f.climb_ticks_run, f.climb_quiescent ? " (quiescent)" : "",
                f.first_climb_desired, f.last_climb_desired);
    std::printf("  GROUND UNCOVERED         %d of %d climb ticks "
                "(longest run %d)\n",
                f.climb_ticks_uncovered, f.climb_ticks_run,
                f.climb_uncovered_run);
    std::printf("  CLIMB REGRESSIONS        %d   (must be 0)\n",
                f.climb_regressions);
    std::printf("  ... under fallback       %d\n",
                f.climb_regressions_under_fallback);
    std::printf("  climb refused / dropped  %llu / %llu (+%llu reserve)\n",
                static_cast<unsigned long long>(f.climb_refused),
                static_cast<unsigned long long>(f.climb_dropped),
                static_cast<unsigned long long>(f.climb_retained_dropped));
    std::printf("  resident HIGH WATER      %zu  (arena %zu)\n",
                f.resident_peak, f.arena_slots);
    std::printf("  repair rounds peak       %d  (backstop %d)\n",
                f.repair_peak, pt::planet::kMaxRepairRounds);

    // --- Non-vacuity, first ----------------------------------------------
    // The warm-up really did leave the arena holding the WHOLE cut and not
    // just its frontier. This is the fix stated directly, and it is what
    // makes the coverage assertion below mean something: without it the
    // climb would be asking a half-empty arena to cover ground.
    CHECK(f.warm_quiescent);
    CHECK(f.warm_cut > f.warm_desired);          // there ARE interior nodes
    CHECK(f.warm_cut_resident == f.warm_cut);    // and every one is resident
    // A cut with L leaves has exactly (L - 6) / 3 interior nodes -- the
    // identity the arena is sized from. If this ever fails, either the
    // selector stopped producing a proper quadtree cut or CutAncestors is
    // computing something else.
    CHECK(f.warm_cut - f.warm_desired == (f.warm_desired - 6) / 3);
    // The climb really did merge: the selector wanted materially fewer
    // leaves at 60 km than at 4 km, which is what makes ancestors necessary.
    CHECK(f.last_climb_desired < f.first_climb_desired);
    CHECK(f.climb_quiescent);
    CHECK(f.climb_ticks_run > plan.climb_ticks);
    CHECK(f.distinct_desired_digests > 4);

    // --- The claim -------------------------------------------------------
    // Ground drawn on one tick of the climb is still drawn on the next, on
    // EVERY tick. Not "mostly", and not "except under the fallback": with
    // the whole cut resident the merge publishes a chunk it already has, so
    // no substitution is needed and no fallback has anything to fire about.
    // The two fallback counters are asserted to zero for exactly that
    // reason -- if either moved, the classification above would start
    // absorbing real regressions into `under fallback`, which is how a
    // coverage test goes quietly vacuous.
    CHECK(f.climb_ticks_uncovered == 0);
    CHECK(f.climb_regressions == 0);
    CHECK(f.climb_regressions_under_fallback == 0);
    CHECK(f.climb_refused == 0u);
    CHECK(f.climb_dropped == 0u);
    CHECK(f.climb_retained_dropped == 0u);

    // And the rest of #308's guarantees still hold through the climb.
    CHECK(f.overlap_failures   == 0);
    CHECK(f.imbalance_failures == 0);
    CHECK(f.orphan_failures    == 0);
    CHECK(f.resident_peak <= f.arena_slots);
    CHECK(f.repair_peak < pt::planet::kMaxRepairRounds);
}

TEST_CASE("paced residency: a teleport into a tight arena never overruns it") {
    // The pressure case, and the one where the policy is REQUIRED to give
    // up. A 224-chunk arena against a camera that jumps a quarter of the way
    // round the planet in one frame: every resident chunk becomes unwanted
    // at once, nothing that replaces it is baked, and the stand-ins the
    // policy would like to hold are sitting on the very slots the incoming
    // set needs.
    //
    // What must survive that is not smoothness. It is:
    //   * the arena is never overrun -- residency stays inside the cap that
    //     sized the vertex buffer and the TLAS;
    //   * the cover never overlaps and never opens a two-level step, even
    //     while it is being torn down and rebuilt;
    //   * and the streamer does not deadlock. The failure mode this shape
    //     of policy invites is self-sustaining -- a held chunk occupying the
    //     slot its own replacement needs -- so the run has to come out the
    //     far side with nothing stuck.
    FlightPlan plan;
    // --- WHY THESE NUMBERS MOVED AT #319 ---------------------------------
    //
    // This case used to be tau 6 px, ceiling 8, and a 224-chunk budget: 189
    // leaves before the jump against 213 after, which could not both fit,
    // so the arena filled and the release fired. Retaining the whole cut
    // took the pressure out of exactly that configuration, and the reason
    // is the fix rather than an accident. The abandoned fine leaves can
    // only be retired once something covers their ground, and after the
    // jump that something is the coarse chunk over the old region -- which
    // is now an interior node of the new cut and therefore already
    // resident. The old set retires almost immediately instead of squatting
    // while its replacement bakes. Measured at the old parameters against
    // the new arena of WholeCutSlots(224) = 296: high water 278, zero
    // starved, zero released, converged in 46 ticks. A green run that
    // exercises none of the release path is worth nothing, so the arena is
    // squeezed harder rather than the assertions relaxed.
    //
    // Ceiling 12 and a 150-leaf budget: 501 leaves uncapped at 4 km, so the
    // selector IS on its cap here, and that is deliberate. The old comment
    // avoided the cap because EnforceBudget's bisection under-counts during
    // a descent (its trial sets treat an unmeasured node as a leaf) and
    // could collapse the set -- measured 126 leaves at one longitude and 12
    // at another with a 128-chunk cap at ceiling 8. That mattered when the
    // case asserted something about WHAT the selector chose. It asserts
    // nothing of the kind: the claims below are the arena is never
    // overrun, the cover never overlaps or breaks 2:1, the streamer does
    // not wedge, and the release path fires. Sitting on the cap is the only
    // way left to squeeze the arena, and the collapse does not happen here
    // -- 150 of 150 settled, 150 desired after the jump, checked below.
    //
    // Measured over four runs at these parameters: high water 198 against
    // an arena of exactly 198 on every one, peak starved 53-61, stand-ins
    // released 93 every run, reserve released 5-8. The arena is genuinely
    // full and both halves of the release fire -- the reserve first, being
    // insurance rather than pixels, and the stand-ins after.
    plan.tau_px        = 6.0;
    plan.chunk_budget  = 150;
    plan.max_level     = 12;
    plan.settle_first  = true;    // fill the arena on every host, not just fast ones
    plan.descent_ticks = 0;
    plan.teleport      = true;
    plan.recover_until_converged = true;
    // The shipping interactive default, where it used to be 8 ms. The
    // budget is what decides how much of the baked backlog can reach the
    // arena per tick, so it is also what decides whether "blocked demand"
    // ever exceeds the free list -- and blocked demand is what authorises a
    // release. At 8 ms the builds kept up with the bakes and the deficit
    // stayed at zero on some runs and not others: measured peak starved 0,
    // 2, 3 and 7 over four runs at otherwise identical parameters, which is
    // a coin-toss non-vacuity check. At 2 ms it is 53-61 over four runs
    // with 93 stand-ins and 5-8 reserve chunks released every time, and the
    // high water sits at exactly 198 of 198 slots on every run. Slower is
    // also more honest here: 2 ms is what a session actually runs at.
    plan.blas_budget_ms = 2.0;
    plan.frame_sleep_ms = 4;
    const FlightResult f = FlyThePacedPath(pt::rhi::BackendType::Software, plan);
    if (!f.ran) {
        MESSAGE("SKIPPED: " << f.skip_reason);
        return;
    }
    std::size_t peak_starved = 0, peak_held = 0;
    for (const auto& t : f.ticks) {
        peak_starved = std::max(peak_starved, t.starved);
        peak_held    = std::max(peak_held,    t.held);
    }

    // The hard cap, which is the whole reason the hold is released on
    // demand rather than kept on principle.
    CHECK(f.resident_peak <= f.arena_slots);
    // Still no coincident surfaces and still no crack, under pressure.
    CHECK(f.overlap_failures   == 0);
    CHECK(f.imbalance_failures == 0);
    CHECK(f.orphan_failures    == 0);
    // NO DEADLOCK, and this is the one claim in the file that is NOT
    // asserted per tick, because it cannot honestly be. The failure this
    // shape of policy invites is a stand-in squatting the slot its own
    // replacement needs, so the arena stays full of ground nobody is
    // looking at; ruling that out means watching the streamer finish, and
    // how long finishing takes is a property of the host.
    //
    // So the two claims are split deliberately, and the split is the point:
    //
    //   COVERAGE, OVERLAP, 2:1 AND THE ARENA are asserted STRICTLY, on
    //   every single tick, above. They are the fix, and they do not get to
    //   be conditional on anything.
    //
    //   CONVERGENCE is bounded by WORK REMAINING instead. The loop ran
    //   until resident == published == desired with nothing held, or until
    //   the streamer went quiescent -- no bake queued, none in flight, no
    //   BLAS built for 60 consecutive ticks -- while still behind. The
    //   second is what stuck looks like and it involves no clock.
    //
    // An earlier version asserted "published >= what was resident before
    // the jump" after a fixed 150 ticks, which is the same claim with a
    // stopwatch in it: green on an M4 Max, red on a Windows Debug runner
    // that was 51 chunks of 210 into a perfectly healthy recovery. The
    // per-tick invariants held on that run -- zero coverage regressions,
    // high water 201 inside a 224-slot arena -- which is exactly why the
    // stopping rule and not the invariant was the thing that was wrong.
    CHECK(f.converged);
    CHECK_FALSE(f.ended_quiescent);   // no work left and still behind = wedged
    CHECK_FALSE(f.ended_on_cap);      // ran out of patience, not out of work
    CHECK(f.final_held == 0u);
    // Non-vacuous, on any host. The barrier really did fill the arena
    // before the jump -- Settle() is machine-independent, which is the
    // whole reason it is used here -- and the jump really did turn that
    // arena into stand-ins and starve the incoming set, so the release path
    // below was exercised rather than merely present.
    CHECK(f.settled_resident >= 150u);         // the selector's own cap
    CHECK(f.settled_desired  == f.settled_resident);   // and no collapse
    CHECK(peak_held  >= 100u);                 // stand-ins, held at once
    CHECK(peak_starved > 0u);                  // demand really was blocked
    // SOMETHING was released to serve that demand. The sum rather than
    // either half: since #319 the reserve of retained ancestors goes first
    // -- it is insurance against a later zoom-out, while a stand-in is
    // ground on screen right now -- so which of the two fires depends on
    // how much of the reserve the arena happened to be holding when the
    // jump landed. Measured over four runs: reserve 5-8, stand-ins 93.
    CHECK(f.retained_dropped + f.holds_dropped > 0u);
    CHECK(f.repair_peak < pt::planet::kMaxRepairRounds);

    std::printf("\n--- teleport into a tight arena ---\n");
    std::printf("  settled residency        %zu of %zu desired\n",
                f.settled_resident, f.settled_desired);
    std::printf("  resident HIGH WATER      %zu  (arena %zu)\n",
                f.resident_peak, f.arena_slots);
    std::printf("  final published          %zu  (resident %zu, desired %zu)\n",
                f.final_published, f.final_resident, f.final_desired);
    std::printf("  peak starved / held      %zu / %zu\n", peak_starved, peak_held);
    std::printf("  final held               %zu   (settled %zu)\n",
                f.final_held, f.settled_resident);
    std::printf("  recovery                 %s after %d ticks%s\n",
                f.converged ? "CONVERGED" : "DID NOT CONVERGE",
                f.recovery_ticks,
                f.ended_quiescent ? "  (WEDGED: no work left, still behind)"
                                  : (f.ended_on_cap ? "  (hit the runaway cap)"
                                                    : ""));
    std::printf("  cover complete           %s   (%d of %zu ticks)\n",
                f.final_cover_complete ? "yes" : "still catching up",
                f.ticks_complete, f.ticks.size());
    std::printf("  stand-ins released       %llu\n",
                static_cast<unsigned long long>(f.holds_dropped));
    std::printf("  reserve released         %llu\n",
                static_cast<unsigned long long>(f.retained_dropped));
    std::printf("  repair rounds peak       %d  (backstop %d)\n",
                f.repair_peak, pt::planet::kMaxRepairRounds);
    std::printf("  substitutions refused    %llu\n",
                static_cast<unsigned long long>(f.holds_refused));
    std::printf("  coverage regressions     %d (bug) / %d (deliberate)\n",
                f.coverage_regressions, f.regressions_under_fallback);
}

int main(int argc, char** argv) {
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int res = ctx.run();
    if (ctx.shouldExit()) return res;

    // Print the ledger unconditionally. A reader of a GREEN run has to be
    // able to tell "the paced stream held chunks and never holed out" from
    // "the device did not come up and nothing ran" -- the difference this
    // project has been bitten by before.
    std::printf("\n--- paced terrain residency ---\n");
    if (!g_flight.ran) {
        std::printf("  SKIPPED: %s\n", g_flight.skip_reason.c_str());
        // 125 is the ctest SKIP_RETURN_CODE convention used by
        // rhi_accel_update; a green tick over zero coverage is worse than
        // an honest skip.
        return (res == 0) ? 125 : res;
    }
    std::size_t peak_desired = 0, peak_pub = 0, peak_held = 0, peak_starved = 0;
    for (const auto& t : g_flight.ticks) {
        peak_desired = std::max(peak_desired, t.desired);
        peak_pub     = std::max(peak_pub,     t.published);
        peak_held    = std::max(peak_held,    t.held);
        peak_starved = std::max(peak_starved, t.starved);
    }
    std::printf("  ticks                    %zu\n", g_flight.ticks.size());
    std::printf("  distinct desired sets    %d\n", g_flight.distinct_desired_digests);
    std::printf("  peak desired             %zu  (budget %d)\n",
                peak_desired, kChunkBudget);
    std::printf("  peak published           %zu\n", peak_pub);
    std::printf("  peak held (stand-ins)    %zu\n", peak_held);
    std::size_t peak_retained = 0, peak_cut = 0;
    for (const auto& t : g_flight.ticks) {
        peak_retained = std::max(peak_retained, t.retained);
        peak_cut      = std::max(peak_cut,      t.cut);
    }
    std::size_t peak_cut_res = 0;
    for (const auto& t : g_flight.ticks) {
        peak_cut_res = std::max(peak_cut_res, t.cut_resident);
    }
    std::printf("  peak retained ancestors  %zu  (whole cut peak %zu, "
                "cut resident peak %zu)\n",
                peak_retained, peak_cut, peak_cut_res);
    std::printf("  resident HIGH WATER      %zu  (arena %zu, headroom %zu)\n",
                g_flight.resident_peak, g_flight.arena_slots,
                g_flight.arena_slots - g_flight.resident_peak);
    std::printf("  peak starved             %zu\n", peak_starved);
    std::printf("  ticks holding            %d\n", g_flight.ticks_with_holds);
    std::printf("  ticks mid-stream         %d\n", g_flight.ticks_incomplete);
    std::printf("  holds dropped (arena)    %llu\n",
                static_cast<unsigned long long>(g_flight.holds_dropped));
    std::printf("  holds refused (2:1)      %llu\n",
                static_cast<unsigned long long>(g_flight.holds_refused));
    std::printf("  reserve released         %llu\n",
                static_cast<unsigned long long>(g_flight.retained_dropped));
    std::printf("  repair rounds peak       %d  (backstop %d)\n",
                g_flight.repair_peak, pt::planet::kMaxRepairRounds);
    std::printf("  COVERAGE REGRESSIONS     %d   (must be 0)\n",
                g_flight.coverage_regressions);
    std::printf("  ... under fallback       %d   (deliberate holes)\n",
                g_flight.regressions_under_fallback);
    std::printf("  overlapping covers       %d\n", g_flight.overlap_failures);
    std::printf("  unbalanced covers        %d\n", g_flight.imbalance_failures);
    return res;
}

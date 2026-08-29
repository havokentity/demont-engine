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
//   4. EVERY PUBLISHED CHUNK IS RESIDENT, and residency never exceeds
//      r_planet_chunk_budget. The arena is a hard cap and holding an
//      outgoing chunk alongside its incoming replacement is exactly the
//      pressure that could break it.
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
    std::size_t desired    = 0;
    std::size_t resident   = 0;
    std::size_t published  = 0;
    std::size_t held       = 0;
    std::size_t starved    = 0;
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
    std::uint64_t holds_dropped = 0;
    std::uint64_t holds_refused = 0;
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

    auto tick = [&](double altitude_m) {
        pt::planet::LodParams lod = cfg.lod;
        lod.camera_w = camera_at(lat_rad, lon_rad, altitude_m);
        lon_rad += lon_rate;

        terrain.Update(lod, glm::dvec3(0.0));

        const std::set<ChunkKey>& pub = terrain.Published();
        const auto& st = terrain.Stats();

        // 1. Coverage never regresses -- unless the policy deliberately
        //    chose a hole this very tick, which is the only licence it has.
        const bool fallback_fired = (st.holds_refused != prev_refused) ||
                                    (st.holds_dropped != prev_dropped);
        prev_refused = st.holds_refused;
        prev_dropped = st.holds_dropped;
        if (!prev_published.empty() && !CoversAll(pub, prev_published)) {
            if (fallback_fired) ++fr.regressions_under_fallback;
            else                ++fr.coverage_regressions;
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
        rec.desired   = st.desired;
        rec.resident  = st.resident;
        rec.published = st.published;
        rec.held      = st.held;
        rec.starved   = st.starved;
        fr.ticks.push_back(rec);
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
    if (plan.teleport) {
        // A quarter of the way round the planet in one frame, at the SAME
        // altitude. Same demand, none of it resident: every chunk in the
        // arena becomes a stand-in for ground nobody is looking at, and the
        // incoming set has 128 slots' worth of nowhere to go.
        lon_rad += 3.14159265358979323846 * 0.5;
    }
    for (int i = 0; i < plan.climb_ticks; ++i) {
        if (plan.settle_first) {
            // Hold the altitude. The teleport alone is the disturbance; a
            // climb on top of it would confound "the arena was overrun" with
            // "the selector wanted less anyway".
            tick(kAltBottom_m);
            continue;
        }
        const double t = static_cast<double>(i) / (plan.climb_ticks - 1);
        tick(kAltBottom_m + (kAltStart_m - kAltBottom_m) * t);
    }

    fr.resident_peak = terrain.Stats().resident_peak;
    fr.final_held      = terrain.Stats().held;
    fr.final_published = terrain.Stats().published;
    fr.final_resident  = terrain.Stats().resident;
    fr.final_desired   = terrain.Stats().desired;
    fr.final_cover_complete = CoversAll(terrain.Published(), terrain.Desired());
    fr.holds_dropped = terrain.Stats().holds_dropped;
    fr.holds_refused = terrain.Stats().holds_refused;
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

    // The arena is a hard cap and the transient must live inside it.
    CHECK(g_flight.resident_peak <= static_cast<std::size_t>(kChunkBudget));
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
    // tau 6 px rather than the flight's 1.5, and a 256-slot arena. Both
    // numbers exist to keep the SELECTOR off its own budget cap while the
    // ARENA is squeezed hard, and that is not fussiness: TerrainQuadtree's
    // EnforceBudget raises tau by bisection over trial sets that treat an
    // unmeasured node as a leaf, so on the cap during a descent it
    // under-counts and settles on a far coarser tau than a fully-informed
    // bisection would -- the cvar help for r_planet_chunk_budget says so and
    // says it is worth closing "if a fixture ever needs to sit on the cap".
    // Measured here at 4 km altitude, max level 8: 657 leaves uncapped
    // against 126 with a 128-chunk cap at one longitude and 12 at another,
    // the second collapsing to level 1 over the whole planet. Sitting there
    // would make this case a test of that quirk instead of a test of the
    // arena. At tau 6 the selector wants 189 leaves before the jump and 213
    // after, so 224 slots is genuine pressure -- 189 stand-ins and 213
    // incomers cannot both fit -- while still leaving room for the recovery
    // to complete, and with the selector's own cap never binding.
    plan.tau_px        = 6.0;
    plan.chunk_budget  = 224;
    plan.max_level     = 8;
    plan.settle_first  = true;    // fill the arena on every host, not just fast ones
    plan.descent_ticks = 0;
    plan.climb_ticks   = 150;
    plan.teleport      = true;
    // 8 ms rather than the interactive 2. What this case is about is the
    // arena, and the recovery has 213 chunks to build; at the interactive
    // budget a Debug host would still be halfway through when the ticks ran
    // out, and "did not finish" is not the same claim as "did not deadlock".
    plan.blas_budget_ms = 8.0;
    plan.frame_sleep_ms = 8;
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
    CHECK(f.resident_peak <= 224u);
    // Still no coincident surfaces and still no crack, under pressure.
    CHECK(f.overlap_failures   == 0);
    CHECK(f.imbalance_failures == 0);
    CHECK(f.orphan_failures    == 0);
    // NO DEADLOCK. The failure this shape of policy invites is a stand-in
    // squatting the slot its own replacement needs, so that the arena stays
    // full of ground nobody is looking at and the new region trickles in a
    // slot at a time. Two statements rule it out, and neither asks how fast
    // the host is:
    //
    //   * the arena is no longer dominated by the abandoned region -- most
    //     of the 189 stand-ins the jump created are gone;
    //   * and the streamer is drawing at least as much as it was before the
    //     jump, so the slots they were holding went to the new region
    //     rather than nowhere.
    //
    // Deliberately NOT "resident == desired" or "the cover is complete".
    // Whether a 224-slot arena finishes refining 213 chunks inside the tick
    // budget is a question about how fast this host bakes and builds -- a
    // Debug runner is still 50 stand-ins short at the end -- and being
    // behind is not the same thing as being stuck. Both numbers are printed
    // below so a reader can see which it was.
    CHECK(f.final_held * 2 <= f.settled_resident);
    CHECK(f.final_published >= f.settled_resident);
    // Non-vacuous, on any host. The barrier really did fill the arena
    // before the jump -- Settle() is machine-independent, which is the
    // whole reason it is used here -- and the jump really did turn that
    // arena into stand-ins and starve the incoming set, so the release path
    // below was exercised rather than merely present.
    CHECK(f.settled_resident >= 150u);         // the selector wanted 189
    CHECK(peak_held  >= 100u);                 // stand-ins, held at once
    CHECK(peak_starved > 0u);                  // demand really was blocked
    CHECK(f.holds_dropped > 0u);

    std::printf("\n--- teleport into a 224-chunk arena ---\n");
    std::printf("  settled residency        %zu of %zu desired\n",
                f.settled_resident, f.settled_desired);
    std::printf("  resident HIGH WATER      %zu  (arena 224)\n", f.resident_peak);
    std::printf("  final published          %zu  (resident %zu, desired %zu)\n",
                f.final_published, f.final_resident, f.final_desired);
    std::printf("  peak starved / held      %zu / %zu\n", peak_starved, peak_held);
    std::printf("  final held               %zu   (settled %zu)\n",
                f.final_held, f.settled_resident);
    std::printf("  cover complete           %s   (%d of %zu ticks)\n",
                f.final_cover_complete ? "yes" : "still catching up",
                f.ticks_complete, f.ticks.size());
    std::printf("  stand-ins released       %llu\n",
                static_cast<unsigned long long>(f.holds_dropped));
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
    std::printf("  resident HIGH WATER      %zu  (arena %d, headroom %zu)\n",
                g_flight.resident_peak, kChunkBudget,
                static_cast<std::size_t>(kChunkBudget) - g_flight.resident_peak);
    std::printf("  peak starved             %zu\n", peak_starved);
    std::printf("  ticks holding            %d\n", g_flight.ticks_with_holds);
    std::printf("  ticks mid-stream         %d\n", g_flight.ticks_incomplete);
    std::printf("  holds dropped (arena)    %llu\n",
                static_cast<unsigned long long>(g_flight.holds_dropped));
    std::printf("  holds refused (2:1)      %llu\n",
                static_cast<unsigned long long>(g_flight.holds_refused));
    std::printf("  COVERAGE REGRESSIONS     %d   (must be 0)\n",
                g_flight.coverage_regressions);
    std::printf("  ... under fallback       %d   (deliberate holes)\n",
                g_flight.regressions_under_fallback);
    std::printf("  overlapping covers       %d\n", g_flight.overlap_failures);
    std::printf("  unbalanced covers        %d\n", g_flight.imbalance_failures);
    return res;
}

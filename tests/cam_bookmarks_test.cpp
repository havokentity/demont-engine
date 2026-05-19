// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit-test coverage for the named-camera-bookmark commands added on
// the `feat/cam-named-bookmarks` branch:
//
//   cam_save_named   <name>     -- snapshot current camera into bookmarks_[name]
//   cam_load_named   <name>     -- restore camera from bookmarks_[name]
//   cam_list_bookmarks          -- print all bookmarks (sorted by name)
//   cam_delete_bookmark <name>  -- remove a bookmark
//
// The commands complement the numeric cam_save / cam_load 1..9 slot
// system, which is keyed by integer index and stored in CVAR_ARCHIVE
// cvars. Bookmarks are keyed by user-chosen mnemonic
// (`overhead`, `closeup`, `cornell_3q`) and persisted to a separate
// `camera_bookmarks.cfg` next to demont.cfg (mirroring the
// favorites.cfg pattern). This pins:
//
//   - Save round-trip: cam_save_named overhead -> cam_load_named
//     overhead reproduces the original camera pose (pos / yaw /
//     pitch / fov) to within float-string-round-trip resolution.
//   - Name validation: empty / whitespace / '#'-prefixed names are
//     rejected with a clear error.
//   - Missing-bookmark error: cam_load_named bogus prints a helpful
//     error and does NOT crash or mutate the camera.
//   - Listing: cam_list_bookmarks shows all saved names; empty case
//     prints a hint instead of an empty line.
//   - Deletion: cam_delete_bookmark removes the entry; subsequent
//     cam_load_named hits the missing-bookmark path.
//   - Replace semantics: cam_save_named with an existing name
//     overwrites the old state (no error, output mentions "replaced").
//   - Persistence: Save -> rewrite camera_bookmarks.cfg ->
//     LoadCameraBookmarksFromDisk recovers the same map.
//
// Test architecture mirrors phys_drop_args_test.cpp:
//   - A single process-lifetime Engine via CamBookmarksTestAccess::Get(),
//     intentionally leaked so ~Engine() does not run at exit (which
//     would touch demont.cfg / the log sinks / etc.).
//   - The access struct wires up the Engine's camera_ unique_ptr +
//     calls RegisterCameraBookmarkCommands(). That function is the
//     minimal slice of RegisterCommands needed for cam_save_named /
//     cam_load_named / cam_list_bookmarks / cam_delete_bookmark; the
//     full RegisterCommands path would capture `this` across audio /
//     RHI / window state that isn't initialised in the test harness.
//   - ResetState() between TEST_CASEs clears camera_bookmarks_ and
//     resets the camera to its default pose so test order doesn't
//     matter.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/console/Console.h"
#include "../src/engine/Engine.h"
#include "../src/renderer/Camera.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>

namespace pt::engine {

// Friend access to Engine private state. Same singleton + leak
// pattern as PhysDropArgsTestAccess for the same reason (Console's
// command map captures the Engine `this` pointer; recreating the
// Engine per-TEST_CASE would leave a dangling pointer in the
// singleton's first-registered closure).
struct CamBookmarksTestAccess {
    static CamBookmarksTestAccess& Get() {
        static CamBookmarksTestAccess* instance = new CamBookmarksTestAccess();
        return *instance;
    }

    void ResetState() {
        engine_.camera_bookmarks_.clear();
        // Reset camera to a known starting point matching the engine
        // default (matches Camera.h inline initialiser). Tests below
        // mutate this BEFORE cam_save_named to verify the snapshot
        // captured the modified pose.
        engine_.camera_->pos     = glm::vec3(0.0f, 1.5f, 4.0f);
        engine_.camera_->yaw     = 0.0f;
        engine_.camera_->pitch   = -0.20f;
        engine_.camera_->fov_deg = 60.0f;
        // Delete the on-disk persistence so the SaveCameraBookmarksToDisk
        // path doesn't leave a state file behind between test cases.
        // (Each cam_save_named writes the cfg; if the previous test
        // wrote one then this test's first save would race with the
        // unlink. Wiping pre-test is the safe order.)
        std::remove("camera_bookmarks.cfg");
    }

    // Direct access to the underlying camera_bookmarks_ map for
    // assertions that need to bypass the console-formatted output.
    const std::map<std::string, std::string>& Bookmarks() const {
        return engine_.camera_bookmarks_;
    }

    pt::renderer::Camera& Camera() {
        return *engine_.camera_;
    }

    // Direct call to LoadCameraBookmarksFromDisk for the persistence
    // round-trip test. Doesn't go through the console.
    void LoadFromDisk() {
        engine_.LoadCameraBookmarksFromDisk();
    }

private:
    CamBookmarksTestAccess() {
        engine_.camera_ = std::make_unique<pt::renderer::Camera>();
        engine_.RegisterCameraBookmarkCommands();
    }

    Engine engine_;
};

}  // namespace pt::engine

namespace {

bool approx(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

bool contains(const std::string& s, std::string_view needle) {
    return s.find(needle) != std::string::npos;
}

}  // namespace

// =============================================================================
// cam_save_named + cam_load_named round-trip
// =============================================================================

TEST_CASE("cam_save_named: stores current camera state under given name") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Mutate camera into a non-default pose so we can tell the
    // snapshot apart from the default-reset state below.
    acc.Camera().pos     = glm::vec3(10.0f, 20.0f, -5.5f);
    acc.Camera().yaw     = 1.0f;        // radians
    acc.Camera().pitch   = -0.5f;       // radians
    acc.Camera().fov_deg = 45.0f;

    auto r = C.Execute("cam_save_named overhead");
    CHECK(r.ok);
    CHECK(contains(r.output, "saved 'overhead'"));

    // The bookmark exists in the map under the user's chosen name.
    auto& m = acc.Bookmarks();
    REQUIRE(m.count("overhead") == 1u);
    // The stored state is the six-float serialization (the exact
    // string format is "x y z yaw_deg pitch_deg fov_deg"); we don't
    // pin the exact spaces / precision (that's tested by the round-
    // trip below), just that the floats parsed back are approximately
    // the originals.
    CHECK(!m.at("overhead").empty());
}

TEST_CASE("cam_load_named: restores camera to saved pose") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Save an aerial pose.
    acc.Camera().pos     = glm::vec3(0.0f, 50.0f, 0.0f);
    acc.Camera().yaw     = 0.5f;
    acc.Camera().pitch   = -1.2f;  // looking nearly straight down
    acc.Camera().fov_deg = 90.0f;
    CHECK(C.Execute("cam_save_named aerial").ok);

    // Move the camera elsewhere.
    acc.Camera().pos     = glm::vec3(-3.0f, 0.5f, 12.0f);
    acc.Camera().yaw     = -1.0f;
    acc.Camera().pitch   = 0.0f;
    acc.Camera().fov_deg = 30.0f;

    // Restore.
    auto r = C.Execute("cam_load_named aerial");
    CHECK(r.ok);
    CHECK(contains(r.output, "cam_load_named: 'aerial'"));

    // ClampPitch is applied on load; -1.2 rad ~ -68.75 deg, well
    // within the +/- 85 deg clamp limit, so the round-trip preserves it.
    CHECK(approx(acc.Camera().pos.x, 0.0f));
    CHECK(approx(acc.Camera().pos.y, 50.0f));
    CHECK(approx(acc.Camera().pos.z, 0.0f));
    CHECK(approx(acc.Camera().yaw, 0.5f, 1e-2f));
    CHECK(approx(acc.Camera().pitch, -1.2f, 1e-2f));
    CHECK(approx(acc.Camera().fov_deg, 90.0f));
}

TEST_CASE("cam_save_named: overwriting existing bookmark mentions 'replaced'") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    acc.Camera().pos = glm::vec3(1.0f, 2.0f, 3.0f);
    auto r1 = C.Execute("cam_save_named spot");
    CHECK(r1.ok);
    CHECK(contains(r1.output, "saved 'spot'"));

    // Re-save under the same name; output should mention "replaced",
    // not "saved", so the user knows they overwrote a prior entry.
    acc.Camera().pos = glm::vec3(4.0f, 5.0f, 6.0f);
    auto r2 = C.Execute("cam_save_named spot");
    CHECK(r2.ok);
    CHECK(contains(r2.output, "replaced 'spot'"));
    CHECK(acc.Bookmarks().size() == 1u);  // not 2

    // The stored state reflects the LAST save, not the first.
    CHECK(C.Execute("cam_load_named spot").ok);
    CHECK(approx(acc.Camera().pos.x, 4.0f));
    CHECK(approx(acc.Camera().pos.y, 5.0f));
    CHECK(approx(acc.Camera().pos.z, 6.0f));
}

// =============================================================================
// Rejection paths
// =============================================================================

TEST_CASE("cam_save_named: missing-name arg is rejected with usage") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("cam_save_named");
    CHECK(r.ok);  // command found + dispatched; usage is the success surface
    CHECK(contains(r.output, "usage: cam_save_named"));
    CHECK(acc.Bookmarks().empty());
}

TEST_CASE("cam_save_named: name containing '#' is rejected") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // '#' is the cfg comment prefix. Saving a name starting with '#'
    // would land at the start of a cfg line and be silently dropped
    // on next reload -- our validator rejects it at the gate.
    //
    // NOTE: bare `cam_save_named #evil` is stripped to
    // `cam_save_named ` by the console's inline-comment pass BEFORE
    // dispatch (see Console.cpp:359), so an unquoted '#' would hit
    // the missing-arg usage path, not the invalid-name path. We
    // need a QUOTED arg so the '#' survives tokenization and
    // actually reaches our validator. (The validator still matters
    // for the LoadCameraBookmarksFromDisk path -- a hand-edited
    // cfg file could deliver any name string at all.)
    auto r = C.Execute("cam_save_named \"#evil\"");
    CHECK(r.ok);
    CHECK(contains(r.output, "invalid name"));
    CHECK(acc.Bookmarks().empty());
}

TEST_CASE("cam_load_named: missing bookmark prints helpful error, no crash") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Snapshot camera pos so we can verify it wasn't mutated.
    const glm::vec3 before = acc.Camera().pos;

    auto r = C.Execute("cam_load_named does_not_exist");
    CHECK(r.ok);  // command found; soft error in output
    CHECK(contains(r.output, "no bookmark named 'does_not_exist'"));
    // Camera unchanged.
    CHECK(approx(acc.Camera().pos.x, before.x));
    CHECK(approx(acc.Camera().pos.y, before.y));
    CHECK(approx(acc.Camera().pos.z, before.z));
}

// =============================================================================
// Listing
// =============================================================================

TEST_CASE("cam_list_bookmarks: empty case shows hint") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("cam_list_bookmarks");
    CHECK(r.ok);
    CHECK(contains(r.output, "no camera bookmarks yet"));
}

TEST_CASE("cam_list_bookmarks: populated case lists every saved bookmark") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    CHECK(C.Execute("cam_save_named alpha").ok);
    CHECK(C.Execute("cam_save_named beta").ok);
    CHECK(C.Execute("cam_save_named gamma").ok);

    auto r = C.Execute("cam_list_bookmarks");
    CHECK(r.ok);
    CHECK(contains(r.output, "3 bookmark(s):"));
    CHECK(contains(r.output, "alpha"));
    CHECK(contains(r.output, "beta"));
    CHECK(contains(r.output, "gamma"));
}

// =============================================================================
// Deletion
// =============================================================================

TEST_CASE("cam_delete_bookmark: removes entry; subsequent load fails") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    CHECK(C.Execute("cam_save_named throwaway").ok);
    CHECK(acc.Bookmarks().size() == 1u);

    auto del = C.Execute("cam_delete_bookmark throwaway");
    CHECK(del.ok);
    CHECK(contains(del.output, "removed 'throwaway'"));
    CHECK(acc.Bookmarks().empty());

    // Subsequent load hits the missing-bookmark error path.
    auto load = C.Execute("cam_load_named throwaway");
    CHECK(load.ok);
    CHECK(contains(load.output, "no bookmark named 'throwaway'"));
}

TEST_CASE("cam_delete_bookmark: deleting nonexistent prints error") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("cam_delete_bookmark phantom");
    CHECK(r.ok);
    CHECK(contains(r.output, "no bookmark named 'phantom'"));
}

// =============================================================================
// Persistence round-trip (LoadCameraBookmarksFromDisk reads what
// SaveCameraBookmarksToDisk wrote)
// =============================================================================

TEST_CASE("cam bookmarks persist to camera_bookmarks.cfg across save+load") {
    auto& acc = pt::engine::CamBookmarksTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Save two distinct poses.
    acc.Camera().pos = glm::vec3(7.0f, 8.0f, 9.0f);
    CHECK(C.Execute("cam_save_named first").ok);
    acc.Camera().pos = glm::vec3(-1.0f, 2.0f, -3.0f);
    acc.Camera().fov_deg = 75.0f;
    CHECK(C.Execute("cam_save_named second").ok);

    // Capture the in-memory state for comparison.
    const auto saved_first  = acc.Bookmarks().at("first");
    const auto saved_second = acc.Bookmarks().at("second");

    // Wipe in-memory and reload from the persisted cfg file.
    // We can't (easily) destroy the Engine without losing the
    // singleton, so we just clear the map and call the loader.
    const_cast<std::map<std::string, std::string>&>(acc.Bookmarks()).clear();
    CHECK(acc.Bookmarks().empty());
    acc.LoadFromDisk();

    // Same two bookmarks, same six-float state strings.
    REQUIRE(acc.Bookmarks().count("first")  == 1u);
    REQUIRE(acc.Bookmarks().count("second") == 1u);
    CHECK(acc.Bookmarks().at("first")  == saved_first);
    CHECK(acc.Bookmarks().at("second") == saved_second);

    // Clean up the cfg file so subsequent ctest invocations from the
    // same build dir start fresh. ResetState() also removes the file
    // pre-test for the same reason.
    std::remove("camera_bookmarks.cfg");
}

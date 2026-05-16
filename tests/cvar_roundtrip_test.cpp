// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Cvar persistence round-trip. Catches:
//
//   - SaveArchivedCvars emitting a malformed line that ExecuteScript
//     can't parse back (broken quoting, missing space, etc.)
//   - ExecuteScript silently dropping a value during the load
//   - Default-skip logic accidentally writing defaults (would
//     bloat demont.cfg) or skipping non-defaults (would lose state)
//   - Quote handling for values containing spaces
//
// Strategy: register cvars with unique test-* names so we don't fight
// the engine's own registered cvars. Set them to non-default values.
// Save to a temp file. Set them to OTHER values (so the load has
// something to restore). Read the file content + replay via
// ExecuteScript. Assert the cvars now hold the saved values.
//
// Console is a singleton; each test executable is its own process so
// cross-test global-state taint isn't a concern.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/console/Console.h"

#include <cstdio>      // std::remove
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// Read whole file into a string. Used to feed ExecuteScript a snapshot
// of what SaveArchivedCvars just wrote. Returns empty string on missing
// file (test asserts catch this -- saving should never silently fail).
std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Per-process temp path under the system tmpdir. doctest doesn't ship a
// tempdir helper; std::filesystem::temp_directory_path() works
// cross-platform (Mac /var/folders/.../T, Windows %TEMP%, Linux /tmp).
fs::path temp_cfg_path(std::string_view test_name) {
    auto p = fs::temp_directory_path() /
             fs::path(std::string("demont_") + std::string(test_name) + ".cfg");
    return p;
}

}  // namespace

TEST_CASE("cvar round-trip: save + load restores values") {
    auto& C = pt::console::Console::Get();

    // Register four test cvars covering the format-relevant cases:
    //   - integer-string value
    //   - float-string value
    //   - value with no spaces
    //   - value containing a space (forces the SaveArchivedCvars
    //     quoting branch at Console.cpp:519)
    C.RegisterCVar("test_rt_int",      "0",         "test int",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_float",    "0.0",       "test float",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_string",   "default",   "test string",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_quoted",   "default",   "test quoted",
                   pt::console::CVAR_ARCHIVE);

    // Set them to known non-default values.
    REQUIRE(C.Execute("test_rt_int 42").ok);
    REQUIRE(C.Execute("test_rt_float 3.14159").ok);
    REQUIRE(C.Execute("test_rt_string hello_world").ok);
    REQUIRE(C.Execute("test_rt_quoted \"a value with spaces\"").ok);

    // Save snapshot.
    const auto path = temp_cfg_path("rt_save");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n > 0);
    REQUIRE(fs::exists(path));

    SUBCASE("file contains expected lines") {
        const std::string body = slurp(path);
        CHECK(body.find("test_rt_int 42")            != std::string::npos);
        CHECK(body.find("test_rt_float 3.14159")     != std::string::npos);
        CHECK(body.find("test_rt_string hello_world")!= std::string::npos);
        // Quoted form: value contains spaces, must be wrapped in "".
        CHECK(body.find("test_rt_quoted \"a value with spaces\"") != std::string::npos);
    }

    SUBCASE("load restores values after they've been changed") {
        // Mutate to clearly-different values so the load has something
        // to actually restore. If the load path silently dropped a
        // line, the assertion below would still see the mutated value
        // instead of the saved one -- that's the failure mode the
        // round-trip is specifically checking for.
        REQUIRE(C.Execute("test_rt_int 999").ok);
        REQUIRE(C.Execute("test_rt_float 99.0").ok);
        REQUIRE(C.Execute("test_rt_string mutated").ok);
        REQUIRE(C.Execute("test_rt_quoted mutated_quoted").ok);

        // Replay the saved snapshot. ExecuteScript handles multi-line
        // bodies with `// comments`, blank lines, and semicolon-
        // separated statements -- same as how Engine::Init replays
        // demont.cfg on launch.
        const std::string body = slurp(path);
        auto r = C.ExecuteScript(body);
        CHECK(r.ok);

        // Values should match what we saved, not what we mutated to.
        auto* cv_int    = C.FindCVar("test_rt_int");
        auto* cv_float  = C.FindCVar("test_rt_float");
        auto* cv_str    = C.FindCVar("test_rt_string");
        auto* cv_quoted = C.FindCVar("test_rt_quoted");
        REQUIRE(cv_int    != nullptr);
        REQUIRE(cv_float  != nullptr);
        REQUIRE(cv_str    != nullptr);
        REQUIRE(cv_quoted != nullptr);

        CHECK(cv_int->GetInt()    == 42);
        CHECK(cv_float->GetFloat() == doctest::Approx(3.14159f));
        CHECK(cv_str->value     == "hello_world");
        CHECK(cv_quoted->value  == "a value with spaces");
    }

    // Cleanup. std::remove tolerates a missing file; we want best-
    // effort cleanup either way (the temp_directory_path tree is
    // OS-managed so leftover files are cosmetic, not leaks).
    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("cvar round-trip: defaults are NOT serialized") {
    // SaveArchivedCvars deliberately skips cvars whose current value
    // equals their default (Console.cpp:518). This keeps demont.cfg
    // tight -- only changes from the default get persisted. Test the
    // skip logic: a freshly-registered cvar at its default value
    // should not appear in the output.
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_unset", "default_value", "unset test",
                   pt::console::CVAR_ARCHIVE);

    // Don't set it -- value == default.
    const auto path = temp_cfg_path("rt_defaults");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n >= 0);  // success; could be 0 or any positive count from other tests

    const std::string body = slurp(path);
    CHECK(body.find("test_rt_unset") == std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("cvar round-trip: non-archive cvars are NOT serialized") {
    // CVAR_ARCHIVE is the gate for SaveArchivedCvars (Console.cpp:517).
    // Cvars without it -- READONLY engine state, internal dev knobs,
    // temporary state -- must not leak into demont.cfg even if mutated.
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_non_archive", "0", "non-archive test",
                   pt::console::CVAR_NONE);
    REQUIRE(C.Execute("test_rt_non_archive 7").ok);

    const auto path = temp_cfg_path("rt_non_archive");
    C.SaveArchivedCvars(path.string());

    const std::string body = slurp(path);
    CHECK(body.find("test_rt_non_archive") == std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

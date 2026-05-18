// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pins two UX behaviours on the cvar surface:
//
//   1. CVAR_PLATFORM_{MAC,WIN}: cvars carrying the wrong-platform bit
//      stay registered (set / get still work, so demont.cfg keeps
//      round-tripping on a shared drive) but vanish from
//      EnumerateCVars output -- the source for `list_cvars` and for
//      autocomplete via BuildCompletions / Console::EnumerateCVars.
//
//   2. "0" as off-shorthand: setting an enum-style cvar (one with
//      allowed_values that includes an "off"/"none"/"disabled" token)
//      to the literal "0" is rewritten to that off token before the
//      allowed_values gate fires. Only "0" gets the rewrite -- "1",
//      other digits, and bool cvars with allowed_values = {"0","1"}
//      keep their existing semantics.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/console/Console.h"

#include <string>

namespace {

// Walk EnumerateCVars and return true iff a cvar with `name` is in
// the visible set. The platform filter lives inside EnumerateCVars
// itself; FindCVar deliberately does NOT filter (the cvar must still
// be accessible by name for shared-cfg round-trip), so this helper
// distinguishes "registered but hidden" from "absent".
bool cvar_visible(const std::string& name) {
    auto& C = pt::console::Console::Get();
    bool seen = false;
    C.EnumerateCVars("", [&](pt::console::CVar& v) {
        if (v.name == name) seen = true;
    });
    return seen;
}

}  // namespace

TEST_CASE("cvar platform-filter: wrong-platform bit hides from listing") {
    auto& C = pt::console::Console::Get();

    // Three witnesses: an unflagged baseline, plus one of each
    // platform bit. The wrong-platform one for THIS build is the
    // one that should be hidden; the other should still show up.
    C.RegisterCVar("test_ux_plain", "0", "no-platform cvar",
                   pt::console::CVAR_NONE);
    C.RegisterCVar("test_ux_mac_only", "0", "mac-only cvar",
                   pt::console::CVAR_PLATFORM_MAC);
    C.RegisterCVar("test_ux_win_only", "0", "win-only cvar",
                   pt::console::CVAR_PLATFORM_WIN);

    // FindCVar must return all three regardless of platform -- the cvar
    // stays registered everywhere so a shared demont.cfg can set it
    // without the parser tripping over an unknown name on the other host.
    CHECK(C.FindCVar("test_ux_plain")    != nullptr);
    CHECK(C.FindCVar("test_ux_mac_only") != nullptr);
    CHECK(C.FindCVar("test_ux_win_only") != nullptr);

    // Plain cvar is always visible.
    CHECK(cvar_visible("test_ux_plain"));

#if defined(__APPLE__)
    CHECK(cvar_visible("test_ux_mac_only"));
    CHECK_FALSE(cvar_visible("test_ux_win_only"));
#elif defined(_WIN32)
    CHECK(cvar_visible("test_ux_win_only"));
    CHECK_FALSE(cvar_visible("test_ux_mac_only"));
#else
    // Linux / other: both platform-flagged cvars are hidden.
    CHECK_FALSE(cvar_visible("test_ux_mac_only"));
    CHECK_FALSE(cvar_visible("test_ux_win_only"));
#endif

    // The hidden cvar still accepts Execute writes (so cfg load on
    // the wrong host doesn't bounce a saved value back to default).
    CHECK(C.Execute("test_ux_mac_only 7").ok);
    CHECK(C.Execute("test_ux_win_only 7").ok);
    CHECK(C.FindCVar("test_ux_mac_only")->value == "7");
    CHECK(C.FindCVar("test_ux_win_only")->value == "7");
}

TEST_CASE("cvar zero-as-off: '0' resolves to off on enum cvars with off token") {
    auto& C = pt::console::Console::Get();

    // Mirror the r_denoiser shape: allowed_values list begins with
    // "off" and continues with several non-numeric tokens. The cvar
    // gate normally rejects "0" since "0" is not in the allowed set
    // -- the new alias path should rewrite "0" to "off" before the
    // gate fires.
    C.RegisterCVar("test_ux_enum_off", "alpha", "enum with off token",
                   pt::console::CVAR_NONE);
    auto* v = C.FindCVar("test_ux_enum_off");
    REQUIRE(v != nullptr);
    v->allowed_values = {"off", "alpha", "beta"};

    CHECK(C.Execute("test_ux_enum_off 0").ok);
    CHECK(v->value == "off");

    // Round-trip: a subsequent set to a real token works the same as
    // before (no lingering effect of the alias rewrite).
    CHECK(C.Execute("test_ux_enum_off beta").ok);
    CHECK(v->value == "beta");

    // "1" must NOT be rewritten -- the alias is intentionally
    // asymmetric. "1" is not in the allowed set so the gate rejects.
    auto r1 = C.Execute("test_ux_enum_off 1");
    CHECK_FALSE(r1.ok);
    CHECK(v->value == "beta");
}

TEST_CASE("cvar zero-as-off: 'none' / 'disabled' / case variants also accepted") {
    auto& C = pt::console::Console::Get();

    // "none" form: r_perf_overlay / r_capture_format share this shape.
    C.RegisterCVar("test_ux_enum_none", "rhi", "enum with none token",
                   pt::console::CVAR_NONE);
    auto* vn = C.FindCVar("test_ux_enum_none");
    REQUIRE(vn != nullptr);
    vn->allowed_values = {"none", "rhi", "renderer"};
    CHECK(C.Execute("test_ux_enum_none 0").ok);
    CHECK(vn->value == "none");

    // "disabled" form: no production cvar uses this today but the
    // shorthand is documented as accepting it, so pin the contract.
    C.RegisterCVar("test_ux_enum_disabled", "on", "enum with disabled token",
                   pt::console::CVAR_NONE);
    auto* vd = C.FindCVar("test_ux_enum_disabled");
    REQUIRE(vd != nullptr);
    vd->allowed_values = {"disabled", "on"};
    CHECK(C.Execute("test_ux_enum_disabled 0").ok);
    CHECK(vd->value == "disabled");

    // Case-insensitive match on the allowed_values token: a cvar that
    // registers "OFF" (uppercase) still picks up the "0" alias.
    C.RegisterCVar("test_ux_enum_upper", "A", "enum with OFF token",
                   pt::console::CVAR_NONE);
    auto* vu = C.FindCVar("test_ux_enum_upper");
    REQUIRE(vu != nullptr);
    vu->allowed_values = {"OFF", "A", "B"};
    CHECK(C.Execute("test_ux_enum_upper 0").ok);
    CHECK(vu->value == "OFF");
}

TEST_CASE("cvar zero-as-off: bool-style {0,1} cvars are unaffected") {
    auto& C = pt::console::Console::Get();

    // A bool cvar stored as plain int (e.g. r_clouds shape with
    // allowed_values = {"0","1"}) must keep accepting "0" as the
    // literal value "0", not get rewritten to anything else. The
    // alias only fires when the allowed set contains an off-style
    // token; "0" itself doesn't trigger it.
    C.RegisterCVar("test_ux_bool", "1", "bool cvar",
                   pt::console::CVAR_NONE);
    auto* v = C.FindCVar("test_ux_bool");
    REQUIRE(v != nullptr);
    v->allowed_values = {"0", "1"};

    CHECK(C.Execute("test_ux_bool 0").ok);
    CHECK(v->value == "0");
    CHECK(C.Execute("test_ux_bool 1").ok);
    CHECK(v->value == "1");
}

TEST_CASE("cvar zero-as-off: cvars without allowed_values are unaffected") {
    auto& C = pt::console::Console::Get();

    // Free-form cvars (no allowed_values) accept any string; "0"
    // stays "0" even when an off-style token wouldn't match anything.
    C.RegisterCVar("test_ux_freeform", "x", "free-form cvar",
                   pt::console::CVAR_NONE);
    auto* v = C.FindCVar("test_ux_freeform");
    REQUIRE(v != nullptr);

    CHECK(C.Execute("test_ux_freeform 0").ok);
    CHECK(v->value == "0");
}

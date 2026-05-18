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

// ---------------------------------------------------------------------------
// Issue #161: per-value platform gating + cross-cvar dep warnings.
// ---------------------------------------------------------------------------

TEST_CASE("cvar per-value platform: wrong-platform value errors w/ available list") {
    auto& C = pt::console::Console::Get();

    // Three-value enum: one universal, one Mac-only, one Win-only.
    // The set path should error when the user picks a wrong-platform
    // value, listing only the available-on-this-platform values.
    // The value is still WRITTEN (so a shared demont.cfg round-trips
    // across hosts), but on_change does NOT fire -- the engine
    // handler can rely on `result.ok == true` to mean "switch
    // backends now".
    C.RegisterCVar("test_ux_pv_enum", "any_ok", "per-value-gated enum",
                   pt::console::CVAR_NONE);
    auto* v = C.FindCVar("test_ux_pv_enum");
    REQUIRE(v != nullptr);
    v->allowed_values = {"any_ok", "mac_only", "win_only"};
    v->allowed_value_flags = {
        pt::console::CVAR_VALUE_ANY,
        pt::console::CVAR_VALUE_MAC,
        pt::console::CVAR_VALUE_WIN,
    };

    // The universal value always succeeds without error.
    {
        auto r = C.Execute("test_ux_pv_enum any_ok");
        CHECK(r.ok);
        CHECK(v->value == "any_ok");
    }

    // Wrong-platform value: error, but value is still committed
    // (cfg round-trip preserved).
#if defined(__APPLE__)
    {
        auto r = C.Execute("test_ux_pv_enum win_only");
        CHECK_FALSE(r.ok);
        CHECK(v->value == "win_only");  // cfg round-trip
        CHECK(r.error.find("Windows-only") != std::string::npos);
        CHECK(r.error.find("not available on macOS") != std::string::npos);
        // available-on-this-platform list excludes the wrong-platform value.
        CHECK(r.error.find("any_ok") != std::string::npos);
        CHECK(r.error.find("mac_only") != std::string::npos);
    }
    // Correct-platform value succeeds on Mac.
    {
        auto r = C.Execute("test_ux_pv_enum mac_only");
        CHECK(r.ok);
        CHECK(v->value == "mac_only");
    }
#elif defined(_WIN32)
    {
        auto r = C.Execute("test_ux_pv_enum mac_only");
        CHECK_FALSE(r.ok);
        CHECK(v->value == "mac_only");
        CHECK(r.error.find("macOS-only") != std::string::npos);
        CHECK(r.error.find("not available on Windows") != std::string::npos);
    }
    {
        auto r = C.Execute("test_ux_pv_enum win_only");
        CHECK(r.ok);
        CHECK(v->value == "win_only");
    }
#else
    // Linux: both mac_only and win_only are wrong-platform.
    {
        auto r = C.Execute("test_ux_pv_enum mac_only");
        CHECK_FALSE(r.ok);
        CHECK(v->value == "mac_only");
    }
#endif
}

TEST_CASE("cvar per-value platform: universal-only set unaffected") {
    auto& C = pt::console::Console::Get();

    // Cvar with allowed_value_flags but everything universal -- the
    // gate path should be a no-op (no error, value commits, on_change
    // fires as normal).
    bool fired = false;
    C.RegisterCVar("test_ux_pv_universal", "a",
                   "all-universal enum", pt::console::CVAR_NONE,
                   [&fired](const pt::console::CVar&) { fired = true; });
    auto* v = C.FindCVar("test_ux_pv_universal");
    REQUIRE(v != nullptr);
    v->allowed_values = {"a", "b"};
    v->allowed_value_flags = {
        pt::console::CVAR_VALUE_ANY,
        pt::console::CVAR_VALUE_ANY,
    };

    auto r = C.Execute("test_ux_pv_universal b");
    CHECK(r.ok);
    CHECK(v->value == "b");
    CHECK(fired);
}

TEST_CASE("cvar dep warning: unmet predicate prints [warn] hint in output") {
    auto& C = pt::console::Console::Get();

    // Set up a pair: dep_target depends on dep_source being "on".
    C.RegisterCVar("test_ux_dep_source", "off", "predicate source",
                   pt::console::CVAR_NONE);
    C.RegisterCVar("test_ux_dep_target", "0", "predicate-gated target",
                   pt::console::CVAR_NONE);
    auto* src = C.FindCVar("test_ux_dep_source");
    auto* tgt = C.FindCVar("test_ux_dep_target");
    REQUIRE(src != nullptr);
    REQUIRE(tgt != nullptr);
    src->allowed_values = {"on", "off"};

    tgt->requires_predicate = []() {
        auto* s = pt::console::Console::Get().FindCVar("test_ux_dep_source");
        return s != nullptr && s->value == "on";
    };
    tgt->requires_hint = "test_ux_dep_target requires test_ux_dep_source=on; "
                         "hint: cvar set test_ux_dep_source on";

    // Predicate false: warning fires + appears in output.
    {
        CHECK(src->value == "off");
        auto r = C.Execute("test_ux_dep_target 5");
        CHECK(r.ok);
        CHECK(tgt->value == "5");
        CHECK(r.output.find("[warn]") != std::string::npos);
        CHECK(r.output.find("test_ux_dep_source=on") != std::string::npos);
        CHECK(r.output.find("hint:") != std::string::npos);
    }

    // Make predicate true; the next set has no [warn] line.
    {
        CHECK(C.Execute("test_ux_dep_source on").ok);
        auto r = C.Execute("test_ux_dep_target 7");
        CHECK(r.ok);
        CHECK(tgt->value == "7");
        CHECK(r.output.find("[warn]") == std::string::npos);
    }
}

TEST_CASE("cvar dep warning: suppressed during cfg replay") {
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_ux_dep2_source", "off", "predicate source",
                   pt::console::CVAR_NONE);
    C.RegisterCVar("test_ux_dep2_target", "0", "predicate-gated target",
                   pt::console::CVAR_NONE);
    auto* src = C.FindCVar("test_ux_dep2_source");
    auto* tgt = C.FindCVar("test_ux_dep2_target");
    REQUIRE(src != nullptr);
    REQUIRE(tgt != nullptr);

    tgt->requires_predicate = []() {
        auto* s = pt::console::Console::Get().FindCVar("test_ux_dep2_source");
        return s != nullptr && s->value == "on";
    };
    tgt->requires_hint = "test_ux_dep2_target requires test_ux_dep2_source=on";

    // cfg-load-style suppression: same set, no [warn] in output.
    C.SetSuppressDepWarnings(true);
    {
        auto r = C.Execute("test_ux_dep2_target 11");
        CHECK(r.ok);
        CHECK(r.output.find("[warn]") == std::string::npos);
    }
    C.SetSuppressDepWarnings(false);
    // Once suppression lifts, the warning re-engages.
    {
        auto r = C.Execute("test_ux_dep2_target 13");
        CHECK(r.ok);
        CHECK(r.output.find("[warn]") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Issue #162: ResolveCommand -- smart prefix-resolution.
// ---------------------------------------------------------------------------

TEST_CASE("smart resolve: exact match wins immediately, no log line") {
    auto& C = pt::console::Console::Get();
    C.RegisterCVar("test_ux_sr_exact", "x", "smart-resolve exact",
                   pt::console::CVAR_NONE);

    auto r = C.ResolveCommand("test_ux_sr_exact");
    CHECK(r.canonical_name == "test_ux_sr_exact");
    CHECK(r.is_exact_match);
    CHECK(r.ambiguous_matches.empty());

    // Execute path: no `[console] resolved` log line for exact match.
    auto er = C.Execute("test_ux_sr_exact");
    CHECK(er.ok);
    CHECK(er.output.find("[console] resolved") == std::string::npos);
}

TEST_CASE("smart resolve: unique prefix returns canonical + log line in Execute") {
    auto& C = pt::console::Console::Get();
    // Single cvar in this namespace prefix -- prefix `test_ux_sr_uniq_p`
    // is a unique match for `test_ux_sr_uniq_payload`.
    C.RegisterCVar("test_ux_sr_uniq_payload", "v", "smart-resolve unique",
                   pt::console::CVAR_NONE);

    auto r = C.ResolveCommand("test_ux_sr_uniq_p");
    CHECK(r.canonical_name == "test_ux_sr_uniq_payload");
    CHECK_FALSE(r.is_exact_match);
    CHECK(r.ambiguous_matches.empty());

    // Execute auto-rewrites the prefix; the output gets the
    // `[console] resolved ... -> ... (top match)` log line followed
    // by the standard cvar-read line.
    auto er = C.Execute("test_ux_sr_uniq_p");
    CHECK(er.ok);
    CHECK(er.output.find("[console] resolved") != std::string::npos);
    CHECK(er.output.find("test_ux_sr_uniq_payload") != std::string::npos);
}

TEST_CASE("smart resolve: ambiguous prefix errors with candidate list") {
    auto& C = pt::console::Console::Get();
    // Two cvars sharing the same prefix -- the resolver must surface
    // both as ambiguous candidates rather than picking one.
    C.RegisterCVar("test_ux_sr_amb_alpha", "1", "amb a", pt::console::CVAR_NONE);
    C.RegisterCVar("test_ux_sr_amb_beta",  "2", "amb b", pt::console::CVAR_NONE);

    auto r = C.ResolveCommand("test_ux_sr_amb_");
    CHECK(r.canonical_name.empty());
    CHECK_FALSE(r.is_exact_match);
    CHECK(r.ambiguous_matches.size() >= 2);

    auto er = C.Execute("test_ux_sr_amb_");
    CHECK_FALSE(er.ok);
    CHECK(er.error.find("ambiguous prefix") != std::string::npos);
    CHECK(er.error.find("test_ux_sr_amb_alpha") != std::string::npos);
    CHECK(er.error.find("test_ux_sr_amb_beta")  != std::string::npos);
}

TEST_CASE("smart resolve: no match falls through to unknown-cvar error") {
    auto& C = pt::console::Console::Get();
    auto r = C.ResolveCommand("zzz_test_ux_sr_nomatch_xyz");
    CHECK(r.canonical_name.empty());
    CHECK_FALSE(r.is_exact_match);
    CHECK(r.ambiguous_matches.empty());

    auto er = C.Execute("zzz_test_ux_sr_nomatch_xyz");
    CHECK_FALSE(er.ok);
    CHECK(er.error.find("unknown command or cvar") != std::string::npos);
}

TEST_CASE("smart resolve: r_-prefix strip lets `deno` find r_denoiser") {
    auto& C = pt::console::Console::Get();
    // Production cvars are uniformly prefixed with `r_`. The
    // resolver also tries the prefix after stripping a leading
    // `r_` from canonical names so a user's `deno metalfx` finds
    // `r_denoiser`. The acceptance example in issue #162 demands
    // this behaviour explicitly.
    C.RegisterCVar("r_ux_sr_strip_target", "v", "strip-r_ target",
                   pt::console::CVAR_NONE);
    // Make sure no competing cvar starts with `ux_sr_strip` (cleanup
    // would also work; we just pick a unique enough body).
    auto r = C.ResolveCommand("ux_sr_strip");
    CHECK(r.canonical_name == "r_ux_sr_strip_target");
    CHECK_FALSE(r.is_exact_match);
}

TEST_CASE("smart resolve: disabled via r_console_smart_resolve = 0") {
    auto& C = pt::console::Console::Get();
    // Register the gating cvar at test-local default 1 so we can flip
    // it off, observe strict matching, then restore. (If the test
    // runs after the production registration -- it doesn't here,
    // since this is the cvar_ux unit test binary -- the registration
    // is a no-op and we still control the value.)
    C.RegisterCVar("r_console_smart_resolve", "1",
                   "console UX toggle", pt::console::CVAR_NONE);
    C.RegisterCVar("test_ux_sr_off_target", "v", "off-target",
                   pt::console::CVAR_NONE);

    // Sanity: with smart-resolve on, prefix lookup works.
    {
        auto er = C.Execute("test_ux_sr_off_t");
        CHECK(er.ok);
        CHECK(er.output.find("[console] resolved") != std::string::npos);
    }

    // Flip the cvar off and confirm prefix is now unknown.
    CHECK(C.Execute("r_console_smart_resolve 0").ok);
    {
        auto er = C.Execute("test_ux_sr_off_t");
        CHECK_FALSE(er.ok);
        CHECK(er.error.find("unknown command or cvar") != std::string::npos);
    }
    // Restore.
    CHECK(C.Execute("r_console_smart_resolve 1").ok);
}

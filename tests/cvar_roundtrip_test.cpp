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
// Test strategy notes:
//
//   - Console is a singleton; SUBCASE re-runs the TEST_CASE body each
//     time, so any "second time we hit RegisterCVar with the same
//     name" path through Console's de-dup logic could make a SUBCASE-
//     based test fragile if more tests are added that re-register the
//     same names. To eliminate that class of fragility we use TWO
//     measures: (1) every TEST_CASE registers its own uniquely-named
//     cvars (test_rt_file_*, test_rt_load_*, test_rt_default_*,
//     test_rt_nonarchive_*) so cross-TEST_CASE singleton state can't
//     bleed; (2) no SUBCASEs -- each test is one straight-line case
//     that does its own setup -> save -> assert -> cleanup.
//   - Each test_executable() is its own process under ctest, so global
//     singleton state across THIS file's tests doesn't bleed into
//     other test files even if they happened to share cvar names.

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
// file -- callers always REQUIRE(fs::exists(path)) first so the
// false-positive "test passed because slurp() returned empty + npos
// triggered" path can't happen.
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

// --- Test 1: SaveArchivedCvars output format ------------------------------
// Verifies the on-disk format SaveArchivedCvars emits. Covers all four
// value shapes (int, float, bareword, space-quoted) in one pass so the
// per-shape format rules stay traceable to one test.
TEST_CASE("cvar round-trip: save emits expected on-disk format") {
    auto& C = pt::console::Console::Get();

    // Unique names for this TEST_CASE (test_rt_file_*) so the
    // singleton-singleton state can't bleed into the "load restores
    // values" TEST_CASE below.
    C.RegisterCVar("test_rt_file_int",    "0",       "int test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_file_float",  "0.0",     "float test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_file_string", "default", "string test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_file_quoted", "default", "quoted test",
                   pt::console::CVAR_ARCHIVE);

    REQUIRE(C.Execute("test_rt_file_int 42").ok);
    REQUIRE(C.Execute("test_rt_file_float 3.14159").ok);
    REQUIRE(C.Execute("test_rt_file_string hello_world").ok);
    REQUIRE(C.Execute("test_rt_file_quoted \"a value with spaces\"").ok);

    const auto path = temp_cfg_path("rt_file");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n   != -1);     // -1 means file-open failure
    REQUIRE(n   >   0);     // we set 4 archive cvars, expect >=4 lines
    REQUIRE(fs::exists(path));

    const std::string body = slurp(path);
    CHECK(body.find("test_rt_file_int 42")            != std::string::npos);
    CHECK(body.find("test_rt_file_float 3.14159")     != std::string::npos);
    CHECK(body.find("test_rt_file_string hello_world")!= std::string::npos);
    // Quoted form: value contains spaces, must be wrapped in "".
    CHECK(body.find("test_rt_file_quoted \"a value with spaces\"")
          != std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

// --- Test 2: round-trip restores values ----------------------------------
// The actual "load reverses save" assertion. Save, mutate, replay the
// saved file via ExecuteScript, verify the values came back. Unique
// cvar names (test_rt_load_*) keep this test independent of Test 1's
// singleton state.
TEST_CASE("cvar round-trip: load reverses save after mid-flight mutation") {
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_load_int",    "0",       "int test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_load_float",  "0.0",     "float test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_load_string", "default", "string test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_load_quoted", "default", "quoted test",
                   pt::console::CVAR_ARCHIVE);

    // Set initial values that we'll save.
    REQUIRE(C.Execute("test_rt_load_int 42").ok);
    REQUIRE(C.Execute("test_rt_load_float 3.14159").ok);
    REQUIRE(C.Execute("test_rt_load_string hello_world").ok);
    REQUIRE(C.Execute("test_rt_load_quoted \"a value with spaces\"").ok);

    const auto path = temp_cfg_path("rt_load");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n   != -1);
    REQUIRE(n   >   0);
    REQUIRE(fs::exists(path));

    // Mutate to clearly-different values so the load has something to
    // actually restore. If the load path silently dropped a line, the
    // assertion below would still see the mutated value instead of
    // the saved one -- that's the failure mode this test is
    // specifically checking for.
    REQUIRE(C.Execute("test_rt_load_int 999").ok);
    REQUIRE(C.Execute("test_rt_load_float 99.0").ok);
    REQUIRE(C.Execute("test_rt_load_string mutated").ok);
    REQUIRE(C.Execute("test_rt_load_quoted mutated_quoted").ok);

    // Replay the saved snapshot. ExecuteScript handles multi-line
    // bodies with `// comments`, blank lines, and semicolon-separated
    // statements -- same as how Engine::Init replays demont.cfg on
    // launch.
    const std::string body = slurp(path);
    REQUIRE_FALSE(body.empty());
    auto r = C.ExecuteScript(body);
    CHECK(r.ok);

    auto* cv_int    = C.FindCVar("test_rt_load_int");
    auto* cv_float  = C.FindCVar("test_rt_load_float");
    auto* cv_str    = C.FindCVar("test_rt_load_string");
    auto* cv_quoted = C.FindCVar("test_rt_load_quoted");
    REQUIRE(cv_int    != nullptr);
    REQUIRE(cv_float  != nullptr);
    REQUIRE(cv_str    != nullptr);
    REQUIRE(cv_quoted != nullptr);

    // Values should match what we saved, not what we mutated to.
    CHECK(cv_int->GetInt()    == 42);
    CHECK(cv_float->GetFloat() == doctest::Approx(3.14159f));
    CHECK(cv_str->value     == "hello_world");
    CHECK(cv_quoted->value  == "a value with spaces");

    std::error_code ec;
    fs::remove(path, ec);
}

// --- Test 3: cvars at their default value are not serialized -------------
TEST_CASE("cvar round-trip: defaults are NOT serialized") {
    // SaveArchivedCvars deliberately skips cvars whose current value
    // equals their default (Console.cpp:518). This keeps demont.cfg
    // tight -- only changes from the default get persisted. Test the
    // skip logic: a freshly-registered cvar at its default value
    // should not appear in the output.
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_default_unset", "default_value", "unset test",
                   pt::console::CVAR_ARCHIVE);
    // Don't set it -- value == default.

    const auto path = temp_cfg_path("rt_default");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n != -1);        // -1 means file-open failed; without this
                             // gate a save failure would produce an
                             // empty file + the "find == npos" check
                             // below would falsely pass.
    REQUIRE(fs::exists(path));

    const std::string body = slurp(path);
    CHECK(body.find("test_rt_default_unset") == std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

// --- Test 4: cvars without CVAR_ARCHIVE are not serialized --------------
TEST_CASE("cvar round-trip: non-archive cvars are NOT serialized") {
    // CVAR_ARCHIVE is the gate for SaveArchivedCvars (Console.cpp:517).
    // Cvars without it -- READONLY engine state, internal dev knobs,
    // temporary state -- must not leak into demont.cfg even if mutated.
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_nonarchive", "0", "non-archive test",
                   pt::console::CVAR_NONE);
    REQUIRE(C.Execute("test_rt_nonarchive 7").ok);

    const auto path = temp_cfg_path("rt_nonarchive");
    const int  n    = C.SaveArchivedCvars(path.string());
    REQUIRE(n != -1);        // see Test 3 for rationale on this gate
    REQUIRE(fs::exists(path));

    const std::string body = slurp(path);
    CHECK(body.find("test_rt_nonarchive") == std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

// --- Test 5: allowed_values gate (r_software_blit_recreate shape) -------
// Pins the cvar shape used by PR-B (auto / prompt / warn, archived,
// validation by Console::Execute). Regression catch: if a future refactor
// stops enforcing allowed_values on Execute, r_software_blit_recreate=foo
// would silently overwrite the value and the next vulkan->software gdi
// switch would silently fall through to none of the three documented
// branches, leaving the engine in undefined behaviour with no operator
// signal.
TEST_CASE("cvar allowed_values: auto/prompt/warn gate accepts valid, rejects others") {
    auto& C = pt::console::Console::Get();

    // Mirror r_software_blit_recreate's registration shape, with a
    // unique test_rt_blit_recreate name so this TEST_CASE doesn't
    // collide with the production cvar (which is registered by
    // Engine.cpp's static-init when the engine is linked).
    C.RegisterCVar("test_rt_blit_recreate", "auto",
                   "test mirror of r_software_blit_recreate",
                   pt::console::CVAR_ARCHIVE);
    auto* v = C.FindCVar("test_rt_blit_recreate");
    REQUIRE(v != nullptr);
    v->allowed_values = {"auto", "prompt", "warn"};

    // Default value should be "auto".
    CHECK(v->value == "auto");

    // All three documented values must Execute cleanly.
    CHECK(C.Execute("test_rt_blit_recreate auto").ok);
    CHECK(v->value == "auto");
    CHECK(C.Execute("test_rt_blit_recreate prompt").ok);
    CHECK(v->value == "prompt");
    CHECK(C.Execute("test_rt_blit_recreate warn").ok);
    CHECK(v->value == "warn");

    // Unrecognised values must be rejected with a non-empty error
    // string. The cvar's stored value should NOT change.
    auto r_bad = C.Execute("test_rt_blit_recreate foobar");
    CHECK_FALSE(r_bad.ok);
    CHECK_FALSE(r_bad.error.empty());
    CHECK(v->value == "warn");  // unchanged from previous successful set

    // Mixed-case is also rejected: allowed_values uses exact string
    // match, so "Auto" doesn't slip through. (The engine's HWND-
    // recreate dispatch hits its `mode == "auto"` branch -- not
    // case-insensitive -- so case-folding here would create a silent
    // fall-through to the warn branch.)
    auto r_case = C.Execute("test_rt_blit_recreate Auto");
    CHECK_FALSE(r_case.ok);
    CHECK(v->value == "warn");
}

// --- Test 6: ExecuteScript treats ';' inside '#' / '//' comments as data --
// Regression for the smoke-exec fatal-init bug discovered 2026-05-18.
// ExecuteScript splits its body into statements at '\n' and ';', then
// hands each statement to Execute() which strips '#' / '//' comments.
// Previously the script-level split was naive: a fixture comment like
//   # (golden-hour, not yet twilight; civil twilight is sun 0 to -6)
// split into a comment-prefix line plus a bogus `civil twilight ...`
// line, and Execute() reported `unknown command or cvar: 'civil'`.
// The smoke-runner treats Execute errors as fatal -> engine init
// aborted -> no golden PNG -> CI failure. Same trap for `;` inside a
// `// ...` C++-style full-line comment and inside a quoted string.
//
// The fix lives in Console.cpp ScanScriptStatementEnd() which makes
// the script-level scanner comment-aware and quote-aware. This test
// pins the behaviour so the fixture-comment cleanup doesn't have to
// be repeated for every new fixture file we add.
TEST_CASE("cvar parser: ';' inside '#' comments is NOT a statement separator") {
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("test_rt_parser_aurora", "0", "aurora gate test",
                   pt::console::CVAR_ARCHIVE);

    // The exact comment shape that broke procedural_evening.cfg /
    // lunar_night.cfg / bsc_night_clouds.cfg in CI run 26037168862.
    // Inline '#' comment containing a ';' should be inert; the
    // subsequent r_aurora set on the next line should still apply.
    const std::string body =
        "# (golden-hour, not yet twilight; civil twilight is sun 0 to -6)\n"
        "test_rt_parser_aurora 1\n";
    auto r = C.ExecuteScript(body);
    CHECK(r.ok);                                            // no spurious unknown-cvar errors
    CHECK(r.error.find("civil")   == std::string::npos);    // 'civil' not parsed as a command
    CHECK(r.error.find("unknown") == std::string::npos);
    auto* v = C.FindCVar("test_rt_parser_aurora");
    REQUIRE(v != nullptr);
    CHECK(v->GetInt() == 1);                                // the real set after the comment took effect

    // Same with a '//' full-line comment containing ';'.
    C.Execute("test_rt_parser_aurora 0");
    REQUIRE(v->GetInt() == 0);
    const std::string body2 =
        "// note: tonemap; bloom; lens flare all off below\n"
        "test_rt_parser_aurora 1\n";
    auto r2 = C.ExecuteScript(body2);
    CHECK(r2.ok);
    CHECK(r2.error.find("unknown") == std::string::npos);
    CHECK(v->GetInt() == 1);

    // Inline `r_x 0 # comment with ;` form (the documented
    // shell-style trailing comment) must still drop the comment AND
    // not treat the ';' as a statement separator that would start a
    // new statement with leftover comment text.
    C.Execute("test_rt_parser_aurora 0");
    const std::string body3 = "test_rt_parser_aurora 1 # turn it on; final state\n";
    auto r3 = C.ExecuteScript(body3);
    CHECK(r3.ok);
    CHECK(r3.error.find("unknown") == std::string::npos);
    CHECK(v->GetInt() == 1);

    // Sanity: real `;` outside a comment STILL works as a separator
    // (the user-typed "set two cvars on one line" use case).
    C.RegisterCVar("test_rt_parser_pair_a", "0", "pair a", pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_parser_pair_b", "0", "pair b", pt::console::CVAR_ARCHIVE);
    auto r4 = C.ExecuteScript("test_rt_parser_pair_a 7; test_rt_parser_pair_b 9");
    CHECK(r4.ok);
    auto* a = C.FindCVar("test_rt_parser_pair_a");
    auto* b = C.FindCVar("test_rt_parser_pair_b");
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(a->GetInt() == 7);
    CHECK(b->GetInt() == 9);

    // Sanity: quoted `;` is data, not a separator.
    C.RegisterCVar("test_rt_parser_quoted", "default", "quoted ; test",
                   pt::console::CVAR_ARCHIVE);
    auto r5 = C.ExecuteScript("test_rt_parser_quoted \"a;b;c\"");
    CHECK(r5.ok);
    auto* q = C.FindCVar("test_rt_parser_quoted");
    REQUIRE(q != nullptr);
    CHECK(q->value == "a;b;c");

    // GPT-5.5 / Copilot cross-confirmed regression: a `//` appearing
    // MID-STATEMENT (unquoted) must be data, not a comment marker.
    // The script-level scanner used to swallow `//host/path; ...` as a
    // comment, breaking URL-shaped values and silently absorbing the
    // user's statement separator. Execute() only treats `//` as a
    // comment at statement start (after optional whitespace); the
    // scanner now mirrors that exactly.
    //
    // Use UNQUOTED URLs here -- the quoted form goes through the
    // `in_quote=true` branch and never reaches the `//` check, so it
    // wouldn't actually exercise the fix.
    C.RegisterCVar("test_rt_parser_url",  "default",  "url-shaped value test",
                   pt::console::CVAR_ARCHIVE);
    C.RegisterCVar("test_rt_parser_post", "0",        "post-url cvar test",
                   pt::console::CVAR_ARCHIVE);
    auto r6 = C.ExecuteScript(
        "test_rt_parser_url http://example.com/path; test_rt_parser_post 1");
    CHECK(r6.ok);
    auto* url  = C.FindCVar("test_rt_parser_url");
    auto* post = C.FindCVar("test_rt_parser_post");
    REQUIRE(url  != nullptr);
    REQUIRE(post != nullptr);
    // URL value preserved through the `://`: scanner did NOT treat
    // the inline `//` as a comment marker.
    CHECK(url->value  == "http://example.com/path");
    // Statement separator still split: post-cvar got set.
    CHECK(post->GetInt() == 1);

    // Full-line `//` comment with leading whitespace -- the `//` is
    // still recognised as comment-start because at_stmt_start is true
    // until non-whitespace is seen.
    C.Execute("test_rt_parser_post 0");
    REQUIRE(post->GetInt() == 0);
    auto r7 = C.ExecuteScript(
        "    // leading-whitespace comment with ; semicolon in it\n"
        "test_rt_parser_post 1\n");
    CHECK(r7.ok);
    CHECK(r7.error.find("unknown") == std::string::npos);
    CHECK(post->GetInt() == 1);
}

// --- Test 7: ResetAllCVarsToDefaults -------------------------------------
// Clean-room testing helper: wipe every cvar back to its registered
// default in one call, with a single undo entry so the whole reset can
// be rolled back. Pair with the `--no-cfg` CLI flag (skips demont.cfg
// and autoexec.cfg load) for tests that need a guaranteed-clean cvar
// state without manually `mv`-ing config files around.
TEST_CASE("cvar reset: ResetAllCVarsToDefaults restores every cvar in one shot") {
    auto& C = pt::console::Console::Get();

    // Three differently-shaped cvars: int with on_change, plain float,
    // and a string. Verifies on_change fires AND a non-defaulted cvar
    // gets counted AND a defaulted cvar contributes nothing.
    int on_change_fires = 0;
    auto* cv_int   = C.RegisterCVar("test_rt_reset_int",  "0",        "int + on_change",
                                    pt::console::CVAR_ARCHIVE,
                                    [&](const pt::console::CVar&) { ++on_change_fires; });
    auto* cv_float = C.RegisterCVar("test_rt_reset_float","3.14",     "float",
                                    pt::console::CVAR_ARCHIVE);
    auto* cv_str   = C.RegisterCVar("test_rt_reset_str",  "hello",    "string",
                                    pt::console::CVAR_ARCHIVE);
    REQUIRE(cv_int   != nullptr);
    REQUIRE(cv_float != nullptr);
    REQUIRE(cv_str   != nullptr);

    // Mutate two of three; leave the float at default.
    REQUIRE(C.Execute("test_rt_reset_int 42").ok);
    REQUIRE(C.Execute("test_rt_reset_str world").ok);
    REQUIRE(cv_int->GetInt() == 42);
    REQUIRE(cv_str->value == "world");
    REQUIRE(cv_float->GetFloat() == doctest::Approx(3.14f));
    on_change_fires = 0;  // discard the on_change from the set above

    // Reset. Expect: 2 cvars changed (int, str); float was already at
    // default and contributes 0. The on_change hook fires once for the
    // int (not for the float that didn't change, not for the str that
    // has no hook).
    std::size_t n = C.ResetAllCVarsToDefaults();
    CHECK(n >= 2);  // there ARE other archive cvars in this singleton
                    // so n may be larger if those got reset too; the
                    // strict lower bound is 2 (our two mutated cvars).
    CHECK(cv_int->GetInt()      == 0);            // default restored
    CHECK(cv_str->value         == "hello");      // default restored
    CHECK(cv_float->GetFloat()  == doctest::Approx(3.14f));  // unchanged
    CHECK(on_change_fires       == 1);            // int's hook fired once

    // Calling reset again is a no-op: every cvar is now at default.
    std::size_t n2 = C.ResetAllCVarsToDefaults();
    CHECK(n2 == 0);

    // Single undo entry covers the whole reset -- one Undo() call
    // restores both our cvars in one transaction.
    auto changes = C.Undo();
    CHECK(changes.size() >= 2);
    CHECK(cv_int->GetInt() == 42);                // pre-reset value
    CHECK(cv_str->value    == "world");           // pre-reset value
}

// --- Test 8: favourites ---------------------------------------------------
// Console::favorites_ + AddFavorite + RemoveFavorite + ClearFavorites +
// the f<N> magic-invocation path. Engine-side persistence (favorites.cfg)
// is not exercised here -- that's tested implicitly by the engine's
// LoadFavoritesFromDisk / SaveFavoritesToDisk on real runs.
TEST_CASE("favourites: add / list / fN-invoke / remove / clear round-trip") {
    auto& C = pt::console::Console::Get();
    // Start from a known state. Other TEST_CASEs may have left
    // favourites in the singleton.
    C.ClearFavorites();
    REQUIRE(C.FavoriteCount() == 0);

    // Register a target cvar we can set + check side-effects against.
    C.RegisterCVar("test_rt_fav_target", "0", "fav target", pt::console::CVAR_ARCHIVE);
    auto* v = C.FindCVar("test_rt_fav_target");
    REQUIRE(v != nullptr);

    // 1) AddFavorite stores the line.
    C.AddFavorite("test_rt_fav_target 7");
    CHECK(C.FavoriteCount() == 1);
    CHECK(C.Favorites()[0] == "test_rt_fav_target 7");

    // 2) Typing `f1` executes the saved line. The Execute() intercept
    //    runs BEFORE smart-resolve so it wins over any fuzzy match.
    auto r = C.Execute("f1");
    CHECK(r.ok);
    CHECK(v->GetInt() == 7);
    // Output should include the fav-dispatch log line.
    CHECK(r.output.find("[fav] f1 ->") != std::string::npos);

    // 3) Add a second favourite. Verify f1 and f2 dispatch independently.
    C.AddFavorite("test_rt_fav_target 13");
    CHECK(C.FavoriteCount() == 2);
    C.Execute("test_rt_fav_target 0");  // reset
    REQUIRE(v->GetInt() == 0);
    auto r2 = C.Execute("f2");
    CHECK(r2.ok);
    CHECK(v->GetInt() == 13);
    // f1 still works after f2.
    C.Execute("f1");
    CHECK(v->GetInt() == 7);

    // 4) Out-of-range fN falls through to "unknown command or cvar"
    //    (it's not in the favourites vector, so the magic-intercept
    //    bypasses it and the standard error path fires).
    auto r3 = C.Execute("f99");
    CHECK_FALSE(r3.ok);
    CHECK(r3.error.find("unknown") != std::string::npos);

    // 5) RemoveFavorite shifts subsequent indices down.
    C.RemoveFavorite(1);
    CHECK(C.FavoriteCount() == 1);
    CHECK(C.Favorites()[0] == "test_rt_fav_target 13");  // was f2 -> now f1
    C.Execute("test_rt_fav_target 0");
    C.Execute("f1");
    CHECK(v->GetInt() == 13);

    // 6) Whitespace-only or empty AddFavorite is a no-op.
    C.AddFavorite("");
    C.AddFavorite("   \t  ");
    CHECK(C.FavoriteCount() == 1);

    // 7) Trim leading/trailing whitespace on save.
    C.AddFavorite("  test_rt_fav_target 42  ");
    CHECK(C.Favorites().back() == "test_rt_fav_target 42");

    // 8) ClearFavorites wipes everything.
    C.ClearFavorites();
    CHECK(C.FavoriteCount() == 0);
    // After clear, f1 is unknown.
    auto r4 = C.Execute("f1");
    CHECK_FALSE(r4.ok);
}

// --- Test 9: favourites + recursion guard ---------------------------------
// A pathological case: someone saves "f1" itself as a favourite. The
// in_fav_dispatch_ guard prevents the dispatch from recursing forever.
TEST_CASE("favourites: self-referential favourite (f1 saved as `f1`) doesn't loop") {
    auto& C = pt::console::Console::Get();
    C.ClearFavorites();
    REQUIRE(C.FavoriteCount() == 0);
    C.AddFavorite("f1");  // saved line IS "f1"
    REQUIRE(C.FavoriteCount() == 1);

    // Execute f1. The outer call sees first-token "f1", index in range,
    // calls Execute("f1") with in_fav_dispatch_=true. The inner call
    // skips the fav-magic branch (guard is set) and falls through to
    // smart-resolve -> no match -> "unknown command or cvar: f1".
    // Result: error, NO infinite recursion.
    auto r = C.Execute("f1");
    CHECK_FALSE(r.ok);  // inner Execute reports unknown
    CHECK(r.error.find("unknown") != std::string::npos);

    C.ClearFavorites();
}

// --- Test 10: console input history round-trip ----------------------------
// Console::history_ + PushHistory + SetHistory + History +
// ClearHistory. The Engine-side persistence (console_history.txt via
// Load/SaveConsoleHistoryToDisk) is exercised implicitly by real demont
// runs; this test pins the in-memory Console API + the rules that the
// on-disk format depends on (trim, blank-line reject, dedup,
// kMaxHistoryDepth cap).
TEST_CASE("console history: push / list / set / clear round-trip") {
    auto& C = pt::console::Console::Get();
    // Start clean -- other TEST_CASEs may have pushed entries into the
    // singleton.
    C.ClearHistory();
    REQUIRE(C.HistoryCount() == 0);

    // 1) PushHistory appends.
    C.PushHistory("r_exposure 0.05");
    CHECK(C.HistoryCount() == 1);
    CHECK(C.History()[0] == "r_exposure 0.05");

    // 2) Push more, verify FIFO ordering (oldest at front, newest at back).
    C.PushHistory("phys_drop_sphere 1 2 3");
    C.PushHistory("r_spp 4");
    CHECK(C.HistoryCount() == 3);
    CHECK(C.History()[0] == "r_exposure 0.05");
    CHECK(C.History()[1] == "phys_drop_sphere 1 2 3");
    CHECK(C.History()[2] == "r_spp 4");

    // 3) Blank / whitespace-only pushes are no-ops. The persisted file
    //    ignores blank lines, and walking up-arrow onto a blank would
    //    surprise the user.
    C.PushHistory("");
    C.PushHistory("   \t  ");
    CHECK(C.HistoryCount() == 3);

    // 4) Push trims leading + trailing whitespace.
    C.PushHistory("  csg_dump  ");
    CHECK(C.HistoryCount() == 4);
    CHECK(C.History().back() == "csg_dump");

    // 5) Consecutive duplicates are coalesced (bash HISTCONTROL=ignoredups).
    C.PushHistory("csg_dump");
    CHECK(C.HistoryCount() == 4);  // unchanged
    // But a NON-consecutive repeat IS kept -- previous entry must
    // differ.
    C.PushHistory("r_spp 8");
    C.PushHistory("csg_dump");
    CHECK(C.HistoryCount() == 6);
    CHECK(C.History().back() == "csg_dump");

    // 6) SetHistory replaces the whole vector (mirrors what
    //    LoadConsoleHistoryFromDisk does at startup).
    std::vector<std::string> seeded{
        "fav r_exposure 0.05",
        "fav r_spp 16",
        "r_theme synthwave",
    };
    C.SetHistory(seeded);
    CHECK(C.HistoryCount() == 3);
    CHECK(C.History()[0] == "fav r_exposure 0.05");
    CHECK(C.History()[1] == "fav r_spp 16");
    CHECK(C.History()[2] == "r_theme synthwave");

    // 7) ClearHistory wipes everything.
    C.ClearHistory();
    CHECK(C.HistoryCount() == 0);
    CHECK(C.History().empty());
}

// --- Test 11: console history -- depth cap --------------------------------
// kMaxHistoryDepth caps the buffer at the most recent N entries. Both
// PushHistory (one-at-a-time) and SetHistory (bulk-load) must enforce
// the cap so a long-running session OR a hand-edited
// console_history.txt with too many lines can't unbounded-grow.
TEST_CASE("console history: depth cap enforced on push + bulk-load") {
    auto& C = pt::console::Console::Get();
    C.ClearHistory();

    const std::size_t kCap = pt::console::Console::kMaxHistoryDepth;
    REQUIRE(kCap > 0);

    // 1) Push (kCap + 50) UNIQUE entries one-by-one. The cap should
    //    trim the oldest 50, leaving the most recent kCap.
    for (std::size_t i = 0; i < kCap + 50; ++i) {
        C.PushHistory(std::string("cmd_") + std::to_string(i));
    }
    CHECK(C.HistoryCount() == kCap);
    // First retained entry is the (50)th push (cmd_50); last is
    // the (kCap+49)th push (cmd_<kCap+49>).
    CHECK(C.History().front() == std::string("cmd_") + std::to_string(50));
    CHECK(C.History().back()  == std::string("cmd_") + std::to_string(kCap + 49));

    // 2) SetHistory with an oversized vector trims to kCap from the
    //    front (oldest dropped). Mirrors a hand-edited
    //    console_history.txt with too many lines.
    std::vector<std::string> over;
    over.reserve(kCap + 25);
    for (std::size_t i = 0; i < kCap + 25; ++i) {
        over.push_back(std::string("seed_") + std::to_string(i));
    }
    C.SetHistory(over);
    CHECK(C.HistoryCount() == kCap);
    CHECK(C.History().front() == std::string("seed_") + std::to_string(25));
    CHECK(C.History().back()  == std::string("seed_") + std::to_string(kCap + 24));

    C.ClearHistory();
}

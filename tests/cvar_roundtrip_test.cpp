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

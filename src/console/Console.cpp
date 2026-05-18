// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Console.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <fstream>

namespace pt::console {

// ---------- CVar coercion --------------------------------------------------

int CVar::GetInt() const {
    int n = 0;
    auto [_, ec] = std::from_chars(value.data(),
                                   value.data() + value.size(), n);
    if (ec != std::errc()) {
        std::from_chars(default_value.data(),
                        default_value.data() + default_value.size(), n);
    }
    return n;
}

float CVar::GetFloat() const {
    char* end = nullptr;
    float f = std::strtof(value.c_str(), &end);
    if (end == value.c_str()) {
        f = std::strtof(default_value.c_str(), nullptr);
    }
    return f;
}

bool CVar::GetBool() const {
    if (value == "1" || value == "true" || value == "on" || value == "yes") return true;
    if (value == "0" || value == "false" || value == "off" || value == "no") return false;
    return GetInt() != 0;
}

// ---------- Tokenizer ------------------------------------------------------

std::vector<std::string_view> TokenizeLine(std::string_view line, std::string& storage) {
    storage.clear();
    storage.reserve(line.size());
    std::vector<std::pair<std::size_t, std::size_t>> spans;

    std::size_t i = 0;
    auto skip_ws = [&]() {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    };

    while (i < line.size()) {
        skip_ws();
        if (i >= line.size()) break;
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '/') break;

        std::size_t start = storage.size();
        if (line[i] == '"') {
            ++i;
            while (i < line.size() && line[i] != '"') {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    char c = line[i + 1];
                    switch (c) {
                        case 'n': storage.push_back('\n'); break;
                        case 't': storage.push_back('\t'); break;
                        case '"': storage.push_back('"');  break;
                        case '\\': storage.push_back('\\'); break;
                        default:  storage.push_back(c);    break;
                    }
                    i += 2;
                } else {
                    storage.push_back(line[i++]);
                }
            }
            if (i < line.size()) ++i;  // consume closing quote
        } else {
            while (i < line.size() && line[i] != ' ' && line[i] != '\t') {
                storage.push_back(line[i++]);
            }
        }
        spans.emplace_back(start, storage.size() - start);
    }

    std::vector<std::string_view> out;
    out.reserve(spans.size());
    for (auto [off, len] : spans) {
        out.emplace_back(storage.data() + off, len);
    }
    return out;
}

// ---------- Console singleton ---------------------------------------------

Console& Console::Get() {
    static Console instance;
    return instance;
}

CVar* Console::RegisterCVar(std::string name, std::string default_value,
                            std::string description, std::uint32_t flags,
                            std::function<void(const CVar&)> on_change) {
    auto [it, inserted] = cvars_.try_emplace(name);
    auto& v = it->second;
    if (inserted) {
        v.name          = std::move(name);
        v.value         = default_value;
        v.default_value = default_value;
        v.description   = std::move(description);
        v.flags         = flags;
        v.on_change     = std::move(on_change);
    }
    return &v;
}

Command* Console::RegisterCommand(std::string name, std::string description,
                                  CommandCallback callback) {
    auto [it, inserted] = commands_.try_emplace(name);
    auto& c = it->second;
    if (inserted) {
        c.name        = std::move(name);
        c.description = std::move(description);
        c.callback    = std::move(callback);
    }
    return &c;
}

CVar* Console::FindCVar(std::string_view name) {
    auto it = cvars_.find(name);
    return it == cvars_.end() ? nullptr : &it->second;
}

Command* Console::FindCommand(std::string_view name) {
    auto it = commands_.find(name);
    return it == commands_.end() ? nullptr : &it->second;
}

bool Console::SetCVarOverride(std::string_view name, std::string_view value) {
    auto* v = FindCVar(name);
    if (v == nullptr) return false;
    v->value.assign(value);
    if (v->on_change) v->on_change(*v);
    return true;
}

// ---------- Platform-visibility / per-value-gate helpers ------------------
// Defined here (before Execute / ResolveCommand) because both methods
// consult them. The free-function versions
// (CVarValueAllowedOnThisPlatform, CurrentPlatformName) are part of the
// public surface declared in Console.h; the anon-namespace helpers
// (CVarVisibleOnThisPlatform, AllowedValuesForCurrentPlatformCsv,
// ValueFlagsFor, PlatformsFromMask) are TU-local.

namespace {

// A cvar carrying a CVAR_PLATFORM_* bit that doesn't match the current
// build is hidden from listing + autocomplete. Registration / set /
// archive paths still work so demont.cfg sharing across hosts keeps
// round-tripping.
inline bool CVarVisibleOnThisPlatform(std::uint32_t flags) {
#if defined(__APPLE__)
    if ((flags & CVAR_PLATFORM_WIN) != 0) return false;
#elif defined(_WIN32)
    if ((flags & CVAR_PLATFORM_MAC) != 0) return false;
#else
    if ((flags & (CVAR_PLATFORM_MAC | CVAR_PLATFORM_WIN)) != 0) return false;
#endif
    return true;
}

// Build a CSV of allowed values that pass the per-value platform
// filter on this host. Used by the error message when the user picks
// a wrong-platform enum value, so the message lists only values that
// would actually be accepted.
std::string AllowedValuesForCurrentPlatformCsv(const CVar& v) {
    std::string out;
    for (std::size_t i = 0; i < v.allowed_values.size(); ++i) {
        std::uint32_t mask = (i < v.allowed_value_flags.size())
                                 ? v.allowed_value_flags[i]
                                 : 0u;
        if (!CVarValueAllowedOnThisPlatform(mask)) continue;
        if (!out.empty()) out += ", ";
        out += v.allowed_values[i];
    }
    return out;
}

// Find the CVAR_VALUE_* mask aligned with `value` inside the cvar's
// allowed set. Returns 0 if the value isn't in the set (caller still
// has to consult allowed_values).
std::uint32_t ValueFlagsFor(const CVar& v, std::string_view value) {
    for (std::size_t i = 0; i < v.allowed_values.size(); ++i) {
        if (v.allowed_values[i] != value) continue;
        return (i < v.allowed_value_flags.size())
                   ? v.allowed_value_flags[i]
                   : 0u;
    }
    return 0u;
}

// Translate a value-platform mask into a human-readable platform list
// ("macOS", "Windows/Linux", etc) for the platform-mismatch error.
std::string PlatformsFromMask(std::uint32_t mask) {
    std::string s;
    auto add = [&](const char* p) {
        if (!s.empty()) s += "/";
        s += p;
    };
    if (mask & CVAR_VALUE_MAC)   add("macOS");
    if (mask & CVAR_VALUE_WIN)   add("Windows");
    if (mask & CVAR_VALUE_LINUX) add("Linux");
    return s;
}

}  // namespace

bool CVarValueAllowedOnThisPlatform(std::uint32_t value_flags) {
    if (value_flags == 0u) return true;     // CVAR_VALUE_ANY
#if defined(__APPLE__)
    return (value_flags & CVAR_VALUE_MAC) != 0u;
#elif defined(_WIN32)
    return (value_flags & CVAR_VALUE_WIN) != 0u;
#else
    return (value_flags & CVAR_VALUE_LINUX) != 0u;
#endif
}

const char* CurrentPlatformName() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(_WIN32)
    return "Windows";
#else
    return "Linux";
#endif
}

Console::Resolution Console::ResolveCommand(std::string_view typed) {
    Resolution r;
    if (typed.empty()) return r;

    // 1. Exact match wins immediately, no resolution log needed.
    if (auto it = cvars_.find(typed); it != cvars_.end()) {
        r.canonical_name = it->second.name;
        r.is_exact_match = true;
        return r;
    }
    if (auto it = commands_.find(typed); it != commands_.end()) {
        r.canonical_name = it->second.name;
        r.is_exact_match = true;
        return r;
    }

    // 2. Gather every cvar/command whose canonical name starts with
    //    `typed`. For cvars conventionally prefixed with "r_" we also
    //    accept the post-"r_" body as a typed prefix so `deno` ->
    //    `r_denoiser` works per the issue #162 acceptance examples.
    //    This is a small one-step relaxation, not a fuzzy/substring
    //    search; the test in `cvar_ux_test.cpp` covers both the
    //    raw-prefix and the strip-r_ paths.
    //
    //    Cvars come first in the candidate list (tie-break: cvars
    //    preferred over commands, per issue #162 spec). The maps are
    //    std::map so iteration is alphabetical, satisfying the
    //    final deterministic tie-break.
    auto matches_prefix = [&](const std::string& name) {
        if (name.size() >= typed.size() &&
            name.compare(0, typed.size(), typed) == 0) {
            return true;
        }
        // Strip an "r_" prefix off the registered name and re-check.
        // Only fires for cvars; commands like `phys_drop_sphere`
        // already match via the raw-prefix branch.
        if (name.size() >= 2 && name[0] == 'r' && name[1] == '_') {
            std::string_view body(name.data() + 2, name.size() - 2);
            if (body.size() >= typed.size() &&
                body.compare(0, typed.size(), typed) == 0) {
                return true;
            }
        }
        return false;
    };

    std::vector<std::string> cvar_hits;
    std::vector<std::string> cmd_hits;
    for (const auto& [_, v] : cvars_) {
        if (!CVarVisibleOnThisPlatform(v.flags)) continue;
        if (matches_prefix(v.name)) cvar_hits.push_back(v.name);
    }
    for (const auto& [_, c] : commands_) {
        if (matches_prefix(c.name)) cmd_hits.push_back(c.name);
    }

    const std::size_t total = cvar_hits.size() + cmd_hits.size();
    if (total == 0) return r;       // no match -- caller falls back to "unknown" error.
    if (total == 1) {
        r.canonical_name = cvar_hits.empty() ? cmd_hits.front() : cvar_hits.front();
        return r;
    }
    // Multiple hits: ambiguous. Pack both lists into ambiguous_matches
    // (cvars first, alphabetical, then commands alphabetical) so the
    // caller can render the list with the canonical tie-break order
    // baked in.
    r.ambiguous_matches.reserve(total);
    for (auto& n : cvar_hits) r.ambiguous_matches.push_back(std::move(n));
    for (auto& n : cmd_hits)  r.ambiguous_matches.push_back(std::move(n));
    return r;
}

ExecuteResult Console::Execute(std::string_view line) {
    ExecuteResult result;

    // Strip leading whitespace and skip blank/comment lines. Accept
    // `//` (Quake-style) full-line and `#` (shell-style) anywhere on
    // the line: anything from a non-quoted `#` to end-of-line is
    // dropped. Lets users paste an annotated block like
    //     r_volumetric 0   # kill volumetric clouds
    //     r_show_stars 0   # disable BSC starmap
    // without splitting it into separate sends.
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
    }
    if (line.empty()) return result;
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/') return result;
    {
        // Inline `#` comment stripping. Track quote state so a `#`
        // inside a quoted string is treated as data, not a comment.
        // Also handle backslash escaping so `\"` doesn't flip the
        // quote state mid-string -- without this, an escaped quote
        // in a quoted token would silently end the quote and a
        // following `#` would chop the rest of the command off.
        bool in_quote = false;
        bool escape   = false;
        std::size_t cut = line.size();
        for (std::size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') { escape = true; continue; }
            if (c == '"')  { in_quote = !in_quote; continue; }
            if (c == '#' && !in_quote) { cut = i; break; }
        }
        if (cut < line.size()) line = line.substr(0, cut);
    }
    // Trim trailing whitespace including '\r' so CRLF-terminated input
    // (config files, pasted text) doesn't end up with the carriage
    // return swallowed into the last token, breaking command/cvar
    // lookup.
    while (!line.empty()
           && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
        line.remove_suffix(1);
    }
    if (line.empty()) return result;

    std::string storage;
    auto tokens = TokenizeLine(line, storage);
    if (tokens.empty()) return result;

    auto name = tokens[0];

    // Smart command resolution (issue #162). If the first token isn't
    // an exact match, run a prefix search across cvars + commands.
    // - Unique winner: rewrite `name` to the canonical and prepend an
    //   info line so the user sees what we picked.
    // - Ambiguous (>1 match): return an error listing candidates,
    //   capped at 8 with `... (N more)`, asking for more chars.
    // - Zero match: fall through to the existing "unknown command or
    //   cvar" error path.
    // Gated by the r_console_smart_resolve cvar so users who prefer
    // strict matching can turn it off (default 1 = on).
    bool smart_on = true;
    if (auto* sm = FindCVar("r_console_smart_resolve");
        sm != nullptr) {
        smart_on = sm->GetBool();
    }

    std::string resolution_log;
    std::string canonical_storage;
    if (smart_on) {
        Resolution r = ResolveCommand(name);
        if (!r.canonical_name.empty() && !r.is_exact_match) {
            // Rewrite the first token to the canonical name. The
            // tokens span points into `storage`; rather than mutate
            // storage in place (which would invalidate other tokens'
            // string_views), park the canonical name in
            // `canonical_storage` and rebind `name` to view it.
            canonical_storage = r.canonical_name;
            resolution_log = fmt::format(
                "[console] resolved `{}` -> `{}` (top match)\n",
                std::string(name), canonical_storage);
            name = std::string_view(canonical_storage);
        } else if (r.canonical_name.empty() && !r.ambiguous_matches.empty()) {
            // Ambiguous -- print at most 8 candidates and a count.
            constexpr std::size_t kCap = 8;
            std::string msg = fmt::format(
                "ambiguous prefix `{}`: ", std::string(name));
            const std::size_t n = r.ambiguous_matches.size();
            const std::size_t shown = std::min(n, kCap);
            for (std::size_t i = 0; i < shown; ++i) {
                if (i) msg += ", ";
                msg += r.ambiguous_matches[i];
            }
            if (n > kCap) {
                msg += fmt::format(", ... ({} more)", n - kCap);
            }
            msg += "\n        type more characters to disambiguate";
            result.ok = false;
            result.error = std::move(msg);
            return result;
        }
        // else: zero match -- fall through to the existing "unknown"
        // path below so error formatting matches PR #159 behaviour.
    }

    // Try command first.
    if (auto* cmd = FindCommand(name); cmd != nullptr) {
        Output out;
        std::span<const std::string_view> args(tokens.data() + 1, tokens.size() - 1);
        cmd->callback(args, out);
        result.output = resolution_log + out.Buffer();
        return result;
    }

    // Then cvar: with no argument we read; with one or more we set.
    if (auto* v = FindCVar(name); v != nullptr) {
        if (tokens.size() == 1) {
            result.output = resolution_log + fmt::format(
                "{} = \"{}\"  (default \"{}\")",
                v->name, v->value, v->default_value);
            return result;
        }
        if ((v->flags & CVAR_READONLY) != 0) {
            result.ok = false;
            result.error = fmt::format("cvar '{}' is read-only", v->name);
            return result;
        }
        if ((v->flags & CVAR_CHEAT) != 0) {
            auto* cheat = FindCVar("dev_cheats");
            if (cheat == nullptr || !cheat->GetBool()) {
                result.ok = false;
                result.error = fmt::format(
                    "cvar '{}' requires dev_cheats 1", v->name);
                return result;
            }
        }

        // Join remaining tokens (handles both `cvar one_value` and
        // `cvar "0.05 0.05 0.06"` since the tokenizer already strips quotes).
        std::string new_value;
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (i > 1) new_value.push_back(' ');
            new_value.append(tokens[i]);
        }

        // "0" as off-shorthand. When the user types a literal "0" on an
        // enum-style cvar (one with allowed_values) and the allowed set
        // contains exactly one of "off" / "none" / "disabled", rewrite
        // the incoming value to that canonical token so the standard
        // allowed_values gate accepts it. Only "0" gets this treatment
        // -- "1" / "true" / any other digit must still hit the normal
        // validation path so a bool-stored-as-int cvar like r_clouds
        // (allowed_values = {"0","1"}) keeps its existing behaviour.
        if (new_value == "0" && !v->allowed_values.empty()) {
            auto ieq = [](std::string_view a, std::string_view b) {
                if (a.size() != b.size()) return false;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    char ca = a[i], cb = b[i];
                    if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
                    if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
                    if (ca != cb) return false;
                }
                return true;
            };
            for (const auto& a : v->allowed_values) {
                if (ieq(a, "off") || ieq(a, "none") || ieq(a, "disabled")) {
                    new_value = a;
                    break;
                }
            }
        }

        // Enforce allowed_values when defined.
        if (!v->allowed_values.empty()) {
            bool ok2 = false;
            for (const auto& a : v->allowed_values) {
                if (a == new_value) { ok2 = true; break; }
            }
            if (!ok2) {
                std::string allowed;
                for (std::size_t i = 0; i < v->allowed_values.size(); ++i) {
                    if (i) allowed += '|';
                    allowed += v->allowed_values[i];
                }
                result.ok = false;
                result.error = fmt::format(
                    "{}: invalid value '{}' (expected one of: {})",
                    v->name, new_value, allowed);
                return result;
            }
        }

        // Per-value platform gate (issue #161). A cross-platform cvar
        // (e.g. r_denoiser) can carry a per-allowed-value platform
        // mask; picking a wrong-platform value at the console errors
        // with a platform-mismatch message that lists only the
        // available-on-this-platform values. NOTE: the value still
        // gets written so a cfg-load that's running through Execute
        // doesn't bounce a saved-on-other-host value back to default
        // -- the error path here only fires when allowed_value_flags
        // is populated AND the user-typed value is gated off, AND
        // we're inside an interactive Execute call. We could also
        // check that the value is allowed_values-valid first (which
        // it must be by this point), so we don't have to redo that.
        if (!v->allowed_value_flags.empty()) {
            std::uint32_t mask = ValueFlagsFor(*v, new_value);
            if (!CVarValueAllowedOnThisPlatform(mask)) {
                std::string platforms = PlatformsFromMask(mask);
                std::string available = AllowedValuesForCurrentPlatformCsv(*v);
                // Still write the value so a portable demont.cfg
                // round-trips cleanly: same model as PR #159's
                // CVAR_PLATFORM_* flag-bit cvars. The error is
                // returned so the interactive caller knows the
                // value won't take effect *on this host*.
                std::string old_value = v->value;
                v->value = new_value;
                // Don't fire on_change for an inactive value -- the
                // engine handler would just try to switch backends
                // we can't actually use.
                result.ok = false;
                result.error = fmt::format(
                    "{}={} is {}-only; not available on {}.\n"
                    "        Available on this platform: {}",
                    v->name, new_value, platforms,
                    CurrentPlatformName(), available);
                // result.output records that the value was still
                // written (cfg round-trip preserved) so the caller
                // / cfg loader sees the state mutation.
                result.output = resolution_log + fmt::format(
                    "{}: \"{}\" -> \"{}\" (inactive: platform-gated)",
                    v->name, old_value, v->value);
                return result;
            }
        }

        std::string old_value = v->value;
        v->value = std::move(new_value);
        if (v->on_change) v->on_change(*v);

        // Cross-cvar dependency warning (issue #161). Evaluated AFTER
        // the value is committed -- the predicate inspects current
        // engine state; if it's false, we emit a one-line `[warn]`
        // hint that includes the prerequisite and a fix suggestion.
        // Never blocks the set. This matters for cfg-load order: a
        // user's demont.cfg that sets the dependent cvar before the
        // dependency would otherwise produce a spurious warning at
        // every line, so we suppress warnings while replaying scripts
        // (the cfg loader uses ExecuteScript which sets a flag --
        // see ExecuteScript() below) and only warn on real
        // interactive sets.
        std::string warn_line;
        if (!dep_warn_suppressed_ &&
            v->requires_predicate && !v->requires_predicate()) {
            warn_line = fmt::format("[warn] {}", v->requires_hint);
        }

        result.output = resolution_log + fmt::format(
            "{}: \"{}\" -> \"{}\"", v->name, old_value, v->value);
        if (!warn_line.empty()) {
            result.output.push_back('\n');
            result.output.append(warn_line);
        }
        return result;
    }

    result.ok = false;
    result.error = fmt::format("unknown command or cvar: '{}'", name);
    return result;
}

namespace {

// Find the end of the next logical statement starting at body[i].
// Statements end at '\n' OR at ';' that is OUTSIDE of quoted strings
// AND OUTSIDE of '#' / '//' comments. The body slice [i, end) is then
// passed to Execute(), which handles in-line comment stripping itself.
//
// Keeps the user-typed `r_a 0; r_b 1` two-statement form working while
// preventing a stray ';' inside a `# ...` or `// ...` comment from
// erroneously splitting a single comment line into a comment plus a
// junk command. Without this, a fixture like
//     # (golden-hour, not yet twilight; civil twilight is sun 0 to -6)
// would split into a (correctly-stripped) `# (golden-hour, not yet
// twilight` plus a bogus ` civil twilight is sun 0 to -6)` which the
// engine reports as `unknown command or cvar: 'civil'` and the smoke
// runner treats as a fatal init error.
//
// Mirrors the quote/escape handling Execute() does so cfg files can
// quote a literal `;` inside a string token and have it survive the
// script-level split.
//
// `//` is recognised as a comment marker ONLY at statement start (after
// optional leading whitespace), matching Execute()'s full-line `//`
// rule at line ~327. Treating `//` as a comment anywhere outside quotes
// would mis-parse values like `r_url http://host/path; r_y 1` -- the
// scanner would swallow the `;` along with everything after `://` into
// a comment, breaking the user's statement separator. `#` remains
// inline because Execute() also strips inline `#`, so the two halves
// agree on what's a comment.
std::size_t ScanScriptStatementEnd(std::string_view body, std::size_t i) {
    bool in_quote      = false;
    bool escape        = false;
    bool in_comment    = false;
    bool at_stmt_start = true;  // true until we see any non-whitespace
    std::size_t end = i;
    while (end < body.size()) {
        char c = body[end];
        if (c == '\n') break;
        if (in_comment) { ++end; continue; }
        if (escape)     { escape = false;  ++end; at_stmt_start = false; continue; }
        if (c == '\\')  { escape = true;   ++end; at_stmt_start = false; continue; }
        if (c == '"')   { in_quote = !in_quote; ++end; at_stmt_start = false; continue; }
        if (!in_quote) {
            // '#' is an inline comment marker anywhere (matches Execute()).
            if (c == '#') { in_comment = true; ++end; continue; }
            // '//' is ONLY a comment marker at statement start (matches
            // Execute()'s `if (line.size() >= 2 && line[0] == '/' && line[1] == '/')`).
            // Mid-statement `//` is data -- preserves URL-shaped values.
            if (at_stmt_start && c == '/' && end + 1 < body.size() && body[end + 1] == '/') {
                in_comment = true; end += 2; continue;
            }
            if (c == ';') break;
        }
        if (c != ' ' && c != '\t') at_stmt_start = false;
        ++end;
    }
    return end;
}

}  // namespace

ExecuteResult Console::ExecuteScript(std::string_view body) {
    ExecuteResult agg;
    // Capture pre-transaction values of every cvar that the script
    // tries to set. Implemented by parsing the first token of each
    // statement; if that token names a known cvar, snapshot its
    // current value. Statements that target commands (no cvar match)
    // contribute nothing to the snapshot.
    CvarSnapshot pre;
    if (!in_undo_redo_) {
        std::size_t i = 0;
        while (i < body.size()) {
            std::size_t end = ScanScriptStatementEnd(body, i);
            auto line = body.substr(i, end - i);
            // Find first non-whitespace, then first whitespace (token end).
            std::size_t a = 0;
            while (a < line.size() && (line[a] == ' ' || line[a] == '\t')) ++a;
            std::size_t b = a;
            while (b < line.size() && line[b] != ' ' && line[b] != '\t') ++b;
            if (b > a) {
                std::string_view name = line.substr(a, b - a);
                auto it = cvars_.find(std::string(name));
                if (it != cvars_.end() && pre.find(it->first) == pre.end()) {
                    pre[it->first] = it->second.value;
                }
            }
            i = (end < body.size()) ? end + 1 : end;
        }
    }

    std::size_t i = 0;
    while (i < body.size()) {
        std::size_t end = ScanScriptStatementEnd(body, i);
        auto line = body.substr(i, end - i);
        auto r = Execute(line);
        if (!r.ok) {
            agg.ok = false;
            if (!agg.error.empty()) agg.error.push_back('\n');
            agg.error.append(r.error);
        }
        if (!r.output.empty()) {
            if (!agg.output.empty() && agg.output.back() != '\n') {
                agg.output.push_back('\n');
            }
            agg.output.append(r.output);
        }
        i = (end < body.size()) ? end + 1 : end;
    }

    // Build the actual diff: only push cvars whose value really
    // changed during this script. Skip the entry entirely if nothing
    // changed (typing a no-op or a query).
    if (!in_undo_redo_) {
        CvarSnapshot diff;
        for (auto& [name, old] : pre) {
            auto it = cvars_.find(name);
            if (it != cvars_.end() && it->second.value != old) {
                diff[name] = old;
            }
        }
        if (!diff.empty()) {
            undo_stack_.push_back(std::move(diff));
            if (undo_stack_.size() > kMaxHistory) undo_stack_.pop_front();
            // Any new edit invalidates the redo branch.
            redo_stack_.clear();
        }
    }
    return agg;
}

std::vector<Console::CvarChange> Console::Undo() {
    std::vector<CvarChange> changes;
    if (undo_stack_.empty()) return changes;
    in_undo_redo_ = true;
    CvarSnapshot snap = std::move(undo_stack_.back());
    undo_stack_.pop_back();
    // Capture the current state of these same cvars so Redo can
    // reapply the forward edit.
    CvarSnapshot fwd;
    for (auto& [name, old] : snap) {
        auto it = cvars_.find(name);
        if (it == cvars_.end()) continue;
        std::string current = it->second.value;
        fwd[name] = current;
        // Set value via Execute so any allowed_values check, on_change
        // hook, and cascading on_change side effects fire as if the
        // user typed the rollback.
        std::string line = name + " " + old;
        Execute(line);
        changes.push_back({name, std::move(current), old});
    }
    redo_stack_.push_back(std::move(fwd));
    if (redo_stack_.size() > kMaxHistory) redo_stack_.pop_front();
    in_undo_redo_ = false;
    return changes;
}

std::vector<Console::CvarChange> Console::Redo() {
    std::vector<CvarChange> changes;
    if (redo_stack_.empty()) return changes;
    in_undo_redo_ = true;
    CvarSnapshot snap = std::move(redo_stack_.back());
    redo_stack_.pop_back();
    CvarSnapshot back;
    for (auto& [name, val] : snap) {
        auto it = cvars_.find(name);
        if (it == cvars_.end()) continue;
        std::string current = it->second.value;
        back[name] = current;
        std::string line = name + " " + val;
        Execute(line);
        changes.push_back({name, std::move(current), val});
    }
    undo_stack_.push_back(std::move(back));
    if (undo_stack_.size() > kMaxHistory) undo_stack_.pop_front();
    in_undo_redo_ = false;
    return changes;
}

void Console::QueueExecute(std::string line, Responder responder) {
    std::lock_guard lock(queue_mutex_);
    queue_.push_back({std::move(line), std::move(responder)});
}

void Console::Drain() {
    std::deque<PendingExec> local;
    {
        std::lock_guard lock(queue_mutex_);
        std::swap(local, queue_);
    }
    for (auto& pe : local) {
        // Trim leading/trailing whitespace to detect lone `[` / `]`.
        std::string_view trimmed = pe.line;
        while (!trimmed.empty() &&
               (trimmed.front() == ' ' || trimmed.front() == '\t'  ||
                trimmed.front() == '\r' || trimmed.front() == '\n')) {
            trimmed.remove_prefix(1);
        }
        while (!trimmed.empty() &&
               (trimmed.back() == ' ' || trimmed.back() == '\t'  ||
                trimmed.back() == '\r' || trimmed.back() == '\n')) {
            trimmed.remove_suffix(1);
        }

        if (!batch_active_) {
            if (trimmed == "[") {
                batch_active_ = true;
                batch_buffer_.clear();
                ExecuteResult r;
                r.output = "batch: open  (send ']' on its own line to commit; "
                           "the whole bundle is one undo step)";
                if (pe.responder) pe.responder(r);
                continue;
            }
            // Normal path: ExecuteScript splits on '\n' / ';' so
            // multi-line paste in one shot still works.
            auto result = ExecuteScript(pe.line);
            if (pe.responder) pe.responder(result);
            continue;
        }

        // batch_active_ == true: collect or commit
        if (trimmed == "]") {
            batch_active_ = false;
            std::string body = std::move(batch_buffer_);
            batch_buffer_.clear();
            // Empty bundle: nothing to do. Otherwise run as one
            // ExecuteScript transaction so the existing snapshot
            // mechanism captures all touched cvars in one entry.
            ExecuteResult result;
            if (body.empty()) {
                result.output = "batch: empty (nothing committed)";
            } else {
                result = ExecuteScript(body);
                std::string header = "batch: committed (one undo step rolls back all)";
                if (result.output.empty()) {
                    result.output = std::move(header);
                } else {
                    result.output = header + "\n" + std::move(result.output);
                }
            }
            if (pe.responder) pe.responder(result);
        } else {
            // Cap the bundle size so a misbehaving / disconnected
            // client that opens '[' but never sends ']' can't grow
            // the buffer without bound -- the engine is shared
            // process-wide state and an OOM here would take
            // everything down.  1 MiB is far more than any sane
            // batched command list (cvar dumps + replay scripts in
            // this codebase top out at < 64 KiB) but tiny relative
            // to engine memory headroom, so it's safe to be
            // generous before auto-aborting.
            constexpr std::size_t kBatchMaxBytes = 1u * 1024u * 1024u;
            const std::size_t projected =
                batch_buffer_.size() + pe.line.size() + 1u;
            if (projected > kBatchMaxBytes) {
                batch_active_ = false;
                batch_buffer_.clear();
                ExecuteResult r;
                r.output = fmt::format(
                    "batch: aborted (would exceed {} byte cap; nothing committed). "
                    "Send '[' again to retry with a smaller bundle.",
                    kBatchMaxBytes);
                if (pe.responder) pe.responder(r);
                continue;
            }
            // Append the line to the batch buffer (preserve raw line,
            // not the trimmed view -- ExecuteScript will trim per-line).
            if (!batch_buffer_.empty()) batch_buffer_.push_back('\n');
            batch_buffer_.append(pe.line);
            ExecuteResult r;
            r.output = fmt::format("batch: queued  ({} byte{} buffered, send ']' to commit)",
                                   batch_buffer_.size(),
                                   batch_buffer_.size() == 1 ? "" : "s");
            if (pe.responder) pe.responder(r);
        }
    }
}

void Console::EnumerateCVars(std::string_view prefix,
                             const std::function<void(CVar&)>& visitor) {
    for (auto& [_, v] : cvars_) {
        if (!CVarVisibleOnThisPlatform(v.flags)) continue;
        if (prefix.empty() || v.name.starts_with(prefix)) visitor(v);
    }
}

void Console::EnumerateCommands(std::string_view prefix,
                                const std::function<void(Command&)>& visitor) {
    for (auto& [_, c] : commands_) {
        if (prefix.empty() || c.name.starts_with(prefix)) visitor(c);
    }
}

int Console::SaveArchivedCvars(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return -1;

    f << "// DeMonT Engine -- archived cvars (auto-generated on quit)\n";
    f << "// Hand edits get overwritten on the next clean exit.\n\n";

    int n = 0;
    for (auto& [name, cv] : cvars_) {
        if ((cv.flags & CVAR_ARCHIVE) == 0) continue;
        if (cv.value == cv.default_value)   continue;       // don't bloat with defaults
        const bool needs_quotes = (cv.value.find(' ') != std::string::npos);
        f << cv.name << ' ';
        if (needs_quotes) f << '"' << cv.value << '"';
        else              f << cv.value;
        f << '\n';
        ++n;
    }
    return n;
}

}  // namespace pt::console

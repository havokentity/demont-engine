// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Completion.h"

#include "Console.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

namespace pt::console {

namespace {

// ASCII-only lowercase compare. Cvar / command names in this engine
// are pure ASCII identifiers (alpha + digit + `_`) so we deliberately
// skip locale-aware std::tolower -- it'd be both slower per char and
// surprising at non-ASCII inputs.
inline char to_lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

}  // namespace

TokenInfo CurrentToken(std::string_view input, std::size_t cursor) {
    TokenInfo t;
    if (cursor > input.size()) cursor = input.size();
    // Treat both ' ' and '\t' as a token delimiter. Console::TokenizeLine
    // (the engine's actual parser) treats both as whitespace, and the
    // Win32 paste path preserves tabs, so completion has to classify
    // tokens the same way the executor will -- otherwise pasted input
    // containing a tab would get a wrong start/end and splice the
    // candidate at the wrong byte offset.
    auto is_ws = [](char c) { return c == ' ' || c == '\t'; };

    // Word boundaries around `cursor`: walk left until whitespace (or
    // SOL), walk right until whitespace (or EOL). Empty input or a
    // cursor on whitespace gives start == end == cursor.
    std::size_t s = cursor;
    while (s > 0 && !is_ws(input[s - 1])) --s;
    std::size_t e = cursor;
    while (e < input.size() && !is_ws(input[e])) ++e;
    t.start = s;
    t.end   = e;
    t.text  = std::string(input.substr(s, e - s));

    // First-token detection: scan from start of input through `s`;
    // collect the first non-whitespace run as first_tok. is_token0
    // == true iff the cursor sits AT or BEFORE the first non-
    // whitespace run -- i.e. the user is typing / browsing where
    // token 0 will land. A cursor in leading whitespace (s < t0_start)
    // is logically still "at the start of the command line" since
    // Console::Execute strips leading whitespace before tokenising,
    // so Tab / Ctrl+Space at that position should offer cvar/command
    // names rather than value candidates for some downstream first
    // token the user hasn't reached yet.
    std::size_t i = 0;
    while (i < input.size() && is_ws(input[i])) ++i;
    std::size_t t0_start = i;
    while (i < input.size() && !is_ws(input[i])) ++i;
    t.first_tok  = std::string(input.substr(t0_start, i - t0_start));
    t.is_token0  = (s <= t0_start);
    return t;
}

int ScoreMatch(std::string_view name, std::string_view query,
               std::vector<std::pair<std::size_t, std::size_t>>* out_spans) {
    if (out_spans) out_spans->clear();
    if (query.empty()) return 1;     // anything matches an empty query
    if (name.empty())  return 0;

    // round-to-nearest integer division: matches JS's
    // Math.round((numerator) / denominator). Integer floor would
    // produce different rankings for otherwise-close candidates
    // (e.g. lengths 7 vs 8 in the denominator), which would diverge
    // the C++ ordering from the web JS ordering.
    auto rdiv = [](std::size_t num, std::size_t den) -> int {
        if (den == 0) return 0;
        return static_cast<int>((num + den / 2) / den);
    };

    // PREFIX: name starts with query (case-insensitive).
    if (name.size() >= query.size()) {
        bool prefix = true;
        for (std::size_t i = 0; i < query.size(); ++i) {
            if (to_lower_ascii(name[i]) != to_lower_ascii(query[i])) {
                prefix = false;
                break;
            }
        }
        if (prefix) {
            const int tightness = rdiv(query.size() * 200, name.size());
            if (out_spans) out_spans->emplace_back(0, query.size());
            return 1000 + tightness;
        }
    }

    // SUBSTRING: case-insensitive find. Word-boundary bonus when the
    // match starts at position 0 or right after `_`.
    {
        std::size_t found = std::string::npos;
        for (std::size_t i = 0; i + query.size() <= name.size(); ++i) {
            bool eq = true;
            for (std::size_t j = 0; j < query.size(); ++j) {
                if (to_lower_ascii(name[i + j]) != to_lower_ascii(query[j])) {
                    eq = false;
                    break;
                }
            }
            if (eq) { found = i; break; }
        }
        if (found != std::string::npos) {
            const bool word_start = (found == 0) || name[found - 1] == '_';
            const int  tightness  = rdiv(query.size() * 100, name.size());
            const int  word_bonus = word_start ? 100 : 0;
            // The `- idx` term gives a small preference to matches
            // closer to the front of the name, breaking ties between
            // otherwise-identical substring hits.
            const int score = 500 + word_bonus + tightness
                            - static_cast<int>(found);
            if (out_spans) out_spans->emplace_back(found, found + query.size());
            return score;
        }
    }

    // FUZZY: chars of query appear in order in name, possibly with
    // gaps. Per-char score: word-start chars (pos 0 / after `_`) earn
    // 8, others 4; +4 streak bonus per consecutive match. Spans
    // collect contiguous matched-char runs for highlight rendering.
    {
        std::size_t qi = 0;
        int score = 0;
        long long last_match = -2;
        std::size_t run_start = std::string::npos;
        for (std::size_t i = 0; i < name.size() && qi < query.size(); ++i) {
            if (to_lower_ascii(name[i]) != to_lower_ascii(query[qi])) {
                if (run_start != std::string::npos && out_spans) {
                    out_spans->emplace_back(run_start, i);
                }
                run_start = std::string::npos;
                continue;
            }
            const bool word_start = (i == 0) || name[i - 1] == '_';
            score += word_start ? 8 : 4;
            if (static_cast<long long>(i) == last_match + 1) score += 4;
            last_match = static_cast<long long>(i);
            if (run_start == std::string::npos) run_start = i;
            ++qi;
        }
        if (qi < query.size()) {
            if (out_spans) out_spans->clear();
            return 0;
        }
        if (run_start != std::string::npos && out_spans) {
            out_spans->emplace_back(run_start, static_cast<std::size_t>(last_match) + 1);
        }
        const int density = rdiv(query.size() * 50, name.size());
        return 100 + score + density;
    }
}

std::vector<CompletionMatch> BuildCompletions(const TokenInfo& token,
                                              std::size_t max_results,
                                              std::size_t description_clip) {
    auto& C = pt::console::Console::Get();
    std::vector<CompletionMatch> pool;

    if (token.is_token0) {
        // Token 0: every cvar + every command.
        C.EnumerateCVars("", [&](pt::console::CVar& v) {
            CompletionMatch m;
            m.name        = v.name;
            m.kind        = CompletionKind::Cvar;
            m.value       = v.value;
            m.description = v.description;
            pool.push_back(std::move(m));
        });
        C.EnumerateCommands("", [&](pt::console::Command& c) {
            CompletionMatch m;
            m.name        = c.name;
            m.kind        = CompletionKind::Command;
            m.value       = c.default_args;
            m.description = c.description;
            pool.push_back(std::move(m));
        });
    } else if (token.first_tok == "toggle") {
        // `toggle` special-case: token 1 is itself a cvar name, and
        // only cvars with allowed_values are meaningful targets.
        C.EnumerateCVars("", [&](pt::console::CVar& v) {
            if (v.allowed_values.empty()) return;
            CompletionMatch m;
            m.name        = v.name;
            m.kind        = CompletionKind::Cvar;
            m.value       = v.value;
            m.description = v.description;
            pool.push_back(std::move(m));
        });
    } else if (token.first_tok == "exec") {
        // Filesystem completion for `exec <path>`. Lists the
        // directory referenced by the typed prefix and offers:
        //   - subdirectories (suffixed with '/'), and
        //   - .cfg files (the only file type `exec` consumes).
        // Path is resolved relative to the CWD demont was launched
        // from (matches Engine.cpp's exec_smoke fopen path). Empty
        // typed text lists the current directory.
        //
        // The pushed `name` is the FULL path-so-far (so TAB-commit
        // replaces the typed text wholesale and the user can keep
        // typing into the new prefix to drill down). The scorer at
        // the bottom of this function will rank by ScoreMatch against
        // the typed prefix -- PREFIX/SUBSTRING/FUZZY all work
        // naturally because the candidate name string contains the
        // typed prefix verbatim when it's a real path match.
        namespace fs = std::filesystem;
        const std::string& typed = token.text;
        std::string dir_str;
        std::string partial;
        if (typed.empty()) {
            dir_str = ".";
            partial = "";
        } else if (typed.back() == '/' || typed.back() == '\\') {
            dir_str = typed.substr(0, typed.size() - 1);
            if (dir_str.empty()) dir_str = "/";
            partial = "";
        } else {
            const auto slash = typed.find_last_of("/\\");
            if (slash == std::string::npos) {
                dir_str = ".";
                partial = typed;
            } else {
                dir_str = typed.substr(0, slash);
                if (dir_str.empty()) dir_str = "/";
                partial = typed.substr(slash + 1);
            }
        }
        std::error_code ec;
        fs::path dir_path(dir_str);
        if (fs::is_directory(dir_path, ec)) {
            for (const auto& entry : fs::directory_iterator(dir_path, ec)) {
                const std::string fname = entry.path().filename().string();
                if (fname.empty() || fname[0] == '.') continue;  // skip dotfiles
                // Prefix filter (case-insensitive) against `partial`.
                if (!partial.empty()) {
                    if (fname.size() < partial.size()) continue;
                    bool prefix_ok = true;
                    for (std::size_t i = 0; i < partial.size(); ++i) {
                        if (to_lower_ascii(fname[i]) != to_lower_ascii(partial[i])) {
                            prefix_ok = false;
                            break;
                        }
                    }
                    if (!prefix_ok) continue;
                }
                const bool is_dir = entry.is_directory(ec);
                if (!is_dir && entry.path().extension() != ".cfg") continue;
                // Reconstruct the full typed-path-replacement string.
                std::string commit;
                if (dir_str == ".") {
                    commit = fname;
                } else {
                    commit = dir_str;
                    if (commit.back() != '/' && commit.back() != '\\') commit += '/';
                    commit += fname;
                }
                if (is_dir) commit += '/';
                CompletionMatch m;
                m.name        = std::move(commit);
                m.kind        = CompletionKind::Value;
                m.value       = is_dir ? "(dir)" : "(cfg)";
                pool.push_back(std::move(m));
            }
        }
    } else {
        // Value position. Three sources, in priority order:
        //   1. CVar with allowed_values  -- list those, tag the
        //      current / default ones via `value`. Descriptions are
        //      LEFT EMPTY here because the cvar's help text would be
        //      identical on every row (same cvar) -- visual noise.
        //      Web/console.js takes the same approach.
        //   2. CVar without allowed_values -- offer [current, default]
        //      as one-shot suggestions. Description kept because each
        //      row has the same cvar help, but there are only 1-2
        //      rows so the repetition is bounded.
        //   3. Command with default_args -- show the default
        //      invocation as a single suggestion. Restores the prior
        //      ghost-era affordance for things like
        //      `screenshot demonte_screen.ppm`.
        // First try exact-match on the typed first token. If no exact
        // hit, fall through to smart-resolve (same ScoreMatch the
        // Execute path uses) and use the canonical cvar's value list.
        // Lets users type a shorthand like `den ` + TAB and get
        // `r_denoiser`'s allowed-value completions, instead of an
        // empty suggestion list. The resolution surfaces in the menu
        // header at the call site -- here we just pick the right
        // cvar's value vector.
        auto* cv = C.FindCVar(token.first_tok);
        if (cv == nullptr) {
            auto rr = C.ResolveCommand(token.first_tok);
            if (!rr.canonical_name.empty()) {
                cv = C.FindCVar(rr.canonical_name);
            }
        }
        if (cv != nullptr) {
            if (!cv->allowed_values.empty()) {
                // Filter out values whose per-value platform flag
                // (CVAR_VALUE_*) doesn't permit the current build's
                // host (issue #161 / PR #163). Without this, typing
                // `r_denoiser <TAB>` on Mac would offer `nrd` and the
                // OptiX family as completions even though setting any
                // of them would be rejected at Execute time with a
                // platform-mismatch error -- the user saw an
                // autocomplete option, picked it, and then got told
                // "no, not this platform". Mirrors the same per-value
                // filter that AllowedValuesForCurrentPlatformCsv()
                // uses at error-message time so the two surfaces
                // (suggest + reject) stay consistent.
                for (std::size_t i = 0; i < cv->allowed_values.size(); ++i) {
                    const std::uint32_t mask =
                        (i < cv->allowed_value_flags.size())
                            ? cv->allowed_value_flags[i]
                            : 0u;  // CVAR_VALUE_ANY: visible everywhere
                    if (!CVarValueAllowedOnThisPlatform(mask)) continue;
                    const auto& v = cv->allowed_values[i];
                    CompletionMatch m;
                    m.name = v;
                    m.kind = CompletionKind::Value;
                    if (v == cv->value)              m.value = "current";
                    else if (v == cv->default_value) m.value = "default";
                    // description left empty -- same text per row is noise.
                    pool.push_back(std::move(m));
                }
            } else {
                CompletionMatch cur;
                cur.name  = cv->value;
                cur.kind  = CompletionKind::Value;
                cur.value = "current";
                cur.description = cv->description;
                pool.push_back(std::move(cur));
                if (cv->default_value != cv->value) {
                    CompletionMatch dflt;
                    dflt.name  = cv->default_value;
                    dflt.kind  = CompletionKind::Value;
                    dflt.value = "default";
                    dflt.description = cv->description;
                    pool.push_back(std::move(dflt));
                }
            }
        } else if (auto* cmd = C.FindCommand(token.first_tok); cmd != nullptr) {
            if (!cmd->default_args.empty()) {
                CompletionMatch m;
                m.name        = cmd->default_args;
                m.kind        = CompletionKind::Value;
                m.value       = "default";
                m.description = cmd->description;
                pool.push_back(std::move(m));
            }
        }
    }

    if (pool.empty()) return {};

    // Score every candidate; keep score-bearing ones. Value-position
    // rows tagged "current" / "default" get a small score bonus so
    // they sort to the top of the popup when the user opens it at
    // `<cvar> ` (empty query, value position) -- this is the "show
    // the current value first" affordance. Bonuses are small enough
    // that any real prefix / substring / fuzzy match on a typed query
    // still beats them (prefix gives 1000+, substring 500+, fuzzy
    // 100+; +5 / +2 are noise at those magnitudes).
    std::vector<CompletionMatch> ranked;
    ranked.reserve(pool.size());
    for (auto& m : pool) {
        const int s = ScoreMatch(m.name, token.text, &m.spans);
        if (s <= 0) continue;
        m.score = s;
        if      (m.value == "current") m.score += 5;
        else if (m.value == "default") m.score += 2;
        ranked.push_back(std::move(m));
    }
    if (ranked.empty()) return {};

    std::sort(ranked.begin(), ranked.end(),
              [](const CompletionMatch& a, const CompletionMatch& b) {
                  if (a.score != b.score) return a.score > b.score;
                  if (a.name.size() != b.name.size()) return a.name.size() < b.name.size();
                  return a.name < b.name;
              });

    if (max_results > 0 && ranked.size() > max_results) {
        ranked.resize(max_results);
    }
    if (description_clip > 0) {
        for (auto& m : ranked) {
            // UTF-8 safe -- some cvar descriptions contain non-ASCII
            // (e.g. "≈" in Engine.cpp), and std::string::resize at a
            // raw byte index can split a multibyte codepoint, leaving
            // an invalid UTF-8 sequence that downstream renderers
            // silently drop.
            Utf8SafeTruncate(m.description, description_clip);
        }
    }
    return ranked;
}

}  // namespace pt::console

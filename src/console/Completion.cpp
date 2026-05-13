// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Completion.h"

#include "Console.h"

#include <algorithm>
#include <cctype>

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
    // Word boundaries around `cursor`: walk left until a space (or
    // SOL), walk right until a space (or EOL). Empty input or a
    // cursor on a space gives start == end == cursor.
    std::size_t s = cursor;
    while (s > 0 && input[s - 1] != ' ') --s;
    std::size_t e = cursor;
    while (e < input.size() && input[e] != ' ') ++e;
    t.start = s;
    t.end   = e;
    t.text  = std::string(input.substr(s, e - s));

    // First-token detection: scan from start of input through `s`;
    // collect the first non-space run as first_tok. is_token0 == true
    // iff s coincides with the start of that first run.
    std::size_t i = 0;
    while (i < input.size() && input[i] == ' ') ++i;
    std::size_t t0_start = i;
    while (i < input.size() && input[i] != ' ') ++i;
    t.first_tok  = std::string(input.substr(t0_start, i - t0_start));
    t.is_token0  = (s == t0_start);
    return t;
}

int ScoreMatch(std::string_view name, std::string_view query,
               std::vector<std::pair<std::size_t, std::size_t>>* out_spans) {
    if (out_spans) out_spans->clear();
    if (query.empty()) return 1;     // anything matches an empty query
    if (name.empty())  return 0;

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
            const int tightness = static_cast<int>(
                (query.size() * 200) / name.size());
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
            const int  tightness  = static_cast<int>(
                (query.size() * 100) / name.size());
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
        const int density = static_cast<int>((query.size() * 50) / name.size());
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
    } else {
        // Value position. Pull from the named cvar's allowed_values,
        // or fall back to [current, default] for free-form cvars.
        auto* cv = C.FindCVar(token.first_tok);
        if (cv != nullptr) {
            if (!cv->allowed_values.empty()) {
                for (const auto& v : cv->allowed_values) {
                    CompletionMatch m;
                    m.name = v;
                    m.kind = CompletionKind::Value;
                    if (v == cv->value)            m.value = "current";
                    else if (v == cv->default_value) m.value = "default";
                    m.description = cv->description;
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
            if (m.description.size() > description_clip) {
                m.description.resize(description_clip);
            }
        }
    }
    return ranked;
}

}  // namespace pt::console

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

    // Try command first.
    if (auto* cmd = FindCommand(name); cmd != nullptr) {
        Output out;
        std::span<const std::string_view> args(tokens.data() + 1, tokens.size() - 1);
        cmd->callback(args, out);
        result.output = out.Buffer();
        return result;
    }

    // Then cvar: with no argument we read; with one or more we set.
    if (auto* v = FindCVar(name); v != nullptr) {
        if (tokens.size() == 1) {
            result.output = fmt::format("{} = \"{}\"  (default \"{}\")",
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

        std::string old_value = v->value;
        v->value = std::move(new_value);
        if (v->on_change) v->on_change(*v);
        result.output = fmt::format("{}: \"{}\" -> \"{}\"",
                                    v->name, old_value, v->value);
        return result;
    }

    result.ok = false;
    result.error = fmt::format("unknown command or cvar: '{}'", name);
    return result;
}

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
            std::size_t end = i;
            while (end < body.size() && body[end] != '\n' && body[end] != ';') ++end;
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
        std::size_t end = i;
        while (end < body.size() && body[end] != '\n' && body[end] != ';') ++end;
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

}  // namespace

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

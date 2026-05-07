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

    // Strip leading whitespace and skip blank/comment lines. We accept
    // both `//` (Quake-style) and `#` (shell-style) so users can paste
    // mixed-source snippets without stripping comments first.
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
    }
    if (line.empty()) return result;
    if (line[0] == '#') return result;
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/') return result;

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
    return agg;
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
        auto result = Execute(pe.line);
        if (pe.responder) pe.responder(result);
    }
}

void Console::EnumerateCVars(std::string_view prefix,
                             const std::function<void(CVar&)>& visitor) {
    for (auto& [_, v] : cvars_) {
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

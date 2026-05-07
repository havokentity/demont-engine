// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <deque>

#include <fmt/format.h>

namespace pt::console {

enum CVarFlag : std::uint32_t {
    CVAR_NONE      = 0,
    CVAR_ARCHIVE   = 1u << 0,  // persist to config.cfg
    CVAR_READONLY  = 1u << 1,  // engine-set, user cannot mutate
    CVAR_CHEAT     = 1u << 2,  // requires dev_cheats 1
};

struct CVar {
    std::string name;
    std::string value;
    std::string default_value;
    std::string description;
    std::uint32_t flags = 0;

    // Optional set of accepted values.  When non-empty, Console::Execute
    // rejects writes whose value isn't in the set (free-form cvars like
    // r_clear_color leave it empty).  Also drives tab-completion of the
    // value position in both UIs.
    std::vector<std::string> allowed_values;

    // Optional numeric range. When slider_max > slider_min, the web UI
    // renders a draggable range slider for the cvar (with a numeric
    // readout). slider_step controls the granularity (defaults to a
    // sensible value if 0). Leave both at 0 for cvars that aren't
    // numeric.
    float slider_min  = 0.0f;
    float slider_max  = 0.0f;
    float slider_step = 0.0f;

    std::function<void(const CVar&)> on_change;

    // Coerced accessors.  All values are stored as strings; these parse on
    // demand and gracefully return defaults on parse failure.
    int   GetInt()   const;
    float GetFloat() const;
    bool  GetBool()  const;
};

// Output sink passed to command callbacks.  The captured string is what
// gets returned to the caller of Console::Execute (and ultimately to the
// remote client over WS / TCP).
class Output {
public:
    void Print(std::string_view s) { buf_.append(s.data(), s.size()); }
    void PrintLine(std::string_view s) { Print(s); buf_.push_back('\n'); }

    template <typename... Args>
    void Format(fmt::format_string<Args...> f, Args&&... args) {
        Print(fmt::format(f, std::forward<Args>(args)...));
    }
    template <typename... Args>
    void FormatLine(fmt::format_string<Args...> f, Args&&... args) {
        PrintLine(fmt::format(f, std::forward<Args>(args)...));
    }

    const std::string& Buffer() const { return buf_; }

private:
    std::string buf_;
};

using CommandCallback =
    std::function<void(std::span<const std::string_view> args, Output& out)>;

struct Command {
    std::string name;
    std::string description;
    CommandCallback callback;
};

struct ExecuteResult {
    bool ok = true;
    std::string output;
    std::string error;
};

// Response handler invoked on the main thread after Drain() executes a
// queued command.  Implementations forward the result back over the wire.
using Responder = std::function<void(const ExecuteResult&)>;

class Console {
public:
    static Console& Get();

    CVar* RegisterCVar(std::string name, std::string default_value,
                       std::string description, std::uint32_t flags = 0,
                       std::function<void(const CVar&)> on_change = {});
    Command* RegisterCommand(std::string name, std::string description,
                             CommandCallback callback);

    CVar*    FindCVar(std::string_view name);
    Command* FindCommand(std::string_view name);

    // Force a CVar value past READONLY (engine use only).
    bool SetCVarOverride(std::string_view name, std::string_view value);

    // Synchronously tokenize and dispatch a single console line on the
    // calling thread.
    ExecuteResult Execute(std::string_view line);

    // Multi-line input (newline OR ';' separated); each statement runs in
    // turn, output concatenated.  Honors `//` line comments.
    ExecuteResult ExecuteScript(std::string_view body);

    // Thread-safe enqueue from the network threads.  The main thread drains
    // these once per Engine::Tick.
    void QueueExecute(std::string line, Responder responder);
    void Drain();

    // Iteration (sorted by name).  Optional prefix filter for tab-completion.
    void EnumerateCVars(std::string_view prefix,
                        const std::function<void(CVar&)>& visitor);
    void EnumerateCommands(std::string_view prefix,
                           const std::function<void(Command&)>& visitor);

    // Persistence (P11). Writes every CVAR_ARCHIVE cvar whose current
    // value differs from its default as `<name> <value>` lines to `path`.
    // Returns the number of lines written, or -1 on file error. Quoting
    // for values that contain spaces is wrapped in double quotes.
    int SaveArchivedCvars(const std::string& path);

    // Cvar undo / redo. Each ExecuteScript transaction captures the
    // pre-values of every cvar it touched and pushes them as one
    // entry on the undo stack. Undo() pops the top and restores;
    // Redo() reapplies. Stack capped at kMaxHistory entries.
    // Returns one entry per cvar reverted in this transaction
    // (empty if the stack was empty). Multi-cvar transactions
    // (semicolon-bundle) come back as a single Undo() call with
    // multiple entries -- caller can format each one.
    struct CvarChange {
        std::string name;
        std::string from;   // value at undo time (the rolled-back-from value)
        std::string to;     // value after undo (the restored value)
    };
    static constexpr std::size_t kMaxHistory = 50;
    std::vector<CvarChange> Undo();
    std::vector<CvarChange> Redo();
    std::size_t UndoDepth() const { return undo_stack_.size(); }
    std::size_t RedoDepth() const { return redo_stack_.size(); }

private:
    Console() = default;

    std::map<std::string, CVar, std::less<>>    cvars_;
    std::map<std::string, Command, std::less<>> commands_;

    struct PendingExec {
        std::string line;
        Responder   responder;
    };
    std::mutex                queue_mutex_;
    std::deque<PendingExec>   queue_;

    // Single transaction snapshot: name -> pre-transaction value.
    using CvarSnapshot = std::map<std::string, std::string>;
    std::deque<CvarSnapshot>  undo_stack_;
    std::deque<CvarSnapshot>  redo_stack_;
    bool                      in_undo_redo_ = false;   // suppress nested capture

    // Bracket-batch mode (`[` ... `]`). When the user sends a line
    // containing only `[`, subsequent lines are buffered (not
    // executed) until a line containing only `]` arrives, at which
    // point the entire buffer is run through ExecuteScript as one
    // transaction (so undo reverts the whole bundle in one step).
    // Single-threaded: only Drain() touches these.
    std::string               batch_buffer_;
    bool                      batch_active_ = false;
};

// Tokenize a single console line.  Quote-aware ("a b" stays one token).
// Stops at newline; '//' begins a comment for the rest of the line.
std::vector<std::string_view> TokenizeLine(std::string_view line,
                                           std::string& storage);

}  // namespace pt::console

// Lightweight registration macros.  Variables receive an opaque pointer to
// the registered object so that subsystem code can call back into them
// (e.g. read GetInt() at runtime).  Static-init order is safe: Console::Get
// uses Construct-On-First-Use.
// Use C++20's __VA_OPT__ instead of GCC's `, ##__VA_ARGS__` extension --
// both elide the trailing comma when the variadic pack is empty, but
// __VA_OPT__ is standard so Apple Clang doesn't warn -Wgnu-zero-variadic-
// macro-arguments on every PT_CVAR call (~50 warnings before this).
#define PT_CVAR(varname, default_value, description, ...)                     \
    static ::pt::console::CVar* varname =                                     \
        ::pt::console::Console::Get().RegisterCVar(                           \
            #varname, default_value, description __VA_OPT__(,) __VA_ARGS__)

#define PT_COMMAND(varname, description, lambda)                              \
    static ::pt::console::Command* varname =                                  \
        ::pt::console::Console::Get().RegisterCommand(                        \
            #varname, description, lambda)

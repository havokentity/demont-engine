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
    // Platform-availability hints. The cvar is still registered on
    // every platform (so a shared demont.cfg round-trips cleanly across
    // hosts) but is filtered out of listing / autocomplete pools on the
    // wrong platform. Default of zero on both bits = "available
    // everywhere". Mutually exclusive in practice; setting both is
    // legal but masks the cvar everywhere it isn't built for.
    CVAR_PLATFORM_MAC = 1u << 3,
    CVAR_PLATFORM_WIN = 1u << 4,
};

// Per-allowed-value platform tags (issue #161). Lets a cross-platform
// cvar like r_denoiser carry a per-value platform mask: e.g.
// "metalfx" is Mac-only, "nrd_relax" is Win/Linux-only, but
// "svgf_atrous" is available everywhere. At cvar-set time the gate
// produces a platform-mismatch error listing only the
// available-on-this-platform values; the cfg-load path keeps writing
// the value through anyway (warn-but-no-op) so a shared demont.cfg
// round-trips across hosts. Mirror of CVAR_PLATFORM_* but per-enum-
// value.
enum CVarValueFlag : std::uint32_t {
    CVAR_VALUE_ANY   = 0,           // available on every platform (default)
    CVAR_VALUE_MAC   = 1u << 0,
    CVAR_VALUE_WIN   = 1u << 1,
    CVAR_VALUE_LINUX = 1u << 2,
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

    // Optional per-allowed-value platform mask (issue #161). Aligned
    // 1:1 with `allowed_values` -- a 0 entry (CVAR_VALUE_ANY) means
    // the value works everywhere; non-zero bits gate the value to
    // specific platforms. Either left empty (every value is universal)
    // or sized == allowed_values.size(). When sized, the cvar set
    // path emits a platform-mismatch error AT CONSOLE-SET TIME, but
    // the registration / cfg-load path still accepts the write (we
    // want shared demont.cfg files to round-trip without rejection).
    std::vector<std::uint32_t> allowed_value_flags;

    // Optional cross-cvar dependency predicate (issue #161). Evaluated
    // *after* a successful set to print an informational warning when
    // the cvar's effect is gated on another cvar's state. The lambda
    // returns true iff the prerequisite holds; on false, the engine
    // prints `requires_hint` as a one-line `[warn]` so the user sees
    // *why* their value isn't taking effect. Never blocks the set --
    // a cfg-load script may set the dependency cvar second, so the
    // warning is purely informational.
    std::function<bool()> requires_predicate;
    std::string           requires_hint;

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

    // Optional default argument string shown as a ghost-suggestion when
    // the user has typed `<command> ` and is at the value position.
    // Free-form -- the registrant picks something representative of a
    // typical invocation (e.g. screenshot's "demonte_screen.ppm").
    // Empty means the command has no value-position ghost.
    std::string default_args;
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

    // Cross-cvar dependency warnings (issue #161) are suppressed when
    // a cfg replay is in progress -- the dependency cvar might be set
    // on a later line, so warning on every entry would be noisy. Set
    // to true around the cfg-load ExecuteScript call, then back to
    // false. No-op otherwise (Execute always honours the predicate).
    void SetSuppressDepWarnings(bool suppress) { dep_warn_suppressed_ = suppress; }
    bool DepWarningsSuppressed() const { return dep_warn_suppressed_; }

    // Smart-resolve a typed first-token to its canonical
    // command/cvar name (issue #162). When the user types
    // `deno metalfx`, this finds `r_denoiser` as the unique prefix
    // match and lets Execute() dispatch the canonical name with the
    // remaining args. Tie-breaking: exact match > unique prefix >
    // cvars over commands > alphabetical. Honors the per-platform
    // visibility filter (wrong-platform cvars don't compete in the
    // candidate pool). Gated by the r_console_smart_resolve cvar
    // (defaults on; off = strict exact-match like before).
    struct Resolution {
        std::string canonical_name;             // non-empty on success
        std::vector<std::string> ambiguous_matches;  // populated when no winner
        bool is_exact_match = false;
    };
    Resolution ResolveCommand(std::string_view typed);

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

    // Suppress cross-cvar dependency warnings while a cfg replay is
    // running -- the dependency cvar may be set on a later line.
    // Toggled by the engine around the cfg-load ExecuteScript call.
    bool                      dep_warn_suppressed_ = false;
};

// Tokenize a single console line.  Quote-aware ("a b" stays one token).
// Stops at newline; '//' begins a comment for the rest of the line.
std::vector<std::string_view> TokenizeLine(std::string_view line,
                                           std::string& storage);

// True iff the given CVAR_VALUE_* mask permits the value on the
// current build's host platform. A mask of 0 (CVAR_VALUE_ANY) always
// returns true. Used by the cvar-set gate to print a platform-
// mismatch error when the user picks a wrong-platform enum value.
bool CVarValueAllowedOnThisPlatform(std::uint32_t value_flags);

// Per-platform tag for the host this build is running on. Used in
// error messages so the user knows *which* platform is blocking
// their value (e.g. "metalfx is macOS-only; not available on
// Windows").
const char* CurrentPlatformName();

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

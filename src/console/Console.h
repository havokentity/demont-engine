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
#define PT_CVAR(varname, default_value, description, ...)                     \
    static ::pt::console::CVar* varname =                                     \
        ::pt::console::Console::Get().RegisterCVar(                           \
            #varname, default_value, description, ##__VA_ARGS__)

#define PT_COMMAND(varname, description, lambda)                              \
    static ::pt::console::Command* varname =                                  \
        ::pt::console::Console::Get().RegisterCommand(                        \
            #varname, description, lambda)

#include "Log.h"

#include <fmt/format.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>

namespace pt::log {

namespace {

constexpr std::size_t kMaxSinks = 4;
std::array<Sink, kMaxSinks> g_sinks{};
std::size_t g_sink_count = 0;
std::mutex  g_sink_mutex;

// Cache the tty check; stderr won't change handle types mid-run.
bool TtyEnabled() {
    static const bool v = ::isatty(fileno(stderr)) != 0;
    return v;
}

// 24-bit ANSI palette matching the rest of the project's UI.
constexpr const char* kReset      = "\033[0m";
constexpr const char* kDim        = "\033[2;38;2;111;120;146m";   // dim grey for ts
constexpr const char* kCyan       = "\033[38;2;0;240;255m";       // INFO tag
constexpr const char* kAmber      = "\033[38;2;255;200;58m";      // WARN tag/body
constexpr const char* kAmberBold  = "\033[1;38;2;255;200;58m";
constexpr const char* kRed        = "\033[38;2;255;74;94m";       // ERROR tag/body
constexpr const char* kRedBold    = "\033[1;38;2;255;74;94m";
constexpr const char* kFg         = "\033[38;2;231;234;242m";     // INFO body

}  // namespace

void AddSink(Sink sink) {
    std::lock_guard lock(g_sink_mutex);
    if (g_sink_count < kMaxSinks) {
        g_sinks[g_sink_count++] = sink;
    }
}

void RemoveAllSinks() {
    std::lock_guard lock(g_sink_mutex);
    g_sink_count = 0;
}

void Emit(Level level, fmt::string_view fmt_str, fmt::format_args args) {
    const char* tag      = "INFO ";
    const char* tag_col  = "";
    const char* body_col = "";
    switch (level) {
        case Level::Info:  tag = "INFO ";  tag_col = kCyan;     body_col = kFg;       break;
        case Level::Warn:  tag = "WARN ";  tag_col = kAmberBold; body_col = kAmber;   break;
        case Level::Error: tag = "ERROR";  tag_col = kRedBold;  body_col = kRed;      break;
    }

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);

    auto body = fmt::vformat(fmt_str, args);

    if (TtyEnabled()) {
        fmt::print(stderr,
                   "{ts_col}[{:02d}:{:02d}:{:02d}]{rs} {tag_col}[{}]{rs} {body_col}{}{rs}\n",
                   tm.tm_hour, tm.tm_min, tm.tm_sec, tag, body,
                   fmt::arg("ts_col",   kDim),
                   fmt::arg("tag_col",  tag_col),
                   fmt::arg("body_col", body_col),
                   fmt::arg("rs",       kReset));
    } else {
        fmt::print(stderr, "[{:02d}:{:02d}:{:02d}] [{}] {}\n",
                   tm.tm_hour, tm.tm_min, tm.tm_sec, tag, body);
    }
    std::fflush(stderr);

    // Snapshot the sink table under the lock; invoke without holding it so
    // a slow sink cannot block other log-emitting threads.
    std::array<Sink, kMaxSinks> snapshot{};
    std::size_t count = 0;
    {
        std::lock_guard lock(g_sink_mutex);
        snapshot = g_sinks;
        count = g_sink_count;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (snapshot[i] != nullptr) snapshot[i](level, body);
    }
}

}  // namespace pt::log

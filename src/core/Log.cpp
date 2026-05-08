// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Log.h"

#include <fmt/format.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

// Portable isatty: POSIX uses <unistd.h>::isatty + ::fileno; MSVC's
// runtime exposes _isatty + _fileno in <io.h>. Both check whether
// the given C stdio FILE* is bound to a terminal.
#if defined(_WIN32)
#include <io.h>
#define PT_ISATTY(fd) (_isatty(fd) != 0)
#define PT_FILENO(f)  _fileno(f)
#else
#include <unistd.h>
#define PT_ISATTY(fd) (::isatty(fd) != 0)
#define PT_FILENO(f)  fileno(f)
#endif

namespace pt::log {

namespace {

constexpr std::size_t kMaxSinks = 4;
std::array<Sink, kMaxSinks> g_sinks{};
std::size_t g_sink_count = 0;
std::mutex  g_sink_mutex;

// Cache the tty check; stderr won't change handle types mid-run.
bool TtyEnabled() {
    static const bool v = PT_ISATTY(PT_FILENO(stderr));
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
    const char* tag      = "INFO";
    const char* tag_col  = "";
    const char* body_col = "";
    switch (level) {
        case Level::Info:  tag = "INFO";  tag_col = kCyan;      body_col = kFg;     break;
        case Level::Warn:  tag = "WARN";  tag_col = kAmberBold; body_col = kAmber;  break;
        case Level::Error: tag = "ERROR"; tag_col = kRedBold;   body_col = kRed;    break;
    }

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    // Portable localtime: POSIX has localtime_r(time_t*, tm*); MSVC has
    // localtime_s(tm*, time_t*) -- note the reversed argument order.
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    auto body = fmt::vformat(fmt_str, args);

    if (TtyEnabled()) {
        // Pre-render the bracketed timestamp so the outer print uses
        // only named args (fmt 11 forbids mixing manual & auto indexing).
        auto ts = fmt::format("[{:02d}:{:02d}:{:02d}]",
                              tm.tm_hour, tm.tm_min, tm.tm_sec);
        fmt::print(stderr,
                   "{ts_col}{ts}{rs} {tag_col}[{tag}]{rs} {body_col}{body}{rs}\n",
                   fmt::arg("ts_col",   kDim),
                   fmt::arg("ts",       ts),
                   fmt::arg("tag_col",  tag_col),
                   fmt::arg("tag",      tag),
                   fmt::arg("body_col", body_col),
                   fmt::arg("body",     body),
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

#include "Log.h"

#include <fmt/format.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace pt::log {

namespace {

constexpr std::size_t kMaxSinks = 4;
std::array<Sink, kMaxSinks> g_sinks{};
std::size_t g_sink_count = 0;
std::mutex  g_sink_mutex;

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
    const char* tag = "INFO ";
    switch (level) {
        case Level::Info:  tag = "INFO "; break;
        case Level::Warn:  tag = "WARN "; break;
        case Level::Error: tag = "ERROR"; break;
    }

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);

    auto body = fmt::vformat(fmt_str, args);
    fmt::print(stderr, "[{:02d}:{:02d}:{:02d}] [{}] {}\n",
               tm.tm_hour, tm.tm_min, tm.tm_sec, tag, body);
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

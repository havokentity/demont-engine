#include "Log.h"

#include <fmt/format.h>
#include <chrono>
#include <ctime>
#include <cstdio>

namespace pt::log {

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
}

}  // namespace pt::log

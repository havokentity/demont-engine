// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <fmt/format.h>

namespace pt::log {

enum class Level { Info, Warn, Error };

void Emit(Level level, fmt::string_view fmt_str, fmt::format_args args);

// A sink receives every log line in addition to stderr.  Used by the
// console server to push log events to subscribed WebSocket clients.
using Sink = void(*)(Level, const std::string&);
void AddSink(Sink sink);
void RemoveAllSinks();

template <typename... Args>
inline void Info(fmt::format_string<Args...> f, Args&&... args) {
    Emit(Level::Info, f, fmt::make_format_args(args...));
}

template <typename... Args>
inline void Warn(fmt::format_string<Args...> f, Args&&... args) {
    Emit(Level::Warn, f, fmt::make_format_args(args...));
}

template <typename... Args>
inline void Error(fmt::format_string<Args...> f, Args&&... args) {
    Emit(Level::Error, f, fmt::make_format_args(args...));
}

}  // namespace pt::log

#define LOG_INFO(...)  ::pt::log::Info(__VA_ARGS__)
#define LOG_WARN(...)  ::pt::log::Warn(__VA_ARGS__)
#define LOG_ERROR(...) ::pt::log::Error(__VA_ARGS__)

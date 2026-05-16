// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <string_view>

// Output-format enum + parser shared between FrameCapture's per-frame
// dispatch and the cvar surface. Header-only so unit tests can pull it
// in without linking the encoder (the encoder TU is what owns the stb
// implementation; this file owns only the format identity).
//
// The cvar `r_capture_format` is registered in CaptureEncoder.cpp with
// `allowed_values = {"png","ppm"}` so the Console::Execute path
// auto-rejects unknown strings with a uniform error. ParseOutputFormat
// is the read-side counterpart used at capture time.

namespace pt::engine::capture {

enum class OutputFormat : std::uint8_t {
    Png = 0,
    Ppm = 1,
};

// Cvar name + default value + accepted set, exported as constants so
// registrant and unit tests reference the same source of truth.
inline constexpr const char* kCaptureFormatCvar    = "r_capture_format";
inline constexpr const char* kCaptureFormatDefault = "png";

inline bool ParseOutputFormat(std::string_view s, OutputFormat& out) noexcept {
    if (s == "png") { out = OutputFormat::Png; return true; }
    if (s == "ppm") { out = OutputFormat::Ppm; return true; }
    return false;
}

inline std::string_view OutputFormatExtension(OutputFormat fmt) noexcept {
    switch (fmt) {
        case OutputFormat::Png: return "png";
        case OutputFormat::Ppm: return "ppm";
    }
    return "bin";
}

}  // namespace pt::engine::capture

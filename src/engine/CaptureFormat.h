// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

// Output-format enum + parser shared between FrameCapture's per-frame
// dispatch and the cvar surface. Header-only so unit tests can pull it
// in without linking the encoder (the encoder TU is what owns the stb
// implementation; this file owns only the format identity).
//
// The cvar `r_capture_format` is registered in FrameCapture.cpp at
// static-init time (sentinel `kCaptureFormatRegistered`), with
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

// Normalize an operator-supplied output path to match `fmt`. The cvar
// is the single source of truth for the on-disk format, so any
// extension the user typed is unconditionally replaced with the one
// derived from `fmt`. Examples:
//   ("foo",       Png) -> "foo.png"
//   ("foo.ppm",   Png) -> "foo.png"   (override)
//   ("foo.png",   Ppm) -> "foo.ppm"   (override)
//   ("sub/foo",   Png) -> "sub/foo.png"
//   ("foo.bar",   Png) -> "foo.png"   (drops unknown ext too)
//
// Rationale: a `screenshot foo.ppm` while r_capture_format=png that
// silently wrote a PPM would defeat the cvar's purpose -- the user
// would have two ways to choose the format and one would win
// silently. Single source of truth is less surprising.
inline std::filesystem::path ResolveCapturePath(std::string_view user_path,
                                                OutputFormat       fmt) {
    std::filesystem::path p(user_path);
    p.replace_extension(std::string(".") + std::string(OutputFormatExtension(fmt)));
    return p;
}

// Convenience read-side helper: parse `s` into an OutputFormat, with
// `fallback` returned on parse failure. CaptureFormat.h has no LOG_*
// dep, so callers that want a warning emit it themselves.
inline OutputFormat ParseOutputFormatOr(std::string_view s,
                                        OutputFormat     fallback) noexcept {
    OutputFormat out = fallback;
    (void)ParseOutputFormat(s, out);
    return out;
}

}  // namespace pt::engine::capture

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "CaptureFormat.h"
#include "FrameCapture.h"   // CaptureSourceKind

#include <cstdint>
#include <filesystem>
#include <vector>

// Pixel encoders for FrameCapture. Split out of FrameCapture.cpp so the
// stb_image_write implementation (the only third-party dependency in
// the engine that emits a non-trivial /W4 warning flood on MSVC) is
// confined to its own static library `pt_capture_encoder`, and so the
// pure-compute encode/round-trip can be unit-tested without dragging
// in the RHI / renderer / Vulkan toolchain.
//
// Same ACES filmic + sRGB OETF as the screenshot path in Engine.cpp's
// `screenshot` command and as PathTrace.slang's inline tonemap; the
// host-side pipeline matches the GPU pipeline byte-for-byte modulo
// post-tonemap effects (bloom / lens flare / perf overlay) that don't
// land in the readback texture.

namespace pt::engine::capture {

// Tonemap `raw` (readback bytes for `kind` at `w`x`h`, exposure-scaled)
// into a fresh `w * h * 3` sRGB byte buffer. Shared between the PPM and
// PNG encoders so the two encoder paths cannot drift on gamma / pack /
// channel order.
//
// `kind` selects how to interpret `raw`:
//   CaptureSourceKind::Accum         -> RGBA32F linear HDR
//   CaptureSourceKind::DenoiseColor  -> RGBA16F linear HDR (half decode)
std::vector<std::uint8_t> BuildSrgbBuffer(
    const std::vector<std::uint8_t>& raw,
    std::uint32_t                    w,
    std::uint32_t                    h,
    CaptureSourceKind                kind,
    float                            exposure);

// Encode `raw` and write as PPM `P6` (binary RGB) to `path`. Returns
// true on success. Header is plain-text "P6\n<w> <h>\n255\n", body is
// `BuildSrgbBuffer`'s output as a raw fwrite -- portable, dependency-
// free, and the legacy path that pre-dates stb_image_write being in-
// tree. Retained so a `PT_BUILD_TESTS=OFF` / no-stb build can still
// capture.
bool EncodeAndWritePpm(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure);

// Encode `raw` and write as 8-bit RGB PNG via `stbi_write_png`.
// Returns true on success. PNG is the default format and the one
// `imgdiff` (PR #92) + the golden-image regression matrix (#45)
// consume.
bool EncodeAndWritePng(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure);

}  // namespace pt::engine::capture

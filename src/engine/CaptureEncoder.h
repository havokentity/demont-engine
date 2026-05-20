// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "CaptureFormat.h"
#include "FrameCapture.h"   // CaptureSourceKind

#include <cstdint>
#include <filesystem>
#include <string_view>
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

// --- Wave 9 tonemap (#27 follow-up) ---
// Map the r_tonemap_op cvar string to the operator enum shared by every
// tonemap site (the GPU shaders' kTonemap* constants, the host capture
// encoders, the SoftwareTracer mirror): 0 aces / 1 agx / 2 khronos_pbr
// _neutral / 3 reinhard / 4 linear. Unknown / empty strings map to 0
// (aces) so a malformed config can't black the screen. Single source of
// truth so the inline engine path and the capture path can't drift.
std::uint32_t ParseTonemapOp(std::string_view op);

// Tonemap `raw` (readback bytes for `kind` at `w`x`h`, exposure-scaled)
// into a fresh `w * h * 3` sRGB byte buffer. Shared between the PPM and
// PNG encoders so the two encoder paths cannot drift on gamma / pack /
// channel order.
//
// `kind` selects how to interpret `raw`:
//   CaptureSourceKind::Accum         -> RGBA32F linear HDR
//   CaptureSourceKind::DenoiseColor  -> RGBA16F linear HDR (half decode)
//
// `tonemap_op` (Wave 9 #27 follow-up) selects the HDR->LDR display
// transform applied after exposure: 0 aces (default) / 1 agx / 2 khronos
// _pbr_neutral / 3 reinhard / 4 linear. The host-side curves mirror the
// GPU paths (shaders/PathTrace.slang + Tonemap.slang) so a captured PNG
// matches what's on-screen. op 0 (aces) keeps the previous per-channel
// path byte-for-byte, so all existing goldens are unchanged; defaulted so
// callers that don't care (screenshot depth/motion targets, unit tests)
// compile + behave exactly as before.
std::vector<std::uint8_t> BuildSrgbBuffer(
    const std::vector<std::uint8_t>& raw,
    std::uint32_t                    w,
    std::uint32_t                    h,
    CaptureSourceKind                kind,
    float                            exposure,
    std::uint32_t                    tonemap_op = 0u);

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
                       float                            exposure,
                       std::uint32_t                    tonemap_op = 0u);

// Encode `raw` and write as 8-bit RGB PNG via `stbi_write_png`.
// Returns true on success. PNG is the default format and the one
// `imgdiff` (PR #92) + the golden-image regression matrix (#45)
// consume.
bool EncodeAndWritePng(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure,
                       std::uint32_t                    tonemap_op = 0u);

// --- Wave 9 tonemap (#27 follow-up) ---
// Tonemap one linear-HDR pixel (pre-exposure) with operator `tonemap_op`
// (0 aces / 1 agx / 2 khronos_pbr_neutral / 3 reinhard / 4 linear), writing
// three sRGB-quantised bytes to `dst` (>= 3 bytes). Exposes the same
// per-pixel transform `BuildSrgbBuffer` applies so the interactive
// `screenshot` command (which has its own decode loop for depth/motion
// targets) can share the exact operator math instead of duplicating it.
// op 0 (aces) is byte-identical to the legacy per-channel ACES path.
void TonemapHdrPixel(float r, float g, float b, float exposure,
                     std::uint32_t tonemap_op, std::uint8_t* dst);

// Write a pre-built tightly-packed 8-bit RGB buffer (`w * h * 3` bytes)
// to `path` in `fmt`. Used by callers that build the rgb buffer
// themselves (e.g. the `screenshot` command's depth / motion / swap
// targets, which use mappings that don't fit the ACES + sRGB pipeline
// the `EncodeAndWrite*` helpers bake in). Returns true on success;
// on failure the file is closed but may be partially written.
//
// `rgb` must point to at least `w * h * 3` bytes, with no padding
// between rows.
bool WriteRgb8(const std::filesystem::path& path,
               const std::uint8_t*          rgb,
               std::uint32_t                w,
               std::uint32_t                h,
               OutputFormat                 fmt);

}  // namespace pt::engine::capture

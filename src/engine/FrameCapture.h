// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace pt::rhi { class Device; }

namespace pt::engine {

// Frame-capture / RMSE-comparator support. Drives objective denoiser
// A/B comparison: capture the same frame from svgf_atrous vs optix_hdr
// vs optix_hdr_aov vs ... and feed the resulting PPMs through
// `scripts/compare_denoisers.py` to get RMSE / SSIM numbers instead of
// squinting at screenshots.
//
// The capture path piggy-backs on the existing `ReadbackTexture` RHI
// primitive that the `screenshot` command uses. Two source textures
// are supported, picked automatically by the engine:
//   * `accum`         -- RGBA32F path-tracer accumulator (denoiser off)
//   * `denoise_color` -- RGBA16F denoiser output (denoiser on)
// Both go through the same on-CPU ACES tonemap + sRGB OETF as the
// screenshot path so the resulting PPMs are directly comparable.
//
// Determinism: the path tracer's PRNG is seeded purely from
// (pixel_id, frame_index) (see PathTrace.slang's `pcgHash(...)` line),
// and `frame_index_` starts at 0 on every cold launch. So two demont.exe
// runs with identical settings produce bitwise-identical accum at the
// same frame_n. `r_capture_seed` lets the engine explicitly reset
// frame_index_ to a known base to make this guarantee survive cvar
// flips during a single session.
//
// State lives in a file-static inside FrameCapture.cpp -- writes come
// from cvar on_change handlers (main thread) and reads come from the
// post-present hook (also main thread), so no synchronisation is
// needed. Keeping state out of `Engine` avoids touching Engine.h while
// the parallel Phase-1b branch is editing it.

// Source texture kind. Engine picks one based on whether a denoiser is
// active; FrameCapture itself is denoiser-agnostic.
//
// `DenoiseColor` is the denoiser's *output* image -- in the engine
// that's `post_denoise_hdr_tex_id_` (the post-denoise pre-tonemap HDR
// buffer that bloom + tonemap then read). The enum name doesn't track
// the engine's internal texture name `denoise_color_tex_id_`, which is
// confusingly the denoiser's *input* (path tracer's per-frame HDR
// write). Capturing the input would make svgf_atrous / optix_hdr /
// optix_hdr_aov produce bitwise-identical PPMs since the same image
// feeds into all three denoisers -- not what we want for RMSE
// comparison.
enum class CaptureSourceKind {
    Accum,         // RGBA32F linear HDR accumulator (denoiser off)
    DenoiseColor,  // RGBA16F post-denoise HDR output (denoiser on)
};

namespace capture {

// Configure a one-shot capture at absolute frame `frame_n`. Pass 0 to
// cancel an outstanding one-shot. Called from r_capture_frame_at's
// on_change handler.
void SetOneShotFrame(std::uint32_t frame_n);

// Start a sequence: capture `count` frames at `interval` frames apart,
// starting from the next render frame, named with `prefix`. Pass
// count=0 to cancel an outstanding sequence. Called from r_capture_seq's
// on_change handler.
void StartSequence(std::string_view prefix,
                   std::uint32_t    count,
                   std::uint32_t    interval);

// Drop any pending capture state. Used by r_capture_seed's on_change
// handler when the user resets the deterministic seed.
void Reset();

// What MaybeCapture did this frame. The boolean trio lets the engine
// keep the user-facing cvar surface honest: when the one-shot capture
// fires the engine resets `r_capture_frame_at` back to "0", and when a
// sequence completes the engine clears `r_capture_seq` back to "".
// Without these flags MaybeCapture's callers can't tell apart "no
// capture armed" from "capture just fired and disarmed itself".
struct MaybeCaptureResult {
    bool wrote          = false;  // any PPM was written this frame
    bool one_shot_fired = false;  // r_capture_frame_at trigger fired
    bool seq_completed  = false;  // r_capture_seq decremented to 0
};

// Single per-frame entry point. Called from the engine's render loop
// after `device_->EndFrame()` (post-present).
//
// `denoiser_label` is just a tag for the output filename (e.g.
// "off" / "svgf_atrous" / "optix_hdr_aov"); FrameCapture doesn't
// interpret it. `exposure_state_buf_id` may be 0; if non-zero, the
// live GPU-resident exposure scalar is read back so the on-CPU ACES
// tonemap matches what the GPU's PathTrace.slang / Tonemap.slang
// applies on-screen (mirroring the screenshot-command behaviour).
//
// Path-traversal safety: `denoiser_label` and the internal sequence
// `prefix` (set via StartSequence) are sanitised to a `[A-Za-z0-9_-]+`
// subset before being formatted into the output filename, so a hostile
// or accidentally weird cvar value (e.g. "../../etc") cannot escape
// the captures/ directory.
MaybeCaptureResult MaybeCapture(pt::rhi::Device*  device,
                                std::uint32_t     frame_index,
                                std::uint64_t     accum_tex_id,
                                std::uint64_t     denoise_color_tex_id,
                                std::uint64_t     exposure_state_buf_id,
                                std::int32_t      accum_w,
                                std::int32_t      accum_h,
                                CaptureSourceKind source,
                                std::string_view  denoiser_label,
                                float             exposure_fallback);

// Cheap predicate: is any capture currently armed? Engine can use this
// to skip allocations / readback paths when nothing is requested.
bool IsArmed();

}  // namespace capture
}  // namespace pt::engine

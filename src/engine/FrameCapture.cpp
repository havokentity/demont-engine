// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "FrameCapture.h"

#include "CaptureEncoder.h"
#include "CaptureFormat.h"

#include "../console/Console.h"
#include "../core/Diag.h"
#include "../core/Log.h"
#include "../rhi/Device.h"
#include "../rhi/Resources.h"

#include <fmt/format.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

namespace pt::engine::capture {

namespace {

// Persistent state. All writes come from cvar on_change handlers
// (Console::Drain context, main thread); all reads come from the
// post-present hook (also main thread). No synchronisation needed.
struct State {
    std::uint32_t one_shot_frame  = 0;       // r_capture_frame_at; 0 = disarmed
    std::string   seq_prefix;
    std::uint32_t seq_remaining   = 0;
    std::uint32_t seq_interval    = 1;
    std::uint32_t seq_next_frame  = 0;       // absolute frame_index
};

State& GetState() {
    static State s;
    return s;
}

// Register `r_capture_format` at static-init time. PT_CVAR doesn't
// surface `allowed_values`, so we register manually via Console::Get
// and mutate the returned CVar*. `allowed_values` makes Console::
// Execute reject unknown strings with a uniform "got 'X', expected
// 'png|ppm'" error -- no per-cvar on_change validator needed.
//
// Static-init runs once at program startup before any FrameCapture
// function is reachable, so by the time DoCapture queries the cvar
// it's guaranteed to exist with a valid `value`. Sentinel bool keeps
// the initializer from being optimised away under LTO.
const bool kCaptureFormatRegistered = [] {
    auto* cv = pt::console::Console::Get().RegisterCVar(
        kCaptureFormatCvar,
        kCaptureFormatDefault,
        "Frame-capture output format. 'png' (default, 8-bit RGB via "
        "stb_image_write) is consumed by `imgdiff` and the golden-image "
        "regression matrix. 'ppm' (legacy P6 plain-header) is dependency-"
        "free and retained as a fallback. Filename suffix derives from "
        "this value, not from the prefix passed to r_capture_seq.",
        pt::console::CVAR_ARCHIVE);
    if (cv != nullptr) {
        cv->allowed_values = {"png", "ppm"};
    }
    return cv != nullptr;
}();

// Read the current value of r_capture_format and parse into OutputFormat.
// allowed_values gates the write side at Console::Execute, but startup
// races (cvar registered after a malformed demont.cfg load) and
// defence-in-depth motivate the LOG_WARN-and-default-to-PNG fallback.
OutputFormat ResolveCaptureFormat() {
    auto* cv = pt::console::Console::Get().FindCVar(kCaptureFormatCvar);
    if (cv == nullptr) return OutputFormat::Png;
    OutputFormat fmt = OutputFormat::Png;
    if (!ParseOutputFormat(cv->value, fmt)) {
        LOG_WARN("FrameCapture: unrecognized r_capture_format='{}' "
                 "(allowed: png|ppm); defaulting to png", cv->value);
        return OutputFormat::Png;
    }
    return fmt;
}

// Pull the live GPU-resident exposure scalar so the host-side tonemap
// matches the GPU paths. Mirrors the screenshot command's behaviour.
// Falls back to `fallback` (typically the r_exposure cvar value) if the
// readback isn't supported (e.g. software backend).
float ResolveExposure(pt::rhi::Device* device,
                      std::uint64_t    exposure_state_buf_id,
                      float            fallback) {
    if (device == nullptr || exposure_state_buf_id == 0) return fallback;
    float v = fallback;
    bool ok = device->ReadbackBuffer(pt::rhi::BufferHandle{exposure_state_buf_id},
                                     &v, sizeof(float));
    return ok ? v : fallback;
}

// captures/ subdir relative to the working directory. Created on first
// use, ignored on already-exists. CWD-relative matches the screenshot
// command's path-handling so dev-checkout vs install-tree behaves the
// same for both.
std::filesystem::path EnsureCapturesDir() {
    namespace fs = std::filesystem;
    fs::path dir = fs::path("captures");
    std::error_code ec;
    fs::create_directories(dir, ec);  // best-effort; non-fatal if it fails
    return dir;
}

// Sanitize a user-controllable string (sequence prefix, denoiser
// label) before letting it land in a filename. Maps any character
// outside [A-Za-z0-9_-] to '_'; collapses repeated underscores and
// trims leading/trailing underscores; clamps to a reasonable length
// so a pathological input doesn't blow up MAX_PATH (260 on legacy
// Windows). Returns the sanitized string; if the input was entirely
// stripped, falls back to `fallback` so we never produce a filename
// segment of just ".ppm".
std::string SanitizeFilenamePart(std::string_view in,
                                 std::string_view fallback) {
    constexpr std::size_t kMaxLen = 64;
    std::string out;
    out.reserve(std::min<std::size_t>(in.size(), kMaxLen));
    bool last_underscore = false;
    for (char c : in) {
        const bool ok =
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-';
        if (ok) {
            out.push_back(c);
            last_underscore = (c == '_');
        } else if (!last_underscore) {
            out.push_back('_');
            last_underscore = true;
        }
        if (out.size() >= kMaxLen) break;
    }
    // Trim leading / trailing underscores so we don't emit names
    // like "_foo_" / "..ppm".
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back()  == '_') out.pop_back();
    if (out.empty()) out.assign(fallback);
    return out;
}

std::string FormatTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return fmt::format("{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}",
                       tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                       tm.tm_hour, tm.tm_min, tm.tm_sec);
}

bool DoCapture(pt::rhi::Device*  device,
               std::uint32_t     frame_index,
               std::uint64_t     accum_tex_id,
               std::uint64_t     denoise_color_tex_id,
               std::uint64_t     exposure_state_buf_id,
               std::int32_t      accum_w,
               std::int32_t      accum_h,
               CaptureSourceKind source,
               std::string_view  denoiser_label,
               float             exposure_fallback,
               std::string_view  filename_prefix) {
    if (device == nullptr) return false;
    if (accum_w <= 0 || accum_h <= 0) return false;

    std::uint64_t tex_id = 0;
    int           bytes_per_pixel = 0;
    switch (source) {
        case CaptureSourceKind::Accum:
            tex_id = accum_tex_id;
            bytes_per_pixel = 16;  // RGBA32F
            break;
        case CaptureSourceKind::DenoiseColor:
            tex_id = denoise_color_tex_id;
            bytes_per_pixel = 8;   // RGBA16F
            break;
    }
    if (tex_id == 0) {
        LOG_WARN("FrameCapture: requested source texture is not allocated "
                 "(denoiser={} frame={})",
                 denoiser_label, frame_index);
        return false;
    }

    std::vector<std::uint8_t> raw(std::size_t(accum_w) * accum_h * bytes_per_pixel);
    std::uint32_t w = 0, h = 0;
    if (!device->ReadbackTexture(pt::rhi::TextureHandle{tex_id},
                                 raw.data(), raw.size(),
                                 &w, &h)) {
        LOG_WARN("FrameCapture: ReadbackTexture failed (denoiser={} frame={})",
                 denoiser_label, frame_index);
        return false;
    }
    if (w == 0 || h == 0) {
        LOG_WARN("FrameCapture: ReadbackTexture returned empty extent "
                 "(denoiser={} frame={})", denoiser_label, frame_index);
        return false;
    }

    const float exposure = ResolveExposure(device, exposure_state_buf_id,
                                           exposure_fallback);

    // Sanitize the operator-controllable filename parts so e.g.
    // `r_capture_seq "../../etc 5 1"` or `r_denoiser` containing
    // path separators can't escape captures/. Both come from cvar
    // values and should always be reasonable, but defence-in-depth
    // is cheap.
    auto dir       = EnsureCapturesDir();
    auto safe_pref = SanitizeFilenamePart(filename_prefix, "capture");
    auto safe_lbl  = SanitizeFilenamePart(denoiser_label,  "unknown");

    // Format selection: r_capture_format gates which encoder runs.
    // Filename suffix derives from the resolved format (.png / .ppm),
    // not from the prefix, so a sequence prefix that already contains
    // an extension produces consistent on-disk names.
    const OutputFormat       output_fmt = ResolveCaptureFormat();
    const std::string_view   ext        = OutputFormatExtension(output_fmt);
    auto fname     = fmt::format("{}_{:06d}_{}_{}.{}",
                                 safe_pref, frame_index,
                                 safe_lbl, FormatTimestamp(), ext);
    auto path      = dir / fname;

    bool ok = false;
    switch (output_fmt) {
        case OutputFormat::Png:
            ok = EncodeAndWritePng(path, raw, w, h, source, exposure);
            break;
        case OutputFormat::Ppm:
            ok = EncodeAndWritePpm(path, raw, w, h, source, exposure);
            break;
    }
    if (!ok) {
        LOG_WARN("FrameCapture: cannot encode/write '{}' (format={})",
                 path.string(), ext);
        return false;
    }

    LOG_INFO("FrameCapture: wrote {} ({}x{} {}, exposure={:.3f}, format={})",
             path.string(), w, h,
             source == CaptureSourceKind::Accum ? "accum_hdr" : "denoise_color",
             exposure, ext);
    return true;
}

}  // namespace

void SetOneShotFrame(std::uint32_t frame_n) {
    auto& s = GetState();
    s.one_shot_frame = frame_n;
    if (frame_n != 0) {
        PT_DIAG_TIER1("capture",
                      "one-shot capture armed for frame={}", frame_n);
    }
}

void StartSequence(std::string_view prefix,
                   std::uint32_t    count,
                   std::uint32_t    interval) {
    auto& s = GetState();
    if (count == 0) {
        s.seq_remaining = 0;
        s.seq_prefix.clear();
        s.seq_next_frame = 0;
        s.seq_interval   = 1;
        return;
    }
    s.seq_prefix.assign(prefix);
    s.seq_remaining  = count;
    s.seq_interval   = (interval == 0) ? 1u : interval;
    // Sentinel: fire on the very next render frame regardless of its
    // absolute index. The post-present hook detects this via
    // `frame_index >= seq_next_frame` with seq_next_frame=0 being
    // always-true, then advances seq_next_frame to frame+interval.
    s.seq_next_frame = 0;
    PT_DIAG_TIER1("capture",
                  "sequence capture armed: prefix={} count={} interval={}",
                  s.seq_prefix, count, s.seq_interval);
}

void Reset() {
    auto& s = GetState();
    s = State{};
}

bool IsArmed() {
    const auto& s = GetState();
    return s.one_shot_frame != 0 || s.seq_remaining > 0;
}

MaybeCaptureResult MaybeCapture(pt::rhi::Device*  device,
                                std::uint32_t     frame_index,
                                std::uint64_t     accum_tex_id,
                                std::uint64_t     denoise_color_tex_id,
                                std::uint64_t     exposure_state_buf_id,
                                std::int32_t      accum_w,
                                std::int32_t      accum_h,
                                CaptureSourceKind source,
                                std::string_view  denoiser_label,
                                float             exposure_fallback) {
    MaybeCaptureResult r;
    auto& s = GetState();
    if (s.one_shot_frame == 0 && s.seq_remaining == 0) return r;

    // r_capture_frame_at: one-shot trigger. Disarms only on a
    // successful write -- if DoCapture fails (e.g. ReadbackTexture
    // returned an empty extent), keep the trigger armed and try
    // again next frame so a transient hiccup doesn't lose the shot.
    // The engine's post-present hook clears the cvar surface once
    // `one_shot_fired` is reported.
    if (s.one_shot_frame != 0 && frame_index == s.one_shot_frame) {
        const bool ok = DoCapture(device, frame_index,
                                  accum_tex_id, denoise_color_tex_id,
                                  exposure_state_buf_id,
                                  accum_w, accum_h, source,
                                  denoiser_label, exposure_fallback,
                                  /*filename_prefix=*/"capture");
        if (ok) {
            r.wrote = true;
            r.one_shot_fired = true;
            s.one_shot_frame = 0;
        }
        // On failure leave one_shot_frame set; next frame past the
        // target it will skip the equality check, so we'd actually
        // lose the trigger anyway. Bump it to (frame_index + 1) so
        // the retry hits next frame instead.
        else {
            s.one_shot_frame = frame_index + 1;
        }
    }

    // r_capture_seq: emit at every `seq_interval` frames until
    // `seq_remaining` decrements to zero. Same retry semantics as
    // the one-shot path: only decrement / advance the schedule on
    // a successful write, so a readback hiccup costs at most one
    // frame of latency, not a missing capture in the middle of the
    // sequence.
    if (s.seq_remaining > 0 && frame_index >= s.seq_next_frame) {
        const bool ok = DoCapture(device, frame_index,
                                  accum_tex_id, denoise_color_tex_id,
                                  exposure_state_buf_id,
                                  accum_w, accum_h, source,
                                  denoiser_label, exposure_fallback,
                                  /*filename_prefix=*/s.seq_prefix);
        if (ok) {
            r.wrote = true;
            --s.seq_remaining;
            s.seq_next_frame = frame_index + s.seq_interval;
            if (s.seq_remaining == 0) {
                PT_DIAG_TIER1("capture",
                              "sequence capture done (prefix={})",
                              s.seq_prefix);
                s.seq_prefix.clear();
                s.seq_next_frame = 0;
                s.seq_interval   = 1;
                r.seq_completed  = true;
            }
        } else {
            // Don't burn a count on a failed write. Re-schedule for
            // next frame (interval-spacing is sacrificed once on a
            // hiccup, but the user still gets `count` total
            // captures) -- the failure already logged a LOG_WARN
            // inside DoCapture so this is silent retry on top of
            // visible diagnostic.
            s.seq_next_frame = frame_index + 1;
        }
    }

    return r;
}

}  // namespace pt::engine::capture

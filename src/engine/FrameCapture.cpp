// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "FrameCapture.h"

#include "../core/Diag.h"
#include "../core/Log.h"
#include "../rhi/Device.h"
#include "../rhi/Resources.h"

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
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

// Portable IEEE 754 binary16 -> binary32 decode. Lifted from the
// screenshot path in Engine.cpp's RegisterCommands -- duplicated rather
// than refactored so this PR doesn't churn the existing screenshot
// command. A future PR can deduplicate by pulling both the half decode
// and the ACES + sRGB OETF into a small `pt::renderer::HostTonemap`
// header.
float HalfToFloat(std::uint16_t h_) {
    const std::uint32_t sign = (h_ >> 15) & 0x1u;
    const std::uint32_t exp  = (h_ >> 10) & 0x1Fu;
    const std::uint32_t mant = h_ & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // signed zero
        } else {
            // Subnormal: renormalise into a regular float32. Same shift
            // logic as the screenshot path; see that comment for the
            // detailed exponent-rebias derivation.
            std::uint32_t e = 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            m &= 0x3FFu;
            f = (sign << 31)
              | ((127u - 15u - e + 2u) << 23)
              | (m << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31)
          | ((exp - 15u + 127u) << 23)
          | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

std::uint8_t AcesPlusSrgb(float linear, float exposure) {
    float c = linear * exposure;
    // ACES filmic (Hill / Krzysztof Narkowicz fit) -- same constants as
    // the screenshot path and PathTrace.slang's inline tonemap, so PPMs
    // match what's on-screen pixel-for-pixel modulo bloom / lens flare /
    // perf overlay (all of which only land in the swapchain after this
    // intermediate).
    const float a = 2.51f, b = 0.03f, d = 2.43f, e = 0.59f, f = 0.14f;
    float x = (c * (a * c + b)) / (c * (d * c + e) + f);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    // sRGB OETF: piecewise to match the shader's srgb_oetf so the LDR
    // PPM is the same byte-for-byte as what would land in the
    // swapchain's sRGB-formatted attachment.
    if (x <= 0.0031308f) {
        x = x * 12.92f;
    } else {
        x = 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    }
    if (x > 1.0f) x = 1.0f;
    return static_cast<std::uint8_t>(x * 255.0f + 0.5f);
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

// Decode `raw` (the readback bytes for `kind`) into 8-bit sRGB and
// write a PPM P6 to `path`. Returns true on success.
bool EncodeAndWritePpm(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure) {
    std::vector<std::uint8_t> rgb(std::size_t(w) * h * 3);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            const std::size_t pi = std::size_t(y) * w + x;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            switch (kind) {
                case CaptureSourceKind::Accum: {
                    // RGBA32F linear HDR.
                    const float* src = reinterpret_cast<const float*>(raw.data()) + pi * 4;
                    r = src[0]; g = src[1]; b = src[2];
                    break;
                }
                case CaptureSourceKind::DenoiseColor: {
                    // RGBA16F linear HDR (denoiser output).
                    const std::uint16_t* src =
                        reinterpret_cast<const std::uint16_t*>(raw.data()) + pi * 4;
                    r = HalfToFloat(src[0]);
                    g = HalfToFloat(src[1]);
                    b = HalfToFloat(src[2]);
                    break;
                }
            }
            std::uint8_t* dst = rgb.data() + pi * 3;
            dst[0] = AcesPlusSrgb(r, exposure);
            dst[1] = AcesPlusSrgb(g, exposure);
            dst[2] = AcesPlusSrgb(b, exposure);
        }
    }

    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (f == nullptr) return false;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
    return true;
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

    auto dir   = EnsureCapturesDir();
    auto fname = fmt::format("{}_{:06d}_{}_{}.ppm",
                             filename_prefix.empty()
                                 ? std::string_view{"capture"}
                                 : filename_prefix,
                             frame_index,
                             denoiser_label.empty()
                                 ? std::string_view{"unknown"}
                                 : denoiser_label,
                             FormatTimestamp());
    auto path  = dir / fname;

    if (!EncodeAndWritePpm(path, raw, w, h, source, exposure)) {
        LOG_WARN("FrameCapture: cannot open '{}' for writing",
                 path.string());
        return false;
    }

    LOG_INFO("FrameCapture: wrote {} ({}x{} {}, exposure={:.3f})",
             path.string(), w, h,
             source == CaptureSourceKind::Accum ? "accum_hdr" : "denoise_color",
             exposure);
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

bool MaybeCapture(pt::rhi::Device*  device,
                  std::uint32_t     frame_index,
                  std::uint64_t     accum_tex_id,
                  std::uint64_t     denoise_color_tex_id,
                  std::uint64_t     exposure_state_buf_id,
                  std::int32_t      accum_w,
                  std::int32_t      accum_h,
                  CaptureSourceKind source,
                  std::string_view  denoiser_label,
                  float             exposure_fallback) {
    auto& s = GetState();
    if (s.one_shot_frame == 0 && s.seq_remaining == 0) return false;

    bool wrote = false;

    // r_capture_frame_at: one-shot trigger. Cleared after firing.
    if (s.one_shot_frame != 0 && frame_index == s.one_shot_frame) {
        wrote |= DoCapture(device, frame_index,
                           accum_tex_id, denoise_color_tex_id,
                           exposure_state_buf_id,
                           accum_w, accum_h, source,
                           denoiser_label, exposure_fallback,
                           /*filename_prefix=*/"capture");
        s.one_shot_frame = 0;  // belt-and-braces; cvar on_change clears too
    }

    // r_capture_seq: emit at every `seq_interval` frames until
    // `seq_remaining` decrements to zero.
    if (s.seq_remaining > 0 && frame_index >= s.seq_next_frame) {
        wrote |= DoCapture(device, frame_index,
                           accum_tex_id, denoise_color_tex_id,
                           exposure_state_buf_id,
                           accum_w, accum_h, source,
                           denoiser_label, exposure_fallback,
                           /*filename_prefix=*/s.seq_prefix);
        --s.seq_remaining;
        s.seq_next_frame = frame_index + s.seq_interval;
        if (s.seq_remaining == 0) {
            PT_DIAG_TIER1("capture",
                          "sequence capture done (prefix={})",
                          s.seq_prefix);
            s.seq_prefix.clear();
            s.seq_next_frame = 0;
            s.seq_interval   = 1;
        }
    }

    return wrote;
}

}  // namespace pt::engine::capture

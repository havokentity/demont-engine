// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "CaptureEncoder.h"

// stb_image_write declarations only -- the actual implementation
// (`STB_IMAGE_WRITE_IMPLEMENTATION`) lives in `stb_impl.cpp`, which
// gets per-file `/w` (MSVC) / `-w` (Clang/GCC) so the impl's warning
// flood doesn't leak into the rest of pt_capture_encoder.
#include "../../third_party/stb/stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace pt::engine::capture {

namespace {

// Portable IEEE 754 binary16 -> binary32 decode. Lifted from the
// previous in-place version inside FrameCapture.cpp (before this
// PR split the encoders out). Same screenshot-command logic as in
// Engine.cpp's RegisterCommands -- duplicated rather than refactored
// upstream so a future PR can deduplicate by pulling the half-decode
// and the ACES + sRGB OETF into a small `pt::renderer::HostTonemap`
// header that all three paths (screenshot, FrameCapture, PR #94's
// CaptureEncoder) consume.
float HalfToFloat(std::uint16_t h_) {
    const std::uint32_t sign = (h_ >> 15) & 0x1u;
    const std::uint32_t exp  = (h_ >> 10) & 0x1Fu;
    const std::uint32_t mant = h_ & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // signed zero
        } else {
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
    // the screenshot path and PathTrace.slang's inline tonemap so PPMs
    // / PNGs match what's on-screen pixel-for-pixel modulo bloom / lens
    // flare / perf overlay (all of which only land in the swapchain
    // after this intermediate).
    const float a = 2.51f, b = 0.03f, d = 2.43f, e = 0.59f, f = 0.14f;
    float x = (c * (a * c + b)) / (c * (d * c + e) + f);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    // sRGB OETF: piecewise to match the shader's srgb_oetf.
    if (x <= 0.0031308f) {
        x = x * 12.92f;
    } else {
        x = 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    }
    if (x > 1.0f) x = 1.0f;
    return static_cast<std::uint8_t>(x * 255.0f + 0.5f);
}

}  // namespace

std::vector<std::uint8_t> BuildSrgbBuffer(
    const std::vector<std::uint8_t>& raw,
    std::uint32_t                    w,
    std::uint32_t                    h,
    CaptureSourceKind                kind,
    float                            exposure) {
    // Per-component memcpy rather than `reinterpret_cast<float*>` /
    // `reinterpret_cast<uint16_t*>` on the byte buffer. `std::vector<
    // uint8_t>` only guarantees 1-byte alignment per the standard
    // (std::allocator returns max-align-aligned storage in practice on
    // every platform we target, so the typed loads work in production,
    // but the standard-level UB trips UBSan and is brittle if the
    // readback API ever switches to a custom allocator). memcpy is
    // the canonical way to do "load a value with this representation
    // from this byte stream" without strict-aliasing or alignment
    // assumptions; modern compilers fold it to a single MOV/LDR.
    std::vector<std::uint8_t> rgb(std::size_t(w) * h * 3);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            const std::size_t pi = std::size_t(y) * w + x;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            switch (kind) {
                case CaptureSourceKind::Accum: {
                    // RGBA32F: 16 bytes/pixel. Load R/G/B as three
                    // independent float reads from the raw byte stream.
                    const std::uint8_t* src = raw.data() + pi * 16;
                    std::memcpy(&r, src + 0,  sizeof(float));
                    std::memcpy(&g, src + 4,  sizeof(float));
                    std::memcpy(&b, src + 8,  sizeof(float));
                    break;
                }
                case CaptureSourceKind::DenoiseColor: {
                    // RGBA16F: 8 bytes/pixel. Load each half-float as
                    // uint16 then decode to float32 portably.
                    const std::uint8_t* src = raw.data() + pi * 8;
                    std::uint16_t hr = 0, hg = 0, hb = 0;
                    std::memcpy(&hr, src + 0, sizeof(std::uint16_t));
                    std::memcpy(&hg, src + 2, sizeof(std::uint16_t));
                    std::memcpy(&hb, src + 4, sizeof(std::uint16_t));
                    r = HalfToFloat(hr);
                    g = HalfToFloat(hg);
                    b = HalfToFloat(hb);
                    break;
                }
            }
            std::uint8_t* dst = rgb.data() + pi * 3;
            dst[0] = AcesPlusSrgb(r, exposure);
            dst[1] = AcesPlusSrgb(g, exposure);
            dst[2] = AcesPlusSrgb(b, exposure);
        }
    }
    return rgb;
}

bool EncodeAndWritePpm(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure) {
    const auto rgb = BuildSrgbBuffer(raw, w, h, kind, exposure);
    return WriteRgb8(path, rgb.data(), w, h, OutputFormat::Ppm);
}

bool EncodeAndWritePng(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure) {
    const auto rgb = BuildSrgbBuffer(raw, w, h, kind, exposure);
    return WriteRgb8(path, rgb.data(), w, h, OutputFormat::Png);
}

bool WriteRgb8(const std::filesystem::path& path,
               const std::uint8_t*          rgb,
               std::uint32_t                w,
               std::uint32_t                h,
               OutputFormat                 fmt) {
    if (rgb == nullptr || w == 0 || h == 0) return false;

    switch (fmt) {
        case OutputFormat::Png: {
            // Stride = w * 3 (tightly packed 8-bit RGB, no row padding).
            // stbi_write_png returns non-zero on success.
            const int rc = stbi_write_png(
                path.string().c_str(),
                static_cast<int>(w),
                static_cast<int>(h),
                /*comp=*/3,
                rgb,
                /*stride_in_bytes=*/static_cast<int>(w) * 3);
            return rc != 0;
        }
        case OutputFormat::Ppm: {
            std::FILE* f = std::fopen(path.string().c_str(), "wb");
            if (f == nullptr) return false;
            std::fprintf(f, "P6\n%u %u\n255\n", w, h);
            const std::size_t want  = std::size_t(w) * h * 3;
            const std::size_t wrote = std::fwrite(rgb, 1, want, f);
            std::fclose(f);
            return wrote == want;
        }
    }
    return false;
}

}  // namespace pt::engine::capture

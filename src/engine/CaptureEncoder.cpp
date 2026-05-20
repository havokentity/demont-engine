// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "CaptureEncoder.h"

// stb_image_write declarations only -- the actual implementation
// (`STB_IMAGE_WRITE_IMPLEMENTATION`) lives in `stb_impl.cpp`, which
// gets per-file `/w` (MSVC) / `-w` (Clang/GCC) so the impl's warning
// flood doesn't leak into the rest of pt_capture_encoder.
#include "../../third_party/stb/stb_image_write.h"

#include <algorithm>   // std::min, std::max
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

// sRGB OETF + 8-bit quantise. Shared tail of every operator. Piecewise to
// match the shader's srgb_oetf.
std::uint8_t SrgbQuantize(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    if (x <= 0.0031308f) {
        x = x * 12.92f;
    } else {
        x = 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    }
    if (x > 1.0f) x = 1.0f;
    return static_cast<std::uint8_t>(x * 255.0f + 0.5f);
}

float Aces1(float c) {
    // ACES filmic (Hill / Krzysztof Narkowicz fit) -- same constants as
    // the screenshot path and PathTrace.slang's inline tonemap so PPMs
    // / PNGs match what's on-screen pixel-for-pixel modulo bloom / lens
    // flare / perf overlay (all of which only land in the swapchain
    // after this intermediate). Per-channel.
    const float a = 2.51f, b = 0.03f, d = 2.43f, e = 0.59f, f = 0.14f;
    float x = (c * (a * c + b)) / (c * (d * c + e) + f);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    return x;
}

// --- Wave 9 tonemap (#27 follow-up) ----------------------------------------
// Per-pixel (RGB-coupled) operators. Inputs are scene-linear, post-exposure;
// outputs are display-linear sRGB (the SrgbQuantize tail applies the OETF).
// Constants are kept identical to the shader copies in PathTrace.slang /
// Tonemap.slang and the SoftwareTracer host mirror so every path agrees.
inline float Clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// AgX -- "Minimal AgX" (Benjamin Wrensch fit of Troy Sobotka's AgX).
void AgxTonemap(float& r, float& g, float& b) {
    auto contrast = [](float x) {
        float x2 = x * x, x4 = x2 * x2;
        return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4
             - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
    };
    // Inset matrix (rows).
    float v0 = 0.842479062253094f * r + 0.0784335999999992f * g + 0.0792237451477643f * b;
    float v1 = 0.0423282422610123f * r + 0.878468636469772f * g + 0.0791661274605434f * b;
    float v2 = 0.0423756549057051f * r + 0.0784336f * g + 0.879142973793104f * b;

    const float kMinEv = -12.47393f, kMaxEv = 4.026069f;
    auto enc = [&](float v) {
        v = std::log2(v < 1e-10f ? 1e-10f : v);
        if (v < kMinEv) v = kMinEv;
        if (v > kMaxEv) v = kMaxEv;
        return (v - kMinEv) / (kMaxEv - kMinEv);
    };
    v0 = contrast(enc(v0));
    v1 = contrast(enc(v1));
    v2 = contrast(enc(v2));

    // sat = slope = power = 1, offset = 0 -> the luma blend is a no-op,
    // kept explicit for parity with the shader.
    float luma = 0.2126f * v0 + 0.7152f * v1 + 0.0722f * v2;
    v0 = luma + (v0 - luma);
    v1 = luma + (v1 - luma);
    v2 = luma + (v2 - luma);

    // Outset matrix (rows).
    float o0 =  1.19687900512017f * v0 - 0.0980208811401368f * v1 - 0.0990297440797205f * v2;
    float o1 = -0.0528968517574562f * v0 + 1.15190312990417f * v1 - 0.0989611768448433f * v2;
    float o2 = -0.0529716355144438f * v0 - 0.0980434501171241f * v1 + 1.15107367264116f * v2;
    r = Clamp01(o0); g = Clamp01(o1); b = Clamp01(o2);
}

// Khronos PBR Neutral -- the official glTF reference operator.
void KhronosPbrNeutralTonemap(float& r, float& g, float& b) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation     = 0.15f;
    float x = std::min(r, std::min(g, b));
    float offset = (x < 0.08f) ? (x - 6.25f * x * x) : 0.04f;
    r -= offset; g -= offset; b -= offset;
    float peak = std::max(r, std::max(g, b));
    if (peak < startCompression) return;
    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    float s = newPeak / peak;
    r *= s; g *= s; b *= s;
    float gg = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    r = r + (newPeak - r) * gg;
    g = g + (newPeak - g) * gg;
    b = b + (newPeak - b) * gg;
}

// --- end Wave 9 tonemap ----------------------------------------------------

}  // namespace

std::uint32_t ParseTonemapOp(std::string_view op) {
    if (op == "agx")                 return 1u;
    if (op == "khronos_pbr_neutral") return 2u;
    if (op == "reinhard")            return 3u;
    if (op == "linear")              return 4u;
    return 0u;  // "aces" (default) + unknown
}

// Public per-pixel operator dispatch (declared in CaptureEncoder.h). op 0
// routes through the per-channel ACES path so it stays byte-identical to
// the pre-Wave-9 encoder. Inputs are linear-HDR (pre-exposure).
void TonemapHdrPixel(float r, float g, float b, float exposure,
                     std::uint32_t tonemap_op, std::uint8_t* dst) {
    if (tonemap_op > 4u) tonemap_op = 0u;
    r *= exposure; g *= exposure; b *= exposure;
    switch (tonemap_op) {
        case 1u: AgxTonemap(r, g, b); break;
        case 2u: KhronosPbrNeutralTonemap(r, g, b); break;
        case 3u: r = Clamp01(r / (1.0f + r));
                 g = Clamp01(g / (1.0f + g));
                 b = Clamp01(b / (1.0f + b)); break;
        case 4u: r = Clamp01(r); g = Clamp01(g); b = Clamp01(b); break;
        default: // aces -- per-channel, byte-identical to the legacy path.
                 r = Aces1(r); g = Aces1(g); b = Aces1(b); break;
    }
    dst[0] = SrgbQuantize(r);
    dst[1] = SrgbQuantize(g);
    dst[2] = SrgbQuantize(b);
}

std::vector<std::uint8_t> BuildSrgbBuffer(
    const std::vector<std::uint8_t>& raw,
    std::uint32_t                    w,
    std::uint32_t                    h,
    CaptureSourceKind                kind,
    float                            exposure,
    std::uint32_t                    tonemap_op) {
    // Defence-in-depth: clamp an out-of-range op to ACES (0) so a stray
    // value can't fall through the dispatch's default twice.
    if (tonemap_op > 4u) tonemap_op = 0u;
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
            // --- Wave 9 tonemap --- per-pixel operator dispatch. op 0
            // (aces) routes through the per-channel Aces1 path inside
            // TonemapHdrPixel, byte-identical to the previous AcesPlusSrgb
            // calls, so existing goldens are unchanged.
            TonemapHdrPixel(r, g, b, exposure, tonemap_op, dst);
        }
    }
    return rgb;
}

bool EncodeAndWritePpm(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure,
                       std::uint32_t                    tonemap_op) {
    const auto rgb = BuildSrgbBuffer(raw, w, h, kind, exposure, tonemap_op);
    return WriteRgb8(path, rgb.data(), w, h, OutputFormat::Ppm);
}

bool EncodeAndWritePng(const std::filesystem::path&     path,
                       const std::vector<std::uint8_t>& raw,
                       std::uint32_t                    w,
                       std::uint32_t                    h,
                       CaptureSourceKind                kind,
                       float                            exposure,
                       std::uint32_t                    tonemap_op) {
    const auto rgb = BuildSrgbBuffer(raw, w, h, kind, exposure, tonemap_op);
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

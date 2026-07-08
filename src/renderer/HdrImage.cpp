// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "HdrImage.h"

#include "../core/Diag.h"
#include "../core/Log.h"
#include "../core/Tracy.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <cstdio>
#include <cstring>

namespace pt::renderer {

namespace {

bool ReadLine(std::FILE* f, std::string& out) {
    out.clear();
    int c;
    while ((c = std::fgetc(f)) != EOF) {
        if (c == '\n') return true;
        out.push_back(static_cast<char>(c));
    }
    return !out.empty();
}

void DecodeRGBE(const std::uint8_t* src, float* dst) {
    if (src[3] == 0) {
        dst[0] = dst[1] = dst[2] = 0.0f;
        return;
    }
    const float scale = std::ldexp(1.0f, int(src[3]) - 128 - 8);
    dst[0] = float(src[0]) * scale;
    dst[1] = float(src[1]) * scale;
    dst[2] = float(src[2]) * scale;
}

// New (post-2.0) RLE format: each scanline starts with a 4-byte marker
// (2, 2, len_high, len_low) where len = width. Then 4 RLE-compressed
// channels (R, G, B, E) follow. Each channel is: a sequence of either
// runs (lead byte > 128: count = byte - 128, repeated value follows)
// or literals (lead byte <= 128: count = byte, that many raw values).
bool DecodeScanlineRle(std::FILE* f, std::uint32_t width,
                       std::vector<std::uint8_t>& scanline_rgbe) {
    if (width < 8 || width > 0x7fff) {
        // Outside the range the new-RLE encoding is defined for -- the
        // Radiance spec stores such scanlines FLAT (width*4 raw RGBE
        // bytes). Returning false here was treated as fatal by the
        // caller, so any legitimate .hdr narrower than 8 px or wider
        // than 32767 px refused to load with "scanline decode failed".
        scanline_rgbe.resize(std::size_t(width) * 4);
        return std::fread(scanline_rgbe.data(), 1, std::size_t(width) * 4, f)
               == std::size_t(width) * 4;
    }
    std::uint8_t marker[4];
    if (std::fread(marker, 1, 4, f) != 4) return false;
    if (marker[0] != 2 || marker[1] != 2 || (marker[2] & 0x80) != 0) {
        // Not new RLE format -- caller has to handle.
        // Push the 4 bytes back conceptually: re-decode them as the
        // first pixel.
        scanline_rgbe.assign(marker, marker + 4);
        scanline_rgbe.resize(std::size_t(width) * 4);
        if (std::fread(scanline_rgbe.data() + 4, 1, std::size_t(width) * 4 - 4, f)
                != std::size_t(width) * 4 - 4) return false;
        return true;
    }
    const std::uint32_t expected = (std::uint32_t(marker[2]) << 8) | marker[3];
    if (expected != width) return false;

    scanline_rgbe.assign(std::size_t(width) * 4, 0);
    for (int chan = 0; chan < 4; ++chan) {
        std::uint32_t pos = 0;
        while (pos < width) {
            std::uint8_t lead;
            if (std::fread(&lead, 1, 1, f) != 1) return false;
            if (lead > 128) {
                const std::uint32_t count = lead - 128u;
                if (count == 0 || pos + count > width) return false;
                std::uint8_t value;
                if (std::fread(&value, 1, 1, f) != 1) return false;
                for (std::uint32_t i = 0; i < count; ++i) {
                    scanline_rgbe[(std::size_t(pos + i) << 2) | chan] = value;
                }
                pos += count;
            } else {
                const std::uint32_t count = lead;
                if (count == 0 || pos + count > width) return false;
                for (std::uint32_t i = 0; i < count; ++i) {
                    std::uint8_t value;
                    if (std::fread(&value, 1, 1, f) != 1) return false;
                    scanline_rgbe[(std::size_t(pos + i) << 2) | chan] = value;
                }
                pos += count;
            }
        }
    }
    return true;
}

}  // namespace

HdrImage LoadRadianceHdr(const std::string& path, std::string* out_error) {
    PT_ZONE_SCOPED_N("renderer::LoadRadianceHdr");
    HdrImage img;
    auto fail = [&](const char* msg) -> HdrImage {
        if (out_error) *out_error = msg;
        LOG_WARN("HDR '{}': {}", path, msg);
        return HdrImage{};
    };

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return fail("cannot open file");

    // Header.
    std::string line;
    if (!ReadLine(f, line) || line.find("RADIANCE") == std::string::npos) {
        // Some files use "#?RGBE" which is also valid.
        if (line.find("RGBE") == std::string::npos &&
            line.find("RADIANCE") == std::string::npos) {
            std::fclose(f);
            return fail("not a Radiance HDR (missing #?RADIANCE)");
        }
    }
    std::string format;
    while (ReadLine(f, line)) {
        if (line.empty()) break;          // blank line ends header
        if (line.rfind("FORMAT=", 0) == 0) format = line.substr(7);
    }
    if (!format.empty() && format != "32-bit_rle_rgbe") {
        std::fclose(f);
        return fail("unsupported FORMAT (only 32-bit_rle_rgbe)");
    }

    // Resolution. We only support `-Y h +X w`; no rotated/flipped variants.
    if (!ReadLine(f, line)) { std::fclose(f); return fail("missing resolution line"); }
    int height = 0, width = 0;
    if (std::sscanf(line.c_str(), "-Y %d +X %d", &height, &width) != 2) {
        std::fclose(f);
        return fail("only -Y +X resolution layout supported");
    }
    if (width <= 0 || height <= 0) { std::fclose(f); return fail("bad dimensions"); }

    img.width  = static_cast<std::uint32_t>(width);
    img.height = static_cast<std::uint32_t>(height);
    img.rgb.assign(std::size_t(width) * height * 3, 0.0f);

    std::vector<std::uint8_t> scanline;
    for (int y = 0; y < height; ++y) {
        if (!DecodeScanlineRle(f, img.width, scanline)) {
            std::fclose(f);
            return fail("scanline decode failed");
        }
        float* row = img.rgb.data() + std::size_t(y) * width * 3;
        for (int x = 0; x < width; ++x) {
            DecodeRGBE(&scanline[std::size_t(x) * 4], &row[std::size_t(x) * 3]);
        }
    }

    std::fclose(f);
    if (out_error) out_error->clear();
    PT_DIAG_TIER1("renderer", "HDR '{}' loaded: {}x{}, {} pixels",
                  path, img.width, img.height,
                  img.width * img.height);
    return img;
}

double BuildEnvCdf(const std::vector<float>& rgb,
                   std::uint32_t W, std::uint32_t H,
                   const std::vector<std::uint8_t>& light_mask,
                   std::vector<float>& out_marginal,
                   std::vector<float>& out_conditional) {
    out_marginal.assign(H, 0.0f);
    out_conditional.assign(std::size_t(W) * H, 0.0f);
    const bool use_mask = !light_mask.empty();
    double total = 0.0;
    for (std::uint32_t v = 0; v < H; ++v) {
        // Lat-long solid-angle Jacobian: rows near the poles subtend
        // less solid angle, so weight by sin(theta) at the row centre.
        const double sin_theta =
            std::sin(std::numbers::pi * (double(v) + 0.5) / double(H));
        double row_sum = 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            const std::size_t pi = std::size_t(v) * W + u;
            const float r = rgb[pi * 3 + 0];
            const float g = rgb[pi * 3 + 1];
            const float b = rgb[pi * 3 + 2];
            const float lum_raw = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float lum = (use_mask && light_mask[pi]) ? 0.0f : lum_raw;
            row_sum += double(lum) * sin_theta;
            out_conditional[pi] = float(row_sum);   // unnormalized prefix sum
        }
        // Normalize within the row to [0, 1].
        const double norm = (row_sum > 0.0) ? (1.0 / row_sum) : 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            const std::size_t pi = std::size_t(v) * W + u;
            out_conditional[pi] = float(double(out_conditional[pi]) * norm);
        }
        out_marginal[v] = float(row_sum);
        total += row_sum;
    }
    // Marginal: prefix-sum + normalize so marginal[H-1] == 1.0.
    {
        const double norm = (total > 0.0) ? (1.0 / total) : 0.0;
        double prefix = 0.0;
        for (std::uint32_t v = 0; v < H; ++v) {
            prefix += out_marginal[v];
            out_marginal[v] = float(prefix * norm);
        }
        // Force the last entry to exactly 1.0 to absorb prefix-sum FP
        // drift -- but only when there IS luminance. An all-black env
        // map must yield an all-zero marginal (there is no drift to
        // guard, and "no light" should not look like a valid CDF); the
        // shader gates env-NEE on env_total_luminance <= 0 either way.
        if (H > 0 && total > 0.0) out_marginal[H - 1] = 1.0f;
    }
    return total;
}

}  // namespace pt::renderer

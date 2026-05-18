// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "MoonTexture.h"

#include "../core/Tracy.h"

#include <algorithm>
#include <cmath>

namespace pt::moon {

namespace {

float hash3d(float x, float y, float z) {
    // Simple sin-hash; biased but adequate for "scatter craters across
    // a sphere." Not the same hash as the cloud shader -- moon noise
    // is generated once at startup, no need to share GPU code.
    float h = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
    return h - std::floor(h);
}

float valueNoise3d(float x, float y, float z) {
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    int zi = static_cast<int>(std::floor(z));
    float fx = x - xi;
    float fy = y - yi;
    float fz = z - zi;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    fz = fz * fz * (3.0f - 2.0f * fz);
    auto h = [](int i, int j, int k) {
        return hash3d(static_cast<float>(i),
                      static_cast<float>(j),
                      static_cast<float>(k));
    };
    float a = h(xi,     yi,     zi);
    float b = h(xi + 1, yi,     zi);
    float c = h(xi,     yi + 1, zi);
    float d = h(xi + 1, yi + 1, zi);
    float e = h(xi,     yi,     zi + 1);
    float f = h(xi + 1, yi,     zi + 1);
    float g = h(xi,     yi + 1, zi + 1);
    float k = h(xi + 1, yi + 1, zi + 1);
    auto lerp1 = [](float p, float q, float t) { return p + (q - p) * t; };
    return lerp1(
        lerp1(lerp1(a, b, fx), lerp1(c, d, fx), fy),
        lerp1(lerp1(e, f, fx), lerp1(g, k, fx), fy),
        fz);
}

float fbm(float x, float y, float z, int octaves) {
    float v = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        v += amp * valueNoise3d(x * freq, y * freq, z * freq);
        amp  *= 0.5f;
        freq *= 2.02f;
    }
    return v;
}

}  // namespace

void generateMoonTexture(int width, int height, std::vector<float>& rgba_out) {
    PT_ZONE_SCOPED_N("moon::generateMoonTexture");
    rgba_out.assign(static_cast<std::size_t>(width) * height * 4, 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float u = (i + 0.5f) / float(width);            // 0..1
            float v = (j + 0.5f) / float(height);           // 0..1
            float lon = u * 2.0f * kPi - kPi;               // -pi..pi
            float lat = (0.5f - v) * kPi;                   // -pi/2..pi/2
            float cl = std::cos(lat);
            float px = cl * std::cos(lon);
            float py = std::sin(lat);
            float pz = cl * std::sin(lon);

            // Highland base: bright mottled grey-tan.
            float base = 0.85f + 0.13f * fbm(px * 8.0f, py * 8.0f, pz * 8.0f, 4);

            // Mare ("seas") -- large dark basaltic basins. Frequency
            // 0.6 (was 1.5) gives features that span a meaningful
            // fraction of the visible disc even at default r_moon_size,
            // so the dark patches don't average out under bilinear
            // filtering when the on-screen disc is only ~9-15px wide.
            // Mare albedo dropped to 0.10 for stronger contrast vs the
            // 0.85-0.98 highlands.
            float mare_n = fbm(px * 0.6f, py * 0.6f, pz * 0.6f, 3);
            float mare_factor = std::clamp((0.55f - mare_n) * 5.0f, 0.0f, 1.0f);
            base = base + (0.10f - base) * mare_factor;

            // Mid-frequency crater fields -- wider features so they
            // survive the bilinear average at small disc sizes.
            float crater_n = fbm(px * 14.0f, py * 14.0f, pz * 14.0f, 3);
            float crater_dark   = std::clamp(crater_n - 0.55f, 0.0f, 1.0f) * 0.35f;
            float crater_bright = std::clamp(0.40f - crater_n, 0.0f, 1.0f) * 0.25f;
            base = std::clamp(base - crater_dark + crater_bright, 0.08f, 1.00f);

            // Lunar regolith tint: slightly warm grey (sandy / tan).
            float r = base * 1.00f;
            float g = base * 0.95f;
            float b = base * 0.88f;

            int idx = (j * width + i) * 4;
            rgba_out[idx + 0] = r;
            rgba_out[idx + 1] = g;
            rgba_out[idx + 2] = b;
            rgba_out[idx + 3] = 1.0f;
        }
    }
}

}  // namespace pt::moon

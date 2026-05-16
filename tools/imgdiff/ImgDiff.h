// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Header-only image diff compute kernel. Pure RGBA8 -> stats; no I/O,
// no allocations beyond what the caller passes in. Lives in a header so
// both the imgdiff CLI driver and the doctest unit tests link the
// exact same code without a separate static lib in the way.
//
// Inputs are tightly-packed RGBA8 (4 bytes/pixel, no row padding). The
// caller (main.cpp via stb_image; tests via inline arrays) is responsible
// for the format conversion -- stb_image_load with desired_channels=4
// satisfies the contract.
//
// Threshold semantics, intentional and important to get right:
//
//   --max-delta N  : classifier. A pixel is "bad" iff its L2 distance
//                    exceeds N. NOT a hard pass/fail gate on its own.
//   --fail-percent P : pass requires badPixels/totalPixels * 100 <= P.
//   --mean-delta M : pass requires mean(L2) across all pixels <= M.
//
// So --max-delta+--fail-percent works together as the "% over tolerance"
// gate; --mean-delta is an independent ceiling on global drift even
// when no single pixel crosses the per-pixel classifier. With all three
// at 0 (the default), the only passing comparison is byte-identical.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pt::imgdiff {

struct DiffStats {
    // Max per-pixel L2 across the whole image. Informational; not gated
    // directly. Useful for tuning thresholds when adding new goldens.
    double   maxDelta    = 0.0;
    // Mean per-pixel L2. Gated by Thresholds::meanDelta.
    double   meanDelta   = 0.0;
    // sqrt( sum(L2(p)^2) / n ). Informational. Issue #69 lists this in
    // the stats summary; cheap to compute alongside the rest.
    double   rmsDelta    = 0.0;
    // Fraction of "bad" pixels, expressed 0..100. Gated by Thresholds::failPercent.
    double   badPercent  = 0.0;
    uint64_t totalPixels = 0;
    uint64_t badPixels   = 0;
};

struct Thresholds {
    // L2 distance above which a pixel is classified "bad". Default 0
    // -> any non-zero per-pixel diff is bad. Tighten upward when goldens
    // contain antialiasing or dithering noise.
    double maxDelta    = 0.0;
    double meanDelta   = 0.0;
    double failPercent = 0.0;
};

// L2 distance between two RGBA8 pixels treated as 4D vectors. Output
// is in [0, sqrt(4*255^2)] = [0, 510].
inline double PixelL2(const uint8_t* a, const uint8_t* b) {
    const int dr = int(a[0]) - int(b[0]);
    const int dg = int(a[1]) - int(b[1]);
    const int db = int(a[2]) - int(b[2]);
    const int da = int(a[3]) - int(b[3]);
    return std::sqrt(double(dr * dr + dg * dg + db * db + da * da));
}

// One-shot compare. `a` and `b` must each point to width*height*4 bytes
// of RGBA8. `outDiff`, if non-null, must also point to width*height*4
// bytes and receives a colorized heatmap of the per-pixel L2 distances
// (black=no diff, hot colors = larger diff). Returns aggregate stats.
inline DiffStats Compute(const uint8_t* a,
                          const uint8_t* b,
                          uint32_t width,
                          uint32_t height,
                          double badThreshold,
                          uint8_t* outDiff = nullptr) {
    DiffStats s;
    const uint64_t n = uint64_t(width) * uint64_t(height);
    s.totalPixels = n;
    if (n == 0) return s;

    double sumDelta   = 0.0;
    double sumSqDelta = 0.0;
    double maxDelta   = 0.0;
    uint64_t bad      = 0;

    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* pa = a + i * 4;
        const uint8_t* pb = b + i * 4;
        const double d = PixelL2(pa, pb);
        sumDelta   += d;
        sumSqDelta += d * d;
        if (d > maxDelta)     maxDelta = d;
        if (d > badThreshold) ++bad;
    }

    s.maxDelta   = maxDelta;
    s.meanDelta  = sumDelta / double(n);
    s.rmsDelta   = std::sqrt(sumSqDelta / double(n));
    s.badPixels  = bad;
    s.badPercent = (double(bad) / double(n)) * 100.0;

    if (outDiff) {
        // Per-pixel L2 normalized against the observed max delta and
        // mapped through a 3-segment color ramp (black -> purple ->
        // orange -> yellow). When two images are identical maxDelta is
        // 0 so the heatmap is all-black, which is the right read for
        // "no differences".
        const double scale = (maxDelta > 0.0) ? (1.0 / maxDelta) : 0.0;
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t* pa = a + i * 4;
            const uint8_t* pb = b + i * 4;
            const double d = PixelL2(pa, pb);
            const double t = std::clamp(d * scale, 0.0, 1.0);

            double r, g, bch;
            if (t < 0.33) {
                const double u = t / 0.33;
                r   = u * 128.0;
                g   = 0.0;
                bch = u * 192.0;
            } else if (t < 0.66) {
                const double u = (t - 0.33) / 0.33;
                r   = 128.0 + u * 127.0;
                g   = u * 80.0;
                bch = 192.0 * (1.0 - u);
            } else {
                const double u = (t - 0.66) / 0.34;
                r   = 255.0;
                g   = 80.0 + u * 175.0;
                bch = u * 100.0;
            }

            // Round-on-cast (not truncate) when packing into uint8_t.
            // At the ramp endpoints, t = d * scale = maxDelta / maxDelta
            // is 1 ULP under 1.0 in double precision, so u in the upper
            // branch lands at 0.99999... and naive trunc-casts give
            // 254/99 instead of the intended 255/100 at the hottest
            // pixel. Rounding eliminates that one-bit loss without
            // changing the ramp shape elsewhere.
            uint8_t* po = outDiff + i * 4;
            po[0] = uint8_t(std::clamp(r,   0.0, 255.0) + 0.5);
            po[1] = uint8_t(std::clamp(g,   0.0, 255.0) + 0.5);
            po[2] = uint8_t(std::clamp(bch, 0.0, 255.0) + 0.5);
            po[3] = 255;
        }
    }
    return s;
}

inline bool Passes(const DiffStats& s, const Thresholds& t) {
    if (s.meanDelta  > t.meanDelta)   return false;
    if (s.badPercent > t.failPercent) return false;
    return true;
}

} // namespace pt::imgdiff

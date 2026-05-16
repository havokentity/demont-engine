// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::imgdiff::Compute / Passes. Exercises the pure
// compute kernel directly with hand-crafted RGBA8 buffers -- no PNG
// I/O involved, so the test is fast and entirely deterministic. The
// imgdiff CLI driver in tools/imgdiff/main.cpp adds stb_image load +
// argv parsing on top; those layers are exercised by the smoke test
// in CI's golden-image workflow (#45) rather than here.
//
// Coverage:
//   - Identical buffers -> zero stats, passes by default.
//   - Single pixel delta -> max/mean reflect it; percentage gate.
//   - badThreshold classifier behaviour (pixel-by-pixel).
//   - mean-delta gate independent of bad-pixel gate.
//   - 0x0 empty image is safe (no UB, returns zero stats).
//   - Heatmap buffer is populated when supplied.
//   - Channel-swap regression (R<->B) is caught.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../tools/imgdiff/ImgDiff.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using pt::imgdiff::Compute;
using pt::imgdiff::DiffStats;
using pt::imgdiff::Passes;
using pt::imgdiff::PixelL2;
using pt::imgdiff::Thresholds;

namespace {

// Build a width*height RGBA8 image filled with a single color. Helper
// keeps the tests readable -- the comparisons are what the test is
// about, not the buffer plumbing.
std::vector<uint8_t> SolidImage(uint32_t w, uint32_t h, uint8_t r, uint8_t g,
                                 uint8_t b, uint8_t a = 255) {
    std::vector<uint8_t> v(size_t(w) * size_t(h) * 4u);
    for (size_t i = 0; i < v.size(); i += 4) {
        v[i + 0] = r;
        v[i + 1] = g;
        v[i + 2] = b;
        v[i + 3] = a;
    }
    return v;
}

} // namespace

TEST_CASE("imgdiff: PixelL2 sanity") {
    const std::array<uint8_t, 4> p0 = {0,   0, 0, 255};
    const std::array<uint8_t, 4> p1 = {0,   0, 0, 255};
    CHECK(PixelL2(p0.data(), p1.data()) == doctest::Approx(0.0));

    // Red(255) vs black -> L2 = sqrt(255^2) = 255.
    const std::array<uint8_t, 4> red   = {255, 0, 0, 255};
    const std::array<uint8_t, 4> black = {0,   0, 0, 255};
    CHECK(PixelL2(red.data(), black.data()) == doctest::Approx(255.0));

    // 3-3-4 diff -> sqrt(9 + 9 + 16) = sqrt(34).
    const std::array<uint8_t, 4> a = {10, 10, 10, 255};
    const std::array<uint8_t, 4> b = {13, 13, 14, 255};
    CHECK(PixelL2(a.data(), b.data()) == doctest::Approx(std::sqrt(34.0)));
}

TEST_CASE("imgdiff: identical images pass with default thresholds") {
    const auto img = SolidImage(8, 8, 128, 64, 200);

    const DiffStats s = Compute(img.data(), img.data(), 8, 8, 0.0);
    CHECK(s.totalPixels == 64);
    CHECK(s.badPixels   == 0);
    CHECK(s.maxDelta    == doctest::Approx(0.0));
    CHECK(s.meanDelta   == doctest::Approx(0.0));
    CHECK(s.rmsDelta    == doctest::Approx(0.0));
    CHECK(s.badPercent  == doctest::Approx(0.0));

    const Thresholds t{}; // all zeros -- strictest
    CHECK(Passes(s, t));
}

TEST_CASE("imgdiff: single-pixel delta is detected") {
    // 10x10 image, one pixel made red instead of black.
    auto golden = SolidImage(10, 10, 0,   0, 0);
    auto actual = SolidImage(10, 10, 0,   0, 0);
    // Pixel (3, 4): index 4*10 + 3 = 43, byte offset 43*4 = 172.
    actual[172 + 0] = 255;

    const DiffStats s = Compute(actual.data(), golden.data(), 10, 10, 0.0);
    CHECK(s.totalPixels == 100);
    CHECK(s.badPixels   == 1);
    CHECK(s.maxDelta    == doctest::Approx(255.0));
    // Mean: 255 / 100 = 2.55.
    CHECK(s.meanDelta   == doctest::Approx(2.55));
    // RMS: sqrt(255^2 / 100) = 25.5.
    CHECK(s.rmsDelta    == doctest::Approx(25.5));
    // 1 of 100 = 1%.
    CHECK(s.badPercent  == doctest::Approx(1.0));

    // Strictest thresholds fail.
    Thresholds strict{};
    CHECK_FALSE(Passes(s, strict));

    // Loosen fail-percent above 1% but keep mean strict -> still fails
    // on mean.
    Thresholds onePctFail = strict;
    onePctFail.failPercent = 1.5;
    CHECK_FALSE(Passes(s, onePctFail));

    // Loosen both -> passes.
    Thresholds loose;
    loose.failPercent = 1.5;
    loose.meanDelta   = 3.0;
    CHECK(Passes(s, loose));
}

TEST_CASE("imgdiff: badThreshold classifier excludes small deltas") {
    // 4x1 image. Differences of 0 / 3 / 7 / 50 between actual & golden.
    std::vector<uint8_t> golden = {0, 0, 0, 255,
                                    0, 0, 0, 255,
                                    0, 0, 0, 255,
                                    0, 0, 0, 255};
    std::vector<uint8_t> actual = {0, 0, 0, 255,    // delta 0
                                    3, 0, 0, 255,    // delta 3
                                    7, 0, 0, 255,    // delta 7
                                    50, 0, 0, 255};  // delta 50

    SUBCASE("badThreshold = 0: every non-zero diff is bad") {
        const DiffStats s = Compute(actual.data(), golden.data(), 4, 1, 0.0);
        CHECK(s.badPixels == 3);
        CHECK(s.badPercent == doctest::Approx(75.0));
    }
    SUBCASE("badThreshold = 5: only deltas > 5 are bad") {
        const DiffStats s = Compute(actual.data(), golden.data(), 4, 1, 5.0);
        CHECK(s.badPixels == 2);
        CHECK(s.badPercent == doctest::Approx(50.0));
    }
    SUBCASE("badThreshold = 100: nothing is bad") {
        const DiffStats s = Compute(actual.data(), golden.data(), 4, 1, 100.0);
        CHECK(s.badPixels == 0);
        CHECK(s.badPercent == doctest::Approx(0.0));
        // maxDelta still reports the largest L2 observed, independent
        // of the classifier.
        CHECK(s.maxDelta == doctest::Approx(50.0));
    }
}

TEST_CASE("imgdiff: percentage threshold edge cases") {
    // 100-pixel image (10x10). Make exactly 5 pixels bad with delta=10.
    auto golden = SolidImage(10, 10, 0, 0, 0);
    auto actual = golden;
    for (int i = 0; i < 5; ++i) {
        actual[i * 4 + 0] = 10;
    }

    const DiffStats s = Compute(actual.data(), golden.data(), 10, 10, 0.0);
    CHECK(s.badPixels  == 5);
    CHECK(s.badPercent == doctest::Approx(5.0));

    // Boundary: failPercent exactly == observed badPercent -> passes.
    // The gate uses `>`, not `>=`, so 5.0 <= 5.0 is OK.
    Thresholds atBoundary;
    atBoundary.failPercent = 5.0;
    atBoundary.meanDelta   = 1.0;  // mean = 50/100 = 0.5; well under.
    CHECK(Passes(s, atBoundary));

    // Just below the observed value -> fails.
    Thresholds justBelow = atBoundary;
    justBelow.failPercent = 4.999;
    CHECK_FALSE(Passes(s, justBelow));
}

TEST_CASE("imgdiff: mean-delta gate is independent of fail-percent") {
    // Construct a case where bad-percent is 0 (no pixel above threshold)
    // but mean is non-zero -- only possible if badThreshold is high
    // enough that all the small deltas are "good" but they still
    // accumulate into a non-zero mean.
    auto golden = SolidImage(4, 4, 0, 0, 0);
    auto actual = SolidImage(4, 4, 1, 0, 0);  // every pixel delta=1

    // badThreshold = 5, so all 16 pixels are "good" -> badPixels = 0.
    const DiffStats s = Compute(actual.data(), golden.data(), 4, 4, 5.0);
    CHECK(s.badPixels  == 0);
    CHECK(s.maxDelta   == doctest::Approx(1.0));
    CHECK(s.meanDelta  == doctest::Approx(1.0));
    CHECK(s.badPercent == doctest::Approx(0.0));

    // fail-percent gate passes (0 <= 0), but mean-delta gate fails.
    Thresholds t;
    t.failPercent = 0.0;
    t.meanDelta   = 0.5;
    CHECK_FALSE(Passes(s, t));

    // Raise mean-delta tolerance above 1.0 -> passes.
    t.meanDelta = 1.0;
    CHECK(Passes(s, t));
}

TEST_CASE("imgdiff: empty image is safe") {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    const DiffStats s = Compute(a.data(), b.data(), 0, 0, 0.0);
    CHECK(s.totalPixels == 0);
    CHECK(s.badPixels   == 0);
    CHECK(s.maxDelta    == doctest::Approx(0.0));
    CHECK(s.meanDelta   == doctest::Approx(0.0));
    CHECK(s.rmsDelta    == doctest::Approx(0.0));
    CHECK(s.badPercent  == doctest::Approx(0.0));
    // Defaults pass: there's nothing to fail on.
    CHECK(Passes(s, Thresholds{}));
}

TEST_CASE("imgdiff: heatmap is populated when buffer is provided") {
    auto golden = SolidImage(2, 2, 0, 0, 0);
    auto actual = SolidImage(2, 2, 0, 0, 0);
    // One pixel pure red (large delta), one pixel slightly red (small).
    actual[0]  = 255;  // pixel 0: full red
    actual[4]  = 16;   // pixel 1: small delta
    // pixels 2, 3: unchanged (delta 0).

    std::vector<uint8_t> heatmap(2 * 2 * 4, 0xCC);
    const DiffStats s = Compute(actual.data(), golden.data(), 2, 2, 0.0,
                                 heatmap.data());

    // The identical pixels (2, 3) should map to black (0, 0, 0) -- the
    // ramp at t=0 returns all zeros for RGB. Alpha is always 255.
    CHECK(heatmap[2 * 4 + 0] == 0);
    CHECK(heatmap[2 * 4 + 1] == 0);
    CHECK(heatmap[2 * 4 + 2] == 0);
    CHECK(heatmap[2 * 4 + 3] == 255);
    CHECK(heatmap[3 * 4 + 0] == 0);
    CHECK(heatmap[3 * 4 + 3] == 255);

    // The hottest pixel (pixel 0, delta = maxDelta = 255) should map to
    // the top of the ramp -- R=255, G=255, B=100. Test the exact upper
    // values so a future ramp tweak doesn't go unnoticed.
    CHECK(heatmap[0 * 4 + 0] == 255);
    CHECK(heatmap[0 * 4 + 1] == 255);
    CHECK(heatmap[0 * 4 + 2] == 100);
    CHECK(heatmap[0 * 4 + 3] == 255);

    // Some non-zero output for the small-delta pixel (1).
    const bool anySmallNonZero =
        heatmap[1 * 4 + 0] != 0 ||
        heatmap[1 * 4 + 1] != 0 ||
        heatmap[1 * 4 + 2] != 0;
    CHECK(anySmallNonZero);

    // Sanity on the stats while we're here.
    CHECK(s.maxDelta == doctest::Approx(255.0));
}

TEST_CASE("imgdiff: catches R<->B channel swap") {
    // Regression-spec case from #69: a channel-swap mistake (the
    // renderer producing BGRA where the golden is RGBA) should fail
    // even though luminance / hash-distance would be 'similar'.
    auto golden = SolidImage(4, 4, 200, 50, 10);
    auto actual = SolidImage(4, 4, 10, 50, 200);  // R and B swapped

    const DiffStats s = Compute(actual.data(), golden.data(), 4, 4, 0.0);
    // L2 per pixel = sqrt((200-10)^2 + (50-50)^2 + (10-200)^2 + 0^2)
    //              = sqrt(190^2 + 190^2) = 190 * sqrt(2)
    const double expected = 190.0 * std::sqrt(2.0);
    CHECK(s.maxDelta  == doctest::Approx(expected));
    CHECK(s.meanDelta == doctest::Approx(expected));
    CHECK(s.badPixels == 16);
    CHECK_FALSE(Passes(s, Thresholds{}));
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Numeric regression test for the Hosek-Wilkie sky cook (engine/HosekSkyModel.h).
//
// This pins the PHYSICS the previous hand-fabricated coefficient tables got
// wrong -- the reason this whole file exists. The old tables were round
// hand-authored numbers falsely attributed to a "Helmer fit"; they crushed
// the blue channel at the zenith (dark olive sky) and, at high turbidity,
// drove the zenith blue channel negative so max(F,0) clamped it to exactly 0.
// The fix embeds the real ArHosekSkyModel RGB dataset and cooks it exactly
// like the reference. These checks would catch either a regression back to a
// non-blue zenith OR silent corruption of the embedded dataset -- WITHOUT
// needing the GPU golden (which only runs on the user's Mac).
//
// All reference values are hardcoded (computed once from the reference
// implementation, cross-checked by the CPU cook): no clock, no network. Same
// source + same build => same numbers on every platform.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/engine/HosekSkyModel.h"

#include <cmath>

namespace {

constexpr double kPi     = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

// Straight-up (zenith) radiance for a sun at `elev_deg` above the horizon.
// theta = 0 (cos_theta = 1); gamma (zenith-to-sun angle) = 90deg - elev.
struct RGB { double r, g, b; };
RGB zenithRGB(double turbidity, double albedo, double elev_deg) {
    const double elev = elev_deg * kPi / 180.0;
    const pt::hosek::Cooked ck = pt::hosek::Cook(turbidity, albedo, elev);
    const double gamma = kHalfPi - elev;
    const double cg    = std::cos(gamma);
    RGB out{};
    out.r = pt::hosek::RadianceInternal(ck.cfg[0], 1.0, cg, gamma) * ck.rad[0];
    out.g = pt::hosek::RadianceInternal(ck.cfg[1], 1.0, cg, gamma) * ck.rad[1];
    out.b = pt::hosek::RadianceInternal(ck.cfg[2], 1.0, cg, gamma) * ck.rad[2];
    return out;
}

}  // namespace

TEST_CASE("hosek zenith is blue-dominant on a clear day") {
    // T=3 (clear day), typical soil albedo, sun at 50deg. The real dataset
    // gives a strongly blue zenith; the old fabricated tables gave b/g ~0.31.
    const RGB z = zenithRGB(3.0, 0.1, 50.0);
    CHECK(z.b > z.g);            // blue dominates green
    CHECK(z.g > z.r);            // green dominates red -> proper sky hue
    CHECK(z.b / z.r > 2.5);      // strongly blue (reference ~3.33)
    CHECK(z.b / z.g > 1.6);      // reference ~2.0
    // Spot-check absolute magnitudes against the reference cook so a corrupted
    // dataset (wrong parse / truncated table) is caught, not just the ratios.
    CHECK(z.b == doctest::Approx(8.604).epsilon(0.01));
    CHECK(z.g == doctest::Approx(4.283).epsilon(0.01));
    CHECK(z.r == doctest::Approx(2.582).epsilon(0.01));
}

TEST_CASE("hosek zenith blue never clamps to zero as turbidity rises") {
    // The specific failure the audit flagged: at T=8..10 the old tables drove
    // the zenith blue channel negative (clamped to 0 -> pure yellow-grey sky).
    // The real model desaturates toward white but keeps every channel > 0.
    for (double T : {1.0, 3.0, 6.0, 8.0, 10.0}) {
        const RGB z = zenithRGB(T, 0.1, 50.0);
        CAPTURE(T);
        CHECK(z.r > 0.0);
        CHECK(z.g > 0.0);
        CHECK(z.b > 0.0);
        CHECK(z.b >= z.g);       // blue stays >= green at every turbidity
    }
    // Monotone desaturation: the blue/green ratio must fall as haze rises
    // (never invert, which is what the fabricated tables did).
    const double bg_clear = [] { auto z = zenithRGB(2.0, 0.1, 50.0); return z.b / z.g; }();
    const double bg_hazy  = [] { auto z = zenithRGB(9.0, 0.1, 50.0); return z.b / z.g; }();
    CHECK(bg_clear > bg_hazy);
    CHECK(bg_hazy >= 1.0);       // still (barely) blue at heavy haze, not olive
}

TEST_CASE("hosek ground albedo brightens the sky (physically correct sign)") {
    // Higher ground albedo -> more upwelling light bounced back down -> a
    // BRIGHTER dome. The audit found the old response was near-inert and
    // partly backwards; the real dataset's albedo blend gets the sign right.
    const RGB dark  = zenithRGB(3.0, 0.0, 50.0);
    const RGB bright = zenithRGB(3.0, 0.8, 50.0);
    const double lum_dark   = 0.2126 * dark.r   + 0.7152 * dark.g   + 0.0722 * dark.b;
    const double lum_bright = 0.2126 * bright.r + 0.7152 * bright.g + 0.0722 * bright.b;
    CHECK(lum_bright > lum_dark);
    // And it stays blue-dominant (albedo must not wash the hue to grey here).
    CHECK(bright.b > bright.g);
}

TEST_CASE("hosek cook stays finite at the domain edges") {
    // Turbidity/elevation clamping must never index the dataset out of range
    // or pow() a negative base (NaN). Probe the corners + out-of-range inputs.
    for (double T : {1.0, 10.0, 0.5, 12.0}) {
        for (double e : {0.0, 90.0, -5.0, 95.0}) {
            const RGB z = zenithRGB(T, 0.1, e);
            CAPTURE(T); CAPTURE(e);
            CHECK(std::isfinite(z.r));
            CHECK(std::isfinite(z.g));
            CHECK(std::isfinite(z.b));
        }
    }
}

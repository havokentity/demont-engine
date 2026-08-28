// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P5 (#259): the planetary ocean.
//
// WHAT THIS FILE IS FOR
//
// The phase's headline acceptance criterion is a claim about a SEQUENCE and
// not about a frame: "the ocean reads as water from a boat AND as a smooth
// specular sphere from orbit, with no visible transition between them". A
// golden PNG cannot settle that. Two PNGs at the ends of the climb cannot
// either -- they can both be right while everything between them jumps.
//
// So the claim is turned into numbers, three ways, in increasing strength:
//
//   1. THE ENDPOINTS ARE MEASURED CONSTANTS. At infinite footprint the
//      surface's microfacet alpha^2 is exactly Cox & Munk's 1954 sea-slope
//      variance for the wind. That is an oceanographic measurement, not a
//      dial, and the from-orbit golden pair (r_ocean 1 vs r_ocean 0) is the
//      rendered form of the same statement.
//
//   2. THE TWO HALVES SUM TO IT AT EVERY POINT BETWEEN. sigma^2_geometry(w)
//      + alpha^2_brdf(w) == sigma^2_CoxMunk for every footprint w. This is
//      an identity, so it is asserted with no tolerance beyond float
//      rounding. A "fade" cannot satisfy it; only a handover can.
//
//   3. AND THE PATH BETWEEN THEM IS LIPSCHITZ. #260's refinement test, which
//      needs no tolerance at all: sample alpha^2 along a geometric ladder in
//      footprint, and the largest adjacent step must HALVE as the rungs
//      double. A continuous function shrinks; a jump plateaus. The control
//      case in this file is the thing P5 must not be -- a hard switch at
//      some distance -- and it is asserted to plateau, so the test cannot go
//      vacuous by measuring nothing.
//
// The rest pins the pieces those three rest on: the cascade band partition,
// the anchor lattice's exactness, and the crossing test that has been
// above-to-below only since #25.
//
// WHY IT MIRRORS THE SHADER
//
// oceanCascadeResolved and oceanBrdfAlpha2 live in PathTrace.slang and there
// is no host entry point to call, so this file transcribes them -- same
// operations, same order -- exactly as tests/pt_atmosphere_test.cpp and
// tests/pt_math_altitude_test.cpp do for their kernels. A mirror that has
// drifted is worthless, so the last case re-reads the .slang and pins what
// the transcription depends on. It COUNTS occurrences rather than testing
// find() != npos: a substring pin is satisfied by one correct copy however
// many wrong ones exist elsewhere, which is exactly how issue #276 stayed
// live for a whole cycle underneath a passing test.
//
// The tangent-frame lattice is NOT mirrored -- pt::ocean::OceanTangentAnchor
// is a free function over glm precisely so the test can call the real code.
//
// Deterministic: every input is a literal or derived from literals.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/physics/OceanFFT.h"
#include "../src/renderer/Planet/CubedSphere.h"
#include "../src/renderer/Planet/TerrainChunk.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using pt::ocean::CoxMunkAlpha;
using pt::ocean::CoxMunkMeanSquareSlope;
using pt::ocean::CoxMunkRoughness;
using pt::ocean::OceanFFT;

namespace {

// --- The cascade layout, mirroring Engine.h ------------------------------
// Engine.h's kOceanCascadePeriodM / kOceanBandDivisor. Pinned against the
// header text in the mirror case below.
constexpr int    kCascades      = 3;
constexpr double kPeriodM[3]    = {1793.0, 211.0, 23.0};
constexpr double kBandDivisor   = 8.0;
constexpr double kDefaultWind   = 12.0;

double bandHi(int c) {
    return (c == 0) ? kPeriodM[0] : kPeriodM[c] / kBandDivisor;
}
double bandLo(int c, unsigned grid) {
    return (c + 1 < kCascades)
               ? kPeriodM[c + 1] / kBandDivisor
               : 2.0 * kPeriodM[c] / static_cast<double>(grid);
}

// --- shader mirror: oceanCascadeResolved (PathTrace.slang) ---------------
//
//   float P  = oceanCascadePeriod(c);
//   float hi = (c == 0u) ? P : (P / kOceanBandDivisor);
//   float lo = (c + 1u < count) ? (oceanCascadePeriod(c + 1u) / kOceanBandDivisor)
//                               : (2.0 * P / float(max(grid, 1u)));
//   if (!(hi > lo)) return 1.0;
//   float wc = clamp(w, lo, hi);
//   return log(hi / wc) / log(hi / lo);
double oceanCascadeResolved(int c, unsigned grid, double w) {
    const double hi = bandHi(c);
    const double lo = bandLo(c, grid);
    if (!(hi > lo)) return 1.0;
    const double wc = std::clamp(w, lo, hi);
    return std::log(hi / wc) / std::log(hi / lo);
}

// --- shader mirror: oceanBrdfAlpha2 (PathTrace.slang) --------------------
//
//   float geom = dot(resolved, ocean_slope.xyz);
//   return max(ocean_slope.w + planet_ocean.z - geom, 0.0);
double oceanBrdfAlpha2(const double sigma2_c[3], double cox_munk,
                       double alpha2_base, unsigned grid, double w) {
    double geom = 0.0;
    for (int c = 0; c < kCascades; ++c) {
        geom += oceanCascadeResolved(c, grid, w) * sigma2_c[c];
    }
    return std::max(alpha2_base + cox_munk - geom, 0.0);
}

double geometrySlopeVariance(const double sigma2_c[3], unsigned grid,
                             double w) {
    double geom = 0.0;
    for (int c = 0; c < kCascades; ++c) {
        geom += oceanCascadeResolved(c, grid, w) * sigma2_c[c];
    }
    return geom;
}

// The cascade slope budget the engine actually pushes: the Cox-Munk total,
// times the log-share of it the resolvable band holds, split across the
// three cascades in proportion to their own band log-lengths. Derived from
// the same law the shader's ramp is, so the two cannot disagree about the
// endpoints.
void cascadeSigma2(unsigned grid, double wind, double out[3]) {
    const double total = CoxMunkMeanSquareSlope(wind) *
                         pt::ocean::SlopeVarianceFractionInBand(
                             bandLo(kCascades - 1, grid), kPeriodM[0], wind);
    double logs[3];
    double sum = 0.0;
    for (int c = 0; c < kCascades; ++c) {
        const double hi = std::min(bandHi(c),
                                   pt::ocean::PiersonMoskowitzPeakWavelengthM(wind));
        const double lo = bandLo(c, grid);
        logs[c] = (hi > lo) ? std::log(hi / lo) : 0.0;
        sum += logs[c];
    }
    for (int c = 0; c < kCascades; ++c) {
        out[c] = (sum > 0.0) ? total * logs[c] / sum : 0.0;
    }
}

std::string tighten(const char* path) {
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    std::string tight;
    tight.reserve(src.size());
    for (char ch : src) {
        if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
    }
    return tight;
}

std::size_t countOf(const std::string& hay, const std::string& needle) {
    std::size_t n = 0, at = 0;
    while ((at = hay.find(needle, at)) != std::string::npos) { ++n; ++at; }
    return n;
}

}  // namespace

// ===========================================================================
// 1. THE ENDPOINT: Cox & Munk 1954, and what "roughness" means here.
// ===========================================================================

TEST_CASE("Cox-Munk sea slope is the constant the ocean converges to") {
    // Cox, C. & Munk, W. (1954), J. Opt. Soc. Am. 44(11) 838-850, table 1,
    // clean surface: sigma^2_total = 0.003 + 0.00512 U.
    CHECK(CoxMunkMeanSquareSlope(0.0)  == doctest::Approx(0.003));
    CHECK(CoxMunkMeanSquareSlope(10.0) == doctest::Approx(0.0542));
    CHECK(CoxMunkMeanSquareSlope(12.0) == doctest::Approx(0.06444));
    // Linear in wind, which is the whole content of the fit.
    const double d1 = CoxMunkMeanSquareSlope(11.0) - CoxMunkMeanSquareSlope(10.0);
    const double d2 = CoxMunkMeanSquareSlope(21.0) - CoxMunkMeanSquareSlope(20.0);
    CHECK(d1 == doctest::Approx(0.00512));
    CHECK(d2 == doctest::Approx(d1));
    // Negative wind is not an input; it must not produce a negative
    // variance that later sqrts to a NaN.
    CHECK(CoxMunkMeanSquareSlope(-5.0) == doctest::Approx(0.003));

    // Beckmann's slope distribution is a 2-D Gaussian of variance alpha^2/2
    // per axis, so its TOTAL mean-square slope is alpha^2 and matching the
    // measurement gives alpha = sigma.
    CHECK(CoxMunkAlpha(12.0) == doctest::Approx(std::sqrt(0.06444)));
    CHECK(CoxMunkAlpha(12.0) == doctest::Approx(0.253850).epsilon(1e-4));

    // ...and PathTrace.slang's anisoAlpha() squares the material's
    // roughness to get alpha, so the engine-facing number is sqrt(alpha).
    CHECK(CoxMunkRoughness(12.0) == doctest::Approx(std::sqrt(CoxMunkAlpha(12.0))));
    CHECK(CoxMunkRoughness(12.0) == doctest::Approx(0.503836).epsilon(1e-4));

    // THE 0.18 IN THE ISSUE IS NOT THIS NUMBER, and the difference is not
    // rounding. 0.18 is sqrt(sigma^2 / 2) -- the PER-AXIS slope RMS -- which
    // would be the answer only if `roughness` were alpha directly AND
    // alpha^2 were the per-axis variance. Neither holds here, and the two
    // mistakes do not cancel: they compound to sqrt(2) x sqrt(2). Pinned so
    // the discrepancy is a recorded decision and not a drift.
    CHECK(std::sqrt(0.06444 / 2.0) == doctest::Approx(0.17950).epsilon(1e-3));
    CHECK(CoxMunkRoughness(12.0) > 2.0 * std::sqrt(0.06444 / 2.0) * 0.5);

    // The glitter half-angle Cox & Munk photographed: atan(sigma) = 14.3 deg.
    const double half_angle_deg =
        std::atan(std::sqrt(CoxMunkMeanSquareSlope(12.0))) * 180.0 / 3.14159265358979;
    CHECK(half_angle_deg == doctest::Approx(14.24).epsilon(2e-3));
}

TEST_CASE("the slope band has two ends and both are measured") {
    // Gravity-capillary transition: the wavelength of minimum phase speed,
    // lambda_m = 2 pi sqrt(sigma / (rho g)) = 1.71 cm (Lamb 1932 sec. 267).
    CHECK(pt::ocean::GravityCapillaryWavelengthM() ==
          doctest::Approx(0.017117).epsilon(1e-3));

    // Pierson & Moskowitz 1964 fully-developed peak,
    // lambda_p = 2 pi U^2 / (0.877^2 g).
    CHECK(pt::ocean::PiersonMoskowitzPeakWavelengthM(12.0) ==
          doctest::Approx(119.93).epsilon(1e-3));
    // Quadratic in wind -- doubling U quadruples the peak wavelength, which
    // is why a 25 m/s sea would outgrow a band top capped at a fixed length
    // and why cascade 0 is left unbounded above.
    CHECK(pt::ocean::PiersonMoskowitzPeakWavelengthM(24.0) ==
          doctest::Approx(4.0 * pt::ocean::PiersonMoskowitzPeakWavelengthM(12.0)));

    // The fraction of Cox & Munk's constant a window holds. The whole band
    // is 1 by construction; a window outside it is 0; and the default
    // cascade set at grid 256 holds 73%.
    CHECK(pt::ocean::SlopeVarianceFractionInBand(1e-6, 1e6, 12.0) ==
          doctest::Approx(1.0));
    CHECK(pt::ocean::SlopeVarianceFractionInBand(1000.0, 5000.0, 12.0) ==
          doctest::Approx(0.0));
    const double f256 =
        pt::ocean::SlopeVarianceFractionInBand(bandLo(2, 256u), kPeriodM[0], 12.0);
    CHECK(f256 == doctest::Approx(0.734).epsilon(2e-2));
    // A COARSER grid resolves less and hands more to the BRDF. That
    // monotonicity is the reason the split is computed from the grid rather
    // than baked: it has to be right at every r_ocean_grid_size.
    const double f64 =
        pt::ocean::SlopeVarianceFractionInBand(bandLo(2, 64u), kPeriodM[0], 12.0);
    CHECK(f64 < f256);
}

// ===========================================================================
// 2. THE IDENTITY: geometry + BRDF == the measured constant, everywhere.
// ===========================================================================

TEST_CASE("the geometry-to-BRDF handover conserves the Cox-Munk constant") {
    const unsigned grid = 256u;
    const double   cm   = CoxMunkMeanSquareSlope(kDefaultWind);
    double sigma2[3];
    cascadeSigma2(grid, kDefaultWind, sigma2);

    // A ladder of footprints spanning eleven decades: a millimetre at the
    // bow of a boat to a 100 km pixel from beyond geostationary.
    for (int i = 0; i <= 220; ++i) {
        const double w = 1e-3 * std::pow(10.0, double(i) / 20.0);
        CAPTURE(w);
        const double geom  = geometrySlopeVariance(sigma2, grid, w);
        const double brdf  = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, w);
        // The identity. No tolerance beyond double rounding: this is a
        // rearrangement of one expression, not two models agreeing.
        CHECK(geom + brdf == doctest::Approx(cm).epsilon(1e-12));
        // And neither half is ever negative, which is what makes the split
        // a partition rather than an algebraic accident.
        CHECK(geom >= 0.0);
        CHECK(brdf >= 0.0);
    }
}

TEST_CASE("the ocean converges to Cox-Munk from orbit and to a near-mirror at the bow") {
    const unsigned grid = 256u;
    const double   cm   = CoxMunkMeanSquareSlope(kDefaultWind);
    double sigma2[3];
    cascadeSigma2(grid, kDefaultWind, sigma2);

    // From orbit the cone outgrows every cascade and the BRDF carries the
    // whole measured constant. 2 460 m is the pixel footprint at 400 km on
    // a 1080p 60-degree camera (400e3 * 1.07e-3 * ... -- the number that
    // matters is that it is far past cascade 0's 1793 m period).
    const double far = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, 1.0e5);
    CHECK(far == doctest::Approx(cm).epsilon(1e-12));
    CHECK(std::sqrt(std::sqrt(far)) ==
          doctest::Approx(CoxMunkRoughness(kDefaultWind)).epsilon(1e-9));

    // At the bow the cascades carry everything they can and what is left is
    // the sub-grid band -- short gravity waves and capillary ripples, which
    // are genuinely below a 9 cm grid spacing and genuinely still there.
    const double near_a2 = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, 0.0);
    CHECK(near_a2 > 0.0);
    CHECK(near_a2 < 0.4 * cm);
    // ...and it is exactly the complement of the resolved fraction.
    const double f = pt::ocean::SlopeVarianceFractionInBand(
        bandLo(2, grid), kPeriodM[0], kDefaultWind);
    CHECK(near_a2 == doctest::Approx(cm * (1.0 - f)).epsilon(1e-9));

    // Monotone in between: a wider footprint never resolves MORE.
    double prev = -1.0;
    for (int i = 0; i <= 400; ++i) {
        const double w = 1e-3 * std::pow(10.0, double(i) / 40.0);
        const double a2 = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, w);
        CAPTURE(w);
        CHECK(a2 >= prev - 1e-15);
        prev = a2;
    }
}

// ===========================================================================
// 3. THE SEQUENCE: continuity by refinement, with a control that plateaus.
// ===========================================================================

TEST_CASE("the boat-to-orbit handover is continuous, by refinement") {
    // #259's headline is a claim about a CLIMB, not about two frames. A
    // golden PNG cannot settle it and neither can a threshold on adjacent
    // frames -- adjacent frames are SUPPOSED to differ, because the ocean
    // really is changing as it recedes.
    //
    // Continuity has an exact definition that needs no tolerance: refine the
    // sampling and the largest adjacent step must shrink with it. For a
    // Lipschitz function sampled at spacing h the largest adjacent
    // difference is bounded by K*h, so halving h halves it; for a function
    // with a jump of size J the largest difference is at least J however
    // fine the sampling gets. The two behaviours are qualitatively different
    // and no constant has to be chosen to tell them apart.
    //
    // The bar is 0.6 rather than the ideal 0.5 for #260's reason: the
    // location of the maximum moves between refinements, so the halved
    // ladder does not sample the same worst point.
    const unsigned grid = 256u;
    const double   cm   = CoxMunkMeanSquareSlope(kDefaultWind);
    double sigma2[3];
    cascadeSigma2(grid, kDefaultWind, sigma2);

    // The ladder is geometric in footprint because the cone is: cone_width
    // is cone_spread times distance, so a linear ladder that resolved the
    // 9 cm grid spacing would need ten million rungs to reach orbit.
    auto worstStep = [&](int rungs) {
        double worst = 0.0;
        double prev = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, 1e-3);
        for (int i = 1; i <= rungs; ++i) {
            const double f = double(i) / double(rungs);
            const double w = 1e-3 * std::pow(1e8, f);
            const double cur = oceanBrdfAlpha2(sigma2, cm, 0.0, grid, w);
            worst = std::max(worst, std::fabs(cur - prev));
            prev = cur;
        }
        return worst;
    };
    const double w1 = worstStep(512);
    const double w2 = worstStep(1024);
    const double w3 = worstStep(2048);
    CAPTURE(w1); CAPTURE(w2); CAPTURE(w3);
    CHECK(w1 > 0.0);
    CHECK(w2 < 0.6 * w1);
    CHECK(w3 < 0.6 * w2);

    // THE CONTROL, so the case above cannot pass by measuring nothing. This
    // is the thing #259 must not be: hand the whole slope budget over at a
    // fixed distance -- a fade with a threshold, which is what "just switch
    // to a rough sphere past N km" would be. Its worst step is the size of
    // the jump at every refinement, so the ratio plateaus at 1.
    auto worstStepSwitch = [&](int rungs) {
        auto hard = [&](double w) {
            return (w < 1000.0) ? 0.0 : cm;
        };
        double worst = 0.0;
        double prev = hard(1e-3);
        for (int i = 1; i <= rungs; ++i) {
            const double f = double(i) / double(rungs);
            const double w = 1e-3 * std::pow(1e8, f);
            const double cur = hard(w);
            worst = std::max(worst, std::fabs(cur - prev));
            prev = cur;
        }
        return worst;
    };
    const double s1 = worstStepSwitch(512);
    const double s2 = worstStepSwitch(1024);
    const double s3 = worstStepSwitch(2048);
    CAPTURE(s1); CAPTURE(s2); CAPTURE(s3);
    CHECK(s2 == doctest::Approx(s1));
    CHECK(s3 == doctest::Approx(s1));
    CHECK(s2 > 0.6 * s1);
}

TEST_CASE("the sub-pixel-crest bailout can only move the surface half a pixel") {
    // oceanRayMarchShell stops marching the crests when 2 * h_max <=
    // cone_width and takes the analytic shell instead. That is a HARD
    // switch, so the honest question is how far the surface can move across
    // it -- and the answer is bounded by the threshold's own definition.
    //
    // At the switch the displacement band is exactly one cone width across.
    // The surface can be anywhere in the band, so the position can move by
    // at most h_max = cone_width / 2, and cone_width is one pixel by
    // construction (cone_spread = 2 tan(fov/2) / height). So the jump is at
    // most half a pixel, at every resolution and field of view, without any
    // distance appearing in the argument.
    const double fov_rad = 60.0 * 3.14159265358979 / 180.0;
    const double height_px = 1080.0;
    const double cone_spread = 2.0 * std::tan(0.5 * fov_rad) / height_px;
    CHECK(cone_spread == doctest::Approx(1.0691e-3).epsilon(1e-3));

    for (double h_max : {0.5, 1.5, 3.0, 8.0}) {
        // Distance at which the band becomes one pixel wide.
        const double d = 2.0 * h_max / cone_spread;
        CAPTURE(h_max);
        CAPTURE(d);
        // The worst positional error, in pixels, is h_max over the pixel
        // footprint at that distance -- one half, identically.
        const double pixels = h_max / (cone_spread * d);
        CHECK(pixels == doctest::Approx(0.5));
    }
    // And for the engine's own peak crest at 12 m/s the switch lands where
    // the study says it does: a couple of km, not tens.
    CHECK(2.0 * 1.5 / cone_spread == doctest::Approx(2806.0).epsilon(1e-2));
}

// ===========================================================================
// 4. THE CASCADE LAYOUT.
// ===========================================================================

TEST_CASE("the three cascades partition one spectrum with no gap and no overlap") {
    const unsigned grid = 256u;
    // Contiguous: each cascade's band bottom is the next coarser one's top.
    for (int c = 0; c + 1 < kCascades; ++c) {
        CAPTURE(c);
        CHECK(bandLo(c, grid) == doctest::Approx(bandHi(c + 1)));
    }
    // Every band is non-empty and resolvable by its OWN grid: the shortest
    // wave it carries must be at least the Nyquist of its tile.
    for (int c = 0; c < kCascades; ++c) {
        CAPTURE(c);
        CHECK(bandHi(c) > bandLo(c, grid));
        const double nyquist = 2.0 * kPeriodM[c] / double(grid);
        CHECK(bandLo(c, grid) >= nyquist * (1.0 - 1e-12));
    }
    // Each tile holds at least kBandDivisor periods of its longest wave, so
    // the coarsest ring of its lattice carries ~2*pi*8 = 50 independent
    // Fourier modes and the field reads as stochastic rather than as a
    // pattern. Cascade 0 is the exception BY CONSTRUCTION -- it is unbounded
    // above so a high-wind spectral peak is not truncated -- and its
    // energy-bearing content still sits far below its period.
    for (int c = 1; c < kCascades; ++c) {
        CAPTURE(c);
        CHECK(kPeriodM[c] / bandHi(c) == doctest::Approx(kBandDivisor));
    }
    CHECK(pt::ocean::PiersonMoskowitzPeakWavelengthM(kDefaultWind) <
          kPeriodM[0] / kBandDivisor);
}

TEST_CASE("the cascade periods do not share a beat inside the visible world") {
    // 1793 = 11 * 163, 211 prime, 23 prime: no common factor, so the summed
    // field repeats only at the product.
    auto gcd = [](long long a, long long b) {
        while (b) { const long long t = a % b; a = b; b = t; }
        return a;
    };
    CHECK(gcd(1793, 211) == 1);
    CHECK(gcd(1793, 23) == 1);
    CHECK(gcd(211, 23) == 1);
    const long long lcm = 1793LL * 211LL * 23LL;
    CHECK(lcm == 8701429LL);

    // Against the horizon distance sqrt(2 R h) at the altitudes where the
    // waves are still resolved at all. Even from 400 km, where the crests
    // are 300 pixels below sub-pixel, the pattern cannot close.
    const double R = pt::planet::kIuggMeanRadius;
    for (double h : {1.7, 10.0e3, 400.0e3}) {
        const double horizon = std::sqrt(2.0 * R * h + h * h);
        CAPTURE(h);
        CAPTURE(horizon);
        CHECK(double(lcm) > horizon);
    }
}

TEST_CASE("band-limiting a Phillips cascade really removes the other bands") {
    // The window is what makes three tiles a partition of one spectrum
    // rather than three copies of it. Measured through the solver's own
    // spectral slope variance, which is a sum over H0 and therefore sees
    // exactly what was and was not synthesised.
    OceanFFT full;
    full.MutableConfig().grid_size    = 64;
    full.MutableConfig().patch_size_m = 200.0f;
    full.MutableConfig().amplitude    = 1.0f;
    full.EnsureSpectrum();
    const double all = full.SlopeVarianceAbove(0.0);
    CHECK(all > 0.0);

    OceanFFT windowed;
    windowed.MutableConfig() = full.GetConfig();
    windowed.MutableConfig().band_lo_m = 20.0f;
    windowed.MutableConfig().band_hi_m = 0.0f;   // unbounded above
    windowed.EnsureSpectrum();

    // Everything the window kept is above 20 m, so restricting the query to
    // >= 20 m changes nothing, and restricting it to the removed band
    // returns exactly zero rather than something small.
    const double kept = windowed.SlopeVarianceAbove(0.0);
    CHECK(kept == doctest::Approx(windowed.SlopeVarianceAbove(20.0)));
    CHECK(kept > 0.0);
    CHECK(kept < all);
    // The complementary window, and the two summing back to the whole.
    //
    // The union is itself a WINDOWED solver rather than `full`: a windowed
    // spectrum carries the k-space cell area (2*pi/L)^2 and the legacy one
    // does not, so comparing across that boundary would measure the
    // normalisation rather than the partition. Two windows and their union,
    // all on the same side of it, is the comparison that means something.
    OceanFFT lower;
    lower.MutableConfig() = full.GetConfig();
    lower.MutableConfig().band_lo_m = 1.0e-6f;
    lower.MutableConfig().band_hi_m = 20.0f;
    lower.EnsureSpectrum();
    OceanFFT both;
    both.MutableConfig() = full.GetConfig();
    both.MutableConfig().band_lo_m = 1.0e-6f;
    both.MutableConfig().band_hi_m = 1.0e9f;   // above every lattice mode
    both.EnsureSpectrum();
    // Exactly, not approximately: the window is half-open, so the ring of
    // modes that lands on the shared boundary belongs to one side only.
    CHECK(kept + lower.SlopeVarianceAbove(0.0) ==
          doctest::Approx(both.SlopeVarianceAbove(0.0)).epsilon(1e-12));

    // The window does NOT disturb the random draw: the same bins carry the
    // same amplitudes, because the Gaussian pair is drawn whether or not
    // the window zeroed that bin's spectral density. Had the window skipped
    // the draw, every bin after the first rejected one would shift and the
    // three cascades would stop being one sea.
    CHECK(windowed.SlopeVarianceAbove(20.0) ==
          doctest::Approx(both.SlopeVarianceAbove(20.0)));
}

TEST_CASE("the windowed spectrum is scale-invariant and the legacy one is not") {
    // The k-space cell area (2*pi/L)^2 is what makes the synthesis a
    // spectral DENSITY rather than a per-mode amplitude. Without it the
    // wave height scales with the tile size -- invisible with one tile,
    // fatal with three, because the cascades' relative weights come out
    // wrong by (L_0/L_2)^2.
    auto sigma2At = [](float patch, bool windowed) {
        OceanFFT o;
        o.MutableConfig().grid_size    = 64;
        o.MutableConfig().patch_size_m = patch;
        o.MutableConfig().amplitude    = 1.0f;
        if (windowed) {
            // A window that is well inside both tiles' resolvable range, so
            // the comparison is over the SAME physical band.
            o.MutableConfig().band_lo_m = 12.0f;
            o.MutableConfig().band_hi_m = 60.0f;
        }
        o.EnsureSpectrum();
        return o.SlopeVarianceAbove(0.0);
    };
    // Windowed: doubling the tile leaves the band's slope variance alone
    // (up to the realisation noise of a finite lattice).
    const double w1 = sigma2At(200.0f, true);
    const double w2 = sigma2At(400.0f, true);
    CAPTURE(w1); CAPTURE(w2);
    CHECK(w1 > 0.0);
    CHECK(w2 == doctest::Approx(w1).epsilon(0.35));

    // Legacy: the same doubling multiplies it by ~4, which is the bug this
    // pins so a future "tidy-up" cannot quietly reintroduce it on the
    // windowed path. Compared over the same band so the two rows differ in
    // exactly one thing.
    OceanFFT a; a.MutableConfig().grid_size = 64;
    a.MutableConfig().patch_size_m = 200.0f; a.MutableConfig().amplitude = 1.0f;
    a.EnsureSpectrum();
    OceanFFT b; b.MutableConfig().grid_size = 64;
    b.MutableConfig().patch_size_m = 400.0f; b.MutableConfig().amplitude = 1.0f;
    b.EnsureSpectrum();
    const double la = a.SlopeVarianceAbove(12.0) - a.SlopeVarianceAbove(60.0);
    const double lb = b.SlopeVarianceAbove(12.0) - b.SlopeVarianceAbove(60.0);
    CAPTURE(la); CAPTURE(lb);
    CHECK(lb > 2.0 * la);
}

// ===========================================================================
// 5. THE ANCHOR LATTICE.
// ===========================================================================

TEST_CASE("crossing an anchor cell boundary does not move the wave field") {
    // A freely sliding tangent origin drags the field along with the camera
    // and deletes parallax; a snapped one only helps if the snap itself is
    // invisible. It is, and exactly rather than approximately: the lattice
    // spacing IS cascade 0's period, so a one-cell step translates every
    // point's tangent coordinate by a whole number of cascade-0 tiles.
    const double R = pt::planet::kIuggMeanRadius;
    const glm::dvec3 pole(0.0, 0.0, 1.0);
    const glm::dvec3 prime(1.0, 0.0, 0.0);

    // Two camera positions either side of an easting cell boundary. The
    // boundary is at half a cell, so step across it from just inside.
    auto camAt = [&](double lat, double lon, double alt) {
        const double r = R + alt;
        return glm::dvec3(r * std::cos(lat) * std::cos(lon),
                          r * std::cos(lat) * std::sin(lon),
                          r * std::sin(lat));
    };
    const double lat0 = 0.35;                       // ~20 degrees
    // FIND the boundary rather than predicting it. The easting lattice is
    // measured along the SNAPPED parallel, so its cell width depends on the
    // snapped latitude and not on lat0 -- computing the half-cell from
    // cos(lat0) lands next to the boundary rather than on it, and the case
    // then silently tests two cameras in the SAME cell. Bisect to a
    // millimetre instead: deterministic, and it cannot go vacuous because
    // the index step is asserted below.
    const double cell_lon = kPeriodM[0] / (R * std::cos(lat0));
    double lo_lon = 4.0 * cell_lon;
    double hi_lon = 6.0 * cell_lon;
    const auto jAt = [&](double lon) {
        return pt::ocean::OceanTangentAnchor(camAt(lat0, lon, 2.0), R, pole,
                                             prime, kPeriodM, kCascades)
            .lattice_j;
    };
    const double j_lo = jAt(lo_lon);
    REQUIRE(jAt(hi_lon) > j_lo);
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo_lon + hi_lon);
        if (jAt(mid) > j_lo) hi_lon = mid; else lo_lon = mid;
    }
    const double lon_mid = 0.5 * (lo_lon + hi_lon);

    const auto fa = pt::ocean::OceanTangentAnchor(
        camAt(lat0, lo_lon, 2.0), R, pole, prime, kPeriodM, kCascades);
    const auto fb = pt::ocean::OceanTangentAnchor(
        camAt(lat0, hi_lon, 2.0), R, pole, prime, kPeriodM, kCascades);
    REQUIRE(fa.valid);
    REQUIRE(fb.valid);
    // The snap really did fire -- otherwise this case pins nothing.
    CHECK(fb.lattice_j == doctest::Approx(fa.lattice_j + 1.0));
    CHECK(fb.lattice_i == doctest::Approx(fa.lattice_i));

    // Cascade 0's phase is 0 on BOTH sides, identically: its lookup cannot
    // move, because the lattice is its own period.
    CHECK(fa.phase[0][0] == 0.0);
    CHECK(fb.phase[0][0] == 0.0);
    CHECK(fa.phase[0][1] == 0.0);
    CHECK(fb.phase[0][1] == 0.0);

    // The two terms of the bound, from the SNAPPED latitude the lattice
    // actually used rather than from lat0.
    const double sin_phi = fa.origin.z / R;
    const double cos_phi = std::sqrt(std::max(1.0 - sin_phi * sin_phi, 1e-12));
    const double tan_phi = std::fabs(sin_phi / cos_phi);
    const double bound_m = kPeriodM[0] * kPeriodM[0] * tan_phi / (2.0 * R);
    CHECK(bound_m == doctest::Approx(0.0922).epsilon(5e-2));

    // For a fixed WORLD point, the tangent coordinate that reaches the
    // texture -- (e, n)/P_c plus the phase -- must land on the same texel.
    // Sample a ring of points out to 2 km, the range at which crests are
    // still resolved at all.
    for (int k = 0; k < 24; ++k) {
        const double th = 2.0 * 3.14159265358979 * double(k) / 24.0;
        for (double d : {10.0, 100.0, 1000.0, 2000.0}) {
            // A world point d metres from the (unsnapped) camera ground
            // point, in the shared tangent plane.
            const glm::dvec3 p = camAt(lat0, lon_mid, 0.0) +
                                 d * (std::cos(th) * fa.east +
                                      std::sin(th) * fa.north);
            for (int c = 0; c < kCascades; ++c) {
                const double ea = glm::dot(p - fa.origin, fa.east);
                const double na = glm::dot(p - fa.origin, fa.north);
                const double eb = glm::dot(p - fb.origin, fb.east);
                const double nb = glm::dot(p - fb.origin, fb.north);
                const double ua = ea / kPeriodM[c] + fa.phase[c][0];
                const double ub = eb / kPeriodM[c] + fb.phase[c][0];
                const double va = na / kPeriodM[c] + fa.phase[c][1];
                const double vb = nb / kPeriodM[c] + fb.phase[c][1];
                // Compare modulo one tile -- a whole-tile offset is a no-op
                // for a periodic field, which is the entire point.
                auto wrapDiff = [](double x, double y) {
                    double d2 = x - y;
                    d2 -= std::floor(d2 + 0.5);
                    return std::fabs(d2);
                };
                CAPTURE(c); CAPTURE(d); CAPTURE(th);
                CAPTURE(bound_m);
                const double lever = glm::length(p - fa.origin);
                // The bound is DERIVED, in two terms, and the derivation is
                // the reason this case is worth having:
                //
                //  (a) THE PARALLEL IS NOT A GEODESIC. Stepping one lattice
                //      cell east along the parallel lands
                //      R cos(phi) sin(phi) (1 - cos(dlon)) = P0^2 tan(phi) /
                //      (2R) NORTH of a pure east step -- 9.2 cm at 20
                //      degrees for P0 = 1793 m. Constant, independent of the
                //      lever arm, and it is most of what this case measures.
                //  (b) ORIENTATION. The two bases differ by dlon about the
                //      POLE, whose in-plane part is dlon sin(phi) =
                //      P0 tan(phi) / R, so a point L from the ANCHOR moves
                //      by L P0 tan(phi) / R.
                //
                // The lever arm is measured from the ANCHOR, not from the
                // camera: the anchor is up to a lattice cell away from the
                // camera's ground point, so even a point AT the camera has
                // a kilometre of lever. Getting that wrong made the first
                // version of this case fail by 20x while the code was right.
                const double bound = (bound_m + lever * tan_phi *
                                      kPeriodM[0] / R) / kPeriodM[c] + 1e-9;
                CHECK(wrapDiff(ua, ub) <= bound);
                CHECK(wrapDiff(va, vb) <= bound);
            }
        }
    }
}

TEST_CASE("the anchor is a pure function of camera position") {
    // No hysteresis, no path dependence -- which is what the golden matrix
    // requires and what a "re-anchor once you have drifted half a cell"
    // scheme could not give. Reaching the same camera position by two
    // different routes must produce the same frame, bit for bit.
    const double R = pt::planet::kIuggMeanRadius;
    const glm::dvec3 pole(0.0, 0.0, 1.0), prime(1.0, 0.0, 0.0);
    const glm::dvec3 cam(0.3 * R, 0.4 * R, 0.86 * R);
    const auto f1 = pt::ocean::OceanTangentAnchor(cam, R, pole, prime,
                                                  kPeriodM, kCascades);
    // Same query after a long detour of other queries: the function holds
    // no state, so the second answer is identical to the bit.
    for (int i = 0; i < 50; ++i) {
        (void)pt::ocean::OceanTangentAnchor(
            cam + glm::dvec3(double(i) * 1e4, 0.0, 0.0), R, pole, prime,
            kPeriodM, kCascades);
    }
    const auto f2 = pt::ocean::OceanTangentAnchor(cam, R, pole, prime,
                                                  kPeriodM, kCascades);
    CHECK(f1.origin == f2.origin);
    CHECK(f1.east   == f2.east);
    CHECK(f1.north  == f2.north);
    CHECK(f1.lattice_i == f2.lattice_i);
    CHECK(f1.lattice_j == f2.lattice_j);
    for (int c = 0; c < kCascades; ++c) {
        CHECK(f1.phase[c][0] == f2.phase[c][0]);
        CHECK(f1.phase[c][1] == f2.phase[c][1]);
    }
}

TEST_CASE("the tangent frame is orthonormal, right-handed and on the shell") {
    const double R = pt::planet::kIuggMeanRadius;
    const glm::dvec3 pole(0.0, 0.0, 1.0), prime(1.0, 0.0, 0.0);
    // Sweep well past the polar collapse band so the degenerate branch is
    // exercised too rather than merely defended.
    for (int i = 0; i <= 40; ++i) {
        const double lat = -1.5707963 + 3.1415926 * double(i) / 40.0;
        for (int j = 0; j < 7; ++j) {
            const double lon = -3.14159 + 6.28318 * double(j) / 7.0;
            const glm::dvec3 cam(
                (R + 100.0) * std::cos(lat) * std::cos(lon),
                (R + 100.0) * std::cos(lat) * std::sin(lon),
                (R + 100.0) * std::sin(lat));
            const auto f = pt::ocean::OceanTangentAnchor(cam, R, pole, prime,
                                                         kPeriodM, kCascades);
            CAPTURE(lat); CAPTURE(lon);
            REQUIRE(f.valid);
            CHECK(glm::length(f.east)  == doctest::Approx(1.0));
            CHECK(glm::length(f.north) == doctest::Approx(1.0));
            CHECK(glm::dot(f.east, f.north) == doctest::Approx(0.0).epsilon(1e-9));
            // The origin sits ON the shell, to the metre.
            CHECK(glm::length(f.origin) == doctest::Approx(R).epsilon(1e-12));
            // East x North = Up, i.e. the outward radial. Right-handed ENU.
            const glm::dvec3 up = f.origin / R;
            const glm::dvec3 cr = glm::cross(f.east, f.north);
            CHECK(glm::dot(cr, up) == doctest::Approx(1.0).epsilon(1e-9));
            // ...and the frame really is tangent: both basis vectors are
            // perpendicular to the local vertical.
            CHECK(glm::dot(f.east,  up) == doctest::Approx(0.0).epsilon(1e-9));
            CHECK(glm::dot(f.north, up) == doctest::Approx(0.0).epsilon(1e-9));
            // The anchor is within half a lattice cell of the camera's own
            // ground point in the northing direction -- that is what makes
            // the projection's (d/R)^2 distortion small where it matters.
            const glm::dvec3 ground = R * glm::normalize(cam);
            CHECK(glm::length(ground - f.origin) <= kPeriodM[0]);
        }
    }
}

// ===========================================================================
// 6. THE CROSSING TEST, both ways.
// ===========================================================================

namespace {
// --- shader mirror: oceanCrossed / oceanSecantDenom (PathTrace.slang) ----
bool oceanCrossed(double diff_prev, double diff_cur) {
    return (diff_prev > 0.0 && diff_cur <= 0.0) ||
           (diff_prev < 0.0 && diff_cur >= 0.0);
}
double oceanSecantDenom(double diff_prev, double diff_cur) {
    const double den = diff_prev - diff_cur;
    return (den >= 0.0) ? std::max(den, 1e-5) : std::min(den, -1e-5);
}
// The Wave 8 test, kept so the difference can be asserted rather than
// described.
bool oceanCrossedWave8(double diff_prev, double diff_cur) {
    return diff_prev > 0.0 && diff_cur <= 0.0;
}
}  // namespace

TEST_CASE("a ray rising from under the surface now registers a crossing") {
    // PathTrace.slang:5417 (as filed) tested `diff_prev > 0 && diff_cur <= 0`
    // -- above to below ONLY. A ray originating below the surface and
    // travelling up starts with diff_prev < 0 and could never register a
    // crossing; it fell through to the analytic surface, so Snell's window
    // seen from underwater has been geometrically wrong since #25.
    //
    // The below-to-above case, which Wave 8 misses and P5 catches.
    CHECK_FALSE(oceanCrossedWave8(-1.0, +1.0));
    CHECK(oceanCrossed(-1.0, +1.0));
    CHECK(oceanCrossed(-0.001, 0.0));

    // STRICTLY ADDITIVE. Over a dense grid of sign pairs, every case the
    // Wave 8 test fired on fires identically here -- that is what makes the
    // fix safe for the existing flat-water goldens, and it is asserted
    // rather than argued.
    for (int i = -20; i <= 20; ++i) {
        for (int j = -20; j <= 20; ++j) {
            const double a = double(i) * 0.1;
            const double b = double(j) * 0.1;
            CAPTURE(a); CAPTURE(b);
            if (oceanCrossedWave8(a, b)) CHECK(oceanCrossed(a, b));
        }
    }
    // Neither test fires on a sample that starts exactly ON the surface,
    // which is the Wave 8 behaviour and is deliberate: a zero diff has no
    // side to have come from.
    CHECK_FALSE(oceanCrossed(0.0, 1.0));
    CHECK_FALSE(oceanCrossed(0.0, -1.0));
    CHECK_FALSE(oceanCrossedWave8(0.0, -1.0));

    // The secant weight has to land in [0, 1] in BOTH directions or the
    // refined hit lands outside the step it was bracketed in. The Wave 8
    // guard was max(den, 1e-5), which is right for a positive denominator
    // and produces a hugely negative weight for the new case.
    for (int i = 1; i <= 40; ++i) {
        const double mag = double(i) * 0.05;
        // above -> below
        double w = mag / oceanSecantDenom(mag, -mag);
        CAPTURE(mag);
        CHECK(w >= 0.0);
        CHECK(w <= 1.0);
        // below -> above
        w = (-mag) / oceanSecantDenom(-mag, mag);
        CHECK(w >= 0.0);
        CHECK(w <= 1.0);
        // Symmetric magnitudes bracket the midpoint in both directions.
        CHECK(w == doctest::Approx(0.5));
    }
    // And the guard never divides by zero even when the two samples agree.
    CHECK(std::isfinite(1.0 / oceanSecantDenom(0.0, 0.0)));
    CHECK(std::isfinite(1.0 / oceanSecantDenom(1e-30, 1e-30)));
}

// ===========================================================================
// 7. SEA LEVEL IS THE TERRAIN'S ZERO ELEVATION.
// ===========================================================================

TEST_CASE("the ocean shell and the terrain agree about where sea level is") {
    // The shell radius is the SAME scalar that places planet_center_radius
    // and, with terrain streaming on, the site's geocentric radius -- which
    // is the radius of the sphere TANGENT to the WGS-84 ellipsoid at the
    // site. So at the site the two surfaces coincide exactly and the
    // coastline falls where the DEM's own sign change is.
    const double lat = 36.6 * 3.14159265358979 / 180.0;   // a coastal site
    const double lon = -122.0 * 3.14159265358979 / 180.0;
    const auto site = pt::planet::PlanetSite::FromGeodetic(lat, lon);

    // The site's own surface point is the world origin, at exactly the
    // shell radius from the centre. Zero, not "small".
    const glm::dvec3 centre = site.CenterWorld();
    CHECK(glm::length(glm::dvec3(0.0) - centre) ==
          doctest::Approx(site.site_radius_m).epsilon(1e-15));

    // AND WHAT IT COSTS AWAY FROM THE SITE. The terrain is on the ellipsoid;
    // the shell is a sphere tangent to it. Walk out along a geodesic and
    // measure the separation directly, then check it against the second
    // -order prediction d^2 / (2 R) x (R / R_curv - 1) that the code
    // comments quote. Both must be metres at 100 km and millimetres at the
    // horizon -- if this ever grows the ocean starts cutting into beaches.
    const glm::dvec3 east_ecef =
        glm::transpose(site.ecef_to_world) * glm::dvec3(1.0, 0.0, 0.0);
    const glm::dvec3 site_ecef = pt::planet::GeodeticToEcef(lat, lon);
    struct Row { double d; double bound; };
    const Row rows[] = {
        {4.65e3, 0.02},     // the horizon from eye height
        {1.0e4,  0.10},
        {1.0e5,  6.0},
    };
    for (const Row& r : rows) {
        // Step d metres along the great circle toward local east.
        const double ang = r.d / site.site_radius_m;
        const glm::dvec3 p_ecef = std::cos(ang) * site_ecef +
                                  std::sin(ang) * site.site_radius_m * east_ecef;
        // Where the ELLIPSOID's surface is along that same direction...
        const glm::dvec3 dir = pt::planet::EcefToFieldDirection(p_ecef);
        const glm::dvec3 surf = pt::planet::EllipsoidSurface(dir);
        // ...against where the SHELL is: both measured from the centre.
        const double shell = site.site_radius_m;
        const double ellip = glm::length(surf);
        CAPTURE(r.d);
        CAPTURE(std::fabs(ellip - shell));
        CHECK(std::fabs(ellip - shell) < r.bound);
    }

    // The sign test the acceptance criterion actually asks for: wherever
    // the terrain's ellipsoidal height is bigger than that separation, the
    // terrain vertex is on the correct side of the shell. Land above water,
    // seabed below, from one grid -- checked as an inequality with the
    // separation in it, not as a hope.
    for (double h : {-4000.0, -50.0, -1.0, 1.0, 50.0, 4000.0}) {
        const glm::dvec3 surf = pt::planet::EllipsoidSurface(
            pt::planet::EcefToFieldDirection(site_ecef));
        const glm::dvec3 n = pt::planet::GeodeticNormal(surf);
        const double r = glm::length(surf + n * h);
        CAPTURE(h);
        CHECK(((h > 0.0) == (r > site.site_radius_m)));
    }
}

// ===========================================================================
// 8. THE MIRROR.
// ===========================================================================

TEST_CASE("shader and engine mirrors are still faithful") {
    // The transcriptions above are only worth something while they match
    // the shader. Whitespace is stripped so reformatting cannot fail this,
    // and occurrences are COUNTED rather than merely found.
    const std::string pt = tighten(PT_SHADER_PATHTRACE_PATH);

    // --- THE PUSH LAYOUT, BOTH HALVES ---------------------------------
    //
    // PathTrace.slang declares the push block TWICE -- a Metal cbuffer and
    // a SPIR-V Frame block -- and this build cannot compile the SPIR-V one
    // (PT_ENABLE_VULKAN_BACKEND is off on Apple), so a field added to one
    // half and forgotten in the other would ship silently and every Vulkan
    // field after it would read from the wrong offset. Two of each, and one
    // of each in the host struct, is the only guard available here.
    //
    // What IS verifiable on this host, and was: PathTrace.slang compiles to
    // SPIR-V and passes spirv-val with these lanes present, in both the
    // RayQuery and the PT_SPIRV_NO_RAYQUERY variant.
    for (const char* lane : {"float4planet_ocean;", "float4ocean_tan_e;",
                             "float4ocean_tan_n;", "float4ocean_tan_o;",
                             "float4ocean_phase;", "float4ocean_slope;"}) {
        CAPTURE(lane);
        CHECK(countOf(pt, lane) == 2u);
    }
    // ...and the host mirror, once each, in PtPush's tail. Appended rather
    // than inserted, so SoftwareTracer.cpp's raw byte offsets stay valid --
    // the trap that once turned every non-ACES tonemap into ACES.
    const std::string ec = tighten(PT_ENGINE_CPP_PATH);
    for (const char* lane : {"floatplanet_ocean[4];", "floatocean_tan_e[4];",
                             "floatocean_tan_n[4];", "floatocean_tan_o[4];",
                             "floatocean_phase[4];", "floatocean_slope[4];"}) {
        CAPTURE(lane);
        CHECK(countOf(ec, lane) == 1u);
    }
    // The Vulkan spilled-tail budget guard has to account for them: six
    // vec4 is 96 B on a tail that was 1712 and is now 1808, against
    // kFrameUboSize 2048. A silent overflow renders corrupt on Vulkan and
    // nothing on this host would notice.
    CHECK(countOf(ec, "+96/*PlanetaryP5(#259):planet_ocean+ocean_tan_e+") == 1u);
    CHECK(countOf(ec, "static_assert(sizeof(PtPush)-112<=2048,") == 1u);

    // The band divisor is a wire format between Engine.h and the shader.
    CHECK(countOf(pt, "kOceanBandDivisor=8.0") == 1u);
    const std::string eh = tighten(PT_ENGINE_H_PATH);
    CHECK(countOf(eh, "kOceanBandDivisor=8.0") == 1u);
    CHECK(countOf(eh, "kOceanCascadePeriodM[3]={1793.0,211.0,23.0}") == 1u);
    CHECK(countOf(eh, "kOceanCascades=3") == 1u);
    // ...and the shader derives its band edges the same way: cascade 0
    // unbounded above, everyone else at P/divisor, finest at its Nyquist.
    CHECK(countOf(pt, "floathi=(c==0u)?P:(P/kOceanBandDivisor);") == 1u);
    CHECK(countOf(pt, "?(oceanCascadePeriod(c+1u)/kOceanBandDivisor)") == 1u);
    CHECK(countOf(pt, ":(2.0*P/float(max(grid,1u)));") == 1u);

    // The log-uniform ramp itself -- the Phillips equilibrium range's own
    // answer, and the thing a smoothstep would quietly replace.
    CHECK(countOf(pt, "returnlog(hi/wc)/log(hi/lo);") == 1u);
    CHECK(countOf(pt, "floatwc=clamp(w,lo,hi);") == 1u);

    // The conservation law, in one expression and only one.
    CHECK(countOf(pt, "floatgeom=dot(resolved,ocean_slope.xyz);") == 1u);
    CHECK(countOf(pt, "returnmax(ocean_slope.w+planet_ocean.z-geom,0.0);") == 1u);

    // sqrt of the retained VARIANCE fraction, not the fraction itself:
    // scaling a cascade's amplitude by a scales its slope variance by a^2,
    // and dropping the sqrt would silently break the identity above.
    CHECK(countOf(pt, "float3amp=sqrt(max(resolved,float3(0.0,0.0,0.0)));") == 1u);

    // The crossing test, both disjuncts, exactly once.
    CHECK(countOf(pt, "return(diff_prev>0.0&&diff_cur<=0.0)||"
                      "(diff_prev<0.0&&diff_cur>=0.0);") == 1u);
    CHECK(countOf(pt, "return(den>=0.0)?max(den,1e-5):min(den,-1e-5);") == 1u);
    // ...and that no copy of the Wave 8 one-sided test survives anywhere
    // outside oceanCrossed. Both marches call the helper.
    CHECK(countOf(pt, "if(diff_prev>0.0&&diff_cur<=0.0){") == 0u);
    CHECK(countOf(pt, "oceanCrossed(diff_prev,diff_cur)") == 2u);

    // The sub-pixel-crest bailout, as the comparison the half-pixel bound
    // above is derived from.
    CHECK(countOf(pt, "if(2.0*h_max<=cone_w){") == 1u);

    // The shell is intersected at the SAME radius lane the atmosphere and
    // the analytic body read, and its centre is planet_center_radius.
    CHECK(countOf(pt, "intersectSphere(ro,rd,planet_center_radius.xyz,"
                      "planet_ocean.x,t_min,t_w)") == 1u);

    // The horizontality gate is dot(n, localUp) and not n.y, in both the
    // G-buffer pass and the shading pass.
    CHECK(countOf(pt, "abs(dot(h.normal,localUp(ro+rd*h.t)))>0.5") == 2u);
    CHECK(countOf(pt, "abs(dot(h0.normal,localUp(ro0+rd0*h0.t)))>0.5") == 1u);
    CHECK(countOf(pt, "abs(h.normal.y)>0.5") == 0u);
    CHECK(countOf(pt, "abs(h0.normal.y)>0.5") == 0u);

    // The sun-disc suppression that keeps the specular NEE and the bounce
    // ray's sky from both counting the sun: set on the reflected lobe,
    // consumed once, cleared in the same breath.
    CHECK(countOf(pt, "ocean_sun_nee=did_ocean_sun_nee;") == 1u);
    CHECK(countOf(pt, "boolocean_sun_nee_prev=ocean_sun_nee;ocean_sun_nee=false;") == 1u);
    CHECK(countOf(pt, "ocean_sun_nee_prev?skyColorNoSunDisc(ro,rd,seed)") == 1u);
    // ...and that the NEE integrates the SAME radiance the disc carries.
    // sunDiscPhysical returns T*80/omega, sunRadianceAt returns 80*T, so
    // the disc's radiance times its solid angle IS the NEE's irradiance.
    CHECK(countOf(pt, "s-=sunDiscPhysical(ro,rd,sun_and_mode.xyz);") == 1u);
    CHECK(countOf(pt, "returnfloat3(80.0)*sunSlantTransmittance(p,sun_dir);") == 1u);
    CHECK(countOf(pt, "returnT*(80.0/max(omega,1.0e-12))*bright;") == 1u);
}

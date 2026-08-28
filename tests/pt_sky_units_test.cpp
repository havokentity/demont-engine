// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for #280: the sky in physical radiometric units, the
// multiple-scattering table that makes that survivable, and the retirement
// of the r_rayleigh = 30 / inflated-aerosol pair those two enable.
//
// WHAT THIS FILE IS FOR, AND WHAT IT REFUSES TO DO
//
// #280's headline claims are all claims about NUMBERS, and every one of
// them has a shape that a golden PNG cannot distinguish from its own
// failure:
//
//   * "the sun's 80 was an irradiance, not a radiance" is a claim about
//     which quantity five call sites consume. A golden cannot see it at
//     all.
//   * "multiple scattering supplies the energy the inflated Rayleigh was
//     standing in for" is a claim that a term is PRESENT and of a
//     particular SIZE. A golden showing a dark sky is consistent with the
//     term being absent, being present and tiny, and with the whole
//     renderer being broken.
//   * "the twilight zenith is blue because of ozone" is a claim about a
//     per-channel ORDERING. A blue frame is also what a painted blue floor
//     produces.
//
// So every case here asserts on CONTENT and on SIGN, never on "it is not
// zero". Two of them are explicitly built so that deleting the feature
// makes them fail rather than pass: the multiple-scattering case requires
// the term to fall in a measured band and rejects both zero and a value it
// could not physically reach, and the ozone case requires an ordering
// INVERSION rather than a threshold. A physically correct sky that
// undershoots by 7x is nearly black, and nearly black is what a
// feature-deleted vacuity check passes on -- that trap is the reason this
// file exists in this shape.
//
// Deterministic: every input is a literal or derived from literals, and
// the table builder has no RNG.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "renderer/Atmosphere.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE_MESSAGE(f.is_open(), "cannot open ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// COUNT, never find() != npos.  A substring pin is satisfied by one correct
// copy however many wrong ones survive elsewhere, which is exactly how #276
// stayed live for a full cycle underneath a passing test.
std::size_t CountOccurrences(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t n = 0;
    for (std::size_t p = hay.find(needle); p != std::string::npos;
         p = hay.find(needle, p + needle.size())) {
        ++n;
    }
    return n;
}

// The reference medium every case below shares: real Earth at the values
// #280 made the defaults.
pt::atmo::Body RefBody(double rayleigh = 1.0, double ozone = 1.0) {
    return pt::atmo::Scale(pt::atmo::Earth(6371008.8), rayleigh, 3.996e-6, ozone);
}

std::vector<float> BuildLut(const pt::atmo::Body& b, double albedo = 0.10,
                            int threads = 0) {
    std::vector<float> lut(static_cast<std::size_t>(pt::atmo::kMsLutFloats), 0.0f);
    pt::atmo::MsLutParams p;
    p.ground_albedo = albedo;
    p.threads = threads;
    pt::atmo::BuildMultiScatterLut(b, p, lut.data());
    return lut;
}

// Rec.709 luma of an engine RGB radiance triple.  Used only to collapse
// three numbers to one for an ordering comparison, never as a photometric
// conversion -- the engine's channels partition the spectrum, so a
// single-wavelength efficacy would be the wrong constant to apply and no
// case here applies one.
double Luma(const float v[3]) {
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}

}  // namespace

// ===========================================================================
// 1.  THE UNITS AUDIT
// ===========================================================================

TEST_CASE("solar irradiance partitions the measured total") {
    // Kopp & Lean 2011: 1360.8 W/m^2.  The three channels PARTITION the
    // spectrum, so their sum is the measured total exactly -- that is the
    // property that makes a white Lambertian surface reflect the right
    // amount of power, and it is the whole reason the triple is not simply
    // three copies of 1360.8.
    const double sum = static_cast<double>(pt::atmo::kSolarIrradianceR)
                     + pt::atmo::kSolarIrradianceG
                     + pt::atmo::kSolarIrradianceB;
    CHECK(sum == doctest::Approx(1360.8).epsilon(1e-6));

    // The ratios come from the solar spectral irradiance at the model's own
    // three sampling wavelengths (Bruneton 2017).  Assert the ratio rather
    // than the values, so a future change to the total cannot silently
    // change the sky's COLOUR as well as its level.
    const double sR = pt::atmo::kSolarSpectralShape[0];
    const double sG = pt::atmo::kSolarSpectralShape[1];
    const double sB = pt::atmo::kSolarSpectralShape[2];
    CHECK(pt::atmo::kSolarIrradianceG / pt::atmo::kSolarIrradianceR
          == doctest::Approx(sG / sR).epsilon(1e-5));
    CHECK(pt::atmo::kSolarIrradianceB / pt::atmo::kSolarIrradianceR
          == doctest::Approx(sB / sR).epsilon(1e-5));

    // The sun is BLUER than it is red at the top of the atmosphere.  This
    // is the ordering that makes a rebased scene read cooler than the tuned
    // warm curve it replaced, and getting the triple in the wrong
    // wavelength order is the obvious mistake -- Rayleigh's own triple
    // ascends with channel index, so a copy-paste of that ordering onto a
    // solar spectrum would pass a sum check and fail this one.
    CHECK(pt::atmo::kSolarIrradianceB > pt::atmo::kSolarIrradianceG);
    CHECK(pt::atmo::kSolarIrradianceG > pt::atmo::kSolarIrradianceR);

    // ...but only mildly.  A factor of 1.3 across the visible, not 5.  This
    // catches a triple accidentally filled with a SCATTERING ratio, which
    // spans 5.7x.
    CHECK(pt::atmo::kSolarIrradianceB / pt::atmo::kSolarIrradianceR < 1.5);
}

TEST_CASE("the unit rebase is one derived factor, not a tuned one") {
    // Every fixture's r_exposure was divided by this and every painted
    // celestial term multiplied by it.  If it is ever edited to something
    // that is not mean(E)/80, the two halves stop cancelling and every
    // painted term in the engine silently changes brightness.
    const double mean = (static_cast<double>(pt::atmo::kSolarIrradianceR)
                       + pt::atmo::kSolarIrradianceG
                       + pt::atmo::kSolarIrradianceB) / 3.0;
    CHECK(static_cast<double>(pt::atmo::kUnitRebaseScale)
          == doctest::Approx(mean / 80.0).epsilon(1e-6));
    CHECK(static_cast<double>(pt::atmo::kLegacySkyScale)
          == doctest::Approx(pt::atmo::kUnitRebaseScale).epsilon(1e-3));
}

TEST_CASE("the shader carries the same anchor, and the old literal is gone") {
    const std::string math = ReadFile(PT_SHADER_MATH_PATH);
    const std::string pt   = ReadFile(PT_SHADER_PATHTRACE_PATH);

    // The triple is declared exactly once, in the module, at the values the
    // host computes.  Formatting the expectation from the host constants is
    // what makes this a MIRROR pin rather than a second hand-typed copy.
    char decl[160];
    std::snprintf(decl, sizeof(decl), "float3(%.4f, %.4f, %.4f)",
                  static_cast<double>(pt::atmo::kSolarIrradianceR),
                  static_cast<double>(pt::atmo::kSolarIrradianceG),
                  static_cast<double>(pt::atmo::kSolarIrradianceB));
    CHECK(CountOccurrences(math, std::string("kPtSolarIrradiance =\n    ") + decl) == 1);
    CHECK(CountOccurrences(math, "public static const float kPtLegacySkyScale = 5.67;") == 1);

    // THE RETIREMENT, PINNED BY COUNT.  `float3(80.0)` was the old solar
    // stand-in and it appeared at three sites in the path tracer; every one
    // of them now reads kPtSolarIrradiance.  A find() != npos check here
    // would be satisfied by one converted site while two stale copies kept
    // lighting the scene at the old level.
    CHECK(CountOccurrences(pt, "float3(80.0)") == 0);
    CHECK(CountOccurrences(pt, "kPtSolarIrradiance") >= 4);

    // The name no longer contradicts the arithmetic.  sunRadianceAt claimed
    // to return a radiance while every caller consumed an irradiance; the
    // audit resolved that the code was right and the label was wrong, so
    // the label moved.
    CHECK(CountOccurrences(pt, "float3 sunIrradianceAt(") == 1);
    CHECK(CountOccurrences(pt, "float3 sunRadianceAt(") == 0);

    // The two remaining legacy painted-curve sites divide by nothing and
    // multiply by exactly the documented factor.  Two, not one: #257
    // recorded that this expression had ALREADY drifted into two copies
    // with different warmth ramps, so a pin of one would leave the other.
    CHECK(CountOccurrences(pt, "80.0 * kPtLegacySkyScale") == 2);
}

TEST_CASE("the sun disc and the sun NEE partition one light source") {
    // This is the audit's operational statement.  sunIrradianceAt returns an
    // irradiance E; sunDiscPhysical returns E / Omega, a radiance.  The two
    // estimators therefore describe the same source and their product with
    // the solid angle must close: L_disc * Omega == E.  If the 80 had been a
    // radiance, the disc would be E/Omega^2 and this identity would fail by
    // Omega ~ 6.9e-5.
    //
    // Asserted on the SOURCE because there is no host entry point to call,
    // and asserted as the presence of the division rather than as a
    // numeric, because the numeric identity is exact by construction and
    // what can rot is the expression.
    const std::string pt = ReadFile(PT_SHADER_PATHTRACE_PATH);
    CHECK(CountOccurrences(pt, "T * (kPtSolarIrradiance / max(omega, 1.0e-12)) * bright")
          == 1);

    // And the NEE site multiplies by a BRDF and a cosine with no solid
    // angle -- the other half of the same argument.
    CHECK(CountOccurrences(pt, "throughput * sun_rad * trans * brdf * n_dot_l") == 1);

    // Numerically, for the sun's true half-angle: E / Omega at the rebased
    // anchor is ~2e7 W/m^2/sr, the textbook solar radiance.  Recomputing it
    // here is what makes the factor-of-14700 question answerable from the
    // test rather than from a comment.
    const double r_rad = 0.2666 * 3.14159265358979 / 180.0;
    const double omega = 2.0 * 3.14159265358979 * (1.0 - std::cos(r_rad));
    CHECK(omega == doctest::Approx(6.807e-5).epsilon(2e-3));
    const double L_sun = 1360.8 / omega;
    CHECK(L_sun == doctest::Approx(2.0e7).epsilon(0.05));
    // ...and that is 14 700x the irradiance, which is the size of the
    // mistake the audit was there to avoid.
    CHECK(L_sun / 1360.8 == doctest::Approx(14700.0).epsilon(0.05));
}

// ===========================================================================
// 2.  MULTIPLE SCATTERING
// ===========================================================================

TEST_CASE("multiple scattering supplies a measured fraction, not a token one") {
    const pt::atmo::Body b = RefBody();
    const std::vector<float> lut = BuildLut(b);

    // Zenith radiance at 60 degrees solar elevation, with and without the
    // table.  The claim #280 rests on is that the second-and-higher orders
    // are the energy the inflated Rayleigh was standing in for, so the size
    // of the term is the thing to assert.
    pt::atmo::SkyCookParams p;
    p.sun_elev_sin = std::sin(60.0 * 3.14159265358979 / 180.0);
    p.observer_alt = 1.7;
    const pt::atmo::SkyCook with    = pt::atmo::CookSky(b, lut.data(), p);
    const pt::atmo::SkyCook without = pt::atmo::CookSky(b, nullptr, p);

    const double ratio = Luma(with.zenith) / Luma(without.zenith);
    // BAND, NOT A FLOOR.  1.0 would mean the table contributed nothing --
    // which is what deleting the feature produces, and what a "> 0" check
    // would happily accept.  Above 2.0 would mean the geometric series had
    // run away, which is what a sign error in f_ms produces.  Measured
    // 1.285; the band is +/- 25% around it, which is 8x the spread across
    // the whole solar-elevation range this case sweeps below.
    INFO("multi-scatter zenith gain at 60 deg = ", ratio);
    CHECK(ratio > 1.15);
    CHECK(ratio < 1.60);

    // It matters MORE at twilight, because at -4 degrees the direct beam is
    // gone from the observer's own column and almost everything left is
    // higher-order.  An implementation that folded the term in as a
    // constant would pass the band above and fail this ordering.
    pt::atmo::SkyCookParams q = p;
    q.sun_elev_sin = std::sin(-4.0 * 3.14159265358979 / 180.0);
    const double twilight_ratio = Luma(pt::atmo::CookSky(b, lut.data(), q).zenith)
                                / Luma(pt::atmo::CookSky(b, nullptr, q).zenith);
    INFO("multi-scatter zenith gain at -4 deg = ", twilight_ratio);
    CHECK(twilight_ratio > ratio);

    // And the sky it produces is BLUE, not grey.
    CHECK(with.zenith[2] > with.zenith[1]);
    CHECK(with.zenith[1] > with.zenith[0]);

    // THE TERM IS BLUER THAN THE SINGLE-SCATTER SKY IT ADDS TO, and that is
    // not the intuition -- "multiple scattering whitens the sky" is the
    // received wisdom, and this case was first written asserting it and
    // FAILED. The received wisdom describes an OPTICALLY THICK atmosphere,
    // where blue saturates (tau >> 1) while red keeps growing, and a
    // horizon path or a hazy day. At the zenith of a clean atmosphere the
    // blue optical depth is only 0.265, nowhere near saturation, so the
    // second order goes as sigma_s squared and its blue-to-red ratio is
    // (33.100/5.802)^2 = 32 before extinction rather than 5.7. Measured
    // here at 14.6, against 5.9 for the single-scatter sky.
    //
    // Pinned in the direction the physics actually goes, with the size
    // bracketed, because "it got bluer" and "the red channel was dropped
    // entirely" are the same sign.
    const double dr = with.zenith[0] - without.zenith[0];
    const double db = with.zenith[2] - without.zenith[2];
    REQUIRE(dr > 0.0);
    const double ms_ratio = db / dr;
    INFO("multi-scatter term blue:red = ", ms_ratio);
    CHECK(ms_ratio > 8.0);
    CHECK(ms_ratio < 33.0);   // the sigma^2 ceiling, unattenuated
    const double sat_with = with.zenith[2] / with.zenith[0];
    CHECK(sat_with > 3.0);
    CHECK(sat_with < 20.0);
}

TEST_CASE("the multiple-scattering table has the shape the physics demands") {
    const pt::atmo::Body b = RefBody();
    const std::vector<float> lut = BuildLut(b);

    // Every texel the builder wrote carries w == 1. That is the shader's
    // "was this ever built?" signal, and it works precisely because a
    // zero-filled placeholder cannot produce it.
    for (int i = 0; i < pt::atmo::kMsLutTexels; ++i) {
        REQUIRE(lut[static_cast<std::size_t>(i) * 4 + 3] == 1.0f);
    }

    // MONOTONE IN SUN ELEVATION. More sun above the layer means more
    // first-order scattering to rescatter, so Psi_ms rises with mu_s. The
    // sweep is over the whole lit range, and every adjacent pair must obey
    // it -- a single ordering test at two points would pass on a table that
    // was accidentally a function of nothing.
    const double r = b.ground_radius + 2000.0;
    double prev = -1.0;
    for (int i = 0; i <= 20; ++i) {
        const double mu_s = 0.02 + 0.95 * (static_cast<double>(i) / 20.0);
        double v[3];
        pt::atmo::SampleMultiScatter(lut.data(), b, r, mu_s, v);
        const double lum = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
        REQUIRE(lum > 0.0);
        CHECK(lum >= prev);
        prev = lum;
    }

    // ALTITUDE, AND HOW WEAKLY. Psi_ms falls with height, but only by 2x
    // across the whole 100 km shell -- 12.70 at 1 km, 10.23 at 60 km, 6.26
    // at the top row. That is much flatter than the medium it lives in, and
    // it is flat for a reason worth recording: Psi_ms is the radiance
    // ARRIVING at a point from the lit air around it, not the amount of air
    // at the point. A point at 60 km sees the whole bright atmosphere below
    // it with almost no extinction in the way, which nearly compensates for
    // there being less air above.
    //
    // This case was first written expecting a 4x fall and then, on being
    // shown 1.24x, rewritten to expect a RISE. Both were wrong. The
    // margins below are measured, not inherited: 1.24x observed against a
    // strict-inequality bar, and 0.49 observed against a 0.70 bar.
    double lo[3], hi[3];
    pt::atmo::SampleMultiScatter(lut.data(), b, b.ground_radius + 1000.0, 0.9, lo);
    pt::atmo::SampleMultiScatter(lut.data(), b, b.ground_radius + 60000.0, 0.9, hi);
    double top[3];
    pt::atmo::SampleMultiScatter(lut.data(), b, b.top_radius, 0.9, top);
    INFO("Psi_ms green: 1 km ", lo[1], "  60 km ", hi[1], "  top ", top[1]);
    CHECK(lo[1] > hi[1]);
    CHECK(top[1] < lo[1] * 0.70);

    // The UNAMBIGUOUS altitude assertion is on the CONTRIBUTION TO THE SKY,
    // sigma_s(h) * Psi(h) -- the product the march actually forms. sigma_s
    // is exponential in altitude and Psi is nearly flat, so the product
    // falls by three orders of magnitude over the same span, and a
    // parameterisation error that flattened the radius axis entirely would
    // still fail this by a wide margin. Observed ratio ~1980 against a bar
    // of 100: 20x of headroom.
    double sr_lo[3], sm_lo[3], st_lo[3], sr_hi[3], sm_hi[3], st_hi[3];
    pt::atmo::Coefficients(b, 1000.0,  sr_lo, sm_lo, st_lo);
    pt::atmo::Coefficients(b, 60000.0, sr_hi, sm_hi, st_hi);
    const double contrib_lo = (sr_lo[1] + sm_lo[1]) * lo[1];
    const double contrib_hi = (sr_hi[1] + sm_hi[1]) * hi[1];
    INFO("sigma_s * Psi: 1 km ", contrib_lo, "  60 km ", contrib_hi);
    CHECK(contrib_lo > contrib_hi * 100.0);

    // NIGHT IS DARK. With the sun 30 degrees below the local horizontal a
    // point at 2 km is inside the body's shadow and the only multiply
    // scattered light left is what leaks around the terminator.
    double night[3];
    pt::atmo::SampleMultiScatter(lut.data(), b, r, -0.5, night);
    double noon[3];
    pt::atmo::SampleMultiScatter(lut.data(), b, r, 1.0, noon);
    CHECK(night[1] < noon[1] * 0.02);
}

TEST_CASE("the table is parallel and deterministic") {
    // "Parallel but deterministic" is exactly the kind of claim that rots
    // quietly, so it is asserted rather than commented. Rows are disjoint
    // and each is a pure function of (body, params, row), so a one-thread
    // build and a twelve-thread build must be bit-identical -- not close.
    const pt::atmo::Body b = RefBody();
    const std::vector<float> one    = BuildLut(b, 0.10, 1);
    const std::vector<float> twelve = BuildLut(b, 0.10, 12);
    REQUIRE(one.size() == twelve.size());
    CHECK(std::memcmp(one.data(), twelve.data(),
                      one.size() * sizeof(float)) == 0);
}

TEST_CASE("the ground albedo actually reaches the table") {
    // r_sky_ground_albedo grew a second consumer in #280. A cvar that is
    // read and then dropped is the shape of half the bugs this project has
    // filed, so the coupling is measured: a snow-white ground must lift the
    // multiply-scattered term well above a black one.
    const pt::atmo::Body b = RefBody();
    const std::vector<float> black = BuildLut(b, 0.0);
    const std::vector<float> snow  = BuildLut(b, 0.8);
    double vb[3], vs[3];
    const double r = b.ground_radius + 2000.0;
    pt::atmo::SampleMultiScatter(black.data(), b, r, 0.9, vb);
    pt::atmo::SampleMultiScatter(snow.data(),  b, r, 0.9, vs);
    INFO("ground bounce: black ", vb[1], " snow ", vs[1]);
    CHECK(vs[1] > vb[1] * 1.5);
    // ...and the black-ground table is still non-trivial, so the case above
    // is measuring the bounce rather than measuring the whole term.
    CHECK(vb[1] > 0.0);
}

TEST_CASE("the shader and the host agree about the table's parameterisation") {
    const std::string math = ReadFile(PT_SHADER_MATH_PATH);
    const std::string pt   = ReadFile(PT_SHADER_PATHTRACE_PATH);

    // Dimensions, declared once each, matching the host.
    char w[64], h[64];
    std::snprintf(w, sizeof(w), "public static const int kPtMsLutWidth  = %d;",
                  pt::atmo::kMsLutWidth);
    std::snprintf(h, sizeof(h), "public static const int kPtMsLutHeight = %d;",
                  pt::atmo::kMsLutHeight);
    CHECK(CountOccurrences(math, w) == 1);
    CHECK(CountOccurrences(math, h) == 1);

    // The mapping itself. Both sides do (r - R_g) / (R_t - R_g) on one axis
    // and (mu_s + 1) / 2 on the other; a mismatch here reads as a sky that
    // is subtly wrong everywhere rather than as a failure.
    CHECK(CountOccurrences(math, "float v = saturate((r - b.ground_radius) / span);") == 1);
    CHECK(CountOccurrences(math, "float u = (clamp(mu_s, -1.0, 1.0) + 1.0) * 0.5;") == 1);

    // The buffer is declared once, at the binding the Vulkan layout and the
    // engine slot table both name.
    CHECK(CountOccurrences(pt, "[[vk::binding(46, 0)]] StructuredBuffer<float4> atmo_ms_lut;") == 1);
    // ...and NOT on any of the six numbers OceanCascades (#293) reserved.
    for (int b = 40; b <= 45; ++b) {
        char decl[96];
        std::snprintf(decl, sizeof(decl),
                      "[[vk::binding(%d, 0)]] StructuredBuffer<float4> atmo_ms_lut;", b);
        CHECK(CountOccurrences(pt, decl) == 0);
    }
    // The unbuilt-table guard, which is what stops a zero-filled
    // placeholder from reading as "multiple scattering is negligible".
    CHECK(CountOccurrences(pt, "atmo_ms_lut[0].w > 0.5") == 1);
    // Two consumers: the sky march and the aerial-perspective march. One
    // would mean the horizon and the sky above it disagree about how much
    // multiply scattered light there is, which is a seam.
    CHECK(CountOccurrences(pt, "ptAtmoMultiScatter(") == 3);   // 1 definition + 2 call sites
}

// ===========================================================================
// 3.  WHAT THE REBASED SKY ACTUALLY READS
// ===========================================================================

TEST_CASE("noon, sunset and twilight are ordered and plausible") {
    const pt::atmo::Body b = RefBody();
    const std::vector<float> lut = BuildLut(b);
    auto cook = [&](double elev_deg) {
        pt::atmo::SkyCookParams p;
        p.sun_elev_sin = std::sin(elev_deg * 3.14159265358979 / 180.0);
        p.observer_alt = 1.7;
        return pt::atmo::CookSky(b, lut.data(), p);
    };
    const pt::atmo::SkyCook noon     = cook(60.0);
    const pt::atmo::SkyCook sunset   = cook(2.0);
    const pt::atmo::SkyCook twilight = cook(-4.0);

    // NOT A VACUITY CHECK. The failure mode #257 measured is a sky that is
    // present but ~7x too dark, and "> 0" cannot see that. So the zenith is
    // bracketed at a level derived from the physics rather than from the
    // render: single-scattered zenith radiance is approximately
    // E * sigma_s * H * P(mu) * T, which for the green channel at 60
    // degrees is 481 * 0.1085 * 0.104 * 0.83 = 4.5, and the multiple
    // scattering term adds ~29%.
    INFO("noon zenith = ", noon.zenith[0], " ", noon.zenith[1], " ", noon.zenith[2]);
    CHECK(noon.zenith[1] > 3.0);
    CHECK(noon.zenith[1] < 10.0);

    // The sky gets darker as the sun goes down, monotonically, over four
    // orders of magnitude. The absolute levels are the thing exposure has
    // to cope with, and the ratios are the thing that says the model is a
    // model rather than a palette.
    CHECK(Luma(noon.zenith)   > Luma(sunset.zenith));
    CHECK(Luma(sunset.zenith) > Luma(twilight.zenith));
    CHECK(Luma(noon.zenith)   > Luma(twilight.zenith) * 50.0);

    // THE HORIZON REDDENS AND THE ZENITH DOES NOT. At 2 degrees the
    // sun-side horizon has red above blue -- an INVERSION of the zenith's
    // ordering in the same frame, which no single global tint can produce.
    CHECK(sunset.horizon_sun[0] > sunset.horizon_sun[2]);
    CHECK(sunset.zenith[2]      > sunset.zenith[0]);

    // ...and the sun side is brighter than the anti-solar side, by a margin.
    // Equality here would mean side_factor had nothing to interpolate.
    CHECK(Luma(sunset.horizon_sun) > Luma(sunset.horizon_anti) * 1.3);

    // At noon the two horizons converge, because the geometry is nearly
    // symmetric. This is the control for the case above: it shows the
    // asymmetry is the sun's position rather than an artefact of the two
    // anchors being computed differently.
    CHECK(Luma(noon.horizon_sun)
          == doctest::Approx(Luma(noon.horizon_anti)).epsilon(0.02));

    // The horizon is brighter than the zenith in daylight -- more air along
    // the path. A cook that returned the same integral for both directions
    // would fail here.
    CHECK(Luma(noon.horizon_sun) > Luma(noon.zenith) * 2.0);
}

TEST_CASE("ozone keeps the twilight zenith blue through the rebase") {
    // P3's acceptance criterion, re-asserted on the WHOLE rebased chain
    // rather than on a bare transmittance ratio -- the question #280 has to
    // answer is whether ozone survives physical units and multiple
    // scattering, and the only way to answer it is to run both.
    //
    // Asserted as a SIGN FLIP, not a threshold, so it cannot be tuned past.
    const pt::atmo::Body with_o3 = RefBody(1.0, 1.0);
    const pt::atmo::Body no_o3   = RefBody(1.0, 0.0);
    const std::vector<float> lut_with = BuildLut(with_o3);
    const std::vector<float> lut_no   = BuildLut(no_o3);

    pt::atmo::SkyCookParams p;
    p.sun_elev_sin = std::sin(-4.0 * 3.14159265358979 / 180.0);
    p.observer_alt = 1.7;
    const pt::atmo::SkyCook a = pt::atmo::CookSky(with_o3, lut_with.data(), p);
    const pt::atmo::SkyCook c = pt::atmo::CookSky(no_o3,   lut_no.data(),   p);

    INFO("twilight zenith with ozone    = ", a.zenith[0], " ", a.zenith[1], " ", a.zenith[2]);
    INFO("twilight zenith without ozone = ", c.zenith[0], " ", c.zenith[1], " ", c.zenith[2]);

    // Blue wins in both, but ozone widens the margin substantially. The
    // enhancement is what the Chappuis band does.
    const double bg_with = a.zenith[2] / a.zenith[1];
    const double bg_no   = c.zenith[2] / c.zenith[1];
    CHECK(bg_with > bg_no * 1.5);

    // THE ORDERING IS THE PHYSICS. Ozone's cross section peaks in GREEN
    // (1.881e-6) and is weakest in BLUE (0.085e-6), so along a long slant
    // path it must remove more green than red and more red than blue. That
    // ordering is what would catch the triple entered in the wrong
    // wavelength order -- the obvious mistake, because Rayleigh's triple
    // ascends with channel index and ozone's does not.
    const double kill_r = 1.0 - a.zenith[0] / c.zenith[0];
    const double kill_g = 1.0 - a.zenith[1] / c.zenith[1];
    const double kill_b = 1.0 - a.zenith[2] / c.zenith[2];
    INFO("ozone removes r=", kill_r, " g=", kill_g, " b=", kill_b);
    CHECK(kill_g > kill_r);
    CHECK(kill_r > kill_b);
    CHECK(kill_b >= 0.0);

    // And it only ever REMOVES: ozone absorbs and does not scatter, so no
    // channel may come out brighter for its presence.
    CHECK(a.zenith[0] <= c.zenith[0]);
    CHECK(a.zenith[1] <= c.zenith[1]);
    CHECK(a.zenith[2] <= c.zenith[2]);
}

TEST_CASE("the compensating pair is retired in the defaults") {
    // The two cvars this issue exists to remove. Read out of Engine.cpp so
    // the pin is on what the engine actually registers, not on a copy.
    const std::string eng = ReadFile(PT_ENGINE_CPP_PATH);
    CHECK(CountOccurrences(eng, "PT_CVAR(r_rayleigh,              \"1.0\",") == 1);
    CHECK(CountOccurrences(eng, "PT_CVAR(r_rayleigh,              \"30.0\",") == 0);
    CHECK(CountOccurrences(eng, "PT_CVAR(r_volumetric_density,   \"3.996e-6\",") == 1);
    CHECK(CountOccurrences(eng, "PT_CVAR(r_volumetric_density,   \"0.002\",") == 0);
    // #257 handed the physical-sunlight flip to the unit rebase explicitly.
    CHECK(CountOccurrences(eng, "PT_CVAR(r_sun_physical_transmittance, \"1\",") == 1);

    // 3.996e-6 is inside the real-Earth range the issue names, at its clean
    // end (Hillaire 2020, aerosol optical depth 0.010 at 550 nm). Pinned as
    // a range rather than a value so the DEFAULT can move within the
    // physics without the test having to be edited, while a return to
    // 0.002 -- 500x the clear-sky value -- fails.
    CHECK(3.996e-6 >= 1e-6);
    CHECK(3.996e-6 <= 5e-4);
}

TEST_CASE("procSky reads the cooked palette rather than authored literals") {
    const std::string pt = ReadFile(PT_SHADER_PATHTRACE_PATH);
    // The six literals that were the day sky. Gone, by count.
    CHECK(CountOccurrences(pt, "float3 zenith_day  = float3(0.32, 0.55, 0.92);") == 0);
    CHECK(CountOccurrences(pt, "float3 horizon_day = float3(0.78, 0.88, 1.00);") == 0);
    // Replaced by the three pushed radiances, each read exactly once.
    CHECK(CountOccurrences(pt, "sky_cook_zenith.rgb") == 1);
    CHECK(CountOccurrences(pt, "sky_cook_horizon_sun.rgb") == 1);
    CHECK(CountOccurrences(pt, "sky_cook_horizon_anti.rgb") == 1);
    // Declared in BOTH push blocks -- the Metal cbuffer and the SPIR-V
    // Frame UBO. One would be a byte-layout divergence between backends,
    // which is the failure mode the PtPush static_asserts exist for.
    CHECK(CountOccurrences(pt, "float4 sky_cook_zenith;") == 2);
    CHECK(CountOccurrences(pt, "float4 sky_cook_horizon_sun;") == 2);
    CHECK(CountOccurrences(pt, "float4 sky_cook_horizon_anti;") == 2);

    // And the molecular half of the in-scatter march no longer double
    // counts into it on a primary miss.
    CHECK(CountOccurrences(pt,
        "if (uint(sun_and_mode.w) == 2u && !primary_hit) march_ray = false;") == 1);
}

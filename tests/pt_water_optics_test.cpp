// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P7 (#261): underwater.
//
// WHAT THIS FILE IS FOR
//
// The phase's headline acceptance criterion is a claim about a SEQUENCE and
// not about a frame: "descending from the surface to 200 m darkens and
// shifts hue per Pope & Fry -- red extinguishing first, blue last". A golden
// PNG cannot settle that, and neither can two PNGs at the ends of the
// descent: they can both be right while everything between them jumps. Worse
// than for P5 or P6, an underwater frame is ALREADY DARK, so the usual
// feature-deleted vacuity check -- "does the frame change when I turn the
// feature off" -- is satisfied by a black image. Every assertion below is on
// CONTENT.
//
// So the claim is turned into numbers, four ways, in increasing strength:
//
//   1. THE COEFFICIENTS ARE A PUBLISHED MEASUREMENT. Not "look, water is
//      blue" but Pope & Fry 1997's integrating-cavity table, transcribed
//      into src/physics/WaterOptics.cpp and checked here against the
//      anchors the paper itself quotes. Same for the scattering: Morel
//      1974's beta_w(90) fit, with the sphere-integral factor DERIVED from
//      the depolarisation ratio rather than quoted as 16.06.
//
//   2. THE ORDERING HOLDS AT EVERY DEPTH BETWEEN. T_red(d) < T_green(d) <
//      T_blue(d) strictly for every d > 0, and the blue-to-red ratio grows
//      without bound. This is an inequality between exponentials with
//      ordered exponents, so it is asserted with NO tolerance -- it cannot
//      be satisfied by a tuned triple that happens to look blue.
//
//   3. AND THE PATH BETWEEN THEM IS LIPSCHITZ. #260's refinement test,
//      which needs no tolerance at all: sample the observed radiance along
//      a ladder in depth, and the largest adjacent step must HALVE as the
//      rungs double. A continuous descent shrinks; a jump plateaus. The
//      control is the thing this phase must not be -- "black below 185 m",
//      the optical-black depth used as a switch instead of as a bound --
//      and it is asserted to plateau, so the test cannot go vacuous by
//      measuring nothing.
//
//   4. AND THE MARCH ACTUALLY COMPUTES THAT PATH. The shader's stratified
//      Riemann sum is mirrored here and driven against the closed-form
//      homogeneous single-scatter integral it is approximating. Its error
//      QUARTERS as the sample count doubles, which is the signature of a
//      second-order rule -- and which the march only has because it takes
//      its eye transmittance in closed form at the sample rather than as
//      the air march's running product from the cell start. The same case
//      measures what the sample count is really for, and finds that it is
//      NOT the quadrature: two samples already hold the smooth part inside
//      one 8-bit level over the whole 404 m span, so the default of 16 is
//      buying resolution of the sun VISIBILITY, which is a property of the
//      scene and cannot be derived from the water. Stated as a convention,
//      not dressed up as a derivation.
//
// The rest pins the pieces those four rest on: the medium stack (which
// replaces a bit that could not represent nesting), Snell's window from
// below, and the refracted sun direction -- which is the difference between
// a window that is in the right PLACE (all P5's geometric fix could give)
// and one that is correctly SHADED.
//
// WHY IT MIRRORS THE SHADER
//
// The in-scatter march, the medium stack and the phase functions live in
// PathTrace.slang / PathTraceMath.slang and there is no host entry point to
// call, so this file transcribes them -- same operations, same order --
// exactly as tests/pt_planet_ocean_test.cpp and tests/pt_atmosphere_test.cpp
// do for their kernels. A mirror that has drifted is worthless, so the last
// case re-reads the .slang and pins what the transcription depends on. It
// COUNTS occurrences rather than testing find() != npos: a substring pin is
// satisfied by one correct copy however many wrong ones exist elsewhere,
// which is exactly how issue #276 stayed live for a whole cycle underneath a
// passing test.
//
// pt::water is NOT mirrored -- it is a free-function module over glm
// precisely so the tests call the real code.
//
// Deterministic: every input is a literal or derived from literals.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/physics/WaterOptics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pt::water;

namespace {

constexpr double kPi = 3.14159265358979323846;

// --- shader mirror: the medium stack (PathTrace.slang) --------------------
//
//   uint ptMediumDepth(uint s) { return (s >> 16) & 0xFu; }
//   uint ptMediumTop(uint s) {
//       uint d = ptMediumDepth(s);
//       return (d == 0u) ? MEDIUM_AIR : ((s >> ((d - 1u) * 2u)) & 3u);
//   }
//   uint ptMediumPush(uint s, uint id) {
//       uint d = ptMediumDepth(s);
//       if (d >= kMediumStackMax) return s;
//       uint entries = (s & 0xFFFFu) & ~(3u << (d * 2u));
//       entries |= (id & 3u) << (d * 2u);
//       return entries | ((d + 1u) << 16);
//   }
//   uint ptMediumPop(uint s) {
//       uint d = ptMediumDepth(s);
//       if (d == 0u) return s;
//       uint entries = (s & 0xFFFFu) & ~(3u << ((d - 1u) * 2u));
//       return entries | ((d - 1u) << 16);
//   }
constexpr std::uint32_t kMediumAir        = 0u;
constexpr std::uint32_t kMediumWater      = 1u;
constexpr std::uint32_t kMediumDielectric = 2u;
constexpr std::uint32_t kMediumStackMax   = 8u;

std::uint32_t mediumDepth(std::uint32_t s) { return (s >> 16) & 0xFu; }

std::uint32_t mediumTop(std::uint32_t s) {
    const std::uint32_t d = mediumDepth(s);
    return (d == 0u) ? kMediumAir : ((s >> ((d - 1u) * 2u)) & 3u);
}

std::uint32_t mediumPush(std::uint32_t s, std::uint32_t id) {
    const std::uint32_t d = mediumDepth(s);
    if (d >= kMediumStackMax) return s;
    std::uint32_t entries = (s & 0xFFFFu) & ~(3u << (d * 2u));
    entries |= (id & 3u) << (d * 2u);
    return entries | ((d + 1u) << 16);
}

std::uint32_t mediumPop(std::uint32_t s) {
    const std::uint32_t d = mediumDepth(s);
    if (d == 0u) return s;
    std::uint32_t entries = (s & 0xFFFFu) & ~(3u << ((d - 1u) * 2u));
    return entries | ((d - 1u) << 16);
}

bool mediumIsWater(std::uint32_t s) { return mediumTop(s) == kMediumWater; }

// --- shader mirror: ptPhaseRayleighDepol / ptPhaseHenyeyGreenstein --------
//
//   float k = (1.0 - d) / (1.0 + d);
//   return (1.0 + k * mu * mu) / (4.0 * PI * (1.0 + k / 3.0));
//
//   float g2 = g * g;
//   return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0*g*mu, 1e-4), 1.5));
double phaseHG(double mu, double g) {
    const double g2 = g * g;
    return (1.0 - g2) /
           (4.0 * kPi * std::pow(std::max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// --- The engine's own pure-seawater working point -------------------------
// Absorption is r_water_absorption_*'s DEFAULT (Pope & Fry); scattering is
// what the host pushes unconditionally (Morel). Both read from the real
// module, so a change to either shows up in every case below.
glm::dvec3 sigmaA() { return PureWaterAbsorptionRgb(); }
glm::dvec3 sigmaS() { return PureSeawaterScatteringRgb(); }
glm::dvec3 sigmaT() { return sigmaA() + sigmaS(); }

// --- shader mirror: the in-scatter march ----------------------------------
//
// One channel of the loop in PathTrace.slang, for the geometry the descent
// fixture actually renders: the camera at depth D looking straight up, so a
// sample `t` from the camera sits at depth D - t and the sun's slant path
// back to the surface is (D - t) / cos(theta_water).
//
//   for (int wi = 0; wi < N_w; ++wi) {
//       float t = (float(wi) + jit_w) * dt_w;
//       float3 trans_eye_w = exp(-sigma_t * t);
//       ... v_color_w += sun_rad_w * trans_w * in_scatter_w * dt_w
//                      * trans_eye_w;
//   }
double marchOne(double sigma_s_phase, double sigma_t, double depth_m,
                double cos_theta_w, double t_far, int n, double jitter) {
    const double dt = t_far / static_cast<double>(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double t = (static_cast<double>(i) + jitter) * dt;
        if (t >= t_far) break;
        const double trans_sun =
            std::exp(-sigma_t * (depth_m - t) / cos_theta_w);
        const double trans_eye = std::exp(-sigma_t * t);
        sum += trans_sun * sigma_s_phase * dt * trans_eye;
    }
    return sum;
}

// The closed form the march is approximating. Homogeneous medium, so
// exp(-a(D-t)/c) exp(-a t) integrates in one line.
double marchClosedForm(double sigma_s_phase, double sigma_t, double depth_m,
                       double cos_theta_w, double t_far) {
    const double k = sigma_t * (1.0 / cos_theta_w - 1.0);
    const double lead = std::exp(-sigma_t * depth_m / cos_theta_w);
    if (std::fabs(k) < 1e-12) return sigma_s_phase * lead * t_far;
    return sigma_s_phase * lead * (std::exp(k * t_far) - 1.0) / k;
}

// What a camera at depth D looking straight up actually measures: the sky
// through Snell's window, attenuated by the column, PLUS everything the
// column scattered into the eye on the way. One channel.
double observedUpward(double sky_radiance, double sun_radiance,
                      double sigma_s_phase, double sigma_t, double depth_m,
                      double cos_theta_w) {
    return sky_radiance * std::exp(-sigma_t * depth_m) +
           sun_radiance * marchClosedForm(sigma_s_phase, sigma_t, depth_m,
                                          cos_theta_w, depth_m);
}

// --- mirror-drift helpers (same shape as pt_planet_ocean_test) ------------
std::string readAll(const char* path) {
    std::ifstream f(path);
    REQUIRE_MESSAGE(f.good(), "cannot open ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string stripSpace(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out.push_back(c);
    }
    return out;
}

std::size_t countOf(const std::string& hay, const std::string& needle) {
    std::size_t n = 0, at = 0;
    while ((at = hay.find(needle, at)) != std::string::npos) { ++n; ++at; }
    return n;
}

}  // namespace

// ===========================================================================
// 1. THE COEFFICIENTS ARE A PUBLISHED MEASUREMENT.
// ===========================================================================

TEST_CASE("the Pope & Fry table is the paper's, not a look-alike") {
    const double* a = PopeFryAbsorptionTable();

    // The grid is exactly the range the paper measured.
    CHECK(kPopeFrySamples == 65u);
    CHECK(kPopeFryLambdaMinNm + (kPopeFrySamples - 1) * kPopeFryStepNm ==
          doctest::Approx(kPopeFryLambdaMaxNm));

    // ANCHOR 1 -- the paper's abstract quotes the absorption MINIMUM as
    // 0.0044 +/- 0.0006 /m at 418 nm. That single number is the headline
    // result of the whole paper (it is several times lower than every
    // pre-1990 determination, which is why an older table makes clear water
    // render like tea), so a transcription that gets it right is a
    // transcription of the right dataset.
    std::size_t argmin = 0;
    for (std::size_t i = 1; i < kPopeFrySamples; ++i) {
        if (a[i] < a[argmin]) argmin = i;
    }
    const double lambda_min =
        kPopeFryLambdaMinNm + static_cast<double>(argmin) * kPopeFryStepNm;
    CAPTURE(lambda_min);
    CAPTURE(a[argmin]);
    // 415 nm on THIS 5 nm grid; the paper's own 2.5 nm grid puts the true
    // minimum at 417.5 nm / 0.00442, and 418 is where the paper rounds it.
    // Assert against the paper's uncertainty band rather than against the
    // decimated grid point, because the band is the published claim.
    CHECK(std::fabs(lambda_min - 418.0) <= 5.0);
    CHECK(std::fabs(a[argmin] - 0.0044) <= 0.0006);

    // ANCHOR 2 -- the red end. 0.624 /m at 700 nm is 141x the minimum, and
    // that ratio is what makes red the first channel to die.
    CHECK(a[kPopeFrySamples - 1] == doctest::Approx(0.624));
    CHECK(a[kPopeFrySamples - 1] / a[argmin] > 140.0);

    // STRUCTURE no typo survives: strictly rising from the minimum to the
    // red end. A transposed digit anywhere in the last 57 entries breaks
    // this, and so does a dropped or duplicated row.
    for (std::size_t i = argmin; i + 1 < kPopeFrySamples; ++i) {
        CAPTURE(i);
        CHECK(a[i + 1] > a[i]);
    }
    // ...and strictly falling from 380 nm down to it.
    for (std::size_t i = 0; i < argmin; ++i) {
        CAPTURE(i);
        CHECK(a[i] > a[i + 1]);
    }
}

TEST_CASE("the RGB triple is the table sampled, and #261's red row is wrong") {
    const glm::dvec3 a = PureWaterAbsorptionRgb();

    // The three named wavelengths, straight out of the table.
    CHECK(a.x == doctest::Approx(0.2755));    // 620 nm
    CHECK(a.y == doctest::Approx(0.0565));    // 550 nm
    CHECK(a.z == doctest::Approx(0.00922));   // 450 nm

    // The 1/e depths. #261's table quotes 4.1 / 17.7 / 109 m; the green and
    // blue rows match the paper and reproduce here exactly, and the red one
    // does not.
    const glm::dvec3 len = AttenuationLengths(a);
    CHECK(len.y == doctest::Approx(17.699).epsilon(1e-3));
    CHECK(len.z == doctest::Approx(108.46).epsilon(1e-3));
    CHECK(len.x == doctest::Approx(3.6298).epsilon(1e-3));

    // THE CORRECTION, PINNED AS A DECISION AND NOT LEFT AS A DRIFT.
    // #261 gives a(620) = 0.245 /m, 1/e = 4.1 m. Pope & Fry's tabulated
    // value at 620 nm is 0.2755; 0.245 falls at about 602 nm on their own
    // curve, so the issue's row reads like a transposed 602 -> 620. The
    // engine uses the paper. This case exists so that if someone later
    // "fixes" the code back to the issue's number, a test says why not.
    CHECK(a.x != doctest::Approx(0.245));
    // Where 0.245 actually lives: bracket it and check the bracket is near
    // 602 nm, not near 620.
    double lam_lo = kPopeFryLambdaMinNm;
    for (double l = kPopeFryLambdaMinNm; l <= kPopeFryLambdaMaxNm; l += 0.5) {
        if (PureWaterAbsorption(l) <= 0.245) lam_lo = l;
    }
    CAPTURE(lam_lo);
    CHECK(lam_lo > 598.0);
    CHECK(lam_lo < 606.0);

    // Interpolation is exact at grid points and clamped outside the
    // measured range -- past 700 nm the real spectrum climbs through the
    // 760 nm band by another order of magnitude, and a linear continuation
    // of the last two samples would be fiction presented as data.
    CHECK(PureWaterAbsorption(500.0) == doctest::Approx(0.0204));
    CHECK(PureWaterAbsorption(200.0) ==
          doctest::Approx(PureWaterAbsorption(380.0)));
    CHECK(PureWaterAbsorption(900.0) ==
          doctest::Approx(PureWaterAbsorption(700.0)));
    // Midpoint of a grid cell is the mean of its ends, by construction.
    CHECK(PureWaterAbsorption(497.5) ==
          doctest::Approx(0.5 * (PureWaterAbsorption(495.0) +
                                 PureWaterAbsorption(500.0))));
}

TEST_CASE("Morel's scattering factor is derived, not the quoted 16.06") {
    // Ocean optics uses "b_w = 16.06 beta_w(90)" as a constant. It is not a
    // constant, it is a function of the depolarisation ratio, and writing it
    // down as 16.06 is exactly how a factor and the ratio it came from drift
    // apart. Derive it: integrating beta(90)[1 + k mu^2] over the sphere
    // with k = (1-d)/(1+d) gives 4 pi (1 + k/3).
    CHECK(RayleighDepolarisedSphereFactor(kSeawaterDepolarisation) ==
          doctest::Approx(16.0648).epsilon(1e-4));
    // At zero depolarisation it must reduce to pure Rayleigh's 16 pi / 3.
    CHECK(RayleighDepolarisedSphereFactor(0.0) ==
          doctest::Approx(16.0 * kPi / 3.0));
    // ...and at total depolarisation, to isotropic 4 pi.
    CHECK(RayleighDepolarisedSphereFactor(1.0) == doctest::Approx(4.0 * kPi));

    // Morel's fit at its own reference wavelength and mean ocean salinity.
    // #261 quotes ~0.0029 /m from Smith & Baker 1981; this reproduces it
    // from Morel's beta_w(90) rather than from a second table, which is the
    // point -- the two published routes agree.
    CHECK(PureSeawaterScattering(500.0, kOceanSalinityPpt) ==
          doctest::Approx(2.846e-3).epsilon(1e-3));
    // Salinity term: pure water (S = 0) is 1 / (1 + 0.3 * 35 / 37) of it.
    CHECK(PureSeawaterScattering(500.0, 0.0) /
              PureSeawaterScattering(500.0, kOceanSalinityPpt) ==
          doctest::Approx(1.0 / (1.0 + 0.3 * 35.0 / 37.0)));

    // The spectral shape is exactly lambda^-4.32 -- a ratio between any two
    // wavelengths must be the power law with no residual, which is a
    // statement about the code and needs no tolerance beyond rounding.
    for (double l : {380.0, 450.0, 550.0, 620.0, 700.0}) {
        CAPTURE(l);
        CHECK(PureSeawaterScattering(l, kOceanSalinityPpt) /
                  PureSeawaterScattering(500.0, kOceanSalinityPpt) ==
              doctest::Approx(std::pow(l / 500.0, -4.32)).epsilon(1e-12));
    }

    // The RGB triple, and the ordering that makes water BLUE rather than
    // merely dark: blue scatters four times as much as red.
    const glm::dvec3 b = PureSeawaterScatteringRgb();
    CHECK(b.z > b.y);
    CHECK(b.y > b.x);
    CHECK(b.z / b.x == doctest::Approx(std::pow(450.0 / 620.0, -4.32)));
}

TEST_CASE("the depolarised Rayleigh phase function integrates to one") {
    // It is a ratio of two expressions that share their only constant, so
    // this is closer to an identity than to a numerical agreement. The
    // tolerance is set by the QUADRATURE, not by the physics: Simpson on
    // 2000 intervals over a smooth quadratic-in-mu integrand is exact to
    // rounding, so 1e-10 is generous by many orders and any real failure
    // would be a factor-of-something, not a last-digit drift.
    for (double d : {0.0, 0.09, 0.5, 1.0}) {
        CAPTURE(d);
        const int n = 2000;
        double sum = 0.0;
        for (int i = 0; i <= n; ++i) {
            const double th = kPi * static_cast<double>(i) /
                              static_cast<double>(n);
            const double w = (i == 0 || i == n) ? 1.0 : ((i % 2) ? 4.0 : 2.0);
            sum += w * RayleighDepolarisedPhase(std::cos(th), d) *
                   std::sin(th);
        }
        sum *= (kPi / static_cast<double>(n)) / 3.0 * 2.0 * kPi;
        CHECK(sum == doctest::Approx(1.0).epsilon(1e-10));
    }
    // At d = 0 it IS ptPhaseRayleigh, 3/(16 pi) (1 + mu^2). The shader
    // keeps both functions, so pin that they agree where they overlap --
    // otherwise a change to one silently forks the physics of the other.
    for (double mu : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
        CAPTURE(mu);
        CHECK(RayleighDepolarisedPhase(mu, 0.0) ==
              doctest::Approx(3.0 / (16.0 * kPi) * (1.0 + mu * mu)));
    }
    // And it is NOT the forward-peaked lobe #261 proposes for everything.
    // The Petzold HG at g = 0.924 is three orders of magnitude more
    // forward-biased than seawater's own molecules; using one lobe for both
    // would make PURE water forward-scattering and kill the blue glow that
    // is the whole reason a clear ocean looks like water.
    const double fwd_ratio = phaseHG(1.0, kPetzoldAsymmetry) /
                             RayleighDepolarisedPhase(1.0,
                                                      kSeawaterDepolarisation);
    CAPTURE(fwd_ratio);
    // MEASURED: 232x. The bar is 200 rather than the measured value so a
    // small change to the Petzold asymmetry does not fail the case for the
    // wrong reason, but 232 is recorded here so a change that halves it is
    // visible as a change rather than as noise.
    CHECK(fwd_ratio == doctest::Approx(232.06).epsilon(1e-3));
    CHECK(fwd_ratio > 200.0);
    // The molecular lobe is near-symmetric: forward and backward within 15%.
    const double sym = RayleighDepolarisedPhase(1.0, kSeawaterDepolarisation) /
                       RayleighDepolarisedPhase(0.0, kSeawaterDepolarisation);
    CAPTURE(sym);
    CHECK(sym < 1.85);
    CHECK(sym > 1.0);
}

TEST_CASE("pure seawater's single-scatter albedo is why the ocean is blue") {
    const glm::dvec3 a = sigmaA();
    const glm::dvec3 b = sigmaS();
    const glm::dvec3 w = b / (a + b);
    CAPTURE(w.x); CAPTURE(w.y); CAPTURE(w.z);
    // Red scatters essentially nothing; blue scatters a third of what it
    // removes. That is a two-order-of-magnitude spread and it is the whole
    // mechanism -- absorption alone gives a DARK column, not a blue one.
    CHECK(w.x == doctest::Approx(0.004062).epsilon(1e-3));
    CHECK(w.y == doctest::Approx(0.032293).epsilon(1e-3));
    CHECK(w.z == doctest::Approx(0.32731).epsilon(1e-3));
    CHECK(w.z / w.x > 75.0);
}

TEST_CASE("the optical-black depth is derived from whatever water is in force") {
    // Pure seawater: absorption plus BOTH scattering species.
    const double d_pure = OpticalBlackDepth(sigmaT());
    CAPTURE(d_pure);
    CHECK(d_pure == doctest::Approx(404.3).epsilon(1e-3));
    // ...and it really is one 8-bit level at that depth, in every channel.
    const glm::dvec3 t = glm::exp(-sigmaT() * d_pure);
    CHECK(std::max(t.x, std::max(t.y, t.z)) ==
          doctest::Approx(1.0 / 255.0).epsilon(1e-9));

    // #261's own ~185 m is the same expression at its coastal c_B = 0.03,
    // and reproducing it is what shows the two are one formula rather than
    // two numbers. This is the number that hard-caps seabed chunk LOD.
    CHECK(OpticalBlackDepth(glm::dvec3(1.0, 0.5, 0.03)) ==
          doctest::Approx(184.7).epsilon(1e-3));

    // Turbid water goes black sooner, monotonically, with no special case.
    double prev = 1e30;
    for (double bp : {0.0, 0.05, 0.1, 0.3, 0.5}) {
        const double d = OpticalBlackDepth(sigmaT() + glm::dvec3(bp));
        CAPTURE(bp);
        CHECK(d < prev);
        prev = d;
    }
    // Degenerate input must not produce a NaN span for the march to walk.
    CHECK(OpticalBlackDepth(glm::dvec3(0.0)) == 0.0);
}

// ===========================================================================
// 2. THE ORDERING HOLDS AT EVERY DEPTH BETWEEN. No tolerance.
// ===========================================================================

TEST_CASE("red extinguishes first and blue last, at every depth") {
    const glm::dvec3 st = sigmaT();
    // The premise, as an inequality on the coefficients themselves.
    REQUIRE(st.x > st.y);
    REQUIRE(st.y > st.z);

    // 400 rungs from 1 cm to 200 m. Strict inequalities: this is
    // exp(-a d) with ordered a, so there is nothing to tolerance.
    double prev_ratio = 0.0;
    for (int i = 1; i <= 400; ++i) {
        const double d = 0.01 * std::pow(20000.0, static_cast<double>(i) / 400.0);
        const glm::dvec3 t = glm::exp(-st * d);
        CAPTURE(d);
        CHECK(t.x < t.y);
        CHECK(t.y < t.z);
        // Everything darkens, monotonically, with no rung going back up.
        CHECK(t.x <= 1.0);
        // ...and the hue shifts toward blue without bound, which is the
        // "shifts hue" half of the claim. A tint that merely dimmed would
        // hold this ratio constant.
        const double ratio = t.z / t.x;
        CHECK(ratio > prev_ratio);
        prev_ratio = ratio;
    }
    CAPTURE(prev_ratio);
    CHECK(prev_ratio > 1e22);

    // The depths #261 asks about by name, as e-foldings rather than as an
    // eyeball. "Red gone by ~10 m" is 2.8 e-foldings; "blue persisting to
    // ~150 m" is 2.1.
    CHECK(st.x * 10.0 > 2.7);      // red: gone at 10 m
    CHECK(st.y * 50.0 > 2.8);      // green: gone at 50 m
    CHECK(st.z * 150.0 < 2.5);     // blue: still there at 150 m
    CHECK(std::exp(-st.z * 150.0) > 0.12);
}

// ===========================================================================
// 3. THE SEQUENCE: the descent is continuous, by refinement.
// ===========================================================================

TEST_CASE("the descent to 200 m is continuous, by refinement") {
    // #261's headline is a claim about a DESCENT, not about two frames. A
    // golden PNG cannot settle it and neither can a threshold on adjacent
    // frames -- adjacent frames are SUPPOSED to differ, because the water
    // really is getting darker.
    //
    // Continuity has an exact definition that needs no tolerance: refine
    // the sampling and the largest adjacent step must shrink with it. For a
    // Lipschitz function sampled at spacing h the largest adjacent
    // difference is bounded by K*h, so halving h halves it; for a function
    // with a jump of size J the largest difference is at least J however
    // fine the sampling gets. The two behaviours are qualitatively
    // different and no constant has to be chosen to tell them apart.
    //
    // The bar is 0.6 rather than the ideal 0.5 for #260's reason: the
    // location of the maximum moves between refinements, so the halved
    // ladder does not sample the same worst point.
    const glm::dvec3 st = sigmaT();
    const glm::dvec3 ss = sigmaS();
    // Sun 60 degrees up, refracted into the water: 30 degrees from the
    // zenith in air becomes asin(sin 30 / 1.33) = 22.08 degrees below.
    const double cos_w = std::cos(std::asin(std::sin(30.0 * kPi / 180.0) / 1.33));
    const double phase = RayleighDepolarisedPhase(cos_w, kSeawaterDepolarisation);

    auto observedLuma = [&](double depth) {
        double sum = 0.0;
        for (int c = 0; c < 3; ++c) {
            sum += observedUpward(1.0, 80.0, ss[c] * phase, st[c], depth,
                                  cos_w);
        }
        return sum;
    };

    auto worstStep = [&](int rungs) {
        double worst = 0.0;
        double prev = observedLuma(0.0);
        for (int i = 1; i <= rungs; ++i) {
            const double d = 200.0 * static_cast<double>(i) /
                             static_cast<double>(rungs);
            const double cur = observedLuma(d);
            worst = std::max(worst, std::fabs(cur - prev));
            prev = cur;
        }
        return worst;
    };
    const double w1 = worstStep(256);
    const double w2 = worstStep(512);
    const double w3 = worstStep(1024);
    CAPTURE(w1); CAPTURE(w2); CAPTURE(w3);
    CHECK(w1 > 0.0);
    CHECK(w2 < 0.6 * w1);
    CHECK(w3 < 0.6 * w2);

    // THE CONTROL, so the case above cannot pass by measuring nothing. This
    // is the thing #261 must not be: use the optical-black depth as a
    // SWITCH instead of as a bound -- "below 185 m the water is black, so
    // stop" -- which is exactly the shape of shortcut the 185 m figure
    // invites. Its worst step is the size of the jump at every refinement,
    // so the ratio plateaus at 1.
    auto worstStepSwitch = [&](int rungs) {
        auto hard = [&](double d) {
            return (d < 184.7) ? observedLuma(d) : 0.0;
        };
        double worst = 0.0;
        double prev = hard(0.0);
        for (int i = 1; i <= rungs; ++i) {
            const double d = 200.0 * static_cast<double>(i) /
                             static_cast<double>(rungs);
            const double cur = hard(d);
            worst = std::max(worst, std::fabs(cur - prev));
            prev = cur;
        }
        return worst;
    };
    const double s1 = worstStepSwitch(256);
    const double s2 = worstStepSwitch(512);
    const double s3 = worstStepSwitch(1024);
    CAPTURE(s1); CAPTURE(s2); CAPTURE(s3);
    // The control does NOT hold s exactly constant, and the reason is worth
    // stating rather than papering over with a loose Approx: the jump's
    // measured size is the value at the last rung BELOW the cutoff, and
    // that rung moves as the ladder refines. Its drift is bounded by one
    // rung's worth of the underlying continuous function, i.e. by w1 --
    // which is 0.202 against a jump of 0.592, so the plateau can wobble by
    // at most a third of a per cent per refinement. MEASURED: 0.59206 ->
    // 0.59206 -> 0.59096, a 0.19% total drift over two refinements.
    CHECK(s2 == doctest::Approx(s1).epsilon(0.01));
    CHECK(s3 == doctest::Approx(s1).epsilon(0.01));
    // THE QUALITATIVE STATEMENT, which is the one that matters: refining
    // does not shrink it. The continuous case above more than halves twice
    // over; this one does not shrink by even 40% across four times the
    // rungs.
    CHECK(s2 > 0.6 * s1);
    CHECK(s3 > 0.6 * s2);
}

TEST_CASE("the descent darkens, and the in-scatter is what keeps it water") {
    const glm::dvec3 st = sigmaT();
    const glm::dvec3 ss = sigmaS();
    const double cos_w = std::cos(std::asin(std::sin(30.0 * kPi / 180.0) / 1.33));
    const double phase = RayleighDepolarisedPhase(cos_w, kSeawaterDepolarisation);

    auto observed = [&](double depth, bool with_inscatter) {
        glm::dvec3 out(0.0);
        for (int c = 0; c < 3; ++c) {
            out[c] = 1.0 * std::exp(-st[c] * depth);
            if (with_inscatter) {
                out[c] += 80.0 * marchClosedForm(ss[c] * phase, st[c], depth,
                                                 cos_w, depth);
            }
        }
        return out;
    };

    // THE DESCENT IS NOT MONOTONE FROM THE FIRST METRE, AND THAT IS THE
    // PHYSICS, NOT A BUG. Writing the closed form out makes it exact:
    //
    //   in_scatter(D) = A [ exp(-sigma D) - exp(-sigma D / c) ],
    //   A = sigma_s p / (sigma (1/c - 1))
    //
    // Both terms are positive and the second decays faster, so the
    // in-scattered glow RISES from zero, peaks, and then falls. Its peak is
    // where the two decay rates balance:
    //
    //   D* = ln(1/c) / (sigma (1/c - 1))
    //
    // That is a derived depth, not a fitted one, and it is a real thing
    // divers see: the water gets bluer and brighter for the first few tens
    // of metres before absorption takes over. Asserting "every channel
    // darkens from the surface down" would be asserting that this does not
    // happen -- and it would have been satisfiable only by leaving the
    // in-scatter out, which is the state #261 is fixing.
    glm::dvec3 peak_depth(0.0);
    for (int c = 0; c < 3; ++c) {
        peak_depth[c] = std::log(1.0 / cos_w) / (st[c] * (1.0 / cos_w - 1.0));
    }
    CAPTURE(peak_depth.x); CAPTURE(peak_depth.y); CAPTURE(peak_depth.z);
    // And the separation between the three peaks is "red first, blue last"
    // said a second way: red's glow is over by 3.5 m, blue's peaks at 70.
    CHECK(peak_depth.x == doctest::Approx(3.48).epsilon(1e-2));
    CHECK(peak_depth.y == doctest::Approx(16.5).epsilon(1e-2));
    CHECK(peak_depth.z == doctest::Approx(70.2).epsilon(1e-2));
    CHECK(peak_depth.z / peak_depth.x > 19.0);

    // Verify the derived peak against the function itself: each channel's
    // in-scatter really does turn over there, to within the sampling.
    for (int c = 0; c < 3; ++c) {
        const double dpk = peak_depth[c];
        const double at   = marchClosedForm(ss[c] * phase, st[c], dpk, cos_w, dpk);
        const double before = marchClosedForm(ss[c] * phase, st[c], dpk * 0.9,
                                              cos_w, dpk * 0.9);
        const double after  = marchClosedForm(ss[c] * phase, st[c], dpk * 1.1,
                                              cos_w, dpk * 1.1);
        CAPTURE(c);
        CHECK(at > before);
        CHECK(at > after);
    }

    // BEYOND ITS OWN PEAK, EVERY CHANNEL DARKENS MONOTONICALLY, all the way
    // to 200 m. This is the acceptance criterion's "darkens", stated where
    // it is actually true and with the boundary derived rather than picked.
    glm::dvec3 prev = observed(peak_depth.z, true);
    for (int i = 1; i <= 400; ++i) {
        const double d = peak_depth.z +
                         (200.0 - peak_depth.z) * static_cast<double>(i) / 400.0;
        const glm::dvec3 cur = observed(d, true);
        CAPTURE(d);
        CHECK(cur.x <= prev.x);
        CHECK(cur.y <= prev.y);
        CHECK(cur.z <= prev.z);
        prev = cur;
    }
    // ...and the transmitted half -- the Pope & Fry claim proper -- darkens
    // from the very first millimetre, in every channel, with no exception.
    glm::dvec3 tprev(1.0);
    for (int i = 1; i <= 400; ++i) {
        const double d = 200.0 * static_cast<double>(i) / 400.0;
        const glm::dvec3 cur = observed(d, false);
        CAPTURE(d);
        CHECK(cur.x < tprev.x);
        CHECK(cur.y < tprev.y);
        CHECK(cur.z < tprev.z);
        tprev = cur;
    }
    // AT 200 m IT IS NOT YET OVER, AND THAT IS WORTH PINNING BECAUSE #261
    // IMPLIES OTHERWISE. The issue says "effectively black by ~185 m", but
    // that figure comes from its own coastal c_B = 0.03 /m, not from the
    // pure-water table the same issue quotes two lines above it. In real
    // open ocean the blue channel still transmits 6.4% at 200 m -- sixteen
    // of 255 levels, plainly visible -- and one level is not reached until
    // 404 m. Both numbers are right for their own water; the engine
    // computes whichever is in force, and this case is here so nobody
    // hard-codes 185.
    CHECK(observed(200.0, false).z == doctest::Approx(0.06449).epsilon(1e-3));
    CHECK(observed(200.0, false).z * 255.0 > 16.0);
    CHECK(observed(200.0, false).x < 1.0 / 255.0);       // red: long gone
    CHECK(observed(200.0, false).y < 1.0 / 255.0);       // green: gone too
    // Where it IS over, for this water: the derived optical-black depth.
    CHECK(observed(OpticalBlackDepth(st), false).z <=
          doctest::Approx(1.0 / 255.0));

    // THE ABLATION, ASSERTED ON CONTENT AND NOT ON "IT CHANGED". An
    // underwater frame is already dark, so "the picture differs with the
    // feature off" is satisfied by black. What must be true is a specific
    // SHAPE: without in-scatter the column's colour is pure transmitted
    // sky, so the blue-to-red ratio is the transmittance ratio and nothing
    // else; with it, blue picks up scattered light the red channel cannot,
    // and the ratio is strictly larger at every depth.
    for (double d : {1.0, 5.0, 20.0, 50.0, 100.0, 200.0}) {
        const glm::dvec3 on  = observed(d, true);
        const glm::dvec3 off = observed(d, false);
        CAPTURE(d);
        // In-scatter only ever ADDS light -- it is an emission term.
        CHECK(on.x >= off.x);
        CHECK(on.z > off.z);
        // ...and it adds relatively more blue than red, at every depth.
        CHECK(on.z / on.x > off.z / off.x);
    }

    // AND THE MAGNITUDE MATTERS, which is what makes this not vacuous. Past
    // the green channel's 1/e depth the in-scattered blue is not a garnish:
    // it is most of what is left of the frame. Under 20% would be a term
    // nobody would notice was missing, which is the state #261 describes.
    const glm::dvec3 deep_on  = observed(60.0, true);
    const glm::dvec3 deep_off = observed(60.0, false);
    const double share = (deep_on.z - deep_off.z) / deep_on.z;
    CAPTURE(share);
    CHECK(share > 0.20);
}

// ===========================================================================
// 4. THE MARCH COMPUTES THAT PATH, and its sample count is derived.
// ===========================================================================

TEST_CASE("the in-scatter march converges to the closed form, second order") {
    // The shader's loop is a stratified quadrature of an integral that, for
    // a homogeneous column, has a closed form. So the march can be checked
    // against the exact answer rather than against itself at a different
    // resolution -- and the RATE tells us whether the quadrature is the one
    // we think it is.
    //
    // WHY THIS CASE EXISTS. The march originally reused the air march's
    // running-product transmittance, which evaluates exp(-sigma_t s) at the
    // CELL START while the sample sits at (wi + jitter) dt. That is an
    // O(dt) bias that no amount of temporal accumulation converges away.
    // With the closed-form per-sample transmittance the midpoint rule is
    // second order, and the error QUARTERS per doubling. A first-order rule
    // would only halve, so this ladder distinguishes the two without any
    // threshold being chosen.
    const glm::dvec3 st = sigmaT();
    const glm::dvec3 ss = sigmaS();
    const double cos_w = std::cos(std::asin(std::sin(30.0 * kPi / 180.0) / 1.33));
    const double phase = RayleighDepolarisedPhase(cos_w, kSeawaterDepolarisation);
    const double depth = 60.0;

    const double exact =
        marchClosedForm(ss.z * phase, st.z, depth, cos_w, depth);
    REQUIRE(exact > 0.0);

    double prev_err = 0.0;
    for (int n : {4, 8, 16, 32, 64}) {
        const double got =
            marchOne(ss.z * phase, st.z, depth, cos_w, depth, n, 0.5);
        const double err = std::fabs(got - exact) / exact;
        CAPTURE(n);
        CAPTURE(err);
        if (prev_err > 0.0) {
            // Second order: quartering, with 0.30 rather than 0.25 as the
            // bar because the constant in the O(h^2) term is only
            // asymptotically constant.
            CHECK(err < 0.30 * prev_err);
        }
        prev_err = err;
    }

    // UNBIASED IN THE JITTER. The shader jitters per pixel, not at 0.5, so
    // the estimator has to be right in expectation and not merely at the
    // midpoint. Average over a fine sweep of jitters and the mean lands on
    // the exact answer to the quadrature's own accuracy.
    double mean = 0.0;
    const int kJit = 512;
    for (int j = 0; j < kJit; ++j) {
        mean += marchOne(ss.z * phase, st.z, depth, cos_w, depth, 16,
                         (static_cast<double>(j) + 0.5) /
                             static_cast<double>(kJit));
    }
    mean /= static_cast<double>(kJit);
    CHECK(mean == doctest::Approx(exact).epsilon(2e-3));
}

TEST_CASE("the quadrature is not what sets the sample count, and that is measured") {
    // The default has to come from somewhere, and "16 looked fine" is the
    // kind of number that survives for years. So MEASURE what the sample
    // count actually buys, and say which of the two jobs it is doing.
    //
    // The bar is display-referred: an absolute error smaller than one 8-bit
    // level of the BRIGHTEST value the in-scatter term ever reaches. It has
    // to be absolute rather than relative, because relative error at 400 m
    // is a ratio of two numbers around e^-120 -- meaningless, and it is
    // what a naive relative bound would spend the whole sample budget
    // chasing.
    const glm::dvec3 st = sigmaT();
    const glm::dvec3 ss = sigmaS();
    const double cos_w = std::cos(std::asin(std::sin(30.0 * kPi / 180.0) / 1.33));
    const double phase = RayleighDepolarisedPhase(cos_w, kSeawaterDepolarisation);
    const double black = OpticalBlackDepth(st);

    double peak = 0.0;
    for (int i = 1; i <= 200; ++i) {
        const double depth = black * static_cast<double>(i) / 200.0;
        for (int c = 0; c < 3; ++c) {
            peak = std::max(peak, marchClosedForm(ss[c] * phase, st[c], depth,
                                                  cos_w, depth));
        }
    }
    REQUIRE(peak > 0.0);

    auto worstAt = [&](int n) {
        double worst = 0.0;
        for (int i = 1; i <= 200; ++i) {
            const double depth = black * static_cast<double>(i) / 200.0;
            for (int c = 0; c < 3; ++c) {
                const double exact = marchClosedForm(ss[c] * phase, st[c],
                                                     depth, cos_w, depth);
                const double got = marchOne(ss[c] * phase, st[c], depth,
                                            cos_w, depth, n, 0.5);
                worst = std::max(worst, std::fabs(got - exact) / peak);
            }
        }
        return worst;
    };

    // THE FINDING, and it is not the one the sample count implies. TWO
    // samples already hold the smooth part of the integral inside one 8-bit
    // level over the entire 0 - 404 m span. That is not luck: for a
    // homogeneous column the integrand is exp(-sigma_t D / c) times
    // exp(sigma_t t (1/c - 1)), and at the blue channel's sigma_t = 0.0137
    // and a 60-degree sun that exponent moves by 0.44 across four hundred
    // metres. There is almost nothing there to resolve.
    const double w2  = worstAt(2);
    const double w16 = worstAt(16);
    CAPTURE(w2);
    CAPTURE(w16);
    CHECK(w2 < 1.0 / 255.0);
    CHECK(w16 < w2);
    // Measured margins, recorded so the next reader does not have to
    // re-derive them: N = 2 lands at 2.21e-4 of peak, 18x under the 3.92e-3
    // one-level bound; N = 16 at 3.46e-6, 1134x under it.
    CHECK(w2  == doctest::Approx(2.211e-4).epsilon(0.05));
    CHECK(w16 == doctest::Approx(3.455e-6).epsilon(0.05));

    // SO WHAT IS r_water_inscatter_samples FOR? The other half of the
    // integrand, which this closed form cannot model: the per-sample sun
    // NEE is a VISIBILITY test, and a shaft edge is a step function in t.
    // Its Nyquist limit is a property of the scene's occluders, not of the
    // water, so it cannot be derived from the coefficients here and the
    // default is set by consistency instead -- 16, the same as
    // r_volumetric_samples, so an underwater frame and an air frame spend
    // the same number of shadow rays per marched pixel. That is a stated
    // convention rather than a physical derivation, and this case exists so
    // it is not mistaken for one.
    //
    // The cost consequence is worth writing down: the march runs ONLY when
    // the camera is submerged, so this is 16 extra shadow rays per pixel on
    // underwater frames and exactly zero on every other frame in the
    // matrix.
    CHECK(w16 * 255.0 < 1.0e-2);
}

// ===========================================================================
// 5. THE MEDIUM STACK. It replaces a bit that could not represent nesting.
// ===========================================================================

TEST_CASE("the medium stack round-trips at every depth") {
    // An empty stack is air, and that is what a camera in air starts with.
    CHECK(mediumDepth(0u) == 0u);
    CHECK(mediumTop(0u) == kMediumAir);
    CHECK_FALSE(mediumIsWater(0u));

    // Push the full alphabet and read every entry back on the way down. If
    // one push clobbered a neighbour's two bits this fails on the pop, not
    // on the push, which is the failure mode a bitfield stack actually has.
    std::vector<std::uint32_t> ids;
    std::uint32_t s = 0u;
    for (std::uint32_t i = 0; i < kMediumStackMax; ++i) {
        const std::uint32_t id = (i % 2u) ? kMediumDielectric : kMediumWater;
        ids.push_back(id);
        s = mediumPush(s, id);
        CAPTURE(i);
        CHECK(mediumDepth(s) == i + 1u);
        CHECK(mediumTop(s) == id);
    }
    for (std::uint32_t i = kMediumStackMax; i-- > 0;) {
        CAPTURE(i);
        CHECK(mediumTop(s) == ids[i]);
        s = mediumPop(s);
        CHECK(mediumDepth(s) == i);
    }
    CHECK(s == 0u);
    CHECK(mediumTop(s) == kMediumAir);
}

TEST_CASE("the medium stack's two edge cases are defined, not undefined") {
    // OVERFLOW SATURATES. A ray that has crossed eight nested interfaces
    // without leaving one is a pathological scene; the conservative answer
    // is to keep the current top and never invent a medium. Critically the
    // stack must not CORRUPT -- the entries already on it stay readable, so
    // the ray recovers as it pops back out.
    std::uint32_t s = 0u;
    for (std::uint32_t i = 0; i < kMediumStackMax; ++i) {
        s = mediumPush(s, kMediumWater);
    }
    const std::uint32_t full = s;
    for (int i = 0; i < 5; ++i) {
        s = mediumPush(s, kMediumDielectric);
        CHECK(s == full);                    // byte-identical, not merely equal top
        CHECK(mediumDepth(s) == kMediumStackMax);
        CHECK(mediumTop(s) == kMediumWater);
    }
    // ...and popping all the way out still gives exactly air.
    for (std::uint32_t i = 0; i < kMediumStackMax; ++i) s = mediumPop(s);
    CHECK(s == 0u);

    // UNDERFLOW STAYS AIR. A ray that exits a surface it never entered --
    // which happens on the very first bounce of a camera inside a body
    // whose entry was never traced -- must land in air, not in whatever
    // bits happen to be below the stack pointer.
    std::uint32_t e = 0u;
    for (int i = 0; i < 5; ++i) {
        e = mediumPop(e);
        CHECK(e == 0u);
        CHECK(mediumTop(e) == kMediumAir);
    }
    // Reserved bits stay clear so the whole stack is comparable as a uint.
    std::uint32_t r = mediumPush(mediumPush(0u, kMediumWater),
                                 kMediumDielectric);
    CHECK((r & 0xFFF00000u) == 0u);
}

TEST_CASE("above water the stack reproduces the bit it replaces") {
    // THE BIT-EXACTNESS ARGUMENT, as a test rather than as a comment. Every
    // branch that used to read `in_water == 1` now reads
    // ptMediumIsWater(stack). For a camera in air crossing any sequence of
    // GLASS interfaces the two must give the same answer at every step,
    // because that is what makes every existing non-water golden unmoved.
    std::uint32_t s = 0u;
    int legacy_bit = 0;                 // the old one-bit model
    for (int i = 0; i < 6; ++i) {       // enter, enter, exit, enter, exit, exit
        const bool entering = (i == 0 || i == 1 || i == 3);
        s = entering ? mediumPush(s, kMediumDielectric) : mediumPop(s);
        // The old model did not flip on dielectric at all.
        CAPTURE(i);
        CHECK(mediumIsWater(s) == (legacy_bit == 1));
    }

    // AND THE CASE THE BIT GOT WRONG. In water, entering glass: the old
    // model kept in_water == 1 and went on applying WATER absorption over
    // the segment inside the glass. The stack says dielectric, so it does
    // not. This is the nested-media fix, stated as the behavioural
    // difference it is.
    std::uint32_t w = mediumPush(0u, kMediumWater);
    CHECK(mediumIsWater(w));
    const std::uint32_t in_glass = mediumPush(w, kMediumDielectric);
    CHECK_FALSE(mediumIsWater(in_glass));      // the old model said true
    CHECK(mediumTop(in_glass) == kMediumDielectric);
    // ...and leaving the glass returns to water, which "set in_water = 0 on
    // exit" could never do.
    CHECK(mediumIsWater(mediumPop(in_glass)));
}

// ===========================================================================
// 6. SNELL'S WINDOW, AND THE SUN THAT IS NOT WHERE IT LOOKS.
// ===========================================================================

TEST_CASE("Snell's window is the critical-angle cone, from below") {
    const double n = 1.33;
    const double crit = std::asin(1.0 / n);
    CHECK(crit * 180.0 / kPi == doctest::Approx(48.75).epsilon(1e-3));
    // The window is the full cone, so its apex angle is twice that. #261
    // and the planet_ocean_snell fixture both quote 97.2 degrees, and the
    // fixture's 130-degree field of view is chosen to hold it with the
    // total-internal-reflection region visible around it.
    CHECK(2.0 * crit * 180.0 / kPi == doctest::Approx(97.5).epsilon(1e-3));

    // THE WHOLE SKY FITS INSIDE IT. Every direction in the upper
    // hemisphere, from the zenith to the horizon, maps into the cone and
    // the horizon maps exactly to its rim. That is the compression the
    // window IS, and it is a statement about Snell's law with no tolerance
    // beyond the trig.
    for (int i = 0; i <= 90; ++i) {
        const double theta_air = static_cast<double>(i) * kPi / 180.0;
        const double theta_w = std::asin(std::sin(theta_air) / n);
        CAPTURE(i);
        CHECK(theta_w <= crit + 1e-12);
        CHECK(theta_w < theta_air + 1e-12);      // bends toward the vertical
    }
    CHECK(std::asin(std::sin(kPi / 2.0) / n) == doctest::Approx(crit));

    // OUTSIDE THE CONE THERE IS NO SKY AT ALL -- total internal
    // reflection, which is what makes the rim an edge rather than a fade.
    // Beyond the critical angle the refracted direction has no real
    // solution, i.e. sin(theta_air) would exceed 1.
    for (double deg : {49.0, 60.0, 80.0, 89.0}) {
        const double s = n * std::sin(deg * kPi / 180.0);
        CAPTURE(deg);
        CHECK(s > 1.0);
    }
}

TEST_CASE("the underwater sun is not where the air sun is") {
    // THE BUG THIS PREVENTS, AND IT IS NOT HYPOTHETICAL. The four NEE call
    // sites this engine already had all sample the sun's AIR direction and
    // trace toward it from underwater. That is the reverse path, and it
    // does not arrive: a ray leaving the water at angle theta from the
    // vertical exits at asin(n sin theta), which is not theta. So the
    // estimator either misses the sun or -- past the critical angle --
    // total-internal-reflects and returns black for a sun that is plainly
    // visible in the window.
    const double n = 1.33;
    const double crit = std::asin(1.0 / n);

    for (double elev_deg : {5.0, 20.0, 45.0, 60.0, 89.0}) {
        const double theta_air = (90.0 - elev_deg) * kPi / 180.0;
        // The direction a submerged point must look. Snell, entering the
        // denser medium: no total internal reflection is possible.
        const double theta_w = std::asin(std::sin(theta_air) / n);
        CAPTURE(elev_deg);
        // It is always inside the window...
        CHECK(theta_w < crit);
        // ...and always closer to the zenith than the air direction, by an
        // amount a renderer cannot ignore.
        CHECK(theta_w < theta_air);
        // Tracing toward the AIR direction from below exits at this angle
        // instead, which is not the sun's:
        const double sin_out = n * std::sin(theta_air);
        if (sin_out <= 1.0) {
            const double theta_out = std::asin(sin_out);
            CAPTURE(theta_out * 180.0 / kPi);
            CHECK(std::fabs(theta_out - theta_air) > 1e-3);
        } else {
            // ...or does not exit at all. At a 20-degree sun the air
            // direction is 70 degrees from the vertical, well past the
            // 48.75-degree critical angle, so the naive estimator returns
            // exactly zero for a sun that is bright and overhead-ish in
            // the window. This is the case that makes the refraction
            // mandatory rather than a refinement.
            CHECK(elev_deg <= 45.0);
        }
    }

    // THE WORST ERROR IS LARGE. Over the whole sky the two directions
    // differ by up to 41 degrees, which at the sun's 0.53-degree angular
    // diameter is 78 sun-widths -- there is no sense in which sampling the
    // wrong one is an approximation.
    double worst = 0.0;
    for (int i = 0; i <= 90; ++i) {
        const double ta = static_cast<double>(i) * kPi / 180.0;
        const double tw = std::asin(std::sin(ta) / n);
        worst = std::max(worst, ta - tw);
    }
    CAPTURE(worst * 180.0 / kPi);
    CHECK(worst * 180.0 / kPi > 40.0);
    CHECK(worst * 180.0 / kPi / 0.53 > 75.0);

    // AND THE MIRROR OF WHAT THE SHADER ACTUALLY WRITES. It computes
    // refract(-sun_dir, up, 1/n) and negates. For a sun at elevation e in
    // the plane, sun_dir = (cos e, sin e, 0) and up = (0, 1, 0); the
    // refracted direction's angle from -up must be theta_w.
    for (double elev_deg : {10.0, 30.0, 60.0}) {
        const double e = elev_deg * kPi / 180.0;
        const glm::dvec3 sun_dir(std::cos(e), std::sin(e), 0.0);
        const glm::dvec3 up(0.0, 1.0, 0.0);
        const glm::dvec3 inc = -sun_dir;
        const double eta = 1.0 / n;
        const double cosi = -glm::dot(up, inc);
        const double k = 1.0 - eta * eta * (1.0 - cosi * cosi);
        REQUIRE(k > 0.0);                       // entering: never TIR
        const glm::dvec3 into =
            eta * inc + (eta * cosi - std::sqrt(k)) * up;
        const glm::dvec3 look = -glm::normalize(into);
        const double theta_w =
            std::acos(std::clamp(glm::dot(look, up), -1.0, 1.0));
        const double expect =
            std::asin(std::sin((90.0 - elev_deg) * kPi / 180.0) / n);
        CAPTURE(elev_deg);
        CHECK(theta_w == doctest::Approx(expect).epsilon(1e-9));
    }
}

// ===========================================================================
// 7. THE MIRRORS ARE STILL FAITHFUL.
// ===========================================================================

TEST_CASE("shader and engine mirrors are still faithful") {
    // Every transcription above is only as good as the code it copies, so
    // re-read the sources and pin what the copies depend on. Whitespace is
    // stripped so reformatting does not break the pins, and occurrences are
    // COUNTED rather than merely found: `find() != npos` is satisfied by
    // one correct copy however many wrong ones exist elsewhere, which is
    // how #276 stayed live for a cycle under a passing test.
    const std::string pt = stripSpace(readAll(PT_SHADER_PATHTRACE_PATH));
    const std::string pm = stripSpace(readAll(PT_SHADER_PATHTRACEMATH_PATH));
    const std::string ec = stripSpace(readAll(PT_ENGINE_CPP_PATH));

    // --- the medium stack, verbatim -----------------------------------
    CHECK(countOf(pt, "uintptMediumDepth(uints){return(s>>16)&0xFu;}") == 1u);
    CHECK(countOf(pt, "return(d==0u)?MEDIUM_AIR:((s>>((d-1u)*2u))&3u);") == 1u);
    CHECK(countOf(pt, "if(d>=kMediumStackMax)returns;") == 1u);
    CHECK(countOf(pt, "uintentries=(s&0xFFFFu)&~(3u<<(d*2u));") == 1u);
    CHECK(countOf(pt, "entries|=(id&3u)<<(d*2u);") == 1u);
    CHECK(countOf(pt, "returnentries|((d+1u)<<16);") == 1u);
    CHECK(countOf(pt, "if(d==0u)returns;") == 1u);
    // Pop CLEARS the entry, so the encoding is canonical: one bit pattern
    // per (depth, contents). The non-clearing form must not survive.
    CHECK(countOf(pt, "uintentries=(s&0xFFFFu)&~(3u<<((d-1u)*2u));") == 1u);
    CHECK(countOf(pt, "returnentries|((d-1u)<<16);") == 1u);
    CHECK(countOf(pt, "return(s&0xFFFFu)|((d-1u)<<16);") == 0u);
    CHECK(countOf(pt, "staticconstuintkMediumStackMax=8u;") == 1u);
    CHECK(countOf(pt, "staticconstuintMEDIUM_WATER=1u;") == 1u);
    CHECK(countOf(pt, "staticconstuintMEDIUM_DIELECTRIC=2u;") == 1u);

    // THE BIT IS GONE. Not "a stack exists somewhere" -- the one-bit model
    // must have no surviving copies, or the two would drift and half the
    // branches would read the stale one.
    CHECK(countOf(pt, "intin_water=") == 0u);
    CHECK(countOf(pt, "in_water=1;") == 0u);
    CHECK(countOf(pt, "in_water=0;") == 0u);
    CHECK(countOf(pt, "sr_in_water=1-sr_in_water;") == 0u);
    // ...and every NEE site now forwards the stack. Six call sites carried
    // the bit; all six must carry the stack.
    // Six NEE sites carried the bit, and the ambient gate below adds a
    // seventh. The "medium);" form counts six of those seven; the odd one
    // out is the Lambert sun NEE, whose call is the middle of a ternary and
    // so has no semicolon after it. Both are pinned, because asserting only
    // the total would be satisfied by seven copies of the wrong one.
    CHECK(countOf(pt, "cone_hit,medium)") == 7u);
    CHECK(countOf(pt, "cone_hit,medium);") == 6u);

    // --- the phase functions ------------------------------------------
    CHECK(countOf(pm, "floatk=(1.0-d)/(1.0+d);") == 1u);
    CHECK(countOf(pm,
                  "return(1.0+k*mu*mu)/(4.0*3.14159265358979*(1.0+k/3.0));")
          == 1u);
    CHECK(countOf(pm, "publicfloatptPhaseHenyeyGreenstein(floatmu,floatg)")
          == 1u);
    CHECK(countOf(pm, "publicfloatptPhaseRayleighDepol(floatmu,floatd)") == 1u);

    // --- the march ------------------------------------------------------
    // The transmittance is the CLOSED FORM at the sample, not the air
    // march's running product. This is the line the second-order
    // convergence case above rests on, and reverting it silently halves
    // the order.
    CHECK(countOf(pt, "float3trans_eye_w=exp(-sigma_t*t);") == 1u);
    CHECK(countOf(pt, "trans_eye_w*=exp(-sigma_t*dt_w);") == 0u);
    // ln(255), the same one-8-bit-level bound pt::water::OpticalBlackDepth
    // uses. If these two ever disagree the march span stops meaning what
    // the test says it means.
    CHECK(countOf(pt, "floatt_black=5.5412635451584/sigma_t_lo;") == 1u);
    // Two species, two phase functions -- not one HG lobe for both.
    CHECK(countOf(pt, "sigma_s_mol*ptPhaseRayleighDepol(mu_w,depol)") == 1u);
    CHECK(countOf(pt, "float3(sigma_s_par)*ptPhaseHenyeyGreenstein(mu_w,g_par)")
          == 1u);
    // The sun is refracted into the water before it is sampled.
    CHECK(countOf(pt, "float3into=refract(-sun_dir_w,up_w,eta_in);") == 1u);
    CHECK(countOf(pt, "float3sun_w=normalize(-into);") == 1u);
    CHECK(countOf(pt, "float3sun_jit=sampleSunDisc(sun_w,seed);") == 1u);
    // Extinction is absorption PLUS both scattering species.
    CHECK(countOf(pt,
                  "float3sigma_t=max(water_params0.xyz,float3(0.0))+sigma_s;")
          == 1u);

    // --- the ambient gate ------------------------------------------------
    // skyColorBase must be attenuated when the stack says water, and the
    // ungated form must not survive anywhere.
    CHECK(countOf(pt, "float3ambient=skyColorBase(hit_pt,nf,seed);") == 1u);
    CHECK(countOf(pt, "radiance+=throughput*h.albedo*ambient;") == 1u);
    CHECK(countOf(pt,
                  "radiance+=throughput*h.albedo*skyColorBase(hit_pt,nf,seed);")
          == 0u);
    CHECK(countOf(pt, "if(ptMediumIsWater(medium)){ambient*=transmittance(")
          == 1u);

    // --- the host side ---------------------------------------------------
    // The coefficients come from the module, not from literals at the push
    // site. Three literals here would be exactly the fabricated-Hosek
    // decision made again.
    CHECK(countOf(ec, "constglm::dvec3b_mol=pt::water::PureSeawaterScatteringRgb();")
          == 1u);
    CHECK(countOf(ec, "push.water_scatter2[0]=static_cast<float>(pt::water::kPetzoldAsymmetry);")
          == 1u);
    CHECK(countOf(ec, "push.water_scatter2[3]=static_cast<float>(pt::water::kSeawaterDepolarisation);")
          == 1u);
    // The Pope & Fry defaults reached the cvars.
    CHECK(countOf(ec, "PT_CVAR(r_water_absorption_r,\"0.2755\",") == 1u);
    CHECK(countOf(ec, "PT_CVAR(r_water_absorption_g,\"0.0565\",") == 1u);
    CHECK(countOf(ec, "PT_CVAR(r_water_absorption_b,\"0.00922\",") == 1u);
    CHECK(countOf(ec, "PT_CVAR(r_water_absorption_r,\"0.45\",") == 0u);
    // The Vulkan spilled-tail budget guard accounts for the two new lanes.
    CHECK(countOf(ec, "+32/*PlanetaryP7(#261):water_scatter+") == 1u);
    CHECK(countOf(ec, "static_assert(sizeof(PtPush)-112<=2048,") == 1u);
    // Both new push lanes are 16-byte aligned, asserted at compile time.
    CHECK(countOf(ec, "static_assert(offsetof(PtPush,water_scatter)%16==0,") == 1u);
    CHECK(countOf(ec, "static_assert(offsetof(PtPush,water_scatter2)%16==0,") == 1u);

    // --- the Pope & Fry table is where the header says it is -------------
    const std::string wo = stripSpace(readAll(PT_WATER_OPTICS_CPP_PATH));
    CHECK(countOf(wo, "0.00922,") == 1u);
    CHECK(countOf(wo, "0.05650,") == 1u);
    CHECK(countOf(wo, "0.27550,") == 1u);
    CHECK(countOf(wo, "0.62400,") == 1u);
    CHECK(countOf(wo, "constexprdoublekMorelBeta90At500=1.38e-4;") == 1u);
    CHECK(countOf(wo, "constexprdoublekMorelExponent=-4.32;") == 1u);
}

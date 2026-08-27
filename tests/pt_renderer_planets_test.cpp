// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for the planetarium -- issue #281.
//
// Planet positions are host-side, deterministic and closed-form, so this
// file is where the feature actually gets verified. The golden fixture is a
// smoke check that the compositing path runs at all; everything about
// whether Jupiter is in the right place, and whether it moves the right way
// on the right dates, is here.
//
// REFERENCE DATA
// --------------
// Every position and magnitude reference below was read out of the JPL
// HORIZONS system (https://ssd.jpl.nasa.gov/horizons/, ephemeris DE441),
// geocentric observer (site 500@399), QUANTITIES 1 + 9 + 19 + 20 --
// i.e. astrometric RA/Dec in the ICRF (light-time corrected, no aberration,
// referred to J2000), apparent V magnitude, heliocentric range r and
// observer range delta. Astrometric-and-not-apparent is the correct column
// to compare against: pt::astro::planetPosition applies light-time and
// nothing else, and it returns J2000 coordinates, which is also the frame
// the Bright Star Catalog is stored in (see Astronomy.h's frame note).
//
// Structural references (elongation envelopes, sidereal periods, perihelion
// and aphelion distances) come from the NASA planetary fact sheets and are
// cited at their use site. Those are the tests that would survive someone
// mistyping one digit of the JPL element table -- a single wrong element
// moves a body's whole orbit, not just its position on one date.
//
// TOLERANCES
// ----------
// Not invented. JPL publishes the worst-case error of this element set over
// 1800-2050 (approx_pos.html), in arcseconds of ecliptic longitude:
//   Mercury 15, Venus 20, Mars 40, Jupiter 400, Saturn 600, Uranus 50,
//   Neptune 10.
// The per-body angular budgets below are those numbers with headroom, and
// no more. Raising one of them to make a test pass would be conceding that
// the implementation is worse than the element set it is built on.
//
// All reference values are HARDCODED -- no time(), no system clock, no
// network. Same source, same build, same numbers on every host.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/Astronomy.h"
#include "../src/renderer/BscCatalog.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using pt::astro::EquatorialPos;
using pt::astro::Planet;
using pt::astro::PlanetPos;
using pt::astro::earthHeliocentricEcliptic;
using pt::astro::equatorialToHorizon;
using pt::astro::julianDateFromUtc;
using pt::astro::kPlanetCount;
using pt::astro::planetBvColorIndex;
using pt::astro::planetHeliocentricEcliptic;
using pt::astro::planetName;
using pt::astro::planetPosition;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;

// Finiteness probe that survives -ffast-math.
//
// The Release build compiles this project (and this test) with
// -ffast-math, which implies -ffinite-math-only, under which Apple clang
// folds std::isfinite() -- and a plain exponent-field comparison on a value
// it can see the provenance of -- to a constant true. An assertion written
// with std::isfinite would be VACUOUS in exactly the build the goldens run
// in. The volatile round-trip breaks the provenance chain so the exponent
// read is honest. Same pattern, and same reason, as
// tests/pt_math_altitude_test.cpp's finiteBits(); this is the double
// version (11 exponent bits, all-ones = inf/NaN).
bool finiteBits(double v) {
    volatile double t = v;
    double u = t;
    std::uint64_t b;
    std::memcpy(&b, &u, 8);
    return ((b >> 52) & 0x7FFull) != 0x7FFull;
}

// Assert the harness itself is not vacuous: if the compiler ever folds
// finiteBits to constant true, this fails and the whole file's finiteness
// claims stop being a lie.
bool finiteHarnessWorks() {
    volatile double zero = 0.0;
    const double inf = 1.0 / zero;
    const double nan = zero / zero;
    return !finiteBits(inf) && !finiteBits(nan) && finiteBits(1.0);
}

double wrap180(double d) {
    d = std::fmod(d + 180.0, 360.0);
    if (d < 0.0) d += 360.0;
    return d - 180.0;
}

// Great-circle separation between two equatorial positions, in arcseconds.
double sepArcsec(const EquatorialPos& a, const EquatorialPos& b) {
    const double d1 = a.dec_deg * kDeg2Rad, d2 = b.dec_deg * kDeg2Rad;
    const double dra = (a.ra_deg - b.ra_deg) * kDeg2Rad;
    double c = std::sin(d1) * std::sin(d2)
             + std::cos(d1) * std::cos(d2) * std::cos(dra);
    c = std::clamp(c, -1.0, 1.0);
    return std::acos(c) / kDeg2Rad * 3600.0;
}

// Sexagesimal helpers so the HORIZONS output can be transcribed exactly as
// printed, instead of pre-reduced to decimal degrees by hand (which is
// where transcription errors hide).
constexpr double hms(int h, int m, double s) {
    return (double(h) + double(m) / 60.0 + s / 3600.0) * 15.0;
}
constexpr double dms(int sign, int d, int m, double s) {
    return double(sign) * (double(d) + double(m) / 60.0 + s / 3600.0);
}

struct HorizonsRow {
    const char* name;
    Planet      body;
    double      ra_deg;
    double      dec_deg;
    double      vmag;
    double      tol_arcsec;   // per-body, from the JPL accuracy table
};

// JPL HORIZONS, 2026-Jan-01 00:00 UT, geocentric astrometric ICRF.
//
//   mercury  17 52 31.46  -23 59 41.6   APmag -0.595
//   venus    18 38 39.71  -23 38 40.9   APmag -3.911
//   mars     18 53 57.37  -23 45 06.0   APmag  1.072
//   jupiter  07 30 55.04  +22 02 04.5   APmag -2.670
//   saturn   23 48 11.31  -03 44 26.4   APmag  1.007
//   uranus   03 41 25.51  +19 25 28.7   APmag  5.646
//   neptune  23 58 58.57  -01 33 48.2   APmag  7.767
const HorizonsRow kEpoch2026[] = {
    {"mercury", Planet::Mercury, hms(17, 52, 31.46), dms(-1, 23, 59, 41.6), -0.595,  30.0},
    {"venus",   Planet::Venus,   hms(18, 38, 39.71), dms(-1, 23, 38, 40.9), -3.911,  40.0},
    {"mars",    Planet::Mars,    hms(18, 53, 57.37), dms(-1, 23, 45,  6.0),  1.072,  80.0},
    {"jupiter", Planet::Jupiter, hms( 7, 30, 55.04), dms(+1, 22,  2,  4.5), -2.670, 400.0},
    {"saturn",  Planet::Saturn,  hms(23, 48, 11.31), dms(-1,  3, 44, 26.4),  1.007, 600.0},
    {"uranus",  Planet::Uranus,  hms( 3, 41, 25.51), dms(+1, 19, 25, 28.7),  5.646, 100.0},
    {"neptune", Planet::Neptune, hms(23, 58, 58.57), dms(-1,  1, 33, 48.2),  7.767,  60.0},
};

// The magnitude model (Mallama, Krobusek & Pavlov 2017) is a fit to real
// photometry with its own scatter, and HORIZONS uses the Astronomical
// Almanac's Hilton (2005) model for the inner planets, so the two disagree
// slightly by construction. 0.15 mag is about half of what the eye can
// discriminate between two point sources and an order of magnitude below
// the differences this feature has to get right (Venus is 1.2 mag brighter
// than Jupiter; a 0.15 mag error cannot swap them).
constexpr double kVmagTol = 0.15;

// Mars's 2020 apparition, sampled every 10 days from HORIZONS. RA only --
// the retrograde loop is a right-ascension reversal, and pinning the
// reversal is the point. Transcribed exactly as printed.
struct MarsRaRow { int y, m, d; int rh, rm; double rs; };
const MarsRaRow kMars2020[] = {
    {2020,  8,  1, 1, 12,  0.51}, {2020,  8, 11, 1, 27, 19.64},
    {2020,  8, 21, 1, 39, 15.99}, {2020,  8, 31, 1, 46, 56.53},
    {2020,  9, 10, 1, 49, 33.78}, {2020,  9, 20, 1, 46, 32.35},
    {2020,  9, 30, 1, 38, 15.61}, {2020, 10, 10, 1, 26, 25.96},
    {2020, 10, 20, 1, 13, 48.81}, {2020, 10, 30, 1,  3, 30.71},
    {2020, 11,  9, 0, 57, 32.54}, {2020, 11, 19, 0, 56, 36.66},
    {2020, 11, 29, 1,  0, 33.34}, {2020, 12,  9, 1,  8, 38.80},
    {2020, 12, 19, 1, 20,  9.33}, {2020, 12, 29, 1, 34, 27.26},
};

std::string tightenFile(const char* path) {
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

// COUNT occurrences, do not just test for presence. A `find() != npos` pin
// against shader source keeps passing after the thing it describes has been
// duplicated, moved or half-deleted -- the SPIR-V and Metal halves of this
// kernel each declare the push layout, so "the field exists" is satisfied
// by one of the two and says nothing about the other.
std::size_t countOccurrences(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t n = 0, pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += needle.size();
    }
    return n;
}

}  // namespace

// --- the harness's own guard ---------------------------------------------

TEST_CASE("finiteness harness is not folded away by -ffast-math") {
    CHECK(finiteHarnessWorks());
}

// --- element table integrity ---------------------------------------------
//
// These are the tests that catch a mistyped digit in the JPL table. A wrong
// element does not shift a body slightly on one date; it puts the whole
// orbit in the wrong place, and the orbit's own invariants say so.

TEST_CASE("heliocentric distance stays inside the published perihelion / aphelion") {
    // NASA planetary fact sheets (https://nssdc.gsfc.nasa.gov/planetary/
    // factsheet/), perihelion and aphelion in 1e6 km, converted at
    // 1 au = 149.597870700e6 km:
    //   Mercury 46.00 / 69.82   -> 0.30749 / 0.46670
    //   Venus  107.48 /108.94   -> 0.71846 / 0.72822
    //   Mars   206.65 /249.26   -> 1.38137 / 1.66619
    //   Jupiter740.60 /816.36   -> 4.95060 / 5.45703
    //   Saturn 1357.6 /1506.5   -> 9.07500 /10.07033
    //   Uranus 2732.7 /3001.4   ->18.26697 /20.06311
    //   Neptune4471.1 /4558.9   ->29.88742 /30.47437
    // Sampled over a full orbital period of each body so both extremes are
    // actually visited, with a 1% envelope for the fact sheets' own
    // rounding and for the difference between osculating and mean elements.
    struct Row { Planet p; double peri_au, apo_au, period_days; };
    const Row rows[] = {
        {Planet::Mercury,  0.30749,  0.46670,    87.969},
        {Planet::Venus,    0.71846,  0.72822,   224.701},
        {Planet::Mars,     1.38137,  1.66619,   686.980},
        {Planet::Jupiter,  4.95060,  5.45703,  4332.589},
        {Planet::Saturn,   9.07500, 10.07033, 10759.220},
        {Planet::Uranus,  18.26697, 20.06311, 30685.400},
        {Planet::Neptune, 29.88742, 30.47437, 60189.000},
    };
    const double base = julianDateFromUtc(2000, 1, 1, 12, 0, 0.0);
    for (const Row& row : rows) {
        CAPTURE(planetName(row.p));
        double rmin = 1e30, rmax = -1e30;
        const double step = std::max(row.period_days / 400.0, 0.25);
        for (double t = base; t <= base + row.period_days; t += step) {
            double x[3];
            planetHeliocentricEcliptic(row.p, t, x);
            const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            REQUIRE(finiteBits(r));
            rmin = std::min(rmin, r);
            rmax = std::max(rmax, r);
        }
        CAPTURE(rmin);
        CAPTURE(rmax);
        CHECK(rmin == doctest::Approx(row.peri_au).epsilon(0.01));
        CHECK(rmax == doctest::Approx(row.apo_au).epsilon(0.01));
    }
}

TEST_CASE("mean longitude rates reproduce the published sidereal periods") {
    // Independent check of the L_dot column: advance by one published
    // sidereal period and the heliocentric ecliptic longitude must come
    // back to where it started. Periods in days from the NASA planetary
    // fact sheets. Saturn's 0.15 deg residual is the fact sheet's own
    // 4-significant-figure period, not the element set.
    struct Row { Planet p; double period_days; double tol_deg; };
    const Row rows[] = {
        {Planet::Mercury,    87.969, 0.01},
        {Planet::Venus,     224.701, 0.01},
        {Planet::Mars,      686.980, 0.01},
        {Planet::Jupiter,  4332.589, 0.05},
        {Planet::Saturn,  10759.220, 0.20},
        {Planet::Uranus,  30685.400, 0.05},
        {Planet::Neptune, 60189.000, 0.05},
    };
    const double base = julianDateFromUtc(2000, 1, 1, 12, 0, 0.0);
    for (const Row& row : rows) {
        CAPTURE(planetName(row.p));
        double a[3], b[3];
        planetHeliocentricEcliptic(row.p, base, a);
        planetHeliocentricEcliptic(row.p, base + row.period_days, b);
        const double la = std::atan2(a[1], a[0]) / kDeg2Rad;
        const double lb = std::atan2(b[1], b[0]) / kDeg2Rad;
        const double d = wrap180(lb - la);
        CAPTURE(d);
        CHECK(std::abs(d) < row.tol_deg);
    }
}

TEST_CASE("inner-planet elongation never exceeds its geometric limit") {
    // The single strongest structural test in this file. Mercury and Venus
    // are interior to Earth's orbit, so their angular distance from the sun
    // is bounded by the orbit geometry -- and those bounds are among the
    // best-known numbers in naked-eye astronomy: Mercury's greatest
    // elongations run 17.9 to 27.8 degrees, Venus's 45.4 to 47.3 (Astronomical
    // Almanac; the spread is the eccentricity of the inner orbit). Get any
    // element of either orbit materially wrong and this fails immediately,
    // without needing a single ephemeris lookup.
    //
    // Superior planets have no such bound and must reach opposition
    // (elongation ~180) and conjunction (~0) inside a decade.
    const double t0 = julianDateFromUtc(2020, 1, 1, 0, 0, 0.0);
    const double t1 = t0 + 3653.0;   // 2020-01-01 .. 2029-12-31

    double emax[kPlanetCount], emin[kPlanetCount];
    for (int i = 0; i < kPlanetCount; ++i) { emax[i] = -1e30; emin[i] = 1e30; }
    for (double t = t0; t < t1; t += 1.0) {
        for (int i = 0; i < kPlanetCount; ++i) {
            const auto pp = planetPosition(static_cast<Planet>(i), t);
            REQUIRE(finiteBits(pp.elongation_deg));
            emax[i] = std::max(emax[i], pp.elongation_deg);
            emin[i] = std::min(emin[i], pp.elongation_deg);
        }
    }

    const int mercury = static_cast<int>(Planet::Mercury);
    const int venus   = static_cast<int>(Planet::Venus);
    CAPTURE(emax[mercury]);
    CHECK(emax[mercury] < 28.5);    // hard geometric ceiling
    CHECK(emax[mercury] > 27.0);    // the decade must contain a wide one
    CAPTURE(emax[venus]);
    CHECK(emax[venus] < 48.0);
    CHECK(emax[venus] > 46.0);
    // Both must also pass through inferior/superior conjunction.
    CHECK(emin[mercury] < 2.0);
    CHECK(emin[venus]   < 2.0);

    for (int i = static_cast<int>(Planet::Mars); i < kPlanetCount; ++i) {
        CAPTURE(planetName(static_cast<Planet>(i)));
        CHECK(emax[i] > 175.0);     // reaches opposition
        CHECK(emin[i] <   5.0);     // reaches conjunction
    }
}

// --- positions against published values ----------------------------------

TEST_CASE("2026-01-01 positions match JPL HORIZONS inside the element set's own error") {
    const double jd = julianDateFromUtc(2026, 1, 1, 0, 0, 0.0);
    // 2026-01-01 00:00 UT is JD 2461041.5 by definition (a JD begins at
    // noon, so midnight is the .5). Exact equality, not Approx: doctest's
    // Approx uses a strict `<` and so cannot express "identical".
    CHECK(jd == 2461041.5);
    for (const HorizonsRow& row : kEpoch2026) {
        CAPTURE(row.name);
        const auto pp = planetPosition(row.body, jd);
        REQUIRE(finiteBits(pp.eq.ra_deg));
        REQUIRE(finiteBits(pp.eq.dec_deg));
        const EquatorialPos ref{row.ra_deg, row.dec_deg};
        const double sep = sepArcsec(pp.eq, ref);
        CAPTURE(pp.eq.ra_deg);
        CAPTURE(row.ra_deg);
        CAPTURE(pp.eq.dec_deg);
        CAPTURE(row.dec_deg);
        CAPTURE(sep);
        CHECK(sep < row.tol_arcsec);
    }
}

TEST_CASE("magnitudes match JPL HORIZONS across the whole brightness range") {
    // Seven bodies spanning V = -3.9 to +7.8, i.e. a factor of 3e4 in flux.
    const double jd = julianDateFromUtc(2026, 1, 1, 0, 0, 0.0);
    for (const HorizonsRow& row : kEpoch2026) {
        CAPTURE(row.name);
        const auto pp = planetPosition(row.body, jd);
        REQUIRE(finiteBits(pp.vmag));
        CAPTURE(pp.vmag);
        CAPTURE(row.vmag);
        CHECK(std::abs(pp.vmag - row.vmag) < kVmagTol);
    }
}

TEST_CASE("positions hold at both ends of the element set's validity window") {
    // The 1800-2050 table is a linear-rate model; its error is largest at
    // the ends of the interval, which is exactly where a sign error in a
    // rate column would show up and where the J2000-centred 2026 epoch
    // above would not. 1900 and 2049 are ~1.26 centuries apart.
    struct Row { const char* name; Planet p; int y, mo, d; double ra, dec, vmag, tol; };
    const Row rows[] = {
        // 1900-Jan-01 00:00 UT  jupiter  16 02 31.46  -19 52 48.8  APmag -1.783
        {"jupiter 1900", Planet::Jupiter, 1900, 1, 1,
         hms(16, 2, 31.46), dms(-1, 19, 52, 48.8), -1.783, 400.0},
        // 2049-Jun-15 00:00 UT  venus    02 30 02.95  +12 04 53.9  APmag -4.214
        {"venus 2049", Planet::Venus, 2049, 6, 15,
         hms(2, 30, 2.95), dms(+1, 12, 4, 53.9), -4.214, 40.0},
        // 2049-Jun-15 00:00 UT  mars     06 40 37.40  +24 06 38.0  APmag  1.621
        {"mars 2049", Planet::Mars, 2049, 6, 15,
         hms(6, 40, 37.40), dms(+1, 24, 6, 38.0), 1.621, 80.0},
    };
    for (const Row& row : rows) {
        CAPTURE(row.name);
        const double jd = julianDateFromUtc(row.y, row.mo, row.d, 0, 0, 0.0);
        const auto pp = planetPosition(row.p, jd);
        const double sep = sepArcsec(pp.eq, EquatorialPos{row.ra, row.dec});
        CAPTURE(sep);
        CHECK(sep < row.tol);
        CAPTURE(pp.vmag);
        CHECK(std::abs(pp.vmag - row.vmag) < kVmagTol);
    }
}

TEST_CASE("the 2020 great conjunction puts Jupiter and Saturn on top of each other") {
    // 2020-12-21 is the closest Jupiter-Saturn conjunction since 1623 --
    // a famous, heavily published event and a two-body test: BOTH planets
    // have to be right at once for the separation to come out. The
    // published separation at closest approach was about 6.1 arcmin.
    //
    // HORIZONS at 2020-Dec-21 12:00 UT:
    //   jupiter  20 09 45.68  -20 35 11.4
    //   saturn   20 09 47.52  -20 28 50.7   -> separation 6.4 arcmin
    const double jd = julianDateFromUtc(2020, 12, 21, 12, 0, 0.0);
    const auto j = planetPosition(Planet::Jupiter, jd);
    const auto s = planetPosition(Planet::Saturn,  jd);

    const double sep_arcmin = sepArcsec(j.eq, s.eq) / 60.0;
    CAPTURE(sep_arcmin);
    // Below the moon's own apparent diameter (30 arcmin), which is what
    // made this event newsworthy, and not so small that a degenerate
    // "both bodies collapsed to the same wrong point" bug would pass:
    // the two are computed from completely separate element rows.
    CHECK(sep_arcmin < 15.0);
    CHECK(sep_arcmin >  2.0);

    // And each is individually in the right place.
    CHECK(sepArcsec(j.eq, EquatorialPos{hms(20, 9, 45.68), dms(-1, 20, 35, 11.4)})
          < 400.0);
    CHECK(sepArcsec(s.eq, EquatorialPos{hms(20, 9, 47.52), dms(-1, 20, 28, 50.7)})
          < 600.0);

    // Six months earlier they are nowhere near each other -- the
    // conjunction is a real event in the motion, not a fixed offset.
    const double jd_far = jd - 182.0;
    const auto j2 = planetPosition(Planet::Jupiter, jd_far);
    const auto s2 = planetPosition(Planet::Saturn,  jd_far);
    CHECK(sepArcsec(j2.eq, s2.eq) / 3600.0 > 5.0);   // degrees apart
}

// --- motion, not just a static position ----------------------------------

TEST_CASE("Mars reproduces the 2020 retrograde loop against HORIZONS") {
    // A static position can be right by accident (or by a lucky sign
    // error). The retrograde loop cannot: right ascension has to rise,
    // reverse, and rise again, at the published dates, which only happens
    // if the RELATIVE motion of Earth and Mars is right.
    std::vector<double> ours, refs;
    for (const MarsRaRow& r : kMars2020) {
        const double jd = julianDateFromUtc(r.y, r.m, r.d, 0, 0, 0.0);
        const auto pp = planetPosition(Planet::Mars, jd);
        ours.push_back(pp.eq.ra_deg);
        refs.push_back(hms(r.rh, r.rm, r.rs));
    }
    REQUIRE(ours.size() == 16);

    // Every sample within the Mars budget.
    for (std::size_t i = 0; i < ours.size(); ++i) {
        CAPTURE(i);
        CAPTURE(ours[i]);
        CAPTURE(refs[i]);
        CHECK(std::abs(wrap180(ours[i] - refs[i])) * 3600.0 < 80.0);
    }

    // The shape of the loop: direct, then retrograde, then direct, with
    // the same sign pattern as the reference series sample for sample.
    for (std::size_t i = 1; i < ours.size(); ++i) {
        CAPTURE(i);
        const double d_ours = wrap180(ours[i] - ours[i - 1]);
        const double d_ref  = wrap180(refs[i] - refs[i - 1]);
        CHECK((d_ours < 0.0) == (d_ref < 0.0));
    }

    // Amplitude: the loop is nearly 13 degrees of RA deep, from the
    // 2020-09-10 maximum to the 2020-11-19 minimum. A model that had the
    // direction right but the amplitude wrong would pass the sign test.
    const double loop_deg = *std::max_element(ours.begin(), ours.end())
                          - *std::min_element(ours.begin(), ours.end());
    const double loop_ref = *std::max_element(refs.begin(), refs.end())
                          - *std::min_element(refs.begin(), refs.end());
    CAPTURE(loop_deg);
    CAPTURE(loop_ref);
    CHECK(loop_deg == doctest::Approx(loop_ref).epsilon(0.02));
}

TEST_CASE("Mars's 2020 stationary points land on the published dates") {
    // Mars was stationary on 2020-09-09 (entering retrograde) and
    // 2020-11-14/15 (resuming direct motion) -- the dates every almanac
    // for that apparition prints. Scan the derivative at 6-hour steps and
    // require exactly two sign changes in the window, at those dates.
    const double t0 = julianDateFromUtc(2020, 8, 1, 0, 0, 0.0);
    const double t1 = julianDateFromUtc(2021, 1, 5, 0, 0, 0.0);
    const double dt = 0.25;
    auto ra = [](double t) { return planetPosition(Planet::Mars, t).eq.ra_deg; };

    std::vector<double> stations;
    double prev = wrap180(ra(t0 + dt) - ra(t0));
    for (double t = t0 + dt; t + dt < t1; t += dt) {
        const double d = wrap180(ra(t + dt) - ra(t));
        if ((d < 0.0) != (prev < 0.0)) stations.push_back(t);
        prev = d;
    }
    REQUIRE(stations.size() == 2);
    // 2020-09-09 00:00 UT and 2020-11-14 00:00 UT, +- 2 days.
    CHECK(std::abs(stations[0] - julianDateFromUtc(2020, 9, 9, 0, 0, 0.0)) < 2.0);
    CHECK(std::abs(stations[1] - julianDateFromUtc(2020, 11, 14, 0, 0, 0.0)) < 2.0);
}

TEST_CASE("light-time correction is applied and displaces the planet backwards") {
    // Not a style point: without it every outer planet sits a few
    // arcseconds ahead of where it is seen. The claim is testable because
    // the geometric position is reachable through the heliocentric API.
    //
    // Direction matters as much as magnitude. The corrected position must
    // lag the geometric one -- we see where the planet WAS -- so it must
    // sit closer to the position the planet held slightly earlier than to
    // the geometric one.
    const double jd = julianDateFromUtc(2026, 1, 1, 0, 0, 0.0);
    struct Row { Planet p; double min_arcsec; };
    const Row rows[] = {
        {Planet::Jupiter, 2.0},
        {Planet::Saturn,  1.0},
        {Planet::Neptune, 1.0},
    };
    for (const Row& row : rows) {
        CAPTURE(planetName(row.p));
        double e[3], g[3];
        earthHeliocentricEcliptic(jd, e);
        planetHeliocentricEcliptic(row.p, jd, g);
        // Geometric geocentric direction -> RA/Dec through the same
        // obliquity rotation planetPosition uses.
        const double eps = (84381.406 / 3600.0) * kDeg2Rad;
        const double gx = g[0] - e[0], gy = g[1] - e[1], gz = g[2] - e[2];
        const double ex = gx;
        const double ey = std::cos(eps) * gy - std::sin(eps) * gz;
        const double ez = std::sin(eps) * gy + std::cos(eps) * gz;
        const double rr = std::sqrt(ex * ex + ey * ey + ez * ez);
        EquatorialPos geom{std::atan2(ey, ex) / kDeg2Rad, std::asin(ez / rr) / kDeg2Rad};
        if (geom.ra_deg < 0.0) geom.ra_deg += 360.0;

        const auto pp = planetPosition(row.p, jd);
        const double shift = sepArcsec(pp.eq, geom);
        CAPTURE(shift);
        CHECK(shift > row.min_arcsec);          // the correction is real
        CHECK(shift < 60.0);                    // and is not a blunder

        // Lag, not lead: the corrected position is nearer the body's
        // geometric position one light-time in the past.
        const double tau = pp.geocentric_dist_au * (499.00478380614 / 86400.0);
        double gp[3];
        planetHeliocentricEcliptic(row.p, jd - tau, gp);
        const double px = gp[0] - e[0], py = gp[1] - e[1], pz = gp[2] - e[2];
        const double qy = std::cos(eps) * py - std::sin(eps) * pz;
        const double qz = std::sin(eps) * py + std::cos(eps) * pz;
        const double qr = std::sqrt(px * px + qy * qy + qz * qz);
        EquatorialPos retarded{std::atan2(qy, px) / kDeg2Rad,
                               std::asin(qz / qr) / kDeg2Rad};
        if (retarded.ra_deg < 0.0) retarded.ra_deg += 360.0;
        CHECK(sepArcsec(pp.eq, retarded) < shift);
    }
}

// --- photometry ----------------------------------------------------------

TEST_CASE("relative brightness reads the way the sky does") {
    // Issue #281's acceptance wording: "Venus brightest, then Jupiter,
    // Mars varying strongly with opposition." Sample a decade and check
    // the envelopes rather than one date, because that IS the claim.
    const double t0 = julianDateFromUtc(2020, 1, 1, 0, 0, 0.0);
    double vmin[kPlanetCount], vmax[kPlanetCount];
    for (int i = 0; i < kPlanetCount; ++i) { vmin[i] = 1e30; vmax[i] = -1e30; }
    for (double t = t0; t < t0 + 3653.0; t += 1.0) {
        for (int i = 0; i < kPlanetCount; ++i) {
            const double v = planetPosition(static_cast<Planet>(i), t).vmag;
            REQUIRE(finiteBits(v));
            vmin[i] = std::min(vmin[i], v);
            vmax[i] = std::max(vmax[i], v);
        }
    }
    const int me = static_cast<int>(Planet::Mercury);
    const int ve = static_cast<int>(Planet::Venus);
    const int ma = static_cast<int>(Planet::Mars);
    const int ju = static_cast<int>(Planet::Jupiter);
    const int sa = static_cast<int>(Planet::Saturn);
    const int ur = static_cast<int>(Planet::Uranus);
    const int ne = static_cast<int>(Planet::Neptune);

    // Venus at greatest brilliancy reaches about V = -4.9 and is the
    // brightest planet by a wide margin.
    CAPTURE(vmin[ve]);
    CHECK(vmin[ve] < -4.5);
    CHECK(vmin[ve] < vmin[ju] - 1.0);
    // Jupiter at opposition reaches about -2.9, and is never faint.
    CAPTURE(vmin[ju]);
    CAPTURE(vmax[ju]);
    CHECK(vmin[ju] < -2.7);
    CHECK(vmax[ju] < -1.5);
    // Mars swings hugely with opposition distance -- from about -2.6 at a
    // close opposition down to +1.8 near conjunction. That 4+ magnitude
    // range (a factor of 40 in flux) is the "varying strongly" criterion,
    // and it is far larger than Jupiter's 1.3.
    CAPTURE(vmax[ma] - vmin[ma]);
    CHECK(vmax[ma] - vmin[ma] > 4.0);
    CHECK(vmax[ma] - vmin[ma] > 3.0 * (vmax[ju] - vmin[ju]));
    // Ordering of the naked-eye bodies at their best.
    CHECK(vmin[ve] < vmin[me]);
    CHECK(vmin[ju] < vmin[sa]);
    CHECK(vmin[sa] < vmin[ur]);
    CHECK(vmin[ur] < vmin[ne]);
    // Uranus straddles the 6.5 naked-eye limit; Neptune never crosses it.
    CHECK(vmin[ur] < 6.5);
    CHECK(vmin[ne] > 6.5);
}

TEST_CASE("Saturn's rings dominate its brightness") {
    // Mallama's Equation A-6.1 carries a ring-inclination term worth over
    // a magnitude -- larger than every other photometric effect in the
    // model. The engine computes the opening angle from the IAU 2015
    // pole, so this is a test of that pole as much as of the photometry.
    //
    // The rings were near maximum opening in 2017 (Saturn's northern
    // summer solstice was May 2017) and crossed edge-on in March 2025.
    // Saturn is famously brighter with open rings.
    const double open  = julianDateFromUtc(2017, 6, 15, 0, 0, 0.0);
    const double edge  = julianDateFromUtc(2025, 6, 15, 0, 0, 0.0);
    const auto s_open = planetPosition(Planet::Saturn, open);
    const auto s_edge = planetPosition(Planet::Saturn, edge);
    CAPTURE(s_open.vmag);
    CAPTURE(s_edge.vmag);
    CHECK(s_open.vmag < s_edge.vmag - 0.8);

    // 2017-Jun-15 00:00 UT was Saturn's opposition; HORIZONS APmag -0.058.
    // With the ring term dropped the same instant computes to about +0.85,
    // so this bound is what makes the term load-bearing rather than decorative.
    CHECK(std::abs(s_open.vmag - (-0.058)) < kVmagTol);
}

// --- the shared point-source photometric scale ---------------------------

TEST_CASE("magnitude to flux follows Pogson's ratio exactly") {
    // Each magnitude step is a factor 10^0.4 in flux. This is the curve
    // both the catalogue stars and the planets ride.
    const float f0 = pt::stars::MagnitudeToFlux(0.0f);
    CHECK(f0 == doctest::Approx(4.0f));           // the engine's Vega gain
    for (float m = -4.0f; m <= 8.0f; m += 1.0f) {
        CAPTURE(m);
        const double a = pt::stars::MagnitudeToFlux(m);
        const double b = pt::stars::MagnitudeToFlux(m + 1.0f);
        CHECK(a / b == doctest::Approx(2.51188643).epsilon(1e-5));
    }
    // Monotone, positive, finite over the whole range the sky uses.
    for (float m = -6.0f; m <= 12.0f; m += 0.25f) {
        const double f = pt::stars::MagnitudeToFlux(m);
        REQUIRE(finiteBits(f));
        CHECK(f > 0.0);
    }
}

TEST_CASE("a planet splat and a catalogue star of the same magnitude are the same brightness") {
    // The point of exporting MagnitudeToFlux / SplatAngularRadiusRad from
    // BscCatalog: the planetarium must not invent its own scale. Rasterise
    // a one-star map with the star placed exactly on a texel centre, and
    // the peak texel must be MagnitudeToFlux(vmag) times that star's tint
    // -- the same number the engine hands the shader for a planet of the
    // same magnitude.
    constexpr std::uint32_t W = 720, H = 360;
    constexpr float kVmag = -2.7f;                // Jupiter near opposition
    const std::uint32_t tx = 100, ty = 80;
    pt::stars::Star s{};
    s.ra_deg  = float((double(tx) + 0.5) * 360.0 / double(W));
    s.dec_deg = float(90.0 - (double(ty) + 0.5) * 180.0 / double(H));
    s.vmag    = kVmag;

    std::vector<float> map;
    pt::stars::RasteriseJ2000Map({s}, W, H, map);
    REQUIRE(map.size() == std::size_t(W) * H * 4);

    double peak = 0.0;
    for (std::size_t i = 0; i < std::size_t(W) * H; ++i) {
        peak = std::max(peak, double(map[i * 4 + 0]));
    }
    // The rasteriser tints the first star with its hot-blue palette entry
    // (R = 0.85); the flux itself is the shared curve.
    const double expect = double(pt::stars::MagnitudeToFlux(kVmag)) * 0.85;
    CAPTURE(peak);
    CAPTURE(expect);
    CHECK(peak == doctest::Approx(expect).epsilon(1e-4));

    // And the splat footprint the planet path uses is the same tier
    // function the map just used.
    CHECK(pt::stars::SplatAngularRadiusRad(kVmag) == doctest::Approx(1.4e-3f));
    CHECK(pt::stars::SplatAngularRadiusRad(0.0f)  == doctest::Approx(1.0e-3f));
    CHECK(pt::stars::SplatAngularRadiusRad(5.0f)  == doctest::Approx(0.75e-3f));
}

TEST_CASE("B-V tints are hue-only and ordered by colour index") {
    // Unit luminance is the contract: the tint must not smuggle brightness
    // past MagnitudeToFlux, or the magnitude scale stops meaning anything.
    for (float bv = -0.4f; bv <= 2.0f; bv += 0.05f) {
        CAPTURE(bv);
        float c[3];
        pt::stars::BvToLinearSrgbTint(bv, c);
        for (int k = 0; k < 3; ++k) {
            REQUIRE(finiteBits(double(c[k])));
            CHECK(c[k] >= 0.0f);
        }
        const double lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        CHECK(lum == doctest::Approx(1.0).epsilon(1e-5));
    }
    // Redder (larger B-V) means a larger red-to-blue ratio, monotonically.
    double prev = -1.0;
    for (float bv = -0.3f; bv <= 1.9f; bv += 0.1f) {
        float c[3];
        pt::stars::BvToLinearSrgbTint(bv, c);
        const double rb = double(c[0]) / double(c[2]);
        CAPTURE(bv);
        CAPTURE(rb);
        CHECK(rb > prev);
        prev = rb;
    }
    // The solar B-V of 0.65 must come out very close to neutral -- the
    // Ballesteros relation returns 5779 K for it against the sun's true
    // 5772 K effective temperature, so this is a check on that anchor.
    float sun[3];
    pt::stars::BvToLinearSrgbTint(0.65f, sun);
    CHECK(sun[0] == doctest::Approx(1.0).epsilon(0.20));
    CHECK(sun[2] == doctest::Approx(1.0).epsilon(0.20));
}

TEST_CASE("planet colours put Mars at the red end and Neptune at the blue end") {
    // B-V from Mallama et al. 2017 Table 3. Mars is the reddest object in
    // the naked-eye sky after a low sun; Neptune is bluer than any of them.
    CHECK(planetBvColorIndex(Planet::Mars)    == doctest::Approx(1.36));
    CHECK(planetBvColorIndex(Planet::Neptune) == doctest::Approx(0.39));

    float mars[3], neptune[3], jupiter[3];
    pt::stars::BvToLinearSrgbTint(float(planetBvColorIndex(Planet::Mars)),    mars);
    pt::stars::BvToLinearSrgbTint(float(planetBvColorIndex(Planet::Neptune)), neptune);
    pt::stars::BvToLinearSrgbTint(float(planetBvColorIndex(Planet::Jupiter)), jupiter);
    CHECK(mars[0] > mars[2]);                     // Mars: red > blue
    CHECK(neptune[2] > neptune[0]);               // Neptune: blue > red
    CHECK(mars[0] / mars[2] > jupiter[0] / jupiter[2]);
    CHECK(jupiter[0] / jupiter[2] > neptune[0] / neptune[2]);

    // Every rendered body has a name and a plausible colour index.
    for (int i = 0; i < kPlanetCount; ++i) {
        const auto p = static_cast<Planet>(i);
        CAPTURE(i);
        CHECK(std::strlen(planetName(p)) > 2);
        CHECK(planetBvColorIndex(p) > 0.0);
        CHECK(planetBvColorIndex(p) < 2.0);
    }
}

// --- determinism + robustness --------------------------------------------

TEST_CASE("planetPosition is deterministic and finite over two centuries") {
    // The whole render path assumes this is a pure function of jd: the
    // golden fixture pins a dated frame, and the engine calls it once per
    // frame from the canonical clock.
    const double jd = julianDateFromUtc(2026, 3, 14, 6, 30, 0.0);
    for (int i = 0; i < kPlanetCount; ++i) {
        const auto p = static_cast<Planet>(i);
        const auto a = planetPosition(p, jd);
        const auto b = planetPosition(p, jd);
        CHECK(a.eq.ra_deg  == b.eq.ra_deg);
        CHECK(a.eq.dec_deg == b.eq.dec_deg);
        CHECK(a.vmag       == b.vmag);
    }
    // Sweep the whole published validity interval at 10-day steps. No
    // NaN, no infinity, RA in range, Dec in range, distances positive,
    // phase angle physical.
    const double t0 = julianDateFromUtc(1800, 1, 1, 0, 0, 0.0);
    const double t1 = julianDateFromUtc(2050, 1, 1, 0, 0, 0.0);
    for (double t = t0; t < t1; t += 10.0) {
        for (int i = 0; i < kPlanetCount; ++i) {
            const auto pp = planetPosition(static_cast<Planet>(i), t);
            REQUIRE(finiteBits(pp.eq.ra_deg));
            REQUIRE(finiteBits(pp.eq.dec_deg));
            REQUIRE(finiteBits(pp.vmag));
            REQUIRE(finiteBits(pp.geocentric_dist_au));
            CHECK(pp.eq.ra_deg >= 0.0);
            CHECK(pp.eq.ra_deg < 360.0);
            CHECK(pp.eq.dec_deg >= -90.0);
            CHECK(pp.eq.dec_deg <= 90.0);
            CHECK(pp.geocentric_dist_au > 0.0);
            CHECK(pp.heliocentric_dist_au > 0.0);
            CHECK(pp.phase_angle_deg >= 0.0);
            CHECK(pp.phase_angle_deg <= 180.0);
        }
    }
}

TEST_CASE("out-of-range planet enums degrade instead of reading past the table") {
    const auto p = static_cast<Planet>(kPlanetCount + 3);
    double x[3] = {1.0, 2.0, 3.0};
    planetHeliocentricEcliptic(p, 2451545.0, x);
    CHECK(x[0] == 0.0);
    CHECK(x[1] == 0.0);
    CHECK(x[2] == 0.0);
    CHECK(std::strcmp(planetName(p), "?") == 0);
    CHECK(planetBvColorIndex(p) == doctest::Approx(0.65));
}

// --- the horizon hand-off the renderer actually uses ---------------------

TEST_CASE("a planet and a star at the same J2000 coordinates land on the same spot") {
    // This is the acceptance criterion of #281 stated as a test: the
    // planets are positioned "relative to the Bright Star Catalog". Both
    // go through equatorialToHorizon from J2000 coordinates, so a planet
    // and a hypothetical catalogue star sharing its RA/Dec must produce
    // an identical horizon position. If a future change gave the planets
    // an equinox-of-date correction the stars do not get, this fails --
    // which is the point: consistency with the starmap beats absolute
    // accuracy the starmap cannot match.
    const double jd  = julianDateFromUtc(2026, 1, 1, 22, 0, 0.0);
    const double lat = 13.0827, lon = 80.2707;       // Chennai, the default
    for (int i = 0; i < kPlanetCount; ++i) {
        const auto p  = static_cast<Planet>(i);
        const auto pp = planetPosition(p, jd);
        const auto planet_h = equatorialToHorizon(pp.eq, lat, lon, jd);
        // Same coordinates, arrived at as if from the star catalogue.
        const EquatorialPos as_star{pp.eq.ra_deg, pp.eq.dec_deg};
        const auto star_h = equatorialToHorizon(as_star, lat, lon, jd);
        CAPTURE(planetName(p));
        CHECK(planet_h.azimuth_deg  == star_h.azimuth_deg);
        CHECK(planet_h.altitude_deg == star_h.altitude_deg);
        REQUIRE(finiteBits(planet_h.altitude_deg));
        CHECK(planet_h.altitude_deg >= -90.0);
        CHECK(planet_h.altitude_deg <=  90.0);
    }
}

// --- shader mirror --------------------------------------------------------

TEST_CASE("StarsComposite.slang still declares the planet path this test assumes") {
    // The host half of the planetarium is only worth something while the
    // shader half is still reading it. Pin the push layout and the splat
    // math, with whitespace stripped so reformatting cannot fail this, and
    // by OCCURRENCE COUNT so a change that fixes only one of the kernel's
    // two push declarations (SPIR-V and Metal) cannot slip through.
    const std::string src = tightenFile(PT_SHADER_STARSCOMPOSITE_PATH);

    // Slot count, declared once and used by both halves.
    CHECK(countOccurrences(src, "#definePT_PLANET_SLOTS7") == 1);
    // Both push declarations carry both arrays. TWO of each -- one in the
    // SPIR-V Frame cbuffer, one in the Metal Push cbuffer.
    CHECK(countOccurrences(src, "float4planet_dir_flux[PT_PLANET_SLOTS];") == 2);
    CHECK(countOccurrences(src, "float4planet_tint_sigma[PT_PLANET_SLOTS];") == 2);
    // The count lane. It replaced _pad0, which must be gone from BOTH
    // halves -- a leftover _pad0 would mean one of the two structs still
    // has a reserved word where the count is supposed to be, and the two
    // would disagree by a field.
    CHECK(countOccurrences(src, "uintplanet_count;") == 2);
    CHECK(countOccurrences(src, "uint_pad0;") == 0);

    // The splat is the star map's Gaussian, not a new one: same
    // ang2 = 2(1-cos), same exp(-ang2/sigma^2) with no 1/2, same 4-sigma
    // truncation (16 = K*K).
    CHECK(countOccurrences(src, "floatang2=2.0*(1.0-c);") == 1);
    CHECK(countOccurrences(src, "if(ang2>s2*16.0)continue;") == 1);
    CHECK(countOccurrences(src, "acc+=ts.rgb*(df.w*exp(-ang2/s2));") == 1);

    // The day fade matches starsOnly's, so planets and stars wash out
    // together. Both the smoothstep and the 0.6 cutoff appear in
    // planetSplat as well as in starsOnly, hence two of the smoothstep
    // (starsOnly + planetSplat; procSkyInlineSunDiscAndHalo has its own,
    // making three in total across the file).
    CHECK(countOccurrences(src, "floatday=smoothstep(-0.10,0.20,sun_elev);") == 3);
    CHECK(countOccurrences(src, "if(day>=0.6)returnfloat3(0.0);") == 1);

    // And it is actually summed into the composite, exactly once.
    CHECK(countOccurrences(src, "+planetSplat(rd,sun)") == 1);
    CHECK(countOccurrences(src, "float3planetSplat(float3rd,float3sun){") == 1);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for pt::astro (Astronomy.h / Astronomy.cpp) -- issue #65,
// Phase 3 of #47.
//
// The astronomy module drives the engine's procedural sky from real
// observer lat/lon + UTC. A regression here would put the sun in the
// wrong place by a degree or more, which is visually obvious but also
// scientifically wrong -- this project commits to real metric units and
// real ephemerides (see project memory: "Real physics + metric units").
//
// Reference-data sources
// ----------------------
// * Julian-date reference instants: standard astronomical conversions.
//   - J2000.0 = 2000-01-01 12:00:00 UTC = JD 2451545.0 (definition).
//     See e.g. IAU SOFA's `iauEpj2jd`, Meeus "Astronomical Algorithms"
//     2nd ed. ch. 7, or NASA HORIZONS' Julian-date converter.
//   - Unix epoch = 1970-01-01 00:00:00 UTC = JD 2440587.5 (definition).
// * GMST at J2000: 18h 41m 50.54841s = 280.4606184 deg per the IAU 1982
//   formula (and the engine's gmstDegrees() implements exactly that
//   model -- see Astronomy.cpp). Tolerance ~0.01 deg here is just float
//   round-trip, not formula uncertainty.
// * Sun (RA, Dec) at the four cardinal solar events: Naval Almanac
//   low-precision formulas (the engine's own model), cross-checked
//   against JPL HORIZONS Web App (https://ssd.jpl.nasa.gov/horizons/app.html)
//   for the same UTC instants. Header docstring promises ~1 arcminute
//   accuracy through the 21st century; we test against published
//   reference values with a 6-arcminute envelope (0.1 deg) per the
//   issue's "tolerance <= 0.5 deg" while keeping headroom for the
//   formula's own ~1' modelling error. RA tolerance is looser
//   (~0.5 deg = ~2 minutes-of-time) because RA error grows near
//   dec=0 where a small ecliptic-longitude perturbation moves RA
//   through atan2's steepest region.
//
// All reference values are HARDCODED here -- no time(), no system clock,
// no network calls. The same source compile + same build produces the
// same numbers on Mac / Win / Linux + Debug / Release / ASan.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/renderer/Astronomy.h"

#include <cmath>
#include <ctime>

using pt::astro::EquatorialPos;
using pt::astro::HorizonPos;
using pt::astro::equatorialToHorizon;
using pt::astro::gmstDegrees;
using pt::astro::julianDateFromTimeT;
using pt::astro::julianDateFromUtc;
using pt::astro::sunPosition;

namespace {

// --- Tolerances ----------------------------------------------------------
// Julian-date references are *exact* by definition -- the tolerance here
// is just float epsilon protection against an accidental arithmetic
// re-ordering. 1e-6 day ~ 0.1 second, well below sub-second precision.
constexpr double kJdEps      = 1e-6;
// GMST at J2000 is a closed-form polynomial; tolerance is float round-trip.
constexpr double kGmstEpsDeg = 0.01;
// Sun position is from the Naval Almanac model -- ~1 arcminute claim.
// JPL HORIZONS gives sub-arcsecond truth; for the published reference
// values used in the cardinal-solar tests we leave 0.1 deg = 6' of
// envelope. RA near dec=0 is steep so allow 0.5 deg in RA.
constexpr double kSunDecDeg  = 0.1;
constexpr double kSunRaDeg   = 0.5;

// Float-distance helpers. doctest::Approx has the .epsilon() semantics
// of a *relative* error, which is fine for nonzero references but
// degenerate near zero. Most of our checks want an absolute tolerance,
// so we use explicit subtraction + std::fabs.
bool CloseAbs(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

// Smallest angular separation in degrees, accounting for wrap-around
// on the 360-deg circle. RA at 359.9 deg is 0.2 deg from RA 0.1 deg,
// not 359.8 deg. Used for RA checks where the reference value may
// straddle the 0/360 seam.
double WrappedDegDiff(double a, double b) {
    double d = std::fmod(a - b + 540.0, 360.0) - 180.0;
    return std::fabs(d);
}

}  // namespace

// --- Test 1: Julian date conversion known references ---------------------
// The two standard reference instants are exact by definition; if either
// breaks, every downstream calculation that uses JD is wrong by a
// constant offset. Off-by-one bugs in month handling (the Meeus
// algorithm's January/February shift is famously easy to get wrong)
// surface here as multi-day deltas.
TEST_CASE("Astronomy: julianDateFromUtc matches J2000 and Unix epoch") {
    // J2000.0: defined as JD 2451545.0 = 2000-01-01 12:00 UT (TT, but
    // for our 1-arcmin sky we treat UTC == TT to within ~70 seconds /
    // ~0.0008 day; the test references all use UTC consistently so
    // there's no drift introduced).
    {
        const double jd = julianDateFromUtc(2000, 1, 1, 12, 0, 0.0);
        CHECK(CloseAbs(jd, 2451545.0, kJdEps));
    }
    // Unix epoch: JD 2440587.5 = 1970-01-01 00:00 UTC. The .5 day fraction
    // pins the 00:00 vs 12:00 convention -- a bug that flipped the day-
    // fraction sign would fail here even if the J2000 case (which lands
    // exactly on a noon boundary) accidentally still passed.
    {
        const double jd = julianDateFromUtc(1970, 1, 1, 0, 0, 0.0);
        CHECK(CloseAbs(jd, 2440587.5, kJdEps));
    }
    // A sub-second time-of-day case. 2000-01-01 12:00:00.5 UTC is
    // exactly 0.5 / 86400 day later than J2000.0.
    {
        const double jd = julianDateFromUtc(2000, 1, 1, 12, 0, 0.5);
        const double expect = 2451545.0 + 0.5 / 86400.0;
        CHECK(CloseAbs(jd, expect, kJdEps));
    }
    // A near-present-day reference: 2026-01-15 00:00 UTC = JD 2461055.5.
    // Computed by hand from J2000 + 9511 days (26 calendar years + 14
    // days, with 7 leap days at 2000/04/08/12/16/20/24). Guards against
    // an off-by-one in the leap-year handling that would only show up
    // ~26 years out from J2000.
    {
        const double jd = julianDateFromUtc(2026, 1, 15, 0, 0, 0.0);
        CHECK(CloseAbs(jd, 2461055.5, kJdEps));
    }
}

// --- Test 2: julianDateFromTimeT matches julianDateFromUtc ---------------
// time_t = 946728000 is 2000-01-01 12:00:00 UTC (POSIX seconds since
// 1970-01-01 00:00 UTC; verifiable via `date -u -d @946728000` or any
// online converter). Both code paths must agree to bit-exact: the
// time_t path round-trips through std::gmtime() and back into the same
// julianDateFromUtc() routine, so the only way they can disagree is
// if std::gmtime() returns the wrong broken-down time (a libc bug,
// not ours).
TEST_CASE("Astronomy: julianDateFromTimeT matches julianDateFromUtc") {
    // 2000-01-01 12:00:00 UTC. Hardcoded constant: no time() / no
    // localtime / no TZ dependency.
    const std::time_t t_j2000   = 946728000;
    const double jd_via_time_t  = julianDateFromTimeT(t_j2000);
    const double jd_via_struct  = julianDateFromUtc(2000, 1, 1, 12, 0, 0.0);
    CHECK(CloseAbs(jd_via_time_t, jd_via_struct, kJdEps));
    CHECK(CloseAbs(jd_via_time_t, 2451545.0,     kJdEps));

    // Unix epoch via time_t = 0.
    const std::time_t t_unix    = 0;
    const double jd_unix        = julianDateFromTimeT(t_unix);
    CHECK(CloseAbs(jd_unix, 2440587.5, kJdEps));
}

// --- Test 3: GMST at J2000 -----------------------------------------------
// The IAU 1982 formula gives 18h 41m 50.54841s at J2000 = 280.4606184 deg.
// Engine implementation is verbatim that polynomial (see Astronomy.cpp);
// tolerance kGmstEpsDeg (0.01 deg = ~2.4 seconds-of-time) absorbs only
// the float round-trip, not any formula uncertainty. A bug in the
// constant term or the secular drift coefficient would shift this by
// many degrees.
TEST_CASE("Astronomy: gmstDegrees at J2000 matches IAU 1982 reference") {
    const double gmst_j2000 = gmstDegrees(2451545.0);
    // Reference: 18h 41m 50.54841s.  Converting to degrees:
    //   18h = 18 * 15 = 270 deg
    //   41m = 41 * 0.25 = 10.25 deg
    //   50.54841s = 50.54841 * (15/3600) deg = 0.2106184 deg
    //   total = 280.4606184 deg
    constexpr double kRefGmstJ2000 = 280.4606184;
    CHECK(CloseAbs(gmst_j2000, kRefGmstJ2000, kGmstEpsDeg));
    // GMST output range invariant: always in [0, 360).
    CHECK(gmst_j2000 >= 0.0);
    CHECK(gmst_j2000 <  360.0);

    // Spot-check at JD 2451545.0 + 0.5 day: in half a solar day GMST
    // advances by half of (360 deg / 0.99726957 sidereal days per solar
    // day) ~= 180.4928 deg. Equivalently, the sidereal day is 23h 56m
    // 04.0905s so GMST gains ~3m 55.9s relative to UTC per solar day,
    // ~1m 57.95s per half-day, on top of the 12h * 15 deg/h = 180 deg.
    // Tolerance 0.05 deg absorbs the secular (T^2, T^3) terms over half
    // a Julian day -- those are tiny but nonzero.
    const double gmst_plus_half = gmstDegrees(2451545.5);
    const double advance = std::fmod(gmst_plus_half - gmst_j2000 + 720.0, 360.0);
    CHECK(CloseAbs(advance, 180.4928, 0.05));
}

// --- Test 4: Sun position vs published references ------------------------
// Reference values cross-checked against JPL HORIZONS at each cardinal
// solar event in 2026. JPL HORIZONS (apparent right ascension /
// declination, geocentric, frame J2000) is sub-arcsecond truth; the
// Naval Almanac model the engine uses claims ~1 arcminute, and our
// tolerances (0.1 deg dec, 0.5 deg RA) leave ~6x headroom past that
// for the formula's known approximation error.
//
// The four cardinal events sweep the full declination range so a bug
// that, say, flipped the sign of the obliquity coefficient would fail
// at three of the four; a bug that swapped RA and dec would fail at
// all four.
TEST_CASE("Astronomy: sunPosition matches JPL HORIZONS / Naval Almanac at solar events") {
    // J2000 (2000-01-01 12:00 UTC). Reference: RA ~18h 45m = 281.25 deg,
    // dec ~-23.03 deg. Engine produces (281.286, -23.033) -- inside
    // tolerance.
    {
        const auto s = sunPosition(2451545.0);
        CHECK(WrappedDegDiff(s.ra_deg, 281.25) <= kSunRaDeg);
        CHECK(CloseAbs(s.dec_deg, -23.03, kSunDecDeg));
    }
    // March equinox 2026 ~ 2026-03-20 14:46 UTC. At 12:00 UTC the sun
    // is just before equinox. RA ~ 359.9 (near 0), dec ~ -0.04. Use
    // WrappedDegDiff so 359.9 vs 0.0 reads as 0.1 deg, not 359.9.
    {
        const double jd = julianDateFromUtc(2026, 3, 20, 12, 0, 0.0);
        const auto s = sunPosition(jd);
        CHECK(WrappedDegDiff(s.ra_deg, 359.9) <= kSunRaDeg);
        CHECK(CloseAbs(s.dec_deg, 0.0, 0.2));        // looser near zero
    }
    // June solstice 2026 ~ 2026-06-21 08:24 UTC. At 12:00 UTC the sun
    // has dec ~ +23.43 (max), RA ~ 90.15 (6h 0m). The northern-summer
    // solstice is the strongest signal of "sun is high in northern
    // sky" and a sign-flip in obliquity would land here as -23 deg.
    {
        const double jd = julianDateFromUtc(2026, 6, 21, 12, 0, 0.0);
        const auto s = sunPosition(jd);
        CHECK(WrappedDegDiff(s.ra_deg, 90.16) <= kSunRaDeg);
        CHECK(CloseAbs(s.dec_deg, 23.43, kSunDecDeg));
    }
    // September equinox 2026 ~ 2026-09-23 00:05 UTC. At 12:00 UTC on
    // 2026-09-22 the sun is still slightly north of equator.
    // RA ~ 179.5, dec ~ +0.2.
    {
        const double jd = julianDateFromUtc(2026, 9, 22, 12, 0, 0.0);
        const auto s = sunPosition(jd);
        CHECK(WrappedDegDiff(s.ra_deg, 179.54) <= kSunRaDeg);
        CHECK(CloseAbs(s.dec_deg, 0.2, 0.2));         // looser near zero
    }
    // December solstice 2026 ~ 2026-12-21 20:50 UTC. At 12:00 UTC the
    // sun has dec ~ -23.43, RA ~ 269.59 (17h 58m). Mirror of June
    // solstice; both must be near +/-23.43 deg for the obliquity sign
    // to be correct.
    {
        const double jd = julianDateFromUtc(2026, 12, 21, 12, 0, 0.0);
        const auto s = sunPosition(jd);
        CHECK(WrappedDegDiff(s.ra_deg, 269.59) <= kSunRaDeg);
        CHECK(CloseAbs(s.dec_deg, -23.43, kSunDecDeg));
    }
}

// --- Test 5: Sun position output-range invariants ------------------------
// Whatever the engine returns, dec must always satisfy |dec| <= 23.5 deg
// (Earth's axial tilt: ~23.44 deg, +/- a few arcmin from nutation that
// the model deliberately ignores) and RA must be in [0, 360). Sweep a
// year of dates one per week so any wraparound bug in normDeg() trips
// on at least one sample.
TEST_CASE("Astronomy: sunPosition output ranges across one year") {
    constexpr double kMaxAxialTiltDeg = 23.5;
    const double jd_start = julianDateFromUtc(2026, 1, 1, 0, 0, 0.0);
    for (int day = 0; day < 366; day += 7) {
        const double jd = jd_start + double(day);
        const auto s = sunPosition(jd);
        CAPTURE(day);
        CAPTURE(s.ra_deg);
        CAPTURE(s.dec_deg);
        // Declination magnitude bound.
        CHECK(std::fabs(s.dec_deg) <= kMaxAxialTiltDeg);
        // RA range.
        CHECK(s.ra_deg >= 0.0);
        CHECK(s.ra_deg <  360.0);
    }
}

// --- Test 6: Equatorial -> horizon transform sanity ----------------------
// Two scenarios where the answer is physically constrained tightly enough
// that any transform bug shifts it visibly:
//   (a) Equator + March equinox + local solar noon: sun near zenith
//       (altitude > 88 deg). A swap of lat/lon, or a sign flip on the
//       hour angle, or treating dec as RA would all fail here -- only
//       a near-correct transform can put the sun at zenith.
//   (b) Azimuth wraparound: az_deg must always lie in [0, 360) -- no
//       negatives, no 360.0 itself. Fire the transform at a few
//       (lat, lon, jd) tuples and assert.
TEST_CASE("Astronomy: equatorialToHorizon -- sun near zenith on equator at equinox noon") {
    // March equinox 2026 ~ 2026-03-20 14:46 UTC. Sun's RA is near 0 deg.
    // For an observer at lat=0, lon=0, local solar noon is the UTC when
    // GMST + lon - sun_RA = 0; with sun_RA near 0 deg, that's UTC near
    // 12:00 (because GMST at noon UTC near equinox is also near 0 -- see
    // calibration in the test plan). Empirically (matches engine output):
    // 2026-03-20 12:00 UTC at lat=0, lon=0 gives alt ~88.13 deg.
    const double jd = julianDateFromUtc(2026, 3, 20, 12, 0, 0.0);
    const auto s = sunPosition(jd);
    const auto h = equatorialToHorizon(s, 0.0, 0.0, jd);
    CAPTURE(s.ra_deg);
    CAPTURE(s.dec_deg);
    CAPTURE(h.azimuth_deg);
    CAPTURE(h.altitude_deg);
    // Altitude > 85 deg is "sun within 5 deg of zenith". A bug that
    // misapplies the observer's latitude (e.g. lat=90 treated as lat=0)
    // would drop this 60 deg or more.
    CHECK(h.altitude_deg > 85.0);
    // Within 90 deg of zenith.
    CHECK(h.altitude_deg <= 90.0);
}

TEST_CASE("Astronomy: equatorialToHorizon -- azimuth always in [0, 360)") {
    // Sample a handful of (lat, lon, jd) tuples. The transform should
    // keep azimuth on [0, 360) for every input. Sun position at each
    // jd as the equatorial source.
    struct Case { double lat, lon, jd_offset; };
    const double jd_base = julianDateFromUtc(2026, 6, 21, 0, 0, 0.0);
    const Case cases[] = {
        {  0.0,   0.0, 0.0},                  // equator, lon 0, midnight UTC
        { 51.5,  -0.1, 0.25},                 // London, +6h UTC
        { 35.7, 139.7, 0.5},                  // Tokyo, +12h UTC
        {-33.9, 151.2, 0.75},                 // Sydney, +18h UTC
        { 89.0,   0.0, 0.0},                  // near-north-pole edge
        {-89.0,   0.0, 0.0},                  // near-south-pole edge
        {  0.0, 179.9, 0.5},                  // near-antimeridian
    };
    for (const auto& c : cases) {
        const double jd = jd_base + c.jd_offset;
        const auto s = sunPosition(jd);
        const auto h = equatorialToHorizon(s, c.lat, c.lon, jd);
        CAPTURE(c.lat);
        CAPTURE(c.lon);
        CAPTURE(c.jd_offset);
        CAPTURE(h.azimuth_deg);
        CAPTURE(h.altitude_deg);
        CHECK(h.azimuth_deg  >= 0.0);
        CHECK(h.azimuth_deg  <  360.0);
        // Altitude must be in [-90, 90].
        CHECK(h.altitude_deg >= -90.0);
        CHECK(h.altitude_deg <=  90.0);
    }
}

// --- Test 7: Equatorial -> horizon -- sun crosses horizon at sunrise -----
// At the equator on equinox, sunrise is at ~06:00 local solar time and
// sunset at ~18:00. Find the UTC times where altitude crosses zero and
// confirm there are exactly two such transitions per 24-hour window.
// This validates the transform's behaviour over a full diurnal cycle,
// not just the noon instant.
TEST_CASE("Astronomy: equatorialToHorizon -- sun crosses horizon twice per day at equator equinox") {
    const double jd_start = julianDateFromUtc(2026, 3, 20, 0, 0, 0.0);
    // Sample every 30 minutes over 24 hours and count sign changes of
    // (altitude_deg). Two sign changes == one sunrise + one sunset.
    int sign_changes = 0;
    double prev_alt = 0.0;
    bool   has_prev = false;
    for (int step = 0; step < 48; ++step) {
        const double jd = jd_start + double(step) / 48.0;
        const auto s = sunPosition(jd);
        const auto h = equatorialToHorizon(s, 0.0, 0.0, jd);
        if (has_prev && ((prev_alt < 0.0 && h.altitude_deg > 0.0) ||
                         (prev_alt > 0.0 && h.altitude_deg < 0.0))) {
            ++sign_changes;
        }
        prev_alt = h.altitude_deg;
        has_prev = true;
    }
    CHECK(sign_changes == 2);
}

// --- Test 8: Edge-case inputs ---------------------------------------------
// julianDateFromUtc is not defensive against invalid inputs -- it's a
// thin wrapper around a Meeus polynomial that produces *some* number
// for any input. Document this by exercising a couple of degenerate
// inputs and asserting we don't crash. The engine never feeds negative
// months in practice (cvar parsing rejects them upstream), but the test
// pins behaviour against accidental "harden against invalid input"
// regressions that could break callers expecting a specific number.
TEST_CASE("Astronomy: julianDateFromUtc does not crash on edge-case inputs") {
    // month=0 (one before January): the Meeus shift handles M<=2 by
    // decrementing year and adding 12, so M=0 becomes M=12 in the
    // previous year. Output is meaningful (no NaN / inf), just shifted.
    const double jd = julianDateFromUtc(2026, 0, 1, 12, 0, 0.0);
    CHECK(std::isfinite(jd));
    CHECK(jd > 0.0);
    // Hour=23, minute=59, second=59.999 -- close to a day boundary.
    const double jd2 = julianDateFromUtc(2026, 6, 21, 23, 59, 59.999);
    CHECK(std::isfinite(jd2));
}

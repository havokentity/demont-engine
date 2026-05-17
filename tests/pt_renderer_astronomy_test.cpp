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
//   - J2000.0 is *strictly* defined at 2000-01-01 12:00 TT (Terrestrial
//     Time), not UTC; the difference (TT - UTC = ~64.184 s in 2000)
//     amounts to ~7e-4 day. This engine treats UTC == TT for the
//     ~1-arcminute sky model (the sun moves ~0.04 deg / hour at the
//     equator, so 65 s offsets the apparent position by ~7e-4 deg,
//     well below the model's claimed accuracy and our tolerances).
//     `Astronomy.h` documents the TT definition; tests below pass
//     UTC arguments and accept the result as "JD 2451545.0" within
//     the kJdEps tolerance.  References: IAU SOFA's `iauEpj2jd`,
//     Meeus "Astronomical Algorithms" 2nd ed. ch. 7, or NASA HORIZONS'
//     Julian-date converter.
//   - Unix epoch = 1970-01-01 00:00:00 UTC = JD 2440587.5 (definition,
//     same UTC vs TT note applies).
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
using pt::astro::moonDistanceKm;
using pt::astro::moonPhaseAngle;
using pt::astro::moonPosition;
using pt::astro::sunDistanceAu;
using pt::astro::sunPosition;
using pt::astro::worldToJ2000Matrix;

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

// --- Test 9: Moon position invariants -------------------------------------
// `moonPosition` returns RA in [0, 360) and dec in a band bounded by the
// sum of the ecliptic obliquity (23.4 deg) and the lunar inclination to
// the ecliptic (~5.1 deg). The full range is therefore |dec| <= 28.6 deg
// or so; we test against a slightly looser ~29 deg envelope. These are
// model-invariant truths (any conforming Meeus implementation must
// respect them), not reference values that need JPL HORIZONS truth.
TEST_CASE("Astronomy: moonPosition invariants across the J2000 epoch") {
    // Walk a full 29.5-day synodic cycle at half-day resolution starting
    // from J2000. ~60 samples; cheap and exhaustive within the range
    // restriction we're checking.
    constexpr double kDeclEnvelopeDeg = 29.0;
    for (int step = 0; step <= 59; ++step) {
        const double jd = 2451545.0 + 0.5 * step;
        const EquatorialPos p = moonPosition(jd);
        CAPTURE(jd);
        CAPTURE(p.ra_deg);
        CAPTURE(p.dec_deg);
        CHECK(std::isfinite(p.ra_deg));
        CHECK(std::isfinite(p.dec_deg));
        CHECK(p.ra_deg  >= 0.0);
        CHECK(p.ra_deg  <  360.0);
        CHECK(p.dec_deg >= -kDeclEnvelopeDeg);
        CHECK(p.dec_deg <=  kDeclEnvelopeDeg);
    }
}

TEST_CASE("Astronomy: moonPosition is deterministic for the same JD") {
    // Same JD twice -- function is pure, must return bit-identical
    // values. Guards against an accidental introduction of any
    // global state (caching that depends on call order, etc.).
    const double jd = 2451559.5;   // J2000 + 14.5 days (near first full moon)
    const EquatorialPos a = moonPosition(jd);
    const EquatorialPos b = moonPosition(jd);
    CHECK(a.ra_deg  == b.ra_deg);
    CHECK(a.dec_deg == b.dec_deg);
}

// --- Test 10: Moon phase angle cycles through 0..pi over a synodic month -
// The synodic month is ~29.53 days. Walking the cycle starting from
// J2000 -- when the moon is approximately 5.6 days past new -- we should
// see the phase angle traverse from near zero (new) through pi (full)
// and back. The agent doesn't need to identify which sample is new or
// full to test the range invariant; verifying the angle is always in
// [0, pi] and that the sample-to-sample spread covers a meaningful
// fraction of the range is enough to catch a sign-flipped or
// constant-output regression.
TEST_CASE("Astronomy: moonPhaseAngle stays in [0, pi] and covers the cycle") {
    constexpr double kPi = 3.14159265358979323846;
    double min_phase =  kPi + 1.0;     // sentinel above range
    double max_phase = -1.0;           // sentinel below range
    for (int step = 0; step <= 59; ++step) {
        const double jd = 2451545.0 + 0.5 * step;
        const EquatorialPos sun  = sunPosition(jd);
        const EquatorialPos moon = moonPosition(jd);
        const double phase = moonPhaseAngle(sun, moon);
        CAPTURE(jd);
        CAPTURE(phase);
        CHECK(std::isfinite(phase));
        CHECK(phase >= 0.0);
        CHECK(phase <= kPi);
        if (phase < min_phase) min_phase = phase;
        if (phase > max_phase) max_phase = phase;
    }
    // Over a synodic month we should see phase swing through a wide
    // range -- minimum near 0 (new) and maximum near pi (full). Allow
    // generous slack: min < 0.4 rad and max > kPi - 0.4 rad.
    CHECK(min_phase < 0.4);
    CHECK(max_phase > kPi - 0.4);
}

// --- Test 11: Moon distance stays in physical perigee..apogee range ------
// Real lunar orbit: perigee ~363,300 km, apogee ~405,500 km (per the
// header docstring). Engine model claims ~14% angular-size variation,
// matching this range. Test envelope kept loose at [350,000, 410,000]
// km to absorb the simplified-Meeus model's accuracy bound.
TEST_CASE("Astronomy: moonDistanceKm stays in perigee..apogee envelope") {
    for (int step = 0; step <= 59; ++step) {
        const double jd = 2451545.0 + 0.5 * step;
        const double d = moonDistanceKm(jd);
        CAPTURE(jd);
        CAPTURE(d);
        CHECK(std::isfinite(d));
        CHECK(d >= 350000.0);
        CHECK(d <= 410000.0);
    }
}

// --- Test 12: Sun distance stays in perihelion..aphelion envelope ---------
// Earth orbital eccentricity 0.0167 -> perihelion 0.983 AU, aphelion
// 1.017 AU per Astronomy.h. Use a 0.97..1.03 envelope to absorb the
// model's own ~0.001 AU residual.
TEST_CASE("Astronomy: sunDistanceAu stays in perihelion..aphelion envelope") {
    // Sample at month intervals across a year starting from J2000.
    for (int month_step = 0; month_step <= 12; ++month_step) {
        const double jd = 2451545.0 + 30.4 * month_step;
        const double d = sunDistanceAu(jd);
        CAPTURE(jd);
        CAPTURE(d);
        CHECK(std::isfinite(d));
        CHECK(d >= 0.97);
        CHECK(d <= 1.03);
    }
    // Mean distance check: the kSunDistanceMeanAu constant should be
    // close to 1.0 by construction.
    CHECK(CloseAbs(pt::astro::kSunDistanceMeanAu, 1.0, 0.01));
}

// --- Test 13: worldToJ2000Matrix orthonormality -------------------------
// The world->J2000 transform is a pure rotation: rows must be unit
// length, mutually orthogonal, and the matrix determinant must be
// +1 (right-handed; no reflection). A bug that swapped two rows or
// transposed a sub-block would fail here. Tolerance is float epsilon
// magnified to 1e-5 to absorb double->float rounding in the
// row-major output buffer.
TEST_CASE("Astronomy: worldToJ2000Matrix is a right-handed orthonormal rotation") {
    constexpr double kOrthoEps = 1e-5;
    // A spread of (observer, JD) tuples: London at J2000, Tokyo at
    // 2026 summer solstice, Sydney at 2026 winter solstice.
    struct Sample { double lat; double lon; double jd; };
    const Sample samples[] = {
        {51.4769, -0.0005,    2451545.0},                         // Greenwich at J2000
        {35.6762, 139.6503,   julianDateFromUtc(2026, 6, 21, 6, 0, 0.0)},
        {-33.8688, 151.2093,  julianDateFromUtc(2026, 12, 21, 18, 0, 0.0)},
    };
    for (const auto& s : samples) {
        float M[9] = {};
        worldToJ2000Matrix(s.lat, s.lon, s.jd, M);
        CAPTURE(s.lat);
        CAPTURE(s.lon);
        CAPTURE(s.jd);
        // All entries finite.
        for (int i = 0; i < 9; ++i) {
            CHECK(std::isfinite(M[i]));
        }
        // Row dot products: rows 0,1,2 each unit length, mutually orthogonal.
        auto rowDot = [&](int a, int b) -> double {
            return double(M[a*3+0]) * M[b*3+0]
                 + double(M[a*3+1]) * M[b*3+1]
                 + double(M[a*3+2]) * M[b*3+2];
        };
        CHECK(CloseAbs(rowDot(0, 0), 1.0, kOrthoEps));
        CHECK(CloseAbs(rowDot(1, 1), 1.0, kOrthoEps));
        CHECK(CloseAbs(rowDot(2, 2), 1.0, kOrthoEps));
        CHECK(CloseAbs(rowDot(0, 1), 0.0, kOrthoEps));
        CHECK(CloseAbs(rowDot(0, 2), 0.0, kOrthoEps));
        CHECK(CloseAbs(rowDot(1, 2), 0.0, kOrthoEps));
        // Determinant = +1 (right-handed) computed via cofactor expansion
        // on row 0.  det = r0 . (r1 x r2). Equivalent to triple product.
        const double cx = double(M[3]) * M[7] - double(M[4]) * M[6];
        const double cy = double(M[5]) * M[6] - double(M[3]) * M[8];
        const double cz = double(M[4]) * M[8] - double(M[5]) * M[7];
        // The cofactor cross product is (r1 x r2), and r1 x r2 should
        // equal r0 (or -r0 for left-handed). Use row-0 dot (r1 x r2).
        // Note the sign convention above produces (r1 x r2) directly:
        //   (r1 x r2).x = r1.y * r2.z - r1.z * r2.y -> M[4]*M[8] - M[5]*M[7]
        // Recompute with the correct formula for clarity:
        const double crossYZ = double(M[4]) * M[8] - double(M[5]) * M[7];
        const double crossZX = double(M[5]) * M[6] - double(M[3]) * M[8];
        const double crossXY = double(M[3]) * M[7] - double(M[4]) * M[6];
        const double det = double(M[0]) * crossYZ
                         + double(M[1]) * crossZX
                         + double(M[2]) * crossXY;
        CHECK(CloseAbs(det, 1.0, kOrthoEps));
        (void)cx; (void)cy; (void)cz;
    }
}

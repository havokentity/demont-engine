#pragma once

// Tiny astronomy helpers: Julian date arithmetic, Sun position
// (low-precision Naval Almanac formulas, ~1 arcmin accuracy --
// good enough for "sun rises in roughly the right direction"),
// and equatorial -> horizon coordinate transform.
//
// Used by the engine to drive the procedural sky from real lat/lon
// + UTC, so e.g. r_sky_use_astronomical 1 puts the sun where it
// actually is for the user's location and time. Stars-by-catalog
// will reuse the same eq->horizon transform later.

#include <ctime>

namespace pt::astro {

// Days since J2000.0 (= JD 2451545.0, 2000-01-01 12:00 TT).
double julianDateFromUtc(int year, int month, int day,
                         int hour, int minute, double second);

// Convenience: convert std::time_t (UTC) to JD.
double julianDateFromTimeT(std::time_t t);

// Greenwich Mean Sidereal Time in degrees [0, 360).
double gmstDegrees(double jd);

struct EquatorialPos {
    double ra_deg;   // right ascension, J2000 of date, [0, 360)
    double dec_deg;  // declination, [-90, 90]
};
struct HorizonPos {
    double azimuth_deg;   // 0 = north, 90 = east, [0, 360)
    double altitude_deg;  // 0 = horizon, 90 = zenith, [-90, 90]
};

// Sun position at the given Julian date. Naval Almanac low-precision
// formulas; accurate to ~1 arcminute through the 21st century.
EquatorialPos sunPosition(double jd);

// Convert equatorial (ra, dec) to local horizon (az, alt) for an
// observer at lat/lon (degrees, +N / +E) and the given JD.
HorizonPos equatorialToHorizon(EquatorialPos eq,
                               double observer_lat_deg,
                               double observer_lon_deg,
                               double jd);

}  // namespace pt::astro

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
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

// Moon position at the given Julian date. Meeus chapter 47 simplified
// formulas keeping the dominant ELP-2000/82 periodic terms; accurate
// to ~10 arcminutes for ra/dec, ~1 day for phase calculations.
EquatorialPos moonPosition(double jd);

// Phase angle of the moon (Sun-Earth-Moon angle), radians. 0 = new
// (moon between sun and earth, dark side facing earth), pi = full.
double moonPhaseAngle(EquatorialPos sun, EquatorialPos moon);

// Earth-Moon distance in kilometres at the given JD. Real moon orbit
// is elliptical: perigee ~363,300 km, apogee ~405,500 km; the apparent
// angular size varies ~14% across this range. Renderer uses this to
// scale the moon disc.
double moonDistanceKm(double jd);

// Earth-Sun distance in astronomical units at the given JD. Earth's
// orbit eccentricity is 0.0167 -- perihelion ~0.983 AU (early Jan),
// aphelion ~1.017 AU (early Jul). Apparent sun size varies ~3.4%.
double sunDistanceAu(double jd);

// Mean reference distances. Use these as the "1.0x apparent size"
// reference; scale = mean / current.
constexpr double kMoonDistanceMeanKm = 384400.0;
constexpr double kSunDistanceMeanAu  = 1.000001018;  // ~149.6 Mkm

// Convert equatorial (ra, dec) to local horizon (az, alt) for an
// observer at lat/lon (degrees, +N / +E) and the given JD.
HorizonPos equatorialToHorizon(EquatorialPos eq,
                               double observer_lat_deg,
                               double observer_lon_deg,
                               double jd);

// Build the 3x3 rotation that maps a unit vector in the engine's world
// frame (+X east, +Y up, +Z south) into J2000 equatorial coordinates
// (+X = vernal equinox, +Z = north celestial pole). Used by the path
// tracer to look up a star at a given world ray direction inside a
// J2000-frame starmap texture: e_j = M * w. Output is row-major:
// out[0..2] = row 0, out[3..5] = row 1, out[6..8] = row 2. Precession
// from J2000 to current epoch is ignored (worth ~0.4 deg over 26 years
// — invisible at our angular resolution).
void worldToJ2000Matrix(double observer_lat_deg,
                        double observer_lon_deg,
                        double jd,
                        float  out_row_major[9]);

}  // namespace pt::astro

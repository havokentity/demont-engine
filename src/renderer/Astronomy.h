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

// --- Planetarium (issue #281) --------------------------------------------
//
// The eight major planets, positioned from the JPL "approximate positions"
// Keplerian element set and photometered from the Mallama et al. reference
// magnitudes. Earth is excluded from the render list for the obvious reason;
// its heliocentric position is still needed (and exposed) because every
// geocentric planet vector is planet_helio - earth_helio.
//
// ACCURACY TIER. This module deliberately matches what the Sun (Naval
// Almanac, ~1') and Moon (Meeus ch. 47 simplified, ~10') already ship.
// JPL states the following worst-case errors for the 1800-2050 element set
// over that whole interval (approx_pos.html, "accuracy" table, ecliptic
// longitude / latitude in arcseconds):
//
//     Mercury 15" / 1"     Jupiter 400" / 10"
//     Venus   20" / 1"     Saturn  600" / 25"
//     E-M Bary 20" / 8"    Uranus   50" / 2"
//     Mars    40" / 2"     Neptune  10" / 1"
//
// So the inner planets land inside the Sun's ~1' tier and the two gas
// giants inside the Moon's ~10' tier. We do NOT claim better. Specifically
// omitted, all individually below the tier:
//   * planetary aberration from the observer's velocity (<= 20.5"),
//   * the Earth / Earth-Moon-barycentre offset (<= 4700 km, ~6" at 1 au),
//   * precession from J2000 to the equinox of date (see the frame note
//     below -- deliberate, not an oversight),
//   * topocentric parallax (<= 35" for Venus at closest approach; the
//     engine has no geocentric-to-topocentric step for the Moon either,
//     where the effect is a full degree).
// Light-time is NOT omitted -- it is applied, because it is three lines.
//
// FRAME. planetPosition() returns astrometric J2000 (ICRF-aligned) RA/Dec,
// exactly the frame the Bright Star Catalog is stored in, so a planet and a
// star with the same catalogue coordinates land on the same pixel. That is
// the property issue #281 actually asks for ("planets appear at their
// correct positions RELATIVE TO the Bright Star Catalog"). The hour-angle
// step in equatorialToHorizon() treats GMST as if it were the J2000
// equinox's hour angle, which is the same simplification worldToJ2000Matrix
// already documents; planets and stars inherit that offset identically, so
// it cancels in the planet-vs-star comparison and only shows up as the
// starmap's existing shared ~0.4 deg absolute-azimuth error.
enum class Planet : int {
    Mercury = 0,
    Venus   = 1,
    Mars    = 2,
    Jupiter = 3,
    Saturn  = 4,
    Uranus  = 5,
    Neptune = 6,
};
// Number of rendered planets (Earth excluded). The shader-side push
// constant sizes its arrays from this, so it is a hard contract with
// StarsComposite.slang -- see the PT_PLANET_SLOTS comment there.
constexpr int kPlanetCount = 7;

// Human-readable name, lowercase ("mercury", "venus", ...). Used by the
// dump_planet_pos console command; stable identifiers, not localised.
const char* planetName(Planet p);

// Johnson-Cousins B-V colour index. Derived as (B - V) from the
// "Reference" rows of Table 3 in Mallama, Krobusek & Pavlov (2017),
// "Comprehensive wide-band magnitudes and albedos for the planets, with
// applications to exo-planets and Planet Nine", Icarus 282, 19-33
// (preprint arXiv:1609.05048). Mars at +1.36 is the reddest; Neptune at
// +0.39 the bluest -- which is exactly how they look through a telescope.
double planetBvColorIndex(Planet p);

// Everything the renderer needs about one planet at one instant.
struct PlanetPos {
    EquatorialPos eq;              // geocentric astrometric J2000 RA/Dec
    double geocentric_dist_au;     // delta -- Earth to planet
    double heliocentric_dist_au;   // r     -- Sun to planet
    double phase_angle_deg;        // alpha -- Sun-planet-Earth angle
    double elongation_deg;         // Sun-Earth-planet angle (0 = at the sun)
    double vmag;                   // apparent visual magnitude
};

// Full solution for one planet at the given Julian date: Kepler propagation
// of the JPL element set, light-time-corrected geocentric vector, and the
// Mallama et al. V-band photometry. See Astronomy.cpp for the element table
// and every constant's citation.
PlanetPos planetPosition(Planet p, double jd);

// Heliocentric J2000-ecliptic rectangular coordinates in AU. Exposed
// separately so tests can pin the orbital mechanics (semi-major axis,
// eccentricity envelope, sidereal period) without going through the
// geocentric / photometric layers on top.
void planetHeliocentricEcliptic(Planet p, double jd, double out_xyz_au[3]);

// Same, for the Earth-Moon barycentre (the JPL table's "EM Bary" row).
// The barycentre stands in for Earth throughout: the offset is at most
// ~4700 km = 3.1e-5 au, worth ~6 arcseconds on a body at 1 au, an order
// of magnitude below the tier this module targets.
void earthHeliocentricEcliptic(double jd, double out_xyz_au[3]);

// --- end Planetarium ------------------------------------------------------

}  // namespace pt::astro

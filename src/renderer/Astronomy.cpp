// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Astronomy.h"

#include <algorithm>
#include <cmath>

namespace pt::astro {

namespace {
constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;

double normDeg(double d) {
    d = std::fmod(d, 360.0);
    if (d < 0.0) d += 360.0;
    return d;
}
}  // namespace

double julianDateFromUtc(int year, int month, int day,
                         int hour, int minute, double second) {
    // Meeus, "Astronomical Algorithms", Ch. 7. Gregorian calendar
    // (year >= 1582-10-15) — we use it unconditionally; close enough
    // for any time the user is going to type.
    int Y = year, M = month;
    if (M <= 2) { Y -= 1; M += 12; }
    int A = Y / 100;
    int B = 2 - A + A / 4;
    double dayFrac = double(day) +
                     (double(hour) + double(minute) / 60.0 + second / 3600.0) / 24.0;
    return std::floor(365.25 * (Y + 4716))
         + std::floor(30.6001 * (M + 1))
         + dayFrac + B - 1524.5;
}

double julianDateFromTimeT(std::time_t t) {
    std::tm gm = *std::gmtime(&t);
    return julianDateFromUtc(gm.tm_year + 1900, gm.tm_mon + 1, gm.tm_mday,
                             gm.tm_hour, gm.tm_min, double(gm.tm_sec));
}

double gmstDegrees(double jd) {
    // IAU 1982 model, accurate to a few arcsec for our timeframe.
    double T = (jd - 2451545.0) / 36525.0;
    double gmst_sec = 67310.54841
                    + (876600.0 * 3600.0 + 8640184.812866) * T
                    + 0.093104 * T * T
                    - 6.2e-6  * T * T * T;
    double gmst_deg = gmst_sec * (360.0 / 86400.0);
    return normDeg(gmst_deg);
}

EquatorialPos sunPosition(double jd) {
    // Naval Almanac low-precision formulas (~1 arcmin accuracy).
    double n = jd - 2451545.0;
    double L = normDeg(280.460 + 0.9856474 * n);     // mean longitude
    double g = normDeg(357.528 + 0.9856003 * n);     // mean anomaly
    double g_r = g * kDeg2Rad;
    double lambda = L + 1.915 * std::sin(g_r) + 0.020 * std::sin(2.0 * g_r);
    double epsilon = 23.439 - 0.0000004 * n;          // obliquity
    double l_r = lambda * kDeg2Rad;
    double e_r = epsilon * kDeg2Rad;

    double ra  = std::atan2(std::cos(e_r) * std::sin(l_r), std::cos(l_r)) * kRad2Deg;
    double dec = std::asin(std::sin(e_r) * std::sin(l_r)) * kRad2Deg;
    return EquatorialPos{ normDeg(ra), dec };
}

EquatorialPos moonPosition(double jd) {
    // Meeus chapter 47 simplified, keeping the largest periodic terms.
    // Accurate to ~10 arcmin for ra/dec, plenty for "moon is roughly
    // there in the sky." For sub-arcminute precision we'd need the
    // full ELP-2000/82 series (~70 longitude terms, ~30 latitude).
    double T = (jd - 2451545.0) / 36525.0;            // Julian centuries

    // Fundamental arguments, all in degrees.
    double L  = 218.3164591 + 481267.88134236 * T;    // moon mean longitude
    double D  = 297.8502042 + 445267.1115168 * T;     // mean elongation
    double M  = 357.5291092 + 35999.0502909 * T;      // sun mean anomaly
    double Mm = 134.9634114 + 477198.8676313 * T;     // moon mean anomaly
    double F  =  93.2720993 + 483202.0175273 * T;     // argument of latitude

    L  = normDeg(L);  D = normDeg(D);  M = normDeg(M);
    Mm = normDeg(Mm); F = normDeg(F);

    double Mr  = M * kDeg2Rad;
    double Mmr = Mm * kDeg2Rad;
    double Dr  = D * kDeg2Rad;
    double Fr  = F * kDeg2Rad;

    // Six dominant longitude terms (deg).
    double dL =  6.289 * std::sin(Mmr)
              - 1.274 * std::sin(Mmr - 2.0 * Dr)
              + 0.658 * std::sin(2.0 * Dr)
              - 0.186 * std::sin(Mr)
              - 0.114 * std::sin(2.0 * Fr)
              + 0.059 * std::sin(2.0 * Mmr - 2.0 * Dr);

    // Three dominant latitude terms (deg).
    double dB =  5.128 * std::sin(Fr)
              + 0.281 * std::sin(Mmr + Fr)
              + 0.278 * std::sin(Mmr - Fr)
              - 0.173 * std::sin(2.0 * Dr - Fr);

    double lambda = L + dL;                    // ecliptic longitude
    double beta   = dB;                        // ecliptic latitude
    double epsilon = 23.439 - 0.0000004 * (jd - 2451545.0);

    double l_r = lambda  * kDeg2Rad;
    double b_r = beta    * kDeg2Rad;
    double e_r = epsilon * kDeg2Rad;

    // Ecliptic -> equatorial.
    double sin_a = std::sin(l_r) * std::cos(e_r) - std::tan(b_r) * std::sin(e_r);
    double cos_a = std::cos(l_r);
    double ra    = std::atan2(sin_a, cos_a) * kRad2Deg;
    double dec   = std::asin(std::sin(b_r) * std::cos(e_r)
                           + std::cos(b_r) * std::sin(e_r) * std::sin(l_r)) * kRad2Deg;
    return EquatorialPos{ normDeg(ra), dec };
}

double moonDistanceKm(double jd) {
    // Meeus chapter 47, distance terms (km). Uses the same fundamental
    // arguments as moonPosition; extracts the dominant 6 cosine terms
    // of the radial-distance series.
    double T = (jd - 2451545.0) / 36525.0;
    double D  = normDeg(297.8502042 + 445267.1115168 * T) * kDeg2Rad;
    double M  = normDeg(357.5291092 +  35999.0502909 * T) * kDeg2Rad;
    double Mm = normDeg(134.9634114 + 477198.8676313 * T) * kDeg2Rad;

    double dist = 385000.56
                + (-20905.355) * std::cos(Mm)
                + ( -3699.111) * std::cos(2.0 * D - Mm)
                + ( -2955.968) * std::cos(2.0 * D)
                + (  -569.925) * std::cos(2.0 * Mm)
                + (    48.888) * std::cos(M)
                + (    -3.149) * std::cos(2.0 * (Mm - D));   // small terms
    return dist;
}

double sunDistanceAu(double jd) {
    // Naval Almanac low-precision: r = 1.00014 - 0.01671 cos(g)
    //                                       - 0.00014 cos(2g)
    // where g is the sun's mean anomaly. Accurate to ~5e-5 AU.
    double n = jd - 2451545.0;
    double g = normDeg(357.528 + 0.9856003 * n) * kDeg2Rad;
    return 1.00014 - 0.01671 * std::cos(g) - 0.00014 * std::cos(2.0 * g);
}

double moonPhaseAngle(EquatorialPos sun, EquatorialPos moon) {
    // Phase = angular separation between the moon and the sun as seen
    // from earth. New moon: sun and moon at the same place in the sky
    // (separation = 0). Full moon: opposite (separation = pi). The
    // shader expects 0 = new (no light), pi = full (max light).
    //
    // Earlier this returned acos(-cos_sep), which inverted the
    // convention -- new moon read as full-strength and vice versa.
    // That's the "no shadows on full moon, max shadows on new moon"
    // bug the user spotted.
    double s_ra = sun.ra_deg * kDeg2Rad;
    double s_dec = sun.dec_deg * kDeg2Rad;
    double m_ra = moon.ra_deg * kDeg2Rad;
    double m_dec = moon.dec_deg * kDeg2Rad;
    double cos_sep = std::sin(s_dec) * std::sin(m_dec)
                   + std::cos(s_dec) * std::cos(m_dec)
                       * std::cos(s_ra - m_ra);
    cos_sep = std::clamp(cos_sep, -1.0, 1.0);
    return std::acos(cos_sep);     // 0 at new, pi at full
}

HorizonPos equatorialToHorizon(EquatorialPos eq,
                               double observer_lat_deg,
                               double observer_lon_deg,
                               double jd) {
    double gmst = gmstDegrees(jd);
    double lst  = normDeg(gmst + observer_lon_deg);   // local sidereal time
    double H    = normDeg(lst - eq.ra_deg);            // hour angle, deg
    double H_r  = H * kDeg2Rad;
    double dec_r = eq.dec_deg * kDeg2Rad;
    double lat_r = observer_lat_deg * kDeg2Rad;

    double sin_alt = std::sin(dec_r) * std::sin(lat_r)
                   + std::cos(dec_r) * std::cos(lat_r) * std::cos(H_r);
    double alt_r = std::asin(std::clamp(sin_alt, -1.0, 1.0));

    double cos_az = (std::sin(dec_r) - std::sin(alt_r) * std::sin(lat_r))
                  / (std::cos(alt_r) * std::cos(lat_r) + 1e-12);
    cos_az = std::clamp(cos_az, -1.0, 1.0);
    double az_r = std::acos(cos_az);
    // Resolve quadrant: if the hour angle is positive (sun in the
    // western sky after meridian transit), azimuth is 360 - az.
    double az_deg = az_r * kRad2Deg;
    if (std::sin(H_r) > 0.0) az_deg = 360.0 - az_deg;

    return HorizonPos{ normDeg(az_deg), alt_r * kRad2Deg };
}

void worldToJ2000Matrix(double observer_lat_deg,
                        double observer_lon_deg,
                        double jd,
                        float  out[9]) {
    // World frame in this engine: +X east, +Y up, +Z south.
    // ENU (east, north, up) and J2000 (x toward vernal equinox, z toward
    // celestial north pole) relate via three rotations parameterised by
    // observer lat (phi) and Local Sidereal Time (theta = GMST + lon).
    //
    // Composing { J2000 -> ENU -> world } and inverting gives the matrix
    // below. See the derivation in the design notes; spot-check: the
    // zenith vector (0,1,0) -> (cos phi*cos theta, cos phi*sin theta,
    // sin phi), which has RA=theta=LST and Dec=phi (lat) -- the local
    // zenith's celestial coordinates. Looking due north on the horizon
    // (0,0,-1) at the equator (phi=0) yields (0,0,1), the north
    // celestial pole, as it should be on the equator's north horizon.
    const double theta = (gmstDegrees(jd) + observer_lon_deg) * kDeg2Rad;
    const double phi   = observer_lat_deg * kDeg2Rad;
    const double ct = std::cos(theta), st = std::sin(theta);
    const double cp = std::cos(phi),   sp = std::sin(phi);

    // Row 0
    out[0] = float(-st);
    out[1] = float( cp * ct);
    out[2] = float( sp * ct);
    // Row 1
    out[3] = float( ct);
    out[4] = float( cp * st);
    out[5] = float( sp * st);
    // Row 2
    out[6] = 0.0f;
    out[7] = float( sp);
    out[8] = float(-cp);
}

}  // namespace pt::astro

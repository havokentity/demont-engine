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

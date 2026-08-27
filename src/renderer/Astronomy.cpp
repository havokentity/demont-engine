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

// --- Planetarium (issue #281) --------------------------------------------

namespace {

// JPL approximate Keplerian elements and their per-Julian-century rates,
// valid 1800 AD - 2050 AD.
//
// SOURCE (verbatim, Table 1):
//   E. M. Standish, "Keplerian Elements for Approximate Positions of the
//   Major Planets", JPL Solar System Dynamics,
//   https://ssd.jpl.nasa.gov/planets/approx_pos.html
//   (companion of the Explanatory Supplement to the Astronomical Almanac,
//   3rd ed., ch. 8). Elements are referred to the mean ecliptic and
//   equinox of J2000.
//
// Column meaning, in the order stored below:
//   a      semi-major axis                        au,  au/century
//   e      eccentricity                           --,  1/century
//   inc    inclination                            deg, deg/century
//   L      mean longitude                         deg, deg/century
//   peri   longitude of perihelion (varpi)        deg, deg/century
//   node   longitude of the ascending node        deg, deg/century
//
// The published table's `e` rate column is headed "rad/Cy"; that is a known
// typo in the source page -- eccentricity is dimensionless and the rate is
// per century, which is how it is used in the accompanying algorithm.
//
// Table 2 (the 3000 BC - 3000 AD variant) adds b/c/s/f correction terms to
// the mean longitude of Jupiter..Neptune. We deliberately do NOT carry
// them: they are identically zero over 1800-2050, which is the interval
// this engine's date cvars (r_sky_year documents 1900..2100) actually
// exercise, and carrying an unused correction would be dead weight.
struct KeplerElements {
    double a,    a_dot;      // au,  au/century
    double e,    e_dot;      // --,  1/century
    double inc,  inc_dot;    // deg, deg/century
    double L,    L_dot;      // deg, deg/century
    double peri, peri_dot;   // deg, deg/century
    double node, node_dot;   // deg, deg/century
};

// Index order matches pt::astro::Planet. kEmBaryElements is stored
// separately because Earth is never rendered, only differenced against.
constexpr KeplerElements kPlanetElements[kPlanetCount] = {
    // Mercury
    {  0.38709927,  0.00000037,
       0.20563593,  0.00001906,
       7.00497902, -0.00594749,
     252.25032350, 149472.67411175,
      77.45779628,  0.16047689,
      48.33076593, -0.12534081 },
    // Venus
    {  0.72333566,  0.00000390,
       0.00677672, -0.00004107,
       3.39467605, -0.00078890,
     181.97909950, 58517.81538729,
     131.60246718,  0.00268329,
      76.67984255, -0.27769418 },
    // Mars
    {  1.52371034,  0.00001847,
       0.09339410,  0.00007882,
       1.84969142, -0.00813131,
      -4.55343205, 19140.30268499,
     -23.94362959,  0.44441088,
      49.55953891, -0.29257343 },
    // Jupiter
    {  5.20288700, -0.00011607,
       0.04838624, -0.00013253,
       1.30439695, -0.00183714,
      34.39644051, 3034.74612775,
      14.72847983,  0.21252668,
     100.47390909,  0.20469106 },
    // Saturn
    {  9.53667594, -0.00125060,
       0.05386179, -0.00050991,
       2.48599187,  0.00193609,
      49.95424423, 1222.49362201,
      92.59887831, -0.41897216,
     113.66242448, -0.28867794 },
    // Uranus
    { 19.18916464, -0.00196176,
       0.04725744, -0.00004397,
       0.77263783, -0.00242939,
     313.23810451, 428.48202785,
     170.95427630,  0.40805281,
      74.01692503,  0.04240589 },
    // Neptune
    { 30.06992276,  0.00026291,
       0.00859048,  0.00005105,
       1.77004347,  0.00035372,
     -55.12002969, 218.45945325,
      44.96476227, -0.32241464,
     131.78422574, -0.00508664 },
};

// "EM Bary" row of the same table -- the Earth-Moon barycentre.
constexpr KeplerElements kEmBaryElements = {
      1.00000261,  0.00000562,
      0.01671123, -0.00004392,
     -0.00001531, -0.01294668,
    100.46457166, 35999.37244981,
    102.93768193,  0.32327364,
      0.0,         0.0,
};

// Obliquity of the ecliptic at J2000. IAU 2006 value 84381.406" =
// 23.439279444...deg (Capitaine, Wallace & Chapront 2003, adopted by IAU
// 2006 Resolution B1). The JPL approx_pos page uses 23.43928 deg for the
// same rotation; identical to 7 significant figures.
constexpr double kObliquityJ2000Deg = 84381.406 / 3600.0;

// Light travel time across one astronomical unit, in days.
// The IAU 2012 definition fixes 1 au = 149597870700 m exactly, and the SI
// second fixes c = 299792458 m/s exactly, so this is an exact ratio:
//   149597870700 / 299792458 = 499.0047838061... s = 0.005775518331... day
constexpr double kLightTimePerAuDays = 499.00478380614 / 86400.0;

// Saturn's north pole of rotation, ICRF/J2000, from the IAU Working Group
// on Cartographic Coordinates and Rotational Elements:
//   Archinal, B. A., et al. (2018), "Report of the IAU Working Group on
//   Cartographic Coordinates and Rotational Elements: 2015", Celestial
//   Mechanics and Dynamical Astronomy 130:22, Table 2:
//       alpha0 = 40.589 - 0.036 T   deg
//       delta0 = 83.537 - 0.004 T   deg     (T in Julian centuries TDB)
// Needed only to compute the ring-opening angle that dominates Saturn's
// apparent magnitude (a ~1.1 mag swing between edge-on and fully open).
constexpr double kSaturnPoleRaDeg      = 40.589;
constexpr double kSaturnPoleRaRateDeg  = -0.036;
constexpr double kSaturnPoleDecDeg     = 83.537;
constexpr double kSaturnPoleDecRateDeg = -0.004;

// Johnson-Cousins B-V, = (B - V) of the "Reference" rows of Table 3 in
// Mallama, Krobusek & Pavlov (2017), Icarus 282, 19-33:
//   Mercury  0.28 - (-0.69) = 0.97      Saturn  -7.84 - (-8.91) = 1.07
//   Venus   -3.68 - (-4.38) = 0.70      Uranus  -6.61 - (-7.11) = 0.50
//   Mars    -0.24 - (-1.60) = 1.36      Neptune -6.55 - (-6.94) = 0.39
//   Jupiter -8.54 - (-9.40) = 0.86
constexpr double kPlanetBv[kPlanetCount] = {
    0.97, 0.70, 1.36, 0.86, 1.07, 0.50, 0.39
};

const char* const kPlanetNames[kPlanetCount] = {
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"
};

// Wrap to (-180, +180]. The JPL algorithm's step 4 requires the mean
// anomaly in this range before the Kepler iteration, otherwise the
// Newton step starts far from the root for the outer planets whose
// L_dot * T has accumulated many revolutions.
double wrap180(double d) {
    d = std::fmod(d + 180.0, 360.0);
    if (d < 0.0) d += 360.0;
    return d - 180.0;
}

// Solve Kepler's equation M = E - e* sin(E) with e* = (180/pi) e, i.e. the
// degree-argument form the JPL page uses so M and E stay in degrees.
//
// The published recipe is a fixed tolerance of 1e-6 deg reached by the
// Newton correction dE = (M - (E - e* sin E)) / (1 - e cos E). We keep the
// same iteration but cap it: Mercury's e = 0.2056 converges in 4 steps,
// and the cap only exists so a future caller feeding a hyperbolic-ish e
// cannot spin forever.
double solveKeplerDeg(double M_deg, double e) {
    const double e_star = kRad2Deg * e;          // "e*" in the JPL text
    double E = M_deg + e_star * std::sin(M_deg * kDeg2Rad);
    for (int i = 0; i < 32; ++i) {
        const double E_r = E * kDeg2Rad;
        const double dM  = M_deg - (E - e_star * std::sin(E_r));
        const double dE  = dM / (1.0 - e * std::cos(E_r));
        E += dE;
        if (std::abs(dE) <= 1e-9) break;         // 1e-9 deg = 3.6 micro-arcsec
    }
    return E;
}

// Steps 1-5 of the JPL recipe: propagate the elements to `T` Julian
// centuries past J2000 and produce heliocentric J2000-ecliptic rectangular
// coordinates in AU.
void keplerToEclipticAu(const KeplerElements& k, double T, double out[3]) {
    const double a    = k.a    + k.a_dot    * T;
    const double e    = k.e    + k.e_dot    * T;
    const double inc  = k.inc  + k.inc_dot  * T;
    const double L    = k.L    + k.L_dot    * T;
    const double peri = k.peri + k.peri_dot * T;
    const double node = k.node + k.node_dot * T;

    // Step 2: argument of perihelion and mean anomaly.
    const double omega = peri - node;            // small omega
    const double M     = wrap180(L - peri);

    // Step 3: eccentric anomaly.
    const double E_r = solveKeplerDeg(M, e) * kDeg2Rad;

    // Step 4: heliocentric coordinates in the orbital plane, x' toward
    // perihelion.
    const double xp = a * (std::cos(E_r) - e);
    const double yp = a * std::sqrt(std::max(0.0, 1.0 - e * e)) * std::sin(E_r);

    // Step 5: rotate the orbital plane into the J2000 ecliptic.
    const double co = std::cos(omega * kDeg2Rad), so = std::sin(omega * kDeg2Rad);
    const double cn = std::cos(node  * kDeg2Rad), sn = std::sin(node  * kDeg2Rad);
    const double ci = std::cos(inc   * kDeg2Rad), si = std::sin(inc   * kDeg2Rad);

    out[0] = ( co * cn - so * sn * ci) * xp + (-so * cn - co * sn * ci) * yp;
    out[1] = ( co * sn + so * cn * ci) * xp + (-so * sn + co * cn * ci) * yp;
    out[2] = ( so * si)                * xp + ( co * si)                * yp;
}

double julianCenturies(double jd) { return (jd - 2451545.0) / 36525.0; }

// Saturnicentric latitude of the Earth referred to the ring plane, in
// degrees -- the "ring opening angle" B. Positive = the northern face of
// the rings is presented. Geometric (planetocentric): the rings are a flat
// disc in Saturn's equatorial plane, so the oblateness correction that
// turns this into the planetographic sub-observer latitude HORIZONS
// reports does not apply to the ring geometry.
//
// sin B = -(pole_hat . earth_to_saturn_hat): the dot product is taken
// against the Saturn->Earth direction, hence the sign flip.
double saturnRingOpeningDeg(double jd, const EquatorialPos& saturn_eq) {
    const double T   = julianCenturies(jd);
    const double a0  = (kSaturnPoleRaDeg  + kSaturnPoleRaRateDeg  * T) * kDeg2Rad;
    const double d0  = (kSaturnPoleDecDeg + kSaturnPoleDecRateDeg * T) * kDeg2Rad;
    const double ra  = saturn_eq.ra_deg  * kDeg2Rad;
    const double dec = saturn_eq.dec_deg * kDeg2Rad;
    const double dot = std::cos(d0) * std::cos(dec) * std::cos(a0 - ra)
                     + std::sin(d0) * std::sin(dec);
    return std::asin(std::clamp(-dot, -1.0, 1.0)) * kRad2Deg;
}

// Apparent V magnitude.
//
// SOURCE for every coefficient below: Mallama, Krobusek & Pavlov (2017),
// "Comprehensive wide-band magnitudes and albedos for the planets, with
// applications to exo-planets and Planet Nine", Icarus 282, 19-33
// (arXiv:1609.05048), appendix tables A-1.2 (Mercury), A-2.2 (Venus),
// A-4.2 (Mars), A-5.2 (Jupiter), A-6.2 (Saturn) and A-7.2 (Uranus),
// V-band column throughout; Neptune from the Table 3 "Reference" V
// magnitude (the paper gives no illumination phase function for Neptune
// because its phase angle never exceeds ~2 deg).
//
// The common form is the Harris (1961) polynomial the paper's Equation 2
// restates:  V = M1(alpha) + 5 log10(r * delta), with M1(alpha) = C0 +
// C1 alpha + C2 alpha^2 + ...  Saturn (Eq. A-6.1) and Uranus (Eq. A-7.1)
// carry extra geometric terms, handled inline.
//
// Deliberately omitted, each documented with its cost:
//   * Mars rotational + orbital-longitude terms (Tables A-4.4 / A-4.6):
//     +-0.05 mag in V.
//   * Uranus sub-latitude term C1 = -0.00084 per degree (Table A-7.2):
//     <= 0.07 mag across the planet's whole 84-year cycle.
//   * Saturn's C1/C3 ring terms ARE included -- they are worth over a
//     magnitude and are the single largest photometric effect in the set.
double planetVmag(Planet p, double jd,
                  const EquatorialPos& eq,
                  double r_au, double delta_au, double alpha_deg) {
    const double dist_term = 5.0 * std::log10(std::max(r_au * delta_au, 1e-12));
    const double a1 = alpha_deg;
    const double a2 = a1 * a1;
    const double a3 = a2 * a1;
    const double a4 = a2 * a2;

    switch (p) {
    case Planet::Mercury: {
        // Table A-1.2, 7th-order in alpha, fitted over 2..170 deg.
        const double a5 = a4 * a1, a6 = a5 * a1, a7 = a6 * a1;
        return -0.694
             +  6.617e-2  * a1
             -  1.867e-3  * a2
             +  4.103e-5  * a3
             -  4.583e-7  * a4
             +  2.643e-9  * a5
             -  7.012e-12 * a6
             +  6.592e-15 * a7
             + dist_term;
    }
    case Planet::Venus:
        // Table A-2.2, V column. Fitted to 163.7 deg; past that the
        // forward-scattering crescent needs a separate branch we do not
        // model (Venus is inside 10 deg of the sun there anyway, so the
        // renderer's daylight fade has already hidden it).
        return -4.384
             -  1.044e-3 * a1
             +  3.687e-4 * a2
             -  2.814e-6 * a3
             +  8.938e-9 * a4
             + dist_term;
    case Planet::Mars:
        // Table A-4.2, V column (illumination term only).
        return -1.601 + 0.02267 * a1 - 0.0001302 * a2 + dist_term;
    case Planet::Jupiter:
        // Table A-5.2, V column.
        return -9.395 - 0.00037 * a1 + 0.000616 * a2 + dist_term;
    case Planet::Saturn: {
        // Equation A-6.1 with the Table A-6.2 V column:
        //   M1 = C0 + C1 sin(beta) + C2 alpha - C3 sin(beta) exp(C4 alpha)
        // beta is the ring inclination, taken as |B| so that presenting
        // either ring face brightens the planet (C1 is negative).
        const double beta   = std::abs(saturnRingOpeningDeg(jd, eq));
        const double sin_b  = std::sin(beta * kDeg2Rad);
        return -8.914
             -  1.825 * sin_b
             +  0.026 * a1
             -  0.378 * sin_b * std::exp(-2.25 * a1)
             + dist_term;
    }
    case Planet::Uranus:
        // Table A-7.2, V column, C0 only (see the omission note above).
        return -7.110 + dist_term;
    case Planet::Neptune:
        // Table 3 "Reference" V.
        return -6.94 + dist_term;
    }
    return 0.0;   // unreachable; enum is exhaustive
}

}  // namespace

const char* planetName(Planet p) {
    const int i = static_cast<int>(p);
    if (i < 0 || i >= kPlanetCount) return "?";
    return kPlanetNames[i];
}

double planetBvColorIndex(Planet p) {
    const int i = static_cast<int>(p);
    if (i < 0 || i >= kPlanetCount) return 0.65;   // solar B-V fallback
    return kPlanetBv[i];
}

void planetHeliocentricEcliptic(Planet p, double jd, double out_xyz_au[3]) {
    const int i = static_cast<int>(p);
    if (i < 0 || i >= kPlanetCount) {
        out_xyz_au[0] = out_xyz_au[1] = out_xyz_au[2] = 0.0;
        return;
    }
    keplerToEclipticAu(kPlanetElements[i], julianCenturies(jd), out_xyz_au);
}

void earthHeliocentricEcliptic(double jd, double out_xyz_au[3]) {
    keplerToEclipticAu(kEmBaryElements, julianCenturies(jd), out_xyz_au);
}

PlanetPos planetPosition(Planet p, double jd) {
    PlanetPos out{};

    double earth[3];
    earthHeliocentricEcliptic(jd, earth);

    // Light-time correction. The planet we see now is where it was
    // delta/c ago; iterate the (position -> distance -> retarded epoch)
    // loop until the epoch stops moving. Three passes converge to well
    // under a millisecond of light time for every body in the table --
    // Neptune's 4.2-hour light time changes its position by ~7", and the
    // second pass changes that by ~1e-4 of itself.
    double planet[3];
    planetHeliocentricEcliptic(p, jd, planet);
    double delta = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        const double dx = planet[0] - earth[0];
        const double dy = planet[1] - earth[1];
        const double dz = planet[2] - earth[2];
        delta = std::sqrt(dx * dx + dy * dy + dz * dz);
        planetHeliocentricEcliptic(p, jd - delta * kLightTimePerAuDays, planet);
    }

    const double gx = planet[0] - earth[0];
    const double gy = planet[1] - earth[1];
    const double gz = planet[2] - earth[2];
    delta = std::sqrt(gx * gx + gy * gy + gz * gz);

    const double r_helio = std::sqrt(planet[0] * planet[0]
                                   + planet[1] * planet[1]
                                   + planet[2] * planet[2]);
    const double r_earth = std::sqrt(earth[0] * earth[0]
                                   + earth[1] * earth[1]
                                   + earth[2] * earth[2]);

    // Ecliptic -> equatorial, a single rotation about the +X (equinox)
    // axis by the obliquity. Same chain sunPosition / moonPosition use,
    // just in rectangular form because we already have a 3-vector.
    const double eps = kObliquityJ2000Deg * kDeg2Rad;
    const double ce = std::cos(eps), se = std::sin(eps);
    const double ex = gx;
    const double ey = ce * gy - se * gz;
    const double ez = se * gy + ce * gz;

    out.eq.ra_deg  = normDeg(std::atan2(ey, ex) * kRad2Deg);
    out.eq.dec_deg = std::asin(std::clamp(ez / std::max(delta, 1e-12), -1.0, 1.0))
                   * kRad2Deg;
    out.geocentric_dist_au   = delta;
    out.heliocentric_dist_au = r_helio;

    // Phase angle: the Sun-planet-Earth angle, from the triangle whose
    // sides are r (sun-planet), delta (planet-earth) and R (sun-earth).
    const double cos_alpha = (r_helio * r_helio + delta * delta - r_earth * r_earth)
                           / std::max(2.0 * r_helio * delta, 1e-12);
    out.phase_angle_deg = std::acos(std::clamp(cos_alpha, -1.0, 1.0)) * kRad2Deg;

    // Elongation: the Sun-Earth-planet angle, the one an observer sees as
    // "how far from the sun in the sky".
    const double cos_elong = (r_earth * r_earth + delta * delta - r_helio * r_helio)
                           / std::max(2.0 * r_earth * delta, 1e-12);
    out.elongation_deg = std::acos(std::clamp(cos_elong, -1.0, 1.0)) * kRad2Deg;

    out.vmag = planetVmag(p, jd, out.eq, r_helio, delta, out.phase_angle_deg);
    return out;
}

}  // namespace pt::astro

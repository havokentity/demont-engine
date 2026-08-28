// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for the physical atmosphere model added in planetary P3
// (issue #257): a body-parameterised medium with three species, and the
// ray-versus-spherical-shell intersection that replaced the cloud layer's
// flat slab.
//
// WHAT THIS FILE IS FOR
//
// Two of P3's acceptance criteria are visual claims -- "the twilight
// zenith reads blue, not brown" and "distant clouds merge with the
// horizon at a ground camera".  A golden PNG can show that something
// changed; it cannot show WHY, and it cannot distinguish "the right term
// was added" from "some term was added".  Both claims are actually
// statements about numbers, and both are pinned here as numbers:
//
//   * the blue claim is that ozone's Chappuis band removes red-orange
//     along the long twilight slant path, so the surviving light is
//     BLUER with ozone than without -- a per-channel transmittance
//     ratio, computed and asserted below;
//   * the horizon claim is that a horizontal ray from eye height enters
//     a cloud layer above it at a FINITE range on a sphere, where on a
//     slab that range is infinite -- a closed-form chord length,
//     computed and asserted below.
//
// WHY IT MIRRORS THE SHADER
//
// The model lives in shaders/PathTraceMath.slang and there is no host
// entry point to call, so this file transcribes it -- same operations,
// same order -- exactly as tests/pt_math_sphere_test.cpp and
// tests/pt_math_altitude_test.cpp do for their kernels.  A mirror that
// has drifted is worthless, so the last case re-reads the .slang and
// pins what the transcription depends on.  It COUNTS occurrences rather
// than testing find() != npos: a substring pin is satisfied by one
// correct copy however many wrong ones exist elsewhere, which is exactly
// how issue #276 stayed live for a whole cycle underneath a passing
// test.
//
// FAST MATH IS PART OF THE CONTRACT
//
// Metal compiles these modules with its default (fast) math mode, so
// this target is built with -ffast-math (see tests/CMakeLists.txt) and
// the host mirror faces the same reassociation the GPU applies.  Note
// the trap recorded in #275: -ffast-math implies __FINITE_MATH_ONLY__,
// under which Apple clang folds std::isfinite() AND a plain exponent-bit
// test to constant true.  finiteBits() below routes through a volatile
// load, which carries no fast-math flags, and isFiniteHarnessWorks()
// asserts that this is still true on whatever compiler is running -- so
// the harness cannot go quietly vacuous.
//
// Deterministic: every input is a literal or derived from literals.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {

#if defined(_MSC_VER)
#  define PT_MIRROR __declspec(noinline)
#else
#  define PT_MIRROR __attribute__((noinline))
#endif

// See the header note: std::isfinite() and a plain bit test both fold to
// `true` under -ffast-math.  The volatile round-trip breaks the chain.
inline bool finiteBits(float v) {
    volatile float t = v;
    float u = t;
    std::uint32_t b;
    std::memcpy(&b, &u, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu;
}

struct F3 {
    float x, y, z;
};
inline F3 operator+(F3 a, F3 b) { return F3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline F3 operator-(F3 a, F3 b) { return F3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline F3 operator*(F3 a, float s) { return F3{a.x * s, a.y * s, a.z * s}; }
inline F3 operator*(float s, F3 a) { return a * s; }
inline float dot(F3 a, F3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float len(F3 a) { return std::sqrt(dot(a, a)); }
inline F3 norm(F3 a) { float l = len(a); return F3{a.x / l, a.y / l, a.z / l}; }

// --- shader mirror: PtAtmoBody / ptAtmoEarth ------------------------------
// Line-for-line with shaders/PathTraceMath.slang.  Citations live on the
// shader side; the numbers are repeated here so a silent edit to either
// copy fails the pin at the bottom of this file.
struct AtmoBody {
    float ground_radius;
    float top_radius;
    F3    rayleigh_sigma_s;
    float rayleigh_scale_h;
    F3    mie_sigma_s;
    F3    mie_sigma_a;
    float mie_scale_h;
    float mie_g;
    F3    ozone_sigma_a;
    float ozone_center;
    float ozone_half_width;
};

AtmoBody ptAtmoEarth(float ground_radius) {
    AtmoBody b{};
    b.ground_radius    = ground_radius;
    b.top_radius       = ground_radius + 100000.0f;
    b.rayleigh_sigma_s = F3{5.802e-6f, 13.558e-6f, 33.100e-6f};
    b.rayleigh_scale_h = 8000.0f;
    b.mie_sigma_s      = F3{3.996e-6f, 3.996e-6f, 3.996e-6f};
    b.mie_sigma_a      = F3{4.400e-6f, 4.400e-6f, 4.400e-6f};
    b.mie_scale_h      = 1200.0f;
    b.mie_g            = 0.8f;
    b.ozone_sigma_a    = F3{0.650e-6f, 1.881e-6f, 0.085e-6f};
    b.ozone_center     = 25000.0f;
    b.ozone_half_width = 15000.0f;
    return b;
}

AtmoBody ptAtmoScale(AtmoBody b, float rayleigh_scale, float mie_sigma_s_abs,
                     float ozone_scale) {
    b.rayleigh_sigma_s = b.rayleigh_sigma_s * std::max(rayleigh_scale, 0.0f);
    float ss0 = b.mie_sigma_s.y;
    F3 co_albedo = (ss0 > 0.0f)
                 ? F3{b.mie_sigma_a.x / ss0, b.mie_sigma_a.y / ss0,
                      b.mie_sigma_a.z / ss0}
                 : F3{0.0f, 0.0f, 0.0f};
    float ss = std::max(mie_sigma_s_abs, 0.0f);
    b.mie_sigma_s = F3{ss, ss, ss};
    b.mie_sigma_a = co_albedo * ss;
    b.ozone_sigma_a = b.ozone_sigma_a * std::max(ozone_scale, 0.0f);
    return b;
}

PT_MIRROR float ptAtmoOzoneProfile(const AtmoBody& b, float h) {
    return std::max(0.0f, 1.0f - std::fabs(h - b.ozone_center) / b.ozone_half_width);
}

PT_MIRROR void ptAtmoCoefficients(const AtmoBody& b, float h,
                                  F3& sigma_s_rayleigh, F3& sigma_s_mie,
                                  F3& sigma_t) {
    float hc = std::max(h, 0.0f);
    float dr = std::exp(-hc / b.rayleigh_scale_h);
    float dm = std::exp(-hc / b.mie_scale_h);
    float doz = ptAtmoOzoneProfile(b, hc);
    sigma_s_rayleigh = b.rayleigh_sigma_s * dr;
    sigma_s_mie      = b.mie_sigma_s * dm;
    sigma_t = sigma_s_rayleigh
            + (b.mie_sigma_s + b.mie_sigma_a) * dm
            + b.ozone_sigma_a * doz;
}

// Reference optical depth: exact-as-we-can-get altitude, arbitrarily many
// Simpson subintervals.  Deliberately NOT the shader's hoisted quadratic
// -- it is the independent yardstick the 8-step production rule is
// measured against, so it must not share the production code's structure.
double refOpticalDepthChannel(const AtmoBody& b, F3 ro, F3 rd, F3 centre,
                              double t, int channel, int steps) {
    auto sigma_t_at = [&](double s) {
        double px = double(ro.x) + double(rd.x) * s;
        double py = double(ro.y) + double(rd.y) * s;
        double pz = double(ro.z) + double(rd.z) * s;
        double dx = px - centre.x, dy = py - centre.y, dz = pz - centre.z;
        double h = std::sqrt(dx * dx + dy * dy + dz * dz) - double(b.ground_radius);
        h = std::max(h, 0.0);
        double dr = std::exp(-h / double(b.rayleigh_scale_h));
        double dm = std::exp(-h / double(b.mie_scale_h));
        double doz = std::max(0.0, 1.0 - std::fabs(h - double(b.ozone_center))
                                          / double(b.ozone_half_width));
        double sr = (channel == 0 ? b.rayleigh_sigma_s.x
                   : channel == 1 ? b.rayleigh_sigma_s.y
                                  : b.rayleigh_sigma_s.z) * dr;
        double sm = ((channel == 0 ? b.mie_sigma_s.x
                    : channel == 1 ? b.mie_sigma_s.y : b.mie_sigma_s.z)
                   + (channel == 0 ? b.mie_sigma_a.x
                    : channel == 1 ? b.mie_sigma_a.y : b.mie_sigma_a.z)) * dm;
        double so = (channel == 0 ? b.ozone_sigma_a.x
                   : channel == 1 ? b.ozone_sigma_a.y
                                  : b.ozone_sigma_a.z) * doz;
        return sr + sm + so;
    };
    double dt = t / steps;
    double acc = 0.0;
    for (int i = 0; i < steps; ++i) {
        double l = i * dt, m = (i + 0.5) * dt, r = (i + 1.0) * dt;
        acc += dt / 6.0 * (sigma_t_at(l) + 4.0 * sigma_t_at(m) + sigma_t_at(r));
    }
    return acc;
}

// --- shader mirror: the hoisted ray altitude (#271) -----------------------
// Only what ptAtmoOpticalDepth consumes.  The kernel itself is pinned in
// full by tests/pt_math_altitude_test.cpp; this is the caller's view.
struct RayAlt {
    double k0, b0;
    float  rad;
};
RayAlt rayAltBegin(F3 ro, F3 rd, F3 c, float rad) {
    RayAlt a{};
    double ox = double(ro.x) - c.x, oy = double(ro.y) - c.y, oz = double(ro.z) - c.z;
    a.k0 = ox * ox + oy * oy + oz * oz - double(rad) * double(rad);
    a.b0 = ox * double(rd.x) + oy * double(rd.y) + oz * double(rd.z);
    a.rad = rad;
    return a;
}
double rayAltAt(const RayAlt& a, double s) {
    double k = s * s + 2.0 * a.b0 * s + a.k0;
    double d = double(a.rad) + std::sqrt(std::max(double(a.rad) * a.rad + k, 0.0));
    return (d > 0.0) ? k / d : 0.0;
}

bool ptSphereRoots(F3 ro, F3 rd, F3 c, float rad, float& t0, float& t1);

PT_MIRROR F3 ptAtmoOpticalDepth(const AtmoBody& b, F3 ro, F3 rd, F3 centre,
                                float t, int steps) {
    if (!(t > 0.0f)) return F3{0.0f, 0.0f, 0.0f};
    // Planetary P6 (#260): clip the interval to the medium before
    // spending the step budget on it.  See the derivation above the
    // shader function -- this is the black-disc fix.
    float u0, u1;
    if (!ptSphereRoots(ro, rd, centre, b.top_radius, u0, u1)) {
        return F3{0.0f, 0.0f, 0.0f};
    }
    float lo = std::max(u0, 0.0f);
    float hi = std::min(u1, t);
    if (!(hi > lo)) return F3{0.0f, 0.0f, 0.0f};
    int N = std::max(steps, 1);
    float dt = (hi - lo) / float(N);
    float oneSixth = 1.0f / 6.0f;
    RayAlt alt = rayAltBegin(ro, rd, centre, b.ground_radius);
    F3 tau{0.0f, 0.0f, 0.0f};
    F3 sr_l, sm_l, st_prev;
    ptAtmoCoefficients(b, float(std::max(rayAltAt(alt, lo), 0.0)), sr_l, sm_l, st_prev);
    for (int i = 0; i < N; ++i) {
        float s_mid   = lo + (float(i) + 0.5f) * dt;
        float s_right = lo + (float(i) + 1.0f) * dt;
        F3 sr_m, sm_m, st_mid;
        F3 sr_r, sm_r, st_right;
        ptAtmoCoefficients(b, float(std::max(rayAltAt(alt, s_mid), 0.0)),
                           sr_m, sm_m, st_mid);
        ptAtmoCoefficients(b, float(std::max(rayAltAt(alt, s_right), 0.0)),
                           sr_r, sm_r, st_right);
        tau = tau + (st_prev + 4.0f * st_mid + st_right) * (dt * oneSixth);
        st_prev = st_right;
    }
    return tau;
}

// --- shader mirror: ptSphereRoots / ptRayShell ----------------------------
// The roots come through the same Vieta rearrangement the shader uses;
// the accumulator itself is pinned by pt_math_sphere_test.cpp, so this
// mirror uses double for the two dot products rather than reproducing it,
// and the shell LOGIC -- which is what this file is testing -- is
// transcribed exactly.
bool ptSphereRoots(F3 ro, F3 rd, F3 c, float rad, float& t0, float& t1) {
    t0 = 0.0f; t1 = 0.0f;
    if (!(rad > 0.0f)) return false;
    double ox = double(ro.x) - c.x, oy = double(ro.y) - c.y, oz = double(ro.z) - c.z;
    double b = ox * double(rd.x) + oy * double(rd.y) + oz * double(rd.z);
    double k = ox * ox + oy * oy + oz * oz - double(rad) * double(rad);
    double h = b * b - k;
    if (h < 0.0) return false;
    h = std::sqrt(h);
    double q = (b >= 0.0) ? -(b + h) : (h - b);
    double ta = q;
    double tb = (q != 0.0) ? k / q : 0.0;
    t0 = float(std::min(ta, tb));
    t1 = float(std::max(ta, tb));
    return true;
}

PT_MIRROR bool ptRayShell(F3 ro, F3 rd, F3 centre, float r_inner, float r_outer,
                          float t_max, float& t_in, float& t_out) {
    t_in = 0.0f; t_out = 0.0f;
    if (!(r_outer > 0.0f) || !(t_max > 0.0f)) return false;
    float o0, o1;
    if (!ptSphereRoots(ro, rd, centre, r_outer, o0, o1)) return false;
    float lo = std::max(o0, 0.0f);
    float hi = std::min(o1, t_max);
    if (!(hi > lo)) return false;
    float i0, i1;
    if (r_inner > 0.0f && ptSphereRoots(ro, rd, centre, r_inner, i0, i1)) {
        if (i1 > lo) {
            if (i0 > lo) { hi = std::min(hi, i0); }
            else         { lo = i1; }
        }
    }
    if (!(hi > lo)) return false;
    t_in = lo; t_out = hi;
    return true;
}

// The IUGG mean radius, r_planet_radius's default.
constexpr float kR = 6371008.8f;
const F3 kCentre{0.0f, -kR, 0.0f};      // y = 0 tangent to the surface

std::string tighten(const char* path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str(), tight;
    tight.reserve(src.size());
    for (char ch : src) {
        if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
    }
    return tight;
}

std::size_t countOf(const std::string& hay, const std::string& needle) {
    std::size_t n = 0, at = 0;
    while ((at = hay.find(needle, at)) != std::string::npos) { ++n; ++at; }
    return n;
}

// --- shader mirror: sunSlantTransmittance / skyPhysical (P6, #260) ---------
// The two functions the ground-to-orbit claim is actually about.  The
// jitter is an explicit parameter here rather than randf(seed), so the
// sequence below is a deterministic function of altitude -- which is
// exactly what a continuity claim needs.
PT_MIRROR F3 sunSlantTransmittance(const AtmoBody& b, F3 p, F3 sun_dir,
                                   F3 centre) {
    float g0, g1;
    // Occluded by the body: the ray leaves p, descends, and re-enters the
    // surface before it can reach space.  This is night, stated as
    // geometry.
    if (ptSphereRoots(p, sun_dir, centre, b.ground_radius, g0, g1) && g0 > 0.0f) {
        return F3{0.0f, 0.0f, 0.0f};
    }
    float s0, s1;
    if (!ptSphereRoots(p, sun_dir, centre, b.top_radius, s0, s1)) {
        return F3{1.0f, 1.0f, 1.0f};      // already outside the shell
    }
    float t = std::max(s1, 0.0f);
    if (!(t > 0.0f)) return F3{1.0f, 1.0f, 1.0f};
    F3 tau = ptAtmoOpticalDepth(b, p, sun_dir, centre, t, 8);
    return F3{std::exp(-tau.x), std::exp(-tau.y), std::exp(-tau.z)};
}

float phaseRayleigh(float mu) { return 0.0596831036594607f * (1.0f + mu * mu); }
float phaseCornetteShanks(float mu, float g) {
    float g2 = g * g;
    float num = 3.0f * (1.0f - g2) * (1.0f + mu * mu);
    float den = 8.0f * 3.14159265358979f * (2.0f + g2)
              * std::pow(std::max(1.0f + g2 - 2.0f * g * mu, 1e-4f), 1.5f);
    return num / den;
}

PT_MIRROR F3 skyPhysical(const AtmoBody& b, F3 ro, F3 rd, F3 centre,
                         F3 sun_dir, float g_mie, float t_max, int steps,
                         float xi) {
    float t_in, t_out;
    if (!ptRayShell(ro, rd, centre, 0.0f, b.top_radius, t_max, t_in, t_out)) {
        return F3{0.0f, 0.0f, 0.0f};
    }
    float g0, g1;
    if (ptSphereRoots(ro, rd, centre, b.ground_radius, g0, g1) && g0 > 0.0f) {
        t_out = std::min(t_out, g0);
    }
    if (!(t_out > t_in)) return F3{0.0f, 0.0f, 0.0f};

    float mu = dot(rd, sun_dir);
    float ph_r = phaseRayleigh(mu);
    float ph_m = phaseCornetteShanks(mu, g_mie);
    int N = std::max(steps, 1);
    float dt = (t_out - t_in) / float(N);
    RayAlt alt = rayAltBegin(ro, rd, centre, b.ground_radius);
    F3 trans{1.0f, 1.0f, 1.0f};
    F3 acc{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < N; ++i) {
        float s = t_in + (float(i) + xi) * dt;
        if (s >= t_out) break;
        float h = float(std::max(rayAltAt(alt, s), 0.0));
        F3 sr, sm, st;
        ptAtmoCoefficients(b, h, sr, sm, st);
        F3 sp{ro.x + rd.x * s, ro.y + rd.y * s, ro.z + rd.z * s};
        F3 T = sunSlantTransmittance(b, sp, sun_dir, centre);
        F3 sun_at{80.0f * T.x, 80.0f * T.y, 80.0f * T.z};
        if (sun_at.x > 0.0f || sun_at.y > 0.0f || sun_at.z > 0.0f) {
            acc.x += trans.x * (sr.x * ph_r + sm.x * ph_m) * sun_at.x * dt;
            acc.y += trans.y * (sr.y * ph_r + sm.y * ph_m) * sun_at.y * dt;
            acc.z += trans.z * (sr.z * ph_r + sm.z * ph_m) * sun_at.z * dt;
        }
        trans.x *= std::exp(-st.x * dt);
        trans.y *= std::exp(-st.y * dt);
        trans.z *= std::exp(-st.z * dt);
    }
    return acc;
}

}  // namespace

TEST_CASE("the finiteness harness is not vacuous under -ffast-math") {
    // #275's trap: under __FINITE_MATH_ONLY__ both std::isfinite() and a
    // plain exponent read fold to `true`, so an assertion written the
    // obvious way passes on a value that really is infinite.  Produce an
    // infinity ARITHMETICALLY -- a literal would be constant-folded --
    // and require the harness to see it.
    volatile float big = 3.0e38f;
    float inf = big * big;
    REQUIRE_FALSE(finiteBits(inf));
    REQUIRE(finiteBits(1.0f));
}

TEST_CASE("ozone is a stratospheric layer, not an exponential") {
    AtmoBody b = ptAtmoEarth(kR);
    // The whole reason ozone gets a tent rather than the exp(-h/H) the
    // other two species use: there is essentially no ozone at the
    // surface.  An exponential profile is MAXIMAL there, so it would
    // redden a short vertical path -- where real ozone does nothing --
    // and under-absorb the long twilight slant path, where real ozone
    // does everything.  Getting the shape wrong inverts the effect.
    CHECK(ptAtmoOzoneProfile(b, 0.0f) == 0.0f);
    CHECK(ptAtmoOzoneProfile(b, 10000.0f) == doctest::Approx(0.0f));
    CHECK(ptAtmoOzoneProfile(b, 25000.0f) == doctest::Approx(1.0f));
    CHECK(ptAtmoOzoneProfile(b, 40000.0f) == doctest::Approx(0.0f));
    CHECK(ptAtmoOzoneProfile(b, 60000.0f) == 0.0f);
    // Symmetric about the peak, and linear on each flank.
    CHECK(ptAtmoOzoneProfile(b, 17500.0f) == doctest::Approx(0.5f));
    CHECK(ptAtmoOzoneProfile(b, 32500.0f) == doctest::Approx(0.5f));

    // The vertical ozone column: a tent of unit peak and half-width w has
    // area w, so tau = sigma_peak * half_width.  1.881e-6 * 15000 = 0.028
    // in green.  Sanity against the real world: the observed vertical
    // ozone optical depth in the Chappuis band is a few hundredths, which
    // is the order this lands on.  Pinned because it is what bounds the
    // Simpson error argument in ptAtmoOpticalDepth's comment.
    CHECK(double(b.ozone_sigma_a.y) * double(b.ozone_half_width)
          == doctest::Approx(0.0282).epsilon(0.01));
}

TEST_CASE("ozone absorbs and never scatters") {
    AtmoBody b = ptAtmoEarth(kR);
    AtmoBody no_o3 = ptAtmoEarth(kR);
    no_o3.ozone_sigma_a = F3{0.0f, 0.0f, 0.0f};

    for (float h : {0.0f, 5000.0f, 25000.0f, 40000.0f}) {
        F3 sr_a, sm_a, st_a, sr_b, sm_b, st_b;
        ptAtmoCoefficients(b, h, sr_a, sm_a, st_a);
        ptAtmoCoefficients(no_o3, h, sr_b, sm_b, st_b);
        CAPTURE(h);
        // Scattering is bit-identical with and without ozone...
        CHECK(sr_a.x == sr_b.x);
        CHECK(sr_a.y == sr_b.y);
        CHECK(sr_a.z == sr_b.z);
        CHECK(sm_a.y == sm_b.y);
        // ...and extinction is not, wherever the layer is present.
        if (ptAtmoOzoneProfile(b, h) > 0.0f) {
            CHECK(st_a.y > st_b.y);
        } else {
            CHECK(st_a.y == st_b.y);
        }
    }
}

TEST_CASE("ozone's absorption peaks in green -- the Chappuis band") {
    AtmoBody b = ptAtmoEarth(kR);
    // This ordering IS the physics.  The Chappuis band sits in the
    // red-orange/yellow-green, so ozone removes the middle and long
    // wavelengths and leaves the short one nearly untouched.  If the
    // triple were ever mistyped in wavelength order -- the obvious
    // mistake, since Rayleigh's triple ASCENDS with wavelength index --
    // twilight would go the wrong colour and no golden would say why.
    CHECK(b.ozone_sigma_a.y > b.ozone_sigma_a.x);   // green > red
    CHECK(b.ozone_sigma_a.x > b.ozone_sigma_a.z);   // red   > blue
    // Blue is more than an order of magnitude below green: ozone barely
    // touches the short wavelengths, which is exactly why what survives
    // a long slant path is blue.
    CHECK(b.ozone_sigma_a.y / b.ozone_sigma_a.z > 10.0f);
}

TEST_CASE("ozone keeps the twilight slant path blue instead of brown") {
    // The acceptance claim, as a number.
    //
    // At civil twilight the sunlight that eventually scatters to a
    // zenith-looking observer has already crossed the atmosphere at a
    // grazing angle, so it passes through a very long chord of the ozone
    // layer.  Model that directly: a ray at eye height travelling at
    // -4 degrees below the local horizontal, integrated to the far side
    // of the shell.  What reaches the scattering point is E_sun * T, and
    // its COLOUR is what decides whether the zenith reads blue or brown.
    const float R = kR;
    AtmoBody with_o3 = ptAtmoEarth(R);
    AtmoBody no_o3   = ptAtmoEarth(R);
    no_o3.ozone_sigma_a = F3{0.0f, 0.0f, 0.0f};

    // The path is a LIMB RAY: sunlight that has grazed the atmosphere
    // and is now heading for a scattering point over the observer.  Its
    // defining parameter is the perigee altitude -- how close to the
    // surface it came -- and for a sun a few degrees below a ground
    // observer's horizon that is the lower stratosphere.  15 km is used
    // here, which puts the path straight through the ozone layer's lower
    // flank, and the integral runs from perigee out to the top of the
    // shell (one half-chord; the colour ratio is unchanged by doubling
    // it, since both channels scale together).
    //
    // Constructing it from the perigee rather than from an elevation
    // angle at the observer is deliberate: an elevation angle at eye
    // height describes a ray that runs into the ground within 25 m,
    // whose transmittance underflows and carries no colour information
    // at all.  The perigee IS the physical parameter of a twilight ray.
    const float perigee_alt = 20000.0f;
    F3 ro{0.0f, perigee_alt, 0.0f};
    F3 rd{1.0f, 0.0f, 0.0f};              // perpendicular to the radius
    float s0, s1;
    REQUIRE(ptSphereRoots(ro, rd, kCentre, with_o3.top_radius, s0, s1));
    const float t = s1;
    REQUIRE(t > 0.0f);
    // ~1.04e6 m of air: the reason a twilight path is coloured at all.
    CHECK(double(t) > 9.0e5);

    F3 tau_a = ptAtmoOpticalDepth(with_o3, ro, rd, kCentre, t, 256);
    F3 tau_b = ptAtmoOpticalDepth(no_o3,   ro, rd, kCentre, t, 256);
    F3 T_a{std::exp(-tau_a.x), std::exp(-tau_a.y), std::exp(-tau_a.z)};
    F3 T_b{std::exp(-tau_b.x), std::exp(-tau_b.y), std::exp(-tau_b.z)};

    // THE SIGN FLIP.  This is the criterion, and it is a change of
    // ordering rather than a threshold, so it cannot be tuned past.
    //
    // Without ozone, Rayleigh alone has scattered the short wavelengths
    // out of a very long path and what survives is GREEN-dominant over
    // blue and RED-dominant over both: yellow-brown.  That is precisely
    // the twilight this engine rendered before #257, and it is what a
    // physically-incomplete model must produce -- not a bug in the
    // Rayleigh term, an absence in the model.
    //
    // With ozone, the Chappuis band removes the green and orange that
    // dominated, and the surviving light becomes BLUE-dominant over
    // green.  Same geometry, same Rayleigh, same aerosol; one absorbing
    // species added, and the colour ordering inverts.
    CHECK(T_b.z < T_b.y);       // without ozone: green beats blue
    CHECK(T_a.z > T_a.y);       // with ozone:    blue beats green
    // Red still beats blue on a path this long either way -- a twilight
    // sky is not uniformly blue, it is blue at the zenith over an orange
    // horizon, and this path is the horizon-most one.
    CHECK(T_b.x > T_b.z);

    // The same thing as ratios, so a regression reports how far it moved
    // rather than only that it moved.  Blue-versus-green is the primary
    // signal because green sits nearest the Chappuis peak; blue-versus-
    // red is real but weaker, because ozone's red cross-section is only
    // a third of its green one.
    const double ratio_with = double(T_a.z) / double(T_a.x);
    const double ratio_without = double(T_b.z) / double(T_b.x);
    const double bg_with = double(T_a.z) / double(T_a.y);
    const double bg_without = double(T_b.z) / double(T_b.y);
    CAPTURE(ratio_with);
    CAPTURE(ratio_without);
    CAPTURE(bg_with);
    CAPTURE(bg_without);
    CHECK(ratio_with > ratio_without);
    CHECK(bg_with > bg_without);
    // Bars set below the measured values (1.83 and 1.21 on this path) so
    // a modest change to the ozone profile does not fail the test, while
    // removing ozone -- which takes both enhancements to exactly 1.0 --
    // always does.  The green margin must exceed the red one: that
    // ordering IS the Chappuis band's shape, and a triple entered in the
    // wrong wavelength order would satisfy the magnitudes but not this.
    CHECK(bg_with / bg_without > 1.5);
    CHECK(ratio_with / ratio_without > 1.10);
    CHECK(bg_with / bg_without > ratio_with / ratio_without);

    // Ozone only ever removes light.  A "blue twilight" produced by
    // ADDING blue would be a different, wrong mechanism, so pin the
    // direction too.
    CHECK(T_a.x < T_b.x);
    CHECK(T_a.y < T_b.y);
    CHECK(T_a.z < T_b.z);

    // The zenith is where the blue hour is observed, and it is the case
    // the exponential-profile mistake would get wrong: a SHORT vertical
    // path must be almost unaffected by ozone, because there is no ozone
    // near the ground.  Same body, same integrator, straight up.
    F3 eye{0.0f, 1.7f, 0.0f};
    F3 up{0.0f, 1.0f, 0.0f};
    F3 tau_up_a = ptAtmoOpticalDepth(with_o3, eye, up, kCentre, 5000.0f, 256);
    F3 tau_up_b = ptAtmoOpticalDepth(no_o3,   eye, up, kCentre, 5000.0f, 256);
    CHECK(tau_up_a.y == doctest::Approx(tau_up_b.y).epsilon(1e-6));
}

TEST_CASE("Mie absorption is real and was previously absent") {
    AtmoBody b = ptAtmoEarth(kR);
    // The engine modelled no aerosol absorption at all before #257, i.e.
    // an implied single-scattering albedo of 1.  Earth's aerosol
    // absorbs slightly more than it scatters.
    const double albedo = double(b.mie_sigma_s.y)
                        / (double(b.mie_sigma_s.y) + double(b.mie_sigma_a.y));
    CHECK(albedo == doctest::Approx(0.4760).epsilon(0.001));

    // The user scale keeps that albedo fixed while changing loading:
    // raising r_volumetric_density means more of the SAME aerosol.
    AtmoBody hazy = ptAtmoScale(b, 1.0f, 1.0e-4f, 1.0f);
    const double albedo_hazy = double(hazy.mie_sigma_s.y)
                             / (double(hazy.mie_sigma_s.y) + double(hazy.mie_sigma_a.y));
    CHECK(albedo_hazy == doctest::Approx(albedo).epsilon(1e-5));
    CHECK(hazy.mie_sigma_s.y == doctest::Approx(1.0e-4f));

    // Rayleigh and ozone scales are independent of it.
    CHECK(hazy.rayleigh_sigma_s.y == doctest::Approx(b.rayleigh_sigma_s.y));
    AtmoBody dim = ptAtmoScale(b, 0.0f, 0.0f, 0.0f);
    CHECK(dim.rayleigh_sigma_s.y == 0.0f);
    CHECK(dim.mie_sigma_s.y == 0.0f);
    CHECK(dim.mie_sigma_a.y == 0.0f);
    CHECK(dim.ozone_sigma_a.y == 0.0f);
}

TEST_CASE("the Mie scale height correction is not cosmetic") {
    // The engine used 1500 m where Hillaire 2020 gives 1200 m.  A scale
    // height sets how much of the species sits ABOVE a given altitude,
    // and the vertical column is sigma * H, so 1500 vs 1200 is 25% more
    // aerosol in every vertical path -- and much more than 25% at the
    // altitudes where aircraft and cloud tops live, because the profiles
    // diverge exponentially.
    AtmoBody now = ptAtmoEarth(kR);
    AtmoBody old = ptAtmoEarth(kR);
    old.mie_scale_h = 1500.0f;
    CHECK(double(old.mie_scale_h) / double(now.mie_scale_h)
          == doctest::Approx(1.25));
    // At 5 km the old profile carries exp(5000/1200 - 5000/1500) = 2.30x
    // the aerosol density of the corrected one -- the profiles diverge
    // exponentially, so the error at altitude is far worse than the 25%
    // the column integral suggests.
    const double ratio = std::exp(-5000.0 / 1500.0) / std::exp(-5000.0 / 1200.0);
    CHECK(ratio == doctest::Approx(2.3010).epsilon(0.001));
}

TEST_CASE("8-step Simpson tracks a 4096-step reference") {
    // The production rule is 8 subintervals.  Its error is bounded here
    // against an independent 4096-step reference computed in double from
    // a materialised position -- deliberately NOT the shader's hoisted
    // quadratic, so the two do not share a structure that could be wrong
    // together.
    //
    // TOLERANCE, DERIVED.  Composite Simpson's error is
    // (b-a) h^4 f''''(xi) / 180.  For the exponential species the
    // integrand along the ray is exp(-h(s)/H) with h quadratic in s, so
    // the fourth derivative scales as (dh/ds)^4 / H^4 and the relative
    // error over N subintervals goes as (L/N)^4 * (1/H)^4 for a ray of
    // length L climbing at order-1 slope.  At N = 8 over the 100 km
    // paths swept below that is a few times 1e-3 for Rayleigh (H = 8 km)
    // and worse for Mie (H = 1.2 km).  Ozone's tent has TWO CORNERS
    // Simpson cannot see at all, and no smooth-function bound applies to
    // it -- but the whole ozone term is bounded by its vertical column,
    // 0.028 in green, so its absolute contribution to the error is
    // bounded even where its relative one is not.
    //
    // The bar below is therefore stated on the quantity that matters:
    // the resulting TRANSMITTANCE, exp(-tau), which is what multiplies
    // radiance.  1% of transmittance is under half a step of an 8-bit
    // channel at any exposure that is not clipping, so it is the point
    // below which the quadrature cannot be what a golden is showing.
    const AtmoBody b = ptAtmoEarth(kR);
    const float alts[5] = {1.7f, 1000.0f, 10000.0f, 40000.0f, 90000.0f};

    auto sweep = [&](float t_cap, double& worst, double& at_elev, double& at_alt) {
        worst = 0.0; at_elev = 0.0; at_alt = 0.0;
        for (int ai = 0; ai < 5; ++ai) {
            const float alt = alts[ai];
            for (int ei = 0; ei <= 32; ++ei) {
                const double elev = -90.0 + 180.0 * (ei / 32.0);
                const double e = elev * 3.14159265358979 / 180.0;
                F3 ro{0.0f, alt, 0.0f};
                F3 rd = norm(F3{float(std::cos(e)), float(std::sin(e)), 0.0f});
                // Clip to the shell or to the ground, whichever comes
                // first, so the integral is over air rather than rock or
                // vacuum -- then to the caller's cap.
                float g0, g1, s0, s1;
                const bool hits_ground = ptSphereRoots(ro, rd, kCentre, kR, g0, g1);
                if (!ptSphereRoots(ro, rd, kCentre, b.top_radius, s0, s1)) continue;
                float t = s1;
                if (hits_ground && g0 > 0.0f) t = std::min(t, g0);
                t = std::min(t, t_cap);
                if (!(t > 0.0f)) continue;
                for (int ch = 0; ch < 3; ++ch) {
                    F3 tau8 = ptAtmoOpticalDepth(b, ro, rd, kCentre, t, 8);
                    const double got = (ch == 0 ? tau8.x : ch == 1 ? tau8.y : tau8.z);
                    const double want =
                        refOpticalDepthChannel(b, ro, rd, kCentre, t, ch, 4096);
                    const double d = std::fabs(std::exp(-got) - std::exp(-want));
                    if (d > worst) { worst = d; at_elev = elev; at_alt = alt; }
                }
            }
        }
    };

    // (a) The band the engine actually integrates.  The in-scatter march
    // caps at 60 km (`t_far = min(primary_t, 60000)`) and the sky depth
    // sentinel is 50 km, so no production call is longer than this.
    double worst_prod, e_prod, a_prod;
    sweep(60000.0f, worst_prod, e_prod, a_prod);
    CAPTURE(worst_prod);
    CAPTURE(e_prod);
    CAPTURE(a_prod);
    CHECK(worst_prod < 0.01);

    // (b) The full shell crossing, which the engine does not currently
    // ask for but P6 will.  Recorded rather than asserted tightly,
    // because it is a genuine limitation of an 8-step rule and not a
    // defect: a grazing ray from the ground to the top of the shell is
    // ~500 km long, so one subinterval is ~62 km against a 1.2 km Mie
    // scale height, and the integrand is concentrated in the first
    // subinterval. Refining it is what a transmittance LUT buys -- each
    // (r, mu) integrated once, offline, with as fine a rule as wanted.
    // The bar here is set at the measured worst case plus headroom so
    // this records the number instead of asserting a wish.
    double worst_full, e_full, a_full;
    sweep(1.0e9f, worst_full, e_full, a_full);
    CAPTURE(worst_full);
    CAPTURE(e_full);
    CAPTURE(a_full);
    CHECK(worst_full < 0.03);
    // ...and that the production band really is the easier of the two,
    // so nobody reads (a) as covering (b).
    CHECK(worst_prod < worst_full);
}

// The pre-#260 rule, verbatim: N uniform Simpson subintervals over the
// WHOLE of [0, t], with no clip to the medium.  Kept as a live function
// rather than as prose so the black-disc case below can assert what the
// engine actually did, instead of asserting that today differs from a
// version nobody can run.
double unclippedOpticalDepthChannel(const AtmoBody& b, F3 ro, F3 rd, F3 centre,
                                    float t, int channel, int steps) {
    if (!(t > 0.0f)) return 0.0;
    int N = std::max(steps, 1);
    float dt = t / float(N);
    RayAlt alt = rayAltBegin(ro, rd, centre, b.ground_radius);
    auto st_at = [&](double s) {
        F3 sr, sm, st;
        ptAtmoCoefficients(b, float(std::max(rayAltAt(alt, s), 0.0)), sr, sm, st);
        return double(channel == 0 ? st.x : channel == 1 ? st.y : st.z);
    };
    double tau = 0.0, st_prev = st_at(0.0);
    for (int i = 0; i < N; ++i) {
        const double m = (i + 0.5) * double(dt), r = (i + 1.0) * double(dt);
        const double st_mid = st_at(m), st_right = st_at(r);
        tau += double(dt) / 6.0 * (st_prev + 4.0 * st_mid + st_right);
        st_prev = st_right;
    }
    return tau;
}

TEST_CASE("the planet is not a black disc from orbit") {
    // #260's founding symptom, as a number.
    //
    // Point a camera at the planet from above the atmosphere and the
    // surface renders BLACK on main -- reproduced at cam_pos 0 30000000 0
    // with r_planet_ground 1, and noted in planet_aerial.cfg's header as
    // the reason that fixture stops at 4 km rather than going to orbit.
    //
    // It is not a shading bug and it is not the sky model.  It is the
    // aerial-perspective transmittance, and specifically its QUADRATURE.
    // Eight uniform Simpson subintervals over [0, t] with t = 30 000 km
    // put seven of them in hard vacuum; the eighth is 3 750 km long and
    // its RIGHT endpoint is on the ground, where extinction is at its
    // sea-level maximum.  Simpson then charges 3 750 km of path at
    // (roughly) sea-level density.  The air the ray really crosses is
    // one vertical column -- optical depth 0.15 in green, transmittance
    // 0.86.
    const AtmoBody b = ptAtmoEarth(kR);
    const F3 down{0.0f, -1.0f, 0.0f};

    struct Case { float alt; const char* what; };
    const Case cases[2] = {{400000.0f, "400 km, ISS"},
                           {30000000.0f, "30 000 km, the repro in the P4 header"}};

    for (const Case& c : cases) {
        CAPTURE(c.what);
        // Camera on the local vertical above the surface, looking down.
        // kCentre is (0, -R, 0), so world y IS altitude here.
        const F3 ro{0.0f, c.alt, 0.0f};
        const float t = c.alt;

        for (int ch = 0; ch < 3; ++ch) {
            CAPTURE(ch);
            // The truth: a nadir column, whatever the camera's altitude.
            // It cannot depend on how far above the air the observer is,
            // because that distance is vacuum.
            const double tau_ref =
                refOpticalDepthChannel(b, ro, down, kCentre, t, ch, 200000);
            const double T_ref = std::exp(-tau_ref);
            CHECK(T_ref > 0.75);      // 0.936 / 0.864 / 0.759 R / G / B

            // RED: what the engine did before this phase.
            const double tau_old =
                unclippedOpticalDepthChannel(b, ro, down, kCentre, t, ch, 8);
            const double T_old = std::exp(-tau_old);

            // GREEN: what it does now.
            const F3 tau_new_v = ptAtmoOpticalDepth(b, ro, down, kCentre, t, 8);
            const double tau_new =
                double(ch == 0 ? tau_new_v.x : ch == 1 ? tau_new_v.y : tau_new_v.z);
            const double T_new = std::exp(-tau_new);

            // The claim is about CONTENT, not merely about difference:
            // the surface has to survive the air column with most of its
            // light, and that has to hold at both altitudes with the
            // same number, because the physics is the same integral.
            CHECK(T_new > 0.70);
            CHECK(std::fabs(T_new - T_ref) < 0.02);

            if (c.alt > 1.0e7f) {
                // ...and at 30 Mm the old rule really did produce black,
                // so the fixture header's "the shading gives black" is
                // this line and not a shading path at all.
                CHECK(T_old < 1.0e-3);
            } else {
                // At 400 km it is not black, but aerial perspective is
                // already ~10 points of transmittance too dark -- which
                // is why the acceptance altitude is not a place the old
                // rule was quietly fine either.
                CHECK(T_old < T_ref - 0.05);
            }
        }
    }
}

TEST_CASE("clipping to the medium discards a bounded, negligible tail") {
    // The clip is exact only if the medium really stops at top_radius.
    // It does not: the Rayleigh and Mie profiles are exponentials and
    // the tent is the only species that is genuinely zero up there.  So
    // the clip DOES throw something away, and the honest thing is to
    // bound it rather than to call the boundary exact.
    //
    // The worst case is the ray that grazes the top of the shell: this
    // function now returns exactly zero for it, while the true integral
    // is the Chapman chord of an exponential of scale height H_R about
    // its tangent point,
    //
    //     tau_true <= sigma_t(R_top) * sqrt(2 pi R_top H_R)
    //
    // with sigma_t(R_top) in green = 13.558e-6 * exp(-100/8)
    //                              +  8.396e-6 * exp(-100/1.2)
    //                              = 5.05e-11 m^-1,
    // and sqrt(2 pi * 6.471e6 * 8000) = 5.70e5 m, so tau_true <= 2.9e-5.
    //
    // Optical depth converts to transmittance one-for-one at this size
    // (1 - exp(-x) ~ x), so the discarded term is under one part in
    // 34 000 -- an order of magnitude below a single 8-bit quantum
    // (1/255 = 3.9e-3).  Measured here against the unclipped reference
    // rather than trusted from the algebra.
    const AtmoBody b = ptAtmoEarth(kR);

    // A ray tangent to the top of the shell, integrated over four
    // Chapman chords either side of the tangent point -- far enough out
    // that the exponential has fallen by e^-16 and the remainder is
    // below the printing precision of the bound.
    const double chapman = std::sqrt(2.0 * 3.14159265358979
                                     * double(b.top_radius) * double(b.rayleigh_scale_h));
    const double reach = 4.0 * chapman;
    const F3 ro{float(-reach), float(double(b.top_radius) - double(kR)), 0.0f};
    const F3 rd{1.0f, 0.0f, 0.0f};
    const float t = float(2.0 * reach);

    for (int ch = 0; ch < 3; ++ch) {
        CAPTURE(ch);
        const double discarded =
            refOpticalDepthChannel(b, ro, rd, kCentre, t, ch, 200000);
        CAPTURE(discarded);
        CHECK(discarded < 1.0e-4);
        CHECK(discarded < 1.0 / 255.0);
        // And the clipped rule really does return zero for it, so the
        // number above is the WHOLE of what the clip costs, not a part.
        const F3 got = ptAtmoOpticalDepth(b, ro, rd, kCentre, t, 8);
        const double g = double(ch == 0 ? got.x : ch == 1 ? got.y : got.z);
        CHECK(g == 0.0);
    }
}

TEST_CASE("the clip is a no-op for every ray that starts inside the shell") {
    // Why the pre-#260 goldens are bit-stable and not merely close.
    //
    // For an origin inside top_radius the near root is negative, so
    // lo = max(u0, 0) is +0.0 EXACTLY, and hi = min(u1, t) is t exactly
    // whenever t is the shorter.  IEEE-754 addition of +0.0 is the
    // identity on every finite value, so `lo + x` is `x` bit-for-bit and
    // the Simpson loop evaluates the same floats it did before.  The
    // highest camera in the pre-#260 fixture set is sunset_altitude at
    // 10 km, so this covers all of them.
    const AtmoBody b = ptAtmoEarth(kR);
    // 10 km is sunset_altitude, the highest camera that existed before
    // this phase; 80 m is clouds_raymarched; 1.7 m is the ground scenes.
    // 99 km is deliberately NOT in this list: a 30 km ray upward from
    // there leaves the shell after 1 km, so the clip fires and the
    // integral legitimately differs.  That case is covered by the
    // sub-case below rather than papered over here.
    const float alts[3] = {1.7f, 80.0f, 10000.0f};
    for (int ai = 0; ai < 3; ++ai) {
        CAPTURE(alts[ai]);
        const F3 ro{0.0f, alts[ai], 0.0f};
        for (int ei = 0; ei <= 16; ++ei) {
            const double e = (-90.0 + 180.0 * (ei / 16.0)) * 3.14159265358979 / 180.0;
            const F3 rd = norm(F3{float(std::cos(e)), float(std::sin(e)), 0.0f});
            // A t that stays inside the shell: the ground is at most
            // `alt` away straight down, and 30 km of reach from 10 km up
            // cannot escape a 100 km shell in any direction.
            const float t = 30000.0f;
            const F3 clipped = ptAtmoOpticalDepth(b, ro, rd, kCentre, t, 8);
            const double u0 = unclippedOpticalDepthChannel(b, ro, rd, kCentre, t, 0, 8);
            const double u1 = unclippedOpticalDepthChannel(b, ro, rd, kCentre, t, 1, 8);
            const double u2 = unclippedOpticalDepthChannel(b, ro, rd, kCentre, t, 2, 8);
            // The unclipped reference accumulates in double, so it is
            // not bit-comparable; equality to within a float ULP of the
            // value is what "the same floats went in" looks like from
            // outside.
            CHECK(double(clipped.x) == doctest::Approx(u0).epsilon(1e-6));
            CHECK(double(clipped.y) == doctest::Approx(u1).epsilon(1e-6));
            CHECK(double(clipped.z) == doctest::Approx(u2).epsilon(1e-6));
        }
    }

    // ...and the boundary, so "no-op" is bounded rather than believed.
    // From 99 km, straight up, 30 km of requested path: the shell ends
    // 1 km along, so the clip removes 29 km of near-vacuum that the old
    // rule spread its whole step budget across.  The clipped answer is
    // SMALLER (the removed span had nonzero, if tiny, extinction) and it
    // is the one that concentrates all eight subintervals on the 1 km
    // that has any air in it.
    {
        const F3 ro{0.0f, 99000.0f, 0.0f};
        const F3 rd{0.0f, 1.0f, 0.0f};
        const F3 clipped = ptAtmoOpticalDepth(b, ro, rd, kCentre, 30000.0f, 8);
        const double unclipped =
            unclippedOpticalDepthChannel(b, ro, rd, kCentre, 30000.0f, 2, 8);
        CHECK(double(clipped.z) < unclipped);
        // Both are utterly negligible -- which is the point: the clip
        // only ever removes air that could not have mattered.
        CHECK(unclipped < 1.0e-4);
    }
}

TEST_CASE("a horizontal ray reaches the cloud layer at a FINITE range") {
    // #51's founding symptom, as geometry.
    //
    // On a flat slab a perfectly horizontal ray from eye height NEVER
    // enters a layer above it -- the entry range is infinite -- so the
    // cloud deck stops at a hard line with a strip of bare sky beneath
    // it and can never touch the horizon.  On a shell the ground curves
    // away, so the same ray climbs into the layer at
    //     t = sqrt((R + h_layer)^2 - (R + h_eye)^2)
    // which for h_eye << h_layer << R is approximately sqrt(2 R dh).
    const float base = 200.0f, top = 500.0f;
    F3 ro{0.0f, 1.7f, 0.0f};
    F3 rd{1.0f, 0.0f, 0.0f};                     // exactly horizontal
    float t_in = 0.0f, t_out = 0.0f;
    REQUIRE(ptRayShell(ro, rd, kCentre, kR + base, kR + top, 1.0e9f, t_in, t_out));

    auto chord = [&](float h_layer) {
        const double a = double(kR) + h_layer;
        const double c = double(kR) + 1.7;
        return std::sqrt(a * a - c * c);
    };
    CHECK(double(t_in)  == doctest::Approx(chord(base)).epsilon(1e-3));
    CHECK(double(t_out) == doctest::Approx(chord(top)).epsilon(1e-3));
    // Both finite -- the property the slab could not provide.
    CHECK(finiteBits(t_in));
    CHECK(finiteBits(t_out));
    // Concretely: the deck begins about 50 km out and ends about 80 km
    // out.
    CHECK(t_in  > 49000.0f);
    CHECK(t_in  < 51000.0f);
    CHECK(t_out > 79000.0f);
    CHECK(t_out < 81000.0f);

    // And this is why the inline march's 30 km span clamp had to become
    // planar-frame-only.  At EYE height the span is 29.4 km, a hair
    // under the cap, so the clamp happens not to fire -- which is the
    // kind of coincidence that makes a bug survive review.  Raise the
    // camera ten metres and it does fire, truncating the deck short of
    // the horizon: the exact gap #51 exists to close, reintroduced by a
    // different route.
    CHECK(double(t_out - t_in) == doctest::Approx(29417.0).epsilon(1e-3));
    CHECK(t_out - t_in < 30000.0f);

    F3 ro100{0.0f, 100.0f, 0.0f};
    float u_in, u_out;
    REQUIRE(ptRayShell(ro100, rd, kCentre, kR + base, kR + top, 1.0e9f,
                       u_in, u_out));
    CHECK(u_out - u_in > 30000.0f);
    // The clamp would end the deck here instead of at the horizon,
    // losing the last several kilometres of it.
    CHECK(double(u_out) - double(u_in + 30000.0f) > 4000.0);
}

TEST_CASE("the shell interval is correct from every origin") {
    const float base = 200.0f, top = 500.0f;
    const float ri = kR + base, ro_r = kR + top;
    float t_in, t_out;

    SUBCASE("ground camera looking up enters at the layer base") {
        F3 ro{0.0f, 1.7f, 0.0f};
        F3 rd = norm(F3{1.0f, 1.0f, 0.0f});
        REQUIRE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1.0e9f, t_in, t_out));
        // sin 45 deg climb: entry at ~ (200 - 1.7)/sin45 = 280 m.
        CHECK(double(t_in) == doctest::Approx(280.0).epsilon(0.01));
        CHECK(double(t_out) == doctest::Approx(704.7).epsilon(0.01));
    }
    SUBCASE("camera inside the layer starts at zero") {
        F3 ro{0.0f, 300.0f, 0.0f};
        F3 rd = norm(F3{0.0f, 1.0f, 0.0f});
        REQUIRE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1.0e9f, t_in, t_out));
        CHECK(t_in == 0.0f);
        CHECK(double(t_out) == doctest::Approx(200.0).epsilon(0.01));
    }
    SUBCASE("camera above the layer looking down enters at the top") {
        F3 ro{0.0f, 10000.0f, 0.0f};
        F3 rd = norm(F3{0.0f, -1.0f, 0.0f});
        REQUIRE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1.0e9f, t_in, t_out));
        CHECK(double(t_in) == doctest::Approx(9500.0).epsilon(0.01));
        CHECK(double(t_out) == doctest::Approx(9800.0).epsilon(0.01));
    }
    SUBCASE("camera above the layer looking up misses it entirely") {
        F3 ro{0.0f, 10000.0f, 0.0f};
        F3 rd = norm(F3{0.0f, 1.0f, 0.0f});
        CHECK_FALSE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1.0e9f, t_in, t_out));
    }
    SUBCASE("the far side of the shell is never returned") {
        // A ground camera looking horizontally must not be handed the
        // segment of the layer on the far side of the planet -- it is
        // behind an opaque body, and marching it would put clouds from
        // the other side of the world into the frame.  The interval
        // returned ends at the near tangent, well inside a quarter
        // circumference.
        F3 ro{0.0f, 1.7f, 0.0f};
        F3 rd{1.0f, 0.0f, 0.0f};
        REQUIRE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1.0e9f, t_in, t_out));
        const double quarter = 0.5 * 3.14159265358979 * double(kR);
        CHECK(double(t_out) < 0.05 * quarter);
    }
    SUBCASE("t_max clips the interval") {
        F3 ro{0.0f, 1.7f, 0.0f};
        F3 rd{1.0f, 0.0f, 0.0f};
        REQUIRE(ptRayShell(ro, rd, kCentre, ri, ro_r, 60000.0f, t_in, t_out));
        CHECK(t_out == 60000.0f);
        // And a t_max in front of the layer removes it altogether.
        CHECK_FALSE(ptRayShell(ro, rd, kCentre, ri, ro_r, 1000.0f, t_in, t_out));
    }
}

// The sun 30 degrees above the horizon at the sub-camera point, in the
// plane of the view rays below, so every case here has a lit sky and a
// terminator somewhere on the body.
const F3 kSun = norm(F3{std::cos(0.5236f), std::sin(0.5236f), 0.0f});

// Sky radiance looking along `rd` from altitude `a` on the local
// vertical.  kCentre is (0, -R, 0), so world y IS altitude.
double skyLumAt(const AtmoBody& b, double a, F3 rd, int steps) {
    const F3 ro{0.0f, float(a), 0.0f};
    // xi = 0.5 -- the midpoint rule.  The shader jitters instead, which
    // makes it an unbiased estimator that the accumulator converges; for
    // a continuity claim we want the deterministic sibling, because
    // otherwise every "jump" measured would be sampling noise.
    const F3 L = skyPhysical(b, ro, rd, kCentre, kSun, b.mie_g, 1.0e30f,
                             steps, 0.5f);
    return 0.2126 * L.x + 0.7152 * L.y + 0.0722 * L.z;
}

TEST_CASE("the sky is continuous in altitude from the ground to orbit") {
    // #260's acceptance criterion is a claim about a SEQUENCE, not about
    // any frame: a climb from 1.7 m to 400 km with no discontinuity in
    // sky colour.  A golden PNG cannot settle that, and neither can a
    // threshold on adjacent frames -- adjacent frames are SUPPOSED to
    // differ, because the sky really is changing.
    //
    // Continuity has an exact definition that needs no tolerance at all:
    // refine the sampling and the largest step must shrink with it.  For
    // a Lipschitz function sampled at spacing h the largest adjacent
    // difference is bounded by K*h, so halving h halves it; for a
    // function with a jump of size J the largest difference is at least J
    // however fine the sampling gets.  The two behaviours are
    // qualitatively different and no constant has to be chosen to tell
    // them apart.
    //
    // The bar below is 0.6 rather than the ideal 0.5 only because the
    // location of the maximum moves between refinements, so the halved
    // ladder does not sample the same worst point.  A genuine
    // discontinuity gives a ratio of 1.0.
    const AtmoBody b = ptAtmoEarth(kR);

    struct Dir { F3 rd; const char* what; };
    const Dir dirs[5] = {
        {F3{0.0f, 1.0f, 0.0f},                          "straight up"},
        {norm(F3{0.7071f, 0.7071f, 0.0f}),              "45 deg up, toward the sun"},
        {norm(F3{-0.7071f, 0.7071f, 0.0f}),             "45 deg up, away from the sun"},
        {F3{1.0f, 0.0f, 0.0f},                          "horizontal"},
        {norm(F3{0.9848f, -0.1736f, 0.0f}),             "10 deg below horizontal -- the limb from orbit"},
    };

    auto worstStep = [&](const Dir& d, int rungs) {
        // Geometric ladder from 1 m to 500 km.  Geometric rather than
        // linear because the atmosphere's own structure is exponential:
        // a linear ladder that resolves the 1.2 km Mie scale height would
        // need 400 000 rungs to reach orbit.
        double worst = 0.0;
        double prev = skyLumAt(b, 1.0, d.rd, 64);
        for (int i = 1; i <= rungs; ++i) {
            const double f = double(i) / double(rungs);
            const double a = 1.0 * std::pow(500000.0 / 1.0, f);
            const double cur = skyLumAt(b, a, d.rd, 64);
            worst = std::max(worst, std::fabs(cur - prev));
            prev = cur;
        }
        return worst;
    };

    for (int di = 0; di < 5; ++di) {
        CAPTURE(dirs[di].what);
        const double w1 = worstStep(dirs[di], 256);
        const double w2 = worstStep(dirs[di], 512);
        const double w3 = worstStep(dirs[di], 1024);
        CAPTURE(w1);
        CAPTURE(w2);
        CAPTURE(w3);
        // Refining halves the worst step, twice over.  A jump would
        // plateau instead.
        CHECK(w2 < 0.6 * w1);
        CHECK(w3 < 0.6 * w2);
    }
}

TEST_CASE("the sky resolves to black on the way out, and does it smoothly") {
    // The other half of the acceptance criterion, and the one the
    // painted modes cannot meet at all.  procSky returns a lerp of
    // float3(0.32, 0.55, 0.92) and float3(0.78, 0.88, 1.00) at every
    // altitude, so its zenith at 400 km is its zenith at eye height.
    // Measured on a rendered climb it plateaus at 56.3/255 of luminance
    // from 200 km upward and the black-space fraction of the frame stays
    // at exactly 0.000 all the way to 400 km -- a frozen value, which is
    // worse than a jump because it never resolves.
    //
    // The physical sky has nothing to freeze: at 400 km looking up the
    // chord through the shell is empty, so the integral is over an empty
    // interval, so it is zero.  No fade, no threshold, no cvar.
    const AtmoBody b = ptAtmoEarth(kR);
    const F3 up{0.0f, 1.0f, 0.0f};

    const double ground = skyLumAt(b, 1.7, up, 64);
    CHECK(ground > 0.0);

    // Monotone all the way out: every rung is dimmer than the one below
    // it.  (Strictly: the remaining column above the camera shrinks
    // monotonically, and single-scattered radiance is monotone in it.)
    double prev = ground;
    for (int i = 1; i <= 200; ++i) {
        const double a = 1.7 * std::pow(500000.0 / 1.7, double(i) / 200.0);
        const double cur = skyLumAt(b, a, up, 64);
        CAPTURE(a);
        CHECK(cur <= prev * (1.0 + 1e-6));
        prev = cur;
    }

    // Above the shell, looking away from the body, the interval really
    // is empty and the answer is EXACTLY zero rather than small.
    CHECK(skyLumAt(b, double(b.top_radius - kR) + 1.0, up, 64) == 0.0);
    CHECK(skyLumAt(b, 400000.0, up, 64) == 0.0);
    // ...while at the same altitude the limb still has a chord through
    // it and is the brightest thing in the frame, so "black" is a
    // statement about direction and not about the mode having died.
    //
    // The band is narrow and worth stating in numbers, because it is the
    // reason the sky-view LUT the roadmap started from could not survive
    // this: from r = 6 771 km the top of the shell is at
    // asin(6471/6771) = 72.94 deg from nadir, i.e. 17.06 deg below local
    // horizontal, and the hard limb is at asin(6371/6771) = 70.15 deg,
    // i.e. 19.85 deg below.  The WHOLE atmosphere is those 2.8 degrees.
    //
    // Directions are built from the tangent altitude they are aiming at
    // rather than written as angles, because the angles are what the
    // reader would have to re-derive: sin(theta from nadir) =
    // (R + h_tangent) / r_camera.
    const double r_cam = double(kR) + 400000.0;
    auto aimAtTangent = [&](double h_t) {
        const double sth = std::min((double(kR) + h_t) / r_cam, 1.0);
        const double cth = std::sqrt(std::max(1.0 - sth * sth, 0.0));
        return norm(F3{float(sth), float(-cth), 0.0f});   // nadir rotated by theta
    };
    // 150 km grazes above the top of the shell: no chord, exactly zero.
    CHECK(skyLumAt(b, 400000.0, aimAtTangent(150000.0), 64) == 0.0);
    // 12 km grazes through the bright part: a chord of
    // 2*sqrt(r_top^2 - (R+12km)^2) = 2*sqrt(6471^2 - 6383^2) = 2 128 km.
    const F3 limb = aimAtTangent(12000.0);
    CHECK(skyLumAt(b, 400000.0, limb, 64) > 0.0);
    // And the limb is far brighter than the zenith was at the SURFACE:
    // a grazing chord crosses hundreds of km of the densest air, which
    // is why the limb reads as a bright rim in every orbital photograph.
    CHECK(skyLumAt(b, 400000.0, limb, 64) > ground);
}

TEST_CASE("the night side is dark because the body is in the way") {
    // The terminator, as geometry rather than as a threshold.  There is
    // no elevation cutoff anywhere in the physical path: a sample whose
    // sun ray runs into the body gets exactly zero, and the graded region
    // between "sun ray clears the limb through 40 air masses" and "sun
    // ray hits rock" IS the terminator.
    const AtmoBody b = ptAtmoEarth(kR);
    // A point 30 km up, and a sun direction that puts it on the far side
    // of the body: sun_dir pointing steeply DOWN through the planet.
    const F3 p{0.0f, 30000.0f, 0.0f};
    const F3 sun_down{0.0f, -1.0f, 0.0f};
    const F3 T_night = sunSlantTransmittance(b, p, sun_down, kCentre);
    CHECK(T_night.x == 0.0f);
    CHECK(T_night.y == 0.0f);
    CHECK(T_night.z == 0.0f);

    // Straight up, no body in the way, and the vertical column above
    // 30 km is thin -- so nearly all of the sunlight arrives.
    const F3 sun_up{0.0f, 1.0f, 0.0f};
    const F3 T_noon = sunSlantTransmittance(b, p, sun_up, kCentre);
    CHECK(T_noon.y > 0.98f);

    // And the terminator is GRADED, not a step: sweep the sun through
    // the local horizon and the transmittance falls smoothly, reddening
    // as it goes, before the body cuts it off.  Blue dies first --
    // Rayleigh scatters it ~6x harder -- which is why a sunset is red.
    double prev_g = 1.0;
    bool saw_partial = false, saw_red = false;
    for (int i = 0; i <= 60; ++i) {
        const double e = (10.0 - 20.0 * (i / 60.0)) * 3.14159265358979 / 180.0;
        const F3 sd = norm(F3{float(std::cos(e)), float(std::sin(e)), 0.0f});
        const F3 T = sunSlantTransmittance(b, p, sd, kCentre);
        CHECK(double(T.y) <= prev_g + 1e-6);      // monotone
        if (T.y > 0.0f && T.y < 0.99f) saw_partial = true;
        if (T.y > 0.0f && T.x > T.z * 1.5f) saw_red = true;
        prev_g = T.y;
    }
    CHECK(saw_partial);
    CHECK(saw_red);
    // ...and it does reach zero, so the night side exists.
    CHECK(prev_g == 0.0);
}

TEST_CASE("stars stop at the body, not 2.9 degrees below the horizontal") {
    // The star gate was `cosTheta > -0.05` in both copies of starsOnly --
    // a flat-Earth statement that everything more than 2.87 degrees below
    // the observer's horizontal is ground.  At eye height the visible
    // horizon dips 3.7 arcmin, so the 2.87 degrees is slack nobody
    // noticed.  From 400 km the hard limb is 19.85 degrees down, so the
    // gate cuts a horizontal line across the frame with stars above it
    // and nothing below, and the band it wrongly blanks widens as the
    // camera climbs -- the discontinuity in star visibility named in this
    // phase's acceptance.
    const AtmoBody b = ptAtmoEarth(kR);
    const F3 eye{0.0f, 400000.0f, 0.0f};
    const double r_cam = double(kR) + 400000.0;
    // The hard limb, from the geometry rather than from a constant.
    const double limb_deg = 90.0 - std::asin(double(kR) / r_cam)
                                     * 180.0 / 3.14159265358979;
    CHECK(limb_deg == doctest::Approx(19.79).epsilon(1e-3));

    auto hitsBody = [&](double below_deg) {
        const double e = -below_deg * 3.14159265358979 / 180.0;
        const F3 rd = norm(F3{float(std::cos(e)), float(std::sin(e)), 0.0f});
        float t0, t1;
        return ptSphereRoots(eye, rd, kCentre, b.ground_radius, t0, t1)
               && t0 > 0.0f;
    };

    // Everything shallower than the limb is open sky and must show stars.
    CHECK_FALSE(hitsBody(0.0));
    CHECK_FALSE(hitsBody(5.0));
    CHECK_FALSE(hitsBody(10.0));
    CHECK_FALSE(hitsBody(19.0));
    // Everything past it is planet.
    CHECK(hitsBody(21.0));
    CHECK(hitsBody(45.0));
    CHECK(hitsBody(90.0));
    // The old gate would have blanked 16.9 of those 19.79 degrees: at
    // 320x240 over a 55 degree vertical FOV that is 74 rows of the frame,
    // roughly a third of it, on the night side where the stars are the
    // only thing in it.
    CHECK(limb_deg - 2.87 > 16.9);
    CHECK((limb_deg - 2.87) / 55.0 * 240.0 > 73.0);

    // At eye height the two criteria agree to within the horizon dip,
    // which is why every pre-#260 fixture is unaffected: 3.7 arcmin is
    // 0.062 degrees, far inside the 2.87 the old gate allowed.
    const F3 ground_eye{0.0f, 1.7f, 0.0f};
    const double dip_deg = std::acos(double(kR) / (double(kR) + 1.7))
                             * 180.0 / 3.14159265358979;
    CHECK(dip_deg < 0.07);
    {
        const double e = -0.5 * 3.14159265358979 / 180.0;   // 0.5 deg down
        const F3 rd = norm(F3{float(std::cos(e)), float(std::sin(e)), 0.0f});
        float t0, t1;
        CHECK(ptSphereRoots(ground_eye, rd, kCentre, b.ground_radius, t0, t1));
        CHECK(t0 > 0.0f);        // ground, under both criteria
    }
}

TEST_CASE("shader mirror is still faithful") {
    const std::string math = tighten(PT_SHADER_MATH_PATH);
    REQUIRE_FALSE(math.empty());

    // One body constructor, and every constant this file transcribes.
    // Counted, not merely present: a second constructor with different
    // numbers would satisfy find() and silently split the model in two,
    // which is the failure mode #276 demonstrated in this codebase.
    CHECK(countOf(math, "publicPtAtmoBodyptAtmoEarth(floatground_radius){") == 1u);
    CHECK(countOf(math, "b.top_radius=ground_radius+100000.0;") == 1u);
    CHECK(countOf(math, "b.rayleigh_sigma_s=float3(5.802e-6,13.558e-6,33.100e-6);") == 1u);
    CHECK(countOf(math, "b.rayleigh_scale_h=8000.0;") == 1u);
    CHECK(countOf(math, "b.mie_sigma_s=float3(3.996e-6,3.996e-6,3.996e-6);") == 1u);
    CHECK(countOf(math, "b.mie_sigma_a=float3(4.400e-6,4.400e-6,4.400e-6);") == 1u);
    CHECK(countOf(math, "b.mie_scale_h=1200.0;") == 1u);
    CHECK(countOf(math, "b.mie_g=0.8;") == 1u);
    CHECK(countOf(math, "b.ozone_sigma_a=float3(0.650e-6,1.881e-6,0.085e-6);") == 1u);
    CHECK(countOf(math, "b.ozone_center=25000.0;") == 1u);
    CHECK(countOf(math, "b.ozone_half_width=15000.0;") == 1u);

    // The tent, and the three-species extinction sum.
    CHECK(countOf(math, "returnmax(0.0,1.0-abs(h-b.ozone_center)/b.ozone_half_width);") == 1u);
    CHECK(countOf(math,
        "sigma_t=sigma_s_rayleigh+(b.mie_sigma_s+b.mie_sigma_a)*dm+b.ozone_sigma_a*doz;") == 1u);
    // Scattering must NOT include the absorbing terms.
    CHECK(countOf(math, "sigma_s_rayleigh=b.rayleigh_sigma_s*dr;") == 1u);
    CHECK(countOf(math, "sigma_s_mie=b.mie_sigma_s*dm;") == 1u);

    // The user scales, and that the aerosol keeps its albedo.
    CHECK(countOf(math, "b.mie_sigma_a=co_albedo*ss;") == 1u);

    // The shell.  The near-piece selection is the part that is easy to
    // get subtly wrong, so both branches are pinned.
    CHECK(countOf(math, "if(i0>lo){hi=min(hi,i0);}") == 1u);
    CHECK(countOf(math, "else{lo=i1;}") == 1u);

    // Planetary P6 (#260): the optical-depth integral clips to the
    // medium before spending its step budget.  Pinned on the whole
    // clause and COUNTED, because the mirror above transcribes it and a
    // mirror that has drifted is worthless -- and because the black-disc
    // case is the only other thing standing between this repo and a
    // planet that renders as a hole in the sky.  Deleting the clause
    // from the shader while leaving `lo` in the loop would still satisfy
    // pt_math_altitude_test's hoist pins, which is exactly the
    // find()-shaped hole #276 lived in.
    CHECK(countOf(math,
        "if(!ptSphereRoots(ro,rd,centre,b.top_radius,u0,u1)){"
        "returnfloat3(0.0,0.0,0.0);//rayneverentersthemedium}") == 1u);
    CHECK(countOf(math, "floatlo=max(u0,0.0);floathi=min(u1,t);") == 1u);
    CHECK(countOf(math, "floatdt=(hi-lo)/float(N);") == 1u);
    // And that the pre-#260 form is gone rather than merely shadowed.
    CHECK(countOf(math, "floatdt=t/float(N);") == 0u);
    // ptSphereRoots must be DEFINED before the integral that calls it --
    // Slang has no forward declaration here, so a later edit that moves
    // it back below would fail to compile; pinning the order makes the
    // reason visible rather than leaving it as a compile mystery.
    CHECK(math.find("publicboolptSphereRoots(")
          < math.find("publicfloat3ptAtmoOpticalDepth("));

    // Earth must be ASSIGNED once, in the constructor -- the point of
    // the struct.  Before #257 the Rayleigh scale height was a literal
    // in four files.  Counted on the assignment rather than on the bare
    // number so the prose above the constructor, which names the value
    // while explaining why it used to be scattered, does not have to be
    // written around the test.
    CHECK(countOf(math, "b.rayleigh_scale_h=8000.0;") == 1u);
    CHECK(countOf(math, "b.mie_scale_h=1200.0;") == 1u);
    CHECK(countOf(math, "scale_h=1500.0") == 0u);   // the pre-#257 value

    // And the phase functions the sky modes share.
    CHECK(countOf(math, "return0.0596831036594607*(1.0+mu*mu);") == 1u);
    CHECK(countOf(math, "publicfloatptPhaseCornetteShanks(floatmu,floatg){") == 1u);

    // No copy of the medium survives in the kernels: PathTrace.slang and
    // CloudsRaymarch.slang must go through the body.
    const std::string pt = tighten(PT_SHADER_PATHTRACE_PATH);
    REQUIRE_FALSE(pt.empty());
    CHECK(countOf(pt, "kSigmaR") == 0u);
    CHECK(countOf(pt, "kHmie") == 0u);
    CHECK(countOf(pt, "kHrayleigh") == 0u);
    CHECK(countOf(pt, "PtAtmoBodyatmosphereBody(){") == 1u);
    CHECK(countOf(pt, "returnptAtmoScale(ptAtmoEarth(R),clouds_p3.w,vol_params.x,clouds_p4.w);") == 1u);

    const std::string cr = tighten(PT_SHADER_CLOUDSRAYMARCH_PATH);
    REQUIRE_FALSE(cr.empty());
    CHECK(countOf(cr, "/8000.0") == 0u);
    CHECK(countOf(cr, "atm_body.rayleigh_scale_h") == 1u);

    // The cloud layer is a shell in both kernels, from one helper each,
    // and the density gate is on the SAME field the geometry is -- if
    // those two ever disagree the layer's geometry and its density
    // function stop agreeing about where a given altitude is.
    CHECK(countOf(pt, "boolcloudLayerInterval(") == 1u);
    CHECK(countOf(cr, "boolcloudLayerInterval(") == 1u);
    CHECK(countOf(pt, "cloudLayerInterval(") == 4u);   // 1 definition + 3 uses
    CHECK(countOf(cr, "cloudLayerInterval(") == 2u);   // 1 definition + 1 use

    // --- Planetary P6 (#260): the physical sky ------------------------
    // The mirror above transcribes skyPhysical and sunSlantTransmittance
    // and the ground-to-orbit continuity claim rests entirely on that
    // transcription, so every line the mirror depends on is counted here.
    CHECK(countOf(pt, "float3skyPhysical(float3ro,float3rd,floatt_max,"
                      "intsteps,inoutuintseed){") == 1u);
    // Clip to the shell with r_inner = 0 (the NEAR piece, correct for an
    // origin below, inside or above), then stop at the body.  Both halves
    // are load-bearing: without the first the mode integrates vacuum from
    // orbit, without the second it integrates through the planet.
    CHECK(countOf(pt, "if(!ptRayShell(ro,rd,pc,0.0,body.top_radius,"
                      "t_max,t_in,t_out)){") == 1u);
    CHECK(countOf(pt, "if(ptSphereRoots(ro,rd,pc,planet_R,g0,g1)&&g0>0.0)"
                      "{t_out=min(t_out,g0);}") == 1u);
    // The sun term is the SAME slant-path integral the NEE site and the
    // cloud march use, which is what makes the terminator geometry rather
    // than a threshold -- and there must be exactly one of it.
    CHECK(countOf(pt, "float3sun_at=float3(80.0)*sunSlantTransmittance(sp,"
                      "sun_dir);") == 1u);
    CHECK(countOf(pt, "acc+=trans*(sigma_s_ray*ph_r+sigma_s_mie*ph_m)"
                      "*sun_at*dt;") == 1u);
    // No elevation gate anywhere in the physical path.  procSky, hosek,
    // sunDisc and starsOnly all fade on a scalar sun elevation; the whole
    // point of mode 4 is that it does not, so a smoothstep on sun
    // elevation appearing inside skyPhysical would be a regression that
    // no image test would obviously catch.
    {
        const std::size_t at = pt.find("float3skyPhysical(");
        const std::size_t end = pt.find("float3sunDiscPhysical(");
        REQUIRE(at != std::string::npos);
        REQUIRE(end != std::string::npos);
        REQUIRE(end > at);
        const std::string body = pt.substr(at, end - at);
        CHECK(countOf(body, "smoothstep") == 0u);
        CHECK(countOf(body, "sun_and_mode.y") == 0u);
    }
    // Aerial perspective: the 60 km cap is legacy-only, and the physical
    // mode clips to the shell instead.  From 400 km the FIRST 60 km of a
    // nadir ray is vacuum and all the air is in the last 100, so capping
    // there gives the surface extinction with no in-scatter -- the black
    // disc's failure one term further along.  Both branches pinned, so
    // neither can be deleted or have the other's behaviour.
    CHECK(countOf(pt, "floatt_far_v=min(primary_t,60000.0);") == 1u);
    CHECK(countOf(pt, "span_ok_v=(planet_R_v>0.0)&&ptRayShell(primary_ro,"
                      "primary_rd,pc_v,0.0,ptAtmoEarth(planet_R_v).top_radius,"
                      "primary_t,t_near_v,t_far_v);") == 1u);
    CHECK(countOf(pt, "span_ok_v=false;//skyPhysicalalreadydidthisray") == 1u);
    // The two marches meet at the limb -- one pixel hits the body and gets
    // its in-scatter from the AP march, the pixel above it misses and gets
    // it from skyPhysical -- so they must agree about the aerosol phase
    // function or there is a seam along the limb.  Legacy keeps plain HG
    // bit-for-bit; the physical mode uses the same Cornette-Shanks helper
    // skyPhysical does.
    CHECK(countOf(pt, "floatphase_mie_v=physical_sky_v?ptPhaseCornetteShanks"
                      "(cos_theta,g):phase;") == 1u);
    CHECK(countOf(pt, "ptPhaseCornetteShanks(") == 2u);   // skyPhysical + the AP march

    // The disc radiance is E / Omega, derived, with Omega taken from the
    // RENDERED half-angle so r_sun_size stays energy-conserving.
    CHECK(countOf(pt, "floatomega=6.28318530718*(1.0-kCosR);") == 1u);
    CHECK(countOf(pt, "returnT*(80.0/max(omega,1.0e-12))*bright;") == 1u);

    // ...and the host half of the same contract.  Mode 4 IMPLIES the
    // physical sun transmittance: skyPhysical lights every sample with
    // sunSlantTransmittance, so if the NEE site were still on
    // exp(-0.30 / max(sun_elev, 0.04)) the air and the ground would
    // disagree about the colour of sunlight and the sun would keep
    // lighting the night side.  A scene that set r_sky_mode physical and
    // forgot r_sun_physical_transmittance would look subtly wrong in a
    // way that is very hard to attribute, so it is not settable wrong.
    const std::string eng = tighten(PT_ENGINE_CPP_PATH);
    REQUIRE_FALSE(eng.empty());
    CHECK(countOf(eng, "if(sky_mode_id==4u)push.sun_extra2[1]=1.0f;") == 1u);
    CHECK(countOf(eng, "elseif(sky_mode_str==\"physical\"){") == 1u);
    CHECK(countOf(eng, "\"physical\"};") == 1u);   // the allowed_values list

    // The star gate, in BOTH copies.  StarsComposite.slang imports no
    // modules and carries a hand-written mirror of localUp() for that
    // reason, so it carries a hand-written mirror of the body test too --
    // and two copies of one gate is exactly the drift hazard this file
    // exists to catch, so both are counted and the flat-Earth form is
    // asserted GONE from each rather than merely outnumbered.
    CHECK(countOf(pt, "boolstar_visible=(cosTheta>-0.05);") == 1u);
    CHECK(countOf(pt, "if(star_visible&&exposure_pad.z>0.5&&day<0.6){") == 1u);
    CHECK(countOf(pt, "}elseif(star_visible&&exposure_pad.y>0.5&&day<0.6){") == 1u);
    CHECK(countOf(pt, "cosTheta>-0.05&&exposure_pad") == 0u);
    const std::string sc = tighten(PT_SHADER_STARSCOMPOSITE_PATH);
    REQUIRE_FALSE(sc.empty());
    CHECK(countOf(sc, "boolstarRayHitsBody(float3rd){") == 1u);
    CHECK(countOf(sc, "if(star_visible&&exposure_pad.z>0.5&&day<0.6){") == 1u);
    CHECK(countOf(sc, "}elseif(star_visible&&exposure_pad.y>0.5&&day<0.6){") == 1u);
    CHECK(countOf(sc, "cosTheta>-0.05&&exposure_pad") == 0u);

    const std::string pc = tighten(PT_SHADER_PATHTRACECLOUD_PATH);
    REQUIRE_FALSE(pc.empty());
    CHECK(countOf(pc,
        "floatalt=(planet_cr.w>0.0)?ptAltitudeAboveSphere(pos,planet_cr.xyz,planet_cr.w):pos.y;")
        == 1u);
    CHECK(countOf(pc, "if(alt<base_y||alt>top_y)return0.0;") == 1u);
    // The old flat gate is gone.
    CHECK(countOf(pc, "if(pos.y<base_y||pos.y>top_y)") == 0u);
}

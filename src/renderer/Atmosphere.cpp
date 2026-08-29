// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Implementation of the host physical atmosphere.  See Atmosphere.h for
// the units audit, the RGB convention and the citations; this file is the
// arithmetic.

#include "renderer/Atmosphere.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace pt::atmo {
namespace {

struct V3 {
    double x, y, z;
};

inline V3 operator+(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return V3{a.x * s, a.y * s, a.z * s}; }
inline double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double Len(V3 a) { return std::sqrt(Dot(a, a)); }
inline V3 Norm(V3 a) {
    const double l = Len(a);
    return (l > 0.0) ? V3{a.x / l, a.y / l, a.z / l} : V3{0.0, 0.0, 0.0};
}
inline V3 Load(const double v[3]) { return V3{v[0], v[1], v[2]}; }

constexpr double kPi = 3.14159265358979323846;
// 3 / (16 pi) -- ptPhaseRayleigh's constant, to the same digits.
constexpr double kRayleighPhaseK = 0.0596831036594607;
// 1 / (4 pi) -- the isotropic phase Hillaire 2020 section 4 uses for both
// the second-order source and the rescatter transfer.
constexpr double kIsotropicPhase = 1.0 / (4.0 * kPi);

// Both roots of the ray-sphere quadratic.  The shader's ptSphereRoots
// carries an accumulated dot product and a Vieta root-pair recovery
// because it runs in fp32 at |oc| ~ 6.4e6; in double the plain form
// already keeps ~10 significant digits of k at that magnitude, so the
// host uses the plain form and gets a MORE accurate answer, not a
// different model.
bool SphereRoots(V3 ro, V3 rd, V3 c, double rad, double& t0, double& t1) {
    t0 = 0.0;
    t1 = 0.0;
    if (!(rad > 0.0)) return false;
    const V3 oc = ro - c;
    const double b = Dot(oc, rd);
    const double k = Dot(oc, oc) - rad * rad;
    double h = b * b - k;
    if (h < 0.0) return false;
    h = std::sqrt(h);
    const double q = (b >= 0.0) ? -(b + h) : (h - b);
    const double ta = q;
    const double tb = (q != 0.0) ? k / q : 0.0;
    t0 = std::min(ta, tb);
    t1 = std::max(ta, tb);
    return true;
}

double PhaseRayleigh(double mu) { return kRayleighPhaseK * (1.0 + mu * mu); }

double PhaseCornetteShanks(double mu, double g) {
    const double g2 = g * g;
    const double num = 3.0 * (1.0 - g2) * (1.0 + mu * mu);
    const double den = 8.0 * kPi * (2.0 + g2)
                     * std::pow(std::max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5);
    return num / den;
}

double OzoneProfile(const Body& b, double h) {
    return std::max(0.0, 1.0 - std::fabs(h - b.ozone_center) / b.ozone_half_width);
}

inline double Altitude(const Body& b, V3 p, V3 centre) {
    return Len(p - centre) - b.ground_radius;
}

}  // namespace

Body Earth(double ground_radius) {
    Body b;
    b.ground_radius    = ground_radius;
    b.top_radius       = ground_radius + 100000.0;
    b.rayleigh_sigma_s[0] = 5.802e-6;
    b.rayleigh_sigma_s[1] = 13.558e-6;
    b.rayleigh_sigma_s[2] = 33.100e-6;
    b.rayleigh_scale_h = 8000.0;
    b.mie_sigma_s[0] = b.mie_sigma_s[1] = b.mie_sigma_s[2] = 3.996e-6;
    b.mie_sigma_a[0] = b.mie_sigma_a[1] = b.mie_sigma_a[2] = 4.400e-6;
    b.mie_scale_h = 1200.0;
    b.mie_g       = 0.8;
    b.ozone_sigma_a[0] = 0.650e-6;
    b.ozone_sigma_a[1] = 1.881e-6;
    b.ozone_sigma_a[2] = 0.085e-6;
    b.ozone_center     = 25000.0;
    b.ozone_half_width = 15000.0;
    return b;
}

Body Scale(Body b, double rayleigh_scale, double mie_sigma_s_abs,
           double ozone_scale) {
    const double rs = std::max(rayleigh_scale, 0.0);
    for (int c = 0; c < 3; ++c) b.rayleigh_sigma_s[c] *= rs;
    const double ss0 = b.mie_sigma_s[1];
    double co_albedo[3] = {0.0, 0.0, 0.0};
    if (ss0 > 0.0) {
        for (int c = 0; c < 3; ++c) co_albedo[c] = b.mie_sigma_a[c] / ss0;
    }
    const double ss = std::max(mie_sigma_s_abs, 0.0);
    for (int c = 0; c < 3; ++c) {
        b.mie_sigma_s[c] = ss;
        b.mie_sigma_a[c] = co_albedo[c] * ss;
    }
    const double oz = std::max(ozone_scale, 0.0);
    for (int c = 0; c < 3; ++c) b.ozone_sigma_a[c] *= oz;
    return b;
}

void Coefficients(const Body& b, double h, double sigma_s_rayleigh[3],
                  double sigma_s_mie[3], double sigma_t[3]) {
    const double hc  = std::max(h, 0.0);
    const double dr  = std::exp(-hc / b.rayleigh_scale_h);
    const double dm  = std::exp(-hc / b.mie_scale_h);
    const double doz = OzoneProfile(b, hc);
    for (int c = 0; c < 3; ++c) {
        sigma_s_rayleigh[c] = b.rayleigh_sigma_s[c] * dr;
        sigma_s_mie[c]      = b.mie_sigma_s[c] * dm;
        sigma_t[c] = sigma_s_rayleigh[c]
                   + (b.mie_sigma_s[c] + b.mie_sigma_a[c]) * dm
                   + b.ozone_sigma_a[c] * doz;
    }
}

void OpticalDepth(const Body& b, const double ro_[3], const double rd_[3],
                  const double centre_[3], double t, int steps,
                  double tau_out[3]) {
    tau_out[0] = tau_out[1] = tau_out[2] = 0.0;
    if (!(t > 0.0)) return;
    const V3 ro = Load(ro_), rd = Load(rd_), centre = Load(centre_);
    double u0 = 0.0, u1 = 0.0;
    if (!SphereRoots(ro, rd, centre, b.top_radius, u0, u1)) return;
    const double lo = std::max(u0, 0.0);
    const double hi = std::min(u1, t);
    if (!(hi > lo)) return;
    const int N = std::max(steps, 1);
    const double dt = (hi - lo) / static_cast<double>(N);
    double sr[3], sm[3], st_prev[3], st_mid[3], st_right[3];
    Coefficients(b, Altitude(b, ro + rd * lo, centre), sr, sm, st_prev);
    for (int i = 0; i < N; ++i) {
        const double s_mid   = lo + (static_cast<double>(i) + 0.5) * dt;
        const double s_right = lo + (static_cast<double>(i) + 1.0) * dt;
        Coefficients(b, Altitude(b, ro + rd * s_mid, centre), sr, sm, st_mid);
        Coefficients(b, Altitude(b, ro + rd * s_right, centre), sr, sm, st_right);
        for (int c = 0; c < 3; ++c) {
            tau_out[c] += dt * (1.0 / 6.0)
                        * (st_prev[c] + 4.0 * st_mid[c] + st_right[c]);
            st_prev[c] = st_right[c];
        }
    }
}

void SunSlantTransmittance(const Body& b, const double p_[3],
                           const double sun_dir_[3], const double centre_[3],
                           double out[3], int steps) {
    const V3 p = Load(p_), sun = Load(sun_dir_), centre = Load(centre_);
    double g0 = 0.0, g1 = 0.0;
    if (SphereRoots(p, sun, centre, b.ground_radius, g0, g1) && g0 > 0.0) {
        out[0] = out[1] = out[2] = 0.0;   // the body is in the way: night
        return;
    }
    double s0 = 0.0, s1 = 0.0;
    if (!SphereRoots(p, sun, centre, b.top_radius, s0, s1)) {
        out[0] = out[1] = out[2] = 1.0;   // already outside the shell
        return;
    }
    const double t = std::max(s1, 0.0);
    if (!(t > 0.0)) {
        out[0] = out[1] = out[2] = 1.0;
        return;
    }
    double tau[3];
    OpticalDepth(b, p_, sun_dir_, centre_, t, steps, tau);
    for (int c = 0; c < 3; ++c) out[c] = std::exp(-tau[c]);
}

// --- Multiple scattering ---------------------------------------------------
//
// Hillaire 2020 section 4, transcribed rather than re-derived.  For a
// point at radius r under a sun at cosine mu_s:
//
//   L_2   = mean over sphere directions of
//             integral of  T(p->x) sigma_s(x) (1/4pi) E T_sun(x) dx
//           plus the ground-bounce term when the direction hits the body
//   f_ms  = mean over sphere directions of  integral of T(p->x) sigma_s(x) dx
//   Psi   = L_2 / (1 - f_ms)
//
// The 1/(1 - f_ms) is the geometric series over scattering orders under
// Hillaire's assumption that each further order redistributes energy the
// same way; it is what turns a second-order term into an all-orders one
// for the price of a second accumulator.
namespace {

void BuildMsRow(const Body& b, const MsLutParams& p, int iy, float* out) {
    const V3 centre{0.0, 0.0, 0.0};
    const double E[3] = {kSolarIrradianceR, kSolarIrradianceG, kSolarIrradianceB};
    const int n_sqrt = std::max(2, static_cast<int>(std::lround(
        std::sqrt(static_cast<double>(std::max(p.directions, 4))))));
    const int n_dir  = n_sqrt * n_sqrt;
    const int steps  = std::max(4, p.march_steps);

    {
        // Radius axis, linear in altitude between the ground and the top of
        // the shell.  Linear rather than Bruneton's rho-remap because this
        // table's integrand is smooth in altitude -- the remap exists to
        // condition GRAZING RAYS in the transmittance table, and there is
        // no ray direction on this axis at all.
        const double v = (static_cast<double>(iy) + 0.5)
                       / static_cast<double>(kMsLutHeight);
        const double r = b.ground_radius
                       + v * (b.top_radius - b.ground_radius);
        for (int ix = 0; ix < kMsLutWidth; ++ix) {
            const double u = (static_cast<double>(ix) + 0.5)
                           / static_cast<double>(kMsLutWidth);
            const double mu_s = u * 2.0 - 1.0;     // [-1, 1]
            const double sin_s = std::sqrt(std::max(0.0, 1.0 - mu_s * mu_s));
            // Local frame: up is +Y, the sun sits in the XY plane.
            const V3 pos{0.0, r, 0.0};
            const V3 sun{sin_s, mu_s, 0.0};
            const double sun_arr[3] = {sun.x, sun.y, sun.z};

            double L2[3]  = {0.0, 0.0, 0.0};
            double fms[3] = {0.0, 0.0, 0.0};

            for (int d = 0; d < n_dir; ++d) {
                // Uniform sphere sampling on an n_sqrt x n_sqrt lattice,
                // cell-centred.  Deterministic and stratified; no RNG, so
                // two builds of the same body are bit-identical.
                const int di = d % n_sqrt;
                const int dj = d / n_sqrt;
                const double a0 = (static_cast<double>(di) + 0.5)
                                / static_cast<double>(n_sqrt);
                const double a1 = (static_cast<double>(dj) + 0.5)
                                / static_cast<double>(n_sqrt);
                const double cz  = 1.0 - 2.0 * a0;               // cos(theta)
                const double sz  = std::sqrt(std::max(0.0, 1.0 - cz * cz));
                const double phi = 2.0 * kPi * a1;
                const V3 dir{sz * std::cos(phi), cz, sz * std::sin(phi)};

                // Clip to the shell, then to the ground.
                double t_in = 0.0, t_out = 0.0;
                double o0 = 0.0, o1 = 0.0;
                if (!SphereRoots(pos, dir, centre, b.top_radius, o0, o1)) continue;
                t_in  = std::max(o0, 0.0);
                t_out = o1;
                bool hits_ground = false;
                double g0 = 0.0, g1 = 0.0;
                if (SphereRoots(pos, dir, centre, b.ground_radius, g0, g1)
                    && g0 > 0.0) {
                    t_out = std::min(t_out, g0);
                    hits_ground = true;
                }
                if (!(t_out > t_in)) continue;

                const double dt = (t_out - t_in) / static_cast<double>(steps);
                double trans[3] = {1.0, 1.0, 1.0};
                for (int i = 0; i < steps; ++i) {
                    const double s = t_in + (static_cast<double>(i) + 0.5) * dt;
                    const V3 x = pos + dir * s;
                    const double h = Altitude(b, x, centre);
                    double sr[3], sm[3], st[3];
                    Coefficients(b, h, sr, sm, st);
                    const double xa[3] = {x.x, x.y, x.z};
                    double tsun[3];
                    const double c0[3] = {0.0, 0.0, 0.0};
                    SunSlantTransmittance(b, xa, sun_arr, c0, tsun);
                    for (int c = 0; c < 3; ++c) {
                        const double ss = sr[c] + sm[c];   // total scattering
                        L2[c]  += trans[c] * ss * kIsotropicPhase
                                * E[c] * tsun[c] * dt;
                        fms[c] += trans[c] * ss * dt;
                        trans[c] *= std::exp(-st[c] * dt);
                    }
                }
                if (hits_ground) {
                    // The planet's own albedo lighting the air above it.
                    // Lambertian, so albedo/pi times the cosine, carried
                    // back up through the transmittance already
                    // accumulated along this direction.
                    const V3 gp = pos + dir * t_out;
                    const V3 n  = Norm(gp - centre);
                    const double ndl = std::max(0.0, Dot(n, sun));
                    if (ndl > 0.0) {
                        const double ga[3] = {gp.x, gp.y, gp.z};
                        double tsun[3];
                        const double c0[3] = {0.0, 0.0, 0.0};
                        SunSlantTransmittance(b, ga, sun_arr, c0, tsun);
                        for (int c = 0; c < 3; ++c) {
                            L2[c] += trans[c] * tsun[c] * ndl
                                   * (p.ground_albedo / kPi) * E[c];
                        }
                    }
                }
            }

            const double inv_n = 1.0 / static_cast<double>(n_dir);
            float* texel = out + (static_cast<std::size_t>(iy) * kMsLutWidth
                                  + static_cast<std::size_t>(ix)) * 4;
            for (int c = 0; c < 3; ++c) {
                const double l  = L2[c] * inv_n;
                const double f  = fms[c] * inv_n;
                // f is a rescatter fraction and is < 1 for any physical
                // medium; the clamp is a guard against a user driving
                // r_rayleigh to an unphysical value, not a tuning knob.
                const double series = 1.0 / std::max(1.0 - f, 1.0e-3);
                texel[c] = static_cast<float>(l * series);
            }
            texel[3] = 1.0f;
        }
    }
}

}  // namespace

void BuildMultiScatterLut(const Body& b, const MsLutParams& p, float* out) {
    if (out == nullptr) return;
    int nt = p.threads;
    if (nt <= 0) {
        nt = static_cast<int>(std::thread::hardware_concurrency());
        if (nt <= 0) nt = 1;
    }
    nt = std::min(nt, kMsLutHeight);
    if (nt <= 1) {
        for (int iy = 0; iy < kMsLutHeight; ++iy) BuildMsRow(b, p, iy, out);
        return;
    }
    // Static row partition.  Rows are disjoint in `out` and every row is a
    // pure function of (b, p, iy), so the table is bit-identical whatever
    // the thread count is -- a property the unit test asserts rather than
    // assumes, because "parallel but deterministic" is exactly the kind of
    // claim that rots.
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        pool.emplace_back([&b, &p, out, t, nt]() {
            for (int iy = t; iy < kMsLutHeight; iy += nt) {
                BuildMsRow(b, p, iy, out);
            }
        });
    }
    for (auto& th : pool) th.join();
}

void SampleMultiScatter(const float* lut, const Body& b, double r,
                        double mu_s, double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    if (lut == nullptr) return;
    const double span = std::max(b.top_radius - b.ground_radius, 1.0);
    double v = (r - b.ground_radius) / span;
    v = std::min(std::max(v, 0.0), 1.0);
    double u = (std::min(std::max(mu_s, -1.0), 1.0) + 1.0) * 0.5;
    const double fx = u * kMsLutWidth - 0.5;
    const double fy = v * kMsLutHeight - 0.5;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const double tx = fx - static_cast<double>(x0);
    const double ty = fy - static_cast<double>(y0);
    auto clampi = [](int a, int lo, int hi) {
        return a < lo ? lo : (a > hi ? hi : a);
    };
    const int xa = clampi(x0, 0, kMsLutWidth - 1);
    const int xb = clampi(x0 + 1, 0, kMsLutWidth - 1);
    const int ya = clampi(y0, 0, kMsLutHeight - 1);
    const int yb = clampi(y0 + 1, 0, kMsLutHeight - 1);
    for (int c = 0; c < 3; ++c) {
        const double c00 = lut[(ya * kMsLutWidth + xa) * 4 + c];
        const double c10 = lut[(ya * kMsLutWidth + xb) * 4 + c];
        const double c01 = lut[(yb * kMsLutWidth + xa) * 4 + c];
        const double c11 = lut[(yb * kMsLutWidth + xb) * 4 + c];
        const double a = c00 + (c10 - c00) * tx;
        const double bb = c01 + (c11 - c01) * tx;
        out[c] = a + (bb - a) * ty;
    }
}

void SkyRadiance(const Body& b, const float* ms_lut, const double ro_[3],
                 const double rd_[3], const double centre_[3],
                 const double sun_dir_[3], int steps, double mie_g,
                 bool include_mie, double out[3], double step_power) {
    out[0] = out[1] = out[2] = 0.0;
    const V3 ro = Load(ro_), rd = Norm(Load(rd_)), centre = Load(centre_);
    const V3 sun = Norm(Load(sun_dir_));
    double o0 = 0.0, o1 = 0.0;
    if (!SphereRoots(ro, rd, centre, b.top_radius, o0, o1)) return;
    double t_in  = std::max(o0, 0.0);
    double t_out = o1;
    double g0 = 0.0, g1 = 0.0;
    if (SphereRoots(ro, rd, centre, b.ground_radius, g0, g1) && g0 > 0.0) {
        t_out = std::min(t_out, g0);
    }
    if (!(t_out > t_in)) return;

    const double E[3] = {kSolarIrradianceR, kSolarIrradianceG, kSolarIrradianceB};
    const double mu   = Dot(rd, sun);
    const double ph_r = PhaseRayleigh(mu);
    const double ph_m = PhaseCornetteShanks(mu, mie_g);
    const int N = std::max(steps, 1);
    const double span = t_out - t_in;
    const double kp   = std::max(step_power, 1.0);
    double trans[3] = {1.0, 1.0, 1.0};
    double s_prev = t_in;
    for (int i = 0; i < N; ++i) {
        // Segment boundaries at t_in + span * (i/N)^kp, midpoint rule on
        // the resulting non-uniform segments.  kp == 1 is exactly the
        // uniform rule the shader runs, to the last bit of the arithmetic
        // it shares.
        const double x1 = static_cast<double>(i + 1) / static_cast<double>(N);
        const double s_next = (kp == 1.0)
            ? t_in + span * x1
            : t_in + span * std::pow(x1, kp);
        const double dt = s_next - s_prev;
        const double s  = 0.5 * (s_prev + s_next);
        s_prev = s_next;
        if (!(dt > 0.0)) continue;
        const V3 x = ro + rd * s;
        const double h = Altitude(b, x, centre);
        double sr[3], sm[3], st[3];
        Coefficients(b, h, sr, sm, st);
        const double xa[3] = {x.x, x.y, x.z};
        double tsun[3];
        SunSlantTransmittance(b, xa, sun_dir_, centre_, tsun);
        double ms[3] = {0.0, 0.0, 0.0};
        if (ms_lut != nullptr) {
            const double r  = Len(x - centre);
            const V3 up     = Norm(x - centre);
            SampleMultiScatter(ms_lut, b, r, Dot(up, sun), ms);
        }
        for (int c = 0; c < 3; ++c) {
            const double s_ray = sr[c];
            const double s_mie = include_mie ? sm[c] : 0.0;
            // Single scattering with the real phase functions, plus the
            // isotropic all-orders term.  The multi-scatter table already
            // carries the solar irradiance (see BuildMultiScatterLut), so
            // it multiplies the scattering coefficient and nothing else.
            const double single = (s_ray * ph_r + s_mie * ph_m) * E[c] * tsun[c];
            const double multi  = (s_ray + s_mie) * ms[c];
            out[c] += trans[c] * (single + multi) * dt;
            trans[c] *= std::exp(-st[c] * dt);
        }
    }
}

SkyCook CookSky(const Body& b, const float* ms_lut, const SkyCookParams& p) {
    SkyCook ck{};
    const double centre[3] = {0.0, 0.0, 0.0};
    const double r = b.ground_radius + std::max(p.observer_alt, 0.0);
    const double ro[3] = {0.0, r, 0.0};
    const double mu_s = std::min(std::max(p.sun_elev_sin, -1.0), 1.0);
    const double sin_s = std::sqrt(std::max(0.0, 1.0 - mu_s * mu_s));
    const double sun[3] = {sin_s, mu_s, 0.0};

    // The three anchor directions.  "Horizon" is the geometric horizontal
    // at the observer, not the dipped visible horizon: procSky applies its
    // own dip to the gradient parameter, so cooking a dipped direction
    // here would apply it twice.
    const double dir_zenith[3] = {0.0, 1.0, 0.0};
    const double dir_sun[3]    = {1.0, 0.0, 0.0};
    const double dir_anti[3]   = {-1.0, 0.0, 0.0};

    double z[3], hs[3], ha[3];
    // include_mie = false: the aerosol's forward lobe is the marched term.
    SkyRadiance(b, ms_lut, ro, dir_zenith, centre, sun, p.march_steps,
                b.mie_g, false, z, 2.0);
    SkyRadiance(b, ms_lut, ro, dir_sun, centre, sun, p.march_steps,
                b.mie_g, false, hs, 2.0);
    SkyRadiance(b, ms_lut, ro, dir_anti, centre, sun, p.march_steps,
                b.mie_g, false, ha, 2.0);
    for (int c = 0; c < 3; ++c) {
        ck.zenith[c]       = static_cast<float>(z[c]);
        ck.horizon_sun[c]  = static_cast<float>(hs[c]);
        ck.horizon_anti[c] = static_cast<float>(ha[c]);
    }
    return ck;
}

}  // namespace pt::atmo

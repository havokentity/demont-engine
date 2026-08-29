// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "WaterOptics.h"

#include <algorithm>
#include <cmath>

namespace pt::water {
namespace {

// Pope & Fry 1997, table as redistributed by
// omlc.org/spectra/water/data/pope97.txt, converted 1/cm -> 1/m and
// decimated 2.5 nm -> 5 nm. 380 nm first, 700 nm last, 65 entries.
//
// Transcription is verified in tests/pt_water_optics_test.cpp against the
// paper's own quoted anchors (the 0.0044 /m minimum near 418 nm and
// a(700) = 0.624 /m) plus structural properties no typo survives: the
// count, monotone rise from the minimum to the red end, and the three
// named-wavelength values.
constexpr double kPopeFry[kPopeFrySamples] = {
    // 380  385      390      395      400
    0.01137, 0.00941, 0.00851, 0.00813, 0.00663,
    // 405  410      415      420      425
    0.00530, 0.00473, 0.00444, 0.00454, 0.00478,
    // 430  435      440      445      450
    0.00495, 0.00530, 0.00635, 0.00751, 0.00922,
    // 455  460      465      470      475
    0.00962, 0.00979, 0.01011, 0.01060, 0.01140,
    // 480  485      490      495      500
    0.01270, 0.01360, 0.01500, 0.01730, 0.02040,
    // 505  510      515      520      525
    0.02560, 0.03250, 0.03960, 0.04090, 0.04170,
    // 530  535      540      545      550
    0.04340, 0.04520, 0.04740, 0.05110, 0.05650,
    // 555  560      565      570      575
    0.05960, 0.06190, 0.06420, 0.06950, 0.07720,
    // 580  585      590      595      600
    0.08960, 0.11000, 0.13510, 0.16720, 0.22240,
    // 605  610      615      620      625
    0.25770, 0.26440, 0.26780, 0.27550, 0.28340,
    // 630  635      640      645      650
    0.29160, 0.30120, 0.31080, 0.32500, 0.34000,
    // 655  660      665      670      675
    0.37100, 0.41000, 0.42900, 0.43900, 0.44800,
    // 680  685      690      695      700
    0.46500, 0.48600, 0.51600, 0.55900, 0.62400,
};

// Morel's fit constants, in the form Zhang, Hu & He 2009 eq. (1) states
// them. Named so a reader can check each one against the paper rather than
// against a single fused literal.
constexpr double kMorelBeta90At500  = 1.38e-4;   // 1/m/sr, S = 0
constexpr double kMorelLambda0Nm    = 500.0;
constexpr double kMorelExponent     = -4.32;
constexpr double kMorelSalinityGain = 0.3;
constexpr double kMorelSalinityRef  = 37.0;      // ppt, Morel's sample

}  // namespace

const double* PopeFryAbsorptionTable() noexcept { return kPopeFry; }

double PureWaterAbsorption(double lambda_nm) noexcept {
    // Clamped, not extrapolated: past 700 nm the real spectrum climbs
    // through the 760 nm band by another order of magnitude and a linear
    // continuation of the last two samples would be fiction.
    const double clamped =
        std::clamp(lambda_nm, kPopeFryLambdaMinNm, kPopeFryLambdaMaxNm);
    const double x = (clamped - kPopeFryLambdaMinNm) / kPopeFryStepNm;
    const double f = std::floor(x);
    const auto   i = static_cast<std::size_t>(f);
    if (i + 1 >= kPopeFrySamples) return kPopeFry[kPopeFrySamples - 1];
    const double t = x - f;
    return kPopeFry[i] * (1.0 - t) + kPopeFry[i + 1] * t;
}

glm::dvec3 PureWaterAbsorptionRgb() noexcept {
    return {PureWaterAbsorption(kLambdaRedNm),
            PureWaterAbsorption(kLambdaGreenNm),
            PureWaterAbsorption(kLambdaBlueNm)};
}

double MorelBeta90(double lambda_nm, double salinity_ppt) noexcept {
    const double spectral =
        std::pow(std::max(lambda_nm, 1.0) / kMorelLambda0Nm, kMorelExponent);
    const double salinity =
        1.0 + kMorelSalinityGain * std::max(salinity_ppt, 0.0) /
                  kMorelSalinityRef;
    return kMorelBeta90At500 * spectral * salinity;
}

double RayleighDepolarisedSphereFactor(double depolarisation) noexcept {
    // b = 2 pi * beta(90) * int_0^pi [1 + k cos^2 t] sin t dt with
    // k = (1 - d) / (1 + d), and int cos^2 t sin t dt over [0, pi] = 2/3.
    // So b / beta(90) = 4 pi (1 + k / 3). No table, no rounded 16.06.
    const double k = (1.0 - depolarisation) / (1.0 + depolarisation);
    return 4.0 * 3.14159265358979323846 * (1.0 + k / 3.0);
}

double PureSeawaterScattering(double lambda_nm,
                              double salinity_ppt) noexcept {
    return RayleighDepolarisedSphereFactor(kSeawaterDepolarisation) *
           MorelBeta90(lambda_nm, salinity_ppt);
}

glm::dvec3 PureSeawaterScatteringRgb() noexcept {
    return {PureSeawaterScattering(kLambdaRedNm, kOceanSalinityPpt),
            PureSeawaterScattering(kLambdaGreenNm, kOceanSalinityPpt),
            PureSeawaterScattering(kLambdaBlueNm, kOceanSalinityPpt)};
}

double RayleighDepolarisedPhase(double cos_theta,
                                double depolarisation) noexcept {
    const double k = (1.0 - depolarisation) / (1.0 + depolarisation);
    // beta(theta) / b -- the same two expressions divided, so the
    // normalisation is exact by construction rather than by a constant.
    return (1.0 + k * cos_theta * cos_theta) /
           RayleighDepolarisedSphereFactor(depolarisation);
}

glm::dvec3 AttenuationLengths(const glm::dvec3& extinction) noexcept {
    return {extinction.x > 0.0 ? 1.0 / extinction.x : 0.0,
            extinction.y > 0.0 ? 1.0 / extinction.y : 0.0,
            extinction.z > 0.0 ? 1.0 / extinction.z : 0.0};
}

double OpticalBlackDepth(const glm::dvec3& extinction) noexcept {
    const double lo =
        std::min(extinction.x, std::min(extinction.y, extinction.z));
    if (!(lo > 0.0)) return 0.0;
    // One 8-bit level: exp(-c d) < 1/255.
    return std::log(255.0) / lo;
}

// --- 5. Water-leaving radiance (#305) -------------------------------------

namespace {
constexpr double kPi = 3.14159265358979323846;

// Henyey-Greenstein, normalised to 1 over the sphere. The exact mirror of
// ptPhaseHenyeyGreenstein in shaders/PathTraceMath.slang.
double HenyeyGreenstein(double cos_theta, double g) noexcept {
    const double g2 = g * g;
    const double d  = std::max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * kPi * std::pow(d, 1.5));
}
}  // namespace

SingleScatterGeometry RefractSunAndView(double sun_zenith_rad,
                                        double view_zenith_rad,
                                        double delta_azimuth_rad,
                                        double n) noexcept {
    SingleScatterGeometry g;
    const double sin_s_a = std::sin(sun_zenith_rad);
    const double sin_v_a = std::sin(view_zenith_rad);
    g.mu_sun_air  = std::cos(sun_zenith_rad);
    g.mu_view_air = std::cos(view_zenith_rad);
    // Snell. Entering the denser medium never total-internal-reflects, so
    // both roots exist for every real air-side angle.
    const double sin_s_w = std::clamp(sin_s_a / n, -1.0, 1.0);
    const double sin_v_w = std::clamp(sin_v_a / n, -1.0, 1.0);
    g.mu_sun_water  = std::sqrt(std::max(0.0, 1.0 - sin_s_w * sin_s_w));
    g.mu_view_water = std::sqrt(std::max(0.0, 1.0 - sin_v_w * sin_v_w));
    // The scattering angle between the DOWNWARD beam's propagation
    // direction and the UPWARD direction the scattered light leaves in.
    // Put the sun's azimuth at 0 and the viewer's at delta_azimuth:
    //   beam propagation  d_s = ( sin_s_w, 0, -mu_sun_water )
    //   scattered leaving d_v = ( sin_v_w cos dphi, sin_v_w sin dphi,
    //                             +mu_view_water )
    // cos psi = dot(d_s, d_v).
    g.cos_scatter = sin_s_w * sin_v_w * std::cos(delta_azimuth_rad)
                  - g.mu_sun_water * g.mu_view_water;
    return g;
}

glm::dvec3 VolumeScatteringFunction(const glm::dvec3& b_molecular,
                                    double b_particulate,
                                    double cos_scatter,
                                    double depolarisation,
                                    double petzold_g) noexcept {
    const double p_mol = RayleighDepolarisedPhase(cos_scatter, depolarisation);
    const double p_par = HenyeyGreenstein(cos_scatter, petzold_g);
    return b_molecular * p_mol + glm::dvec3(std::max(b_particulate, 0.0) * p_par);
}

double RayleighDepolarisedBackscatterFraction(double depolarisation) noexcept {
    // int_{-1}^{0} 2 pi p(mu) dmu with p = (1 + k mu^2) / (4 pi (1 + k/3)).
    // The mu integral gives (1 + k/3)/2, which cancels the normaliser
    // exactly -- so this is 1/2 for every k, i.e. for every depolarisation
    // ratio. Written as the ratio of the two so the cancellation is code
    // and not a comment.
    const double k = (1.0 - depolarisation) / (1.0 + depolarisation);
    return (0.5 * (1.0 + k / 3.0)) / (1.0 + k / 3.0);
}

double HenyeyGreensteinBackscatterFraction(double g) noexcept {
    // Equation (4). The g -> 0 limit is 1/2 (isotropic), taken explicitly
    // because the closed form divides by g.
    const double a = std::fabs(g);
    if (a < 1e-6) return 0.5;
    return (1.0 - g) / (2.0 * g) *
           ((1.0 + g) / std::sqrt(1.0 + g * g) - 1.0);
}

glm::dvec3 BackscatteringCoefficient(const glm::dvec3& b_molecular,
                                     double b_particulate,
                                     double depolarisation,
                                     double petzold_g) noexcept {
    return glm::max(b_molecular, glm::dvec3(0.0)) *
               RayleighDepolarisedBackscatterFraction(depolarisation) +
           glm::dvec3(std::max(b_particulate, 0.0) *
                      HenyeyGreensteinBackscatterFraction(petzold_g));
}

glm::dvec3 EffectiveAttenuation(const glm::dvec3& absorption,
                                const glm::dvec3& b_molecular,
                                double b_particulate,
                                double depolarisation,
                                double petzold_g) noexcept {
    return glm::max(absorption, glm::dvec3(0.0)) +
           BackscatteringCoefficient(b_molecular, b_particulate,
                                     depolarisation, petzold_g);
}

glm::dvec3 SubsurfaceRrsDeep(const glm::dvec3& absorption,
                             const glm::dvec3& b_molecular,
                             double b_particulate,
                             const SingleScatterGeometry& g,
                             double depolarisation,
                             double petzold_g) noexcept {
    const glm::dvec3 beta = VolumeScatteringFunction(
        b_molecular, b_particulate, g.cos_scatter, depolarisation, petzold_g);
    const glm::dvec3 c = EffectiveAttenuation(absorption, b_molecular,
                                              b_particulate, depolarisation,
                                              petzold_g);
    const double     m = g.mu_sun_water + g.mu_view_water;
    if (!(m > 0.0)) return glm::dvec3(0.0);
    return {c.x > 0.0 ? beta.x / (c.x * m) : 0.0,
            c.y > 0.0 ? beta.y / (c.y * m) : 0.0,
            c.z > 0.0 ? beta.z / (c.z * m) : 0.0};
}

glm::dvec3 SubsurfaceRrsSaturationRate(const glm::dvec3& absorption,
                                       const glm::dvec3& b_molecular,
                                       double b_particulate,
                                       const SingleScatterGeometry& g,
                                       double depolarisation,
                                       double petzold_g) noexcept {
    const glm::dvec3 c = EffectiveAttenuation(absorption, b_molecular,
                                              b_particulate, depolarisation,
                                              petzold_g);
    const double     m = g.mu_sun_water + g.mu_view_water;
    if (!(g.mu_sun_water > 0.0)) return glm::dvec3(0.0);
    return c * (m / g.mu_sun_water);
}

glm::dvec3 SubsurfaceRrsMarched(const glm::dvec3& absorption,
                                const glm::dvec3& b_molecular,
                                double b_particulate,
                                const SingleScatterGeometry& g,
                                double depolarisation,
                                double petzold_g,
                                double path_length_m,
                                int    samples) noexcept {
    // The submerged march's loop, verbatim in shape: stratified midpoint
    // samples over [0, T), the eye transmittance evaluated exactly AT the
    // sample (not at the cell start -- see the note in PathTrace.slang on
    // why that is second order rather than first), and the beam's own
    // attenuation from the surface down to the sample's DEPTH.
    if (samples < 1 || !(path_length_m > 0.0)) return glm::dvec3(0.0);
    if (!(g.mu_sun_water > 0.0)) return glm::dvec3(0.0);
    const glm::dvec3 beta = VolumeScatteringFunction(
        b_molecular, b_particulate, g.cos_scatter, depolarisation, petzold_g);
    const glm::dvec3 c  = EffectiveAttenuation(absorption, b_molecular,
                                               b_particulate, depolarisation,
                                               petzold_g);
    const double     dt = path_length_m / static_cast<double>(samples);
    glm::dvec3 sum(0.0);
    for (int i = 0; i < samples; ++i) {
        const double t = (static_cast<double>(i) + 0.5) * dt;
        const double z = t * g.mu_view_water;          // depth of the sample
        // E_perp(z) / E_d(0-) = exp(-c z / mu_s) / mu_s.
        const glm::dvec3 beam = glm::exp(-c * (z / g.mu_sun_water))
                              / g.mu_sun_water;
        const glm::dvec3 eye  = glm::exp(-c * t);
        sum += beta * beam * eye * dt;
    }
    return sum;
}

double FresnelUnpolarised(double cos_i, double n_i, double n_t) noexcept {
    const double ci = std::clamp(cos_i, 0.0, 1.0);
    const double si = std::sqrt(std::max(0.0, 1.0 - ci * ci));
    const double st = n_i * si / n_t;
    if (st >= 1.0) return 1.0;                       // total internal reflection
    const double ct = std::sqrt(std::max(0.0, 1.0 - st * st));
    const double rs = (n_i * ci - n_t * ct) / (n_i * ci + n_t * ct);
    const double rp = (n_i * ct - n_t * ci) / (n_i * ct + n_t * ci);
    return 0.5 * (rs * rs + rp * rp);
}

glm::dvec3 RemoteSensingReflectance(const glm::dvec3& absorption,
                                    const glm::dvec3& b_molecular,
                                    double b_particulate,
                                    const SingleScatterGeometry& g,
                                    double depolarisation,
                                    double petzold_g,
                                    double n) noexcept {
    const glm::dvec3 r_rs = SubsurfaceRrsDeep(
        absorption, b_molecular, b_particulate, g, depolarisation, petzold_g);
    // Downward: E_d(0-) = E_d(0+) (1 - F(th_a)). Upward: the radiance
    // scaling 1/n^2 and the transmittance (1 - F(th_w)) of the SAME
    // interface, which by reciprocity is F evaluated at either conjugate.
    const double t_down = 1.0 - FresnelUnpolarised(g.mu_sun_air, 1.0, n);
    const double t_up   = 1.0 - FresnelUnpolarised(g.mu_view_water, n, 1.0);
    return r_rs * (t_down * t_up / (n * n));
}

double MultipleInternalReflectionGain(double irradiance_reflectance) noexcept {
    const double d = 1.0 - kInternalReflectanceRBar *
                               std::clamp(irradiance_reflectance, 0.0, 0.99);
    return d > 0.0 ? 1.0 / d : 0.0;
}

// --- 6. The published relations the engine is checked against (#305) -------

double GordonSubsurfaceRrs(double bb_over_a_plus_bb) noexcept {
    const double x = std::clamp(bb_over_a_plus_bb, 0.0, 1.0);
    return kGordon1988G1 * x + kGordon1988G2 * x * x;
}

double LeeAboveWaterRrs(double subsurface_rrs) noexcept {
    const double d = 1.0 - kLee2002GamQ * subsurface_rrs;
    return d > 0.0 ? kLee2002T * subsurface_rrs / d : 0.0;
}

namespace {

// Bricaud et al. 1998 A(lambda) / E(lambda), from the tabulation the Ocean
// Optics Web Book redistributes (2 nm grid; this is that grid decimated to
// the wavelengths this engine and the ocean-colour satellites actually use,
// with 443 and 555 linearly interpolated between the neighbouring 2 nm
// entries and marked as such).
//
//   440 / 450 / 490 / 550 / 620 are table entries verbatim.
//   443 is between 442 (0.0514, 0.6298) and 444 (0.0505, 0.6232).
//   555 is between 554 (0.0110, 0.8419) and 556 (0.0107, 0.8438).
struct BricaudRow { double lambda_nm, A, E; };
constexpr BricaudRow kBricaud[] = {
    {412.0, 0.0473,  0.6862},
    {440.0, 0.0520,  0.6350},
    {443.0, 0.05095, 0.6265},   // interpolated 442 <-> 444
    {450.0, 0.0479,  0.6151},
    {490.0, 0.0341,  0.6200},
    {510.0, 0.0232,  0.7060},
    {550.0, 0.0118,  0.8385},
    {555.0, 0.01085, 0.84285},  // interpolated 554 <-> 556
    {620.0, 0.0090,  0.8438},
    {670.0, 0.0199,  0.8177},
};
constexpr std::size_t kBricaudCount = sizeof(kBricaud) / sizeof(kBricaud[0]);

// Linear in lambda between table rows, clamped at both ends. The table is
// coarse away from the satellite bands, which is why this is only ever
// evaluated AT them.
void BricaudAE(double lambda_nm, double& A, double& E) noexcept {
    if (lambda_nm <= kBricaud[0].lambda_nm) {
        A = kBricaud[0].A; E = kBricaud[0].E; return;
    }
    for (std::size_t i = 1; i < kBricaudCount; ++i) {
        if (lambda_nm <= kBricaud[i].lambda_nm) {
            const double t = (lambda_nm - kBricaud[i - 1].lambda_nm) /
                             (kBricaud[i].lambda_nm - kBricaud[i - 1].lambda_nm);
            A = kBricaud[i - 1].A + t * (kBricaud[i].A - kBricaud[i - 1].A);
            E = kBricaud[i - 1].E + t * (kBricaud[i].E - kBricaud[i - 1].E);
            return;
        }
    }
    A = kBricaud[kBricaudCount - 1].A;
    E = kBricaud[kBricaudCount - 1].E;
}

}  // namespace

double BricaudParticleAbsorption(double lambda_nm, double chl_mg_m3) noexcept {
    if (!(chl_mg_m3 > 0.0)) return 0.0;
    double A = 0.0, E = 0.0;
    BricaudAE(lambda_nm, A, E);
    return A * std::pow(chl_mg_m3, E);
}

Case1Iops Case1FromChlorophyll(double lambda_nm, double chl_mg_m3) noexcept {
    Case1Iops out;
    const double chl = std::max(chl_mg_m3, 0.0);
    // --- absorption, MM01 (16) with Bricaud 1998 standing in for the
    //     untabulated A_chl, and the yellow-substance term as the THREE
    //     equations Appendix B insists on rather than the bracket.
    const double a_w    = PureWaterAbsorption(lambda_nm);
    const double a_p    = BricaudParticleAbsorption(lambda_nm, chl);
    const double a_w440 = PureWaterAbsorption(440.0);
    const double a_p440 = BricaudParticleAbsorption(440.0, chl);
    const double a_y440 = 0.2 * (a_w440 + a_p440);                    // (18)
    const double a_y    = a_y440 * std::exp(-0.014 * (lambda_nm - 440.0)); // (17)
    out.a = a_w + a_p + a_y;                                          // (16)
    // --- scattering, MM01 (12) with the lambda^-1 slope the paper states
    const double bp550 = 0.416 * std::pow(std::max(chl, 1e-12), 0.766); // (12)
    out.bp = bp550 * (550.0 / std::max(lambda_nm, 1.0));
    // --- particle BACKscattering, MM01 (13)/(14). Note 0.01 and
    //     (lambda/550)^v -- see the header on why the 0.02 / (550/lambda)
    //     pair is the superseded equation (10) and doubles this.
    const double log_chl = std::log10(std::max(chl, 1e-12));
    const double v = (chl > 2.0) ? 0.0 : 0.5 * (log_chl - 0.3);        // (14)
    const double eff = 0.002 + 0.01 * (0.5 - 0.25 * log_chl) *
                                   std::pow(lambda_nm / 550.0, v);
    out.bbp = std::max(eff, 0.0) * bp550;                              // (13)
    out.bw  = PureSeawaterScattering(lambda_nm, kOceanSalinityPpt);
    out.bb  = 0.5 * out.bw + out.bbp;                                  // (11)
    return out;
}

}  // namespace pt::water

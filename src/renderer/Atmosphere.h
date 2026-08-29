// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Host-side physical atmosphere: the radiometric anchor, the multiple-
// scattering table, and the sky cook that both the GPU and the software
// tracer evaluate.  Planetary P3 follow-up, issue #280.
//
// ===========================================================================
// 1.  THE UNITS AUDIT.  IS THE SUN'S "80" A RADIANCE OR AN IRRADIANCE?
// ===========================================================================
//
// Before #280 the sun entered this engine as the literal `float3(80.0)`,
// carried at five sites in shaders/PathTrace.slang, under a doc comment
// that read:
//
//     "Sun RADIANCE arriving at p, in the engine's tonemap-relative units
//      where 80 is the unattenuated SOLAR CONSTANT."
//
// Those two words name different quantities.  The solar constant is an
// IRRADIANCE, 1360.8 W/m^2 (Kopp & Lean 2011).  Solar RADIANCE is that
// divided by the sun's solid angle, 1360.8 / 6.807e-5 = 2.0e7 W/(m^2 sr).
// The two differ by a factor of ~14 700, so which one the 80 stood for
// decides every absolute brightness downstream by four orders of
// magnitude.  Reading the comment does not settle it.  The arithmetic
// does, and it settles it three independent ways:
//
//   (a) THE NEE SITE.  PathTrace.slang integrates direct sun as
//
//           radiance += throughput * sun_rad * trans * brdf * n_dot_l;
//
//       with NO solid-angle factor anywhere on the line.  For a distant
//       source the reflected radiance is  f_r * L_i * cos(theta) * Omega,
//       and the only way that expression is dimensionally closed is if
//       `sun_rad` is already L_i * Omega -- i.e. the irradiance on a
//       surface facing the sun.  A radiance there would be short by
//       Omega and the sun would light nothing.
//
//   (b) THE DISC SITE.  sunDiscPhysical returns `T * 80 / omega`, and its
//       own comment derives that as "a disc of solid angle Omega carrying
//       irradiance E has radiance E / Omega".  The code therefore already
//       treats the 80 as an E and divides by Omega to get an L.  The two
//       estimators partition one light source and agree, which they could
//       not do if the same symbol were a radiance on one side.
//
//   (c) THE IN-SCATTER SITE.  skyPhysical accumulates
//
//           (sigma_s_R * P_R + sigma_s_M * P_M) * sun_at * dt
//
//       For a delta source, L_i(w) = E_perp * delta(w - w_sun), so the
//       in-scattered radiance is sigma_s * P(mu) * E_perp.  Again an
//       irradiance, again with no solid angle in sight.
//
// ANSWER: 80 is an IRRADIANCE -- a stand-in for the solar constant, used
// consistently as one at every site.  The word "radiance" in the comment,
// and in the function's own name `sunRadianceAt`, was wrong; the code was
// right.  So the unit rebase this file performs is a factor of ~5.7 (see
// below), NOT the factor of 14 700 that a literal reading of the comment
// would have implied.  The function is renamed sunIrradianceAt as part of
// #280 so the name can no longer contradict the arithmetic.
//
// The audit also settles a second, quieter question: is there any unit
// system for this engine to be rebased ONTO, or does #280 have to invent
// one?  It does not have to invent one.  src/engine/Engine.cpp's analytic
// light primitives have been in SI radiometric units since #73 --
// `light_point` is documented "radiant intensity in W/sr", `light_sphere`
// and `light_quad` "surface radiance in W/m^2/sr", with candela and lumen
// authoring converted through the SI 683.002 lm/W definition of the
// candela.  The sun was the one emitter in the engine NOT in those units.
// Rebasing it is therefore a correction toward a convention the engine
// already documents, not a new convention imposed on it.
//
// ===========================================================================
// 2.  THE RGB CONVENTION, AND WHAT "1360.8" BECOMES IN THREE CHANNELS
// ===========================================================================
//
// This renderer has three channels, so a spectral irradiance has to be
// binned.  The convention adopted here, stated once so every later number
// is reproducible:
//
//   THE THREE CHANNELS PARTITION THE SOLAR SPECTRUM.  R + G + B is the
//   total, so a white Lambertian surface facing the sun reflects the
//   measured 1360.8 W/m^2 and no energy is silently created or discarded.
//   The RATIO between the bins is the solar spectral irradiance at the
//   three wavelengths this atmosphere model already samples its cross
//   sections at -- 680 / 550 / 440 nm -- which is the same sampling
//   Bodhaine 1999's Rayleigh triple, Hillaire 2020's aerosol pair and
//   Bruneton 2017's ozone triple are all tabulated against.  Using a
//   different sampling for the source than for the medium would be the
//   inconsistency; using the same one is why this is a convention and not
//   a fudge.
//
// Inputs, both cited, neither derived here:
//
//   Total solar irradiance at 1 AU   1360.8 +/- 0.5 W/m^2
//                                    Kopp & Lean 2011, Geophys. Res. Lett.
//                                    38, L01706, "A new, lower value of
//                                    total solar irradiance".
//   Solar spectral irradiance at
//   680 / 550 / 440 nm               (1.474, 1.8504, 1.91198) W/m^2/nm
//                                    Bruneton 2017 reference implementation
//                                    of "Precomputed Atmospheric
//                                    Scattering" (Bruneton & Neyret 2008),
//                                    its `kSolarIrradiance` sampled at the
//                                    model's three RGB wavelengths.
//
// The partition is then
//
//   E_c = 1360.8 * s_c / (s_R + s_G + s_B),   s = (1.474, 1.8504, 1.91198)
//       = (383.014, 480.914, 496.872) W/m^2,  sum 1360.800.
//
// SANITY CHECK, so this is not merely arithmetic.  Photometric luminance
// of the resulting clear-sky zenith, at the engine's own 683.002 lm/W
// convention and a Rec.709 Y weighting, lands at ~2900 cd/m^2 from single
// scattering alone and ~4000 cd/m^2 once section 3's multiple scattering
// is added.  Measured clear-sky zenith luminance is 2000-6000 cd/m^2.  The
// sun disc lands at 4.6e9 cd/m^2 against a measured ~1.6e9; the residual
// factor of ~3 is the documented single-wavelength (555 nm) approximation
// the engine's photometric light authoring already carries, not a further
// error introduced here.
//
// CONSEQUENCE, STATED RATHER THAN HIDDEN.  Against 80, the new triple is
// 4.79x / 6.01x / 6.21x brighter (5.67x on the mean).  Every surface lit
// by the sun gets that much brighter and slightly cooler, so every
// fixture's exposure moves by the reciprocal of a single documented
// number.  And because the engine converts authored candela through
// 683.002 lm/W -- the monochromatic 555 nm efficacy -- while sunlight's
// broadband efficacy is ~93 lm/W, an authored lamp is now ~7.3x brighter
// relative to the sun than a photometer would measure.  That discrepancy
// is a property of the pre-existing single-wavelength light-authoring
// approximation (Engine.cpp, kLuminousEfficacy555nm), not of this change;
// it was simply invisible while the sun was 5.7x too dim to compare with.
//
// ===========================================================================
// 3.  MULTIPLE SCATTERING
// ===========================================================================
//
// A single-scattering sky is too dark, and #257 measured how much: with
// the compensating r_rayleigh = 30 retired, the noon zenith fell to about
// a seventh of a physically consistent value.  The missing energy is the
// second and higher scattering orders, which for a clear Rayleigh
// atmosphere are a large fraction of the zenith signal.  That is exactly
// what the inflated Rayleigh multiplier was standing in for, which is why
// #280 cannot retire it without supplying the real term.
//
// The term here is Hillaire 2020 section 4 ("A Scalable and Production
// Ready Sky and Atmosphere Rendering Technique", EGSR 2020): a 32x32
// table over (radius, sun-zenith cosine) holding the isotropic
// second-and-higher-order in-scattered radiance at a point in the medium.
// Both of its axes are properties of a POINT, not of a camera, so the
// table is valid at every altitude including outside the shell, and #260's
// ground-to-orbit continuity survives it untouched.
//
// IT IS BUILT ON THE CPU, INTO A STORAGE BUFFER, NOT BY A COMPUTE PASS.
// #257's own recommendation, and it buys two things.  The builder IS the
// software mirror -- one implementation of the physics, evaluated by the
// Metal kernel, the Vulkan kernel and src/rhi_software/SoftwareTracer.cpp
// alike -- and the engine's compute-pipeline count does not move, which
// matters because the first cold-cache MoltenVK pipeline build already
// sits at ~115 s against a 120 s budget (project memory, #257 risk 5).
//
// Everything below is computed in double and stored as float.  The
// shader's cancellation-free altitude machinery (ptRayAltitudeAt, #271)
// and its stable sphere solve (ptSphereRoots, #254/#275) exist because
// those kernels must run in fp32 at |oc| ~ 6.4e6; the host has no such
// constraint, so it uses the precision instead of the workaround.

#ifndef PT_RENDERER_ATMOSPHERE_H
#define PT_RENDERER_ATMOSPHERE_H

#include <array>
#include <cstdint>

namespace pt::atmo {

// --- The radiometric anchor -----------------------------------------------

// Total solar irradiance at 1 AU.  Kopp & Lean 2011, GRL 38 L01706.
inline constexpr double kTotalSolarIrradiance = 1360.8;   // W/m^2

// Solar spectral irradiance at 680 / 550 / 440 nm, W/m^2/nm.  Bruneton
// 2017's reference implementation, sampled at this model's three RGB
// wavelengths.  Used ONLY for the ratio between channels; the absolute
// scale comes from kTotalSolarIrradiance above.
inline constexpr double kSolarSpectralShape[3] = {1.474, 1.8504, 1.91198};

// The partition.  See section 2 of the header: R + G + B == the measured
// total, ratios from the measured spectrum.  Evaluated at compile time so
// the three numbers cannot drift from the two citations they come from.
inline constexpr double kSolarShapeSum =
    kSolarSpectralShape[0] + kSolarSpectralShape[1] + kSolarSpectralShape[2];

inline constexpr float kSolarIrradianceR = static_cast<float>(
    kTotalSolarIrradiance * kSolarSpectralShape[0] / kSolarShapeSum);
inline constexpr float kSolarIrradianceG = static_cast<float>(
    kTotalSolarIrradiance * kSolarSpectralShape[1] / kSolarShapeSum);
inline constexpr float kSolarIrradianceB = static_cast<float>(
    kTotalSolarIrradiance * kSolarSpectralShape[2] / kSolarShapeSum);

// The single number every pre-#280 absolute brightness has to be divided
// by to keep its meaning.  80 was the old stand-in; this is what replaced
// it, on the mean.  Fixture exposures move by exactly its reciprocal, and
// stating it as a named constant is what makes that a derivation rather
// than a retune.
inline constexpr float kLegacySunIrradiance = 80.0f;
inline constexpr float kUnitRebaseScale =
    (kSolarIrradianceR + kSolarIrradianceG + kSolarIrradianceB)
    / (3.0f * kLegacySunIrradiance);

// The same number, rounded to the three figures the shaders carry it at.
// PathTraceMath.slang's kPtLegacySkyScale and SoftwareTracer.cpp's painted
// sky both use this; tests/pt_atmosphere_test.cpp pins the three copies
// equal, and pins that it really is mean(E)/80 rather than a fourth tuned
// constant that happens to be near it.
inline constexpr float kLegacySkyScale = 5.67f;

// --- The medium ------------------------------------------------------------
//
// Field-for-field mirror of PtAtmoBody in shaders/PathTraceMath.slang.
// Every citation lives on the shader side (ptAtmoEarth's header); the
// numbers are repeated here because this file is the host evaluator of the
// same model, and tests/pt_atmosphere_test.cpp pins the two copies against
// each other by counting occurrences in the .slang source.
struct Body {
    double ground_radius    = 0.0;   // R_g [m]
    double top_radius       = 0.0;   // R_t [m]
    double rayleigh_sigma_s[3]{};    // [m^-1] at R_g
    double rayleigh_scale_h = 0.0;   // [m]
    double mie_sigma_s[3]{};         // [m^-1] at R_g
    double mie_sigma_a[3]{};         // [m^-1] at R_g
    double mie_scale_h      = 0.0;   // [m]
    double mie_g            = 0.0;
    double ozone_sigma_a[3]{};       // [m^-1] at the tent peak
    double ozone_center     = 0.0;   // [m]
    double ozone_half_width = 0.0;   // [m]
};

Body Earth(double ground_radius);

// r_rayleigh / r_volumetric_density / r_ozone, applied exactly as
// ptAtmoScale does on the GPU: the aerosol keeps its single-scattering
// albedo as loading changes, the other two are multiples of the real
// column.
Body Scale(Body b, double rayleigh_scale, double mie_sigma_s_abs,
           double ozone_scale);

// Per-altitude coefficients.  Mirror of ptAtmoCoefficients.
void Coefficients(const Body& b, double h,
                  double sigma_s_rayleigh[3],
                  double sigma_s_mie[3],
                  double sigma_t[3]);

// Optical depth along [ro, ro + rd*t], clipped to the shell, by composite
// Simpson.  Mirror of ptAtmoOpticalDepth (which the shader runs at 8
// steps); the host default is finer because it can afford to be.
void OpticalDepth(const Body& b, const double ro[3], const double rd[3],
                  const double centre[3], double t, int steps,
                  double tau_out[3]);

// Transmittance from `p` to space along `sun_dir`, zero when the body
// occludes.  Mirror of sunSlantTransmittance, which calls the shader's
// 8-step Simpson; the default here matches it so the host and the GPU
// agree about the colour of sunlight to within the rule's own error.
void SunSlantTransmittance(const Body& b, const double p[3],
                           const double sun_dir[3], const double centre[3],
                           double out[3], int steps = 8);

// --- Multiple scattering ---------------------------------------------------

inline constexpr int kMsLutWidth  = 32;   // sun-zenith cosine axis
inline constexpr int kMsLutHeight = 32;   // radius axis
inline constexpr int kMsLutTexels = kMsLutWidth * kMsLutHeight;
// float4 per texel: rgb = Psi_ms with the solar irradiance already folded
// in, w = 1 so a shader-side "was this ever built?" check is a compare
// against a value the zero-fill cannot produce.
inline constexpr int kMsLutFloats = kMsLutTexels * 4;

struct MsLutParams {
    double ground_albedo = 0.10;   // r_sky_ground_albedo
    // 64 sphere samples per texel, on an 8x8 lattice -- Hillaire 2020's own
    // count, and the one every production implementation of the technique
    // uses.  The sphere integral is the slower-converging of the two, so
    // this is the axis with the residual quadrature bias; it is a bias in a
    // correction term, not in the leading term, and it is recorded rather
    // than tuned away.
    int    directions    = 64;
    // 32 march steps.  Measured convergence of the blue channel of
    // Psi_ms(1 km, mu_s = 0.9): 12 steps 24.4, 24 steps 22.3, 48 steps
    // 21.1, i.e. ~5% per doubling and falling.  32 sits inside 3% of the
    // 48-step value for two thirds of the cost.
    int    march_steps   = 32;
    // 0 = std::thread::hardware_concurrency().  The table is built by rows
    // and rows are disjoint, so the result does not depend on the thread
    // count: a 1-thread and a 12-thread build are bit-identical.
    int    threads       = 0;
};

// Build the (radius, mu_s) multiple-scattering table.  `out` must hold
// kMsLutFloats floats.  Deterministic: no RNG, no threading-order
// dependence.
void BuildMultiScatterLut(const Body& b, const MsLutParams& p, float* out);

// Bilinear fetch, mirror of ptAtmoMultiScatter in PathTraceMath.slang.
// `r` is the sample's radius from the body centre, `mu_s` the cosine
// between local up and the sun.
void SampleMultiScatter(const float* lut, const Body& b, double r,
                        double mu_s, double out[3]);

// --- The sky cook ----------------------------------------------------------
//
// procSky is a painted gradient with a hand-authored palette.  #280
// replaces the palette with three radiances that are INTEGRATED, once per
// frame, on the host: the sky at the zenith, at the horizon toward the
// sun, and at the horizon away from it.  The painted smoothstep that
// interpolates between them stays -- it is the cheap dome shape, and that
// is what `procedural` is for -- but the three numbers it interpolates are
// now physics with a citation instead of literals with a taste.
//
// The cook carries Rayleigh + ozone + multiple scattering, and NOT the
// aerosol.  Two reasons, both structural: a three-anchor interpolation
// cannot represent the sharp forward lobe an aerosol has around the sun,
// and the primary-ray in-scatter march next door already computes exactly
// that lobe and would double-count anything the cook duplicated.  So the
// split is by frequency content -- the smooth term is cooked, the peaked
// term is marched -- and at the real sea-level aerosol loading the cooked
// term carries the whole of a clear sky's brightness anyway: the vertical
// Rayleigh optical depth in blue is 0.265 against the aerosol's 0.0048.
struct SkyCook {
    float zenith[3]{};        // W/m^2/sr
    float horizon_sun[3]{};   // horizon, azimuth toward the sun
    float horizon_anti[3]{};  // horizon, azimuth away from the sun
};

struct SkyCookParams {
    double sun_elev_sin = 0.0;   // sin(solar elevation) at the observer
    double observer_alt = 0.0;   // metres above the ground radius
    // 256 steps, quadratically distributed.  Three rays, once per frame, on
    // the host -- so the budget is chosen to make the QUADRATURE stop being
    // a term in the answer rather than to hit a cost target.  Sun-side
    // horizon, blue, 60 deg elevation, against a 3 072-step reference of
    // 38.94: 48 steps 43.91 (+12.8%), 96 steps 41.33 (+6.1%), 192 steps
    // 40.08 (+2.9%), 256 steps within 2%.  Measured cost below 0.3 ms.
    int    march_steps  = 256;
};

SkyCook CookSky(const Body& b, const float* ms_lut, const SkyCookParams& p);

// Radiance along one ray through the medium: single scattering with the
// real phase functions, plus the multiple-scattering term.  This is the
// host mirror of skyPhysical, and it is what CookSky evaluates three
// times.  `ro` / `rd` / `centre` in metres, `rd` unit length.
//
// `step_power` shapes the quadrature.  1.0 is uniform in distance, which
// is what shaders/PathTrace.slang's skyPhysical does and what a mirror
// test must ask for.  2.0 places the segment boundaries quadratically,
// concentrating them near the origin, and that is what the cook uses --
// because a HORIZONTAL ray at eye height has a 2 260 km chord with almost
// all of its optical depth in the first few hundred kilometres, and
// uniform stepping over it is badly wrong.  Measured on the sun-side
// horizon at 60 deg solar elevation, blue channel: uniform 48 steps gives
// 61.4 against a 3 072-step reference of 39.8, a 54% overestimate;
// quadratic 48 steps gives it to within a few percent.
//
// The GPU's skyPhysical deliberately keeps the uniform rule: #260 proves
// its ground-to-orbit continuity from the sample distribution being
// independent of camera altitude, and an origin-anchored distribution is
// not.  So the same overestimate is present in mode 4's horizon today.
// That is a finding of #280, recorded here and in the PR, not something
// #280 changes -- fixing it means re-deriving the continuity proof.
void SkyRadiance(const Body& b, const float* ms_lut, const double ro[3],
                 const double rd[3], const double centre[3],
                 const double sun_dir[3], int steps, double mie_g,
                 bool include_mie, double out[3], double step_power = 1.0);

}  // namespace pt::atmo

#endif  // PT_RENDERER_ATMOSPHERE_H

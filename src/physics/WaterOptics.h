// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstddef>

#include <glm/glm.hpp>

namespace pt::water {

// --- Planetary P7 (#261): seawater optics, from the measurements ----------
//
// Everything in this header is a published measurement or a closed-form
// consequence of one. Nothing here is a dial, and the two tunables the
// engine does expose (r_water_absorption_*, r_water_scatter_particulate)
// take their DEFAULTS from these functions rather than from a taste call.
//
// This follows the precedent commit b2111dd set when it replaced fabricated
// Hosek coefficients with the genuine ArHosekSkyModel 1.4a dataset, and the
// one #258 set when it chose ETOPO over a procedural planet. A three-float
// water tint that nobody can trace to a paper is that decision made again.
//
//
// 1. ABSORPTION -- Pope & Fry 1997
//
// R. M. Pope and E. S. Fry, "Absorption spectrum (380-700 nm) of pure
// water. II. Integrating cavity measurements", Applied Optics 36(33),
// 8710-8723 (1997). The definitive integrating-cavity measurement; its
// values in the blue are several times LOWER than every pre-1990
// determination, which is why using an older table makes clear water look
// like tea.
//
// The table below is that paper's data as redistributed by the Oregon
// Medical Laser Center's water-absorption compendium
// (omlc.org/spectra/water/data/pope97.txt), converted from the compendium's
// 1/cm to this engine's metric 1/m, and decimated from its native 2.5 nm to
// 5 nm over 380-700 nm. 65 samples. Two anchors the paper itself quotes,
// asserted in tests/pt_water_optics_test.cpp:
//
//   * the absorption MINIMUM, 0.0044 +/- 0.0006 /m near 418 nm (this grid
//     puts it at 415 nm, 0.00444 /m -- the 2.5 nm grid's true minimum is
//     0.00442 /m at 417.5 nm);
//   * a(700) = 0.624 /m, the red end, 140x the minimum.
//
//
// 2. WHICH THREE NUMBERS AN RGB RENDERER GETS, AND WHY THESE THREE
//
// An RGB channel is a band, not a wavelength, so reducing a spectrum to a
// triple is a CONVENTION and has to be stated rather than assumed.
//
// The obvious-looking rigorous route -- integrate the transmitted spectrum
// against the CIE 1931 2-degree observer, convert to linear Rec.709, and
// read off the per-channel decay rate -- was tried first and REJECTED on
// evidence. The XYZ->linear-sRGB matrix has large negative entries (the B
// row carries -0.2040 on ybar) and zbar overlaps the wavelengths where
// water is at its most transparent, so the blue channel's weighted mean
// comes out at -0.00225 /m: a NEGATIVE absorption, i.e. water that
// amplifies blue. That is not a rounding problem, it is the sRGB gamut
// being smaller than the spectral locus, and no reference depth fixes it.
// Recorded here so it is not re-attempted.
//
// What this file does instead is sample the table at three named
// wavelengths and say so:
//
//   R = 620 nm    G = 550 nm    B = 450 nm
//
// which are the representative RGB wavelengths issue #261 and the phase
// study both quote, and which are close to the sRGB primaries' dominant
// wavelengths (611 / 549 / 464 nm). The convention is theirs; the VALUES
// are the paper's, and that distinction is the whole point of this header.
//
// ONE CORRECTION TO THE ISSUE. #261's table gives a(620) = 0.245 /m and a
// 1/e depth of 4.1 m. Pope & Fry's tabulated value at 620 nm is 0.2755 /m
// (1/e = 3.63 m); 0.245 /m falls at about 602 nm on their curve, so the
// issue's row appears to be a transposed 602 -> 620. The G and B rows match
// the paper exactly. This file uses the paper.
//
//
// 3. SCATTERING -- Morel 1974, via Einstein-Smoluchowski
//
// Molecular scattering by pure seawater, in the form universally quoted
// after A. Morel, "Optical properties of pure water and pure sea water", in
// Optical Aspects of Oceanography (1974), and restated as equation (1) of
// X. Zhang, L. Hu and M.-X. He, "Scattering by pure seawater: effect of
// salinity", Optics Express 17(7) 5698-5710 (2009):
//
//     beta_w(90 deg, lambda, S) = 1.38e-4 * (lambda / 500 nm)^-4.32
//                                        * (1 + 0.3 * S / 37)   [1/m/sr]
//
// with a depolarisation ratio d = 0.09. The TOTAL scattering coefficient
// follows in closed form rather than from a second table. A Rayleigh phase
// function with depolarisation d is
//
//     beta(theta) = beta(90 deg) * [1 + ((1-d)/(1+d)) cos^2 theta]
//
// and integrating it over the sphere gives
//
//     b = 4 pi * beta(90 deg) * [1 + (1/3)(1-d)/(1+d)]
//       = 16.0648 * beta(90 deg)        at d = 0.09
//
// which is the "b_w = 16.06 beta_w(90)" that ocean optics uses as a
// constant. It is derived here, not quoted, so the depolarisation ratio and
// the factor cannot drift apart.
//
// At S = 35 ppt that gives b_w(500) = 2.846e-3 /m, which is the ~0.0029 /m
// #261 quotes from Smith & Baker 1981 and is the number this file produces.
//
// THE PHASE FUNCTION IS NOT ONE FUNCTION, and that matters for the look.
// Molecular scattering is Rayleigh-like and very nearly symmetric
// (g = 0 for the d = 0 case, and 0 exactly for the d = 0.09 case too, since
// the cos^2 term is even). Particulate scattering is violently forward, and
// the standard is the Petzold "average particle" phase function with
// g = 0.924 (T. J. Petzold, "Volume scattering functions for selected ocean
// waters", SIO ref. 72-78, 1972; asymmetry parameter as tabulated by
// Mobley). Collapsing both into one HG lobe at g = 0.92 -- which #261
// suggests -- would make PURE water forward-scattering, and pure water's
// near-isotropic blue scatter is exactly why a clear ocean is blue rather
// than merely dark. The shader therefore carries two species with two phase
// functions, in the same shape the atmosphere march already uses for Mie
// and Rayleigh.
//
//
// 4. WHERE THE WATER GOES BLACK
//
// The single-scatter albedo of pure seawater at these three wavelengths is
// 0.0041 / 0.0323 / 0.3273 -- red scatters essentially nothing and blue
// scatters a third of what it removes. The optical-black depth (below which
// the least-attenuated channel transmits under 1/255, i.e. under one 8-bit
// level) is ln(255) / min(a + b) = 401 m in pure seawater. #261's ~185 m is
// the same calculation at a coastal c_B of 0.03 /m. Both are used: the
// engine computes it from whatever coefficients are actually in force, so
// the march span is derived per-frame rather than being a constant.

// --- The Pope & Fry table -------------------------------------------------

// 380 nm to 700 nm inclusive, 5 nm spacing.
inline constexpr double kPopeFryLambdaMinNm = 380.0;
inline constexpr double kPopeFryLambdaMaxNm = 700.0;
inline constexpr double kPopeFryStepNm      = 5.0;
inline constexpr std::size_t kPopeFrySamples = 65;

// Pure-water absorption coefficient, 1/m. Index i is
// kPopeFryLambdaMinNm + i * kPopeFryStepNm.
const double* PopeFryAbsorptionTable() noexcept;

// Linear interpolation of the table. Clamped to the measured range -- the
// paper measured 380-700 nm and this does not extrapolate a spectrum that
// rises by two orders of magnitude just past the top end.
double PureWaterAbsorption(double lambda_nm) noexcept;

// --- The three named wavelengths -----------------------------------------

inline constexpr double kLambdaRedNm   = 620.0;
inline constexpr double kLambdaGreenNm = 550.0;
inline constexpr double kLambdaBlueNm  = 450.0;

// Pure-water absorption as the engine's RGB triple, 1/m.
glm::dvec3 PureWaterAbsorptionRgb() noexcept;

// --- Morel molecular scattering ------------------------------------------

// Morel's beta_w(90 deg) fit, 1/m/sr, at salinity S in ppt.
double MorelBeta90(double lambda_nm, double salinity_ppt) noexcept;
// The Rayleigh-with-depolarisation sphere integral, b / beta(90 deg).
// A pure function of the depolarisation ratio; 16.0648 at d = 0.09.
double RayleighDepolarisedSphereFactor(double depolarisation) noexcept;
inline constexpr double kSeawaterDepolarisation = 0.09;
// Mean ocean salinity, ppt. UNESCO 1981 practical salinity; the global
// open-ocean mean is 34.7 and 35 is the conventional round figure the
// Morel fit's own salinity term is quoted against.
inline constexpr double kOceanSalinityPpt = 35.0;

// Total molecular scattering coefficient of pure seawater, 1/m.
double PureSeawaterScattering(double lambda_nm, double salinity_ppt) noexcept;
// ...as the engine's RGB triple at kOceanSalinityPpt.
glm::dvec3 PureSeawaterScatteringRgb() noexcept;

// --- Phase functions ------------------------------------------------------

// Petzold "average particle" asymmetry parameter.
inline constexpr double kPetzoldAsymmetry = 0.924;

// Normalised Rayleigh-with-depolarisation phase function, 1/sr.
// Integrates to 1 over the sphere by construction.
double RayleighDepolarisedPhase(double cos_theta, double depolarisation) noexcept;

// --- Derived depths -------------------------------------------------------

// Depth at which a channel's transmittance falls to 1/e, in metres.
glm::dvec3 AttenuationLengths(const glm::dvec3& extinction) noexcept;

// Depth below which every channel transmits less than one 8-bit level, so
// the water column is optically black. ln(255) / min(extinction).
double OpticalBlackDepth(const glm::dvec3& extinction) noexcept;

// --- 5. WATER-LEAVING RADIANCE (#305) -------------------------------------
//
// Everything above describes what a column of water DOES to light. None of
// it describes what a camera OUTSIDE the water sees, and that omission is
// what #305 measured: with every in-scatter term in the megakernel gated on
// the camera being submerged, a ray refracting in from outside got Beer's
// law and no source, so the water body contributed no colour at all and
// every bit of blue in the rendered ocean was atmospheric Rayleigh.
//
// Ocean colour IS a source term. It is the upwelling radiance that leaves
// the surface after light has entered, been absorbed, and been scattered
// back out -- "water-leaving radiance" L_w, and normalised by the
// downwelling irradiance it is the remote-sensing reflectance R_rs, the
// quantity every ocean-colour satellite retrieves and the quantity this
// engine can therefore be checked against instead of eyeballed.
//
//
// 5.1 THE TRANSPORT, DERIVED
//
// Take a horizontally homogeneous, semi-infinite column with absorption a,
// scattering b, beam attenuation c = a + b, and volume scattering function
// beta(psi) [1/m/sr]. Light enters as a collimated beam refracted to nadir
// cosine mu_s; the camera looks along an in-water direction of nadir cosine
// mu_v. Writing the beam's irradiance on a plane PERPENDICULAR to itself as
// E_perp(z) = E_perp(0) exp(-c z / mu_s), the source function toward the
// viewer is beta(psi) E_perp(z), and integrating it up the viewing path
// (path length t, depth z = t mu_v, eye transmittance exp(-c t)) gives
//
//     L_u(0-) = E_perp(0) beta(psi) INT_0^T exp(-c t (1 + mu_v/mu_s)) dt
//             = E_d(0-) beta(psi) [1 - exp(-k T)] / (c (mu_s + mu_v))
//     k       = c (mu_s + mu_v) / mu_s                                  (1)
//
// with E_d(0-) = E_perp(0) mu_s the downwelling irradiance just below the
// surface. T -> infinity is the optically deep limit; finite T is a bottom,
// and it is the SAME expression, which is why the engine has no shallow /
// deep regime switch and therefore no transition for the two to disagree
// at. Equation (1) is exactly the integral the submerged in-scatter march
// in PathTrace.slang evaluates numerically, so the two agree by
// construction and the ladder in tests/pt_water_optics_test.cpp measures
// how fast.
//
//
// 5.2 THE TWO INTERFACE FACTORS THAT ARE EASY TO GET WRONG
//
// DOWNWARD. Radiance across an interface obeys L_w = L_a (1 - F) n^2, and
// the solid angle compresses. But the compression is NOT 1/n^2:
// differentiating Snell's law n_a sin th_a = n_w sin th_w gives
//
//     dOmega_w / dOmega_a = mu_a / (n^2 mu_w)                           (2)
//
// so the beam irradiance in water is E_perp,w = E_perp,a (1 - F) mu_a/mu_s
// -- there is a surviving mu_a/mu_s that the naive "the n^2 cancels"
// argument drops. Equivalently, and as the textbook states it, the
// horizontal downwelling irradiance obeys E_d(0-) = E_d(0+) (1 - F): the
// beam bends toward the vertical, so the same power crosses a larger
// in-water cross-section but lands on the same horizontal area. The
// engine's submerged march carried the cancelled version until #305; the
// factor is 1 at the zenith and 0.66 at a grazing sun.
//
// UPWARD. Radiance leaving water for air scales by 1/n^2, and beyond the
// critical angle asin(1/n) = 48.3 deg it does not leave at all. TIR cannot
// bite a camera in air -- the in-water direction is obtained by refracting
// a real air-side ray, so it is inside Snell's window by construction --
// but the light that DOES total-internally-reflect and eventually escapes
// on a later attempt is dropped by this single-scattering treatment. That
// is the classical multiple-internal-reflection factor 1/(1 - r_bar R),
// with r_bar ~ 0.48 the internal Fresnel reflectance of the upwelling
// field; see MultipleInternalReflectionGain() below, which bounds it.
//
//
// 5.3 WHAT IS APPROXIMATED, AND BY HOW MUCH
//
//   * SINGLE SCATTERING ONLY. Photons scattered twice are not counted.
//     The error scales with the single-scattering albedo b/c, which for
//     pure seawater in the blue is 0.33 and in the red 0.004. The check
//     against the published Gordon et al. 1988 semi-analytic model (which
//     is a fit to full Monte-Carlo radiative transfer) is what bounds
//     this, and it is asserted, not asserted-about.
//   * DIRECT SUN ONLY. Sky radiance entering the water and scattering back
//     out is not part of this term. Measured against the diffuse fraction
//     of E_d, this is the dominant remaining gap at low sun.
//   * HORIZONTALLY HOMOGENEOUS. No chlorophyll gradient with depth, no
//     deep chlorophyll maximum.
//   * A FLAT INTERFACE FOR THE STRATIFICATION. mu_s and mu_v are taken
//     against the local vertical, not the wave normal, because the medium
//     is stratified in depth. The refraction itself uses the wave normal.

// The refraction geometry a single-scatter evaluation needs, all cosines
// against the local vertical, all angles in radians.
struct SingleScatterGeometry {
    double mu_sun_air   = 1.0;  // cos of the solar zenith angle, in air
    double mu_sun_water = 1.0;  // ...of the refracted beam, in water
    double mu_view_air  = 1.0;  // cos of the view nadir angle, in air
    double mu_view_water= 1.0;  // ...of the refracted view ray, in water
    double cos_scatter  = -1.0; // cos of the in-water scattering angle
};

// Refract a (sun, view) pair through a flat interface of index `n`.
// `delta_azimuth_rad` is the azimuth between the solar and viewing planes;
// 0 puts the viewer looking toward the sun's azimuth (forward scatter),
// pi puts the sun behind the viewer, which is the geometry ocean-colour
// radiometry is quoted at.
SingleScatterGeometry RefractSunAndView(double sun_zenith_rad,
                                        double view_zenith_rad,
                                        double delta_azimuth_rad,
                                        double n) noexcept;

// beta(psi), 1/m/sr: the volume scattering function of the two-species
// medium the shader carries -- molecular (Rayleigh with depolarisation,
// per channel) plus particulate (Henyey-Greenstein at the Petzold
// asymmetry, grey).
glm::dvec3 VolumeScatteringFunction(const glm::dvec3& b_molecular,
                                    double b_particulate,
                                    double cos_scatter,
                                    double depolarisation,
                                    double petzold_g) noexcept;

// Equation (1)'s bracket-free amplitude, divided by E_d(0-): the
// subsurface remote-sensing reflectance r_rs = L_u(0-) / E_d(0-), 1/sr,
// for an optically DEEP column.
glm::dvec3 SubsurfaceRrsDeep(const glm::dvec3& absorption,
                             const glm::dvec3& b_molecular,
                             double b_particulate,
                             const SingleScatterGeometry& g,
                             double depolarisation,
                             double petzold_g) noexcept;

// Equation (1)'s saturation rate k, 1/m of PATH (not of depth). A column
// whose viewing path is T metres long returns
// SubsurfaceRrsDeep * (1 - exp(-k T)).
glm::dvec3 SubsurfaceRrsSaturationRate(const glm::dvec3& absorption,
                                       const glm::dvec3& b_molecular,
                                       double b_particulate,
                                       const SingleScatterGeometry& g) noexcept;

// The same integral done the slow way: N stratified steps of the submerged
// march's integrand along a viewing path of `path_length_m`. This is the
// reference the closed form is bounded against, and it is the SHAPE of
// the loop PathTrace.slang runs for a submerged camera.
glm::dvec3 SubsurfaceRrsMarched(const glm::dvec3& absorption,
                                const glm::dvec3& b_molecular,
                                double b_particulate,
                                const SingleScatterGeometry& g,
                                double depolarisation,
                                double petzold_g,
                                double path_length_m,
                                int    samples) noexcept;

// Above-water remote-sensing reflectance R_rs = L_w(0+) / E_d(0+), 1/sr.
// Applies the downward interface loss (1 - F(th_a)), the upward one
// (1 - F(th_w)) and the 1/n^2 radiance scaling -- the three factors the
// shader spells out at the MAT_WATER interface.
glm::dvec3 RemoteSensingReflectance(const glm::dvec3& absorption,
                                    const glm::dvec3& b_molecular,
                                    double b_particulate,
                                    const SingleScatterGeometry& g,
                                    double depolarisation,
                                    double petzold_g,
                                    double n) noexcept;

// Unpolarised Fresnel reflectance at a dielectric interface, exact
// (Fresnel's equations, not Schlick). cos_i is measured in the medium of
// index n_i.
double FresnelUnpolarised(double cos_i, double n_i, double n_t) noexcept;

// The gain the neglected multiple internal reflections would add:
// 1 / (1 - r_bar * R), with R the irradiance reflectance just below the
// surface and r_bar the mean internal Fresnel reflectance of the upwelling
// field. Austin (1974) / Morel & Prieur (1977) quote r_bar = 0.48 for a
// flat surface; this is the term's SIZE, computed so the omission is
// bounded rather than hand-waved.
inline constexpr double kInternalReflectanceRBar = 0.48;
double MultipleInternalReflectionGain(double irradiance_reflectance) noexcept;

}  // namespace pt::water

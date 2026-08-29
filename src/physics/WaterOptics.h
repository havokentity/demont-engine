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

// --- 5.4 THE ATTENUATION THAT GOVERNS A REFLECTANCE ------------------------
//
// Equation (1) is written with a beam attenuation. WHICH one is the single
// most consequential choice in the whole transport, and getting it wrong is
// not a small error: measured against Gordon et al. (1988)'s Monte-Carlo-
// fitted semi-analytic model, using the full c = a + b puts the engine a
// factor of FOUR below the published reflectance as soon as any particulate
// load is present, because it charges the reflectance for every
// forward-scattered photon as if that photon were lost.
//
// It is not lost. A photon scattered 10 degrees off a downward beam is
// still going down; a photon scattered 10 degrees off an upwelling ray
// still reaches the sensor. What removes a photon from the reflectance is
// BACKSCATTERING, and the attenuation that governs the diffuse field is
// therefore
//
//     c_eff = a + b_b                                                   (3)
//
// This is the quasi-single-scattering approximation -- H. R. Gordon,
// "Simple calculation of the diffuse reflectance of the ocean", Applied
// Optics 12(12), 2803-2804 (1973) -- and it is why ocean optics quotes
// K_d ~ (a + b_b)/mu rather than c/mu. Substituting (3) into (1) brings
// the engine to within 0.5% of Gordon's published relation for pure
// seawater in the blue; the strict single-scattering form is 14% low there
// and 4x low at an open-ocean particle load. Measured in
// tests/pt_water_optics_test.cpp, both ways, so the choice is a
// measurement and not an assertion.
//
// THE SAME c_eff GOES INTO THE SUBMERGED MARCH. It has to: a camera just
// above and just below the surface would otherwise disagree by 31% in the
// blue across an interface where the only physical discontinuity is the
// (1 - F)/n^2 of the interface itself.
//
// b_b IS DERIVED FROM THE PHASE FUNCTIONS ALREADY IN USE, not tabulated.
// The Rayleigh-with-depolarisation phase function is even in cos(theta),
// so its backscatter fraction is EXACTLY 1/2 for every depolarisation
// ratio. Henyey-Greenstein integrates in closed form to
//
//     B(g) = (1 - g) / (2 g) * [ (1 + g)/sqrt(1 + g^2) - 1 ]            (4)
//
// which at the Petzold asymmetry g = 0.924 is 0.016989 -- against the
// 0.018 that Petzold's measured "average particle" volume scattering
// function actually integrates to (Mobley, Light and Water, table 3.10),
// a 6% agreement that is a check on the HG summary rather than a
// coincidence.

// Backscatter fraction of the depolarised Rayleigh phase function.
// Exactly 1/2, for every depolarisation ratio, because the phase function
// is even in cos(theta). Returned as a function rather than a constant so
// the property is stated where it is used.
double RayleighDepolarisedBackscatterFraction(double depolarisation) noexcept;
// Backscatter fraction of Henyey-Greenstein, equation (4).
double HenyeyGreensteinBackscatterFraction(double g) noexcept;
// b_b, 1/m: the backscattering coefficient of the engine's two-species
// medium.
glm::dvec3 BackscatteringCoefficient(const glm::dvec3& b_molecular,
                                     double b_particulate,
                                     double depolarisation,
                                     double petzold_g) noexcept;
// c_eff = a + b_b, equation (3), 1/m.
glm::dvec3 EffectiveAttenuation(const glm::dvec3& absorption,
                                const glm::dvec3& b_molecular,
                                double b_particulate,
                                double depolarisation,
                                double petzold_g) noexcept;

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
                                       const SingleScatterGeometry& g,
                                       double depolarisation,
                                       double petzold_g) noexcept;

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

// --- 6. THE PUBLISHED RELATIONS THE ENGINE IS CHECKED AGAINST (#305) -------
//
// These are NOT part of the renderer. They are the literature's own answers
// to the same question, kept here so the acceptance check in
// tests/pt_water_optics_test.cpp compares against a published model rather
// than against a number somebody liked.
//
// 6.1 GORDON ET AL. 1988. H. R. Gordon, O. B. Brown, R. H. Evans, J. W.
// Brown, R. C. Smith, K. S. Baker and D. K. Clark, "A semianalytic radiance
// model of ocean color", J. Geophys. Res. 93(D9), 10909-10924 (1988). The
// SUBSURFACE remote-sensing reflectance r_rs = L_u(0-)/E_d(0-), in 1/sr,
// as a quadratic in X = b_b/(a + b_b):
//
//     r_rs = 0.0949 X + 0.0794 X^2
//
// The coefficients are quoted here from IOCCG Report 5, "Remote Sensing of
// Inherent Optical Properties: Fundamentals, Tests of Algorithms, and
// Applications" (ed. Z.-P. Lee, 2006), equation (8.1), and from Lee, Carder
// and Arnone 2002 p. 5757, because the 1988 paper itself is not openly
// accessible and quoting an equation number in it would be quoting
// something unread. STATED CONDITIONS: nadir-viewed radiance, oceanic Case
// 1 water, and the fit is over a RANGE of solar zenith angles rather than
// one geometry (Park and Ruddick, Applied Optics 44(7), 1236-1249, 2005,
// section 1) -- so this is the right thing to compare an engine to at
// several sun angles, and the wrong thing to compare it to at one and call
// exact. It is a fit to full Monte-Carlo radiative transfer, so it carries
// the multiple scattering the engine's single-backscattering form does not,
// and the ratio between them is the size of that omission.
inline constexpr double kGordon1988G1 = 0.0949;   // 1/sr
inline constexpr double kGordon1988G2 = 0.0794;   // 1/sr
double GordonSubsurfaceRrs(double bb_over_a_plus_bb) noexcept;

// 6.2 LEE ET AL. 2002. Z. P. Lee, K. L. Carder and R. A. Arnone,
// "Deriving inherent optical properties from water color: a multiband
// quasi-analytical algorithm for optically deep waters", Applied Optics
// 41(27), 5755-5772 (2002), equation (8), rearranged:
//
//     r_rs = R_rs / (T + gamma Q R_rs)   ->   R_rs = T r_rs / (1 - gamma Q r_rs)
//     T ~ 0.52        gamma Q ~ 1.7
//
// WHAT THE TWO CONSTANTS ARE, because both are routinely misdescribed.
// T is NOT one transmittance: the paper defines it as t^- t^+ / n^2, an
// upward RADIANCE transmittance times a downward IRRADIANCE transmittance
// over n^2 -- which is exactly the three-factor product this engine
// computes from Fresnel at the interface, and 0.98^2/1.34^2 = 0.535 is the
// sanity check the paper's own 0.52 is measured against. gamma Q is NOT
// the internal reflection coefficient alone: gamma is ~0.48 and Q, the
// upwelling irradiance-to-radiance ratio below the surface, is 3-6. Both
// numbers are empirical HYDROLIGHT fits for OPTICALLY DEEP water and a
// NADIR-VIEWING sensor.
inline constexpr double kLee2002T     = 0.52;
inline constexpr double kLee2002GamQ  = 1.7;
double LeeAboveWaterRrs(double subsurface_rrs) noexcept;

// 6.3 CASE 1 WATER FROM CHLOROPHYLL. Morel, A., and S. Maritorena,
// "Bio-optical properties of oceanic waters: A reappraisal", J. Geophys.
// Res. 106(C4), 7163-7180 (2001).
//
//   a(lambda)   = a_w + a_p + a_y                                  (16)
//   a_y(440)    = 0.2 [ a_w(440) + a_p(440) ]                      (18)
//   a_y(lambda) = a_y(440) exp[-0.014 (lambda - 440)]              (17)
//   b_p(550)    = 0.416 Chl^0.766                                  (12)
//   b_bp(lambda)= { 0.002 + 0.01 [0.5 - 0.25 log10 Chl]
//                              (lambda/550)^v } b_p(550)           (13)
//   v           = 0.5 (log10 Chl - 0.3),  0.02 < Chl < 2           (14)
//                 0,                      Chl > 2
//   b_b(lambda) = 0.5 b_w(lambda) + b_bp(lambda)                   (11)
//
// THREE THINGS THAT ARE EASY TO GET WRONG HERE, and MM01's own Appendix B
// exists because the first of them was got wrong in the literature for a
// decade:
//
//  1. THE YELLOW-SUBSTANCE TERM IS THREE EQUATIONS, NOT A BRACKET. The
//     widely-copied form a = [a_w + 0.06 A_c C^0.65][1 + 0.2 exp(-0.014
//     (lambda - 440))] is the one MM01 Appendix B calls "mistakenly
//     expressed through only two equations". a_y is 20% of the water plus
//     algal absorption AT 440 nm, then propagated spectrally by (17) --
//     not 20% of the local value at every wavelength. A consequence MM01
//     states explicitly: this does not reduce to pure water at Chl = 0, a
//     background remains.
//  2. THE BACKSCATTER COEFFICIENT IN (13) IS 0.01, NOT 0.02, and the
//     spectral factor is (lambda/550)^v, not (550/lambda). The 0.02 form
//     is MM01's own SUPERSEDED equation (10); the paper resets the maximal
//     backscattering efficiency to 1% in the text. Using the old pair
//     doubles the particle backscatter.
//  3. A_chl(lambda) IS NOT TABULATED ANYWHERE ACCESSIBLE. MM01 gives only
//     A_chl(440) = 1 and inherits the spectrum from Prieur and
//     Sathyendranath 1981, which is paywalled with no repository copy. So
//     the algal absorption here is taken from the model that SUPERSEDED
//     it and IS tabulated -- Bricaud, A., A. Morel, M. Babin, K. Allali
//     and H. Claustre, "Variations of light absorption by suspended
//     particles with chlorophyll a concentration in oceanic (case 1)
//     waters", J. Geophys. Res. 103(C13), 31033-31044 (1998), in the form
//     a_p(lambda) = A(lambda) Chl^E(lambda), with A and E read from the
//     dataset the Ocean Optics Web Book redistributes. The two agree to
//     10% at 440 nm over the Case 1 range (MM01's 0.06 C^0.65 against
//     Bricaud's 0.0520 C^0.635), which is the cross-check that the
//     substitution is a substitution and not a different model.

// Bricaud et al. 1998 algal particle absorption, 1/m.
double BricaudParticleAbsorption(double lambda_nm, double chl_mg_m3) noexcept;

// The IOPs a Case 1 water of this chlorophyll concentration has.
struct Case1Iops {
    double a  = 0.0;   // total absorption, 1/m
    double bp = 0.0;   // particle scattering, 1/m
    double bbp= 0.0;   // particle BACKscattering, 1/m (MM01 eq. 13)
    double bw = 0.0;   // pure seawater molecular scattering, 1/m
    double bb = 0.0;   // total backscattering, 1/m  (MM01 eq. 11)
};
Case1Iops Case1FromChlorophyll(double lambda_nm, double chl_mg_m3) noexcept;

// Global mean open-ocean surface chlorophyll, mg/m^3. Antoine, D., J.-M.
// Andre and A. Morel, "Oceanic primary production: 2. Estimation at global
// scale from satellite (Coastal Zone Color Scanner) chlorophyll", Global
// Biogeochem. Cycles 10(1), 57-69 (1996) -- quoted at second hand from
// Werdell and Bailey, Remote Sens. Environ. 98(1), 122-140 (2005) section
// 3.2, "the global ocean mean of 0.19 mg m-3 reported in Antoine and
// co-authors (1996)", because the 1996 paper is closed access. FLAGGED as
// a secondary citation: whether it is an arithmetic or geometric mean, and
// whether it is area-weighted, could not be verified at the primary source.
inline constexpr double kGlobalMeanChlorophyllMgM3 = 0.19;

}  // namespace pt::water

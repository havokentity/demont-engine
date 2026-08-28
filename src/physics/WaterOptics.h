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

}  // namespace pt::water

// SPDX-License-Identifier: MIT
// -----------------------------------------------------------------------------
// Hosek-Wilkie (2012) analytic sky-dome cook + per-direction radiance eval.
//
// This is a faithful, header-only port of the reference "cook" step from the
// ArHosekSkyModel implementation (version 1.4a) that accompanies:
//
//   L. Hosek and A. Wilkie, "An Analytic Model for Full Spectral Sky-Dome
//   Radiance", ACM SIGGRAPH 2012. Reference under 3-clause BSD (see
//   HosekSkyModelData.h for the upstream copyright + the embedded dataset).
//
// The "cook" turns (turbidity, ground albedo, solar elevation) -- all three
// per-frame uniforms -- into the 9 Perez-style shape coefficients A..I plus a
// radiance magnitude L_M, PER RGB CHANNEL, by a bilinear blend (over turbidity
// and albedo) of quintic-Bezier interpolations (over solar elevation) of the
// embedded dataset. It is cheap and runs ONCE per frame on the CPU. The
// per-pixel shape function F(theta, gamma) is evaluated separately (on the GPU
// in PathTrace.slang::hosekSky, and on the CPU here for the software backend),
// fed the cooked coefficients -- exactly the reference architecture (cook once,
// evaluate per view direction).
//
// This replaces the previous hand-authored coefficient tables (which were not
// a fit to any published dataset and crushed the blue channel at the zenith).
// Every number now traces back to the reference dataset in HosekSkyModelData.h.
// -----------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>

#include "HosekSkyModelData.h"

namespace pt::hosek {

// Single scalar mapping the reference model's radiance units into the
// engine's linear-HDR working space. The reference RGB dataset returns
// calibrated relative radiance; this constant sets the absolute exposure
// so a clear daytime dome reads at a sensible brightness under the golden
// fixtures' r_exposure. It is channel-UNIFORM (so it never distorts hue)
// and turbidity/elevation-INDEPENDENT (a pure units constant, not a
// per-config fudge). Both the GPU push (Engine.cpp) and the software
// tracer multiply the cooked radiance by this before use, so they stay in
// lockstep. Tune here if the overall sky brightness needs adjusting.
inline constexpr double kHosekRadianceScale = 1.0;

// Cooked per-frame state. cfg[channel][0..8] hold the 9 shape coefficients in
// the reference index order (see RadianceInternal below for the mapping):
//   [0]=A [1]=B [2]=C [3]=D [4]=E [5]=F [6]=G [7]=I(zenith) [8]=H(Mie g).
// rad[channel] is the channel radiance magnitude L_M. Channels are R,G,B.
struct Cooked {
    double cfg[3][9];
    double rad[3];
};

// Quintic Bezier over 6 control points at parameter t in [0,1]. `base` is the
// index of control point 0 in `m`; successive control points are `stride`
// doubles apart (9 for the config dataset -- one per param -- and 1 for the
// radiance dataset). Mirrors the reference exactly.
inline double Quintic(const double* m, int base, int stride, double t) {
    const double it = 1.0 - t;
    return          it * it * it * it * it              * m[base + 0 * stride]
         +  5.0 * it * it * it * it * t                 * m[base + 1 * stride]
         + 10.0 * it * it * it * t * t                  * m[base + 2 * stride]
         + 10.0 * it * it * t * t * t                   * m[base + 3 * stride]
         +  5.0 * it * t * t * t * t                    * m[base + 4 * stride]
         +        t * t * t * t * t                      * m[base + 5 * stride];
}

// Bilinear (albedo x turbidity) blend of the quintic elevation interpolation
// for a single scalar coefficient. `dataset` is the flattened per-channel
// table; `param_off` selects the coefficient within a turbidity bin (0..8 for
// the 9-config dataset, always 0 for the 1-wide radiance dataset); `param_n`
// is the number of coefficients per (albedo, turbidity, control-point) cell
// (9 for config, 1 for radiance). Reproduces ArHosekSkyModel_Cook*Configuration.
inline double CookScalar(const double* dataset, int param_off, int param_n,
                         int int_turb, double turb_rem, double albedo, double se) {
    const int bin = param_n * 6;                 // doubles per turbidity bin
    const int alb1 = bin * 10;                   // start of the albedo=1 block
    double v =
        (1.0 - albedo) * (1.0 - turb_rem) *
            Quintic(dataset, bin * (int_turb - 1) + param_off, param_n, se)
      + (albedo)       * (1.0 - turb_rem) *
            Quintic(dataset, alb1 + bin * (int_turb - 1) + param_off, param_n, se);
    if (int_turb != 10) {
        v += (1.0 - albedo) * (turb_rem) *
                 Quintic(dataset, bin * (int_turb) + param_off, param_n, se)
           + (albedo)       * (turb_rem) *
                 Quintic(dataset, alb1 + bin * (int_turb) + param_off, param_n, se);
    }
    return v;
}

// Cook the 9 coefficients + radiance magnitude for all three RGB channels.
// turbidity is clamped to [1,10], albedo to [0,1], and solar elevation (the
// sun's angle ABOVE the horizon, in radians) to [0, pi/2] so the dataset is
// never indexed or pow()-ed out of its valid domain (below-horizon suns are a
// night case the caller handles separately).
inline Cooked Cook(double turbidity, double albedo, double solar_elev_rad) {
    constexpr double kHalfPi = 1.5707963267948966;
    turbidity      = std::clamp(turbidity, 1.0, 10.0);
    albedo         = std::clamp(albedo, 0.0, 1.0);
    solar_elev_rad = std::clamp(solar_elev_rad, 0.0, kHalfPi);

    const int    int_turb = static_cast<int>(turbidity);
    const double turb_rem = turbidity - static_cast<double>(int_turb);
    // Reference elevation warp: cube-root packs resolution near the horizon.
    const double se = std::pow(solar_elev_rad / kHalfPi, 1.0 / 3.0);

    Cooked out{};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 9; ++i) {
            out.cfg[c][i] = CookScalar(kHosekDatasetRGB[c], i, 9,
                                       int_turb, turb_rem, albedo, se);
        }
        out.rad[c] = CookScalar(kHosekDatasetRGBRad[c], 0, 1,
                                int_turb, turb_rem, albedo, se);
    }
    return out;
}

// Per-direction radiance shape function F(theta, gamma) for one channel, times
// nothing (the caller multiplies by rad[c]). theta = angle from zenith (so
// cos_theta = up-component of the view ray), gamma = angle from the sun.
// Byte-for-byte the reference ArHosekSkyModel_GetRadianceInternal:
//   F = (1 + A*exp(B/(cos_theta+0.01)))
//     * (C + D*exp(E*gamma) + F*cos^2(gamma) + G*chi(H,gamma) + I*sqrt(cos_theta))
//   chi(g,a) = (1 + cos^2 a) / (1 + g^2 - 2 g cos a)^1.5
// The +0.01 fudge on cos_theta is the reference's horizon guard.
inline double RadianceInternal(const double cfg[9], double cos_theta,
                               double cos_gamma, double gamma) {
    const double A = cfg[0], B = cfg[1], C = cfg[2], D = cfg[3], E = cfg[4];
    const double F = cfg[5], G = cfg[6], I = cfg[7], H = cfg[8];  // I=zenith, H=Mie g
    const double expM = std::exp(E * gamma);
    const double cg2  = cos_gamma * cos_gamma;
    const double mie_den = std::pow(std::max(1.0 + H * H - 2.0 * H * cos_gamma, 1e-4), 1.5);
    const double chi  = (1.0 + cg2) / mie_den;
    const double expo = 1.0 + A * std::exp(B / (cos_theta + 0.01));
    const double lobe = C + D * expM + F * cg2 + G * chi + I * std::sqrt(std::max(cos_theta, 0.0));
    return expo * lobe;
}

}  // namespace pt::hosek

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <glm/glm.hpp>

namespace pt::ocean {

// --- Planetary P5 (#259): Cox & Munk, and what "roughness" means here -----
//
// Cox, C. & Munk, W. (1954), "Measurement of the roughness of the sea
// surface from photographs of the sun's glitter", J. Opt. Soc. Am. 44(11)
// 838-850. Their table 1 clean-surface fit, in the paper's own notation,
// is the TOTAL mean-square slope (the sum of the up-wind and cross-wind
// components):
//
//     sigma^2 = 0.003 + 0.00512 * U        U in m/s, measured at 12.5 m
//
// That is a MEASURED oceanographic constant, not a dial, and it is the
// number a planetary ocean seen from orbit has to converge to.
//
// TURNING IT INTO A MICROFACET ROUGHNESS. The Beckmann NDF with roughness
// alpha has the slope distribution
//
//     P22(s) = exp(-(sx^2 + sy^2) / alpha^2) / (pi * alpha^2)
//
// i.e. a 2-D Gaussian with variance alpha^2 / 2 PER AXIS, so its TOTAL
// mean-square slope is exactly alpha^2. Matching the measured total to the
// distribution therefore gives
//
//     alpha = sigma = sqrt(0.003 + 0.00512 * U)
//
// and at the engine's default r_ocean_wind_speed 12 that is alpha =
// sqrt(0.0645) = 0.25397 -- a glitter half-angle of atan(0.254) = 14.3
// degrees, which is the width Cox & Munk photographed.
//
// THE ENGINE'S ROUGHNESS IS NOT ALPHA. PathTrace.slang's anisoAlpha()
// squares it: `float alpha = r * r`. So the value that goes in a material's
// roughness field to produce alpha is sqrt(alpha), and the far-field ocean
// roughness at 12 m/s is sqrt(0.25397) = 0.50396.
//
// This is deliberately NOT the 0.18 that issue #259 quotes. 0.18 is
// sqrt(sigma^2 / 2) -- the PER-AXIS slope RMS -- which would be the answer
// if `roughness` were alpha directly AND alpha^2 were the per-axis variance.
// Under this engine's alpha = roughness^2 convention neither holds, and the
// two mistakes do not cancel: 0.18 renders a glitter path a factor sqrt(2)
// x sqrt(2) too tight. The derivation above is what the code and the tests
// use; see tests/pt_ocean_fft_test.cpp, which pins every step of it.
double CoxMunkMeanSquareSlope(double wind_mps) noexcept;
// The Beckmann/GGX alpha with that total mean-square slope.
double CoxMunkAlpha(double wind_mps) noexcept;
// The engine-facing perceptual roughness r for which anisoAlpha gives that
// alpha, i.e. sqrt(alpha).
double CoxMunkRoughness(double wind_mps) noexcept;

// --- The band the measured slope lives in ---------------------------------
//
// Cox & Munk photographed the WHOLE sea surface, capillary ripples included,
// so their sigma^2 is the integral over every wavelength the ocean carries.
// A finite grid carries a window of that, and to split the constant between
// geometry and BRDF the split has to be a number rather than a guess.
//
// Phillips 1958 (J. Fluid Mech. 4(4) 426-434) gives the equilibrium range as
// k^-4, so the slope-variance integrand is k^2 * k^-4 * k dk = dk/k =
// d(ln k): MEAN-SQUARE SLOPE IS UNIFORM IN LOG WAVELENGTH. The fraction of
// sigma^2 inside a window is then the window's log-length over the whole
// range's, and the whole range has two ends with real names:
//
//   TOP -- the spectral peak. Pierson & Moskowitz 1964 (JGR 69(24)
//   5181-5190) put a fully developed sea's peak at omega_p = 0.877 g / U, so
//   lambda_p = 2 pi U^2 / (0.877^2 g): 120 m at 12 m/s. Above the peak the
//   spectrum rolls off and carries essentially no slope.
//
//   BOTTOM -- the gravity-capillary transition, the wavelength of MINIMUM
//   phase speed, lambda_m = 2 pi sqrt(sigma / (rho g)) with the air-water
//   surface tension sigma = 0.0728 N/m at 20 C: 1.712 cm. (Lamb,
//   "Hydrodynamics" 6th ed. 1932, section 267; Phillips, "The Dynamics of
//   the Upper Ocean" 2nd ed. 1977, section 3.2.) Below it capillarity, not
//   gravity, restores the surface and the deep-water dispersion this solver
//   uses stops being the right one anyway.
double GravityCapillaryWavelengthM() noexcept;
double PiersonMoskowitzPeakWavelengthM(double wind_mps) noexcept;

// The fraction of Cox & Munk's mean-square slope that lives in the
// wavelength window [lo, hi], under the log-uniform law above. Clamped to
// [0, 1]; the window is intersected with [lambda_m, lambda_p] first, since
// outside that the law does not hold and the spectrum has nothing anyway.
double SlopeVarianceFractionInBand(double lo_m, double hi_m,
                                   double wind_mps) noexcept;

// --- Planetary P5 (#259): the camera-anchored, quantised tangent frame ----
//
// THE PROBLEM. The FFT tile is planar and the ocean is not. Projecting the
// world into the East/North basis at ONE anchor point on the shell costs
// (d/R)^2 of arc-length distortion -- 2.5e-4 at 100 km, i.e. 25 m over
// 100 km, on a field whose finest cascade tiles every 23 m and whose crests
// are sub-pixel past ~2.8 km. Every alternative is worse: cubed-sphere face
// UV rotates discontinuously across the 12 cube seams and stretches the
// field by the tangent warp's +/-13%.
//
// WHY THE ANCHOR MUST BE QUANTISED, AND WHY THE QUANTISATION IS EXACT. An
// anchor that slides with the camera drags the whole wave field along with
// it: parallax dies and the sea looks painted on the screen. So the anchor
// snaps to a lattice -- and the spacing is exactly cascade 0's period, which
// makes the snap a NO-OP for the lookup rather than something to smooth over.
//
//   * Northing is quantised in ARC LENGTH along the meridian:
//     i = round(R * lat / P0). Adjacent lattice rows are exactly P0 apart.
//   * Easting is quantised in arc length along the SNAPPED parallel:
//     j = round(R * cos(lat_a) * lon / P0). Adjacent lattice columns in one
//     row are exactly P0 apart, because the parallel's radius is fixed by
//     the snapped latitude and not by the camera's own.
//
// Moving the anchor one cell therefore translates every world point's
// tangent coordinate by exactly P0, and cascade 0's uv -- that coordinate
// over P0 -- does not move at all. Cascades 1 and 2 pick up
// frac(j * P0 / P_c), computed here in f64 from the INTEGER lattice indices,
// so it is exact and is a pure function of camera position: no hysteresis,
// no path dependence, which is what the golden matrix requires and what a
// "re-anchor once you have drifted half a cell" scheme could not give.
//
// WHAT IS LEFT, stated rather than hidden. Two second-order residuals:
//   * ORIENTATION. Adjacent anchors P0 apart differ in basis orientation by
//     P0/R = 2.8e-4 rad, so a point d away sees its tangent coordinate move
//     by d * P0 / R: 0.28 m at 1 km, 2.8 m at 10 km. On a 30 m wave that is
//     a 3% phase error at 1 km.
//   * MERIDIAN CONVERGENCE. Crossing a latitude row changes cos(lat_a), so a
//     point Dlon east of the anchor shifts by Dlon * sin(lat) * P0 -- 1.6 m
//     for a point 5 km east at 45 degrees latitude.
// Both are the tangent plane not being the sphere, i.e. the same (d/R)^2 the
// parameterisation already accepts.
inline constexpr int kOceanMaxCascades = 3;

struct OceanTangentFrame {
    bool       valid = false;
    // All three in the frame the caller passed its pole and prime meridian
    // in. `origin` is RELATIVE TO THE SPHERE CENTRE; the caller adds the
    // centre, because that addition is a conversion-boundary decision.
    glm::dvec3 origin{0.0};
    glm::dvec3 east{1.0, 0.0, 0.0};
    glm::dvec3 north{0.0, 0.0, 1.0};
    // Tile-space phase per cascade. Cascade 0's is exactly 0 by
    // construction: the lattice spacing IS cascade 0's period.
    double     phase[kOceanMaxCascades][2] = {{0.0, 0.0}, {0.0, 0.0},
                                              {0.0, 0.0}};
    // The lattice cell. Exposed so a test can assert that crossing a
    // boundary really is a whole-cell step and not a fraction of one.
    double     lattice_i = 0.0;   // northing index
    double     lattice_j = 0.0;   // easting index
};

// `cam_rel_centre` is the camera position relative to the sphere centre;
// `pole` and `prime` are an orthonormal pair fixing the lattice's latitude
// axis and its zero meridian. `period_m[0]` is the lattice spacing.
OceanTangentFrame OceanTangentAnchor(const glm::dvec3& cam_rel_centre,
                                     double radius_m,
                                     const glm::dvec3& pole,
                                     const glm::dvec3& prime,
                                     const double period_m[kOceanMaxCascades],
                                     int cascades) noexcept;

// OceanFFT -- Wave 8 (#25): Tessendorf-style FFT ocean surface.
//
// Statistical-spectrum ocean simulation following Tessendorf 2001
// ("Simulating Ocean Water"). The surface is synthesised in the
// frequency domain from the Phillips spectrum, time-evolved by the
// deep-water gravity-wave dispersion relation, and inverse-FFT'd to
// a real-space displacement + slope field that tiles seamlessly over
// an arbitrarily large ocean plane.
//
// Pipeline (all CPU, run once per frame in Engine::StepOcean):
//   1. Build the static base spectrum H0(k) ONCE from the Phillips
//      spectrum P(k) = A * exp(-1/(k*L)^2) / k^4 * |k_hat . wind_hat|^2,
//      L = windSpeed^2 / g, seeded with Gaussian-distributed complex
//      amplitudes (Box-Muller). Rebuilt only when wind / amplitude /
//      tile-size change.
//   2. Time-evolve: H(k,t) = H0(k)*exp(i*w(k)*t)
//                          + conj(H0(-k))*exp(-i*w(k)*t),
//      dispersion w(k) = sqrt(g * |k|). This keeps h(x,t) real.
//   3. Choppy/Gerstner horizontal displacement spectra:
//        Dx(k) = -i * (kx/|k|) * H(k,t)
//        Dz(k) = -i * (kz/|k|) * H(k,t)
//      and slope spectra Sx(k) = i*kx*H, Sz(k) = i*kz*H (for normals).
//   4. Inverse-FFT (radix-2, separable 2D) each spectrum -> real
//      fields. Apply the (-1)^(x+z) sign-flip that folds the
//      iFFT's centre-origin convention into a corner-origin tile.
//   5. Pack into the displacement texture (RGBA16F: xyz = world-space
//      displacement in metres, w = foam) and a normal texture
//      (RGBA16F: xyz = unit world normal). Foam accumulates where the
//      horizontal-displacement Jacobian folds (det < threshold) AND on
//      wind-driven whitecap crests (Wave 9), then persists with a decay
//      trail so broken crests fade over seconds rather than blinking off
//      the instant the Jacobian unfolds.
//
// Real units throughout: 1 world unit = 1 metre, g = 9.81 m/s^2,
// windSpeed in m/s. No demo-scaled fudge factors -- the Phillips
// amplitude `A` and the choppiness scalar are the only artistic
// dials, both exposed as cvars.
//
// Grid size N is a power of two (256 default; 512 for higher detail).
// Per-frame cost at N=256 is ~5 inverse FFTs of N^2 points each on
// one thread -- a few ms, measured acceptable for a real-time toy.
class OceanFFT {
public:
    struct Config {
        // FFT grid resolution (power of two). 256 is the Tessendorf
        // reference default; 512 sharpens crests at ~4x the cost.
        std::uint32_t grid_size = 256u;

        // Physical tile size in metres. The N x N grid spans this many
        // metres in both X and Z; the displacement texture tiles every
        // `patch_size_m` metres of world space. Larger = lower spatial
        // frequency (broad swells); smaller = tighter chop.
        float patch_size_m = 50.0f;

        // Wind speed (m/s). Drives the Phillips characteristic length
        // L = windSpeed^2 / g and hence the dominant wavelength. 12 m/s
        // is a fresh breeze (Beaufort 6) -- visible whitecaps, ~30 m
        // dominant wavelength at L scale.
        float wind_speed = 12.0f;

        // Wind direction in the XZ plane, radians (0 = +X). Waves
        // travelling against the wind are suppressed by the Phillips
        // directional term |k_hat . wind_hat|^2.
        float wind_dir_rad = 0.0f;

        // Phillips amplitude A (dimensionless spectral scale). Sets the
        // overall RMS wave height. At the 50 m default tile + 12 m/s wind,
        // 0.0002 gives roughly +/-1.5 m peak crests (a moderate sea); the
        // cvar scales it linearly. The un-normalised inverse DFT (see
        // Ifft2D) means a little goes a long way.
        float amplitude = 0.0002f;

        // Choppiness lambda (0 = round sinusoidal swells, >0 sharpens
        // crests via horizontal Gerstner displacement). 1.0 is a strong
        // choppy sea; clamped to [0, 1.5] on the host.
        float choppiness = 1.0f;

        // Foam Jacobian threshold. Foam accumulates where the
        // horizontal-displacement Jacobian determinant drops below this
        // (folding crests). Lower = less foam. 0.5 is a moderate spray.
        float foam_threshold = 0.5f;

        // --- Wave 9 ocean-foam (#27 sibling) ---------------------------------
        // Foam intensity multiplier on the instantaneous Jacobian-fold foam
        // (and the persistence trail). 1 = the raw [0,1] crest coverage;
        // >1 spreads brighter, broader whitecaps; 0 disables crest foam.
        // Clamped >= 0 on the host.
        float foam_amount = 1.0f;

        // Foam persistence (lifetime) in [0, 1). Foam lingers after a crest
        // breaks instead of vanishing the instant the Jacobian unfolds: each
        // frame the accumulated foam decays by `foam_persistence^dt`-ish and
        // is re-maxed with the fresh instantaneous foam (so a passing crest
        // leaves a fading streak). 0 = no memory (pure instantaneous foam,
        // matching the Wave 8 behaviour); 0.92 is a few-second trail at 60
        // fps. Clamped to [0, 0.999] on the host.
        float foam_persistence = 0.92f;

        // Whitecap-coverage exponent driving the wind dependence. Real
        // oceanography (Monahan/Wu): whitecap fractional area is ~zero in
        // light air and rises sharply above the ~7 m/s whitecap-onset wind.
        // We model the coverage bias as a smooth ramp centred on
        // kWhitecapOnsetMps, and `foam_coverage` scales how aggressively the
        // ramp lifts the foam floor: 1 = the reference Beaufort ramp, >1
        // pushes whitecaps onto lower-wind seas, 0 removes the wind term
        // (crest foam only). Clamped >= 0 on the host.
        float foam_coverage = 1.0f;
        // --- end Wave 9 ocean-foam -------------------------------------------

        // Gravity (m/s^2). Earth deep-water dispersion w = sqrt(g*k).
        float gravity = 9.81f;

        // PRNG seed for the H0 Gaussian amplitudes. Fixed so a given
        // config produces a deterministic base spectrum (the animation
        // is then a pure function of sim time).
        std::uint32_t seed = 1337u;

        // --- Planetary P5 (#259): spectral band limits ------------------
        // Wavelength window, in metres, that this solver's spectrum is
        // restricted to. A bin is kept when
        //     band_lo_m <= lambda <= band_hi_m,  lambda = 2*pi / |k|.
        // Both 0 means "no window" -- the Wave 8 behaviour, bit-for-bit,
        // which is what the single-cascade legacy ocean still uses.
        //
        // WHY A WINDOW AND NOT JUST DIFFERENT TILE SIZES. A planetary
        // ocean needs several cascades at mutually non-commensurate
        // periods so no tile grid or beat pattern is visible across
        // thousands of km. Running the SAME Phillips spectrum at three
        // tile sizes triple-counts every wavelength the three grids can
        // all resolve: the 23 m and the 1793 m tiles both carry the 20 m
        // waves, so the sea comes out sqrt(3) times steeper than the
        // spectrum says and Cox-Munk stops being reachable. Windowing
        // each cascade to the band its own grid resolves makes the three
        // a PARTITION of one spectrum instead of three copies of it, so
        // the cascade set's mean-square slope is the spectrum's.
        //
        // The window is sharp rather than tapered on purpose. This is
        // spectral SYNTHESIS from random phases, not filtering of a
        // signal, so a sharp edge produces no ringing -- it just decides
        // which cascade owns which wavenumber.
        float band_lo_m = 0.0f;
        float band_hi_m = 0.0f;
    };

    OceanFFT()  = default;
    ~OceanFFT() = default;

    OceanFFT(const OceanFFT&)            = delete;
    OceanFFT& operator=(const OceanFFT&) = delete;

    // Mutable config. The engine writes from cvars each frame; the base
    // spectrum is lazily rebuilt inside Update() when a spectrum-
    // affecting field changed since the last rebuild.
    Config&       MutableConfig()       { return cfg_; }
    const Config& GetConfig()     const { return cfg_; }

    std::uint32_t GridSize() const { return cfg_.grid_size; }

    // Advance the ocean to absolute sim time `t_seconds` (NOT a delta --
    // the spectrum evolution H0*exp(i*w*t) is a closed-form function of
    // absolute time, so passing the accumulated wall-clock time keeps
    // the surface frame-rate independent). Rebuilds the base spectrum
    // first if the config changed. Fills the internal displacement +
    // normal float buffers (see DisplacementRGBA / NormalRGBA).
    void Update(double t_seconds);

    // Packed RGBA32F displacement field, grid_size x grid_size, row-
    // major, 4 floats per texel: xyz = world-space displacement (m),
    // w = foam coverage in [0, 1]. Valid after Update().
    const std::vector<float>& DisplacementRGBA() const { return disp_rgba_; }

    // Packed RGBA32F normal field, grid_size x grid_size, row-major,
    // 4 floats per texel: xyz = unit world-space normal, w = unused (1).
    // Valid after Update().
    const std::vector<float>& NormalRGBA() const { return normal_rgba_; }

    // Peak absolute vertical displacement over the last Update(), in
    // metres. The engine pushes this to the shader as the heightfield
    // ray-march's vertical search bound (so the march only needs to
    // bracket [-h_max, +h_max] above/below the analytic plane).
    float MaxDisplacementY() const { return max_disp_y_; }

    // Force a base-spectrum rebuild on the next Update() (e.g. after a
    // seed / grid-size change the engine can't detect via the float
    // config deltas alone).
    void Invalidate() { dirty_ = true; }

    // --- Planetary P5 (#259): mean-square slope from the spectrum -------
    //
    // The TOTAL mean-square slope (both axes summed) this cascade's
    // surface carries, restricted to the bins whose wavelength is at
    // least `min_wavelength_m`. Pass 0 for the whole spectrum.
    //
    // Derived, not sampled. The packed height field is
    //     h(x) = sum_k h~(k,t) exp(i k.x)
    // (Ifft2D's un-normalised inverse DFT), so its gradient is
    //     grad h = sum_k (i k) h~(k,t) exp(i k.x)
    // and Parseval for that convention gives
    //     mean_x |grad h|^2 = sum_k |k|^2 |h~(k,t)|^2.
    // h~ = H0(k) exp(i w t) + conj(H0(-k)) exp(-i w t), whose expected
    // square modulus is |H0(k)|^2 + |H0(-k)|^2 -- independent of t, which
    // is what makes this a property of the SPECTRUM rather than of the
    // frame, and therefore something the shader can be handed once per
    // rebuild instead of once per frame.
    //
    // Averaging the packed gx/gz grid instead would be one realisation of
    // the same random variable, would move every frame, and could not be
    // band-restricted without a second FFT.
    double SlopeVarianceAbove(double min_wavelength_m) const noexcept;

    // Grid spacing in metres: patch_size_m / grid_size. The finest
    // wavelength this cascade can carry is twice this (Nyquist).
    double GridSpacingM() const noexcept {
        return static_cast<double>(cfg_.patch_size_m) /
               static_cast<double>(cfg_.grid_size ? cfg_.grid_size : 1u);
    }

    // Build (or rebuild) the base spectrum if the config moved, WITHOUT
    // running the per-frame inverse FFTs. SlopeVarianceAbove reads H0, so
    // the amplitude a cascade needs can be solved for at a fraction of the
    // cost of a full Update().
    void EnsureSpectrum();

    // --- Base-spectrum accessors ----------------------------------------
    //
    // These exposed H0(k) for the (now removed) planetary GPU cascade
    // pre-pass to upload once per rebuild -- the seeded std::mt19937
    // Gaussian amplitudes are not something a GPU can reproduce, so the host
    // stayed the single source of the spectrum. They are harmless read-only
    // accessors and are kept for tests/pt_ocean_fft_test.cpp.
    //
    // `SpectrumRevision` increments on every RebuildBaseSpectrum, so a
    // caller can tell "the spectrum I uploaded is still the spectrum this
    // solver holds" without comparing every config field itself.
    std::uint64_t SpectrumRevision() const noexcept { return spectrum_rev_; }
    // H0(k) and its conjugate mirror conj(H0(-k)), both N x N row-major.
    // Empty until the first EnsureSpectrum() / Update().
    const std::vector<std::complex<float>>& H0()     const { return h0_; }
    const std::vector<std::complex<float>>& H0Conj() const { return h0_conj_; }

private:
    void RebuildBaseSpectrum();

    // In-place 1D radix-2 inverse FFT over `n` complex samples with
    // stride `stride` starting at `data + offset`. `n` must be a power
    // of two. Used separably for rows then columns.
    static void Ifft1D(std::complex<float>* data, std::uint32_t n,
                       std::uint32_t stride, std::uint32_t offset);
    // Separable 2D inverse FFT of an N x N complex grid (row-major).
    void Ifft2D(std::vector<std::complex<float>>& grid) const;

    Config cfg_{};

    // Static base spectrum H0(k) and its conjugate-mirror partner
    // conj(H0(-k)), both N x N row-major. Rebuilt on config change.
    std::vector<std::complex<float>> h0_;
    std::vector<std::complex<float>> h0_conj_;
    // Precomputed dispersion w(k) = sqrt(g*|k|) per grid cell.
    std::vector<float>               omega_;

    // Per-frame spectra scratch (reused across Update calls).
    std::vector<std::complex<float>> hkt_;   // height
    std::vector<std::complex<float>> dx_;    // x displacement
    std::vector<std::complex<float>> dz_;    // z displacement
    std::vector<std::complex<float>> sx_;    // x slope
    std::vector<std::complex<float>> sz_;    // z slope

    // Output fields.
    std::vector<float> disp_rgba_;
    std::vector<float> normal_rgba_;
    float              max_disp_y_ = 0.0f;

    // --- Wave 9 ocean-foam (#27 sibling) -------------------------------------
    // Persistent foam buffer (one scalar per grid cell, row-major). Foam
    // breaks on a crest, then lingers and fades: each Update() decays this
    // by foam_persistence^dt and re-maxes with the fresh instantaneous
    // Jacobian-fold foam, so the value packed into disp_rgba_[.w] carries
    // both the live crest froth AND the fading trail of recently-broken
    // crests. Sized lazily to grid_size^2; reset on a grid-size change.
    std::vector<float> foam_accum_;
    // Absolute sim time of the previous Update(), to derive the per-frame
    // dt for the persistence decay (Update takes absolute time, not a delta).
    // Negative => no previous frame yet (first Update seeds the buffer).
    double             last_t_ = -1.0;
    // --- end Wave 9 ocean-foam -----------------------------------------------

    // Cached config snapshot used to detect when the base spectrum must
    // be rebuilt (any of grid_size / patch / wind / amplitude / seed).
    bool          dirty_           = true;
    std::uint32_t built_grid_      = 0u;
    float         built_patch_     = 0.0f;
    float         built_wind_      = 0.0f;
    float         built_wind_dir_  = 0.0f;
    float         built_amplitude_ = 0.0f;
    float         built_gravity_   = 0.0f;
    std::uint32_t built_seed_      = 0u;
    float         built_band_lo_   = 0.0f;
    float         built_band_hi_   = 0.0f;
    // Bumped by every RebuildBaseSpectrum. 0 means "never built", which is
    // a value no rebuilt spectrum can have, so a caller's "have I uploaded
    // this one?" comparison starts out false without a separate flag.
    std::uint64_t spectrum_rev_    = 0u;
};

}  // namespace pt::ocean

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <glm/glm.hpp>

namespace pt::ocean {

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
};

}  // namespace pt::ocean

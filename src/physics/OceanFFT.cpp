// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "OceanFFT.h"

#include <algorithm>
#include <cmath>

namespace pt::ocean {

namespace {
constexpr float kTwoPi = 6.28318530717958647692f;

// Bit-reverse `x` over `bits` bits. Radix-2 FFT permutation step.
inline std::uint32_t BitReverse(std::uint32_t x, std::uint32_t bits) {
    std::uint32_t r = 0u;
    for (std::uint32_t i = 0u; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

inline std::uint32_t Log2u(std::uint32_t n) {
    std::uint32_t l = 0u;
    while ((1u << l) < n) ++l;
    return l;
}
}  // namespace

// In-place radix-2 inverse FFT over a strided 1D slice. The "inverse"
// here uses the +i sign convention and does NOT apply the 1/n
// normalisation -- the caller (Ifft2D) folds the single 1/N^2 scale in
// once at the end. This matches Tessendorf's "iFFT up to a constant"
// formulation where the spectral amplitudes already carry the physical
// scale.
void OceanFFT::Ifft1D(std::complex<float>* data, std::uint32_t n,
                      std::uint32_t stride, std::uint32_t offset) {
    const std::uint32_t bits = Log2u(n);
    // Bit-reversal permutation.
    for (std::uint32_t i = 0u; i < n; ++i) {
        std::uint32_t j = BitReverse(i, bits);
        if (j > i) {
            std::swap(data[offset + i * stride], data[offset + j * stride]);
        }
    }
    // Danielson-Lanczos butterflies. +i exponent sign = inverse
    // transform (forward would be -i).
    for (std::uint32_t len = 2u; len <= n; len <<= 1) {
        const float ang = kTwoPi / static_cast<float>(len);  // +2pi/len => inverse
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (std::uint32_t i = 0u; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            const std::uint32_t half = len >> 1;
            for (std::uint32_t k = 0u; k < half; ++k) {
                std::complex<float>& a = data[offset + (i + k) * stride];
                std::complex<float>& b = data[offset + (i + k + half) * stride];
                const std::complex<float> t = w * b;
                b = a - t;
                a = a + t;
                w *= wlen;
            }
        }
    }
}

void OceanFFT::Ifft2D(std::vector<std::complex<float>>& grid) const {
    const std::uint32_t N = cfg_.grid_size;
    // Rows.
    for (std::uint32_t y = 0u; y < N; ++y) {
        Ifft1D(grid.data(), N, /*stride=*/1u, /*offset=*/y * N);
    }
    // Columns.
    for (std::uint32_t x = 0u; x < N; ++x) {
        Ifft1D(grid.data(), N, /*stride=*/N, /*offset=*/x);
    }
    // NO 1/N^2 normalisation. Tessendorf's ocean synthesis evaluates
    //   h(x,t) = sum_k h~(k,t) * exp(i k.x)
    // which is the *un-normalised* inverse DFT at the grid points -- the
    // spectral amplitudes h~ already carry the physical per-mode height
    // contribution, so dividing by N^2 here would shrink the surface by
    // the grid area (e.g. 1/65536 at N=256) and flatten the ocean to
    // microns. The Phillips amplitude cvar (r_ocean_amplitude) is the
    // physical RMS-height dial on top of this.
}

void OceanFFT::RebuildBaseSpectrum() {
    const std::uint32_t N = cfg_.grid_size;
    const std::size_t   cells = static_cast<std::size_t>(N) * N;
    h0_.assign(cells, std::complex<float>(0.0f, 0.0f));
    h0_conj_.assign(cells, std::complex<float>(0.0f, 0.0f));
    omega_.assign(cells, 0.0f);

    const float L_patch = cfg_.patch_size_m;                 // metres per tile
    const float g       = cfg_.gravity;
    // Phillips characteristic length: largest wave from a continuous
    // wind of `wind_speed` m/s. L = V^2 / g.
    const float V       = std::max(cfg_.wind_speed, 1e-3f);
    const float L_phil  = V * V / g;
    // Suppress waves much smaller than this (numerical-spike guard,
    // Tessendorf eq. 24 small-wave cutoff). 0.1% of the dominant length.
    const float l_small = L_phil * 1e-3f;
    const glm::vec2 wind_hat(std::cos(cfg_.wind_dir_rad),
                             std::sin(cfg_.wind_dir_rad));
    const float A = std::max(cfg_.amplitude, 0.0f);

    std::mt19937 rng(cfg_.seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // k = 2*pi*(n, m) / L for n,m in [-N/2, N/2). Row-major (y outer).
    const int half = static_cast<int>(N) / 2;
    for (std::uint32_t iy = 0u; iy < N; ++iy) {
        const int m = static_cast<int>(iy) - half;
        const float kz = kTwoPi * static_cast<float>(m) / L_patch;
        for (std::uint32_t ix = 0u; ix < N; ++ix) {
            const int n = static_cast<int>(ix) - half;
            const float kx = kTwoPi * static_cast<float>(n) / L_patch;
            const std::size_t idx = static_cast<std::size_t>(iy) * N + ix;

            const float k2 = kx * kx + kz * kz;
            const float k  = std::sqrt(k2);

            // Dispersion w(k) = sqrt(g*k). DC term (k=0) has no motion.
            omega_[idx] = std::sqrt(g * k);

            if (k < 1e-6f) {
                // DC: no wave, leave H0 at zero.
                continue;
            }

            // Phillips spectrum P(k).
            const glm::vec2 k_hat(kx / k, kz / k);
            float kdotw = k_hat.x * wind_hat.x + k_hat.y * wind_hat.y;
            // Directional term |k_hat . wind_hat|^2 -- waves perpendicular
            // to the wind vanish; we keep the sign-squared so up/down-wind
            // are symmetric (Tessendorf uses the squared dot).
            const float dir = kdotw * kdotw;
            const float kL  = k * L_phil;
            float ph = A * std::exp(-1.0f / (kL * kL)) / (k2 * k2) * dir;
            // Small-wave suppression: multiply by exp(-k^2 * l^2).
            ph *= std::exp(-k2 * l_small * l_small);
            if (ph < 0.0f) ph = 0.0f;

            const float amp = std::sqrt(ph * 0.5f);
            // Gaussian complex amplitude (xi_r + i*xi_i) * sqrt(Ph/2).
            h0_[idx] = std::complex<float>(gauss(rng) * amp, gauss(rng) * amp);
        }
    }

    // Precompute conj(H0(-k)) for each cell: mirror index is the cell
    // whose (n, m) is negated. With the [-N/2, N/2) layout, the mirror
    // of grid index (ix, iy) is ((N - ix) % N, (N - iy) % N).
    for (std::uint32_t iy = 0u; iy < N; ++iy) {
        const std::uint32_t my = (N - iy) % N;
        for (std::uint32_t ix = 0u; ix < N; ++ix) {
            const std::uint32_t mx = (N - ix) % N;
            const std::size_t idx  = static_cast<std::size_t>(iy) * N + ix;
            const std::size_t midx = static_cast<std::size_t>(my) * N + mx;
            h0_conj_[idx] = std::conj(h0_[midx]);
        }
    }

    // Mark the built snapshot.
    built_grid_      = cfg_.grid_size;
    built_patch_     = cfg_.patch_size_m;
    built_wind_      = cfg_.wind_speed;
    built_wind_dir_  = cfg_.wind_dir_rad;
    built_amplitude_ = cfg_.amplitude;
    built_gravity_   = cfg_.gravity;
    built_seed_      = cfg_.seed;
    dirty_           = false;
}

void OceanFFT::Update(double t_seconds) {
    // Rebuild the base spectrum if any spectrum-affecting field moved.
    if (dirty_ || built_grid_ != cfg_.grid_size ||
        built_patch_ != cfg_.patch_size_m ||
        built_wind_ != cfg_.wind_speed ||
        built_wind_dir_ != cfg_.wind_dir_rad ||
        built_amplitude_ != cfg_.amplitude ||
        built_gravity_ != cfg_.gravity ||
        built_seed_ != cfg_.seed) {
        RebuildBaseSpectrum();
    }

    const std::uint32_t N = cfg_.grid_size;
    const std::size_t   cells = static_cast<std::size_t>(N) * N;
    const float t = static_cast<float>(t_seconds);
    const float L_patch = cfg_.patch_size_m;

    hkt_.assign(cells, std::complex<float>(0.0f, 0.0f));
    dx_.assign(cells, std::complex<float>(0.0f, 0.0f));
    dz_.assign(cells, std::complex<float>(0.0f, 0.0f));
    sx_.assign(cells, std::complex<float>(0.0f, 0.0f));
    sz_.assign(cells, std::complex<float>(0.0f, 0.0f));

    const int half = static_cast<int>(N) / 2;
    for (std::uint32_t iy = 0u; iy < N; ++iy) {
        const int m = static_cast<int>(iy) - half;
        const float kz = kTwoPi * static_cast<float>(m) / L_patch;
        for (std::uint32_t ix = 0u; ix < N; ++ix) {
            const int n = static_cast<int>(ix) - half;
            const float kx = kTwoPi * static_cast<float>(n) / L_patch;
            const std::size_t idx = static_cast<std::size_t>(iy) * N + ix;

            const float w  = omega_[idx];
            const float wt = w * t;
            // exp(+i*wt) and exp(-i*wt).
            const std::complex<float> ep(std::cos(wt), std::sin(wt));
            const std::complex<float> en(std::cos(wt), -std::sin(wt));

            // H(k,t) = H0(k)*exp(i*w*t) + conj(H0(-k))*exp(-i*w*t).
            const std::complex<float> h = h0_[idx] * ep + h0_conj_[idx] * en;
            hkt_[idx] = h;

            const float k2 = kx * kx + kz * kz;
            const float k  = std::sqrt(k2);

            // Slope spectra: i*k * H  (gradient of height).
            sx_[idx] = std::complex<float>(0.0f, kx) * h;
            sz_[idx] = std::complex<float>(0.0f, kz) * h;

            // Choppy horizontal displacement: -i*(k/|k|)*H.
            if (k > 1e-6f) {
                const float inv_k = 1.0f / k;
                dx_[idx] = std::complex<float>(0.0f, -kx * inv_k) * h;
                dz_[idx] = std::complex<float>(0.0f, -kz * inv_k) * h;
            }
        }
    }

    Ifft2D(hkt_);
    Ifft2D(dx_);
    Ifft2D(dz_);
    Ifft2D(sx_);
    Ifft2D(sz_);

    // Pack to RGBA32F. The (-1)^(x+z) sign flip converts the FFT's
    // centred-origin output (DC at grid centre) into a corner-origin
    // tile that wraps seamlessly when sampled by world XZ -> UV.
    disp_rgba_.assign(cells * 4u, 0.0f);
    normal_rgba_.assign(cells * 4u, 0.0f);

    const float lambda = std::max(cfg_.choppiness, 0.0f);
    const float foam_thresh = cfg_.foam_threshold;
    // Finite-difference spacing for the Jacobian: one grid cell in metres.
    const float dxz = L_patch / static_cast<float>(N);
    max_disp_y_ = 0.0f;

    for (std::uint32_t iy = 0u; iy < N; ++iy) {
        for (std::uint32_t ix = 0u; ix < N; ++ix) {
            const std::size_t idx = static_cast<std::size_t>(iy) * N + ix;
            const float sign = ((ix + iy) & 1u) ? -1.0f : 1.0f;

            const float h  = hkt_[idx].real() * sign;
            const float dX = dx_[idx].real() * sign * lambda;
            const float dZ = dz_[idx].real() * sign * lambda;
            const float gx = sx_[idx].real() * sign;   // dH/dx
            const float gz = sz_[idx].real() * sign;   // dH/dz

            // World displacement: y = height, xz = choppy lateral push.
            disp_rgba_[idx * 4u + 0u] = dX;
            disp_rgba_[idx * 4u + 1u] = h;
            disp_rgba_[idx * 4u + 2u] = dZ;
            // Foam computed below once neighbours are available; init 0.
            disp_rgba_[idx * 4u + 3u] = 0.0f;

            max_disp_y_ = std::max(max_disp_y_, std::fabs(h));

            // Surface normal from the height gradient. n = normalize(
            // (-dH/dx, 1, -dH/dz)). The slope iFFT gives dH/dx, dH/dz
            // directly (i*k*H -> spatial derivative).
            glm::vec3 nrm = glm::normalize(glm::vec3(-gx, 1.0f, -gz));
            normal_rgba_[idx * 4u + 0u] = nrm.x;
            normal_rgba_[idx * 4u + 1u] = nrm.y;
            normal_rgba_[idx * 4u + 2u] = nrm.z;
            normal_rgba_[idx * 4u + 3u] = 1.0f;
        }
    }

    // Foam from the horizontal-displacement Jacobian. Where the choppy
    // displacement folds the surface (det J < threshold) the crest is
    // pinching -- accumulate foam there. Uses central differences of the
    // already-packed lateral displacement field (wraps at the tile edge
    // because the field is periodic).
    //
    // --- Wave 9 ocean-foam (#27 sibling) -------------------------------------
    // Three additions over the Wave 8 instantaneous-Jacobian foam:
    //   1. Wind-driven whitecap coverage. Real oceanography (Monahan 1980,
    //      Wu 1979): whitecap fractional area W is ~0 in light air and rises
    //      sharply past the ~7 m/s whitecap-onset wind. We model a smooth
    //      [0,1] coverage bias `wind_cov` that lifts the foam floor on the
    //      wave crests as wind increases, gated by the local (normalised)
    //      crest height so the froth concentrates on the tops of waves
    //      rather than smearing across troughs.
    //   2. foam_amount intensity multiplier on the combined foam.
    //   3. Temporal persistence: foam lingers after a crest breaks. We keep
    //      a per-cell accumulation buffer, decay it by foam_persistence^dt
    //      each frame, and re-max it with the fresh instantaneous foam, so
    //      a passing crest leaves a fading streak (matching how real
    //      whitecaps dissipate over seconds rather than vanishing instantly).
    const float foam_amount   = std::max(cfg_.foam_amount, 0.0f);
    const float foam_cov_gain = std::max(cfg_.foam_coverage, 0.0f);
    const float persistence   = std::clamp(cfg_.foam_persistence, 0.0f, 0.999f);

    // Whitecap-onset coverage ramp. Centred on kWhitecapOnsetMps (~7 m/s,
    // the Beaufort-4/5 threshold where whitecaps become widespread); the
    // smoothstep over [onset-width, onset+width] gives the characteristic
    // sharp rise. `foam_coverage` scales the result and can push the onset
    // earlier (>1) or remove the wind term entirely (0).
    constexpr float kWhitecapOnsetMps = 7.0f;
    constexpr float kWhitecapWidthMps = 5.0f;
    const float ws = cfg_.wind_speed;
    float wind_cov = (ws - (kWhitecapOnsetMps - kWhitecapWidthMps)) /
                     (2.0f * kWhitecapWidthMps);
    wind_cov = std::clamp(wind_cov, 0.0f, 1.0f);
    wind_cov = wind_cov * wind_cov * (3.0f - 2.0f * wind_cov);  // smoothstep
    wind_cov *= foam_cov_gain;
    wind_cov = std::clamp(wind_cov, 0.0f, 1.0f);

    // Per-frame dt for the persistence decay. Update() takes ABSOLUTE sim
    // time, so the dt is the delta from the previous call. First frame (or
    // a backward/zero step, e.g. a paused sim) seeds with no decay.
    float dt = (last_t_ >= 0.0) ? static_cast<float>(t_seconds - last_t_) : 0.0f;
    if (dt < 0.0f) dt = 0.0f;
    last_t_ = t_seconds;
    // decay = persistence^dt, normalised so a frame at ~60 fps (dt~0.0167)
    // barely fades and a long pause fades fully. Guard persistence==0 ->
    // decay 0 (pure instantaneous foam, the Wave 8 behaviour).
    const float decay = (persistence > 0.0f) ? std::pow(persistence, dt * 60.0f)
                                             : 0.0f;

    // (Re)size the persistence buffer; reset on a grid-size change.
    if (foam_accum_.size() != cells) {
        foam_accum_.assign(cells, 0.0f);
    }

    // Crest-height normaliser for the wind-coverage gate: bias foam toward
    // the upper portion of the wave field. max_disp_y_ is this frame's peak
    // |height|; cells above ~40% of the peak count as "crest".
    const float h_norm = (max_disp_y_ > 1e-4f) ? (1.0f / max_disp_y_) : 0.0f;

    for (std::uint32_t iy = 0u; iy < N; ++iy) {
        const std::uint32_t yp = (iy + 1u) % N;
        const std::uint32_t ym = (iy + N - 1u) % N;
        for (std::uint32_t ix = 0u; ix < N; ++ix) {
            const std::uint32_t xp = (ix + 1u) % N;
            const std::uint32_t xm = (ix + N - 1u) % N;
            const std::size_t c  = static_cast<std::size_t>(iy) * N + ix;
            const std::size_t rx = static_cast<std::size_t>(iy) * N + xp;
            const std::size_t lx = static_cast<std::size_t>(iy) * N + xm;
            const std::size_t uz = static_cast<std::size_t>(yp) * N + ix;
            const std::size_t dz = static_cast<std::size_t>(ym) * N + ix;

            // Instantaneous Jacobian-fold foam (Wave 8). Only meaningful when
            // the choppy lateral displacement is active (lambda > 0); a
            // round-swell sea (lambda == 0) never folds, so its foam comes
            // purely from the wind-coverage term below.
            float foam_inst = 0.0f;
            if (lambda > 0.0f) {
                const float dDxdx = (disp_rgba_[rx * 4u + 0u] - disp_rgba_[lx * 4u + 0u]) / (2.0f * dxz);
                const float dDxdz = (disp_rgba_[uz * 4u + 0u] - disp_rgba_[dz * 4u + 0u]) / (2.0f * dxz);
                const float dDzdx = (disp_rgba_[rx * 4u + 2u] - disp_rgba_[lx * 4u + 2u]) / (2.0f * dxz);
                const float dDzdz = (disp_rgba_[uz * 4u + 2u] - disp_rgba_[dz * 4u + 2u]) / (2.0f * dxz);

                // Jacobian det of the map (x,z) -> (x+Dx, z+Dz).
                const float jxx = 1.0f + dDxdx;
                const float jzz = 1.0f + dDzdz;
                const float det = jxx * jzz - dDxdz * dDzdx;

                // Foam coverage: 0 above threshold, ramps to 1 as det
                // crosses zero (full fold). Clamp to [0, 1].
                foam_inst = (foam_thresh - det) / std::max(foam_thresh, 1e-3f);
                foam_inst = std::clamp(foam_inst, 0.0f, 1.0f);
            }

            // Wind-driven whitecap contribution, gated by crest height: a
            // smooth [0,1] ramp over the top 60% of the wave field, scaled
            // by the wind coverage. This is the "whitecaps grow with wind"
            // term -- it adds froth to crests even where the Jacobian hasn't
            // folded, and rises sharply once wind passes the onset speed.
            float wind_foam = 0.0f;
            if (wind_cov > 0.0f && h_norm > 0.0f) {
                const float hn = disp_rgba_[c * 4u + 1u] * h_norm;  // height / peak, ~[-1,1]
                float crest = (hn - 0.4f) / 0.6f;                   // 0 at 40% peak -> 1 at peak
                crest = std::clamp(crest, 0.0f, 1.0f);
                crest = crest * crest * (3.0f - 2.0f * crest);      // smoothstep
                wind_foam = wind_cov * crest;
            }

            // Combine: the stronger of the two foam sources, lifted by the
            // intensity dial. Both are physically "fraction of surface that
            // is froth", so a max (not a sum) keeps coverage in [0,1].
            float foam_now = std::max(foam_inst, wind_foam) * foam_amount;
            foam_now = std::clamp(foam_now, 0.0f, 1.0f);

            // Temporal persistence: decay the trail, re-max with the live
            // foam. The accumulated value (clamped) is what the shader reads.
            float acc = foam_accum_[c] * decay;
            acc = std::max(acc, foam_now);
            acc = std::clamp(acc, 0.0f, 1.0f);
            foam_accum_[c] = acc;
            disp_rgba_[c * 4u + 3u] = acc;
        }
    }
    // --- end Wave 9 ocean-foam -----------------------------------------------
}

}  // namespace pt::ocean

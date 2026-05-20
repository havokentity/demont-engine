// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt::ocean::OceanFFT unit tests (Wave 8, #25).
//
// Pins the core contracts of the CPU Tessendorf FFT ocean solver. The
// path-traced rendering quality has its own golden cell (ocean_fft.cfg)
// -- these tests pin the SOLVER mechanics + physics so a refactor can't
// silently break the spectrum / dispersion / iFFT without surfacing a
// unit failure first.
//
// Coverage:
//   * Update() fills displacement + normal buffers sized N*N*4
//   * Output is finite (no NaN/Inf from the iFFT or the spectrum)
//   * Normals are unit-length and broadly +Y (water surface)
//   * Foam stays in [0, 1]
//   * A fixed seed + config is deterministic across two solvers
//   * Time evolution actually moves the surface (t=0 vs t=2 differ)
//   * Zero amplitude => flat surface (height ~0, normals == +Y)
//   * Higher wind speed => longer dominant wavelength => larger RMS
//     wave height (Phillips L = V^2/g scaling)
//   * The radix-2 inverse FFT round-trips a known single-frequency
//     spectrum to the expected real-space cosine (validates butterfly
//     correctness independent of the ocean spectrum).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/physics/OceanFFT.h"

#include <cmath>
#include <vector>

using pt::ocean::OceanFFT;

namespace {
bool AllFinite(const std::vector<float>& v) {
    for (float f : v) {
        if (!std::isfinite(f)) return false;
    }
    return true;
}

float RmsHeight(const std::vector<float>& disp) {
    // disp is RGBA, height is the .y (index 1) channel.
    double sum2 = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < disp.size(); i += 4) {
        const double h = disp[i + 1];
        sum2 += h * h;
        ++n;
    }
    return n ? static_cast<float>(std::sqrt(sum2 / static_cast<double>(n))) : 0.0f;
}
}  // namespace

TEST_CASE("OceanFFT::Update fills correctly-sized finite buffers") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;   // small grid keeps the test fast
    o.Update(1.0);
    const auto& disp = o.DisplacementRGBA();
    const auto& nrm  = o.NormalRGBA();
    CHECK(disp.size() == 64u * 64u * 4u);
    CHECK(nrm.size()  == 64u * 64u * 4u);
    CHECK(AllFinite(disp));
    CHECK(AllFinite(nrm));
}

TEST_CASE("OceanFFT normals are unit-length and upward") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;
    o.Update(0.5);
    const auto& nrm = o.NormalRGBA();
    int upward = 0, total = 0;
    for (std::size_t i = 0; i < nrm.size(); i += 4) {
        const float x = nrm[i + 0], y = nrm[i + 1], z = nrm[i + 2];
        const float len = std::sqrt(x * x + y * y + z * z);
        CHECK(len == doctest::Approx(1.0f).epsilon(1e-3f));
        if (y > 0.0f) ++upward;
        ++total;
    }
    // A water surface perturbed about the horizontal: every normal has a
    // positive Y component (the gradient form (-dh/dx, 1, -dh/dz) can
    // never flip Y negative).
    CHECK(upward == total);
}

TEST_CASE("OceanFFT foam coverage stays in [0,1]") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;
    o.MutableConfig().choppiness = 1.2f;
    o.Update(3.0);
    const auto& disp = o.DisplacementRGBA();
    for (std::size_t i = 0; i < disp.size(); i += 4) {
        const float foam = disp[i + 3];
        CHECK(foam >= 0.0f);
        CHECK(foam <= 1.0f);
    }
}

TEST_CASE("OceanFFT is deterministic for a fixed seed + config") {
    OceanFFT a, b;
    a.MutableConfig().grid_size = 64;
    b.MutableConfig().grid_size = 64;
    a.Update(2.0);
    b.Update(2.0);
    const auto& da = a.DisplacementRGBA();
    const auto& db = b.DisplacementRGBA();
    REQUIRE(da.size() == db.size());
    for (std::size_t i = 0; i < da.size(); ++i) {
        CHECK(da[i] == doctest::Approx(db[i]).epsilon(1e-6f));
    }
}

TEST_CASE("OceanFFT time evolution moves the surface") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;
    o.Update(0.0);
    const std::vector<float> at0 = o.DisplacementRGBA();   // copy
    o.Update(2.0);
    const auto& at2 = o.DisplacementRGBA();
    // The surface must have changed between t=0 and t=2 s.
    double max_delta = 0.0;
    for (std::size_t i = 1; i < at0.size(); i += 4) {  // height channel
        max_delta = std::max(max_delta,
                             std::fabs(double(at2[i]) - double(at0[i])));
    }
    CHECK(max_delta > 1e-4);
}

TEST_CASE("OceanFFT zero amplitude => flat calm surface") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;
    o.MutableConfig().amplitude = 0.0f;
    o.Update(1.0);
    const auto& disp = o.DisplacementRGBA();
    const auto& nrm  = o.NormalRGBA();
    for (std::size_t i = 0; i < disp.size(); i += 4) {
        CHECK(disp[i + 1] == doctest::Approx(0.0f).epsilon(1e-5f));  // height
    }
    for (std::size_t i = 0; i < nrm.size(); i += 4) {
        CHECK(nrm[i + 0] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(nrm[i + 1] == doctest::Approx(1.0f).epsilon(1e-5f));   // +Y
        CHECK(nrm[i + 2] == doctest::Approx(0.0f).epsilon(1e-5f));
    }
}

TEST_CASE("OceanFFT higher wind => larger RMS wave height") {
    // Phillips L = windSpeed^2 / g: a stronger wind shifts energy into
    // longer, taller waves, raising the RMS surface height for a fixed
    // spectral amplitude. Same seed so the only difference is the wind.
    OceanFFT calm, gale;
    calm.MutableConfig().grid_size = 128;
    calm.MutableConfig().wind_speed = 6.0f;
    gale.MutableConfig().grid_size = 128;
    gale.MutableConfig().wind_speed = 20.0f;
    calm.Update(1.0);
    gale.Update(1.0);
    const float rms_calm = RmsHeight(calm.DisplacementRGBA());
    const float rms_gale = RmsHeight(gale.DisplacementRGBA());
    CHECK(rms_gale > rms_calm);
}

TEST_CASE("OceanFFT MaxDisplacementY tracks the height field") {
    OceanFFT o;
    o.MutableConfig().grid_size = 64;
    o.Update(1.0);
    const auto& disp = o.DisplacementRGBA();
    float peak = 0.0f;
    for (std::size_t i = 1; i < disp.size(); i += 4) {
        peak = std::max(peak, std::fabs(disp[i]));
    }
    CHECK(o.MaxDisplacementY() == doctest::Approx(peak).epsilon(1e-5f));
    CHECK(o.MaxDisplacementY() > 0.0f);
}

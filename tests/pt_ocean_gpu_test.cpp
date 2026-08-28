// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// CPU-vs-GPU equivalence for the ocean cascade solver
// (issue #259 planetary P5, issue #133 water roadmap Phase 2).
//
// WHY THIS TEST IS THE ACCEPTANCE CRITERION
// -----------------------------------------
// shaders/OceanCascades.slang moved the per-frame half of
// pt::ocean::OceanFFT to the GPU: the time evolution, the five spectra,
// the fifteen inverse FFTs, the pack, the normals and the foam. The host
// solver stays as the reference -- it still builds the seeded Phillips
// base spectrum both paths consume, and tests/pt_ocean_fft_test.cpp and
// tests/pt_planet_ocean_test.cpp still pin the physics through it. What
// nothing else can pin is that the two solvers agree, and "the goldens
// still pass" cannot stand in for it: a golden is 512x384 8-bit pixels of
// one camera, and the field is 3 x 256^2 RGBA32F.
//
// So this renders nothing. It runs both solvers on the SAME base spectrum
// and compares the fields texel by texel, against a bound derived from the
// arithmetic rather than from what happened to pass.
//
// THE TWO SOLVERS ARE NOT EXPECTED TO AGREE BIT FOR BIT, AND THE REASON IS
// SPECIFIC. OceanFFT::Ifft1D advances the twiddle factor by repeated
// complex multiplication:
//
//     w = 1;  for k in 0..len/2:  ...;  w *= wlen;
//
// so the twiddle at index k carries k multiplications' worth of accumulated
// rounding. The kernel evaluates cos/sin at the exact angle instead. That
// term is ~96% of the bound below and it is the HOST's error, not the
// GPU's -- the derivation says so, which is the point of deriving it.
//
// WHAT IS BIT-IDENTICAL, AND HAS TO BE. The GPU field, run to run. Every
// reduction in the kernel is over max(), which is exact and order-free in
// IEEE-754, and it is still done in a fixed groupshared tree rather than
// through an atomic. The three P5 golden cells render byte-identical run
// to run in Release and Debug, and that is asserted here directly rather
// than inferred: RunGpu twice, require memcmp equality.
//
// Metal only, and skipped (exit 125) when no Metal device or no
// "ocean_cascades" pipeline is available -- the same discovery-not-
// assumption shape tests/rhi_accel_update_test.cpp uses. The kernel is
// compiled to SPIR-V too (and passes spirv-val) but VulkanDevice builds no
// pipeline from it, so there is nothing to compare there yet.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "physics/OceanCascadeDispatch.h"
#include "physics/OceanFFT.h"
#include "rhi/Device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pt::rhi;
using pt::ocean::OceanFFT;

namespace {

// --- The float32 arithmetic the bound is derived from ---------------------
//
// u is the unit roundoff of IEEE-754 binary32: half an ulp at 1.0, i.e.
// 2^-24. Every constant below is a MULTIPLE OF u with a stated origin; none
// of them was chosen to make a measurement pass.
constexpr double kU = 5.9604644775390625e-08;   // 2^-24

// One radix-2 butterfly, excluding the twiddle's own inaccuracy.
//
//   * one complex multiply, evaluated as (ac - bd, ad + bc). The standard
//     bound is |fl(zw) - zw| <= sqrt(2) * gamma_2 * |z||w| with
//     gamma_2 = 2u/(1-2u) ~ 2u, i.e. 2*sqrt(2)*u relative.
//   * two complex adds, each contributing at most u to the stage's output.
//     They act on the same 2-norm, so they count once.
//
// (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
// section 3.6 for the complex multiply and section 24.1 for the FFT.)
constexpr double kButterflyU = 2.0 * 1.4142135623730951 + 1.0;   // 3.828 u

// A directly evaluated twiddle, cos/sin of ang = 2*pi*k/len.
//
//   * the angle is two roundings (the multiply, then the divide), so it
//     carries <= 2u of relative error; ang <= pi and |d cos/d ang| <= 1, so
//     that is <= 2*pi*u absolute on a unit-modulus number;
//   * the transcendental itself: <= 2 ulp, and ulp(x) <= 2u|x| for a
//     normalised |x| <= 1, so <= 4u.
constexpr double kTwiddleDirectU = 2.0 * 3.141592653589793 + 4.0;   // 10.28 u

// A recurrence twiddle, w *= wlen. Each step costs one complex multiply
// (2*sqrt(2)*u) on top of the error already in wlen, which is itself a
// directly evaluated cos/sin pair. The error after k steps is k times this.
constexpr double kTwiddleStepU = 2.0 * 1.4142135623730951 + kTwiddleDirectU;

// The spectrum evaluation that feeds the transform: two cos/sin (<= 2 ulp
// = 4u each, and the argument wt is BIT-IDENTICAL on both sides because
// omega = sqrt(g|k|) is recomputed from the same kx/kz in the same order),
// two complex multiplies and one complex add, then one more complex
// multiply for the i*k or -i*k/|k| factor.
constexpr double kSpectrumU = 2.0 * 4.0 + 3.0 * 2.0 * 1.4142135623730951 + 1.0;

double Log2u(std::uint32_t n) {
    double l = 0.0;
    while ((1u << static_cast<std::uint32_t>(l)) < n) l += 1.0;
    return l;
}

// Relative 2-norm error bound for ONE 1-D radix-2 inverse FFT of length N.
//
// Each stage of the transform is sqrt(2) times a unitary map, so a relative
// perturbation introduced at one stage is neither amplified nor damped by
// the later ones in the 2-norm; the total is the sum over stages.
//
// With directly evaluated twiddles every stage costs the same. With the
// recurrence, stage s (length 2^s) uses twiddle indices up to 2^(s-1) - 1,
// and the 2-norm bound takes the worst index in the stage, so the twiddle
// term sums to (N - 1 - log2 N) steps over the whole transform.
double Fft1dRelErr(std::uint32_t N, bool recurrence_twiddle) {
    const double s = Log2u(N);
    if (recurrence_twiddle) {
        return (s * kButterflyU + kTwiddleStepU * (double(N) - 1.0 - s)) * kU;
    }
    return s * (kButterflyU + kTwiddleDirectU) * kU;
}

// The full CPU-vs-GPU bound on the relative 2-norm difference of any field
// the two solvers pack: a separable 2-D transform is a row pass followed by
// a column pass, and the two solvers' errors add.
double CpuGpuRelErr(std::uint32_t N) {
    const double cpu = 2.0 * Fft1dRelErr(N, /*recurrence_twiddle=*/true);
    const double gpu = 2.0 * Fft1dRelErr(N, /*recurrence_twiddle=*/false);
    return cpu + gpu + kSpectrumU * kU;
}

// --- The cascade layout, read from Engine.h rather than copied -----------
//
// The three periods and the band divisor are the engine's, and
// tests/pt_planet_ocean_test.cpp pins them by text in Engine.h. Parsing
// them here rather than retyping them means this test measures the
// cascades the engine actually runs, and cannot quietly keep testing 1793 m
// after someone changes it.
struct CascadeLayout {
    double period_m[3] = {0.0, 0.0, 0.0};
    double divisor     = 0.0;
    bool   ok          = false;
};

std::string SlurpTight(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out.push_back(c);
    }
    return out;
}

CascadeLayout ReadCascadeLayout() {
    CascadeLayout L;
    const std::string eh = SlurpTight(PT_ENGINE_H_PATH);
    if (eh.empty()) return L;
    const std::string kPer = "kOceanCascadePeriodM[3]={";
    const std::string kDiv = "kOceanBandDivisor=";
    std::size_t p = eh.find(kPer);
    if (p == std::string::npos) return L;
    p += kPer.size();
    const std::size_t close = eh.find('}', p);
    if (close == std::string::npos) return L;
    std::string body = eh.substr(p, close - p);
    for (char& c : body) if (c == ',') c = ' ';
    std::istringstream bs(body);
    for (int i = 0; i < 3; ++i) {
        if (!(bs >> L.period_m[i])) return L;
    }
    std::size_t d = eh.find(kDiv);
    if (d == std::string::npos) return L;
    L.divisor = std::atof(eh.c_str() + d + kDiv.size());
    L.ok = (L.period_m[0] > 0.0 && L.period_m[1] > 0.0 &&
            L.period_m[2] > 0.0 && L.divisor > 0.0);
    return L;
}

double BandHiM(const CascadeLayout& L, int c) {
    return (c == 0) ? L.period_m[0] : L.period_m[c] / L.divisor;
}
double BandLoM(const CascadeLayout& L, int c, std::uint32_t grid) {
    return (c + 1 < 3) ? L.period_m[c + 1] / L.divisor
                       : 2.0 * L.period_m[c] / double(grid ? grid : 1u);
}

// --- The three cascade solvers, configured exactly as the engine does ----
constexpr int   kCascades  = 3;
constexpr float kWindMps   = 12.0f;   // the phase's Beaufort-6 reference
constexpr float kChoppy    = 1.0f;
constexpr float kFoamThr   = 0.5f;
constexpr float kFoamAmt   = 1.0f;
constexpr float kGravity   = 9.81f;
constexpr std::uint32_t kSeed = 1337u;

struct Cascades {
    std::vector<std::unique_ptr<OceanFFT>> s;
    float amplitude = 0.0f;
};

Cascades MakeCascades(const CascadeLayout& L, std::uint32_t grid) {
    Cascades cc;
    for (int c = 0; c < kCascades; ++c) {
        auto solver = std::make_unique<OceanFFT>();
        auto& cfg = solver->MutableConfig();
        cfg.grid_size    = grid;
        cfg.patch_size_m = static_cast<float>(L.period_m[c]);
        cfg.wind_speed   = kWindMps;
        cfg.wind_dir_rad = 0.0f;
        cfg.choppiness   = kChoppy;
        cfg.foam_threshold   = kFoamThr;
        cfg.foam_amount      = kFoamAmt;
        cfg.foam_persistence = 0.92f;
        cfg.foam_coverage    = 1.0f;
        cfg.gravity      = kGravity;
        cfg.seed         = kSeed + 7919u * static_cast<std::uint32_t>(c);
        cfg.band_hi_m    = (c == 0) ? 0.0f
                                    : static_cast<float>(BandHiM(L, c));
        cfg.band_lo_m    = static_cast<float>(BandLoM(L, c, grid));
        cc.s.push_back(std::move(solver));
    }
    // The Cox-Munk amplitude, solved the way Engine::StepOceanCascades
    // solves it: the cascade window's share of the measured mean-square
    // slope, divided by what the un-normalised spectrum delivers. One
    // evaluation, because scaling the spectrum by A scales sigma^2 by A.
    const double lambda_bot = BandLoM(L, kCascades - 1, grid);
    const double lambda_top = L.period_m[0];
    const double target =
        pt::ocean::CoxMunkMeanSquareSlope(kWindMps) *
        pt::ocean::SlopeVarianceFractionInBand(lambda_bot, lambda_top,
                                               kWindMps);
    double raw = 0.0;
    for (auto& s : cc.s) {
        s->MutableConfig().amplitude = 1.0f;
        s->EnsureSpectrum();
        raw += s->SlopeVarianceAbove(0.0);
    }
    cc.amplitude = (raw > 0.0) ? static_cast<float>(target / raw) : 0.0f;
    for (auto& s : cc.s) {
        s->MutableConfig().amplitude = cc.amplitude;
        s->EnsureSpectrum();
    }
    return cc;
}

// --- GPU side -------------------------------------------------------------
struct GpuRun {
    std::vector<float> disp;    // grid x (cascades*grid + 1) RGBA32F
    std::vector<float> nrm;     // grid x (cascades*grid)     RGBA32F
    bool ok = false;
};

struct GpuResources {
    BufferHandle  h0{}, scratch{}, foam{}, reduce{};
    TextureHandle disp{}, nrm{};
    bool ok = false;
};

GpuResources AllocGpu(Device& dev, std::uint32_t grid, std::uint32_t cascades) {
    GpuResources r;
    const std::size_t cells = std::size_t(grid) * grid * cascades;
    const std::size_t tiles_x = (grid + pt::ocean::kGpuGroupDim - 1u) /
                                pt::ocean::kGpuGroupDim;
    r.h0      = dev.CreateBuffer({cells * 4u * sizeof(float),
                                  BufferUsage::Storage, "t_ocean_h0"});
    r.scratch = dev.CreateBuffer({cells * pt::ocean::kGpuFields * 2u *
                                      sizeof(float),
                                  BufferUsage::Storage, "t_ocean_scratch"});
    r.foam    = dev.CreateBuffer({cells * sizeof(float),
                                  BufferUsage::Storage, "t_ocean_foam"});
    r.reduce  = dev.CreateBuffer({(tiles_x * tiles_x * cascades + cascades) *
                                      sizeof(float),
                                  BufferUsage::Storage, "t_ocean_reduce"});
    r.disp = dev.CreateTexture({grid, cascades * grid + 1u,
                                TextureFormat::RGBA32F, TextureUsage::Storage,
                                "t_ocean_disp"});
    r.nrm  = dev.CreateTexture({grid, cascades * grid,
                                TextureFormat::RGBA32F, TextureUsage::Storage,
                                "t_ocean_nrm"});
    r.ok = r.h0.id && r.scratch.id && r.foam.id && r.reduce.id &&
           r.disp.id && r.nrm.id;
    // The foam accumulator is persistent state; a fresh buffer's contents
    // are unspecified, and the CPU solver's equivalent starts at zero.
    if (r.ok) {
        std::vector<float> zeros(cells, 0.0f);
        dev.WriteBuffer(r.foam, zeros.data(), zeros.size() * sizeof(float));
    }
    return r;
}

void FreeGpu(Device& dev, GpuResources& r) {
    if (r.h0.id)      dev.DestroyBuffer(r.h0);
    if (r.scratch.id) dev.DestroyBuffer(r.scratch);
    if (r.foam.id)    dev.DestroyBuffer(r.foam);
    if (r.reduce.id)  dev.DestroyBuffer(r.reduce);
    if (r.disp.id)    dev.DestroyTexture(r.disp);
    if (r.nrm.id)     dev.DestroyTexture(r.nrm);
    r = GpuResources{};
}

// The host-side scalars the engine evaluates for the foam model, so the
// kernel carries no transcendental of its own beyond the FFT's cos/sin.
float FoamDecay(float persistence, float dt) {
    return (persistence > 0.0f) ? std::pow(persistence, dt * 60.0f) : 0.0f;
}
float WindCoverage(float wind_mps, float coverage_gain) {
    constexpr float kOnset = 7.0f, kWidth = 5.0f;
    float w = (wind_mps - (kOnset - kWidth)) / (2.0f * kWidth);
    w = std::clamp(w, 0.0f, 1.0f);
    w = w * w * (3.0f - 2.0f * w);
    w *= std::max(coverage_gain, 0.0f);
    return std::clamp(w, 0.0f, 1.0f);
}

GpuRun RunGpu(Device& dev, PipelineHandle pipe, GpuResources& r,
              const CascadeLayout& L, const Cascades& cc, std::uint32_t grid,
              double t_seconds, float dt) {
    GpuRun out;
    const std::size_t per = std::size_t(grid) * grid;

    // Upload the base spectrum -- BOTH halves, H0(k) and conj(H0(-k)) --
    // exactly as Engine::UploadOceanGpuSpectrum packs it, so the two
    // solvers start from identical inputs and the comparison isolates the
    // transform.
    std::vector<float> h0(per * kCascades * 4u, 0.0f);
    for (int c = 0; c < kCascades; ++c) {
        const auto& a = cc.s[std::size_t(c)]->H0();
        const auto& b = cc.s[std::size_t(c)]->H0Conj();
        REQUIRE(a.size() == per);
        REQUIRE(b.size() == per);
        float* dst = h0.data() + per * std::size_t(c) * 4u;
        for (std::size_t i = 0; i < per; ++i) {
            dst[i * 4u + 0u] = a[i].real();
            dst[i * 4u + 1u] = a[i].imag();
            dst[i * 4u + 2u] = b[i].real();
            dst[i * 4u + 3u] = b[i].imag();
        }
    }
    dev.WriteBuffer(r.h0, h0.data(), h0.size() * sizeof(float));

    pt::ocean::GpuCascadeDispatch d;
    d.pipeline    = pipe.id;
    d.h0_buf      = r.h0.id;
    d.scratch_buf = r.scratch.id;
    d.foam_buf    = r.foam.id;
    d.reduce_buf  = r.reduce.id;
    d.disp_tex    = r.disp.id;
    d.normal_tex  = r.nrm.id;
    d.grid        = grid;
    d.cascades    = kCascades;
    d.t_seconds     = t_seconds;
    d.choppiness    = kChoppy;
    d.foam_threshold = kFoamThr;
    d.foam_amount   = kFoamAmt;
    d.foam_decay    = FoamDecay(0.92f, dt);
    d.wind_coverage = WindCoverage(kWindMps, 1.0f);
    d.bracket_scale = 1.25f;
    d.bracket_bias  = 0.05f;
    d.gravity       = kGravity;
    for (int c = 0; c < 3; ++c) d.period_m[c] = L.period_m[c];

    CommandBuffer* cb = dev.AcquireCommandBuffer();
    if (cb == nullptr) return out;
    pt::ocean::RecordOceanCascades(cb, d);
    dev.Submit(cb);
    dev.WaitIdle();

    out.disp.assign(per * kCascades * 4u + std::size_t(grid) * 4u, 0.0f);
    out.nrm.assign(per * kCascades * 4u, 0.0f);
    std::uint32_t w = 0, h = 0;
    const bool a = dev.ReadbackTexture(r.disp, out.disp.data(),
                                       out.disp.size() * sizeof(float), &w, &h);
    const bool b = dev.ReadbackTexture(r.nrm, out.nrm.data(),
                                       out.nrm.size() * sizeof(float), &w, &h);
    out.ok = a && b;
    return out;
}

// --- Comparison bookkeeping ----------------------------------------------
struct FieldStats {
    double rms_ref  = 0.0;   // RMS of the reference field
    double rms_diff = 0.0;   // RMS of (gpu - cpu)
    double max_diff = 0.0;
    double l2_ref   = 0.0;   // sqrt(sum ref^2) -- the 2-norm the bound uses
    double l2_diff  = 0.0;
    std::size_t n   = 0;
};

FieldStats CompareStride(const std::vector<float>& gpu,
                         const std::vector<float>& cpu,
                         std::size_t count, std::size_t stride,
                         std::size_t gpu_off, std::size_t cpu_off) {
    FieldStats st;
    double sr = 0.0, sd = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double a = gpu[gpu_off + i * stride];
        const double b = cpu[cpu_off + i * stride];
        const double dd = a - b;
        sr += b * b;
        sd += dd * dd;
        mx = std::max(mx, std::fabs(dd));
    }
    st.n = count;
    st.l2_ref  = std::sqrt(sr);
    st.l2_diff = std::sqrt(sd);
    st.rms_ref  = std::sqrt(sr / double(count));
    st.rms_diff = std::sqrt(sd / double(count));
    st.max_diff = mx;
    return st;
}

// --- Backend discovery ----------------------------------------------------
bool g_ran = false;
std::string g_skip;

struct Env {
    std::unique_ptr<Device> dev;
    PipelineHandle pipe{};
    CascadeLayout  layout;
};

Env MakeEnv() {
    Env e;
    e.layout = ReadCascadeLayout();
    if (!e.layout.ok) {
        g_skip = "could not read the cascade layout out of Engine.h";
        return e;
    }
    NativeWindowHandle window{};
    window.opaque = nullptr;
    window.width  = 64;
    window.height = 64;
    e.dev = Device::Create(BackendType::Metal, window);
    if (!e.dev) {
        g_skip = "no Metal device on this host";
        return e;
    }
    ComputePipelineDesc pd{};
    pd.kernel_name = "ocean_cascades";
    pd.debug_name  = "ocean_cascades";
    e.pipe = e.dev->CreateComputePipeline(pd);
    if (!e.pipe) {
        g_skip = "ocean_cascades pipeline unavailable on this device";
        e.dev.reset();
    }
    return e;
}

}  // namespace

// ==========================================================================
TEST_CASE("ocean cascades: the derived bound is what it says it is") {
    // The bound is a function of N alone and is dominated by the HOST's
    // recurrence twiddle. Pin both facts, because if a future edit made the
    // bound grow with something else, or made the GPU term dominant, the
    // number below would stop meaning what its derivation says.
    const double b256 = CpuGpuRelErr(256u);
    const double cpu  = 2.0 * Fft1dRelErr(256u, true);
    const double gpu  = 2.0 * Fft1dRelErr(256u, false);
    CHECK(cpu > 0.9 * b256);          // the CPU's recurrence is >90% of it
    CHECK(gpu < 0.05 * b256);
    // ~4.0e-4 at N = 256. Stated so a reader can check the arithmetic
    // without re-running it: 2 * [8 * 3.828 + 13.11 * (256 - 1 - 8)] u
    // for the host, 2 * 8 * 14.11 u for the kernel.
    CHECK(b256 == doctest::Approx(4.03e-4).epsilon(0.02));
    // It grows ~linearly in N, because the recurrence's worst twiddle index
    // does. Doubling the grid roughly doubles the allowed disagreement.
    CHECK(CpuGpuRelErr(512u) / b256 == doctest::Approx(2.0).epsilon(0.05));
}

TEST_CASE("ocean cascades: GPU field matches the CPU reference (metal)") {
    Env env = MakeEnv();
    if (!env.dev) {
        MESSAGE("SKIP: " << g_skip);
        return;
    }
    g_ran = true;

    // Every grid size Engine::StepOcean snaps r_ocean_grid_size to. 64 and
    // 128 leave most of a 64-lane threadgroup idle in the FFT passes and
    // 512 makes each lane run four butterflies per stage, so the striding
    // is exercised in both directions rather than only at the default.
    for (std::uint32_t grid : {64u, 128u, 256u, 512u}) {
        CAPTURE(grid);
        GpuResources res = AllocGpu(*env.dev, grid, kCascades);
        REQUIRE(res.ok);

        Cascades cc = MakeCascades(env.layout, grid);
        // A time a golden capture actually reaches: 48 frames of
        // wall-clock dt is a fraction of a second, and 120 frames a couple
        // of seconds. 1.75 s is inside that and is not a multiple of any
        // cascade's period, so no mode sits at a trivial phase.
        const double t  = 1.75;
        const float  dt = 1.0f / 60.0f;

        GpuRun g = RunGpu(*env.dev, env.pipe, res, env.layout, cc, grid, t, dt);
        REQUIRE(g.ok);

        // The host reference, on the SAME spectrum objects the upload came
        // from. Update() re-checks the spectrum and finds it already built,
        // so H0 is not re-drawn and the two paths really do share an input.
        for (auto& s : cc.s) s->Update(t);

        const double bound = CpuGpuRelErr(grid);
        const std::size_t per = std::size_t(grid) * grid;

        double worst_h_ratio = 0.0;
        for (int c = 0; c < kCascades; ++c) {
            CAPTURE(c);
            const auto& cpu_d = cc.s[std::size_t(c)]->DisplacementRGBA();
            const auto& cpu_n = cc.s[std::size_t(c)]->NormalRGBA();
            REQUIRE(cpu_d.size() == per * 4u);
            const std::size_t off = per * std::size_t(c) * 4u;

            // --- The height field. This is the one the bound is about:
            // it is the inverse transform's own output, with nothing but
            // the (-1)^(x+y) sign fold applied on top.
            FieldStats h = CompareStride(g.disp, cpu_d, per, 4u, off + 1u, 1u);
            // The field has to BE something. Cox & Munk at 12 m/s through
            // the cascade window puts metres of swell in cascade 0 and
            // centimetres of chop in cascade 2; anything at 1e-6 m would
            // mean the solver produced nothing and every ratio below would
            // pass vacuously.
            CHECK(h.rms_ref > 1.0e-4);
            // The derived statement, exactly: ||gpu - cpu||_2 <= eps * ||cpu||_2.
            CHECK(h.l2_diff <= bound * h.l2_ref);
            worst_h_ratio = std::max(worst_h_ratio,
                                     h.l2_diff / std::max(h.l2_ref, 1e-30));
            MESSAGE("grid " << grid << " cascade " << c
                    << ": h rms " << h.rms_ref << " m, l2 diff/ref "
                    << (h.l2_diff / h.l2_ref) << " (bound " << bound << ")");

            // --- The two lateral displacement channels, same statement.
            FieldStats dx = CompareStride(g.disp, cpu_d, per, 4u, off + 0u, 0u);
            FieldStats dz = CompareStride(g.disp, cpu_d, per, 4u, off + 2u, 2u);
            CHECK(dx.l2_diff <= bound * dx.l2_ref);
            CHECK(dz.l2_diff <= bound * dz.l2_ref);

            // --- The normal. It is normalize(-dh/dx, 1, -dh/dz), so its
            // components are a smooth function of the slope field with
            // gradient bounded by 1 (the map s -> s/sqrt(1+|s|^2) is
            // 1-Lipschitz), plus one rsqrt and one multiply. The slopes
            // themselves carry the transform's error scaled by |k| <= the
            // Nyquist wavenumber, but that scaling is ALREADY inside the
            // transform here -- the solver transforms i*k*H directly, so
            // the packed gradient is a transformed field in its own right
            // and obeys the same relative bound. Two extra roundings for
            // the normalisation.
            const double nb = bound + 2.0 * kU;
            for (std::size_t comp : {std::size_t(0), std::size_t(1),
                                     std::size_t(2)}) {
                FieldStats n = CompareStride(g.nrm, cpu_n, per, 4u,
                                             off + comp, comp);
                CAPTURE(comp);
                CHECK(n.l2_diff <= nb * n.l2_ref);
            }

            // --- Foam. NOT a transformed field: it is
            //   clamp((thr - det J) / thr, 0, 1)
            // maxed with a crest-gated wind term, then run through the
            // persistence trail. The clamp makes it 1-Lipschitz in its
            // argument and the division by thr scales by 1/thr, while
            // det J is built from central differences of the lateral
            // displacement over one grid cell, which divides by 2*dxz.
            // So a displacement error of e metres moves foam by at most
            //   2 * (1 + |dD/dx|) * e / (2 * dxz * thr),
            // and the (1 + |dD/dx|) factor is bounded by the Jacobian
            // staying near 1 on an unfolded surface. Rather than assert
            // that chain with a hand-picked |dD/dx|, assert the two
            // consequences that matter: foam is a COVERAGE, so it must
            // stay in [0,1], and the two solvers must agree on it to well
            // inside the 1/255 an 8-bit golden can even represent.
            double foam_max = 0.0, foam_l2 = 0.0, foam_ref_l2 = 0.0;
            for (std::size_t i = 0; i < per; ++i) {
                const double a = g.disp[off + i * 4u + 3u];
                const double b = cpu_d[i * 4u + 3u];
                CHECK(a >= 0.0);
                CHECK(a <= 1.0);
                foam_max = std::max(foam_max, std::fabs(a - b));
                foam_l2 += (a - b) * (a - b);
                foam_ref_l2 += b * b;
            }
            CHECK(foam_max < 1.0 / 255.0);
            MESSAGE("grid " << grid << " cascade " << c << ": foam rms "
                    << std::sqrt(foam_ref_l2 / double(per)) << ", max diff "
                    << foam_max);
        }

        // --- The march bracket in the metadata row. -----------------------
        // The GPU reduces the per-cascade peak |height| and writes
        // scale * sum + bias into ocean_displacement[uint2(x, cascades*grid)]
        // for every x. oceanRayMarchShell reads texel 0 of that row; the
        // whole row is written so nothing there is ever undefined.
        {
            const std::size_t meta = per * std::size_t(kCascades) * 4u;
            double cpu_sum = 0.0;
            for (int c = 0; c < kCascades; ++c) {
                cpu_sum += double(cc.s[std::size_t(c)]->MaxDisplacementY());
            }
            const double cpu_bracket = cpu_sum * 1.25 + 0.05;
            const double gpu_bracket = g.disp[meta];
            CHECK(cpu_bracket > 0.05);       // there IS a wave band
            // The peak is a max over the same field, so it inherits the
            // field's bound -- via the peak texel's own error, which the
            // 2-norm bound covers, plus the float sum of three peaks where
            // the host sums in double.
            const double bracket_tol =
                1.25 * (bound * cpu_sum + 3.0 * kU * cpu_sum) + 1e-9;
            CHECK(std::fabs(gpu_bracket - cpu_bracket) <= bracket_tol);
            for (std::uint32_t x = 1u; x < grid; ++x) {
                REQUIRE(g.disp[meta + std::size_t(x) * 4u] == g.disp[meta]);
            }
        }

        // --- RED-THEN-GREEN, built in. -----------------------------------
        // A bound this loose has to be shown to reject something. Compare
        // cascade 0's GPU height against cascade 1's CPU height: same grid,
        // same spectrum family, same amplitude solve, different band. If
        // that passed, the test would be measuring nothing.
        if (kCascades > 1) {
            const auto& other = cc.s[1]->DisplacementRGBA();
            FieldStats wrong = CompareStride(g.disp, other, per, 4u, 1u, 1u);
            CHECK(wrong.l2_diff > bound * wrong.l2_ref);
        }
        // ...and the real comparison is not passing by being trivially
        // zero either: the two solvers DO differ, in the last bits, which
        // is what the bound exists to allow.
        CHECK(worst_h_ratio > 0.0);

        FreeGpu(*env.dev, res);
    }
}

TEST_CASE("ocean cascades: the GPU field is bit-identical run to run (metal)") {
    // The three P5 golden cells render byte-identical run to run in
    // Release and in Debug, and a GPU solver that gave that up would be
    // worse than the CPU one however fast it was. The kernel's only
    // reduction is over max(), which is exact and order-free in IEEE-754,
    // and it runs as a fixed groupshared tree rather than through an
    // atomic -- so the order is not merely irrelevant, it is pinned.
    //
    // Asserted as memcmp over the WHOLE field, both atlases, rather than
    // inferred from a rendered image: a golden is 8-bit and would hide a
    // difference in the low mantissa bits that a longer march could later
    // amplify into a visible one.
    Env env = MakeEnv();
    if (!env.dev) {
        MESSAGE("SKIP: " << g_skip);
        return;
    }
    g_ran = true;

    const std::uint32_t grid = 256u;
    GpuResources res = AllocGpu(*env.dev, grid, kCascades);
    REQUIRE(res.ok);
    Cascades cc = MakeCascades(env.layout, grid);

    GpuRun first;
    for (int run = 0; run < 8; ++run) {
        CAPTURE(run);
        // Re-zero the foam accumulator so each run starts from the same
        // state; the trail is the only thing that carries between frames.
        const std::size_t cells = std::size_t(grid) * grid * kCascades;
        std::vector<float> zeros(cells, 0.0f);
        env.dev->WriteBuffer(res.foam, zeros.data(),
                             zeros.size() * sizeof(float));
        GpuRun g = RunGpu(*env.dev, env.pipe, res, env.layout, cc, grid,
                          1.75, 1.0f / 60.0f);
        REQUIRE(g.ok);
        if (run == 0) {
            first = std::move(g);
            // Vacuity guard: a field of all zeros would be bit-identical
            // eight times over and would mean nothing.
            double e = 0.0;
            for (std::size_t i = 1; i < first.disp.size(); i += 4) {
                e += double(first.disp[i]) * double(first.disp[i]);
            }
            REQUIRE(e > 0.0);
            continue;
        }
        REQUIRE(g.disp.size() == first.disp.size());
        REQUIRE(g.nrm.size()  == first.nrm.size());
        CHECK(std::memcmp(g.disp.data(), first.disp.data(),
                          g.disp.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(g.nrm.data(), first.nrm.data(),
                          g.nrm.size() * sizeof(float)) == 0);
    }
    FreeGpu(*env.dev, res);
}

int main(int argc, char** argv) {
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int rc = ctx.run();
    if (ctx.shouldExit()) return rc;
    if (rc != 0) return rc;
    // A test that silently passes when it did not run is the #252 failure
    // mode. Report exit 125 (ctest SKIP_RETURN_CODE) so a host without a
    // usable Metal device reads as skipped, never as green.
    if (!g_ran) {
        std::fprintf(stderr, "pt_ocean_gpu: SKIPPED -- %s\n",
                     g_skip.empty() ? "no reason recorded" : g_skip.c_str());
        return 125;
    }
    return 0;
}

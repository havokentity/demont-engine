// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Land cover (#300): the surface albedo raster, the two physical models
// that modulate it, and the wire formats between them.
//
// WHY THIS FILE EXISTS
//
// The visible half of this phase is a golden image, and a golden image is
// exactly the wrong instrument for most of what the phase claims:
//
//   1. THE FIXTURE THIS PHASE IS ABOUT IS A QUARTER OF ONE TEXEL ACROSS.
//      planet_surface stands at 27.99 N with a 4.7 km horizon, and the
//      raster's texel is 19.5 km. The raster contributes ONE colour to
//      that whole frame, so no surface-level golden can tell a correct
//      raster from a wrong one -- only from an absent one.
//   2. A GOLDEN CANNOT FAIL FOR THE RIGHT REASON. "Earth from orbit is
//      recognisable" is a claim about SPECTRAL SHAPE -- that the Congo is
//      green because chlorophyll absorbs red and blue, and the Sahara is
//      tan because iron oxide absorbs blue. A pixel comparison against a
//      committed PNG passes just as well if both are grey.
//   3. THE CPU AND GPU SAMPLERS ARE ONE ALGORITHM WITH TWO
//      IMPLEMENTATIONS. src/renderer/Planet/SurfaceAlbedo.cpp and
//      ptLandAlbedoSample() in shaders/PathTrace.slang have no generated
//      code between them, and src/rhi_software has no terrain path at all,
//      so nothing at runtime compares them.
//
// COUNT OCCURRENCES, DO NOT TEST FOR PRESENCE, and NORMALISE LINE ENDINGS.
// A `find() != npos` pin passes as soon as one site matches, which is how
// #276 shipped a stale duplicate of planetAltitude while its test stayed
// green. And a pattern containing '\n' never matches a CRLF checkout, which
// is how #280's kPtSolarIrradiance pin passed on macOS and failed only on
// Windows.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "renderer/Planet/SurfaceAlbedo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pt::planet;

namespace {

std::string NormaliseNewlines(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    return s;
}

std::string Slurp(const char* path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE_MESSAGE(f.good(), "cannot open ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return NormaliseNewlines(ss.str());
}

// The inverse, so the CRLF case below can hand the pins what Git gives a
// Windows checkout.
std::string ToCrlf(const std::string& s) {
    std::string out;
    out.reserve(s.size() + s.size() / 16);
    for (char c : s) {
        if (c == '\n') out.push_back('\r');
        out.push_back(c);
    }
    return out;
}

std::string StripLineComments(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool in_comment = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (!in_comment && src[i] == '/' && i + 1 < src.size()
            && src[i + 1] == '/') {
            in_comment = true;
        }
        if (src[i] == '\n') in_comment = false;
        if (!in_comment) out.push_back(src[i]);
    }
    return out;
}

std::size_t CountOccurrences(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos;
         at = hay.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// The shipped raster, loaded once. Every case that needs it REQUIREs it,
// because a missing asset must fail these tests rather than silently make
// them vacuous -- "the file is not there so the claim holds" is the exact
// shape of a test that cannot fail.
const SurfaceAlbedoMap& Shipped() {
    static SurfaceAlbedoMap m;
    static bool tried = false;
    if (!tried) {
        tried = true;
        std::string err;
        if (!m.Load(PT_PLANET_ALBEDO_PATH, err)) {
            std::fprintf(stderr, "[albedo] %s\n", err.c_str());
        }
    }
    return m;
}

glm::vec4 SampleDeg(const SurfaceAlbedoMap& m, double lat, double lon) {
    return m.SampleAt(lat * kDeg, lon * kDeg);
}

double Luminance(const glm::vec4& c) {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// The law #300 replaced, kept here so the comparison is a measurement and
// not a claim in a comment.
double LegacyCosSnowline(double lat_rad, double anchor_m) {
    return anchor_m * std::max(std::cos(lat_rad), 0.05);
}

// The rock ramp #300 replaced. Reproduced exactly, so the bug it had can be
// pinned as a FAILURE rather than described.
double LegacyRockFraction(double slope01) {
    return std::clamp((slope01 - 0.5) * 2.5, 0.0, 1.0);
}

double Slope01FromTiltDeg(double deg) { return 1.0 - std::cos(deg * kDeg); }

}  // namespace

// ===========================================================================
// The container and the sampler
// ===========================================================================

TEST_CASE("the raster round-trips linear reflectance through the encoding") {
    const std::uint32_t W = 8, H = 4;
    std::vector<float> rgb(static_cast<std::size_t>(W) * H * 3);
    std::vector<std::uint8_t> cov(static_cast<std::size_t>(W) * H, 255);
    // A ramp that spans the interesting range: the dark end is where the
    // gamma-2.0 encoding earns its place.
    for (std::size_t i = 0; i < static_cast<std::size_t>(W) * H; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(W * H - 1);
        rgb[i * 3 + 0] = t;
        rgb[i * 3 + 1] = t * t;
        rgb[i * 3 + 2] = 0.018f;      // closed-canopy forest
    }
    SurfaceAlbedoMap m;
    m.BuildFromLinear(W, H, rgb, cov, kAlbFlagMeasuredAlbedo);
    REQUIRE(m.Width() == W);
    REQUIRE(m.Height() == H);
    CHECK(m.IsMeasuredAlbedo());

    // Sample at exact texel CENTRES, where the smoothstep weights are 0 and
    // the interpolation is the identity, so this measures the encoding and
    // nothing else.
    double worst_rel = 0.0;
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            const double lon = -kPi + (x + 0.5) * 2.0 * kPi / W;
            const double lat = kPi * 0.5 - (y + 0.5) * kPi / H;
            const glm::vec4 got = m.SampleAt(lat, lon);
            const std::size_t i = static_cast<std::size_t>(y) * W + x;
            CHECK(got.w == doctest::Approx(1.0).epsilon(0.005));
            for (int c = 0; c < 3; ++c) {
                const double want = rgb[i * 3 + c];
                const double have = (c == 0) ? got.x : (c == 1) ? got.y : got.z;
                if (want > 1e-4) {
                    worst_rel = std::max(worst_rel,
                                         std::abs(have - want) / want);
                }
            }
        }
    }
    // The gamma-2.0 step at 0.018 is 0.0011, i.e. 5.8% -- so 7% is the
    // encoding's own resolution and not a tolerance chosen to pass.
    CHECK(worst_rel < 0.07);
    CHECK(worst_rel > 0.0);   // not vacuous: quantisation IS observable
}

TEST_CASE("a linear byte would have been four times worse at the dark end") {
    // The reason kAlbedoGamma exists, as a measurement rather than a claim
    // in the header. If this ever stops holding, the encoding choice has
    // stopped being justified.
    const double a = 0.018;                       // closed-canopy forest
    const double linear_step = 1.0 / 255.0;
    const double gamma_step = 2.0 * std::sqrt(a) / 255.0;
    CHECK(linear_step / a > 0.20);                // 22%
    CHECK(gamma_step / a < 0.07);                 // 5.8%
    CHECK(linear_step / gamma_step > 3.5);
}

TEST_CASE("longitude wraps and latitude clamps, exactly as the DEM's do") {
    const std::uint32_t W = 16, H = 8;
    std::vector<float> rgb(static_cast<std::size_t>(W) * H * 3, 0.0f);
    std::vector<std::uint8_t> cov(static_cast<std::size_t>(W) * H, 255);
    // Mark the first and last COLUMN differently so a wrap is observable.
    for (std::uint32_t y = 0; y < H; ++y) {
        rgb[(static_cast<std::size_t>(y) * W + 0) * 3 + 0] = 1.0f;
        rgb[(static_cast<std::size_t>(y) * W + (W - 1)) * 3 + 2] = 1.0f;
    }
    SurfaceAlbedoMap m;
    m.BuildFromLinear(W, H, rgb, cov, 0);

    // A longitude one full turn away must give the same answer bit for bit.
    for (double lon = -179.0; lon < 180.0; lon += 37.0) {
        const glm::vec4 a = SampleDeg(m, 12.0, lon);
        const glm::vec4 b = SampleDeg(m, 12.0, lon + 360.0);
        CHECK(a.x == doctest::Approx(b.x));
        CHECK(a.z == doctest::Approx(b.z));
    }
    // Past the pole, latitude clamps rather than wrapping into the other
    // hemisphere. Sampling beyond +90 must equal sampling AT the cap row.
    const glm::vec4 cap = SampleDeg(m, 89.999, 40.0);
    const glm::vec4 over = SampleDeg(m, 120.0, 40.0);
    CHECK(over.x == doctest::Approx(cap.x).epsilon(1e-5));
}

TEST_CASE("the interpolation is C1, which plain bilinear is not") {
    // The banding claim, measured. A checkerboard is the worst case: plain
    // bilinear's derivative flips sign at every texel edge, and magnified
    // 20x -- which is what a 19.5 km texel is from orbit -- that shows as a
    // diamond lattice over every continent.
    const std::uint32_t W = 16, H = 8;
    std::vector<float> rgb(static_cast<std::size_t>(W) * H * 3, 0.0f);
    std::vector<std::uint8_t> cov(static_cast<std::size_t>(W) * H, 255);
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            const float v = ((x + y) & 1u) ? 1.0f : 0.0f;
            rgb[(static_cast<std::size_t>(y) * W + x) * 3 + 1] = v;
        }
    }
    SurfaceAlbedoMap m;
    m.BuildFromLinear(W, H, rgb, cov, 0);

    // Walk a row of longitude and measure the SECOND difference. For a C1
    // interpolant it stays bounded; for a C0 one it spikes at every texel
    // edge, where the first derivative jumps.
    //
    // AT A TEXEL-ROW CENTRE, not at latitude 0. The first version of this
    // case sampled the equator, which for H = 8 lands exactly between rows
    // 3 and 4 -- and in a checkerboard those rows are complements, so the
    // 50/50 blend is a CONSTANT and the whole measurement was of a flat
    // line. It passed the C1 bound trivially. The non-vacuity check below
    // is what caught it, which is the argument for having one.
    const std::size_t probe_row = 3;
    const double lat = kPi * 0.5 - (probe_row + 0.5) * kPi / H;
    const double lon0 = -kPi;
    const int N = 4000;
    const double dl = 2.0 * kPi / N;
    double worst_second = 0.0;
    std::vector<double> v(N + 2);
    for (int i = 0; i < N + 2; ++i) {
        v[static_cast<std::size_t>(i)] = m.SampleAt(lat, lon0 + i * dl).y;
    }
    for (int i = 1; i + 1 < N + 2; ++i) {
        const double d2 = v[static_cast<std::size_t>(i + 1)]
                        - 2.0 * v[static_cast<std::size_t>(i)]
                        + v[static_cast<std::size_t>(i - 1)];
        worst_second = std::max(worst_second, std::abs(d2));
    }
    // Analytic bound: the smoothstep-weighted interpolant's second
    // derivative is at most 6*|delta| per unit parameter, so over a step of
    // dl covering (dl*W/2pi) of a texel the second difference is bounded by
    // 6*(dl*W/2pi)^2. Plain bilinear's is |delta| itself at every edge, a
    // factor of ~1/(step^2) larger.
    const double frac_step = dl * W / (2.0 * kPi);
    const double c1_bound = 6.0 * frac_step * frac_step * 1.05;
    CHECK(worst_second < c1_bound);
    // Not vacuous: the signal really does swing the full range.
    CHECK(*std::max_element(v.begin(), v.end())
          - *std::min_element(v.begin(), v.end()) > 0.5);
    // And the negative pole: a plain-bilinear reference over the SAME data
    // must break the same bound, or the bound is not measuring anything.
    double worst_linear = 0.0;
    std::vector<double> lv(N + 2);
    for (int i = 0; i < N + 2; ++i) {
        const double lon = lon0 + i * dl;
        const double fx = (lon + kPi) / (2.0 * kPi) * W - 0.5;
        const double x0 = std::floor(fx);
        const double t = fx - x0;                     // NO smoothstep
        auto fetch = [&](long xx) {
            const long w = static_cast<long>(W);
            xx = ((xx % w) + w) % w;
            return static_cast<double>(
                rgb[(probe_row * W + static_cast<std::size_t>(xx)) * 3 + 1]);
        };
        lv[static_cast<std::size_t>(i)] =
            fetch(static_cast<long>(x0)) * (1.0 - t)
            + fetch(static_cast<long>(x0) + 1) * t;
    }
    for (int i = 1; i + 1 < N + 2; ++i) {
        const double d2 = lv[static_cast<std::size_t>(i + 1)]
                        - 2.0 * lv[static_cast<std::size_t>(i)]
                        + lv[static_cast<std::size_t>(i - 1)];
        worst_linear = std::max(worst_linear, std::abs(d2));
    }
    CHECK(worst_linear > c1_bound * 10.0);
}

TEST_CASE("a truncated raster is refused rather than baked into a planet") {
    // The DEM's scar, applied here from the start: the first ETOPO download
    // stopped at 30%, curl exited 0, and `file` still called it valid HDF5
    // because the header survived.
    const std::uint32_t W = 32, H = 16;
    std::vector<float> rgb(static_cast<std::size_t>(W) * H * 3, 0.25f);
    std::vector<std::uint8_t> cov(static_cast<std::size_t>(W) * H, 255);
    SurfaceAlbedoMap src;
    src.BuildFromLinear(W, H, rgb, cov, 0);

    const std::string path = "pt_albedo_trunc_test.ptalb";
    const std::size_t body = static_cast<std::size_t>(W) * H * 4;
    {
        std::ofstream f(path, std::ios::binary);
        f.write(kAlbedoMagic, 8);
        const std::uint32_t w = W, h = H, flags = 0, res = 0;
        const double scale = 1.0, off = 0.0;
        f.write(reinterpret_cast<const char*>(&w), 4);
        f.write(reinterpret_cast<const char*>(&h), 4);
        f.write(reinterpret_cast<const char*>(&scale), 8);
        f.write(reinterpret_cast<const char*>(&off), 8);
        f.write(reinterpret_cast<const char*>(&flags), 4);
        f.write(reinterpret_cast<const char*>(&res), 4);
        // 30% of the samples, exactly the DEM's failure.
        std::vector<char> zeros(body * 3 / 10, 0);
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    SurfaceAlbedoMap m;
    std::string err;
    CHECK_FALSE(m.Load(path, err));
    CHECK(m.Empty());
    CHECK(err.find("length") != std::string::npos);
    std::remove(path.c_str());

    // RED THEN GREEN: the same header with the FULL body must load, or the
    // check above is rejecting everything rather than rejecting truncation.
    {
        std::ofstream f(path, std::ios::binary);
        f.write(kAlbedoMagic, 8);
        const std::uint32_t w = W, h = H, flags = 0, res = 0;
        const double scale = 1.0, off = 0.0;
        f.write(reinterpret_cast<const char*>(&w), 4);
        f.write(reinterpret_cast<const char*>(&h), 4);
        f.write(reinterpret_cast<const char*>(&scale), 8);
        f.write(reinterpret_cast<const char*>(&off), 8);
        f.write(reinterpret_cast<const char*>(&flags), 4);
        f.write(reinterpret_cast<const char*>(&res), 4);
        std::vector<char> full(body, 0);
        f.write(full.data(), static_cast<std::streamsize>(full.size()));
    }
    SurfaceAlbedoMap ok;
    CHECK(ok.Load(path, err));
    CHECK_FALSE(ok.Empty());
    std::remove(path.c_str());
}

TEST_CASE("bad magic is refused") {
    const std::string path = "pt_albedo_magic_test.ptalb";
    {
        std::ofstream f(path, std::ios::binary);
        const char bad[8] = {'P','T','D','E','M','0','0','1'};   // the DEM's
        f.write(bad, 8);
        std::vector<char> rest(64, 0);
        f.write(rest.data(), 64);
    }
    SurfaceAlbedoMap m;
    std::string err;
    CHECK_FALSE(m.Load(path, err));
    CHECK(err.find("magic") != std::string::npos);
    std::remove(path.c_str());
}

// ===========================================================================
// The snowline
// ===========================================================================

TEST_CASE("the snowline model reproduces published equilibrium-line altitudes") {
    // Four latitudes with published ELAs, none of which were used to fit
    // the model -- the only anchor is the tropical value the cvar carries.
    struct Row { double lat_deg; double observed_m; const char* site; };
    const Row rows[] = {
        {  0.0, 4900.0, "tropical Andes / East Africa" },
        { 45.0, 2900.0, "European Alps" },
        { 60.0, 1400.0, "southern Norway" },
        { 80.0, 1000.0, "Ellesmere Island" },
    };
    double new_err = 0.0, old_err = 0.0;
    for (const Row& r : rows) {
        const double lat = r.lat_deg * kDeg;
        const double got = SnowlineAltitudeM(lat, kTropicalSnowlineM);
        const double leg = LegacyCosSnowline(lat, kTropicalSnowlineM);
        new_err += std::abs(got - r.observed_m);
        old_err += std::abs(leg - r.observed_m);
        // Every row within 700 m of the published value. That bound is
        // MEASURED -- the worst row is Norway at 655 m, where a zonal mean
        // cannot know about maritime moderation -- not inherited.
        CHECK_MESSAGE(std::abs(got - r.observed_m) < 700.0,
                      r.site, ": modelled ", got, " m against ",
                      r.observed_m, " m");
    }
    new_err /= 4.0;
    old_err /= 4.0;
    // The honest claim: not uniformly better, but roughly half the mean
    // error. If a future change makes it worse than the shape it replaced,
    // that change should have to argue for itself.
    CHECK(new_err < 300.0);
    CHECK(old_err > 400.0);
    CHECK(old_err / new_err > 1.5);
}

TEST_CASE("the snowline falls monotonically from equator to pole") {
    // The property that made the old law LOOK right, and which any
    // replacement still has to have.
    double prev = 1e30;
    int steps = 0;
    for (double d = 0.0; d <= 90.0; d += 2.5) {
        const double s = SnowlineAltitudeM(d * kDeg, kTropicalSnowlineM);
        CHECK(s <= prev + 1e-6);
        CHECK(s >= 0.0);
        prev = s;
        ++steps;
    }
    CHECK(steps >= 30);                       // not vacuous
    // And it is SYMMETRIC about the equator: the model is a function of
    // sin^2(latitude), so the southern hemisphere is not an afterthought.
    for (double d = 5.0; d <= 85.0; d += 10.0) {
        CHECK(SnowlineAltitudeM(d * kDeg, kTropicalSnowlineM)
              == doctest::Approx(SnowlineAltitudeM(-d * kDeg,
                                                   kTropicalSnowlineM)));
    }
    // The anchor means what the cvar says: at the equator it IS the cvar.
    CHECK(SnowlineAltitudeM(0.0, 4900.0) == doctest::Approx(4900.0));
    CHECK(SnowlineAltitudeM(0.0, 1234.0) == doctest::Approx(1234.0));
}

TEST_CASE("the warm-season temperature model is the published zonal fit") {
    // Spot-check the two endpoints of North, Cahalan & Coakley (1981) eq. 2
    // before the seasonal term, so a future edit to the coefficients has to
    // move a number that is tied to a citation.
    //   annual(0)  = T0 + T2/2 = 15 + 14 = 29 degC
    //   annual(90) = T0 - T2   = 15 - 28 = -13 degC
    CHECK(WarmSeasonSeaLevelTempC(0.0)
          == doctest::Approx(29.0 + kSeasonAmpEqK));
    CHECK(WarmSeasonSeaLevelTempC(90.0 * kDeg)
          == doctest::Approx(-13.0 + kSeasonAmpEqK + kSeasonAmpPolarK));
    CHECK(kLapseRateKPerM == doctest::Approx(0.0065));   // ICAO / ISO 2533
}

// ===========================================================================
// Threshold-hillslope rock exposure -- and the bug it replaces
// ===========================================================================

TEST_CASE("the OLD slope ramp exposed no rock below a sixty-degree face") {
    // This is why the Himalayas fixture rendered uniform white, pinned as a
    // measurement so the fix cannot be quietly undone. saturate((slope01 -
    // 0.5) * 2.5) is zero until slope01 = 0.5, which is a tilt of SIXTY
    // DEGREES, and does not saturate until slope01 = 0.9, which is 84.3.
    for (double deg = 0.0; deg <= 60.0; deg += 5.0) {
        CHECK_MESSAGE(LegacyRockFraction(Slope01FromTiltDeg(deg)) == 0.0,
                      "old ramp at ", deg, " deg");
    }
    // It does eventually reach 1 -- at an angle terrain does not have.
    CHECK(LegacyRockFraction(Slope01FromTiltDeg(70.0)) < 0.5);
    CHECK(LegacyRockFraction(Slope01FromTiltDeg(84.3))
          == doctest::Approx(1.0).epsilon(0.01));
    // The replacement is already fully exposed at 45, where the old one is
    // still exactly zero. That gap IS the fix.
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(45.0)) == doctest::Approx(1.0));
    CHECK(LegacyRockFraction(Slope01FromTiltDeg(45.0)) == 0.0);
}

TEST_CASE("rock is exposed across the measured threshold-hillslope band") {
    // Burbank et al. 1996 give ~30 deg as the landsliding threshold in the
    // northwest Himalaya; bedrock cliffs exceed it, so the ramp runs to 45.
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(0.0)) == doctest::Approx(0.0));
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(20.0)) == doctest::Approx(0.0));
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(30.0)) == doctest::Approx(0.0));
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(37.5)) > 0.25);
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(37.5)) < 0.75);
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(45.0)) == doctest::Approx(1.0));
    CHECK(SlopeRockFraction(Slope01FromTiltDeg(60.0)) == doctest::Approx(1.0));
    // Monotone, and it actually MOVES -- the old ramp was monotone too, and
    // was still wrong, because it never left zero.
    double prev = -1.0, span_lo = 1.0, span_hi = 0.0;
    for (double d = 0.0; d <= 90.0; d += 1.0) {
        const double f = SlopeRockFraction(Slope01FromTiltDeg(d));
        CHECK(f >= prev - 1e-9);
        prev = f;
        span_lo = std::min(span_lo, f);
        span_hi = std::max(span_hi, f);
    }
    CHECK(span_hi - span_lo > 0.99);
    CHECK(kThresholdHillslopeDeg == doctest::Approx(30.0));
    CHECK(kRockFullExposureDeg == doctest::Approx(45.0));
}

// ===========================================================================
// The shipped raster -- content, and variation
// ===========================================================================

TEST_CASE("the shipped raster is real land cover, not a brightness map") {
    const SurfaceAlbedoMap& m = Shipped();
    REQUIRE_MESSAGE(!m.Empty(),
                    "assets/planet/earth_lite.ptalb did not load. These "
                    "assertions are about its CONTENT and are worthless "
                    "without it -- bake it with tools/fetch_planet_albedo.py");
    CHECK(m.Width() == 2048);
    CHECK(m.Height() == 1024);
    CHECK_MESSAGE(m.IsMeasuredAlbedo(),
                  "the shipped raster's header says it is NOT measured "
                  "albedo. If the source was deliberately changed, change "
                  "assets/planet/PROVENANCE.md in the same commit");

    // THE CHECK A GREY SPHERE CANNOT PASS. Chlorophyll absorbs at 430 and
    // 662 nm, so vegetation must be greener than it is red or blue. Iron
    // oxide absorbs toward the blue, so soil must run R > G > B. Neither
    // ordering can arise from a uniform albedo, a luminance ramp, or a
    // wrongly-registered raster.
    struct Veg { double lat, lon; const char* name; };
    const Veg veg[] = {
        { -5.0, -62.0, "Amazon" },
        {  0.0,  22.0, "Congo" },
        { 60.0, 100.0, "boreal Siberia" },
        {  2.0, 113.0, "Borneo" },
    };
    for (const Veg& v : veg) {
        const glm::vec4 c = SampleDeg(m, v.lat, v.lon);
        CHECK_MESSAGE(c.w > 0.5, v.name, ": no coverage");
        CHECK_MESSAGE(c.y > c.x, v.name, ": green ", c.y, " not above red ",
                      c.x);
        CHECK_MESSAGE(c.y > c.z, v.name, ": green ", c.y, " not above blue ",
                      c.z);
    }
    const Veg soil[] = {
        {  23.0,  12.0, "Sahara" },
        {  22.0,  45.0, "Arabia" },
        { -25.0, 130.0, "Australian interior" },
        {  33.0,  88.0, "Tibetan plateau" },
    };
    for (const Veg& s : soil) {
        const glm::vec4 c = SampleDeg(m, s.lat, s.lon);
        CHECK_MESSAGE(c.w > 0.5, s.name, ": no coverage");
        CHECK_MESSAGE(c.x > c.y, s.name, ": red ", c.x, " not above green ",
                      c.y);
        CHECK_MESSAGE(c.y > c.z, s.name, ": green ", c.y, " not above blue ",
                      c.z);
    }
}

TEST_CASE("the shipped raster's brightness ordering is Earth's") {
    const SurfaceAlbedoMap& m = Shipped();
    REQUIRE(!m.Empty());
    // ORDERING, not magnitude. The absolute level depends on which MODIS
    // product a future bake uses; the ordering is a property of the planet
    // and holds for every one of them.
    const double amazon = Luminance(SampleDeg(m, -5.0, -62.0));
    const double sahara = Luminance(SampleDeg(m,  23.0,  12.0));
    const double tibet  = Luminance(SampleDeg(m,  33.0,  88.0));
    const double antarc = Luminance(SampleDeg(m, -80.0,  20.0));
    CHECK(sahara > tibet);
    CHECK(tibet > amazon);
    CHECK(antarc > sahara);
    // Rainforest is the darkest land there is, and a raster that has it
    // above 0.10 has lost its dark end to a bad encoding or a bad stretch.
    CHECK(amazon < 0.10);
    CHECK(amazon > 0.001);
    // And the desert is unambiguously bright.
    CHECK(sahara > 0.15);
}

TEST_CASE("the shipped raster VARIES -- a uniform one is the bug being fixed") {
    const SurfaceAlbedoMap& m = Shipped();
    REQUIRE(!m.Empty());
    // A feature-deleted vacuity check passes on a black frame and on a
    // uniform one, so this asserts on the SPREAD. Walk the covered land and
    // measure how much of the reflectance range it actually occupies.
    double lo = 1e30, hi = -1e30, sum = 0.0, wsum = 0.0;
    std::size_t covered = 0, total = 0;
    int buckets[10] = {0};
    for (double lat = -85.0; lat <= 85.0; lat += 0.5) {
        for (double lon = -179.0; lon < 180.0; lon += 0.5) {
            ++total;
            const glm::vec4 c = SampleDeg(m, lat, lon);
            if (c.w < 0.5) continue;
            ++covered;
            const double y = Luminance(c);
            lo = std::min(lo, y);
            hi = std::max(hi, y);
            const double w = std::cos(lat * kDeg);
            sum += y * w;
            wsum += w;
            buckets[std::min(9, static_cast<int>(y * 10.0))]++;
        }
    }
    REQUIRE(covered > 1000);
    // Land is ~29% of the globe; this walk is uniform in lat/lon so it
    // over-samples the poles, where the ice caps are. 20%-50% is the band
    // that leaves.
    const double frac = static_cast<double>(covered) / static_cast<double>(total);
    CHECK(frac > 0.20);
    CHECK(frac < 0.50);
    // The spread. A uniform raster gives hi - lo == 0 and one non-empty
    // bucket; this is the assertion the old single-albedo terrain fails.
    CHECK(hi - lo > 0.6);
    int nonempty = 0;
    for (int b : buckets) if (b > covered / 500) ++nonempty;
    CHECK_MESSAGE(nonempty >= 5,
                  "only ", nonempty, " of 10 reflectance deciles are "
                  "populated -- the raster is not carrying land cover");
    // Area-weighted mean land reflectance. Trenberth, Fasullo & Kiehl 2009
    // give 0.15 for the SHORTWAVE land mean; the visible-band mean must be
    // in the same neighbourhood and below 0.30, which is where an
    // unweighted mean (over-counting the icy poles) would land.
    const double mean = sum / wsum;
    CHECK(mean > 0.08);
    CHECK(mean < 0.30);
}

TEST_CASE("ocean has no coverage, so the water material keeps it") {
    const SurfaceAlbedoMap& m = Shipped();
    REQUIRE(!m.Empty());
    // Deep ocean, far from any coast at this raster's 19.5 km texel.
    const std::pair<double, double> sea[] = {
        {  0.0, -30.0},      // equatorial Atlantic
        {-40.0, -120.0},     // south Pacific
        { 20.0,  -50.0},     // Sargasso
        {-55.0,   90.0},     // southern Indian
    };
    for (const auto& s : sea) {
        const glm::vec4 c = SampleDeg(m, s.first, s.second);
        CHECK_MESSAGE(c.w < 0.5, "coverage ", c.w, " over open ocean at ",
                      s.first, ",", s.second);
    }
}

// ===========================================================================
// The two implementations of one algorithm
// ===========================================================================

TEST_CASE("the shader mirrors the CPU sampler, and says so once each") {
    const std::string raw = Slurp(PT_SHADER_PATHTRACE_PATH);
    const std::string src = StripLineComments(raw);

    // The binding. 47 and slot 22, and NOT any of the reserved numbers.
    CHECK(CountOccurrences(src,
        "[[vk::binding(47, 0)]] StructuredBuffer<uint> land_albedo;") == 1);
    for (const char* reserved : {"vk::binding(40, 0)", "vk::binding(41, 0)",
                                 "vk::binding(42, 0)", "vk::binding(43, 0)",
                                 "vk::binding(44, 0)", "vk::binding(45, 0)"}) {
        CHECK_MESSAGE(CountOccurrences(src, reserved) == 0,
                      reserved, " is RESERVED by OceanCascades.slang (#293) "
                      "and is deliberately absent from the shared Vulkan "
                      "layout -- declaring it is a hard "
                      "vkCreateComputePipelines failure");
    }

    // The registration expression, which must be the DEM's byte for byte or
    // a coastline and the colour beside it index different cells.
    CHECK(CountOccurrences(src, "(lon_rad + kPiF) / (2.0 * kPiF) * W - 0.5") == 1);
    CHECK(CountOccurrences(src, "(kPiF * 0.5 - lat_rad) / kPiF * H - 0.5") == 1);

    // The geodetic conversion, mirroring DirectionToGeodetic. Getting this
    // wrong by using the GEOCENTRIC latitude shifts every biome band by a
    // texel at mid-latitude.
    CHECK(CountOccurrences(src,
        "atan2(kWgs84SemiMajorM * e.z, kWgs84SemiMinorM * p)") == 1);

    // The gamma-2.0 decode, once and only once.
    CHECK(CountOccurrences(src, "float4(r * r, g * g, b * b, a)") == 1);

    // The snowline model's coefficients, tied to their citations by the
    // header. If someone edits one side only, this fails.
    CHECK(CountOccurrences(src, "(15.0 - 28.0 * p2) + (2.0 + 18.0 * s2)") == 1);

    // The threshold-hillslope band, and the ABSENCE of the ramp it
    // replaced. Pinning the removal matters as much as pinning the
    // addition: a stale second copy is exactly the #276 failure.
    CHECK(CountOccurrences(src, "smoothstep(0.1340, 0.2929, slope01)") == 1);
    CHECK_MESSAGE(CountOccurrences(src, "(slope01 - 0.5) * 2.5") == 0,
                  "the 60-degree rock ramp that made the Himalayas white is "
                  "still in the shader");

    // The old entry point must be GONE, not merely shadowed.
    CHECK(CountOccurrences(src, "terrainBiomeAlbedo") == 0);
    CHECK(CountOccurrences(src, "terrainSurfaceAlbedo(") == 2);   // def + call
    CHECK(CountOccurrences(src, "terrainProceduralAlbedo(") == 2);
}

TEST_CASE("the source pins survive a CRLF checkout") {
    // #280's kPtSolarIrradiance pin passed on macOS and Linux and failed
    // only on Windows, because Git checks out CRLF there and a pattern
    // containing '\n' can never match. Every pin above is single-line, and
    // this case proves it by running them against CRLF text.
    const std::string lf = Slurp(PT_SHADER_PATHTRACE_PATH);
    const std::string crlf = NormaliseNewlines(ToCrlf(lf));
    CHECK(crlf == lf);
    const std::string src = StripLineComments(crlf);
    CHECK(CountOccurrences(src,
        "[[vk::binding(47, 0)]] StructuredBuffer<uint> land_albedo;") == 1);
    CHECK(CountOccurrences(src, "(15.0 - 28.0 * p2) + (2.0 + 18.0 * s2)") == 1);
    // And the bug: an un-normalised CRLF read must NOT match a multi-line
    // pattern, which is the failure mode being guarded.
    const std::string raw_crlf = ToCrlf(lf);
    CHECK(raw_crlf.find("land_albedo;\n") == std::string::npos);
}

TEST_CASE("the Vulkan layout declares binding 47 and still omits 40..45") {
    const std::string src = Slurp(PT_VULKAN_DEVICE_PATH);
    // Slot table, layout and pool must move together. Any one of the three
    // alone is a silent failure on a backend this machine cannot run.
    CHECK(CountOccurrences(src,
        "47, // engine slot 22 -> shader binding 47 (land_albedo)") == 1);
    CHECK(CountOccurrences(src,
        "add_binding(47, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);") == 1);
    CHECK(CountOccurrences(src, "kTotalSets * 22 + 8") == 1);
    CHECK_MESSAGE(CountOccurrences(src, "kTotalSets * 21 + 8") == 0,
                  "the pre-#300 storage-buffer pool count is still here -- "
                  "pin the ABSENCE of the old number, not just the presence "
                  "of the new one");
    for (int b = 40; b <= 45; ++b) {
        const std::string pat = "add_binding(" + std::to_string(b) + ",";
        CHECK_MESSAGE(CountOccurrences(src, pat) == 0,
                      "binding ", b, " is reserved by OceanCascades.slang");
    }
    // Slot 22 was the last free entry in kSlotToBufBinding. Pin that the
    // next phase is told so.
    CHECK(CountOccurrences(src, "0,  // engine slot 23 unused") == 1);
}

TEST_CASE("the push-constant budget is one shared number, guarded at compile time") {
    // The bug this phase tripped: MetalDevice.h and VulkanDevice.h each had
    // a bare 2048-byte staging array, sizeof(PtPush) reached 2064, and BOTH
    // silently dropped the last 16 bytes -- so the terrain ignored its new
    // raster, on two backends, with no error anywhere.
    const std::string types = Slurp(PT_RHI_TYPES_PATH);
    CHECK(CountOccurrences(types,
        "inline constexpr std::size_t kMaxPushConstantBytes = 4096;") == 1);

    const std::string metal = Slurp(PT_METAL_DEVICE_PATH);
    const std::string vulkan = Slurp(PT_VULKAN_DEVICE_H_PATH);
    CHECK(CountOccurrences(metal,
        "push_buf_[pt::rhi::kMaxPushConstantBytes]") == 1);
    CHECK(CountOccurrences(vulkan,
        "push_buf_[pt::rhi::kMaxPushConstantBytes]") == 1);
    // The bare literals must be GONE from both, or one backend can drift.
    CHECK(CountOccurrences(metal, "push_buf_[2048]") == 0);
    CHECK(CountOccurrences(vulkan, "push_buf_[2048]") == 0);

    // And the compile-time guard that makes the next overflow a build error
    // rather than a field that reads zero.
    const std::string engine = Slurp(PT_ENGINE_CPP_PATH);
    CHECK(CountOccurrences(engine,
        "static_assert(sizeof(PtPush) <= pt::rhi::kMaxPushConstantBytes,") == 1);
    // The three tail lanes this phase added, mirrored on all three sides.
    CHECK(CountOccurrences(engine, "float land_basis_e[4];") == 1);
    CHECK(CountOccurrences(engine, "float land_basis_u[4];") == 1);
    CHECK(CountOccurrences(engine, "float land_basis_s[4];") == 1);
    const std::string shader = StripLineComments(Slurp(PT_SHADER_PATHTRACE_PATH));
    // Once in the Metal Push cbuffer, once in the SPIR-V Frame mirror.
    CHECK(CountOccurrences(shader, "float4 land_basis_e;") == 2);
    CHECK(CountOccurrences(shader, "float4 land_basis_u;") == 2);
    CHECK(CountOccurrences(shader, "float4 land_basis_s;") == 2);
}

TEST_CASE("the engine states the snowline anchor unscaled") {
    // planet_terrain.x changed meaning: it used to be pre-multiplied by
    // cos(site latitude) on the host, which is what made a planet have one
    // snowline. The shader now derives the local value, so the host must
    // pass the anchor through untouched.
    const std::string engine = StripLineComments(Slurp(PT_ENGINE_CPP_PATH));
    CHECK(CountOccurrences(engine,
        "push.planet_terrain[0] = static_cast<float>(snowline);") == 1);
    CHECK_MESSAGE(CountOccurrences(engine, "snowline * std::max(std::cos(lat)")
                  == 0,
                  "the host is still folding the SITE's latitude into the "
                  "snowline, which gives a whole planet one value");
    // The cvars this phase added.
    CHECK(CountOccurrences(engine, "PT_CVAR(r_planet_albedo_map,") == 1);
    CHECK(CountOccurrences(engine, "PT_CVAR(r_planet_land_albedo,") == 1);
}

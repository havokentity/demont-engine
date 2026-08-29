// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "SurfaceAlbedo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace pt::planet {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Cubic Hermite smoothstep, the C1 interpolation weight. Shared by the
// raster sampler and the slope ramp so "smoothstep" means one thing here.
double Smoothstep01(double t) noexcept {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The second Legendre polynomial, the shape the energy-balance models fit
// the observed zonal-mean temperature with.
double LegendreP2(double x) noexcept { return 0.5 * (3.0 * x * x - 1.0); }

// The ramp's endpoints, in the shader's slope01 = 1 - cos(tilt) measure and
// in tangent. Evaluated once at static-init rather than per call: the area
// integral below evaluates the ramp tens of times per vertex, and two
// cosines of a compile-time constant inside that loop was measurably most
// of a coarse chunk's bake.
const double kRockRampLo01  = 1.0 - std::cos(kThresholdHillslopeDeg * kPi / 180.0);
const double kRockRampHi01  = 1.0 - std::cos(kRockFullExposureDeg   * kPi / 180.0);
const double kRockRampLoTan = std::tan(kThresholdHillslopeDeg * kPi / 180.0);
const double kRockRampHiTan = std::tan(kRockFullExposureDeg   * kPi / 180.0);

}  // namespace

// --- The snowline ----------------------------------------------------------

double WarmSeasonSeaLevelTempC(double lat_rad) noexcept {
    const double s  = std::sin(lat_rad);
    const double s2 = s * s;
    // North, Cahalan & Coakley (1981) eq. 2: the annual zonal mean.
    const double annual_c = kZonalMeanT0C - kZonalMeanT2K * LegendreP2(s);
    // Half the annual range, growing with latitude. The fit, not a
    // derivation -- see the header.
    const double amplitude_k = kSeasonAmpEqK + kSeasonAmpPolarK * s2;
    return annual_c + amplitude_k;
}

double SnowlineAltitudeM(double lat_rad, double tropical_anchor_m) noexcept {
    // The freezing level of the warm-season air column, by the environmental
    // lapse rate. Anchored so the equator reproduces the measured tropical
    // equilibrium-line altitude the caller passes in -- so the cvar keeps
    // meaning exactly what its docstring says, and the model supplies only
    // the latitude DEPENDENCE.
    const double level_here  = WarmSeasonSeaLevelTempC(lat_rad) / kLapseRateKPerM;
    const double level_equat = WarmSeasonSeaLevelTempC(0.0) / kLapseRateKPerM;
    if (!(level_equat > 0.0)) return std::max(tropical_anchor_m, 0.0);
    const double m = level_here * (tropical_anchor_m / level_equat);
    // A negative freezing level means the warm season never gets above
    // freezing at sea level: snow AT sea level, not snow below it.
    return std::max(m, 0.0);
}

// --- Threshold hillslope ---------------------------------------------------

double SlopeRockFraction(double slope01) noexcept {
    if (!(kRockRampHi01 > kRockRampLo01)) return 0.0;
    return Smoothstep01((slope01 - kRockRampLo01)
                        / (kRockRampHi01 - kRockRampLo01));
}

double RockFractionFromRmsSlope(double rms_slope_tan_per_axis) noexcept {
    const double s = rms_slope_tan_per_axis;
    if (!(s > 0.0)) return 0.0;
    if (!(kRockRampHiTan > kRockRampLoTan)) return 0.0;
    const double inv2s2 = 0.5 / (s * s);

    // The Rayleigh SURVIVAL function, exp(-t^2 / 2 sigma^2), is elementary,
    // and the ramp is exactly 1 above tan(45 deg) -- so the whole upper
    // tail is closed form and only the ramp's own 0.5774..1.0 band has to
    // be integrated. That is a FIXED, FINITE interval: the quadrature cost
    // does not grow with sigma, the tail is exact rather than truncated,
    // and the integrand is smooth across it.
    const double tail = std::exp(-kRockRampHiTan * kRockRampHiTan * inv2s2);

    // Composite Simpson over [tan 30, tan 45], a 0.4226-wide interval. 64
    // intervals put the abscissae 6.6e-3 apart against a smoothstep whose
    // full width is the interval itself, so the quadrature error is far
    // below the 2e-3 the terrain test compares it against.
    constexpr int kSteps = 64;                 // even, for Simpson
    const double dt = (kRockRampHiTan - kRockRampLoTan) / kSteps;
    auto f = [&](double t) -> double {
        // slope01 = 1 - cos(atan t) = 1 - 1/sqrt(1 + t^2), formed without
        // the trig round trip.
        const double slope01 = 1.0 - 1.0 / std::sqrt(1.0 + t * t);
        const double pdf = t * (2.0 * inv2s2) * std::exp(-t * t * inv2s2);
        return SlopeRockFraction(slope01) * pdf;
    };
    double acc = f(kRockRampLoTan) + f(kRockRampHiTan);
    for (int i = 1; i < kSteps; ++i) {
        acc += ((i & 1) ? 4.0 : 2.0)
             * f(kRockRampLoTan + static_cast<double>(i) * dt);
    }
    return std::clamp(tail + acc * dt / 3.0, 0.0, 1.0);
}

// --- The raster ------------------------------------------------------------

void SurfaceAlbedoMap::BuildFromLinear(std::uint32_t width, std::uint32_t height,
                                       const std::vector<float>& rgb,
                                       const std::vector<std::uint8_t>& coverage,
                                       std::uint32_t flags) {
    packed_.clear();
    width_ = height_ = 0;
    flags_ = flags;
    const std::size_t n = static_cast<std::size_t>(width) * height;
    if (n == 0 || rgb.size() < n * 3 || coverage.size() < n) return;
    width_ = width;
    height_ = height;
    packed_.resize(n);
    const double inv_gamma = 1.0 / kAlbedoGamma;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t v = 0;
        for (int c = 0; c < 3; ++c) {
            const double lin = std::clamp(static_cast<double>(rgb[i * 3 + c]),
                                          0.0, 1.0);
            const double enc = std::pow(lin, inv_gamma);
            const auto   b   = static_cast<std::uint32_t>(
                std::lround(std::clamp(enc, 0.0, 1.0) * 255.0));
            v |= (b & 0xFFu) << (8 * c);
        }
        v |= static_cast<std::uint32_t>(coverage[i]) << 24;
        packed_[i] = v;
    }
}

bool SurfaceAlbedoMap::Load(const std::string& path, std::string& out_error) {
    packed_.clear();
    width_ = height_ = flags_ = 0;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        out_error = "cannot open " + path;
        return false;
    }
    AlbedoHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(h))) {
        out_error = path + ": truncated header";
        return false;
    }
    if (std::memcmp(h.magic, kAlbedoMagic, sizeof(kAlbedoMagic)) != 0) {
        out_error = path + ": bad magic (expected PTALB001)";
        return false;
    }
    // A raster wider than this is not a bad idea, it is a corrupt header
    // being trusted into a multi-gigabyte allocation.
    constexpr std::uint32_t kMaxDim = 65536;
    if (h.width == 0 || h.height == 0 || h.width > kMaxDim || h.height > kMaxDim) {
        out_error = path + ": implausible dimensions";
        return false;
    }
    const std::size_t n     = static_cast<std::size_t>(h.width) * h.height;
    const std::size_t bytes = n * 4u;

    // LENGTH VERIFICATION, for the same reason tools/fetch_planet_dem.py has
    // it: the first ETOPO download truncated at 30% and every check short of
    // a length comparison called the result valid. A short raster bakes into
    // a plausible-looking planet with half a continent missing.
    f.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uintmax_t>(f.tellg());
    const std::uintmax_t want = sizeof(AlbedoHeader) + bytes;
    if (file_size != want) {
        out_error = path + ": length " + std::to_string(file_size)
                  + " does not match the header's " + std::to_string(want)
                  + " (" + std::to_string(h.width) + "x"
                  + std::to_string(h.height) + " RGBA8 plus a 40-byte header)";
        return false;
    }
    f.seekg(static_cast<std::streamoff>(sizeof(AlbedoHeader)), std::ios::beg);

    std::vector<std::uint8_t> raw(bytes);
    f.read(reinterpret_cast<char*>(raw.data()),
           static_cast<std::streamsize>(bytes));
    if (!f || f.gcount() != static_cast<std::streamsize>(bytes)) {
        out_error = path + ": truncated samples";
        return false;
    }

    // Little-endian assembly by hand rather than a memcpy of uint32, so the
    // byte order of the file is a property of the format and not of the host.
    packed_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        packed_[i] = static_cast<std::uint32_t>(raw[i * 4 + 0])
                   | (static_cast<std::uint32_t>(raw[i * 4 + 1]) << 8)
                   | (static_cast<std::uint32_t>(raw[i * 4 + 2]) << 16)
                   | (static_cast<std::uint32_t>(raw[i * 4 + 3]) << 24);
    }
    width_  = h.width;
    height_ = h.height;
    flags_  = h.flags;
    return true;
}

glm::vec4 SurfaceAlbedoMap::Fetch(std::int64_t x, std::int64_t y) const noexcept {
    const auto w = static_cast<std::int64_t>(width_);
    const auto hh = static_cast<std::int64_t>(height_);
    if (w <= 0 || hh <= 0) return glm::vec4(0.0f);
    // Longitude WRAPS, latitude CLAMPS -- the DEM's rule, so the two grids
    // behave identically at the antimeridian and at the poles.
    x = ((x % w) + w) % w;
    y = std::clamp<std::int64_t>(y, 0, hh - 1);
    const std::uint32_t v = packed_[static_cast<std::size_t>(y) * width_
                                    + static_cast<std::size_t>(x)];
    const float inv = 1.0f / 255.0f;
    const float r = static_cast<float>((v >> 0) & 0xFFu) * inv;
    const float g = static_cast<float>((v >> 8) & 0xFFu) * inv;
    const float b = static_cast<float>((v >> 16) & 0xFFu) * inv;
    const float a = static_cast<float>((v >> 24) & 0xFFu) * inv;
    // gamma-2.0 decode: one multiply, see kAlbedoGamma.
    return glm::vec4(r * r, g * g, b * b, a);
}

glm::vec4 SurfaceAlbedoMap::SampleAt(double lat_rad,
                                     double lon_rad) const noexcept {
    if (packed_.empty()) return glm::vec4(0.0f);
    // Pixel-centre registration, byte for byte the DEM's expression:
    //   lon = -pi + (x + 0.5) * 2pi / W,  lat = +pi/2 - (y + 0.5) * pi / H
    const double fx = (lon_rad + kPi) / (2.0 * kPi)
                          * static_cast<double>(width_) - 0.5;
    const double fy = (kPi * 0.5 - lat_rad) / kPi
                          * static_cast<double>(height_) - 0.5;
    const double x0d = std::floor(fx);
    const double y0d = std::floor(fy);
    const auto x0 = static_cast<std::int64_t>(x0d);
    const auto y0 = static_cast<std::int64_t>(y0d);
    // C1 weights. See the header: plain bilinear's derivative jump at every
    // texel edge is the diamond lattice this phase must not produce.
    const float tx = static_cast<float>(Smoothstep01(fx - x0d));
    const float ty = static_cast<float>(Smoothstep01(fy - y0d));

    const glm::vec4 c00 = Fetch(x0,     y0);
    const glm::vec4 c10 = Fetch(x0 + 1, y0);
    const glm::vec4 c01 = Fetch(x0,     y0 + 1);
    const glm::vec4 c11 = Fetch(x0 + 1, y0 + 1);
    const glm::vec4 top = c00 + (c10 - c00) * tx;
    const glm::vec4 bot = c01 + (c11 - c01) * tx;
    return top + (bot - top) * ty;
}

}  // namespace pt::planet

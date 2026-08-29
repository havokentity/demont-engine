// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "ElevationField.h"

#include "../../core/Log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace pt::planet {

namespace {

constexpr double kPi = 3.14159265358979323846;

// PCG-style integer avalanche. Three rounds of xor-shift-multiply over the
// packed lattice coordinate. Chosen over a float-based hash (sin/fract) on
// purpose: the classic `fract(sin(dot(p, k)) * 43758.5)` is not portable
// between a CPU bake and any other evaluation, and its output is visibly
// structured at large coordinates. This one is integer-exact everywhere.
inline std::uint32_t HashU32(std::uint32_t x) noexcept {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Signed uniform in [-1, 1] from the vertex's exact integer lattice
// position on the unwarped cube, the subdivision level, and the seed.
inline double HashSigned(std::int64_t ix, std::int64_t iy, std::int64_t iz,
                         int level, std::uint32_t seed) noexcept {
    // Fold the 64-bit lattice coordinates into 32 bits each. Level 19's
    // lattice runs to 64 << 20 = 6.7e7, so the low 32 bits are lossless.
    std::uint32_t h = seed;
    h = HashU32(h ^ static_cast<std::uint32_t>(ix));
    h = HashU32(h ^ static_cast<std::uint32_t>(iy) ^ 0x9e3779b9u);
    h = HashU32(h ^ static_cast<std::uint32_t>(iz) ^ 0x85ebca6bu);
    h = HashU32(h ^ static_cast<std::uint32_t>(level) ^ 0xc2b2ae35u);
    // [0, 1) with 24 bits of mantissa, then to [-1, 1).
    const double u = static_cast<double>(h >> 8) * (1.0 / 16777216.0);
    return u * 2.0 - 1.0;
}

// RMS of U(-1,1) is 1/sqrt(3); scale so the displacement's RMS is exactly
// the amplitude the fractal continuation asks for.
constexpr double kUniformRmsScale = 1.7320508075688772;  // sqrt(3)

}  // namespace

// --- DigitalElevationModel ------------------------------------------------

bool DigitalElevationModel::Load(const std::string& path, std::string& out_error) {
    width_ = height_ = 0;
    samples_.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) { out_error = "cannot open " + path; return false; }

    DemHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f) { out_error = "short read on header"; return false; }
    if (std::memcmp(hdr.magic, kDemMagic, sizeof(kDemMagic)) != 0) {
        out_error = "bad magic (expected PTDEM001)";
        return false;
    }
    if (hdr.width < 4 || hdr.height < 2 ||
        hdr.width > (1u << 20) || hdr.height > (1u << 20)) {
        out_error = "implausible dimensions";
        return false;
    }
    if (!(hdr.scale_m > 0.0)) { out_error = "non-positive height scale"; return false; }

    const std::size_t count = static_cast<std::size_t>(hdr.width) * hdr.height;
    samples_.resize(count);
    f.read(reinterpret_cast<char*>(samples_.data()),
           static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
    if (!f) {
        samples_.clear();
        out_error = "short read on sample data";
        return false;
    }
    width_  = hdr.width;
    height_ = hdr.height;
    scale_  = hdr.scale_m;
    offset_ = hdr.offset_m;
    BuildReliefMap();
    return true;
}

double DigitalElevationModel::Fetch(std::int64_t x, std::int64_t y) const noexcept {
    const std::int64_t w = static_cast<std::int64_t>(width_);
    const std::int64_t h = static_cast<std::int64_t>(height_);
    // Longitude wraps, latitude clamps.
    std::int64_t xi = x % w;
    if (xi < 0) xi += w;
    const std::int64_t yi = std::clamp<std::int64_t>(y, 0, h - 1);
    const std::uint16_t v = samples_[static_cast<std::size_t>(yi * w + xi)];
    return static_cast<double>(v) * scale_ + offset_;
}

double DigitalElevationModel::HeightAt(double lat_rad, double lon_rad) const noexcept {
    if (samples_.empty()) return 0.0;
    // Pixel-centre registration: texel (x, y) centre sits at
    //   lon = -pi + (x + 0.5) * 2pi / W,  lat = +pi/2 - (y + 0.5) * pi / H
    const double fx = (lon_rad + kPi) / (2.0 * kPi) * static_cast<double>(width_) - 0.5;
    const double fy = (kPi * 0.5 - lat_rad) / kPi * static_cast<double>(height_) - 0.5;
    const double x0 = std::floor(fx);
    const double y0 = std::floor(fy);
    const double tx = fx - x0;
    const double ty = fy - y0;
    const auto ix = static_cast<std::int64_t>(x0);
    const auto iy = static_cast<std::int64_t>(y0);
    const double h00 = Fetch(ix,     iy);
    const double h10 = Fetch(ix + 1, iy);
    const double h01 = Fetch(ix,     iy + 1);
    const double h11 = Fetch(ix + 1, iy + 1);
    return (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) +
           (h01 * (1.0 - tx) + h11 * tx) * ty;
}

float DigitalElevationModel::FetchRelief(std::int64_t x, std::int64_t y) const noexcept {
    const std::int64_t w = static_cast<std::int64_t>(width_);
    const std::int64_t h = static_cast<std::int64_t>(height_);
    std::int64_t xi = x % w;
    if (xi < 0) xi += w;
    const std::int64_t yi = std::clamp<std::int64_t>(y, 0, h - 1);
    return relief_[static_cast<std::size_t>(yi * w + xi)];
}

void DigitalElevationModel::BuildReliefMap() {
    // Per-texel central-difference RMS height change, in metres per texel.
    // This IS sigma(L_dem): the measured amplitude of the terrain at the
    // data's own lag, which the fractal continuation extrapolates from.
    relief_.assign(samples_.size(), 0.0f);
    const std::int64_t w = static_cast<std::int64_t>(width_);
    const std::int64_t h = static_cast<std::int64_t>(height_);
    for (std::int64_t y = 0; y < h; ++y) {
        for (std::int64_t x = 0; x < w; ++x) {
            const double gx = 0.5 * (Fetch(x + 1, y) - Fetch(x - 1, y));
            const double gy = 0.5 * (Fetch(x, y + 1) - Fetch(x, y - 1));
            relief_[static_cast<std::size_t>(y * w + x)] =
                static_cast<float>(std::sqrt(0.5 * (gx * gx + gy * gy)));
        }
    }
}

void DigitalElevationModel::SampleAt(double lat_rad, double lon_rad,
                                     double& out_height_m,
                                     double& out_relief_m) const noexcept {
    if (samples_.empty()) { out_height_m = 0.0; out_relief_m = 0.0; return; }
    const double fx = (lon_rad + kPi) / (2.0 * kPi) * static_cast<double>(width_) - 0.5;
    const double fy = (kPi * 0.5 - lat_rad) / kPi * static_cast<double>(height_) - 0.5;
    const double x0 = std::floor(fx);
    const double y0 = std::floor(fy);
    const double tx = fx - x0;
    const double ty = fy - y0;
    const auto ix = static_cast<std::int64_t>(x0);
    const auto iy = static_cast<std::int64_t>(y0);
    const double h00 = Fetch(ix,     iy);
    const double h10 = Fetch(ix + 1, iy);
    const double h01 = Fetch(ix,     iy + 1);
    const double h11 = Fetch(ix + 1, iy + 1);
    out_height_m = (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) +
                   (h01 * (1.0 - tx) + h11 * tx) * ty;
    const double r00 = FetchRelief(ix,     iy);
    const double r10 = FetchRelief(ix + 1, iy);
    const double r01 = FetchRelief(ix,     iy + 1);
    const double r11 = FetchRelief(ix + 1, iy + 1);
    out_relief_m = (r00 * (1.0 - tx) + r10 * tx) * (1.0 - ty) +
                   (r01 * (1.0 - tx) + r11 * tx) * ty;
}

void DirectionToGeodetic(const glm::dvec3& dir_unit,
                         double& out_lat_rad, double& out_lon_rad) noexcept {
    const double p = std::sqrt(dir_unit.x * dir_unit.x + dir_unit.y * dir_unit.y);
    out_lat_rad = std::atan2(kWgs84A * dir_unit.z, kWgs84B * p);
    out_lon_rad = std::atan2(dir_unit.y, dir_unit.x);
}

double DigitalElevationModel::TexelAngularSize() const noexcept {
    if (width_ == 0) return 0.0;
    return 2.0 * kPi / static_cast<double>(width_);
}

// --- The broken power law -------------------------------------------------

double RelativeStructureFunction(double lag_m, double floor_m, double break_m,
                                 double h_coarse, double h_fine) noexcept {
    if (!(lag_m > 0.0) || !(floor_m > 0.0)) return 0.0;
    if (!(break_m > 0.0)) {
        // No break: the single self-affine power law. Divergent below the
        // hillslope scale; kept only so the A/B is one cvar away.
        return std::pow(lag_m / floor_m, h_coarse);
    }
    // pow(1, x) is exactly 1 in IEEE-754, so S(floor_m) == 1.0 exactly and
    // the continuation joins the DEM without a step.
    return std::pow(lag_m / floor_m, h_fine) *
           std::pow((lag_m + break_m) / (floor_m + break_m), h_coarse - h_fine);
}

// --- ElevationField -------------------------------------------------------

void ElevationField::SampleBaseAndRelief(const glm::dvec3& dir_unit,
                                         double& out_height_m,
                                         double& out_relief_m) const noexcept {
    if (dem_ == nullptr || dem_->Empty()) {
        out_height_m = 0.0;
        out_relief_m = params_.procedural_relief_m;
        return;
    }
    double lat = 0.0, lon = 0.0;
    DirectionToGeodetic(dir_unit, lat, lon);
    dem_->SampleAt(lat, lon, out_height_m, out_relief_m);
}

double ElevationField::BaseHeight(const glm::dvec3& dir_unit) const noexcept {
    double h = 0.0, r = 0.0;
    SampleBaseAndRelief(dir_unit, h, r);
    return h;
}

double ElevationField::Relief(const glm::dvec3& dir_unit) const noexcept {
    double h = 0.0, r = 0.0;
    SampleBaseAndRelief(dir_unit, h, r);
    return r;
}

double ElevationField::DataFloorMetres() const noexcept {
    if (dem_ == nullptr || dem_->Empty()) return params_.procedural_floor_m;
    return dem_->TexelAngularSize() * kIuggMeanRadius;
}

int ElevationField::FirstDetailLevel() const noexcept {
    const double floor_m = DataFloorMetres();
    if (!(floor_m > 0.0)) return 0;
    for (int l = 0; l <= kMaxLevel + 1; ++l) {
        if (ChunkVertexSpacing(l) < floor_m) return l;
    }
    return kMaxLevel + 1;
}

void ElevationField::GenerateChunkHeights(const ChunkKey& key, int level, int halo,
                                          std::vector<double>& out) const {
    const int target = std::max<int>(level, key.level);
    const int steps  = target - static_cast<int>(key.level);
    const std::int64_t hl = std::max(0, halo);
    const std::int64_t n = static_cast<std::int64_t>(kChunkQuads) << steps;  // cells
    const std::int64_t verts = n + 1 + 2 * hl;
    out.assign(static_cast<std::size_t>(verts * verts), 0.0);

    // Face-axis cell count at subdivision level l.
    auto grid_cells = [](int l) -> std::int64_t {
        return static_cast<std::int64_t>(kChunkQuads) << l;
    };
    const std::int64_t span = static_cast<std::int64_t>(1) << key.level;

    // Integer window of the chunk's quad on the level-l lattice, inclusive
    // of both endpoints. Widened by the requested halo at the target level
    // and by one cell at every coarser level -- which is exactly what the
    // interpolation stencil needs, because
    //     lo(l) - 1 >= 2*(lo(l-1) - 1)   and   hi(l) + 1 <= 2*(hi(l-1) + 1)
    // for the floor/ceil definitions below, so a parent is always in range.
    // Indices may go NEGATIVE (or past the face) at a cube seam; that is
    // deliberate and handled by TangentWarp's linear continuation.
    auto window = [&](int l, std::int64_t idx, std::int64_t& lo, std::int64_t& hi) {
        const std::int64_t G = grid_cells(l);
        const std::int64_t pad = (l == target) ? std::max<std::int64_t>(hl, 1) : 1;
        lo = (idx * G) / span - pad;
        hi = ((idx + 1) * G + span - 1) / span + pad;   // ceil, then pad
    };

    // Levels below the first detail level carry zero displacement (their
    // amplitude is zero and the recursion starts from zero), so the walk
    // can start there instead of at level 0.
    const int l0 = std::min(FirstDetailLevel() - 1, target);
    const double floor_m = DataFloorMetres();
    const double H = std::clamp(params_.hurst, 0.05, 1.0);
    // H_fine >= H_coarse: the surface may only get smoother as it refines.
    // The upper bound of 2 admits Perron et al.'s measured sub-break
    // exponents (H ~ 1.25-1.6) for anyone who wants the fully diffusive
    // landscape; the default 1.0 is the marginal, slope-invariant case.
    const double Hf = std::clamp(params_.hurst_fine, H, 2.0);
    const double break_m = std::max(params_.hillslope_break_m, 0.0);

    std::vector<double> prev, cur;
    std::int64_t plo_x = 0, phi_x = 0, plo_y = 0, phi_y = 0;
    {
        const int l = std::max(l0, 0);
        window(l, key.i, plo_x, phi_x);
        window(l, key.j, plo_y, phi_y);
        prev.assign(static_cast<std::size_t>((phi_x - plo_x + 1) * (phi_y - plo_y + 1)), 0.0);
    }

    for (int l = std::max(l0, 0) + 1; l <= target; ++l) {
        std::int64_t lo_x, hi_x, lo_y, hi_y;
        window(l, key.i, lo_x, hi_x);
        window(l, key.j, lo_y, hi_y);
        const std::int64_t w = hi_x - lo_x + 1;
        cur.assign(static_cast<std::size_t>(w * (hi_y - lo_y + 1)), 0.0);

        const std::int64_t G  = grid_cells(l);
        const std::int64_t Gh = G / 2;
        const double spacing = ChunkVertexSpacing(l);
        // A pure function of (level, params, DEM): the level-consistency
        // guarantee needs the amplitude to be identical for a parent and a
        // child evaluating the same level, and it is -- nothing here reads
        // the chunk key or the vertex.
        const double amp_scale =
            (spacing < floor_m && floor_m > 0.0)
                ? params_.detail_gain * kUniformRmsScale *
                      RelativeStructureFunction(spacing, floor_m, break_m, H, Hf)
                : 0.0;
        // The displacement at this level acts over one level-l spacing, so
        // capping it at max_slope * spacing caps the slope it can add.
        const double slope_cap = std::max(params_.max_slope, 0.0) * spacing;

        const glm::dvec3 nf = FaceNormal(key.face);
        const glm::dvec3 rf = FaceRight(key.face);
        const glm::dvec3 uf = FaceUp(key.face);

        const std::int64_t pw = phi_x - plo_x + 1;
        auto parent = [&](std::int64_t gx, std::int64_t gy) -> double {
            const std::int64_t px = std::clamp(gx, plo_x, phi_x) - plo_x;
            const std::int64_t py = std::clamp(gy, plo_y, phi_y) - plo_y;
            return prev[static_cast<std::size_t>(py * pw + px)];
        };

        for (std::int64_t gy = lo_y; gy <= hi_y; ++gy) {
            for (std::int64_t gx = lo_x; gx <= hi_x; ++gx) {
                const bool ox = (gx & 1) != 0;
                const bool oy = (gy & 1) != 0;
                double v;
                if (!ox && !oy) {
                    // Inherited vertex: EXACTLY the parent value. This is
                    // the property that makes chunk levels agree.
                    v = parent(gx / 2, gy / 2);
                } else {
                    if (ox && !oy) {
                        v = 0.5 * (parent((gx - 1) / 2, gy / 2) +
                                   parent((gx + 1) / 2, gy / 2));
                    } else if (!ox && oy) {
                        v = 0.5 * (parent(gx / 2, (gy - 1) / 2) +
                                   parent(gx / 2, (gy + 1) / 2));
                    } else {
                        v = 0.25 * (parent((gx - 1) / 2, (gy - 1) / 2) +
                                    parent((gx + 1) / 2, (gy - 1) / 2) +
                                    parent((gx - 1) / 2, (gy + 1) / 2) +
                                    parent((gx + 1) / 2, (gy + 1) / 2));
                    }
                    if (amp_scale > 0.0) {
                        // The exact integer lattice position of this vertex
                        // on the UNWARPED cube, scaled by G/2. Face-basis
                        // vectors are signed coordinate axes, so this is a
                        // permutation of (+-G/2, gx - G/2, gy - G/2) and two
                        // faces meeting at a seam produce the same triple.
                        const glm::dvec3 iv = static_cast<double>(Gh) * nf
                                            + static_cast<double>(gx - Gh) * rf
                                            + static_cast<double>(gy - Gh) * uf;
                        const auto ix = static_cast<std::int64_t>(std::llround(iv.x));
                        const auto iy = static_cast<std::int64_t>(std::llround(iv.y));
                        const auto iz = static_cast<std::int64_t>(std::llround(iv.z));
                        const double s = GridParam(gx,
                                                   G);
                        const double t = GridParam(gy,
                                                   G);
                        const glm::dvec3 dir = FaceParamToDirection(key.face, s, t);
                        double base_unused = 0.0, relief = 0.0;
                        SampleBaseAndRelief(dir, base_unused, relief);
                        const double disp = amp_scale * relief *
                                            HashSigned(ix, iy, iz, l, params_.seed);
                        // Threshold-hillslope cap. See ElevationParams::max_slope.
                        v += std::clamp(disp, -slope_cap, slope_cap);
                    }
                }
                cur[static_cast<std::size_t>((gy - lo_y) * w + (gx - lo_x))] = v;
            }
        }
        prev.swap(cur);
        plo_x = lo_x; phi_x = hi_x; plo_y = lo_y; phi_y = hi_y;
    }

    // Add the DEM base and copy out the (verts x verts) window. The base is
    // a pure function of direction and therefore level-independent, which
    // is what keeps the sum consistent across levels.
    const std::int64_t G = grid_cells(target);
    const std::int64_t base_x = (key.i * G) / span - hl;
    const std::int64_t base_y = (key.j * G) / span - hl;
    const std::int64_t pw = phi_x - plo_x + 1;
    for (std::int64_t y = 0; y < verts; ++y) {
        for (std::int64_t x = 0; x < verts; ++x) {
            const std::int64_t gx = base_x + x;
            const std::int64_t gy = base_y + y;
            const double s = GridParam(gx,
                                       G);
            const double t = GridParam(gy,
                                       G);
            const glm::dvec3 dir = FaceParamToDirection(key.face, s, t);
            const double detail =
                prev[static_cast<std::size_t>((gy - plo_y) * pw + (gx - plo_x))];
            out[static_cast<std::size_t>(y * verts + x)] = BaseHeight(dir) + detail;
        }
    }
}

double ElevationField::HeightAtDirection(const glm::dvec3& dir_unit, int level) const {
    const int lvl = std::clamp(level, 0, kMaxLevel);
    int face = 0;
    double s = 0.0, t = 0.0;
    DirectionToFaceParam(dir_unit, face, s, t);
    const auto span = static_cast<double>(1u << lvl);
    auto cell = [span](double p) -> std::uint32_t {
        const double c = (p + 1.0) * 0.5 * span;
        const auto idx = static_cast<std::int64_t>(std::floor(c));
        return static_cast<std::uint32_t>(
            std::clamp<std::int64_t>(idx, 0, static_cast<std::int64_t>(span) - 1));
    };
    const ChunkKey key{static_cast<std::uint8_t>(face), static_cast<std::uint8_t>(lvl),
                       cell(s), cell(t)};
    std::vector<double> grid;
    GenerateChunkHeights(key, lvl, 0, grid);

    // Bilinear within the chunk's 65x65 grid.
    const std::int64_t G = static_cast<std::int64_t>(key.Span());
    const double s0 = GridParam(key.i,     G);
    const double s1 = GridParam(key.i + 1, G);
    const double t0 = GridParam(key.j,     G);
    const double t1 = GridParam(key.j + 1, G);
    const double fu = std::clamp((s - s0) / (s1 - s0), 0.0, 1.0) * kChunkQuads;
    const double fv = std::clamp((t - t0) / (t1 - t0), 0.0, 1.0) * kChunkQuads;
    const int xi = std::clamp(static_cast<int>(std::floor(fu)), 0, kChunkQuads - 1);
    const int yi = std::clamp(static_cast<int>(std::floor(fv)), 0, kChunkQuads - 1);
    const double ax = fu - xi;
    const double ay = fv - yi;
    auto at = [&](int x, int y) { return grid[static_cast<std::size_t>(y) * kChunkVerts + x]; };
    return (at(xi, yi) * (1.0 - ax) + at(xi + 1, yi) * ax) * (1.0 - ay) +
           (at(xi, yi + 1) * (1.0 - ax) + at(xi + 1, yi + 1) * ax) * ay;
}

}  // namespace pt::planet

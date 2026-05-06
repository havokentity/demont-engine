#include "BscCatalog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace pt::stars {

namespace {
constexpr double kPi    = 3.14159265358979323846;
constexpr double kRad2Deg = 180.0 / kPi;

// All known macOS/x86/arm targets are little-endian; the BSC5 file
// shipped with this engine is the canonical Harvard distribution which
// is also LE. We assume LE without runtime checks; on a hypothetical
// big-endian build the parse would silently produce wrong numbers and
// the loader would reject most stars on the sanity guards below.
template <typename T>
T read_le(const std::uint8_t*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}
}  // namespace

std::vector<Star> LoadBsc5(const std::string& path, std::string* err) {
    std::vector<Star> stars;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "failed to open " + path;
        return stars;
    }
    in.seekg(0, std::ios::end);
    auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 28) {
        if (err) *err = "file shorter than BSC5 header";
        return stars;
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(buf.data()), size);

    const std::uint8_t* p = buf.data();
    const std::int32_t star0  = read_le<std::int32_t>(p);  // unused
    const std::int32_t star1  = read_le<std::int32_t>(p);  // unused
    const std::int32_t starn  = read_le<std::int32_t>(p);  // negative if J2000 floats present
    const std::int32_t stnum  = read_le<std::int32_t>(p);  // unused
    const std::int32_t mprop  = read_le<std::int32_t>(p);  // unused
    const std::int32_t nmag   = read_le<std::int32_t>(p);  // unused
    const std::int32_t nbent  = read_le<std::int32_t>(p);
    (void)star0; (void)star1; (void)stnum; (void)mprop; (void)nmag;

    if (nbent != 32) {
        if (err) *err = "unexpected record size " + std::to_string(nbent);
        return stars;
    }
    const std::size_t count = static_cast<std::size_t>(std::abs(starn));
    if (28 + count * 32 > buf.size()) {
        if (err) *err = "file truncated";
        return stars;
    }

    stars.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* r = buf.data() + 28 + i * 32;
        // float xno (4) | double ra_rad (8) | double dec_rad (8) |
        // char[2] sp | int16 mag*100 | float ra_pm | float dec_pm
        double ra_rad, dec_rad;
        std::int16_t mag_x100;
        std::memcpy(&ra_rad,    r + 4,  sizeof(double));
        std::memcpy(&dec_rad,   r + 12, sizeof(double));
        std::memcpy(&mag_x100,  r + 22, sizeof(std::int16_t));

        // Sanity gates -- the catalog has a handful of placeholder
        // entries with garbage RA/Dec and impossible magnitudes.
        if (!std::isfinite(ra_rad) || !std::isfinite(dec_rad)) continue;
        if (ra_rad < -1.0 || ra_rad > 2.0 * kPi + 1.0) continue;
        if (dec_rad < -kPi || dec_rad > kPi) continue;
        const float vmag = float(mag_x100) * 0.01f;
        if (vmag < -2.0f || vmag > 9.0f) continue;

        // Normalise RA to [0, 360).
        double ra_deg = ra_rad * kRad2Deg;
        ra_deg = std::fmod(ra_deg, 360.0);
        if (ra_deg < 0.0) ra_deg += 360.0;
        const double dec_deg = std::clamp(dec_rad * kRad2Deg, -90.0, 90.0);

        Star s{};
        s.ra_deg  = float(ra_deg);
        s.dec_deg = float(dec_deg);
        s.vmag    = vmag;
        stars.push_back(s);
    }

    std::sort(stars.begin(), stars.end(),
              [](const Star& a, const Star& b) { return a.vmag < b.vmag; });
    return stars;
}

void RasteriseJ2000Map(const std::vector<Star>& stars,
                       std::uint32_t W, std::uint32_t H,
                       std::vector<float>& out) {
    out.assign(std::size_t(W) * H * 4, 0.0f);
    if (stars.empty() || W == 0 || H == 0) return;

    // Magnitude -> linear flux. Vega (mag 0) is 1.0; dimmer stars fall
    // off by 2.512^(-mag). We multiply by an overall gain so naked-eye
    // stars punch through ACES tonemapping at typical exposure.
    constexpr float kVegaFlux  = 1.0f;
    // 4.0 picked so naked-eye limit (vmag=6) lands at ~0.06 per-texel,
    // which after exposure*1.5 and ACES tonemap reads as ~50/255 --
    // visibly speckled against a deep night sky. Bright stars saturate
    // ACES regardless, so this scale doesn't blow them out further.
    constexpr float kFluxScale = 4.0f;
    constexpr float kPogson    = 2.51188643f; // 10^(0.4)

    // Angular Gaussian sigma (radians) per star. Most stars get a
    // sub-arcmin sigma so they read as crisp single-texel points;
    // only the very brightest get any visible halo. Earlier values
    // (~6-20 arcmin for bright stars) made Sirius / Vega look like
    // small moons -- which is way bigger than what you see by eye
    // or in a long-exposure photo. The full moon is 30 arcmin
    // across; the human eye's optical PSF for a star is 1-2 arcmin.
    //
    // 1 arcmin = 2.909e-4 rad. Texel pitch at 4096x2048 is ~5.3 arcmin
    // (1.53e-3 rad), so anything below ~1 texel sigma is sub-pixel
    // and reads as a hot point. Per-texel peak brightness comes from
    // the magnitude-driven flux, which is independent of sigma --
    // shrinking sigma here doesn't dim the dim stars.
    // Sigma must be >= the texel pitch (~5.3 arcmin / 1.54e-3 rad at
    // 4Kx2K) for sub-texel-positioned stars to reliably write their
    // peak flux into at least one texel; below ~half-texel-pitch the
    // Gaussian's mass falls between texels and dim stars lose 90%+
    // of their amplitude. Brighter tiers get a wider halo on top.
    auto angular_radius = [](float vmag) {
        if (vmag < -1.0f) return 2.4e-3f;  // ~8.3 arcmin -- Sirius/Canopus halo
        if (vmag <  1.0f) return 1.8e-3f;  // ~6.2 arcmin -- top ~15 stars, gentle bloom
        if (vmag <  3.0f) return 1.5e-3f;  // ~5.2 arcmin -- ~1 texel
        return                  1.5e-3f;   // ~5.2 arcmin -- ~1 texel for the rest
    };

    auto color_for_index = [](std::uint32_t i) {
        const float palette[3][3] = {
            {0.85f, 0.92f, 1.10f},     // hot blue-white
            {1.00f, 0.98f, 0.95f},     // neutral white (most common)
            {1.05f, 0.85f, 0.70f},     // warm amber (red giants)
        };
        const std::uint32_t h = (i * 2654435761u) >> 30;  // 0..3
        const int idx = (h < 1) ? 0 : (h < 3) ? 1 : 2;
        return std::tuple<float, float, float>{
            palette[idx][0], palette[idx][1], palette[idx][2]};
    };

    constexpr double kPi    = 3.14159265358979323846;
    constexpr double kDeg2R = kPi / 180.0;
    const double dphi   = (2.0 * kPi) / double(W);  // RA per u-step (rad)
    const double dtheta = kPi / double(H);          // colatitude per v-step (rad)

    for (std::size_t i = 0; i < stars.size(); ++i) {
        const Star& s = stars[i];
        const float flux = kVegaFlux * std::pow(kPogson, -s.vmag) * kFluxScale;

        // Star direction in J2000: same convention as the shader looks
        // up later (atan2(j.y, j.x) -> RA, asin(j.z) -> dec).
        const double ra  = s.ra_deg  * kDeg2R;
        const double dec = s.dec_deg * kDeg2R;
        const double cdec = std::cos(dec);
        const double sdec = std::sin(dec);
        const double sx = cdec * std::cos(ra);
        const double sy = cdec * std::sin(ra);
        const double sz = sdec;

        // Equirectangular center texel.
        const float u = s.ra_deg / 360.0f;
        const float v = float(0.5 - s.dec_deg / 180.0);   // dec=+90 -> v=0
        const float fx = u * float(W);
        const float fy = v * float(H);

        const auto [r_tint, g_tint, b_tint] = color_for_index(std::uint32_t(i));
        const float r_ang  = angular_radius(s.vmag);
        const float r_ang2 = r_ang * r_ang;
        const int   K      = 4;     // truncate Gaussian past 4-sigma

        // Splat extent in texel space. dec direction is uniform; ra
        // direction widens by 1/cos(dec) so a star near a pole still
        // covers its full angular footprint. Cap at a half-sphere to
        // avoid degenerate sweeps when the star is at the celestial
        // pole exactly (the entire row is "within range" angularly,
        // and we'd visit W texels per star).
        const float half_v = float(K) * r_ang / float(dtheta);
        const double cos_dec_safe = std::max(std::cos(dec), 0.05);   // ~3 deg from pole
        const float half_u = std::min(
            float(K) * r_ang / (float(dphi) * float(cos_dec_safe)),
            float(W) * 0.5f);

        const int iy0 = std::max(0, int(std::floor(fy - half_v)));
        const int iy1 = std::min(int(H) - 1, int(std::ceil (fy + half_v)));
        const int ix0 = int(std::floor(fx - half_u));
        const int ix1 = int(std::ceil (fx + half_u));

        for (int y = iy0; y <= iy1; ++y) {
            // texel-center direction in J2000 for this row of texels
            const double theta_t = (double(y) + 0.5) * dtheta;     // colatitude
            const double sint    = std::sin(theta_t);
            const double cost    = std::cos(theta_t);              // = sin(dec_t)
            for (int xRaw = ix0; xRaw <= ix1; ++xRaw) {
                int x = xRaw;
                while (x < 0)         x += int(W);
                while (x >= int(W))   x -= int(W);
                // u = ra/(2π); the rasteriser maps RA=0 -> u=0, so the
                // texel-center azimuth is phi_t = u*2π *without* a -pi
                // recentre. (An earlier draft subtracted pi here, which
                // pointed every texel exactly opposite its star and made
                // the whole map zero.)
                const double phi_t  = (double(x) + 0.5) * dphi;
                const double tx = sint * std::cos(phi_t);
                const double ty = sint * std::sin(phi_t);
                const double tz = cost;
                // Angular distance via dot product. cos(angle) = s . t.
                // For small angles, angle^2 ≈ 2 * (1 - cos(angle)).
                double dotv = sx * tx + sy * ty + sz * tz;
                if (dotv > 1.0) dotv = 1.0;
                if (dotv < -1.0) dotv = -1.0;
                const double ang2 = 2.0 * (1.0 - dotv);
                if (ang2 > double(r_ang2) * double(K * K)) continue;
                const float gauss = std::exp(-float(ang2) / r_ang2);
                if (gauss < 1e-4f) continue;
                const float intensity = flux * gauss;
                std::size_t off = (std::size_t(y) * W + std::size_t(x)) * 4;
                out[off + 0] += intensity * r_tint;
                out[off + 1] += intensity * g_tint;
                out[off + 2] += intensity * b_tint;
                out[off + 3]  = 1.0f;
            }
        }
    }
}

}  // namespace pt::stars

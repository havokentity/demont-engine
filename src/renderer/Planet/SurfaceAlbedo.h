// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary land cover (#300): real surface albedo ----------------------
//
// The terrain shaded with ONE albedo (r_planet_ground_albedo) plus a fixed
// snowline. At 6 264 m the Himalayas came out uniform white, and from orbit
// the planet was a well-lit uniform sphere. From orbit Earth is blue ocean,
// green forest, tan desert, grey-brown rock and white ice, and those
// boundaries are most of what the eye uses to recognise a planet.
//
// This file is the DATA half of the fix: an equirectangular reflectance
// raster with the same registration as the DEM, plus the two physical
// models that modulate it (a temperature-derived snowline and a
// threshold-hillslope rock exposure). The shading half is
// terrainSurfaceAlbedo() in shaders/PathTrace.slang, which mirrors the
// sampler below texel for texel.
//
// WHY A RASTER AND NOT A PROCEDURAL CLASSIFICATION
// ------------------------------------------------
// The same argument tools/fetch_planet_dem.py makes for ETOPO over a
// procedural planet. Commit b2111dd replaced fabricated Hosek-Wilkie sky
// coefficients with the published dataset; assets/stars/BSC5.dat is the
// genuine Yale Bright Star Catalogue. A procedural biome map would be that
// decision made a third time. Procedural classification survives here in
// its proper place -- terrainProceduralAlbedo() in the shader, the path for
// bodies that HAVE no data -- and the raster carries Earth.
//
// SOURCE, AND THE ONE THING IT IS NOT
// -----------------------------------
// See assets/planet/PROVENANCE.md for the full record. In short: the
// physically correct product is MODIS MCD43 per-band albedo, which is
// measured, atmospherically corrected bihemispherical reflectance. It is
// public domain, and tools/fetch_planet_albedo.py implements it -- but it
// is served only behind a NASA Earthdata Login, which an unattended bake
// cannot obtain. The SHIPPED asset is therefore derived from NASA Blue
// Marble Next Generation, whose absolute level is display-tuned rather than
// radiometric. kAlbFlagMeasuredAlbedo records which of the two a given file
// is, so the distinction is in the container and not only in a document.
//
// How far off the shipped one is, is measured rather than asserted: the
// bake's linearised luminance against published visible-band hemispherical
// reflectance is tabulated in PROVENANCE.md and pinned by
// tests/pt_planet_albedo_test.cpp.

#include "CubedSphere.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pt::planet {

// --- Container ------------------------------------------------------------
//
// Deliberately the same 40-byte shape as DemHeader: one wire format to
// learn, one baker to read, and the two grids register on the same
// pixel-centre equirectangular convention so a coastline and the colour on
// either side of it cannot disagree.
inline constexpr char kAlbedoMagic[8] = {'P','T','A','L','B','0','0','1'};

// The encoding, and why it is not linear 8-bit.
//
//     reflectance = (value / 255)^2 * scale + offset,  value uint8
//
// A LINEAR uint8 would put its 1/255 = 0.0039 step against a closed-canopy
// forest reflectance of ~0.018, i.e. a 22% quantisation error on the
// darkest land there is. The gamma-2.0 encoding spends its codes where the
// data is: the step at 0.018 is 0.0011 (5.8%) and at snow's 0.95 it is
// 0.0076 (0.8%). The decode is one multiply, which matters because the
// shader does four of them per bilinear tap.
//
// float16 was considered and rejected for the same reason as in the DEM:
// three halves per texel is 6 bytes against 4, for precision the source
// data does not have.
inline constexpr double kAlbedoGamma = 2.0;

// Flags. Bit 0 says whether the file carries MEASURED albedo (MODIS MCD43
// per-band bihemispherical reflectance) or a radiance-derived stand-in
// (Blue Marble). The engine logs which, once, so a render is never quietly
// attributed to a measurement it did not come from.
inline constexpr std::uint32_t kAlbFlagMeasuredAlbedo = 1u << 0;

struct AlbedoHeader {
    char          magic[8];
    std::uint32_t width;
    std::uint32_t height;
    double        scale;
    double        offset;
    std::uint32_t flags;
    std::uint32_t reserved;
};
static_assert(sizeof(AlbedoHeader) == 40,
              "AlbedoHeader must be 40 bytes on disk");

// --- Published hemispherical reflectances ---------------------------------
//
// Used by the procedural no-data path and by the slope/snow modulation that
// sits on top of the raster. These are MEASUREMENTS, not picked colours:
// Budyko 1974 ("Climate and Life") and Ahrens, "Meteorology Today" 11th ed.
// table 2.2 give fresh snow 0.80-0.90, dry sand 0.30-0.40, grassland
// 0.16-0.20, bare rock 0.20-0.30, deep water 0.06-0.10. Earth's mean land
// albedo of 0.15 (Trenberth, Fasullo & Kiehl 2009, BAMS 90:311) falls out
// of the mixture rather than being imposed on it.
//
// The RGB split within each is the visible-band spectral shape of the
// material, not a hue choice: sand and rock are red-weighted by iron oxide
// absorption in the blue, vegetation is green-weighted by chlorophyll's
// twin absorption bands at 430 and 662 nm, and snow is very slightly
// blue-weighted.
inline constexpr glm::vec3 kAlbedoSnow   {0.85f, 0.87f, 0.90f};
inline constexpr glm::vec3 kAlbedoRock   {0.26f, 0.24f, 0.22f};
inline constexpr glm::vec3 kAlbedoSand   {0.40f, 0.36f, 0.28f};
inline constexpr glm::vec3 kAlbedoGrass  {0.13f, 0.17f, 0.09f};
inline constexpr glm::vec3 kAlbedoSeabed {0.055f, 0.060f, 0.070f};

// --- The snowline, derived rather than dialled -----------------------------
//
// r_planet_snowline used to be one altitude scaled by cos(site latitude),
// which is a shape and not a physical model -- and because it was scaled by
// the SITE's latitude it was a single global scalar, so a planet seen from
// orbit had one snowline everywhere.
//
// What actually sets a snowline is where the warm-season air temperature
// reaches freezing. That is three measured quantities:
//
//   1. THE ENVIRONMENTAL LAPSE RATE, 6.5 K/km. ICAO Standard Atmosphere /
//      ISO 2533:1975 troposphere gradient. This is why altitude buys cold.
//
//   2. THE ZONAL-MEAN SURFACE AIR TEMPERATURE. Energy-balance climate
//      models fit the observed annual zonal mean with a second Legendre
//      polynomial in sin(latitude) -- North, Cahalan & Coakley (1981),
//      "Energy balance climate models", Rev. Geophys. 19(1):91-121, eq. 2;
//      the form goes back to Budyko (1969), Tellus 21:611 and Sellers
//      (1969), J. Appl. Meteorol. 8:392:
//
//          T_annual(phi) = T0 - T2 * P2(sin phi),   P2(x) = (3x^2 - 1)/2
//
//      with T0 ~ 15 degC (the global mean) and T2 ~ 28 K (the equator-to-
//      pole contrast). That puts the equator near +29 degC and the poles
//      near -13 degC.
//
//   3. THE SEASONAL AMPLITUDE. Freezing level in the WARM season is what a
//      snowline tracks, and the annual range grows with latitude and
//      continentality: ~2 K half-range in the humid tropics against ~18 K
//      over high-latitude land. Modelled as A(phi) = A_eq + A_pol*sin^2 phi.
//      This is the weakest of the three -- a two-parameter fit to the
//      observed zonal seasonal cycle rather than a derivation -- and it is
//      labelled as such here rather than dressed up.
//
// Composed, T_warm(phi) = T_annual(phi) + A(phi) and the freezing level is
// T_warm / lapse rate. The model is then anchored so that its equatorial
// value equals r_planet_snowline, whose default 4 900 m is the measured
// tropical equilibrium-line altitude (Kaser & Osmaston, "Tropical
// Glaciers", CUP 2002).
//
// VALIDATION -- the reason to trust it past the anchor, and the honest
// size of the improvement. Predicted against published equilibrium-line
// altitudes at three latitudes that were NOT used to fit it, alongside the
// cos(latitude) law it replaces:
//
//   latitude   this model   cos law    observed ELA
//   0 deg        4 900 m    4 900 m     ~4 900 m  tropical Andes / E Africa
//   45 deg       3 003 m    3 465 m     ~2 900 m  European Alps
//   60 deg       2 055 m    2 450 m     ~1 400 m  southern Norway
//   80 deg       1 221 m      851 m     ~1 000 m  Ellesmere Island
//
//   mean |error|:  245 m  against  441 m  -- a factor of 1.80.
//
// It is NOT uniformly better: at 80 deg the cos law happens to land closer.
// The case for this one is that it halves the mean error, that its inputs
// are measured constants rather than a shape chosen for its endpoints, and
// -- the part that actually matters for a planet -- that it is a function
// of the SHADED POINT's latitude instead of the site's, so a globe seen
// from orbit stops having one snowline everywhere.
//
// The residual is structural and worth naming: a zonal mean knows nothing
// about continentality or about elevated heat sources, which is why Norway
// (maritime, so a smaller seasonal amplitude than the zonal fit assumes) is
// the worst row, and why the model puts the Everest site's snowline at
// 4 064 m against a south-face observation nearer 5 500 m -- the Tibetan
// Plateau warms its own troposphere and no zonal fit can see that.
//
// tests/pt_planet_albedo_test.cpp pins every row of the table AND the ratio,
// so the improvement stays measured rather than asserted here.
inline constexpr double kLapseRateKPerM   = 0.0065;   // ICAO / ISO 2533
inline constexpr double kZonalMeanT0C     = 15.0;     // North et al. 1981
inline constexpr double kZonalMeanT2K     = 28.0;     // North et al. 1981
inline constexpr double kSeasonAmpEqK     = 2.0;      // fit, see above
inline constexpr double kSeasonAmpPolarK  = 18.0;     // fit, see above
inline constexpr double kTropicalSnowlineM = 4900.0;  // Kaser & Osmaston 2002

// Warm-season mean sea-level air temperature at a geodetic latitude, degC.
double WarmSeasonSeaLevelTempC(double lat_rad) noexcept;

// Snowline altitude in metres at a geodetic latitude, given the tropical
// anchor (r_planet_snowline). Clamped at zero: a snowline below sea level
// means snow at sea level, not a negative altitude.
double SnowlineAltitudeM(double lat_rad, double tropical_anchor_m) noexcept;

// --- Threshold hillslope ---------------------------------------------------
//
// Slope drives rock exposure because a steep enough face sheds soil, snow
// and vegetation -- something no raster can know, since the raster's texel
// is 20-40 km across and a mountain face is metres.
//
// The two angles are measured. Actively uplifting hillslopes steepen only
// to a threshold set by landsliding, ~30 deg modal slope in the northwest
// Himalaya (Burbank et al. 1996, Nature 379:505); dry granular material
// stands at its angle of repose, ~34 deg (Al-Hashemi & Al-Amoudi 2018,
// Powder Technology 330:397). Bedrock cliffs exceed both, so rock exposure
// ramps from the threshold hillslope to 45 deg rather than switching at one
// angle.
//
// The shader's slope measure is slope01 = 1 - |cos(tilt from local up)|,
// so the ramp bounds are 1 - cos(30 deg) = 0.1340 and 1 - cos(45 deg) =
// 0.2929.
//
// THIS IS THE BUG THAT MADE THE HIMALAYAS WHITE. The pre-#300 ramp was
// saturate((slope01 - 0.5) * 2.5), which is ZERO until slope01 = 0.5 and
// does not saturate until 0.9. Those are tilts of 60 and 84.3 degrees:
//
//   tilt      slope01    old rock fraction    this model
//    20 deg    0.0603          0.000             0.000
//    30 deg    0.1340          0.000             0.000
//    40 deg    0.2340          0.000             0.689
//    45 deg    0.2929          0.000             1.000
//    60 deg    0.5000          0.000             1.000
//    70 deg    0.6580          0.395             1.000
//    84.3 deg  0.9000          1.000             1.000
//
// So the old ramp exposed NO rock below a sixty-degree face, and needed
// eighty-four to expose it fully. Terrain that steep barely exists at a
// 38 m vertex spacing, so in practice rock was never exposed at all and
// every land surface above the snowline was snow and nothing else -- which
// is exactly the uniform white the fixture shows at 6 264 m.
//
// The measured angles put the transition where the geomorphology does:
// nothing below 30 degrees, half by 37.5, bare by 45.
// THE ANGLES CARRY A BASELINE, AND #307 IS ABOUT HONOURING IT. 30 deg is a
// measurement made on a 3-arcsecond DEM with a one-cell-either-side slope
// operator -- Gabet, Pratt-Sitaula & Burbank (2004), Geology 32(7):629, say
// so in as many words, and Burbank et al. (1996) is the same range and the
// same authors. Feeding this ramp a slope measured over some other baseline
// is a units error, and it was one: the shader fed it the MESH normal,
// whose baseline is two chunk cells and therefore changes with the LOD. The
// baseline the ramp is now evaluated at is kRefSlopeHalfLagM /
// kRefSlopeLevel in TerrainChunk.h.
inline constexpr double kThresholdHillslopeDeg = 30.0;
inline constexpr double kRockFullExposureDeg   = 45.0;
double SlopeRockFraction(double slope01) noexcept;

// --- Rock exposure as an AREA FRACTION (#307) ------------------------------
//
// The ramp above answers "is THIS point bare rock". A chunk whose vertices
// are further apart than the reference baseline cannot ask that question --
// its mesh does not carry a 180 m signal, and point-sampling one at 5 km
// intervals is aliasing, not detail. What such a chunk needs is the AREA
// MEAN of the ramp over its footprint, which is the same quantity
// prefiltered, exactly as a mip is.
//
// The mean is available in closed form because the fractal continuation is
// a stationary random field given the local relief: its two slope
// components are the sum of many independent per-level displacements and go
// Gaussian by the central limit theorem, so the slope MAGNITUDE
// t = |grad h| is Rayleigh-distributed,
//
//     p(t) = (t / sigma^2) exp(-t^2 / (2 sigma^2)),
//
// with sigma the per-axis RMS slope tangent. The area fraction is then
// the ramp integrated against p:
//
//     rock = integral_0^inf  SlopeRockFraction(1 - cos(atan t)) p(t) dt
//
// The ramp is exactly 0 below tan(30 deg) and exactly 1 above tan(45 deg),
// so the two tails are elementary -- the upper one is the Rayleigh
// survival function exp(-t^2 / 2 sigma^2) -- and only the 0.5774..1.0 band
// between them needs quadrature. The cost is therefore fixed rather than a
// function of sigma, and the tail is exact rather than truncated.
//
// The Gaussian assumption behind the Rayleigh is not asserted:
// tests/pt_planet_terrain_test.cpp measures the pointwise branch's own area
// mean at the crossover level and compares, and the residual is recorded
// with the constant it calibrates.
double RockFractionFromRmsSlope(double rms_slope_tan_per_axis) noexcept;

// --- The raster ------------------------------------------------------------
//
// An equirectangular RGBA8 grid, row 0 at the north pole, pixel-centre
// registered on exactly the DEM's convention:
//
//     lon = -pi + (x + 1/2) * 2pi / W        lat = +pi/2 - (y + 1/2) * pi / H
//
// RGB is gamma-2.0-encoded linear reflectance; A is COVERAGE -- 255 where
// the source had real land data, 0 over ocean and over no-data. Coverage is
// what lets a body with no raster, or a texel the source never saw, fall
// back to the procedural path per-texel rather than all-or-nothing.
class SurfaceAlbedoMap {
public:
    // Returns false and leaves the map empty if the file is missing or
    // malformed; out_error gets a human-readable reason.
    bool Load(const std::string& path, std::string& out_error);

    // Build directly from linear RGB + coverage, for tests and for a
    // future in-engine baker. `rgb` is width*height*3 linear reflectances.
    void BuildFromLinear(std::uint32_t width, std::uint32_t height,
                         const std::vector<float>& rgb,
                         const std::vector<std::uint8_t>& coverage,
                         std::uint32_t flags);

    bool          Empty()  const noexcept { return packed_.empty(); }
    std::uint32_t Width()  const noexcept { return width_; }
    std::uint32_t Height() const noexcept { return height_; }
    std::uint32_t Flags()  const noexcept { return flags_; }
    bool IsMeasuredAlbedo() const noexcept {
        return (flags_ & kAlbFlagMeasuredAlbedo) != 0u;
    }

    // Linear RGB reflectance in .xyz and coverage in [0,1] in .w, at a
    // geodetic latitude and longitude in radians. Longitude wraps; latitude
    // clamps at the poles -- the same boundary rules DigitalElevationModel
    // uses, because the two grids must agree at the seam and at the caps.
    //
    // SMOOTHSTEP-WEIGHTED bilinear, not plain bilinear. Plain bilinear is
    // only C0: magnified 20x (which is what a 39 km texel is from orbit) its
    // derivative discontinuity at every texel edge reads as a diamond
    // lattice across continents -- the banding this phase must not have.
    // Cubic Hermite weights on the fractional coordinates make the
    // interpolant C1 for two extra multiplies per axis.
    //
    // MIRRORED BY ptLandAlbedoSample() in shaders/PathTrace.slang. The two
    // are one algorithm with two implementations and there is no generated
    // code between them, so tests/pt_planet_albedo_test.cpp pins the
    // shader's text against this one.
    glm::vec4 SampleAt(double lat_rad, double lon_rad) const noexcept;

    // Packed RGBA8, one uint per texel, row-major, ready for the storage
    // buffer at shader binding 47. Byte 0 is R, byte 3 is coverage.
    const std::vector<std::uint32_t>& Packed() const noexcept { return packed_; }

private:
    glm::vec4 Fetch(std::int64_t x, std::int64_t y) const noexcept;

    std::uint32_t width_  = 0;
    std::uint32_t height_ = 0;
    std::uint32_t flags_  = 0;
    std::vector<std::uint32_t> packed_;
};

}  // namespace pt::planet

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// --- Planetary P4 (#258): the elevation field ------------------------------
//
// Real ETOPO 2022 topography AND bathymetry as the base, self-affine
// fractal continuation below the data floor.
//
// WHY NOT PURE PROCEDURAL, AND WHY NOT PURE DATA
// ----------------------------------------------
// Pure procedural gives infinite detail and zero fidelity -- no Himalayas,
// no Mid-Atlantic Ridge, no Mediterranean. From orbit, which is the money
// shot of this whole arc, it looks like noise because it is. This project
// already made that call once: commit b2111dd replaced fabricated
// Hosek-Wilkie coefficients with the real ArHosekSkyModel 1.4a dataset, and
// assets/stars/BSC5.dat is the genuine Yale Bright Star Catalogue. A
// procedural planet would be the fabricated-coefficients decision made
// again.
//
// Pure data stops at the grid's resolution and cannot be walked on. Between
// a 20 km/texel "Earth lite" and level 19's 30 cm there are ~16 octaves the
// data does not have.
//
// SOURCE
// ------
// ETOPO 2022 (NOAA National Centers for Environmental Information,
// doi:10.25921/fd45-gt74). A US Government work: public domain, so no
// attribution obligation attaches to an MIT-licensed engine. Decisively, it
// is a SINGLE grid carrying both topography and bathymetry, which is what a
// shoreline needs -- GEBCO requires attribution and has mixed third-party
// provenance, SRTM/NASADEM is land-only and ~100 GB.
//
// The full grid is not committed. tools/fetch_planet_dem.py downloads it
// from NCEI and bakes assets/planet/earth_lite.ptdem; the engine runs
// without it (see HasData()) and says so once, loudly.
//
// ENCODING (see tools/fetch_planet_dem.py, which must stay in step)
// -----------------------------------------------------------------
//   height_m = value * kDemScaleM + kDemOffsetM,  value uint16
// with the affine chosen from the real bounds rather than round numbers:
//   offset -11 000 m clears Challenger Deep, -10 935 m (Gardner, Armstrong
//     & Calder 2014) by 65 m;
//   scale 0.303 m/count puts the top of the range at
//     -11000 + 65535*0.303 = +8 857.1 m, clearing Everest's 8 848.86 m
//     (2020 China/Nepal joint survey) by 8.2 m;
//   quantisation is therefore 30.3 cm.
// float16 is NOT usable here: its 11-bit mantissa gives an 8 m ULP at
// 8 000 m, which would quantise the Himalayas into terraces.
//
// FRACTAL CONTINUATION -- AND WHY IT IS INTERPOLATORY
// ---------------------------------------------------
// Below the data floor the field continues with Fournier, Fussell &
// Carpenter (1982) midpoint displacement -- "Computer rendering of
// stochastic models", CACM 25(6) -- on the cubed-sphere vertex hierarchy.
// The Hurst exponent is a measured constant, not a dial: continental
// topography has a power spectrum with beta ~ 2, i.e. fractal dimension
// D ~ 2.5 for the surface, and D = 3 - H gives H = 0.5 (Turcotte,
// "Fractals and Chaos in Geology and Geophysics", 2nd ed. 1997, ch. 7).
// The RMS height difference over lag l therefore scales as
//
//     sigma(l) = sigma(L_dem) * (l / L_dem)^H
//
// and sigma(L_dem) is *measured from the DEM itself* at each point (the
// local RMS inter-texel height difference), so the Andes get sharp octaves
// and the abyssal plains stay flat without anyone tuning anything.
//
// The choice of an INTERPOLATORY subdivision -- displacement is added only
// at vertices that are new at that level, and is exactly zero at every
// vertex inherited from the level above -- is the load-bearing one. It
// makes the field satisfy
//
//     h(level L, x) == h(level L-1, x)   for every x on the level-(L-1) grid
//
// EXACTLY, in floating point. Two chunks at different levels that share a
// vertex therefore agree bit-for-bit, and the whole class of LOD cracks --
// including the corner cases a 2:1 edge restriction does not cover --
// stops being representable rather than being defended against. A
// sum-of-octaves fBm cannot give that: band-limiting per chunk level makes
// neighbours at different levels disagree by exactly the octave one of them
// dropped.
//
// The price is a documented artefact: the displacement vanishes on the
// coarsest lattice it is applied at. That lattice is the one whose spacing
// matches the DEM texel (~19.6 km for Earth lite), where the DEM's own real
// data already fixes the height -- so the pinning is to measurement, not to
// nothing -- and 19.6 km is a scale band that is either far too far away to
// resolve the fractal detail or far too close to see the lattice.
//
// SEAM CONSISTENCY
// ----------------
// The per-vertex hash keys off the vertex's exact INTEGER lattice position
// on the unwarped cube, not off (face, i, j) and not off the float
// direction. A vertex on one of the 12 cube seams has the same integer
// triple computed from either adjoining face -- the algebra in
// NeighborChunk shows the cube-space coordinates are exact dyadic
// rationals -- so the two faces produce identical displacement. Hashing
// (face, i, j) would crack every seam; hashing the normalised direction
// would crack them too, because normalize() of a permuted triple can differ
// by one ULP and a hash amplifies one ULP into a full-amplitude difference.

#include "CubedSphere.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pt::planet {

// The uint16 affine. Mirrored in tools/fetch_planet_dem.py.
inline constexpr double kDemScaleM  = 0.303;
inline constexpr double kDemOffsetM = -11000.0;

// On-disk container. 40-byte header, then width*height little-endian
// uint16 rows north-to-south, columns west-to-east, pixel-centre
// registered on a plate-carree (equirectangular) grid.
inline constexpr char kDemMagic[8] = {'P','T','D','E','M','0','0','1'};

struct DemHeader {
    char          magic[8];
    std::uint32_t width;
    std::uint32_t height;
    double        scale_m;
    double        offset_m;
    std::uint32_t flags;      // 0 today
    std::uint32_t reserved;
};
static_assert(sizeof(DemHeader) == 40, "DemHeader must be 40 bytes on disk");

// Digital elevation model: an equirectangular uint16 grid plus the
// sampling that turns it into a continuous function of direction.
class DigitalElevationModel {
public:
    // Returns false (and leaves the model empty) if the file is missing or
    // malformed. `out_error` gets a human-readable reason.
    bool Load(const std::string& path, std::string& out_error);

    bool Empty() const noexcept { return samples_.empty(); }
    std::uint32_t Width()  const noexcept { return width_; }
    std::uint32_t Height() const noexcept { return height_; }

    // Bilinear height in metres at geodetic (lat, lon) in radians.
    // Longitude wraps; latitude clamps at the poles.
    double HeightAt(double lat_rad, double lon_rad) const noexcept;

    // Height AND the local relief in one bilinear setup. `relief` is
    // sigma(L_dem) in the fractal continuation: the RMS inter-texel height
    // difference, precomputed per texel at load by central differences and
    // then bilinearly interpolated here.
    //
    // Interpolating a precomputed map rather than differencing at sample
    // time is not just a speed choice. The continuation's amplitude must be
    // a CONTINUOUS pure function of position: a pure function so two chunk
    // levels agree at a shared vertex (a per-chunk or per-texel constant
    // would crack every LOD boundary), and continuous so the fractal
    // roughness does not step at texel edges.
    void SampleAt(double lat_rad, double lon_rad,
                  double& out_height_m, double& out_relief_m) const noexcept;

    // Angular size of one texel at the equator, radians. The data floor.
    double TexelAngularSize() const noexcept;

private:
    double Fetch(std::int64_t x, std::int64_t y) const noexcept;
    float  FetchRelief(std::int64_t x, std::int64_t y) const noexcept;
    void   BuildReliefMap();

    std::uint32_t width_  = 0;
    std::uint32_t height_ = 0;
    double        scale_  = kDemScaleM;
    double        offset_ = kDemOffsetM;
    std::vector<std::uint16_t> samples_;
    std::vector<float>         relief_;   // metres RMS per texel
};

// Geodetic latitude / longitude of a unit direction in the ellipsoid
// frame, without forming the surface point. Longitude is the direction's
// own azimuth; the geodetic latitude follows from the normal
// n ~ (dx/a, dy/a, dz/b), i.e. tan(lat) = a*dz / (b*hypot(dx, dy)).
void DirectionToGeodetic(const glm::dvec3& dir_unit,
                         double& out_lat_rad, double& out_lon_rad) noexcept;

// Parameters the engine hands the field. All are real quantities; the two
// that are not measured (procedural relief when there is no DEM, and the
// detail gain) are called out in the cvar docstrings as such.
struct ElevationParams {
    // Relief used when no DEM is loaded, metres RMS at the notional data
    // floor. Not a measurement -- see r_planet_procedural_relief.
    double procedural_relief_m = 900.0;
    // Notional data-floor lag when no DEM is loaded, metres.
    double procedural_floor_m  = 20000.0;
    // Hurst exponent. 0.5 is the measured continental value; exposed so a
    // user can render a smoother (H -> 1) or rougher (H -> 0) body.
    double hurst = 0.5;
    // Multiplier on the continuation amplitude. 1.0 = the measured
    // continuation; the cvar exists for A/B, not for tuning the look.
    double detail_gain = 1.0;
    // Threshold-hillslope cap on the slope the continuation may ADD at any
    // one level, as a tangent. A self-affine surface with H < 1 is nowhere
    // differentiable -- sigma(l)/l = sigma(L) * l^(H-1) * L^-H diverges as
    // l -> 0 -- so continuing it unbounded to level 19 produces vertical
    // walls and, at 0.30 m spacing, overhangs a heightfield cannot even
    // represent. Real terrain has a lower cutoff: hillslopes steepen only
    // to a threshold set by landsliding, measured at ~30 deg modal slope in
    // the actively uplifting northwest Himalaya (Burbank et al. 1996,
    // Nature 379:505), and granular material stands at its angle of repose,
    // ~34 deg for dry sand (Al-Hashemi & Al-Amoudi 2018, Powder Technology
    // 330:397). Bedrock cliffs exceed both locally, so the default is
    // tan(45 deg) = 1.0 rather than tan(34 deg): a cap that admits real
    // cliffs while refusing the mathematically unbounded ones.
    //
    // The clamp is a pure function of the vertex (amplitude, relief, hash
    // and spacing all are), so it does not disturb the level-consistency
    // the whole scheme rests on.
    double max_slope = 1.0;
    // Deterministic seed folded into the vertex hash.
    std::uint32_t seed = 0x5eed1234u;
};

// The elevation field: DEM base plus the interpolatory fractal hierarchy.
//
// Thread-safe for concurrent reads once constructed (the DEM is immutable
// and every evaluation is a pure function). Chunk generation calls
// GenerateChunkHeights from several JobSystem workers at once.
class ElevationField {
public:
    void SetDem(const DigitalElevationModel* dem) noexcept { dem_ = dem; }
    void SetParams(const ElevationParams& p) noexcept { params_ = p; }
    const ElevationParams& Params() const noexcept { return params_; }
    bool HasData() const noexcept { return dem_ != nullptr && !dem_->Empty(); }

    // The DEM-only base height at a unit ellipsoid-frame direction, metres
    // of ellipsoidal height. A pure function of the direction, so two
    // chunks that share a vertex agree to within the ~1 ULP the direction
    // itself can differ by across a cube seam -- and because this is a
    // smooth bilinear interpolation rather than a hash, 1 ULP of direction
    // is ~1e-10 m of height.
    double BaseHeight(const glm::dvec3& dir_unit) const noexcept;

    // sigma(L_dem) at a direction: the local RMS relief the continuation
    // extrapolates from.
    double Relief(const glm::dvec3& dir_unit) const noexcept;

    // Both in one DEM bilinear setup. The generation loop calls this per
    // vertex per subdivision level, so halving the trig and the fetches
    // matters: a level-19 chunk evaluates it ~22 000 times.
    void SampleBaseAndRelief(const glm::dvec3& dir_unit,
                             double& out_height_m,
                             double& out_relief_m) const noexcept;

    // The lag (metres) at which the base data stops carrying information.
    double DataFloorMetres() const noexcept;

    // The subdivision level whose vertex spacing first goes below the data
    // floor -- the first level that carries fractal displacement.
    int FirstDetailLevel() const noexcept;

    // Fill `out` (size (n+1)*(n+1), row-major, y outer) with the ellipsoidal
    // height in metres at the level-`level` grid over the quad of `key`,
    // sampled at n+1 points per axis where n = 64 << (level - key.level).
    //
    // `level` must be >= key.level. The engine asks for key.level (the
    // 65x65 chunk grid) and key.level + 1 (the 129x129 fine grid used for
    // e_L and the slope statistics).
    //
    // Deterministic and seed-stable: the result depends only on (key,
    // level, the DEM, params). Baking the same chunk twice, on any thread,
    // in any order, produces byte-identical output -- which the golden
    // matrix requires because chunks are generated asynchronously.
    void GenerateChunkHeights(const ChunkKey& key, int level, int halo,
                              std::vector<double>& out) const;

    // Single-point evaluation at the level-`level` vertex hierarchy, for
    // host queries that are not chunk-shaped (physics ground height,
    // camera altitude). Walks the same recursion, memo-free, so it is
    // O(level) DEM samples rather than O(1) -- fine at a handful of calls
    // per frame, not for bulk work.
    double HeightAtDirection(const glm::dvec3& dir_unit, int level) const;

private:
    const DigitalElevationModel* dem_ = nullptr;
    ElevationParams params_{};
};

}  // namespace pt::planet

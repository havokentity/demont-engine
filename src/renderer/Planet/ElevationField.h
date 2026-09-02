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
// THE RELIEF PLANE -- WHY THE MEAN IS NOT ENOUGH (#318)
// ----------------------------------------------------
// The file carries a SECOND uint16 plane after the elevation grid: the
// per-texel relief, sigma(L_dem), quantised at the same 0.303 m/count as
// the heights (offset 0, since relief >= 0; the largest height difference
// on Earth is 8 848.86 - -10 935 = 19 784 m < 65535*0.303 = 19 857 m, so
// the plane provably never clamps). Its presence is flagged by
// kDemFlagReliefPlane in the header and by the magic PTDEM002.
//
// It exists because the DEM is baked by AREA-AVERAGING the full-resolution
// ETOPO grid down to 2048x1024, and an area average is a low-pass filter:
// it preserves the MEAN of each output texel -- which is why the heights
// validate (land fraction, mean ocean depth, Everest and the Marianas all
// land correctly) -- but it destroys the SECOND MOMENT, the inter-texel
// height variance that sigma(L_dem) measures and that the entire fractal
// continuation below is anchored on. Deriving relief by differencing the
// already-averaged grid (which is what pre-#318 did, in BuildReliefMap)
// cannot recover information the averaging discarded: it reads the
// difference of block MEANS, which for a rough-but-slowly-trending massif
// like the Khumbu is far smaller than the true point-to-point relief. The
// symptom was land relief p99 = 473 m against a rock ramp that needs
// >= 500 m, so bare rock fired on ~1% of land, and a reference-slope
// median of ~12 deg at the Khumbu where Gabet, Pratt-Sitaula & Burbank
// (2004), Geology 32(7):629, publish ~30 deg. The mountains were averaged
// away before the engine ever saw them.
//
// So tools/fetch_planet_dem.py computes sigma(L_dem) on the FULL-resolution
// grid -- the same central-difference RMS relief BuildReliefMap computes,
// but at the output-texel lag evaluated over every full-resolution cell in
// the footprint and RMS-aggregated -- and stores it as this plane. The
// loader reads it verbatim; the averaging never touches it. Old PTDEM001
// files (no plane) still load, falling back to the suppressed
// BuildReliefMap relief with a single loud log line, because a suppressed
// second moment is a wrong planet, not a broken one.
//
// FRACTAL CONTINUATION -- AND WHY IT IS INTERPOLATORY
// ---------------------------------------------------
// Below the data floor the field continues with Fournier, Fussell &
// Carpenter (1982) midpoint displacement -- "Computer rendering of
// stochastic models", CACM 25(6) -- on the cubed-sphere vertex hierarchy.
// The RMS height difference over lag l scales as
//
//     sigma(l) = sigma(L_dem) * S(l)
//
// and sigma(L_dem) is *measured from the full-resolution ETOPO grid* per
// texel and carried in the file (the relief plane above; the local RMS
// inter-texel height difference), so the Andes get sharp octaves and the
// abyssal plains stay flat without anyone tuning anything. S is
// the normalised structure function, S(L_dem) = 1, and the shape of S is
// the subject of the next section.
//
// THE SCALE BREAK -- WHY S IS NOT ONE POWER LAW (#304)
// ---------------------------------------------------
// Continental topography has a power spectrum with beta ~ 2, i.e. fractal
// dimension D ~ 2.5 for the surface, and D = 3 - H gives H = 0.5
// (Turcotte, "Fractals and Chaos in Geology and Geophysics", 2nd ed. 1997,
// ch. 7). That exponent is real, but it was measured in the band where
// fluvial dissection and tectonics set the form -- kilometres to a few
// hundred metres. Taking S(l) = (l / L_dem)^0.5 all the way to level 19's
// 0.30 m extrapolates it ~16 octaves past its own evidence, and because
// H < 1 the SLOPE it implies,
//
//     sigma(l) / l = sigma(L) * l^(H-1) * L^-H,
//
// diverges as l -> 0. That is not a subtle error. Measured over
// earth_lite's own relief, area-weighted, the single power law adds a
// MEDIAN of 34.8 deg RMS slope at level 19's 0.30 m spacing, 74.6 deg at
// the 90th percentile; over Kansas (13.5 m of relief per 19.5 km texel) it
// adds 10.0 deg where real farmland is a degree or so. Terrain that steep
// everywhere does not read as rock; it reads as noise, which is exactly
// what it is.
//
// Landscapes are not self-affine at all scales. Perron, Kirchner &
// Dietrich (2008), "Spectral signatures of characteristic spatial scales
// and nonfractal structure in landscapes", JGR Earth Surface 113:F04003
// (doi:10.1029/2007JF000866), show that topographic power spectra carry a
// distinct roll-off rather than one scaling regime: above the roll-off the
// spectrum is shallow (beta = 2.8 at Gabilan Mesa, 3.1 at the South Fork
// Eel River, on the 2D convention beta = 2H + 2, so H = 0.4 and 0.55 --
// Turcotte's continental value); below it the spectrum steepens sharply
// (beta = 5.2 and 4.5, i.e. H = 1.6 and 1.25) because diffusive hillslope
// sediment transport smooths the surface faster than channel incision
// roughens it. The break is the fluvial-to-hillslope transition.
//
// So S gets two regimes and one crossover:
//
//     S(l) = (l / L)^H_fine * ((l + L_b) / (L + L_b))^(H_coarse - H_fine)
//
// which is exactly 1 at l = L, tends to (l/L)^H_coarse for l >> L_b, tends
// to const * l^H_fine for l << L_b, is smooth everywhere in between, and
// has local log-log slope exactly (H_coarse + H_fine)/2 at l = L_b -- so
// L_b is the break, by definition rather than by fitting.
//
// L_b -- THE BREAK LENGTH, 106 m
// ------------------------------
// Perron et al. (2008) measure the spectral roll-off at 5.6e-3 m^-1
// (wavelength 180 m) at Gabilan Mesa and 4.0e-3 m^-1 (250 m) at the South
// Fork Eel River. A spectral feature at wavelength lambda appears in a
// structure function at lag lambda/2: sigma^2(l) = 2 * integral of
// P(k) (1 - cos(2 pi k l)) dk, whose kernel peaks at k l = 1/2. The two
// sites therefore break at lags of 90 m and 125 m, and their geometric
// mean, 106 m, is the constant used here.
//
// It is deliberately NOT a function of local relief. The hillslope length
// is set by the ratio of diffusive to advective transport efficiency --
// Perron, Dietrich & Kirchner (2008), "Controls on the spacing of
// first-order valleys", JGR 113:F04016 (doi:10.1029/2007JF000977), show
// valley spacing collapses onto a dimensionless Peclet number built from
// D and K, which are climate and lithology properties -- so tying L_b to
// relief would be a fit dressed as physics. The crossover is smooth over
// roughly a decade in l either way, so the field is insensitive to L_b
// anywhere in the 90-125 m the two sites bracket; r_planet_hillslope_break
// exists to A/B that, not to tune a screenshot.
//
// H_fine -- WHY 1.0 AND NOT THE MEASURED 1.6
// ------------------------------------------
// H = 1 is the marginal exponent: sigma(l)/l is then scale-invariant, so
// slope neither diverges nor decays as the hierarchy refines. Perron's
// sub-break spectra actually give H ~ 1.25-1.6, which would make the
// surface locally planar and erase every sub-metre feature. That is right
// for a soil-mantled hillslope at 10 m and wrong at 0.30 m, because a
// third regime -- clast, tussock and rill roughness -- reappears below
// about a metre. Shepard et al. (2001), "The roughness of natural terrain:
// A planetary and remote sensing perspective", JGR Planets 106(E12):32777
// (doi:10.1029/2000JE001429), fit 60 natural surfaces over centimetre-to-
// hundred-metre baselines and find H clustering near 0.5: at sub-metre
// lags real ground is still self-affine and rough, decidedly not the
// locally planar surface H > 1 implies. H_fine = 1.0 is the conservative
// choice between the two regimes -- rougher than pure diffusion, and the
// largest exponent for which the slope does not run away.
//
// The consequence is that ElevationParams::max_slope stops being
// load-bearing. It used to truncate a divergence, and the truncation was
// most of the surface: area-weighted over earth_lite, 31.6% of level-19
// midpoints hit the clamp (4.4% at level 14, 11.5% at level 16 -- growing
// without bound as the LOD refines, because that is what a divergence
// does). Those vertices were not fractal at all, they were pinned at a
// uniform 45 deg, which is precisely what read as noise rather than as
// rock. With the break in place the added slope below L_b no longer
// depends on l, so the engagement rate stops growing with level: 0.19% at
// level 14, 0.22% at 16, 0.23% at 19. What is left is the 1.2% of Earth
// whose DEM relief exceeds ~830 m per texel -- the Karakoram front, the
// Andean scarp, the trench walls -- where a threshold-hillslope cap is
// exactly the right physics rather than a patch over a bad extrapolation.
//
// THE STRUCTURAL LIMIT OF THIS LAW -- WHY CORRECT RELIEF STILL READS ~20 deg
// AT THE KHUMBU AT THE 270 m LAG (#330)
// -------------------------------------------------------------------------
// #318 restored the relief second moment (Khumbu texel relief up to ~500 m),
// yet the reference-scale slope this law derives at the Gabet 180 m baseline
// (kRefSlopeHalfLagM; the operator Gabet, Pratt-Sitaula & Burbank 2004 used,
// ~270 m length scale) still reads a MEDIAN of only ~20 deg at the Everest
// site, against ~30 deg published. #330 is the measurement of why, and the
// finding is that the shortfall is NOT a mis-set parameter -- it is the
// ceiling of a single-anchor, single-exponent, linear-in-relief law.
//
// Measured (production ReferenceSlope01 over a 4x4 level-13 block; the
// analytic sigma(l)/l is smaller because it omits the finite-difference
// accumulation kRefSlopeRmsGain folds back in):
//
//   * The DECOMPOSITION. The fine detail is not the problem: the baked
//     field's RMS slope already exceeds 30 deg below ~40 m (34 deg at 19 m,
//     50 deg at 0.3 m at Everest). The shortfall is confined to the
//     100-500 m band. There the continuation is anchored on relief at the
//     19.5 km floor with H_coarse = 0.5, and the reference operator reads a
//     median of only ~20 deg -- because slope(l) = relief * S(l) / l has
//     grown too slowly from the floor to reach the threshold by the
//     hillslope band. It is candidate (a): the single anchor extrapolated
//     ~7 octaves undershoots the intermediate band for high-relief terrain.
//     The break (candidate b) is secondary: removing it entirely (L_b -> 0)
//     lifts Everest only 20.6 -> 24.4 deg while it detonates the 1 m slope
//     back to the #304 divergence (Kansas 1.7 -> 11.6 deg, abyssal
//     3.3 -> 21.3 deg). Lengthening the break (candidate c) LOWERS the
//     reference slope, so the break is not too short. Even the divergent
//     no-break law never reaches 30 deg at Everest: relief 500 m under
//     H = 0.5 cannot.
//
//   * WHY NO GLOBAL KNOB FIXES IT. Dropping H_coarse to ~0.37 does lift
//     Everest to ~30 deg -- but the same drop pushes the Annapurna site
//     (the planet_hillslope_rock camera) from 44 to ~53 deg, over-rocking a
//     cell that already renders correctly at 43 deg, and lifts the abyssal
//     Pacific 1 m slope past its Shepard et al. (2001) band. The reference
//     slope is EXACTLY linear in relief (asserted in the terrain tests), so
//     one exponent maps relief to slope by one proportionality for every
//     terrain class at once. Real hillslope angle is a SATURATING function
//     of relief -- it rises then pins at the landsliding threshold, ~30-35
//     deg (Burbank et al. 1996; Montgomery & Brandon 2002, EPSL 201:481;
//     DiBiase et al. 2010, EPSL 289:134). A linear law cannot be both steep
//     enough at moderate relief (Everest, 500 m) and bounded at extreme
//     relief (Annapurna, 1450 m); it undershoots the first and overshoots
//     the second, which is exactly what is measured. A saturating transfer
//     would fix the overshoot but, being concave, cannot RAISE the moderate
//     end -- so it does not close the Everest gap either.
//
//   * THE SITE. Gabet, Pratt-Sitaula & Burbank (2004) measured the
//     Marsyandi valley, CENTRAL Nepal -- the Annapurna Himal, which is where
//     planet_hillslope_rock stands (28.389 N 84.111 E) and where this law
//     ALREADY yields a ~44 deg median, at or above their ~30 deg. The
//     Everest/Khumbu site (27.9881 N 86.9250 E, planet_surface) is EASTERN
//     Nepal, glaciated, and 2.2 km above the derived snowline -- a camera
//     planet_hillslope_rock.cfg itself documents as a correct snowfield.
//     So the ~20 deg there is not obviously wrong; the ~30 deg target may be
//     the central-Nepal number read at the wrong texel.
//
// The honest resolution is therefore NOT a re-tuned amplitude or exponent
// (either regresses a class this arc already got right). It is one of:
// (1) a threshold-hillslope SATURATING transfer, relief -> slope, which is a
//     new sub-model needing S_c and a relief scale calibrated to the papers
//     above and its own cross-class validation; (2) a second measured anchor
//     at an intermediate scale, which the 19.5 km DEM cannot supply below
//     its floor; or (3) accepting that the Everest texel is correctly ~20
//     deg and correcting the target-site attribution. #330 scopes all three
//     and changes no coefficient, so no golden moves.
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

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace pt::planet {

// The uint16 affine. Mirrored in tools/fetch_planet_dem.py.
inline constexpr double kDemScaleM  = 0.303;
inline constexpr double kDemOffsetM = -11000.0;

// The relief plane (#318) reuses the height quantisation step but with a
// zero offset, since relief is a non-negative height difference. The top
// of the range, 65535*0.303 = 19 857 m, clears the largest possible height
// difference on Earth (Everest 8 848.86 m minus Challenger Deep -10 935 m
// = 19 784 m), so the plane provably never clamps -- which matters, because
// a silent clamp would reintroduce exactly the second-moment suppression
// this plane exists to fix.
inline constexpr double kDemReliefScaleM  = kDemScaleM;
inline constexpr double kDemReliefOffsetM = 0.0;

// On-disk container. 40-byte header, then width*height little-endian
// uint16 rows north-to-south, columns west-to-east, pixel-centre
// registered on a plate-carree (equirectangular) grid. PTDEM002 appends a
// second width*height uint16 plane, the relief (see kDemFlagReliefPlane).
inline constexpr char kDemMagic[8]   = {'P','T','D','E','M','0','0','2'};
// PTDEM001 predates the relief plane (#318). Still loaded, with relief
// derived from the decimated grid as a fallback; see DigitalElevationModel
// ::Load and the "RELIEF PLANE" section of this file's header comment.
inline constexpr char kDemMagicV1[8] = {'P','T','D','E','M','0','0','1'};

// Header `flags` bits.
//   bit 0: a relief plane follows the elevation plane. Set by PTDEM002
//          bakes; absent from PTDEM001 (whose flags word is always 0).
inline constexpr std::uint32_t kDemFlagReliefPlane = 1u << 0;

struct DemHeader {
    char          magic[8];
    std::uint32_t width;
    std::uint32_t height;
    double        scale_m;
    double        offset_m;
    std::uint32_t flags;      // kDemFlagReliefPlane et al.
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

    // True when the relief plane came from the file (PTDEM002), false when
    // it was derived from the decimated grid as a PTDEM001 fallback. The
    // caller logs the difference, because a fallback is a suppressed-relief
    // planet, not a broken one (see the "RELIEF PLANE" header section).
    bool ReliefFromFile() const noexcept { return relief_from_file_; }

    // Bilinear height in metres at geodetic (lat, lon) in radians.
    // Longitude wraps; latitude clamps at the poles.
    double HeightAt(double lat_rad, double lon_rad) const noexcept;

    // Height AND the local relief in one bilinear setup. `relief` is
    // sigma(L_dem) in the fractal continuation: the RMS inter-texel height
    // difference. For a PTDEM002 file it is read straight from the relief
    // plane, measured on the full-resolution ETOPO grid before decimation
    // (#318); for a PTDEM001 file it is derived from the decimated grid by
    // central differences (BuildReliefMap) -- suppressed, but consistent.
    // Either way it is bilinearly interpolated here.
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
    // PTDEM001 fallback: derive relief by central-differencing the decimated
    // grid. This is the pre-#318 path and is known to suppress the relief
    // second moment (it reads the difference of block means); kept only so a
    // legacy file still renders.
    void   BuildReliefMap();

    std::uint32_t width_  = 0;
    std::uint32_t height_ = 0;
    double        scale_  = kDemScaleM;
    double        offset_ = kDemOffsetM;
    bool          relief_from_file_ = false;
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
    // Hurst exponent ABOVE the hillslope break. 0.5 is the measured
    // continental value; exposed so a user can render a smoother (H -> 1)
    // or rougher (H -> 0) body.
    double hurst = 0.5;
    // Hurst exponent BELOW the hillslope break, where diffusive sediment
    // transport dominates. 1.0 makes RMS slope scale-invariant there; see
    // the "H_fine" section of this file's header comment. Clamped to be at
    // least `hurst`, so the surface can only get smoother as it refines,
    // never rougher.
    double hurst_fine = 1.0;
    // The fluvial-to-hillslope break lag, metres. 106 m is the geometric
    // mean of the half-wavelengths of the two spectral roll-offs measured
    // by Perron, Kirchner & Dietrich (2008); see the "L_b" section of the
    // header comment. Zero disables the break entirely and restores the
    // single unbroken power law -- which is the pre-#304 behaviour and
    // diverges, so it exists for A/B only.
    double hillslope_break_m = 106.0;
    // Multiplier on the continuation amplitude. 1.0 = the measured
    // continuation; the cvar exists for A/B, not for tuning the look.
    double detail_gain = 1.0;
    // Threshold-hillslope cap on the slope the continuation may ADD at any
    // one level, as a tangent. Real terrain has an upper cutoff on slope:
    // hillslopes steepen only to a threshold set by landsliding, measured
    // at ~30 deg modal slope in the actively uplifting northwest Himalaya
    // (Burbank et al. 1996, Nature 379:505), and granular material stands
    // at its angle of repose, ~34 deg for dry sand (Al-Hashemi &
    // Al-Amoudi 2018, Powder Technology 330:397). Bedrock cliffs exceed
    // both locally, so the default is tan(45 deg) = 1.0 rather than
    // tan(34 deg): a cap that admits real cliffs.
    //
    // This is a BACKSTOP, not the fix for a divergence. It used to be the
    // latter, and it did not work: an unbroken H = 0.5 continuation drove
    // 31.6% of Earth's level-19 midpoints into the clamp, which does not
    // remove a divergence -- it truncates one and leaves a constant 45 deg
    // behind. `hillslope_break_m` is what bounds the slope now; with it in
    // place the clamp engages on 0.23% of the surface and, decisively,
    // that number stops growing as the LOD refines.
    //
    // The clamp is a pure function of the vertex (amplitude, relief, hash
    // and spacing all are), so it does not disturb the level-consistency
    // the whole scheme rests on.
    double max_slope = 1.0;
    // Deterministic seed folded into the vertex hash.
    std::uint32_t seed = 0x5eed1234u;
};

// The normalised structure function S(l): the RMS height difference the
// continuation puts at lag `lag_m`, as a multiple of the relief measured at
// the data floor `floor_m`. S(floor_m) == 1.0 exactly.
//
//     S(l) = (l/L)^H_fine * ((l + L_b) / (L + L_b))^(H_coarse - H_fine)
//
// A broken power law with a smooth crossover at L_b = `break_m`: exponent
// H_coarse (fluvial) well above the break, H_fine (diffusive hillslope)
// well below it, and exactly their mean at l = L_b. `break_m` <= 0 gives
// the unbroken single power law. See this file's header comment for the
// derivation and the citations.
//
// Exposed rather than kept private because it is the whole physics of the
// continuation, and the terrain tests measure it directly.
double RelativeStructureFunction(double lag_m, double floor_m, double break_m,
                                 double h_coarse, double h_fine) noexcept;

// The elevation field: DEM base plus the interpolatory fractal hierarchy.
//
// Thread-safe for concurrent reads once constructed (the DEM is immutable
// and every evaluation is a pure function). Chunk generation calls
// GenerateChunkHeights from several JobSystem workers at once.
class ElevationField {
public:
    ElevationField() = default;

    // COPIES AND MOVES GET A FRESH STAMP, because the alternative makes
    // Generation()'s guarantee false in a way a memo cannot see. The
    // implicit copy constructor would duplicate generation_, so two live
    // objects would answer the same number -- and TerrainChunk.cpp's
    // reference-grid memo keys on that number ALONE, having dropped the
    // object's address once the generation stamp was shown to subsume it.
    //
    // Copies are not hypothetical: MakeProceduralField in
    // tests/pt_planet_terrain_test.cpp returns one by value, and every
    // planet test starts with one.
    //
    // A shared stamp would in fact still serve the right grid today, since
    // a copy has the same dem_ and params_ and therefore generates the same
    // heights. That argument is exactly the kind that stops being true
    // quietly -- add one mutable member and it fails with no compiler
    // anywhere to notice -- so the invariant is made real rather than
    // reasoned about.
    ElevationField(const ElevationField& o) noexcept
        : dem_(o.dem_), params_(o.params_) {}
    ElevationField(ElevationField&& o) noexcept
        : dem_(o.dem_), params_(o.params_) {}
    ElevationField& operator=(const ElevationField& o) noexcept {
        if (this != &o) {
            dem_ = o.dem_;
            params_ = o.params_;
            generation_ = NextGeneration();
        }
        return *this;
    }
    ElevationField& operator=(ElevationField&& o) noexcept {
        return *this = static_cast<const ElevationField&>(o);
    }

    void SetDem(const DigitalElevationModel* dem) noexcept {
        dem_ = dem;
        generation_ = NextGeneration();
    }
    void SetParams(const ElevationParams& p) noexcept {
        params_ = p;
        generation_ = NextGeneration();
    }
    const ElevationParams& Params() const noexcept { return params_; }
    bool HasData() const noexcept { return dem_ != nullptr && !dem_->Empty(); }

    // Moved by every mutator, to a value NO OTHER FIELD IN THE PROCESS HAS
    // EVER HELD. A caller that MEMOISES a result of this field --
    // TerrainChunk.cpp caches the shared level-11 reference grid, which
    // sixteen level-13 chunks otherwise regenerate identically -- holds
    // this alongside the memo and discards it when the number moves.
    //
    // PROCESS-UNIQUE, AND UNIQUE FROM CONSTRUCTION -- INCLUDING COPY
    // CONSTRUCTION, which is why the copy and move members above exist at
    // all. Every ElevationField that has ever existed in this process has
    // held a different value, so this ALONE identifies both which field a
    // memo came from and which configuration it was in. A per-object counter would not: a destroyed
    // field and a freshly constructed one at the same address with the
    // same number of mutations would look identical to a cache keyed on
    // (pointer, generation), and they are not identical if they were
    // handed different DEMs. Drawing from one monotonic source makes that
    // collision impossible rather than improbable -- and lets the memo drop
    // the address from its key entirely, which is better than carrying a
    // term that can never differ.
    //
    // It is still only a NECESSARY condition for a memo to be valid, not a
    // sufficient one, because DigitalElevationModel::Load rewrites a DEM IN
    // PLACE: the pointer can be unchanged while the data is not.
    // PlanetTerrain::Configure calls SetDem after every successful Load,
    // which moves this; a FAILED Load leaves the model empty, which
    // HasData() reports. A memo key should therefore carry HasData() and
    // the params as well, and TerrainChunk.cpp's does.
    std::uint64_t Generation() const noexcept { return generation_; }

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
    // Monotonic across the process. Relaxed ordering is enough: the value
    // only has to be UNIQUE, and every consumer reads it through the same
    // happens-before the field's own configuration already establishes
    // (SetDem/SetParams run before the workers that bake against them).
    static std::uint64_t NextGeneration() noexcept {
        static std::atomic<std::uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    const DigitalElevationModel* dem_ = nullptr;
    ElevationParams params_{};
    // Stamped AT CONSTRUCTION, not left at zero, so that the stamp is a
    // complete identity for the field as well as for its configuration
    // state. Two never-configured fields would otherwise share a stamp --
    // harmless in itself, since they behave identically, but it would mean
    // a memo keyed on this had to carry the object's address too, and an
    // address is exactly the thing that gets recycled.
    std::uint64_t   generation_ = NextGeneration();
};

}  // namespace pt::planet

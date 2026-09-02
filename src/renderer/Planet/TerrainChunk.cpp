// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "TerrainChunk.h"

#include "SurfaceAlbedo.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace pt::planet {

namespace {

// The fine grid is one level below the chunk, with a two-vertex halo so
// every chunk vertex has both of its coarse neighbours in range and the
// vertex normal can be a true central difference right up to the chunk's
// own boundary. Without the halo the boundary would fall back to a
// one-sided difference and two chunks sharing an edge would disagree about
// the normal there -- a shading seam along every chunk boundary on the
// planet.
constexpr int kFineHalo  = 2;                                  // fine vertices
constexpr int kFineCells = kChunkQuads * 2;                    // 128
constexpr int kFineDim   = kFineCells + 1 + 2 * kFineHalo;     // 133

inline int FineIdx(int fx, int fy) { return fy * kFineDim + fx; }

// Chunk vertex (x, y) lives at fine grid (2x + halo, 2y + halo).
inline int FineOfChunk(int x) { return 2 * x + kFineHalo; }

// The reference-lattice grid: a level-kRefSlopeLevel chunk with the same
// two-vertex halo. TWO IS THE MAXIMUM the interpolatory recursion can
// deliver exactly: GenerateChunkHeights pads coarser levels by one cell,
// and the parent-in-range condition lo(l)/2 >= lo(l-1) reduces to
// halo <= 2. It is also exactly what is needed here -- the difference
// samples sit kRefSlopeHalfLagM / ChunkVertexSpacing(kRefSlopeLevel) =
// 1.179 cells outside the chunk at worst.
constexpr int kRefHalo = 2;
constexpr int kRefDim  = kChunkQuads + 1 + 2 * kRefHalo;   // 69

// The one calibration in the coarse (area-mean) branch, and it is a
// MEASURED ratio rather than a derived one.
//
// The continuation is built so that the RMS displacement added at level l
// is exactly relief * S(spacing(l)) -- amp_scale folds in the sqrt(3) that
// turns U(-1,1) into unit RMS. What that does NOT give in closed form is
// the RMS of a finite DIFFERENCE over a baseline L, because every level
// contributes: levels far coarser than L through their gradient, levels
// near L at nearly full amplitude, and levels far finer than L at their
// own (small) amplitude. The sum has no tidy form.
//
// So it is measured. kRefSlopeRmsGain is
//
//     RMS(per-axis reference slope from the POINTWISE branch)
//     -------------------------------------------------------
//              relief * S(2 * kRefSlopeHalfLagM) / (2 * kRefSlopeHalfLagM)
//
// evaluated over the pointwise branch itself, so the coarse branch is
// calibrated against the fine one rather than against a hope.
// tests/pt_planet_terrain_test.cpp re-measures it and fails if it drifts,
// which is what stops this from being a number nobody can re-derive.
constexpr double kRefSlopeRmsGain = 1.4689;

}  // namespace

PlanetSite PlanetSite::FromGeodetic(double lat_rad, double lon_rad) noexcept {
    PlanetSite s;
    s.lat_rad = lat_rad;
    s.lon_rad = lon_rad;
    const glm::dvec3 p = GeodeticToEcef(lat_rad, lon_rad);
    const double r = glm::length(p);
    s.site_radius_m = (r > 0.0) ? r : kIuggMeanRadius;
    const glm::dvec3 up = (r > 0.0) ? p / r : glm::dvec3(0.0, 0.0, 1.0);
    // East is horizontal and perpendicular to the meridian plane; it is the
    // same vector for geocentric and geodetic frames.
    glm::dvec3 east(-std::sin(lon_rad), std::cos(lon_rad), 0.0);
    // At a pole the meridian plane degenerates; fall back to the prime
    // meridian's east so the frame stays well defined.
    if (!(glm::length(east) > 1e-12)) east = glm::dvec3(0.0, 1.0, 0.0);
    east = glm::normalize(east - up * glm::dot(east, up));
    const glm::dvec3 south = glm::cross(east, up);
    // glm::dmat3(a, b, c) builds COLUMNS; transposing makes them rows, so
    // the matrix maps an ECEF vector to (dot(east, v), dot(up, v),
    // dot(south, v)) -- world (X = East, Y = Up, Z = South).
    s.ecef_to_world = glm::transpose(glm::dmat3(east, up, south));
    return s;
}

void ReferenceSlope01(const ChunkKey& key,
                      const ElevationField& field,
                      const PlanetSite& site,
                      std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kChunkVertexCount), 0.0);
    const int drop = static_cast<int>(key.level) - kRefSlopeLevel;
    if (drop < 0) return;

    // The reference lattice is the chunk's ancestor at kRefSlopeLevel, so
    // every descendant of that ancestor reads the SAME grid -- which is
    // where the bit-exact level consistency comes from.
    const ChunkKey anc{key.face, static_cast<std::uint8_t>(kRefSlopeLevel),
                       key.i >> drop, key.j >> drop};

    // ...and "the same grid" is a statement about COST as well as about
    // correctness. Sixteen level-13 chunks share one level-11 ancestor;
    // 256 level-15 chunks do. Regenerating that 69x69 grid per chunk was
    // 0.27 ms of the 0.43 ms this function cost -- the single largest term
    // in #307's bake regression, and every one of those regenerations
    // produced byte-identical output by construction.
    //
    // So it is memoised. This is not an approximation and cannot become
    // one: a hit returns the bytes a miss would have computed, which is
    // what makes the LOD-independence equality survive it unchanged.
    //
    // THREAD-LOCAL rather than shared-with-a-mutex. Bakes run on several
    // JobSystem workers at once; a global cache would need a lock on the
    // hot path and would serialise them. Per-worker costs 4 x 37.2 KB --
    // 149 KB a thread, 0.7 MB at the default four workers plus the main
    // one, against the 83 MB vertex arena -- and it still hits, because
    // the streamer hands each worker spatially adjacent chunks. It also
    // keeps BuildTerrainChunk a pure function of its arguments from every
    // caller's point of view -- there is no cross-thread state to reason
    // about.
    //
    // THE KEY IS THE FIELD'S IDENTITY, NOT JUST THE CHUNK'S. Three terms,
    // and every one of them is REACHABLE -- a key term that cannot ever
    // differ is not caution, it is decoration, and the test below is what
    // established which is which:
    //
    //   generation -- ElevationField stamps every object at construction
    //                and again on every SetDem/SetParams from ONE
    //                process-wide counter, so this single term separates a
    //                reconfigured field from its earlier self, one live
    //                field from another, and a recycled address from
    //                whatever used to live there. The object's ADDRESS was
    //                in this key until the red-then-green pass showed no
    //                way to make it matter; it is gone rather than kept as
    //                reassurance.
    //   anc face/i/j -- which patch of ground the grid covers. NOT
    //                anc.level: that is kRefSlopeLevel for every entry by
    //                construction, so comparing it is a term that cannot
    //                differ, and the red-then-green pass confirmed no
    //                mutation of the test can make it differ. Left out
    //                rather than carried.
    //   has_data  -- the one mutation the stamp does NOT see.
    //                DigitalElevationModel::Load empties the model at its
    //                first statement, so a FAILED reload leaves the field
    //                pointing at an emptied DEM with no SetDem to stamp
    //                it. (The test below cannot construct that case: it
    //                needs a real .ptdem on disk, which the unit suite
    //                deliberately does not depend on. Recorded here so the
    //                term is not mistaken for covered.)
    //
    // A cache that answers from the wrong field is exactly the class of
    // defect this project keeps finding, which is why the test below
    // reproduces each reachable way of getting it wrong before asserting
    // that this one does not.
    struct RefGridEntry {
        std::uint64_t         generation = 0;
        bool                  has_data = false;
        ChunkKey              anc{};
        bool                  valid = false;
        std::vector<double>   grid;
    };
    constexpr int kRefGridCacheSize = 4;
    thread_local std::array<RefGridEntry, kRefGridCacheSize> cache{};
    thread_local int cache_next = 0;

    const RefGridEntry* hit = nullptr;
    for (const RefGridEntry& e : cache) {
        if (e.valid && e.generation == field.Generation()
            && e.has_data == field.HasData()
            && e.anc.face == anc.face
            && e.anc.i == anc.i && e.anc.j == anc.j) {
            hit = &e;
            break;
        }
    }
    if (hit == nullptr) {
        RefGridEntry& e = cache[static_cast<std::size_t>(cache_next)];
        cache_next = (cache_next + 1) % kRefGridCacheSize;
        field.GenerateChunkHeights(anc, kRefSlopeLevel, kRefHalo, e.grid);
        e.generation = field.Generation();
        e.has_data   = field.HasData();
        e.anc        = anc;
        e.valid      = true;
        hit = &e;
    }
    const std::vector<double>& rg = hit->grid;

    const std::int64_t Gr =
        static_cast<std::int64_t>(kChunkQuads) << kRefSlopeLevel;
    const double rbase_x = static_cast<double>(
        static_cast<std::int64_t>(anc.i) * kChunkQuads - kRefHalo);
    const double rbase_y = static_cast<double>(
        static_cast<std::int64_t>(anc.j) * kChunkQuads - kRefHalo);

    // Bilinear height at a fractional coordinate in the reference grid.
    auto ref_h = [&](double fx, double fy) -> double {
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, kRefDim - 2);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, kRefDim - 2);
        const double ax = fx - static_cast<double>(x0);
        const double ay = fy - static_cast<double>(y0);
        auto at = [&](int x, int y) {
            return rg[static_cast<std::size_t>(y) * kRefDim + x];
        };
        return (at(x0, y0)     * (1.0 - ax) + at(x0 + 1, y0)     * ax) * (1.0 - ay)
             + (at(x0, y0 + 1) * (1.0 - ax) + at(x0 + 1, y0 + 1) * ax) * ay;
    };
    // The published operator: one 3-arcsecond cell either side.
    const double d_cells = kRefSlopeHalfLagM / ChunkVertexSpacing(kRefSlopeLevel);
    // Reference-lattice coordinates per level-key.level cell. A negative
    // power of two, so every product below is EXACT and two chunk levels
    // over the same ground land on the same fractional coordinate.
    const double per_cell = 1.0 / static_cast<double>(std::int64_t{1} << drop);

    // --- The five samples per vertex share their axes -----------------------
    //
    // Each vertex needs ref_world at (fx +- d, fy), (fx, fy +- d) and
    // (fx, fy) -- five points, 21 125 per chunk. The naive form evaluated
    // FaceParamToDirection at every one of them, i.e. 42 250 calls to
    // std::tan per chunk, which was over half of this function's cost.
    //
    // But the sample set is a PRODUCT of two small axis sets: the x
    // coordinate only ever takes one of {fx-d, fx, fx+d} for the 65 chunk
    // columns (195 distinct values) and likewise for y. TangentWarp and the
    // face-basis scaling depend on one axis each, so both are computed 195
    // times instead of 21 125 -- a 108x reduction in transcendental calls.
    //
    // BIT-EXACT, and it has to be: the whole point of #307 is that two
    // chunk levels over the same ground produce IDENTICAL rock exposure,
    // and that property is asserted as `a == b`. Every `s` below is formed
    // by the same expression the per-sample version used, on the same
    // inputs, so each cached TangentWarp is the same double the inline call
    // returned. The test that pins the equality is the check on this.
    const glm::dvec3 face_n = FaceNormal(key.face);
    const glm::dvec3 face_r = FaceRight(key.face);
    const glm::dvec3 face_u = FaceUp(key.face);

    // Axis tables, three entries per chunk vertex index: [0] = -d_cells,
    // [1] = the vertex itself, [2] = +d_cells.
    constexpr int kOff = 3;
    std::vector<double>     ax_f(static_cast<std::size_t>(kChunkVerts) * kOff);
    std::vector<double>     ay_f(static_cast<std::size_t>(kChunkVerts) * kOff);
    std::vector<glm::dvec3> ax_pr(static_cast<std::size_t>(kChunkVerts) * kOff);
    std::vector<glm::dvec3> ay_qu(static_cast<std::size_t>(kChunkVerts) * kOff);
    for (int x = 0; x <= kChunkQuads; ++x) {
        const double gx = static_cast<double>(
            static_cast<std::int64_t>(key.i) * kChunkQuads + x) * per_cell;
        const double fxc = gx - rbase_x;
        const double fv[kOff] = {fxc - d_cells, fxc, fxc + d_cells};
        for (int k = 0; k < kOff; ++k) {
            const std::size_t o = static_cast<std::size_t>(x) * kOff + k;
            ax_f[o] = fv[k];
            const double s =
                -1.0 + 2.0 * ((rbase_x + fv[k]) / static_cast<double>(Gr));
            ax_pr[o] = TangentWarp(s) * face_r;
        }
    }
    for (int y = 0; y <= kChunkQuads; ++y) {
        const double gy = static_cast<double>(
            static_cast<std::int64_t>(key.j) * kChunkQuads + y) * per_cell;
        const double fyc = gy - rbase_y;
        const double fv[kOff] = {fyc - d_cells, fyc, fyc + d_cells};
        for (int k = 0; k < kOff; ++k) {
            const std::size_t o = static_cast<std::size_t>(y) * kOff + k;
            ay_f[o] = fv[k];
            const double t =
                -1.0 + 2.0 * ((rbase_y + fv[k]) / static_cast<double>(Gr));
            ay_qu[o] = TangentWarp(t) * face_u;
        }
    }

    // World position from a pair of cached axis contributions. The same
    // construction as fine_world in BuildTerrainChunk, one lattice coarser,
    // so the reference slope is a TRUE 3D slope and not a height difference
    // over a nominal spacing -- the cubed sphere's metric is not uniform and
    // the two differ by up to 8% here.
    auto ref_world = [&](std::size_t ox, std::size_t oy) -> glm::dvec3 {
        const glm::dvec3 dir  = glm::normalize(face_n + ax_pr[ox] + ay_qu[oy]);
        const glm::dvec3 surf = EllipsoidSurface(dir);
        const glm::dvec3 nn   = GeodeticNormal(surf);
        return site.EcefToWorld(surf + ref_h(ax_f[ox], ay_f[oy]) * nn);
    };

    for (int y = 0; y <= kChunkQuads; ++y) {
        const std::size_t yb = static_cast<std::size_t>(y) * kOff;
        for (int x = 0; x <= kChunkQuads; ++x) {
            const std::size_t xb = static_cast<std::size_t>(x) * kOff;
            const glm::dvec3 dxr = ref_world(xb + 2, yb + 1)
                                 - ref_world(xb + 0, yb + 1);
            const glm::dvec3 dyr = ref_world(xb + 1, yb + 2)
                                 - ref_world(xb + 1, yb + 0);
            glm::dvec3 nr = glm::cross(dxr, dyr);
            const double nl = glm::length(nr);
            if (!(nl > 0.0)) continue;
            nr /= nl;
            const glm::dvec3 up = site.WorldUp(ref_world(xb + 1, yb + 1));
            // Outward, by the same guard BuildTerrainChunk uses on the mesh
            // normal: the face bases are right-handed so cross(du, dv)
            // already points away from the centre, and the dot makes the
            // assumption checkable instead of assumed.
            if (glm::dot(nr, up) < 0.0) nr = -nr;
            out[static_cast<std::size_t>(y) * kChunkVerts + x] =
                std::clamp(1.0 - std::abs(glm::dot(nr, up)), 0.0, 1.0);
        }
    }
}

namespace {

// The part of ReferenceRmsSlopePerAxis that does not depend on POSITION.
//
// `S`, the lag, the two Hurst exponents, the break and the detail gain are
// all properties of the FIELD, so on a coarse chunk's 4 225-vertex sweep
// they were being recomputed 4 225 times to produce one number -- including
// RelativeStructureFunction, which is two calls to std::pow. Splitting them
// out lets the per-vertex path be a multiply and a divide.
//
// The factorisation keeps the ORIGINAL EVALUATION ORDER exactly:
//   gain * detail_gain * Relief * S / lag
// associates left, so `(gain * detail_gain)` is a subexpression of it and
// hoisting that pair changes no rounding. Everything downstream still
// multiplies by Relief, then by S, then divides by lag, in that order.
struct RefRmsSlopeConstants {
    double gain_detail;   // kRefSlopeRmsGain * max(detail_gain, 0)
    double S;
    double lag;
};

RefRmsSlopeConstants RefRmsSlopeConstantsFor(const ElevationField& field) noexcept {
    const ElevationParams& p = field.Params();
    const double floor_m = field.DataFloorMetres();
    const double H  = std::clamp(p.hurst, 0.05, 1.0);
    const double Hf = std::clamp(p.hurst_fine, H, 2.0);
    const double brk = std::max(p.hillslope_break_m, 0.0);
    const double lag = 2.0 * kRefSlopeHalfLagM;
    return RefRmsSlopeConstants{
        kRefSlopeRmsGain * std::max(p.detail_gain, 0.0),
        RelativeStructureFunction(lag, floor_m, brk, H, Hf),
        lag};
}

inline double RefRmsSlopeAt(const RefRmsSlopeConstants& c,
                            const ElevationField& field,
                            const glm::dvec3& dir_unit) noexcept {
    return c.gain_detail * field.Relief(dir_unit) * c.S / c.lag;
}

}  // namespace

double ReferenceRmsSlopePerAxis(const ElevationField& field,
                                const glm::dvec3& dir_unit) noexcept {
    return RefRmsSlopeAt(RefRmsSlopeConstantsFor(field), field, dir_unit);
}

void BuildTerrainChunk(const ChunkKey& key,
                       const ElevationField& field,
                       const PlanetSite& site,
                       TerrainChunkData& out) {
    out.key = key;
    out.vertex_spacing_m = ChunkVertexSpacing(key.level);

    // One evaluation pass at level+1 with a halo. Everything below --
    // vertices, normals, e_L, the slope statistics -- reads this grid.
    std::vector<double> fine;
    field.GenerateChunkHeights(key, static_cast<int>(key.level) + 1, kFineHalo, fine);

    const std::int64_t Gf = static_cast<std::int64_t>(kChunkQuads)
                          << (static_cast<int>(key.level) + 1);
    const std::int64_t span = static_cast<std::int64_t>(1) << key.level;
    const std::int64_t base_x = (static_cast<std::int64_t>(key.i) * Gf) / span - kFineHalo;
    const std::int64_t base_y = (static_cast<std::int64_t>(key.j) * Gf) / span - kFineHalo;

    // World position of a fine-grid vertex.
    auto fine_world = [&](int fx, int fy) -> glm::dvec3 {
        const double s = GridParam(base_x + fx, Gf);
        const double t = GridParam(base_y + fy, Gf);
        const glm::dvec3 dir = FaceParamToDirection(key.face, s, t);
        const glm::dvec3 surf = EllipsoidSurface(dir);
        const glm::dvec3 n = GeodeticNormal(surf);
        const double h = fine[static_cast<std::size_t>(FineIdx(fx, fy))];
        return site.EcefToWorld(surf + h * n);
    };

    // --- Chunk-local origin ------------------------------------------------
    // The centre vertex at its own elevation. Any point inside the chunk
    // works; the centre minimises the local coordinate magnitude and
    // therefore maximises the float32 resolution of the vertex array.
    const glm::dvec3 origin_w = fine_world(FineOfChunk(kChunkQuads / 2),
                                           FineOfChunk(kChunkQuads / 2));
    out.origin_w = origin_w;

    // --- Reference-scale rock exposure (#307) ------------------------------
    //
    // A pure function of position, evaluated on the level-kRefSlopeLevel
    // lattice over a kRefSlopeHalfLagM baseline. See the header for why
    // that operator and not the mesh's own.
    //
    // Two branches, and they are the two halves of one mip chain rather
    // than two models: POINTWISE where the mesh can carry a 180 m signal
    // (vertex spacing <= kRefSlopeHalfLagM, i.e. level >= kRefSlopeLevel),
    // and the AREA MEAN of the same ramp where it cannot. The crossover is
    // at the Nyquist of the reference baseline, not at a level someone
    // picked, and the two agree in the mean there by measurement -- see
    // kRefSlopeRmsGain and tests/pt_planet_terrain_test.cpp.
    std::vector<float> rock01(static_cast<std::size_t>(kChunkVertexCount), 0.0f);
    if (static_cast<int>(key.level) >= kRefSlopeLevel) {
        std::vector<double> slope_ref;
        ReferenceSlope01(key, field, site, slope_ref);
        for (std::size_t i = 0; i < rock01.size(); ++i) {
            rock01[i] = static_cast<float>(SlopeRockFraction(slope_ref[i]));
        }
    } else {
        const std::int64_t Gc = static_cast<std::int64_t>(kChunkQuads) << key.level;
        const std::int64_t cbase_x = static_cast<std::int64_t>(key.i) * kChunkQuads;
        const std::int64_t cbase_y = static_cast<std::int64_t>(key.j) * kChunkQuads;
        // Same two hoists as ReferenceSlope01's: the field constants come
        // out of the 4 225-vertex sweep, and TangentWarp is a function of
        // ONE axis so it is evaluated 65 times per axis rather than 4 225
        // times per axis. Both are pure common-subexpression removal -- the
        // per-vertex arithmetic below is the identical expression on the
        // identical doubles.
        const RefRmsSlopeConstants rc = RefRmsSlopeConstantsFor(field);
        const glm::dvec3 face_n = FaceNormal(key.face);
        const glm::dvec3 face_r = FaceRight(key.face);
        const glm::dvec3 face_u = FaceUp(key.face);
        std::vector<glm::dvec3> cx(static_cast<std::size_t>(kChunkVerts));
        std::vector<glm::dvec3> cy(static_cast<std::size_t>(kChunkVerts));
        for (int x = 0; x <= kChunkQuads; ++x) {
            cx[static_cast<std::size_t>(x)] =
                TangentWarp(GridParam(cbase_x + x, Gc)) * face_r;
        }
        for (int y = 0; y <= kChunkQuads; ++y) {
            cy[static_cast<std::size_t>(y)] =
                TangentWarp(GridParam(cbase_y + y, Gc)) * face_u;
        }
        // The threshold-hillslope saturation (#330) draws the pointwise
        // reference slope toward S_c; the area-mean branch must prefilter the
        // SAME saturated slope or the two mip levels disagree where relief is
        // high. The saturation is a per-vertex map on the SLOPE, so its
        // reduction factor is set by the slope at the reference LAG itself --
        // relief*S(lag)/lag, the true scale slope -- NOT by the finite-
        // difference RMS, which kRefSlopeRmsGain inflates by folding in the
        // finer levels the transfer barely touches. Saturating the inflated
        // RMS as one slope over-reduces; saturating the true scale slope and
        // applying that factor to the RMS tracks the pointwise branch to a few
        // percent across the earth_lite relief range (re-measured in
        // pt_planet_terrain_test.cpp). rc.S and rc.lag are exactly S(lag) and
        // the reference lag already hoisted for the RMS.
        const double thr   = field.Params().hillslope_threshold_slope;
        const double dgain = std::max(field.Params().detail_gain, 0.0);
        for (int y = 0; y <= kChunkQuads; ++y) {
            for (int x = 0; x <= kChunkQuads; ++x) {
                const glm::dvec3 dir = glm::normalize(
                    face_n + cx[static_cast<std::size_t>(x)]
                           + cy[static_cast<std::size_t>(y)]);
                const double sigma_lin = RefRmsSlopeAt(rc, field, dir);
                // The scale slope at the reference lag, and the factor the
                // saturation applies to it -- inherited by the reference RMS.
                const double sigma_ref = dgain * field.Relief(dir) * rc.S / rc.lag;
                const double factor = (thr > 0.0 && sigma_ref > 0.0)
                    ? SaturateHillslopeSlope(sigma_ref, 1.0, thr) / sigma_ref
                    : 1.0;
                rock01[static_cast<std::size_t>(y) * kChunkVerts + x] =
                    static_cast<float>(RockFractionFromRmsSlope(factor * sigma_lin));
            }
        }
    }

    // --- Vertices, normals, elevation extent ------------------------------
    out.positions.assign(static_cast<std::size_t>(kChunkVertexCount) * 3u, 0.0f);
    out.shader_verts.assign(
        static_cast<std::size_t>(kChunkVertexCount) * kVertexPayloadFloats, 0.0f);
    double h_min =  1e30, h_max = -1e30;
    glm::dvec3 lo( 1e30), hi(-1e30);

    for (int y = 0; y <= kChunkQuads; ++y) {
        for (int x = 0; x <= kChunkQuads; ++x) {
            const int fx = FineOfChunk(x);
            const int fy = FineOfChunk(y);
            const glm::dvec3 p = fine_world(fx, fy);
            const glm::vec3 local(p - origin_w);
            const std::size_t vi = static_cast<std::size_t>(y) * kChunkVerts + x;
            out.positions[vi * 3 + 0] = local.x;
            out.positions[vi * 3 + 1] = local.y;
            out.positions[vi * 3 + 2] = local.z;

            // Central difference over the COARSE neighbours (two fine
            // steps), so the normal describes the surface the 65x65 mesh
            // actually renders rather than the finer one underneath it.
            const glm::dvec3 dx = fine_world(fx + 2, fy) - fine_world(fx - 2, fy);
            const glm::dvec3 dy = fine_world(fx, fy + 2) - fine_world(fx, fy - 2);
            glm::dvec3 n = glm::cross(dx, dy);
            const double nl = glm::length(n);
            if (nl > 0.0) {
                n /= nl;
            } else {
                n = site.WorldUp(p);
            }
            // Orient outward. The face bases are right-handed with
            // cross(right, up) == normal, so cross(dx, dy) already points
            // away from the planet centre; the guard costs one dot and
            // makes the winding assumption checkable rather than assumed.
            if (glm::dot(n, site.WorldUp(p)) < 0.0) n = -n;

            const double h = fine[static_cast<std::size_t>(FineIdx(fx, fy))];
            const std::size_t so = vi * static_cast<std::size_t>(kVertexPayloadFloats);
            out.shader_verts[so + 0] = static_cast<float>(n.x);
            out.shader_verts[so + 1] = static_cast<float>(n.y);
            out.shader_verts[so + 2] = static_cast<float>(n.z);
            out.shader_verts[so + 3] = static_cast<float>(h);
            out.shader_verts[so + 4] = rock01[vi];

            h_min = std::min(h_min, h);
            h_max = std::max(h_max, h);
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
    }
    out.h_min_m = h_min;
    out.h_max_m = h_max;
    out.bound_center_w = 0.5 * (lo + hi);
    out.bound_radius_m = 0.5 * glm::length(hi - lo);

    // --- e_L: the error of NOT splitting ----------------------------------
    //
    // For every vertex of the level+1 grid, compare its true 3D position
    // against the point this chunk's triangulated 65x65 surface puts at the
    // same parameter. Even-even fine vertices coincide with chunk vertices
    // and contribute exactly zero (the elevation field is interpolatory),
    // so the max is over the three other parities. The triangulation is the
    // shared index arena's: quad (x, y) splits into (v00, v10, v11) and
    // (v00, v11, v01), i.e. the diagonal runs from (x, y) to (x+1, y+1).
    //
    // MEASURED IN 3D, NOT IN HEIGHT. The first version of this compared
    // heights only, and on a smooth body that is identically zero: the
    // selector then never split anything, because the sphere's own
    // curvature -- the coarse mesh cutting a chord where the fine mesh
    // follows the arc -- is invisible to a height comparison. That sag is
    // 477 m on a level-0 chunk and 7.3 mm on a level-8 one, and it is the
    // ONLY thing that drives tessellation on a flat sea.
    auto chunk_p = [&](int x, int y) -> glm::dvec3 {
        const std::size_t vi = static_cast<std::size_t>(y) * kChunkVerts + x;
        return glm::dvec3(out.positions[vi * 3 + 0],
                          out.positions[vi * 3 + 1],
                          out.positions[vi * 3 + 2]);
    };
    auto chunk_h = [&](int x, int y) -> double {
        return fine[static_cast<std::size_t>(FineIdx(FineOfChunk(x), FineOfChunk(y)))];
    };
    double e_l = 0.0;
    for (int fy = 0; fy <= kFineCells; ++fy) {
        for (int fx = 0; fx <= kFineCells; ++fx) {
            if (((fx | fy) & 1) == 0) continue;          // a chunk vertex
            const double u = 0.5 * static_cast<double>(fx);   // in chunk cells
            const double v = 0.5 * static_cast<double>(fy);
            const int cx = std::min(static_cast<int>(u), kChunkQuads - 1);
            const int cy = std::min(static_cast<int>(v), kChunkQuads - 1);
            const double a = u - cx;
            const double b = v - cy;
            const glm::dvec3 p00 = chunk_p(cx,     cy);
            const glm::dvec3 p10 = chunk_p(cx + 1, cy);
            const glm::dvec3 p01 = chunk_p(cx,     cy + 1);
            const glm::dvec3 p11 = chunk_p(cx + 1, cy + 1);
            const glm::dvec3 interp = (b <= a)
                ? p00 + (p10 - p00) * a + (p11 - p10) * b      // lower triangle
                : p00 + (p11 - p01) * a + (p01 - p00) * b;     // upper triangle
            const glm::dvec3 truth =
                fine_world(fx + kFineHalo, fy + kFineHalo) - origin_w;
            e_l = std::max(e_l, glm::length(truth - interp));
        }
    }
    out.e_l_m = e_l;

    // --- Per-mip mean-square slope ----------------------------------------
    //
    // Bruneton, Neyret & Holzschuch (2010) fold the sub-footprint slope
    // distribution into the microfacet roughness rather than letting an
    // averaged normal silently REMOVE specular energy. sigma2[m] is the
    // variance of the surface slope that a footprint of 2^m vertex spacings
    // averages away, computed as the mean over blocks of
    //     E_block[|s|^2] - |E_block[s]|^2.
    // Track two pyramids: the mean slope vector and the mean squared slope
    // magnitude. This is what makes "seamless surface to orbit" work -- from
    // orbit a mountain range must not go smooth, it must go ROUGH, which is
    // what satellite imagery shows.
    const double spacing = std::max(out.vertex_spacing_m, 1e-6);
    std::vector<double> m1x(kChunkQuads * kChunkQuads);
    std::vector<double> m1y(kChunkQuads * kChunkQuads);
    std::vector<double> m2 (kChunkQuads * kChunkQuads);
    for (int y = 0; y < kChunkQuads; ++y) {
        for (int x = 0; x < kChunkQuads; ++x) {
            const double sx = (chunk_h(x + 1, y) - chunk_h(x, y)) / spacing;
            const double sy = (chunk_h(x, y + 1) - chunk_h(x, y)) / spacing;
            const std::size_t c = static_cast<std::size_t>(y) * kChunkQuads + x;
            m1x[c] = sx;
            m1y[c] = sy;
            m2[c]  = sx * sx + sy * sy;
        }
    }
    out.sigma2.fill(0.0f);
    int dim = kChunkQuads;
    for (int m = 1; m < kSlopeMips && dim > 1; ++m) {
        const int nd = dim / 2;
        double acc = 0.0;
        std::vector<double> nx(static_cast<std::size_t>(nd) * nd);
        std::vector<double> ny(static_cast<std::size_t>(nd) * nd);
        std::vector<double> n2(static_cast<std::size_t>(nd) * nd);
        for (int y = 0; y < nd; ++y) {
            for (int x = 0; x < nd; ++x) {
                const std::size_t a = static_cast<std::size_t>(2 * y)     * dim + 2 * x;
                const std::size_t b = a + 1;
                const std::size_t c = a + dim;
                const std::size_t d = c + 1;
                const double ax = 0.25 * (m1x[a] + m1x[b] + m1x[c] + m1x[d]);
                const double ay = 0.25 * (m1y[a] + m1y[b] + m1y[c] + m1y[d]);
                const double a2 = 0.25 * (m2[a] + m2[b] + m2[c] + m2[d]);
                const std::size_t o = static_cast<std::size_t>(y) * nd + x;
                nx[o] = ax; ny[o] = ay; n2[o] = a2;
                acc += std::max(0.0, a2 - (ax * ax + ay * ay));
            }
        }
        out.sigma2[static_cast<std::size_t>(m)] =
            static_cast<float>(acc / static_cast<double>(nd * nd));
        m1x.swap(nx); m1y.swap(ny); m2.swap(n2);
        dim = nd;
    }
    // Levels past the chunk's own extent keep the coarsest measured value:
    // a footprint wider than the chunk cannot remove more slope variance
    // than the whole chunk contains.
    for (int m = 1; m < kSlopeMips; ++m) {
        if (out.sigma2[static_cast<std::size_t>(m)] == 0.0f) {
            out.sigma2[static_cast<std::size_t>(m)] =
                out.sigma2[static_cast<std::size_t>(m - 1)];
        }
    }
}

}  // namespace pt::planet

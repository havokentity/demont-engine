// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Yale Bright Star Catalog (BSC5) loader + J2000 equirectangular
// starmap rasteriser. The on-disk catalog (assets/stars/BSC5.dat) is
// the original Harvard CDC binary -- 28-byte header, 9110 fixed-width
// 32-byte records (HR number, RA/Dec in J2000 radians as float64,
// spectral type, V mag * 100, proper motions). We don't model proper
// motion -- a few arcsec/yr is invisible at any zoom we actually
// render, and the visual layout of constellations hasn't shifted
// noticeably since J2000 anyway.

#include <cstdint>
#include <string>
#include <vector>

namespace pt::stars {

struct Star {
    float ra_deg;    // J2000 right ascension, [0, 360)
    float dec_deg;   // J2000 declination, [-90, 90]
    float vmag;      // visual magnitude (lower = brighter)
};

// Load and parse BSC5 binary at `path`. On success returns a vector
// sorted by ascending vmag (brightest first), filtered to entries with
// finite RA/Dec and a sensible vmag. On failure returns an empty
// vector and writes a reason to `err` if non-null.
std::vector<Star> LoadBsc5(const std::string& path, std::string* err = nullptr);

// Rasterise the catalog into an RGBA16F equirectangular texture in the
// J2000 frame (RA across X in [0, 2pi], Dec across Y from +90 deg at
// y=0 down to -90 deg at y=H-1). Stars are splatted as Gaussian dots
// whose intensity follows the standard astronomical magnitude scale
// (each step of 1 mag ~= factor 2.512 in flux) and whose tint is
// derived from B-V color heuristics (we don't have B-V here, so we
// pick from a small palette by RA/Dec hash for visual variety).
//
// Output: row-major float-RGBA, len = W*H*4. Caller passes this to
// the RHI as RGBA16F (the floats fit easily in half-float range; the
// brightest star in the catalog tops out around intensity ~25).
void RasteriseJ2000Map(const std::vector<Star>& stars,
                       std::uint32_t W, std::uint32_t H,
                       std::vector<float>& out_rgba);

}  // namespace pt::stars

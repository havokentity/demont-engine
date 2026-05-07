// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Procedural lunar surface texture generator. Produces a cylindrical
// (equirectangular) RGBA16F-half-friendly float map of the moon's
// near side -- highlands base, dark mare basins, sub-arcsecond crater
// detail. Run once at engine init (~50ms for 512x256) and uploaded as
// a 2D texture sampled by the path tracer's moonDisc().
//
// Why procedural: shipping a real NASA SVS / USGS lunar mosaic would
// add a ~1MB+ binary asset to the repo. The procedural look is good
// enough to read as "moon" through the small angular footprint of the
// disc on screen (~9 px at 1080p, 60deg FOV).

#pragma once

#include <vector>

namespace pt::moon {

// Fill `rgba_out` with a procedural moon surface in float RGBA, in
// equirectangular projection (width = 2 * height for proper sphere
// coverage). Output is in [0, 1] linear space; engine converts to
// half-precision before upload. Recommended size: 512 x 256.
void generateMoonTexture(int width, int height, std::vector<float>& rgba_out);

}  // namespace pt::moon

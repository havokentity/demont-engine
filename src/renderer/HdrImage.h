// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Radiance .hdr (RGBE) image loader. Single-purpose: enough to support
// IBL via lat-long environment maps in the path tracer. Returns a flat
// float[w*h*3] buffer (HDR linear, no tonemap).
//
// Format reference: https://www.graphics.cornell.edu/online/formats/rgbe/
// Header is text terminated by a blank line, then a Y/X resolution
// specifier, then run-length-encoded RGBE pixels (4 bytes each: r, g,
// b, exponent). Linear value per channel = (byte / 256) * 2^(exp - 128).

#include <cstdint>
#include <string>
#include <vector>

namespace pt::renderer {

struct HdrImage {
    std::uint32_t      width  = 0;
    std::uint32_t      height = 0;
    std::vector<float> rgb;     // tightly packed, w * h * 3 floats

    bool Empty() const { return rgb.empty() || width == 0 || height == 0; }
};

// Returns Empty() image on failure. `out_error` (optional) gets a short
// human-readable diagnostic.
HdrImage LoadRadianceHdr(const std::string& path, std::string* out_error = nullptr);

}  // namespace pt::renderer

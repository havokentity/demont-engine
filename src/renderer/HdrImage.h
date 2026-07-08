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

// Build the 2D importance-sampling CDF the path tracer's env-map NEE
// walks: a per-row conditional CDF over u, and a marginal CDF over v,
// both weighted by luminance * sin(theta) (the lat-long solid-angle
// Jacobian). Returns the total (unnormalized) weight, which the engine
// publishes as env_total_luminance.
//
// `light_mask` (optional; empty = none) zeroes the CDF luminance of any
// pixel belonging to an extracted bright cluster, so env-map NEE skips
// those -- the shader lights them through the separate directional
// hdri_lights NEE instead, and double-counting them here would bias the
// estimator. The env_map TEXTURE keeps the bright pixels intact so
// camera-direct rays still see the sun.
//
// Outputs are sized W*H (conditional) and H (marginal). marginal[H-1] is
// forced to exactly 1.0 to absorb prefix-sum FP drift when the image has
// any luminance; an all-black image yields an all-zero marginal (and a
// zero return), which the shader treats as "env-NEE disabled".
//
// This is the single source of truth: Engine::ReloadEnvMap and
// tests/hdrimage_cdf_test both call it, so the test can no longer pass
// against a private re-implementation that has drifted from the shipped
// code (it had -- the mirror never applied light_mask).
double BuildEnvCdf(const std::vector<float>& rgb,
                   std::uint32_t W, std::uint32_t H,
                   const std::vector<std::uint8_t>& light_mask,
                   std::vector<float>& out_marginal,
                   std::vector<float>& out_conditional);

}  // namespace pt::renderer

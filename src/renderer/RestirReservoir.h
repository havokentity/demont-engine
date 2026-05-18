// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// ReSTIR DI Phase A (issue #78) -- Reservoir-based Spatio-Temporal
// Importance Resampling for Direct Illumination.
//
// This header is the HOST-side mirror of the GPU-side `Reservoir`
// struct declared in shaders/RestirReservoir.slang. They MUST stay
// layout-byte-identical because the engine writes the per-pixel
// reservoir SSBO with PathTrace.slang's WRS candidate-generation
// pass and reads it back via the RestirTemporal / RestirSpatial /
// RestirFinal kernels. A drift between the two would silently corrupt
// the resampling chain (NaN survivor weights / out-of-bounds light
// indices) — hence the static_assert below.
//
// Layout philosophy: 64 bytes per pixel, aligned to 16-byte float4 lanes
// so Slang's StructuredBuffer<Reservoir>-or-StructuredBuffer<float4>
// loaders stay simple. At 1920x1080 the storage cost is ~127 MB per
// buffer; we double-buffer for temporal reuse so total VRAM is ~254 MB
// — comfortable on the M4 Max's 64 GB unified memory.
//
// Field semantics (matches the WRS algorithm in Bitterli et al.
// "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing
// with Dynamic Direct Lighting", SIGGRAPH 2020):
//
//   y          : the currently-chosen sample (light index + sample point).
//   w_sum      : running sum of source pdfs / target pdfs for WRS.
//   M          : number of candidates merged so far (capped by
//                kTemporalMaxM to bound temporal accumulation).
//   W          : final resampling weight = w_sum / (M * p_hat(y))
//                used as the per-sample weight in the final estimator.
//   p_hat      : target pdf of y at the receiver (the un-shadowed
//                Lambert estimate length(Le * brdf * cos) / dist^2).
//   pdf_solid  : original sampling pdf in solid-angle terms (selection
//                pmf * per-light area-to-solid-angle conversion).
//                Carried so the final shading pass can divide by the
//                true Veach-style pdf, not by p_hat (which would re-
//                introduce bias).
//   dist       : world-space distance from receiver to the chosen
//                sample point on the light surface. The final shadow
//                ray uses this as t_max.
//
// Encoded as a 4-float4 record so loadReservoir / storeReservoir on the
// GPU can just do 4 buffer.Load4 calls (no scalar-stride packing
// gymnastics).

#pragma once

#include <cstdint>

namespace pt::renderer {

// --- ReSTIR DI Phase A constants -------------------------------------------
// Per-pixel candidate budget for the initial Weighted Reservoir Sampling
// pass. K=8 is the AAA default (NVIDIA Falcor RTXDI ships 8 as the
// "balanced" preset). The shader caps this at 64 to bound the inner
// loop's worst-case ALU cost; the host cvar `r_restir_k_candidates`
// is clamped to [1, 64] before push-constant upload.
inline constexpr std::uint32_t kRestirMaxK = 64u;

// Temporal-reuse cap on Reservoir::M. Without a cap the merged sample
// count grows monotonically and biases the WRS toward the OLDEST sample
// — a moving camera produces visible smear because the survivor never
// "lets go" of stale candidates. Bitterli §5.3 recommends 20-30. We
// pick 20 (matches the original paper's reference figures) and keep it
// hardcoded rather than cvar-exposed; the bias tradeoff is a per-pixel
// constant, not a runtime knob users would reach for.
inline constexpr std::uint32_t kRestirTemporalMaxM = 20u;

// Spatial-reuse neighbour count per pass and search radius. 3-5 is the
// AAA-default range; 3 is the cheaper end (Cyberpunk 2077 RT mode uses
// 3-5 adaptively). 10-pixel radius keeps the reservoir hits within a
// roughly-disocclusion-safe window at 1080p.
inline constexpr std::uint32_t kRestirSpatialNeighbours = 4u;
inline constexpr float         kRestirSpatialRadiusPx   = 10.0f;

// Bias-mode tag. Selectable via r_restir_bias.
//   Biased (0)   : straight WRS resampling, no per-pair MIS weight.
//                   Cheaper (~30% off the spatial pass cost). Bias
//                   manifests as a slight under-estimate where two
//                   reservoirs disagree on surface orientation; SVGF /
//                   MetalFX absorbs it visually.
//   Unbiased (1) : per-pair MIS balance heuristic at every spatial
//                   merge — Bitterli §6.2 "RIS with unbiased weights."
//                   The correct estimator, but ~30% slower. Used for
//                   ground-truth comparisons in offline benches.
inline constexpr std::uint32_t kRestirBiasBiased   = 0u;
inline constexpr std::uint32_t kRestirBiasUnbiased = 1u;

// --- GPU-mirror struct -----------------------------------------------------
// Total = 64 bytes = 4 × float4. Slang's StructuredBuffer<Reservoir> on
// MSL lays this out without padding (all natural alignments line up at
// 4 bytes), and on SPIR-V std430 the same is true. Keep ALL fields
// scalar (no bare bool / smaller-than-4-byte types) so both compilers
// emit the same byte layout.
struct alignas(16) Reservoir {
    // Lane 0
    std::uint32_t light_idx;     // index into light_prims; UINT32_MAX = invalid (empty reservoir)
    float         p_hat;         // target pdf at the receiver (un-shadowed Lambert estimate)
    float         w_sum;         // running sum of (source_pdf / target_pdf) weights
    std::uint32_t M;             // # of candidates merged so far (capped by kTemporalMaxM)

    // Lane 1: sample point on the light surface (world space)
    float         light_pos_x;
    float         light_pos_y;
    float         light_pos_z;
    float         W;             // final resampling weight = w_sum / (M * p_hat)

    // Lane 2: light-side normal (sphere/quad emitters); spot/point write float3(0,1,0)
    float         light_n_x;
    float         light_n_y;
    float         light_n_z;
    float         pdf_solid;     // original sampling pdf in solid-angle units

    // Lane 3: cached Le * pre-shadow scale + distance for the final shadow ray
    float         Le_r;
    float         Le_g;
    float         Le_b;
    float         dist;          // shadow-ray t_max
};

static_assert(sizeof(Reservoir) == 64,
              "Reservoir host struct must be 64B to match RestirReservoir.slang's GPU layout");
static_assert(alignof(Reservoir) == 16,
              "Reservoir host struct must be 16B-aligned (std430 / MSL float4 rule)");

// Sentinel for "this pixel has no candidate yet" — written by the
// initial WRS pass when the candidate budget is exhausted with no
// non-zero source-pdf samples (every candidate was back-facing or
// behind a light's emission lobe). The temporal/spatial passes treat
// it as M=0 and skip the merge; the final pass writes zero radiance.
inline constexpr std::uint32_t kRestirInvalidLight = 0xFFFFFFFFu;

}  // namespace pt::renderer

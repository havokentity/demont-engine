// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <cstddef>

#include "rhi/CommandBuffer.h"
#include "rhi/Handles.h"

// --- #259 / #133 Phase 2: recording the ocean cascade pre-pass ------------
//
// The push-constant layout and the five-pass dispatch sequence for
// shaders/OceanCascades.slang, in ONE place, so the engine's driver and
// tests/pt_ocean_gpu_test.cpp are not two transcriptions of the same wire
// format that can drift apart. Header-only and dependent on nothing but
// the RHI's abstract CommandBuffer, so it adds no link edge to pt_physics
// and no library to the test.
//
// This is deliberately a RECORDER, not an owner: allocation, the spectrum
// upload and the lifetime of every handle belong to the caller (Engine for
// the render path, the test for the comparison). What is shared is exactly
// the part that has to agree with the shader.
namespace pt::ocean {

// Pass indices. MIRRORED as PASS_* in shaders/OceanCascades.slang.
inline constexpr std::uint32_t kGpuPassRows   = 0u;
inline constexpr std::uint32_t kGpuPassCols   = 1u;
inline constexpr std::uint32_t kGpuPassPack   = 2u;
inline constexpr std::uint32_t kGpuPassReduce = 3u;
inline constexpr std::uint32_t kGpuPassFoam   = 4u;
// Complex spectra transformed per cascade: height, dh/dx, dh/dz, Dx, Dz.
// MIRRORED as kFields in shaders/OceanCascades.slang.
inline constexpr std::uint32_t kGpuFields     = 5u;
// Threads per group in each axis. MetalCommandBuffer::Dispatch hardcodes
// [numthreads(8,8,1)] for every pipeline in the engine, so the pack / foam
// dispatch shapes are in units of 8x8 texel tiles and the FFT lanes stride
// by 64.
inline constexpr std::uint32_t kGpuGroupDim   = 8u;

// Byte-identical to the `Push` cbuffer in shaders/OceanCascades.slang.
// 80 B, under VulkanDevice::kPushSplitOffset (112 B), so unlike PathTrace
// and CloudsRaymarch there is no Frame-UBO tail to keep in step.
struct GpuCascadePush {
    // .x pass index, .y grid size N, .z cascade count, .w reserved.
    std::uint32_t ctl[4];
    // .x absolute sim time t (s), .y choppiness lambda,
    // .z foam Jacobian threshold, .w foam amount.
    float         p0[4];
    // .x foam persistence decay for this frame's dt, .y wind whitecap
    // coverage in [0,1], .z march-bracket scale, .w march-bracket bias (m).
    float         p1[4];
    // Cascade tile periods (m). .w unused.
    float         period[4];
    // .x gravity (m/s^2).
    float         p2[4];
};
static_assert(sizeof(GpuCascadePush) == 80,
              "GpuCascadePush must match OceanCascades.slang's Push block");

// Everything the five passes need. Handles are raw ids so the caller can
// keep using whatever wrapper it already has.
struct GpuCascadeDispatch {
    std::uint64_t pipeline    = 0;
    std::uint64_t h0_buf      = 0;
    std::uint64_t scratch_buf = 0;
    std::uint64_t foam_buf    = 0;
    std::uint64_t reduce_buf  = 0;
    std::uint64_t disp_tex    = 0;
    std::uint64_t normal_tex  = 0;

    std::uint32_t grid     = 0;
    std::uint32_t cascades = 0;

    double t_seconds     = 0.0;
    float  choppiness    = 1.0f;
    float  foam_threshold = 0.5f;
    float  foam_amount   = 1.0f;
    // Host-evaluated: std::pow(persistence, dt*60) and the whitecap-onset
    // smoothstep. Evaluated on the host rather than in the kernel so no
    // transcendental other than the FFT's own cos/sin contributes to the
    // CPU-vs-GPU difference the equivalence test bounds.
    float  foam_decay    = 1.0f;
    float  wind_coverage = 0.0f;
    // The slack the ray-march bracket carries over the summed peak crest:
    // bracket = scale * sum(peak |h|) + bias. The same 1.25 / 0.05 m the
    // CPU path puts on ocean_params0.w.
    float  bracket_scale = 1.25f;
    float  bracket_bias  = 0.05f;
    float  gravity       = 9.81f;
    double period_m[3]   = {0.0, 0.0, 0.0};
};

inline GpuCascadePush MakeGpuCascadePush(const GpuCascadeDispatch& d) {
    GpuCascadePush p{};
    p.ctl[0] = 0u;
    p.ctl[1] = d.grid;
    p.ctl[2] = d.cascades;
    p.ctl[3] = 0u;
    p.p0[0] = static_cast<float>(d.t_seconds);
    p.p0[1] = d.choppiness;
    p.p0[2] = d.foam_threshold;
    p.p0[3] = d.foam_amount;
    p.p1[0] = d.foam_decay;
    p.p1[1] = d.wind_coverage;
    p.p1[2] = d.bracket_scale;
    p.p1[3] = d.bracket_bias;
    p.period[0] = static_cast<float>(d.period_m[0]);
    p.period[1] = static_cast<float>(d.period_m[1]);
    p.period[2] = static_cast<float>(d.period_m[2]);
    p.period[3] = 0.0f;
    p.p2[0] = d.gravity;
    p.p2[1] = 0.0f; p.p2[2] = 0.0f; p.p2[3] = 0.0f;
    return p;
}

// Record the five passes into `cb`. The caller must already have the
// textures allocated at grid x (cascades*grid + 1) for the displacement
// atlas (the extra row is the march-bracket metadata) and grid x
// (cascades*grid) for the normal atlas.
//
// Ordering between the passes is the compute encoder's: Metal's default
// encoder is MTLDispatchTypeSerial, so dispatch N+1 sees everything
// dispatch N wrote. The explicit Barrier calls state the dependency
// anyway, and are what a Vulkan port would consume.
inline void RecordOceanCascades(pt::rhi::CommandBuffer* cb,
                                const GpuCascadeDispatch& d) {
    if (cb == nullptr || d.pipeline == 0 || d.grid == 0 || d.cascades == 0) {
        return;
    }
    cb->BindComputePipeline(pt::rhi::PipelineHandle{d.pipeline});
    cb->BindBuffer(0, pt::rhi::BufferHandle{d.h0_buf});
    cb->BindBuffer(1, pt::rhi::BufferHandle{d.scratch_buf});
    cb->BindBuffer(2, pt::rhi::BufferHandle{d.foam_buf});
    cb->BindBuffer(3, pt::rhi::BufferHandle{d.reduce_buf});
    cb->BindStorageTexture(0, pt::rhi::TextureHandle{d.disp_tex});
    cb->BindStorageTexture(1, pt::rhi::TextureHandle{d.normal_tex});

    GpuCascadePush p = MakeGpuCascadePush(d);
    const std::uint32_t tiles = (d.grid + kGpuGroupDim - 1u) / kGpuGroupDim;
    auto pass = [&](std::uint32_t index, std::uint32_t gx, std::uint32_t gy,
                    std::uint32_t gz) {
        p.ctl[0] = index;
        cb->PushConstants(&p, sizeof(p));
        cb->Dispatch(gx, gy, gz);
        cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                     pt::rhi::BarrierDesc::Stage::ComputeRead});
    };
    // One threadgroup per (line, field, cascade) for the separable
    // transform; one thread per texel for the pack and the foam; one
    // threadgroup for the whole reduce.
    pass(kGpuPassRows,   d.grid, kGpuFields, d.cascades);
    pass(kGpuPassCols,   d.grid, kGpuFields, d.cascades);
    pass(kGpuPassPack,   tiles,  tiles,      d.cascades);
    pass(kGpuPassReduce, 1u,     1u,         1u);
    pass(kGpuPassFoam,   tiles,  tiles,      d.cascades);
}

}  // namespace pt::ocean

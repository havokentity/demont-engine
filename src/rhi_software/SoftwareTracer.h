// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>

namespace pt::rhi::sw {

class SoftwareDevice;
class SoftwareCommandBuffer;

// CPU path-tracer kernel entry point. Called from
// SoftwareCommandBuffer::Dispatch when the bound pipeline is named
// "pathtrace". Reads the engine's push-constant struct (PtPush) out
// of cmd.push_constants_buf and the engine-bound buffers / textures /
// acceleration structures out of cmd.binds, and writes the final
// per-pixel colour into the slot-0 output texture's CPU backing.
//
// Current scope (session 1 minimum viable):
//   * Camera ray gen from the push struct's basis vectors.
//   * Analytic-primitive intersection (sphere + plane), linear loop --
//     the analytic-BVH path is GPU-only and not yet ported.
//   * Triangle-mesh intersection via Embree (the BLAS/TLAS path).
//   * Lambert direct lighting only -- no metal, no dielectric, no MIS,
//     no env-map sampling. Light comes from a procedural sky lerp
//     (gold-tinted toward the sun direction).
//   * Inline ACES tonemap + sRGB OETF; writes RGBA8-equivalent
//     floats into the output texture (the present blit clamps).
//   * Single sample per pixel, no accumulation across frames.
//
// Out of scope until later sessions:
//   * HDRI env-map sampling + MIS (would need env_map texture access,
//     marginal/conditional CDF buffers, etc.).
//   * Metal / dielectric BRDFs.
//   * Accumulator + denoiser chain (accum_hdr + autoexpose +
//     metalfx / SVGF / NRD don't apply to a CPU reference renderer).
//   * Volumetric cloud march, stars, moon, atmosphere LUTs.
//   * Multi-bounce indirect lighting (we do primary hit + 1 NEE for
//     direct light + miss-to-sky only).
void RunPathTraceKernel(SoftwareDevice& device,
                        const SoftwareCommandBuffer& cmd);

// CPU mirror of shaders/AccelProbe.slang (issue #254 P0 / issue #251).
// Called from SoftwareCommandBuffer::Dispatch when the bound pipeline
// is named "accel_probe". Traces one ray per element against the TLAS
// bound at accel slot 2 and writes two uints per ray -- hit flag and
// the instance id Embree's RTCHit::instID resolves to -- into the
// storage buffer bound at buffer slot 1, matching the Slang kernel's
// binding numbers and output layout exactly so the cross-backend test
// can compare like for like.
//
// This is the CPU backend's answer to "what does the shader see", the
// same way RunPathTraceKernel is the CPU backend's answer to
// PathTrace.slang. It exists only for the RHI acceleration-structure
// tests; the engine never dispatches it.
void RunAccelProbeKernel(SoftwareDevice& device,
                         const SoftwareCommandBuffer& cmd);

}  // namespace pt::rhi::sw

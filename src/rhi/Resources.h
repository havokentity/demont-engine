// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Handles.h"
#include "Types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace pt::rhi {

struct BufferDesc {
    std::size_t size       = 0;
    BufferUsage usage      = BufferUsage::Storage;
    std::string_view debug_name;   // optional, for capture tools
};

struct TextureDesc {
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
    TextureFormat format = TextureFormat::RGBA8_UNORM;
    TextureUsage  usage  = TextureUsage::Sampled;
    std::string_view debug_name;
};

// Compute-kernel descriptor.  In P2 the kernel is identified by a name
// (the software backend looks it up in a small built-in table); from P3 on
// it points at a Slang-compiled blob (MSL / SPIR-V / generated C++).
struct ComputePipelineDesc {
    std::string_view kernel_name;
    std::span<const std::uint8_t> bytecode;   // empty in software
    std::string_view debug_name;
};

// Acceleration-structure build hints (issue #254 P0). One field covers
// the three build policies the engine needs, mapped per backend:
//
//   PreferFastTrace  Vulkan VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
//                    Metal  (default usage -- Metal has no explicit
//                           "fast trace" bit; it is the absence of
//                           PreferFastBuild)
//                    SW     Embree RTC_BUILD_QUALITY_MEDIUM
//                    For geometry built once and traced for minutes:
//                    the static CSG mesh. This is the default so every
//                    existing call site keeps byte-identical behaviour.
//
//   PreferFastBuild  Vulkan ..._PREFER_FAST_BUILD_BIT_KHR
//                    Metal  MTL::AccelerationStructureUsagePreferFastBuild
//                    SW     Embree RTC_BUILD_QUALITY_LOW
//                    For geometry that churns: streaming terrain chunks
//                    that are built once and traced for a second or two.
//
//   AllowUpdate      Vulkan ..._ALLOW_UPDATE_BIT_KHR (a prerequisite for
//                           VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
//                    Metal  MTL::AccelerationStructureUsageRefit
//                    SW     Embree RTC_SCENE_FLAG_DYNAMIC
//                    Required before Device::UpdateTLASInstances can take
//                    the cheap refit path on a TLAS. Costs a slightly
//                    larger / slower initial build and (on the GPU
//                    backends) a retained scratch allocation, so it is
//                    opt-in rather than the default.
//
// PreferFastTrace and PreferFastBuild are mutually exclusive; if both are
// set the backends prefer fast build (matching Vulkan's own rule that a
// driver picks one).
enum class AccelBuildFlags : std::uint32_t {
    None            = 0u,
    PreferFastTrace = 1u << 0,
    PreferFastBuild = 1u << 1,
    AllowUpdate     = 1u << 2,
};

constexpr AccelBuildFlags operator|(AccelBuildFlags a, AccelBuildFlags b) {
    return static_cast<AccelBuildFlags>(static_cast<std::uint32_t>(a) |
                                        static_cast<std::uint32_t>(b));
}
constexpr AccelBuildFlags operator&(AccelBuildFlags a, AccelBuildFlags b) {
    return static_cast<AccelBuildFlags>(static_cast<std::uint32_t>(a) &
                                        static_cast<std::uint32_t>(b));
}
constexpr bool HasAccelFlag(AccelBuildFlags v, AccelBuildFlags f) {
    return (static_cast<std::uint32_t>(v) & static_cast<std::uint32_t>(f)) != 0u;
}

// Acceleration-structure descriptors.
//
// BLAS: a single triangle geometry built from CPU-provided arrays. The
// backend uploads them into device memory itself (so this struct holds
// raw pointers + counts rather than RHI buffer handles -- meshes don't
// need to be CPU-readable after the build).
struct BLASDesc {
    const float* vertex_positions = nullptr;  // tightly packed float3 per vertex
    std::uint32_t vertex_count    = 0;
    const std::uint32_t* indices  = nullptr;  // 3 per triangle
    std::uint32_t index_count     = 0;
    // Build policy. AllowUpdate makes the structure refit-capable on
    // every backend; the verb that feeds it new vertices is skeletal
    // animation's (#81) and does not exist yet, so setting it today
    // only pays the build cost. PreferFastBuild / PreferFastTrace are
    // live now.
    AccelBuildFlags flags = AccelBuildFlags::PreferFastTrace;
    std::string_view debug_name;
};

// One TLAS instance: a 4x3 row-major transform + which BLAS to instance.
// instance_id is delivered to the shader as InstanceCustomIndex / its
// equivalent so the shader can look up materials.
struct TLASInstance {
    AccelStructHandle blas;
    float             transform[12] {1,0,0,0, 0,1,0,0, 0,0,1,0};
    std::uint32_t     instance_id   = 0;
    std::uint32_t     mask          = 0xFF;
};

struct TLASDesc {
    std::span<const TLASInstance> instances;
    // The instance count given here is also the TLAS's *capacity*:
    // Device::UpdateTLASInstances accepts any count in [1, capacity]
    // but never grows the structure (growing means new storage, which
    // means a new handle). Add AllowUpdate to get the cheap refit path
    // for same-count / same-BLAS-set updates.
    AccelBuildFlags flags = AccelBuildFlags::PreferFastTrace;
    std::string_view debug_name;
};

struct BarrierDesc {
    enum class Stage : std::uint8_t { ComputeRead, ComputeWrite, Transfer, Present };
    Stage from = Stage::ComputeWrite;
    Stage to   = Stage::ComputeRead;
};

}  // namespace pt::rhi

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
    std::string_view debug_name;
};

struct BarrierDesc {
    enum class Stage : std::uint8_t { ComputeRead, ComputeWrite, Transfer, Present };
    Stage from = Stage::ComputeWrite;
    Stage to   = Stage::ComputeRead;
};

}  // namespace pt::rhi

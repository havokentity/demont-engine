#pragma once

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

// Acceleration-structure descriptors land in P8+.  Defined now so the RHI
// interface compiles at full shape.
struct BLASDesc {
    BufferHandle vertex_buffer;
    std::uint32_t vertex_stride = 0;
    std::uint32_t vertex_count  = 0;
    BufferHandle index_buffer;
    std::uint32_t index_count   = 0;
};

struct TLASDesc {
    std::span<const AccelStructHandle> blas_instances;
};

struct BarrierDesc {
    enum class Stage : std::uint8_t { ComputeRead, ComputeWrite, Transfer, Present };
    Stage from = Stage::ComputeWrite;
    Stage to   = Stage::ComputeRead;
};

}  // namespace pt::rhi

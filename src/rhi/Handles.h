#pragma once

#include <cstdint>

namespace pt::rhi {

// Opaque 64-bit handles.  Comparable, hashable, cheap to copy. id == 0
// always means "invalid/null".  Backends keep their own slot allocator
// internally and return packed (generation, slot) ids.
struct BufferHandle      { std::uint64_t id = 0; explicit operator bool() const { return id != 0; } };
struct TextureHandle     { std::uint64_t id = 0; explicit operator bool() const { return id != 0; } };
struct PipelineHandle    { std::uint64_t id = 0; explicit operator bool() const { return id != 0; } };
struct AccelStructHandle { std::uint64_t id = 0; explicit operator bool() const { return id != 0; } };
struct ShaderHandle      { std::uint64_t id = 0; explicit operator bool() const { return id != 0; } };

inline bool operator==(BufferHandle a, BufferHandle b) noexcept           { return a.id == b.id; }
inline bool operator!=(BufferHandle a, BufferHandle b) noexcept           { return a.id != b.id; }
inline bool operator==(TextureHandle a, TextureHandle b) noexcept         { return a.id == b.id; }
inline bool operator!=(TextureHandle a, TextureHandle b) noexcept         { return a.id != b.id; }
inline bool operator==(PipelineHandle a, PipelineHandle b) noexcept       { return a.id == b.id; }
inline bool operator!=(PipelineHandle a, PipelineHandle b) noexcept       { return a.id != b.id; }
inline bool operator==(AccelStructHandle a, AccelStructHandle b) noexcept { return a.id == b.id; }
inline bool operator!=(AccelStructHandle a, AccelStructHandle b) noexcept { return a.id != b.id; }
inline bool operator==(ShaderHandle a, ShaderHandle b) noexcept           { return a.id == b.id; }
inline bool operator!=(ShaderHandle a, ShaderHandle b) noexcept           { return a.id != b.id; }

}  // namespace pt::rhi

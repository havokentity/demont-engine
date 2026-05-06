#pragma once

#include <cstdint>

namespace pt::rhi {

enum class BackendType : std::uint8_t {
    None = 0,
    Software,
    Metal,
    Vulkan,
};

constexpr const char* BackendName(BackendType b) {
    switch (b) {
        case BackendType::None:     return "none";
        case BackendType::Software: return "software";
        case BackendType::Metal:    return "metal";
        case BackendType::Vulkan:   return "vulkan";
    }
    return "?";
}

// Pixel formats we care about for the path tracer.  More can land in P3+.
enum class TextureFormat : std::uint8_t {
    Unknown   = 0,
    RGBA8_UNORM,
    RGBA8_SRGB,
    RGBA16F,
    RGBA32F,
    R32_UINT,
    R32F,         // P10 denoiser depth
    RG16F,        // P10 denoiser motion vectors
};

enum class TextureUsage : std::uint32_t {
    None    = 0,
    Sampled = 1u << 0,   // shader read
    Storage = 1u << 1,   // shader read/write (RWTexture2D)
    Present = 1u << 2,   // can be presented to the swapchain
};
inline constexpr TextureUsage operator|(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr bool HasUsage(TextureUsage v, TextureUsage f) {
    return (static_cast<std::uint32_t>(v) & static_cast<std::uint32_t>(f)) != 0;
}

enum class BufferUsage : std::uint32_t {
    None        = 0,
    Vertex      = 1u << 0,
    Index       = 1u << 1,
    Storage     = 1u << 2,
    Uniform     = 1u << 3,
    Indirect    = 1u << 4,
    AccelInput  = 1u << 5,
    Upload      = 1u << 6,   // CPU-writable, GPU-readable
};
inline constexpr BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

}  // namespace pt::rhi

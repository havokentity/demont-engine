// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstddef>
#include <cstdint>

namespace pt::rhi {

// Capacity of every backend's CommandBuffer::PushConstants staging buffer.
//
// WHY THIS IS SHARED RATHER THAN PER-BACKEND. It used to be a bare 2048 in
// MetalDevice.h and another bare 2048 in VulkanDevice.h, each with a comment
// asking whoever grew PtPush to remember to grow it too. Land cover (#300)
// pushed sizeof(PtPush) to 2064 and BOTH silently dropped the last 16 bytes
// -- the terrain simply ignored its new albedo raster, with no error, on two
// backends at once. That is the third time this exact truncation has shipped:
// the header's own comment already records a black-frame regression (#24) and
// a mesh-motion-blur one (#21) from the same cause.
//
// So the number now lives in one place, and Engine.cpp static_asserts
// sizeof(PtPush) against it. The next overflow is a build error on every
// platform instead of a field that reads zero on some of them.
//
// 4096 is the CEILING, not a round number: Metal's
// setBytes:length:atIndex: accepts at most 4 KB, and past that the push
// block has to become a real buffer. Vulkan's own split (112 B of true push
// constants, the rest through VulkanDevice::kFrameUboSize) is independent
// of this and has its own guard.
inline constexpr std::size_t kMaxPushConstantBytes = 4096;

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
    R32F,         // P10 denoiser depth (storage image)
    RG16F,        // P10 denoiser motion vectors (storage image)
    RG32F,        // Two-channel 32-bit float storage image. No current
                  // texture-backed user -- SVGF luminance moments + variance
                  // live in storage *buffers* now (under Metal's 8-RW-texture
                  // compute cap), but the format is kept registered for any
                  // future caller that needs a 2x32f image.
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

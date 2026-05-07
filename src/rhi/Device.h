// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "CommandBuffer.h"
#include "Handles.h"
#include "Resources.h"
#include "Swapchain.h"
#include "Types.h"

#include <cstddef>
#include <memory>

namespace pt::rhi {

// Opaque platform handle passed to Device::Create.  On macOS this wraps an
// NSWindow* (or its content view); on Windows it'd hold an HWND.
struct NativeWindowHandle {
    void* opaque = nullptr;     // NSWindow* / HWND
    int   width  = 0;
    int   height = 0;
};

class Device {
public:
    virtual ~Device() = default;

    // Factory.  Each backend registers itself by linking its own
    // Create<Backend>Device shim that this dispatches to.
    static std::unique_ptr<Device> Create(BackendType type,
                                          const NativeWindowHandle& window);

    // Resource creation.
    virtual BufferHandle      CreateBuffer(const BufferDesc&)             = 0;
    virtual TextureHandle     CreateTexture(const TextureDesc&)           = 0;
    virtual PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) = 0;
    virtual AccelStructHandle CreateBLAS(const BLASDesc&)                 = 0;
    virtual AccelStructHandle CreateTLAS(const TLASDesc&)                 = 0;

    virtual void DestroyBuffer(BufferHandle)             = 0;
    virtual void DestroyTexture(TextureHandle)           = 0;
    virtual void DestroyPipeline(PipelineHandle)         = 0;
    virtual void DestroyAccelStruct(AccelStructHandle)   = 0;

    // CPU-side write into an Upload buffer.  The backend chooses the most
    // efficient mapping it can.
    virtual void WriteBuffer(BufferHandle, const void* src, std::size_t size,
                             std::size_t dst_offset = 0) = 0;

    // CPU-side write of an entire 2D texture (mip 0). `src_size` is the
    // total bytes of the source: width * height * bytes_per_pixel for
    // the texture's pixel format. Returns false if the texture isn't
    // CPU-writable (private storage) or sizes mismatch. Used for
    // one-shot uploads like env maps.
    virtual bool WriteTexture(TextureHandle, const void* /*src*/,
                              std::size_t /*src_size*/) {
        return false;
    }

    // Per-frame contract: BeginFrame returns the swapchain image to render
    // into, EndFrame submits and presents.
    virtual FrameContext BeginFrame() = 0;
    virtual void         EndFrame(CommandBuffer*) = 0;

    virtual CommandBuffer* AcquireCommandBuffer() = 0;
    virtual void           Submit(CommandBuffer*) = 0;
    virtual void           WaitIdle() = 0;

    // Swapchain control.
    virtual void Resize(int w, int h) = 0;

    // Read a texture back to CPU. The destination buffer must be at
    // least width * height * bytes_per_pixel(format). Returns true on
    // success. Implementations may stall the GPU (waitUntilCompleted)
    // since this is intended for screenshot / debug use only.
    // `out_w` / `out_h` get the texture's dimensions so the caller can
    // size the buffer correctly without a separate query.
    virtual bool ReadbackTexture(TextureHandle, void* /*dst*/, std::size_t /*dst_size*/,
                                 std::uint32_t* /*out_w*/, std::uint32_t* /*out_h*/) {
        return false;
    }

    // Capability + introspection.
    virtual BackendType  Type()             const = 0;
    virtual bool         SupportsHardwareRT() const = 0;
    virtual const char*  DeviceName()       const = 0;
    virtual std::size_t  CurrentAllocatedBytes() const = 0;

    // P10 denoiser hook. Backends without a denoiser implementation
    // (software / vulkan today) leave the default no-op. The Metal
    // backend implements it via MTLFXTemporalDenoisedScaler. Must be
    // called AFTER the path-tracer Submit so the input textures are
    // ready, and BEFORE EndFrame so it can encode into the same
    // command buffer flow.
    struct DenoiseDesc {
        TextureHandle color_in;       // RGBA16F linear (per-frame, not accumulated)
        TextureHandle depth_in;       // R32F clip-space depth (z/w in [0,1])
        TextureHandle motion_in;      // RG16F pixel-space (prev - curr)
        TextureHandle output;         // typically the swapchain image
        float jitter_x       = 0.0f;
        float jitter_y       = 0.0f;
        bool  reset_history  = false; // true on backend switch / scene reset
        // Required by MetalFX TemporalDenoisedScaler. Column-major 4x4
        // (16 floats each). Pass nullptr only if the backend doesn't need
        // them (currently: nothing -- both Metal and any future Vulkan
        // denoiser need motion math).
        const float* world_to_view = nullptr;
        const float* view_to_clip  = nullptr;
    };
    virtual bool SupportsDenoise() const { return false; }
    virtual void Denoise(const DenoiseDesc& /*d*/) {}
};

}  // namespace pt::rhi

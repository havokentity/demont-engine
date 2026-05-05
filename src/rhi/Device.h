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

    // Per-frame contract: BeginFrame returns the swapchain image to render
    // into, EndFrame submits and presents.
    virtual FrameContext BeginFrame() = 0;
    virtual void         EndFrame(CommandBuffer*) = 0;

    virtual CommandBuffer* AcquireCommandBuffer() = 0;
    virtual void           Submit(CommandBuffer*) = 0;
    virtual void           WaitIdle() = 0;

    // Swapchain control.
    virtual void Resize(int w, int h) = 0;

    // Capability + introspection.
    virtual BackendType  Type()             const = 0;
    virtual bool         SupportsHardwareRT() const = 0;
    virtual const char*  DeviceName()       const = 0;
    virtual std::size_t  CurrentAllocatedBytes() const = 0;
};

}  // namespace pt::rhi

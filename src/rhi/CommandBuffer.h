#pragma once

#include "Handles.h"
#include "Resources.h"

#include <cstdint>

namespace pt::rhi {

// Records GPU work.  Acquired from Device::AcquireCommandBuffer, returned
// via Device::Submit which gives ownership back to the device.  Backends
// pool/reuse these internally.
class CommandBuffer {
public:
    virtual ~CommandBuffer() = default;

    virtual void BindComputePipeline(PipelineHandle p) = 0;
    virtual void BindBuffer(std::uint32_t slot, BufferHandle b,
                            std::size_t offset = 0) = 0;
    virtual void BindStorageTexture(std::uint32_t slot, TextureHandle t) = 0;
    virtual void BindAccelStruct(std::uint32_t slot, AccelStructHandle a) = 0;

    // Push constants are layered atop the pipeline's reflected layout.  In
    // P2 we just copy bytes that the kernel reads as-is.
    virtual void PushConstants(const void* data, std::size_t size) = 0;

    virtual void Dispatch(std::uint32_t gx, std::uint32_t gy,
                          std::uint32_t gz) = 0;

    virtual void CopyBufferToTexture(BufferHandle src, TextureHandle dst) = 0;

    virtual void Barrier(const BarrierDesc& d) = 0;
};

}  // namespace pt::rhi

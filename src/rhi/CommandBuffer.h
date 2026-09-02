// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
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

    // Clear a storage / colour-attachment texture to a uniform RGBA
    // value. Used as a minimal "loading frame" path while the Vulkan
    // backend's async pipeline build is in flight: BeginFrame ->
    // ClearStorageTexture(swapchain) -> Submit -> EndFrame writes a
    // defined dark frame to the swapchain instead of leaving it as
    // an undefined-layout post-acquire state. Backends may issue
    // any necessary layout transitions internally.
    virtual void ClearStorageTexture(TextureHandle t, const float rgba[4]) = 0;

    virtual void Barrier(const BarrierDesc& d) = 0;

    // --- Per-pass GPU timestamps (#320) ----------------------------------
    // Open a GPU-timed pass. Closes the previously-open timed pass (if any)
    // and marks `slot` as the region the backend samples the GPU clock
    // across. The engine assigns slots 0..N-1 in record order and holds the
    // matching human-readable labels; `slot` must be < the capacity passed
    // to Device::SetGpuTimestampsEnabled.
    //
    // Backend mapping:
    //   * Metal (Apple silicon supports ONLY stage-boundary counter
    //     sampling) -- each timed pass gets its own MTLComputeCommandEncoder
    //     created from an MTLComputePassDescriptor that samples the
    //     timestamp counter set at the encoder's start and end boundaries.
    //     Encoder creation is deferred to the pass's first Dispatch, so a
    //     mark that is immediately followed by a FlushEncoder (the denoiser
    //     path) records no empty encoder.
    //   * Vulkan -- vkCmdWriteTimestamp(BOTTOM_OF_PIPE) into a
    //     VK_QUERY_TYPE_TIMESTAMP pool at the pass's start, and again at its
    //     end when the next BeginGpuPass / EndGpuPass closes it.
    //
    // No-op unless the device has timestamps enabled for the current frame
    // (Device::SetGpuTimestampsEnabled). Default no-op so backends without
    // a timestamp implementation (software) ignore it.
    virtual void BeginGpuPass(std::uint32_t /*slot*/) {}

    // Close the final timed pass of the frame. Called once, right before
    // Device::Submit, so the last pass's end boundary is sampled. No-op
    // when timestamps are disabled or no pass is open.
    virtual void EndGpuPass() {}
};

}  // namespace pt::rhi

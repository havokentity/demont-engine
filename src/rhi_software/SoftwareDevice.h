#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// Forward decls for metal-cpp pointer types.  Real defs come from
// <Metal/Metal.hpp> in the .cpp.
namespace MTL {
class Device;
class CommandQueue;
}
namespace CA { class MetalLayer; class MetalDrawable; }

namespace pt::rhi::sw {

class SoftwarePipeline {
public:
    explicit SoftwarePipeline(std::string name);
    void SetClearColor(float r, float g, float b, float a);
    void GetClearColor(float& r, float& g, float& b, float& a) const;
    const std::string& Name() const { return name_; }
private:
    std::string name_;
    float clear_[4] { 0, 0, 0, 1 };
};

class SoftwareDevice;

class SoftwareCommandBuffer : public CommandBuffer {
public:
    explicit SoftwareCommandBuffer(SoftwareDevice* d) : device_(d) {}

    void BindComputePipeline(PipelineHandle p) override;
    void BindBuffer(std::uint32_t, BufferHandle, std::size_t) override {}
    void BindStorageTexture(std::uint32_t, TextureHandle) override {}
    void BindAccelStruct(std::uint32_t, AccelStructHandle) override {}
    void PushConstants(const void* data, std::size_t size) override;
    void Dispatch(std::uint32_t, std::uint32_t, std::uint32_t) override;
    void CopyBufferToTexture(BufferHandle, TextureHandle) override {}
    void Barrier(const BarrierDesc&) override {}

    PipelineHandle bound_pipeline{0};
    std::uint8_t   push_constants_buf[128] {};
    std::size_t    push_constants_size = 0;

private:
    SoftwareDevice* device_ = nullptr;
};

class SoftwareDevice : public Device {
public:
    explicit SoftwareDevice(const NativeWindowHandle& window);
    ~SoftwareDevice() override;

    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override { return {0}; }
    AccelStructHandle CreateTLAS(const TLASDesc&) override { return {0}; }

    void DestroyBuffer(BufferHandle h) override;
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle h) override;
    void DestroyAccelStruct(AccelStructHandle) override {}

    void WriteBuffer(BufferHandle, const void*, std::size_t,
                     std::size_t) override {}

    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override {}
    void           Resize(int w, int h) override;

    BackendType  Type()             const override { return BackendType::Software; }
    bool         SupportsHardwareRT() const override { return false; }
    const char*  DeviceName()       const override { return "DeMonT Software (CPU)"; }
    std::size_t  CurrentAllocatedBytes() const override;

    void StashClear(float r, float g, float b, float a);
    SoftwarePipeline* GetPipeline(PipelineHandle h);

private:
    void* ns_window_ = nullptr;
    int   width_  = 0;
    int   height_ = 0;
    std::uint32_t frame_index_ = 0;

    float pending_clear_[4] { 0.18f, 0.05f, 0.28f, 1.0f };

    // Metal is used only for the present blit -- the rendering work itself
    // is meant to be CPU-side (P5+ will fill a CPU framebuffer here and
    // upload to a Metal texture). For P3 the kernel is just "clear", so
    // we use a render pass with loadAction=Clear, no upload needed yet.
    MTL::Device*       mtl_device_ = nullptr;
    MTL::CommandQueue* mtl_queue_  = nullptr;
    CA::MetalLayer*    mtl_layer_  = nullptr;

    std::unique_ptr<SoftwareCommandBuffer> cmd_buf_;

    std::mutex resource_mutex_;
    std::uint64_t next_id_ = 1;
    std::unordered_map<std::uint64_t, std::unique_ptr<SoftwarePipeline>> pipelines_;

    std::atomic<std::size_t> bytes_held_{0};
};

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w);
}

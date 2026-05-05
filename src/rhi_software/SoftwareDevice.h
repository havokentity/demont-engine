#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

struct GLFWwindow;

namespace pt::rhi::sw {

// Per-pipeline kernel.  P2 supports just "clear" -- writes a 4-float
// push-constant colour into the rendering state.  P5+ will extend this
// to host the path-tracer integrator.
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
    std::uint8_t   push_constants_buf[64] {};
    std::size_t    push_constants_size = 0;

private:
    SoftwareDevice* device_ = nullptr;
};

class SoftwareDevice : public Device {
public:
    explicit SoftwareDevice(GLFWwindow* window);
    ~SoftwareDevice() override;

    // ---- Resources -------------------------------------------------------
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

    // ---- Frame -----------------------------------------------------------
    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override {}
    void           Resize(int w, int h) override;

    // ---- Introspection ---------------------------------------------------
    BackendType  Type()             const override { return BackendType::Software; }
    bool         SupportsHardwareRT() const override { return false; }
    const char*  DeviceName()       const override { return "PathTracer Software (CPU)"; }
    std::size_t  CurrentAllocatedBytes() const override;

    // Internal: command buffer feeds back the clear color for the next
    // present.
    void StashClear(float r, float g, float b, float a);

    SoftwarePipeline* GetPipeline(PipelineHandle h);

private:
    GLFWwindow* window_ = nullptr;
    int         width_  = 0;
    int         height_ = 0;
    std::uint32_t frame_index_ = 0;

    float pending_clear_[4] { 0.05f, 0.05f, 0.06f, 1.0f };

    std::unique_ptr<SoftwareCommandBuffer> cmd_buf_;

    // Resource tables.  Trivial id allocator.
    std::mutex resource_mutex_;
    std::uint64_t next_id_ = 1;
    std::unordered_map<std::uint64_t, std::unique_ptr<SoftwarePipeline>> pipelines_;

    // For CurrentAllocatedBytes -- sum of CPU buffers/textures we own.
    std::atomic<std::size_t> bytes_held_{0};
};

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w);
}  // namespace pt::rhi

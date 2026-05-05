#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

// Forward declares for metal-cpp pointer types so this header doesn't drag
// the full Metal SDK into every consumer. The real impl uses MTL:: etc.
namespace MTL {
class Device;
class CommandQueue;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
class Library;
class Texture;
class Buffer;
class AccelerationStructure;
}
namespace CA {
class MetalLayer;
class MetalDrawable;
}
namespace NS {
class AutoreleasePool;
}

namespace pt::rhi::mtl {

class MetalDevice;

class MetalCommandBuffer : public CommandBuffer {
public:
    explicit MetalCommandBuffer(MetalDevice* d) : device_(d) {}

    void BindComputePipeline(PipelineHandle p) override;
    void BindBuffer(std::uint32_t slot, BufferHandle b, std::size_t off) override;
    void BindStorageTexture(std::uint32_t slot, TextureHandle t) override;
    void BindAccelStruct(std::uint32_t slot, AccelStructHandle a) override;
    void PushConstants(const void* data, std::size_t size) override;
    void Dispatch(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) override;
    void CopyBufferToTexture(BufferHandle, TextureHandle) override {}
    void Barrier(const BarrierDesc&) override {}

    void Reset(MTL::CommandBuffer* cb);
    MTL::CommandBuffer* RawCmdBuf() const { return mtl_cb_; }

private:
    void EnsureEncoder();
    void EndEncoderIfActive();

    MetalDevice*               device_      = nullptr;
    MTL::CommandBuffer*        mtl_cb_      = nullptr;
    MTL::ComputeCommandEncoder* encoder_    = nullptr;

    PipelineHandle             bound_pso_{0};
    TextureHandle              bound_tex_[8] {};
    BufferHandle               bound_buf_[8] {};
    std::size_t                bound_buf_off_[8] {};
    AccelStructHandle          bound_accel_[4] {};

    std::uint8_t  push_buf_[128] {};
    std::size_t   push_size_ = 0;
};

class MetalDevice : public Device {
public:
    explicit MetalDevice(const NativeWindowHandle& window);
    ~MetalDevice() override;

    // ---- Resources ------------------------------------------------------
    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override;
    AccelStructHandle CreateTLAS(const TLASDesc&) override;

    void DestroyBuffer(BufferHandle h) override;
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle h) override;
    void DestroyAccelStruct(AccelStructHandle h) override;

    void WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                     std::size_t dst_offset = 0) override;

    // ---- Frame ----------------------------------------------------------
    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override;
    void           Resize(int w, int h) override;

    // ---- Introspection --------------------------------------------------
    BackendType  Type()             const override { return BackendType::Metal; }
    bool         SupportsHardwareRT() const override { return true; }
    const char*  DeviceName()       const override { return device_name_.c_str(); }
    std::size_t  CurrentAllocatedBytes() const override;

    // ---- Internal lookup ------------------------------------------------
    MTL::ComputePipelineState* LookupPipeline(PipelineHandle h);
    MTL::Texture*              LookupTexture(TextureHandle h);
    MTL::Buffer*               LookupBuffer(BufferHandle h);
    MTL::AccelerationStructure* LookupAccelStruct(AccelStructHandle h);

    // Mark all known acceleration structures as used by the current
    // encoder. Required because a TLAS internally references its BLASes
    // and Metal needs every one declared as a dependency on the encoder
    // that ray-queries the TLAS.
    void UseAllAccelStructs(MTL::ComputeCommandEncoder* enc);

    static constexpr std::uint64_t kSwapchainTextureId = 1;

private:
    void* ns_window_ = nullptr;     // NSWindow*
    int   width_  = 0;
    int   height_ = 0;
    std::uint32_t frame_index_ = 0;
    std::string device_name_ = "Metal Device";

    MTL::Device*        device_  = nullptr;
    MTL::CommandQueue*  queue_   = nullptr;
    CA::MetalLayer*     layer_   = nullptr;
    CA::MetalDrawable*  current_drawable_ = nullptr;
    NS::AutoreleasePool* frame_pool_ = nullptr;

    std::mutex                                         resource_mutex_;
    std::uint64_t                                      next_id_ = kSwapchainTextureId + 1;
    std::unordered_map<std::uint64_t, MTL::ComputePipelineState*>  pipelines_;
    std::unordered_map<std::uint64_t, MTL::Texture*>               textures_;
    std::unordered_map<std::uint64_t, MTL::Buffer*>                buffers_;
    std::unordered_map<std::uint64_t, MTL::AccelerationStructure*> accels_;
    // Built-in pipelines indexed by Slang kernel name.  P3+ shaders are
    // pre-compiled at device construction; CreateComputePipeline looks up
    // by name.
    std::unordered_map<std::string, std::uint64_t>     named_pipelines_;

    std::unique_ptr<MetalCommandBuffer> cmd_;
};

}  // namespace pt::rhi::mtl

namespace pt::rhi {
std::unique_ptr<Device> CreateMetalDevice(const NativeWindowHandle&);
}

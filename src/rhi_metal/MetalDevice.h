// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
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
    // No-op on Metal: pipelines build synchronously and quickly, so
    // the engine's loading-frame path (which is the only call site)
    // never fires on this backend.
    void ClearStorageTexture(TextureHandle, const float[4]) override {}
    void Barrier(const BarrierDesc&) override {}

    void Reset(MTL::CommandBuffer* cb);
    MTL::CommandBuffer* RawCmdBuf() const { return mtl_cb_; }

    // Forces the currently-open compute encoder closed. Required before
    // any non-RHI code (e.g. MetalFX) wants to attach its own encoder
    // to the same MTLCommandBuffer.
    void FlushEncoder();

private:
    void EnsureEncoder();
    void EndEncoderIfActive();

    MetalDevice*               device_      = nullptr;
    MTL::CommandBuffer*        mtl_cb_      = nullptr;
    MTL::ComputeCommandEncoder* encoder_    = nullptr;

    PipelineHandle             bound_pso_{0};
    // Texture slot map: 11 slots covering every kernel's max engine
    // slot. PathTrace fills slots 0..10:
    //   0  output / swapchain
    //   1  accum_hdr
    //   2  denoise_color
    //   3  depth_tex (denoiser-only)
    //   4  motion_tex (denoiser-only)
    //   5  env_map (HDRI)
    //   6  star_map (BSC)
    //   7  moon_map
    //   8  normal_tex (SVGF / NRD / OptiX-AOV)
    //   9  albedo_tex (OptiX-AOV only)
    //   10 cloud_trans_tex (issue #46 follow-up: R32F per-pixel cloud
    //      transmittance the path tracer writes and StarsComposite
    //      reads). On Metal slots above the kernel's actual texture
    //      count are silently dropped; on Vulkan the slot table in
    //      VulkanDevice.cpp maps them to vk::binding numbers.
    TextureHandle              bound_tex_[11] {};
    // 12 buffer slots. Slots 0..7 are the original engine layout
    // (mesh_positions / mesh_indices, primitives, marginal /
    // conditional CDFs, exposure_state, analytic-prim bvh_nodes).
    // Slots 8/9 were added in the PR #106 follow-up for the triangle
    // BVH (tri_bvh_nodes / tri_bvh_permuted_ids -- replaces the SW
    // Möller-Trumbore linear-scan path with a stack-based BVH walk).
    // Slot 10 was added by SDF Phase 1 (#97) for the SDF cluster
    // storage buffer (MSL slot 10 in the path tracer's dynamic
    // buffer-slot layout; moved from slot 8 to make room for the
    // triangle BVH). Slot 11 was added by issue #115 for the SIGMA
    // shadow visibility buffer (R32F per pixel) -- declared as a
    // storage buffer rather than an RWTexture2D because PathTrace
    // already sits exactly at Apple Silicon's 8-RW-texture compute
    // cap; storage buffers escape that quota the same way SVGF's
    // variance / moments buffers do. Keep in sync with the engine's
    // BindBuffer(11,...) call site.
    BufferHandle               bound_buf_[12] {};
    std::size_t                bound_buf_off_[12] {};
    AccelStructHandle          bound_accel_[4] {};

    // Push-constant buffer. Sized to fit the unified PathTrace push
    // plus growth headroom for upcoming features (DOF, MIS PDFs,
    // bloom params, ...). PathTrace push is currently 336 bytes; the
    // 512-byte buffer leaves room without forcing another resize.
    // PushConstants() silently truncates beyond this size, so when
    // adding push fields make sure the buffer is still big enough --
    // a too-small buffer manifests as the last fields being all-zero
    // at runtime (e.g. DOF appearing to do nothing). 1024B fits today's
    // TonePush (576B with the physical-flare ghost array) and PtPush
    // (352B); Metal's setBytes accepts up to 4KB so bump as needed.
    std::uint8_t  push_buf_[1024] {};
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
    bool WriteTexture(TextureHandle h, const void* src, std::size_t src_size) override;

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

    // ---- P10 denoiser ---------------------------------------------------
    bool SupportsDenoise() const override { return true; }
    void Denoise(const DenoiseDesc& d) override;

    bool ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                         std::uint32_t* out_w, std::uint32_t* out_h) override;
    bool ReadbackBuffer (BufferHandle  h, void* dst, std::size_t bytes) override;

    // ---- Internal lookup ------------------------------------------------
    MTL::ComputePipelineState* LookupPipeline(PipelineHandle h);
    MTL::Texture*              LookupTexture(TextureHandle h);
    MTL::Buffer*               LookupBuffer(BufferHandle h);
    MTL::AccelerationStructure* LookupAccelStruct(AccelStructHandle h);

    // Used by MetalSvgfDenoiser to build its own pipelines + textures.
    MTL::Device* RawDevice() const { return device_; }

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

    // MetalFX TemporalDenoisedScaler. Allocated lazily on first
    // Denoise() call; recreated when the swapchain size changes.
    void*         metalfx_scaler_   = nullptr;
    std::uint32_t metalfx_width_    = 0;
    std::uint32_t metalfx_height_   = 0;

    // In-house SVGF (svgf_basic / svgf_atrous). Same Slang sources as
    // the Vulkan backend, cross-compiled to MSL. Allocated lazily on
    // first SVGF-mode Denoise() call.
    std::unique_ptr<class MetalSvgfDenoiser> svgf_denoiser_;

    // SVGF -> MetalFX chain (Kind::SvgfMetalFx): intermediate RGBA16F
    // texture SVGF writes to before MetalFX reads it as the "color"
    // input. Lazily allocated on first chain-mode Denoise() and
    // resized when the swapchain dims change.
    MTL::Texture* svgf_metalfx_intermediate_   = nullptr;
    std::uint32_t svgf_metalfx_intermediate_w_ = 0;
    std::uint32_t svgf_metalfx_intermediate_h_ = 0;
};

}  // namespace pt::rhi::mtl

namespace pt::rhi {
std::unique_ptr<Device> CreateMetalDevice(const NativeWindowHandle&);
}

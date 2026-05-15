// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pt::rhi::vk { class VulkanNrdDenoiser; }
#if defined(PT_ENABLE_OPTIX)
namespace pt::rhi::vk { class VulkanOptixDenoiser; }
#endif

struct GLFWwindow;

namespace pt::rhi::vk {

class VulkanDevice;

class VulkanCommandBuffer : public CommandBuffer {
public:
    explicit VulkanCommandBuffer(VulkanDevice* d) : device_(d) {}

    void BindComputePipeline(PipelineHandle p) override;
    void BindBuffer(std::uint32_t slot, BufferHandle b, std::size_t off) override;
    void BindStorageTexture(std::uint32_t slot, TextureHandle t) override;
    void BindAccelStruct(std::uint32_t slot, AccelStructHandle a) override;
    void PushConstants(const void* data, std::size_t size) override;
    void Dispatch(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) override;
    void CopyBufferToTexture(BufferHandle, TextureHandle) override {}
    void ClearStorageTexture(TextureHandle t, const float rgba[4]) override;
    void Barrier(const BarrierDesc& d) override;

    void Reset(VkCommandBuffer cb);
    VkCommandBuffer Raw() const { return cb_; }

private:
    VulkanDevice*  device_ = nullptr;
    VkCommandBuffer cb_     = VK_NULL_HANDLE;
    PipelineHandle bound_pipeline_{0};
    // 12 texture slots gives engine slot 8 (normal_tex for the SVGF/NRD
    // denoiser, mapped to vk::binding 16) plus 3 spare. Metal stays at
    // 8 because MetalFX doesn't take a normal input and the path-tracer
    // shader gates the normal write on PT_TARGET_SPIRV.
    TextureHandle  bound_tex_[12] {};
    BufferHandle   bound_buf_[8] {};
    std::size_t    bound_buf_off_[8] {};
    AccelStructHandle bound_accel_[4] {};
    // Push-constant staging. Sized to fit the full PtPush (~448B today)
    // plus growth headroom so the engine can keep treating push as one
    // contiguous blob across backends. On SPIR-V, Dispatch byte-splits:
    // first kPushConstantSize bytes -> vkCmdPushConstants (within hw
    // limit), remainder -> per-frame Frame UBO at vk::binding(14, 0).
    std::uint8_t   push_buf_[1024] {};
    std::size_t    push_size_ = 0;
};

class VulkanDevice : public Device {
public:
    explicit VulkanDevice(const NativeWindowHandle& w);
    ~VulkanDevice() override;

    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override;
    AccelStructHandle CreateTLAS(const TLASDesc&) override;

    void DestroyBuffer(BufferHandle h) override;
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle) override {}
    void DestroyAccelStruct(AccelStructHandle h) override;

    // Batched-destroy fast paths. The public Destroy* above each call
    // vkDeviceWaitIdle() internally so a single destroy is always safe.
    // Callers that tear down N resources back-to-back (denoiser scratch
    // reallocation, swapchain rebuild, shutdown) pay an N-way idle stall
    // otherwise. The NoWait variants skip the internal wait; the caller
    // is responsible for issuing a single WaitIdle() before the batch.
    // Strictly Vulkan-specific (not on the abstract Device interface).
    void DestroyBufferNoWait(BufferHandle h);
    void DestroyTextureNoWait(TextureHandle h);

    void WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                     std::size_t dst_offset) override;
    bool WriteTexture(TextureHandle h, const void* src, std::size_t src_size) override;
    bool ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                         std::uint32_t* out_w, std::uint32_t* out_h) override;
    bool ReadbackBuffer (BufferHandle  h, void* dst, std::size_t bytes) override;

    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override;
    void           Resize(int w, int h) override;

    BackendType  Type()             const override { return BackendType::Vulkan; }
    bool         SupportsHardwareRT() const override { return rt_supported_; }
    const char*  DeviceName()       const override { return device_name_.c_str(); }
    std::size_t  CurrentAllocatedBytes() const override;

    // SVGF/NRD denoiser. The denoiser pipelines + scratch textures are
    // NOT built by the async worker -- they're constructed lazily on
    // the first Denoise() call (a few ms one-time hitch on the first
    // frame after r_denoiser is toggled to a Vulkan kind). Until then,
    // SupportsDenoise gates on the main async pipeline build (so the
    // engine doesn't flag denoiser as available before the path-tracer
    // pipeline is even ready). After lazy init, the cached `ready_`
    // flag short-circuits this check.
    bool SupportsDenoise() const override;
    void Denoise(const DenoiseDesc& d) override;

    // Internal accessors used by the command buffer.
    VkDevice         RawDevice()     const { return device_; }
    VkPhysicalDevice RawPhysicalDevice() const { return phys_device_; }
    VkQueue          RawGraphicsQueue() const { return graphics_queue_; }
    std::uint32_t    GraphicsQueueFamily() const { return graphics_qfi_; }
    VkPipeline       LookupPipeline(PipelineHandle h);
    VkPipelineLayout LookupPipelineLayout(PipelineHandle h);
    VkImageView      CurrentSwapchainImageView() const;

#if defined(PT_ENABLE_OPTIX)
    // Run JUST the SVGF DenoiseFinalize compute pass (ACES + sRGB
    // tonemap to swapchain), exposed for the OptiX denoiser path so
    // it doesn't have to roll its own tonemap. Lazy-creates and
    // initialises the SVGF VulkanNrdDenoiser (only the finalize
    // pipeline is needed; the SVGF history textures stay
    // unallocated until/unless the user actually picks svgf_*).
    // No-op if PT_ENABLE_OPTIX is off; safe to call on any frame.
    //
    // Caller responsible for both image layouts being GENERAL before
    // the call. See VulkanNrdDenoiser::EncodeFinalizeOnly for the
    // detailed contract.
    void EncodeDenoiseFinalize(VkCommandBuffer cb,
                               VkImageView    color_in_view,
                               VkImageView    final_output_view,
                               VkBuffer       exposure_state_buf,
                               std::uint32_t  width,
                               std::uint32_t  height,
                               bool           hdr_pipeline);

    // CUDA-Vulkan interop hook for VulkanOptixDenoiser.
    //
    // VulkanOptixDenoiser::Encode records the input copy into the
    // engine's main command buffer (so it runs as part of the engine's
    // submit), but CUDA needs to know when that copy is done before
    // it can read the OptiX denoiser input. Vulkan can only signal
    // semaphores at submit time, so this method lets the OptiX path
    // request that the next call to Submit() additionally signals
    // a timeline semaphore at the given value -- letting CUDA's
    // cudaWaitExternalSemaphoresAsync gate on it.
    //
    // Cleared automatically after one Submit. Call once per frame
    // when the OptiX path is active. Setting sem == VK_NULL_HANDLE
    // disables (and is the post-Submit reset state).
    void RequestExtraSubmitSignal(VkSemaphore sem, std::uint64_t timeline_value);
#endif

    static constexpr std::uint64_t kSwapchainTextureId = 1;

    // ---- Push-constant / descriptor-binding constants -------------------
    // CROSS-FILE INVARIANTS -- THREE FILES MUST AGREE:
    //   (a) src/engine/Engine.cpp PtPush struct:
    //       - first kPushSplitOffset bytes hold the small Metal-Push
    //         tail (frame_index, reset_accum, max_bounces, ...).
    //       - bytes [kPushSplitOffset..sizeof(PtPush)] hold the
    //         spilled tail (matrices, sun/moon, clouds, hdri lights).
    //       Engine.cpp has a static_assert(sizeof(PtPush) == ...) at the
    //       dispatch site (search "static_assert(sizeof(PtPush)")
    //       which fires at compile-time if anything drifts.
    //   (b) shaders/PathTrace.slang SPIR-V path:
    //       - cbuffer Push  at vk::push_constant   = first kPushSplit
    //         Offset bytes.
    //       - cbuffer Frame at vk::binding(kFrameUboBinding, 0) =
    //         the spilled tail layout.  The Frame { ... } block in
    //         PathTrace.slang must match Engine.cpp's PtPush tail
    //         FIELD-BY-FIELD AND BYTE-BY-BYTE; std140 uses 16-byte
    //         alignment so all fields are float4-padded.
    //   (c) This file: the kPush*/kFrame* constants below.
    //
    // If you reorder/resize PtPush, you MUST update the matching Frame
    // cbuffer in PathTrace.slang AND verify the static_assert in
    // Engine.cpp still fires green; the runtime symptom of a desync
    // is rendering corruption, not a build error.  A future cleanup
    // (logged in FOLLOW_UPS.md) is to drive these from a single
    // shared header or Slang reflection.
    static constexpr std::uint32_t kPushSplitOffset = 112;
    static constexpr std::uint32_t kFrameUboBinding = 14;
    // Sized to fit the PtPush tail (sizeof(PtPush) - kPushSplitOffset) with
    // room to grow. The static_assert at Engine.cpp's dispatch site checks
    // sizeof(PtPush) against the field-by-field sum; the runtime guard in
    // VulkanCommandBuffer::Dispatch LOG_ERRORs once if the tail ever exceeds
    // this size, since silently clamping the memcpy truncates the GPU-side
    // view of trailing fields (write_hdr_aux, write_normal_gbuffer etc.) and
    // turns rendering corruption into a silent visual bug. PtPush tail today
    // is 608 B (HDRI multi-light + flags); bump this constant whenever the
    // static_assert fires.
    static constexpr std::size_t   kFrameUboSize    = 1024;

private:
    void DestroyDevice();
    bool RecreateSwapchain();

    GLFWwindow* glfw_window_ = nullptr;
    int         width_       = 0;
    int         height_      = 0;
    std::string device_name_ = "Vulkan Device";
    bool        rt_supported_ = false;
    std::uint32_t frame_index_ = 0;
    std::uint32_t max_push_constant_size_ = 128;

    // Core handles
    VkInstance              instance_      = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_msgr_   = VK_NULL_HANDLE;
    VkSurfaceKHR            surface_       = VK_NULL_HANDLE;
    VkPhysicalDevice        phys_device_   = VK_NULL_HANDLE;
    VkDevice                device_        = VK_NULL_HANDLE;
    std::uint32_t           graphics_qfi_  = 0;
    VkQueue                 graphics_queue_ = VK_NULL_HANDLE;

    // Accel-struct & buffer-device-address extension function pointers --
    // resolved post-vkCreateDevice via vkGetDeviceProcAddr. Null when the
    // corresponding extension wasn't enabled (e.g. driver too old; we
    // gracefully degrade to "no hardware RT" in that case).
    PFN_vkGetAccelerationStructureBuildSizesKHR    pfn_GetAccelStructBuildSizes_ = nullptr;
    PFN_vkCreateAccelerationStructureKHR           pfn_CreateAccelStruct_        = nullptr;
    PFN_vkDestroyAccelerationStructureKHR          pfn_DestroyAccelStruct_       = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR        pfn_CmdBuildAccelStructs_     = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_GetAccelStructAddr_       = nullptr;
    PFN_vkGetBufferDeviceAddressKHR                pfn_GetBufferDeviceAddr_      = nullptr;

    // Swapchain
    VkSwapchainKHR              swapchain_     = VK_NULL_HANDLE;
    VkFormat                    swap_format_   = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D                  swap_extent_   { 0, 0 };
    std::vector<VkImage>        swap_images_;
    std::vector<VkImageView>    swap_views_;

    // Per-frame in flight: 2-deep
    static constexpr int kFramesInFlight = 2;
    int                       current_frame_ = 0;
    std::uint32_t             current_swap_index_ = 0;
    VkCommandPool             cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer           cmds_[kFramesInFlight] {};
    VkSemaphore               sem_image_avail_[kFramesInFlight] {};
    VkFence                   fence_in_flight_[kFramesInFlight] {};
    // Render-finished semaphore is keyed by SWAPCHAIN IMAGE INDEX, not
    // frame-in-flight, because the semaphore stays in use until the
    // present completes; if we keyed by frame we'd risk reusing a
    // semaphore that's still pending on the previous present of the
    // same image.
    std::vector<VkSemaphore>  sem_render_done_;

    std::unique_ptr<VulkanCommandBuffer> wrapped_cb_;

    // Pipelines
    struct PipelineEntry {
        VkPipeline pipeline;
    };
    std::mutex resource_mutex_;
    std::uint64_t next_id_ = kSwapchainTextureId + 1;
    std::unordered_map<std::uint64_t, PipelineEntry>     pipelines_;
    std::unordered_map<std::string, std::uint64_t>       named_pipelines_;

    // Shared layouts -- created once at construction. The layout has
    // 19 bindings matching the maximally-expanded shader: the smaller
    // shaders (Tonemap, BloomDown/Up) reference only a subset; Vulkan
    // allows pipelines whose shader doesn't use a binding to be created
    // against the larger layout. nullDescriptor lets the unused slots
    // bind VK_NULL_HANDLE without validation noise.
    VkDescriptorSetLayout shared_dset_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      shared_pipe_layout_ = VK_NULL_HANDLE;

    // Persistent VkPipelineCache. Loaded from
    // %LOCALAPPDATA%/demont/pipeline.cache (or $XDG_CACHE_HOME/demont
    // on POSIX) at device init, and the cache blob is rewritten to
    // disk in the destructor. The driver compiles SPIR-V -> native
    // bytecode at vkCreateComputePipelines time; first launch pays
    // that cost (PathTrace is the long pole, ~1-3 s), but subsequent
    // launches see near-zero pipeline-creation latency because the
    // driver finds a hit in this cache. Cache header is version-tagged
    // by the driver, so a mismatched (stale) blob is silently rejected
    // and rebuilt -- no need for engine-side validation.
    VkPipelineCache       pipeline_cache_     = VK_NULL_HANDLE;
    void LoadPipelineCache();
    void SavePipelineCache();
    static std::string PipelineCachePath();

    // Async pipeline build. The path-tracer pipeline can take
    // 1-3s on a cold pipeline cache; running that on the main thread
    // freezes the window during init.  Move it to a worker that runs
    // alongside the main thread's swapchain setup and first frames.
    // While the worker is in flight:
    //   - named_pipelines_ is empty for any kernel still being built,
    //     so CreateComputePipeline-by-name returns id=0.
    //   - The engine treats id=0 as "no-op dispatch" so RenderFrame
    //     skips the path-trace work cleanly each frame.
    //   - The engine re-resolves cached pipeline ids each frame via
    //     EnsurePipelineHandles() until all return non-zero.
    // Vulkan permits vkCreateComputePipelines in parallel with queue
    // submission on the same device; pipelines_/named_pipelines_
    // mutations are already serialised through resource_mutex_, which
    // LookupPipeline + CreateComputePipeline-by-name also take.
    std::thread           pipeline_build_thread_;
    std::atomic<bool>     pipelines_ready_{false};

    // SVGF/NRD denoiser. Pointer rather than embedded so the heavy
    // VulkanDenoiser.h doesn't bleed into every translation unit that
    // pulls in this header. Allocated lazily by Denoise() the first
    // time it's called with a non-zero output texture; freed in
    // DestroyDevice() before any VkPipeline / VkDescriptorPool teardown.
    std::unique_ptr<VulkanNrdDenoiser> denoiser_;

#if defined(PT_ENABLE_OPTIX)
    // OptiX denoiser. Sibling to denoiser_ above, gated by build-time
    // PT_ENABLE_OPTIX. Allocated lazily by Denoise() on the first call
    // with d.kind == OptixHdr / OptixHdrAov. The two never run on the
    // same frame (cvar pick is exclusive) and can coexist as members
    // because each owns its own scratch resources -- only the active
    // one consumes GPU memory after Init().
    std::unique_ptr<VulkanOptixDenoiser> optix_denoiser_;

    // Pending extra signal semaphore for the next Submit. See
    // RequestExtraSubmitSignal(). Reset to (VK_NULL_HANDLE, 0) after
    // each Submit. Only the OptiX path uses this today.
    VkSemaphore   extra_submit_signal_sem_   = VK_NULL_HANDLE;
    std::uint64_t extra_submit_signal_value_ = 0;
#endif

    VkDescriptorPool dpool_ = VK_NULL_HANDLE;
    VkDescriptorSet  dsets_[kFramesInFlight] {};

    struct ImageEntry {
        VkImage        image  = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView    view   = VK_NULL_HANDLE;
        VkFormat       format = VK_FORMAT_UNDEFINED;
        VkExtent2D     extent { 0, 0 };
    };
    std::unordered_map<std::uint64_t, ImageEntry> images_;

    struct BufferEntry {
        VkBuffer        buffer         = VK_NULL_HANDLE;
        VkDeviceMemory  memory         = VK_NULL_HANDLE;
        VkDeviceSize    size           = 0;
        VkDeviceAddress device_address = 0;     // 0 if buffer-device-address not enabled
        void*           mapped         = nullptr; // non-null iff host-visible & persistent
    };
    std::unordered_map<std::uint64_t, BufferEntry> buffers_;

    struct AccelEntry {
        VkAccelerationStructureKHR accel          = VK_NULL_HANDLE;
        VkBuffer                   buffer         = VK_NULL_HANDLE;
        VkDeviceMemory             memory         = VK_NULL_HANDLE;
        VkDeviceAddress            device_address = 0;
    };
    std::unordered_map<std::uint64_t, AccelEntry> accels_;

    // Per-frame UBO holding the spilled tail of the engine PtPush.
    // Persistently host-visible so VulkanCommandBuffer::Dispatch can
    // memcpy on each dispatch without staging.
    BufferEntry frame_ubos_[kFramesInFlight] {};

    // ReadbackTexture is implemented as a one-shot synchronous path
    // (alloc staging buffer + cmd + fence, submit copy, wait, memcpy,
    // free). The previous async-slot-ring infrastructure was built
    // around a per-8-frames CPU autoexpose readback that's been
    // retired -- auto-expose now lives entirely on the GPU (see
    // shaders/AutoExposure.slang + exposure_state buffer). The only
    // remaining caller is the `screenshot` console command, which is
    // user-triggered and low-frequency, so a queue stall during
    // readback is fine. When async per-frame readback returns (likely
    // for GPU-physics event queues, not texture data), it'll land as
    // a separate ReadbackBuffer API designed for that use case.

    // Helpers (impl in .cpp)
    bool CreateBufferImpl(VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags props,
                          BufferEntry& out,
                          bool persistent_map);
    void DestroyBufferImpl(BufferEntry& e);
    std::uint32_t FindMemoryType(std::uint32_t bits, VkMemoryPropertyFlags props) const;
    bool BuildAccelerationStructure(VkAccelerationStructureBuildGeometryInfoKHR& build_info,
                                    const VkAccelerationStructureBuildRangeInfoKHR* range,
                                    AccelEntry& entry,
                                    VkAccelerationStructureTypeKHR type,
                                    VkDeviceSize as_size,
                                    VkDeviceSize scratch_size);

public:
    VkImageView         LookupImageView(TextureHandle h);
    VkImage             LookupImage(TextureHandle h);
    VkExtent2D          LookupImageExtent(TextureHandle h);
    VkBuffer            LookupBuffer(BufferHandle h);
    VkAccelerationStructureKHR LookupAccel(AccelStructHandle h);
    VkDescriptorSet     CurrentDescriptorSet() const { return dsets_[current_frame_]; }
    VkPipelineLayout    SharedPipelineLayout() const { return shared_pipe_layout_; }
    const BufferEntry&  CurrentFrameUbo() const { return frame_ubos_[current_frame_]; }
    std::uint32_t       MaxPushConstantSize() const { return max_push_constant_size_; }
};

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle&);
}

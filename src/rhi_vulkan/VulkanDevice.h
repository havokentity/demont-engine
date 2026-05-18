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
    // 14 texture slots: slots 0..7 are output / accum / denoise_color /
    // depth / motion / env_map / star_map / moon_map; slot 8 is
    // normal_tex (SVGF/NRD + OptiX-AOV + MetalFX), slot 9 is albedo_tex
    // (OptiX-AOV + MetalFX), slot 10 is cloud_trans_tex (issue #46
    // follow-up: R32F per-pixel cloud transmittance G-buffer the path
    // tracer writes and StarsComposite reads). Slots 11/12/13 (issue
    // #118) are the MetalFX specular-guidance trio: specular_albedo
    // (RGBA16F F0), roughness (R32F surface roughness), and
    // specular_hit_distance (R32F reflection depth). Vulkan mirrors
    // them for SPIR-V slot stability but the in-house NRD/SVGF chain
    // doesn't consume them today (#50 will). Keep in sync with
    // kSlotToTexBinding[] in VulkanDevice.cpp.
    TextureHandle  bound_tex_[14] {};
    // 14 buffer slots:
    //   0..7   original engine layout (mesh_positions / mesh_indices,
    //          primitives, marginal / conditional CDFs, exposure_state,
    //          analytic-prim bvh_nodes; slot 0 unused).
    //   8, 9   triangle BVH (tri_bvh_nodes, tri_bvh_permuted_ids -- the
    //          PR #106 follow-up that replaces the SW Möller-Trumbore
    //          linear-scan path with a stack-based BVH walk).
    //   10     SDF cluster storage buffer (SDF Phase 1, #97;
    //          vk::binding 21; moved from binding 19 to make room for
    //          the triangle BVH).
    //   11     SIGMA shadow visibility R32F-per-pixel buffer (#115;
    //          vk::binding 23 -- a storage buffer, not an image,
    //          because Metal's 8-RW-texture cap on PathTrace was
    //          already saturated).
    //   12     light primitives (#73), the analytic light list at
    //          vk::binding 27.
    //   13     hierarchical light tree (#129), packed-node SSBO at
    //          vk::binding 28.
    //   14     ReSTIR DI reservoir SSBO (#78) at vk::binding 29.
    //   15     smoke emitters (#136, Fluid Phase 1), the density-
    //          injection SSBO at vk::binding 30.
    // Pre-#136 the array was [12] which silently dropped any
    // BindBuffer(>= 12, ...) at the bounds-check in BindBuffer().
    // Now [16]. Keep this in sync with kSlotToBufBinding[] in
    // VulkanDevice.cpp.
    BufferHandle   bound_buf_[16] {};
    std::size_t    bound_buf_off_[16] {};
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

    // True iff the constructor finished without early-returning on a
    // missing required feature / failed Vulkan call. The factory in
    // Device.cpp uses this to surface a clean nullptr on partial init.
    bool IsInitialized() const { return init_ok_; }

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
    bool ReadbackSwapchain(void* dst, std::size_t dst_size,
                           std::uint32_t* out_w, std::uint32_t* out_h,
                           SwapFormat* out_format) override;
    bool SupportsSwapchainReadback() const override { return swap_supports_readback_; }

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
    bool         IsDeviceLost() const override { return device_lost_; }

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

    // Predictive pipeline JIT prewarming (see Device::EnsurePipelineWarmed).
    // On Vulkan, the constructor already launches an async worker that
    // builds every known kernel name eagerly, so by the time the engine
    // calls EnsurePipelineWarmed the relevant build is either:
    //   * already complete (named_pipelines_.find hits)        -> no-op
    //   * still in flight inside the worker's hardcoded list   -> no-op
    //                                                              (the
    //     worker will reach it on its own)
    //   * an unknown name (no SPIR-V blob embedded for it)     -> logged
    //                                                              once
    //     at tier-2 diag so engine→device API drift surfaces visibly.
    // Method exists as the documented engine→device handshake so future
    // dynamic pipelines (loaded from disk, generated at runtime) have a
    // clean queue-on-demand entry point without each pipeline owner
    // touching the device construction sequence.
    void EnsurePipelineWarmed(const char* kernel_name) override;

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
    //
    // bloom_in_view / bloom_intensity: optional bloom-pyramid mip 0
    // composite. Pass VK_NULL_HANDLE / 0.0f to skip the bloom add (the
    // shader gates the read on intensity > 0). When supplied, the view
    // must reference an RGBA16F storage image in GENERAL layout, with
    // any prior compute-write ordered by the caller (the queue submit
    // boundary between the engine cb that wrote the pyramid and this
    // private cb is sufficient on its own).
    void EncodeDenoiseFinalize(VkCommandBuffer cb,
                               VkImageView    color_in_view,
                               VkImageView    final_output_view,
                               VkBuffer       exposure_state_buf,
                               VkImageView    bloom_in_view,
                               float          bloom_intensity,
                               std::uint32_t  width,
                               std::uint32_t  height,
                               bool           hdr_pipeline,
                               VkImageView    stars_in_view);  // accum_stars (issue #46)

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

    // Try to record the pending swapchain-readback copy into `cb`,
    // sourcing from `swap_image` at `extent`. Returns true if a pending
    // request was claimed (caller MUST call PublishSwapchainCaptureConsumed
    // after their queue-submit returns).
    //
    // Contract:
    //   * swap_image must be in VK_IMAGE_LAYOUT_GENERAL when this method
    //     records into `cb`.
    //   * The prior writes to swap_image must execute on
    //     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT with
    //     VK_ACCESS_SHADER_WRITE_BIT; this method inserts the
    //     compute->transfer + buffer->host barriers internally.
    //   * Caller is responsible for the subsequent layout transition
    //     to PRESENT_SRC (or whatever else) after this method returns.
    //
    // The OptiX path uses this from inside its private cb (after
    // EncodeDenoiseFinalize composites bloom + tonemap into the swap
    // image) so the screenshot reflects the image that gets PRESENT'd.
    // The SVGF path uses it from Submit() against the engine cb, where
    // the SVGF DenoiseFinalize compute pass already wrote the swap.
    // Atomically exchanges the request flag so only one cb records.
    bool TryRecordSwapchainCapture(VkCommandBuffer cb,
                                   VkImage         swap_image,
                                   VkExtent2D      extent);

    // Publish the swap_capture_consumed_ flag. Call this AFTER the
    // queue-submit that contains the recorded copy returns, so a
    // concurrent ReadbackSwapchain poller that observes the flag and
    // immediately calls vkDeviceWaitIdle actually waits for the copy
    // to complete instead of finding nothing queued. Pairs with
    // TryRecordSwapchainCapture above.
    void PublishSwapchainCaptureConsumed();

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
    // True iff the constructor ran to completion. The factory in
    // Device.cpp checks this and returns nullptr on partial init, so
    // Engine never sees a half-built device whose VkDevice handle is
    // VK_NULL_HANDLE -- that previously cascaded into a wall of
    // "buffer creation failed" / "texture create/upload failed" errors
    // (one per resource the engine then tried to create against the
    // dead device) after the first init check failed.
    bool        init_ok_      = false;
    // VkResult correctness floor (#???): set by any submit / wait / idle
    // call that returns VK_ERROR_DEVICE_LOST. Once latched the device is
    // dead -- subsequent vkQueueSubmit / vkWaitForFences will keep
    // returning the same error, and there's no recovery path short of a
    // full backend re-init (which the engine doesn't attempt today).
    // Engine::RenderFrame polls IsDeviceLost() at the top so the per-
    // frame work no-ops cleanly, and the smoke harness trips
    // smoke_test_failed_ so ctest sees a non-zero exit instead of
    // silently writing an all-zero PNG against a hung GPU.
    bool        device_lost_  = false;
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
    // True iff the platform's surface advertised TRANSFER_SRC_BIT and
    // RecreateSwapchain successfully included it in the swap's
    // imageUsage. Read by SupportsSwapchainReadback() / the
    // ReadbackSwapchain implementation so the screenshot-swap path
    // refuses to engage on platforms that strip TRANSFER_SRC.
    bool                        swap_supports_readback_ = false;

    // ReadbackSwapchain handshake: two atomics + a persistent host-
    // coherent staging buffer (declared further down, alongside the
    // BufferEntry definition).
    //   swap_capture_requested_: set by the caller, consumed by the
    //                            next Submit() (which inserts the
    //                            vkCmdCopyImageToBuffer + sets the
    //                            "consumed" flag below).
    //   swap_capture_consumed_:  set by Submit() once the copy has
    //                            been recorded into the cmd buffer.
    //                            The caller WaitIdle's after seeing
    //                            this so the GPU work also completes.
    std::atomic<bool>           swap_capture_requested_ { false };
    std::atomic<bool>           swap_capture_consumed_  { false };
    // Latched at the recording site that emits vkCmdCopyImageToBuffer.
    // ReadbackSwapchain reads from this (NOT current swap_extent_)
    // when computing the memcpy size, so a swap-resize between
    // record and read can't desync the staging-vs-caller byte count.
    VkExtent2D                  swap_capture_extent_ { 0, 0 };

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
    // against the larger layout. Every binding is flagged
    // PARTIALLY_BOUND so dispatches can leave unused slots unwritten
    // without validation noise (no dependence on
    // VK_EXT_robustness2.nullDescriptor, which MoltenVK doesn't expose).
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
    // Shutdown-abort signal for the async pipeline builder. DestroyDevice
    // sets this true BEFORE join() so the worker can short-circuit any
    // remaining build_pipeline() calls in its list -- without it, quit
    // blocks for the FULL pipeline-build duration (~25s release-build,
    // cold NVIDIA pipeline cache) when the user closes the window before
    // the worker finishes its 7+ kernel sweep. vkCreateComputePipelines
    // itself is not abortable mid-flight, so the in-flight build still
    // completes; the abort prevents starting any new builds after it.
    std::atomic<bool>     pipeline_build_abort_{false};

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
    // Ring of descriptor sets for the shared 20-binding layout. Why a
    // ring and not one set per frame: the shared layout uses
    // UPDATE_AFTER_BIND on every binding so the engine can rewrite
    // bindings between dispatches in the same cmd buffer. Per the
    // Vulkan spec, the GPU reads UPDATE_AFTER_BIND descriptors at
    // command-EXECUTION time, not record time -- which means if we
    // share one set across multiple recorded dispatches and update
    // its bindings between them, ALL dispatches end up reading the
    // LATEST update at execution. The path-tracer's `output` binding
    // would observe whatever the last bloom_down rebinding left, and
    // its output.GetDimensions() would return the wrong dimensions.
    // The ring sidesteps this: each Dispatch() takes a fresh set from
    // dsets_[current_frame_][next_dispatch_set_], so each dispatch's
    // bindings are isolated. next_dispatch_set_ is reset to 0 in
    // AcquireCommandBuffer. Total live sets = kFramesInFlight *
    // kDispatchSetsPerFrame -- conservative cap that covers PathTrace
    // + autoexpose + 9 bloom (5 down + 4 up) + perfoverlay + room.
    static constexpr int kDispatchSetsPerFrame = 24;
    VkDescriptorSet  dsets_[kFramesInFlight][kDispatchSetsPerFrame] {};
    int              next_dispatch_set_ = 0;

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

    // Persistent staging buffer for ReadbackSwapchain. Allocated /
    // re-allocated lazily on the first capture request (and on swap
    // resize), so projects that never trigger the screenshot-swap
    // path don't pay the ~3MB cost (1280x720x4) at startup. See
    // ReadbackSwapchain for the rest of the contract.
    BufferEntry                 swap_capture_staging_ {};
    std::size_t                 swap_capture_staging_capacity_ = 0;

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
    // Hand out the next descriptor set in this frame's ring and
    // advance the cursor. Called once per Dispatch so each dispatch
    // gets its own set (see kDispatchSetsPerFrame's comment for why
    // we can't reuse one set across multiple dispatches with
    // UPDATE_AFTER_BIND). Wraps if a frame issues more than
    // kDispatchSetsPerFrame dispatches -- which would re-introduce
    // the sharing bug for whichever dispatches collide; bump the
    // constant if you see it.
    VkDescriptorSet AcquireDispatchDescriptorSet() {
        VkDescriptorSet s = dsets_[current_frame_][next_dispatch_set_];
        next_dispatch_set_ = (next_dispatch_set_ + 1) % kDispatchSetsPerFrame;
        return s;
    }
    VkPipelineLayout    SharedPipelineLayout() const { return shared_pipe_layout_; }
    const BufferEntry&  CurrentFrameUbo() const { return frame_ubos_[current_frame_]; }
    std::uint32_t       MaxPushConstantSize() const { return max_push_constant_size_; }
};

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle&);
}

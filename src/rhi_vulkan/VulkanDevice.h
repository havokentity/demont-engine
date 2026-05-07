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
#include <unordered_map>
#include <vector>

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
    void Barrier(const BarrierDesc&) override {}

    void Reset(VkCommandBuffer cb);
    VkCommandBuffer Raw() const { return cb_; }

private:
    VulkanDevice*  device_ = nullptr;
    VkCommandBuffer cb_     = VK_NULL_HANDLE;
    PipelineHandle bound_pipeline_{0};
    TextureHandle  bound_tex_[8] {};
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

    void WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                     std::size_t dst_offset) override;
    bool WriteTexture(TextureHandle h, const void* src, std::size_t src_size) override;
    bool ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                         std::uint32_t* out_w, std::uint32_t* out_h) override;

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

    // Internal accessors used by the command buffer.
    VkDevice         RawDevice()     const { return device_; }
    VkPipeline       LookupPipeline(PipelineHandle h);
    VkPipelineLayout LookupPipelineLayout(PipelineHandle h);
    VkImageView      CurrentSwapchainImageView() const;

    static constexpr std::uint64_t kSwapchainTextureId = 1;

    // ---- Push-constant / descriptor-binding constants -------------------
    // Bytes [0..kPushSplitOffset] of the engine-side PtPush go into actual
    // VkPushConstants; bytes [kPushSplitOffset..push_size_] go into the
    // per-frame Frame UBO at binding kFrameUboBinding. Matches the layout
    // of Push (small) + Frame (UBO) cbuffers in PathTrace.slang's SPIR-V
    // path. Tied to the byte boundary inside PtPush in Engine.cpp; if you
    // reorder fields, update these too.
    static constexpr std::uint32_t kPushSplitOffset = 112;
    static constexpr std::uint32_t kFrameUboBinding = 14;
    static constexpr std::size_t   kFrameUboSize    = 512;  // 336B + headroom

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
    // 16 bindings matching the maximally-expanded shader: the smaller
    // shaders (Tonemap, BloomDown/Up) reference only a subset; Vulkan
    // allows pipelines whose shader doesn't use a binding to be created
    // against the larger layout. nullDescriptor lets the unused slots
    // bind VK_NULL_HANDLE without validation noise.
    VkDescriptorSetLayout shared_dset_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      shared_pipe_layout_ = VK_NULL_HANDLE;

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

    // ---- Async readback ring ------------------------------------------
    // ReadbackTexture is called from the engine every 8 frames (auto-
    // exposure update) and on demand (screenshots). Doing a synchronous
    // vkQueueWaitIdle each time stalls the GPU and produces visible
    // stutters. Instead, we maintain a small ring of staging buffers +
    // fences. Each call:
    //   1. Polls fences, captures any completed copies.
    //   2. If a completed slot matches the requested texture handle,
    //      returns its data immediately (likely 1-2 ticks old; fine for
    //      auto-exposure's eye-adaptation smoothing).
    //   3. Submits a new copy command for the requested texture with
    //      its own fence. Does NOT wait.
    //   4. If no completed data was available (cold start, different
    //      texture), falls back to a synchronous wait on the slot we
    //      just submitted -- one stutter at startup, then the ring
    //      stays warm.
    struct ReadbackSlot {
        BufferEntry     staging;
        VkFence         fence       = VK_NULL_HANDLE;
        VkCommandBuffer cmd         = VK_NULL_HANDLE;
        bool            in_flight   = false;
        bool            data_ready  = false;
        std::uint64_t   src_id      = 0;
        std::uint32_t   width       = 0;
        std::uint32_t   height      = 0;
        std::size_t     bytes       = 0;
    };
    static constexpr int kReadbackSlots = 3;
    ReadbackSlot readback_slots_[kReadbackSlots] {};

    void PollReadbacks();
    bool SubmitReadback(ReadbackSlot& slot, VkImage img, VkFormat fmt,
                        VkExtent2D extent, std::uint64_t src_id,
                        std::size_t bytes, std::size_t bpp);

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
    VkBuffer            LookupBuffer(BufferHandle h);
    VkAccelerationStructureKHR LookupAccel(AccelStructHandle h);
    VkDescriptorSet     CurrentDescriptorSet() const { return dsets_[current_frame_]; }
    VkPipelineLayout    SharedPipelineLayout() const { return shared_pipe_layout_; }
    const BufferEntry&  CurrentFrameUbo() const { return frame_ubos_[current_frame_]; }
};

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle&);
}

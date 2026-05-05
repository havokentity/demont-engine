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
    void BindBuffer(std::uint32_t, BufferHandle, std::size_t) override {}
    void BindStorageTexture(std::uint32_t slot, TextureHandle t) override;
    void BindAccelStruct(std::uint32_t, AccelStructHandle) override {}
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
    std::uint8_t   push_buf_[128] {};
    std::size_t    push_size_ = 0;
};

class VulkanDevice : public Device {
public:
    explicit VulkanDevice(const NativeWindowHandle& w);
    ~VulkanDevice() override;

    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override { return {0}; }
    AccelStructHandle CreateTLAS(const TLASDesc&) override { return {0}; }

    void DestroyBuffer(BufferHandle) override {}
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle) override {}
    void DestroyAccelStruct(AccelStructHandle) override {}

    void WriteBuffer(BufferHandle, const void*, std::size_t,
                     std::size_t) override {}

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

private:
    void DestroyDevice();
    bool RecreateSwapchain();

    GLFWwindow* glfw_window_ = nullptr;
    int         width_       = 0;
    int         height_      = 0;
    std::string device_name_ = "Vulkan Device";
    bool        rt_supported_ = false;
    std::uint32_t frame_index_ = 0;

    // Core handles
    VkInstance              instance_      = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_msgr_   = VK_NULL_HANDLE;
    VkSurfaceKHR            surface_       = VK_NULL_HANDLE;
    VkPhysicalDevice        phys_device_   = VK_NULL_HANDLE;
    VkDevice                device_        = VK_NULL_HANDLE;
    std::uint32_t           graphics_qfi_  = 0;
    VkQueue                 graphics_queue_ = VK_NULL_HANDLE;

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

    // Pipelines.  All three (clear, scene, pathtrace) share a single
    // pipeline layout (2 storage-image bindings + 128B push range);
    // unused bindings are tolerated by Vulkan when the shader doesn't
    // reference them.
    struct PipelineEntry {
        VkPipeline pipeline;
    };
    std::mutex resource_mutex_;
    std::uint64_t next_id_ = kSwapchainTextureId + 1;
    std::unordered_map<std::uint64_t, PipelineEntry>     pipelines_;
    std::unordered_map<std::string, std::uint64_t>       named_pipelines_;

    // Shared layouts -- created once at construction.
    VkDescriptorSetLayout shared_dset_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      shared_pipe_layout_ = VK_NULL_HANDLE;

    // One descriptor pool, one set per frame, updated each dispatch.
    VkDescriptorPool dpool_ = VK_NULL_HANDLE;
    VkDescriptorSet  dsets_[kFramesInFlight] {};

    // Allocated textures (CreateTexture). Each has its own image, memory,
    // and view. Looked up by id from CommandBuffer's bound_tex_ slots.
    struct ImageEntry {
        VkImage        image  = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView    view   = VK_NULL_HANDLE;
        VkFormat       format = VK_FORMAT_UNDEFINED;
        VkExtent2D     extent { 0, 0 };
    };
    std::unordered_map<std::uint64_t, ImageEntry> images_;

public:
    VkImageView         LookupImageView(TextureHandle h);
    VkImage             LookupImage(TextureHandle h);
    VkDescriptorSet     CurrentDescriptorSet() const { return dsets_[current_frame_]; }
    VkPipelineLayout    SharedPipelineLayout() const { return shared_pipe_layout_; }
};

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle&);
}

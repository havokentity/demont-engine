// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Minimal Vulkan-blit present path for the Software backend on Win32.
// Exists so that switching from the Vulkan backend to Software at
// runtime doesn't run into the DXGI flip-model lockout: once Vulkan
// has called vkQueuePresentKHR on an HWND, Microsoft's compositor
// (DWM) commits that window to flip-model presentation for the
// window's remaining lifetime, and GDI writes (SetDIBitsToDevice +
// friends) are silently ignored. See VK_KHR_win32_surface spec note
// "creating a VkSwapchainKHR over a window object can also alter the
// object for its remaining lifetime" and the DXGI flip-model docs
// note "after the first successful call to IDXGISwapChain1::Present1
// on a flip-model swap chain, GDI no longer works with the HWND
// associated with that swap chain, even after the destruction of the
// swap chain".
//
// The fix is to never leave Vulkan presentation: the Software backend
// runs its CPU+Embree path tracer as before, packs RGBA32F -> BGRA8
// in PresentOutput()'s scratch buffer as before, then this class
// uploads that scratch into a host-visible staging buffer and
// vkCmdCopyBufferToImage's it into the next swapchain image, with
// vkQueuePresentKHR finishing the cycle. The window stays in
// flip-model presentation for its whole life.
//
// This class owns its OWN VkInstance / VkDevice / VkSwapchainKHR --
// it does NOT share with VulkanDevice, because the two RHI backends
// are mutually exclusive (TearDownDevice destroys one before the
// other is created). Keeping them independent avoids any
// cross-backend lifetime entanglement.
//
// Win32 only. On Mac the equivalent is the existing CAMetalLayer
// upload path inside SoftwareDevice::EndFrame; Mac's compositor
// doesn't have the GDI/DXGI lock-in problem.

#pragma once

#if defined(_WIN32)

#include <cstdint>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.h>

namespace pt::rhi::sw {

class SoftwareVulkanPresent {
public:
    SoftwareVulkanPresent() = default;
    ~SoftwareVulkanPresent();

    SoftwareVulkanPresent(const SoftwareVulkanPresent&)            = delete;
    SoftwareVulkanPresent& operator=(const SoftwareVulkanPresent&) = delete;

    // One-time setup. Returns true on success. On failure, the caller
    // is expected to log + fall back to the GDI present path.
    bool Init(void* hwnd, int width, int height);

    // Recreate swapchain at new dimensions. Cheap when (w,h) matches
    // the current swapchain extent (no-op).
    void Resize(int width, int height);

    // Upload a pre-packed BGRA8 scratch buffer (same layout the GDI
    // path uses, 0xAARRGGBB) into the next swapchain image, then
    // vkQueuePresentKHR. width/height must match the swapchain extent
    // (call Resize first if the window resized).
    bool Present(const std::vector<std::uint32_t>& bgra8_scratch,
                 int width, int height);

private:
    void Shutdown();
    bool CreateInstance();
    bool PickPhysicalDevice();
    bool CreateDevice();
    bool CreateSurface(void* hwnd);
    bool CreateSwapchain(int width, int height);
    void DestroySwapchain();
    bool CreateStagingBuffer(std::uint64_t size_bytes);
    void DestroyStagingBuffer();
    bool CreateSyncObjects();
    void DestroySyncObjects();
    bool CreateCommandPool();
    void DestroyCommandPool();

    // Find a memory type matching the requested type-bits + property
    // flags. Returns UINT32_MAX on failure.
    std::uint32_t FindMemoryType(std::uint32_t type_filter,
                                 VkMemoryPropertyFlags props) const;

    VkInstance       instance_         = VK_NULL_HANDLE;
    VkPhysicalDevice phys_             = VK_NULL_HANDLE;
    VkDevice         device_           = VK_NULL_HANDLE;
    std::uint32_t    queue_family_     = UINT32_MAX;
    VkQueue          queue_            = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_          = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_        = VK_NULL_HANDLE;
    VkFormat         swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       swapchain_extent_ = {0, 0};
    std::vector<VkImage> swap_images_;

    VkCommandPool    cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf_  = VK_NULL_HANDLE;

    // Single-frame-in-flight model: we don't try to overlap upload +
    // present (the kernel takes long enough that the sync overhead
    // is in the noise). Keeps the impl small and easier to reason
    // about across backend-switch teardown ordering.
    VkSemaphore      sem_image_avail_  = VK_NULL_HANDLE;
    VkSemaphore      sem_render_done_  = VK_NULL_HANDLE;
    VkFence          fence_in_flight_  = VK_NULL_HANDLE;

    // Staging buffer: host-visible, host-coherent. memcpy from
    // BGRA8 scratch into staging.mapped_, then vkCmdCopyBufferToImage
    // to the swapchain image.
    VkBuffer         staging_buf_     = VK_NULL_HANDLE;
    VkDeviceMemory   staging_mem_     = VK_NULL_HANDLE;
    void*            staging_mapped_  = nullptr;
    std::uint64_t    staging_size_    = 0;

    bool             initialized_     = false;
};

}  // namespace pt::rhi::sw

#endif  // _WIN32

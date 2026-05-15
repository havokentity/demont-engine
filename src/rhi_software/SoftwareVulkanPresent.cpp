// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// See SoftwareVulkanPresent.h for the design rationale.

#include "SoftwareVulkanPresent.h"

#if defined(_WIN32)

#include "../core/Log.h"

#include <algorithm>
#include <cstring>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>

namespace pt::rhi::sw {

SoftwareVulkanPresent::~SoftwareVulkanPresent() { Shutdown(); }

bool SoftwareVulkanPresent::Init(void* hwnd, int width, int height) {
    if (initialized_) return true;
    if (hwnd == nullptr || width <= 0 || height <= 0) {
        LOG_ERROR("SoftwareVulkanPresent::Init: invalid HWND/dims "
                  "(hwnd={}, w={}, h={})", hwnd, width, height);
        return false;
    }
    if (!CreateInstance())          { Shutdown(); return false; }
    if (!CreateSurface(hwnd))       { Shutdown(); return false; }
    if (!PickPhysicalDevice())      { Shutdown(); return false; }
    if (!CreateDevice())            { Shutdown(); return false; }
    if (!CreateCommandPool())       { Shutdown(); return false; }
    if (!CreateSyncObjects())       { Shutdown(); return false; }
    if (!CreateSwapchain(width, height)) { Shutdown(); return false; }
    initialized_ = true;
    LOG_INFO("Software backend (Win32): Vulkan-blit present path online "
             "({}x{}, BGRA8 swapchain on HWND, {} swap images, single "
             "frame-in-flight)",
             width, height, swap_images_.size());
    return true;
}

void SoftwareVulkanPresent::Resize(int width, int height) {
    if (!initialized_) return;
    if (width <= 0 || height <= 0) return;
    if (static_cast<std::uint32_t>(width)  == swapchain_extent_.width &&
        static_cast<std::uint32_t>(height) == swapchain_extent_.height) {
        return;  // already correctly sized
    }
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    DestroySwapchain();
    if (!CreateSwapchain(width, height)) {
        LOG_ERROR("SoftwareVulkanPresent::Resize: CreateSwapchain failed "
                  "({}x{})", width, height);
    } else {
        LOG_INFO("SoftwareVulkanPresent: swapchain resized to {}x{} "
                 "({} swap images)",
                 width, height, swap_images_.size());
    }
}

bool SoftwareVulkanPresent::Present(
        const std::vector<std::uint32_t>& bgra8_scratch,
        int width, int height) {
    if (!initialized_) return false;
    if (bgra8_scratch.empty() || width <= 0 || height <= 0) return false;

    if (static_cast<std::uint32_t>(width)  != swapchain_extent_.width ||
        static_cast<std::uint32_t>(height) != swapchain_extent_.height) {
        Resize(width, height);
        if (!initialized_) return false;
    }

    const std::uint64_t need_bytes =
        std::uint64_t(width) * std::uint64_t(height) *
        std::uint64_t(sizeof(std::uint32_t));
    if (need_bytes > staging_size_) {
        DestroyStagingBuffer();
        if (!CreateStagingBuffer(need_bytes)) {
            LOG_ERROR("SoftwareVulkanPresent::Present: staging buffer "
                      "alloc failed ({} bytes)", need_bytes);
            return false;
        }
    }

    // Wait for the previous frame's GPU work to retire so we can
    // safely overwrite staging + cmd_buf_ + signal sem_image_avail_.
    vkWaitForFences(device_, 1, &fence_in_flight_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_in_flight_);

    std::memcpy(staging_mapped_, bgra8_scratch.data(),
                static_cast<size_t>(need_bytes));

    std::uint32_t img_idx = 0;
    VkResult ar = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                        sem_image_avail_, VK_NULL_HANDLE,
                                        &img_idx);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
        Resize(width, height);
        return true;  // skip this frame; next call will use new swapchain
    }
    if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
        LOG_WARN("SoftwareVulkanPresent: vkAcquireNextImageKHR failed "
                 "(VkResult={})", static_cast<int>(ar));
        return false;
    }

    vkResetCommandBuffer(cmd_buf_, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf_, &bi);

    // Transition swapchain image: undefined -> transfer-dst-optimal.
    VkImage dst = swap_images_[img_idx];
    VkImageMemoryBarrier to_dst{};
    to_dst.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_dst.srcAccessMask    = 0;
    to_dst.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image            = dst;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd_buf_,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_dst);

    // Copy staging buffer -> swapchain image.
    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;  // tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset       = {0, 0, 0};
    region.imageExtent       = {static_cast<std::uint32_t>(width),
                                static_cast<std::uint32_t>(height),
                                1};
    vkCmdCopyBufferToImage(cmd_buf_, staging_buf_, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);

    // Transition swapchain image: transfer-dst-optimal -> present-src.
    VkImageMemoryBarrier to_present{};
    to_present.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_present.srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_present.dstAccessMask    = 0;
    to_present.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_present.newLayout        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_present.image            = dst;
    to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd_buf_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_present);

    vkEndCommandBuffer(cmd_buf_);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &sem_image_avail_;
    si.pWaitDstStageMask    = &wait_stage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd_buf_;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &sem_render_done_;
    if (vkQueueSubmit(queue_, 1, &si, fence_in_flight_) != VK_SUCCESS) {
        LOG_ERROR("SoftwareVulkanPresent: vkQueueSubmit failed");
        return false;
    }

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &sem_render_done_;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &img_idx;
    VkResult pr = vkQueuePresentKHR(queue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        Resize(width, height);
    } else if (pr != VK_SUCCESS) {
        LOG_WARN("SoftwareVulkanPresent: vkQueuePresentKHR failed "
                 "(VkResult={})", static_cast<int>(pr));
        return false;
    }
    return true;
}

void SoftwareVulkanPresent::Shutdown() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    DestroySyncObjects();
    DestroyStagingBuffer();
    DestroyCommandPool();
    DestroySwapchain();
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    phys_         = VK_NULL_HANDLE;
    queue_        = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    initialized_  = false;
}

bool SoftwareVulkanPresent::CreateInstance() {
    VkApplicationInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName   = "demont-software-present";
    ai.apiVersion         = VK_API_VERSION_1_2;

    const char* exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    };
    VkInstanceCreateInfo ici{};
    ici.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo        = &ai;
    ici.enabledExtensionCount   = 2;
    ici.ppEnabledExtensionNames = exts;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        LOG_ERROR("SoftwareVulkanPresent: vkCreateInstance failed");
        return false;
    }
    return true;
}

bool SoftwareVulkanPresent::CreateSurface(void* hwnd) {
    VkWin32SurfaceCreateInfoKHR sci{};
    sci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hwnd      = static_cast<HWND>(hwnd);
    sci.hinstance = GetModuleHandleW(nullptr);
    if (vkCreateWin32SurfaceKHR(instance_, &sci, nullptr, &surface_)
            != VK_SUCCESS) {
        LOG_ERROR("SoftwareVulkanPresent: vkCreateWin32SurfaceKHR failed");
        return false;
    }
    return true;
}

bool SoftwareVulkanPresent::PickPhysicalDevice() {
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) {
        LOG_ERROR("SoftwareVulkanPresent: no Vulkan physical devices");
        return false;
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    // Pick first device that has a graphics + present queue family.
    // We don't care about features beyond that since we only do
    // copy-buffer-to-image + present.
    for (auto pd : devs) {
        std::uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qprops.data());
        for (std::uint32_t i = 0; i < qn; ++i) {
            if (!(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present_ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface_, &present_ok);
            if (present_ok != VK_TRUE) continue;
            phys_         = pd;
            queue_family_ = i;
            return true;
        }
    }
    LOG_ERROR("SoftwareVulkanPresent: no graphics+present queue family found");
    return false;
}

bool SoftwareVulkanPresent::CreateDevice() {
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &qprio;

    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    if (vkCreateDevice(phys_, &dci, nullptr, &device_) != VK_SUCCESS) {
        LOG_ERROR("SoftwareVulkanPresent: vkCreateDevice failed");
        return false;
    }
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    return true;
}

bool SoftwareVulkanPresent::CreateSwapchain(int width, int height) {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps);

    // Pick BGRA8 to match the present_scratch_ packing the existing
    // GDI path also uses (0xAARRGGBB byte layout = BGRA in memory).
    std::uint32_t fmt_n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &fmt_n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmt_n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &fmt_n, fmts.data());
    VkSurfaceFormatKHR chosen = fmts[0];
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    swapchain_format_ = chosen.format;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width  = std::clamp(static_cast<std::uint32_t>(width),
                                   caps.minImageExtent.width,
                                   caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<std::uint32_t>(height),
                                   caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }
    swapchain_extent_ = extent;

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = surface_;
    sci.minImageCount    = image_count;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;  // always supported
    sci.clipped          = VK_TRUE;
    sci.oldSwapchain     = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_)
            != VK_SUCCESS) {
        LOG_ERROR("SoftwareVulkanPresent: vkCreateSwapchainKHR failed");
        return false;
    }

    std::uint32_t img_n = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_n, nullptr);
    swap_images_.resize(img_n);
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_n, swap_images_.data());
    return true;
}

void SoftwareVulkanPresent::DestroySwapchain() {
    if (swapchain_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }
    swapchain_ = VK_NULL_HANDLE;
    swap_images_.clear();
    swapchain_extent_ = {0, 0};
}

bool SoftwareVulkanPresent::CreateStagingBuffer(std::uint64_t size_bytes) {
    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size_bytes;
    bi.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bi, nullptr, &staging_buf_) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device_, staging_buf_, &mr);
    std::uint32_t mtype = FindMemoryType(
        mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mtype == UINT32_MAX) {
        LOG_ERROR("SoftwareVulkanPresent: no host-visible+coherent "
                  "memory type for staging buffer");
        vkDestroyBuffer(device_, staging_buf_, nullptr);
        staging_buf_ = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(device_, &ai, nullptr, &staging_mem_) != VK_SUCCESS) {
        vkDestroyBuffer(device_, staging_buf_, nullptr);
        staging_buf_ = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0);
    if (vkMapMemory(device_, staging_mem_, 0, VK_WHOLE_SIZE, 0,
                    &staging_mapped_) != VK_SUCCESS) {
        return false;
    }
    staging_size_ = mr.size;
    return true;
}

void SoftwareVulkanPresent::DestroyStagingBuffer() {
    if (staging_mem_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        if (staging_mapped_) vkUnmapMemory(device_, staging_mem_);
        vkFreeMemory(device_, staging_mem_, nullptr);
    }
    if (staging_buf_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, staging_buf_, nullptr);
    }
    staging_mem_    = VK_NULL_HANDLE;
    staging_buf_    = VK_NULL_HANDLE;
    staging_mapped_ = nullptr;
    staging_size_   = 0;
}

bool SoftwareVulkanPresent::CreateSyncObjects() {
    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(device_, &si, nullptr, &sem_image_avail_)
            != VK_SUCCESS ||
        vkCreateSemaphore(device_, &si, nullptr, &sem_render_done_)
            != VK_SUCCESS) {
        return false;
    }
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // first wait must succeed
    if (vkCreateFence(device_, &fi, nullptr, &fence_in_flight_)
            != VK_SUCCESS) {
        return false;
    }
    return true;
}

void SoftwareVulkanPresent::DestroySyncObjects() {
    if (sem_image_avail_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, sem_image_avail_, nullptr);
    }
    if (sem_render_done_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, sem_render_done_, nullptr);
    }
    if (fence_in_flight_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_in_flight_, nullptr);
    }
    sem_image_avail_ = VK_NULL_HANDLE;
    sem_render_done_ = VK_NULL_HANDLE;
    fence_in_flight_ = VK_NULL_HANDLE;
}

bool SoftwareVulkanPresent::CreateCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmd_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd_buf_) != VK_SUCCESS) {
        return false;
    }
    return true;
}

void SoftwareVulkanPresent::DestroyCommandPool() {
    if (cmd_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    }
    cmd_pool_ = VK_NULL_HANDLE;
    cmd_buf_  = VK_NULL_HANDLE;
}

std::uint32_t SoftwareVulkanPresent::FindMemoryType(
        std::uint32_t type_filter,
        VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

}  // namespace pt::rhi::sw

#endif  // _WIN32

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Vulkan backend.  Single-file impl (~600 LOC) covering instance creation,
// swapchain, one compute pipeline ("clear"), per-frame descriptor sets,
// and the command-buffer wrapper.  No VMA yet -- we don't allocate any
// non-swapchain images/buffers in P4; that lands in P5+.

#include "VulkanDevice.h"

#include "../core/Log.h"
#include "../core/Memory/MemTag.h"
#include "../core/Memory/Memory.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstring>

extern "C" {
extern const unsigned char shader_PathTrace_spirv_data[];
extern const unsigned long shader_PathTrace_spirv_size;
}

namespace pt::rhi::vk {

namespace {

#ifndef NDEBUG
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (data == nullptr || data->pMessage == nullptr) return VK_FALSE;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LOG_ERROR("VK: {}", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("VK: {}", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        // Quiet -- info-level messages flood the console.
    }
    return VK_FALSE;
}

VkShaderModule MakeShaderModule(VkDevice dev, const std::uint8_t* bytes,
                                std::size_t n) {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = n;
    ci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes);
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS) {
        LOG_ERROR("vkCreateShaderModule failed");
        return VK_NULL_HANDLE;
    }
    return m;
}

}  // namespace

// =====================================================================
// VulkanCommandBuffer
// =====================================================================

void VulkanCommandBuffer::Reset(VkCommandBuffer cb) {
    cb_ = cb;
    bound_pipeline_ = PipelineHandle{0};
    push_size_ = 0;
    for (auto& t : bound_tex_) t = TextureHandle{0};
}

void VulkanCommandBuffer::BindComputePipeline(PipelineHandle p) {
    bound_pipeline_ = p;
    if (cb_ == VK_NULL_HANDLE) return;
    auto pipe = device_->LookupPipeline(p);
    if (pipe != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    }
}

void VulkanCommandBuffer::BindStorageTexture(std::uint32_t slot, TextureHandle t) {
    if (slot < std::size(bound_tex_)) bound_tex_[slot] = t;
}

void VulkanCommandBuffer::PushConstants(const void* data, std::size_t size) {
    if (size > sizeof(push_buf_)) size = sizeof(push_buf_);
    std::memcpy(push_buf_, data, size);
    push_size_ = size;
}

void VulkanCommandBuffer::Dispatch(std::uint32_t gx, std::uint32_t gy,
                                   std::uint32_t gz) {
    if (cb_ == VK_NULL_HANDLE || !bound_pipeline_) return;
    VkPipelineLayout layout = device_->SharedPipelineLayout();
    if (layout == VK_NULL_HANDLE) return;

    // Update current frame's descriptor set with bound storage textures.
    // Slots 0 and 1 are storage-image bindings in the shared layout;
    // anything else is currently a no-op.
    VkDescriptorImageInfo dii[2] {};
    VkWriteDescriptorSet  writes[2] {};
    std::uint32_t writeCount = 0;
    auto dset = device_->CurrentDescriptorSet();
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
        VkImageView v = device_->LookupImageView(bound_tex_[slot]);
        if (v == VK_NULL_HANDLE) continue;
        dii[writeCount].imageView   = v;
        dii[writeCount].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[writeCount].sType    = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeCount].dstSet   = dset;
        writes[writeCount].dstBinding = slot;
        writes[writeCount].descriptorCount = 1;
        writes[writeCount].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[writeCount].pImageInfo      = &dii[writeCount];
        ++writeCount;
    }
    if (writeCount > 0) {
        vkUpdateDescriptorSets(device_->RawDevice(), writeCount, writes, 0, nullptr);
    }
    vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &dset, 0, nullptr);

    if (push_size_ > 0) {
        vkCmdPushConstants(cb_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(push_size_), push_buf_);
    }
    vkCmdDispatch(cb_, gx, gy, gz);
}

// =====================================================================
// VulkanDevice
// =====================================================================

VulkanDevice::VulkanDevice(const NativeWindowHandle& nw) {
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    glfw_window_ = static_cast<GLFWwindow*>(nw.opaque);
    width_  = nw.width;
    height_ = nw.height;

    // ---- Instance ------------------------------------------------------
    VkApplicationInfo ai{};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "DeMonT Engine";
    ai.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    ai.pEngineName      = "DeMonT";
    ai.apiVersion       = VK_API_VERSION_1_2;

    std::uint32_t glfw_ext_n = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_n);
    std::vector<const char*> exts(glfw_exts, glfw_exts + glfw_ext_n);
#if defined(__APPLE__)
    // MoltenVK-on-Mac is a portability driver, has to be enumerated.
    // On Windows / Linux native Vulkan loaders advertise the GPU
    // directly so the portability extensions are unavailable AND
    // unnecessary.
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
    if (kEnableValidation) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> layers;
    if (kEnableValidation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ici{};
    ici.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo        = &ai;
#if defined(__APPLE__)
    ici.flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
    ici.flags                   = 0;
#endif
    ici.enabledExtensionCount   = static_cast<std::uint32_t>(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    ici.enabledLayerCount       = static_cast<std::uint32_t>(layers.size());
    ici.ppEnabledLayerNames     = layers.data();

    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        LOG_ERROR("vkCreateInstance failed");
        return;
    }

    if (kEnableValidation) {
        auto pfn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (pfn != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT mci{};
            mci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mci.messageType    = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mci.pfnUserCallback = DebugCallback;
            pfn(instance_, &mci, nullptr, &debug_msgr_);
        }
    }

    // ---- Surface (GLFW handles the metal_surface dance for us) --------
    if (glfwCreateWindowSurface(instance_, glfw_window_, nullptr, &surface_) != VK_SUCCESS) {
        LOG_ERROR("glfwCreateWindowSurface failed");
        return;
    }

    // ---- Physical device ---------------------------------------------
    std::uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(instance_, &pd_count, nullptr);
    if (pd_count == 0) {
        LOG_ERROR("No Vulkan physical devices found");
        return;
    }
    std::vector<VkPhysicalDevice> pds(pd_count);
    vkEnumeratePhysicalDevices(instance_, &pd_count, pds.data());
    // Prefer a discrete GPU (NVIDIA / AMD dGPU) over an integrated one
    // when multiple are present. On Mac there's only MoltenVK so the
    // first device is fine; on Windows boxes with Intel iGPU + NVIDIA
    // RTX dGPU we want the RTX.
    phys_device_ = pds[0];
    for (auto pd : pds) {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(pd, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            phys_device_ = pd;
            break;
        }
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_device_, &props);
    device_name_ = props.deviceName;
    LOG_INFO("Vulkan device: {}", device_name_);

    // ---- Queue family ------------------------------------------------
    std::uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qf_count, qfs.data());
    bool found_qf = false;
    for (std::uint32_t i = 0; i < qf_count; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(phys_device_, i, surface_, &present);
        if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
            graphics_qfi_ = i;
            found_qf = true;
            break;
        }
    }
    if (!found_qf) {
        LOG_ERROR("No suitable Vulkan queue family (compute + present)");
        return;
    }

    // ---- Logical device ----------------------------------------------
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = graphics_qfi_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &qprio;

    std::vector<const char*> dexts;
    dexts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if defined(__APPLE__)
    // MoltenVK requires the portability_subset extension on the device.
    // Native Vulkan drivers (NVIDIA / AMD on Windows/Linux) don't.
    dexts.push_back("VK_KHR_portability_subset");
#endif

    // Slang's SPIR-V output uses StorageImageReadWithoutFormat /
    // StorageImageWriteWithoutFormat capabilities. Enable the matching
    // device features or vkCreateShaderModule fails validation.
    VkPhysicalDeviceFeatures feats{};
    feats.shaderStorageImageReadWithoutFormat  = VK_TRUE;
    feats.shaderStorageImageWriteWithoutFormat = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(dexts.size());
    dci.ppEnabledExtensionNames = dexts.data();
    dci.pEnabledFeatures        = &feats;
    if (vkCreateDevice(phys_device_, &dci, nullptr, &device_) != VK_SUCCESS) {
        LOG_ERROR("vkCreateDevice failed");
        return;
    }
    vkGetDeviceQueue(device_, graphics_qfi_, 0, &graphics_queue_);

    // ---- Command pool + per-frame command buffers, sync objects -------
    VkCommandPoolCreateInfo cpci{};
    cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = graphics_qfi_;
    vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = cmd_pool_;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kFramesInFlight;
    vkAllocateCommandBuffers(device_, &cbai, cmds_);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < kFramesInFlight; ++i) {
        vkCreateSemaphore(device_, &sci, nullptr, &sem_image_avail_[i]);
        vkCreateFence(device_,     &fci, nullptr, &fence_in_flight_[i]);
    }
    // sem_render_done_ is per-swapchain-image, sized in RecreateSwapchain.

    // ---- Descriptor pool: 1 storage image per frame -------------------
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps.descriptorCount = kFramesInFlight * 4;   // headroom for multi-binding
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = kFramesInFlight;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool_);

    // ---- Build shared layouts (used by all 3 pipelines) ---------------
    {
        // 2 storage-image bindings (output + accum). Pipelines whose
        // shader doesn't read binding 1 simply ignore it.
        VkDescriptorSetLayoutBinding b[2] {};
        for (int i = 0; i < 2; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 2;
        dslci.pBindings    = b;
        vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &shared_dset_layout_);

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = 128;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &shared_dset_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(device_, &plci, nullptr, &shared_pipe_layout_);

        // Per-frame descriptor sets.
        std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, shared_dset_layout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = kFramesInFlight;
        dsai.pSetLayouts        = layouts.data();
        vkAllocateDescriptorSets(device_, &dsai, dsets_);
    }

    // ---- Build the three compute pipelines ---------------------------
    auto build_pipeline = [&](const char* name,
                              const unsigned char* spirv,
                              std::size_t          spirv_size) {
        auto mod = MakeShaderModule(device_, spirv, spirv_size);
        if (mod == VK_NULL_HANDLE) return;
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci2{};
        cpci2.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci2.layout = shared_pipe_layout_;
        cpci2.stage  = stage;
        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci2,
                                     nullptr, &pipe) == VK_SUCCESS) {
            std::lock_guard lock(resource_mutex_);
            auto id = next_id_++;
            pipelines_.emplace(id, PipelineEntry{pipe});
            named_pipelines_.emplace(name, id);
        } else {
            LOG_ERROR("vkCreateComputePipelines({}) failed", name);
        }
        vkDestroyShaderModule(device_, mod, nullptr);
    };

    build_pipeline("pathtrace", shader_PathTrace_spirv_data, shader_PathTrace_spirv_size);

    if (!RecreateSwapchain()) return;

    wrapped_cb_ = std::make_unique<VulkanCommandBuffer>(this);
}

VulkanDevice::~VulkanDevice() {
    DestroyDevice();
}

void VulkanDevice::DestroyDevice() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        for (auto v : swap_views_) if (v) vkDestroyImageView(device_, v, nullptr);
        swap_views_.clear();
        if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        for (int i = 0; i < kFramesInFlight; ++i) {
            if (sem_image_avail_[i]) vkDestroySemaphore(device_, sem_image_avail_[i], nullptr);
            if (fence_in_flight_[i]) vkDestroyFence(device_, fence_in_flight_[i], nullptr);
        }
        for (auto s : sem_render_done_) {
            if (s) vkDestroySemaphore(device_, s, nullptr);
        }
        sem_render_done_.clear();
        if (cmd_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, cmd_pool_, nullptr);

        {
            std::lock_guard lock(resource_mutex_);
            for (auto& [id, e] : pipelines_) {
                if (e.pipeline) vkDestroyPipeline(device_, e.pipeline, nullptr);
            }
            pipelines_.clear();
            named_pipelines_.clear();
            for (auto& [id, im] : images_) {
                if (im.view)   vkDestroyImageView(device_, im.view, nullptr);
                if (im.image)  vkDestroyImage(device_, im.image, nullptr);
                if (im.memory) vkFreeMemory(device_, im.memory, nullptr);
            }
            images_.clear();
        }
        if (shared_pipe_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, shared_pipe_layout_, nullptr);
        if (shared_dset_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, shared_dset_layout_, nullptr);
        if (dpool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, dpool_, nullptr);

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (debug_msgr_ != VK_NULL_HANDLE) {
        auto pfn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (pfn) pfn(instance_, debug_msgr_, nullptr);
        debug_msgr_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::RecreateSwapchain() {
    if (device_ == VK_NULL_HANDLE) return false;
    vkDeviceWaitIdle(device_);

    for (auto v : swap_views_) if (v) vkDestroyImageView(device_, v, nullptr);
    swap_views_.clear();
    swap_images_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device_, surface_, &caps);

    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(glfw_window_, &fb_w, &fb_h);
    swap_extent_.width  = std::clamp(static_cast<std::uint32_t>(fb_w),
                                     caps.minImageExtent.width, caps.maxImageExtent.width);
    swap_extent_.height = std::clamp(static_cast<std::uint32_t>(fb_h),
                                     caps.minImageExtent.height, caps.maxImageExtent.height);
    width_  = static_cast<int>(swap_extent_.width);
    height_ = static_cast<int>(swap_extent_.height);

    // Pick a surface format we can also write to as a storage image.
    std::uint32_t fc = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device_, surface_, &fc, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fc);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device_, surface_, &fc, formats.data());
    swap_format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swap_format_ = f.format;
            color_space  = f.colorSpace;
            break;
        }
    }

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) image_count = std::min(image_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = surface_;
    sci.minImageCount    = image_count;
    sci.imageFormat      = swap_format_;
    sci.imageColorSpace  = color_space;
    sci.imageExtent      = swap_extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                         | VK_IMAGE_USAGE_STORAGE_BIT
                         | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;

    if (vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_) != VK_SUCCESS) {
        LOG_ERROR("vkCreateSwapchainKHR failed");
        return false;
    }

    std::uint32_t ic = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &ic, nullptr);
    swap_images_.resize(ic);
    vkGetSwapchainImagesKHR(device_, swapchain_, &ic, swap_images_.data());

    swap_views_.resize(ic);
    for (std::uint32_t i = 0; i < ic; ++i) {
        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = swap_images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = swap_format_;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.layerCount = 1;
        vci.subresourceRange.levelCount = 1;
        vkCreateImageView(device_, &vci, nullptr, &swap_views_[i]);
    }

    // Resize per-image render-finished semaphores.
    for (auto s : sem_render_done_) {
        if (s) vkDestroySemaphore(device_, s, nullptr);
    }
    sem_render_done_.assign(ic, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo rsci{};
    rsci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (std::uint32_t i = 0; i < ic; ++i) {
        vkCreateSemaphore(device_, &rsci, nullptr, &sem_render_done_[i]);
    }
    return true;
}

VkPipeline VulkanDevice::LookupPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = pipelines_.find(h.id);
    return (it == pipelines_.end()) ? VK_NULL_HANDLE : it->second.pipeline;
}
VkPipelineLayout VulkanDevice::LookupPipelineLayout(PipelineHandle) {
    return shared_pipe_layout_;
}
VkImageView VulkanDevice::CurrentSwapchainImageView() const {
    if (current_swap_index_ >= swap_views_.size()) return VK_NULL_HANDLE;
    return swap_views_[current_swap_index_];
}

VkImageView VulkanDevice::LookupImageView(TextureHandle h) {
    if (h.id == 0) return VK_NULL_HANDLE;
    if (h.id == kSwapchainTextureId) return CurrentSwapchainImageView();
    std::lock_guard lock(resource_mutex_);
    auto it = images_.find(h.id);
    return (it == images_.end()) ? VK_NULL_HANDLE : it->second.view;
}
VkImage VulkanDevice::LookupImage(TextureHandle h) {
    if (h.id == 0) return VK_NULL_HANDLE;
    if (h.id == kSwapchainTextureId) {
        return current_swap_index_ < swap_images_.size()
                 ? swap_images_[current_swap_index_] : VK_NULL_HANDLE;
    }
    std::lock_guard lock(resource_mutex_);
    auto it = images_.find(h.id);
    return (it == images_.end()) ? VK_NULL_HANDLE : it->second.image;
}

BufferHandle  VulkanDevice::CreateBuffer(const BufferDesc&)        { return {0}; }

TextureHandle VulkanDevice::CreateTexture(const TextureDesc& d) {
    if (device_ == VK_NULL_HANDLE) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    switch (d.format) {
        case TextureFormat::RGBA8_UNORM: fmt = VK_FORMAT_R8G8B8A8_UNORM;     break;
        case TextureFormat::RGBA8_SRGB:  fmt = VK_FORMAT_R8G8B8A8_SRGB;       break;
        case TextureFormat::RGBA16F:     fmt = VK_FORMAT_R16G16B16A16_SFLOAT; break;
        case TextureFormat::RGBA32F:     fmt = VK_FORMAT_R32G32B32A32_SFLOAT; break;
        case TextureFormat::R32_UINT:    fmt = VK_FORMAT_R32_UINT;             break;
        case TextureFormat::R32F:        fmt = VK_FORMAT_R32_SFLOAT;           break;
        case TextureFormat::RG16F:       fmt = VK_FORMAT_R16G16_SFLOAT;        break;
        default: break;
    }

    VkImageCreateInfo ici{};
    ici.sType        = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType    = VK_IMAGE_TYPE_2D;
    ici.format       = fmt;
    ici.extent       = { d.width, d.height, 1 };
    ici.mipLevels    = 1;
    ici.arrayLayers  = 1;
    ici.samples      = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling       = VK_IMAGE_TILING_OPTIMAL;
    ici.usage        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode  = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img = VK_NULL_HANDLE;
    if (vkCreateImage(device_, &ici, nullptr, &img) != VK_SUCCESS) return {0};

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(device_, img, &mr);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mp);
    std::uint32_t mem_type = 0;
    bool found_mt = false;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type = i;
            found_mt = true;
            break;
        }
    }
    if (!found_mt) {
        vkDestroyImage(device_, img, nullptr);
        return {0};
    }
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = mem_type;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyImage(device_, img, nullptr);
        return {0};
    }
    vkBindImageMemory(device_, img, mem, 0);

    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = fmt;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.layerCount = 1;
    vci.subresourceRange.levelCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(device_, &vci, nullptr, &view);

    // One-shot transition UNDEFINED -> GENERAL so the storage-image
    // descriptor write is valid the first time we use it.
    {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmd_pool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer once = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &cbai, &once);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(once, &bi);

        VkImageMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image         = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.layerCount = 1;
        b.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(once,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        vkEndCommandBuffer(once);
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &once;
        vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue_);
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &once);
    }

    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    images_[id] = ImageEntry{ img, mem, view, fmt, { d.width, d.height } };
    return TextureHandle{ id };
}

void VulkanDevice::DestroyTexture(TextureHandle h) {
    if (h.id == 0 || h.id == kSwapchainTextureId) return;
    std::lock_guard lock(resource_mutex_);
    auto it = images_.find(h.id);
    if (it == images_.end()) return;
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    if (it->second.view)   vkDestroyImageView(device_, it->second.view, nullptr);
    if (it->second.image)  vkDestroyImage(device_, it->second.image, nullptr);
    if (it->second.memory) vkFreeMemory(device_, it->second.memory, nullptr);
    images_.erase(it);
}
PipelineHandle VulkanDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    std::lock_guard lock(resource_mutex_);
    auto it = named_pipelines_.find(std::string(d.kernel_name));
    if (it == named_pipelines_.end()) return {0};
    return PipelineHandle{ it->second };
}

FrameContext VulkanDevice::BeginFrame() {
    if (device_ == VK_NULL_HANDLE) return {};

    vkWaitForFences(device_, 1, &fence_in_flight_[current_frame_],
                    VK_TRUE, UINT64_MAX);

    VkResult ar = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                        sem_image_avail_[current_frame_],
                                        VK_NULL_HANDLE, &current_swap_index_);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain();
        ar = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                   sem_image_avail_[current_frame_],
                                   VK_NULL_HANDLE, &current_swap_index_);
    }

    vkResetFences(device_, 1, &fence_in_flight_[current_frame_]);
    vkResetCommandBuffer(cmds_[current_frame_], 0);

    return FrameContext{
        .swapchain_image = TextureHandle{kSwapchainTextureId},
        .width  = swap_extent_.width,
        .height = swap_extent_.height,
        .frame_index = frame_index_,
    };
}

CommandBuffer* VulkanDevice::AcquireCommandBuffer() {
    if (device_ == VK_NULL_HANDLE) return nullptr;
    VkCommandBuffer cb = cmds_[current_frame_];
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    // Transition swapchain image: UNDEFINED -> GENERAL (writable from compute).
    VkImageMemoryBarrier toGen{};
    toGen.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGen.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    toGen.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    toGen.srcAccessMask = 0;
    toGen.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGen.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGen.image         = swap_images_[current_swap_index_];
    toGen.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGen.subresourceRange.layerCount = 1;
    toGen.subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toGen);

    // VulkanCommandBuffer::Dispatch handles descriptor updates + binding
    // now (post-P4 parity refactor). Nothing else to do here -- the
    // engine binds the pipeline + textures + push constants then calls
    // Dispatch which does the rest.

    wrapped_cb_->Reset(cb);
    return wrapped_cb_.get();
}

void VulkanDevice::Submit(CommandBuffer* cb) {
    if (cb == nullptr || device_ == VK_NULL_HANDLE) return;
    auto* vcb = static_cast<VulkanCommandBuffer*>(cb);
    VkCommandBuffer cmd = vcb->Raw();
    if (cmd == VK_NULL_HANDLE) return;

    // Transition swapchain image to PRESENT_SRC_KHR.
    VkImageMemoryBarrier toPres{};
    toPres.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toPres.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    toPres.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPres.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toPres.dstAccessMask = 0;
    toPres.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPres.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPres.image         = swap_images_[current_swap_index_];
    toPres.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toPres.subresourceRange.layerCount = 1;
    toPres.subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toPres);

    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &sem_image_avail_[current_frame_];
    si.pWaitDstStageMask    = &wait_stage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &sem_render_done_[current_swap_index_];
    vkQueueSubmit(graphics_queue_, 1, &si, fence_in_flight_[current_frame_]);

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &sem_render_done_[current_swap_index_];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &current_swap_index_;
    VkResult pr = vkQueuePresentKHR(graphics_queue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain();
    }
}

void VulkanDevice::EndFrame(CommandBuffer*) {
    current_frame_ = (current_frame_ + 1) % kFramesInFlight;
    ++frame_index_;
}

void VulkanDevice::WaitIdle() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
}

void VulkanDevice::Resize(int /*w*/, int /*h*/) {
    RecreateSwapchain();
}

std::size_t VulkanDevice::CurrentAllocatedBytes() const {
    // VK_EXT_memory_budget query lands in P5+ alongside VMA.  For now the
    // backend doesn't allocate non-swapchain memory.
    return 0;
}

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle& w) {
    return std::make_unique<vk::VulkanDevice>(w);
}
}  // namespace pt::rhi

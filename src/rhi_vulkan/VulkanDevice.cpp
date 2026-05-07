// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Vulkan backend, P12 expansion: ray-query, acceleration structures,
// real CreateBuffer / WriteBuffer / CreateBLAS / CreateTLAS, expanded
// descriptor set layout (16 bindings) matching the unified PathTrace
// shader, and per-frame UBO ring for the spilled push-constant tail.

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
#include <array>
#include <cstring>

extern "C" {
extern const unsigned char shader_PathTrace_spirv_data[];
extern const unsigned long shader_PathTrace_spirv_size;
extern const unsigned char shader_AutoExposure_spirv_data[];
extern const unsigned long shader_AutoExposure_spirv_size;
}

namespace pt::rhi::vk {

namespace {

#ifndef NDEBUG
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

// Engine slot -> shader vk::binding translation. The unified PathTrace
// shader uses bindings 0-13; Tonemap/Bloom* re-use a subset. Engine
// code in Engine.cpp uses 8-element textures, 8-element buffers, and
// 4-element accel-struct slot arrays as if Metal-style argument tables;
// we flatten these into the single Vulkan descriptor set here.
constexpr std::uint32_t kSlotToTexBinding[8] = {
    0,  // engine slot 0 -> shader binding 0  (output / swapchain)
    1,  // engine slot 1 -> shader binding 1  (accum_hdr)
    6,  // engine slot 2 -> shader binding 6  (denoise_color)
    7,  // engine slot 3 -> shader binding 7  (depth_tex)
    8,  // engine slot 4 -> shader binding 8  (motion_tex)
    9,  // engine slot 5 -> shader binding 9  (env_map)
    12, // engine slot 6 -> shader binding 12 (star_map)
    13, // engine slot 7 -> shader binding 13 (moon_map)
};
constexpr std::uint32_t kSlotToBufBinding[8] = {
    0,  // engine slot 0 unused
    3,  // engine slot 1 -> shader binding 3  (mesh_positions)
    4,  // engine slot 2 -> shader binding 4  (mesh_indices)
    5,  // engine slot 3 -> shader binding 5  (primitives)
    10, // engine slot 4 -> shader binding 10 (marginal_cdf)
    11, // engine slot 5 -> shader binding 11 (conditional_cdf)
    15, // engine slot 6 -> shader binding 15 (exposure_state)
    0,  // engine slot 7 unused
};
// Scene TLAS lives at engine accel-slot 2 -> shader binding 2.

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

bool DeviceSupportsExtension(VkPhysicalDevice pd, const char* name) {
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, props.data());
    for (auto& p : props) {
        if (std::strcmp(p.extensionName, name) == 0) return true;
    }
    return false;
}

}  // namespace

// =====================================================================
// VulkanCommandBuffer
// =====================================================================

void VulkanCommandBuffer::Reset(VkCommandBuffer cb) {
    cb_ = cb;
    bound_pipeline_ = PipelineHandle{0};
    push_size_ = 0;
    for (auto& t : bound_tex_)   t = TextureHandle{0};
    for (auto& b : bound_buf_)   b = BufferHandle{0};
    for (auto& o : bound_buf_off_) o = 0;
    for (auto& a : bound_accel_) a = AccelStructHandle{0};
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

void VulkanCommandBuffer::BindBuffer(std::uint32_t slot, BufferHandle b,
                                     std::size_t off) {
    if (slot < std::size(bound_buf_)) {
        bound_buf_[slot]     = b;
        bound_buf_off_[slot] = off;
    }
}

void VulkanCommandBuffer::BindAccelStruct(std::uint32_t slot, AccelStructHandle a) {
    if (slot < std::size(bound_accel_)) bound_accel_[slot] = a;
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

    VkDevice raw_dev = device_->RawDevice();
    auto dset = device_->CurrentDescriptorSet();

    // Build the full descriptor write list. We touch all 16 bindings on
    // every dispatch -- nullDescriptor (VK_EXT_robustness2), required at
    // device-create time for this path, means slots the engine didn't
    // bind get VK_NULL_HANDLE silently. The cost of re-writing every
    // binding each dispatch is small (16 small structs)
    // and avoids per-pipeline layout management.
    constexpr std::uint32_t kMaxWrites = 16;
    std::array<VkDescriptorImageInfo,  kMaxWrites> img_infos {};
    std::array<VkDescriptorBufferInfo, kMaxWrites> buf_infos {};
    std::array<VkWriteDescriptorSetAccelerationStructureKHR, 1> as_infos {};
    std::array<VkAccelerationStructureKHR, 1> as_handles {};
    std::array<VkWriteDescriptorSet, kMaxWrites> writes {};
    std::uint32_t wc = 0;

    auto add_image = [&](std::uint32_t binding, VkImageView v) {
        auto& ii = img_infos[wc];
        ii.imageView   = v;
        ii.imageLayout = (v == VK_NULL_HANDLE) ? VK_IMAGE_LAYOUT_UNDEFINED
                                               : VK_IMAGE_LAYOUT_GENERAL;
        auto& w = writes[wc];
        w = {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = dset;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo      = &ii;
        ++wc;
    };

    auto add_buffer = [&](std::uint32_t binding, VkBuffer b,
                          VkDeviceSize off, VkDeviceSize sz,
                          VkDescriptorType type) {
        auto& bi = buf_infos[wc];
        bi.buffer = b;
        // Null descriptors must use offset=0, range=VK_WHOLE_SIZE per
        // VUID-VkDescriptorBufferInfo-buffer-02999. Robustness2's
        // nullDescriptor feature lets the access produce zero, but the
        // descriptor write itself still must be spec-compliant.
        bi.offset = (b == VK_NULL_HANDLE) ? 0 : off;
        bi.range  = (b == VK_NULL_HANDLE) ? VK_WHOLE_SIZE : sz;
        auto& w = writes[wc];
        w = {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = dset;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = type;
        w.pBufferInfo     = &bi;
        ++wc;
    };

    auto add_accel = [&](std::uint32_t binding, VkAccelerationStructureKHR a) {
        as_handles[0] = a;
        auto& info = as_infos[0];
        info = {};
        info.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        info.accelerationStructureCount = 1;
        info.pAccelerationStructures    = &as_handles[0];
        auto& w = writes[wc];
        w = {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.pNext           = &info;
        w.dstSet          = dset;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        ++wc;
    };

    // ---- Storage images at bindings 0,1,6,7,8,9,12,13 -----------------
    for (std::uint32_t s = 0; s < std::size(bound_tex_); ++s) {
        VkImageView v = device_->LookupImageView(bound_tex_[s]);
        add_image(kSlotToTexBinding[s], v);
    }

    // ---- Acceleration structure at binding 2 --------------------------
    {
        auto a = device_->LookupAccel(bound_accel_[2]);
        add_accel(2, a);
    }

    // ---- Storage buffers at bindings 3,4,5,10,11,15 ------------------
    // Slot 6 -> binding 15 is exposure_state (GPU-driven auto-exposure
    // scalar; PathTrace reads, AutoExposure writes).
    for (std::uint32_t s = 1; s <= 6; ++s) {
        VkBuffer b = device_->LookupBuffer(bound_buf_[s]);
        VkDeviceSize sz = (b == VK_NULL_HANDLE) ? 0 : VK_WHOLE_SIZE;
        add_buffer(kSlotToBufBinding[s], b, bound_buf_off_[s], sz,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    // ---- Frame UBO at binding 14 (per-frame ring) ---------------------
    {
        const auto& ubo = device_->CurrentFrameUbo();
        // Copy spilled tail if push exceeds the on-chip range.
        if (push_size_ > VulkanDevice::kPushSplitOffset && ubo.mapped != nullptr) {
            std::size_t tail = push_size_ - VulkanDevice::kPushSplitOffset;
            if (tail > VulkanDevice::kFrameUboSize) tail = VulkanDevice::kFrameUboSize;
            std::memcpy(ubo.mapped,
                        push_buf_ + VulkanDevice::kPushSplitOffset, tail);
        }
        add_buffer(VulkanDevice::kFrameUboBinding, ubo.buffer, 0,
                   VulkanDevice::kFrameUboSize,
                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    }

    vkUpdateDescriptorSets(raw_dev, wc, writes.data(), 0, nullptr);
    vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &dset, 0, nullptr);

    // Push constants: send only the on-chip prefix. The remainder lives
    // in the Frame UBO uploaded above.
    if (push_size_ > 0) {
        std::uint32_t to_push = static_cast<std::uint32_t>(
            std::min<std::size_t>(push_size_, VulkanDevice::kPushSplitOffset));
        vkCmdPushConstants(cb_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           to_push, push_buf_);
    }
    vkCmdDispatch(cb_, gx, gy, gz);
}

void VulkanCommandBuffer::Barrier(const BarrierDesc& d) {
    if (cb_ == VK_NULL_HANDLE) return;

    // Translate the engine's coarse Stage enum to Vulkan stage / access
    // masks. We emit a global VkMemoryBarrier rather than enumerating
    // every resource handle: the caller's contract is "between these
    // pipeline stages, make writes visible to reads," which a global
    // memory barrier expresses cleanly without us tracking which
    // resource was last written by which dispatch.
    auto stage_to_vk = [](BarrierDesc::Stage s,
                          VkPipelineStageFlags& stage_mask,
                          VkAccessFlags& access_mask, bool is_dst) {
        switch (s) {
            case BarrierDesc::Stage::ComputeRead:
                stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                access_mask = VK_ACCESS_SHADER_READ_BIT;
                break;
            case BarrierDesc::Stage::ComputeWrite:
                stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                access_mask = is_dst ? (VK_ACCESS_SHADER_READ_BIT
                                      | VK_ACCESS_SHADER_WRITE_BIT)
                                     : VK_ACCESS_SHADER_WRITE_BIT;
                break;
            case BarrierDesc::Stage::Transfer:
                stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
                access_mask = is_dst ? VK_ACCESS_TRANSFER_WRITE_BIT
                                     : VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case BarrierDesc::Stage::Present:
                // Present-side barriers are handled by the swapchain
                // acquire/release dance, not by this generic path.
                stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                access_mask = 0;
                break;
        }
    };

    VkPipelineStageFlags src_stage  = 0;
    VkPipelineStageFlags dst_stage  = 0;
    VkAccessFlags        src_access = 0;
    VkAccessFlags        dst_access = 0;
    stage_to_vk(d.from, src_stage, src_access, /*is_dst=*/false);
    stage_to_vk(d.to,   dst_stage, dst_access, /*is_dst=*/true);

    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb_, src_stage, dst_stage,
                         /*dependencyFlags=*/0,
                         /*memoryBarrierCount=*/1, &mb,
                         /*bufferMemoryBarrierCount=*/0, nullptr,
                         /*imageMemoryBarrierCount=*/0, nullptr);
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
    ai.apiVersion       = VK_API_VERSION_1_3;

    std::uint32_t glfw_ext_n = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_n);
    std::vector<const char*> exts(glfw_exts, glfw_exts + glfw_ext_n);
#if defined(__APPLE__)
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

    // ---- Surface ------------------------------------------------------
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
    max_push_constant_size_ = props.limits.maxPushConstantsSize;
    LOG_INFO("Vulkan device: {}", device_name_);
    LOG_INFO("  maxPushConstantsSize: {}", max_push_constant_size_);

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
    // Probe for the optional extensions first; we'll degrade to "no
    // hardware RT" if the driver lacks them (e.g. very old card).
    bool has_ray_query        = DeviceSupportsExtension(phys_device_, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    bool has_accel_struct     = DeviceSupportsExtension(phys_device_, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    bool has_deferred_host_op = DeviceSupportsExtension(phys_device_, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    bool has_robustness2      = DeviceSupportsExtension(phys_device_, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    rt_supported_ = has_ray_query && has_accel_struct && has_deferred_host_op;

    // Query feature support before enabling anything in vkCreateDevice.
    VkPhysicalDeviceFeatures2 f2_supported{};
    f2_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan12Features v12_supported{};
    v12_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f2_supported.pNext = &v12_supported;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_supported{};
    as_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayQueryFeaturesKHR rq_supported{};
    rq_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceRobustness2FeaturesEXT rob2_supported{};
    rob2_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;

    void** supported_next = &v12_supported.pNext;
    auto chain_supported = [&](void* node, void** node_next) {
        *supported_next = node;
        supported_next = node_next;
    };
    if (rt_supported_) {
        chain_supported(&as_supported, reinterpret_cast<void**>(&as_supported.pNext));
        chain_supported(&rq_supported, reinterpret_cast<void**>(&rq_supported.pNext));
    }
    if (has_robustness2) {
        chain_supported(&rob2_supported, reinterpret_cast<void**>(&rob2_supported.pNext));
    }
    vkGetPhysicalDeviceFeatures2(phys_device_, &f2_supported);

    const bool supports_storage_image_rw_wo_format =
        f2_supported.features.shaderStorageImageReadWithoutFormat == VK_TRUE &&
        f2_supported.features.shaderStorageImageWriteWithoutFormat == VK_TRUE;
    const bool supports_uab_storage_image =
        v12_supported.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
    const bool supports_uab_storage_buffer =
        v12_supported.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
    const bool supports_uab_uniform_buffer =
        v12_supported.descriptorBindingUniformBufferUpdateAfterBind == VK_TRUE;
    const bool supports_update_after_bind =
        supports_uab_storage_image && supports_uab_storage_buffer && supports_uab_uniform_buffer;
    const bool supports_buffer_device_address = v12_supported.bufferDeviceAddress == VK_TRUE;
    const bool supports_null_descriptor = has_robustness2 && (rob2_supported.nullDescriptor == VK_TRUE);

    if (!supports_storage_image_rw_wo_format) {
        LOG_ERROR("Vulkan: missing shaderStorageImageRead/WriteWithoutFormat feature");
        return;
    }
    if (!supports_update_after_bind) {
        LOG_ERROR("Vulkan: UPDATE_AFTER_BIND features are required for shared descriptor-set dispatch");
        return;
    }
    if (!supports_null_descriptor) {
        LOG_ERROR("Vulkan: VK_EXT_robustness2 nullDescriptor is required by Vulkan descriptor binding strategy");
        return;
    }

    if (rt_supported_) {
        const bool supports_rt_features =
            supports_buffer_device_address &&
            as_supported.accelerationStructure == VK_TRUE &&
            as_supported.descriptorBindingAccelerationStructureUpdateAfterBind == VK_TRUE &&
            rq_supported.rayQuery == VK_TRUE;
        if (!supports_rt_features) {
            LOG_WARN("Vulkan: RT extensions present but required RT features are missing; disabling hardware RT");
            rt_supported_ = false;
        }
    }

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = graphics_qfi_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &qprio;

    std::vector<const char*> dexts;
    dexts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if defined(__APPLE__)
    dexts.push_back("VK_KHR_portability_subset");
#endif
    if (rt_supported_) {
        dexts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        dexts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        dexts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    if (supports_null_descriptor) {
        dexts.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    }

    // pNext feature chain. Hold each struct by value so the chain stays
    // valid for the duration of vkCreateDevice. Order in the chain
    // doesn't matter.
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.features.shaderStorageImageReadWithoutFormat  = supports_storage_image_rw_wo_format ? VK_TRUE : VK_FALSE;
    f2.features.shaderStorageImageWriteWithoutFormat = supports_storage_image_rw_wo_format ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = (rt_supported_ && supports_buffer_device_address) ? VK_TRUE : VK_FALSE;
    // UPDATE_AFTER_BIND for the shared descriptor set: lets us record
    // multiple Dispatch calls in one cmd buffer (path-trace then auto-
    // expose then bloom etc.) where each rewrites the descriptor set
    // with different bindings -- without invalidating earlier draws.
    v12.descriptorBindingStorageImageUpdateAfterBind  = supports_uab_storage_image ? VK_TRUE : VK_FALSE;
    v12.descriptorBindingStorageBufferUpdateAfterBind = supports_uab_storage_buffer ? VK_TRUE : VK_FALSE;
    v12.descriptorBindingUniformBufferUpdateAfterBind = supports_uab_uniform_buffer ? VK_TRUE : VK_FALSE;
    v12.descriptorBindingPartiallyBound               =
        (v12_supported.descriptorBindingPartiallyBound == VK_TRUE) ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features v13{};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_feat{};
    as_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_feat.accelerationStructure = VK_TRUE;
    as_feat.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_feat{};
    rq_feat.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_feat.rayQuery = VK_TRUE;

    VkPhysicalDeviceRobustness2FeaturesEXT rob2_feat{};
    rob2_feat.sType          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
    rob2_feat.nullDescriptor = VK_TRUE;

    void** next_slot = &f2.pNext;
    auto chain = [&](void* node, void** node_next) {
        *next_slot = node;
        next_slot = node_next;
    };
    chain(&v12, reinterpret_cast<void**>(&v12.pNext));
    chain(&v13, reinterpret_cast<void**>(&v13.pNext));
    if (rt_supported_) {
        chain(&as_feat, reinterpret_cast<void**>(&as_feat.pNext));
        chain(&rq_feat, reinterpret_cast<void**>(&rq_feat.pNext));
    }
    if (supports_null_descriptor) {
        chain(&rob2_feat, reinterpret_cast<void**>(&rob2_feat.pNext));
    }

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &f2;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(dexts.size());
    dci.ppEnabledExtensionNames = dexts.data();
    // pEnabledFeatures must be NULL when pNext->VkPhysicalDeviceFeatures2.
    dci.pEnabledFeatures        = nullptr;
    if (vkCreateDevice(phys_device_, &dci, nullptr, &device_) != VK_SUCCESS) {
        LOG_ERROR("vkCreateDevice failed");
        return;
    }
    vkGetDeviceQueue(device_, graphics_qfi_, 0, &graphics_queue_);

    // ---- Resolve extension function pointers -------------------------
    pfn_GetBufferDeviceAddr_ = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
        vkGetDeviceProcAddr(device_, "vkGetBufferDeviceAddress"));
    if (rt_supported_) {
        pfn_GetAccelStructBuildSizes_ = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(device_, "vkGetAccelerationStructureBuildSizesKHR"));
        pfn_CreateAccelStruct_ = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device_, "vkCreateAccelerationStructureKHR"));
        pfn_DestroyAccelStruct_ = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device_, "vkDestroyAccelerationStructureKHR"));
        pfn_CmdBuildAccelStructs_ = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(device_, "vkCmdBuildAccelerationStructuresKHR"));
        pfn_GetAccelStructAddr_ = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(device_, "vkGetAccelerationStructureDeviceAddressKHR"));
        if (pfn_CreateAccelStruct_ == nullptr || pfn_CmdBuildAccelStructs_ == nullptr) {
            LOG_WARN("Vulkan: accel-struct extensions enabled but procs missing; disabling RT");
            rt_supported_ = false;
        }
    }
    if (!rt_supported_) {
        LOG_WARN("Vulkan: hardware ray tracing unavailable on this driver");
    }

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

    // ---- Descriptor pool ---------------------------------------------
    // Sizing matches the expanded layout below:
    //   8 storage_image  + 1 accel_struct + 5 storage_buffer + 1 ubo
    //   per set, x kFramesInFlight sets, with headroom.
    std::array<VkDescriptorPoolSize, 4> psizes{};
    psizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           kFramesInFlight * 8 + 4 };
    psizes[1] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kFramesInFlight * 1 + 1 };
    psizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          kFramesInFlight * 6 + 4 };
    psizes[3] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          kFramesInFlight * 1 + 1 };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = kFramesInFlight;
    dpci.poolSizeCount = static_cast<std::uint32_t>(psizes.size());
    dpci.pPoolSizes    = psizes.data();
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                       | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool_);

    // ---- Build the unified 16-binding descriptor set layout ----------
    {
        std::array<VkDescriptorSetLayoutBinding, 16> b{};
        auto fill = [&](std::uint32_t idx, std::uint32_t binding,
                        VkDescriptorType type) {
            b[idx].binding         = binding;
            b[idx].descriptorType  = type;
            b[idx].descriptorCount = 1;
            b[idx].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        // bindings 0-1: output + accum (storage image)
        fill(0,  0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        fill(1,  1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 2: scene_tlas
        fill(2,  2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        // bindings 3-5: mesh_positions, mesh_indices, primitives (storage buffer)
        fill(3,  3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        fill(4,  4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        fill(5,  5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // bindings 6-9: denoise_color, depth_tex, motion_tex, env_map (storage image)
        fill(6,  6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        fill(7,  7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        fill(8,  8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        fill(9,  9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // bindings 10-11: marginal_cdf, conditional_cdf (storage buffer)
        fill(10, 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        fill(11, 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // bindings 12-13: star_map, moon_map (storage image)
        fill(12, 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        fill(13, 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 14: Frame UBO (spilled push tail)
        fill(14, kFrameUboBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        // binding 15: exposure_state (GPU-driven auto-exposure scalar)
        fill(15, 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        // UPDATE_AFTER_BIND for every binding so we can rewrite the
        // shared descriptor set between dispatches in the same cmd
        // buffer without invalidating earlier recorded draws.
        std::array<VkDescriptorBindingFlags, 16> bind_flags{};
        for (auto& f : bind_flags) f = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{};
        bfci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bfci.bindingCount  = static_cast<std::uint32_t>(bind_flags.size());
        bfci.pBindingFlags = bind_flags.data();

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.pNext        = &bfci;
        dslci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        dslci.bindingCount = static_cast<std::uint32_t>(b.size());
        dslci.pBindings    = b.data();
        if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr,
                                        &shared_dset_layout_) != VK_SUCCESS) {
            LOG_ERROR("vkCreateDescriptorSetLayout (16-binding) failed");
            return;
        }

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = std::min<std::uint32_t>(kPushSplitOffset, max_push_constant_size_);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &shared_dset_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(device_, &plci, nullptr,
                                   &shared_pipe_layout_) != VK_SUCCESS) {
            LOG_ERROR("vkCreatePipelineLayout failed");
            return;
        }

        std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, shared_dset_layout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = kFramesInFlight;
        dsai.pSetLayouts        = layouts.data();
        vkAllocateDescriptorSets(device_, &dsai, dsets_);
    }

    // ---- Per-frame Frame UBO ring ------------------------------------
    for (int i = 0; i < kFramesInFlight; ++i) {
        if (!CreateBufferImpl(kFrameUboSize,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              frame_ubos_[i],
                              /*persistent_map=*/true)) {
            LOG_ERROR("Vulkan: failed to allocate Frame UBO {}", i);
            return;
        }
    }

    // ---- Async readback fences + cmd buffers -------------------------
    {
        VkFenceCreateInfo rfci{};
        rfci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkCommandBufferAllocateInfo rcbai{};
        rcbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        rcbai.commandPool        = cmd_pool_;
        rcbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        rcbai.commandBufferCount = 1;
        for (auto& slot : readback_slots_) {
            vkCreateFence(device_, &rfci, nullptr, &slot.fence);
            vkAllocateCommandBuffers(device_, &rcbai, &slot.cmd);
        }
    }

    // ---- Build the path-trace pipeline -------------------------------
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

    build_pipeline("pathtrace",  shader_PathTrace_spirv_data,   shader_PathTrace_spirv_size);
    build_pipeline("autoexpose", shader_AutoExposure_spirv_data, shader_AutoExposure_spirv_size);

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
            DestroyBufferImpl(frame_ubos_[i]);
        }
        for (auto& slot : readback_slots_) {
            if (slot.in_flight) vkWaitForFences(device_, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            if (slot.cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(device_, cmd_pool_, 1, &slot.cmd);
            if (slot.fence != VK_NULL_HANDLE) vkDestroyFence(device_, slot.fence, nullptr);
            DestroyBufferImpl(slot.staging);
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
            for (auto& [id, b] : buffers_) DestroyBufferImpl(b);
            buffers_.clear();
            for (auto& [id, a] : accels_) {
                if (a.accel != VK_NULL_HANDLE && pfn_DestroyAccelStruct_) {
                    pfn_DestroyAccelStruct_(device_, a.accel, nullptr);
                }
                if (a.buffer) vkDestroyBuffer(device_, a.buffer, nullptr);
                if (a.memory) vkFreeMemory(device_, a.memory, nullptr);
            }
            accels_.clear();
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

VkBuffer VulkanDevice::LookupBuffer(BufferHandle h) {
    if (h.id == 0) return VK_NULL_HANDLE;
    std::lock_guard lock(resource_mutex_);
    auto it = buffers_.find(h.id);
    return (it == buffers_.end()) ? VK_NULL_HANDLE : it->second.buffer;
}

VkAccelerationStructureKHR VulkanDevice::LookupAccel(AccelStructHandle h) {
    if (h.id == 0) return VK_NULL_HANDLE;
    std::lock_guard lock(resource_mutex_);
    auto it = accels_.find(h.id);
    return (it == accels_.end()) ? VK_NULL_HANDLE : it->second.accel;
}

// ---- Memory / buffer helpers --------------------------------------------

std::uint32_t VulkanDevice::FindMemoryType(std::uint32_t bits,
                                            VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mp);

    // When the caller wants HOST_VISIBLE (typical for storage buffers
    // we memcpy into), prefer a heap that's also DEVICE_LOCAL. On
    // modern NVIDIA dGPUs with Resizable BAR enabled (most boards
    // since RTX 30, basically all RTX 50) this maps directly to VRAM,
    // so shader reads run at ~1 TB/s instead of crawling across PCIe
    // at ~32 GB/s. Without this preference the first-match loop picks
    // system-RAM-only host-visible memory and the path tracer's inner
    // loop becomes PCIe-bound -- a 5-10x perf cliff on a 5090.
    if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        VkMemoryPropertyFlags preferred = props
            | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & preferred) == preferred) {
                return i;
            }
        }
    }
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool VulkanDevice::CreateBufferImpl(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags props,
                                    BufferEntry& out,
                                    bool persistent_map) {
    out = {};
    if (size == 0) return false;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device_, out.buffer, &mr);
    std::uint32_t mt = FindMemoryType(mr.memoryTypeBits, props);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext           = (flagsInfo.flags != 0) ? &flagsInfo : nullptr;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(device_, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device_, out.buffer, out.memory, 0);
    out.size = size;

    if (persistent_map && (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        vkMapMemory(device_, out.memory, 0, size, 0, &out.mapped);
    }
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) && pfn_GetBufferDeviceAddr_) {
        VkBufferDeviceAddressInfo bdai{};
        bdai.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = out.buffer;
        out.device_address = pfn_GetBufferDeviceAddr_(device_, &bdai);
    }
    return true;
}

void VulkanDevice::DestroyBufferImpl(BufferEntry& e) {
    if (device_ == VK_NULL_HANDLE) { e = {}; return; }
    if (e.mapped) vkUnmapMemory(device_, e.memory);
    if (e.buffer) vkDestroyBuffer(device_, e.buffer, nullptr);
    if (e.memory) vkFreeMemory(device_, e.memory, nullptr);
    e = {};
}

// ---- Public Buffer API --------------------------------------------------

BufferHandle VulkanDevice::CreateBuffer(const BufferDesc& d) {
    if (device_ == VK_NULL_HANDLE || d.size == 0) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    // Engine storage buffers (primitives, mesh data, env CDFs) are
    // pure DEVICE_LOCAL -- shader reads run at full VRAM bandwidth
    // (~1.79 TB/s on the 5090) with no PCIe / cache-coherency
    // bookkeeping. Host writes go through a transient staging buffer
    // in WriteBuffer below; that's a slow path but called only at
    // upload time, not per frame.
    BufferEntry e{};
    if (!CreateBufferImpl(d.size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          e,
                          /*persistent_map=*/false)) {
        LOG_ERROR("Vulkan CreateBuffer({} bytes, '{}') failed", d.size, d.debug_name);
        return {0};
    }
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    buffers_.emplace(id, e);
    return BufferHandle{id};
}

void VulkanDevice::DestroyBuffer(BufferHandle h) {
    if (h.id == 0 || device_ == VK_NULL_HANDLE) return;
    std::lock_guard lock(resource_mutex_);
    auto it = buffers_.find(h.id);
    if (it == buffers_.end()) return;
    vkDeviceWaitIdle(device_);
    DestroyBufferImpl(it->second);
    buffers_.erase(it);
}

void VulkanDevice::WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                               std::size_t dst_offset) {
    if (src == nullptr || size == 0 || device_ == VK_NULL_HANDLE) return;
    VkBuffer dst_buf = VK_NULL_HANDLE;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = buffers_.find(h.id);
        if (it == buffers_.end()) return;
        dst_buf = it->second.buffer;
    }
    if (dst_buf == VK_NULL_HANDLE) return;

    // Stage CPU bytes through a transient host-visible buffer, then
    // vkCmdCopyBuffer to the device-local destination. Synchronous
    // submit-and-wait -- WriteBuffer is called at scene-load time
    // (mesh upload, env CDF upload) not in the per-frame loop, so
    // the stall cost is fine.
    BufferEntry staging{};
    if (!CreateBufferImpl(size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging,
                          /*persistent_map=*/true)) {
        LOG_ERROR("Vulkan WriteBuffer: staging alloc failed ({} bytes)", size);
        return;
    }
    std::memcpy(staging.mapped, src, size);

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

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size      = size;
    vkCmdCopyBuffer(once, staging.buffer, dst_buf, 1, &region);

    vkEndCommandBuffer(once);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &once;
    vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &once);
    DestroyBufferImpl(staging);
}

// ---- Texture creation (unchanged from P4 baseline) ----------------------

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
    std::uint32_t mt = FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) {
        vkDestroyImage(device_, img, nullptr);
        return {0};
    }
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = mt;
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

bool VulkanDevice::WriteTexture(TextureHandle h, const void* src,
                                std::size_t src_size) {
    if (src == nullptr || src_size == 0 || device_ == VK_NULL_HANDLE) return false;

    VkImage img = VK_NULL_HANDLE;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    VkExtent2D extent { 0, 0 };
    {
        std::lock_guard lock(resource_mutex_);
        auto it = images_.find(h.id);
        if (it == images_.end()) return false;
        img    = it->second.image;
        fmt    = it->second.format;
        extent = it->second.extent;
    }
    if (img == VK_NULL_HANDLE) return false;

    std::size_t bpp = 0;
    switch (fmt) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: bpp = 16; break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: bpp = 8;  break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:       bpp = 4;  break;
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R16G16_SFLOAT:       bpp = 4;  break;
        default: return false;
    }
    const std::size_t expected = std::size_t(extent.width) * extent.height * bpp;
    if (src_size != expected) return false;

    // Stage CPU bytes through a host-visible buffer, then vkCmdCopyBufferToImage.
    BufferEntry staging{};
    if (!CreateBufferImpl(src_size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging,
                          /*persistent_map=*/true)) {
        LOG_ERROR("Vulkan WriteTexture: staging buffer alloc failed ({} bytes)", src_size);
        return false;
    }
    std::memcpy(staging.mapped, src, src_size);

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

    auto barrier = [&](VkImageLayout from, VkImageLayout to,
                       VkAccessFlags src_acc, VkAccessFlags dst_acc,
                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout     = from;
        b.newLayout     = to;
        b.srcAccessMask = src_acc;
        b.dstAccessMask = dst_acc;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image         = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.layerCount = 1;
        b.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(once, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // CreateTexture left the image in GENERAL. Transition GENERAL ->
    // TRANSFER_DST for the copy, then back to GENERAL so it's storage-
    // image-ready for the compute kernel. Once-shot resource upload, no
    // need to reason about the pipeline barrier graph.
    barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { extent.width, extent.height, 1 };
    vkCmdCopyBufferToImage(once, staging.buffer, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkEndCommandBuffer(once);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &once;
    vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &once);
    DestroyBufferImpl(staging);
    return true;
}

void VulkanDevice::PollReadbacks() {
    for (auto& slot : readback_slots_) {
        if (slot.in_flight && slot.fence != VK_NULL_HANDLE &&
            vkGetFenceStatus(device_, slot.fence) == VK_SUCCESS) {
            slot.in_flight  = false;
            slot.data_ready = true;
        }
    }
}

bool VulkanDevice::SubmitReadback(ReadbackSlot& slot, VkImage img, VkFormat fmt,
                                  VkExtent2D extent, std::uint64_t src_id,
                                  std::size_t bytes, std::size_t /*bpp*/) {
    (void)fmt;
    // Grow staging buffer if needed (or first-time alloc).
    if (slot.staging.buffer == VK_NULL_HANDLE || slot.staging.size < bytes) {
        if (slot.staging.buffer != VK_NULL_HANDLE) DestroyBufferImpl(slot.staging);
        if (!CreateBufferImpl(bytes,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              slot.staging,
                              /*persistent_map=*/true)) {
            return false;
        }
    }

    vkResetFences(device_, 1, &slot.fence);
    vkResetCommandBuffer(slot.cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(slot.cmd, &bi);

    auto barrier = [&](VkImageLayout from, VkImageLayout to,
                       VkAccessFlags src_acc, VkAccessFlags dst_acc,
                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout     = from;
        b.newLayout     = to;
        b.srcAccessMask = src_acc;
        b.dstAccessMask = dst_acc;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image         = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.layerCount = 1;
        b.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(slot.cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { extent.width, extent.height, 1 };
    vkCmdCopyImageToBuffer(slot.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           slot.staging.buffer, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkEndCommandBuffer(slot.cmd);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &slot.cmd;
    vkQueueSubmit(graphics_queue_, 1, &si, slot.fence);

    slot.in_flight  = true;
    slot.data_ready = false;
    slot.src_id     = src_id;
    slot.width      = extent.width;
    slot.height     = extent.height;
    slot.bytes      = bytes;
    return true;
}

bool VulkanDevice::ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                                   std::uint32_t* out_w, std::uint32_t* out_h) {
    if (dst == nullptr || dst_size == 0 || device_ == VK_NULL_HANDLE) return false;

    VkImage img = VK_NULL_HANDLE;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    VkExtent2D extent { 0, 0 };
    {
        std::lock_guard lock(resource_mutex_);
        auto it = images_.find(h.id);
        if (it == images_.end()) return false;
        img    = it->second.image;
        fmt    = it->second.format;
        extent = it->second.extent;
    }
    if (img == VK_NULL_HANDLE) return false;

    std::size_t bpp = 0;
    switch (fmt) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: bpp = 16; break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: bpp = 8;  break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:       bpp = 4;  break;
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R16G16_SFLOAT:       bpp = 4;  break;
        default: return false;
    }
    const std::size_t bytes = std::size_t(extent.width) * extent.height * bpp;
    if (dst_size < bytes) return false;

    PollReadbacks();

    // Diagnostic: track cache vs sync stats so we can tell whether
    // the async ring is actually serving cache hits or constantly
    // falling back to the sync wait. Logged every 30 calls.
    static std::uint32_t s_dbg_calls = 0;
    static std::uint32_t s_dbg_cache = 0;
    static std::uint32_t s_dbg_sync  = 0;
    static std::uint32_t s_dbg_fail  = 0;
    ++s_dbg_calls;

    // Step 1: serve from the ring if any slot has matching ready data.
    ReadbackSlot* served = nullptr;
    for (auto& slot : readback_slots_) {
        if (slot.data_ready && slot.src_id == h.id && slot.bytes == bytes) {
            std::memcpy(dst, slot.staging.mapped, bytes);
            if (out_w) *out_w = slot.width;
            if (out_h) *out_h = slot.height;
            slot.data_ready = false;  // consumed; slot is now reusable
            served = &slot;
            ++s_dbg_cache;
            break;
        }
    }

    // Step 2: find an idle slot to issue a fresh async copy into.
    ReadbackSlot* idle = nullptr;
    for (auto& slot : readback_slots_) {
        if (!slot.in_flight && !slot.data_ready) { idle = &slot; break; }
    }
    if (idle == nullptr) {
        // All in-flight; pick the slot we just consumed if we served, else
        // any non-in-flight (overwriting an unconsumed ready slot is OK,
        // it's stale data).
        if (served != nullptr) idle = served;
        else for (auto& slot : readback_slots_) {
            if (!slot.in_flight) { idle = &slot; break; }
        }
    }

    if (idle != nullptr) {
        SubmitReadback(*idle, img, fmt, extent, h.id, bytes, bpp);
    }

    if (served != nullptr) {
        if (s_dbg_calls % 30 == 0) {
            LOG_INFO("readback: calls={} cache={} sync={} fail={}",
                     s_dbg_calls, s_dbg_cache, s_dbg_sync, s_dbg_fail);
        }
        return true;
    }

    // Cold start / different texture: fall back to a synchronous wait
    // on the slot we just submitted. One stutter, then the ring stays
    // primed for subsequent calls.
    if (idle == nullptr) {
        ++s_dbg_fail;
        if (s_dbg_calls % 30 == 0) {
            LOG_INFO("readback: calls={} cache={} sync={} fail={}",
                     s_dbg_calls, s_dbg_cache, s_dbg_sync, s_dbg_fail);
        }
        return false;
    }
    ++s_dbg_sync;
    vkWaitForFences(device_, 1, &idle->fence, VK_TRUE, UINT64_MAX);
    idle->in_flight  = false;
    idle->data_ready = false;       // consumed in this call
    std::memcpy(dst, idle->staging.mapped, bytes);
    if (out_w) *out_w = idle->width;
    if (out_h) *out_h = idle->height;

    // Critical: also submit a FRESH async readback into a different
    // slot so the next call has data cached and doesn't re-stutter.
    // Without this, every tick falls into the sync fallback above
    // (resetting all state) and the ring never gets to act as a ring.
    for (auto& slot : readback_slots_) {
        if (&slot == idle) continue;
        if (!slot.in_flight && !slot.data_ready) {
            SubmitReadback(slot, img, fmt, extent, h.id, bytes, bpp);
            break;
        }
    }
    if (s_dbg_calls % 30 == 0) {
        LOG_INFO("readback: calls={} cache={} sync={} fail={}",
                 s_dbg_calls, s_dbg_cache, s_dbg_sync, s_dbg_fail);
    }
    return true;
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

// ---- Acceleration structures --------------------------------------------

bool VulkanDevice::BuildAccelerationStructure(
    VkAccelerationStructureBuildGeometryInfoKHR& build_info,
    const VkAccelerationStructureBuildRangeInfoKHR* range,
    AccelEntry& entry,
    VkAccelerationStructureTypeKHR type,
    VkDeviceSize as_size,
    VkDeviceSize scratch_size) {

    // 1. Storage buffer for the acceleration structure itself.
    BufferEntry storage{};
    if (!CreateBufferImpl(as_size,
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          storage,
                          /*persistent_map=*/false)) {
        LOG_ERROR("Vulkan: failed to allocate AS storage ({} bytes)", as_size);
        return false;
    }

    VkAccelerationStructureCreateInfoKHR aci{};
    aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    aci.buffer = storage.buffer;
    aci.size   = as_size;
    aci.type   = type;
    if (pfn_CreateAccelStruct_(device_, &aci, nullptr, &entry.accel) != VK_SUCCESS) {
        DestroyBufferImpl(storage);
        LOG_ERROR("Vulkan: vkCreateAccelerationStructureKHR failed");
        return false;
    }

    // 2. Scratch buffer for the build itself (transient).
    BufferEntry scratch{};
    if (!CreateBufferImpl(scratch_size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          scratch,
                          /*persistent_map=*/false)) {
        pfn_DestroyAccelStruct_(device_, entry.accel, nullptr);
        DestroyBufferImpl(storage);
        return false;
    }

    build_info.dstAccelerationStructure  = entry.accel;
    build_info.scratchData.deviceAddress = scratch.device_address;

    // 3. One-shot command buffer for the build.
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
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[1] = { range };
    pfn_CmdBuildAccelStructs_(once, 1, &build_info, ranges);
    vkEndCommandBuffer(once);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &once;
    vkQueueSubmit(graphics_queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &once);

    // 4. Free scratch (no longer needed); keep storage with the entry.
    DestroyBufferImpl(scratch);
    entry.buffer = storage.buffer;
    entry.memory = storage.memory;

    VkAccelerationStructureDeviceAddressInfoKHR adi{};
    adi.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    adi.accelerationStructure = entry.accel;
    entry.device_address = pfn_GetAccelStructAddr_(device_, &adi);
    return true;
}

AccelStructHandle VulkanDevice::CreateBLAS(const BLASDesc& d) {
    if (!rt_supported_ || d.vertex_count == 0 || d.index_count == 0) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    // Upload vertex + index data into device-local AS-input buffers.
    VkDeviceSize vbytes = sizeof(float) * 3 * d.vertex_count;
    VkDeviceSize ibytes = sizeof(std::uint32_t) * d.index_count;
    BufferEntry vbuf{}, ibuf{};
    constexpr VkBufferUsageFlags as_input_usage =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags host_props =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateBufferImpl(vbytes, as_input_usage, host_props, vbuf, /*persistent_map=*/true) ||
        !CreateBufferImpl(ibytes, as_input_usage, host_props, ibuf, /*persistent_map=*/true)) {
        DestroyBufferImpl(vbuf);
        DestroyBufferImpl(ibuf);
        LOG_ERROR("Vulkan CreateBLAS: AS-input buffer alloc failed");
        return {0};
    }
    std::memcpy(vbuf.mapped, d.vertex_positions, vbytes);
    std::memcpy(ibuf.mapped, d.indices,          ibytes);

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress = vbuf.device_address;
    geom.geometry.triangles.vertexStride  = sizeof(float) * 3;
    geom.geometry.triangles.maxVertex     = d.vertex_count - 1;
    geom.geometry.triangles.indexType     = VK_INDEX_TYPE_UINT32;
    geom.geometry.triangles.indexData.deviceAddress = ibuf.device_address;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geom;

    std::uint32_t prim_count = d.index_count / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_GetAccelStructBuildSizes_(device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &prim_count, &sizes);

    AccelEntry entry{};
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = prim_count;
    if (!BuildAccelerationStructure(build_info, &range, entry,
                                    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                    sizes.accelerationStructureSize,
                                    sizes.buildScratchSize)) {
        DestroyBufferImpl(vbuf);
        DestroyBufferImpl(ibuf);
        return {0};
    }
    // BLAS contains its own copy now; AS-input buffers can go.
    DestroyBufferImpl(vbuf);
    DestroyBufferImpl(ibuf);

    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    accels_.emplace(id, entry);
    return AccelStructHandle{ id };
}

AccelStructHandle VulkanDevice::CreateTLAS(const TLASDesc& d) {
    if (!rt_supported_ || d.instances.empty()) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    // Build the instance buffer.
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(d.instances.size());
    for (const auto& inst : d.instances) {
        VkAccelerationStructureKHR blas = LookupAccel(inst.blas);
        if (blas == VK_NULL_HANDLE) continue;
        VkAccelerationStructureDeviceAddressInfoKHR adi{};
        adi.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        adi.accelerationStructure = blas;
        VkDeviceAddress blas_addr = pfn_GetAccelStructAddr_(device_, &adi);

        VkAccelerationStructureInstanceKHR vi{};
        // VkTransformMatrixKHR is row-major 3x4 (3 rows, 4 cols), matches
        // our TLASInstance.transform layout exactly: 12 floats, row 0 first.
        std::memcpy(&vi.transform, inst.transform, sizeof(vi.transform));
        vi.instanceCustomIndex                    = inst.instance_id & 0xFFFFFF;
        vi.mask                                   = inst.mask;
        vi.instanceShaderBindingTableRecordOffset = 0;
        vi.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        vi.accelerationStructureReference         = blas_addr;
        instances.push_back(vi);
    }
    if (instances.empty()) return {0};
    VkDeviceSize ibytes = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();

    BufferEntry inst_buf{};
    if (!CreateBufferImpl(ibytes,
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          inst_buf,
                          /*persistent_map=*/true)) {
        LOG_ERROR("Vulkan CreateTLAS: instance buffer alloc failed");
        return {0};
    }
    std::memcpy(inst_buf.mapped, instances.data(), ibytes);

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.data.deviceAddress = inst_buf.device_address;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geom;

    std::uint32_t prim_count = static_cast<std::uint32_t>(instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_GetAccelStructBuildSizes_(device_,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &prim_count, &sizes);

    AccelEntry entry{};
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = prim_count;
    if (!BuildAccelerationStructure(build_info, &range, entry,
                                    VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                    sizes.accelerationStructureSize,
                                    sizes.buildScratchSize)) {
        DestroyBufferImpl(inst_buf);
        return {0};
    }
    DestroyBufferImpl(inst_buf);

    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    accels_.emplace(id, entry);
    return AccelStructHandle{ id };
}

void VulkanDevice::DestroyAccelStruct(AccelStructHandle h) {
    if (h.id == 0 || device_ == VK_NULL_HANDLE) return;
    std::lock_guard lock(resource_mutex_);
    auto it = accels_.find(h.id);
    if (it == accels_.end()) return;
    vkDeviceWaitIdle(device_);
    if (it->second.accel != VK_NULL_HANDLE && pfn_DestroyAccelStruct_) {
        pfn_DestroyAccelStruct_(device_, it->second.accel, nullptr);
    }
    if (it->second.buffer) vkDestroyBuffer(device_, it->second.buffer, nullptr);
    if (it->second.memory) vkFreeMemory(device_, it->second.memory, nullptr);
    accels_.erase(it);
}

// ---- Frame loop ---------------------------------------------------------

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

    wrapped_cb_->Reset(cb);
    return wrapped_cb_.get();
}

void VulkanDevice::Submit(CommandBuffer* cb) {
    if (cb == nullptr || device_ == VK_NULL_HANDLE) return;
    auto* vcb = static_cast<VulkanCommandBuffer*>(cb);
    VkCommandBuffer cmd = vcb->Raw();
    if (cmd == VK_NULL_HANDLE) return;

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
    return 0;
}

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle& w) {
    return std::make_unique<vk::VulkanDevice>(w);
}
}  // namespace pt::rhi

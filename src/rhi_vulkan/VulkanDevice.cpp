// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Vulkan backend, P12 expansion: ray-query, acceleration structures,
// real CreateBuffer / WriteBuffer / CreateBLAS / CreateTLAS, expanded
// descriptor set layout (16 bindings) matching the unified PathTrace
// shader, and per-frame UBO ring for the spilled push-constant tail.

#include "VulkanDevice.h"
#include "VulkanDenoiser.h"
#if defined(PT_ENABLE_OPTIX)
#include "VulkanOptixDenoiser.h"
#endif

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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
extern const unsigned char shader_PathTrace_spirv_data[];
extern const unsigned long shader_PathTrace_spirv_size;
extern const unsigned char shader_AutoExposure_spirv_data[];
extern const unsigned long shader_AutoExposure_spirv_size;
extern const unsigned char shader_PerfOverlay_spirv_data[];
extern const unsigned long shader_PerfOverlay_spirv_size;
}

namespace pt::rhi::vk {

namespace {

#ifndef NDEBUG
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

// Engine slot -> shader vk::binding translation. The unified descriptor
// set has 17 bindings (0-16): storage_image x9 (output / accum_hdr /
// denoise_color / depth / motion / env_map / star_map / moon_map /
// normal_tex), AS x1 (scene_tlas), storage_buffer x6 (mesh / cdf /
// exposure_state), uniform_buffer x1 (Frame UBO at binding 14).
// Tonemap / Bloom* re-use a subset. Engine.cpp uses up-to-9-element
// textures, 8-element buffers, and 4-element accel-struct slot arrays
// as if Metal-style argument tables; we flatten these into the single
// Vulkan descriptor set here. Engine slot 8 (normal_tex / vk::binding 16)
// is Vulkan-only -- only the SVGF/NRD denoiser path tracer writes it.
static constexpr std::uint32_t kNumTexSlots = 9;
constexpr std::uint32_t kSlotToTexBinding[kNumTexSlots] = {
    0,  // engine slot 0 -> shader binding 0  (output / swapchain)
    1,  // engine slot 1 -> shader binding 1  (accum_hdr)
    6,  // engine slot 2 -> shader binding 6  (denoise_color)
    7,  // engine slot 3 -> shader binding 7  (depth_tex)
    8,  // engine slot 4 -> shader binding 8  (motion_tex)
    9,  // engine slot 5 -> shader binding 9  (env_map)
    12, // engine slot 6 -> shader binding 12 (star_map)
    13, // engine slot 7 -> shader binding 13 (moon_map)
    16, // engine slot 8 -> shader binding 16 (normal_tex, SVGF/NRD only)
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

void VulkanCommandBuffer::ClearStorageTexture(TextureHandle t, const float rgba[4]) {
    if (cb_ == VK_NULL_HANDLE) return;
    VkImage img = device_->LookupImage(t);
    if (img == VK_NULL_HANDLE) return;
    VkImageSubresourceRange range{};
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = 0;
    range.levelCount     = 1;
    range.baseArrayLayer = 0;
    range.layerCount     = 1;
    // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL. UNDEFINED is the
    // safe pessimistic source -- caller's contract is "this image
    // doesn't need its current contents preserved", which matches the
    // loading-frame use case (just-acquired swapchain image).
    VkImageMemoryBarrier b{};
    b.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image                = img;
    b.subresourceRange     = range;
    b.srcAccessMask        = 0;
    b.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cb_,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
    VkClearColorValue cv{};
    cv.float32[0] = rgba[0];
    cv.float32[1] = rgba[1];
    cv.float32[2] = rgba[2];
    cv.float32[3] = rgba[3];
    vkCmdClearColorImage(cb_, img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &cv, 1, &range);
    // Leave in GENERAL so subsequent shader dispatches in the same
    // frame (e.g. if pipelines come ready mid-frame and the engine
    // chooses to render on top) see the image in its standard
    // storage-texture layout. EndFrame's transition to PRESENT_SRC
    // takes over from there.
    b.oldLayout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout      = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask  = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb_,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanCommandBuffer::Dispatch(std::uint32_t gx, std::uint32_t gy,
                                   std::uint32_t gz) {
    if (cb_ == VK_NULL_HANDLE || !bound_pipeline_) return;
    VkPipelineLayout layout = device_->SharedPipelineLayout();
    if (layout == VK_NULL_HANDLE) return;

    VkDevice raw_dev = device_->RawDevice();
    auto dset = device_->CurrentDescriptorSet();

    // Build the full descriptor write list. We touch all 17 bindings on
    // every dispatch -- nullDescriptor (VK_EXT_robustness2), required at
    // device-create time for this path, means slots the engine didn't
    // bind get VK_NULL_HANDLE silently. The cost of re-writing every
    // binding each dispatch is small (17 small structs)
    // and avoids per-pipeline layout management.
    constexpr std::uint32_t kMaxWrites = 17;
    std::array<VkDescriptorImageInfo,  kMaxWrites> img_infos {};
    std::array<VkDescriptorBufferInfo, kMaxWrites> buf_infos {};
    std::array<VkWriteDescriptorSetAccelerationStructureKHR, 1> as_infos {};
    std::array<VkAccelerationStructureKHR, 1> as_handles {};
    std::array<VkWriteDescriptorSet, kMaxWrites> writes {};
    std::uint32_t wc = 0;

    auto add_image = [&](std::uint32_t binding, VkImageView v) {
        auto& ii = img_infos[wc];
        ii.imageView   = v;
        // Use VK_IMAGE_LAYOUT_GENERAL for both bound and null storage-image
        // descriptors. Validation layers expect a real, non-UNDEFINED
        // layout for image descriptor writes -- robustness2 nullDescriptor
        // makes the access produce zero, but the descriptor write itself
        // still has to be spec-compliant. A stray UNDEFINED triggered
        // VUID-VkWriteDescriptorSet errors on some drivers.
        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
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

    // ---- Storage images at bindings 0,1,6,7,8,9,12,13,16 --------------
    // Iterate exactly kNumTexSlots even though bound_tex_ has spares;
    // those spares aren't in kSlotToTexBinding so we can't translate
    // them, and the unused vk::binding slots in the layout would still
    // be written nullDescriptor by the AS / buffer / UBO loops below.
    for (std::uint32_t s = 0; s < kNumTexSlots; ++s) {
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
    // in the Frame UBO uploaded above. Clamp to the device's actual
    // pipeline-layout push range so we don't try to push more than the
    // layout declares (max_push_constant_size_ may be < kPushSplitOffset
    // on devices with a tighter VkPhysicalDeviceLimits::maxPushConstantsSize,
    // or future increases of kPushSplitOffset would otherwise trigger
    // VUID-vkCmdPushConstants-offset-01795).
    if (push_size_ > 0) {
        const std::uint32_t layout_push_max = std::min<std::uint32_t>(
            VulkanDevice::kPushSplitOffset, device_->MaxPushConstantSize());
        std::uint32_t to_push = static_cast<std::uint32_t>(
            std::min<std::size_t>(push_size_, layout_push_max));
        if (to_push > 0) {
            vkCmdPushConstants(cb_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               to_push, push_buf_);
        }
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
                // Cover both directions on Transfer because the engine's
                // coarse Stage enum doesn't distinguish "transfer wrote
                // (vkCmdCopy*)" from "transfer will read" -- and the
                // most common Transfer->* barrier in this engine is
                // upload-then-shader-read, which needs srcAccess to
                // include TRANSFER_WRITE_BIT or the visibility step
                // is a no-op. Allowing both bits in both directions
                // is the conservative cross-direction fix.
                access_mask = VK_ACCESS_TRANSFER_READ_BIT
                            | VK_ACCESS_TRANSFER_WRITE_BIT;
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
#if defined(PT_ENABLE_OPTIX)
    // CUDA-Vulkan interop for the OptiX denoiser. The base
    // VK_KHR_external_memory / VK_KHR_external_semaphore are core 1.1
    // (already implicitly available through VkPhysicalDeviceVulkan11/12),
    // but the platform-specific export variants must still be requested
    // explicitly -- they're what supplies vkGetMemoryWin32HandleKHR /
    // vkGetSemaphoreWin32HandleKHR (or the _fd flavours on Linux),
    // which VulkanOptixDenoiser uses to hand VkDeviceMemory and
    // VkSemaphore handles to the CUDA runtime.
    //
    // All NVIDIA RTX-class GPUs / drivers expose these unconditionally,
    // so we don't query support before enabling -- a missing extension
    // would be a fundamental driver-bug situation we can't recover from.
    #if defined(_WIN32)
        dexts.push_back("VK_KHR_external_memory_win32");
        dexts.push_back("VK_KHR_external_semaphore_win32");
    #elif defined(__linux__)
        dexts.push_back("VK_KHR_external_memory_fd");
        dexts.push_back("VK_KHR_external_semaphore_fd");
    #endif
#endif  // PT_ENABLE_OPTIX

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
#if defined(PT_ENABLE_OPTIX)
    // Timeline semaphores: required by VulkanOptixDenoiser to fence
    // CUDA <-> Vulkan work. Without enabling the feature, Vulkan
    // creates a binary semaphore on vkCreateSemaphore (silently
    // ignoring VK_SEMAPHORE_TYPE_TIMELINE), which then makes
    // cudaImportExternalSemaphore reject the handle as 'invalid
    // argument'. Core Vulkan 1.2 feature; universally supported on
    // RTX-class GPUs, so no per-device support query.
    v12.timelineSemaphore =
        (v12_supported.timelineSemaphore == VK_TRUE) ? VK_TRUE : VK_FALSE;
#endif

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
    //   9 storage_image  + 1 accel_struct + 6 storage_buffer + 1 ubo
    //   per set, x kFramesInFlight sets, with headroom.
    std::array<VkDescriptorPoolSize, 4> psizes{};
    psizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           kFramesInFlight * 9 + 4 };
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

    // ---- Build the unified 17-binding descriptor set layout ----------
    {
        std::array<VkDescriptorSetLayoutBinding, 17> b{};
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
        // binding 16: normal_tex (path tracer G-buffer for SVGF/NRD)
        fill(16, 16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

        // UPDATE_AFTER_BIND for every binding so we can rewrite the
        // shared descriptor set between dispatches in the same cmd
        // buffer without invalidating earlier recorded draws.
        std::array<VkDescriptorBindingFlags, 17> bind_flags{};
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

    // ---- Async pipeline build ----------------------------------------
    // The path-tracer pipeline can take 1-3s on a cold pipeline cache;
    // running that synchronously on the main thread blocks the window
    // from appearing.  Move the load+build sequence to a worker that
    // runs in parallel with RecreateSwapchain + the engine's first
    // frames.  vkCreateComputePipelines is thread-safe with respect
    // to the device; pipelines_/named_pipelines_ writes go through
    // resource_mutex_ which the lookup paths already take.
    //
    // While the worker is in flight, named_pipelines_.find() returns
    // empty and CreateComputePipeline-by-name hands back id=0.  The
    // engine's RenderFrame skips dispatches with id=0 and re-resolves
    // its cached ids each frame via Engine::EnsurePipelineHandles(),
    // so as soon as a pipeline lands in named_pipelines_ the engine
    // picks it up on the next frame.
    pipeline_build_thread_ = std::thread([this]() {
        // Pipeline cache load + creation also moved to the worker so
        // the pipeline_cache_ handle isn't half-initialised when the
        // first vkCreateComputePipelines runs against it.
        const auto t_start = std::chrono::steady_clock::now();
        LoadPipelineCache();

        // Per-pipeline timing surfaces cold/warm cache effectiveness
        // and lets us see at a glance which kernel is the long pole
        // (PathTrace today, by an order of magnitude).
        struct PipeBuild { const char* name; double ms; };
        std::vector<PipeBuild> timings;
        timings.reserve(3);
        auto build_pipeline = [&](const char* name,
                                  const unsigned char* spirv,
                                  std::size_t          spirv_size) {
            const auto t0 = std::chrono::steady_clock::now();
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
            if (vkCreateComputePipelines(device_, pipeline_cache_, 1, &cpci2,
                                         nullptr, &pipe) == VK_SUCCESS) {
                std::lock_guard lock(resource_mutex_);
                auto id = next_id_++;
                pipelines_.emplace(id, PipelineEntry{pipe});
                named_pipelines_.emplace(name, id);
            } else {
                LOG_ERROR("vkCreateComputePipelines({}) failed", name);
            }
            vkDestroyShaderModule(device_, mod, nullptr);
            const auto t1 = std::chrono::steady_clock::now();
            timings.push_back({
                name,
                std::chrono::duration<double, std::milli>(t1 - t0).count()
            });
        };

        build_pipeline("pathtrace",   shader_PathTrace_spirv_data,    shader_PathTrace_spirv_size);
        build_pipeline("autoexpose",  shader_AutoExposure_spirv_data, shader_AutoExposure_spirv_size);
        build_pipeline("perfoverlay", shader_PerfOverlay_spirv_data,  shader_PerfOverlay_spirv_size);
        pipelines_ready_.store(true, std::memory_order_release);

        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
        std::string per_pipe;
        for (const auto& p : timings) {
            if (!per_pipe.empty()) per_pipe += ", ";
            per_pipe += fmt::format("{} {:.0f}ms", p.name, p.ms);
        }
        LOG_INFO("Vulkan: async pipeline build done in {:.0f}ms ({})",
                 total_ms, per_pipe);
    });

    if (!RecreateSwapchain()) return;

    wrapped_cb_ = std::make_unique<VulkanCommandBuffer>(this);
}

VulkanDevice::~VulkanDevice() {
    DestroyDevice();
}

void VulkanDevice::DestroyDevice() {
    // Join the async pipeline build worker before touching anything
    // device-related. The worker mutates pipelines_/named_pipelines_
    // and reads pipeline_cache_, so we need it stopped before we
    // destroy any of those.  Safe even if the worker is never
    // started (default-constructed std::thread is not joinable).
    if (pipeline_build_thread_.joinable()) {
        pipeline_build_thread_.join();
    }
    if (device_ != VK_NULL_HANDLE) {
#if defined(PT_ENABLE_OPTIX)
        // Drain the CUDA stream BEFORE vkDeviceWaitIdle. The OptiX
        // denoiser submitted a private output-copy cb whose wait on
        // sem_cuda_to_vk only completes after CUDA signals (in turn
        // gated on engine-cb completion via sem_vk_to_cuda). If CUDA
        // hasn't finished its queued denoise work, the private cb
        // stays pending forever and vkDeviceWaitIdle deadlocks. The
        // drain here forces CUDA to flush, signaling cuda_to_vk for
        // every pending frame -- which lets the private cbs complete
        // and vkDeviceWaitIdle return instantly.
        if (optix_denoiser_) {
            optix_denoiser_->DrainCuda();
        }
#endif
        vkDeviceWaitIdle(device_);
        // Tear down the denoiser before any of its dependencies
        // (VkPipeline / VkDescriptorPool / VkImageView). The denoiser
        // owns a few textures via device_->DestroyTexture and a few
        // raw VK objects via its own dtor; destroying it here does
        // both in the right order while device_ is still live.
        denoiser_.reset();
#if defined(PT_ENABLE_OPTIX)
        // Same rationale for the OptiX denoiser: it holds external
        // VkImage / VkDeviceMemory / VkSemaphore handles whose dtor
        // calls vkDestroy* against device_->RawDevice(). Reset before
        // vkDestroyDevice() below so the VkDevice is still live when
        // those teardowns run -- otherwise the validation layer
        // (correctly) flags the underlying VK objects as leaked.
        optix_denoiser_.reset();
#endif
        for (auto v : swap_views_) if (v) vkDestroyImageView(device_, v, nullptr);
        swap_views_.clear();
        if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        for (int i = 0; i < kFramesInFlight; ++i) {
            if (sem_image_avail_[i]) vkDestroySemaphore(device_, sem_image_avail_[i], nullptr);
            if (fence_in_flight_[i]) vkDestroyFence(device_, fence_in_flight_[i], nullptr);
            DestroyBufferImpl(frame_ubos_[i]);
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

        // Persist pipeline cache before destroying device. Order matters:
        // vkGetPipelineCacheData / vkDestroyPipelineCache need a live device.
        SavePipelineCache();
        if (pipeline_cache_ != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
            pipeline_cache_ = VK_NULL_HANDLE;
        }

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

// ---- VkPipelineCache persistence ----------------------------------------
//
// Path: %LOCALAPPDATA%/demont/pipeline.cache on Windows,
// $XDG_CACHE_HOME/demont/pipeline.cache (or $HOME/.cache/demont/...) on
// POSIX. Per-driver / per-device blobs are NOT separated here -- the
// driver tags the cache with a UUID and silently rejects mismatched
// blobs, so a single file works correctly across reinstalls / GPU
// swaps (just rebuilds on first launch after a change).

std::string VulkanDevice::PipelineCachePath() {
    namespace fs = std::filesystem;
    fs::path base;
#if defined(_WIN32)
    if (const char* appdata = std::getenv("LOCALAPPDATA")) {
        base = fs::path(appdata) / "demont";
    } else {
        base = fs::temp_directory_path() / "demont";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        base = fs::path(xdg) / "demont";
    } else if (const char* home = std::getenv("HOME")) {
        base = fs::path(home) / ".cache" / "demont";
    } else {
        base = fs::temp_directory_path() / "demont";
    }
#endif
    std::error_code ec;
    fs::create_directories(base, ec);     // best-effort; ignore failures
    return (base / "pipeline.cache").string();
}

void VulkanDevice::LoadPipelineCache() {
    if (device_ == VK_NULL_HANDLE) return;
    std::vector<char> blob;
    {
        std::ifstream f(PipelineCachePath(), std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            const auto sz = f.tellg();
            f.seekg(0, std::ios::beg);
            if (sz > 0) {
                blob.resize(static_cast<std::size_t>(sz));
                f.read(blob.data(), sz);
                if (!f) blob.clear();
            }
        }
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = blob.size();
    ci.pInitialData    = blob.empty() ? nullptr : blob.data();
    if (vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        // Driver rejected the blob (header / UUID / version mismatch).
        // Retry with empty initial data so the cache is at least usable
        // for this run -- subsequent vkCreateComputePipelines will
        // populate it and SavePipelineCache will overwrite the stale
        // file on shutdown.
        pipeline_cache_ = VK_NULL_HANDLE;
        VkPipelineCacheCreateInfo empty_ci{};
        empty_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device_, &empty_ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
            pipeline_cache_ = VK_NULL_HANDLE;
            LOG_WARN("Vulkan: pipeline cache create failed; pipelines will compile cold");
        } else if (!blob.empty()) {
            LOG_INFO("Vulkan: pipeline cache rejected stale blob ({} bytes); rebuilding", blob.size());
        }
    } else if (!blob.empty()) {
        LOG_INFO("Vulkan: pipeline cache loaded ({} bytes)", blob.size());
    }
}

void VulkanDevice::SavePipelineCache() {
    if (device_ == VK_NULL_HANDLE || pipeline_cache_ == VK_NULL_HANDLE) return;
    std::size_t sz = 0;
    if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, nullptr) != VK_SUCCESS || sz == 0) {
        return;
    }
    std::vector<char> blob(sz);
    if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, blob.data()) != VK_SUCCESS) {
        return;
    }
    std::ofstream f(PipelineCachePath(), std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(blob.data(), static_cast<std::streamsize>(sz));
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

VkExtent2D VulkanDevice::LookupImageExtent(TextureHandle h) {
    if (h.id == kSwapchainTextureId) return swap_extent_;
    if (h.id == 0) return {0, 0};
    std::lock_guard lock(resource_mutex_);
    auto it = images_.find(h.id);
    return (it == images_.end()) ? VkExtent2D{0, 0} : it->second.extent;
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
        // vkMapMemory failure (rare: out of host address space, driver
        // bug, or trying to map non-host-visible memory we let slip
        // through earlier) leaves out.mapped at nullptr. Subsequent
        // memcpy/WriteBuffer/WriteTexture would deref it and crash, so
        // treat it as a buffer-creation failure: tear down the
        // partially-built buffer and return false to the caller.
        if (vkMapMemory(device_, out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS) {
            LOG_ERROR("Vulkan: vkMapMemory failed on persistent-map buffer (size {})",
                      static_cast<std::uint64_t>(size));
            out.mapped = nullptr;
            vkFreeMemory(device_, out.memory, nullptr);
            vkDestroyBuffer(device_, out.buffer, nullptr);
            out.memory = VK_NULL_HANDLE;
            out.buffer = VK_NULL_HANDLE;
            out.size   = 0;
            return false;
        }
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
    VkBuffer    dst_buf  = VK_NULL_HANDLE;
    std::size_t dst_size = 0;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = buffers_.find(h.id);
        if (it == buffers_.end()) return;
        dst_buf  = it->second.buffer;
        dst_size = it->second.size;
    }
    if (dst_buf == VK_NULL_HANDLE) return;
    // Bounds-check the requested copy region against the destination
    // buffer's allocated size. Without this, a caller passing an
    // incorrect size / offset triggers vkCmdCopyBuffer past the buffer's
    // end -- validation error / undefined behaviour on real drivers.
    if (dst_offset > dst_size || size > dst_size - dst_offset) {
        LOG_ERROR("Vulkan WriteBuffer: copy region [{} +{}] exceeds buffer size {}",
                  static_cast<std::uint64_t>(dst_offset),
                  static_cast<std::uint64_t>(size),
                  static_cast<std::uint64_t>(dst_size));
        return;
    }

    // Stage CPU bytes through a transient host-visible buffer, then
    // vkCmdCopyBuffer to the device-local destination.  Synchronous
    // submit-and-wait -- the call sites are scene-load time (mesh
    // upload, env CDF upload) and rare cvar-change events
    // (r_exposure flip, etc.), not every frame.  The 5-15 ms stall
    // is user-initiated and acceptable on a knob change; we'd want
    // a vkCmdUpdateBuffer-based fast-path queued into the next
    // frame's command buffer if this ever moves into the per-frame
    // loop.  See Raytracer Plan/FOLLOW_UPS.md "WriteBuffer fast
    // path for tiny runtime updates" if you hit a hitch on a knob.
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
    // STORAGE_BIT  : sampling + writes from compute shaders (the
    //                main path for every texture we create here).
    // TRANSFER_DST : WriteTexture host upload + ReadbackTexture's
    //                transient staging fill.
    // TRANSFER_SRC : the SVGF basic-mode vkCmdCopyImage out of the
    //                history texture into post_denoise_hdr; also
    //                lets ReadbackTexture work on any storage image.
    ici.usage        = VK_IMAGE_USAGE_STORAGE_BIT
                     | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                     | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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

    // One-shot synchronous readback path. The only caller today is the
    // `screenshot` console command, which is user-triggered and
    // low-frequency, so a queue stall during the wait is fine. The
    // async-slot-ring infrastructure that lived here was built for the
    // per-8-frames CPU autoexpose readback that's been retired
    // (auto-expose is now entirely GPU-resident, see AutoExposure.slang
    // + exposure_state). When async readback is needed again it'll most
    // likely be buffer-shaped (GPU physics event queues, not texture
    // data) and ship as a separate ReadbackBuffer API designed around
    // that use case rather than retrofitted onto this one.
    BufferEntry staging{};
    if (!CreateBufferImpl(bytes,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging,
                          /*persistent_map=*/true)) {
        return false;
    }

    VkCommandBuffer cb = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmd_pool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &ai, &cb) != VK_SUCCESS) {
            DestroyBufferImpl(staging);
            return false;
        }
    }

    VkFence fence = VK_NULL_HANDLE;
    {
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device_, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, cmd_pool_, 1, &cb);
            DestroyBufferImpl(staging);
            return false;
        }
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);

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
        vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { extent.width, extent.height, 1 };
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.buffer, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(graphics_queue_, 1, &si, fence);
    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

    std::memcpy(dst, staging.mapped, bytes);
    if (out_w) *out_w = extent.width;
    if (out_h) *out_h = extent.height;

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cb);
    DestroyBufferImpl(staging);
    return true;
}

bool VulkanDevice::ReadbackBuffer(BufferHandle h, void* dst, std::size_t bytes) {
    if (dst == nullptr || bytes == 0 || device_ == VK_NULL_HANDLE) return false;

    VkBuffer    src      = VK_NULL_HANDLE;
    std::size_t src_size = 0;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = buffers_.find(h.id);
        if (it == buffers_.end()) return false;
        src      = it->second.buffer;
        src_size = it->second.size;
    }
    if (src == VK_NULL_HANDLE) return false;
    // Bounds-check: a caller passing an oversized `bytes` would otherwise
    // copy past the source buffer's allocation -- spec violation /
    // undefined behaviour.
    if (bytes > src_size) {
        LOG_ERROR("Vulkan ReadbackBuffer: requested {} bytes from buffer of size {}",
                  static_cast<std::uint64_t>(bytes),
                  static_cast<std::uint64_t>(src_size));
        return false;
    }

    // One-shot synchronous readback, mirrors ReadbackTexture's pattern.
    // Engine storage buffers are DEVICE_LOCAL only -- copy through a
    // host-visible staging buffer to read.
    BufferEntry staging{};
    if (!CreateBufferImpl(bytes,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          staging,
                          /*persistent_map=*/true)) {
        return false;
    }

    VkCommandBuffer cb = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmd_pool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &ai, &cb) != VK_SUCCESS) {
            DestroyBufferImpl(staging);
            return false;
        }
    }

    VkFence fence = VK_NULL_HANDLE;
    {
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device_, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, cmd_pool_, 1, &cb);
            DestroyBufferImpl(staging);
            return false;
        }
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);

    // Wait for prior shader writes (e.g. AutoExposure.slang updating
    // exposure_state) to be visible to TRANSFER_READ.
    VkBufferMemoryBarrier bmb{};
    bmb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = src;
    bmb.offset              = 0;
    bmb.size                = bytes;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &bmb, 0, nullptr);

    VkBufferCopy region{};
    region.size = bytes;
    vkCmdCopyBuffer(cb, src, staging.buffer, 1, &region);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    bool ok = (vkQueueSubmit(graphics_queue_, 1, &si, fence) == VK_SUCCESS);
    if (ok) {
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        std::memcpy(dst, staging.mapped, bytes);
    }

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cb);
    DestroyBufferImpl(staging);
    return ok;
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

#if defined(PT_ENABLE_OPTIX)
    // OptiX path: when VulkanOptixDenoiser::Encode requested an extra
    // timeline-semaphore signal for CUDA's cudaWaitExternalSemaphoresAsync
    // to gate on, attach it here. Both signals (the binary swapchain-
    // present sem + the timeline OptiX sem) go in the same vkQueueSubmit
    // -- VkTimelineSemaphoreSubmitInfo's pSignalSemaphoreValues holds
    // values per-position, with binary-semaphore positions ignored.
    VkSemaphore     opt_signals[2] {};
    std::uint64_t   opt_signal_vals[2] {};
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    if (extra_submit_signal_sem_ != VK_NULL_HANDLE) {
        opt_signals[0]      = sem_render_done_[current_swap_index_];
        opt_signals[1]      = extra_submit_signal_sem_;
        opt_signal_vals[0]  = 0;  // ignored for binary
        opt_signal_vals[1]  = extra_submit_signal_value_;
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 2;
        timeline_info.pSignalSemaphoreValues    = opt_signal_vals;
        si.pNext                = &timeline_info;
        si.signalSemaphoreCount = 2;
        si.pSignalSemaphores    = opt_signals;
    }
#endif

    vkQueueSubmit(graphics_queue_, 1, &si, fence_in_flight_[current_frame_]);

#if defined(PT_ENABLE_OPTIX)
    // One-shot: clear the request after each submit so a frame where
    // the OptiX path doesn't activate (cvar transition, denoiser
    // un-ready, etc.) doesn't accidentally signal a stale value.
    extra_submit_signal_sem_   = VK_NULL_HANDLE;
    extra_submit_signal_value_ = 0;

    // Submit the OptiX denoiser's private output-copy cb after the
    // main engine cb. Encode() recorded it but deferred the submit so
    // queue-order serialization works: the private cb waits on a
    // timeline that the engine's cb above signals (via the
    // RequestExtraSubmitSignal path), so it must be submitted AFTER
    // the engine's cb to avoid blocking the queue.
    if (optix_denoiser_) {
        optix_denoiser_->SubmitPostMain();
    }
#endif

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

#if defined(PT_ENABLE_OPTIX)
void VulkanDevice::RequestExtraSubmitSignal(VkSemaphore sem,
                                            std::uint64_t timeline_value) {
    extra_submit_signal_sem_   = sem;
    extra_submit_signal_value_ = timeline_value;
}

void VulkanDevice::EncodeDenoiseFinalize(VkCommandBuffer cb,
                                         VkImageView    color_in_view,
                                         VkImageView    final_output_view,
                                         VkBuffer       exposure_state_buf,
                                         std::uint32_t  width,
                                         std::uint32_t  height,
                                         bool           hdr_pipeline) {
    // Lazy: ensure the NRD denoiser exists + Init'd so finalize_pipe_
    // is built. We only need the finalize pipeline (the temporal /
    // atrous pipelines and history textures remain unbuilt /
    // unallocated as long as Encode() isn't called -- ResizeTextures
    // is gated on Encode(), not Init()).
    if (denoiser_ == nullptr) {
        denoiser_ = std::make_unique<VulkanNrdDenoiser>(this);
    }
    if (!denoiser_->Ready()) {
        if (!denoiser_->Init()) {
            LOG_ERROR("VulkanDevice::EncodeDenoiseFinalize: SVGF denoiser "
                      "Init failed; OptiX path will skip tonemap (image will "
                      "be over-bright on screen until a re-init succeeds)");
            denoiser_.reset();
            return;
        }
    }
    denoiser_->EncodeFinalizeOnly(cb, color_in_view, final_output_view,
                                  exposure_state_buf, width, height,
                                  hdr_pipeline);
}
#endif

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

// ---- SVGF/NRD denoiser --------------------------------------------------

bool VulkanDevice::SupportsDenoise() const {
    // Two-stage readiness:
    //   1) `denoiser_->Ready()` is true once the denoiser has been
    //      lazily initialised by the first Denoise() call (which builds
    //      its 3 compute pipelines, descriptor pool, and dummy textures
    //      synchronously -- a one-time few-ms hitch on first transition
    //      from r_denoiser=off to a Vulkan kind).
    //   2) Before that first init, fall back to `pipelines_ready_`,
    //      which the engine's main async pipeline-build worker flips
    //      true after pathtrace/autoexpose/perfoverlay finish. This is
    //      a soft signal -- it means "the device has finished its
    //      heavy-lift cold-cache work, opting in is safe" -- not "the
    //      denoiser itself is ready". The hitch on opt-in is acceptable
    //      because it's a one-shot user-driven cvar transition.
    // Eagerly building the denoiser in the worker would eliminate the
    // hitch (see the FOLLOW_UPS.md note); deferred since the hitch is
    // small and only happens once per session.
    if (denoiser_ != nullptr && denoiser_->Ready()) return true;
    return pipelines_ready_.load(std::memory_order_acquire);
}

void VulkanDevice::Denoise(const DenoiseDesc& d) {
    if (device_ == VK_NULL_HANDLE) return;
    if (wrapped_cb_ == nullptr || wrapped_cb_->Raw() == VK_NULL_HANDLE) {
        // Engine called Denoise without first AcquireCommandBuffer
        // (loading frame, or path tracer skipped) -- silently no-op.
        return;
    }

    // ---- Dispatch-routing trace. Logged once per Kind transition
    // (off->svgf, svgf->optix, etc.) so the engine -> RHI plumbing
    // boundary is observable for any future denoiser-routing bug.
    // Earned its keep on the OptiX bring-up: revealed that the
    // engine routing was correct (kind=1 dispatched cleanly) and
    // isolated the failures to optixInit + optixDenoiserComputeIntensity
    // upstream. Stays in -- matches the engine's "log on state
    // transitions" philosophy and costs nothing per frame after the
    // first dispatch (static guard latches once kind is logged).
    // Will move to PT_DIAG(2) when the r_diagnostic_level cvar lands
    // (see follow-up).
    {
        static int s_last_logged_kind = -1;
        const int k = static_cast<int>(d.kind);
        if (k != s_last_logged_kind) {
            LOG_INFO("VulkanDevice::Denoise: dispatching kind={} (0=Svgf, 1=OptixHdr, 2=OptixHdrAov), "
                     "color={} out={} normal={} depth={} motion={}",
                     k, d.color_in.id, d.output.id, d.normal_in.id, d.depth_in.id, d.motion_in.id);
            s_last_logged_kind = k;
        }
    }

    // Route based on d.kind. SVGF and OptiX have different input
    // contracts: SVGF wants the full G-buffer (color/depth/motion/
    // normal/output); OptiX HDR only needs color + output; OptiX HDR
    // AOV adds normal_in (and primary_albedo, once Phase 1a step 3
    // lands). Each path validates only what it actually consumes.
#if defined(PT_ENABLE_OPTIX)
    if (d.kind == DenoiseDesc::Kind::OptixHdr ||
        d.kind == DenoiseDesc::Kind::OptixHdrAov) {
        if (d.color_in.id == 0 || d.output.id == 0) {
            LOG_WARN("VulkanDevice::Denoise(OptiX): missing color/output (color={} out={})",
                     d.color_in.id, d.output.id);
            return;
        }
        if (optix_denoiser_ == nullptr) {
            const auto opt_kind = (d.kind == DenoiseDesc::Kind::OptixHdrAov)
                                  ? VulkanOptixDenoiser::Kind::HdrAov
                                  : VulkanOptixDenoiser::Kind::Hdr;
            optix_denoiser_ = std::make_unique<VulkanOptixDenoiser>(this, opt_kind);
        }
        if (!optix_denoiser_->IsReady()) {
            // Init() guards itself against re-entry via init_attempted_,
            // so the first failure logs once (inside Init) and subsequent
            // per-frame calls return false silently. We KEEP the failed
            // instance alive so we don't recreate-and-retry every frame --
            // re-trying after a hardware/driver-level failure has no path
            // to succeed anyway. To force a retry the user must restart.
            if (!optix_denoiser_->Init()) {
                return;
            }
        }
        optix_denoiser_->Encode(wrapped_cb_->Raw(), d);
        return;
    }
#endif

    // ---- SVGF / NRD path (Kind::Svgf, the historical default) ------------
    // Need at least the noisy color, depth, motion, normal, and a
    // valid output target. The engine's allocation gate guarantees
    // these on a denoiser-active frame; if any are missing it's a
    // logic bug, so log + skip rather than dispatch with garbage.
    if (d.color_in.id == 0 || d.depth_in.id == 0 || d.motion_in.id == 0 ||
        d.normal_in.id == 0 || d.output.id == 0) {
        LOG_WARN("VulkanDevice::Denoise: missing G-buffer inputs (color={} depth={} motion={} normal={} out={})",
                 d.color_in.id, d.depth_in.id, d.motion_in.id, d.normal_in.id, d.output.id);
        return;
    }

    // Lazy denoiser init. Safe to call every frame -- Init() is a
    // no-op once ready_.
    if (denoiser_ == nullptr) {
        denoiser_ = std::make_unique<VulkanNrdDenoiser>(this);
    }
    if (!denoiser_->Ready()) {
        if (!denoiser_->Init()) {
            LOG_ERROR("VulkanDevice::Denoise: denoiser init failed; disabling");
            denoiser_.reset();
            return;
        }
    }

    // Resolve the G-buffer dimensions from the noisy color texture
    // so scratch textures track resize + backend toggles automatically.
    auto image_it = images_.find(d.color_in.id);
    if (image_it == images_.end()) {
        LOG_WARN("VulkanDevice::Denoise: color_in image lookup miss");
        return;
    }
    const std::uint32_t w = image_it->second.extent.width;
    const std::uint32_t h = image_it->second.extent.height;
    if (w == 0 || h == 0) return;
    if (!denoiser_->ResizeTextures(w, h)) {
        LOG_ERROR("VulkanDevice::Denoise: ResizeTextures({}x{}) failed", w, h);
        return;
    }

    const bool atrous_enabled =
        (d.quality == DenoiseDesc::Quality::Atrous);
    denoiser_->Encode(wrapped_cb_->Raw(),
                      d.color_in, d.depth_in, d.motion_in,
                      d.normal_in, d.output,
                      d.final_output, d.exposure_state,
                      d.reset_history,
                      atrous_enabled,
                      d.hdr_pipeline);
}

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle& w) {
    return std::make_unique<vk::VulkanDevice>(w);
}
}  // namespace pt::rhi

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Vulkan backend, P12 expansion: ray-query, acceleration structures,
// real CreateBuffer / WriteBuffer / CreateBLAS / CreateTLAS, expanded
// descriptor set layout (19 bindings) matching the unified PathTrace
// shader, and per-frame UBO ring for the spilled push-constant tail.

#include "VulkanDevice.h"
#include "VulkanDenoiser.h"
#if defined(PT_ENABLE_OPTIX)
#include "VulkanOptixDenoiser.h"
#endif

#include "../core/Diag.h"
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
#include <atomic>
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
// no-RayQuery variant. Built by src/rhi_vulkan/CMakeLists.txt with
// -DPT_SPIRV_NO_RAYQUERY; used in place of the RT-enabled blob above
// on backends that don't expose VK_KHR_ray_query (Mac-Vulkan on
// pre-MoltenVK-1.3 builds). The two are functionally equivalent except
// for the mesh-traversal path: gold variant uses RayQuery, norq
// variant falls back to a linear scan via bvh_params.z.
extern const unsigned char shader_PathTrace_norq_spirv_data[];
extern const unsigned long shader_PathTrace_norq_spirv_size;
extern const unsigned char shader_AutoExposure_spirv_data[];
extern const unsigned long shader_AutoExposure_spirv_size;
extern const unsigned char shader_PerfOverlay_spirv_data[];
extern const unsigned long shader_PerfOverlay_spirv_size;
// Bloom pyramid (BloomDown/Up). Compiled to SPIR-V and embedded
// alongside the other compute shaders. Each declares only 2
// storage-image bindings (src + dst) on Vulkan, which is compatible
// with the shared 20-binding pipeline layout -- so we build them with
// the same VkPipelineLayout the path tracer uses and drive them via
// the engine's standard cb->BindComputePipeline + cb->Dispatch path.
extern const unsigned char shader_BloomDown_spirv_data[];
extern const unsigned long shader_BloomDown_spirv_size;
extern const unsigned char shader_BloomUp_spirv_data[];
extern const unsigned long shader_BloomUp_spirv_size;
// Tonemap (composite bloom + apply exposure*ACES*sRGB, plus lens
// flare). Compiled to SPIR-V; the host-side TonePush has 48 bytes of
// padding inserted so the ghost array lands at the kPushSplitOffset
// boundary (offset 112) and naturally spills into the Frame UBO at
// binding 14. The shader's PT_TARGET_SPIRV path reads ghosts[] from
// that UBO; the non-ghost fields live in vkCmdPushConstants like the
// rest of the small-push pipelines.
extern const unsigned char shader_Tonemap_spirv_data[];
extern const unsigned long shader_Tonemap_spirv_size;
}

namespace pt::rhi::vk {

namespace {

#ifndef NDEBUG
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

// Engine slot -> shader vk::binding translation. The unified descriptor
// set has 23 bindings (0-22): storage_image x11 (output / accum_hdr /
// denoise_color / depth / motion / env_map / star_map / moon_map /
// normal_tex / albedo_tex / accum_stars), AS x1 (scene_tlas),
// storage_buffer x10 (mesh / cdf / exposure_state / bvh_nodes /
// tri_bvh_nodes / tri_bvh_permuted_ids / sdf_clusters), uniform_buffer
// x1 (Frame UBO at binding 14). Tonemap / Bloom* re-use a subset.
// Engine.cpp uses up-to-12-element textures, 11-element buffers, and
// 4-element accel-struct slot arrays as if Metal-style argument tables;
// we flatten these into the single Vulkan descriptor set here. Engine
// slot 8 (normal_tex / vk::binding 16) and slot 9 (albedo_tex /
// vk::binding 17) are Vulkan-only -- the SVGF/NRD path tracer writes
// normal; only OptiX AOV (optix_hdr_aov) writes albedo. Engine slot 7
// (bvh_nodes / vk::binding 18) is populated for the analytic-prim BVH
// path on either backend.
// --- PR #106 follow-up (triangle BVH) ---------------------------------------
// Engine buffer slots 8 / 9 -> vk::bindings 19 / 20 are the triangle
// BVH (tri_bvh_nodes + tri_bvh_permuted_ids). Always bound (placeholder
// = primitives buffer) so the descriptor set remains complete; the
// shader gates the read on push.bvh_params.w > 0.
// --- end PR #106 follow-up --------------------------------------------------
// --- SDF Phase 1 (#97) ------------------------------------------------------
// Engine buffer slot 10 -> vk::binding 21 is the SDF cluster storage
// buffer. Moved from engine slot 8 / vk::binding 19 to make room for
// PR #106's triangle BVH. Always bound (placeholder = placeholder
// storage buffer) so the descriptor set remains complete; the shader
// gates the SDF read on push.sdf_params.x > 0.
// --- end SDF Phase 1 --------------------------------------------------------
// --- Cloud transmittance G-buffer (issue #46 follow-up) ---------------------
// Engine texture slot 10 -> vk::binding 22 is cloud_trans_tex, the R32F
// per-pixel cloud transmittance the path tracer writes from its
// volumetric cloud march. Reused the slot number (and binding) that
// accum_stars (#108) briefly occupied -- accum_stars is gone, slot is
// free, and re-using it keeps the descriptor pool sizing arithmetic
// identical to the pre-rewrite state. StarsComposite reads this to
// attenuate the celestial composite by foreground cloud density.
// --- end Cloud transmittance ------------------------------------------------
static constexpr std::uint32_t kNumTexSlots = 11;
constexpr std::uint32_t kSlotToTexBinding[kNumTexSlots] = {
    0,  // engine slot 0  -> shader binding 0  (output / swapchain)
    1,  // engine slot 1  -> shader binding 1  (accum_hdr)
    6,  // engine slot 2  -> shader binding 6  (denoise_color)
    7,  // engine slot 3  -> shader binding 7  (depth_tex)
    8,  // engine slot 4  -> shader binding 8  (motion_tex)
    9,  // engine slot 5  -> shader binding 9  (env_map)
    12, // engine slot 6  -> shader binding 12 (star_map)
    13, // engine slot 7  -> shader binding 13 (moon_map)
    16, // engine slot 8  -> shader binding 16 (normal_tex, SVGF/NRD/OptiX-AOV)
    17, // engine slot 9  -> shader binding 17 (albedo_tex, OptiX AOV only)
    22, // engine slot 10 -> shader binding 22 (cloud_trans_tex, #46 follow-up)
};
constexpr std::uint32_t kSlotToBufBinding[12] = {
    0,  // engine slot 0 unused
    3,  // engine slot 1 -> shader binding 3  (mesh_positions)
    4,  // engine slot 2 -> shader binding 4  (mesh_indices)
    5,  // engine slot 3 -> shader binding 5  (primitives)
    10, // engine slot 4 -> shader binding 10 (marginal_cdf)
    11, // engine slot 5 -> shader binding 11 (conditional_cdf)
    15, // engine slot 6 -> shader binding 15 (exposure_state)
    18, // engine slot 7 -> shader binding 18 (analytic-prim BVH nodes)
    // PR #106 follow-up: triangle BVH (host-built; replaces the
    // O(N) Möller-Trumbore SW linear-scan path that #106 shipped for
    // Mac-Vulkan / MoltenVK without VK_KHR_ray_query).
    19, // engine slot 8 -> shader binding 19 (tri_bvh_nodes)
    20, // engine slot 9 -> shader binding 20 (tri_bvh_permuted_ids)
    // SDF Phase 1 (#97): SDF cluster buffer. Moved here from engine
    // slot 8 / binding 19 to make room for tri_bvh_*.
    21, // engine slot 10 -> shader binding 21 (SDF cluster buffer)
    // Light primitives (#73): analytic light list. Past the SIGMA
    // #115 / MetalFX specular #118 reservations at 23..26.
    27, // engine slot 11 -> shader binding 27 (light_prims)
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
    // Each Dispatch grabs a fresh descriptor set from this frame's
    // ring so its bindings are isolated from other dispatches in the
    // same cmd buffer. With one shared set + UPDATE_AFTER_BIND, the
    // GPU reads the LATEST descriptor at execution time -- which
    // meant PathTrace's `output` ended up reading whatever
    // bloom_pyramid's last bloom_down dispatch had bound there
    // (a 320x180 bloom mip), bounds-checking PathTrace's writes to
    // 320x180 instead of 1280x720. See dsets_ comment in VulkanDevice.h.
    auto dset = device_->AcquireDispatchDescriptorSet();

    // Build the descriptor write list. Each binding in the shared layout
    // is marked VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, so slots the
    // engine didn't bind for this dispatch are simply *not written* --
    // they stay in whatever state the prior dispatch left them in, and
    // the shader is required not to access them (the pipeline's SPIR-V
    // dictates which bindings it reads/writes). This replaces the older
    // strategy of writing VK_NULL_HANDLE to unbound slots via
    // VK_EXT_robustness2.nullDescriptor -- MoltenVK on Apple Silicon
    // reports the extension but with nullDescriptor=false, so the null
    // path doesn't work on Mac. Partially-bound is core Vulkan 1.2 and
    // supported on every target including MoltenVK.
    //
    // Capacity: 11 storage_image (one less than kNumTexSlots because
    // engine texture slot 10 is reserved/unused on the Vulkan path)
    // + 1 accel_struct + 10 storage_buffer + 1 uniform_buffer = 23,
    // sized to the worst-case "PathTrace binds everything" dispatch.
    // kMaxWrites must be >= the total binding count or
    // vkUpdateDescriptorSets reads off the end of these stack arrays.
    // The +4 over the legacy 19 is bindings 19/20 (tri_bvh_nodes /
    // tri_bvh_permuted_ids, PR #106 follow-up host-built triangle BVH),
    // binding 21 (SDF cluster buffer, SDF Phase 1 #97; moved from
    // binding 19 to make room for the tri BVH), and binding 22
    // (cloud_trans_tex, issue #46 follow-up -- the R32F per-pixel
    // cloud transmittance the path tracer writes and StarsComposite
    // reads. Reuses the slot number accum_stars (#108) briefly
    // occupied before the stateless composite rewrite.)
    constexpr std::uint32_t kMaxWrites = 23;
    std::array<VkDescriptorImageInfo,  kMaxWrites> img_infos {};
    std::array<VkDescriptorBufferInfo, kMaxWrites> buf_infos {};
    std::array<VkWriteDescriptorSetAccelerationStructureKHR, 1> as_infos {};
    std::array<VkAccelerationStructureKHR, 1> as_handles {};
    std::array<VkWriteDescriptorSet, kMaxWrites> writes {};
    std::uint32_t wc = 0;

    // Helpers each assume a non-null handle -- callers gate at the call
    // site, so unbound slots are skipped instead of being written as
    // null. This matches the partially-bound contract: writes that
    // happen must be spec-compliant for real resources.
    auto add_image = [&](std::uint32_t binding, VkImageView v) {
        auto& ii = img_infos[wc];
        ii.imageView   = v;
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
        bi.offset = off;
        bi.range  = sz;
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

    // ---- Storage images at bindings 0,1,6,7,8,9,12,13,16,17,22 -------
    // Skip slots the engine didn't bind: partially-bound layout means
    // unwritten slots are simply absent from this dispatch's set view.
    // (Slots a shader actually reads must be bound by the engine ahead
    // of Dispatch; that's the dispatch-pipeline contract.) Engine
    // texture slot 10 is reserved (no binding) -- accum_stars moved to
    // slot 11 / binding 22 in round-2 merge so SDF Phase 1 (#109) could
    // own buffer slot 10 / binding 21 without colliding in the flat
    // Vulkan descriptor space.
    for (std::uint32_t s = 0; s < kNumTexSlots; ++s) {
        VkImageView v = device_->LookupImageView(bound_tex_[s]);
        if (v == VK_NULL_HANDLE) continue;
        add_image(kSlotToTexBinding[s], v);
    }

    // ---- Acceleration structure at binding 2 --------------------------
    {
        auto a = device_->LookupAccel(bound_accel_[2]);
        if (a != VK_NULL_HANDLE) add_accel(2, a);
    }

    // ---- Storage buffers at bindings 3,4,5,10,11,15,18,19,20,21 -----
    // Slot 6 -> binding 15 is exposure_state (GPU-driven auto-exposure
    // scalar; PathTrace reads, AutoExposure writes).
    // Slot 7 -> binding 18 is the analytic-primitive BVH node buffer
    // (StructuredBuffer<float4> pairs, 32B per BvhNode).
    // Slots 8/9 -> bindings 19/20 are tri_bvh_nodes / tri_bvh_permuted_ids
    // (PR #106 follow-up). Bound on every dispatch so the partially-bound
    // semantics work and the shader's SW mesh path doesn't read an
    // unbound slot when the CSG bake has finished.
    // Slot 10 -> binding 21 is the SDF cluster buffer (SDF Phase 1, #97;
    // moved from binding 19 to make room for the triangle BVH).
    for (std::uint32_t s = 1; s < std::size(bound_buf_); ++s) {
        VkBuffer b = device_->LookupBuffer(bound_buf_[s]);
        if (b == VK_NULL_HANDLE) continue;
        add_buffer(kSlotToBufBinding[s], b, bound_buf_off_[s], VK_WHOLE_SIZE,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    // ---- Frame UBO at binding 14 (per-frame ring) ---------------------
    {
        const auto& ubo = device_->CurrentFrameUbo();
        // Copy spilled tail if push exceeds the on-chip range.
        if (push_size_ > VulkanDevice::kPushSplitOffset && ubo.mapped != nullptr) {
            std::size_t tail = push_size_ - VulkanDevice::kPushSplitOffset;
            // Silent clamp would truncate trailing PtPush fields and
            // surface as rendering corruption (the prior svgf_basic
            // regression: `write_hdr_aux` at UBO offset 600 read as 0,
            // so the path tracer skipped its denoise_color write and
            // SVGF sampled zeros -> black frame). LOG_ERROR once when
            // it happens so the next person hits a noisy failure instead
            // of squinting at the framebuffer.
            if (tail > VulkanDevice::kFrameUboSize) {
                static std::atomic<bool> s_warned{false};
                bool expected = false;
                if (s_warned.compare_exchange_strong(expected, true)) {
                    LOG_ERROR("Frame UBO too small: PtPush tail {} > kFrameUboSize {} -- bump kFrameUboSize in VulkanDevice.h; trailing PtPush fields will not reach the GPU and rendering will be corrupted",
                              tail, VulkanDevice::kFrameUboSize);
                }
                tail = VulkanDevice::kFrameUboSize;
            }
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
    rt_supported_ = has_ray_query && has_accel_struct && has_deferred_host_op;
    // One-line log for the bringup gate so the cause of a missing RT
    // path is obvious in the console. Pre-MoltenVK-1.3 builds report
    // 'ray_query=false accel_struct=false' here; on those drivers
    // the engine falls back to a CPU-built triangle BVH for CSG meshes
    // (see Engine::RebuildMeshResources + the bvh_tris software path
    // in PathTrace.slang).
    LOG_INFO("Vulkan RT extension presence: ray_query={} accel_struct={} deferred_host_op={} -> hw_rt={}",
             has_ray_query, has_accel_struct, has_deferred_host_op, rt_supported_);

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

    void** supported_next = &v12_supported.pNext;
    auto chain_supported = [&](void* node, void** node_next) {
        *supported_next = node;
        supported_next = node_next;
    };
    if (rt_supported_) {
        chain_supported(&as_supported, reinterpret_cast<void**>(&as_supported.pNext));
        chain_supported(&rq_supported, reinterpret_cast<void**>(&rq_supported.pNext));
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
    const bool supports_partially_bound =
        v12_supported.descriptorBindingPartiallyBound == VK_TRUE;
    const bool supports_buffer_device_address = v12_supported.bufferDeviceAddress == VK_TRUE;

    if (!supports_storage_image_rw_wo_format) {
        LOG_ERROR("Vulkan: missing shaderStorageImageRead/WriteWithoutFormat feature");
        return;
    }
    if (!supports_update_after_bind) {
        LOG_ERROR("Vulkan: UPDATE_AFTER_BIND features are required for shared descriptor-set dispatch");
        return;
    }
    if (!supports_partially_bound) {
        // PARTIALLY_BOUND lets dispatches leave unused slots in the
        // 20-binding shared layout unwritten. Core Vulkan 1.2 feature;
        // present on every desktop driver and MoltenVK.
        LOG_ERROR("Vulkan: descriptorBindingPartiallyBound is required by Vulkan descriptor binding strategy");
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
    // PARTIALLY_BOUND lets dispatches leave unused slots in the shared
    // 20-binding layout unwritten -- the descriptor-write path skips
    // them, the shader is required not to access them. Replaces the
    // older nullDescriptor-based scheme that MoltenVK doesn't support.
    v12.descriptorBindingPartiallyBound = VK_TRUE;
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
    // Pool sized for the per-frame dispatch ring of shared-layout sets
    // (see kDispatchSetsPerFrame in VulkanDevice.h for why a ring is
    // necessary -- UPDATE_AFTER_BIND descriptors are read at execution
    // time, so a single shared set across multiple recorded dispatches
    // races on the LATEST host-side binding update). Total sets =
    // kFramesInFlight * kDispatchSetsPerFrame, each set needs the full
    // per-binding allocation:
    //   11 storage_image + [1 accel_struct] + 10 storage_buffer + 1 ubo
    // The accel-struct pool entry is only emitted when rt_supported_
    // is true; ASKHR isn't a valid descriptor type at all on drivers
    // without VK_KHR_acceleration_structure (e.g. pre-1.3 MoltenVK).
    // The "+4" / "+1" headroom from the original sizing is preserved.
    // 11 storage_images: +1 over the legacy 10 is the accum_stars
    // binding 22 added for issue #46 (star-split bypass of SVGF;
    // moved from binding 21 in round-2 merge with main after SDF
    // Phase 1 #109 took binding 21 for its storage buffer).
    // PR #106 follow-up bumps storage_buffer per set from 7 to 9 for
    // tri_bvh_nodes / tri_bvh_permuted_ids (bindings 19/20); SDF
    // Phase 1 (#97) bumps it again from 9 to 10 for the SDF cluster
    // buffer at binding 21.
    const std::uint32_t kTotalSets =
        static_cast<std::uint32_t>(kFramesInFlight * kDispatchSetsPerFrame);
    std::vector<VkDescriptorPoolSize> psizes;
    psizes.push_back({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           kTotalSets * 11 + 4 });
    if (rt_supported_) {
        psizes.push_back({ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kTotalSets * 1 + 1 });
    }
    // 10 storage-buffer bindings per dispatch: mesh_positions, mesh_indices,
    // primitives, marginal_cdf, conditional_cdf, exposure_state, analytic
    // bvh_nodes (binding 18), tri_bvh_nodes (binding 19),
    // tri_bvh_permuted_ids (binding 20), sdf_clusters (binding 21).
    psizes.push_back({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          kTotalSets * 10 + 4 });
    psizes.push_back({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          kTotalSets * 1 + 1 });
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = kTotalSets;
    dpci.poolSizeCount = static_cast<std::uint32_t>(psizes.size());
    dpci.pPoolSizes    = psizes.data();
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                       | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    vkCreateDescriptorPool(device_, &dpci, nullptr, &dpool_);

    // ---- Build the unified descriptor set layout ----------------------
    // 23 bindings when hw RT is available (slot 2 = scene_tlas KHR AS),
    // 22 otherwise -- the AS descriptor type itself is invalid on
    // drivers without VK_KHR_acceleration_structure, so the layout has
    // to omit that binding entirely. The matching SPIR-V variant
    // (PathTrace_norq.spv) drops the scene_tlas declaration in lock
    // step so the shader's binding numbering still matches; the SW
    // mesh path reads mesh_positions / mesh_indices (bindings 3/4) and
    // (PR #106 follow-up) the host-built triangle BVH at bindings 19/20
    // exactly like the HW path does. SDF Phase 1 (#97) adds binding 21
    // for the SDF cluster buffer (moved from binding 19 to make room
    // for the triangle BVH). Binding 22 is cloud_trans_tex (R32F
    // per-pixel cloud transmittance G-buffer, issue #46 follow-up);
    // reuses the slot accum_stars (#108) briefly occupied before the
    // stateless composite rewrite freed it.
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(23);
        auto add_binding = [&](std::uint32_t binding, VkDescriptorType type) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding         = binding;
            lb.descriptorType  = type;
            lb.descriptorCount = 1;
            lb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b.push_back(lb);
        };
        // bindings 0-1: output + accum (storage image)
        add_binding(0,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        add_binding(1,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 2: scene_tlas (only when hw RT is supported)
        if (rt_supported_) {
            add_binding(2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
        }
        // bindings 3-5: mesh_positions, mesh_indices, primitives (storage buffer)
        add_binding(3,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        add_binding(4,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        add_binding(5,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // bindings 6-9: denoise_color, depth_tex, motion_tex, env_map (storage image)
        add_binding(6,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        add_binding(7,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        add_binding(8,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        add_binding(9,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // bindings 10-11: marginal_cdf, conditional_cdf (storage buffer)
        add_binding(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        add_binding(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // bindings 12-13: star_map, moon_map (storage image)
        add_binding(12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        add_binding(13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 14: Frame UBO (spilled push tail)
        add_binding(kFrameUboBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        // binding 15: exposure_state (GPU-driven auto-exposure scalar)
        add_binding(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // binding 16: normal_tex (path tracer G-buffer for SVGF/NRD/OptiX-AOV)
        add_binding(16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 17: albedo_tex (path tracer G-buffer for OptiX AOV only)
        add_binding(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // binding 18: bvh_nodes (analytic-prim BVH flat node array)
        add_binding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // bindings 19/20: tri_bvh_nodes / tri_bvh_permuted_ids
        // (PR #106 follow-up -- host-built triangle BVH used by the
        // shader's SW mesh path to replace the O(N) Möller-Trumbore
        // linear-scan loop. Always declared, even on HW-RT builds, so
        // the same SPIR-V variant works whether the engine populated
        // the BVH or left the slots default. PARTIALLY_BOUND means the
        // unbound case is fine; the shader's gate `bvh_params.w > 0`
        // is the runtime "is the BVH populated" signal).
        add_binding(19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        add_binding(20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // --- SDF Phase 1 (#97) ---
        // binding 21: sdf_clusters (flat per-cluster op-tree + AABB).
        // Moved here from binding 19 to make room for #106's
        // tri_bvh_nodes / tri_bvh_permuted_ids. PARTIALLY_BOUND means
        // the unbound case is fine (engine binds a placeholder
        // storage buffer when no SDF clusters exist); the shader's
        // gate `sdf_params.x > 0` is the runtime "have any clusters"
        // signal.
        add_binding(21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // --- end SDF Phase 1 ---
        // Binding 22: cloud_trans_tex. R32F storage image written by
        // PathTrace's volumetric cloud march, read by StarsComposite
        // for foreground-cloud occlusion of stars / sun / moon.
        // Allocated host-side when denoiser_active_; PARTIALLY_BOUND
        // covers the host-side gate's "no denoiser, no binding" case.
        add_binding(22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        // --- Light primitives (#73) ---
        // Binding 27: analytic light list (light_prims). Sits past
        // bindings 23..26 reserved for SIGMA #115 / MetalFX specular
        // #118. The engine binds a placeholder storage buffer when
        // no lights are active; the shader's `light_count > 0` gate
        // is the runtime signal.
        add_binding(27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        // --- end Light primitives ---

        // UPDATE_AFTER_BIND for every binding so we can rewrite the
        // shared descriptor set between dispatches in the same cmd
        // buffer without invalidating earlier recorded draws.
        // PARTIALLY_BOUND so dispatches can leave unused slots unwritten
        // -- the descriptor-write path skips slots the engine didn't
        // bind (the shader is required not to access those slots).
        std::vector<VkDescriptorBindingFlags> bind_flags(b.size(),
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
          | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
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
            LOG_ERROR("vkCreateDescriptorSetLayout (PathTrace shared layout) failed");
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

        // Allocate the full ring: kFramesInFlight * kDispatchSetsPerFrame
        // sets, all sharing the 20-binding layout. We allocate them in
        // one batch (each row corresponds to one frame slot, the
        // columns are the ring slots advanced by AcquireDispatch
        // DescriptorSet). Per the comment on dsets_ in the header,
        // each Dispatch grabs a fresh column so that the
        // UPDATE_AFTER_BIND read-at-execution semantic doesn't cause
        // PathTrace and bloom_pyramid (which both run on this layout)
        // to read each other's bindings.
        std::vector<VkDescriptorSetLayout> layouts(
            kFramesInFlight * kDispatchSetsPerFrame, shared_dset_layout_);
        std::vector<VkDescriptorSet> flat_dsets(
            kFramesInFlight * kDispatchSetsPerFrame, VK_NULL_HANDLE);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount =
            static_cast<std::uint32_t>(kFramesInFlight * kDispatchSetsPerFrame);
        dsai.pSetLayouts        = layouts.data();
        if (vkAllocateDescriptorSets(device_, &dsai, flat_dsets.data()) != VK_SUCCESS) {
            LOG_ERROR("Vulkan: vkAllocateDescriptorSets for shared-layout ring failed");
            return;
        }
        for (int f = 0; f < kFramesInFlight; ++f) {
            for (int s = 0; s < kDispatchSetsPerFrame; ++s) {
                dsets_[f][s] = flat_dsets[f * kDispatchSetsPerFrame + s];
            }
        }
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

        // Pick the PathTrace SPIR-V variant at pipeline build time:
        // rt_supported_ -> RT-enabled blob (RayQuery + scene_tlas binding);
        // !rt_supported_ -> norq blob (no RayQueryKHR capability, mesh
        // traversal degrades to the linear-scan path in PathTrace.slang).
        // The norq variant exists because some MoltenVK builds don't
        // expose VK_KHR_ray_query at all -- loading the RT-enabled blob
        // on those drivers fails vkCreateShaderModule with a
        // "RayQueryKHR capability declared but rayQuery feature missing"
        // validation error, taking the path tracer pipeline down with
        // it. See issue #44.
        if (rt_supported_) {
            build_pipeline("pathtrace", shader_PathTrace_spirv_data, shader_PathTrace_spirv_size);
        } else {
            build_pipeline("pathtrace", shader_PathTrace_norq_spirv_data, shader_PathTrace_norq_spirv_size);
        }
        build_pipeline("autoexpose",  shader_AutoExposure_spirv_data, shader_AutoExposure_spirv_size);
        build_pipeline("perfoverlay", shader_PerfOverlay_spirv_data,  shader_PerfOverlay_spirv_size);
        // Bloom pyramid pipelines. These re-use the shared 20-binding
        // pipeline layout (their only real bindings are 0 and 1, both
        // storage images, which the shared layout supports). The MSL
        // slot-padding dummies that BloomDown.slang / BloomUp.slang
        // emit on Metal are PT_TARGET_METAL-gated so the SPIR-V we
        // load here only declares the real bindings -- no clash with
        // the shared layout. Engine.cpp's existing post-denoise bloom
        // dispatch chain (BindComputePipeline + Dispatch) drives them
        // identically to the Metal path.
        build_pipeline("bloom_down",  shader_BloomDown_spirv_data,    shader_BloomDown_spirv_size);
        build_pipeline("bloom_up",    shader_BloomUp_spirv_data,      shader_BloomUp_spirv_size);
        // Tonemap pipeline. Engine.cpp's post-denoise tonemap chain
        // dispatches this when tonemap_pipeline_id_ != 0 AND
        // use_engine_tonemap is true. On Vulkan that's currently NEVER:
        // bloom-without-denoiser routes through Denoise(Kind::Finalize
        // Only) -> VulkanNrdDenoiser::EncodeFinalizeOnly (a dedicated
        // 4-binding layout that sidesteps the broken Tonemap.slang
        // black-screen path); SVGF/NRD denoiser stays on its own
        // DenoiseFinalize.slang; OptiX stays on its private-cb
        // finalize. Tonemap.slang remains BUILT on the SPIR-V path
        // (compiled and bound by name in case it's needed for future
        // routing) but is NOT exercised at runtime. See Tonemap.slang's
        // `#ifdef PT_TARGET_SPIRV` branch for the push/UBO split that
        // makes the 624-byte TonePush fit Vulkan's 256B push-constant
        // limit.
        build_pipeline("tonemap",     shader_Tonemap_spirv_data,      shader_Tonemap_spirv_size);
        pipelines_ready_.store(true, std::memory_order_release);

        // Skip the per-pipeline timing-string construction below tier 2.
        // The macro itself short-circuits format-args evaluation, but the
        // explicit `per_pipe` loop sits outside the macro and would run
        // unconditionally otherwise -- belt-and-braces with the macro's
        // own gate.
        if (pt::diag::TierEnabled(2)) {
            const double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            std::string per_pipe;
            for (const auto& p : timings) {
                if (!per_pipe.empty()) per_pipe += ", ";
                per_pipe += fmt::format("{} {:.0f}ms", p.name, p.ms);
            }
            PT_DIAG_TIER2("rhi.vulkan",
                          "async pipeline build done in {:.0f}ms ({})",
                          total_ms, per_pipe);
        }
    });

    if (!RecreateSwapchain()) return;

    wrapped_cb_ = std::make_unique<VulkanCommandBuffer>(this);

    // All required Vulkan objects are created. The factory in Device.cpp
    // checks this; without it, partial-init failure (e.g. a required
    // feature missing) silently produced a non-null unique_ptr whose
    // underlying VkDevice was VK_NULL_HANDLE, and the engine's
    // subsequent CreateBuffer/CreateTexture calls all failed with the
    // diagnostic-noise cascade ("exposure_state buffer creation
    // failed", "env_map: texture create/upload failed", ...).
    init_ok_ = true;
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
        if (swap_capture_staging_.buffer != VK_NULL_HANDLE) {
            DestroyBufferImpl(swap_capture_staging_);
            swap_capture_staging_ = {};
            swap_capture_staging_capacity_ = 0;
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
            PT_DIAG_TIER2("rhi.vulkan",
                          "pipeline cache rejected stale blob ({} bytes); rebuilding",
                          blob.size());
        }
    } else if (!blob.empty()) {
        PT_DIAG_TIER2("rhi.vulkan",
                      "pipeline cache loaded ({} bytes)", blob.size());
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
    // TRANSFER_SRC is requested when the platform's surface caps allow it,
    // so the screenshot-swap path's vkCmdCopyImageToBuffer is valid
    // (per VUID-vkCmdCopyImageToBuffer-srcImage-00188 the source image
    // must have VK_IMAGE_USAGE_TRANSFER_SRC_BIT). swap_supports_readback_
    // gates ReadbackSwapchain at runtime so on a stripped-down driver
    // that refuses TRANSFER_SRC the screenshot swap target just reports
    // unsupported instead of silently producing UB.
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        sci.imageUsage          |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        swap_supports_readback_  = true;
    } else {
        swap_supports_readback_  = false;
        LOG_INFO("swap surface does not advertise TRANSFER_SRC_BIT "
                 "(supportedUsageFlags=0x{:x}); ReadbackSwapchain will be unsupported",
                 static_cast<std::uint32_t>(caps.supportedUsageFlags));
    }
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
    // One-shot caller path: wait idle so the GPU isn't still using the
    // buffer. Batch teardown should use DestroyBufferNoWait + a single
    // upfront WaitIdle() instead (this method's wait amortises to one
    // per destroy, which is fine in isolation but N× for batches).
    vkDeviceWaitIdle(device_);
    DestroyBufferNoWait(h);
}

void VulkanDevice::DestroyBufferNoWait(BufferHandle h) {
    if (h.id == 0 || device_ == VK_NULL_HANDLE) return;
    std::lock_guard lock(resource_mutex_);
    auto it = buffers_.find(h.id);
    if (it == buffers_.end()) return;
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
        case TextureFormat::RG32F:       fmt = VK_FORMAT_R32G32_SFLOAT;        break;
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

bool VulkanDevice::ReadbackSwapchain(void* dst, std::size_t dst_size,
                                     std::uint32_t* out_w, std::uint32_t* out_h,
                                     SwapFormat* out_format) {
    if (dst == nullptr || device_ == VK_NULL_HANDLE) return false;
    if (!swap_supports_readback_) return false;

    // Non-blocking. The caller polls across ticks (the public contract
    // in Device.h's ReadbackSwapchain block documents this). First
    // call latches the request; later calls (after Submit() consumes
    // it) return true and memcpy. Blocking here would deadlock when
    // the caller is the same thread that runs the render loop -- the
    // engine's console-drain-on-main-thread case.
    if (!swap_capture_consumed_.load(std::memory_order_acquire)) {
        // Pre-flight extent sanity: we can't size dst_size against
        // the recording-time extent yet (it hasn't been recorded), so
        // use the current swap_extent_ as a conservative estimate.
        // A subsequent swap resize between this call and the next
        // poll is handled below by reading swap_capture_extent_.
        if (swap_extent_.width  == 0 || swap_extent_.height == 0) return false;
        const std::size_t need = std::size_t(swap_extent_.width) *
                                 swap_extent_.height * 4u;
        if (dst_size < need) return false;
        swap_capture_requested_.store(true, std::memory_order_release);
        return false;
    }

    // Capture-time extent (latched by Submit at the moment of
    // vkCmdCopyImageToBuffer) is what's in the staging buffer.
    // Don't use the current swap_extent_ -- a resize between the
    // record and this read would desync the byte count.
    const std::uint32_t cw = swap_capture_extent_.width;
    const std::uint32_t ch = swap_capture_extent_.height;
    const std::size_t bytes = std::size_t(cw) * ch * 4u;
    if (cw == 0 || ch == 0 || bytes == 0) {
        swap_capture_consumed_.store(false, std::memory_order_release);
        return false;
    }
    if (dst_size < bytes) {
        // Caller's buffer is too small for what was captured -- bail
        // and reset state so the next request gets a fresh latch.
        swap_capture_consumed_.store(false, std::memory_order_release);
        return false;
    }

    // WaitIdle here is cheap once the consuming frame's work has
    // already completed in real-time (the typical case once a frame
    // has elapsed since the request). Bounds the worst case if the
    // poll happens to fire before the GPU finishes. Safe to call now
    // that swap_capture_consumed_ is only published after the host
    // Submit returned (so WaitIdle is guaranteed to find the copy
    // queued and wait for its completion rather than no-oping).
    vkDeviceWaitIdle(device_);

    if (swap_capture_staging_.mapped == nullptr) {
        LOG_ERROR("ReadbackSwapchain: staging buffer has no host mapping after consume");
        swap_capture_consumed_.store(false, std::memory_order_release);
        return false;
    }
    std::memcpy(dst, swap_capture_staging_.mapped, bytes);
    swap_capture_consumed_.store(false, std::memory_order_release);

    if (out_w) *out_w = cw;
    if (out_h) *out_h = ch;
    if (out_format) {
        switch (swap_format_) {
            case VK_FORMAT_B8G8R8A8_UNORM: *out_format = SwapFormat::Bgra8Unorm; break;
            case VK_FORMAT_B8G8R8A8_SRGB:  *out_format = SwapFormat::Bgra8Srgb;  break;
            case VK_FORMAT_R8G8B8A8_UNORM: *out_format = SwapFormat::Rgba8Unorm; break;
            case VK_FORMAT_R8G8B8A8_SRGB:  *out_format = SwapFormat::Rgba8Srgb;  break;
            default:                       *out_format = SwapFormat::Other;      break;
        }
    }
    return true;
}

void VulkanDevice::DestroyTexture(TextureHandle h) {
    if (h.id == 0 || h.id == kSwapchainTextureId) return;
    // One-shot caller path: wait idle so the GPU isn't still using the
    // texture. Batch teardown should use DestroyTextureNoWait + a single
    // upfront WaitIdle() instead (this method's wait amortises to one
    // per destroy, which is fine in isolation but N× for batches).
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    DestroyTextureNoWait(h);
}

void VulkanDevice::DestroyTextureNoWait(TextureHandle h) {
    if (h.id == 0 || h.id == kSwapchainTextureId || device_ == VK_NULL_HANDLE) return;
    std::lock_guard lock(resource_mutex_);
    auto it = images_.find(h.id);
    if (it == images_.end()) return;
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
    // Reset the per-frame descriptor-set ring so the first Dispatch
    // of this frame takes set 0. Safe to reset here because BeginFrame
    // already vkWaitForFences'd the previous use of this frame slot
    // (so the GPU is done reading those sets and we can safely
    // re-bind them with fresh values).
    next_dispatch_set_ = 0;
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

    // ReadbackSwapchain: if a caller requested a one-shot capture of
    // the swap image's contents, claim it now and record the copy into
    // the engine cb. Helper handles the atomic exchange + barriers +
    // copy + staging-buffer (re)allocation; we only care about whether
    // it claimed so we know to publish swap_capture_consumed_ after
    // vkQueueSubmit returns.
    //
    // Why publish AFTER submit (not inside the helper): a caller
    // polling on the flag and immediately calling vkDeviceWaitIdle
    // could observe a mid-recording publish, find the queue idle
    // (nothing submitted yet), and memcpy STALE staging. Pushing the
    // publish past vkQueueSubmit means by the time the flag is
    // observable the copy is at least in the queue -- WaitIdle then
    // actually waits for it.
    //
    // The OptiX path also calls TryRecordSwapchainCapture from inside
    // its private cb (see VulkanOptixDenoiser::Encode). The atomic
    // exchange ensures only one cb records the copy per request: on
    // the OptiX path the OptiX private cb claims first (because its
    // Encode runs during engine-cb recording, before Submit), and the
    // call here sees the flag already false and skips. That's the
    // intended behaviour -- the engine cb's pre-finalize swap image
    // doesn't contain the OptiX bloom+tonemap composite, so capturing
    // it here would write a bloom-less screenshot.
    const bool capture_recorded_this_submit =
        TryRecordSwapchainCapture(cmd,
                                  swap_images_[current_swap_index_],
                                  swap_extent_);

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
    // srcStage includes TRANSFER_BIT so the layout transition is
    // ordered after BOTH the engine's compute writes AND any in-flight
    // TRANSFER_READ from TryRecordSwapchainCapture's
    // vkCmdCopyImageToBuffer above. With only COMPUTE_SHADER_BIT,
    // execution-dependency rules would let the transition race the
    // copy on a strict implementation -- the layout is GENERAL on both
    // sides so the layout flip is trivial, but the present's read of
    // the image still needs to happen-after the copy completes.
    // Unconditional widening is harmless when no capture was recorded:
    // no TRANSFER-stage work means no extra synchronisation.
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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

    // ReadbackSwapchain consume publish: AFTER vkQueueSubmit. By the
    // time another thread observes this flag, the copy is at least
    // queued -- so a follow-up vkDeviceWaitIdle on that thread will
    // actually wait for the copy to finish (rather than returning
    // immediately because nothing was submitted, which would let the
    // caller memcpy stale staging). Order matters here: store-release
    // pairs with the load-acquire in ReadbackSwapchain.
    if (capture_recorded_this_submit) {
        PublishSwapchainCaptureConsumed();
    }

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

bool VulkanDevice::TryRecordSwapchainCapture(VkCommandBuffer cb,
                                             VkImage         swap_image,
                                             VkExtent2D      extent) {
    if (cb == VK_NULL_HANDLE || swap_image == VK_NULL_HANDLE) return false;
    if (extent.width == 0 || extent.height == 0)               return false;
    // Atomic exchange: only one cb claims a given request. The OptiX
    // private cb's recording (inside VulkanOptixDenoiser::Encode) races
    // here against the engine cb's recording (in VulkanDevice::Submit)
    // for the same request. Whichever calls this method first wins;
    // the other sees false and skips.
    if (!swap_capture_requested_.exchange(false, std::memory_order_acq_rel)) {
        return false;
    }

    const std::size_t need_bytes =
        std::size_t(extent.width) * extent.height * 4u;
    if (need_bytes > 0 && swap_capture_staging_capacity_ < need_bytes) {
        if (swap_capture_staging_.buffer != VK_NULL_HANDLE) {
            DestroyBufferImpl(swap_capture_staging_);
            swap_capture_staging_ = {};
            swap_capture_staging_capacity_ = 0;
        }
        if (CreateBufferImpl(need_bytes,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             swap_capture_staging_,
                             /*persistent_map=*/true)) {
            swap_capture_staging_capacity_ = need_bytes;
        } else {
            LOG_ERROR("ReadbackSwapchain: staging buffer alloc failed ({} bytes)",
                      need_bytes);
        }
    }
    if (swap_capture_staging_.buffer == VK_NULL_HANDLE ||
        swap_capture_staging_capacity_ < need_bytes) {
        // Staging wasn't usable; drop the claim. The caller's poll will
        // see neither consumed nor requested and re-issue on the next
        // ReadbackSwapchain call.
        return false;
    }

    // Image memory barrier: make the prior dispatches' SHADER_WRITE_BIT
    // on the swap image visible to TRANSFER_READ_BIT by
    // vkCmdCopyImageToBuffer. Without this the copy can observe stale
    // (or partially-written) compute output -- queue order alone
    // doesn't provide the memory dependency
    // (per VUID-vkCmdCopyImageToBuffer-srcImage-00188 / the execution +
    // memory rules around inter-stage hazards).
    VkImageMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    mb.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    mb.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.image         = swap_image;
    mb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    mb.subresourceRange.layerCount = 1;
    mb.subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &mb);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { extent.width, extent.height, 1 };
    vkCmdCopyImageToBuffer(cb, swap_image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           swap_capture_staging_.buffer, 1, &region);
    // Make the transfer-write visible to host reads after the submission
    // completes. ReadbackSwapchain's WaitIdle handles the GPU-side wait;
    // this buffer barrier handshakes the memory visibility step inside
    // the same command buffer.
    VkBufferMemoryBarrier bb{};
    bb.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer        = swap_capture_staging_.buffer;
    bb.offset        = 0;
    bb.size          = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &bb, 0, nullptr);
    // Latch the recording-time extent so ReadbackSwapchain can compute
    // its memcpy size from this (NOT the current swap_extent_, which
    // may change under a resize between record and read).
    swap_capture_extent_.width  = extent.width;
    swap_capture_extent_.height = extent.height;
    return true;
}

void VulkanDevice::PublishSwapchainCaptureConsumed() {
    // store-release pairs with the load-acquire in ReadbackSwapchain.
    swap_capture_consumed_.store(true, std::memory_order_release);
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
                                         VkImageView    bloom_in_view,
                                         float          bloom_intensity,
                                         std::uint32_t  width,
                                         std::uint32_t  height,
                                         bool           hdr_pipeline,
                                         VkImageView    stars_in_view) {
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
    // EncodeFinalizeOnly's contract: a null bloom view forces intensity
    // to 0 internally and binds color_in as a safe-but-unread fallback,
    // so the OptiX bloom-off case (engine passes view=NULL, intensity=0)
    // and the OptiX bloom-on case (view=mip0, intensity>0) both go
    // through the same call. The stars_in_view contract mirrors that
    // exactly: a null view also falls back to color_in plus
    // stars_present=0 internally, so the OptiX path can pass
    // VK_NULL_HANDLE when star_split is off without producing a black
    // frame. When stars are routed the engine passes the real
    // accum_stars accumulator and stars_present resolves to 1 inside
    // EncodeFinalizeOnly.
    denoiser_->EncodeFinalizeOnly(cb, color_in_view, final_output_view,
                                  exposure_state_buf,
                                  bloom_in_view, bloom_intensity,
                                  width, height, hdr_pipeline,
                                  stars_in_view);
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
            LOG_INFO("VulkanDevice::Denoise: dispatching kind={} "
                     "(0=Svgf, 1=OptixHdr, 2=OptixHdrAov, "
                     "3=OptixTemporalHdr, 4=OptixTemporalHdrAov, "
                     "5=MetalFX, 6=SvgfMetalFx, 7=FinalizeOnly), "
                     "color={} out={} normal={} depth={} motion={}",
                     k, d.color_in.id, d.output.id, d.normal_in.id, d.depth_in.id, d.motion_in.id);
            s_last_logged_kind = k;
        }
    }

    // Route based on d.kind. SVGF and OptiX have different input
    // contracts: SVGF wants the full G-buffer (color/depth/motion/
    // normal/output); OptiX HDR only needs color + output; OptiX HDR
    // AOV adds albedo + normal (the AOV model's OptixDenoiserGuideLayer);
    // OptiX TEMPORAL adds motion_in (fed as guide_layer.flow) plus a
    // CUDA-side previous-output history buffer the denoiser maintains
    // internally; OptiX TEMPORAL_AOV is the union (motion + albedo +
    // normal + history).
#if defined(PT_ENABLE_OPTIX)
    // Kind-classification bools live inside the PT_ENABLE_OPTIX guard
    // because they're only consumed by the OptiX dispatch block.
    // Hoisting them out triggers Mac CI's zero-warnings gate
    // (Wunused-variable) since the Mac build defines no OptiX kinds.
    const bool kind_is_optix_temporal =
        (d.kind == DenoiseDesc::Kind::OptixTemporalHdr ||
         d.kind == DenoiseDesc::Kind::OptixTemporalHdrAov);
    const bool kind_is_optix_aov =
        (d.kind == DenoiseDesc::Kind::OptixHdrAov ||
         d.kind == DenoiseDesc::Kind::OptixTemporalHdrAov);
    const bool kind_is_optix =
        (d.kind == DenoiseDesc::Kind::OptixHdr ||
         d.kind == DenoiseDesc::Kind::OptixHdrAov ||
         kind_is_optix_temporal);
    if (kind_is_optix) {
        if (d.color_in.id == 0 || d.output.id == 0) {
            LOG_WARN("VulkanDevice::Denoise(OptiX): missing color/output (color={} out={})",
                     d.color_in.id, d.output.id);
            return;
        }
        if (kind_is_optix_aov) {
            if (d.albedo_in.id == 0 || d.normal_in.id == 0) {
                LOG_WARN("VulkanDevice::Denoise(OptiX AOV): missing albedo/normal "
                         "(albedo={} normal={}) -- engine should have allocated these "
                         "for the optix_*_aov variants; check "
                         "Engine::want_albedo_gbuffer / want_normal_gbuffer paths",
                         d.albedo_in.id, d.normal_in.id);
                return;
            }
        }
        if (kind_is_optix_temporal) {
            if (d.motion_in.id == 0) {
                LOG_WARN("VulkanDevice::Denoise(OptiX Temporal): missing motion "
                         "(motion={}) -- the temporal model needs the per-pixel "
                         "reprojection delta as guide_layer.flow input; engine "
                         "always allocates motion_tex when denoiser_active",
                         d.motion_in.id);
                return;
            }
        }
        VulkanOptixDenoiser::Kind opt_kind;
        switch (d.kind) {
            case DenoiseDesc::Kind::OptixHdr:             opt_kind = VulkanOptixDenoiser::Kind::Hdr; break;
            case DenoiseDesc::Kind::OptixHdrAov:          opt_kind = VulkanOptixDenoiser::Kind::HdrAov; break;
            case DenoiseDesc::Kind::OptixTemporalHdr:     opt_kind = VulkanOptixDenoiser::Kind::TemporalHdr; break;
            case DenoiseDesc::Kind::OptixTemporalHdrAov:  opt_kind = VulkanOptixDenoiser::Kind::TemporalHdrAov; break;
            default: opt_kind = VulkanOptixDenoiser::Kind::Hdr; break;  // unreachable
        }
        // Detect runtime r_denoiser cvar transitions that swap kinds
        // (optix_hdr <-> optix_hdr_aov). The denoiser builds its OptiX
        // state buffer + external buffers around `kind_` at construction,
        // so we can't just flip the field on the live instance -- tear
        // it down and rebuild. Destruction drains CUDA + frees external
        // memory + destroys the OptiX context, so this is safe to do
        // between frames (the engine clears pending_post_main_ inside
        // SubmitPostMain at the end of every frame). Cheap: state +
        // scratch are <200MB and the rebuild's ~tens-of-ms.
        if (optix_denoiser_ != nullptr &&
            optix_denoiser_->GetKind() != opt_kind) {
            LOG_INFO("VulkanDevice::Denoise: OptiX kind transition "
                     "(was {}, now {}); rebuilding denoiser instance",
                     static_cast<int>(optix_denoiser_->GetKind()),
                     static_cast<int>(opt_kind));
            optix_denoiser_.reset();
        }
        if (optix_denoiser_ == nullptr) {
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

    // ---- FinalizeOnly path (Kind::FinalizeOnly) ------------------------
    // Run JUST the swapchain finalize stage (linear HDR -> exposure ->
    // ACES -> sRGB OETF + bloom composite) via VulkanNrdDenoiser::
    // EncodeFinalizeOnly. Skips every temporal / spatial denoising pass.
    // Used by the engine's bloom-without-denoiser path on Vulkan, where
    // the standalone Tonemap.slang compute kernel that Metal uses for
    // the same job produces a black swapchain (descriptor-visibility
    // bug against the shared 20-binding layout, not yet root-caused).
    // EncodeFinalizeOnly has its own dedicated 4-binding layout +
    // descriptor set ring, so it sidesteps the Tonemap.slang issue.
    if (d.kind == DenoiseDesc::Kind::FinalizeOnly) {
        if (d.color_in.id == 0 || d.final_output.id == 0) {
            LOG_WARN("VulkanDevice::Denoise(FinalizeOnly): missing "
                     "color_in / final_output (color={} final={})",
                     d.color_in.id, d.final_output.id);
            return;
        }
        if (denoiser_ == nullptr) {
            denoiser_ = std::make_unique<VulkanNrdDenoiser>(this);
        }
        if (!denoiser_->Ready()) {
            if (!denoiser_->Init()) {
                LOG_ERROR("VulkanDevice::Denoise(FinalizeOnly): NRD "
                          "denoiser Init failed; bloom composite + "
                          "swapchain write will be skipped this frame");
                denoiser_.reset();
                return;
            }
        }
        auto image_it = images_.find(d.color_in.id);
        if (image_it == images_.end()) {
            LOG_WARN("VulkanDevice::Denoise(FinalizeOnly): color_in "
                     "image lookup miss (id={})", d.color_in.id);
            return;
        }
        const std::uint32_t w = image_it->second.extent.width;
        const std::uint32_t h = image_it->second.extent.height;
        if (w == 0 || h == 0) return;

        VkImageView color_view  = LookupImageView(d.color_in);
        VkImageView output_view = LookupImageView(d.final_output);
        VkImageView bloom_view  = (d.bloom_in.id != 0)
                                    ? LookupImageView(d.bloom_in)
                                    : VK_NULL_HANDLE;
        // Star-split accumulator (issue #46). On the FinalizeOnly
        // route (bloom-without-denoiser path) PathTrace.slang skips
        // its accum_stars write because star_split_enabled requires
        // denoiser_enabled. The engine therefore binds a 1x1
        // placeholder here purely for descriptor validity, and the
        // shader's `stars_present` push gate elides the additive
        // read. The denoiser-active path is handled in the SVGF/NRD
        // branch below via d.stars_in (real accumulator when
        // r_star_split is on, placeholder otherwise).
        VkImageView stars_view  = (d.stars_in.id != 0)
                                    ? LookupImageView(d.stars_in)
                                    : VK_NULL_HANDLE;
        VkBuffer    exposure_buf = LookupBuffer(d.exposure_state);

        denoiser_->EncodeFinalizeOnly(wrapped_cb_->Raw(),
                                      color_view, output_view,
                                      exposure_buf,
                                      bloom_view, d.bloom_intensity,
                                      w, h, d.hdr_pipeline, stars_view);
        return;
    }

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
                      d.bloom_in, d.bloom_intensity,
                      d.reset_history,
                      atrous_enabled,
                      d.atrous_passes,
                      d.hdr_pipeline,
                      d.stars_in);
}

}  // namespace pt::rhi::vk

namespace pt::rhi {
std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle& w) {
    auto dev = std::make_unique<vk::VulkanDevice>(w);
    if (!dev->IsInitialized()) {
        // Constructor early-returned (missing required feature, failed
        // vkCreateDevice, swapchain create failed, ...). Drop the
        // half-built object so Engine::RequestBackendSwitch sees the
        // factory return nullptr and rolls back to BackendType::None
        // cleanly, rather than driving render through a dead device.
        return nullptr;
    }
    return dev;
}
}  // namespace pt::rhi

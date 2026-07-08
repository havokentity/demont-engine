// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "VulkanDenoiser.h"
#include "VulkanDevice.h"

#include "../core/Diag.h"
#include "../core/Log.h"
#include "../rhi/Resources.h"

#include <array>
#include <cstring>
#include <vector>

extern "C" {
extern const unsigned char shader_DenoiseTemporal_spirv_data[];
extern const unsigned long shader_DenoiseTemporal_spirv_size;
extern const unsigned char shader_DenoiseAtrous_spirv_data[];
extern const unsigned long shader_DenoiseAtrous_spirv_size;
extern const unsigned char shader_DenoiseFinalize_spirv_data[];
extern const unsigned long shader_DenoiseFinalize_spirv_size;
// Issue #119 -- albedo remod kernel.
extern const unsigned char shader_DenoiseRemod_spirv_data[];
extern const unsigned long shader_DenoiseRemod_spirv_size;
}

namespace pt::rhi::vk {

namespace {

VkShaderModule MakeModule(VkDevice dev,
                          const unsigned char* bytes, std::size_t n) {
    // VkShaderModuleCreateInfo::pCode requires 4-byte-aligned SPIR-V
    // words. The embedded SPIR-V blobs come from EmbedFile.cmake as
    // `const unsigned char[]` -- byte-aligned by the C++ standard.
    // Casting to `const uint32_t*` is undefined behavior if the array
    // happened to land on a non-4-byte boundary (and would trap on
    // strict-alignment archs / be flagged by UBSan). Side-step it by
    // memcpying into an aligned `std::vector<uint32_t>` before handing
    // pCode to the driver. SPIR-V is always a multiple of 4 bytes, so
    // the size check is also a sanity gate against a corrupt blob.
    if (n == 0 || (n % sizeof(std::uint32_t)) != 0) {
        LOG_ERROR("MakeModule: invalid SPIR-V byte length {} (must be >0 and 4-aligned)", n);
        return VK_NULL_HANDLE;
    }
    std::vector<std::uint32_t> aligned(n / sizeof(std::uint32_t));
    std::memcpy(aligned.data(), bytes, n);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = n;
    ci.pCode    = aligned.data();
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

// Push struct shared by both denoise kernels. 48 bytes total (was 32
// pre-#119). Field names are kernel-specific (see DenoiseTemporal.slang
// / DenoiseAtrous.slang for what each lane means at each call site).
// Issue #119 appended demod_enabled + final_remod (atrous only) and
// two padding lanes; std140 push blocks align at 16 bytes so 48 is
// the next multiple after 40.
struct DenoisePush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t step_or_reset;     // temporal: reset_history; atrous: step_size
    std::uint32_t pad0;
    float         a;                  // temporal: depth_tolerance; atrous: sigma_depth
    float         b;                  // temporal: normal_tolerance; atrous: sigma_normal
    float         c;                  // temporal: min_alpha; atrous: sigma_color
    float         pad1;
    std::uint32_t demod_enabled;     // issue #119; 1 = run on demodulated lighting
    std::uint32_t final_remod;       // issue #119; atrous-only; multiply albedo back on output write
    std::uint32_t pad2;
    std::uint32_t pad3;
};
static_assert(sizeof(DenoisePush) == 48, "DenoisePush layout");

// Issue #119 -- 16-byte push struct for the remod-only kernel
// (DenoiseRemod.slang). Just the dispatch size; the kernel reads
// demod_in + albedo_tex and writes color_out.
struct RemodPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pad0;
    std::uint32_t pad1;
};
static_assert(sizeof(RemodPush) == 16, "RemodPush layout");

// Two-stage compute-write -> compute-read barrier. Used between sub-
// passes that write/read the same scratch texture (history_b after
// temporal, atrous ping-pong textures, output).
void ComputeChainBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Generic stage/access barrier. Used to bracket the vkCmdCopyImage in
// the svgf_basic path: the temporal compute write must be visible to
// the transfer read of the same image, and the transfer write to the
// caller's output must be visible to the subsequent bloom/tonemap
// compute reads. All affected images stay in VK_IMAGE_LAYOUT_GENERAL,
// so a memory-only barrier is sufficient -- no image-layout transition
// needed.
void StageBarrier(VkCommandBuffer cb,
                  VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                  VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

}  // namespace

VulkanNrdDenoiser::~VulkanNrdDenoiser() {
    DestroyAll();
}

void VulkanNrdDenoiser::DestroyAll() {
    if (device_ == nullptr) return;
    VkDevice dev = device_->RawDevice();
    if (dev == VK_NULL_HANDLE) return;

    // Batch teardown: one WaitIdle covers everything destroyed below
    // (scratch textures/buffers via DestroyTextures, the three 1x1
    // dummies, the pipelines, and the pool). Per-resource Destroy*
    // calls would otherwise each issue their own vkDeviceWaitIdle ->
    // ~15 idle stalls on a shutdown / re-Init. The pipeline/pool
    // destroys are inherently non-waiting (vkDestroyPipeline /
    // vkDestroyDescriptorPool have no implicit-wait semantic).
    device_->WaitIdle();

    DestroyTexturesAlreadyWaited();
    if (dummy_color_id_     != 0) device_->DestroyTextureNoWait(TextureHandle{dummy_color_id_});
    if (dummy_motion_id_    != 0) device_->DestroyTextureNoWait(TextureHandle{dummy_motion_id_});
    if (dummy_variance_buf_ != 0) device_->DestroyBufferNoWait(BufferHandle{dummy_variance_buf_});
    dummy_color_id_     = 0;
    dummy_motion_id_    = 0;
    dummy_variance_buf_ = 0;

    if (temporal_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, temporal_pipe_, nullptr);
        temporal_pipe_ = VK_NULL_HANDLE;
    }
    if (atrous_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, atrous_pipe_, nullptr);
        atrous_pipe_ = VK_NULL_HANDLE;
    }
    if (remod_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, remod_pipe_, nullptr);
        remod_pipe_ = VK_NULL_HANDLE;
    }
    if (finalize_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, finalize_pipe_, nullptr);
        finalize_pipe_ = VK_NULL_HANDLE;
    }
    if (pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, pipe_layout_, nullptr);
        pipe_layout_ = VK_NULL_HANDLE;
    }
    if (remod_pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, remod_pipe_layout_, nullptr);
        remod_pipe_layout_ = VK_NULL_HANDLE;
    }
    if (finalize_pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, finalize_pipe_layout_, nullptr);
        finalize_pipe_layout_ = VK_NULL_HANDLE;
    }
    if (dset_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, dset_layout_, nullptr);
        dset_layout_ = VK_NULL_HANDLE;
    }
    if (remod_dset_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, remod_dset_layout_, nullptr);
        remod_dset_layout_ = VK_NULL_HANDLE;
    }
    if (finalize_dset_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, finalize_dset_layout_, nullptr);
        finalize_dset_layout_ = VK_NULL_HANDLE;
    }
    if (dpool_ != VK_NULL_HANDLE) {
        // VkDescriptorPool destruction frees all sets; clear the cache
        // first so we don't accidentally reuse stale handles after
        // re-Init().
        for (auto& s : sets_)          s = VK_NULL_HANDLE;
        for (auto& s : finalize_sets_) s = VK_NULL_HANDLE;
        for (auto& s : remod_sets_)    s = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(dev, dpool_, nullptr);
        dpool_ = VK_NULL_HANDLE;
    }
    ready_ = false;
}

void VulkanNrdDenoiser::DestroyTextures() {
    if (device_ == nullptr) return;
    // Batch the 12 scratch resources behind a single WaitIdle. The
    // per-resource Destroy* path each calls vkDeviceWaitIdle internally,
    // which adds up to 12× stalls on a resize (8 textures + 4 storage
    // buffers); a single up-front wait + NoWait destroys is identical
    // in safety but pays one stall.
    device_->WaitIdle();
    DestroyTexturesAlreadyWaited();
}

void VulkanNrdDenoiser::DestroyTexturesAlreadyWaited() {
    if (device_ == nullptr) return;
    auto relTex = [&](std::uint64_t& id) {
        if (id != 0) { device_->DestroyTextureNoWait(TextureHandle{id}); id = 0; }
    };
    auto relBuf = [&](std::uint64_t& id) {
        if (id != 0) { device_->DestroyBufferNoWait(BufferHandle{id}); id = 0; }
    };
    relTex(history_a_id_);
    relTex(history_b_id_);
    relTex(depth_history_a_id_);
    relTex(depth_history_b_id_);
    relTex(normal_history_a_id_);
    relTex(normal_history_b_id_);
    relTex(atrous_a_id_);
    relTex(atrous_b_id_);
    relBuf(moments_history_a_buf_);
    relBuf(moments_history_b_buf_);
    relBuf(variance_a_buf_);
    relBuf(variance_b_buf_);
    cached_w_ = cached_h_ = 0;
    needs_history_clear_ = true;
}

bool VulkanNrdDenoiser::Init() {
    if (ready_) return true;
    if (device_ == nullptr || device_->RawDevice() == VK_NULL_HANDLE) return false;
    if (!BuildLayout())          { DestroyAll(); return false; }
    if (!BuildPipelines())       { DestroyAll(); return false; }
    if (!BuildDescriptorPool())  { DestroyAll(); return false; }

    // Lazy 1x1 RGBA16F placeholder for the atrous-pass color_history /
    // motion bindings (declared in the shader but unused at runtime).
    // Robustness2's nullDescriptor would let us bind VK_NULL_HANDLE,
    // but a real 1x1 image makes the descriptor write trivially valid
    // on drivers / validation modes that flag null storage images as
    // suspect. Storage layout gets these in GENERAL after CreateTexture.
    auto color_h = device_->CreateTexture({
        .width = 1, .height = 1,
        .format = TextureFormat::RGBA16F,
        .usage  = TextureUsage::Storage,
        .debug_name = "denoise_dummy_color",
    });
    auto mot_h = device_->CreateTexture({
        .width = 1, .height = 1,
        .format = TextureFormat::RG16F,
        .usage  = TextureUsage::Storage,
        .debug_name = "denoise_dummy_motion",
    });
    auto var_buf = device_->CreateBuffer({
        .size       = 16,
        .usage      = BufferUsage::Storage,
        .debug_name = "denoise_dummy_variance",
    });
    dummy_color_id_     = color_h.id;
    dummy_motion_id_    = mot_h.id;
    dummy_variance_buf_ = var_buf.id;
    if (dummy_color_id_ == 0 || dummy_motion_id_ == 0 ||
        dummy_variance_buf_ == 0) {
        LOG_ERROR("VulkanNrdDenoiser: dummy resource allocation failed");
        DestroyAll();
        return false;
    }

    ready_ = true;
    PT_DIAG_TIER1("denoiser",
                  "VulkanNrdDenoiser: SVGF compute pipelines + dummy "
                  "history textures ready");
    return true;
}

bool VulkanNrdDenoiser::BuildLayout() {
    VkDevice dev = device_->RawDevice();

    // ---- temporal/atrous: 9 storage images + 4 storage buffers -----
    // Per-slot semantics (shared by both passes; unused bindings are
    // satisfied with dummy / same-format placeholders at runtime):
    //   0  color_in            (storage image, RGBA16F)
    //   1  color_history_in    (storage image, RGBA16F; atrous: dummy)
    //   2  depth_tex           (storage image, R32F)
    //   3  motion_tex          (storage image, RG16F;   atrous: dummy)
    //   4  normal_tex          (storage image, RGBA16F)
    //   5  color_out           (storage image, RGBA16F)
    //   6  depth_history_in    (storage image, R32F;    atrous: any R32F)
    //   7  normal_history_in   (storage image, RGBA16F; atrous: any RGBA16F)
    //   8  moments_history_in  (storage buffer, float2[]; atrous: any)
    //   9  moments_history_out (storage buffer, float2[]; atrous: any)
    //   10 variance_in         (storage buffer, float[];  temporal: dummy)
    //   11 variance_out        (storage buffer, float[])
    //   12 albedo_tex          (storage image, RGBA16F)  -- issue #119
    // The 4 storage buffers replace what would otherwise be 4 more
    // storage images -- Metal compute shaders cap at 8 textures with
    // access::read_write per kernel, and 12 would crash slangc's MSL.
    // Slot 12's albedo is declared `Texture2D` (sample-only in MSL,
    // OpImageRead-only in SPIR-V), so it stays under the cap on Metal
    // and is a STORAGE_IMAGE in Vulkan -- matches the rest of the
    // engine's "no samplers" compute setup.
    {
        // Bindings 0..7 storage images, 8..11 storage buffers, 12 storage image.
        // Non-contiguous: skip 8..11 in the image block and add 12 explicitly.
        constexpr std::uint32_t kStorageImageSlots = kPassImages;   // 9 total: 0..7 + 12
        constexpr std::uint32_t kStorageBufferSlots = kPassBuffers; // 4 total: 8..11
        constexpr std::uint32_t kTotal = kStorageImageSlots + kStorageBufferSlots;
        std::array<VkDescriptorSetLayoutBinding, kTotal> b{};
        for (std::uint32_t i = 0; i < 8u; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        for (std::uint32_t i = 0; i < kPassBuffers; ++i) {
            const std::uint32_t idx = 8u + i;
            b[idx].binding         = idx;
            b[idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[idx].descriptorCount = 1;
            b[idx].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        // Slot 12 (the 9th storage image). Sits in the last array
        // entry; layout creation order doesn't require ascending
        // binding numbers but does require uniqueness, so this lands
        // at array index 12 with binding 12.
        b[12].binding         = 12;
        b[12].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[12].descriptorCount = 1;
        b[12].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<std::uint32_t>(b.size());
        dslci.pBindings    = b.data();
        if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dset_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkCreateDescriptorSetLayout failed");
            return false;
        }
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(DenoisePush);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dset_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipe_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkCreatePipelineLayout failed");
            return false;
        }
    }

    // ---- remod-only layout: 3 storage images (issue #119) -----------
    // Bindings: 0 demod_in, 1 color_out, 12 albedo_tex. Mirrors the
    // DenoiseRemod.slang declarations one-for-one so the shader's
    // descriptor-set zero binds straight in. Same shader-stage flag as
    // temporal/atrous, lives in the same descriptor pool (sized
    // mixed-pool below).
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[2] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<std::uint32_t>(b.size());
        dslci.pBindings    = b.data();
        if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &remod_dset_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: remod descriptor set layout failed");
            return false;
        }
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(RemodPush);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &remod_dset_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &remod_pipe_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: remod pipeline layout failed");
            return false;
        }
    }

    // ---- finalize: 3 storage images + 1 storage buffer --------------
    // 0: hdr_in       (RGBA16F linear-HDR post-denoise)
    // 1: swap_out     (BGRA8 swapchain)
    // 2: exposure_state (storage buffer, written by AutoExposure.slang)
    // 3: bloom_in     (RGBA16F bloom mip 0 when bloom is enabled).
    //                  Disabled-bloom contract: the engine passes
    //                  dd.bloom_in = 0 and dd.bloom_intensity = 0;
    //                  Encode() / EncodeFinalizeOnly() bind v_output
    //                  (color_in_view) as a safe-but-unread fallback
    //                  and force the push's bloom_intensity to 0 so
    //                  the shader's `bloom_intensity > 0` gate skips
    //                  the sample. No 1x1 placeholder texture is
    //                  required on this path (Tonemap.slang's slot 2
    //                  path is a separate Mac-only contract that does
    //                  use the engine's bloom_dummy_tex_id_).
    {
        // Layout: hdr_in / swap_out / exposure_state / bloom_tex.
        // Binding 4 (stars_tex, #108) retired with the stateless
        // StarsComposite rewrite -- celestials composite Metal-side on
        // a dedicated pipeline; the Vulkan denoiser path has no stars
        // compositor in this PR.
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<std::uint32_t>(b.size());
        dslci.pBindings    = b.data();
        if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &finalize_dset_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: finalize descriptor set layout failed");
            return false;
        }
        // Finalize push: width, height, hdr_pipeline, bloom_intensity,
        // and 16 trailing bytes of reserved padding kept for layout
        // stability (DenoiseFinalize.slang still declares the tail as
        // `_reserved_stars + 3 pad uints` to keep std140 alignment).
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = 32;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &finalize_dset_layout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &finalize_pipe_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: finalize pipeline layout failed");
            return false;
        }
    }
    return true;
}

bool VulkanNrdDenoiser::BuildPipelines() {
    VkDevice dev = device_->RawDevice();

    auto build = [&](const unsigned char* spirv, std::size_t spirv_size,
                     VkPipelineLayout layout, VkPipeline& out, const char* label) -> bool {
        VkShaderModule m = MakeModule(dev, spirv, spirv_size);
        if (m == VK_NULL_HANDLE) {
            LOG_ERROR("VulkanNrdDenoiser: shader module failed for {}", label);
            return false;
        }
        VkPipelineShaderStageCreateInfo s{};
        s.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        s.module = m;
        s.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.layout = layout;
        cpci.stage  = s;
        VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &out);
        vkDestroyShaderModule(dev, m, nullptr);
        if (r != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkCreateComputePipelines({}) = {}", label, int(r));
            return false;
        }
        return true;
    };

    if (!build(shader_DenoiseTemporal_spirv_data, shader_DenoiseTemporal_spirv_size,
               pipe_layout_, temporal_pipe_, "DenoiseTemporal")) return false;
    if (!build(shader_DenoiseAtrous_spirv_data, shader_DenoiseAtrous_spirv_size,
               pipe_layout_, atrous_pipe_, "DenoiseAtrous")) return false;
    // Issue #119 -- remod kernel. Built unconditionally so we can flip
    // r_svgf_albedo_demod at runtime without a backend re-init.
    if (!build(shader_DenoiseRemod_spirv_data, shader_DenoiseRemod_spirv_size,
               remod_pipe_layout_, remod_pipe_, "DenoiseRemod")) return false;
    if (!build(shader_DenoiseFinalize_spirv_data, shader_DenoiseFinalize_spirv_size,
               finalize_pipe_layout_, finalize_pipe_, "DenoiseFinalize")) return false;
    return true;
}

bool VulkanNrdDenoiser::BuildDescriptorPool() {
    VkDevice dev = device_->RawDevice();

    // Mixed pool sizing for all three layouts:
    //   - kSetRing sets of (kPassImages storage images + kPassBuffers
    //     storage buffers) for temporal/atrous.
    //   - kFinalizeSetRing sets of (3 storage images + 1 storage
    //     buffer) for the finalize pass: hdr_in, swap_out, bloom_in
    //     (stars_in binding 4 retired with the EMA design).
    //   - kRemodSetRing sets of 3 storage images for the issue #119
    //     albedo remod pass.
    // Plus a small headroom on each type.
    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = static_cast<std::uint32_t>(
        kSetRing * kPassImages + kFinalizeSetRing * 3 + kRemodSetRing * 3 + 4);
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = static_cast<std::uint32_t>(
        kSetRing * kPassBuffers + kFinalizeSetRing * 1 + 4);

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = static_cast<std::uint32_t>(kSetRing + kFinalizeSetRing + kRemodSetRing);
    dpci.poolSizeCount = static_cast<std::uint32_t>(ps.size());
    dpci.pPoolSizes    = ps.data();
    // Sets stay alive for the pool's lifetime; we re-write them via
    // vkUpdateDescriptorSets each Encode rather than freeing/realloc.
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool_) != VK_SUCCESS) {
        LOG_ERROR("VulkanNrdDenoiser: vkCreateDescriptorPool failed");
        return false;
    }

    {
        std::array<VkDescriptorSetLayout, kSetRing> layouts;
        for (auto& l : layouts) l = dset_layout_;
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = kSetRing;
        dsai.pSetLayouts        = layouts.data();
        if (vkAllocateDescriptorSets(dev, &dsai, sets_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkAllocateDescriptorSets (temporal/atrous) failed");
            return false;
        }
    }
    {
        std::array<VkDescriptorSetLayout, kFinalizeSetRing> layouts;
        for (auto& l : layouts) l = finalize_dset_layout_;
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = kFinalizeSetRing;
        dsai.pSetLayouts        = layouts.data();
        if (vkAllocateDescriptorSets(dev, &dsai, finalize_sets_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkAllocateDescriptorSets (finalize) failed");
            return false;
        }
    }
    // Issue #119 -- remod descriptor set ring.
    {
        std::array<VkDescriptorSetLayout, kRemodSetRing> layouts;
        for (auto& l : layouts) l = remod_dset_layout_;
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = dpool_;
        dsai.descriptorSetCount = kRemodSetRing;
        dsai.pSetLayouts        = layouts.data();
        if (vkAllocateDescriptorSets(dev, &dsai, remod_sets_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: vkAllocateDescriptorSets (remod) failed");
            return false;
        }
    }
    return true;
}

bool VulkanNrdDenoiser::ResizeTextures(std::uint32_t w, std::uint32_t h) {
    if (w == 0 || h == 0) return false;
    if (cached_w_ == w && cached_h_ == h && history_a_id_ != 0 &&
        history_b_id_ != 0 && depth_history_a_id_ != 0 &&
        depth_history_b_id_ != 0 && normal_history_a_id_ != 0 &&
        normal_history_b_id_ != 0 && moments_history_a_buf_ != 0 &&
        moments_history_b_buf_ != 0 && atrous_a_id_ != 0 &&
        atrous_b_id_ != 0 && variance_a_buf_ != 0 && variance_b_buf_ != 0) {
        return true;  // already sized
    }
    DestroyTextures();

    auto mkTex = [&](const char* name, TextureFormat fmt) -> std::uint64_t {
        auto h_ = device_->CreateTexture({
            .width  = w, .height = h,
            .format = fmt,
            .usage  = TextureUsage::Storage,
            .debug_name = name,
        });
        return h_.id;
    };
    const std::size_t pix       = std::size_t(w) * std::size_t(h);
    const std::size_t mom_bytes = pix * sizeof(float) * 2;
    const std::size_t var_bytes = pix * sizeof(float);
    auto mkBuf = [&](const char* name, std::size_t bytes) -> std::uint64_t {
        auto h_ = device_->CreateBuffer({
            .size       = bytes,
            .usage      = BufferUsage::Storage,
            .debug_name = name,
        });
        return h_.id;
    };

    history_a_id_         = mkTex("denoise_history_a",         TextureFormat::RGBA16F);
    history_b_id_         = mkTex("denoise_history_b",         TextureFormat::RGBA16F);
    depth_history_a_id_   = mkTex("denoise_depth_history_a",   TextureFormat::R32F);
    depth_history_b_id_   = mkTex("denoise_depth_history_b",   TextureFormat::R32F);
    normal_history_a_id_  = mkTex("denoise_normal_history_a",  TextureFormat::RGBA16F);
    normal_history_b_id_  = mkTex("denoise_normal_history_b",  TextureFormat::RGBA16F);
    atrous_a_id_          = mkTex("denoise_atrous_a",          TextureFormat::RGBA16F);
    atrous_b_id_          = mkTex("denoise_atrous_b",          TextureFormat::RGBA16F);
    moments_history_a_buf_ = mkBuf("denoise_moments_history_a", mom_bytes);
    moments_history_b_buf_ = mkBuf("denoise_moments_history_b", mom_bytes);
    variance_a_buf_        = mkBuf("denoise_variance_a",        var_bytes);
    variance_b_buf_        = mkBuf("denoise_variance_b",        var_bytes);
    if (!history_a_id_ || !history_b_id_ ||
        !depth_history_a_id_ || !depth_history_b_id_ ||
        !normal_history_a_id_ || !normal_history_b_id_ ||
        !atrous_a_id_ || !atrous_b_id_ ||
        !moments_history_a_buf_ || !moments_history_b_buf_ ||
        !variance_a_buf_ || !variance_b_buf_) {
        LOG_ERROR("VulkanNrdDenoiser: scratch resource alloc failed at {}x{}", w, h);
        DestroyTextures();
        return false;
    }
    cached_w_ = w;
    cached_h_ = h;
    needs_history_clear_ = true;
    return true;
}

VkDescriptorSet VulkanNrdDenoiser::NextSet() {
    VkDescriptorSet s = sets_[next_set_];
    next_set_ = (next_set_ + 1) % kSetRing;
    return s;
}

void VulkanNrdDenoiser::RecordPass(VkCommandBuffer cb,
                                   VkPipeline      pipe,
                                   VkDescriptorSet set,
                                   const VkImageView (&views)[kPassImages],
                                   const VkBuffer    (&buffers)[kPassBuffers],
                                   const void*     push,
                                   std::size_t     push_size,
                                   std::uint32_t   gx,
                                   std::uint32_t   gy) {
    constexpr std::uint32_t kTotal = kPassImages + kPassBuffers;
    VkDescriptorImageInfo  img_infos[kPassImages]  {};
    VkDescriptorBufferInfo buf_infos[kPassBuffers] {};
    VkWriteDescriptorSet   writes[kTotal] {};
    // Image binding map: array indices 0..7 -> bindings 0..7 (the
    // engine-managed G-buffer block), array index 8 -> binding 12
    // (issue #119 albedo). The buffer block at bindings 8..11 sits
    // between, so the image array isn't a 1:1 binding index but the
    // mapping is fixed at compile time.
    for (std::uint32_t i = 0; i < kPassImages; ++i) {
        const std::uint32_t binding = (i < 8u) ? i : 12u;
        img_infos[i].imageView   = views[i];
        img_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = binding;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &img_infos[i];
    }
    for (std::uint32_t i = 0; i < kPassBuffers; ++i) {
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range  = VK_WHOLE_SIZE;
        const std::uint32_t w = kPassImages + i;
        writes[w].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[w].dstSet          = set;
        writes[w].dstBinding      = 8u + i;     // storage buffer bindings 8..11
        writes[w].descriptorCount = 1;
        writes[w].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[w].pBufferInfo     = &buf_infos[i];
    }
    vkUpdateDescriptorSets(device_->RawDevice(), kTotal, writes, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<std::uint32_t>(push_size), push);
    vkCmdDispatch(cb, gx, gy, 1);
}

void VulkanNrdDenoiser::DispatchRemod(VkCommandBuffer cb,
                                      VkImageView     demod_in_view,
                                      VkImageView     color_out_view,
                                      VkImageView     albedo_view,
                                      std::uint32_t   gx,
                                      std::uint32_t   gy) {
    // Inter-dispatch barrier: the SVGF chain's last shader write to
    // demod_in_view must be visible to this dispatch's shader read.
    // Mirrors the ComputeChainBarrier we issue between temporal /
    // atrous passes elsewhere.
    ComputeChainBarrier(cb);

    // Pick a remod descriptor set from the ring and update it.
    VkDescriptorSet set = remod_sets_[next_remod_set_];
    next_remod_set_ = (next_remod_set_ + 1) % kRemodSetRing;

    VkDescriptorImageInfo img_infos[3] {};
    img_infos[0].imageView   = demod_in_view;
    img_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_infos[1].imageView   = color_out_view;
    img_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_infos[2].imageView   = albedo_view;
    img_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet writes[3] {};
    // Remod layout bindings: 0 demod_in, 1 color_out, 12 albedo.
    const std::uint32_t bindings[3] = { 0u, 1u, 12u };
    for (int i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = bindings[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &img_infos[i];
    }
    vkUpdateDescriptorSets(device_->RawDevice(), 3, writes, 0, nullptr);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, remod_pipe_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            remod_pipe_layout_, 0, 1, &set, 0, nullptr);
    RemodPush p{};
    p.width  = cached_w_;
    p.height = cached_h_;
    vkCmdPushConstants(cb, remod_pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(p), &p);
    vkCmdDispatch(cb, gx, gy, 1);
    // Post-barrier so the finalize pass (or any downstream consumer)
    // sees the remod write.
    ComputeChainBarrier(cb);
}

void VulkanNrdDenoiser::Encode(VkCommandBuffer cb,
                               TextureHandle   color_in,
                               TextureHandle   depth_in,
                               TextureHandle   motion_in,
                               TextureHandle   normal_in,
                               TextureHandle   albedo_in,
                               TextureHandle   output,
                               TextureHandle   final_output,
                               BufferHandle    exposure_state,
                               TextureHandle   bloom_in,
                               float           bloom_intensity,
                               bool            reset_history,
                               bool            atrous_enabled,
                               std::uint32_t   atrous_passes,
                               bool            hdr_pipeline,
                               TextureHandle   stars_in,
                               bool            albedo_demod_enabled) {
    if (!ready_) return;
    if (cb == VK_NULL_HANDLE) return;
    if (cached_w_ == 0 || cached_h_ == 0) return;

    // Resolve all the engine-owned image views once. Null views are
    // treated as a hard failure -- the engine should have allocated
    // every G-buffer before reaching here.
    VkImageView v_color_in       = device_->LookupImageView(color_in);
    VkImageView v_depth          = device_->LookupImageView(depth_in);
    VkImageView v_motion         = device_->LookupImageView(motion_in);
    VkImageView v_normal         = device_->LookupImageView(normal_in);
    VkImageView v_output         = device_->LookupImageView(output);
    VkImageView v_hist_a         = device_->LookupImageView(TextureHandle{history_a_id_});
    VkImageView v_hist_b         = device_->LookupImageView(TextureHandle{history_b_id_});
    VkImageView v_depth_hist_a   = device_->LookupImageView(TextureHandle{depth_history_a_id_});
    VkImageView v_depth_hist_b   = device_->LookupImageView(TextureHandle{depth_history_b_id_});
    VkImageView v_normal_hist_a  = device_->LookupImageView(TextureHandle{normal_history_a_id_});
    VkImageView v_normal_hist_b  = device_->LookupImageView(TextureHandle{normal_history_b_id_});
    VkImageView v_atrous_a       = device_->LookupImageView(TextureHandle{atrous_a_id_});
    VkImageView v_atrous_b       = device_->LookupImageView(TextureHandle{atrous_b_id_});
    VkImageView v_dummy_c        = device_->LookupImageView(TextureHandle{dummy_color_id_});
    VkImageView v_dummy_m        = device_->LookupImageView(TextureHandle{dummy_motion_id_});
    VkBuffer    b_moments_a      = device_->LookupBuffer(BufferHandle{moments_history_a_buf_});
    VkBuffer    b_moments_b      = device_->LookupBuffer(BufferHandle{moments_history_b_buf_});
    VkBuffer    b_variance_a     = device_->LookupBuffer(BufferHandle{variance_a_buf_});
    VkBuffer    b_variance_b     = device_->LookupBuffer(BufferHandle{variance_b_buf_});
    VkBuffer    b_dummy_var      = device_->LookupBuffer(BufferHandle{dummy_variance_buf_});
    // Issue #119 -- albedo view. Falls through to dummy_color_id_
    // (RGBA16F 1x1) when the engine didn't allocate the albedo
    // G-buffer for this denoiser kind; in that case `demod_active`
    // drops to false so the shader skips every divide / multiply.
    VkImageView v_albedo = (albedo_in.id != 0)
                              ? device_->LookupImageView(albedo_in)
                              : v_dummy_c;
    if (v_color_in == VK_NULL_HANDLE || v_depth == VK_NULL_HANDLE ||
        v_motion == VK_NULL_HANDLE   || v_normal == VK_NULL_HANDLE ||
        v_output == VK_NULL_HANDLE   || v_hist_a == VK_NULL_HANDLE ||
        v_hist_b == VK_NULL_HANDLE   || v_depth_hist_a == VK_NULL_HANDLE ||
        v_depth_hist_b == VK_NULL_HANDLE || v_normal_hist_a == VK_NULL_HANDLE ||
        v_normal_hist_b == VK_NULL_HANDLE || v_atrous_a == VK_NULL_HANDLE ||
        v_atrous_b == VK_NULL_HANDLE ||
        v_dummy_c == VK_NULL_HANDLE  || v_dummy_m == VK_NULL_HANDLE ||
        v_albedo == VK_NULL_HANDLE ||
        b_moments_a == VK_NULL_HANDLE || b_moments_b == VK_NULL_HANDLE ||
        b_variance_a == VK_NULL_HANDLE || b_variance_b == VK_NULL_HANDLE ||
        b_dummy_var == VK_NULL_HANDLE) {
        LOG_WARN("VulkanNrdDenoiser::Encode: missing input resources");
        return;
    }
    // Effective demod state. Mirrors the Metal backend: demod requires
    // both the cvar/desc flag AND a real albedo binding.
    const bool demod_active = albedo_demod_enabled && (albedo_in.id != 0);

    const std::uint32_t gx = (cached_w_ + 7) / 8;
    const std::uint32_t gy = (cached_h_ + 7) / 8;

    // Path-tracer dispatch -> denoiser handoff barrier. The engine
    // only emits an explicit barrier when r_auto_exposure is on (the
    // autoexpose dispatch sits between path tracer + denoiser); when
    // auto-expose is off this is the only barrier between the path
    // tracer's writes and our reads of color/depth/motion/normal.
    ComputeChainBarrier(cb);

    // Frame-parity ping-pong: history read = side[parity], history
    // write = side[1 - parity]. Bumping parity at the end so consecutive
    // Encode calls swap roles.
    const bool parity_is_a = (frame_parity_ == 0);
    const VkImageView v_hist_read           = parity_is_a ? v_hist_a        : v_hist_b;
    const VkImageView v_hist_write          = parity_is_a ? v_hist_b        : v_hist_a;
    const VkImageView v_depth_hist_read     = parity_is_a ? v_depth_hist_a  : v_depth_hist_b;
    const VkImageView v_depth_hist_write    = parity_is_a ? v_depth_hist_b  : v_depth_hist_a;
    const VkImageView v_normal_hist_read    = parity_is_a ? v_normal_hist_a : v_normal_hist_b;
    const VkImageView v_normal_hist_write   = parity_is_a ? v_normal_hist_b : v_normal_hist_a;
    const VkBuffer    b_moments_hist_read   = parity_is_a ? b_moments_a     : b_moments_b;
    const VkBuffer    b_moments_hist_write  = parity_is_a ? b_moments_b     : b_moments_a;
    const std::uint64_t depth_hist_write_id  = parity_is_a ? depth_history_b_id_  : depth_history_a_id_;
    const std::uint64_t normal_hist_write_id = parity_is_a ? normal_history_b_id_ : normal_history_a_id_;

    // Force-reset on first Encode after Init / resize so the read of
    // freshly-allocated history memory doesn't drift undefined values
    // into the temporal blend.
    const bool effective_reset = reset_history || needs_history_clear_;
    needs_history_clear_ = false;

    // In SVGF-atrous mode the temporal output is intermediate scratch
    // (consumed by the single A-Trous pass) and v_hist_write receives
    // the filtered result -- Schied 2017's feedback loop. With variance
    // gating + tight sigma_color, converged regions are barely touched
    // by the filter so feedback is ~no-op there; disoccluded regions
    // benefit because next frame's temporal blend starts from a
    // spatially-denoised estimate, accelerating convergence. SVGF-basic
    // skips the chain so temporal writes straight to v_hist_write.
    const VkImageView v_temporal_color_out = atrous_enabled ? v_atrous_a : v_hist_write;

    // ---- Pass 1: temporal accumulate -----------------------------------
    DenoisePush p1{};
    p1.width         = cached_w_;
    p1.height        = cached_h_;
    p1.step_or_reset = effective_reset ? 1u : 0u;
    p1.a             = 0.10f;   // depth_tolerance (relative)
    p1.b             = 0.85f;   // normal_tolerance (cos angle)
    p1.c             = 0.10f;   // min_alpha (steady-state blend)
    // Issue #119 -- demod gate; temporal never remod's (its color_out
    // is the SVGF feedback target and must stay demodulated).
    p1.demod_enabled = demod_active ? 1u : 0u;
    p1.final_remod   = 0u;
    {
        const VkImageView views[kPassImages] = {
            /*0 color_in          */ v_color_in,
            /*1 color_history_in  */ v_hist_read,
            /*2 depth_tex         */ v_depth,
            /*3 motion_tex        */ v_motion,
            /*4 normal_tex        */ v_normal,
            /*5 color_out         */ v_temporal_color_out,
            /*6 depth_history_in  */ v_depth_hist_read,
            /*7 normal_history_in */ v_normal_hist_read,
            // Array index 8 -> binding 12 (albedo). RecordPass's
            // binding-table maps (i < 8) ? i : 12.
            /*12 albedo_tex       */ v_albedo,
        };
        const VkBuffer buffers[kPassBuffers] = {
            /*8  moments_history_in  */ b_moments_hist_read,
            /*9  moments_history_out */ b_moments_hist_write,
            /*10 variance_in (unused)*/ b_dummy_var,
            /*11 variance_out        */ b_variance_a,
        };
        RecordPass(cb, temporal_pipe_, NextSet(), views, buffers,
                   &p1, sizeof(p1), gx, gy);
    }

    // Keep depth/normal history ping-ponged in lockstep with color history
    // so frame N+1 can validate reprojection taps against frame N surfaces.
    VkImage dst_depth_hist = device_->LookupImage(TextureHandle{depth_hist_write_id});
    VkImage dst_normal_hist = device_->LookupImage(TextureHandle{normal_hist_write_id});
    VkImage src_depth = device_->LookupImage(depth_in);
    VkImage src_normal = device_->LookupImage(normal_in);
    if (src_depth == VK_NULL_HANDLE || src_normal == VK_NULL_HANDLE ||
        dst_depth_hist == VK_NULL_HANDLE || dst_normal_hist == VK_NULL_HANDLE) {
        LOG_WARN("VulkanNrdDenoiser::Encode: history copy image lookup miss");
        return;
    }
    StageBarrier(cb,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy copy_region{};
    // R32F + RGBA16F storage images use COLOR aspect in Vulkan.
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.layerCount = 1;
    copy_region.dstSubresource            = copy_region.srcSubresource;
    copy_region.extent.width  = cached_w_;
    copy_region.extent.height = cached_h_;
    copy_region.extent.depth  = 1;
    vkCmdCopyImage(cb,
                   src_depth, VK_IMAGE_LAYOUT_GENERAL,
                   dst_depth_hist, VK_IMAGE_LAYOUT_GENERAL,
                   1, &copy_region);
    vkCmdCopyImage(cb,
                   src_normal, VK_IMAGE_LAYOUT_GENERAL,
                   dst_normal_hist, VK_IMAGE_LAYOUT_GENERAL,
                   1, &copy_region);
    StageBarrier(cb,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    if (atrous_enabled) {
        // ---- A-Trous chain ---------------------------------------------
        // Caller picks pass count (1..5) via r_svgf_atrous_passes:
        //   1 = step=1                       (5x5  effective)
        //   2 = step=1, 2                    (9x9)
        //   3 = step=1, 2, 4                 (17x17)
        //   4 = step=1, 2, 4, 8              (33x33)
        //   5 = step=1, 2, 4, 8, 16          (65x65, canonical SVGF)
        // Pass 1 always writes to v_hist_write so it doubles as next
        // frame's reprojection source (Schied feedback). For n_passes
        // == 1 we vkCmdCopyImage hist_write -> output below; for
        // multi-pass the LAST pass writes directly into v_output and
        // intermediates ping-pong v_atrous_a / v_atrous_b.
        std::uint32_t n_passes = atrous_passes;
        if (n_passes < 1u) n_passes = 1u;
        if (n_passes > 5u) n_passes = 5u;

        auto atrous_pass = [&](VkImageView color_src, VkImageView color_dst,
                               VkBuffer var_src, VkBuffer var_dst,
                               std::uint32_t step,
                               bool          is_final_pass) {
            DenoisePush p{};
            p.width         = cached_w_;
            p.height        = cached_h_;
            p.step_or_reset = step;
            p.a             = 1.0f;     // sigma_depth (gradient-scaled)
            p.b             = 128.0f;   // sigma_normal (sharper than the old 64.0)
            p.c             = 1.0f;     // sigma_color (variance-scaled). The SVGF
                                        // paper default of 4.0 reads as visibly
                                        // smooth on diffuse surfaces -- 1.0 keeps
                                        // the noise rejection without averaging
                                        // through 4-sigma-wide luminance bands.
            p.demod_enabled = demod_active ? 1u : 0u;
            // Final-pass remod fires only when this pass writes
            // directly to v_output (n_passes >= 2's last pass);
            // n_passes == 1 routes through the remod kernel below so
            // its single pass leaves the signal demodulated.
            p.final_remod   = (demod_active && is_final_pass) ? 1u : 0u;
            const VkImageView views[kPassImages] = {
                /*0 color_in            */ color_src,
                /*1 color_history_in    */ v_dummy_c,
                /*2 depth_tex           */ v_depth,
                /*3 motion_tex          */ v_dummy_m,
                /*4 normal_tex          */ v_normal,
                /*5 color_out           */ color_dst,
                /*6 depth_history_in    */ v_depth_hist_write,
                /*7 normal_history_in   */ v_normal_hist_write,
                /*12 albedo_tex         */ v_albedo,
            };
            // Moments slots 8/9 are declared but unused by atrous; bind
            // the parity-side moments buffer (any valid storage buffer
            // works -- the shader never reads or writes through them).
            const VkBuffer buffers[kPassBuffers] = {
                /*8  moments_in_unused  */ b_moments_hist_write,
                /*9  moments_out_unused */ b_moments_hist_write,
                /*10 variance_in        */ var_src,
                /*11 variance_out       */ var_dst,
            };
            RecordPass(cb, atrous_pipe_, NextSet(), views, buffers,
                       &p, sizeof(p), gx, gy);
            ComputeChainBarrier(cb);
        };
        // Pass schedule. Color ping-pong:
        //   pass 1: v_atrous_a -> v_hist_write     (Schied feedback)
        //   pass i (1<i<N): toggles v_atrous_b / v_atrous_a
        //   pass N (last, N>=2):                  -> v_output
        // Variance ping-pong is independent: starts variance_a -> b on
        // pass 1, then alternates each pass.
        VkImageView color_src = v_atrous_a;
        VkBuffer    var_src   = b_variance_a;
        VkBuffer    var_dst   = b_variance_b;
        for (std::uint32_t i = 1; i <= n_passes; ++i) {
            const std::uint32_t step = 1u << (i - 1u);
            VkImageView color_dst;
            // Identify the pass that writes directly to v_output so it
            // can do an inline remod (multi-pass demod path). For
            // n_passes == 1 the pass writes v_hist_write (feedback
            // target) and the remod compute pass below handles the
            // multiply-back.
            bool writes_to_output = false;
            if (i == 1u) {
                color_dst = v_hist_write;
            } else if (i == n_passes) {
                color_dst = v_output;
                writes_to_output = true;
            } else {
                // After pass 1 v_atrous_a is no longer read, so it's
                // free to re-enter the ping-pong from pass 3 onward.
                color_dst = (i % 2u == 0u) ? v_atrous_b : v_atrous_a;
            }
            atrous_pass(color_src, color_dst, var_src, var_dst, step,
                        writes_to_output);
            color_src = color_dst;
            std::swap(var_src, var_dst);
        }

        if (n_passes == 1u) {
            // One-pass path: pass 1's output sits in v_hist_write (the
            // feedback target). Publish to v_output:
            //   - demod off: vkCmdCopyImage (RGBA16F GENERAL -> GENERAL).
            //   - demod on:  dispatch the remod compute kernel so the
            //     multiply-back happens in `output` while v_hist_write
            //     stays in demodulated lighting space for next frame.
            const std::uint64_t one_pass_src_id =
                parity_is_a ? history_b_id_ : history_a_id_;
            VkImage src_img = device_->LookupImage(TextureHandle{one_pass_src_id});
            VkImage dst_img = device_->LookupImage(output);
            if (src_img != VK_NULL_HANDLE && dst_img != VK_NULL_HANDLE) {
                if (!demod_active) {
                    StageBarrier(cb,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                    VkImageCopy region{};
                    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    region.srcSubresource.layerCount = 1;
                    region.dstSubresource            = region.srcSubresource;
                    region.extent.width  = cached_w_;
                    region.extent.height = cached_h_;
                    region.extent.depth  = 1;
                    vkCmdCopyImage(cb,
                                   src_img, VK_IMAGE_LAYOUT_GENERAL,
                                   dst_img, VK_IMAGE_LAYOUT_GENERAL,
                                   1, &region);
                    StageBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,        VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  VK_ACCESS_SHADER_READ_BIT);
                } else {
                    DispatchRemod(cb, v_hist_write, v_output, v_albedo, gx, gy);
                }
            } else {
                LOG_WARN("VulkanNrdDenoiser::Encode: atrous publish blit lookup miss");
            }
        }
    } else {
        // ---- svgf_basic: temporal-only -> copy history into output ----
        // Skip the spatial filter entirely. The temporal pass already
        // wrote v_hist_write; copy that exact image to the caller's
        // output texture so the bloom + tonemap chain downstream sees
        // the temporally-stable result. Both images are RGBA16F at
        // (cached_w_, cached_h_) and stay in GENERAL layout, so a
        // memory-barrier-only handoff plus a single vkCmdCopyImage is
        // all we need.
        const std::uint64_t hist_write_id =
            (frame_parity_ == 0) ? history_b_id_ : history_a_id_;
        VkImage src_img = device_->LookupImage(TextureHandle{hist_write_id});
        VkImage dst_img = device_->LookupImage(output);
        if (src_img == VK_NULL_HANDLE || dst_img == VK_NULL_HANDLE) {
            LOG_WARN("VulkanNrdDenoiser::Encode: basic-mode VkImage lookup miss "
                     "(src={} dst={})", reinterpret_cast<void*>(src_img),
                     reinterpret_cast<void*>(dst_img));
            return;
        }
        if (!demod_active) {
            // Compute -> transfer handoff. NB: with the history-copy
            // block above (which already issued compute_write -> transfer
            // for the depth/normal hist copies), this barrier is
            // technically redundant for the current flow -- no compute
            // op writes color_hist between the temporal pass and here.
            // Kept defensively so a future refactor that inserts new
            // compute writes between those points doesn't silently
            // regress visibility. dstAccess covers both transfer read
            // of color_hist and transfer write of `output` (theoretical
            // WAW against any future upstream shader write to output).
            StageBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageCopy region{};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.dstSubresource            = region.srcSubresource;
            region.extent.width  = cached_w_;
            region.extent.height = cached_h_;
            region.extent.depth  = 1;
            vkCmdCopyImage(cb,
                           src_img, VK_IMAGE_LAYOUT_GENERAL,
                           dst_img, VK_IMAGE_LAYOUT_GENERAL,
                           1, &region);
            StageBarrier(cb,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,        VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  VK_ACCESS_SHADER_READ_BIT);
        } else {
            // Issue #119 -- svgf_basic with demod on: dispatch the remod
            // compute kernel to multiply v_hist_write (demodulated
            // lighting) by albedo into `output`. Keeps v_hist_write in
            // demod space for next frame's feedback.
            DispatchRemod(cb, v_hist_write, v_output, v_albedo, gx, gy);
        }
    }

    // ---- Finalize: HDR -> ACES + sRGB -> swapchain --------------------
    // Both atrous and basic paths leave the linear-HDR result in
    // `output` (and the basic path also issued a transfer->compute
    // barrier already, which double-covers as our finalize-read
    // barrier; in the atrous path the last ComputeChainBarrier covers
    // it). We dispatch DenoiseFinalize only when the caller supplied
    // a final_output target and an exposure_state buffer -- both
    // required for the kernel's bindings. Skipping the dispatch when
    // either is missing keeps the code defensive against a partially-
    // wired caller (the engine fills both today).
    if (final_output.id != 0 && exposure_state.id != 0) {
        VkImageView v_final = device_->LookupImageView(final_output);
        VkBuffer    b_exp   = device_->LookupBuffer(exposure_state);
        // Bloom is optional. If the caller didn't pass a bloom mip
        // (id=0) OR the lookup misses OR bloom_intensity is zero, we
        // still need to bind *something* at binding 3 because the
        // descriptor set layout declares it. Fall back to v_output
        // (any valid RGBA16F storage image) and force the effective
        // intensity to zero in that case so the shader's
        // `if (bloom_intensity > 0.0)` short-circuits the sample.
        // Tracking `bloom_real` instead of keying intensity on
        // bloom_in.id catches the corner case where the engine passes
        // a non-zero handle but LookupImageView returns NULL_HANDLE
        // (mid-resize, stale handle, etc.) -- without the gate the
        // shader would sample v_output as its own bloom layer and
        // produce a feedback-style artifact.
        bool bloom_real = (bloom_in.id != 0);
        VkImageView v_bloom = VK_NULL_HANDLE;
        if (bloom_real) v_bloom = device_->LookupImageView(bloom_in);
        if (v_bloom == VK_NULL_HANDLE) {
            v_bloom    = v_output;
            bloom_real = false;
        }
        // Stars binding 4 retired with the stateless StarsComposite
        // rewrite (#46). The Vulkan denoiser path doesn't run stars in
        // this PR -- composite is a Metal-side dispatch.
        (void)stars_in;
        if (v_final != VK_NULL_HANDLE && b_exp != VK_NULL_HANDLE) {
            VkDescriptorSet fset = finalize_sets_[next_finalize_set_];
            next_finalize_set_ = (next_finalize_set_ + 1) % kFinalizeSetRing;

            VkDescriptorImageInfo  ii[3] {};
            VkDescriptorBufferInfo bi    {};
            VkWriteDescriptorSet   w[4]  {};

            ii[0].imageView   = v_output;
            ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            ii[1].imageView   = v_final;
            ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            ii[2].imageView   = v_bloom;
            ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            bi.buffer = b_exp;
            bi.offset = 0;
            bi.range  = VK_WHOLE_SIZE;

            for (int i = 0; i < 4; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = fset;
                w[i].descriptorCount = 1;
            }
            w[0].dstBinding     = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo     = &ii[0];
            w[1].dstBinding     = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = &ii[1];
            w[2].dstBinding     = 2;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo    = &bi;
            w[3].dstBinding     = 3;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[3].pImageInfo     = &ii[2];

            vkUpdateDescriptorSets(device_->RawDevice(), 4, w, 0, nullptr);

            // Atrous path's last pass left the chain barrier in place,
            // so the read of `output` is already covered. Basic path
            // emitted a transfer->compute barrier on the same image.
            // Both reach this dispatch with the source image visible
            // for shader read.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, finalize_pipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    finalize_pipe_layout_, 0, 1, &fset, 0, nullptr);

            struct FinalizePush {
                std::uint32_t width;
                std::uint32_t height;
                std::uint32_t hdr_pipeline;     // 1 = ACES + sRGB; 0 = sRGB OETF only
                float         bloom_intensity;  // 0 = skip bloom add
                // r_tonemap_op enum (0 aces / 1 agx / 2 khronos /
                // 3 reinhard / 4 linear). Repurposed from the retired
                // `_reserved_stars` lane -- same offset, layout
                // unchanged -- so the denoiser-on swapchain honours
                // the operator instead of hard-coding ACES.
                std::uint32_t tonemap_op;
                std::uint32_t _pad0;
                std::uint32_t _pad1;
                std::uint32_t _pad2;
            } fp{};
            static_assert(sizeof(FinalizePush) == 32,
                          "FinalizePush layout mismatch with DenoiseFinalize.slang");
            static_assert(sizeof(FinalizePush) % 16 == 0,
                          "FinalizePush must be 16-byte aligned (cbuffer rule)");
            fp.width           = cached_w_;
            fp.height          = cached_h_;
            fp.hdr_pipeline    = hdr_pipeline ? 1u : 0u;
            // Key intensity on the resolved-bloom flag, not on
            // bloom_in.id, so the corner case where the engine passes
            // a non-zero handle but LookupImageView fails also
            // collapses to a no-op (otherwise the v_output fallback
            // above would be sampled as bloom -- HDR feedback artifact).
            fp.bloom_intensity = bloom_real ? bloom_intensity : 0.0f;
            fp.tonemap_op      = tonemap_op_;
            vkCmdPushConstants(cb, finalize_pipe_layout_,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(fp), &fp);
            vkCmdDispatch(cb, gx, gy, 1);

            // Compute->compute barrier so any subsequent dispatch
            // (perfoverlay etc.) reading the swapchain image sees the
            // tonemapped result.
            ComputeChainBarrier(cb);
        } else {
            LOG_WARN("VulkanNrdDenoiser::Encode: finalize lookup miss "
                     "(final_view={} exp_buf={})",
                     reinterpret_cast<void*>(v_final),
                     reinterpret_cast<void*>(b_exp));
        }
    }

    // Advance parity unconditionally. Reset path: history_write now
    // holds this frame's noisy stamp, and next frame's temporal pass
    // reads it as the prior history. Non-reset path: standard
    // ping-pong roll. Both cases want a single XOR.
    frame_parity_ ^= 1u;

    PT_DIAG_TIER3("denoiser",
                  "NRD::Encode: {}x{} parity={} reset={} atrous={} hdr_pipeline={}",
                  cached_w_, cached_h_,
                  static_cast<int>(frame_parity_),
                  reset_history ? 1 : 0,
                  atrous_enabled ? 1 : 0,
                  hdr_pipeline ? 1 : 0);
}

// ---- EncodeFinalizeOnly (standalone tonemap + sRGB compute pass) ---------
//
// Mirrors the finalize dispatch at the tail of Encode() above, but
// works against caller-supplied views/buffers instead of pulling them
// from the SVGF history textures. Used by the OptiX denoiser path so
// it gets the same ACES + sRGB output as SVGF without duplicating the
// pipeline, layout, descriptor pool, or kernel.
//
// Layout assumption: both images already in GENERAL. Caller barriers
// before/after.
void VulkanNrdDenoiser::EncodeFinalizeOnly(VkCommandBuffer cb,
                                           VkImageView    color_in_view,
                                           VkImageView    final_output_view,
                                           VkBuffer       exposure_state_buf,
                                           VkImageView    bloom_in_view,
                                           float          bloom_intensity,
                                           std::uint32_t  width,
                                           std::uint32_t  height,
                                           bool           hdr_pipeline,
                                           VkImageView    stars_in_view) {
    if (!ready_)                                  return;
    if (cb == VK_NULL_HANDLE)                     return;
    if (color_in_view     == VK_NULL_HANDLE)      return;
    if (final_output_view == VK_NULL_HANDLE)      return;
    if (exposure_state_buf == VK_NULL_HANDLE)     return;
    if (width == 0 || height == 0)                return;
    // Bloom slot must be valid (descriptor layout requires it). If the
    // caller didn't pass one, fall back to color_in_view -- the shader
    // gates the read on bloom_intensity > 0 so the slot is never
    // actually sampled.
    VkImageView v_bloom = (bloom_in_view != VK_NULL_HANDLE)
                            ? bloom_in_view
                            : color_in_view;
    const float effective_bloom = (bloom_in_view != VK_NULL_HANDLE)
                                    ? bloom_intensity : 0.0f;
    // Stars binding 4 retired with the stateless StarsComposite
    // rewrite (#46). Vulkan denoiser path has no stars compositor in
    // this PR -- the function signature still carries stars_in_view
    // so VulkanDevice's call sites don't need a signature ripple in
    // this PR, but the parameter is currently unused.
    (void)stars_in_view;

    VkDescriptorSet fset = finalize_sets_[next_finalize_set_];
    next_finalize_set_   = (next_finalize_set_ + 1) % kFinalizeSetRing;

    VkDescriptorImageInfo  ii[3] {};
    VkDescriptorBufferInfo bi    {};
    VkWriteDescriptorSet   w[4]  {};

    ii[0].imageView   = color_in_view;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[1].imageView   = final_output_view;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView   = v_bloom;
    ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    bi.buffer = exposure_state_buf;
    bi.offset = 0;
    bi.range  = VK_WHOLE_SIZE;

    for (int i = 0; i < 4; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = fset;
        w[i].descriptorCount = 1;
    }
    w[0].dstBinding     = 0;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo     = &ii[0];
    w[1].dstBinding     = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo     = &ii[1];
    w[2].dstBinding     = 2;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo    = &bi;
    w[3].dstBinding     = 3;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[3].pImageInfo     = &ii[2];
    vkUpdateDescriptorSets(device_->RawDevice(), 4, w, 0, nullptr);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, finalize_pipe_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            finalize_pipe_layout_, 0, 1, &fset, 0, nullptr);

    struct FinalizePush {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t hdr_pipeline;     // 1 = tonemap + sRGB; 0 = sRGB OETF only
        float         bloom_intensity;  // 0 = skip bloom add
        // r_tonemap_op enum, repurposed from the retired
        // `_reserved_stars` lane (same offset, layout unchanged).
        std::uint32_t tonemap_op;
        std::uint32_t _pad0;
        std::uint32_t _pad1;
        std::uint32_t _pad2;
    } fp{};
    static_assert(sizeof(FinalizePush) == 32,
                  "FinalizePush layout mismatch with DenoiseFinalize.slang");
    static_assert(sizeof(FinalizePush) % 16 == 0,
                  "FinalizePush must be 16-byte aligned (cbuffer rule)");
    fp.width           = width;
    fp.height          = height;
    fp.hdr_pipeline    = hdr_pipeline ? 1u : 0u;
    fp.bloom_intensity = effective_bloom;
    fp.tonemap_op      = tonemap_op_;
    vkCmdPushConstants(cb, finalize_pipe_layout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(fp), &fp);

    const std::uint32_t gx = (width  + 7) / 8;
    const std::uint32_t gy = (height + 7) / 8;
    vkCmdDispatch(cb, gx, gy, 1);
}

}  // namespace pt::rhi::vk

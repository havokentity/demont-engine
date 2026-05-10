// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "VulkanDenoiser.h"
#include "VulkanDevice.h"

#include "../core/Log.h"
#include "../rhi/Resources.h"

#include <array>
#include <cstring>

extern "C" {
extern const unsigned char shader_DenoiseTemporal_spirv_data[];
extern const unsigned long shader_DenoiseTemporal_spirv_size;
extern const unsigned char shader_DenoiseAtrous_spirv_data[];
extern const unsigned long shader_DenoiseAtrous_spirv_size;
extern const unsigned char shader_DenoiseFinalize_spirv_data[];
extern const unsigned long shader_DenoiseFinalize_spirv_size;
}

namespace pt::rhi::vk {

namespace {

VkShaderModule MakeModule(VkDevice dev,
                          const unsigned char* bytes, std::size_t n) {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = n;
    ci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes);
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

// Push struct shared by both denoise kernels. 32 bytes total. Field
// names are kernel-specific (see DenoiseTemporal.slang / DenoiseAtrous
// .slang for what each lane means at each call site).
struct DenoisePush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t step_or_reset;     // temporal: reset_history; atrous: step_size
    std::uint32_t pad0;
    float         a;                  // temporal: depth_tolerance; atrous: sigma_depth
    float         b;                  // temporal: normal_tolerance; atrous: sigma_normal
    float         c;                  // temporal: min_alpha; atrous: sigma_color
    float         pad1;
};
static_assert(sizeof(DenoisePush) == 32, "DenoisePush layout");

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

    DestroyTextures();
    if (dummy_color_id_  != 0) device_->DestroyTexture(TextureHandle{dummy_color_id_});
    if (dummy_motion_id_ != 0) device_->DestroyTexture(TextureHandle{dummy_motion_id_});
    dummy_color_id_  = 0;
    dummy_motion_id_ = 0;

    if (temporal_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, temporal_pipe_, nullptr);
        temporal_pipe_ = VK_NULL_HANDLE;
    }
    if (atrous_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, atrous_pipe_, nullptr);
        atrous_pipe_ = VK_NULL_HANDLE;
    }
    if (finalize_pipe_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, finalize_pipe_, nullptr);
        finalize_pipe_ = VK_NULL_HANDLE;
    }
    if (pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, pipe_layout_, nullptr);
        pipe_layout_ = VK_NULL_HANDLE;
    }
    if (finalize_pipe_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, finalize_pipe_layout_, nullptr);
        finalize_pipe_layout_ = VK_NULL_HANDLE;
    }
    if (dset_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, dset_layout_, nullptr);
        dset_layout_ = VK_NULL_HANDLE;
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
        vkDestroyDescriptorPool(dev, dpool_, nullptr);
        dpool_ = VK_NULL_HANDLE;
    }
    ready_ = false;
}

void VulkanNrdDenoiser::DestroyTextures() {
    if (device_ == nullptr) return;
    if (history_a_id_ != 0) device_->DestroyTexture(TextureHandle{history_a_id_});
    if (history_b_id_ != 0) device_->DestroyTexture(TextureHandle{history_b_id_});
    if (atrous_a_id_  != 0) device_->DestroyTexture(TextureHandle{atrous_a_id_});
    if (atrous_b_id_  != 0) device_->DestroyTexture(TextureHandle{atrous_b_id_});
    history_a_id_ = history_b_id_ = atrous_a_id_ = atrous_b_id_ = 0;
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
    dummy_color_id_  = color_h.id;
    dummy_motion_id_ = mot_h.id;
    if (dummy_color_id_ == 0 || dummy_motion_id_ == 0) {
        LOG_ERROR("VulkanNrdDenoiser: dummy texture allocation failed");
        DestroyAll();
        return false;
    }

    ready_ = true;
    return true;
}

bool VulkanNrdDenoiser::BuildLayout() {
    VkDevice dev = device_->RawDevice();

    // ---- temporal/atrous: 6 storage images --------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 6> b{};
        for (std::uint32_t i = 0; i < b.size(); ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
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

    // ---- finalize: 2 storage images + 1 storage buffer --------------
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<std::uint32_t>(b.size());
        dslci.pBindings    = b.data();
        if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &finalize_dset_layout_) != VK_SUCCESS) {
            LOG_ERROR("VulkanNrdDenoiser: finalize descriptor set layout failed");
            return false;
        }
        // Finalize push: width, height + 8 bytes pad to keep the
        // struct 16-byte aligned (Slang/SPIR-V cbuffer convention).
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = 16;
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
    if (!build(shader_DenoiseFinalize_spirv_data, shader_DenoiseFinalize_spirv_size,
               finalize_pipe_layout_, finalize_pipe_, "DenoiseFinalize")) return false;
    return true;
}

bool VulkanNrdDenoiser::BuildDescriptorPool() {
    VkDevice dev = device_->RawDevice();

    // Mixed pool sizing for both layouts:
    //   - kSetRing sets of (6 storage images) for temporal/atrous.
    //   - kFinalizeSetRing sets of (2 storage images + 1 storage
    //     buffer) for the finalize pass.
    // Plus a small headroom on each type.
    std::array<VkDescriptorPoolSize, 2> ps{};
    ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = static_cast<std::uint32_t>(
        kSetRing * 6 + kFinalizeSetRing * 2 + 4);
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = static_cast<std::uint32_t>(kFinalizeSetRing * 1 + 2);

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = static_cast<std::uint32_t>(kSetRing + kFinalizeSetRing);
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
    return true;
}

bool VulkanNrdDenoiser::ResizeTextures(std::uint32_t w, std::uint32_t h) {
    if (w == 0 || h == 0) return false;
    if (cached_w_ == w && cached_h_ == h && history_a_id_ != 0 &&
        history_b_id_ != 0 && atrous_a_id_ != 0 && atrous_b_id_ != 0) {
        return true;  // already sized
    }
    DestroyTextures();

    auto mk = [&](const char* name) -> std::uint64_t {
        auto h_ = device_->CreateTexture({
            .width  = w, .height = h,
            .format = TextureFormat::RGBA16F,
            .usage  = TextureUsage::Storage,
            .debug_name = name,
        });
        return h_.id;
    };

    history_a_id_ = mk("denoise_history_a");
    history_b_id_ = mk("denoise_history_b");
    atrous_a_id_  = mk("denoise_atrous_a");
    atrous_b_id_  = mk("denoise_atrous_b");
    if (!history_a_id_ || !history_b_id_ || !atrous_a_id_ || !atrous_b_id_) {
        LOG_ERROR("VulkanNrdDenoiser: scratch texture alloc failed at {}x{}", w, h);
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
                                   VkImageView     color_in,
                                   VkImageView     color_history_in,
                                   VkImageView     depth_in,
                                   VkImageView     motion_in,
                                   VkImageView     normal_in,
                                   VkImageView     color_out,
                                   const void*     push,
                                   std::size_t     push_size,
                                   std::uint32_t   gx,
                                   std::uint32_t   gy) {
    VkImageView views[6] = { color_in, color_history_in, depth_in, motion_in, normal_in, color_out };
    VkDescriptorImageInfo infos[6] {};
    VkWriteDescriptorSet  writes[6] {};
    for (std::uint32_t i = 0; i < 6; ++i) {
        infos[i].imageView   = views[i];
        infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &infos[i];
    }
    vkUpdateDescriptorSets(device_->RawDevice(), 6, writes, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<std::uint32_t>(push_size), push);
    vkCmdDispatch(cb, gx, gy, 1);
}

void VulkanNrdDenoiser::Encode(VkCommandBuffer cb,
                               TextureHandle   color_in,
                               TextureHandle   depth_in,
                               TextureHandle   motion_in,
                               TextureHandle   normal_in,
                               TextureHandle   output,
                               TextureHandle   final_output,
                               BufferHandle    exposure_state,
                               bool            reset_history,
                               bool            atrous_enabled) {
    if (!ready_) return;
    if (cb == VK_NULL_HANDLE) return;
    if (cached_w_ == 0 || cached_h_ == 0) return;

    // Resolve all the engine-owned image views once. Null views are
    // treated as a hard failure -- the engine should have allocated
    // every G-buffer before reaching here.
    VkImageView v_color_in  = device_->LookupImageView(color_in);
    VkImageView v_depth     = device_->LookupImageView(depth_in);
    VkImageView v_motion    = device_->LookupImageView(motion_in);
    VkImageView v_normal    = device_->LookupImageView(normal_in);
    VkImageView v_output    = device_->LookupImageView(output);
    VkImageView v_hist_a    = device_->LookupImageView(TextureHandle{history_a_id_});
    VkImageView v_hist_b    = device_->LookupImageView(TextureHandle{history_b_id_});
    VkImageView v_atrous_a  = device_->LookupImageView(TextureHandle{atrous_a_id_});
    VkImageView v_atrous_b  = device_->LookupImageView(TextureHandle{atrous_b_id_});
    VkImageView v_dummy_c   = device_->LookupImageView(TextureHandle{dummy_color_id_});
    VkImageView v_dummy_m   = device_->LookupImageView(TextureHandle{dummy_motion_id_});
    if (v_color_in == VK_NULL_HANDLE || v_depth == VK_NULL_HANDLE ||
        v_motion == VK_NULL_HANDLE   || v_normal == VK_NULL_HANDLE ||
        v_output == VK_NULL_HANDLE   || v_hist_a == VK_NULL_HANDLE ||
        v_hist_b == VK_NULL_HANDLE   || v_atrous_a == VK_NULL_HANDLE ||
        v_atrous_b == VK_NULL_HANDLE) {
        LOG_WARN("VulkanNrdDenoiser::Encode: missing input image views");
        return;
    }

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
    const VkImageView v_hist_read  = (frame_parity_ == 0) ? v_hist_a : v_hist_b;
    const VkImageView v_hist_write = (frame_parity_ == 0) ? v_hist_b : v_hist_a;

    // Force-reset on first Encode after Init / resize so the read of
    // freshly-allocated history memory doesn't drift undefined values
    // into the temporal blend.
    const bool effective_reset = reset_history || needs_history_clear_;
    needs_history_clear_ = false;

    // ---- Pass 1: temporal accumulate -----------------------------------
    DenoisePush p1{};
    p1.width         = cached_w_;
    p1.height        = cached_h_;
    p1.step_or_reset = effective_reset ? 1u : 0u;
    p1.a             = 0.10f;   // depth_tolerance (relative)
    p1.b             = 0.85f;   // normal_tolerance (cos angle)
    p1.c             = 0.10f;   // min_alpha (steady-state blend)
    RecordPass(cb, temporal_pipe_, NextSet(),
               v_color_in, v_hist_read, v_depth, v_motion, v_normal,
               v_hist_write,
               &p1, sizeof(p1), gx, gy);

    if (atrous_enabled) {
        ComputeChainBarrier(cb);

        // ---- Passes 2..4: a-trous chain at step sizes 1, 2, 4 ----------
        // The first pass reads the just-written history (v_hist_write)
        // and outputs to atrous_a; subsequent passes ping-pong between
        // the two atrous textures. The last pass writes directly to
        // `output`, leaving v_hist_write unchanged so next frame's
        // temporal still blends from the pre-spatial-filter accumulation.
        auto atrous_pass = [&](VkImageView in, VkImageView out, std::uint32_t step) {
            DenoisePush p{};
            p.width         = cached_w_;
            p.height        = cached_h_;
            p.step_or_reset = step;
            p.a             = 1.0f;    // sigma_depth (relative)
            p.b             = 64.0f;   // sigma_normal (pow exponent; bigger = sharper edges)
            p.c             = 4.0f;    // sigma_color (luminance Gaussian sigma)
            RecordPass(cb, atrous_pipe_, NextSet(),
                       in, v_dummy_c, v_depth, v_dummy_m, v_normal, out,
                       &p, sizeof(p), gx, gy);
            ComputeChainBarrier(cb);
        };
        atrous_pass(v_hist_write, v_atrous_a, 1u);
        atrous_pass(v_atrous_a,   v_atrous_b, 2u);
        atrous_pass(v_atrous_b,   v_output,   4u);
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
        StageBarrier(cb,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,       VK_ACCESS_TRANSFER_READ_BIT);
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
        if (v_final != VK_NULL_HANDLE && b_exp != VK_NULL_HANDLE) {
            VkDescriptorSet fset = finalize_sets_[next_finalize_set_];
            next_finalize_set_ = (next_finalize_set_ + 1) % kFinalizeSetRing;

            VkDescriptorImageInfo  ii[2] {};
            VkDescriptorBufferInfo bi    {};
            VkWriteDescriptorSet   w[3]  {};

            ii[0].imageView   = v_output;
            ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            ii[1].imageView   = v_final;
            ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            bi.buffer = b_exp;
            bi.offset = 0;
            bi.range  = VK_WHOLE_SIZE;

            for (int i = 0; i < 3; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = fset;
                w[i].dstBinding      = i;
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[0].pImageInfo     = &ii[0];
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = &ii[1];
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo    = &bi;

            vkUpdateDescriptorSets(device_->RawDevice(), 3, w, 0, nullptr);

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
                std::uint32_t pad0;
                std::uint32_t pad1;
            } fp{};
            fp.width  = cached_w_;
            fp.height = cached_h_;
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

    if (!effective_reset) {
        frame_parity_ ^= 1u;
    } else {
        // After a reset, history_write holds this frame's noisy stamp.
        // Advance parity so the next frame's temporal pass reads it.
        frame_parity_ ^= 1u;
    }
}

}  // namespace pt::rhi::vk

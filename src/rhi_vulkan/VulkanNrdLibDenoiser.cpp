// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// VulkanNrdLibDenoiser implementation -- scaffolding stage (issue #50).
//
// What this file does today (v0.3.30):
//   - On Init(): calls nrd::CreateInstance with SIGMA_SHADOW registered,
//     then queries GetLibraryDesc() + GetInstanceDesc() and logs a
//     one-shot summary (NRD version, pipeline count, internal-pool sizes).
//     This is enough to confirm the static-link end-to-end -- if NRD's
//     symbols aren't visible we trip an undefined-reference at link time
//     (caught by CMake's `target_link_libraries(... PRIVATE NRD)`), and
//     if the ABI is bad the CreateInstance call returns a non-SUCCESS
//     result with a recognisable error log.
//
// What it doesn't do yet:
//   - The actual per-frame Dispatch translation. nrd::GetComputeDispatches
//     hands back an array of DispatchDesc{ pipelineIndex, resources[],
//     gridX, gridY, gridZ, constantBufferData } per frame. Translating
//     those into VkPipeline binds + VkDescriptorSet writes + vkCmdDispatch
//     is the bulk of the integration -- punted to the follow-up PR.
//   - Allocating the NRD-internal texture pool (permanentPool + transientPool
//     from the InstanceDesc). For SIGMA_SHADOW today this is ~10 texture
//     slots, all linear-storage-format -- straightforward but boilerplate.
//   - The shadow-visibility G-buffer write site in PathTrace.slang. The
//     existing G-buffer set ends at vk::binding(17) (albedo, OptiX-AOV
//     only); the SIGMA path will add vk::binding(18) shadow_visibility.
//     Gated by PT_TARGET_SPIRV like the existing normal G-buffer.
//
// Until the dispatch walker lands, Encode() does a passthrough vkCmdCopy
// Image -- noisy_color_in -> output. The engine's bloom + tonemap chain
// downstream sees a valid image, just with no denoising applied. This
// keeps a hypothetical `r_denoiser nrd_lib` cvar value live-routable
// without crashing on the unfinished path.

#if defined(PT_ENABLE_NRD)

#include "VulkanNrdLibDenoiser.h"
#include "VulkanDevice.h"

#include "../core/Log.h"

#include <NRD.h>

#include <cstring>

namespace pt::rhi::vk {

namespace {

// Static helper: turn an nrd::Result into a short human-readable string
// for logging. NRD's API uses NRD_API extern "C" so the enum lives in
// the nrd namespace. We don't (yet) include all of NRD's enum string
// helpers; this is enough to identify the four error paths CreateInstance
// can produce.
const char* NrdResultStr(nrd::Result r) {
    switch (r) {
        case nrd::Result::SUCCESS:                 return "SUCCESS";
        case nrd::Result::FAILURE:                 return "FAILURE";
        case nrd::Result::INVALID_ARGUMENT:        return "INVALID_ARGUMENT";
        case nrd::Result::UNSUPPORTED:             return "UNSUPPORTED";
        case nrd::Result::NON_UNIQUE_IDENTIFIER:   return "NON_UNIQUE_IDENTIFIER";
        default:                                   return "UNKNOWN";
    }
}

// Stable identifier for the SIGMA_SHADOW denoiser instance. NRD uses
// 16-bit identifiers; values are user-chosen and only have to be unique
// within a single nrd::Instance. We park ours at a recognisable constant
// so the per-frame GetComputeDispatches(... &kSigmaShadowId, 1, ...) call
// in the follow-up PR has a stable handle to look up.
constexpr nrd::Identifier kSigmaShadowId = 0x5160;  // 'SI' nominal

}  // namespace

VulkanNrdLibDenoiser::~VulkanNrdLibDenoiser() {
    DestroyAll();
}

void VulkanNrdLibDenoiser::DestroyAll() {
    if (nrd_inst_ != nullptr) {
        // DestroyInstance is the only correct way to release NRD's
        // internal state -- it frees the InstanceDesc pipeline cache and
        // any allocator-callbacked memory. Safe to call on a half-built
        // instance per the API docs.
        nrd::DestroyInstance(*nrd_inst_);
        nrd_inst_ = nullptr;
    }
    ready_              = false;
    init_attempted_     = false;
    encode_stub_logged_ = false;
    cached_w_           = 0;
    cached_h_           = 0;
    frame_index_        = 0;
}

bool VulkanNrdLibDenoiser::Init(std::uint32_t width, std::uint32_t height) {
    if (device_ == nullptr || width == 0 || height == 0) {
        return false;
    }

    // Cached + ready means nothing to do. Init can be safely re-entered.
    if (ready_ && cached_w_ == width && cached_h_ == height) {
        return true;
    }

    // If we previously failed, don't retry -- NRD's CreateInstance is
    // deterministic; a failure once will fail every subsequent call. The
    // engine should see Ready()==false and route to the off path.
    //
    // Resizes do flush this gate via DestroyAll -> init_attempted_=false,
    // so a legitimate Init with new dimensions still runs.
    if (init_attempted_ && !ready_) {
        return false;
    }
    init_attempted_ = true;

    // If we're already initialised but dimensions changed, tear down so
    // CreateInstance below builds a fresh w/h-baked instance.
    if (nrd_inst_ != nullptr) {
        nrd::DestroyInstance(*nrd_inst_);
        nrd_inst_ = nullptr;
        ready_    = false;
    }

    // Log NRD library version at the static call site -- if this returns
    // a stale version number something is wrong with the static link
    // (multiple NRD.lib's resolved? wrong tag fetched?).
    const nrd::LibraryDesc* lib_desc = nrd::GetLibraryDesc();
    if (lib_desc == nullptr) {
        LOG_ERROR("VulkanNrdLibDenoiser::Init: nrd::GetLibraryDesc returned null "
                  "-- NRD static-link almost certainly broken");
        return false;
    }
    LOG_INFO("VulkanNrdLibDenoiser: NRD v{}.{}.{} (normal encoding={}, roughness encoding={})",
             lib_desc->versionMajor, lib_desc->versionMinor, lib_desc->versionBuild,
             static_cast<int>(lib_desc->normalEncoding),
             static_cast<int>(lib_desc->roughnessEncoding));

    // Build CreateInstance descriptor with just SIGMA_SHADOW for now.
    // REBLUR / RELAX add additional `DenoiserDesc{ ..., identifier }` rows;
    // each one would allocate its own pipeline + descriptor pool slice.
    nrd::DenoiserDesc sigma_desc{};
    sigma_desc.identifier      = kSigmaShadowId;
    sigma_desc.denoiser        = nrd::Denoiser::SIGMA_SHADOW;
    sigma_desc.renderWidth     = static_cast<uint16_t>(width);
    sigma_desc.renderHeight    = static_cast<uint16_t>(height);

    nrd::InstanceCreationDesc ci{};
    ci.denoisers               = &sigma_desc;
    ci.denoisersNum            = 1;
    // allocationCallbacks: NRD's docs say leave default (= use new/delete).
    // We can route through our PersistentHeap in the follow-up if memory
    // accounting starts mattering; SIGMA's internal scratch is small.

    nrd::Instance* new_inst = nullptr;
    nrd::Result r = nrd::CreateInstance(ci, new_inst);
    if (r != nrd::Result::SUCCESS || new_inst == nullptr) {
        LOG_ERROR("VulkanNrdLibDenoiser::Init: nrd::CreateInstance failed "
                  "({} -- {}x{} SIGMA_SHADOW)",
                  NrdResultStr(r), width, height);
        return false;
    }
    nrd_inst_ = new_inst;

    // Log the instance shape so the follow-up implementor knows what
    // resources they need to wire up. SIGMA_SHADOW at 1080p in NRD 4.17.3
    // produces roughly:
    //   - ~10 compute pipelines (pre-blur, blur, post-blur, history-fix,
    //     temporal-stabilization, copy + a handful of utility passes)
    //   - ~8 internal texture pool slots (penumbra ping-pong + history)
    //   - ~3 KB per-frame constant buffer high-water (dispatch count *
    //     constantBufferMaxDataSize)
    // These numbers grow when REBLUR / RELAX are added in the follow-ups.
    const nrd::InstanceDesc* inst_desc = nrd::GetInstanceDesc(*nrd_inst_);
    if (inst_desc != nullptr) {
        LOG_INFO("VulkanNrdLibDenoiser: SIGMA_SHADOW @ {}x{} ready "
                 "(pipelines={}, permanent textures={}, transient textures={}, "
                 "cbuf max/dispatch={} B)",
                 width, height,
                 inst_desc->pipelinesNum,
                 inst_desc->permanentPoolSize,
                 inst_desc->transientPoolSize,
                 inst_desc->constantBufferMaxDataSize);
    } else {
        // Shouldn't happen if CreateInstance succeeded; log defensively
        // because a null InstanceDesc on a non-null Instance would be an
        // NRD library bug we'd want to catch loudly.
        LOG_WARN("VulkanNrdLibDenoiser::Init: GetInstanceDesc returned null "
                 "after successful CreateInstance -- NRD library inconsistency");
    }

    cached_w_    = width;
    cached_h_    = height;
    frame_index_ = 0;
    ready_       = true;
    return true;
}

bool VulkanNrdLibDenoiser::ResizeTextures(std::uint32_t w, std::uint32_t h) {
    if (w == 0 || h == 0) return false;
    if (cached_w_ == w && cached_h_ == h && ready_) return true;
    // Init() handles the destroy-and-rebuild path internally.
    return Init(w, h);
}

void VulkanNrdLibDenoiser::Encode(VkCommandBuffer cb,
                                  TextureHandle   noisy_color_in,
                                  TextureHandle   /*depth_in*/,
                                  TextureHandle   /*normal_in*/,
                                  TextureHandle   /*motion_in*/,
                                  TextureHandle   /*shadow_visibility_in*/,
                                  TextureHandle   output,
                                  bool            reset_history) {
    if (!ready_ || nrd_inst_ == nullptr || device_ == nullptr) {
        return;
    }
    if (reset_history) {
        frame_index_ = 0;
    }

    // STUB: the real implementation calls nrd::SetCommonSettings + nrd::
    // SetDenoiserSettings, then nrd::GetComputeDispatches and walks the
    // returned DispatchDesc array, recording vkCmdDispatch for each. Until
    // that lands, we do a passthrough copy so the downstream bloom/tonemap
    // chain receives a valid (un-denoised) image.
    if (!encode_stub_logged_) {
        LOG_INFO("VulkanNrdLibDenoiser::Encode: SIGMA dispatch chain not yet "
                 "implemented in v0.3.30 (issue #50 scaffolding stage). "
                 "Passing noisy color through unchanged; bloom + tonemap "
                 "downstream still runs. Switch r_denoiser back to "
                 "svgf_atrous or nrd for actual denoising until the "
                 "follow-up PR lands.");
        encode_stub_logged_ = true;
    }

    // Passthrough copy. Both images are engine-owned, RGBA16F storage,
    // VK_IMAGE_LAYOUT_GENERAL. Caller (VulkanDevice::Denoise) is
    // responsible for the prior compute-write barrier on noisy_color_in;
    // we add a transfer<-compute / compute<-transfer pair around the copy
    // and let the caller's next compute consumer see the result.
    VkImage src = device_->LookupImage(noisy_color_in);
    VkImage dst = device_->LookupImage(output);
    if (src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE) {
        LOG_WARN("VulkanNrdLibDenoiser::Encode: image lookup miss "
                 "(noisy_color_in id={}, output id={})",
                 noisy_color_in.id, output.id);
        return;
    }

    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource            = region.srcSubresource;
    region.extent.width  = cached_w_;
    region.extent.height = cached_h_;
    region.extent.depth  = 1;
    vkCmdCopyImage(cb,
                   src, VK_IMAGE_LAYOUT_GENERAL,
                   dst, VK_IMAGE_LAYOUT_GENERAL,
                   1, &region);

    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    ++frame_index_;
}

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_NRD

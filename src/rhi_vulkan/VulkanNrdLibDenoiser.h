// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// NVIDIA RayTracingDenoiser (NRD) library wrapper -- issue #50.
//
// This is the real NRD library integration, sibling to VulkanNrdDenoiser
// (the in-house SVGF placeholder that currently backs the `r_denoiser nrd`
// cvar value). When PT_ENABLE_NRD is compiled in and the user opts in via
// the (forthcoming) `r_denoiser nrd_lib` cvar value, VulkanDevice routes
// the per-frame Denoise() call here instead of through SVGF.
//
// STATUS (v0.3.30): SCAFFOLDING -- not yet wired into the per-frame
// dispatcher. The class compiles only when PT_ENABLE_NRD is defined; Init()
// creates an nrd::Instance with SIGMA_SHADOW registered, queries the
// library descriptor, and stops. The per-frame Encode() path that walks
// the dispatch list from nrd::GetComputeDispatches and records vkCmdDispatch
// calls is intentionally a TODO -- see "What's left for follow-up" in the
// PR body for #50. The current code is sufficient to:
//   - confirm the FetchContent / static-link chain works end-to-end
//   - exercise the NRD library bootstrap (CreateInstance + GetLibraryDesc
//     + GetInstanceDesc -- the things that fail loudly if the static-lib
//     link is misconfigured)
//   - leave the API surface stable for the follow-up that fills in Encode
//
// Why SIGMA-first (not REBLUR or RELAX):
//   - SIGMA_SHADOW needs only 4 inputs (IN_PENUMBRA + the universal trio
//     IN_NORMAL_ROUGHNESS / IN_VIEWZ / IN_MV) and produces 1 output. The
//     path tracer's existing G-buffer set (color/depth/normal/motion) can
//     feed it with one new texture: a shadow visibility buffer
//     (path.shadow_visibility * 1.0 -- float per pixel). The full
//     REBLUR / RELAX integration needs IN_DIFF_RADIANCE_HITDIST + a
//     hit-distance encoding scheme + NRD-packed normal+roughness, which
//     is structurally more invasive.
//   - SIGMA gives an immediately visible win on the project's existing
//     scenes (the gold-metal hero scene has sharp shadow edges where
//     SVGF's a-trous filter softens transitions).
//
// Lifecycle: lazy. The instance is created on the first Encode() call
// (cheap, ~1 ms). Resize triggers a full instance recreate -- NRD bakes
// w/h into pipelines at create time. Mode switches (SIGMA <-> REBLUR
// later) also recreate; matches NRD's documented usage.
//
// Resource ownership:
//   Owned by this class:
//     - nrd::Instance handle (owned via CreateInstance/DestroyInstance)
//     - Per-pipeline VkPipeline + VkPipelineLayout + VkShaderModule list
//       (built from GetInstanceDesc()->pipelines). Built once per resize.
//     - One VkDescriptorPool sized from GetInstanceDesc()->descriptorPoolDesc.
//     - ~60 internal VkImage / VkImageView slots from
//       GetInstanceDesc()->permanentPool + transientPool. Backed by VMA.
//     - A staging VkBuffer for cbuffer uploads (`constantBufferMaxDataSize`
//       per dispatch, ring-allocated).
//   Borrowed (engine-owned, lifetime guaranteed by VulkanDevice):
//     - The G-buffer inputs (color / depth / normal / motion).
//     - The shadow-visibility input (path tracer writes a new G-buffer
//       at vk::binding(18) when r_denoiser==nrd_lib; gated by
//       PT_TARGET_SPIRV like the existing normal G-buffer).
//     - The denoised-output VkImage.
//
// Why this class isn't VulkanNrdDenoiser (the existing one):
//   The existing class is the SVGF placeholder that backs `r_denoiser nrd`
//   today. The cvar value is user-stable -- swapping its implementation
//   in the existing class mid-PR risks regressing the SVGF path. Splitting
//   into a new class lets the follow-up PR change routing (Engine.cpp's
//   `s == "nrd_lib"` branch) without touching SVGF code.

#pragma once

#if defined(PT_ENABLE_NRD)

#include "../rhi/Handles.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

// Forward-declare NRD opaque types so the header doesn't pull <NRD.h> into
// every TU. The real type is `struct nrd::Instance*`; the forward decl
// matches NRD.h's declaration of Instance as an incomplete struct in the
// nrd namespace.
namespace nrd { struct Instance; }

namespace pt::rhi::vk {

class VulkanDevice;

class VulkanNrdLibDenoiser {
public:
    explicit VulkanNrdLibDenoiser(VulkanDevice* parent) : device_(parent) {}
    ~VulkanNrdLibDenoiser();

    VulkanNrdLibDenoiser(const VulkanNrdLibDenoiser&)            = delete;
    VulkanNrdLibDenoiser& operator=(const VulkanNrdLibDenoiser&) = delete;

    // Build the nrd::Instance with SIGMA_SHADOW registered at `width x
    // height`. Returns false on hard failure (NRD library mismatch /
    // out-of-memory / Vulkan resource creation failure). On false the
    // engine should refuse to advertise the `nrd_lib` denoiser kind and
    // fall back to off (matching the OptiX path's failure handling).
    //
    // Safe to call repeatedly; succeeds-once-then-noop, or rebuilds if
    // (w, h) changes. Engine drives resize through ResizeTextures (below)
    // which calls back into here on dimension change.
    bool Init(std::uint32_t width, std::uint32_t height);

    // Recreate the nrd::Instance (and all backing resources) at the new
    // dimensions. NRD bakes w/h into its internal pipeline LDS sizing,
    // so a true resize requires DestroyInstance + CreateInstance. Cheap
    // (~1 ms) -- amortised across the resize event itself.
    //
    // Idempotent on dimension match.
    bool ResizeTextures(std::uint32_t w, std::uint32_t h);

    // True after Init() succeeded. Engine consults this in the dispatch
    // routing to avoid a wasted Encode call when init has failed.
    bool Ready() const { return ready_; }

    // Encode the NRD dispatch chain onto the given command buffer.
    //
    // STATUS: TODO -- placeholder logs once on first call ("VulkanNrdLib
    // Denoiser::Encode: SIGMA dispatch chain not yet implemented; falling
    // back to passthrough copy"). The placeholder does a vkCmdCopyImage
    // from `noisy_color_in` -> `output` so the engine's downstream
    // bloom + tonemap chain receives a valid image (just with no
    // denoising applied). This lets us land the scaffolding and verify
    // routing without a runtime crash on the unfinished path; the
    // follow-up PR replaces the copy with the real GetComputeDispatches
    // walker.
    //
    // Inputs (all engine-owned VkImages, in VK_IMAGE_LAYOUT_GENERAL):
    //   noisy_color_in     -- linear-HDR path-tracer output
    //   depth_in           -- R32F per-pixel linear-z (the engine's
    //                         existing depth_tex; NRD will eventually
    //                         want this re-encoded to its viewZ format,
    //                         but for scaffolding we pass-through).
    //   normal_in          -- RGBA16F world-space normal (vk::binding 16)
    //   motion_in          -- RG16F screen-space motion vectors
    //   shadow_visibility_in -- R8 or R16F shadow visibility (1 - shadow).
    //                         May be id=0 in the scaffolding stage; the
    //                         path tracer's shadow-visibility G-buffer
    //                         lands in the follow-up PR.
    //   output             -- linear-HDR scratch (denoised result)
    //
    // reset_history zeros NRD's internal temporal buffers (camera teleport,
    // scene change). NRD's CommonSettings::accumulationMode handles this
    // directly; we just translate the engine's flag.
    void Encode(VkCommandBuffer cb,
                TextureHandle   noisy_color_in,
                TextureHandle   depth_in,
                TextureHandle   normal_in,
                TextureHandle   motion_in,
                TextureHandle   shadow_visibility_in,
                TextureHandle   output,
                bool            reset_history);

private:
    void DestroyAll();

    VulkanDevice* device_      = nullptr;
    nrd::Instance* nrd_inst_   = nullptr;
    bool          init_attempted_ = false;
    bool          ready_          = false;
    bool          encode_stub_logged_ = false;  // one-shot log gate for the
                                                // not-yet-implemented dispatch

    // Cached frame dimensions. ResizeTextures() is a no-op when these match.
    std::uint32_t cached_w_ = 0;
    std::uint32_t cached_h_ = 0;

    // Frame counter passed to nrd::SetCommonSettings (NRD uses frameIndex
    // for jitter sequence tracking and history aging). Reset to 0 on
    // reset_history.
    std::uint32_t frame_index_ = 0;
};

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_NRD

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// OptiX denoiser strategy for the Vulkan backend.
//
// Sibling to VulkanNrdDenoiser (the in-house SVGF impl). When the
// engine's r_denoiser cvar selects an `optix_*` value, VulkanDevice
// routes the per-frame Denoise() call here instead of through SVGF.
// The denoiser itself runs entirely on CUDA/OptiX; CUDA imports the
// Vulkan G-buffer textures via VK_KHR_external_memory + (Win32 or FD)
// and writes the denoised result back into a shared output VkImage.
// Sync between the two APIs uses timeline VkSemaphores exported as
// CUDA external semaphores. Vulkan retains ownership of the swapchain
// and presentation; OptiX just denoises.
//
// Variants live behind the `Kind` enum below. Phase 1a ships Hdr and
// HdrAov; Phase 1b adds TemporalHdr and TemporalHdrAov. The variants
// map 1:1 to OptiX's OPTIX_DENOISER_MODEL_KIND_* enum.
//
// Lifecycle: lazy. Init() runs on the first Encode() call where the
// requested kind matches; subsequent Encode()s reuse the built state.
// Switching kinds (e.g. r_denoiser optix_hdr -> optix_hdr_aov) tears
// down and rebuilds; cheap because the OptiX state buffer dominates
// and is small (<10 MB at 1080p).
//
// Linux note: structurally supported via the FD-handle branch in
// ExternalHandles.h, but untested as of v0.3.1. Will be validated on
// a real Linux box when one's available.

#pragma once

#if defined(PT_ENABLE_OPTIX)

#include "../rhi/Device.h"
#include "ExternalHandles.h"

#include <vulkan/vulkan.h>

#include <cstdint>

// Forward-declare OptiX types so the header doesn't drag the full SDK
// into every TU that pulls this in. The real includes happen in the .cpp.
struct OptixDeviceContext_t;
typedef struct OptixDeviceContext_t* OptixDeviceContext;
struct OptixDenoiser_t;
typedef struct OptixDenoiser_t* OptixDenoiser;

namespace pt::rhi::vk {

class VulkanDevice;

class VulkanOptixDenoiser {
public:
    // OptiX denoiser model kind. Maps 1:1 to OPTIX_DENOISER_MODEL_KIND_*.
    //
    //   Hdr           Plain HDR model. Cheapest, no AOV inputs.
    //                 Cvar value: r_denoiser optix_hdr.
    //
    //   HdrAov        HDR model + albedo + normal AOV hints. Better
    //                 quality, especially in shadowed regions where
    //                 plain HDR can over-smooth surface detail. Costs
    //                 the path tracer one additional output (primary
    //                 albedo, RGBA16F at vk::binding 17, gated by
    //                 PT_TARGET_SPIRV).
    //                 Cvar value: r_denoiser optix_hdr_aov.
    //
    //   TemporalHdr     [Phase 1b] Temporal HDR model. Reuses motion
    //                 vectors + previous denoised output for stable
    //                 frame-to-frame coherence.
    //
    //   TemporalHdrAov  [Phase 1b] Temporal HDR + AOV. Highest quality
    //                 of the four; recommended default once landed.
    enum class Kind : std::uint8_t {
        Hdr,
        HdrAov,
        // TemporalHdr,    // Phase 1b
        // TemporalHdrAov, // Phase 1b
    };

    explicit VulkanOptixDenoiser(VulkanDevice* device, Kind kind);
    ~VulkanOptixDenoiser();

    VulkanOptixDenoiser(const VulkanOptixDenoiser&)            = delete;
    VulkanOptixDenoiser& operator=(const VulkanOptixDenoiser&) = delete;

    // Lazy init: creates the CUDA context, loads the OptiX function table,
    // builds the denoiser handle for `kind_`, allocates state + scratch
    // buffers. Returns false if any step fails (no NVIDIA GPU, driver too
    // old, OptiX SDK runtime missing, etc.). On failure the engine falls
    // back to off (matches the existing SupportsDenoise lazy pattern).
    //
    // Safe to call repeatedly; succeeds-once or fails-permanently for the
    // lifetime of this instance.
    bool Init();

    // Per-frame dispatch. Records Vulkan->CUDA sync, runs optixDenoiserInvoke,
    // records CUDA->Vulkan sync. No-op until Init() succeeds.
    void Encode(VkCommandBuffer cb, const Device::DenoiseDesc& d);

    // Whether the requested kind is currently usable (Init succeeded).
    // Engine consults this in the dispatch routing to avoid a wasted
    // Encode call when init has failed.
    bool IsReady() const { return ready_; }

    Kind GetKind() const { return kind_; }

private:
    void DestroyResources();

    VulkanDevice* device_         = nullptr;
    Kind          kind_           = Kind::Hdr;
    bool          init_attempted_ = false;
    bool          ready_          = false;

    // CUDA + OptiX handles. Real types come in once we wire up the
    // implementation -- left as opaque placeholders in the scaffold.
    OptixDeviceContext optix_ctx_      = nullptr;
    OptixDenoiser      optix_denoiser_ = nullptr;
    // CUcontext, CUstream, scratch/state CUdeviceptr land here in
    // the next commit. Keeping the scaffold minimal so the build
    // verification step doesn't depend on resolving the full CUDA
    // type surface.
};

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_OPTIX

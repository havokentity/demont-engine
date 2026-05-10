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
// Resource ownership:
//   Owned by this class (allocated in Init / ResizeTextures, freed
//   in DestroyResources):
//     - CUDA driver context, stream
//     - OptiX device context + denoiser handle
//     - CUDA scratch buffer (state, scratch, hdr-intensity)
//     - Two external Vulkan VkImages (color_in, output) at the
//       current frame size, in VK_IMAGE_LAYOUT_GENERAL, with
//       VK_KHR_external_memory_win32 (or _fd) flags. CUDA imports
//       these as cudaArrays + surface objects.
//     - Two timeline VkSemaphores (vk_signals_cuda_waits and
//       cuda_signals_vk_waits), exported to CUDA.
//   Borrowed:
//     - VulkanDevice* (lifetime guaranteed by VulkanDevice owning us)
//     - DenoiseDesc.color_in / .output (engine-owned VkImages, we
//       only vkCmdCopyImage to/from them; never alias their memory)
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

// Forward-declare CUDA + OptiX types so the header doesn't drag the
// full SDKs into every TU that pulls this in. The real includes
// happen in the .cpp.
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUstream_st;
typedef struct CUstream_st* CUstream;
struct OptixDeviceContext_t;
typedef struct OptixDeviceContext_t* OptixDeviceContext;
struct OptixDenoiser_t;
typedef struct OptixDenoiser_t* OptixDenoiser;

// CUDA driver-API CUdeviceptr is unsigned long long across all
// supported platforms; safe to mirror as a typedef so we don't pull
// in cuda.h here.
typedef unsigned long long CUdeviceptr_v2_local;

// CUDA runtime opaque handles. Match the SDK definitions; we only
// pass them through, never deref.
struct cudaArray;
typedef struct cudaArray* cudaArray_t;
typedef unsigned long long cudaSurfaceObject_t;
struct CUexternalMemory_st;
typedef struct CUexternalMemory_st* cudaExternalMemory_t;
struct CUexternalSemaphore_st;
typedef struct CUexternalSemaphore_st* cudaExternalSemaphore_t;

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
    //                 NOTE: AOV plumbing lands in Phase 1a step 3 --
    //                 this enum value is wired but currently behaves
    //                 the same as Hdr until the path-tracer albedo
    //                 output + AOV inputs land.
    //
    //   TemporalHdr     [Phase 1b] Temporal HDR model.
    //   TemporalHdrAov  [Phase 1b] Temporal HDR + AOV.
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

    // Lazy init: creates the CUDA context, loads the OptiX function
    // table, builds the denoiser handle for `kind_`. Scratch + external
    // textures are sized lazily on the first Encode call (when we
    // know the actual frame size). Returns false if any step fails
    // (no NVIDIA GPU, driver too old, OptiX SDK runtime missing,
    // etc.). On failure the engine falls back to off.
    //
    // Safe to call repeatedly; succeeds-once or fails-permanently for
    // the lifetime of this instance.
    bool Init();

    // Per-frame dispatch. Records Vulkan->CUDA sync, runs
    // optixDenoiserInvoke, records CUDA->Vulkan sync. No-op until
    // Init() succeeds. Resizes the scratch + external textures
    // automatically when the input size changes.
    void Encode(VkCommandBuffer cb, const Device::DenoiseDesc& d);

    // Whether the requested kind is currently usable (Init succeeded
    // AND scratch is sized). Engine consults this in the dispatch
    // routing to avoid a wasted Encode call when init has failed.
    bool IsReady() const { return ready_; }

    Kind GetKind() const { return kind_; }

private:
    bool InitOnce();              // CUDA + OptiX context (one-shot)
    bool ResizeScratch(std::uint32_t w, std::uint32_t h);
    bool ResizeExternalImages(std::uint32_t w, std::uint32_t h);
    bool ResizeExternalSemaphores();
    void DestroyResources();
    void DestroyExternalImage(int slot);   // 0 = color_in, 1 = output
    void DestroyExternalSemaphores();

    VulkanDevice* device_         = nullptr;
    Kind          kind_           = Kind::Hdr;
    bool          init_attempted_ = false;
    bool          ready_          = false;

    // CUDA driver + runtime state (single-context, single-stream).
    CUcontext     cuda_ctx_       = nullptr;
    CUstream      cuda_stream_    = nullptr;

    // OptiX state.
    OptixDeviceContext optix_ctx_      = nullptr;
    OptixDenoiser      optix_denoiser_ = nullptr;
    std::uint64_t      state_size_     = 0;     // bytes
    std::uint64_t      scratch_size_   = 0;     // bytes
    std::uint64_t      overlap_        = 0;     // tile overlap (unused at 1080p)
    // CUDA device pointers backing the scratch/state. Stored as
    // uint64_t to avoid pulling cuda.h into the header; the real
    // type is CUdeviceptr (== unsigned long long).
    std::uint64_t      state_buf_      = 0;
    std::uint64_t      scratch_buf_    = 0;
    std::uint64_t      intensity_buf_  = 0;     // single-float HDR intensity scratch

    // Cached frame dimensions. ResizeXxx is a no-op when these match
    // the requested w/h.
    std::uint32_t cached_w_ = 0;
    std::uint32_t cached_h_ = 0;

    // External Vulkan-side resources. Two pairs (color_in @ slot 0,
    // output @ slot 1) of (VkImage, VkDeviceMemory, cudaExternalMemory,
    // cudaArray, cudaSurfaceObject). All RGBA16F, layout GENERAL.
    struct ExternalImage {
        VkImage              image     = VK_NULL_HANDLE;
        VkDeviceMemory       memory    = VK_NULL_HANDLE;
        VkImageView          view      = VK_NULL_HANDLE;
        cudaExternalMemory_t cuda_ext  = nullptr;
        cudaArray_t          cuda_arr  = nullptr;
        cudaSurfaceObject_t  cuda_surf = 0;
    };
    ExternalImage img_color_in_;
    ExternalImage img_output_;

    // Timeline VkSemaphores exported to CUDA for cross-API sync.
    // vk_signals_cuda_waits_ : Vulkan queue signals after recording
    //                          color_in copy; CUDA stream waits before
    //                          launching the denoise kernel.
    // cuda_signals_vk_waits_ : CUDA stream signals after the denoise
    //                          kernel completes; Vulkan queue waits
    //                          before reading from the output image.
    // The timeline payload counter is monotonically incremented per
    // Encode call; we track the last-used value here so the next
    // Encode picks the next pair.
    VkSemaphore             sem_vk_to_cuda_      = VK_NULL_HANDLE;
    VkSemaphore             sem_cuda_to_vk_      = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cuda_sem_vk_to_cuda_ = nullptr;
    cudaExternalSemaphore_t cuda_sem_cuda_to_vk_ = nullptr;
    std::uint64_t           timeline_counter_    = 0;
};

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_OPTIX

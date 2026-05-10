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

// Forward-declare OptiX types and the CUDA driver/runtime context +
// stream + external-memory/semaphore opaque pointers so we don't
// have to include <optix.h> from every TU that pulls in this header.
//
// The transitive cost from ExternalHandles.h above is non-zero --
// it pulls <cuda.h> + <cuda_runtime.h> for the CUDA-side handle-type
// enums (cudaExternalMemoryHandleType etc.) -- so the "header is
// SDK-light" claim is partial: cuda.h DOES end up in any TU that
// includes us. Acceptable because the only non-OptiX includer is
// VulkanDevice.{h,cpp} (the SVGF path doesn't pull this header), and
// the whole thing is gated behind PT_ENABLE_OPTIX so non-NVIDIA
// builds skip it entirely. Keeping the OptiX forward-decls here
// (vs another #include) still saves ~200 KB of headers per TU --
// non-trivial in the engine's hot compile path.
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUstream_st;
typedef struct CUstream_st* CUstream;
struct OptixDeviceContext_t;
typedef struct OptixDeviceContext_t* OptixDeviceContext;
struct OptixDenoiser_t;
typedef struct OptixDenoiser_t* OptixDenoiser;

// CUDA runtime opaque handles. Match the SDK definitions; we only
// pass them through, never deref.
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
    //   HdrAov        HDR model + albedo + normal AOV hints
    //                 (OPTIX_DENOISER_MODEL_KIND_AOV). Better quality
    //                 than plain Hdr in disocclusion / low-confidence
    //                 regions and at diffuse-color edges where the
    //                 plain HDR model can over-smooth surface detail.
    //                 Costs the path tracer two additional G-buffer
    //                 outputs:
    //                   - primary albedo (RGBA16F at vk::binding 17)
    //                   - primary normal (RGBA16F at vk::binding 16,
    //                     reused from the SVGF/NRD path)
    //                 Both gated by PT_TARGET_SPIRV. The engine
    //                 allocates them only when r_denoiser is
    //                 optix_hdr_aov. Cvar value: r_denoiser optix_hdr_aov.
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

    // Per-frame dispatch. Records the input copy into the engine's
    // command buffer, schedules the async CUDA wait+denoise+signal,
    // and records (but does not submit) the private cb that copies
    // the denoised output back to the engine's d.output. The actual
    // private-cb submit happens in SubmitPostMain() below, after the
    // engine's main vkQueueSubmit completes -- otherwise queue-order
    // serialization deadlocks (engine's cb signals the timeline our
    // private cb waits on).
    //
    // No-op until Init() succeeds.
    void Encode(VkCommandBuffer cb, const Device::DenoiseDesc& d);

    // Called by VulkanDevice::Submit after the main engine cb is
    // submitted. Submits the private output-copy cb (waiting on the
    // CUDA->VK timeline at the value Encode set up), so it executes
    // strictly after CUDA finishes the denoise. Idempotent: a frame
    // where Encode wasn't called (denoiser off) just returns.
    void SubmitPostMain();

    // Block the calling thread until the CUDA stream drains. Called
    // by VulkanDevice::DestroyDevice BEFORE vkDeviceWaitIdle so the
    // private output-copy cb's wait on sem_cuda_to_vk gets satisfied
    // (CUDA signals the timeline as it completes the queued denoise
    // work). Without this, vkDeviceWaitIdle deadlocks at shutdown.
    void DrainCuda();

    // Whether the requested kind is currently usable (Init succeeded
    // AND scratch is sized). Engine consults this in the dispatch
    // routing to avoid a wasted Encode call when init has failed.
    bool IsReady() const { return ready_; }

    Kind GetKind() const { return kind_; }

private:
    bool InitOnce();              // CUDA + OptiX context (one-shot)
    bool ResizeScratch(std::uint32_t w, std::uint32_t h);
    bool ResizeExternalBuffers(std::uint32_t w, std::uint32_t h);
    bool ResizeExternalSemaphores();
    bool EnsureCommandPool();
    void DestroyResources();
    void DestroyExternalBuffer(int slot);   // 0 = color_in, 1 = output
    void DestroyExternalSemaphores();
    void DestroyCommandPool();

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
    // output @ slot 1) of (VkBuffer, VkDeviceMemory, cudaExternalMemory,
    // CUdeviceptr). RGBA16F linear in CUDA's view -- the engine's
    // images get vkCmdCopyImageToBuffer'd in to color_in, OptiX denoises
    // the linear data into output, then vkCmdCopyBufferToImage transfers
    // back to the engine's d.output VkImage.
    //
    // Why VkBuffers and not VkImages: OptiX's OptixImage2D::data takes
    // a CUdeviceptr (linear memory). Vulkan-imaged-as-cudaArray
    // (surface object) doesn't expose linear memory and can't feed
    // OptiX directly. cudaExternalMemoryGetMappedBuffer on an external
    // VkBuffer returns the CUdeviceptr we need.
    //
    // Stored as std::uint64_t to avoid pulling cuda.h into the header;
    // the real type is CUdeviceptr (== unsigned long long).
    struct ExternalBuffer {
        VkBuffer             buffer    = VK_NULL_HANDLE;
        VkDeviceMemory       memory    = VK_NULL_HANDLE;
        VkDeviceSize         size      = 0;       // total bytes
        cudaExternalMemory_t cuda_ext  = nullptr;
        std::uint64_t        cuda_ptr  = 0;       // CUdeviceptr
    };
    ExternalBuffer buf_color_in_;
    ExternalBuffer buf_output_;
    // OptiX AOV-only guide-layer inputs. Allocated alongside
    // color_in/output by ResizeExternalBuffers when kind_ == HdrAov;
    // left in their default-empty state for kind_ == Hdr (the plain-
    // HDR model has no guide layers). Imported with the same external-
    // memory machinery as color_in -- the engine's albedo_tex and
    // normal_tex VkImages get vkCmdCopyImageToBuffer'd in alongside
    // color_in during Encode, and OptiX reads them as
    // OptixDenoiserGuideLayer::albedo / .normal in SubmitPostMain.
    ExternalBuffer buf_albedo_;
    ExternalBuffer buf_normal_;
    // Pixel layout matching what we copy: RGBA16F = 8 bytes per
    // pixel, tightly packed, row stride = width * 8. Albedo and normal
    // share this same layout (OptiX accepts HALF4 for guide layers
    // and ignores the alpha channel).
    static constexpr std::uint32_t kBytesPerPixel = 8;

    // Private Vulkan command pool + per-frame-in-flight command
    // buffer used for the output copy submit. The output copy waits
    // on sem_cuda_to_vk_ before reading our buffer, so it can't be
    // recorded into the engine's main cb (which doesn't wait on
    // CUDA's signal). Owned + freed by this class.
    VkCommandPool   private_pool_      = VK_NULL_HANDLE;
    static constexpr int kFramesInFlight = 2;
    VkCommandBuffer private_cb_out_[kFramesInFlight] {};
    int             private_cb_index_  = 0;

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

    // Per-frame state shared between Encode() and SubmitPostMain().
    // Encode records the private output-copy cb and saves which
    // private_cb_out_ slot it used + the timeline value the cb waits
    // on. SubmitPostMain reads these and does the vkQueueSubmit. The
    // pending flag is reset after each SubmitPostMain so a frame
    // where Encode wasn't called doesn't accidentally re-submit a
    // stale cb.
    bool          pending_post_main_       = false;
    int           pending_pcb_slot_        = 0;
    std::uint64_t pending_wait_value_      = 0;
};

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_OPTIX

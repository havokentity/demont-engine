// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SVGF-style temporal + a-trous compute denoiser for the Vulkan
// backend. Drives the engine's `r_denoiser svgf_basic`,
// `r_denoiser svgf_atrous`, and `r_denoiser nrd` cvar values.
// `nrd` is forward-looking: when NVIDIA's RayTracingDenoiser
// library lands properly (multi-day integration, see Raytracer
// Plan/FOLLOW_UPS.md), the cvar value will switch implementation
// but stay user-stable. Today it aliases svgf_atrous with a
// one-time log on transition.
//
// Pipeline shape, per Encode() call (atrous mode, N from
// r_svgf_atrous_passes; clamped to 1..5):
//   1) DenoiseTemporal: reproject color_history + moments_history
//      + variance, accumulate noisy input. Adaptive alpha falls to 1.0
//      on reprojection failure (off-screen / depth+normal mismatch),
//      with a 7x7 spatial variance fallback for sample_count < 4.
//      Writes color into v_atrous_a (intermediate) when N >= 1.
//   2) DenoiseAtrous pass 1 (step=1): variance-gated 5x5 edge-aware
//      spatial filter v_atrous_a -> v_hist_write (Schied feedback
//      target: next frame's temporal reads this).
//   ...) Passes 2..N-1 ping-pong between v_atrous_b / v_atrous_a with
//      step sizes 2, 4, 8, 16 (footprint doubles per pass).
//   N) Last pass writes directly into the caller's `output`. For N==1
//      we vkCmdCopyImage v_hist_write -> output to publish.
// Variance ping-pongs in lockstep with color through variance_a/b.
// Frame parity flips which history side is read vs written so the next
// frame picks up this frame's spatially-filtered accumulation.
// Basic mode (atrous_enabled == false) skips the chain entirely:
// temporal writes straight to v_hist_write, then copy -> output.
//
// Resource ownership: the denoiser owns 8 scratch textures + 4 storage
// buffers. Textures: color history ping-pong (history_a/history_b,
// RGBA16F), depth history ping-pong (depth_history_a/b, R32F), normal
// history ping-pong (normal_history_a/b, RGBA16F), and a-trous
// color ping-pong (atrous_a/b, RGBA16F). Buffers: moments_history_a/b
// (RG32F mu1/mu2 luminance moments, cross-frame ping-pong) and
// variance_a/b (R32F per-pixel luminance variance, intra-frame
// ping-pong through the a-trous chain). The moments + variance went to
// storage buffers (not RW textures) to clear Metal's 8-RW-texture
// compute cap; the SPIR-V emission is unaffected. The engine continues
// to own the noisy color/depth/motion/normal G-buffers (PathTrace
// bindings 6/7/8/16) plus final `output` (post_denoise_hdr).
//
// Pipeline layout: dedicated, NOT the engine's unified 19-binding
// layout. 12 bindings -- 8 storage images at 0..7 + 4 storage buffers
// at 8..11 (moments_in/out, variance_in/out) -- plus a 32-byte push
// constant range. Each Encode dispatch picks a fresh descriptor set
// from the ring so we don't conflict with prior in-flight dispatches.

#pragma once

#include "../rhi/Handles.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

namespace pt::rhi::vk {

class VulkanDevice;

class VulkanNrdDenoiser {
public:
    explicit VulkanNrdDenoiser(VulkanDevice* parent) : device_(parent) {}
    ~VulkanNrdDenoiser();

    // Builds the layout, descriptor pool, and pipelines. Returns false
    // on hard failure (out-of-memory / shader rejected by validation);
    // the caller should then refuse to advertise SupportsDenoise.
    bool Init();

    // Allocate / reallocate the four scratch textures sized to (w, h).
    // Idempotent on size match. Returns false on allocation failure.
    bool ResizeTextures(std::uint32_t w, std::uint32_t h);

    // Encode the denoise pipeline onto an in-progress command buffer.
    // Caller is responsible for any prior memory barrier on the
    // input textures (the engine already inserts a Barrier() between
    // path-tracer and autoexpose; we add our own between denoise
    // sub-passes). `reset_history` zeros temporal accumulation.
    // `atrous_enabled` selects between the two quality tiers:
    //   false (svgf_basic) -- temporal pass only, then vkCmdCopyImage
    //                         the temporal result into `output`.
    //   true  (svgf_atrous /
    //          nrd alias)  -- temporal pass + N a-trous wavelet passes,
    //                         where N comes from `atrous_passes` (1..5,
    //                         clamped). Footprint doubles per pass:
    //                         1 = 5x5, 2 = 9x9, 3 = 17x17, 4 = 33x33,
    //                         5 = 65x65 (canonical SVGF).
    void Encode(VkCommandBuffer cb,
                TextureHandle   color_in,
                TextureHandle   depth_in,
                TextureHandle   motion_in,
                TextureHandle   normal_in,
                TextureHandle   output,         // linear-HDR scratch
                TextureHandle   final_output,   // tonemapped-LDR target (swapchain)
                BufferHandle    exposure_state,
                TextureHandle   bloom_in,       // bloom mip 0 (id=0 -> bloom_intensity must be 0)
                float           bloom_intensity,
                bool            reset_history,
                bool            atrous_enabled,
                std::uint32_t   atrous_passes,  // 1..5
                bool            hdr_pipeline,
                TextureHandle   stars_in);      // accum_stars (issue #46; id=0 -> dummy)

    // True after Init() succeeded. Used by VulkanDevice::SupportsDenoise.
    bool Ready() const { return ready_; }

    // Encode JUST the DenoiseFinalize compute dispatch (HDR -> ACES + sRGB
    // -> swapchain) without running any of the SVGF passes that
    // normally precede it. Exposed so the OptiX denoiser path can
    // reuse this proven tonemap+sRGB stage instead of rolling its
    // own. Caller is responsible for:
    //   - Both images in VK_IMAGE_LAYOUT_GENERAL before the call
    //   - Any pre-barrier on color_in (the dispatch is a shader read)
    //   - Any post-barrier the next consumer needs
    //
    // Pipeline must be Ready() (call Init() first if not). Width and
    // height drive the dispatch grid; hdr_pipeline picks ACES+sRGB
    // (true) vs sRGB-OETF only (false; for paths that pre-tonemap).
    void EncodeFinalizeOnly(VkCommandBuffer cb,
                            VkImageView    color_in_view,
                            VkImageView    final_output_view,
                            VkBuffer       exposure_state_buf,
                            VkImageView    bloom_in_view,   // bloom mip 0 (placeholder if no bloom)
                            float          bloom_intensity, // 0 = skip bloom add
                            std::uint32_t  width,
                            std::uint32_t  height,
                            bool           hdr_pipeline,
                            VkImageView    stars_in_view);  // accum_stars (issue #46;
                                                            // safe dummy if stars off)

private:
    void DestroyAll();
    bool BuildLayout();
    bool BuildPipelines();
    bool BuildDescriptorPool();
    // Public: WaitIdle()s once then frees all 12 scratch resources via
    // the device's NoWait fast path. Use for standalone resize.
    void DestroyTextures();
    // Same teardown as DestroyTextures but skips the WaitIdle -- caller
    // promises a wait already happened upstream (DestroyAll's single
    // batch wait covers DestroyTextures + dummy resources + pipelines
    // + pool in one stall).
    void DestroyTexturesAlreadyWaited();

    // Acquire the next descriptor set in the ring. Wraps when full.
    VkDescriptorSet NextSet();

    // Bind a single dispatch's resources into `set` and dispatch with
    // `pipe` + `push`. Bindings 0..7 are storage images (engine-managed
    // G-buffer + denoiser texture history); bindings 8..11 are storage
    // buffers (moments + variance). See BuildLayout() for per-slot
    // semantics.
    static constexpr std::uint32_t kPassImages  = 8;
    static constexpr std::uint32_t kPassBuffers = 4;
    void RecordPass(VkCommandBuffer cb,
                    VkPipeline      pipe,
                    VkDescriptorSet set,
                    const VkImageView (&views)[kPassImages],
                    const VkBuffer    (&buffers)[kPassBuffers],
                    const void*     push,
                    std::size_t     push_size,
                    std::uint32_t   gx,
                    std::uint32_t   gy);

    VulkanDevice*         device_       = nullptr;
    bool                  ready_        = false;
    VkDescriptorSetLayout dset_layout_  = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout_  = VK_NULL_HANDLE;
    VkPipeline            temporal_pipe_ = VK_NULL_HANDLE;
    VkPipeline            atrous_pipe_  = VK_NULL_HANDLE;
    VkDescriptorPool      dpool_        = VK_NULL_HANDLE;
    // DenoiseFinalize uses its own layout (2 storage images + 1
    // storage buffer; different from the temporal/atrous 8-image
    // layout, which gained depth/normal history bindings 6/7 in
    // PR #3) so it gets a parallel layout / pipeline / set ring.
    // Same VkDescriptorPool serves both since we sized it for a
    // mixed pool.
    VkDescriptorSetLayout finalize_dset_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      finalize_pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline            finalize_pipe_        = VK_NULL_HANDLE;
    static constexpr int  kFinalizeSetRing      = 4;
    VkDescriptorSet       finalize_sets_[kFinalizeSetRing] {};
    int                   next_finalize_set_    = 0;

    // 4 dispatches per Encode * 2 frames in flight = 8 minimum; round
    // up so a stray reset doesn't wrap mid-frame and recycle a set
    // that's still bound on the GPU.
    static constexpr int kSetRing = 16;
    VkDescriptorSet       sets_[kSetRing] {};
    int                   next_set_     = 0;

    // Owned scratch resources (RHI-managed, freed via DestroyTextures()).
    // Cross-frame ping-pong: history_*, depth_history_*, normal_history_*
    // (textures), moments_history_* (storage buffers; see BuildLayout()
    // for why bindings 8..11 are buffers not images).
    // Within-frame ping-pong: atrous_a/b color scratch (textures) +
    // variance_a/b ping-pong (storage buffers).
    std::uint64_t         history_a_id_         = 0;
    std::uint64_t         history_b_id_         = 0;
    std::uint64_t         depth_history_a_id_   = 0;
    std::uint64_t         depth_history_b_id_   = 0;
    std::uint64_t         normal_history_a_id_  = 0;
    std::uint64_t         normal_history_b_id_  = 0;
    std::uint64_t         moments_history_a_buf_ = 0;
    std::uint64_t         moments_history_b_buf_ = 0;
    std::uint64_t         atrous_a_id_          = 0;
    std::uint64_t         atrous_b_id_          = 0;
    std::uint64_t         variance_a_buf_       = 0;
    std::uint64_t         variance_b_buf_       = 0;
    // 1x1 image placeholders for the bindings the atrous shader
    // declares but does not consume:
    //   dummy_color_id_  -- RGBA16F, plugs atrous slot 1 (color_history)
    //   dummy_motion_id_ -- RG16F,   plugs atrous slot 3 (motion)
    // Allocated once at Init(); live for the denoiser's lifetime.
    std::uint64_t         dummy_color_id_  = 0;
    std::uint64_t         dummy_motion_id_ = 0;
    // Storage-buffer placeholder for temporal's slot-10 (variance_in
    // unused). Sized to a single element; the shader never reads it but
    // Vulkan still validates the descriptor write.
    std::uint64_t         dummy_variance_buf_ = 0;

    std::uint32_t         cached_w_     = 0;
    std::uint32_t         cached_h_     = 0;
    // Flips each Encode so frame N reads history_a / writes history_b,
    // frame N+1 reads history_b / writes history_a. Reset by
    // reset_history.
    std::uint32_t         frame_parity_ = 0;
    bool                  needs_history_clear_ = true;
    // Issue #46 round-3: stars_in lookup-failure log latch. Encode runs
    // every frame, so a persistent stale-handle race would otherwise
    // spam the log once per frame. We emit on the leading edge (first
    // time per session the lookup fails) and clear when it next
    // resolves cleanly. Mirrors the one-shot vs spamming pattern used
    // by other per-frame error paths in this file.
    bool                  stars_lookup_warn_latched_ = false;
};

}  // namespace pt::rhi::vk

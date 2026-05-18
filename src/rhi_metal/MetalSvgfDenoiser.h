// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SVGF-style temporal + a-trous compute denoiser for the Metal backend.
// Drives the engine's `r_denoiser svgf_basic` and `r_denoiser svgf_atrous`
// cvar values on Apple Silicon.
//
// Parity with the Vulkan path: this is the exact same Slang source
// (shaders/DenoiseTemporal.slang + shaders/DenoiseAtrous.slang)
// cross-compiled to MSL via slangc. The dispatch contract matches
// VulkanNrdDenoiser one-for-one:
//   1) Temporal pass: reproject + accumulate denoise_color into the
//      ping-pong color history (history_a / history_b based on frame
//      parity). depth and normal history get a blit-copy in lockstep.
//   2a) svgf_basic: blit history_write -> output and return.
//   2b) svgf_atrous: three a-trous wavelet passes at step sizes 1/2/4
//       ping-ponging atrous_a / atrous_b; the last pass writes directly
//       into output.
//
// What this does NOT do (intentional): the post-denoise tonemap. On
// Metal the engine's existing Tonemap.slang pipeline is built and
// dispatched, reading post_denoise_hdr -> writing swapchain, so
// MetalSvgfDenoiser just leaves its linear-HDR result in `output`
// (= post_denoise_hdr) and the standard bloom + tonemap chain takes
// it from there. This is the only structural difference vs the
// Vulkan SVGF path -- which runs DenoiseFinalize.slang to do the
// swapchain write itself, because Engine.cpp gates
// `use_engine_tonemap` to Metal only (see b2f4bfd). The Vulkan
// Tonemap.slang pipeline IS built (so re-enabling is a single-flag
// flip) but is currently un-routed pending root-cause of the
// descriptor / layout corruption that black-screened the chain on PC.
//
// Lifetime: MetalDevice owns one of these and lazily allocates the
// scratch textures + pipelines on first Encode(). Pipelines stay for
// the device lifetime; textures resize when the swapchain does.

#pragma once

#include "../rhi/Handles.h"

#include <cstdint>
#include <memory>

namespace MTL {
class Device;
class CommandBuffer;
class ComputePipelineState;
class Texture;
class Buffer;
}

namespace pt::rhi::mtl {

class MetalDevice;

class MetalSvgfDenoiser {
public:
    explicit MetalSvgfDenoiser(MetalDevice* parent) : device_(parent) {}
    ~MetalSvgfDenoiser();

    // Build the two MSL pipelines from the embedded slang-compiled MSL
    // blobs. Returns false if either pipeline fails (logged). Idempotent.
    bool Init();

    // Allocate / resize the eight scratch textures sized to (w, h).
    // Idempotent on size match. Returns false on allocation failure.
    bool ResizeTextures(std::uint32_t w, std::uint32_t h);

    // Encode the temporal + (optional) atrous chain onto `cb`. The
    // command buffer's outer compute encoder MUST have been ended by
    // the caller (MetalDevice::Denoise flushes via cmd_->FlushEncoder
    // before this).
    //   color_in / depth_in / motion_in / normal_in -- engine-owned
    //     G-buffer textures (path tracer wrote these earlier in `cb`).
    //   albedo_in -- engine-owned primary-hit albedo G-buffer (issue
    //     #119). Required when `albedo_demod_enabled` is true; the
    //     SVGF chain divides input radiance by this and the remod
    //     pass multiplies back to produce textured radiance. May be
    //     nullptr when demod is disabled -- the temporal/atrous
    //     kernels still BIND a placeholder (dummy_color_) so MSL has
    //     a valid argument for the albedo slot (MSL slot 8 for the
    //     9-texture temporal/atrous kernels, slot 2 for the 3-texture
    //     remod kernel; Slang compacts [[vk::binding(12)]] down to
    //     the next free slot rather than honouring the vk slot number
    //     -- issue #164), and the `demod_enabled` push gate at 0
    //     skips every divide/multiply.
    //   output -- engine-owned linear-HDR target the tonemap chain
    //     reads next (= post_denoise_hdr).
    //   reset_history -- true clears the temporal accumulation.
    //   atrous_enabled -- true runs the spatial filter, false blits
    //     history_write -> output (svgf_basic).
    //   atrous_passes -- number of A-Trous wavelet passes when
    //     atrous_enabled (1..5, clamped here). The A-Trous structure
    //     keeps the same 5x5 binomial kernel but doubles its tap
    //     stride per pass (1, 2, 4, 8, 16). Effective footprint:
    //     1 = 5x5 (default), 2 = 9x9, 3 = 17x17, 4 = 33x33,
    //     5 = 65x65 (canonical SVGF / Schied 2017). Ignored when
    //     atrous_enabled is false.
    //   albedo_demod_enabled -- issue #119. True keeps the entire
    //     SVGF chain in demodulated lighting space; the remod kernel
    //     (svgf_basic, or one-pass atrous) or the final atrous pass
    //     (multi-pass atrous) multiplies albedo back before writing
    //     to `output`.
    void Encode(MTL::CommandBuffer* cb,
                MTL::Texture*       color_in,
                MTL::Texture*       depth_in,
                MTL::Texture*       motion_in,
                MTL::Texture*       normal_in,
                MTL::Texture*       albedo_in,
                MTL::Texture*       output,
                bool                reset_history,
                bool                atrous_enabled,
                std::uint32_t       atrous_passes,
                bool                albedo_demod_enabled);

    bool Ready() const { return ready_; }

private:
    void DestroyAll();
    void DestroyTextures();

    MetalDevice*                device_       = nullptr;
    bool                        ready_        = false;
    MTL::ComputePipelineState*  temporal_pso_ = nullptr;
    MTL::ComputePipelineState*  atrous_pso_   = nullptr;
    // Issue #119 -- remodulation kernel. Reads the SVGF chain's last
    // demodulated write and the albedo G-buffer, writes the textured
    // radiance into `output`. Only dispatched when albedo_demod is
    // on AND the chain's last writer wasn't already remod-capable
    // (the multi-pass atrous final pass handles it inline).
    MTL::ComputePipelineState*  remod_pso_    = nullptr;

    // Scratch resources (owned). Cross-frame ping-pong:
    //   history_*         RGBA16F texture (rgb = first-A-Trous output, a = sample count)
    //   depth_history_*   R32F texture
    //   normal_history_*  RGBA16F texture
    //   moments_history_* RG32F-packed *buffer* (w*h * float2 elements;
    //                     storage buffers don't count against Metal's
    //                     8-RW-texture compute limit)
    // Within-frame ping-pong:
    //   atrous_a/atrous_b RGBA16F color scratch
    //   variance_a/b      R32F *buffer* (temporal seeds one, A-Trous
    //                     filters through both)
    // Plus 1x1 placeholder textures for slots the active pass declares
    // but does not consume (atrous: slot 1 RGBA16F, slot 3 RG16F).
    MTL::Texture* history_a_           = nullptr;
    MTL::Texture* history_b_           = nullptr;
    MTL::Texture* depth_history_a_     = nullptr;
    MTL::Texture* depth_history_b_     = nullptr;
    MTL::Texture* normal_history_a_    = nullptr;
    MTL::Texture* normal_history_b_    = nullptr;
    MTL::Buffer*  moments_history_a_   = nullptr;
    MTL::Buffer*  moments_history_b_   = nullptr;
    MTL::Texture* atrous_a_            = nullptr;
    MTL::Texture* atrous_b_            = nullptr;
    MTL::Buffer*  variance_a_          = nullptr;
    MTL::Buffer*  variance_b_          = nullptr;
    MTL::Buffer*  dummy_variance_buf_  = nullptr;  // for temporal's unused variance_in slot
    MTL::Texture* dummy_color_         = nullptr;  // 1x1 RGBA16F
    MTL::Texture* dummy_motion_        = nullptr;  // 1x1 RG16F

    std::uint32_t cached_w_       = 0;
    std::uint32_t cached_h_       = 0;
    std::uint32_t frame_parity_   = 0;
    bool          needs_history_clear_ = true;
};

}  // namespace pt::rhi::mtl

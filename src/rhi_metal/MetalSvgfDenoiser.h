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
    //   output -- engine-owned linear-HDR target the tonemap chain
    //     reads next (= post_denoise_hdr).
    //   reset_history -- true clears the temporal accumulation.
    //   atrous_enabled -- true runs the 3-pass spatial filter, false
    //     blits history_write -> output (svgf_basic).
    void Encode(MTL::CommandBuffer* cb,
                MTL::Texture*       color_in,
                MTL::Texture*       depth_in,
                MTL::Texture*       motion_in,
                MTL::Texture*       normal_in,
                MTL::Texture*       output,
                bool                reset_history,
                bool                atrous_enabled);

    bool Ready() const { return ready_; }

private:
    void DestroyAll();
    void DestroyTextures();

    MetalDevice*                device_       = nullptr;
    bool                        ready_        = false;
    MTL::ComputePipelineState*  temporal_pso_ = nullptr;
    MTL::ComputePipelineState*  atrous_pso_   = nullptr;

    // Scratch textures (owned). Cross-frame ping-pong:
    //   history_*        RGBA16F (rgb = first-A-Trous output, a = sample count)
    //   depth_history_*  R32F
    //   normal_history_* RGBA16F
    //   moments_history_* RG32F (mu1, mu2 luminance moments)
    // Within-frame ping-pong:
    //   atrous_a_/atrous_b_  RGBA16F color scratch
    //   variance_a_/variance_b_ R32F variance ping-pong (temporal seeds
    //                           one of them, A-Trous filters through both)
    // Plus 1x1 placeholders for slots the active pass declares but
    // does not consume (atrous: slot 1 RGBA16F, slot 3 RG16F; temporal:
    // slot 10 R32F).
    MTL::Texture* history_a_         = nullptr;
    MTL::Texture* history_b_         = nullptr;
    MTL::Texture* depth_history_a_   = nullptr;
    MTL::Texture* depth_history_b_   = nullptr;
    MTL::Texture* normal_history_a_  = nullptr;
    MTL::Texture* normal_history_b_  = nullptr;
    MTL::Texture* moments_history_a_ = nullptr;
    MTL::Texture* moments_history_b_ = nullptr;
    MTL::Texture* atrous_a_          = nullptr;
    MTL::Texture* atrous_b_          = nullptr;
    MTL::Texture* variance_a_        = nullptr;
    MTL::Texture* variance_b_        = nullptr;
    MTL::Texture* dummy_color_       = nullptr;  // 1x1 RGBA16F
    MTL::Texture* dummy_motion_      = nullptr;  // 1x1 RG16F
    MTL::Texture* dummy_variance_    = nullptr;  // 1x1 R32F (temporal slot 10)

    std::uint32_t cached_w_       = 0;
    std::uint32_t cached_h_       = 0;
    std::uint32_t frame_parity_   = 0;
    bool          needs_history_clear_ = true;
};

}  // namespace pt::rhi::mtl

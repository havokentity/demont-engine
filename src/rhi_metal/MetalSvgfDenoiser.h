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
// reads post_denoise_hdr -> writes swapchain, so MetalSvgfDenoiser
// just leaves its linear-HDR result in `output` (= post_denoise_hdr)
// and the standard bloom + tonemap chain takes it from there. This is
// the only structural difference vs the Vulkan SVGF path, which has
// to run DenoiseFinalize because Vulkan's Tonemap pipeline isn't
// built today.
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

    // Scratch textures (owned). All RGBA16F except depth_history_*
    // (R32F) and the two 1x1 placeholders for the atrous shader's
    // declared-but-unread slots 1 (RGBA16F) and 3 (RG16F).
    MTL::Texture* history_a_       = nullptr;
    MTL::Texture* history_b_       = nullptr;
    MTL::Texture* depth_history_a_ = nullptr;
    MTL::Texture* depth_history_b_ = nullptr;
    MTL::Texture* normal_history_a_ = nullptr;
    MTL::Texture* normal_history_b_ = nullptr;
    MTL::Texture* atrous_a_        = nullptr;
    MTL::Texture* atrous_b_        = nullptr;
    MTL::Texture* dummy_color_     = nullptr;  // 1x1 RGBA16F (atrous slot 1)
    MTL::Texture* dummy_motion_    = nullptr;  // 1x1 RG16F   (atrous slot 3)

    std::uint32_t cached_w_       = 0;
    std::uint32_t cached_h_       = 0;
    std::uint32_t frame_parity_   = 0;
    bool          needs_history_clear_ = true;
};

}  // namespace pt::rhi::mtl

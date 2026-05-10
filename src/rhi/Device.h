// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "CommandBuffer.h"
#include "Handles.h"
#include "Resources.h"
#include "Swapchain.h"
#include "Types.h"

#include <cstddef>
#include <memory>

namespace pt::rhi {

// Opaque platform handle passed to Device::Create.  On macOS this wraps an
// NSWindow* (or its content view); on Windows it'd hold an HWND.
struct NativeWindowHandle {
    void* opaque = nullptr;     // NSWindow* / HWND
    int   width  = 0;
    int   height = 0;
};

class Device {
public:
    virtual ~Device() = default;

    // Factory.  Each backend registers itself by linking its own
    // Create<Backend>Device shim that this dispatches to.
    static std::unique_ptr<Device> Create(BackendType type,
                                          const NativeWindowHandle& window);

    // Resource creation.
    virtual BufferHandle      CreateBuffer(const BufferDesc&)             = 0;
    virtual TextureHandle     CreateTexture(const TextureDesc&)           = 0;
    virtual PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) = 0;
    virtual AccelStructHandle CreateBLAS(const BLASDesc&)                 = 0;
    virtual AccelStructHandle CreateTLAS(const TLASDesc&)                 = 0;

    virtual void DestroyBuffer(BufferHandle)             = 0;
    virtual void DestroyTexture(TextureHandle)           = 0;
    virtual void DestroyPipeline(PipelineHandle)         = 0;
    virtual void DestroyAccelStruct(AccelStructHandle)   = 0;

    // CPU-side write into an Upload buffer.  The backend chooses the most
    // efficient mapping it can.
    virtual void WriteBuffer(BufferHandle, const void* src, std::size_t size,
                             std::size_t dst_offset = 0) = 0;

    // CPU-side write of an entire 2D texture (mip 0). `src_size` is the
    // total bytes of the source: width * height * bytes_per_pixel for
    // the texture's pixel format. Returns false if the texture isn't
    // CPU-writable (private storage) or sizes mismatch. Used for
    // one-shot uploads like env maps.
    virtual bool WriteTexture(TextureHandle, const void* /*src*/,
                              std::size_t /*src_size*/) {
        return false;
    }

    // Per-frame contract: BeginFrame returns the swapchain image to render
    // into, EndFrame submits and presents.
    virtual FrameContext BeginFrame() = 0;
    virtual void         EndFrame(CommandBuffer*) = 0;

    virtual CommandBuffer* AcquireCommandBuffer() = 0;
    virtual void           Submit(CommandBuffer*) = 0;
    virtual void           WaitIdle() = 0;

    // Swapchain control.
    virtual void Resize(int w, int h) = 0;

    // Read a texture back to CPU. The destination buffer must be at
    // least width * height * bytes_per_pixel(format). Returns true on
    // success. Implementations may stall the GPU (waitUntilCompleted)
    // since this is intended for screenshot / debug use only.
    // `out_w` / `out_h` get the texture's dimensions so the caller can
    // size the buffer correctly without a separate query.
    virtual bool ReadbackTexture(TextureHandle, void* /*dst*/, std::size_t /*dst_size*/,
                                 std::uint32_t* /*out_w*/, std::uint32_t* /*out_h*/) {
        return false;
    }

    // Read a buffer back to CPU. The destination must be at least
    // `bytes` long. Returns true on success. Mirrors ReadbackTexture's
    // contract: synchronous, may stall the GPU, intended for screenshot
    // / debug paths only. Used today by the `screenshot` command to
    // pull `exposure_state[0]` so the host-side tonemap applies the
    // same exposure scalar the GPU paths use.
    virtual bool ReadbackBuffer(BufferHandle, void* /*dst*/, std::size_t /*bytes*/) {
        return false;
    }

    // Capability + introspection.
    virtual BackendType  Type()             const = 0;
    virtual bool         SupportsHardwareRT() const = 0;
    virtual const char*  DeviceName()       const = 0;
    virtual std::size_t  CurrentAllocatedBytes() const = 0;

    // P10 denoiser hook. Backends without a denoiser implementation
    // (software / vulkan today) leave the default no-op. The Metal
    // backend implements it via MTLFXTemporalDenoisedScaler. Must be
    // called AFTER the path-tracer Submit so the input textures are
    // ready, and BEFORE EndFrame so it can encode into the same
    // command buffer flow.
    struct DenoiseDesc {
        TextureHandle color_in;       // RGBA16F linear (per-frame, not accumulated)
        TextureHandle depth_in;       // R32F clip-space depth (z/w in [0,1])
        TextureHandle motion_in;      // RG16F pixel-space (prev - curr)
        // World-space surface normals at primary hit (RGBA16F, .xyz =
        // unit normal, .w unused). Required by the Vulkan SVGF/NRD
        // denoiser for edge-aware spatial filtering and by the OptiX
        // AOV denoiser (OptixHdrAov) as the normal guide layer; MetalFX
        // ignores it (handle may be 0 on Metal). Engine writes it from
        // the path tracer's primary-ray pass when denoiser_enabled.
        TextureHandle normal_in;
        // Linear-RGB albedo at primary hit (RGBA16F, .rgb = surface
        // diffuse color, .a unused). Consumed by the OptiX AOV denoiser
        // (OptixHdrAov) as the albedo guide layer; ignored by the SVGF/
        // NRD path and by MetalFX (handle may be 0 in those modes).
        // Engine allocates + writes this only when r_denoiser is
        // optix_hdr_aov.
        TextureHandle albedo_in;
        // Linear-HDR target the denoiser writes to. On Mac/MetalFX this
        // is the post_denoise_hdr intermediate that the Tonemap pipeline
        // reads. On Vulkan with SVGF/NRD it's the same intermediate, but
        // the denoiser also runs its own DenoiseFinalize pass that
        // reads from `output` and writes the tonemapped LDR result into
        // `final_output` (typically the swapchain).
        TextureHandle output;
        // Vulkan SVGF/NRD only: tonemapped-LDR target the denoiser's
        // finalize pass writes (typically the swapchain image). Ignored
        // by MetalFX -- the engine's separate Tonemap dispatch reads
        // `output` and writes the swapchain there. May be 0 on Metal.
        TextureHandle final_output;
        // Vulkan SVGF/NRD only: GPU-side exposure scalar buffer
        // (AutoExposure.slang updates it / engine seeds it). The
        // finalize pass reads exposure_state[0] to apply the same
        // exposure the path tracer's inline tonemap would have used.
        // MetalFX ignores it.
        BufferHandle  exposure_state;
        // Vulkan SVGF/NRD only: r_hdr_pipeline value (1 = path tracer
        // wrote raw linear HDR into color_in, denoiser finalize applies
        // ACES + sRGB; 0 = path tracer already tonemapped into color_in,
        // finalize applies sRGB OETF only). MetalFX ignores it -- the
        // existing Tonemap.slang pipeline keys on the same flag via its
        // own push.
        bool hdr_pipeline = true;
        float jitter_x       = 0.0f;
        float jitter_y       = 0.0f;
        bool  reset_history  = false; // true on backend switch / scene reset
        // SVGF/NRD-only: which spatial-filter quality tier to apply
        // after the temporal accumulation pass.
        //   Basic  = temporal only, then a one-shot vkCmdCopyImage to
        //            the output texture. ~1.5 ms at 1080p on a 5090.
        //            Cleaner under fast motion (no atrous lag), slightly
        //            noisier on disocclusions / undersampled regions.
        //   Atrous = temporal pass + 3 a-trous wavelet passes at step
        //            sizes 1/2/4 with depth+normal+luminance edge stops.
        //            ~5 ms; cleaner on disocclusions, mild softening of
        //            micro-detail.
        // MetalFX path ignores this. Defaults to Atrous so existing
        // callers (and the `r_denoiser svgf` alias) keep their previous
        // behaviour.
        enum class Quality : std::uint8_t { Basic, Atrous };
        Quality quality = Quality::Atrous;

        // Which denoiser implementation the backend should route to.
        // The Vulkan backend looks at this to dispatch between
        // VulkanNrdDenoiser (Svgf -- in-house SVGF/atrous chain) and
        // VulkanOptixDenoiser (OptixHdr / OptixHdrAov -- NVIDIA OptiX
        // via CUDA-Vulkan interop). MetalFX ignores it -- the Metal
        // backend has only one denoiser path.
        //
        // Defaults to Svgf so callers still pinned to the PR #2 API
        // shape (Mac/MetalFX path, existing test fixtures) keep their
        // behaviour. Engine sets this explicitly each frame from the
        // active DenoiserKind cvar.
        // OptixTemporalHdr / OptixTemporalHdrAov: same model family as
        // OptixHdr / OptixHdrAov but with an extra motion-vector flow
        // guide + a single-frame denoised-output history buffer fed
        // back as OptixDenoiserLayer::previousOutput. Maps to
        // OPTIX_DENOISER_MODEL_KIND_TEMPORAL{,_AOV} on the OptiX side.
        // Reuses motion_in (the SVGF/MetalFX motion G-buffer) as the
        // flow input -- the engine already produces it on every
        // denoiser-active frame; this just imports it to CUDA.
        enum class Kind : std::uint8_t {
            Svgf,
            OptixHdr,
            OptixHdrAov,
            OptixTemporalHdr,
            OptixTemporalHdrAov,
        };
        Kind kind = Kind::Svgf;
        // Required by MetalFX TemporalDenoisedScaler. Column-major 4x4
        // (16 floats each). Pass nullptr only if the backend doesn't need
        // them (currently: nothing -- both Metal and any future Vulkan
        // denoiser need motion math).
        const float* world_to_view = nullptr;
        const float* view_to_clip  = nullptr;
    };
    virtual bool SupportsDenoise() const { return false; }
    virtual void Denoise(const DenoiseDesc& /*d*/) {}
};

}  // namespace pt::rhi

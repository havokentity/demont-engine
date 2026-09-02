// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "CommandBuffer.h"
#include "Handles.h"
#include "Resources.h"
#include "Swapchain.h"
#include "Types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace pt::rhi {

// --- How far the CPU may run ahead of the GPU (#295) ----------------------
//
// Every backend must guarantee that when Device::BeginFrame() returns for
// frame F, frame F - kMaxFramesInFlight has FINISHED executing. It is not a
// pacing preference: it is the precondition every CPU-written / GPU-read
// resource ring in this engine is sized against.
//
// The rings write shared (CPU-visible) storage with a plain memcpy --
// MetalDevice::WriteTexture is an MTLTexture replaceRegion, VulkanDevice's
// is a mapped-memory copy -- and neither is ordered against a dispatch that
// is already executing. A ring of depth D makes that safe only because the
// slot the CPU writes at frame F was last READ at frame F - D, so the write
// is safe iff frame F - D has completed. With the write happening in the
// engine's Tick, i.e. BEFORE BeginFrame() for the same frame, the frames
// that can still be executing at the write are F-1 ... F-kMaxFramesInFlight,
// so the requirement is
//
//     D >= kMaxFramesInFlight + 1.
//
// Engine::kOceanUploadRing (3) and PlanetTerrain::kRetireFrames (3) are both
// D = 3, so kMaxFramesInFlight = 2 -- which is the depth VulkanDevice has
// enforced with its fence ring since it was written. MEASURED consequence of
// Metal NOT enforcing it (#295): golden_planet_ocean_snell rendered a
// different image in 2 runs out of 10, and the outlier correlated one-for-one
// with the frame at which the Metal backend had FOUR command buffers still
// in flight -- one more than the ocean ring's three slots can absorb.
//
// Raising this constant requires raising every ring above with it.
inline constexpr int kMaxFramesInFlight = 2;

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

    // Re-point an existing TLAS at a new instance array WITHOUT a device
    // or queue drain (issue #254 P0; the verb #81 and the planetary
    // terrain streaming both block on).
    //
    // Contract:
    //   * `instances` must be non-empty and no longer than the count the
    //     TLAS was created with (its capacity). Growing past that needs
    //     new storage, i.e. a new CreateTLAS + handle; the call returns
    //     false rather than silently truncating.
    //   * When the TLAS carries AccelBuildFlags::AllowUpdate AND the
    //     instance count and the set of referenced BLASes are both
    //     unchanged, the backend takes its refit path (Vulkan
    //     MODE_UPDATE_KHR, Metal refitAccelerationStructure, Embree
    //     dynamic-scene re-commit). Otherwise it does a full rebuild
    //     into the SAME storage. Either way the handle stays valid and
    //     any command buffer that has already been submitted keeps
    //     tracing the topology it was submitted against.
    //   * The call is ordered against subsequent submissions on the
    //     device's own queue: work recorded after this returns observes
    //     the new instances, work already submitted observes the old
    //     ones. It does NOT block on the GPU -- see AccelGpuStallCount().
    //
    // Returns false on an unknown handle, a BLAS-less instance array, a
    // capacity overflow, or a backend that has no acceleration
    // structures at all (the default below).
    virtual bool UpdateTLASInstances(AccelStructHandle,
                                     std::span<const TLASInstance>) {
        return false;
    }

    // Diagnostics counter: how many times an acceleration-structure code
    // path has blocked the CALLING THREAD on the GPU -- a queue/device
    // drain (`vkQueueWaitIdle` / `vkDeviceWaitIdle` / Metal
    // `waitUntilCompleted`) or a targeted wait on one in-flight command
    // buffer. Monotonic for the device's lifetime.
    //
    // This exists so tests can assert the *absence* of a stall rather
    // than infer it from wall-clock timing (which is noise-prone on a
    // shared CI runner): CreateBLAS / CreateTLAS still block once each
    // at startup, UpdateTLASInstances must never move the counter.
    // Backends with no GPU (software/Embree) leave it at 0 forever,
    // which is the truth for them.
    virtual std::uint64_t AccelGpuStallCount() const { return 0; }

    // Blocking waits taken by the frame-pacing bound (#295): how many
    // times BeginFrame() had to hold the CPU because frame
    // F - kMaxFramesInFlight had not finished. Unlike the accel counter
    // above, a NON-zero value here is correct behaviour -- it is the
    // count of times the engine was about to violate the precondition
    // every CPU-written / GPU-read ring is sized against. It is exposed
    // so "is the CPU outrunning the GPU?" is a number rather than a
    // guess. Backends with no GPU timeline leave it at 0.
    virtual std::uint64_t FramePacingStallCount() const { return 0; }

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

    // Read the most-recently-submitted swapchain image back to CPU.
    // The destination must be at least width*height*4 bytes; pixels
    // come back as a tight-packed buffer in `out_format` order.
    //
    // Polling contract (NOT blocking -- the previous documentation
    // here said "block until ready" but the only implementation
    // (Vulkan) cannot block because the caller would deadlock
    // Submit; see VulkanDevice::ReadbackSwapchain's comment):
    //
    //   First call:    backend latches a "capture next submit"
    //                  request. Returns false ("not ready, call again
    //                  next tick").
    //   Later calls:   when the render loop's Submit() has inserted
    //                  the copy + the GPU has finished, returns true
    //                  and memcpys staging -> dst.
    //
    // Callers that issue this from the same thread that drives the
    // render loop (e.g. the engine's `screenshot ... swap` console
    // command, which runs inside Console::Drain on the main thread)
    // must NOT busy-wait; poll across ticks. Backends that don't
    // implement this just return false on every call.
    // SupportsSwapchainReadback() above lets callers fail fast
    // instead of polling indefinitely.
    //
    // Use case: debugging / screenshot tools that need the actual
    // presented pixels for comparison against engine-managed scratch
    // textures (post_denoise_hdr, denoise_color, etc.) -- independent
    // of OS-level compositing, window-position, occlusion.
    enum class SwapFormat : std::uint8_t {
        // Note: "Unorm" here refers to the Vulkan FORMAT only, not
        // the engine's tonemap convention. The engine's compute
        // shaders manually apply the sRGB OETF before storing to a
        // VK_FORMAT_B8G8R8A8_UNORM swap so the OS's framebuffer-
        // display path doesn't re-encode -- so on a typical Vulkan
        // PC build the BYTES come back already sRGB-encoded even
        // though Vulkan calls the format Unorm. Consumers comparing
        // against linear-HDR scratch textures should treat all four
        // variants below as "already display-ready (sRGB-encoded)
        // 8-bit bytes" and only do channel-swap (B<->R) when the
        // variant indicates BGRA.
        Bgra8Unorm,    // 8-bit BGRA; bytes are sRGB-encoded by the engine
        Bgra8Srgb,     // 8-bit BGRA; format-level sRGB encode (storage view)
        Rgba8Unorm,    // 8-bit RGBA; bytes are sRGB-encoded by the engine
        Rgba8Srgb,     // 8-bit RGBA; format-level sRGB encode (storage view)
        Other          // backend-specific; caller treats as opaque BGRA-ish
    };
    virtual bool ReadbackSwapchain(void* /*dst*/, std::size_t /*dst_size*/,
                                   std::uint32_t* /*out_w*/, std::uint32_t* /*out_h*/,
                                   SwapFormat* /*out_format*/) {
        return false;
    }
    // True iff the backend actually implements ReadbackSwapchain.
    // Callers (the engine's screenshot-swap path) check this before
    // issuing a capture so they don't poll-and-timeout on backends
    // (Metal, software, or a Vulkan driver that didn't advertise
    // TRANSFER_SRC on the swap surface) that have no implementation.
    virtual bool SupportsSwapchainReadback() const { return false; }

    // Capability + introspection.
    virtual BackendType  Type()             const = 0;
    virtual bool         SupportsHardwareRT() const = 0;
    virtual const char*  DeviceName()       const = 0;
    virtual std::size_t  CurrentAllocatedBytes() const = 0;

    // True iff a previous Vulkan call returned VK_ERROR_DEVICE_LOST (or
    // an equivalent terminal status on another backend). Once latched
    // the backend's per-frame work is effectively a no-op against a
    // dead device -- the engine polls this at the top of RenderFrame
    // and in smoke-test mode trips smoke_test_failed_ so a GPU hang
    // doesn't silently pass CI with all-zero output. Default false for
    // backends that don't track this yet (Metal / software).
    virtual bool IsDeviceLost() const { return false; }

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
        // MetalFX specular-guidance G-buffers (issue #118). Three
        // textures fed to MTLFXTemporalDenoisedScaler so it can tell
        // specular from diffuse response and eliminate the 8x8 halos
        // it otherwise produces around bright reflections / metals.
        //   specular_albedo_in       -- RGBA16F per-pixel F0 (Fresnel
        //                               reflectance at normal incidence;
        //                               metals: F0 = albedo; dielectrics:
        //                               float3(0.04); Lambert: 0).
        //   roughness_in             -- R32F single-channel surface
        //                               roughness in [0, 1].
        //   specular_hit_distance_in -- R32F distance from camera to
        //                               specularly-reflected hit (MVP:
        //                               primary_t * smoothness proxy;
        //                               see PathTrace.slang's matching
        //                               texture declaration for the
        //                               trade-off vs a real second-trace).
        // Engine allocates them only for DenoiserKind::MetalFX /
        // SvgfBasicMetalFx / SvgfAtrousMetalFx; nil for all other kinds.
        // Apple's MTLFXTemporalDenoisedScaler tolerates a nil binding
        // as "no guidance," so backends consuming the trio can pass
        // them straight through. SVGF / NRD / OptiX paths ignore them
        // (their respective issues will wire matching inputs later;
        // #50 covers NRD).
        TextureHandle specular_albedo_in;
        TextureHandle roughness_in;
        TextureHandle specular_hit_distance_in;
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
        // Vulkan SVGF/NRD only: bloom-pyramid mip 0 (half-res linear
        // HDR). The DenoiseFinalize pass bilinear-samples this and
        // adds it pre-tonemap so highlights get the same ACES squash.
        // Disabled-bloom contract: the engine sets bloom_in = 0 and
        // bloom_intensity = 0; VulkanNrdDenoiser then binds its own
        // color_in_view as a safe-but-unread fallback at descriptor
        // binding 3 (the layout requires a valid view) and forces the
        // push intensity to 0 so the shader's gate short-circuits the
        // sample. No 1x1 placeholder texture is created for this
        // path. MetalFX ignores this -- Metal's bloom is mixed in
        // Tonemap.slang after the denoise call returns (which uses
        // the engine's bloom_dummy_tex_id_ 1x1 texture for *its* own
        // disabled-bloom slot).
        TextureHandle bloom_in;
        // Vulkan SVGF/NRD only: linear blend factor of the bloom layer
        // added on top of the HDR image before tonemap. Mirrors the
        // r_bloom_intensity cvar. 0 = skip the bloom add entirely.
        // MetalFX ignores it (Metal's Tonemap.slang has its own
        // bloom_intensity in TonePush).
        float         bloom_intensity = 0.0f;
        // Vulkan SVGF/NRD only: r_hdr_pipeline value (1 = path tracer
        // wrote raw linear HDR into color_in, denoiser finalize applies
        // the selected tonemap + sRGB; 0 = path tracer already
        // tonemapped into color_in, finalize applies sRGB OETF only).
        // MetalFX ignores it -- the existing Tonemap.slang pipeline
        // keys on the same flag via its own push.
        bool hdr_pipeline = true;
        // Vulkan SVGF/NRD only: r_tonemap_op enum (0 aces / 1 agx /
        // 2 khronos_pbr_neutral / 3 reinhard / 4 linear). The finalize
        // kernel used to hard-code ACES, so on Vulkan every denoiser-on
        // frame ignored the operator the swapchain-off path, Metal, and
        // the PPM/PNG captures all honoured -- screen and screenshot
        // disagreed. MetalFX ignores it (Tonemap.slang reads the same
        // enum from TonePush).
        std::uint32_t tonemap_op = 0u;
        // (stars_in retired with the EMA-based star_split design;
        // celestials now composite Metal-side via shaders/StarsComposite.slang
        // before the bloom pyramid. The Vulkan denoiser path has no
        // stars compositor in this PR -- follow-up.)
        TextureHandle stars_in;
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

        // SVGF-atrous only: number of A-Trous wavelet passes to dispatch
        // after the temporal accumulation. Each pass keeps the same 5x5
        // binomial kernel but doubles the tap stride (1 -> 2 -> 4 -> 8
        // -> 16), so the effective edge-aware footprint doubles per
        // pass at constant tap cost. The engine drives this from the
        // r_svgf_atrous_passes cvar; the backend clamps to 1..5
        // internally (5 = canonical SVGF / Schied 2017). Ignored by
        // SVGF-basic (which skips the spatial chain entirely) and by
        // MetalFX/OptiX.
        std::uint32_t atrous_passes = 1;
        // Issue #119 -- SVGF albedo demodulation. When true, the
        // temporal + atrous chain divides the noisy radiance by the
        // primary-hit albedo on input (operates on incident lighting
        // rather than reflected radiance) and a dedicated remod pass
        // multiplies the result back by albedo before the bloom +
        // tonemap chain consumes it. Driven by `r_svgf_albedo_demod`
        // (CVAR_ARCHIVE, default 1). Requires the albedo G-buffer to
        // be allocated (`albedo_in` non-zero); when the engine didn't
        // allocate it (non-SVGF kinds) the backend silently treats
        // this flag as false. Ignored by MetalFX / OptiX / FinalizeOnly.
        bool albedo_demod_enabled = false;

        // Which denoiser implementation the backend should route to.
        // The Vulkan backend looks at this to dispatch between
        // VulkanNrdDenoiser (Svgf -- in-house SVGF/atrous chain) and
        // VulkanOptixDenoiser (OptixHdr / OptixHdrAov -- NVIDIA OptiX
        // via CUDA-Vulkan interop). The Metal backend looks at it to
        // dispatch between MetalSvgfDenoiser (Svgf -- same Slang
        // shaders the Vulkan path uses, cross-compiled to MSL) and
        // MetalFX (Apple's MTLFXTemporalDenoisedScaler).
        //
        // Defaults to Svgf so callers pinned to the PR #2 API shape
        // (existing test fixtures) keep their behaviour. Engine sets
        // this explicitly each frame from the active DenoiserKind
        // cvar.
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
            MetalFX,
            // SVGF followed by MetalFX TemporalDenoisedScaler as a
            // finalizer. Metal only. SVGF kills path-tracing noise;
            // MetalFX then ML-TAAs the result (cleaner edges than the
            // in-shader edge-aware blend). The backend lazily allocates
            // an intermediate scratch texture sized to the swapchain
            // and routes the SVGF output through it before invoking
            // MetalFX. On Vulkan this falls back to Kind::Svgf (the
            // base denoiser still runs; the MetalFX chain is dropped).
            SvgfMetalFx,
            // Skip every temporal / spatial denoising pass and run JUST
            // the swapchain finalize stage (linear HDR -> exposure ->
            // ACES -> sRGB OETF + bloom composite, written into
            // final_output). Used by the engine's "bloom-without-
            // denoiser" path on backends where the standalone Tonemap.
            // slang compute kernel doesn't reach the swapchain
            // correctly (Vulkan today -- documented in Engine.cpp's
            // use_engine_tonemap comment block). Inputs:
            //   - color_in:        linear-HDR source (engine's
            //                      denoise_color, the path tracer's
            //                      per-frame HDR-aux write)
            //   - final_output:    swapchain image
            //   - bloom_in:        bloom_mip[0], or handle 0 if bloom is off
            //   - bloom_intensity: linear blend factor (matches the
            //                      cvar; 0 == bloom add skipped)
            //   - exposure_state:  GPU-resident exposure scalar buffer
            //   - hdr_pipeline:    true = ACES + sRGB, false = sRGB-only
            // Backends that don't implement this kind (Metal, software)
            // can no-op the call -- the engine's caller checks
            // SupportsDenoise() / current_backend_ before issuing.
            FinalizeOnly,
            // Run the SVGF / NRD temporal+spatial chain (same as Kind::
            // Svgf) but STOP before the trailing DenoiseFinalize dispatch
            // -- i.e. emit the denoised linear-HDR result into `output`
            // and skip bloom composite + ACES + sRGB OETF + swapchain
            // write. Used by the engine's Vulkan StarsComposite path
            // (issue #46): the engine needs an opportunity to dispatch
            // its stateless celestial composite kernel between the
            // a-trous chain output and the swap write. A subsequent
            // Denoise(Kind::FinalizeOnly) call then reads the composited
            // result and produces the final swap. Vulkan-only today
            // (Metal still routes StarsComposite through use_engine_
            // tonemap which sits between SVGF output and Tonemap.slang;
            // Vulkan can't because Tonemap.slang produces a black
            // swapchain there). Inputs are identical to Kind::Svgf
            // except `output` MUST be a caller-owned intermediate (the
            // engine's post_denoise_hdr_tex) -- the swap is written by
            // the follow-up FinalizeOnly call.
            SvgfNoFinalize,
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

    // Predictive pipeline JIT prewarming. Engine signals "I will need
    // pipeline `kernel_name` soon" so the backend can start (or finish)
    // building it before the first Dispatch hits a "pipeline not ready"
    // wait. Idempotent and cheap to call every frame -- once the
    // pipeline is built, subsequent calls are a single hash-map lookup.
    //
    // Trigger points:
    //   * Engine::Init, after EnsurePipelineHandles, for every name the
    //     engine knows about.
    //   * Engine::ApplyBackend, after the new device's
    //     EnsurePipelineHandles fires, to kick the new device's worker.
    //   * Driven by the `r_pipeline_prewarm` cvar (default 1).
    //
    // Default base-class impl is a no-op:
    //   * Metal/software backends build pipelines synchronously inside
    //     the device constructor, so by the time Engine sees a usable
    //     device pointer every pipeline is already ready.
    //   * Backends with an async worker (Vulkan) override this to
    //     enqueue the warm or no-op if the worker is already mid-build
    //     of the requested kernel.
    //
    // Contract:
    //   * Safe to call from the main thread at any time after
    //     Device::Create returned a non-null pointer.
    //   * Unknown / unimplemented names are silently ignored -- a
    //     pipeline that the backend's worker has no SPIR-V/MSL blob
    //     for will simply never become available (and the engine's
    //     dispatch-site id==0 gate keeps it inert). Matches the
    //     "Metal-only today, Vulkan follow-up" pattern Engine uses
    //     for stars_composite / sigma_shadow / restir_*.
    virtual void EnsurePipelineWarmed(const char* /*kernel_name*/) {}

    // --- Per-pass GPU timestamps (#320) -----------------------------------
    //
    // A cheap, toggleable instrument that answers "where does the frame go?"
    // with real GPU time per render pass rather than a single trace_ms. The
    // engine gates it on the perf-overlay cvar (tier 4) and drives it with
    // CommandBuffer::BeginGpuPass / EndGpuPass around each pass; the device
    // owns the backend query objects and the tick->millisecond conversion.
    //
    // Honesty contract: a backend/queue that cannot produce trustworthy
    // timestamps reports SupportsGpuTimestamps() == false and never fills a
    // number in. A timestamp is only meaningful with correct period scaling
    // (Vulkan: VkPhysicalDeviceLimits::timestampPeriod, masked to
    // timestampValidBits; Metal: an MTLDevice sampleTimestamps correlation),
    // so a device that can't establish that scale degrades to "unavailable"
    // rather than printing ticks as if they were milliseconds.

    // True iff this backend + its render queue can timestamp GPU work.
    virtual bool SupportsGpuTimestamps() const { return false; }

    // Nanoseconds represented by one native GPU timestamp tick, for
    // reporting / diagnostics only (the resolve path already returns
    // milliseconds). 0.0 when unsupported.
    virtual double GpuTimestampPeriodNs() const { return 0.0; }

    // Enable or disable per-pass timestamp capture for subsequent frames.
    // `pass_capacity` is the maximum number of passes the engine will time
    // in one frame; the backend sizes its query pool / counter sample
    // buffers for 2 * pass_capacity samples. Cheap no-op when the requested
    // (on, capacity) already matches the current state. Turning it off frees
    // the backend query objects so the instrument is zero-overhead when off.
    virtual void SetGpuTimestampsEnabled(bool /*on*/, std::uint32_t /*pass_capacity*/) {}

    // Resolve the most-recently-completed frame's per-pass timings.
    //   out_ms          -- receives up to `capacity` pass durations in
    //                      MILLISECONDS; out_ms[slot] is the GPU time of the
    //                      pass the engine assigned to that slot. Untimed /
    //                      unopened slots are left at 0.
    //   capacity        -- length of out_ms (the engine's pass_capacity).
    //   out_frame_index -- if non-null, receives the frame index the timings
    //                      belong to (resolve lags the live frame by up to
    //                      kMaxFramesInFlight), so the caller can line the
    //                      durations up with the pass labels it recorded for
    //                      that frame.
    // Returns the number of slots that carried a valid, non-negative timing
    // (i.e. how many passes that frame actually recorded), or 0 when no
    // completed frame's timings are ready or the instrument is disabled.
    virtual std::uint32_t ResolveGpuTimestamps(double* /*out_ms*/,
                                               std::uint32_t /*capacity*/,
                                               std::uint64_t* /*out_frame_index*/ = nullptr) {
        return 0;
    }
};

}  // namespace pt::rhi

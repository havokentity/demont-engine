// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Forward declares for metal-cpp pointer types so this header doesn't drag
// the full Metal SDK into every consumer. The real impl uses MTL:: etc.
namespace MTL {
class Device;
class CommandQueue;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
class Library;
class Texture;
class Buffer;
class AccelerationStructure;
}
namespace CA {
class MetalLayer;
class MetalDrawable;
}
namespace NS {
class AutoreleasePool;
}

namespace pt::rhi::mtl {

class MetalDevice;

// A compiled compute pipeline plus the Metal resource layout Slang
// assigned it, parsed from the emitted MSL at build time (see build_pso).
// This makes the shader's own output the single source of truth for where
// Dispatch binds each resource, rather than the RHI assuming a fixed
// convention (AS@buffer0, push@(max_ssbo+1), and every engine slot == its
// Metal index). Those assumptions only held when the acceleration
// structure was the first-declared buffer resource: PathTrace declares it
// first (AS@0, push@18), but RestirFinal declares it AFTER its SSBOs
// (AS@3, push@4), so the old code bound the AS over the reservoir SSBO and
// the push where the shader read the AS -- a silent clobber.
//
// Fields:
//   - `accel_buf_index` / `push_buf_index` : the buffer() index Slang gave
//     the acceleration structure / the [[vk::push_constant]] block, or -1
//     if the kernel declares none.
//   - `ssbo_buf_mask` : bit i set => buffer(i) is an ordinary `device*`
//     SSBO the kernel actually declares. Dispatch binds an engine buffer
//     slot only if its bit is set, so a bind can never land on the AS/push
//     index (the clobber class) or on a declaration Slang dead-code
//     eliminated (harmless, but now skipped rather than blindly issued).
//   - `tex_mask` : bit i set => texture(i) is declared. Same guard for the
//     texture namespace.
// Buffer/texture indices in these kernels are well under 32, so a uint32
// mask covers every slot (bound_buf_ is 24 wide, bound_tex_ 20).
struct MetalPipeline {
    MTL::ComputePipelineState* pso             = nullptr;
    std::int32_t               accel_buf_index = -1;
    std::int32_t               push_buf_index  = -1;
    std::uint32_t              ssbo_buf_mask    = 0;
    std::uint32_t              tex_mask         = 0;
};

class MetalCommandBuffer : public CommandBuffer {
public:
    explicit MetalCommandBuffer(MetalDevice* d) : device_(d) {}

    void BindComputePipeline(PipelineHandle p) override;
    void BindBuffer(std::uint32_t slot, BufferHandle b, std::size_t off) override;
    void BindStorageTexture(std::uint32_t slot, TextureHandle t) override;
    void BindAccelStruct(std::uint32_t slot, AccelStructHandle a) override;
    void PushConstants(const void* data, std::size_t size) override;
    void Dispatch(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) override;
    void CopyBufferToTexture(BufferHandle, TextureHandle) override {}
    // No-op on Metal: pipelines build synchronously and quickly, so
    // the engine's loading-frame path (which is the only call site)
    // never fires on this backend.
    void ClearStorageTexture(TextureHandle, const float[4]) override {}
    void Barrier(const BarrierDesc&) override {}

    void Reset(MTL::CommandBuffer* cb);
    MTL::CommandBuffer* RawCmdBuf() const { return mtl_cb_; }

    // Forces the currently-open compute encoder closed. Required before
    // any non-RHI code (e.g. MetalFX) wants to attach its own encoder
    // to the same MTLCommandBuffer.
    void FlushEncoder();

private:
    void EnsureEncoder();
    void EndEncoderIfActive();

    MetalDevice*               device_      = nullptr;
    MTL::CommandBuffer*        mtl_cb_      = nullptr;
    MTL::ComputeCommandEncoder* encoder_    = nullptr;

    PipelineHandle             bound_pso_{0};
    // Texture slot map: 14 slots covering every kernel's max engine
    // slot. PathTrace fills slots 0..13:
    //   0  output / swapchain
    //   1  accum_hdr
    //   2  denoise_color
    //   3  depth_tex (denoiser-only)
    //   4  motion_tex (denoiser-only)
    //   5  env_map (HDRI)
    //   6  star_map (BSC)
    //   7  moon_map
    //   8  normal_tex (SVGF / NRD / OptiX-AOV / MetalFX)
    //   9  albedo_tex (OptiX-AOV / MetalFX)
    //   10 cloud_trans_tex (issue #46 follow-up: R32F per-pixel cloud
    //      transmittance the path tracer writes and StarsComposite
    //      reads). On Metal slots above the kernel's actual texture
    //      count are silently dropped; on Vulkan the slot table in
    //      VulkanDevice.cpp maps them to vk::binding numbers.
    //   11 specular_albedo_tex (issue #118: RGBA16F per-pixel F0 for
    //      MetalFX TemporalDenoisedScaler specular guidance)
    //   12 roughness_tex (issue #118: R32F per-pixel roughness)
    //   13 specular_hit_distance_tex (issue #118: R32F distance to
    //      specularly-reflected hit)
    //   14 ocean_displacement (Wave 8 #25: RGBA32F FFT ocean displacement
    //      xyz + foam.w; read-only Texture2D on Metal -- OFF the 8-RW cap)
    //   15 ocean_normal (Wave 8 #25: RGBA32F FFT ocean surface normal;
    //      read-only Texture2D on Metal)
    //   16 pbr_atlas (Wave 8 #26: RGBA8 material-texture strip atlas;
    //      sample-only Texture2D on Metal -- escapes the 8-RW cap the
    //      same way env_map / star_map / moon_map / ocean_* do. Lands at
    //      MSL texture index 16 because ocean (#25) took 14/15 on the
    //      integration branch, so PBR rebased one slot up).
    //   18 godrays_mask (Wave 9: RGBA16F scratch the GodRays pass writes
    //      its occlusion/light mask into in pass 0 and reads back in
    //      pass 1; vk::binding 37). Slot 17 is intentionally left as a
    //      gap so the engine slot number matches the documented Wave 9
    //      reservation (slot 18 / binding 37) rather than packing tight.
    //   On Metal slots above the kernel's actual texture count are
    //   silently dropped; on Vulkan the slot table in
    //   VulkanDevice.cpp maps them to vk::binding numbers. Array sized 20
    //   to fit ocean at 14/15 + pbr_atlas at 16 + godrays_mask at 18,
    //   plus one slot of headroom; bump again before adding more.
    TextureHandle              bound_tex_[20] {};
    // Buffer slots. Slots 0..7 are the original engine layout
    // (mesh_positions / mesh_indices, primitives, marginal /
    // conditional CDFs, exposure_state, analytic-prim bvh_nodes).
    // Slots 8/9 were added in the PR #106 follow-up for the triangle
    // BVH (tri_bvh_nodes / tri_bvh_permuted_ids -- replaces the SW
    // Möller-Trumbore linear-scan path with a stack-based BVH walk).
    // Slot 10 was added by SDF Phase 1 (#97) for the SDF cluster
    // storage buffer (MSL slot 10 in the path tracer's dynamic
    // buffer-slot layout; moved from slot 8 to make room for the
    // triangle BVH). Slot 11 was added by issue #115 for the SIGMA
    // shadow visibility buffer (R32F per pixel) -- declared as a
    // storage buffer rather than an RWTexture2D because PathTrace
    // already sits exactly at Apple Silicon's 8-RW-texture compute
    // cap; storage buffers escape that quota the same way SVGF's
    // variance / moments buffers do.
    //
    // Slot 12 was added by light primitives (#73) for the analytic
    // light list (`light_prims`, MSL slot 12; the shader declares it
    // at vk::binding(27)). Declared AFTER shadow_vis_buf so the
    // existing slot 11 stays put. Engine binds it via
    // BindBuffer(12, ...).
    //
    // Slot 13 added by the hierarchical light tree (#129) for the
    // packed-node SSBO consumed by PathTrace.slang's O(log N) NEE
    // picker. MSL slot 13 (declared AFTER light_prims so the existing
    // slots 0..12 stay put); the shader's matching vk::binding is 28.
    //
    // Issue #78 (ReSTIR DI Phase A) extends to slot 14 for the
    // per-pixel reservoir SSBO (`reservoir_curr_buf`, MSL slot 14;
    // vk::binding(29)). Declared AFTER light_tree.
    //
    // Issue #136 (smoke emitters) extends to slot 15 for the smoke
    // emitter list SSBO (`smoke_emitters`, MSL slot 15;
    // vk::binding(30)). Declared AFTER reservoir.
    //
    // Issue #22 (Fluid Phase 3 SPH smoke fluid sim) extends to slot
    // 16 for the SPH particle splat list SSBO (`sph_particles`, MSL
    // slot 16; vk::binding(31)). Declared AFTER smoke_emitters.
    // Array sized 24 to fit all six SSBOs plus a handful of slots of
    // headroom; bump again before adding more.
    BufferHandle               bound_buf_[24] {};
    std::size_t                bound_buf_off_[24] {};
    AccelStructHandle          bound_accel_[4] {};

    // Push-constant buffer. Sized to fit the unified PathTrace push
    // plus growth headroom for upcoming features (DOF, MIS PDFs,
    // bloom params, ...). PathTrace push is currently ~1056 bytes after
    // Wave 7 (#24) added the 16-byte `clouds_runtime` block at the tail
    // and Wave 7 (#21) added 32 bytes for mesh_motion_prev +
    // mesh_motion_curr. Earlier this had to be bumped from 1024 to 2048
    // because the truncation silently zeroed the last bytes and produced
    // a black-frame regression on cloud-bearing scenes (#24) plus a
    // mesh-motion-blur-silently-failing regression (#21). 2048 leaves
    // room for several more 16-byte blocks before another resize.
    // The size is now pt::rhi::kMaxPushConstantBytes, shared with the
    // Vulkan backend and static_asserted against sizeof(PtPush) in
    // Engine.cpp -- because "remember to grow this in lockstep" is exactly
    // what nobody did for #21, #24 and #300, each of which shipped a
    // silently truncated tail (mesh motion blur failing, clouds black,
    // terrain ignoring its albedo raster). PushConstants() now also logs
    // when it has to truncate, so a build that somehow gets past the
    // static_assert still cannot fail quietly.
    std::uint8_t  push_buf_[pt::rhi::kMaxPushConstantBytes] {};
    std::size_t   push_size_ = 0;
};

// Everything the backend retains for one acceleration structure
// (issue #254 P0). Before the update verb existed this was a bare
// `MTL::AccelerationStructure*`; a TLAS that can be refit has to keep
// its build inputs alive instead of destroying them the instant the
// build was drained.
struct MetalAccel {
    MTL::AccelerationStructure* as = nullptr;
    // Byte size the structure was allocated with. An in-place rebuild
    // for a smaller instance count must still fit inside it, so the
    // update path checks the recomputed size against this rather than
    // assuming monotonicity in the instance count.
    std::size_t     as_size = 0;
    bool            is_tlas = false;
    AccelBuildFlags flags   = AccelBuildFlags::PreferFastTrace;

    // --- TLAS only -----------------------------------------------------
    // Scratch retained for the lifetime of the structure, but ONLY for
    // AllowUpdate structures: a one-shot build still frees its scratch
    // the moment the build lands, so the static CSG mesh's memory
    // profile is unchanged. Sized to max(build, refit) scratch and
    // grown in place if a later update needs more.
    MTL::Buffer*    scratch      = nullptr;
    std::size_t     scratch_size = 0;

    // Ring of host-writable instance-descriptor buffers. The build /
    // refit reads one of these on the GPU, so the CPU must not
    // overwrite the slot a still-running command buffer is reading --
    // which is precisely what the old `waitUntilCompleted` bought.
    // A ring of kAccelUpdateRing rotates the write away from anything
    // in flight; `ring_cb[i]` is the last command buffer that read
    // slot i, retained so its completion status is queryable.
    static constexpr std::uint32_t kRing = 3;
    MTL::Buffer*        ring[kRing]    {};
    MTL::CommandBuffer* ring_cb[kRing] {};
    std::uint32_t       ring_cursor    = 0;
    std::size_t         ring_capacity_bytes = 0;

    // Instance count at create time -- the TLAS's capacity. Updates
    // accept [1, capacity].
    std::uint32_t instance_capacity = 0;
    // The BLAS handles the current instance array points at, in order.
    // Used to decide refit (unchanged) vs in-place rebuild (changed),
    // and to rebuild the MTLAccelerationStructure array a Metal
    // instance descriptor indexes into.
    std::vector<std::uint64_t> instance_blas;
};

class MetalDevice : public Device {
public:
    explicit MetalDevice(const NativeWindowHandle& window);
    ~MetalDevice() override;

    // ---- Resources ------------------------------------------------------
    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override;
    AccelStructHandle CreateTLAS(const TLASDesc&) override;
    bool UpdateTLASInstances(AccelStructHandle h,
                             std::span<const TLASInstance> instances) override;
    std::uint64_t AccelGpuStallCount() const override {
        return accel_gpu_stalls_.load(std::memory_order_relaxed);
    }
    std::uint64_t FramePacingStallCount() const override {
        return frame_gpu_stalls_.load(std::memory_order_relaxed);
    }

    void DestroyBuffer(BufferHandle h) override;
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle h) override;
    void DestroyAccelStruct(AccelStructHandle h) override;

    void WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                     std::size_t dst_offset = 0) override;
    bool WriteTexture(TextureHandle h, const void* src, std::size_t src_size) override;

    // ---- Frame ----------------------------------------------------------
    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override;
    void           Resize(int w, int h) override;

    // ---- Introspection --------------------------------------------------
    BackendType  Type()             const override { return BackendType::Metal; }
    // Real capability, not an assumption. See the constructor: a Mac
    // whose GPU cannot build acceleration structures (paravirtualised
    // CI runner, VM, pre-RT discrete GPU) used to be told "yes" here
    // and then aborted inside the first AS encode.
    bool         SupportsHardwareRT() const override { return rt_supported_; }
    const char*  DeviceName()       const override { return device_name_.c_str(); }
    std::size_t  CurrentAllocatedBytes() const override;

    // ---- P10 denoiser ---------------------------------------------------
    bool SupportsDenoise() const override { return true; }
    void Denoise(const DenoiseDesc& d) override;

    bool ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                         std::uint32_t* out_w, std::uint32_t* out_h) override;
    bool ReadbackBuffer (BufferHandle  h, void* dst, std::size_t bytes) override;

    // ---- Internal lookup ------------------------------------------------
    // Returns the pipeline plus its parsed binding indices, or nullptr if
    // the handle is unknown. Dispatch needs the indices, not just the PSO.
    const MetalPipeline*       LookupPipeline(PipelineHandle h);
    MTL::Texture*              LookupTexture(TextureHandle h);
    MTL::Buffer*               LookupBuffer(BufferHandle h);
    MTL::AccelerationStructure* LookupAccelStruct(AccelStructHandle h);

    // Used by MetalSvgfDenoiser to build its own pipelines + textures.
    MTL::Device* RawDevice() const { return device_; }

    // Mark all known acceleration structures as used by the current
    // encoder. Required because a TLAS internally references its BLASes
    // and Metal needs every one declared as a dependency on the encoder
    // that ray-queries the TLAS.
    void UseAllAccelStructs(MTL::ComputeCommandEncoder* enc);

    static constexpr std::uint64_t kSwapchainTextureId = 1;

    // Depth of the frame-pacing ring, i.e. how many frames of GPU work the
    // CPU is allowed to have outstanding. Named separately from
    // pt::rhi::kMaxFramesInFlight only so the array declaration reads as a
    // ring size; they are the same number by construction.
    static constexpr int kFramePacingRing = pt::rhi::kMaxFramesInFlight;
    // frame_index_ is a uint32 that is allowed to wrap, and the slot is
    // frame_index_ % kFramePacingRing. That sequence is only continuous
    // across the wrap if the ring divides 2^32 -- at a ring of 3, 2^32 % 3
    // == 1 and the wrap would skip a slot, waiting on a frame one older
    // than intended exactly once. A power of two makes it a non-question.
    static_assert(kFramePacingRing > 0 &&
                      (kFramePacingRing & (kFramePacingRing - 1)) == 0,
                  "frame-pacing ring must be a power of two so the slot "
                  "sequence stays continuous across the frame-index wrap");

private:
    void* ns_window_ = nullptr;     // NSWindow*
    int   width_  = 0;
    int   height_ = 0;
    std::uint32_t frame_index_ = 0;
    std::string device_name_ = "Metal Device";
    // MTLDevice.supportsRaytracing, latched at construction. Gates every
    // acceleration-structure entry point; false means CreateBLAS /
    // CreateTLAS / UpdateTLASInstances refuse cleanly instead of
    // encoding against a stub AS object and aborting.
    bool rt_supported_ = false;

    MTL::Device*        device_  = nullptr;
    MTL::CommandQueue*  queue_   = nullptr;
    CA::MetalLayer*     layer_   = nullptr;
    CA::MetalDrawable*  current_drawable_ = nullptr;
    NS::AutoreleasePool* frame_pool_ = nullptr;

    std::mutex                                         resource_mutex_;
    std::uint64_t                                      next_id_ = kSwapchainTextureId + 1;
    std::unordered_map<std::uint64_t, MetalPipeline>              pipelines_;
    std::unordered_map<std::uint64_t, MTL::Texture*>               textures_;
    std::unordered_map<std::uint64_t, MTL::Buffer*>                buffers_;
    std::unordered_map<std::uint64_t, MetalAccel>                  accels_;

    // Count of blocking GPU waits taken on an acceleration-structure
    // path (see Device::AccelGpuStallCount). CreateBLAS / CreateTLAS
    // each take one; UpdateTLASInstances is required never to.
    std::atomic<std::uint64_t> accel_gpu_stalls_{0};

    // --- Frame pacing (#295) ---------------------------------------------
    //
    // Metal has no fence object and Submit() does not wait, so before this
    // ring there was NOTHING bounding how far the CPU could run ahead of the
    // GPU: nextDrawable() throttles a windowed session at the swapchain's
    // depth, but a headless capture (pt_render_one_frame, the whole golden
    // matrix) has no layer, takes no drawable, and therefore had no throttle
    // at all. VulkanDevice bounds it with vkWaitForFences on a
    // kFramesInFlight-deep fence ring; this is the same ring with the last
    // command buffer of each frame standing in for the fence, and it exists
    // for the reason spelled out at pt::rhi::kMaxFramesInFlight.
    //
    // frame_cb_[F % kMaxFramesInFlight] holds the last command buffer
    // committed during frame F. BeginFrame() for frame F therefore finds
    // frame F - kMaxFramesInFlight's command buffer in that slot and waits
    // on it -- exactly the frame whose fence VulkanDevice::BeginFrame waits
    // on. Command buffers on one MTLCommandQueue complete in commit order,
    // so waiting on a frame's LAST one implies all of its earlier ones.
    MTL::CommandBuffer* frame_cb_[kFramePacingRing] {};
    // Blocking waits actually taken on that ring. Zero on a GPU-bound run
    // (the frame is long finished); non-zero means the CPU outran the GPU
    // and was held, which is the case this ring exists to make safe.
    std::atomic<std::uint64_t> frame_gpu_stalls_{0};
    // Built-in pipelines indexed by Slang kernel name.  P3+ shaders are
    // pre-compiled at device construction; CreateComputePipeline looks up
    // by name.
    std::unordered_map<std::string, std::uint64_t>     named_pipelines_;

    std::unique_ptr<MetalCommandBuffer> cmd_;

    // MetalFX TemporalDenoisedScaler. Allocated lazily on first
    // Denoise() call; recreated when the swapchain size changes.
    void*         metalfx_scaler_   = nullptr;
    std::uint32_t metalfx_width_    = 0;
    std::uint32_t metalfx_height_   = 0;

    // In-house SVGF (svgf_basic / svgf_atrous). Same Slang sources as
    // the Vulkan backend, cross-compiled to MSL. Allocated lazily on
    // first SVGF-mode Denoise() call.
    std::unique_ptr<class MetalSvgfDenoiser> svgf_denoiser_;

    // SVGF -> MetalFX chain (Kind::SvgfMetalFx): intermediate RGBA16F
    // texture SVGF writes to before MetalFX reads it as the "color"
    // input. Lazily allocated on first chain-mode Denoise() and
    // resized when the swapchain dims change.
    MTL::Texture* svgf_metalfx_intermediate_   = nullptr;
    std::uint32_t svgf_metalfx_intermediate_w_ = 0;
    std::uint32_t svgf_metalfx_intermediate_h_ = 0;
};

}  // namespace pt::rhi::mtl

namespace pt::rhi {
std::unique_ptr<Device> CreateMetalDevice(const NativeWindowHandle&);
}

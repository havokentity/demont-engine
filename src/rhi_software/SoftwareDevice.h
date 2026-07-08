// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Embree's C interface. The kernel uses `RTCDevice` / `RTCScene` / etc.
// directly; we treat each engine BLAS as one RTCScene with a single
// triangle geometry attached, and each engine TLAS as a top-level
// RTCScene that holds instance geometries pointing at the BLAS scenes.
struct RTCDeviceTy;
typedef RTCDeviceTy* RTCDevice;
struct RTCSceneTy;
typedef RTCSceneTy* RTCScene;

// Forward decls for metal-cpp pointer types.  Real defs come from
// <Metal/Metal.hpp> in the .cpp.  Apple-only -- the Windows present
// path uses GDI (see PresentOutput / EndFrame in SoftwareDevice.cpp).
#if defined(__APPLE__)
namespace MTL {
class Device;
class CommandQueue;
class Texture;
}
namespace CA { class MetalLayer; class MetalDrawable; }
#endif

namespace pt::rhi::sw {

// Pipeline metadata for the Software backend. The only thing the
// dispatcher cares about is the pipeline NAME -- different kernels
// (clear / pathtrace / tonemap / etc.) get routed to different CPU
// code paths by name match.
class SoftwarePipeline {
public:
    explicit SoftwarePipeline(std::string name);
    void SetClearColor(float r, float g, float b, float a);
    void GetClearColor(float& r, float& g, float& b, float& a) const;
    const std::string& Name() const { return name_; }
private:
    std::string name_;
    float clear_[4] { 0, 0, 0, 1 };
};

// CPU-side backing for an RHI buffer. The engine WriteBuffer's into
// `data`; the path tracer kernel reads from it via reinterpret_cast
// to the appropriate element type (float / uint32_t / packed struct).
struct BackedBuffer {
    std::vector<std::uint8_t> data;
    std::size_t               size = 0;
    std::string               debug_name;
};

// CPU-side backing for an RHI texture. We store every texture as a
// flat row-major RGBA32F buffer regardless of the GPU-side format the
// engine asked for -- the Software path tracer reads/writes
// `glm::vec4` per texel, and the present blit converts to RGBA8 at
// upload time. Storing RGBA32F internally lets us match Metal/Vulkan's
// HDR accum behaviour without juggling format conversions per binding.
struct BackedTexture {
    std::vector<float> data;       // width * height * 4 floats
    std::uint32_t      width  = 0;
    std::uint32_t      height = 0;
    std::string        debug_name;
};

// CPU-side backing for an acceleration structure. Both BLAS and TLAS
// use the same struct; the TLAS variant holds a non-empty instance
// list so the kernel can transform rays. For a BLAS, `scene` is the
// Embree scene with one triangle geometry attached.
struct BackedAccel {
    RTCScene    scene   = nullptr;     // BLAS or TLAS Embree scene
    bool        is_tlas = false;
    // BLAS-only: retain the geometry buffers so Embree's shared-buffer
    // refs stay valid for the lifetime of the scene. The engine builds
    // BLAS from raw pointers (BLASDesc::vertex_positions / indices)
    // that the engine owns; we copy into our own storage so the
    // engine is free to drop / re-upload its CPU-side vbuf/ibuf
    // without invalidating the Embree BVH.
    std::vector<float>         vertices;   // tightly packed float3 per vertex
    std::vector<std::uint32_t> indices;
    std::uint32_t              instance_id = 0;
};

class SoftwareDevice;
#if defined(_WIN32)
class SoftwareVulkanPresent;  // defined in SoftwareVulkanPresent.h
#endif

// Per-slot bind state, captured by BindBuffer / BindStorageTexture /
// BindAccelStruct between Dispatch calls so the CPU kernel can look
// up the resources by slot index at dispatch time. The slot numbers
// match the engine-side BindBuffer/BindStorageTexture/BindAccelStruct
// calls in Engine.cpp, which are MSL-style indices on Metal and
// Vulkan-binding-mapped indices on Vulkan; we just need them stable.
struct BindState {
    // 24 slots: bumped from 16 to fit Fluid Phase 3 (#22) SPH
    // particle SSBO at engine slot 16. Slot count must match the
    // Metal / Vulkan equivalents (bound_tex_[20] there -- textures
    // was left at 16 when buffers was bumped, silently dropping
    // BindStorageTexture for engine slots 16 (pbr_atlas) and 18
    // (godrays_mask); harmless while the CPU kernel ignores those
    // slots, but the exact 'array too small, feature reads zeros'
    // class the Vulkan ocean-slot regression documented).
    std::uint64_t buffers[24]      {};   // BufferHandle.id by slot
    std::uint64_t textures[20]     {};   // TextureHandle.id by slot
    std::uint64_t accel_structs[4] {};   // AccelStructHandle.id by accel-slot
};

class SoftwareCommandBuffer : public CommandBuffer {
public:
    explicit SoftwareCommandBuffer(SoftwareDevice* d) : device_(d) {}

    void BindComputePipeline(PipelineHandle p) override;
    void BindBuffer(std::uint32_t slot, BufferHandle h, std::size_t /*off*/) override;
    void BindStorageTexture(std::uint32_t slot, TextureHandle h) override;
    void BindAccelStruct(std::uint32_t slot, AccelStructHandle h) override;
    void PushConstants(const void* data, std::size_t size) override;
    void Dispatch(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) override;
    void CopyBufferToTexture(BufferHandle, TextureHandle) override {}
    void ClearStorageTexture(TextureHandle, const float[4]) override {}
    void Barrier(const BarrierDesc&) override {}

    PipelineHandle bound_pipeline{0};
    BindState      binds {};
    // PtPush is ~1040B today (wave-7 #21 added mesh_motion_prev +
    // mesh_motion_curr two float4s). PushConstants() silently truncates
    // beyond this size, so when adding push fields make sure the
    // buffer stays big enough -- a too-small buffer manifests as the
    // last fields reading as zero / garbage at runtime. Mirrors the
    // Metal-side push_buf_ sizing rationale in MetalDevice.h.
    std::uint8_t   push_constants_buf[2048] {};
    std::size_t    push_constants_size = 0;

private:
    SoftwareDevice* device_ = nullptr;
};

class SoftwareDevice : public Device {
public:
    explicit SoftwareDevice(const NativeWindowHandle& window);
    ~SoftwareDevice() override;

    BufferHandle      CreateBuffer(const BufferDesc&) override;
    TextureHandle     CreateTexture(const TextureDesc&) override;
    PipelineHandle    CreateComputePipeline(const ComputePipelineDesc&) override;
    AccelStructHandle CreateBLAS(const BLASDesc&) override;
    AccelStructHandle CreateTLAS(const TLASDesc&) override;

    void DestroyBuffer(BufferHandle h) override;
    void DestroyTexture(TextureHandle h) override;
    void DestroyPipeline(PipelineHandle h) override;
    void DestroyAccelStruct(AccelStructHandle h) override;

    void WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                     std::size_t /*offset*/) override;
    bool WriteTexture(TextureHandle h, const void* src, std::size_t size) override;
    bool ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                          std::uint32_t* out_w, std::uint32_t* out_h) override;

    FrameContext   BeginFrame() override;
    void           EndFrame(CommandBuffer*) override;
    CommandBuffer* AcquireCommandBuffer() override;
    void           Submit(CommandBuffer*) override;
    void           WaitIdle() override {}
    void           Resize(int w, int h) override;

    BackendType  Type()             const override { return BackendType::Software; }
    // Embree gives us a software-emulated BVH + triangle intersection
    // pipeline (CreateBLAS / CreateTLAS / rtcIntersect1) that is
    // semantically equivalent to "hardware ray tracing" from the
    // engine's perspective. Returning true here unlocks
    // Engine::EnsureMeshUpdated()'s CSG-bake + RebuildMeshResources
    // path -- without it the engine treats this backend as RT-less
    // and never builds a TLAS, so the mesh stays invisible no matter
    // what the kernel does. The name is a slight misnomer for a CPU
    // backend, but the contract it gates on is "can the device build
    // an acceleration structure and trace rays against it" which we
    // satisfy via Embree.
    bool         SupportsHardwareRT() const override { return true; }
    const char*  DeviceName()       const override { return "DeMonT Software (CPU + Embree)"; }
    std::size_t  CurrentAllocatedBytes() const override;

    // Stash the present clear colour for the case where the kernel
    // hasn't run yet (the legacy "clear" pipeline path). Once the path
    // tracer runs, the output texture's backing supersedes this and
    // EndFrame uploads it to the Metal drawable instead.
    void StashClear(float r, float g, float b, float a);

    // CPU port of shaders/EditorOverlay.slang. Projects the engine-
    // uploaded gizmo line list (engine buffer slot 1) onto the
    // slot-0 output texture, anti-aliasing thin lines via per-pixel
    // distance-to-segment. Push constants follow the same layout as
    // the Slang push (camera basis + segment count); see
    // SoftwareDevice.cpp's implementation for the parsing detail.
    void RunEditorOverlay(SoftwareCommandBuffer& cb);

    // Internals used by SoftwareCommandBuffer + SoftwareTracer.
    SoftwarePipeline* GetPipeline(PipelineHandle h);
    BackedBuffer*     GetBuffer(BufferHandle h);
    BackedTexture*    GetTexture(TextureHandle h);
    BackedAccel*      GetAccel(AccelStructHandle h);
    RTCDevice         EmbreeDevice() const { return embree_device_; }
    int               Width()  const { return width_; }
    int               Height() const { return height_; }

private:
    void EnsureSwapchainOutput();   // allocate the slot-0 output texture sized to width/height
    void PresentOutput();           // upload output texture to Metal drawable and present

    // Native window handle.  NSWindow* on Mac (via pt_window_native_cocoa
    // -> CAMetalLayer attach), HWND on Windows (via pt_window_native_win32
    // -> GDI SetDIBitsToDevice).  Stored as void* so this header doesn't
    // pull in <Cocoa.h> / <Windows.h>.
    void* native_window_ = nullptr;
    int   width_  = 0;
    int   height_ = 0;
    std::uint32_t frame_index_ = 0;

    float pending_clear_[4] { 0.18f, 0.05f, 0.28f, 1.0f };

#if defined(__APPLE__)
    // Metal-side: only used for the present blit. The CPU kernel does
    // all the real work into BackedTexture::data above and we upload
    // that data into a transient MTLTexture at EndFrame time.
    MTL::Device*       mtl_device_  = nullptr;
    MTL::CommandQueue* mtl_queue_   = nullptr;
    CA::MetalLayer*    mtl_layer_   = nullptr;
    MTL::Texture*      present_tex_ = nullptr;   // RGBA8Unorm, sized to swapchain
    std::uint32_t      present_w_   = 0;
    std::uint32_t      present_h_   = 0;
#endif

    // Cross-platform scratch buffer reused across frames for the
    // RGBA32F -> BGRA8 pack step in PresentOutput().  Allocating a
    // fresh vector every frame at 1080p / 4K was several MB of heap
    // churn on top of the CPU tracer; this stays sized to the current
    // swapchain and only re-allocates on resize.  Mac uploads it into
    // a transient MTLTexture; Windows passes it directly to
    // SetDIBitsToDevice.
    std::vector<std::uint32_t> present_scratch_;
    std::uint32_t              present_scratch_w_ = 0;
    std::uint32_t              present_scratch_h_ = 0;

    std::unique_ptr<SoftwareCommandBuffer> cmd_buf_;

#if defined(_WIN32)
    // Win32-only: Vulkan-blit present path. Owned when r_software_blit
    // selects "vulkan" (the default) and Init succeeded; nullptr when
    // the user opted into "gdi" mode or Vulkan-blit init failed (in
    // which case EndFrame falls back to the existing GDI path).
    // SoftwareVulkanPresent is forward-declared in the same namespace
    // (pt::rhi::sw) at the bottom of SoftwareDevice.h's namespace
    // bracket below, so the public header doesn't pull in
    // <vulkan/vulkan.h>.
    std::unique_ptr<SoftwareVulkanPresent> vk_present_;
#endif

    // Embree state. One device shared across all BLAS/TLAS; each scene
    // is owned by the corresponding BackedAccel entry.
    RTCDevice embree_device_ = nullptr;

    std::mutex resource_mutex_;
    std::uint64_t next_id_ = 1;
    std::unordered_map<std::uint64_t, std::unique_ptr<SoftwarePipeline>> pipelines_;
    std::unordered_map<std::uint64_t, std::unique_ptr<BackedBuffer>>     buffers_;
    std::unordered_map<std::uint64_t, std::unique_ptr<BackedTexture>>    textures_;
    std::unordered_map<std::uint64_t, std::unique_ptr<BackedAccel>>      accels_;

    // The slot-0 output texture's id; allocated lazily on first
    // BindStorageTexture(0, ...) or first PathTrace dispatch and
    // resized on swapchain resize.
    std::uint64_t output_tex_id_ = 0;

    std::atomic<std::size_t> bytes_held_{0};
};

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w);
}

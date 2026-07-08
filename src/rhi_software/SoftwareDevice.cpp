// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Software backend: CPU compute via Embree (triangle BVH) + analytic-
// primitive intersection done inline. The path tracer writes a CPU
// framebuffer; the present path is platform-specific:
//   * Mac: upload to a transient MTLTexture, blit to the drawable, and
//     present via CAMetalLayer.
//   * Windows: SetDIBitsToDevice from the same packed BGRA8 scratch
//     buffer directly to the HWND's device context.  This is a DIB-
//     to-DC blit, not a `BitBlt` -- BitBlt copies between two DCs and
//     isn't what we're doing here.  Grep for `SetDIBitsToDevice` to
//     find the call site.
// No Slang shader is compiled to CPU for this target; the kernel is a
// hand-written C++ port covering the subset documented in
// SoftwareTracer.h.
//
// Known limitation (Windows GDI path): not DPI-aware.  The engine makes
// no SetProcessDpiAwarenessContext call, so on a high-DPI display
// Windows reports virtualised pixels from GetClientRect while
// SetDIBitsToDevice writes physical pixels; the DWM then bilinearly
// stretches our framebuffer to the real display size, softening the
// path-traced output.  Acceptable for a CPU reference renderer.  The
// Vulkan-blit present alternative (r_software_blit=vulkan, the default
// on Win32 since PR #42) sidesteps the GDI path entirely.

#include "SoftwareDevice.h"
#include "SoftwareTracer.h"
#if defined(_WIN32)
#  include "SoftwareVulkanPresent.h"
#  include "../console/Console.h"
#endif

#include "../core/Log.h"
#include "../core/Memory/MemTag.h"
#include "../core/Memory/Memory.h"

#if defined(__APPLE__)
// Headers only -- the metal-cpp PRIVATE_IMPLEMENTATION TU lives in
// rhi_metal/MetalDevice.cpp and supplies the impl symbols we need.
#  include <Foundation/Foundation.hpp>
#  include <Metal/Metal.hpp>
#  include <QuartzCore/QuartzCore.hpp>
#elif defined(_WIN32)
// We pull in <windows.h> for GDI (SetDIBitsToDevice, BITMAPINFO,
// GetClientRect, GetDC/ReleaseDC) and HWND.  NOMINMAX prevents
// <windows.h>'s min/max macros from colliding with std::min/std::max
// used elsewhere in this TU.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

#include <embree4/rtcore.h>

#include <algorithm>
#include <cstring>
#include <cmath>

#if defined(__APPLE__)
extern "C" void  pt_metal_attach_layer(void* ns_window, void* metal_layer);
extern "C" void* pt_window_native_cocoa(void* glfw_window);
#elif defined(_WIN32)
extern "C" void* pt_window_native_win32(void* glfw_window);
#endif

namespace pt::rhi::sw {

// ---------- SoftwarePipeline ------------------------------------------------

SoftwarePipeline::SoftwarePipeline(std::string name) : name_(std::move(name)) {}

void SoftwarePipeline::SetClearColor(float r, float g, float b, float a) {
    clear_[0] = r; clear_[1] = g; clear_[2] = b; clear_[3] = a;
}
void SoftwarePipeline::GetClearColor(float& r, float& g, float& b, float& a) const {
    r = clear_[0]; g = clear_[1]; b = clear_[2]; a = clear_[3];
}

// ---------- SoftwareCommandBuffer -----------------------------------------

void SoftwareCommandBuffer::BindComputePipeline(PipelineHandle p) {
    bound_pipeline = p;
}

void SoftwareCommandBuffer::BindBuffer(std::uint32_t slot, BufferHandle h, std::size_t) {
    if (slot < std::size(binds.buffers)) binds.buffers[slot] = h.id;
}
void SoftwareCommandBuffer::BindStorageTexture(std::uint32_t slot, TextureHandle h) {
    if (slot < std::size(binds.textures)) binds.textures[slot] = h.id;
}
void SoftwareCommandBuffer::BindAccelStruct(std::uint32_t slot, AccelStructHandle h) {
    if (slot < std::size(binds.accel_structs)) binds.accel_structs[slot] = h.id;
}

void SoftwareCommandBuffer::PushConstants(const void* data, std::size_t size) {
    if (size > sizeof(push_constants_buf)) size = sizeof(push_constants_buf);
    std::memcpy(push_constants_buf, data, size);
    push_constants_size = size;
}

void SoftwareCommandBuffer::Dispatch(std::uint32_t /*gx*/, std::uint32_t /*gy*/, std::uint32_t /*gz*/) {
    auto* pl = device_->GetPipeline(bound_pipeline);
    if (pl == nullptr) return;
    const std::string& name = pl->Name();
    if (name == "clear" && push_constants_size >= sizeof(float) * 4) {
        const auto* c = reinterpret_cast<const float*>(push_constants_buf);
        device_->StashClear(c[0], c[1], c[2], c[3]);
        return;
    }
    if (name == "pathtrace") {
        // The big one. Runs the CPU kernel over the entire dispatch
        // grid -- the engine passes (ceil(w/8), ceil(h/8), 1) for an
        // 8x8 work-group; we ignore the group count and parallelise
        // per-pixel internally based on our own thread-pool sizing.
        RunPathTraceKernel(*device_, *this);
        return;
    }
    if (name == "editor_overlay") {
        // 3D-transform gizmo overlay (issue: editor 3D gizmos). CPU
        // port of shaders/EditorOverlay.slang -- projects world-space
        // line segments to screen space and writes anti-aliased lines
        // onto the slot-0 output texture (the engine binds the
        // swapchain-backing scratch texture for Software).
        device_->RunEditorOverlay(*this);
        return;
    }
    // tonemap / autoexpose / bloom_down / bloom_up / perfoverlay --
    // intentionally skipped on Software. The path-trace kernel already
    // ACES-tonemaps inline and writes the final colour to the slot-0
    // output texture; subsequent passes are GPU-only optimisations
    // we don't reproduce on the CPU reference renderer.
}

// ---------- Embree error callback -----------------------------------------

static void EmbreeErrorCallback(void* /*user*/, RTCError code, const char* msg) {
    if (code == RTC_ERROR_NONE) return;
    LOG_ERROR("Embree: code={} msg={}", static_cast<int>(code), msg ? msg : "(null)");
}

// ---------- SoftwareDevice -------------------------------------------------

SoftwareDevice::SoftwareDevice(const NativeWindowHandle& window) {
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    cmd_buf_ = std::make_unique<SoftwareCommandBuffer>(this);

    width_  = window.width;
    height_ = window.height;
    // Cfg-driven backend switch can fire BEFORE the engine constructs
    // its Window (the r_backend cvar's on_change handler runs at
    // demont.cfg load time, which is line ~468 of Engine::Init, but
    // window_ isn't created until line ~490). The handler then passes
    // width=0/height=0 because `window_` is still null. Default to a
    // 1280x720 sentinel here so resource creation doesn't allocate
    // zero-byte storage; the engine's Resize() drives the real size
    // once the swapchain settles.
    if (width_ <= 0)  width_  = 1280;
    if (height_ <= 0) height_ = 720;

#if defined(__APPLE__)
    native_window_ = pt_window_native_cocoa(window.opaque);

    mtl_device_ = MTL::CreateSystemDefaultDevice();
    if (mtl_device_ == nullptr) {
        LOG_ERROR("Software backend: MTL::CreateSystemDefaultDevice failed");
        return;
    }
    mtl_queue_ = mtl_device_->newCommandQueue();
    if (mtl_queue_ == nullptr) {
        LOG_ERROR("Software backend: newCommandQueue failed");
        return;
    }

    mtl_layer_ = CA::MetalLayer::layer();
    mtl_layer_->retain();
    mtl_layer_->setDevice(mtl_device_);
    mtl_layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    mtl_layer_->setFramebufferOnly(false);
    mtl_layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width_),
                                       static_cast<CGFloat>(height_)});

    pt_metal_attach_layer(native_window_, mtl_layer_);
    // When the engine creates the device via a cfg-driven backend
    // switch, the window may not have realised its content-view size
    // yet -- the caller passes width=0/height=0 in that case. Query
    // the layer's drawable size (which CoreAnimation has already
    // populated from the attached NSView) and prefer that when the
    // caller-supplied dimensions are zero. Matches the same fallback
    // BeginFrame() uses each frame.
    if (width_ == 0 || height_ == 0) {
        auto sz = mtl_layer_->drawableSize();
        if (sz.width > 0 && sz.height > 0) {
            width_  = static_cast<int>(sz.width);
            height_ = static_cast<int>(sz.height);
        }
    }
#elif defined(_WIN32)
    // pt_window_native_win32 extracts the HWND from the GLFWwindow*
    // the engine handed us via NativeWindowHandle::opaque.
    native_window_ = pt_window_native_win32(window.opaque);
    if (native_window_ == nullptr) {
        LOG_ERROR("Software backend: pt_window_native_win32 returned null HWND");
        // Don't bail -- the kernel can still run and write pixels into
        // the output texture, the GDI present step will just no-op.
    } else {
        // Prefer the real client-rect size over the caller-supplied
        // width/height when available -- handles the cfg-driven backend
        // switch where the window dimensions arrive as 0x0 before the
        // window has finished realising. Same shape as the Mac block
        // above does via mtl_layer_->drawableSize().
        RECT rc{};
        if (GetClientRect(static_cast<HWND>(native_window_), &rc)) {
            int w = static_cast<int>(rc.right  - rc.left);
            int h = static_cast<int>(rc.bottom - rc.top);
            if (w > 0 && h > 0) { width_ = w; height_ = h; }
        }
    }
#else
    native_window_ = window.opaque;   // unsupported platform; will no-op present
#endif

    embree_device_ = rtcNewDevice(nullptr);
    if (embree_device_ == nullptr) {
        LOG_ERROR("Software backend: rtcNewDevice failed");
        return;
    }
    rtcSetDeviceErrorFunction(embree_device_, EmbreeErrorCallback, nullptr);

#if defined(__APPLE__)
    LOG_INFO("Software backend online (CPU + Embree, Metal present): {}x{}",
             width_, height_);
#elif defined(_WIN32)
    // Pick present path. The default ("vulkan") is the only mode that
    // survives a vulkan -> software backend switch on Windows: once
    // Vulkan touched the HWND, DXGI flip-model permanently locks the
    // window out of GDI presentation (Microsoft spec, see
    // SoftwareVulkanPresent.h for receipts). Users can opt into "gdi"
    // for a software-fresh-start session, but mid-session switches
    // from Vulkan will leave the window stale until restart -- a
    // LOG_WARN is emitted from the cvar's on_change in that case.
    bool vk_present_path = true;
    if (auto* v = pt::console::Console::Get().FindCVar("r_software_blit")) {
        vk_present_path = (v->value == "vulkan");
    }
    if (vk_present_path && native_window_ != nullptr) {
        vk_present_ = std::make_unique<SoftwareVulkanPresent>();
        if (!vk_present_->Init(native_window_, width_, height_)) {
            LOG_WARN("Software backend (Win32): Vulkan-blit present "
                     "init failed; falling back to GDI present (note: "
                     "GDI will be permanently broken if this session "
                     "ever ran the Vulkan backend; restart the app for "
                     "a clean GDI session)");
            vk_present_.reset();
        }
    }
    if (vk_present_) {
        LOG_INFO("Software backend online (CPU + Embree, Vulkan-blit "
                 "present): {}x{}", width_, height_);
    } else {
        LOG_INFO("Software backend online (CPU + Embree, GDI present): "
                 "{}x{}", width_, height_);
    }
#else
    LOG_INFO("Software backend online (CPU + Embree, no present): {}x{}",
             width_, height_);
#endif
}

SoftwareDevice::~SoftwareDevice() {
    cmd_buf_.reset();
    {
        std::lock_guard lock(resource_mutex_);
        // Release Embree scenes before the device they were built on.
        for (auto& [_, a] : accels_) {
            if (a && a->scene) {
                rtcReleaseScene(a->scene);
                a->scene = nullptr;
            }
        }
        accels_.clear();
        pipelines_.clear();
        buffers_.clear();
        textures_.clear();
    }
    if (embree_device_) { rtcReleaseDevice(embree_device_); embree_device_ = nullptr; }
#if defined(__APPLE__)
    if (present_tex_)   { present_tex_->release();          present_tex_   = nullptr; }
    if (mtl_layer_)     { mtl_layer_->release();            mtl_layer_     = nullptr; }
    if (mtl_queue_)     { mtl_queue_->release();            mtl_queue_     = nullptr; }
    if (mtl_device_)    { mtl_device_->release();           mtl_device_    = nullptr; }
#endif
    // Windows: no GDI-side ownership to release. GetDC / ReleaseDC are
    // paired around each Present call so the device-context lifetime
    // never spans frames.
}

BufferHandle SoftwareDevice::CreateBuffer(const BufferDesc& d) {
    auto b = std::make_unique<BackedBuffer>();
    b->data.assign(d.size, 0);
    b->size       = d.size;
    b->debug_name = d.debug_name;
    bytes_held_.fetch_add(d.size, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    buffers_.emplace(id, std::move(b));
    return BufferHandle{ id };
}

TextureHandle SoftwareDevice::CreateTexture(const TextureDesc& d) {
    auto t = std::make_unique<BackedTexture>();
    t->width      = d.width;
    t->height     = d.height;
    t->debug_name = d.debug_name;
    t->data.assign(static_cast<std::size_t>(d.width) * d.height * 4, 0.0f);
    auto bytes = t->data.size() * sizeof(float);
    bytes_held_.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    textures_.emplace(id, std::move(t));
    return TextureHandle{ id };
}

PipelineHandle SoftwareDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    pt::mem::TagScope scope(pt::MemTag::Render);
    auto pl = std::make_unique<SoftwarePipeline>(std::string(d.kernel_name));
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    pipelines_.emplace(id, std::move(pl));
    return PipelineHandle{ id };
}

AccelStructHandle SoftwareDevice::CreateBLAS(const BLASDesc& d) {
    if (embree_device_ == nullptr || d.vertex_count == 0 || d.index_count == 0) {
        LOG_WARN("Software CreateBLAS: empty mesh or Embree device missing");
        return {0};
    }
    auto a = std::make_unique<BackedAccel>();
    a->is_tlas = false;

    // Copy into our own storage so the engine is free to drop / replace
    // its CPU-side mesh data without invalidating the Embree BVH.
    a->vertices.assign(d.vertex_positions,
                       d.vertex_positions + std::size_t(d.vertex_count) * 3u);
    a->indices.assign(d.indices,
                      d.indices + std::size_t(d.index_count));

    RTCScene scene = rtcNewScene(embree_device_);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_NONE);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);

    RTCGeometry geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                               RTC_FORMAT_FLOAT3,
                               a->vertices.data(), 0, sizeof(float) * 3,
                               d.vertex_count);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                               RTC_FORMAT_UINT3,
                               a->indices.data(), 0, sizeof(std::uint32_t) * 3,
                               d.index_count / 3);
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    a->scene = scene;
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    LOG_INFO("Software CreateBLAS: id={} verts={} tris={}",
             id, d.vertex_count, d.index_count / 3);
    accels_.emplace(id, std::move(a));
    return AccelStructHandle{ id };
}

AccelStructHandle SoftwareDevice::CreateTLAS(const TLASDesc& d) {
    if (embree_device_ == nullptr || d.instances.empty()) return {0};

    auto a = std::make_unique<BackedAccel>();
    a->is_tlas = true;

    RTCScene scene = rtcNewScene(embree_device_);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_NONE);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);

    for (const auto& inst : d.instances) {
        BackedAccel* blas = nullptr;
        {
            std::lock_guard lock(resource_mutex_);
            auto it = accels_.find(inst.blas.id);
            if (it != accels_.end()) blas = it->second.get();
        }
        if (blas == nullptr || blas->scene == nullptr) {
            LOG_WARN("Software CreateTLAS: instance references unknown BLAS id={}", inst.blas.id);
            continue;
        }

        RTCGeometry inst_geom = rtcNewGeometry(embree_device_, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(inst_geom, blas->scene);
        // Engine's TLASInstance::transform is row-major 3x4 (12 floats).
        // Embree's RTC_FORMAT_FLOAT3X4_ROW_MAJOR matches exactly.
        rtcSetGeometryTransform(inst_geom, 0,
                                RTC_FORMAT_FLOAT3X4_ROW_MAJOR,
                                inst.transform);
        rtcSetGeometryTimeStepCount(inst_geom, 1);
        rtcCommitGeometry(inst_geom);
        rtcAttachGeometry(scene, inst_geom);
        rtcReleaseGeometry(inst_geom);
    }
    rtcCommitScene(scene);

    a->scene = scene;
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    LOG_INFO("Software CreateTLAS: id={} instances={}", id, d.instances.size());
    accels_.emplace(id, std::move(a));
    return AccelStructHandle{ id };
}

void SoftwareDevice::DestroyBuffer(BufferHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = buffers_.find(h.id); it != buffers_.end()) {
        if (it->second) bytes_held_.fetch_sub(it->second->size, std::memory_order_relaxed);
        buffers_.erase(it);
    }
}
void SoftwareDevice::DestroyTexture(TextureHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = textures_.find(h.id); it != textures_.end()) {
        if (it->second) {
            bytes_held_.fetch_sub(it->second->data.size() * sizeof(float),
                                  std::memory_order_relaxed);
        }
        textures_.erase(it);
    }
    if (h.id == output_tex_id_) output_tex_id_ = 0;
}

void SoftwareDevice::DestroyPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    pipelines_.erase(h.id);
}

void SoftwareDevice::DestroyAccelStruct(AccelStructHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = accels_.find(h.id); it != accels_.end()) {
        if (it->second && it->second->scene) {
            rtcReleaseScene(it->second->scene);
            it->second->scene = nullptr;
        }
        accels_.erase(it);
    }
}

bool SoftwareDevice::ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                                       std::uint32_t* out_w, std::uint32_t* out_h) {
    BackedTexture* t = nullptr;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = textures_.find(h.id);
        if (it != textures_.end()) t = it->second.get();
    }
    if (t == nullptr || dst == nullptr) return false;
    if (out_w) *out_w = t->width;
    if (out_h) *out_h = t->height;
    // Internal format is always RGBA32F (4 floats per pixel). Refuse
    // partial reads so the caller knows it has the full bytes or
    // nothing -- silently truncating would let screenshot / capture
    // callers process incomplete data as if the readback succeeded.
    // For `accum` (RGBA32F GPU format too) the sizes line up; for
    // RGBA16F targets the caller's buffer is half the size we
    // expose, which legitimately fails here -- that's the GPU-only
    // path (denoise_color, depth, motion) and the screenshot
    // command's per-target route already documents it as a
    // Software-backend gap.
    const std::size_t src_size = t->data.size() * sizeof(float);
    if (dst_size < src_size) return false;
    std::memcpy(dst, t->data.data(), src_size);
    return true;
}

bool SoftwareDevice::WriteTexture(TextureHandle h, const void* src, std::size_t size) {
    BackedTexture* t = nullptr;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = textures_.find(h.id);
        if (it != textures_.end()) t = it->second.get();
    }
    if (t == nullptr || src == nullptr) return false;
    // The internal storage is always RGBA32F (4 floats per pixel).
    // The engine may upload in other formats (RGBA16F for star/moon
    // maps, raw RGBA32F for env_map), but the v1 path tracer doesn't
    // sample any of these (it uses a procedural sky), so accept the
    // bytes generically without format conversion: memcpy as many
    // bytes as fit into the float-vector backing. Anything that lands
    // outside that capacity gets silently truncated -- still a
    // success from the engine's perspective, since the failure path
    // disables the texture (`stars disabled`, etc.) which is also
    // fine for the v1 software-backend render. A subsequent session
    // wiring real env-map sampling on Software will revisit this.
    const std::size_t cap_bytes = t->data.size() * sizeof(float);
    const std::size_t to_copy   = std::min(size, cap_bytes);
    std::memcpy(t->data.data(), src, to_copy);
    return true;
}

void SoftwareDevice::WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                                  std::size_t offset) {
    BackedBuffer* b = nullptr;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = buffers_.find(h.id);
        if (it != buffers_.end()) b = it->second.get();
    }
    if (b == nullptr) return;
    // Honor the Device.h contract like the GPU backends: write at
    // `offset`, and reject out-of-range regions loudly (Vulkan-style)
    // instead of clamping -- a clamped partial write is a silent
    // software-vs-GPU divergence waiting for a golden to catch it.
    if (offset > b->data.size() || size > b->data.size() - offset) {
        LOG_ERROR("WriteBuffer: copy region (offset {} + size {}) exceeds "
                  "buffer size {}", offset, size, b->data.size());
        return;
    }
    std::memcpy(b->data.data() + offset, src, size);
}

FrameContext SoftwareDevice::BeginFrame() {
    // Sync our cached width/height with the real window size each
    // frame -- handles late-arriving resize events that the engine
    // hasn't pumped through Resize() yet.
#if defined(__APPLE__)
    if (mtl_layer_) {
        auto sz = mtl_layer_->drawableSize();
        width_  = static_cast<int>(sz.width);
        height_ = static_cast<int>(sz.height);
    }
#elif defined(_WIN32)
    if (HWND hwnd = static_cast<HWND>(native_window_)) {
        RECT rc{};
        if (GetClientRect(hwnd, &rc)) {
            int w = static_cast<int>(rc.right  - rc.left);
            int h = static_cast<int>(rc.bottom - rc.top);
            if (w > 0 && h > 0) { width_ = w; height_ = h; }
        }
    }
#endif
    EnsureSwapchainOutput();
    return FrameContext{
        .swapchain_image = TextureHandle{output_tex_id_},
        .width  = static_cast<std::uint32_t>(width_),
        .height = static_cast<std::uint32_t>(height_),
        .frame_index = frame_index_,
    };
}

void SoftwareDevice::EndFrame(CommandBuffer*) {
#if defined(__APPLE__)
    if (mtl_layer_ == nullptr || mtl_queue_ == nullptr) { ++frame_index_; return; }

    auto* pool = NS::AutoreleasePool::alloc()->init();

    auto* drawable = mtl_layer_->nextDrawable();
    if (drawable == nullptr) { pool->release(); ++frame_index_; return; }

    // If the path tracer wrote into the output texture this frame,
    // upload its CPU backing to a Metal texture and blit-copy to the
    // drawable. Falls back to the clear-color render-pass path when
    // no output is populated yet (very first frame before the engine
    // has dispatched anything).
    bool blitted = false;
    BackedTexture* out_tex = GetTexture(TextureHandle{output_tex_id_});
    if (out_tex != nullptr && out_tex->width > 0 && out_tex->height > 0) {
        PresentOutput();
        // PresentOutput uploaded into present_tex_; now blit to the
        // drawable's texture.
        if (present_tex_ != nullptr) {
            auto* cb  = mtl_queue_->commandBuffer();
            auto* enc = cb->blitCommandEncoder();
            MTL::Origin src_origin{0, 0, 0};
            MTL::Size   src_size{present_w_, present_h_, 1};
            MTL::Origin dst_origin{0, 0, 0};
            enc->copyFromTexture(present_tex_, 0, 0, src_origin, src_size,
                                  drawable->texture(), 0, 0, dst_origin);
            enc->endEncoding();
            cb->presentDrawable(drawable);
            cb->commit();
            blitted = true;
        }
    }
    if (!blitted) {
        // Fallback: clear-only present.
        auto* rpd = MTL::RenderPassDescriptor::renderPassDescriptor();
        auto* attachment = rpd->colorAttachments()->object(0);
        attachment->setTexture(drawable->texture());
        attachment->setLoadAction(MTL::LoadActionClear);
        attachment->setStoreAction(MTL::StoreActionStore);
        attachment->setClearColor(MTL::ClearColor::Make(
            pending_clear_[0], pending_clear_[1],
            pending_clear_[2], pending_clear_[3]));

        auto* cb  = mtl_queue_->commandBuffer();
        auto* enc = cb->renderCommandEncoder(rpd);
        enc->endEncoding();
        cb->presentDrawable(drawable);
        cb->commit();
    }

    pool->release();
#elif defined(_WIN32)
    // Windows present. Two paths:
    //   - Vulkan-blit (vk_present_ non-null, default): pack output via
    //     PresentOutput() then upload + vkCmdCopyBufferToImage +
    //     vkQueuePresentKHR through SoftwareVulkanPresent.
    //   - GDI (vk_present_ null, opt-in): SetDIBitsToDevice from the
    //     BGRA8 scratch buffer to the HWND's HDC. HDC is acquired
    //     and released per frame; no driver-managed swapchain.
    HWND hwnd = static_cast<HWND>(native_window_);
    if (hwnd == nullptr) { ++frame_index_; return; }

    bool blitted = false;
    BackedTexture* out_tex = GetTexture(TextureHandle{output_tex_id_});
    if (out_tex != nullptr && out_tex->width > 0 && out_tex->height > 0) {
        PresentOutput();   // packs RGBA32F -> BGRA8 into present_scratch_
        if (!present_scratch_.empty() &&
            present_scratch_w_ > 0 && present_scratch_h_ > 0) {
            if (vk_present_) {
                if (vk_present_->Present(present_scratch_,
                                         static_cast<int>(present_scratch_w_),
                                         static_cast<int>(present_scratch_h_))) {
                    blitted = true;
                }
                ++frame_index_;
                return;  // GDI fallback below not used in Vulkan-blit mode
            }
            HDC hdc = GetDC(hwnd);
            if (hdc) {
                BITMAPINFO bmi{};
                bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
                bmi.bmiHeader.biWidth       =  static_cast<LONG>(present_scratch_w_);
                // Negative height = top-down DIB.  Our scratch buffer
                // is in top-down row order (row 0 = top of image), so
                // a negative biHeight tells GDI not to flip Y on us.
                bmi.bmiHeader.biHeight      = -static_cast<LONG>(present_scratch_h_);
                bmi.bmiHeader.biPlanes      = 1;
                bmi.bmiHeader.biBitCount    = 32;
                bmi.bmiHeader.biCompression = BI_RGB;
                SetDIBitsToDevice(
                    hdc,
                    /*xDest=*/ 0,             /*yDest=*/ 0,
                    /*w=*/     present_scratch_w_,
                    /*h=*/     present_scratch_h_,
                    /*xSrc=*/  0,             /*ySrc=*/  0,
                    /*startScan=*/ 0,
                    /*cLines=*/    present_scratch_h_,
                    present_scratch_.data(),
                    &bmi,
                    DIB_RGB_COLORS);
                ReleaseDC(hwnd, hdc);
                blitted = true;
            }
        }
    }
    if (!blitted && out_tex == nullptr) {
        // First-frame fallback ONLY (no output texture exists yet).
        // Paint the client rect with the pending clear colour so the
        // window doesn't show garbage on launch before the kernel
        // dispatches its first frame.
        //
        // Why gated on out_tex == nullptr: if out_tex exists but is
        // transiently 0x0 (e.g. one frame after a minimise-to-tray
        // Resize(0, 0)), or PresentOutput legitimately couldn't pack
        // a frame for some other reason, painting clear colour over
        // the previously-blitted content flashes a single ugly frame.
        // GDI doesn't auto-repaint, so leaving the HDC untouched
        // keeps the last good frame on screen until the kernel
        // produces a new one -- much better UX than the flash.  Mac
        // doesn't have this hazard because every Metal frame must
        // present a drawable, so the equivalent fallback there
        // genuinely needs to run.
        HDC hdc = GetDC(hwnd);
        if (hdc) {
            RECT rc{};
            GetClientRect(hwnd, &rc);
            auto to_byte = [](float v) {
                return static_cast<int>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            COLORREF cr = RGB(to_byte(pending_clear_[0]),
                              to_byte(pending_clear_[1]),
                              to_byte(pending_clear_[2]));
            HBRUSH br = CreateSolidBrush(cr);
            if (br) {
                FillRect(hdc, &rc, br);
                DeleteObject(br);
            }
            ReleaseDC(hwnd, hdc);
        }
    }
#endif
    ++frame_index_;
}

CommandBuffer* SoftwareDevice::AcquireCommandBuffer() { return cmd_buf_.get(); }

void SoftwareDevice::Submit(CommandBuffer*) {
    // No-op on Software. Dispatch() has already run the CPU kernel
    // (and any bound clear pipeline's stash) inline at record time,
    // so there is nothing to flush here. Kept for interface parity
    // with the GPU backends.
}

void SoftwareDevice::Resize(int w, int h) {
    width_  = w; height_ = h;
#if defined(__APPLE__)
    if (mtl_layer_) {
        mtl_layer_->setDrawableSize(CGSize{static_cast<CGFloat>(w),
                                           static_cast<CGFloat>(h)});
    }
#elif defined(_WIN32)
    if (vk_present_) vk_present_->Resize(w, h);
#endif
    // Windows GDI fallback: no layer-side state to resize.  GetClientRect
    // is queried fresh each BeginFrame / EndFrame, so the next present
    // will pick up the new size automatically.
    //
    // Cross-platform: force the output texture to re-allocate so the
    // CPU kernel writes into a correctly-sized buffer next frame.  The
    // present scratch buffer also re-sizes lazily inside PresentOutput.
    if (output_tex_id_ != 0) {
        BackedTexture* t = GetTexture(TextureHandle{output_tex_id_});
        if (t != nullptr) {
            t->width  = static_cast<std::uint32_t>(w);
            t->height = static_cast<std::uint32_t>(h);
            t->data.assign(std::size_t(w) * h * 4, 0.0f);
        }
    }
}

std::size_t SoftwareDevice::CurrentAllocatedBytes() const {
    return bytes_held_.load(std::memory_order_relaxed);
}

void SoftwareDevice::StashClear(float r, float g, float b, float a) {
    pending_clear_[0] = r; pending_clear_[1] = g;
    pending_clear_[2] = b; pending_clear_[3] = a;
}

// CPU port of shaders/EditorOverlay.slang. Walks the engine-supplied
// segment list and rasterizes each onto the output texture by per-pixel
// distance-to-segment, alpha-blending the result. The texture is stored
// RGBA32F internally; the path tracer's slot-0 write was already
// clamped to [0,1] for display, and we honour that range here too --
// the present pack step clamps again so out-of-range values would be
// truncated anyway. The segment buffer layout matches the Slang
// declaration: 3 float4s per segment (endpoint A + thickness,
// endpoint B + reserved depth bias, RGB colour + pad).
void SoftwareDevice::RunEditorOverlay(SoftwareCommandBuffer& cb) {
    BackedTexture* out_tex = GetTexture(TextureHandle{cb.binds.textures[0]});
    if (out_tex == nullptr) return;
    BackedBuffer*  segs    = GetBuffer(BufferHandle{cb.binds.buffers[1]});
    if (segs == nullptr) return;

    // Push layout (mirrors shaders/EditorOverlay.slang):
    //   float4 pos_fovtan, fwd_aspect, right_xyz, up_xyz;
    //   uint   num_segments, _pad0, _pad1, _pad2;
    // 4 * 16 + 16 = 80 bytes.
    if (cb.push_constants_size < 80u) return;
    const auto* pf = reinterpret_cast<const float*>(cb.push_constants_buf);
    const auto* pu = reinterpret_cast<const std::uint32_t*>(cb.push_constants_buf);
    const float pos_x   = pf[ 0]; const float pos_y   = pf[ 1];
    const float pos_z   = pf[ 2]; const float fovYTan = pf[ 3];
    const float fwd_x   = pf[ 4]; const float fwd_y   = pf[ 5];
    const float fwd_z   = pf[ 6]; const float aspect  = pf[ 7];
    const float right_x = pf[ 8]; const float right_y = pf[ 9];
    const float right_z = pf[10];
    const float up_x    = pf[12]; const float up_y    = pf[13];
    const float up_z    = pf[14];
    const std::uint32_t num_segments = pu[16];
    if (num_segments == 0u) return;

    const std::uint32_t w = out_tex->width;
    const std::uint32_t h = out_tex->height;
    if (w == 0u || h == 0u) return;

    // Cast segment data. Buffer was written by engine as float4 entries
    // (3 per segment) -- 12 floats per segment.
    const auto* sf = reinterpret_cast<const float*>(segs->data.data());
    const std::size_t total_floats = segs->data.size() / sizeof(float);
    const std::uint32_t max_segs   = static_cast<std::uint32_t>(total_floats / 12u);
    const std::uint32_t n_segs     = std::min(num_segments, max_segs);

    auto srgb_encode = [](float c) {
        if (c <= 0.0f) return 0.0f;
        if (c >= 1.0f) return 1.0f;
        return c <= 0.0031308f ? 12.92f * c
                               : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    };

    // Pre-project every endpoint to screen space, skip segments behind
    // the camera. Cuts the per-pixel cost from "3 dot products + a few
    // divides per segment" to "linear interp from cached float2s".
    struct ProjSeg {
      float ax, ay; // pixel space
      float bx, by;
      float cx, cy;     // filled-triangle C vertex
      float r, g, b;    // sRGB-encoded display colour
      float half_thick; // pixel half-thickness
      bool filled;
    };
    auto fill_color_from_code = [](float code, float &r, float &g, float &b) {
      if (code > 9.5f) {
        r = 1.0f;
        g = 0.95f;
        b = 0.10f;
      } else if (code > 2.5f) {
        r = 0.20f;
        g = 0.40f;
        b = 1.00f;
      } else if (code > 1.5f) {
        r = 0.20f;
        g = 0.85f;
        b = 0.20f;
      } else if (code > 0.5f) {
        r = 0.95f;
        g = 0.15f;
        b = 0.18f;
      } else {
        r = 0.85f;
        g = 0.85f;
        b = 0.85f;
      }
    };
    std::vector<ProjSeg> proj;
    proj.reserve(n_segs);
    for (std::uint32_t i = 0; i < n_segs; ++i) {
      const float *s = sf + std::size_t(i) * 12u;
      const float ax_w = s[0], ay_w = s[1], az_w = s[2];
      const float half_t = s[3];
      const float bx_w = s[4], by_w = s[5], bz_w = s[6];
      // s[7] reserved (depth bias) -- unused in v1.
      const float cr = s[8], cg = s[9], cb_col = s[10];
      const bool filled = half_t < 0.0f;

      // Project A
      float dxa = ax_w - pos_x, dya = ay_w - pos_y, dza = az_w - pos_z;
      float z_a = dxa * fwd_x + dya * fwd_y + dza * fwd_z;
      if (z_a <= 0.001f)
        continue;
      float xs_a = (dxa * right_x + dya * right_y + dza * right_z) / z_a;
      float ys_a = (dxa * up_x + dya * up_y + dza * up_z) / z_a;
      // Project B
      float dxb = bx_w - pos_x, dyb = by_w - pos_y, dzb = bz_w - pos_z;
      float z_b = dxb * fwd_x + dyb * fwd_y + dzb * fwd_z;
      if (z_b <= 0.001f)
        continue;
      float xs_b = (dxb * right_x + dyb * right_y + dzb * right_z) / z_b;
      float ys_b = (dxb * up_x + dyb * up_y + dzb * up_z) / z_b;

      float u_a = xs_a / (fovYTan * aspect);
      float v_a = ys_a / fovYTan;
      float u_b = xs_b / (fovYTan * aspect);
      float v_b = ys_b / fovYTan;

      ProjSeg p;
      p.ax = (u_a * 0.5f + 0.5f) * float(w);
      p.ay = (1.0f - (v_a * 0.5f + 0.5f)) * float(h);
      p.bx = (u_b * 0.5f + 0.5f) * float(w);
      p.by = (1.0f - (v_b * 0.5f + 0.5f)) * float(h);
      p.cx = p.ax;
      p.cy = p.ay;
      p.filled = filled;
      if (filled) {
        const float cx_w = cr, cy_w = cg, cz_w = cb_col;
        float dxc = cx_w - pos_x, dyc = cy_w - pos_y, dzc = cz_w - pos_z;
        float z_c = dxc * fwd_x + dyc * fwd_y + dzc * fwd_z;
        if (z_c <= 0.001f)
          continue;
        float xs_c = (dxc * right_x + dyc * right_y + dzc * right_z) / z_c;
        float ys_c = (dxc * up_x + dyc * up_y + dzc * up_z) / z_c;
        float u_c = xs_c / (fovYTan * aspect);
        float v_c = ys_c / fovYTan;
        p.cx = (u_c * 0.5f + 0.5f) * float(w);
        p.cy = (1.0f - (v_c * 0.5f + 0.5f)) * float(h);
        float lr, lg, lb;
        fill_color_from_code(s[11], lr, lg, lb);
        p.r = srgb_encode(lr);
        p.g = srgb_encode(lg);
        p.b = srgb_encode(lb);
        p.half_thick = 0.0f;
      } else {
        p.r = srgb_encode(cr);
        p.g = srgb_encode(cg);
        p.b = srgb_encode(cb_col);
        p.half_thick = std::max(half_t, 0.5f);
      }
      proj.push_back(p);
    }
    if (proj.empty())
      return;

    // Compute the bounding box across all projected segments to avoid
    // scanning the entire framebuffer when the gizmo only occupies a
    // small region. Pad by 2 pixels for the anti-alias feather.
    float min_x = float(w), min_y = float(h);
    float max_x = 0.0f, max_y = 0.0f;
    for (const auto &p : proj) {
      const float min_px =
          p.filled ? std::min({p.ax, p.bx, p.cx}) : std::min(p.ax, p.bx);
      const float max_px =
          p.filled ? std::max({p.ax, p.bx, p.cx}) : std::max(p.ax, p.bx);
      const float min_py =
          p.filled ? std::min({p.ay, p.by, p.cy}) : std::min(p.ay, p.by);
      const float max_py =
          p.filled ? std::max({p.ay, p.by, p.cy}) : std::max(p.ay, p.by);
      float lo_x = min_px - (p.half_thick + 2.0f);
      float hi_x = max_px + (p.half_thick + 2.0f);
      float lo_y = min_py - (p.half_thick + 2.0f);
      float hi_y = max_py + (p.half_thick + 2.0f);
      if (lo_x < min_x)
        min_x = lo_x;
      if (lo_y < min_y)
        min_y = lo_y;
      if (hi_x > max_x)
        max_x = hi_x;
      if (hi_y > max_y)
        max_y = hi_y;
    }
    const std::int32_t x0 =
        std::clamp(static_cast<std::int32_t>(std::floor(min_x)), 0,
                   static_cast<std::int32_t>(w) - 1);
    const std::int32_t y0 =
        std::clamp(static_cast<std::int32_t>(std::floor(min_y)), 0,
                   static_cast<std::int32_t>(h) - 1);
    const std::int32_t x1 =
        std::clamp(static_cast<std::int32_t>(std::ceil(max_x)), 0,
                   static_cast<std::int32_t>(w) - 1);
    const std::int32_t y1 =
        std::clamp(static_cast<std::int32_t>(std::ceil(max_y)), 0,
                   static_cast<std::int32_t>(h) - 1);
    if (x1 < x0 || y1 < y0)
      return;

    for (std::int32_t py = y0; py <= y1; ++py) {
      float fy = float(py) + 0.5f;
      for (std::int32_t px = x0; px <= x1; ++px) {
        float fx = float(px) + 0.5f;
        float pix_r = out_tex->data[(std::size_t(py) * w + px) * 4 + 0];
        float pix_g = out_tex->data[(std::size_t(py) * w + px) * 4 + 1];
        float pix_b = out_tex->data[(std::size_t(py) * w + px) * 4 + 2];
        bool changed = false;
        for (const auto &s : proj) {
          auto seg_dist_px = [](float px_, float py_, float x0_, float y0_,
                                float x1_, float y1_) {
            float vx_ = x1_ - x0_;
            float vy_ = y1_ - y0_;
            float wx_ = px_ - x0_;
            float wy_ = py_ - y0_;
            float len2_ = vx_ * vx_ + vy_ * vy_;
            if (len2_ < 1e-6f) {
              float dx_ = px_ - x0_, dy_ = py_ - y0_;
              return std::sqrt(dx_ * dx_ + dy_ * dy_);
            }
            float t_ = std::clamp((wx_ * vx_ + wy_ * vy_) / len2_, 0.0f, 1.0f);
            float qx_ = x0_ + t_ * vx_;
            float qy_ = y0_ + t_ * vy_;
            float dx_ = px_ - qx_, dy_ = py_ - qy_;
            return std::sqrt(dx_ * dx_ + dy_ * dy_);
          };
          if (s.filled) {
            auto edge = [](float ax_, float ay_, float bx_, float by_,
                           float px_, float py_) {
              float ex_ = bx_ - ax_;
              float ey_ = by_ - ay_;
              float wx_ = px_ - ax_;
              float wy_ = py_ - ay_;
              return ex_ * wy_ - ey_ * wx_;
            };
            const float e0 = edge(s.ax, s.ay, s.bx, s.by, fx, fy);
            const float e1 = edge(s.bx, s.by, s.cx, s.cy, fx, fy);
            const float e2 = edge(s.cx, s.cy, s.ax, s.ay, fx, fy);
            const bool has_neg = (e0 < 0.0f) || (e1 < 0.0f) || (e2 < 0.0f);
            const bool has_pos = (e0 > 0.0f) || (e1 > 0.0f) || (e2 > 0.0f);
            if (has_neg && has_pos)
              continue;
            const float d0 = seg_dist_px(fx, fy, s.ax, s.ay, s.bx, s.by);
            const float d1 = seg_dist_px(fx, fy, s.bx, s.by, s.cx, s.cy);
            const float d2 = seg_dist_px(fx, fy, s.cx, s.cy, s.ax, s.ay);
            const float edge_d = std::min(d0, std::min(d1, d2));
            const float aa = 0.42f * std::clamp(edge_d + 0.5f, 0.0f, 1.0f);
            pix_r = pix_r * (1.0f - aa) + s.r * aa;
            pix_g = pix_g * (1.0f - aa) + s.g * aa;
            pix_b = pix_b * (1.0f - aa) + s.b * aa;
            changed = true;
            continue;
          }
          float vx = s.bx - s.ax;
          float vy = s.by - s.ay;
          float wx = fx - s.ax;
          float wy = fy - s.ay;
          float len2 = vx * vx + vy * vy;
          float d;
          if (len2 < 1e-6f) {
            float dx = fx - s.ax, dy = fy - s.ay;
            d = std::sqrt(dx * dx + dy * dy);
          } else {
            float t = std::clamp((wx * vx + wy * vy) / len2, 0.0f, 1.0f);
            float qx = s.ax + t * vx;
            float qy = s.ay + t * vy;
            float dx = fx - qx, dy = fy - qy;
            d = std::sqrt(dx * dx + dy * dy);
          }
          if (d > s.half_thick + 1.0f)
            continue;
          float aa = std::clamp((s.half_thick + 1.0f) - d, 0.0f, 1.0f);
          if (aa <= 0.0f)
            continue;
          pix_r = pix_r * (1.0f - aa) + s.r * aa;
          pix_g = pix_g * (1.0f - aa) + s.g * aa;
          pix_b = pix_b * (1.0f - aa) + s.b * aa;
          changed = true;
        }
        if (changed) {
          out_tex->data[(std::size_t(py) * w + px) * 4 + 0] = pix_r;
          out_tex->data[(std::size_t(py) * w + px) * 4 + 1] = pix_g;
          out_tex->data[(std::size_t(py) * w + px) * 4 + 2] = pix_b;
        }
      }
    }
}

SoftwarePipeline* SoftwareDevice::GetPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = pipelines_.find(h.id);
    return it == pipelines_.end() ? nullptr : it->second.get();
}
BackedBuffer* SoftwareDevice::GetBuffer(BufferHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = buffers_.find(h.id);
    return it == buffers_.end() ? nullptr : it->second.get();
}
BackedTexture* SoftwareDevice::GetTexture(TextureHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = textures_.find(h.id);
    return it == textures_.end() ? nullptr : it->second.get();
}
BackedAccel* SoftwareDevice::GetAccel(AccelStructHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = accels_.find(h.id);
    return it == accels_.end() ? nullptr : it->second.get();
}

void SoftwareDevice::EnsureSwapchainOutput() {
    if (width_ <= 0 || height_ <= 0) return;
    if (output_tex_id_ != 0) {
        BackedTexture* t = GetTexture(TextureHandle{output_tex_id_});
        if (t != nullptr && t->width == static_cast<std::uint32_t>(width_)
                         && t->height == static_cast<std::uint32_t>(height_)) {
            return;   // already sized correctly
        }
    }
    TextureDesc d{};
    d.width  = static_cast<std::uint32_t>(width_);
    d.height = static_cast<std::uint32_t>(height_);
    d.format = pt::rhi::TextureFormat::RGBA8_UNORM;
    d.usage  = pt::rhi::TextureUsage::Storage;
    d.debug_name = "sw_output";
    auto handle = CreateTexture(d);
    output_tex_id_ = handle.id;
}

void SoftwareDevice::PresentOutput() {
    BackedTexture* out_tex = GetTexture(TextureHandle{output_tex_id_});
    if (out_tex == nullptr) return;

    const std::uint32_t w = out_tex->width;
    const std::uint32_t h = out_tex->height;
    if (w == 0 || h == 0) return;

    // --- Cross-platform: RGBA32F -> BGRA8 pack into scratch ---------
    // Clamp to [0,1]; the path tracer is responsible for tonemapping
    // into that range. Scratch buffer is a class member so the per-
    // frame heap churn (several MB at 1080p / tens of MB at 4K) goes
    // away after the first resize -- only re-allocates when the
    // swapchain grows.  The packed format 0xAARRGGBB matches both
    // Metal's BGRA8Unorm pixel format (Mac upload path) and GDI's
    // BI_RGB DIB layout (Windows present path).
    const std::size_t pixel_count = std::size_t(w) * h;
    if (present_scratch_.size() < pixel_count) {
        present_scratch_.resize(pixel_count);
    }
    for (std::size_t i = 0; i < pixel_count; ++i) {
        float r = std::clamp(out_tex->data[i * 4 + 0], 0.0f, 1.0f);
        float g = std::clamp(out_tex->data[i * 4 + 1], 0.0f, 1.0f);
        float b = std::clamp(out_tex->data[i * 4 + 2], 0.0f, 1.0f);
        float a = std::clamp(out_tex->data[i * 4 + 3], 0.0f, 1.0f);
        std::uint32_t bi = static_cast<std::uint32_t>(b * 255.0f + 0.5f);
        std::uint32_t gi = static_cast<std::uint32_t>(g * 255.0f + 0.5f);
        std::uint32_t ri = static_cast<std::uint32_t>(r * 255.0f + 0.5f);
        std::uint32_t ai = static_cast<std::uint32_t>(a * 255.0f + 0.5f);
        present_scratch_[i] = (ai << 24) | (ri << 16) | (gi << 8) | bi;
    }
    present_scratch_w_ = w;
    present_scratch_h_ = h;

#if defined(__APPLE__)
    // Mac upload: copy scratch into a transient MTLTexture so the
    // EndFrame blit can src from a GPU-side resource into the drawable.
    if (mtl_device_ == nullptr) return;
    if (present_tex_ == nullptr || present_w_ != w || present_h_ != h) {
        if (present_tex_) { present_tex_->release(); present_tex_ = nullptr; }
        auto* td = MTL::TextureDescriptor::alloc()->init();
        td->setWidth(w);
        td->setHeight(h);
        td->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
        td->setStorageMode(MTL::StorageModeShared);
        td->setUsage(MTL::TextureUsageShaderRead);
        present_tex_ = mtl_device_->newTexture(td);
        td->release();
        present_w_ = w;
        present_h_ = h;
    }
    if (present_tex_ == nullptr) return;
    MTL::Region region = MTL::Region::Make2D(0, 0, w, h);
    present_tex_->replaceRegion(region, 0, present_scratch_.data(), w * 4);
#endif
    // Windows: nothing more to do -- EndFrame() reads present_scratch_
    // directly via SetDIBitsToDevice. No intermediate GPU resource.
}

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w) {
    return std::make_unique<sw::SoftwareDevice>(w);
}
}

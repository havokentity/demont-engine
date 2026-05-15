// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Software backend: CPU compute via Embree (triangle BVH) + analytic-
// primitive intersection done inline. The path tracer writes a CPU
// framebuffer; we upload it to a Metal texture and blit-present each
// frame. No Slang shader is compiled to CPU for this target; the kernel
// is a hand-written C++ port covering the subset documented in
// SoftwareTracer.h.

#include "SoftwareDevice.h"
#include "SoftwareTracer.h"

#include "../core/Log.h"
#include "../core/Memory/MemTag.h"
#include "../core/Memory/Memory.h"

// Headers only -- the metal-cpp PRIVATE_IMPLEMENTATION TU lives in
// rhi_metal/MetalDevice.cpp and supplies the impl symbols we need.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <embree4/rtcore.h>

#include <algorithm>
#include <cstring>
#include <cmath>

extern "C" void  pt_metal_attach_layer(void* ns_window, void* metal_layer);
extern "C" void* pt_window_native_cocoa(void* glfw_window);

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

    ns_window_ = pt_window_native_cocoa(window.opaque);
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

    pt_metal_attach_layer(ns_window_, mtl_layer_);
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

    embree_device_ = rtcNewDevice(nullptr);
    if (embree_device_ == nullptr) {
        LOG_ERROR("Software backend: rtcNewDevice failed");
        return;
    }
    rtcSetDeviceErrorFunction(embree_device_, EmbreeErrorCallback, nullptr);

    LOG_INFO("Software backend online (CPU + Embree, Metal present): {}x{}",
             width_, height_);
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
    if (present_tex_)   { present_tex_->release();          present_tex_   = nullptr; }
    if (mtl_layer_)     { mtl_layer_->release();            mtl_layer_     = nullptr; }
    if (mtl_queue_)     { mtl_queue_->release();            mtl_queue_     = nullptr; }
    if (mtl_device_)    { mtl_device_->release();           mtl_device_    = nullptr; }
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
                                  std::size_t /*offset*/) {
    BackedBuffer* b = nullptr;
    {
        std::lock_guard lock(resource_mutex_);
        auto it = buffers_.find(h.id);
        if (it != buffers_.end()) b = it->second.get();
    }
    if (b == nullptr) return;
    if (size > b->data.size()) size = b->data.size();
    std::memcpy(b->data.data(), src, size);
}

FrameContext SoftwareDevice::BeginFrame() {
    if (mtl_layer_) {
        auto sz = mtl_layer_->drawableSize();
        width_  = static_cast<int>(sz.width);
        height_ = static_cast<int>(sz.height);
    }
    EnsureSwapchainOutput();
    return FrameContext{
        .swapchain_image = TextureHandle{output_tex_id_},
        .width  = static_cast<std::uint32_t>(width_),
        .height = static_cast<std::uint32_t>(height_),
        .frame_index = frame_index_,
    };
}

void SoftwareDevice::EndFrame(CommandBuffer*) {
    if (mtl_layer_ == nullptr || mtl_queue_ == nullptr) return;

    auto* pool = NS::AutoreleasePool::alloc()->init();

    auto* drawable = mtl_layer_->nextDrawable();
    if (drawable == nullptr) { pool->release(); return; }

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
    if (mtl_layer_) {
        mtl_layer_->setDrawableSize(CGSize{static_cast<CGFloat>(w),
                                           static_cast<CGFloat>(h)});
    }
    // Force the output texture and present scratch to resize on next
    // frame.
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
    if (out_tex == nullptr || mtl_device_ == nullptr) return;

    const std::uint32_t w = out_tex->width;
    const std::uint32_t h = out_tex->height;

    // Allocate / re-create the present scratch texture if the
    // swapchain size changed.
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

    // Convert RGBA32F -> BGRA8Unorm and upload. Clamp to [0,1]; the
    // path tracer is responsible for tonemapping into that range.
    // Scratch buffer is a class member so the per-frame heap churn
    // (several MB at 1080p / tens of MB at 4K) goes away after the
    // first resize -- only re-allocates when the swapchain grows.
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
        // BGRA8Unorm packs as 0xAARRGGBB on little-endian.
        present_scratch_[i] = (ai << 24) | (ri << 16) | (gi << 8) | bi;
    }
    MTL::Region region = MTL::Region::Make2D(0, 0, w, h);
    present_tex_->replaceRegion(region, 0, present_scratch_.data(), w * 4);
}

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w) {
    return std::make_unique<sw::SoftwareDevice>(w);
}
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Software backend: CPU work (currently just stash-clear-color), then
// present via a Metal render pass with loadAction=Clear. No OpenGL
// anywhere -- we share the Metal layer attachment with the metal backend.

#include "SoftwareDevice.h"

#include "../core/Log.h"
#include "../core/Memory/MemTag.h"
#include "../core/Memory/Memory.h"

// Headers only -- the metal-cpp PRIVATE_IMPLEMENTATION TU lives in
// rhi_metal/MetalDevice.cpp and supplies the impl symbols we need.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cstring>

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

void SoftwareCommandBuffer::PushConstants(const void* data, std::size_t size) {
    if (size > sizeof(push_constants_buf)) size = sizeof(push_constants_buf);
    std::memcpy(push_constants_buf, data, size);
    push_constants_size = size;
}

void SoftwareCommandBuffer::Dispatch(std::uint32_t, std::uint32_t, std::uint32_t) {
    auto* pl = device_->GetPipeline(bound_pipeline);
    if (pl == nullptr) return;
    if (pl->Name() == "clear" && push_constants_size >= sizeof(float) * 4) {
        const auto* c = reinterpret_cast<const float*>(push_constants_buf);
        device_->StashClear(c[0], c[1], c[2], c[3]);
    }
}

// ---------- SoftwareDevice -------------------------------------------------

SoftwareDevice::SoftwareDevice(const NativeWindowHandle& window) {
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    cmd_buf_ = std::make_unique<SoftwareCommandBuffer>(this);

    ns_window_ = pt_window_native_cocoa(window.opaque);
    width_  = window.width;
    height_ = window.height;

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
    // Match the metal backend's storage layout exactly so brightness
    // doesn't drift between backends (framebufferOnly enables tiled
    // compressed internal formats on Apple Silicon, which round 8-bit
    // values slightly differently from plain BGRA8Unorm).
    mtl_layer_->setFramebufferOnly(false);
    mtl_layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width_),
                                       static_cast<CGFloat>(height_)});

    pt_metal_attach_layer(ns_window_, mtl_layer_);

    LOG_INFO("Software backend online (CPU compute, Metal present): {}x{}",
             width_, height_);
}

SoftwareDevice::~SoftwareDevice() {
    cmd_buf_.reset();
    {
        std::lock_guard lock(resource_mutex_);
        pipelines_.clear();
    }
    if (mtl_layer_)  { mtl_layer_->release();  mtl_layer_  = nullptr; }
    if (mtl_queue_)  { mtl_queue_->release();  mtl_queue_  = nullptr; }
    if (mtl_device_) { mtl_device_->release(); mtl_device_ = nullptr; }
}

BufferHandle SoftwareDevice::CreateBuffer(const BufferDesc& d) {
    bytes_held_.fetch_add(d.size, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    return BufferHandle{ next_id_++ };
}

TextureHandle SoftwareDevice::CreateTexture(const TextureDesc& d) {
    auto bytes = static_cast<std::size_t>(d.width) * d.height * 4;
    bytes_held_.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    return TextureHandle{ next_id_++ };
}

PipelineHandle SoftwareDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    pt::mem::TagScope scope(pt::MemTag::Render);
    auto pl = std::make_unique<SoftwarePipeline>(std::string(d.kernel_name));
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    pipelines_.emplace(id, std::move(pl));
    return PipelineHandle{ id };
}

void SoftwareDevice::DestroyBuffer(BufferHandle)     { /* trivial in P2 */ }
void SoftwareDevice::DestroyTexture(TextureHandle)   { /* trivial in P2 */ }

void SoftwareDevice::DestroyPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    pipelines_.erase(h.id);
}

FrameContext SoftwareDevice::BeginFrame() {
    if (mtl_layer_) {
        auto sz = mtl_layer_->drawableSize();
        width_  = static_cast<int>(sz.width);
        height_ = static_cast<int>(sz.height);
    }
    return FrameContext{
        .swapchain_image = TextureHandle{0},
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

    pool->release();
    ++frame_index_;
}

CommandBuffer* SoftwareDevice::AcquireCommandBuffer() { return cmd_buf_.get(); }

void SoftwareDevice::Submit(CommandBuffer*) { /* clear color already stashed */ }

void SoftwareDevice::Resize(int w, int h) {
    width_  = w; height_ = h;
    if (mtl_layer_) {
        mtl_layer_->setDrawableSize(CGSize{static_cast<CGFloat>(w),
                                           static_cast<CGFloat>(h)});
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

}  // namespace pt::rhi::sw

namespace pt::rhi {
std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w) {
    return std::make_unique<sw::SoftwareDevice>(w);
}
}

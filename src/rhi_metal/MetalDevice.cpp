// metal-cpp single-implementation TU. The PRIVATE_IMPLEMENTATION defines
// must appear in exactly one source file across the whole project.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalDevice.h"
#include "../core/Log.h"
#include "../core/Memory/Memory.h"

#include <fmt/format.h>
#include <cstring>

// Embedded shader (Slang -> MSL source, compiled to a library at runtime).
extern "C" {
extern const unsigned char shader_Clear_metal_data[];
extern const unsigned long shader_Clear_metal_size;
}

// Defined in MetalAttach.mm.
extern "C" void pt_metal_attach_layer(void* ns_window, void* metal_layer);

// Defined in src/app/Window.cpp -- bridges GLFWwindow* to NSWindow*.
extern "C" void* pt_window_native_cocoa(void* glfw_window);

namespace pt::rhi::mtl {

namespace {

NS::String* NsStr(const char* utf8) {
    return NS::String::string(utf8, NS::UTF8StringEncoding);
}

constexpr const char* kClearEntryPoint = "main_0";  // Slang renames `main`

}  // namespace

// ---------- MetalCommandBuffer -------------------------------------------

void MetalCommandBuffer::Reset(MTL::CommandBuffer* cb) {
    EndEncoderIfActive();
    mtl_cb_   = cb;
    encoder_  = nullptr;
    bound_pso_ = PipelineHandle{0};
    push_size_ = 0;
    for (auto& t : bound_tex_) t = TextureHandle{0};
    for (auto& b : bound_buf_) b = BufferHandle{0};
}

void MetalCommandBuffer::EnsureEncoder() {
    if (encoder_ == nullptr && mtl_cb_ != nullptr) {
        encoder_ = mtl_cb_->computeCommandEncoder();
    }
}

void MetalCommandBuffer::EndEncoderIfActive() {
    if (encoder_ != nullptr) {
        encoder_->endEncoding();
        encoder_ = nullptr;
    }
}

void MetalCommandBuffer::BindComputePipeline(PipelineHandle p) {
    bound_pso_ = p;
}
void MetalCommandBuffer::BindBuffer(std::uint32_t slot, BufferHandle b,
                                    std::size_t off) {
    if (slot < std::size(bound_buf_)) {
        bound_buf_[slot] = b;
        bound_buf_off_[slot] = off;
    }
}
void MetalCommandBuffer::BindStorageTexture(std::uint32_t slot, TextureHandle t) {
    if (slot < std::size(bound_tex_)) bound_tex_[slot] = t;
}
void MetalCommandBuffer::PushConstants(const void* data, std::size_t size) {
    if (size > sizeof(push_buf_)) size = sizeof(push_buf_);
    std::memcpy(push_buf_, data, size);
    push_size_ = size;
}

void MetalCommandBuffer::Dispatch(std::uint32_t gx, std::uint32_t gy,
                                  std::uint32_t gz) {
    if (mtl_cb_ == nullptr || !bound_pso_) return;
    EnsureEncoder();

    auto* pso = device_->LookupPipeline(bound_pso_);
    if (pso == nullptr) return;
    encoder_->setComputePipelineState(pso);

    // Storage textures.
    for (std::uint32_t i = 0; i < std::size(bound_tex_); ++i) {
        if (!bound_tex_[i]) continue;
        auto* tex = device_->LookupTexture(bound_tex_[i]);
        if (tex != nullptr) encoder_->setTexture(tex, i);
    }
    // Bound buffers.
    for (std::uint32_t i = 0; i < std::size(bound_buf_); ++i) {
        if (!bound_buf_[i]) continue;
        auto* buf = device_->LookupBuffer(bound_buf_[i]);
        if (buf != nullptr) encoder_->setBuffer(buf, bound_buf_off_[i], i);
    }
    // Push constants land in buffer slot 0 by Slang's convention for
    // [[vk::push_constant]] -> Metal mapping.
    if (push_size_ > 0) {
        encoder_->setBytes(push_buf_, push_size_, 0);
    }

    // dispatchThreads handles the remainder; pso threadExecutionWidth tells
    // us a sane threadgroup size.
    auto tew = pso->threadExecutionWidth();
    auto h   = pso->maxTotalThreadsPerThreadgroup() / tew;
    if (tew == 0) tew = 8;
    if (h   == 0) h   = 8;
    MTL::Size grid    = MTL::Size::Make(gx * 8, gy * 8, gz);
    MTL::Size tgsize  = MTL::Size::Make(tew, h, 1);
    encoder_->dispatchThreads(grid, tgsize);
}

// ---------- MetalDevice ---------------------------------------------------

MetalDevice::MetalDevice(const NativeWindowHandle& window) {
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);

    // GLFW window pointer is stored as `opaque`. The native NSWindow is
    // obtained via the existing Cocoa bridge in pt_app_window, but we
    // already have the GLFW handle here -- glfwGetCocoaWindow lives in
    // the Window layer. The Engine passes the GLFW pointer for software
    // and Metal, so look up NSWindow ourselves via the .mm helper.
    ns_window_ = pt_window_native_cocoa(window.opaque);

    width_  = window.width;
    height_ = window.height;

    device_ = MTL::CreateSystemDefaultDevice();
    if (device_ == nullptr) {
        LOG_ERROR("MTL::CreateSystemDefaultDevice returned null");
        return;
    }
    {
        auto* nm = device_->name();
        if (nm != nullptr) device_name_ = nm->utf8String();
    }
    LOG_INFO("Metal device: {}", device_name_);

    queue_ = device_->newCommandQueue();
    if (queue_ == nullptr) {
        LOG_ERROR("newCommandQueue failed");
        return;
    }

    // Create the CAMetalLayer and attach it to the NSWindow content view.
    layer_ = CA::MetalLayer::layer();
    layer_->setDevice(device_);
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer_->setFramebufferOnly(false);  // we need read-write storage access
    layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width_),
                                   static_cast<CGFloat>(height_)});

    pt_metal_attach_layer(ns_window_, layer_);

    // Compile the embedded MSL clear shader at runtime.
    {
        NS::Error* err = nullptr;
        auto* src = NsStr(reinterpret_cast<const char*>(shader_Clear_metal_data));
        auto* opts = MTL::CompileOptions::alloc()->init();
        auto* lib = device_->newLibrary(src, opts, &err);
        opts->release();
        if (lib == nullptr) {
            LOG_ERROR("Slang clear shader: newLibrary failed: {}",
                      err ? err->localizedDescription()->utf8String() : "?");
        } else {
            auto* fn = lib->newFunction(NsStr(kClearEntryPoint));
            if (fn == nullptr) {
                LOG_ERROR("entry '{}' not found in MSL", kClearEntryPoint);
            } else {
                NS::Error* psoErr = nullptr;
                auto* pso = device_->newComputePipelineState(fn, &psoErr);
                fn->release();
                if (pso != nullptr) {
                    std::lock_guard lock(resource_mutex_);
                    pipelines_.emplace(next_id_++, pso);
                } else {
                    LOG_ERROR("newComputePipelineState failed: {}",
                              psoErr ? psoErr->localizedDescription()->utf8String() : "?");
                }
            }
            lib->release();
        }
    }

    cmd_ = std::make_unique<MetalCommandBuffer>(this);
}

MetalDevice::~MetalDevice() {
    cmd_.reset();
    {
        std::lock_guard lock(resource_mutex_);
        for (auto& [_, p] : pipelines_) if (p) p->release();
        for (auto& [_, t] : textures_)  if (t) t->release();
        for (auto& [_, b] : buffers_)   if (b) b->release();
        pipelines_.clear();
        textures_.clear();
        buffers_.clear();
    }
    if (current_drawable_) { current_drawable_->release(); current_drawable_ = nullptr; }
    if (layer_)            { layer_->release();            layer_ = nullptr; }
    if (queue_)            { queue_->release();            queue_ = nullptr; }
    if (device_)           { device_->release();           device_ = nullptr; }
}

// ---- Resources -----------------------------------------------------------

BufferHandle MetalDevice::CreateBuffer(const BufferDesc& d) {
    if (device_ == nullptr) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    auto* buf = device_->newBuffer(d.size, MTL::ResourceStorageModeShared);
    if (buf == nullptr) return {0};
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    buffers_.emplace(id, buf);
    return BufferHandle{id};
}

TextureHandle MetalDevice::CreateTexture(const TextureDesc& d) {
    if (device_ == nullptr) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    auto* td = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, d.width, d.height, false);
    td->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto* tex = device_->newTexture(td);
    td->release();
    if (tex == nullptr) return {0};
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    textures_.emplace(id, tex);
    return TextureHandle{id};
}

PipelineHandle MetalDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    // For P3 the only kernel is "clear", which we built up front during
    // device construction. Future phases will add a bytecode payload.
    if (d.kernel_name == "clear") {
        std::lock_guard lock(resource_mutex_);
        for (auto& [id, p] : pipelines_) {
            if (p) return PipelineHandle{id};
        }
    }
    return {0};
}

void MetalDevice::DestroyBuffer(BufferHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = buffers_.find(h.id); it != buffers_.end()) {
        if (it->second) it->second->release();
        buffers_.erase(it);
    }
}
void MetalDevice::DestroyTexture(TextureHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = textures_.find(h.id); it != textures_.end()) {
        if (it->second) it->second->release();
        textures_.erase(it);
    }
}
void MetalDevice::DestroyPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    if (auto it = pipelines_.find(h.id); it != pipelines_.end()) {
        if (it->second) it->second->release();
        pipelines_.erase(it);
    }
}

// ---- Frame ---------------------------------------------------------------

FrameContext MetalDevice::BeginFrame() {
    if (frame_pool_) frame_pool_->release();
    frame_pool_ = NS::AutoreleasePool::alloc()->init();

    if (current_drawable_) { current_drawable_->release(); current_drawable_ = nullptr; }
    if (layer_) {
        // Pull current layer size into width/height (window may have resized).
        auto sz = layer_->drawableSize();
        width_  = static_cast<int>(sz.width);
        height_ = static_cast<int>(sz.height);
        current_drawable_ = layer_->nextDrawable();
        if (current_drawable_) current_drawable_->retain();
    }

    return FrameContext{
        .swapchain_image = TextureHandle{kSwapchainTextureId},
        .width  = static_cast<std::uint32_t>(width_),
        .height = static_cast<std::uint32_t>(height_),
        .frame_index = frame_index_,
    };
}

void MetalDevice::EndFrame(CommandBuffer* cb) {
    auto* mcb = static_cast<MetalCommandBuffer*>(cb);
    if (mcb) mcb->Reset(nullptr);  // ends any active encoder
    // Submit + present already happened via Submit() below.
    if (frame_pool_) { frame_pool_->release(); frame_pool_ = nullptr; }
    ++frame_index_;
}

CommandBuffer* MetalDevice::AcquireCommandBuffer() {
    if (queue_ == nullptr) return nullptr;
    auto* mtl_cb = queue_->commandBuffer();
    cmd_->Reset(mtl_cb);
    return cmd_.get();
}

void MetalDevice::Submit(CommandBuffer* cb) {
    auto* mcb = static_cast<MetalCommandBuffer*>(cb);
    if (mcb == nullptr || mcb->RawCmdBuf() == nullptr) return;
    mcb->Reset(mcb->RawCmdBuf());  // ends encoder if open
    auto* mtl_cb = mcb->RawCmdBuf();
    if (current_drawable_ != nullptr) {
        mtl_cb->presentDrawable(current_drawable_);
    }
    mtl_cb->commit();
    mcb->Reset(nullptr);
}

void MetalDevice::WaitIdle() {
    if (queue_ == nullptr) return;
    auto* cb = queue_->commandBuffer();
    cb->commit();
    cb->waitUntilCompleted();
}

void MetalDevice::Resize(int w, int h) {
    width_  = w; height_ = h;
    if (layer_) {
        layer_->setDrawableSize(CGSize{static_cast<CGFloat>(w),
                                       static_cast<CGFloat>(h)});
    }
}

std::size_t MetalDevice::CurrentAllocatedBytes() const {
    if (device_ == nullptr) return 0;
    return device_->currentAllocatedSize();
}

MTL::ComputePipelineState* MetalDevice::LookupPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = pipelines_.find(h.id);
    return it == pipelines_.end() ? nullptr : it->second;
}

MTL::Texture* MetalDevice::LookupTexture(TextureHandle h) {
    if (h.id == kSwapchainTextureId) {
        return current_drawable_ ? current_drawable_->texture() : nullptr;
    }
    std::lock_guard lock(resource_mutex_);
    auto it = textures_.find(h.id);
    return it == textures_.end() ? nullptr : it->second;
}

MTL::Buffer* MetalDevice::LookupBuffer(BufferHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = buffers_.find(h.id);
    return it == buffers_.end() ? nullptr : it->second;
}

}  // namespace pt::rhi::mtl

namespace pt::rhi {

std::unique_ptr<Device> CreateMetalDevice(const NativeWindowHandle& w) {
    return std::make_unique<mtl::MetalDevice>(w);
}

}  // namespace pt::rhi

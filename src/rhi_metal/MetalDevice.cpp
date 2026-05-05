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

// Embedded shaders (Slang -> MSL source, compiled to a library at runtime).
extern "C" {
extern const unsigned char shader_Clear_metal_data[];
extern const unsigned long shader_Clear_metal_size;
extern const unsigned char shader_Scene_metal_data[];
extern const unsigned long shader_Scene_metal_size;
extern const unsigned char shader_PathTrace_metal_data[];
extern const unsigned long shader_PathTrace_metal_size;
extern const unsigned char shader_MeshTrace_metal_data[];
extern const unsigned long shader_MeshTrace_metal_size;
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
    for (auto& t : bound_tex_)   t = TextureHandle{0};
    for (auto& b : bound_buf_)   b = BufferHandle{0};
    for (auto& a : bound_accel_) a = AccelStructHandle{0};
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
void MetalCommandBuffer::BindAccelStruct(std::uint32_t slot, AccelStructHandle a) {
    if (slot < std::size(bound_accel_)) bound_accel_[slot] = a;
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
    // Slang's MSL output puts the AS at [[buffer(0)]] and pushes the
    // push-constant block to [[buffer(1)]] when any AS is referenced;
    // when there's no AS, push lands at [[buffer(0)]]. We compensate
    // here so the same Engine binding code works for both cases.
    bool has_accel = false;
    AccelStructHandle accel_handle{0};
    for (auto& a : bound_accel_) {
        if (a.id != 0) { has_accel = true; accel_handle = a; break; }
    }

    std::uint32_t push_slot = has_accel ? 1u : 0u;
    if (push_size_ > 0) {
        encoder_->setBytes(push_buf_, push_size_, push_slot);
    }
    if (has_accel) {
        auto* as = device_->LookupAccelStruct(accel_handle);
        if (as != nullptr) {
            encoder_->setAccelerationStructure(as, 0);
            // useResource on every known AS so the BLAS instances
            // referenced by the TLAS are also visible to this encoder.
            device_->UseAllAccelStructs(encoder_);
        }
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
    layer_->retain();   // CA::MetalLayer::layer() returns autoreleased
    layer_->setDevice(device_);
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer_->setFramebufferOnly(false);  // we need read-write storage access
    layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width_),
                                   static_cast<CGFloat>(height_)});

    pt_metal_attach_layer(ns_window_, layer_);

    // Build all known compute pipelines up front.  P3 had only "clear";
    // P5 adds "scene". Each Slang source becomes one MTLLibrary +
    // MTLComputePipelineState, looked up later by kernel name.
    auto build_pso = [&](const char* kernel_name,
                         const unsigned char* src_data) {
        NS::Error* err = nullptr;
        auto* src  = NsStr(reinterpret_cast<const char*>(src_data));
        auto* opts = MTL::CompileOptions::alloc()->init();
        auto* lib  = device_->newLibrary(src, opts, &err);
        opts->release();
        if (lib == nullptr) {
            LOG_ERROR("Slang '{}' shader: newLibrary failed: {}", kernel_name,
                      err ? err->localizedDescription()->utf8String() : "?");
            return;
        }
        auto* fn = lib->newFunction(NsStr(kClearEntryPoint));  // Slang renames `main` -> `main_0`
        if (fn == nullptr) {
            LOG_ERROR("entry '{}' not found in MSL for '{}'",
                      kClearEntryPoint, kernel_name);
            lib->release();
            return;
        }
        NS::Error* psoErr = nullptr;
        auto* pso = device_->newComputePipelineState(fn, &psoErr);
        fn->release();
        lib->release();
        if (pso == nullptr) {
            LOG_ERROR("'{}' newComputePipelineState failed: {}", kernel_name,
                      psoErr ? psoErr->localizedDescription()->utf8String() : "?");
            return;
        }
        std::lock_guard lock(resource_mutex_);
        auto id = next_id_++;
        pipelines_.emplace(id, pso);
        named_pipelines_.emplace(kernel_name, id);
    };

    build_pso("clear",     shader_Clear_metal_data);
    build_pso("scene",     shader_Scene_metal_data);
    build_pso("pathtrace", shader_PathTrace_metal_data);
    build_pso("mesh",      shader_MeshTrace_metal_data);

    cmd_ = std::make_unique<MetalCommandBuffer>(this);
}

MetalDevice::~MetalDevice() {
    cmd_.reset();
    {
        std::lock_guard lock(resource_mutex_);
        for (auto& [_, p] : pipelines_) if (p) p->release();
        for (auto& [_, t] : textures_)  if (t) t->release();
        for (auto& [_, b] : buffers_)   if (b) b->release();
        for (auto& [_, a] : accels_)    if (a) a->release();
        pipelines_.clear();
        textures_.clear();
        buffers_.clear();
        accels_.clear();
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

    MTL::PixelFormat fmt = MTL::PixelFormatRGBA8Unorm;
    switch (d.format) {
        case TextureFormat::RGBA8_UNORM: fmt = MTL::PixelFormatRGBA8Unorm;       break;
        case TextureFormat::RGBA8_SRGB:  fmt = MTL::PixelFormatRGBA8Unorm_sRGB;  break;
        case TextureFormat::RGBA16F:     fmt = MTL::PixelFormatRGBA16Float;      break;
        case TextureFormat::RGBA32F:     fmt = MTL::PixelFormatRGBA32Float;      break;
        case TextureFormat::R32_UINT:    fmt = MTL::PixelFormatR32Uint;          break;
        default: break;
    }

    // texture2DDescriptor returns an AUTORELEASED descriptor.  Wrap in a
    // local pool so it gets cleaned up at function exit -- calling
    // release() on it directly was an over-release that crashed once the
    // implicit pool drained.  newTexture(...) is a `new`-prefixed call
    // and returns retained; we own it.
    auto* pool = NS::AutoreleasePool::alloc()->init();
    auto* td = MTL::TextureDescriptor::texture2DDescriptor(
        fmt, d.width, d.height, false);
    td->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto* tex = device_->newTexture(td);
    pool->release();
    if (tex == nullptr) return {0};
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    textures_.emplace(id, tex);
    return TextureHandle{id};
}

PipelineHandle MetalDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    std::lock_guard lock(resource_mutex_);
    std::string key(d.kernel_name);
    auto it = named_pipelines_.find(key);
    if (it == named_pipelines_.end()) return {0};
    return PipelineHandle{ it->second };
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

// ---- Acceleration structures ---------------------------------------------

AccelStructHandle MetalDevice::CreateBLAS(const BLASDesc& d) {
    if (device_ == nullptr || d.vertex_count == 0 || d.index_count == 0) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    auto* pool = NS::AutoreleasePool::alloc()->init();

    std::size_t vbytes = sizeof(float) * 3 * d.vertex_count;
    std::size_t ibytes = sizeof(std::uint32_t) * d.index_count;
    auto* vbuf = device_->newBuffer(d.vertex_positions, vbytes, MTL::ResourceStorageModeShared);
    auto* ibuf = device_->newBuffer(d.indices,          ibytes, MTL::ResourceStorageModeShared);

    auto* geom = MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
    geom->setVertexBuffer(vbuf);
    geom->setVertexBufferOffset(0);
    geom->setVertexStride(sizeof(float) * 3);
    geom->setIndexBuffer(ibuf);
    geom->setIndexBufferOffset(0);
    geom->setIndexType(MTL::IndexTypeUInt32);
    geom->setTriangleCount(d.index_count / 3);

    const NS::Object* descs[1] = { geom };
    NS::Array* geoms = NS::Array::array(descs, 1);
    auto* desc = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
    desc->setGeometryDescriptors(geoms);

    auto sizes = device_->accelerationStructureSizes(desc);
    auto* as = device_->newAccelerationStructure(sizes.accelerationStructureSize);
    auto* scratch = device_->newBuffer(sizes.buildScratchBufferSize,
                                        MTL::ResourceStorageModePrivate);

    auto* cb = queue_->commandBuffer();
    auto* enc = cb->accelerationStructureCommandEncoder();
    enc->buildAccelerationStructure(as, desc, scratch, 0);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    vbuf->release();
    ibuf->release();
    scratch->release();
    pool->release();

    if (as == nullptr) return {0};
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    accels_[id] = as;
    return AccelStructHandle{id};
}

AccelStructHandle MetalDevice::CreateTLAS(const TLASDesc& d) {
    if (device_ == nullptr || d.instances.empty()) return {0};
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    auto* pool = NS::AutoreleasePool::alloc()->init();

    std::vector<MTL::AccelerationStructure*> blas_array;
    blas_array.reserve(d.instances.size());

    struct InstDesc {
        MTL::PackedFloat4x3 transform;
        MTL::AccelerationStructureInstanceOptions options;
        std::uint32_t mask;
        std::uint32_t intersection_function_table_offset;
        std::uint32_t accel_struct_index;
    };
    std::vector<InstDesc> instance_buf;
    instance_buf.reserve(d.instances.size());

    for (std::uint32_t i = 0; i < d.instances.size(); ++i) {
        auto* as = LookupAccelStruct(d.instances[i].blas);
        if (as == nullptr) continue;
        blas_array.push_back(as);
        InstDesc id{};
        std::memcpy(&id.transform, d.instances[i].transform, sizeof(float) * 12);
        id.options = MTL::AccelerationStructureInstanceOptionOpaque;
        id.mask    = d.instances[i].mask;
        id.intersection_function_table_offset = 0;
        id.accel_struct_index                  = static_cast<std::uint32_t>(blas_array.size() - 1);
        instance_buf.push_back(id);
    }
    if (instance_buf.empty()) { pool->release(); return {0}; }

    auto* ibuf = device_->newBuffer(instance_buf.data(),
                                    instance_buf.size() * sizeof(InstDesc),
                                    MTL::ResourceStorageModeShared);

    NS::Array* nsBlasArr = NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(blas_array.data()),
        blas_array.size());

    auto* desc = MTL::InstanceAccelerationStructureDescriptor::descriptor();
    desc->setInstanceCount(static_cast<NS::UInteger>(instance_buf.size()));
    desc->setInstanceDescriptorBuffer(ibuf);
    desc->setInstanceDescriptorStride(sizeof(InstDesc));
    desc->setInstancedAccelerationStructures(nsBlasArr);

    auto sizes = device_->accelerationStructureSizes(desc);
    auto* tlas = device_->newAccelerationStructure(sizes.accelerationStructureSize);
    auto* scratch = device_->newBuffer(sizes.buildScratchBufferSize,
                                        MTL::ResourceStorageModePrivate);

    auto* cb = queue_->commandBuffer();
    auto* enc = cb->accelerationStructureCommandEncoder();
    for (auto* b : blas_array) enc->useResource(b, MTL::ResourceUsageRead);
    enc->buildAccelerationStructure(tlas, desc, scratch, 0);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();

    ibuf->release();
    scratch->release();
    pool->release();

    if (tlas == nullptr) return {0};
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    accels_[id] = tlas;
    return AccelStructHandle{id};
}

void MetalDevice::DestroyAccelStruct(AccelStructHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = accels_.find(h.id);
    if (it == accels_.end()) return;
    if (it->second) it->second->release();
    accels_.erase(it);
}

MTL::AccelerationStructure* MetalDevice::LookupAccelStruct(AccelStructHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = accels_.find(h.id);
    return (it == accels_.end()) ? nullptr : it->second;
}

void MetalDevice::UseAllAccelStructs(MTL::ComputeCommandEncoder* enc) {
    if (enc == nullptr) return;
    std::lock_guard lock(resource_mutex_);
    for (auto& [_, a] : accels_) {
        if (a != nullptr) enc->useResource(a, MTL::ResourceUsageRead);
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

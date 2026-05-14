// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// metal-cpp single-implementation TU. The PRIVATE_IMPLEMENTATION defines
// must appear in exactly one source file across the whole project.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalDevice.h"
#include "MetalSvgfDenoiser.h"
#include "../core/Log.h"
#include "../core/Memory/Memory.h"

#include <fmt/format.h>
#include <cstring>

// Embedded shader (the unified PathTrace kernel: analytic primitives +
// triangle meshes via TLAS + materials + accumulation, all in one).
extern "C" {
extern const unsigned char shader_PathTrace_metal_data[];
extern const unsigned long shader_PathTrace_metal_size;
extern const unsigned char shader_AutoExposure_metal_data[];
extern const unsigned long shader_AutoExposure_metal_size;
extern const unsigned char shader_Tonemap_metal_data[];
extern const unsigned long shader_Tonemap_metal_size;
extern const unsigned char shader_BloomDown_metal_data[];
extern const unsigned long shader_BloomDown_metal_size;
extern const unsigned char shader_BloomUp_metal_data[];
extern const unsigned long shader_BloomUp_metal_size;
extern const unsigned char shader_PerfOverlay_metal_data[];
extern const unsigned long shader_PerfOverlay_metal_size;
}

// MetalFXDenoiser.mm: ObjC++ shim around MTLFXTemporalDenoisedScaler.
extern "C" void* pt_metalfx_create(void* mtl_device, std::uint32_t w, std::uint32_t h);
extern "C" void  pt_metalfx_destroy(void* scaler);
extern "C" void  pt_metalfx_encode(void* scaler, void* mtl_cb,
                                    void* color_in, void* depth_in,
                                    void* motion_in, void* color_out,
                                    float jitter_x, float jitter_y,
                                    const float* world_to_view_4x4,
                                    const float* view_to_clip_4x4,
                                    int reset);

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

void MetalCommandBuffer::FlushEncoder() { EndEncoderIfActive(); }

void MetalCommandBuffer::BindComputePipeline(PipelineHandle p) {
    bound_pso_ = p;
    // Clear all binding state so each pipeline starts fresh. Without
    // this, the previous pipeline's leftover buffer/AS slots would
    // skew the push-constant slot calculation in Dispatch() (which is
    // computed as max-bound-buffer + 1) and the new pipeline's push
    // constants would land at the wrong Metal buffer index. Anything
    // the new pipeline needs must be re-bound after this call.
    push_size_ = 0;
    for (auto& t : bound_tex_)     t = TextureHandle{0};
    for (auto& b : bound_buf_)     b = BufferHandle{0};
    for (auto& o : bound_buf_off_) o = 0;
    for (auto& a : bound_accel_)   a = AccelStructHandle{0};
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
    // Slang's MSL output assigns Metal buffer slots in this order:
    //   AS (if present)        -> slot 0
    //   storage buffers        -> next slots, in declaration order
    //   push constant block    -> next slot after all the above
    // The Engine binds AS via BindAccelStruct and buffers via
    // BindBuffer(slot, ...) using the SAME slot number Slang assigned
    // (no abstraction here yet); push_slot is computed as max(bound
    // buffer/AS slot) + 1 so it lands wherever Slang put it.
    bool has_accel = false;
    AccelStructHandle accel_handle{0};
    for (auto& a : bound_accel_) {
        if (a.id != 0) { has_accel = true; accel_handle = a; break; }
    }

    std::int32_t max_slot = -1;   // -1 = nothing yet
    if (has_accel) max_slot = 0;
    for (std::uint32_t i = 0; i < std::size(bound_buf_); ++i) {
        if (bound_buf_[i]) max_slot = std::max(max_slot, static_cast<std::int32_t>(i));
    }

    if (has_accel) {
        auto* as = device_->LookupAccelStruct(accel_handle);
        if (as != nullptr) {
            encoder_->setAccelerationStructure(as, 0);
            device_->UseAllAccelStructs(encoder_);
        }
    }

    std::uint32_t push_slot = (max_slot < 0) ? 0u
                                             : static_cast<std::uint32_t>(max_slot + 1);
    if (push_size_ > 0) {
        encoder_->setBytes(push_buf_, push_size_, push_slot);
    }

    // Threadgroup size MUST match the kernel's [numthreads(...)]
    // declaration. Every compute kernel in demont declares
    // [numthreads(8, 8, 1)] (PathTrace, AutoExposure, Tonemap,
    // BloomDown, BloomUp), so hardcode the matching shape here.
    //
    // The previous version computed tgsize from the pipeline's
    // threadExecutionWidth (32 on Apple GPUs) and
    // maxTotalThreadsPerThreadgroup (64 here), giving (32, 2, 1).
    // Dispatching grid (8,8,1) with tgsize (32,2,1) fires FOUR
    // threadgroups (ceil(grid/tgsize) = (1,4,1)), not the one the
    // kernel expects. Pixel-per-thread kernels like PathTrace
    // tolerate this because they bounds-check tid against the
    // image dimensions, but reduction kernels like AutoExposure
    // run their full reduction in each threadgroup and race-write
    // to a single output slot, getting nondeterministic results
    // skewed toward whichever threadgroup happened to finish last.
    // This was the actual reason auto-exposure on Mac stabilized
    // at a different value than Win for the same scene.
    //
    // grid = (gx*8, gy*8, gz) is already pre-multiplied by 8 by the
    // engine's caller convention (cb->Dispatch(gx, gy, gz) means
    // "dispatch gx*gy*gz threadgroups of [numthreads]"). With
    // tgsize (8,8,1) Metal computes threadgroupCount = (gx, gy, gz)
    // exactly as intended.
    MTL::Size grid    = MTL::Size::Make(gx * 8, gy * 8, gz);
    MTL::Size tgsize  = MTL::Size::Make(8, 8, 1);
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
    // sRGB swapchain: the shader writes ACES-tonemapped *linear* [0,1]
    // values; the GPU does the linear -> sRGB gamma encode for free on
    // store, so the display sees properly gamma-encoded pixels. Without
    // this we had to either pow(c, 1/2.2) in the shader or live with a
    // dim/desaturated look on the no-denoiser path (MetalFX did the
    // gamma internally, hence the asymmetry).
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    layer_->setFramebufferOnly(false);  // we need read-write storage access
    layer_->setDrawableSize(CGSize{static_cast<CGFloat>(width_),
                                   static_cast<CGFloat>(height_)});

    pt_metal_attach_layer(ns_window_, layer_);

    // Build all known compute pipelines up front.  P3 had only "clear";
    // P5 adds "scene". Each Slang source becomes one MTLLibrary +
    // MTLComputePipelineState, looked up later by kernel name.
    //
    // NB: the embedded MSL blob is a raw byte array, NOT NUL-terminated
    // (cmake/EmbedFile.cmake emits `0x..,` bytes only). We MUST pass the
    // explicit size to NSString so it doesn't read whatever rodata
    // happens to follow in the linker layout -- when the trailing bytes
    // happened to start with high-bit unicode (e.g. `0xD8 0xAE` -> `خ`)
    // the Metal compiler choked with `unknown type name 'خ'`.
    auto build_pso = [&](const char* kernel_name,
                         const unsigned char* src_data,
                         unsigned long        src_size) {
        // metal-cpp's NS::String::string only takes a C string, so we
        // build a one-shot NUL-terminated copy here. Cheap (44 KB once
        // at backend init) and avoids reaching into the ObjC runtime
        // for initWithBytes:length:encoding:.
        std::string nul_terminated(reinterpret_cast<const char*>(src_data),
                                   src_size);
        NS::Error* err = nullptr;
        auto* src  = NsStr(nul_terminated.c_str());
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

    build_pso("pathtrace",   shader_PathTrace_metal_data,    shader_PathTrace_metal_size);
    build_pso("autoexpose",  shader_AutoExposure_metal_data, shader_AutoExposure_metal_size);
    build_pso("tonemap",     shader_Tonemap_metal_data,      shader_Tonemap_metal_size);
    build_pso("bloom_down",  shader_BloomDown_metal_data,    shader_BloomDown_metal_size);
    build_pso("bloom_up",    shader_BloomUp_metal_data,      shader_BloomUp_metal_size);
    build_pso("perfoverlay", shader_PerfOverlay_metal_data,  shader_PerfOverlay_metal_size);

    cmd_ = std::make_unique<MetalCommandBuffer>(this);
}

MetalDevice::~MetalDevice() {
    cmd_.reset();
    // Tear down the SVGF denoiser BEFORE we release device_ / queue_ /
    // pipelines_ -- its textures + pipelines were allocated against
    // device_ and need to release while it's still valid.
    svgf_denoiser_.reset();
    if (metalfx_scaler_) { pt_metalfx_destroy(metalfx_scaler_); metalfx_scaler_ = nullptr; }
    if (svgf_metalfx_intermediate_) {
        svgf_metalfx_intermediate_->release();
        svgf_metalfx_intermediate_   = nullptr;
        svgf_metalfx_intermediate_w_ = 0;
        svgf_metalfx_intermediate_h_ = 0;
    }
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

void MetalDevice::Denoise(const DenoiseDesc& d) {
    if (device_ == nullptr) return;
    if (cmd_ == nullptr || cmd_->RawCmdBuf() == nullptr) {
        // No active command buffer -- caller forgot to AcquireCommandBuffer
        // before Denoise. Silently skip; the path tracer's swapchain write
        // will still be visible.
        return;
    }
    auto* color_in  = LookupTexture(d.color_in);
    auto* depth_in  = LookupTexture(d.depth_in);
    auto* motion_in = LookupTexture(d.motion_in);
    auto* color_out = LookupTexture(d.output);
    if (color_in == nullptr || depth_in == nullptr ||
        motion_in == nullptr || color_out == nullptr) {
        LOG_WARN("MetalDevice::Denoise: missing input textures");
        return;
    }
    const std::uint32_t w = static_cast<std::uint32_t>(color_in->width());
    const std::uint32_t h = static_cast<std::uint32_t>(color_in->height());

    // The compute encoder (if any) must be ended first since both the
    // MetalFX scaler and our SVGF chain create their own encoders.
    cmd_->FlushEncoder();

    const bool kind_is_svgf       = (d.kind == DenoiseDesc::Kind::Svgf);
    const bool kind_is_svgf_chain = (d.kind == DenoiseDesc::Kind::SvgfMetalFx);
    if (kind_is_svgf || kind_is_svgf_chain) {
        // In-house SVGF / atrous chain. normal_in is required (the
        // engine allocates it whenever r_denoiser is svgf_*). The
        // shaders treat a zero-normal sentinel as "no surface" so a
        // missing normal isn't catastrophic, but the spatial filter
        // would lose its normal-weighted edge stop.
        auto* normal_in = LookupTexture(d.normal_in);
        if (normal_in == nullptr) {
            LOG_WARN("MetalDevice::Denoise(svgf): normal_in is null -- engine "
                     "should allocate normal_tex for r_denoiser=svgf_*");
            return;
        }
        if (!svgf_denoiser_) {
            svgf_denoiser_ = std::make_unique<MetalSvgfDenoiser>(this);
            if (!svgf_denoiser_->Init()) {
                LOG_ERROR("MetalSvgfDenoiser::Init() failed");
                svgf_denoiser_.reset();
                return;
            }
        }
        if (!svgf_denoiser_->ResizeTextures(w, h)) {
            LOG_ERROR("MetalSvgfDenoiser::ResizeTextures({}x{}) failed", w, h);
            return;
        }
        const bool atrous_enabled =
            (d.quality == DenoiseDesc::Quality::Atrous);

        // For the SVGF -> MetalFX chain, route SVGF's output through
        // an intermediate texture so MetalFX can read it as its color
        // input. For plain SVGF we write straight to the caller's
        // output texture as before.
        MTL::Texture* svgf_target = color_out;
        if (kind_is_svgf_chain) {
            if (svgf_metalfx_intermediate_ == nullptr ||
                svgf_metalfx_intermediate_w_ != w ||
                svgf_metalfx_intermediate_h_ != h) {
                if (svgf_metalfx_intermediate_ != nullptr) {
                    svgf_metalfx_intermediate_->release();
                    svgf_metalfx_intermediate_ = nullptr;
                }
                NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
                MTL::TextureDescriptor* td =
                    MTL::TextureDescriptor::texture2DDescriptor(
                        MTL::PixelFormatRGBA16Float, w, h, /*mipmapped=*/false);
                td->setUsage(MTL::TextureUsageShaderRead |
                             MTL::TextureUsageShaderWrite);
                td->setStorageMode(MTL::StorageModePrivate);
                svgf_metalfx_intermediate_ = device_->newTexture(td);
                pool->release();
                if (svgf_metalfx_intermediate_ == nullptr) {
                    LOG_ERROR("MetalDevice::Denoise(svgf_metalfx): "
                              "intermediate texture alloc failed at {}x{}", w, h);
                    return;
                }
                svgf_metalfx_intermediate_w_ = w;
                svgf_metalfx_intermediate_h_ = h;
            }
            svgf_target = svgf_metalfx_intermediate_;
        }

        svgf_denoiser_->Encode(cmd_->RawCmdBuf(),
                                color_in, depth_in, motion_in, normal_in,
                                svgf_target,
                                d.reset_history, atrous_enabled,
                                d.atrous_passes);

        if (!kind_is_svgf_chain) return;

        // Chain mode: MetalFX TemporalDenoisedScaler now reads the
        // SVGF-denoised intermediate and applies its ML TAA + edge
        // refinement on top, writing into the caller's color_out.
        // SVGF already ended its compute encoder; pt_metalfx_encode
        // opens its own.
        if (metalfx_scaler_ == nullptr ||
            metalfx_width_  != w ||
            metalfx_height_ != h) {
            if (metalfx_scaler_) {
                pt_metalfx_destroy(metalfx_scaler_);
                metalfx_scaler_ = nullptr;
            }
            metalfx_scaler_ = pt_metalfx_create(static_cast<void*>(device_), w, h);
            if (metalfx_scaler_ == nullptr) {
                LOG_ERROR("MetalDevice::Denoise(svgf_metalfx): MetalFX scaler "
                          "create failed (size {}x{})", w, h);
                return;
            }
            metalfx_width_  = w;
            metalfx_height_ = h;
        }
        pt_metalfx_encode(metalfx_scaler_,
                          static_cast<void*>(cmd_->RawCmdBuf()),
                          static_cast<void*>(svgf_metalfx_intermediate_),
                          static_cast<void*>(depth_in),
                          static_cast<void*>(motion_in),
                          static_cast<void*>(color_out),
                          d.jitter_x, d.jitter_y,
                          d.world_to_view, d.view_to_clip,
                          d.reset_history ? 1 : 0);
        return;
    }

    // MetalFX path (Kind::MetalFX, or any backwards-compat caller that
    // left kind defaulted on Metal -- the engine sets it explicitly).
    if (metalfx_scaler_ == nullptr ||
        metalfx_width_  != w ||
        metalfx_height_ != h) {
        if (metalfx_scaler_) { pt_metalfx_destroy(metalfx_scaler_); metalfx_scaler_ = nullptr; }
        metalfx_scaler_ = pt_metalfx_create(static_cast<void*>(device_), w, h);
        if (metalfx_scaler_ == nullptr) {
            LOG_ERROR("MetalFX scaler creation failed (size {}x{})", w, h);
            return;
        }
        metalfx_width_  = w;
        metalfx_height_ = h;
    }

    pt_metalfx_encode(metalfx_scaler_,
                      static_cast<void*>(cmd_->RawCmdBuf()),
                      static_cast<void*>(color_in),
                      static_cast<void*>(depth_in),
                      static_cast<void*>(motion_in),
                      static_cast<void*>(color_out),
                      d.jitter_x, d.jitter_y,
                      d.world_to_view, d.view_to_clip,
                      d.reset_history ? 1 : 0);
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
        case TextureFormat::R32F:        fmt = MTL::PixelFormatR32Float;         break;
        case TextureFormat::RG16F:       fmt = MTL::PixelFormatRG16Float;        break;
        case TextureFormat::RG32F:       fmt = MTL::PixelFormatRG32Float;        break;
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

void MetalDevice::WriteBuffer(BufferHandle h, const void* src, std::size_t size,
                              std::size_t dst_offset) {
    if (src == nullptr || size == 0) return;
    auto* buf = LookupBuffer(h);
    if (buf == nullptr) return;
    auto* dst = static_cast<std::uint8_t*>(buf->contents());
    if (dst == nullptr) return;   // not shared / not CPU-mapped
    std::memcpy(dst + dst_offset, src, size);
}

bool MetalDevice::WriteTexture(TextureHandle h, const void* src, std::size_t src_size) {
    if (src == nullptr || src_size == 0) return false;
    auto* tex = LookupTexture(h);
    if (tex == nullptr) return false;
    if (tex->storageMode() == MTL::StorageModePrivate) return false;
    const std::size_t w   = static_cast<std::size_t>(tex->width());
    const std::size_t hgt = static_cast<std::size_t>(tex->height());
    std::size_t bpp = 0;
    switch (tex->pixelFormat()) {
        case MTL::PixelFormatRGBA32Float: bpp = 16; break;
        case MTL::PixelFormatRGBA16Float: bpp = 8;  break;
        case MTL::PixelFormatRGBA8Unorm:
        case MTL::PixelFormatRGBA8Unorm_sRGB:
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB: bpp = 4; break;
        case MTL::PixelFormatR32Float:    bpp = 4;  break;
        case MTL::PixelFormatRG16Float:   bpp = 4;  break;
        default: return false;
    }
    if (src_size != w * hgt * bpp) return false;
    MTL::Region region = MTL::Region::Make2D(0, 0, w, hgt);
    tex->replaceRegion(region, 0, src, w * bpp);
    return true;
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
        // The public TLASInstance.transform is row-major 3x4 (column 3 is
        // translation), but MTLPackedFloat4x3 stores 4 columns of float3
        // -- column-major. Flat memcpy garbles the matrix (it produces a
        // singular 3x3 with translation in the wrong slot), so every ray
        // misses. Transpose explicitly here.
        const float* src = d.instances[i].transform;
        float*       dst = reinterpret_cast<float*>(&id.transform);
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 3; ++r) {
                dst[c * 3 + r] = src[r * 4 + c];
            }
        }
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

bool MetalDevice::ReadbackTexture(TextureHandle h, void* dst, std::size_t dst_size,
                                  std::uint32_t* out_w, std::uint32_t* out_h) {
    if (device_ == nullptr || dst == nullptr) return false;
    auto* src = LookupTexture(h);
    if (src == nullptr) return false;

    const std::uint32_t w = static_cast<std::uint32_t>(src->width());
    const std::uint32_t hgt = static_cast<std::uint32_t>(src->height());
    if (out_w) *out_w = w;
    if (out_h) *out_h = hgt;

    // Bytes-per-pixel from format. Caller must ensure dst_size matches.
    std::size_t bpp = 0;
    switch (src->pixelFormat()) {
        case MTL::PixelFormatRGBA32Float: bpp = 16; break;
        case MTL::PixelFormatRGBA16Float: bpp = 8;  break;
        case MTL::PixelFormatRGBA8Unorm:
        case MTL::PixelFormatRGBA8Unorm_sRGB:
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB: bpp = 4; break;
        case MTL::PixelFormatR32Float:    bpp = 4;  break;
        case MTL::PixelFormatRG16Float:   bpp = 4;  break;
        default: return false;
    }
    if (dst_size < std::size_t(w) * hgt * bpp) return false;

    // Storage mode determines whether we can getBytes directly. Shared
    // (the default for our CreateTexture) is fine. Private requires a
    // blit to a managed-storage temp first.
    bool need_temp = (src->storageMode() == MTL::StorageModePrivate);

    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    auto* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Texture* readable = src;
    if (need_temp) {
        auto* td = MTL::TextureDescriptor::texture2DDescriptor(
            src->pixelFormat(), w, hgt, false);
        td->setUsage(MTL::TextureUsageShaderRead);
        td->setStorageMode(MTL::StorageModeShared);
        readable = device_->newTexture(td);
        if (readable == nullptr) { pool->release(); return false; }

        auto* cb = queue_->commandBuffer();
        auto* enc = cb->blitCommandEncoder();
        MTL::Origin origin = {0, 0, 0};
        MTL::Size   size   = {w, hgt, 1};
        enc->copyFromTexture(src, 0, 0, origin, size, readable, 0, 0, origin);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    MTL::Region region = MTL::Region::Make2D(0, 0, w, hgt);
    readable->getBytes(dst, w * bpp, region, 0);

    if (need_temp) readable->release();
    pool->release();
    return true;
}

bool MetalDevice::ReadbackBuffer(BufferHandle h, void* dst, std::size_t bytes) {
    if (device_ == nullptr || dst == nullptr || bytes == 0) return false;
    auto* buf = LookupBuffer(h);
    if (buf == nullptr) return false;
    if (buf->length() < bytes) return false;

    // Engine MTL::Buffer is allocated with StorageModeShared (see
    // CreateBuffer above), so contents() points at host-visible memory
    // that's coherent with the GPU. To make sure any pending GPU
    // writes (e.g. AutoExposure.slang updating exposure_state) are
    // committed before we read, drain the queue with a no-op submit.
    if (queue_ != nullptr) {
        auto* cb = queue_->commandBuffer();
        cb->commit();
        cb->waitUntilCompleted();
    }
    std::memcpy(dst, buf->contents(), bytes);
    return true;
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

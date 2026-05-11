// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// See MetalSvgfDenoiser.h for the per-frame dispatch contract.

#include "MetalSvgfDenoiser.h"

#include "MetalDevice.h"
#include "../core/Log.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstring>
#include <string>

// Embedded MSL blobs for the SVGF kernels. CMake builds these via
// pt_compile_slang(... TARGETS metal) -- same Slang sources the Vulkan
// backend uses, cross-compiled to MSL source which Metal turns into a
// MTLLibrary at runtime via newLibrary().
extern "C" {
extern const unsigned char shader_DenoiseTemporal_metal_data[];
extern const unsigned long shader_DenoiseTemporal_metal_size;
extern const unsigned char shader_DenoiseAtrous_metal_data[];
extern const unsigned long shader_DenoiseAtrous_metal_size;
}

namespace pt::rhi::mtl {

namespace {

// 32-byte push struct shared by both denoise kernels. Field names are
// kernel-specific (see DenoiseTemporal.slang / DenoiseAtrous.slang) but
// the layout matches what DenoisePush in VulkanDenoiser.cpp emits, so
// the Slang MSL emission for the [[vk::push_constant]] cbuffer maps
// cleanly onto this struct.
struct DenoisePush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t step_or_reset;
    std::uint32_t pad0;
    float         a;     // temporal: depth_tolerance; atrous: sigma_depth
    float         b;     // temporal: normal_tolerance; atrous: sigma_normal
    float         c;     // temporal: min_alpha; atrous: sigma_color
    float         pad1;
};
static_assert(sizeof(DenoisePush) == 32, "DenoisePush layout");

constexpr const char* kEntryPoint = "main_0";  // Slang renames `main`

MTL::ComputePipelineState* BuildPso(MTL::Device* dev,
                                    const unsigned char* src_data,
                                    unsigned long        src_size,
                                    const char*          label) {
    // EmbedFile.cmake emits a non-NUL-terminated byte array; copy into
    // a NUL-terminated std::string before handing the C string to
    // NS::String::string (same workaround the PathTrace path uses in
    // MetalDevice.cpp).
    std::string nul_terminated(reinterpret_cast<const char*>(src_data), src_size);
    NS::String* src  = NS::String::string(nul_terminated.c_str(),
                                          NS::UTF8StringEncoding);
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    NS::Error* err = nullptr;
    MTL::Library* lib = dev->newLibrary(src, opts, &err);
    opts->release();
    if (lib == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: newLibrary({}) failed: {}", label,
                  err ? err->localizedDescription()->utf8String() : "?");
        return nullptr;
    }
    NS::String* entry = NS::String::string(kEntryPoint, NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(entry);
    if (fn == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: entry '{}' not found in {}",
                  kEntryPoint, label);
        lib->release();
        return nullptr;
    }
    NS::Error* psoErr = nullptr;
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &psoErr);
    fn->release();
    lib->release();
    if (pso == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: newComputePipelineState({}) failed: {}",
                  label,
                  psoErr ? psoErr->localizedDescription()->utf8String() : "?");
        return nullptr;
    }
    return pso;
}

MTL::Texture* MakeTexture(MTL::Device* dev,
                          MTL::PixelFormat fmt,
                          std::uint32_t w, std::uint32_t h) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(
        fmt, w, h, /*mipmapped=*/false);
    td->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    td->setStorageMode(MTL::StorageModePrivate);
    MTL::Texture* t = dev->newTexture(td);
    pool->release();
    return t;
}

}  // namespace

MetalSvgfDenoiser::~MetalSvgfDenoiser() {
    DestroyAll();
}

void MetalSvgfDenoiser::DestroyAll() {
    DestroyTextures();
    if (dummy_color_)  { dummy_color_->release();  dummy_color_  = nullptr; }
    if (dummy_motion_) { dummy_motion_->release(); dummy_motion_ = nullptr; }
    if (temporal_pso_) { temporal_pso_->release(); temporal_pso_ = nullptr; }
    if (atrous_pso_)   { atrous_pso_->release();   atrous_pso_   = nullptr; }
    ready_ = false;
}

void MetalSvgfDenoiser::DestroyTextures() {
    auto rel = [](MTL::Texture*& t) { if (t) { t->release(); t = nullptr; } };
    rel(history_a_);
    rel(history_b_);
    rel(depth_history_a_);
    rel(depth_history_b_);
    rel(normal_history_a_);
    rel(normal_history_b_);
    rel(atrous_a_);
    rel(atrous_b_);
    cached_w_ = 0;
    cached_h_ = 0;
    needs_history_clear_ = true;
}

bool MetalSvgfDenoiser::Init() {
    if (ready_) return true;
    if (device_ == nullptr) return false;
    MTL::Device* dev = device_->RawDevice();
    if (dev == nullptr) return false;

    temporal_pso_ = BuildPso(dev, shader_DenoiseTemporal_metal_data,
                              shader_DenoiseTemporal_metal_size,
                              "DenoiseTemporal");
    atrous_pso_   = BuildPso(dev, shader_DenoiseAtrous_metal_data,
                              shader_DenoiseAtrous_metal_size,
                              "DenoiseAtrous");
    if (temporal_pso_ == nullptr || atrous_pso_ == nullptr) {
        DestroyAll();
        return false;
    }

    // 1x1 placeholders for the atrous shader's declared-but-unread
    // slots 1 (color_history_unused, RGBA16F) and 3 (motion_unused,
    // RG16F). The MSL emission still expects valid textures at those
    // slots; binding nullptr would either crash or give validation
    // errors. RGBA16F + RG16F match the declared formats.
    dummy_color_  = MakeTexture(dev, MTL::PixelFormatRGBA16Float, 1, 1);
    dummy_motion_ = MakeTexture(dev, MTL::PixelFormatRG16Float,   1, 1);
    if (dummy_color_ == nullptr || dummy_motion_ == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: 1x1 placeholder texture alloc failed");
        DestroyAll();
        return false;
    }

    ready_ = true;
    LOG_INFO("MetalSvgfDenoiser: SVGF compute pipelines (temporal + atrous) ready");
    return true;
}

bool MetalSvgfDenoiser::ResizeTextures(std::uint32_t w, std::uint32_t h) {
    if (w == 0 || h == 0) return false;
    if (cached_w_ == w && cached_h_ == h && history_a_ != nullptr) {
        return true;
    }
    DestroyTextures();
    MTL::Device* dev = device_->RawDevice();
    if (dev == nullptr) return false;

    history_a_        = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    history_b_        = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    depth_history_a_  = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);
    depth_history_b_  = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);
    normal_history_a_ = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    normal_history_b_ = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    atrous_a_         = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    atrous_b_         = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);

    if (history_a_ == nullptr || history_b_ == nullptr ||
        depth_history_a_ == nullptr || depth_history_b_ == nullptr ||
        normal_history_a_ == nullptr || normal_history_b_ == nullptr ||
        atrous_a_ == nullptr || atrous_b_ == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: scratch texture alloc failed at {}x{}", w, h);
        DestroyTextures();
        return false;
    }
    cached_w_ = w;
    cached_h_ = h;
    // History textures are MTLStorageModePrivate -- contents undefined
    // until first write. Force the temporal pass into reset_history
    // mode on the next Encode so we don't read garbage.
    needs_history_clear_ = true;
    return true;
}

void MetalSvgfDenoiser::Encode(MTL::CommandBuffer* cb,
                                MTL::Texture*       color_in,
                                MTL::Texture*       depth_in,
                                MTL::Texture*       motion_in,
                                MTL::Texture*       normal_in,
                                MTL::Texture*       output,
                                bool                reset_history,
                                bool                atrous_enabled) {
    if (!ready_ || cb == nullptr) return;
    if (color_in == nullptr || depth_in == nullptr || motion_in == nullptr ||
        normal_in == nullptr || output == nullptr) {
        LOG_WARN("MetalSvgfDenoiser::Encode: missing input textures");
        return;
    }
    if (cached_w_ == 0 || cached_h_ == 0) {
        LOG_WARN("MetalSvgfDenoiser::Encode: textures not yet sized");
        return;
    }

    // Frame-parity ping-pong: history read = parity-side, history write
    // = other side. Mirror of VulkanNrdDenoiser frame_parity_.
    const bool parity_is_a = (frame_parity_ == 0);
    MTL::Texture* hist_read           = parity_is_a ? history_a_         : history_b_;
    MTL::Texture* hist_write          = parity_is_a ? history_b_         : history_a_;
    MTL::Texture* depth_hist_read     = parity_is_a ? depth_history_a_   : depth_history_b_;
    MTL::Texture* depth_hist_write    = parity_is_a ? depth_history_b_   : depth_history_a_;
    MTL::Texture* normal_hist_read    = parity_is_a ? normal_history_a_  : normal_history_b_;
    MTL::Texture* normal_hist_write   = parity_is_a ? normal_history_b_  : normal_history_a_;

    const bool effective_reset = reset_history || needs_history_clear_;
    needs_history_clear_ = false;

    const std::uint32_t w = cached_w_;
    const std::uint32_t h = cached_h_;
    MTL::Size groups  = MTL::Size::Make((w + 7) / 8, (h + 7) / 8, 1);
    MTL::Size tgsize  = MTL::Size::Make(8, 8, 1);

    // ---- Pass 1: temporal accumulate (compute encoder) -------------
    {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(temporal_pso_);
        // Slang MSL emission for [[vk::binding(N, 0)]] storage images
        // assigns [[texture(N)]] in declaration order (no buffers
        // between them in the denoise shaders), so we bind by the
        // shader's slot number directly.
        enc->setTexture(color_in,         0);
        enc->setTexture(hist_read,        1);
        enc->setTexture(depth_in,         2);
        enc->setTexture(motion_in,        3);
        enc->setTexture(normal_in,        4);
        enc->setTexture(hist_write,       5);
        enc->setTexture(depth_hist_read,  6);
        enc->setTexture(normal_hist_read, 7);

        DenoisePush p{};
        p.width         = w;
        p.height        = h;
        p.step_or_reset = effective_reset ? 1u : 0u;
        p.a             = 0.10f;   // depth_tolerance (relative)
        p.b             = 0.85f;   // normal_tolerance (cos angle)
        p.c             = 0.10f;   // min_alpha (steady-state blend)
        // [[vk::push_constant]] cbuffer Push lands at [[buffer(0)]] on
        // MSL since these shaders declare no StructuredBuffers.
        enc->setBytes(&p, sizeof(p), 0);
        enc->dispatchThreadgroups(groups, tgsize);
        enc->endEncoding();
    }

    // ---- Depth + normal history copy (blit encoder) ----------------
    // Mirrors the vkCmdCopyImage block in VulkanNrdDenoiser::Encode.
    // The next frame's temporal pass reads these as the "previous
    // frame's surfaces" for reprojection validation.
    {
        MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
        MTL::Origin org = MTL::Origin::Make(0, 0, 0);
        MTL::Size   sz  = MTL::Size::Make(w, h, 1);
        blit->copyFromTexture(depth_in,  0, 0, org, sz,
                              depth_hist_write,  0, 0, org);
        blit->copyFromTexture(normal_in, 0, 0, org, sz,
                              normal_hist_write, 0, 0, org);

        if (!atrous_enabled) {
            // svgf_basic path: bypass the spatial filter entirely. The
            // temporal kernel already wrote hist_write; copy that
            // result into the caller's output. Both textures are
            // RGBA16F at (w, h) so it's a same-format blit.
            blit->copyFromTexture(hist_write, 0, 0, org, sz,
                                  output,     0, 0, org);
        }
        blit->endEncoding();
    }

    // ---- Passes 2..4: a-trous chain (atrous_enabled only) ----------
    if (atrous_enabled) {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(atrous_pso_);

        auto atrous_pass = [&](MTL::Texture* in, MTL::Texture* out,
                               std::uint32_t step) {
            // DenoiseAtrous.slang declares 6 bindings (0..5); slots 1
            // and 3 are placeholders (color_history_unused,
            // motion_unused) bound to 1x1 textures so the MSL kernel
            // has valid resources at those texture slots.
            enc->setTexture(in,            0);
            enc->setTexture(dummy_color_,  1);
            enc->setTexture(depth_in,      2);
            enc->setTexture(dummy_motion_, 3);
            enc->setTexture(normal_in,     4);
            enc->setTexture(out,           5);

            DenoisePush p{};
            p.width         = w;
            p.height        = h;
            p.step_or_reset = step;
            p.a             = 1.0f;     // sigma_depth (relative)
            p.b             = 64.0f;    // sigma_normal (pow exponent)
            p.c             = 4.0f;     // sigma_color (luminance sigma)
            enc->setBytes(&p, sizeof(p), 0);
            enc->dispatchThreadgroups(groups, tgsize);
            // Inter-dispatch barrier: each pass reads the previous
            // pass's output. textureBarrier() (renderCommandEncoder
            // equivalent) doesn't exist on compute encoders; instead
            // use memoryBarrierWithScope(textures) which flushes all
            // pending texture writes from this encoder for subsequent
            // reads in the same encoder.
            enc->memoryBarrier(MTL::BarrierScopeTextures);
        };
        atrous_pass(hist_write, atrous_a_, 1u);
        atrous_pass(atrous_a_,  atrous_b_, 2u);
        atrous_pass(atrous_b_,  output,    4u);
        enc->endEncoding();
    }

    frame_parity_ ^= 1u;
}

}  // namespace pt::rhi::mtl

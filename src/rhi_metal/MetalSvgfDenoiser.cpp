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
    if (dummy_color_)    { dummy_color_->release();    dummy_color_    = nullptr; }
    if (dummy_motion_)   { dummy_motion_->release();   dummy_motion_   = nullptr; }
    if (dummy_variance_) { dummy_variance_->release(); dummy_variance_ = nullptr; }
    if (temporal_pso_)   { temporal_pso_->release();   temporal_pso_   = nullptr; }
    if (atrous_pso_)     { atrous_pso_->release();     atrous_pso_     = nullptr; }
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
    rel(moments_history_a_);
    rel(moments_history_b_);
    rel(atrous_a_);
    rel(atrous_b_);
    rel(variance_a_);
    rel(variance_b_);
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

    // 1x1 placeholders for the slots each pass declares but does not
    // consume:
    //   atrous slot 1 (color_history_unused, RGBA16F)
    //   atrous slot 3 (motion_unused, RG16F)
    //   temporal slot 10 (variance_in_unused, R32F)
    // MSL still demands valid textures at every declared [[texture(N)]]
    // index, so we keep one tiny resident texture per format.
    dummy_color_    = MakeTexture(dev, MTL::PixelFormatRGBA16Float, 1, 1);
    dummy_motion_   = MakeTexture(dev, MTL::PixelFormatRG16Float,   1, 1);
    dummy_variance_ = MakeTexture(dev, MTL::PixelFormatR32Float,    1, 1);
    if (dummy_color_ == nullptr || dummy_motion_ == nullptr ||
        dummy_variance_ == nullptr) {
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

    history_a_         = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    history_b_         = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    depth_history_a_   = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);
    depth_history_b_   = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);
    normal_history_a_  = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    normal_history_b_  = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    moments_history_a_ = MakeTexture(dev, MTL::PixelFormatRG32Float,   w, h);
    moments_history_b_ = MakeTexture(dev, MTL::PixelFormatRG32Float,   w, h);
    atrous_a_          = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    atrous_b_          = MakeTexture(dev, MTL::PixelFormatRGBA16Float, w, h);
    variance_a_        = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);
    variance_b_        = MakeTexture(dev, MTL::PixelFormatR32Float,    w, h);

    if (history_a_ == nullptr || history_b_ == nullptr ||
        depth_history_a_ == nullptr || depth_history_b_ == nullptr ||
        normal_history_a_ == nullptr || normal_history_b_ == nullptr ||
        moments_history_a_ == nullptr || moments_history_b_ == nullptr ||
        atrous_a_ == nullptr || atrous_b_ == nullptr ||
        variance_a_ == nullptr || variance_b_ == nullptr) {
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

    // Frame-parity ping-pong for cross-frame textures: history read =
    // parity side, history write = other side. Mirror of
    // VulkanNrdDenoiser frame_parity_.
    const bool parity_is_a = (frame_parity_ == 0);
    MTL::Texture* hist_read           = parity_is_a ? history_a_          : history_b_;
    MTL::Texture* hist_write          = parity_is_a ? history_b_          : history_a_;
    MTL::Texture* depth_hist_read     = parity_is_a ? depth_history_a_    : depth_history_b_;
    MTL::Texture* depth_hist_write    = parity_is_a ? depth_history_b_    : depth_history_a_;
    MTL::Texture* normal_hist_read    = parity_is_a ? normal_history_a_   : normal_history_b_;
    MTL::Texture* normal_hist_write   = parity_is_a ? normal_history_b_   : normal_history_a_;
    MTL::Texture* moments_hist_read   = parity_is_a ? moments_history_a_  : moments_history_b_;
    MTL::Texture* moments_hist_write  = parity_is_a ? moments_history_b_  : moments_history_a_;

    const bool effective_reset = reset_history || needs_history_clear_;
    needs_history_clear_ = false;

    const std::uint32_t w = cached_w_;
    const std::uint32_t h = cached_h_;
    MTL::Size groups  = MTL::Size::Make((w + 7) / 8, (h + 7) / 8, 1);
    MTL::Size tgsize  = MTL::Size::Make(8, 8, 1);

    // In SVGF-atrous mode the temporal output is intermediate scratch
    // (consumed by atrous pass 1) and hist_write is the slot that
    // receives the FIRST atrous output (= what feeds next frame's
    // history, per Schied 2017's feedback loop). In SVGF-basic the
    // atrous chain is skipped, so temporal writes straight to
    // hist_write -- the same texture that backs next frame's history.
    MTL::Texture* temporal_color_out = atrous_enabled ? atrous_a_ : hist_write;

    // ---- Pass 1: temporal accumulate -------------------------------
    {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(temporal_pso_);
        // Slang MSL emission for [[vk::binding(N, 0)]] storage images
        // assigns [[texture(N)]] in declaration order; we bind by
        // shader slot index. Slot 10 (variance_in) is unread by the
        // temporal pass but still needs a valid R32F texture.
        enc->setTexture(color_in,            0);
        enc->setTexture(hist_read,           1);
        enc->setTexture(depth_in,            2);
        enc->setTexture(motion_in,           3);
        enc->setTexture(normal_in,           4);
        enc->setTexture(temporal_color_out,  5);
        enc->setTexture(depth_hist_read,     6);
        enc->setTexture(normal_hist_read,    7);
        enc->setTexture(moments_hist_read,   8);
        enc->setTexture(moments_hist_write,  9);
        enc->setTexture(dummy_variance_,    10);
        enc->setTexture(variance_a_,        11);

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
            // svgf_basic: temporal wrote to hist_write; blit to the
            // caller's output. Same RGBA16F format, same dims.
            blit->copyFromTexture(hist_write, 0, 0, org, sz,
                                  output,     0, 0, org);
        }
        blit->endEncoding();
    }

    // ---- Passes 2..4: a-trous chain (atrous_enabled only) ----------
    if (atrous_enabled) {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(atrous_pso_);

        auto atrous_pass = [&](MTL::Texture* color_src,
                               MTL::Texture* color_dst,
                               MTL::Texture* var_src,
                               MTL::Texture* var_dst,
                               std::uint32_t step) {
            // Slots 1, 3 are dummies (color_history_unused,
            // motion_unused); slots 6-9 carry valid same-format
            // textures the atrous pass declares but doesn't read,
            // saving an R32F + RGBA16F + RG32F dummy each.
            enc->setTexture(color_src,           0);
            enc->setTexture(dummy_color_,        1);
            enc->setTexture(depth_in,            2);
            enc->setTexture(dummy_motion_,       3);
            enc->setTexture(normal_in,           4);
            enc->setTexture(color_dst,           5);
            enc->setTexture(depth_hist_write,    6);
            enc->setTexture(normal_hist_write,   7);
            enc->setTexture(moments_hist_write,  8);
            enc->setTexture(moments_hist_write,  9);
            enc->setTexture(var_src,            10);
            enc->setTexture(var_dst,            11);

            DenoisePush p{};
            p.width         = w;
            p.height        = h;
            p.step_or_reset = step;
            p.a             = 1.0f;     // sigma_depth (gradient-scaled)
            p.b             = 128.0f;   // sigma_normal (sharper than the old 64.0)
            p.c             = 4.0f;     // sigma_color (variance-scaled)
            enc->setBytes(&p, sizeof(p), 0);
            enc->dispatchThreadgroups(groups, tgsize);
            // Inter-dispatch barrier: each pass reads the previous
            // pass's output. memoryBarrier(BarrierScopeTextures)
            // flushes pending texture writes from this encoder so the
            // next dispatch sees them.
            enc->memoryBarrier(MTL::BarrierScopeTextures);
        };
        // Schied 2017 feedback loop: the first A-Trous output is what
        // gets fed back as next frame's color history (the temporal
        // pass's raw blend is noisier and would drift). hist_write
        // captures pass-1 output; passes 2 and 3 ping-pong atrous_b
        // and the caller's output.
        atrous_pass(atrous_a_, hist_write, variance_a_, variance_b_, 1u);
        atrous_pass(hist_write, atrous_b_, variance_b_, variance_a_, 2u);
        atrous_pass(atrous_b_,  output,    variance_a_, variance_b_, 4u);
        enc->endEncoding();
    }

    frame_parity_ ^= 1u;
}

}  // namespace pt::rhi::mtl

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
    auto relT = [](MTL::Texture*& t) { if (t) { t->release(); t = nullptr; } };
    auto relB = [](MTL::Buffer*&  b) { if (b) { b->release(); b = nullptr; } };
    relT(history_a_);
    relT(history_b_);
    relT(depth_history_a_);
    relT(depth_history_b_);
    relT(normal_history_a_);
    relT(normal_history_b_);
    relB(moments_history_a_);
    relB(moments_history_b_);
    relT(atrous_a_);
    relT(atrous_b_);
    relB(variance_a_);
    relB(variance_b_);
    relB(dummy_variance_buf_);
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

    // 1x1 texture placeholders for the slots each pass declares but
    // does not consume:
    //   atrous slot 1 (color_history_unused, RGBA16F)
    //   atrous slot 3 (motion_unused,        RG16F)
    // MSL demands a valid texture at every declared [[texture(N)]] index.
    // The buffer placeholder (dummy_variance_buf_) for temporal's unused
    // variance_in slot is sized lazily in ResizeTextures so it matches
    // the active resolution -- a 1-element buffer is fine since the
    // shader never reads it, but the buffer must be non-null.
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

    // Buffer-backed scratch:
    //   moments_history: width*height * sizeof(float2) per side
    //   variance:        width*height * sizeof(float)  per side
    // MTLStorageModePrivate so the GPU keeps the only copy; the shader
    // initialises every element it reads on first touch (reset path).
    const std::size_t pix       = std::size_t(w) * std::size_t(h);
    const std::size_t mom_bytes = pix * sizeof(float) * 2;
    const std::size_t var_bytes = pix * sizeof(float);
    moments_history_a_  = dev->newBuffer(mom_bytes, MTL::ResourceStorageModePrivate);
    moments_history_b_  = dev->newBuffer(mom_bytes, MTL::ResourceStorageModePrivate);
    variance_a_         = dev->newBuffer(var_bytes, MTL::ResourceStorageModePrivate);
    variance_b_         = dev->newBuffer(var_bytes, MTL::ResourceStorageModePrivate);
    // Temporal's variance_in slot is declared but never read; bind a
    // 16-byte placeholder so the MSL device pointer is non-null.
    dummy_variance_buf_ = dev->newBuffer(16, MTL::ResourceStorageModePrivate);

    if (history_a_ == nullptr || history_b_ == nullptr ||
        depth_history_a_ == nullptr || depth_history_b_ == nullptr ||
        normal_history_a_ == nullptr || normal_history_b_ == nullptr ||
        atrous_a_ == nullptr || atrous_b_ == nullptr ||
        moments_history_a_ == nullptr || moments_history_b_ == nullptr ||
        variance_a_ == nullptr || variance_b_ == nullptr ||
        dummy_variance_buf_ == nullptr) {
        LOG_ERROR("MetalSvgfDenoiser: scratch resource alloc failed at {}x{}", w, h);
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
                                bool                atrous_enabled,
                                std::uint32_t       atrous_passes) {
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
    MTL::Texture* hist_read         = parity_is_a ? history_a_         : history_b_;
    MTL::Texture* hist_write        = parity_is_a ? history_b_         : history_a_;
    MTL::Texture* depth_hist_read   = parity_is_a ? depth_history_a_   : depth_history_b_;
    MTL::Texture* depth_hist_write  = parity_is_a ? depth_history_b_   : depth_history_a_;
    MTL::Texture* normal_hist_read  = parity_is_a ? normal_history_a_  : normal_history_b_;
    MTL::Texture* normal_hist_write = parity_is_a ? normal_history_b_  : normal_history_a_;
    MTL::Buffer*  moments_hist_read  = parity_is_a ? moments_history_a_ : moments_history_b_;
    MTL::Buffer*  moments_hist_write = parity_is_a ? moments_history_b_ : moments_history_a_;

    const bool effective_reset = reset_history || needs_history_clear_;
    needs_history_clear_ = false;

    const std::uint32_t w = cached_w_;
    const std::uint32_t h = cached_h_;
    MTL::Size groups  = MTL::Size::Make((w + 7) / 8, (h + 7) / 8, 1);
    MTL::Size tgsize  = MTL::Size::Make(8, 8, 1);

    // In SVGF-atrous mode the temporal output is intermediate scratch
    // (consumed by the single A-Trous pass) and hist_write receives the
    // filtered result -- Schied 2017's feedback loop. With variance
    // gating + tight sigma_color, converged regions are barely touched
    // by the filter so feedback is ~no-op there; disoccluded regions
    // benefit because next frame's temporal blend starts from a
    // spatially-denoised estimate, accelerating convergence. In
    // SVGF-basic the chain is skipped so temporal writes straight into
    // hist_write.
    MTL::Texture* temporal_color_out = atrous_enabled ? atrous_a_ : hist_write;

    // ---- Pass 1: temporal accumulate -------------------------------
    {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(temporal_pso_);
        // Slang MSL slot map for the temporal kernel (kept stable by
        // Slang's declaration-order policy; verified in the emitted
        // DenoiseTemporal.metal):
        //   [[texture(N)]]  -- N matches the Slang [[vk::binding(N,0)]]
        //                      slot for storage-image bindings 0..7.
        //   [[buffer(0..3)]] -- Slang bindings 8..11 map 1:1 onto MSL
        //                       buffer slots 0..3, in declaration order.
        //                       Slot 2 (variance_in_unused) is DCE'd
        //                       from the kernel signature so any bind
        //                       at slot 2 is silently ignored, but we
        //                       still pass the dummy so Vulkan parity
        //                       is preserved.
        //   [[buffer(4)]]    -- push_constant cbuffer; Slang places it
        //                       AFTER the structured buffers.
        enc->setTexture(color_in,           0);
        enc->setTexture(hist_read,          1);
        enc->setTexture(depth_in,           2);
        enc->setTexture(motion_in,          3);
        enc->setTexture(normal_in,          4);
        enc->setTexture(temporal_color_out, 5);
        enc->setTexture(depth_hist_read,    6);
        enc->setTexture(normal_hist_read,   7);

        enc->setBuffer(moments_hist_read,   0, 0); // moments_history_in  (binding 8)
        enc->setBuffer(moments_hist_write,  0, 1); // moments_history_out (binding 9)
        enc->setBuffer(dummy_variance_buf_, 0, 2); // variance_in_unused  (binding 10, DCE'd)
        enc->setBuffer(variance_a_,         0, 3); // variance_out        (binding 11)

        DenoisePush p{};
        p.width         = w;
        p.height        = h;
        p.step_or_reset = effective_reset ? 1u : 0u;
        p.a             = 0.10f;   // depth_tolerance (relative)
        p.b             = 0.85f;   // normal_tolerance (cos angle)
        p.c             = 0.10f;   // min_alpha (steady-state blend)
        enc->setBytes(&p, sizeof(p), 4);
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

    // ---- A-Trous chain (atrous_enabled only) -----------------------
    // The caller picks pass count (1..5) via r_svgf_atrous_passes:
    //   1 = step=1                       (5x5  effective)
    //   2 = step=1, 2                    (9x9)
    //   3 = step=1, 2, 4                 (17x17)
    //   4 = step=1, 2, 4, 8              (33x33)
    //   5 = step=1, 2, 4, 8, 16          (65x65, canonical SVGF)
    // Pass 1 ALWAYS writes to hist_write so it doubles as next frame's
    // reprojection source (Schied's feedback). For n_passes >= 2 the
    // LAST pass writes to the caller's output; intermediate passes
    // ping-pong atrous_a / atrous_b. With one pass we instead blit
    // hist_write -> output at the end (one pass is both Schied feedback
    // AND user output).
    if (atrous_enabled) {
        std::uint32_t n_passes = atrous_passes;
        if (n_passes < 1u) n_passes = 1u;
        if (n_passes > 5u) n_passes = 5u;

        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(atrous_pso_);

        auto atrous_pass = [&](MTL::Texture* color_src,
                               MTL::Texture* color_dst,
                               MTL::Buffer*  var_src,
                               MTL::Buffer*  var_dst,
                               std::uint32_t step) {
            // Slang MSL slot map for the atrous kernel (same policy as
            // temporal -- bindings 8..11 -> buffers 0..3, push at 4):
            //   texture(1, 3, 6, 7) -- declared but DCE'd from kernel
            //     signature. We still bind dummies / parity textures
            //     because Metal silently ignores binds to slots not
            //     present in the kernel signature, and this keeps the
            //     host code symmetric with the Vulkan path.
            //   buffer(0, 1)        -- moments_in_unused /
            //     moments_out_unused, also DCE'd. We bind the parity-
            //     side moments buffer at both slots; the shader never
            //     reads or writes through them.
            //   buffer(2, 3)        -- variance_in / variance_out
            //     (the only buffers the atrous shader actually uses).
            //   buffer(4)           -- push_constant cbuffer.
            enc->setTexture(color_src,         0);
            enc->setTexture(dummy_color_,      1);
            enc->setTexture(depth_in,          2);
            enc->setTexture(dummy_motion_,     3);
            enc->setTexture(normal_in,         4);
            enc->setTexture(color_dst,         5);
            enc->setTexture(depth_hist_write,  6);
            enc->setTexture(normal_hist_write, 7);

            enc->setBuffer(moments_hist_write, 0, 0); // moments_in_unused  (DCE'd)
            enc->setBuffer(moments_hist_write, 0, 1); // moments_out_unused (DCE'd)
            enc->setBuffer(var_src,            0, 2); // variance_in
            enc->setBuffer(var_dst,            0, 3); // variance_out

            DenoisePush p{};
            p.width         = w;
            p.height        = h;
            p.step_or_reset = step;
            p.a             = 1.0f;     // sigma_depth (gradient-scaled)
            p.b             = 128.0f;   // sigma_normal (sharper than the old 64.0)
            p.c             = 1.0f;     // sigma_color (variance-scaled). The SVGF
                                        // paper default of 4.0 reads as visibly
                                        // smooth on diffuse surfaces -- 1.0 keeps
                                        // the noise rejection without averaging
                                        // through 4-sigma-wide luminance bands.
            enc->setBytes(&p, sizeof(p), 4);
            enc->dispatchThreadgroups(groups, tgsize);
            // Inter-dispatch barrier flushes pending texture AND buffer
            // writes from this encoder so the next dispatch sees them.
            enc->memoryBarrier(MTL::BarrierScope(MTL::BarrierScopeTextures |
                                                MTL::BarrierScopeBuffers));
        };
        // Pass schedule. Color ping-pong:
        //   pass 1: atrous_a_ -> hist_write     (Schied feedback target)
        //   pass 2: hist_write -> atrous_b_     (or -> output if last)
        //   pass i (i>=3, not last): toggles atrous_a_ <-> atrous_b_
        //   pass N (last, N>=2):              -> output
        // Variance ping-pong is independent: starts variance_a -> b on
        // pass 1, then alternates each pass.
        MTL::Texture* color_src   = atrous_a_;
        MTL::Buffer*  var_src     = variance_a_;
        MTL::Buffer*  var_dst     = variance_b_;
        for (std::uint32_t i = 1; i <= n_passes; ++i) {
            const std::uint32_t step = 1u << (i - 1u);
            MTL::Texture* color_dst;
            if (i == 1u) {
                color_dst = hist_write;
            } else if (i == n_passes) {
                color_dst = output;
            } else {
                // Even i -> atrous_b_, odd i -> atrous_a_. After pass 1
                // atrous_a_ is no longer read, so we're free to reuse
                // it as a ping-pong scratch for i >= 3.
                color_dst = (i % 2u == 0u) ? atrous_b_ : atrous_a_;
            }
            atrous_pass(color_src, color_dst, var_src, var_dst, step);
            color_src = color_dst;
            std::swap(var_src, var_dst);
        }
        enc->endEncoding();

        if (n_passes == 1u) {
            // One-pass path: pass 1's output sits in hist_write (the
            // feedback target). Publish a copy to the caller's output.
            // Same RGBA16F format / dims, both in MTLStorageModePrivate.
            MTL::BlitCommandEncoder* blit2 = cb->blitCommandEncoder();
            MTL::Origin org = MTL::Origin::Make(0, 0, 0);
            MTL::Size   sz  = MTL::Size::Make(w, h, 1);
            blit2->copyFromTexture(hist_write, 0, 0, org, sz,
                                   output,     0, 0, org);
            blit2->endEncoding();
        }
    }

    frame_parity_ ^= 1u;
}

}  // namespace pt::rhi::mtl

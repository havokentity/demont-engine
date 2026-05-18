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
// Issue #119 -- albedo remod kernel.
extern const unsigned char shader_DenoiseRemod_metal_data[];
extern const unsigned long shader_DenoiseRemod_metal_size;
}

namespace pt::rhi::mtl {

namespace {

// 48-byte push struct shared by both denoise kernels. Field names are
// kernel-specific (see DenoiseTemporal.slang / DenoiseAtrous.slang) but
// the layout matches what DenoisePush in VulkanDenoiser.cpp emits, so
// the Slang MSL emission for the [[vk::push_constant]] cbuffer maps
// cleanly onto this struct.
//
// Layout history:
//   - Original 32 bytes: width, height, step_or_reset, pad, a, b, c, pad.
//   - Issue #119 grows to 48 bytes by appending demod_enabled +
//     final_remod (atrous-only) + 8 bytes of pad. std140 aligns the
//     push block to 16 bytes; 48 is the next multiple after 32+8 = 40.
struct DenoisePush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t step_or_reset;
    std::uint32_t pad0;
    float         a;     // temporal: depth_tolerance; atrous: sigma_depth
    float         b;     // temporal: normal_tolerance; atrous: sigma_normal
    float         c;     // temporal: min_alpha; atrous: sigma_color
    float         pad1;
    // Issue #119 -- demod gate (both kernels) and remod gate (atrous
    // only). Temporal leaves final_remod unread; atrous reads it on
    // the LAST pass of its chain when also writing into `output`.
    std::uint32_t demod_enabled;
    std::uint32_t final_remod;
    std::uint32_t pad2;
    std::uint32_t pad3;
};
static_assert(sizeof(DenoisePush) == 48, "DenoisePush layout");

// Issue #119 -- 16-byte push struct for the remod-only kernel
// (DenoiseRemod.slang). Just enough to carry the dispatch size; the
// kernel reads `demod_in` + `albedo_tex` and writes `color_out`.
struct RemodPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pad0;
    std::uint32_t pad1;
};
static_assert(sizeof(RemodPush) == 16, "RemodPush layout");

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
    if (remod_pso_)    { remod_pso_->release();    remod_pso_    = nullptr; }
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
    // Issue #119 -- albedo remod pass. Built unconditionally so we can
    // toggle r_svgf_albedo_demod at runtime without a backend re-init.
    remod_pso_    = BuildPso(dev, shader_DenoiseRemod_metal_data,
                              shader_DenoiseRemod_metal_size,
                              "DenoiseRemod");
    if (temporal_pso_ == nullptr || atrous_pso_ == nullptr ||
        remod_pso_ == nullptr) {
        DestroyAll();
        return false;
    }

    // 1x1 texture placeholders for the slots each pass declares but
    // does not consume:
    //   atrous slot 1 (color_history_unused, RGBA16F)
    //   atrous slot 3 (motion_unused,        RG16F)
    // MSL demands a valid texture at every declared [[texture(N)]] index.
    // The buffer placeholder (dummy_variance_buf_) for temporal's unused
    // variance_in slot is a fixed 16-byte storage buffer allocated once
    // in ResizeTextures. The shader never reads it, but the buffer pointer
    // must be non-null so the MSL kernel argument list is valid.
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
                                MTL::Texture*       albedo_in,
                                MTL::Texture*       output,
                                bool                reset_history,
                                bool                atrous_enabled,
                                std::uint32_t       atrous_passes,
                                bool                albedo_demod_enabled) {
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
    // Issue #119 -- demod requires the albedo G-buffer. When the engine
    // didn't allocate it (older r_denoiser kinds, or a transition frame
    // before resize fires) we silently fall back to non-demod behaviour
    // and bind the dummy placeholder so the kernel's albedo slot stays
    // well-formed. The kernel's `demod_enabled = 0` push gate then
    // short-circuits every divide / multiply. (See per-encoder slot
    // maps below for the actual MSL slot the albedo binds at -- it's
    // not slot 12 even though Slang declares [[vk::binding(12)]];
    // issue #164.)
    const bool demod_active = albedo_demod_enabled && (albedo_in != nullptr);
    MTL::Texture* albedo_bind = (albedo_in != nullptr) ? albedo_in : dummy_color_;

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
        // Slang MSL slot map for the temporal kernel (verified in the
        // emitted DenoiseTemporal.metal):
        //   [[texture(N)]]  -- N matches the Slang [[vk::binding(N,0)]]
        //                      slot for the 8 RW textures at bindings
        //                      0..7. Slang preserves those positions
        //                      because they all fit in the [0..7] slot
        //                      window.
        //   [[texture(8)]]  -- Issue #119, sample-only `Texture2D`
        //                      albedo. Declared with [[vk::binding(12)]]
        //                      so Vulkan's descriptor-set lands it at
        //                      slot 12, but Slang's MSL emit COMPACTS
        //                      it down to the next free slot (8) since
        //                      vk-slot 12 is outside the [0..7] window
        //                      it preserved for the RW textures. Sample
        //                      access does NOT count toward Metal's
        //                      8-RW-texture compute cap.
        //   [[buffer(0..3)]] -- Slang bindings 8..11 map 1:1 onto MSL
        //                       buffer slots 0..3, in declaration order.
        //                       Slot 2 (variance_in_unused) is DCE'd
        //                       from the kernel signature so any bind
        //                       at slot 2 is silently ignored, but we
        //                       still pass the dummy so Vulkan parity
        //                       is preserved.
        //   [[buffer(4)]]    -- push_constant cbuffer; Slang places it
        //                       AFTER the structured buffers.
        // Issue #164: the original implementation bound albedo_bind at
        // slot 12 (matching the Vulkan binding). That's the WRONG slot
        // for MSL -- slot 12 is unbound, slot 8 is what the kernel
        // actually reads, and the kernel saw all-zero. isSkyAlbedo()
        // then returned true for every surface pixel, bypassing the
        // demod divide AND the remod multiply -- but the chain still
        // ran a temporal blend on the modulated radiance which produces
        // visible corruption (R/B channels inflated, G preserved =
        // magenta on the dawn ground plane). Bind at slot 8 instead.
        enc->setTexture(color_in,           0);
        enc->setTexture(hist_read,          1);
        enc->setTexture(depth_in,           2);
        enc->setTexture(motion_in,          3);
        enc->setTexture(normal_in,          4);
        enc->setTexture(temporal_color_out, 5);
        enc->setTexture(depth_hist_read,    6);
        enc->setTexture(normal_hist_read,   7);
        enc->setTexture(albedo_bind,        8);

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
        p.demod_enabled = demod_active ? 1u : 0u;
        p.final_remod   = 0u;       // temporal never remod's; its color_out
                                    // is hist_write (the SVGF feedback target)
                                    // which must stay demodulated.
        enc->setBytes(&p, sizeof(p), 4);
        enc->dispatchThreadgroups(groups, tgsize);
        enc->endEncoding();
    }

    // ---- Depth + normal history copy (blit encoder) ----------------
    // The next frame's temporal pass reads these as the "previous
    // frame's surfaces" for reprojection validation.
    //
    // Issue #119 -- with demod ON we'd otherwise have to blit
    // hist_write -> output for svgf_basic, but hist_write is in
    // demodulated lighting space and `output` must hold textured
    // radiance. Defer the svgf_basic publish to a compute remod pass
    // below in that case; the blit here only takes care of depth +
    // normal history (and, for the demod-OFF svgf_basic case, the
    // existing color blit to output).
    {
        MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
        MTL::Origin org = MTL::Origin::Make(0, 0, 0);
        MTL::Size   sz  = MTL::Size::Make(w, h, 1);
        blit->copyFromTexture(depth_in,  0, 0, org, sz,
                              depth_hist_write,  0, 0, org);
        blit->copyFromTexture(normal_in, 0, 0, org, sz,
                              normal_hist_write, 0, 0, org);

        if (!atrous_enabled && !demod_active) {
            // svgf_basic, demod off: temporal wrote modulated radiance
            // to hist_write; blit straight to the caller's output.
            // Same RGBA16F format, same dims.
            blit->copyFromTexture(hist_write, 0, 0, org, sz,
                                  output,     0, 0, org);
        }
        blit->endEncoding();
    }

    // ---- svgf_basic + demod ON: remod hist_write -> output ---------
    // The temporal pass wrote a demodulated-lighting estimate to
    // hist_write (the SVGF feedback target). Multiply by albedo and
    // publish into the caller's output so the bloom + tonemap chain
    // sees textured radiance. Dispatch a compute encoder rather than
    // a blit so we can sample the albedo per-pixel.
    if (!atrous_enabled && demod_active) {
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(remod_pso_);
        // DenoiseRemod.slang slot map (verified in DenoiseRemod.metal):
        //   [[texture(0)]] -- demod_in   (RGBA16F demodulated lighting)
        //   [[texture(1)]] -- color_out  (RGBA16F textured radiance)
        //   [[texture(2)]] -- albedo_tex (sample-only). Declared with
        //                     [[vk::binding(12)]] in DenoiseRemod.slang
        //                     so SPIR-V lands it at descriptor binding
        //                     12, but Slang's MSL emit compacts it down
        //                     to the next free slot (2). Issue #164.
        //   [[buffer(0)]]  -- push constants
        enc->setTexture(hist_write,   0);
        enc->setTexture(output,       1);
        enc->setTexture(albedo_bind,  2);
        RemodPush rp{};
        rp.width  = w;
        rp.height = h;
        enc->setBytes(&rp, sizeof(rp), 0);
        enc->dispatchThreadgroups(groups, tgsize);
        enc->endEncoding();
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
                               std::uint32_t step,
                               bool          is_final_pass) {
            // Slang MSL slot map for the atrous kernel (same buffer
            // policy as temporal -- bindings 8..11 -> buffers 0..3,
            // push at 4):
            //   texture(1, 3, 6, 7) -- declared but DCE'd from kernel
            //     signature. We still bind dummies / parity textures
            //     because Metal silently ignores binds to slots not
            //     present in the kernel signature, and this keeps the
            //     host code symmetric with the Vulkan path.
            //   texture(8)         -- albedo (sample-only); read for
            //     per-tap demod-divide and (on the final pass when
            //     final_remod is set) the multiply-back into color_out.
            //     Declared with [[vk::binding(12)]] but Slang's MSL
            //     emit compacts it to the next free slot (8). Issue
            //     #164: the original code bound this at slot 12 which
            //     was unbound -- the kernel read zeros for albedo,
            //     isSkyAlbedo() returned true everywhere, and the
            //     multiply-back was silently skipped on surface pixels
            //     producing magenta ground.
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
            enc->setTexture(albedo_bind,       8);

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
            p.demod_enabled = demod_active ? 1u : 0u;
            // Final-pass remod fires only when:
            //   - demod is on
            //   - this is the chain's last pass
            //   - the pass is writing directly to `output` (n_passes >=
            //     2; the n_passes == 1 case writes to hist_write first
            //     and a separate remod kernel runs after the loop).
            // Intermediate passes set this 0 so they leave the signal
            // demodulated for the next pass's divide.
            p.final_remod   = (demod_active && is_final_pass) ? 1u : 0u;
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
            // Identify which pass writes directly to the caller's
            // `output` (so the shader can remod inline). For n_passes
            // == 1 the pass writes to hist_write (feedback target) and
            // the remod runs as a separate dispatch below. For
            // n_passes >= 2 the LAST pass writes to `output` and that
            // pass takes `is_final_pass = true` so the shader applies
            // the multiply-back inline.
            bool writes_to_output = false;
            if (i == 1u) {
                color_dst = hist_write;
            } else if (i == n_passes) {
                color_dst = output;
                writes_to_output = true;
            } else {
                // Even i -> atrous_b_, odd i -> atrous_a_. After pass 1
                // atrous_a_ is no longer read, so we're free to reuse
                // it as a ping-pong scratch for i >= 3.
                color_dst = (i % 2u == 0u) ? atrous_b_ : atrous_a_;
            }
            atrous_pass(color_src, color_dst, var_src, var_dst, step,
                        writes_to_output);
            color_src = color_dst;
            std::swap(var_src, var_dst);
        }
        enc->endEncoding();

        if (n_passes == 1u) {
            // One-pass path: pass 1's output sits in hist_write (the
            // feedback target, kept in demodulated lighting space when
            // demod is on so the next frame's temporal divide composes
            // correctly). Publish into the caller's `output`:
            //   - demod off: blit hist_write -> output (legacy path).
            //   - demod on:  dispatch the remod compute kernel so the
            //     albedo multiply-back lands in `output` while
            //     hist_write keeps its demodulated value for feedback.
            if (!demod_active) {
                MTL::BlitCommandEncoder* blit2 = cb->blitCommandEncoder();
                MTL::Origin org = MTL::Origin::Make(0, 0, 0);
                MTL::Size   sz  = MTL::Size::Make(w, h, 1);
                blit2->copyFromTexture(hist_write, 0, 0, org, sz,
                                       output,     0, 0, org);
                blit2->endEncoding();
            } else {
                MTL::ComputeCommandEncoder* enc2 = cb->computeCommandEncoder();
                enc2->setComputePipelineState(remod_pso_);
                // Same slot fix as the svgf_basic remod dispatch above.
                // Issue #164: albedo is at MSL slot 2, not 12.
                enc2->setTexture(hist_write,   0);
                enc2->setTexture(output,       1);
                enc2->setTexture(albedo_bind,  2);
                RemodPush rp{};
                rp.width  = w;
                rp.height = h;
                enc2->setBytes(&rp, sizeof(rp), 0);
                enc2->dispatchThreadgroups(groups, tgsize);
                enc2->endEncoding();
            }
        }
    }

    frame_parity_ ^= 1u;
}

}  // namespace pt::rhi::mtl

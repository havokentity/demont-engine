// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Thin C wrapper around MTLFXTemporalDenoisedScaler. metal-cpp doesn't
// bridge MetalFX, so this is a tiny ObjC++ TU that the C++ code calls
// through extern "C". Same pattern as MetalLayerAttach.mm.
//
// Lifetime: pt_metalfx_create returns an opaque handle that the C++
// side stores on MetalDevice. Per frame, set the four textures + jitter
// + reset and call pt_metalfx_encode against the active MTLCommandBuffer.
// pt_metalfx_destroy releases the scaler, the descriptor, and the
// internal private-storage output texture.
//
// Why an internal output texture? MetalFX requires its output texture
// to have storageMode = MTLStorageModePrivate, but a CAMetalDrawable
// (which is what we hand it as "color_out") is shared/managed. So the
// shim allocates its own private RGBA16F texture, hands THAT to
// MetalFX, and then encodes a blit copy from it onto the caller's
// output texture (swapchain) before returning.
//
// Build portability: MTLFXTemporalDenoisedScaler is macOS 26+. On older
// SDKs (e.g. CI runner macOS 15) the type isn't declared, so the whole
// implementation is gated behind __MAC_OS_X_VERSION_MAX_ALLOWED. When
// built against an older SDK we ship stub no-op entry points so callers
// link, and pt_metalfx_create returning nullptr makes MetalDevice fall
// through to the no-denoiser path -- same as the runtime @available
// branch on a real macOS 15 box.

#include <Availability.h>
#include <cstdint>

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <simd/simd.h>

#include <cstring>

namespace {

struct API_AVAILABLE(macos(26.0)) PtMetalFXState {
    id<MTLFXTemporalDenoisedScaler> scaler = nil;
    id<MTLTexture>                  output_priv = nil;   // private-storage output for MetalFX
    NSUInteger width  = 0;
    NSUInteger height = 0;
};

}  // namespace

extern "C" void* pt_metalfx_create(void* mtl_device,
                                   std::uint32_t width,
                                   std::uint32_t height) {
    if (mtl_device == nullptr || width == 0 || height == 0) return nullptr;
    if (@available(macOS 26.0, *)) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device;
        MTLFXTemporalDenoisedScalerDescriptor* desc =
            [[MTLFXTemporalDenoisedScalerDescriptor alloc] init];
        desc.colorTextureFormat  = MTLPixelFormatRGBA16Float;
        desc.depthTextureFormat  = MTLPixelFormatR32Float;
        desc.motionTextureFormat = MTLPixelFormatRG16Float;
        // Guidance G-buffers. Without these MTLFXTemporalDenoisedScaler
        // has no surface signal to weight its spatial filter -- it falls
        // back to a conservative TAA-style blur that never fully
        // converges on a static camera, looking visibly worse than the
        // path tracer's own accumulated mean. With normal + diffuse
        // albedo bound, MetalFX gets the same kind of edge-aware
        // information SVGF uses, and convergence quality jumps
        // dramatically. Formats match the engine's allocations
        // (normal_tex / albedo_tex are RGBA16F).
        desc.normalTextureFormat        = MTLPixelFormatRGBA16Float;
        desc.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
        // Specular guidance trio (issue #118). Without these MetalFX
        // produces visible 8x8 halos on bright reflections / metals
        // because it can't tell specular response apart from diffuse;
        // with them the scaler reaches DLSS Ray Reconstruction parity
        // on Apple Silicon for the typical mirror-finish / glossy-metal
        // case we have on the diamond / glass sphere scene.
        //
        // Formats:
        //   specularAlbedo  -- RGBA16F (per-pixel F0; metals need RGB
        //                      since gold's F0 is yellow, etc.).
        //   roughness       -- R16F single-channel. The engine allocates
        //                      the backing texture as R32F (the RHI
        //                      doesn't expose R16F today and the
        //                      precision delta is academic for a guidance
        //                      input), so we ask MetalFX to read it as
        //                      R32F here too -- Apple's scaler accepts
        //                      any single-channel float format for this
        //                      input; the descriptor just has to declare
        //                      what the engine actually binds.
        //   specularHitDist -- R16F single-channel; same R32F-on-the-host
        //                      reasoning as roughness.
        desc.specularAlbedoTextureFormat       = MTLPixelFormatRGBA16Float;
        desc.roughnessTextureFormat            = MTLPixelFormatR32Float;
        desc.specularHitDistanceTextureFormat  = MTLPixelFormatR32Float;
        // Linear HDR all the way through MetalFX. Output is RGBA16F so
        // temporal reuse can see the full HDR range; the engine runs
        // a dedicated `tonemap` compute kernel after this pass to
        // apply exposure + ACES and write the sRGB swapchain.
        desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
        desc.inputWidth          = width;
        desc.inputHeight         = height;
        desc.outputWidth         = width;
        desc.outputHeight        = height;
        // Auto-exposure is OFF here -- the engine pre-multiplies the
        // input by its own r_exposure / r_auto_exposure so the
        // denoiser-on and denoiser-off paths render at the same
        // brightness. With autoExposureEnabled = YES, MetalFX would
        // re-scale on top, producing a brighter image than the
        // no-denoiser path.
        desc.autoExposureEnabled = NO;
        desc.requiresSynchronousInitialization = YES;

        id<MTLFXTemporalDenoisedScaler> scaler =
            [desc newTemporalDenoisedScalerWithDevice:dev];
        if (scaler == nil) return nullptr;

        // Private-storage output texture for MetalFX. Apple's docs
        // explicitly require .outputTexture be storageMode == private.
        // RGBA16F (linear HDR) so the post-MetalFX tonemap pass has
        // headroom for exposure + ACES; the engine's color_out
        // (passed to pt_metalfx_encode) is also RGBA16F so the blit
        // below is a same-format copy.
        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                          width:width
                                         height:height
                                      mipmapped:NO];
        // Add RenderTarget so we can clear-fill via a render pass for the
        // diagnostic path below.
        td.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> output_priv = [dev newTextureWithDescriptor:td];
        if (output_priv == nil) {
            scaler = nil;
            return nullptr;
        }

        PtMetalFXState* st = new PtMetalFXState();
        st->scaler      = scaler;
        st->output_priv = output_priv;
        st->width       = width;
        st->height      = height;
        return st;
    } else {
        return nullptr;   // MTLFXTemporalDenoisedScaler is macOS 26+
    }
}

extern "C" void pt_metalfx_destroy(void* state) {
    if (state == nullptr) return;
    // PtMetalFXState is API_AVAILABLE(macos(26.0)) -- the only way to
    // hold a pointer to one is to have called pt_metalfx_create which
    // already runs an @available check, so by the time we land here
    // we're definitely on macOS 26+. The @available block here is just
    // to satisfy the compiler's deployment-target check.
    if (@available(macOS 26.0, *)) {
        PtMetalFXState* st = static_cast<PtMetalFXState*>(state);
        st->scaler      = nil;     // ARC releases
        st->output_priv = nil;
        delete st;
    }
}

// Diagnostic mode picker. PT_METALFX_DEBUG controls what pt_metalfx_encode
// actually does; lets us isolate which stage of the post-pass is broken
// when the user reports black-screen regressions.
//   unset / 0 : real MetalFX scaler + blit (default)
//   1         : skip scaler; render-pass-clear private to red, blit to dst
//   2         : true no-op -- don't touch the command buffer at all (so
//               the path tracer's swapchain write is what reaches present)
//   3         : skip scaler + blit; render-pass-clear DST (swapchain) to
//               green directly. Tests render-pass-against-drawable path.
static int pt_metalfx_debug_mode() {
    static int cached = -2;
    if (cached == -2) {
        const char* v = getenv("PT_METALFX_DEBUG");
        cached = (v == nullptr || v[0] == '\0') ? 0 : atoi(v);
    }
    return cached;
}

extern "C" void pt_metalfx_encode(void* state,
                                  void* mtl_cb,
                                  void* color_in,
                                  void* depth_in,
                                  void* motion_in,
                                  void* normal_in,                  // can be NULL -> skip guidance
                                  void* albedo_in,                  // can be NULL -> skip guidance
                                  void* specular_albedo_in,         // can be NULL -> skip guidance (issue #118)
                                  void* roughness_in,               // can be NULL -> skip guidance (issue #118)
                                  void* specular_hit_distance_in,   // can be NULL -> skip guidance (issue #118)
                                  void* color_out,
                                  float jitter_x,
                                  float jitter_y,
                                  const float* world_to_view_4x4,   // 16 floats, column-major
                                  const float* view_to_clip_4x4,    // 16 floats, column-major
                                  int   reset) {
    if (state == nullptr || mtl_cb == nullptr) return;
    int dbg = pt_metalfx_debug_mode();
    if (dbg == 2) return;     // total no-op: leaves swapchain as the path tracer wrote it
    if (@available(macOS 26.0, *)) {
        PtMetalFXState* st = static_cast<PtMetalFXState*>(state);
        if (st->scaler == nil || st->output_priv == nil) return;

        id<MTLCommandBuffer> cb  = (__bridge id<MTLCommandBuffer>)mtl_cb;
        id<MTLTexture>       dst = (__bridge id<MTLTexture>)color_out;
        if (dst == nil) return;

        if (dbg == 3) {
            // Render-pass-clear directly to the swapchain (green). Tests
            // whether render passes can target the drawable at all.
            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture     = dst;
            rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0.0, 1.0, 0.0, 1.0);
            id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rpd];
            [renc endEncoding];
            return;
        }

        if (dbg == 1) {
            // Render-pass-clear private to red, then blit to swapchain.
            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture     = st->output_priv;
            rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[0].clearColor  = MTLClearColorMake(1.0, 0.0, 0.0, 1.0);
            id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rpd];
            [renc endEncoding];
        } else {
            st->scaler.colorTexture       = (__bridge id<MTLTexture>)color_in;
            st->scaler.depthTexture       = (__bridge id<MTLTexture>)depth_in;
            st->scaler.motionTexture      = (__bridge id<MTLTexture>)motion_in;
            // Guidance G-buffers. Engine allocates + fills these for
            // MetalFX kinds (see want_normal_gbuffer / want_albedo_gbuffer
            // in Engine.cpp). nil is tolerated by the scaler but degrades
            // back to the no-guidance behavior, so we'd rather log loudly
            // if either is missing; the engine's allocation logic should
            // keep both non-null whenever the denoiser is active.
            st->scaler.normalTexture        = (__bridge id<MTLTexture>)normal_in;
            st->scaler.diffuseAlbedoTexture = (__bridge id<MTLTexture>)albedo_in;
            // Specular guidance trio (issue #118). Apple's scaler tolerates
            // a nil binding as "no guidance for this frame," so when the
            // engine didn't allocate one of these (e.g. caller is on the
            // SVGF-only path that doesn't request specular guidance) we
            // pass nil through and the scaler falls back to its
            // no-specular-data behaviour for that input. With all three
            // bound the scaler picks up the F0 / roughness / reflection-
            // depth field and the 8x8 specular halos go away.
            st->scaler.specularAlbedoTexture      = (__bridge id<MTLTexture>)specular_albedo_in;
            st->scaler.roughnessTexture           = (__bridge id<MTLTexture>)roughness_in;
            st->scaler.specularHitDistanceTexture = (__bridge id<MTLTexture>)specular_hit_distance_in;
            st->scaler.outputTexture      = st->output_priv;
            st->scaler.jitterOffsetX      = jitter_x;
            st->scaler.jitterOffsetY      = jitter_y;
            st->scaler.depthReversed      = NO;
            st->scaler.shouldResetHistory = (reset != 0) ? YES : NO;
            st->scaler.motionVectorScaleX = 1.0f;
            st->scaler.motionVectorScaleY = 1.0f;
            // Required matrices. Without these set, the scaler can't do
            // its internal reprojection -> outputs all-black.
            if (world_to_view_4x4 && view_to_clip_4x4) {
                simd_float4x4 w2v, v2c;
                memcpy(&w2v, world_to_view_4x4, sizeof(simd_float4x4));
                memcpy(&v2c, view_to_clip_4x4,  sizeof(simd_float4x4));
                st->scaler.worldToViewMatrix = w2v;
                st->scaler.viewToClipMatrix  = v2c;
            }
            [st->scaler encodeToCommandBuffer:cb];
        }

        // Blit our private output onto the caller's output texture.
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        MTLOrigin origin = MTLOriginMake(0, 0, 0);
        MTLSize   size   = MTLSizeMake(st->width, st->height, 1);
        [blit copyFromTexture:st->output_priv
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:origin
                   sourceSize:size
                    toTexture:dst
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:origin];
        [blit endEncoding];
    }
}

#else  // __MAC_OS_X_VERSION_MAX_ALLOWED < 260000

// SDK doesn't declare MTLFXTemporalDenoisedScaler -- ship stubs so the
// link succeeds and the engine falls through to the no-denoiser path.
extern "C" void* pt_metalfx_create(void*, std::uint32_t, std::uint32_t) {
    return nullptr;
}
extern "C" void pt_metalfx_destroy(void*) {}
// 11 void* params: state, cb, color_in, depth_in, motion_in, normal_in,
// albedo_in, specular_albedo_in (#118), roughness_in (#118),
// specular_hit_distance_in (#118), color_out.
extern "C" void pt_metalfx_encode(void*, void*, void*, void*, void*, void*, void*,
                                   void*, void*, void*, void*,
                                   float, float, const float*, const float*, int) {}

#endif

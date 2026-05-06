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

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <simd/simd.h>

#include <cstdint>
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
        desc.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
        desc.inputWidth          = width;
        desc.inputHeight         = height;
        desc.outputWidth         = width;
        desc.outputHeight        = height;
        desc.autoExposureEnabled = YES;
        desc.requiresSynchronousInitialization = YES;

        id<MTLFXTemporalDenoisedScaler> scaler =
            [desc newTemporalDenoisedScalerWithDevice:dev];
        if (scaler == nil) return nullptr;

        // Private-storage output texture for MetalFX. Apple's docs
        // explicitly require .outputTexture be storageMode == private.
        // BGRA8Unorm so a subsequent blit to the swapchain is a
        // same-format copy.
        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
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
    PtMetalFXState* st = static_cast<PtMetalFXState*>(state);
    st->scaler      = nil;     // ARC releases
    st->output_priv = nil;
    delete st;
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

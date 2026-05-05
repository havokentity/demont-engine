// AppKit bridge for attaching a CAMetalLayer to a GLFW NSWindow's content
// view. Lives in pt_app_window so any backend (software, metal, vulkan)
// can use the same helper without each pulling its own .mm.
//
// Both arguments are opaque void* so this header-less symbol can be
// declared from a plain C++ TU. The caller passes their CA::MetalLayer*
// (metal-cpp) which is bit-identical to a CAMetalLayer*.

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

extern "C" void pt_metal_attach_layer(void* ns_window_ptr, void* metal_layer_ptr) {
    if (ns_window_ptr == nullptr || metal_layer_ptr == nullptr) return;

    NSWindow*    w     = (__bridge NSWindow*)ns_window_ptr;
    CAMetalLayer* layer = (__bridge CAMetalLayer*)metal_layer_ptr;

    NSView* view = w.contentView;
    if (view == nil) return;

    [view setWantsLayer:YES];
    [view setLayer:layer];

    layer.contentsScale    = w.backingScaleFactor;
    layer.opaque           = YES;
    layer.framebufferOnly  = NO;
    layer.frame            = view.bounds;
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
}

// Tiny AppKit bridge to attach a CAMetalLayer to a GLFW-owned NSWindow.
// Kept as a separate .mm so the rest of the Metal backend stays in C++
// using only metal-cpp.

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

extern "C" void pt_metal_attach_layer(void* ns_window_ptr, void* metal_layer_ptr) {
    if (ns_window_ptr == nullptr || metal_layer_ptr == nullptr) return;

    NSWindow* w     = (__bridge NSWindow*)ns_window_ptr;
    CAMetalLayer* layer = (__bridge CAMetalLayer*)metal_layer_ptr;

    NSView* view = w.contentView;
    if (view == nil) return;

    [view setWantsLayer:YES];
    [view setLayer:layer];

    layer.contentsScale = w.backingScaleFactor;
    layer.opaque        = YES;
    layer.framebufferOnly = NO;
    layer.frame         = view.bounds;
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Native macOS performance overlay -- floating NSPanel anchored to
// the top-right of the engine's NSWindow client area, drawing a tiered
// stats readout with NSAttributedString + NSBezierPath for the
// sparkline.  Mirrors the Win32 impl visually; differs only where the
// AppKit/GDI APIs diverge.

#import <Cocoa/Cocoa.h>

#include "PerfOverlay.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Theme {
    const char* name;
    NSColor* panel;
    NSColor* text;
    NSColor* accent;
    NSColor* dim;
    NSColor* graph;
};

static NSColor* RGB8(int r, int g, int b, double a = 1.0) {
    return [NSColor colorWithCalibratedRed:r/255.0 green:g/255.0 blue:b/255.0 alpha:a];
}

Theme PaletteFor(std::string_view name) {
    if (name == "amber")     return {"amber",     RGB8(12,  8,  4, 0.85), RGB8(255,200,140), RGB8(255,140, 40), RGB8(180,128, 80), RGB8(255,140, 40)};
    if (name == "synthwave") return {"synthwave", RGB8(18,  6, 36, 0.85), RGB8(255,180,250), RGB8(255, 80,180), RGB8(180,100,180), RGB8(255, 80,180)};
    if (name == "matrix")    return {"matrix",    RGB8( 0, 12,  4, 0.85), RGB8( 80,255, 80), RGB8(  0,200,  0), RGB8( 40,140, 40), RGB8(  0,255, 80)};
    if (name == "vault")     return {"vault",     RGB8(16, 24, 32, 0.85), RGB8(180,220,255), RGB8( 80,160,255), RGB8(100,140,180), RGB8( 80,160,255)};
    if (name == "sakura")    return {"sakura",    RGB8(36, 12, 26, 0.85), RGB8(255,200,220), RGB8(255,140,180), RGB8(180,140,160), RGB8(255,140,180)};
    if (name == "mono")      return {"mono",      RGB8(12, 12, 12, 0.85), RGB8(220,220,220), RGB8(180,180,180), RGB8(120,120,120), RGB8(220,220,220)};
    return                        {"hardcore",  RGB8(12, 14, 22, 0.85), RGB8(220,220,232), RGB8(  0,240,255), RGB8(110,120,146), RGB8(  0,240,255)};
}

constexpr int kPanelW      = 296;
constexpr int kPanelMargin = 12;
constexpr int kLineHeight  = 16;
constexpr int kPaddingX    = 10;
constexpr int kPaddingY    = 8;
constexpr int kGraphH      = 44;

int LinesForLevel(int level) {
    switch (level) {
        case 1:  return 2;
        case 2:  return 8;
        case 3:  return 8;
        default: return 0;
    }
}

}  // namespace

@interface PtPerfView : NSView
@property (assign) int level;
@property (assign) Theme theme;
@property (assign) pt::app::PerfStats stats;
@property (strong) NSMutableArray<NSNumber*>* history;
- (void)applyStats:(const pt::app::PerfStats&)s;
@end

@interface PtPerfPanel : NSPanel
@property (weak)   NSWindow*   parentRenderWindow;
@property (strong) PtPerfView* view;
- (instancetype)initWithParent:(NSWindow*)parent;
- (void)layoutToParent;
- (void)setLevel:(int)level;
- (void)applyTheme:(std::string_view)name;
@end

// ---------------------------------------------------------------------------
@implementation PtPerfView

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    self.wantsLayer = YES;
    self.layer.cornerRadius = 6.0;
    self.layer.masksToBounds = YES;
    self.history = [NSMutableArray array];
    self.theme = PaletteFor("hardcore");
    return self;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event { (void)event; return NO; }

// Forward all mouse events to the parent window so the overlay never
// captures clicks (it's a read-only HUD over the game viewport).
- (NSView*)hitTest:(NSPoint)point { (void)point; return nil; }

- (void)applyStats:(const pt::app::PerfStats&)s {
    self.stats = s;
    [self.history removeAllObjects];
    for (float v : s.frame_ms_history) {
        [self.history addObject:@(v)];
    }
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirty {
    (void)dirty;
    NSRect b = self.bounds;
    Theme  t = self.theme;
    pt::app::PerfStats s = self.stats;

    // Panel background.
    [t.panel setFill];
    NSRectFillUsingOperation(b, NSCompositingOperationSourceOver);

    // Top accent line.
    [t.accent setFill];
    NSRectFill(NSMakeRect(0, b.size.height - 1, b.size.width, 1));

    NSFont* font = [NSFont monospacedSystemFontOfSize:11
                                               weight:NSFontWeightRegular];

    // AppKit y-up; place text from the top by tracking a top-relative
    // offset that decrements per row. `__block` lets the closure
    // below mutate it (Objective-C captures are const by default).
    __block CGFloat top = b.size.height - kPaddingY - kLineHeight;

    auto drawRow = ^(NSColor* col, NSString* text) {
        NSDictionary* attrs = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: col,
        };
        [text drawAtPoint:NSMakePoint(kPaddingX, top) withAttributes:attrs];
        top -= kLineHeight;
    };

    // Tier-1 lines.
    NSString* fps_line = [NSString stringWithFormat:@"FPS  %5.0f", s.fps];
    NSString* ms_line  = [NSString stringWithFormat:@"ms   %5.2f  (%.2f / %.2f)",
                          s.frame_ms_avg, s.frame_ms_min, s.frame_ms_max];
    drawRow(t.accent, fps_line);
    drawRow(t.text,   ms_line);

    if (self.level >= 2) {
        double mem_mb = (double)s.gpu_bytes / (1024.0 * 1024.0);
        drawRow(t.dim, [NSString stringWithFormat:@"backend     %s",   s.backend ?: ""]);
        drawRow(t.dim, [NSString stringWithFormat:@"resolution  %dx%d", s.width, s.height]);
        drawRow(t.dim, [NSString stringWithFormat:@"gpu memory  %.1f MB", mem_mb]);
        drawRow(t.dim, [NSString stringWithFormat:@"spp         %d",   s.spp]);
        drawRow(t.dim, [NSString stringWithFormat:@"bounces     %d",   s.max_bounces]);
        drawRow(t.dim, [NSString stringWithFormat:@"primitives  %lu",  (unsigned long)s.primitives]);
    }

    if (self.level >= 3 && self.history.count > 0) {
        CGFloat gx = kPaddingX;
        CGFloat gy = kPaddingY;
        CGFloat gw = b.size.width - 2 * kPaddingX;
        CGFloat gh = kGraphH;

        // Border.
        [t.dim setStroke];
        NSBezierPath* frame = [NSBezierPath bezierPathWithRect:NSMakeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1)];
        frame.lineWidth = 1.0;
        [frame stroke];

        // Find peak for Y scale.
        float peak = 1.0f;
        for (NSNumber* n in self.history) {
            peak = std::max(peak, n.floatValue);
        }
        peak = std::max(peak, 0.5f);

        // 60 fps reference at 16.67 ms.
        float ref_ms = 16.667f;
        if (ref_ms < peak) {
            CGFloat ref_y = gy + 1 + (ref_ms / peak) * (gh - 2);
            [t.dim setFill];
            NSRectFill(NSMakeRect(gx + 1, ref_y, gw - 2, 1));
        }

        // Sparkline.
        NSInteger avail = std::max((NSInteger)2, (NSInteger)(gw - 2));
        NSInteger n = (NSInteger)self.history.count;
        NSBezierPath* line = [NSBezierPath bezierPath];
        line.lineWidth = 1.0;
        BOOL first = YES;
        for (NSInteger i = 0; i < std::min(n, avail); ++i) {
            NSInteger hi = n - 1 - (avail - 1 - i);
            if (hi < 0) hi = 0;
            float v = self.history[hi].floatValue;
            CGFloat px = gx + 1 + i;
            CGFloat py = gy + 1 + (v / peak) * (gh - 2);
            NSPoint p = NSMakePoint(px, py);
            if (first) { [line moveToPoint:p]; first = NO; }
            else        [line lineToPoint:p];
        }
        [t.graph setStroke];
        [line stroke];
    }
}

@end

// ---------------------------------------------------------------------------
@implementation PtPerfPanel

+ (BOOL)canBecomeKeyWindow  { return NO; }
+ (BOOL)canBecomeMainWindow { return NO; }

- (BOOL)canBecomeKeyWindow  { return NO; }
- (BOOL)canBecomeMainWindow { return NO; }

- (instancetype)initWithParent:(NSWindow*)parent {
    NSRect frame = NSMakeRect(0, 0, kPanelW, 64);
    self = [super initWithContentRect:frame
                            styleMask:NSWindowStyleMaskBorderless
                              backing:NSBackingStoreBuffered
                                defer:NO];
    if (!self) return nil;

    self.parentRenderWindow = parent;
    self.opaque = NO;
    self.backgroundColor = [NSColor clearColor];
    self.hasShadow = NO;
    self.level = NSStatusWindowLevel;     // floats above the render window
    self.movableByWindowBackground = NO;
    self.releasedWhenClosed = NO;
    self.hidesOnDeactivate = YES;
    self.ignoresMouseEvents = YES;        // never steal clicks

    PtPerfView* v = [[PtPerfView alloc] initWithFrame:NSMakeRect(0, 0, kPanelW, 64)];
    self.contentView = v;
    self.view = v;

    return self;
}

- (void)layoutToParent {
    NSWindow* parent = self.parentRenderWindow;
    if (!parent) return;
    NSRect pf = parent.frame;
    int level = self.view.level;
    int lines = LinesForLevel(level);
    CGFloat h = kPaddingY * 2 + lines * kLineHeight;
    if (level >= 3) h += kGraphH + kPaddingY;
    NSRect f = NSMakeRect(pf.origin.x + pf.size.width - kPanelW - kPanelMargin,
                          pf.origin.y + pf.size.height - h - kPanelMargin,
                          kPanelW, h);
    [self setFrame:f display:YES];
    self.view.frame = NSMakeRect(0, 0, kPanelW, h);
    [self.view setNeedsDisplay:YES];
}

- (void)setLevel:(int)level {
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    self.view.level = level;
    if (level == 0) {
        [self orderOut:nil];
    } else {
        [self layoutToParent];
        [self orderFront:nil];
    }
}

- (void)applyTheme:(std::string_view)name {
    self.view.theme = PaletteFor(name);
    [self.view setNeedsDisplay:YES];
}

@end

// ---------------------------------------------------------------------------
namespace pt::app {

PerfOverlay::PerfOverlay() {
    PtPerfPanel* p = nil;
    opaque_ = (__bridge_retained void*)p;
}
PerfOverlay::~PerfOverlay() {
    if (opaque_) {
        PtPerfPanel* p = (__bridge_transfer PtPerfPanel*)opaque_;
        [p.parentRenderWindow removeChildWindow:p];
        [p orderOut:nil];
        p = nil;
        opaque_ = nullptr;
    }
}

bool PerfOverlay::Init(void* ns_window) {
    if (!ns_window) return false;
    NSWindow* parent = (__bridge NSWindow*)ns_window;
    PtPerfPanel* panel = [[PtPerfPanel alloc] initWithParent:parent];
    if (!panel) return false;
    if (opaque_) {
        // Replace any prior instance (Init shouldn't normally be
        // called twice, but be defensive).
        PtPerfPanel* prev = (__bridge_transfer PtPerfPanel*)opaque_;
        [prev.parentRenderWindow removeChildWindow:prev];
        [prev orderOut:nil];
    }
    // Anchor the panel as a child window of the engine's NSWindow so it
    // stays z-ordered above the render window even after the user clicks
    // back into the main window. Without this, NSStatusWindowLevel alone
    // is not enough -- macOS reorders all of the app's non-child windows
    // when one of them becomes key, and the panel slides behind. Child
    // windows are pinned above their parent regardless of focus changes.
    // Same pattern as ConsoleOverlay.mm.
    [parent addChildWindow:panel ordered:NSWindowAbove];
    opaque_ = (__bridge_retained void*)panel;
    return true;
}

void PerfOverlay::Shutdown() {
    if (!opaque_) return;
    PtPerfPanel* p = (__bridge_transfer PtPerfPanel*)opaque_;
    [p.parentRenderWindow removeChildWindow:p];
    [p orderOut:nil];
    opaque_ = nullptr;
}

void PerfOverlay::SetLevel(int level) {
    if (!opaque_) return;
    [(__bridge PtPerfPanel*)opaque_ setLevel:level];
}

int PerfOverlay::Level() const {
    if (!opaque_) return 0;
    return [(__bridge PtPerfPanel*)opaque_ view].level;
}

void PerfOverlay::Update(const PerfStats& stats) {
    if (!opaque_) return;
    PtPerfPanel* panel = (__bridge PtPerfPanel*)opaque_;
    if (panel.view.level == 0) return;
    [panel.view applyStats:stats];
}

void PerfOverlay::NotifyParentResized(int /*w*/, int /*h*/) {
    if (!opaque_) return;
    PtPerfPanel* panel = (__bridge PtPerfPanel*)opaque_;
    if (panel.view.level == 0) return;
    [panel layoutToParent];
}

void PerfOverlay::ApplyTheme(std::string_view name) {
    if (!opaque_) return;
    [(__bridge PtPerfPanel*)opaque_ applyTheme:name];
}

}  // namespace pt::app

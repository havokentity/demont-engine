// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Native macOS performance overlay -- floating NSPanel anchored to
// the top-right of the engine's NSWindow client area, drawing a tiered
// stats readout with NSAttributedString + NSBezierPath for the
// sparkline.  Mirrors the Win32 impl visually; differs only where the
// AppKit/GDI APIs diverge.

#import <Cocoa/Cocoa.h>

#include "PerfOverlay.h"
#include "../console/Console.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Theme members are NSColor* held in a plain C struct.  Without an
// explicit __strong annotation, ObjC pointers inside C structs default
// to __unsafe_unretained under ARC, which would let the autoreleased
// NSColor*s returned from PaletteFor() get released when the
// surrounding autorelease pool drains -- the Theme struct stored on
// the view via @property(assign) would then carry dangling pointers.
// __strong opts each pointer into ARC retain/release, so struct
// assignment (`self.view.theme = PaletteFor(name)`) correctly
// retains the new colours and releases the old ones.
struct Theme {
    const char* name;
    __strong NSColor* panel;
    __strong NSColor* text;
    __strong NSColor* accent;
    __strong NSColor* dim;
    __strong NSColor* graph;
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

// Baseline (1.0x) sizing constants. Effective layout uses these
// multiplied by the live `r_perf_overlay_scale` cvar value (clamped to
// [0.5, 3.0]) -- mirrors WinPerf::EnsureScale() on Windows.
constexpr int kFontPt      = 11;
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

@class PtPerfPanel;

@interface PtPerfView : NSView
@property (assign) int level;
@property (assign) Theme theme;
@property (assign) pt::app::PerfStats stats;
@property (strong) NSMutableArray<NSNumber*>* history;
// Weak back-reference so a scale change inside drawRect: can ping the
// owning panel to recompute its frame (panel width / row count
// derive from the scale, same as Win32 WinPerf::Reposition()).
@property (weak) PtPerfPanel* panel;
- (void)applyStats:(const pt::app::PerfStats&)s;
// Re-poll the live `r_perf_overlay_scale` cvar. Cheap when unchanged.
// Returns YES if the cached scale changed -- the caller (drawRect:)
// uses that to trigger a panel re-layout before drawing this frame.
- (BOOL)ensureScale;
// Effective layout values, scale * baseline. Read by drawRect: and
// PtPerfPanel's layoutToParent so they share one source of truth.
- (CGFloat)scale;
- (int)scaledPanelW;
- (int)scaledLineHeight;
- (int)scaledGraphH;
- (int)scaledPaddingX;
- (int)scaledPaddingY;
@end

@interface PtPerfPanel : NSPanel
@property (weak)   NSWindow*   parentRenderWindow;
@property (strong) PtPerfView* view;
- (instancetype)initWithParent:(NSWindow*)parent;
- (void)layoutToParent;
// Renamed from setLevel: to avoid colliding with NSWindow/NSPanel's
// own -setLevel:(NSWindowLevel) selector.  The collision meant
// `self.level = NSStatusWindowLevel` in initWithParent: dispatched
// to OUR int-tier setter instead of NSWindow's z-order level
// setter, leaving the panel without its expected status-window
// floating behaviour.
- (void)setOverlayTier:(int)tier;
- (void)applyTheme:(std::string_view)name;
@end

// ---------------------------------------------------------------------------
@implementation PtPerfView {
    // Effective r_perf_overlay_scale value last applied by -ensureScale.
    // Cvar IS the source of truth (matches Win32 WinPerf::EnsureScale).
    // Polled from drawRect: -- one cvar lookup + float compare per
    // paint when unchanged, NSFont rebuild + LOG_INFO when not.
    float    _scale;
    NSFont*  _font;
}

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    self.wantsLayer = YES;
    self.layer.cornerRadius = 6.0;
    self.layer.masksToBounds = YES;
    self.history = [NSMutableArray array];
    self.theme = PaletteFor("hardcore");
    _scale = 1.0f;
    _font  = [NSFont monospacedSystemFontOfSize:kFontPt
                                         weight:NSFontWeightRegular];
    return self;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event { (void)event; return NO; }

// Forward all mouse events to the parent window so the overlay never
// captures clicks (it's a read-only HUD over the game viewport).
- (NSView*)hitTest:(NSPoint)point { (void)point; return nil; }

- (CGFloat)scale            { return (CGFloat)_scale; }
- (int)scaledPanelW         { return std::max(64,  (int)(kPanelW      * _scale + 0.5f)); }
- (int)scaledLineHeight     { return std::max(8,   (int)(kLineHeight  * _scale + 0.5f)); }
- (int)scaledGraphH         { return std::max(16,  (int)(kGraphH      * _scale + 0.5f)); }
- (int)scaledPaddingX       { return std::max(2,   (int)(kPaddingX    * _scale + 0.5f)); }
- (int)scaledPaddingY       { return std::max(2,   (int)(kPaddingY    * _scale + 0.5f)); }

- (BOOL)ensureScale {
    auto* v = pt::console::Console::Get().FindCVar("r_perf_overlay_scale");
    if (v == nullptr) return NO;
    char* end = nullptr;
    float requested = std::strtof(v->value.c_str(), &end);
    // Reject non-finite (NaN / +-Inf) before any math -- same trap
    // WinPerf::EnsureScale guards against. NaN propagates through the
    // < / > clamps unordered, sneaks past the early-exit, and ends up
    // multiplying every layout constant for an undefined-behaviour
    // float-to-int cast.
    if (end == v->value.c_str() || !std::isfinite(requested)) requested = 1.0f;
    if (requested < 0.5f) requested = 0.5f;
    if (requested > 3.0f) requested = 3.0f;
    if (std::fabs(requested - _scale) < 1e-3f) return NO;

    _scale = requested;
    const CGFloat new_pt = (CGFloat)kFontPt * requested;
    _font  = [NSFont monospacedSystemFontOfSize:new_pt
                                         weight:NSFontWeightRegular];
    LOG_INFO("PerfOverlay: r_perf_overlay_scale={:.2f} -> "
             "font={:.1f}pt panel_w={} line_h={} graph_h={}",
             requested, new_pt, [self scaledPanelW],
             [self scaledLineHeight], [self scaledGraphH]);
    return YES;
}

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
    // Polling here (NOT applyStats:) is the single point where any cvar
    // setter -- web GUI, in-game overlay, autoexec -- gets reflected.
    // applyStats only fires on Update(), so a scale change with the
    // overlay paused (level=0 + then re-enabled) wouldn't propagate.
    if ([self ensureScale]) {
        // Panel width / row count derive from scale; ping the parent
        // panel so its frame catches up before this paint draws into
        // a now-wrong-size view. Triggers a follow-up drawRect: with
        // the corrected bounds.
        if (self.panel != nil) [self.panel layoutToParent];
    }
    NSRect b = self.bounds;
    Theme  t = self.theme;
    pt::app::PerfStats s = self.stats;

    // Panel background.
    [t.panel setFill];
    NSRectFillUsingOperation(b, NSCompositingOperationSourceOver);

    // Top accent line.
    [t.accent setFill];
    NSRectFill(NSMakeRect(0, b.size.height - 1, b.size.width, 1));

    NSFont* font = _font;
    const int line_h = [self scaledLineHeight];
    const int padX   = [self scaledPaddingX];
    const int padY   = [self scaledPaddingY];

    // AppKit y-up; place text from the top by tracking a top-relative
    // offset that decrements per row. `__block` lets the closure
    // below mutate it (Objective-C captures are const by default).
    __block CGFloat top = b.size.height - padY - line_h;

    auto drawRow = ^(NSColor* col, NSString* text) {
        NSDictionary* attrs = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: col,
        };
        [text drawAtPoint:NSMakePoint(padX, top) withAttributes:attrs];
        top -= line_h;
    };

    // Tier-1 lines.
    NSString* fps_line = [NSString stringWithFormat:@"FPS  %5.0f", s.fps];
    NSString* ms_line  = [NSString stringWithFormat:@"ms   %5.2f  (%.2f / %.2f)",
                          s.frame_ms_avg, s.frame_ms_min, s.frame_ms_max];
    drawRow(t.accent, fps_line);
    drawRow(t.text,   ms_line);

    if (self.level >= 2) {
        double mem_mb = (double)s.gpu_bytes / (1024.0 * 1024.0);
        drawRow(t.dim, [NSString stringWithFormat:@"backend     %s",   s.backend ? s.backend : ""]);
        drawRow(t.dim, [NSString stringWithFormat:@"resolution  %dx%d", s.width, s.height]);
        drawRow(t.dim, [NSString stringWithFormat:@"gpu memory  %.1f MB", mem_mb]);
        drawRow(t.dim, [NSString stringWithFormat:@"spp         %d",   s.spp]);
        drawRow(t.dim, [NSString stringWithFormat:@"bounces     %d",   s.max_bounces]);
        drawRow(t.dim, [NSString stringWithFormat:@"primitives  %lu",  (unsigned long)s.primitives]);
    }

    if (self.level >= 3 && self.history.count > 0) {
        CGFloat gx = padX;
        CGFloat gy = padY;
        CGFloat gw = b.size.width - 2 * padX;
        CGFloat gh = [self scaledGraphH];

        // Stroke widths scale with the overlay scale -- without this
        // a 2x scale draws a graph that's twice as big BUT outlined
        // and traced with the same 1px lines, so the border + sparkline
        // look anaemic and visually disconnected from the rest of the
        // panel which is fully scaled. Round to nearest pixel,
        // minimum 1, so 0.5x scale doesn't degenerate to invisible.
        CGFloat scale = [self scale];
        CGFloat strokePx = std::max((CGFloat)1.0, std::round(scale));

        // Border. Inset by half the stroke width so the rect's pixel
        // bounds match the graph area (NSBezierPath strokes centred
        // on the path).
        [t.dim setStroke];
        CGFloat half = strokePx * 0.5;
        NSBezierPath* frame = [NSBezierPath bezierPathWithRect:
            NSMakeRect(gx + half, gy + half, gw - strokePx, gh - strokePx)];
        frame.lineWidth = strokePx;
        [frame stroke];

        // Find peak for Y scale.
        float peak = 1.0f;
        for (NSNumber* n in self.history) {
            peak = std::max(peak, n.floatValue);
        }
        peak = std::max(peak, 0.5f);

        // 60 fps reference at 16.67 ms. Inset from the border so the
        // reference rule never overlaps the frame stroke; height
        // matches strokePx so it stays visually proportional with
        // the rest of the graph at all scales.
        float ref_ms = 16.667f;
        if (ref_ms < peak) {
            CGFloat inset = strokePx;
            CGFloat usable_h = gh - 2 * inset;
            CGFloat ref_y = gy + inset + (ref_ms / peak) * usable_h;
            [t.dim setFill];
            NSRectFill(NSMakeRect(gx + inset, ref_y,
                                  gw - 2 * inset, strokePx));
        }

        // Sparkline. Available drawing band is the interior of the
        // border. Older samples are right-aligned to the most-recent
        // edge: when history < available pixels, the line starts in
        // the middle of the graph and grows leftward as history fills,
        // matching the Win32 reference. Stroke width tracks scale.
        CGFloat inset = strokePx;
        NSInteger avail = std::max((NSInteger)2, (NSInteger)(gw - 2 * inset));
        NSInteger n = (NSInteger)self.history.count;
        NSInteger draw = std::min(n, avail);
        NSInteger startCol = avail - draw;       // right-align partial history
        NSInteger startHi  = std::max((NSInteger)0, n - avail);
        NSBezierPath* line = [NSBezierPath bezierPath];
        line.lineWidth = strokePx;
        BOOL first = YES;
        CGFloat usable_h = gh - 2 * inset;
        for (NSInteger c = 0; c < draw; ++c) {
            float v = self.history[startHi + c].floatValue;
            CGFloat px = gx + inset + (CGFloat)(startCol + c);
            CGFloat py = gy + inset + (v / peak) * usable_h;
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
    v.panel  = self;       // weak back-ref so EnsureScale can ping us

    return self;
}

- (void)layoutToParent {
    NSWindow* parent = self.parentRenderWindow;
    if (!parent) return;
    // Pull the live scale BEFORE computing height so a cvar change
    // takes effect on the same NotifyParentResized / SetLevel hop.
    [self.view ensureScale];
    NSRect pf = parent.frame;
    int level = self.view.level;
    int lines = LinesForLevel(level);
    const int line_h = [self.view scaledLineHeight];
    const int padY   = [self.view scaledPaddingY];
    const int panelW = [self.view scaledPanelW];
    CGFloat h = padY * 2 + lines * line_h;
    if (level >= 3) h += [self.view scaledGraphH] + padY;
    NSRect f = NSMakeRect(pf.origin.x + pf.size.width - panelW - kPanelMargin,
                          pf.origin.y + pf.size.height - h - kPanelMargin,
                          panelW, h);
    [self setFrame:f display:YES];
    self.view.frame = NSMakeRect(0, 0, panelW, h);
    [self.view setNeedsDisplay:YES];
}

- (void)setOverlayTier:(int)tier {
    if (tier < 0) tier = 0;
    if (tier > 3) tier = 3;
    self.view.level = tier;
    if (tier == 0) {
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
    [(__bridge PtPerfPanel*)opaque_ setOverlayTier:level];
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

void PerfOverlay::Repaint() {
    // Mac: dispatch setNeedsDisplay: on the main thread (AppKit
    // requirement). Win32 implementation lives in PerfOverlay_Win32.cpp.
    if (!opaque_) return;
    PtPerfPanel* panel = (__bridge PtPerfPanel*)opaque_;
    dispatch_async(dispatch_get_main_queue(), ^{
        [panel.contentView setNeedsDisplay:YES];
    });
}

}  // namespace pt::app

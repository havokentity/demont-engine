// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Native console overlay implementation. AppKit + CoreText.
//
// Architectural note: the panel lives in its own borderless transparent
// NSWindow, added as a CHILD WINDOW of the renderer's NSWindow. Earlier
// implementations added the overlay as a subview, which silently forced
// the parent's content view into layer-backed mode and broke the GLFW
// NSOpenGLContext's drawing path (blank window in the software backend).
// A child window keeps the parent view hierarchy untouched, so all
// three backends (software/metal/vulkan) render unaffected.

#include "ConsoleOverlay.h"

#include "../console/Completion.h"
#include "../console/Console.h"
#include "../core/Log.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <cmath>
#include <cstdlib>

// Tiny logo-drawing NSView. Same glyph as the web console: a hexagon
// (mesh primitive) framing a three-bounce ray with hit-point dots.
// Frame strokes in --accent; ray + dots in --logo-ray (the magenta-pink
// contrast colour) so the icon mirrors the ASCII banner two-tone scheme.
@interface PtLogoView : NSView
@property (strong) NSColor* frameColor;
@property (strong) NSColor* rayColor;
@end

@implementation PtLogoView

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    // Sensible defaults; PtConsoleView pushes the real palette colours
    // immediately after construction and again on every theme change.
    self.frameColor = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:1.0];
    self.rayColor   = [NSColor colorWithCalibratedRed:1.0 green:0.37 blue:0.64 alpha:1.0];
    return self;
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    NSGraphicsContext* gc = [NSGraphicsContext currentContext];
    if (gc == nil) return;

    // Source-of-truth design space is 32x32; scale to whatever frame we got.
    CGFloat s = MIN(self.bounds.size.width, self.bounds.size.height) / 32.0;
    NSAffineTransform* xform = [NSAffineTransform transform];
    [xform translateXBy:(self.bounds.size.width - 32 * s) * 0.5
                    yBy:(self.bounds.size.height - 32 * s) * 0.5];
    [xform scaleBy:s];
    [xform concat];

    // Hex frame (accent colour).
    [self.frameColor setStroke];
    NSBezierPath* hex = [NSBezierPath bezierPath];
    [hex moveToPoint:NSMakePoint(16.0, 2.5)];
    [hex lineToPoint:NSMakePoint(27.5, 9.0)];
    [hex lineToPoint:NSMakePoint(27.5, 23.0)];
    [hex lineToPoint:NSMakePoint(16.0, 29.5)];
    [hex lineToPoint:NSMakePoint(4.5, 23.0)];
    [hex lineToPoint:NSMakePoint(4.5, 9.0)];
    [hex closePath];
    hex.lineWidth = 1.6;
    hex.lineJoinStyle = NSLineJoinStyleRound;
    [hex stroke];

    // Three-bounce ray + hit dots (ray colour).
    [self.rayColor setStroke];
    [self.rayColor setFill];
    NSBezierPath* ray = [NSBezierPath bezierPath];
    [ray moveToPoint:NSMakePoint(7.5, 11.0)];
    [ray lineToPoint:NSMakePoint(13.0, 19.0)];
    [ray lineToPoint:NSMakePoint(21.5, 14.0)];
    [ray lineToPoint:NSMakePoint(24.5, 22.0)];
    ray.lineWidth = 1.6;
    ray.lineCapStyle = NSLineCapStyleRound;
    ray.lineJoinStyle = NSLineJoinStyleRound;
    [ray stroke];

    NSBezierPath* d1 = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(13.0 - 1.4, 19.0 - 1.4, 2.8, 2.8)];
    NSBezierPath* d2 = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(21.5 - 1.4, 14.0 - 1.4, 2.8, 2.8)];
    [d1 fill];
    [d2 fill];
}

@end

#include <atomic>
#include <mutex>
#include <vector>

namespace {
pt::app::ConsoleOverlay* g_instance = nullptr;
}

@class PtConsoleView;
@class PtConsolePopupView;

@interface PtConsoleInputField : NSTextField
@property (weak) PtConsoleView* owner;
@end

// One palette per theme. Mirrors the CSS variables in web/console.css.
typedef struct {
    NSColor* border;          // backdrop hairline outline
    NSColor* accent;          // brand / prompt
    NSColor* status;          // status label tint
    NSColor* logoFrame;       // ASCII frame ░▒▓█ + ╔═╝
    NSColor* logoLetters;     // ASCII letters D M T P A T
    NSColor* logoRay;         // ASCII ray ╲ ╱ ╳ ◉ • ─
    NSColor* warn;            // log warn
    NSColor* error;           // log error
    NSColor* input;           // user input echo
    NSColor* out;             // command output
    NSColor* info;            // muted info
} PtThemePalette;

static NSColor* PtRGB(double r, double g, double b) {
    return [NSColor colorWithCalibratedRed:r green:g blue:b alpha:1.0];
}
static NSColor* PtRGBA(double r, double g, double b, double a) {
    return [NSColor colorWithCalibratedRed:r green:g blue:b alpha:a];
}

static PtThemePalette PtPaletteForTheme(NSString* name) {
    PtThemePalette p{};
    if ([name isEqualToString:@"amber"]) {
        p.border       = PtRGBA(0.91, 0.60, 0.41, 0.45);
        p.accent       = PtRGB(0.91, 0.60, 0.41);
        p.status       = PtRGBA(0.91, 0.60, 0.41, 0.65);
        p.logoFrame    = PtRGB(0.91, 0.60, 0.41);
        p.logoLetters  = PtRGB(0.95, 0.93, 0.86);
        p.logoRay      = PtRGB(0.85, 0.47, 0.34);
        p.warn         = PtRGB(0.72, 0.66, 0.41);
        p.error        = PtRGB(0.85, 0.47, 0.34);
        p.input        = PtRGB(0.91, 0.60, 0.41);
        p.out          = PtRGBA(0.95, 0.95, 0.95, 1.0);
        p.info         = PtRGBA(0.55, 0.55, 0.55, 1.0);
    } else if ([name isEqualToString:@"synthwave"]) {
        p.border       = PtRGBA(1.00, 0.37, 0.75, 0.45);
        p.accent       = PtRGB(1.00, 0.37, 0.75);
        p.status       = PtRGBA(1.00, 0.37, 0.75, 0.65);
        p.logoFrame    = PtRGB(1.00, 0.37, 0.75);
        p.logoLetters  = PtRGB(0.94, 0.90, 1.00);
        p.logoRay      = PtRGB(0.71, 0.42, 1.00);
        p.warn         = PtRGB(1.00, 0.63, 0.25);
        p.error        = PtRGB(1.00, 0.19, 0.38);
        p.input        = PtRGB(1.00, 0.37, 0.75);
        p.out          = PtRGBA(0.95, 0.95, 0.95, 1.0);
        p.info         = PtRGBA(0.55, 0.55, 0.55, 1.0);
    } else if ([name isEqualToString:@"matrix"]) {
        p.border       = PtRGBA(0.00, 1.00, 0.25, 0.45);
        p.accent       = PtRGB(0.00, 1.00, 0.25);
        p.status       = PtRGBA(0.00, 1.00, 0.25, 0.70);
        p.logoFrame    = PtRGB(0.00, 1.00, 0.25);
        p.logoLetters  = PtRGB(0.72, 1.00, 0.72);
        p.logoRay      = PtRGB(0.31, 1.00, 0.50);
        p.warn         = PtRGB(0.81, 1.00, 0.00);
        p.error        = PtRGB(1.00, 0.31, 0.31);
        p.input        = PtRGB(0.00, 1.00, 0.25);
        p.out          = PtRGB(0.72, 0.91, 0.72);
        p.info         = PtRGB(0.35, 0.54, 0.35);
    } else if ([name isEqualToString:@"vault"]) {
        p.border       = PtRGBA(1.00, 0.84, 0.42, 0.55);
        p.accent       = PtRGB(1.00, 0.84, 0.42);
        p.status       = PtRGBA(1.00, 0.72, 0.00, 0.85);
        p.logoFrame    = PtRGB(1.00, 0.72, 0.00);
        p.logoLetters  = PtRGB(1.00, 0.84, 0.42);
        p.logoRay      = PtRGB(1.00, 0.53, 0.33);
        p.warn         = PtRGB(1.00, 0.88, 0.44);
        p.error        = PtRGB(1.00, 0.38, 0.25);
        p.input        = PtRGB(1.00, 0.84, 0.42);
        p.out          = PtRGB(1.00, 0.72, 0.00);
        p.info         = PtRGB(0.67, 0.49, 0.06);
    } else if ([name isEqualToString:@"sakura"]) {
        p.border       = PtRGBA(0.83, 0.25, 0.48, 0.55);
        p.accent       = PtRGB(0.83, 0.25, 0.48);
        p.status       = PtRGBA(0.54, 0.33, 0.44, 0.85);
        p.logoFrame    = PtRGB(0.83, 0.25, 0.48);
        p.logoLetters  = PtRGB(0.29, 0.12, 0.23);
        p.logoRay      = PtRGB(0.82, 0.41, 0.12);
        p.warn         = PtRGB(0.84, 0.50, 0.19);
        p.error        = PtRGB(0.78, 0.19, 0.28);
        p.input        = PtRGB(0.83, 0.25, 0.48);
        p.out          = PtRGB(0.29, 0.12, 0.23);
        p.info         = PtRGB(0.54, 0.33, 0.44);
    } else if ([name isEqualToString:@"mono"]) {
        p.border       = PtRGBA(1.00, 1.00, 1.00, 0.40);
        p.accent       = PtRGB(0.96, 0.96, 0.96);
        p.status       = PtRGBA(0.96, 0.96, 0.96, 0.65);
        p.logoFrame    = PtRGB(0.96, 0.96, 0.96);
        p.logoLetters  = PtRGB(1.00, 1.00, 1.00);
        p.logoRay      = PtRGB(0.55, 0.55, 0.55);
        p.warn         = PtRGB(0.80, 0.80, 0.80);
        p.error        = PtRGB(1.00, 0.19, 0.19);
        p.input        = PtRGB(0.80, 0.80, 0.80);
        p.out          = PtRGB(0.96, 0.96, 0.96);
        p.info         = PtRGB(0.53, 0.53, 0.53);
    } else {
        // hardcore (default)
        p.border       = PtRGBA(0.00, 0.94, 1.00, 0.45);
        p.accent       = PtRGB(0.00, 0.94, 1.00);
        p.status       = PtRGBA(0.00, 0.94, 1.00, 0.65);
        p.logoFrame    = PtRGB(0.00, 0.94, 1.00);
        p.logoLetters  = PtRGB(0.95, 0.97, 1.00);
        p.logoRay      = PtRGB(1.00, 0.37, 0.64);
        p.warn         = PtRGB(1.00, 0.78, 0.23);
        p.error        = PtRGB(1.00, 0.29, 0.37);
        p.input        = PtRGB(1.00, 0.23, 0.55);
        p.out          = PtRGBA(0.95, 0.95, 0.95, 1.0);
        p.info         = PtRGBA(0.52, 0.52, 0.52, 1.0);
    }
    return p;
}

@interface PtConsoleView : NSView <NSTextFieldDelegate, NSTextViewDelegate>

@property (strong) NSVisualEffectView*  backdrop;
@property (strong) NSScrollView*        outputScroll;
@property (strong) NSTextView*          outputView;
@property (strong) PtConsoleInputField* inputField;
@property (strong) NSTextField*         promptLabel;
@property (strong) NSTextField*         statusLabel;
@property (strong) PtLogoView*          logoView;
@property NSUInteger                    bannerEndLocation;
@property (assign) PtThemePalette       palette;

@property NSMutableArray<NSString*>*    history;
@property NSInteger                     historyPos;

// Inline ghost preview (a dim-coloured tail of the highlighted popup
// match, drawn just past the cursor when the match is a case-
// insensitive prefix of the typed token AND the cursor is at end of
// input). Reused from the prior ghost-only era -- the popup state
// machine drives it now via -renderInlineGhost. When the highlighted
// candidate is the cvar's CURRENT value the ghost tints YELLOW
// (RGB 255,200,80), matching web's --warn / Win32's kCurrentHintColor.
@property (strong) NSTextField*         ghostLabel;

// VS Code-style completion popup. The popup view is a sibling NSView
// inside this PtConsoleView (NOT a separate NSPanel) so it inherits
// the panel's animation + theme + occlusion behaviour the same way
// the Win32 popup is a sub-rect of the child HWND. Frame + visibility
// are driven by -layoutPopup which runs whenever popup state changes.
@property (strong) PtConsolePopupView*  popupView;

- (instancetype)initWithFrame:(NSRect)frame;
- (void)appendLine:(NSString*)line level:(NSString*)level;
- (void)submitInput;
- (void)submitLine:(NSString*)line;

// Popup state machine. Mirrors WinOverlay's RefreshCompletions /
// MovePopupSelection / CommitPopup / DismissPopup 1-for-1 -- see
// ConsoleOverlay_Win32.cpp for the reference implementation, and
// src/console/Completion.h for the shared scoring engine the three
// frontends (web JS + Win32 GDI + this AppKit overlay) all use.
- (void)refreshCompletions:(BOOL)forceShow;
- (void)movePopupSelection:(NSInteger)dir;
- (void)commitPopup:(BOOL)chainNext;
- (void)dismissPopup;
- (NSInteger)popupVisibleRows;
- (void)layoutPopup;
- (void)renderInlineGhost;
// Called by PtConsolePopupView's drawRect: -- keeps the popup state
// (items / selected / scroll_offset) private to PtConsoleView.
- (void)drawPopupInRect:(NSRect)bounds;

// Polls the live `con_font_scale` cvar. Cheap when unchanged (one
// cvar lookup + float compare). Called from ConsoleOverlay::Repaint
// whenever a cvar setter pings us, so a value typed into the web
// console / autoexec / CLI propagates to the in-game overlay without
// requiring the user to type something else first. Mirrors Win32's
// EnsureFontScale() polling pattern -- only the AppKit side has to
// rebuild four NSFont instances (input / output / prompt / status)
// instead of the one HFONT Win32 carries.
- (void)applyFontScale;

// Theme-derived popup background colour. A dark base blended with a
// small fraction of the active palette accent so each theme tints the
// popup just enough to read as part of the same visual family --
// matches Win32's DimColor(theme_.panel, 0.55f). Always FULLY OPAQUE
// (NSColor blendedColorWithFraction returns alpha 1.0) so the layer
// + drawRect: fills hide the scrollback behind the popup zone.
- (NSColor*)popupBackgroundColor;
@end

// ---------------------------------------------------------------------------
// Popup background NSView. Lives as a sibling of input/scroll inside the
// PtConsoleView. drawRect: just delegates to the owner so all popup
// state (items / selected / scroll_offset / token) stays in
// PtConsoleView's C++ ivars and we don't have to fan that state across
// two ObjC classes. isFlipped:YES so row-y math runs top-down (matches
// the Win32 implementation -- easier to keep parity).
@interface PtConsolePopupView : NSView
@property (weak) PtConsoleView* owner;
@end

// ---------------------------------------------------------------------------
@interface PtConsolePanel : NSPanel
@property (weak) NSWindow* parentRenderWindow;
@property (strong) PtConsoleView* consoleView;
@property BOOL isShown;
@property id eventMonitor;
- (instancetype)initWithParent:(NSWindow*)parent;
- (void)layoutToParent;
- (void)show;
- (void)hide;
- (void)toggle;
@end

// ---------------------------------------------------------------------------
@implementation PtConsoleInputField

- (BOOL)becomeFirstResponder {
    BOOL ok = [super becomeFirstResponder];
    if (ok) {
        NSText* editor = [self currentEditor];
        if (editor) [editor setSelectedRange:NSMakeRange(self.stringValue.length, 0)];
    }
    return ok;
}
@end

// ---------------------------------------------------------------------------
@implementation PtConsolePopupView
- (BOOL)isFlipped { return YES; }
- (void)drawRect:(NSRect)dirtyRect {
    PtConsoleView* o = self.owner;
    if (o == nil) return;
    [o drawPopupInRect:dirtyRect];
}
// Eat clicks so a stray mouse-down inside the popup zone doesn't punch
// through to the scrollback selection underneath. Today the popup is
// keyboard-driven only (no row-clicks), but eating clicks keeps the
// behaviour predictable and matches the Win32 child-window's natural
// hit-testing.
- (void)mouseDown:(NSEvent*)event { (void)event; }
@end

// ---------------------------------------------------------------------------
@implementation PtConsoleView {
    // VS Code-style completion popup state. Held as plain C++ ivars
    // (not @properties) because std::vector<CompletionMatch> + TokenInfo
    // are non-trivial C++ types -- @property storage doesn't model
    // their constructors/destructors. ARC + Objective-C++ accept C++
    // ivars in the @implementation block.
    //
    // Mirrors WinOverlay::PopupState in ConsoleOverlay_Win32.cpp:
    //   _popupActive        : false when popup hidden, true when shown
    //   _popupItems         : ranked candidates from BuildCompletions
    //   _popupSelected      : highlighted row index, or -1 when empty
    //   _popupToken         : word at cursor at time of last refresh
    //                         (start/end byte indices ready for splice)
    //   _popupScrollOffset  : top-of-window index when items >
    //                         visible-row budget. Kept in sync with
    //                         _popupSelected by movePopupSelection:.
    BOOL                                       _popupActive;
    std::vector<pt::console::CompletionMatch>  _popupItems;
    NSInteger                                  _popupSelected;
    pt::console::TokenInfo                     _popupToken;
    NSInteger                                  _popupScrollOffset;
    // Effective con_font_scale value last applied by -applyFontScale.
    // Driven by the `con_font_scale` cvar; polled from
    // ConsoleOverlay::Repaint so cvar setters propagate live. Default
    // 1.0 = baseline 13/12/9 pt fonts (input / output / status).
    float                                      _fontScale;
}

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;

    self.wantsLayer = YES;
    self.layer.masksToBounds = NO;

    self.history    = [NSMutableArray array];
    self.historyPos = 0;
    self.palette    = PtPaletteForTheme(@"hardcore");

    _popupActive       = NO;
    _popupSelected     = -1;
    _popupScrollOffset = 0;
    _fontScale         = 1.0f;

    // Two-layer backdrop. NSVisualEffectView gives the blurred-edge
    // vibrancy; a dark tint sub-layer on TOP of it guarantees text
    // contrast even when the renderer behind is mid-tone. Without the
    // tint, light cyan/white text washed out against the gray scene
    // background visible through the vibrancy.
    NSVisualEffectView* bg = [[NSVisualEffectView alloc] initWithFrame:self.bounds];
    bg.material = NSVisualEffectMaterialPopover;     // more opaque than HUDWindow
    bg.blendingMode = NSVisualEffectBlendingModeBehindWindow;
    bg.state = NSVisualEffectStateActive;
    bg.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    bg.wantsLayer = YES;
    bg.layer.cornerRadius = 8.0;
    bg.layer.maskedCorners = kCALayerMinXMinYCorner | kCALayerMaxXMinYCorner;
    bg.layer.borderWidth = 1.0;
    bg.layer.borderColor = self.palette.border.CGColor;
    [self addSubview:bg];
    self.backdrop = bg;

    // Dark tint sub-layer on top of the vibrancy so text always has
    // adequate contrast. Fades out slightly at the bottom so the input
    // row sits closer to the rendered scene below.
    NSView* tint = [[NSView alloc] initWithFrame:bg.bounds];
    tint.wantsLayer = YES;
    tint.layer.cornerRadius = 8.0;
    tint.layer.maskedCorners = kCALayerMinXMinYCorner | kCALayerMaxXMinYCorner;
    CAGradientLayer* gradient = [CAGradientLayer layer];
    gradient.frame = tint.bounds;
    gradient.colors = @[
        (id)[NSColor colorWithCalibratedRed:0.024 green:0.031 blue:0.051 alpha:0.86].CGColor,
        (id)[NSColor colorWithCalibratedRed:0.024 green:0.031 blue:0.051 alpha:0.78].CGColor,
    ];
    gradient.startPoint = CGPointMake(0.5, 1.0);
    gradient.endPoint   = CGPointMake(0.5, 0.0);
    tint.layer = gradient;
    tint.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [bg addSubview:tint];

    NSScrollView* scroll = [[NSScrollView alloc]
        initWithFrame:NSMakeRect(0, 36, frame.size.width, frame.size.height - 36)];
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scroll.hasVerticalScroller = YES;
    scroll.borderType = NSNoBorder;
    scroll.drawsBackground = NO;

    NSTextView* tv = [[NSTextView alloc] initWithFrame:scroll.bounds];
    tv.editable = NO;
    tv.selectable = YES;
    tv.drawsBackground = NO;
    tv.textContainerInset = NSMakeSize(14, 8);
    tv.font = [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular];
    tv.textColor = [NSColor colorWithWhite:0.92 alpha:1.0];
    tv.autoresizingMask = NSViewWidthSizable;
    tv.minSize = NSMakeSize(0, scroll.bounds.size.height);
    tv.maxSize = NSMakeSize(FLT_MAX, FLT_MAX);
    tv.verticallyResizable = YES;
    tv.horizontallyResizable = NO;
    tv.textContainer.widthTracksTextView = YES;
    tv.textContainer.containerSize = NSMakeSize(scroll.bounds.size.width, FLT_MAX);
    scroll.documentView = tv;
    [self addSubview:scroll];
    self.outputScroll = scroll;
    self.outputView = tv;

    NSTextField* prompt = [NSTextField labelWithString:@">"];
    prompt.frame = NSMakeRect(14, 8, 14, 22);
    prompt.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightSemibold];
    prompt.textColor = self.palette.accent;
    prompt.bezeled = NO;
    prompt.drawsBackground = NO;
    [self addSubview:prompt];
    self.promptLabel = prompt;

    NSTextField* status = [NSTextField labelWithString:@"DEMONT · PATHTRACER · CONSOLE"];
    status.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightSemibold];
    status.textColor = self.palette.status;
    status.frame = NSMakeRect(frame.size.width - 240, frame.size.height - 24, 200, 16);
    status.alignment = NSTextAlignmentRight;
    status.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [self addSubview:status];
    self.statusLabel = status;

    // Top-left corner logo glyph -- same design as the web console.
    PtLogoView* logo = [[PtLogoView alloc] initWithFrame:NSMakeRect(14, frame.size.height - 28, 18, 18)];
    logo.autoresizingMask = NSViewMinYMargin;
    logo.wantsLayer = YES;
    logo.frameColor = self.palette.accent;
    logo.rayColor   = self.palette.logoRay;
    [self addSubview:logo];
    self.logoView = logo;

    NSRect inputFrame = NSMakeRect(32, 8, frame.size.width - 46, 22);

    // Ghost label sits *behind* the input with the same frame, font and
    // metrics.  Its attributed string has two runs: the typed prefix in
    // a fully-transparent colour (so it occupies the exact width of the
    // user's text) followed by the candidate's tail in the palette
    // dim/info colour.  The transparent prefix keeps the visible tail
    // aligned right after the cursor without pixel-measuring text.
    NSTextField* ghost = [[NSTextField alloc] initWithFrame:inputFrame];
    ghost.autoresizingMask = NSViewWidthSizable;
    ghost.bezeled = NO;
    ghost.bordered = NO;
    ghost.drawsBackground = NO;
    ghost.editable = NO;
    ghost.selectable = NO;
    ghost.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    ghost.stringValue = @"";
    [self addSubview:ghost];
    self.ghostLabel = ghost;

    PtConsoleInputField* in = [[PtConsoleInputField alloc]
        initWithFrame:inputFrame];
    in.owner = self;
    in.autoresizingMask = NSViewWidthSizable;
    in.bezeled = NO;
    in.bordered = NO;
    in.drawsBackground = NO;
    in.focusRingType = NSFocusRingTypeNone;
    in.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    in.textColor = [NSColor whiteColor];
    in.placeholderString = @"sys_info  ·  Tab opens completions  ·  Up/Down history";
    in.delegate = self;
    [self addSubview:in];
    self.inputField = in;

    // Popup background view -- starts hidden; layoutPopup sizes + shows
    // it when the popup state is active. Anchored at the bottom (above
    // the input row) with NSViewMaxYMargin so it stays put when the
    // panel resizes vertically. Width tracks the panel width so the
    // description column expands on a wider window. Z-ordered above the
    // input/ghost so its background hides them while open -- matches
    // the Win32 implementation where the popup is painted last.
    PtConsolePopupView* popup = [[PtConsolePopupView alloc]
        initWithFrame:NSMakeRect(14, 36, frame.size.width - 28, 0)];
    popup.owner = self;
    popup.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
    popup.hidden = YES;
    popup.wantsLayer = YES;
    // Set the LAYER's backing colour so the popup is opaque from the
    // compositor up, not just inside drawRect:'s NSRectFill call. With
    // wantsLayer=YES + the panel's NSVisualEffectView vibrancy host,
    // a drawRect-only fill leaves the layer's compositing alpha
    // unchanged -- and the scrollback NSTextView underneath bleeds
    // through any subsequent non-opaque overlay (e.g. the previous
    // 22%-alpha accent selection band). Setting layer.backgroundColor
    // pins the layer itself opaque, so any fill we do on top inherits
    // a solid base regardless of compositing path. Theme switches
    // refresh this in -applyTheme: so the bg picks up the new accent.
    popup.layer.backgroundColor = [[self popupBackgroundColor] CGColor];
    [self addSubview:popup];
    self.popupView = popup;

    // Hex banner with chevroned inner box and a bouncing-ray scaffold.
    // Cyan frame, magenta-pink rays + hit dots, bright white letters.
    NSArray<NSArray<NSString*>*>* banner = @[
        @[@"frame",   @"        ░▒▓██████████▓▒░"],
        @[@"frame",   @"     ░▒▓██╔═══════════╗▓▒░"],
        @[@"letters", @"   ░▓██╔═╝   D · M · T   ╚═╗██▓░"],
        @[@"ray",     @"  ▒█░  ╔╝  ╲     ◉     ╱  ╚╗  ░█▒"],
        @[@"ray",     @"  ▓█░ ║    ╲   ◉│◉   ╱    ║ ░█▓"],
        @[@"ray",     @"  █░  ║     ╲  ─•─  ╱     ║  ░█"],
        @[@"ray",     @"  ▓█░ ║      ╳  •  ╳      ║ ░█▓"],
        @[@"ray",     @"  █░  ║     ╱  ─•─  ╲     ║  ░█"],
        @[@"ray",     @"  ▓█░ ║    ╱   ◉│◉   ╲    ║ ░█▓"],
        @[@"ray",     @"  ▒█░  ╚╗  ╱     ◉     ╲  ╔╝  ░█▒"],
        @[@"letters", @"   ░▓██╚═╗   P · A · T   ╔═╝██▓░"],
        @[@"frame",   @"     ░▒▓██╚═══════════╝▓▒░"],
        @[@"frame",   @"        ░▒▓██████████▓▒░"],
    ];
    for (NSArray<NSString*>* row in banner) {
        [self appendBannerLine:row[1] level:row[0]];
    }
    [self appendLine:@"" level:@"info"];
    [self appendLine:@"DeMonT Engine · v0.1.0  ·  non-rasterized · path-traced"
              level:@"out"];
    self.bannerEndLocation = self.outputView.textStorage.length;
    [self appendLine:@"console attached. type \"list_commands\" or hit Tab."
              level:@"info"];

    return self;
}

// Re-render the logo banner only (preserves any later log lines below
// it). Called on theme change.
- (void)rebuildBanner {
    NSArray<NSArray<NSString*>*>* banner = @[
        @[@"frame",   @"        ░▒▓██████████▓▒░"],
        @[@"frame",   @"     ░▒▓██╔═══════════╗▓▒░"],
        @[@"letters", @"   ░▓██╔═╝   D · M · T   ╚═╗██▓░"],
        @[@"ray",     @"  ▒█░  ╔╝  ╲     ◉     ╱  ╚╗  ░█▒"],
        @[@"ray",     @"  ▓█░ ║    ╲   ◉│◉   ╱    ║ ░█▓"],
        @[@"ray",     @"  █░  ║     ╲  ─•─  ╱     ║  ░█"],
        @[@"ray",     @"  ▓█░ ║      ╳  •  ╳      ║ ░█▓"],
        @[@"ray",     @"  █░  ║     ╱  ─•─  ╲     ║  ░█"],
        @[@"ray",     @"  ▓█░ ║    ╱   ◉│◉   ╲    ║ ░█▓"],
        @[@"ray",     @"  ▒█░  ╚╗  ╱     ◉     ╲  ╔╝  ░█▒"],
        @[@"letters", @"   ░▓██╚═╗   P · A · T   ╔═╝██▓░"],
        @[@"frame",   @"     ░▒▓██╚═══════════╝▓▒░"],
        @[@"frame",   @"        ░▒▓██████████▓▒░"],
    ];
    NSTextStorage* ts = self.outputView.textStorage;
    [ts beginEditing];
    if (self.bannerEndLocation > 0 && self.bannerEndLocation <= ts.length) {
        [ts deleteCharactersInRange:NSMakeRange(0, self.bannerEndLocation)];
    }
    [ts endEditing];
    // Move insertion to the start so newly appended banner sits above
    // pre-existing later text. We'll do this by inserting at index 0.
    NSAttributedString* (^lineAttr)(NSString*, NSColor*) = ^NSAttributedString*(NSString* s, NSColor* c) {
        NSDictionary* attrs = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName: c,
        };
        return [[NSAttributedString alloc] initWithString:[s stringByAppendingString:@"\n"] attributes:attrs];
    };
    PtThemePalette p = self.palette;
    NSUInteger insertAt = 0;
    [ts beginEditing];
    for (NSArray<NSString*>* row in banner) {
        NSString* role = row[0];
        NSColor* c = p.logoFrame;
        if ([role isEqualToString:@"letters"]) c = p.logoLetters;
        else if ([role isEqualToString:@"ray"]) c = p.logoRay;
        NSAttributedString* a = lineAttr(row[1], c);
        [ts insertAttributedString:a atIndex:insertAt];
        insertAt += a.length;
    }
    NSAttributedString* blank = lineAttr(@"", p.info);
    [ts insertAttributedString:blank atIndex:insertAt];
    insertAt += blank.length;
    NSAttributedString* tag = lineAttr(@"DeMonT Engine · v0.1.0  ·  non-rasterized · path-traced", p.out);
    [ts insertAttributedString:tag atIndex:insertAt];
    insertAt += tag.length;
    [ts endEditing];
    self.bannerEndLocation = insertAt;
}

- (void)applyTheme:(NSString*)name {
    self.palette = PtPaletteForTheme(name);
    PtThemePalette p = self.palette;
    self.backdrop.layer.borderColor = p.border.CGColor;
    self.promptLabel.textColor      = p.accent;
    self.statusLabel.textColor      = p.status;
    self.inputField.textColor       = (p.out != nil) ? p.out : [NSColor whiteColor];
    self.logoView.frameColor        = p.accent;
    self.logoView.rayColor          = p.logoRay;
    [self.logoView setNeedsDisplay:YES];
    // Popup bg is palette-derived (subtle accent tint over a dark
    // base); refresh the LAYER backing colour so a theme switch
    // recolours the popup immediately, even before the next drawRect
    // tick fills the new colour on top.
    self.popupView.layer.backgroundColor =
        [[self popupBackgroundColor] CGColor];
    [self.popupView setNeedsDisplay:YES];
    [self rebuildBanner];
}

- (NSColor*)popupBackgroundColor {
    NSColor* base = PtRGB(0.018, 0.024, 0.042);
    NSColor* accent = self.palette.accent;
    if (accent == nil) return base;
    // 6% accent over 94% dark base -- enough to make the popup feel
    // like part of the active theme without losing the "darker than
    // the panel" cue that separates it visually from the scrollback.
    // blendedColorWithFraction returns alpha=1.0 so the result is
    // safe to use as a fully-opaque layer backing.
    return [base blendedColorWithFraction:0.06 ofColor:accent];
}

// Special-case bannered ASCII line -- colour per role pulled from the
// active theme palette so the banner recolours on theme switch.
- (void)appendBannerLine:(NSString*)line level:(NSString*)role {
    PtThemePalette p = self.palette;
    NSColor* col = p.logoFrame;
    if      ([role isEqualToString:@"letters"]) col = p.logoLetters;
    else if ([role isEqualToString:@"ray"])     col = p.logoRay;
    NSShadow* shadow = [[NSShadow alloc] init];
    shadow.shadowColor = [NSColor colorWithCalibratedWhite:0.0 alpha:0.85];
    shadow.shadowOffset = NSMakeSize(0, -1);
    shadow.shadowBlurRadius = 3.0;
    NSDictionary* attrs = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName: col,
        NSShadowAttributeName: shadow,
    };
    NSString* withNewline = [line stringByAppendingString:@"\n"];
    NSAttributedString* a = [[NSAttributedString alloc]
        initWithString:withNewline attributes:attrs];
    [self.outputView.textStorage appendAttributedString:a];
}

- (NSColor*)colorForLevel:(NSString*)level {
    PtThemePalette p = self.palette;
    if ([level isEqualToString:@"warn"])  return p.warn;
    if ([level isEqualToString:@"error"]) return p.error;
    if ([level isEqualToString:@"input"]) return p.input;
    if ([level isEqualToString:@"out"])   return p.out;
    return p.info;
}

- (void)appendLine:(NSString*)line level:(NSString*)level {
    NSString* prefix = [level isEqualToString:@"input"] ? @"> " : @"";
    NSShadow* shadow = [[NSShadow alloc] init];
    shadow.shadowColor = [NSColor colorWithCalibratedWhite:0.0 alpha:0.85];
    shadow.shadowOffset = NSMakeSize(0, -1);
    shadow.shadowBlurRadius = 2.5;
    NSDictionary* attrs = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName: [self colorForLevel:level],
        NSShadowAttributeName: shadow,
    };
    NSString* withNewline = [NSString stringWithFormat:@"%@%@\n", prefix, line];
    NSAttributedString* a = [[NSAttributedString alloc]
        initWithString:withNewline attributes:attrs];
    [self.outputView.textStorage appendAttributedString:a];
    [self.outputView scrollRangeToVisible:NSMakeRange(self.outputView.textStorage.length, 0)];
}

- (void)submitLine:(NSString*)line {
    if (line.length == 0) return;
    [self.history addObject:line];
    if (self.history.count > 200) [self.history removeObjectAtIndex:0];
    self.historyPos = self.history.count;

    // Mirror into Console::history_ so the cross-platform persistence
    // path (Engine::SaveConsoleHistoryToDisk -> console_history.txt)
    // sees every submitted line. PushHistory dedups against the
    // immediate predecessor and trims to kMaxHistoryDepth -- safe to
    // call unconditionally on every submit.
    pt::console::Console::Get().PushHistory(std::string([line UTF8String]));

    [self appendLine:line level:@"input"];

    // ExecuteScript so pasted multi-statement lines (separated by ';'
    // or '\n') run as a sequence -- e.g. "r_clouds 1; r_volumetric 1".
    auto result = pt::console::Console::Get().ExecuteScript(
        std::string_view([line UTF8String]));
    if (!result.output.empty()) {
        NSString* o = [NSString stringWithUTF8String:result.output.c_str()];
        while (o.length > 0 && [o characterAtIndex:o.length - 1] == '\n') {
            o = [o substringToIndex:o.length - 1];
        }
        [self appendLine:o level:@"out"];
    }
    if (!result.ok) {
        NSString* e = [NSString stringWithUTF8String:result.error.c_str()];
        [self appendLine:[NSString stringWithFormat:@"error: %@", e] level:@"error"];
    }
}

- (void)submitInput {
    NSString* line = [self.inputField.stringValue copy];
    if (line.length == 0) return;
    self.inputField.stringValue = @"";
    // Programmatic setStringValue: doesn't fire controlTextDidChange:,
    // so an active popup (e.g. one auto-opened by the commitPopup path
    // that calls into submitInput) would otherwise linger with stale
    // items pointing at a now-empty field. Tear it down explicitly so
    // every submit leaves a clean prompt.
    if (_popupActive) [self dismissPopup];
    [self submitLine:line];
}

// Paste-to-multiline: when the user pastes text containing newlines, run
// each complete line as its own command and leave the trailing partial
// (anything after the last newline) in the input. Fires from the field
// editor whenever stringValue changes from a user action -- programmatic
// setStringValue: does not re-trigger this, so the writeback below is safe.
- (void)controlTextDidChange:(NSNotification*)note {
    if (note.object != self.inputField) return;

    NSString* value = self.inputField.stringValue;

    // Backtick is the show/hide toggle; stripping it here catches the
    // case where the user presses ` to open AND starts typing fast
    // -- the original ` event arrives at the field editor before the
    // panel becomes the key window, so the NSEvent monitor that's
    // supposed to swallow it doesn't get a chance.
    if ([value rangeOfString:@"`"].location != NSNotFound) {
        value = [value stringByReplacingOccurrencesOfString:@"`" withString:@""];
        self.inputField.stringValue = value;
    }

    if ([value rangeOfString:@"\n"].location != NSNotFound) {
        NSArray<NSString*>* parts = [value componentsSeparatedByString:@"\n"];
        NSString* trailing = parts.lastObject ? parts.lastObject : @"";
        self.inputField.stringValue = trailing;
        for (NSUInteger i = 0; i + 1 < parts.count; ++i) {
            [self submitLine:parts[i]];
        }
    }

    // Re-rank completions on every text mutation. The shared engine's
    // gating decides whether the popup auto-opens (token 0 with empty
    // text -> dismiss; value position with empty text -> open with
    // allowed_values list). Mirrors the web console's `input` event
    // and the Win32 WM_CHAR handler.
    [self refreshCompletions:NO];
}

// NSControlTextEditingDelegate dispatch. The field's `delegate` is set
// to this PtConsoleView, so AppKit calls the method on us (not on the
// field subclass). When the popup is active, Up/Down/Tab/Enter/Esc/End
// drive the popup state machine; otherwise these keys revert to
// history-walk / submit / clear semantics. Cursor-move keys (Left /
// Right / Home / End) re-evaluate completions against the new caret
// position so e.g. arrowing into the trailing space of `r_denoiser `
// pops the value-position list -- mirrors the Win32 implementation.
- (BOOL)control:(NSControl*)control textView:(NSTextView*)textView
      doCommandBySelector:(SEL)cmd {
    (void)control;

    // Popup-active keys. Mirrors the WM_KEYDOWN block in
    // ConsoleOverlay_Win32.cpp:
    //   Up/Down  -- move highlight (do NOT walk history)
    //   Tab      -- commit + chain (open value-position popup if
    //               we just landed on a token-0 cvar/command name)
    //   Enter    -- commit WITHOUT chain, then submit the line
    //   Esc      -- dismiss
    //   End / Right-at-EOL -- commit WITHOUT chain
    //   any other command -- dismiss + fall through to default
    if (_popupActive) {
        if (cmd == @selector(moveUp:))     { [self movePopupSelection:-1]; return YES; }
        if (cmd == @selector(moveDown:))   { [self movePopupSelection:+1]; return YES; }
        if (cmd == @selector(insertTab:))     { [self commitPopup:YES]; return YES; }
        if (cmd == @selector(insertBacktab:)) { [self commitPopup:YES]; return YES; }
        if (cmd == @selector(cancelOperation:)) {
            [self dismissPopup];
            return YES;
        }
        if (cmd == @selector(insertNewline:) ||
            cmd == @selector(insertNewlineIgnoringFieldEditor:)) {
            [self commitPopup:NO];
            [self submitInput];
            return YES;
        }
        if (cmd == @selector(moveToEndOfLine:) ||
            cmd == @selector(moveToEndOfDocument:) ||
            cmd == @selector(moveToEndOfParagraph:)) {
            [self commitPopup:NO];
            return YES;
        }
        if (cmd == @selector(moveRight:)) {
            NSText* ed = [self.inputField currentEditor];
            const NSUInteger end = self.inputField.stringValue.length;
            if (ed && ed.selectedRange.location >= end) {
                [self commitPopup:NO];
                return YES;
            }
            // Right-arrow before EOL: dismiss + let the cursor move
            // and the post-move re-eval (below) re-rank against the
            // new position.
            [self dismissPopup];
            // fall through to cursor-move handling below
        } else {
            // Any other command: tear the popup down and let the
            // keystroke route through default handling. The post-
            // mutation refresh in controlTextDidChange: (or the
            // cursor-move re-eval below) will reopen the popup if
            // the new state still has matches.
            [self dismissPopup];
            // fall through
        }
    }

    if (cmd == @selector(insertTab:)) {
        // Tab with popup hidden = force-show. Mirrors the web
        // console's Tab handler and Win32's VK_TAB branch.
        [self refreshCompletions:YES];
        return YES;
    }
    if (cmd == @selector(insertBacktab:)) {
        // Shift+Tab without popup: same force-show semantics.
        [self refreshCompletions:YES];
        return YES;
    }
    if (cmd == @selector(insertNewline:)) { [self submitInput]; return YES; }
    if (cmd == @selector(moveUp:)) {
        if (self.historyPos > 0) {
            self.historyPos -= 1;
            self.inputField.stringValue = self.history[self.historyPos];
            // Move caret to end of restored line so a follow-up Up/Down
            // step picks up from the right insertion point.
            NSText* ed = [self.inputField currentEditor];
            if (ed) [ed setSelectedRange:NSMakeRange(self.inputField.stringValue.length, 0)];
            // DELIBERATELY do NOT auto-open the popup on a history
            // step. Walking past commands flashes a popup at each step
            // otherwise, which is noise -- the user is reviewing what
            // they ran, not asking for completions. Ctrl+Space is the
            // explicit summon if they want to complete from the
            // restored line. Programmatic setStringValue: above does
            // not fire controlTextDidChange:, so no implicit refresh.
        }
        return YES;
    }
    if (cmd == @selector(moveDown:)) {
        if (self.historyPos < (NSInteger)self.history.count) {
            self.historyPos += 1;
            if (self.historyPos == (NSInteger)self.history.count) {
                self.inputField.stringValue = @"";
            } else {
                self.inputField.stringValue = self.history[self.historyPos];
            }
            NSText* ed = [self.inputField currentEditor];
            if (ed) [ed setSelectedRange:NSMakeRange(self.inputField.stringValue.length, 0)];
            // No auto-popup -- see -moveUp: above. Ctrl+Space summons.
        }
        return YES;
    }
    if (cmd == @selector(cancelOperation:)) {
        // Esc with empty field hides the panel; otherwise clears the field.
        if (self.inputField.stringValue.length == 0) {
            id win = self.window;
            if ([win respondsToSelector:@selector(hide)]) {
                [win performSelector:@selector(hide)];
            }
        } else {
            self.inputField.stringValue = @"";
            [self dismissPopup];
        }
        return YES;
    }

    // Cursor-move keys re-evaluate the popup against the new caret
    // position (fires AFTER the cursor has actually moved -- we run
    // the refresh on the next main-loop tick). Without this, navigating
    // into the trailing space of `r_denoiser ` via Home + End leaves
    // the popup stale -- the user would have to type a char to get the
    // value-position popup to open. Matches the Win32 VK_LEFT / VK_RIGHT
    // / VK_HOME / VK_END branches.
    if (cmd == @selector(moveLeft:) ||
        cmd == @selector(moveRight:) ||
        cmd == @selector(moveToBeginningOfLine:) ||
        cmd == @selector(moveToEndOfLine:) ||
        cmd == @selector(moveToBeginningOfDocument:) ||
        cmd == @selector(moveToEndOfDocument:) ||
        cmd == @selector(moveToBeginningOfParagraph:) ||
        cmd == @selector(moveToEndOfParagraph:) ||
        cmd == @selector(moveWordLeft:) ||
        cmd == @selector(moveWordRight:)) {
        __weak PtConsoleView* weakSelf = self;
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf refreshCompletions:NO];
        });
        return NO;  // let AppKit perform the actual cursor move
    }
    return NO;
}

// ---------------------------------------------------------------------------
// Popup state machine. See WinOverlay::RefreshCompletions et al. in
// ConsoleOverlay_Win32.cpp for the reference implementation -- this is
// a 1-for-1 port with AppKit equivalents for byte-vs-UTF16 cursor math
// and NSAttributedString rendering.

// Convert the current input + UTF-16 cursor into UTF-8 bytes + byte
// offset, the form pt::console::CurrentToken expects. Cvar / command
// names are pure ASCII so the conversion is a no-op for the typical
// case, but a user pasting non-ASCII into the input would otherwise
// splice at the wrong byte offset on commit.
- (std::string)utf8InputAndCursor:(std::size_t*)outCursorBytes {
    NSString* input = self.inputField.stringValue;
    if (input == nil) input = @"";
    const char* utf8 = [input UTF8String];
    std::string s = (utf8 != nullptr) ? std::string(utf8) : std::string();
    NSText* ed = [self.inputField currentEditor];
    NSUInteger cursorUTF16 = (ed != nil) ? ed.selectedRange.location
                                         : input.length;
    if (cursorUTF16 > input.length) cursorUTF16 = input.length;
    NSString* prefix = [input substringToIndex:cursorUTF16];
    NSUInteger cursorBytes = [prefix lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    if (outCursorBytes) *outCursorBytes = cursorBytes;
    return s;
}

- (void)refreshCompletions:(BOOL)forceShow {
    std::size_t cursorBytes = 0;
    std::string input = [self utf8InputAndCursor:&cursorBytes];
    pt::console::TokenInfo token = pt::console::CurrentToken(input, cursorBytes);

    // Empty-query gating mirrors Win32 / web:
    //   - Token 0 with empty text + !forceShow: don't auto-open. The
    //     popup would otherwise list every cvar + command in the engine
    //     every time the input becomes empty (fresh open, post-Submit
    //     clear) -- noise.
    //   - Value position (is_token0 == false) with empty text: DO open.
    //     The user just typed `<cvar> ` and wants to see the value list
    //     including the current value.
    if (!forceShow && token.text.empty() && token.is_token0) {
        [self dismissPopup];
        return;
    }

    auto items = pt::console::BuildCompletions(token,
                                                /*max_results=*/60,
                                                /*description_clip=*/120);
    if (items.empty()) {
        [self dismissPopup];
        return;
    }

    _popupActive       = YES;
    _popupItems        = std::move(items);
    _popupSelected     = 0;
    _popupToken        = std::move(token);
    _popupScrollOffset = 0;
    [self layoutPopup];
    [self renderInlineGhost];
    [self.popupView setNeedsDisplay:YES];
}

- (NSInteger)popupVisibleRows {
    constexpr NSInteger kPopupMaxRows = 10;
    const NSInteger n = (NSInteger)_popupItems.size();
    if (n == 0) return 0;
    NSFont* font = self.inputField.font;
    if (font == nil) font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    const CGFloat lineH = ceil(font.pointSize * 1.35);
    if (lineH <= 0) return 0;

    // Available vertical room between the input row top and the log
    // area's bottom edge. Input field sits at y=8, height=22 -- top
    // edge at y=30 in Mac (bottom-left) coords. The output scroll
    // starts at y=36, so the popup sits in the strip between y=36 and
    // the panel top, anchored at the bottom (above the input). Leave
    // 4 px of breathing room from the scroll's top edge so the popup
    // doesn't visually fight the log header.
    const NSRect bounds = self.bounds;
    const CGFloat panelH    = bounds.size.height;
    const CGFloat popupYBot = 36;            // bottom of popup in Mac coords
    const CGFloat popupYTop = panelH - 36;   // top of available zone
    const CGFloat available = popupYTop - popupYBot - 4;
    if (available < lineH + 8) return 0;
    NSInteger byHeight = (NSInteger)floor((available - 8) / lineH);
    if (byHeight < 1) byHeight = 1;
    NSInteger v = n;
    if (v > kPopupMaxRows) v = kPopupMaxRows;
    if (v > byHeight)      v = byHeight;
    return v;
}

- (void)movePopupSelection:(NSInteger)dir {
    if (!_popupActive || _popupItems.empty()) return;
    const NSInteger n = (NSInteger)_popupItems.size();
    NSInteger sel = ((_popupSelected + dir) % n + n) % n;
    _popupSelected = sel;
    // Keep the highlighted row in the visible window. visible is the
    // actual paintable row count for the current panel size (shared
    // with drawPopupInRect: via -popupVisibleRows), so on a short
    // panel Up/Down scrolls the window through the rest instead of
    // letting selection run off-screen.
    const NSInteger visible = [self popupVisibleRows];
    if (visible > 0) {
        if (sel < _popupScrollOffset) {
            _popupScrollOffset = sel;
        } else if (sel >= _popupScrollOffset + visible) {
            _popupScrollOffset = sel - visible + 1;
        }
    }
    [self renderInlineGhost];
    [self.popupView setNeedsDisplay:YES];
}

// Commit the highlighted candidate by replacing the word at the cursor
// with the candidate name (+ trailing space when we're completing token
// 0 so the user lands at the value position ready to keep typing).
// chainNext == YES auto-reopens the popup at the new cursor (Tab path);
// NO leaves the popup closed (Enter / End / Right-at-EOL paths -- those
// are "accept + stop" gestures).
- (void)commitPopup:(BOOL)chainNext {
    if (!_popupActive ||
        _popupSelected < 0 ||
        _popupSelected >= (NSInteger)_popupItems.size()) {
        [self dismissPopup];
        return;
    }
    const pt::console::CompletionMatch& it = _popupItems[(std::size_t)_popupSelected];
    const pt::console::TokenInfo&        t = _popupToken;

    std::size_t /*cursorBytes*/ _ = 0;
    std::string s = [self utf8InputAndCursor:&_];
    if (t.end > s.size()) {
        // Token boundaries shifted out from under us (the user typed
        // something between refreshCompletions and commit). Bail.
        [self dismissPopup];
        return;
    }
    std::string tail = it.name + (t.is_token0 ? std::string(" ") : std::string());
    std::string newInput = s.substr(0, t.start) + tail + s.substr(t.end);
    const std::size_t newCursorBytes = t.start + tail.size();
    const bool wasToken0 = t.is_token0;

    // Push the new value into the field. setStringValue: doesn't fire
    // controlTextDidChange:, so we explicitly tear the popup down and
    // (optionally) refresh below -- no double-open surprise.
    NSString* ns = [NSString stringWithUTF8String:newInput.c_str()];
    if (ns == nil) ns = @"";
    self.inputField.stringValue = ns;

    // Convert byte cursor back to UTF-16 for AppKit's selectedRange.
    std::string prefixBytes = newInput.substr(0, newCursorBytes);
    NSString* prefixStr = [NSString stringWithUTF8String:prefixBytes.c_str()];
    if (prefixStr == nil) prefixStr = @"";
    NSText* ed = [self.inputField currentEditor];
    if (ed != nil) {
        [ed setSelectedRange:NSMakeRange(prefixStr.length, 0)];
    }

    [self dismissPopup];
    if (chainNext && wasToken0) {
        [self refreshCompletions:YES];
    }
}

- (void)dismissPopup {
    _popupActive       = NO;
    _popupItems.clear();
    _popupSelected     = -1;
    _popupToken        = pt::console::TokenInfo{};
    _popupScrollOffset = 0;
    self.popupView.hidden = YES;
    self.ghostLabel.attributedStringValue =
        [[NSAttributedString alloc] initWithString:@""];
}

// Place the popup view above the input row, sized to the visible-row
// budget for the current panel height. Mac coords are bottom-left, so
// frame origin is at the popup's BOTTOM edge and frame.height grows
// upward. Hidden when not enough room for even one row -- matches
// Win32's PopupVisibleRows() == 0 short-circuit.
- (void)layoutPopup {
    if (!_popupActive || _popupItems.empty()) {
        self.popupView.hidden = YES;
        return;
    }
    const NSInteger visibleN = [self popupVisibleRows];
    if (visibleN <= 0) {
        self.popupView.hidden = YES;
        return;
    }
    NSFont* font = self.inputField.font;
    if (font == nil) font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    const CGFloat lineH = ceil(font.pointSize * 1.35);
    const CGFloat popupH = visibleN * lineH + 8;
    const NSRect bounds = self.bounds;
    NSRect target = NSMakeRect(14, 36,
                               bounds.size.width - 28,
                               popupH);
    self.popupView.frame = target;
    self.popupView.hidden = NO;
    [self.popupView setNeedsDisplay:YES];
}

// Ghost preview just past the cursor: dim tail of the highlighted
// candidate when it's a case-insensitive prefix of the token-at-cursor
// AND the cursor is at end of input. Tinted YELLOW (RGB 255,200,80) on
// current-value rows so cycling Up/Down through allowed_values keeps a
// visible "this is what's set right now" cue inline as well as in the
// popup row -- matches web's `.ghost-tail.is-current` and Win32's
// kCurrentHintColor.
- (void)renderInlineGhost {
    NSAttributedString* empty = [[NSAttributedString alloc] initWithString:@""];
    if (!_popupActive ||
        _popupSelected < 0 ||
        _popupSelected >= (NSInteger)_popupItems.size()) {
        self.ghostLabel.attributedStringValue = empty;
        return;
    }

    NSText* ed = [self.inputField currentEditor];
    NSString* typed = self.inputField.stringValue;
    NSUInteger cursorUTF16 = (ed != nil) ? ed.selectedRange.location
                                         : typed.length;
    if (cursorUTF16 != typed.length) {
        // Only show inline preview at end-of-input; otherwise the dim
        // tail would visually extend past mid-line text and confuse.
        self.ghostLabel.attributedStringValue = empty;
        return;
    }

    const pt::console::CompletionMatch& it = _popupItems[(std::size_t)_popupSelected];
    const pt::console::TokenInfo&        t = _popupToken;
    if (it.name.size() <= t.text.size()) {
        self.ghostLabel.attributedStringValue = empty;
        return;
    }
    auto lc = [](char c) -> char {
        return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    };
    bool prefix = true;
    for (std::size_t i = 0; i < t.text.size(); ++i) {
        if (lc(t.text[i]) != lc(it.name[i])) { prefix = false; break; }
    }
    if (!prefix) {
        self.ghostLabel.attributedStringValue = empty;
        return;
    }
    std::string tailUtf8 = it.name.substr(t.text.size());
    NSString* tail = [NSString stringWithUTF8String:tailUtf8.c_str()];
    if (tail == nil || tail.length == 0) {
        self.ghostLabel.attributedStringValue = empty;
        return;
    }

    NSFont* defaultFont = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    NSFont* font = self.inputField.font ? self.inputField.font : defaultFont;
    NSColor* dim = self.palette.info ? self.palette.info : [NSColor grayColor];
    NSColor* warn = PtRGB(1.0, 200.0/255.0, 80.0/255.0);
    const bool isCurrent = (it.value == "current");

    NSMutableAttributedString* s = [[NSMutableAttributedString alloc] init];
    // Transparent run mirrors the typed text width so the visible tail
    // lands just past the cursor without per-pixel measurement.
    [s appendAttributedString:[[NSAttributedString alloc] initWithString:typed
        attributes:@{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: [NSColor clearColor],
        }]];
    [s appendAttributedString:[[NSAttributedString alloc] initWithString:tail
        attributes:@{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: (isCurrent ? warn : dim),
        }]];
    self.ghostLabel.attributedStringValue = s;
}

// Render the popup. Called by PtConsolePopupView's drawRect:, with the
// view's coordinate system flipped (top-down) so y math reads the same
// as the Win32 reference. Background + outline + per-row content
// (selection bg, current/default left bar, name with match-spans,
// value chip, truncated description). The bg is a dark base + small
// accent blend (-popupBackgroundColor) so each theme tints the popup
// distinctly; everything else (outline, name, chips, value, desc) is
// pulled from the active palette so theme switches recolour
// automatically.
- (void)drawPopupInRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    if (!_popupActive || _popupItems.empty()) return;
    const NSInteger visibleN = [self popupVisibleRows];
    if (visibleN <= 0) return;

    NSFont* font = self.inputField.font;
    if (font == nil) font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    const CGFloat lineH = ceil(font.pointSize * 1.35);

    PtThemePalette p = self.palette;
    NSColor* accent  = p.accent  ? p.accent  : PtRGB(0.0, 0.94, 1.0);
    NSColor* dimCol  = p.info    ? p.info    : [NSColor grayColor];
    NSColor* textCol = p.out     ? p.out     : [NSColor whiteColor];
    NSColor* warn    = PtRGB(1.0, 200.0/255.0, 80.0/255.0);

    const NSRect bounds = self.popupView.bounds;
    // Background -- FULLY opaque so the scrollback rendered behind the
    // popup zone (NSTextView in the NSScrollView underneath) doesn't
    // bleed through. Semi-transparency here is tempting because the
    // panel itself uses NSVisualEffectView vibrancy, but the same
    // vibrancy already shows through the popup's location to the
    // scrollback's text -- a translucent popup on top would render
    // popup rows over a chaos of overlapping log text. Layer.backing
    // is set to the same colour in initWithFrame so the OPAQUE base
    // is there even before drawRect runs. Theme-derived via
    // -popupBackgroundColor so amber/matrix/etc each tint the popup
    // with a hint of their accent.
    NSColor* popupBg = [self popupBackgroundColor];
    [popupBg setFill];
    NSRectFill(bounds);
    // Pre-compute the SOLID selection / left-bar colours by blending
    // the popup bg with the accent. blendedColorWithFraction returns a
    // fully-opaque colour, so the row highlight composites cleanly
    // over the bg with no chance of revealing the scrollback layer
    // underneath -- the prior `accent colorWithAlphaComponent:0.22`
    // approach still pulled the (previously-translucent) layer's
    // compositing alpha through to the user. Three tiers:
    //   selRowBg    -- selected row background (subtle accent wash)
    //   curBarSolid -- current-value left bar (full accent)
    //   defBarSolid -- default-value left bar (mid-blend)
    NSColor* selRowBg    = [popupBg blendedColorWithFraction:0.22 ofColor:accent];
    NSColor* curBarSolid = accent;
    NSColor* defBarSolid = [popupBg blendedColorWithFraction:0.55 ofColor:accent];
    // 1 px accent outline (solid blend; same opacity rationale).
    NSColor* outlineColor = [popupBg blendedColorWithFraction:0.55 ofColor:accent];
    [outlineColor setStroke];
    NSBezierPath* outline = [NSBezierPath bezierPathWithRect:
        NSInsetRect(bounds, 0.5, 0.5)];
    outline.lineWidth = 1.0;
    [outline stroke];

    // Clip subsequent draws to the popup interior so a long
    // description tail (or an unexpectedly wide kind chip) doesn't
    // bleed past the popup's right edge into whatever's behind.
    [NSGraphicsContext saveGraphicsState];
    NSBezierPath* clip = [NSBezierPath bezierPathWithRect:
        NSInsetRect(bounds, 1.0, 1.0)];
    [clip addClip];

    const NSInteger n = (NSInteger)_popupItems.size();
    const NSInteger scroll = _popupScrollOffset;
    const CGFloat   xStart = 8.0;
    CGFloat         rowY   = 4.0;
    for (NSInteger vi = 0; vi < visibleN; ++vi) {
        const NSInteger idx = scroll + vi;
        if (idx < 0 || idx >= n) break;
        const pt::console::CompletionMatch& it = _popupItems[(std::size_t)idx];
        const bool selected   = (idx == _popupSelected);
        const bool isCurrent  = (it.value == "current");
        const bool isDefault  = (it.value == "default");

        // Selected row -- SOLID blended bg (no alpha) so the row never
        // shows scrollback through the highlight. See selRowBg comment
        // above the loop.
        if (selected) {
            NSRect rr = NSMakeRect(2, rowY,
                                   bounds.size.width - 4, lineH);
            [selRowBg setFill];
            NSRectFill(rr);
        }
        // Left-edge bar marks the current value (and a softer mid-
        // blend for the default), independent of selection -- "what's
        // set right now" stays visible while cycling through the
        // value list. Both bar colours are solid blends, no alpha.
        if (isCurrent || isDefault) {
            NSRect bar = NSMakeRect(2, rowY, 2, lineH);
            [(isCurrent ? curBarSolid : defBarSolid) setFill];
            NSRectFill(bar);
        }

        // Walk the match-spans, emitting in-span runs in accent and
        // out-of-span runs in textCol. Current-value rows tint the
        // WHOLE name in accent so the eye lands on it first.
        const std::string& name = it.name;
        NSColor* nameFg = isCurrent ? accent : textCol;
        CGFloat tx = xStart;

        auto drawRun = [&](std::size_t a, std::size_t b, NSColor* col) {
            if (a >= b || a >= name.size()) return;
            const std::size_t end = std::min(b, name.size());
            std::string sub = name.substr(a, end - a);
            NSString* ns = [NSString stringWithUTF8String:sub.c_str()];
            if (ns == nil || ns.length == 0) return;
            NSDictionary* attrs = @{
                NSFontAttributeName:            font,
                NSForegroundColorAttributeName: col,
            };
            // Mac AppKit's drawAtPoint draws with the baseline at point.y
            // when isFlipped is YES (top-down), so the y we pass is the
            // top of the row -- AppKit handles baseline placement.
            [ns drawAtPoint:NSMakePoint(tx, rowY) withAttributes:attrs];
            NSSize sz = [ns sizeWithAttributes:attrs];
            tx += sz.width;
        };

        std::size_t cursorCh = 0;
        for (const auto& sp : it.spans) {
            if (sp.first > cursorCh) {
                drawRun(cursorCh, sp.first, nameFg);
            }
            drawRun(sp.first, sp.second, accent);
            cursorCh = sp.second;
        }
        if (cursorCh < name.size()) {
            drawRun(cursorCh, name.size(), nameFg);
        }

        // Kind chip (cvar / cmd) for token-0 rows. Dim small caps.
        if (it.kind == pt::console::CompletionKind::Cvar ||
            it.kind == pt::console::CompletionKind::Command) {
            NSString* chip =
                (it.kind == pt::console::CompletionKind::Cvar) ? @"cvar" : @"cmd";
            NSDictionary* chipAttrs = @{
                NSFontAttributeName: [NSFont monospacedSystemFontOfSize:font.pointSize - 2
                                                                weight:NSFontWeightRegular],
                NSForegroundColorAttributeName: [dimCol colorWithAlphaComponent:0.85],
            };
            tx += 8;
            [chip drawAtPoint:NSMakePoint(tx, rowY + 1) withAttributes:chipAttrs];
            NSSize sz = [chip sizeWithAttributes:chipAttrs];
            tx += sz.width;
        }

        tx += 12;

        // Value chip ("current" / "default" / current-value string).
        // Accent for current, dim-accent for default, plain dim for
        // anything else (e.g. the cvar's value shown on a token-0 row).
        if (!it.value.empty()) {
            NSString* val = [NSString stringWithUTF8String:it.value.c_str()];
            if (val != nil && val.length > 0) {
                NSColor* vc = isCurrent
                    ? accent
                    : (isDefault ? [accent colorWithAlphaComponent:0.65]
                                 : dimCol);
                if (isCurrent) vc = warn;  // warm tint matches inline ghost
                NSDictionary* vAttrs = @{
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName: vc,
                };
                [val drawAtPoint:NSMakePoint(tx, rowY) withAttributes:vAttrs];
                NSSize sz = [val sizeWithAttributes:vAttrs];
                tx += sz.width + 12;
            }
        }

        // Truncated description, painted in dim. The clip region above
        // hides any overflow pixel-clean; we still cap by char count
        // so the visible part ends in an ellipsis instead of a sharp
        // right-edge cut. ~7 px per char in the default monospace font
        // at 13 pt -- scale roughly with point size.
        if (!it.description.empty()) {
            const CGFloat budgetPx = bounds.size.width - tx - 8;
            if (budgetPx > 24) {
                const CGFloat pxPerCh = MAX((CGFloat)4.0, font.pointSize * 0.55);
                NSInteger maxCh = (NSInteger)floor(budgetPx / pxPerCh);
                std::string desc = it.description;
                if ((NSInteger)desc.size() > maxCh && maxCh > 3) {
                    // UTF-8 safe truncation -- a raw resize can split
                    // a multibyte codepoint, and
                    // [NSString stringWithUTF8String:] returns nil on
                    // invalid UTF-8, dropping the description draw.
                    // Cvar descriptions contain non-ASCII (e.g. "≈"
                    // in Engine.cpp).
                    pt::console::Utf8SafeTruncate(desc, (std::size_t)(maxCh - 1));
                    desc.append("\xE2\x80\xA6");  // UTF-8 ellipsis
                }
                NSString* d = [NSString stringWithUTF8String:desc.c_str()];
                if (d != nil && d.length > 0) {
                    NSDictionary* dAttrs = @{
                        NSFontAttributeName:            font,
                        NSForegroundColorAttributeName: dimCol,
                    };
                    [d drawAtPoint:NSMakePoint(tx, rowY) withAttributes:dAttrs];
                }
            }
        }

        rowY += lineH;
    }

    [NSGraphicsContext restoreGraphicsState];
}

// Forward mouse-wheel events anywhere on the console view to the
// scrollback's NSScrollView, so the user can scroll the log even
// when the cursor is over the input row, banner or padding -- not
// just when it's directly over the output area.  When the cursor IS
// over the scroll view, AppKit hit-tests it directly and this
// override never fires, so no double-scrolling.
- (void)scrollWheel:(NSEvent*)event {
    if (self.outputScroll) {
        [self.outputScroll scrollWheel:event];
    } else {
        [super scrollWheel:event];
    }
}

// Re-flow the popup on every panel resize. The popup-fit math depends
// on bounds.height (visibleN can shrink on a short panel), so a window
// resize that doesn't touch popup state still has to re-evaluate the
// frame -- otherwise a popup opened at 10 rows stays 10 rows tall when
// the user shrinks the panel below that height.
- (void)setFrameSize:(NSSize)newSize {
    [super setFrameSize:newSize];
    if (_popupActive) {
        [self layoutPopup];
    }
}

- (void)applyFontScale {
    auto* v = pt::console::Console::Get().FindCVar("con_font_scale");
    if (v == nullptr) return;
    char* end = nullptr;
    float requested = std::strtof(v->value.c_str(), &end);
    // Reject "no digits parsed" AND non-finite values (NaN / +-Inf):
    // strtof happily returns a quiet NaN for "nan"/"NaN" and +-Inf for
    // "inf"; once non-finite slips into the comparisons below, the < /
    // > clamps return false in BOTH directions (NaN compares unordered),
    // so requested would stay NaN, sneak past the early-exit, and the
    // float * point-size casts in the NSFont calls below would propagate
    // NaN into AppKit -- undefined behaviour. Fall back to 1.0 (the
    // engineering default) before any math. Mirrors the equivalent
    // guard in WinOverlay::EnsureFontScale().
    if (end == v->value.c_str() || !std::isfinite(requested)) requested = 1.0f;
    if (requested < 0.5f) requested = 0.5f;
    if (requested > 3.0f) requested = 3.0f;
    if (std::fabs(requested - _fontScale) < 1e-3f) return;

    _fontScale = requested;
    const CGFloat input_pt  = (CGFloat)13.0 * requested;
    const CGFloat output_pt = (CGFloat)12.0 * requested;
    const CGFloat status_pt = (CGFloat)9.0  * requested;
    self.inputField.font  = [NSFont monospacedSystemFontOfSize:input_pt
                                                        weight:NSFontWeightRegular];
    self.outputView.font  = [NSFont monospacedSystemFontOfSize:output_pt
                                                        weight:NSFontWeightRegular];
    self.promptLabel.font = [NSFont monospacedSystemFontOfSize:input_pt
                                                        weight:NSFontWeightSemibold];
    self.statusLabel.font = [NSFont monospacedSystemFontOfSize:status_pt
                                                        weight:NSFontWeightSemibold];
    self.ghostLabel.font  = [NSFont monospacedSystemFontOfSize:input_pt
                                                        weight:NSFontWeightRegular];
    // Popup row count + line height derive from inputField.font, so a
    // scale change shrinks/grows the popup in lockstep. Existing
    // ghostLabel string still has its own font baked into the
    // attributed runs -- re-render so the dim tail picks up the new
    // size on the next paint.
    if (_popupActive) {
        [self layoutPopup];
        [self renderInlineGhost];
        [self.popupView setNeedsDisplay:YES];
    }
    LOG_INFO("ConsoleOverlay: con_font_scale={:.2f} -> "
             "input={:.1f}pt output={:.1f}pt status={:.1f}pt",
             requested, input_pt, output_pt, status_pt);
}

@end

// ---------------------------------------------------------------------------
@implementation PtConsolePanel

+ (BOOL)canBecomeKeyWindow { return YES; }

- (instancetype)initWithParent:(NSWindow*)parent {
    NSRect frame = parent.frame;
    CGFloat h = MIN(MAX(frame.size.height * 0.45, 240), 560);
    NSRect panelFrame = NSMakeRect(frame.origin.x,
                                    frame.origin.y + frame.size.height - h,
                                    frame.size.width, h);

    self = [super initWithContentRect:panelFrame
                            styleMask:NSWindowStyleMaskBorderless
                              backing:NSBackingStoreBuffered
                                defer:NO];
    if (!self) return nil;

    self.parentRenderWindow = parent;
    self.opaque = NO;
    self.backgroundColor = [NSColor clearColor];
    self.hasShadow = YES;
    self.level = NSFloatingWindowLevel;
    self.movableByWindowBackground = NO;
    self.releasedWhenClosed = NO;
    self.hidesOnDeactivate = NO;

    PtConsoleView* v = [[PtConsoleView alloc] initWithFrame:NSMakeRect(0, 0, panelFrame.size.width, panelFrame.size.height)];
    self.contentView = v;
    self.consoleView = v;

    self.alphaValue = 0.0;
    self.isShown = NO;

    return self;
}

- (BOOL)canBecomeKeyWindow { return YES; }

// Forward standard edit-menu Cmd shortcuts (Cmd+C / V / X / A) to the
// first responder. The console panel is a borderless NSPanel and the
// app process doesn't install an Edit menu, so AppKit's default
// performKeyEquivalent: dispatch never invokes cut:/copy:/paste:/
// selectAll: on the field editor -- without this override Cmd+V is a
// no-op inside the console input. The first responder is the text
// view backing the NSTextField when the panel is key, which is
// exactly the receiver these selectors target. Returning YES from
// tryToPerform tells AppKit we consumed the event so it doesn't
// fall through to a system beep.
- (BOOL)performKeyEquivalent:(NSEvent*)event {
    const NSEventModifierFlags mask =
        NSEventModifierFlagControl | NSEventModifierFlagCommand |
        NSEventModifierFlagOption  | NSEventModifierFlagShift;
    const NSEventModifierFlags mods = event.modifierFlags & mask;
    if (mods == NSEventModifierFlagCommand && event.charactersIgnoringModifiers.length == 1) {
        SEL action = NULL;
        switch ([event.charactersIgnoringModifiers characterAtIndex:0]) {
            case 'c': case 'C': action = @selector(copy:);      break;
            case 'v': case 'V': action = @selector(paste:);     break;
            case 'x': case 'X': action = @selector(cut:);       break;
            case 'a': case 'A': action = @selector(selectAll:); break;
            default: break;
        }
        if (action != NULL && [self.firstResponder tryToPerform:action with:self]) {
            return YES;
        }
    }
    return [super performKeyEquivalent:event];
}

- (void)layoutToParent {
    if (self.parentRenderWindow == nil) return;
    NSRect frame = self.parentRenderWindow.frame;
    CGFloat h = MIN(MAX(frame.size.height * 0.45, 240), 560);
    NSRect target = NSMakeRect(frame.origin.x,
                                frame.origin.y + frame.size.height - h,
                                frame.size.width, h);
    [self setFrame:target display:YES animate:NO];
}

- (void)show {
    if (self.isShown) return;
    self.isShown = YES;

    [self layoutToParent];

    NSRect target = self.frame;
    NSRect start = NSMakeRect(target.origin.x,
                               target.origin.y + target.size.height,
                               target.size.width, target.size.height);
    [self setFrame:start display:NO];

    [self orderFront:nil];
    [self makeKeyWindow];
    [self makeFirstResponder:self.consoleView.inputField];

    // The backtick keystroke that triggered show was caught by GLFW's
    // poll, but AppKit re-dispatches the same NSEvent to the new key
    // window's first responder -- so the character gets inserted into
    // the input field after we focus it. Clear it on the next run-loop
    // tick (after the in-flight keyDown is delivered) and stash the
    // history position so Up arrow still works.
    PtConsoleView* view = self.consoleView;
    dispatch_async(dispatch_get_main_queue(), ^{
        NSString* v = view.inputField.stringValue;
        if (v.length == 1 && [v characterAtIndex:0] == '`') {
            view.inputField.stringValue = @"";
        }
    });

    // Backtick toggles when the panel is the key window.  An NSEvent
    // monitor catches the keystroke before the field editor inserts it,
    // so the user can press ` to close even with the input focused.
    // Ctrl+Space (VS Code / IntelliJ universal "show me my options")
    // force-opens the completion popup -- after the recent auto-trigger
    // improvements it's largely redundant, but kept for the "Esc
    // dismissed it, summon it back" case. We use Ctrl+Space, not
    // Cmd+Space, because Cmd+Space is owned by Spotlight system-wide
    // and never reaches our handler. Returning nil from the monitor
    // consumes the event before AppKit's interpretKeyEvents can
    // synthesise an insertText:" " on the same keystroke -- so unlike
    // the Win32 implementation we don't need a separate
    // s_suppress_next_wm_char gate.
    __weak PtConsolePanel* weakSelf = self;
    self.eventMonitor = [NSEvent
        addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
        handler:^NSEvent*(NSEvent* e) {
            if (e.window != weakSelf) return e;
            NSString* chars = e.charactersIgnoringModifiers;
            if (chars.length == 1 && [chars characterAtIndex:0] == '`') {
                [weakSelf toggle];
                return nil;          // swallow it -- don't insert into field
            }
            // Ctrl+Space -- force-show completion popup. Plain Control
            // mask only (no Cmd, no Option, no Shift) so e.g. Ctrl+Shift
            // +Space doesn't trip it accidentally.
            const NSEventModifierFlags mods =
                e.modifierFlags &
                (NSEventModifierFlagControl | NSEventModifierFlagCommand |
                 NSEventModifierFlagOption  | NSEventModifierFlagShift);
            if (mods == NSEventModifierFlagControl &&
                chars.length == 1 && [chars characterAtIndex:0] == ' ') {
                PtConsoleView* cv = weakSelf.consoleView;
                if (cv != nil) {
                    [cv refreshCompletions:YES];
                }
                return nil;          // swallow -- don't insert space
            }
            return e;
        }];

    [NSAnimationContext runAnimationGroup:^(NSAnimationContext* ctx) {
        ctx.duration = 0.22;
        ctx.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseOut];
        self.animator.alphaValue = 1.0;
        [self.animator setFrame:target display:YES];
    } completionHandler:nil];
}

- (void)hide {
    if (!self.isShown) return;
    self.isShown = NO;

    if (self.eventMonitor) {
        [NSEvent removeMonitor:self.eventMonitor];
        self.eventMonitor = nil;
    }

    NSRect cur = self.frame;
    NSRect target = NSMakeRect(cur.origin.x,
                                cur.origin.y + cur.size.height,
                                cur.size.width, cur.size.height);

    [NSAnimationContext runAnimationGroup:^(NSAnimationContext* ctx) {
        ctx.duration = 0.18;
        ctx.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseIn];
        self.animator.alphaValue = 0.0;
        [self.animator setFrame:target display:YES];
    } completionHandler:^{
        [self orderOut:nil];
        [self.parentRenderWindow makeKeyWindow];
    }];
}

- (void)toggle { if (self.isShown) [self hide]; else [self show]; }

@end

// ===========================================================================
namespace pt::app {

ConsoleOverlay::ConsoleOverlay() = default;
ConsoleOverlay::~ConsoleOverlay() { Shutdown(); }

bool ConsoleOverlay::Init(void* ns_window) {
    if (ns_window == nullptr) return false;
    NSWindow* parent = (__bridge NSWindow*)ns_window;

    PtConsolePanel* panel = [[PtConsolePanel alloc] initWithParent:parent];
    [parent addChildWindow:panel ordered:NSWindowAbove];

    opaque_ = (__bridge_retained void*)panel;
    SetGlobalInstance(this);

    // Pick up the live `con_font_scale` value that may already have
    // been set by autoexec.cfg / CLI args before the overlay was
    // constructed. Without this poll, the first Show would render at
    // 1.0x and only catch up to the configured scale on the next cvar
    // change. Cheap (one cvar lookup + float compare); no-op when the
    // cvar is at its default.
    [panel.consoleView applyFontScale];

    // Seed the up-arrow walk buffer from Console::history_. The Engine
    // populates that vector via LoadConsoleHistoryFromDisk BEFORE
    // ConsoleOverlay::Init runs (see Engine::Init), so previous-
    // session lines are already in place. We mirror them into the
    // NSMutableArray that drives Up/Down arrow walks, then set
    // historyPos to count so the first Up press recalls the most
    // recent entry. Empty history (first launch or --no-cfg) makes
    // this a no-op.
    {
        auto persisted = pt::console::Console::Get().History();
        for (const auto& entry : persisted) {
            NSString* ns = [NSString stringWithUTF8String:entry.c_str()];
            if (ns != nil) [panel.consoleView.history addObject:ns];
        }
        panel.consoleView.historyPos = panel.consoleView.history.count;
    }

    // Forward log lines into the overlay (the panel's contentView).
    return true;
}

void ConsoleOverlay::Shutdown() {
    if (opaque_) {
        PtConsolePanel* panel = (__bridge_transfer PtConsolePanel*)opaque_;
        [panel.parentRenderWindow removeChildWindow:panel];
        [panel orderOut:nil];
        opaque_ = nullptr;
    }
    if (g_instance == this) g_instance = nullptr;
}

void ConsoleOverlay::Show() {
    if (!opaque_) return;
    [(__bridge PtConsolePanel*)opaque_ show];
}
void ConsoleOverlay::Hide() {
    if (!opaque_) return;
    [(__bridge PtConsolePanel*)opaque_ hide];
}
void ConsoleOverlay::Toggle() {
    if (!opaque_) return;
    [(__bridge PtConsolePanel*)opaque_ toggle];
}
bool ConsoleOverlay::IsShown() const {
    if (!opaque_) return false;
    return ((__bridge PtConsolePanel*)opaque_).isShown == YES;
}

void ConsoleOverlay::ApplyTheme(std::string_view name) {
    if (!opaque_) return;
    PtConsolePanel* panel = (__bridge PtConsolePanel*)opaque_;
    NSString* nm = [NSString stringWithUTF8String:std::string(name).c_str()];
    dispatch_async(dispatch_get_main_queue(), ^{
        [panel.consoleView applyTheme:nm];
    });
}

void ConsoleOverlay::Repaint() {
    // Mac path: forward to the consoleView's setNeedsDisplay: (must
    // run on the main thread per AppKit rules). Today every caller
    // reaches Repaint() on the engine main thread -- the civetweb
    // worker thread only enqueues commands via Console::QueueExecute,
    // and the actual Execute() (which fires cvar on_change) runs from
    // Console::Drain() called from Engine::Tick() on main. On macOS
    // the engine main thread IS AppKit's main thread, so the
    // dispatch_async-to-main below is technically a no-op shuffle in
    // the steady state -- but it's a cheap defensive safety net for
    // any future caller that bypasses Drain. Win32 implementation is
    // in ConsoleOverlay_Win32.cpp.
    //
    // applyFontScale is the Mac equivalent of WinOverlay::EnsureFontScale
    // -- polling the live `con_font_scale` cvar so a change typed in
    // the web console / autoexec / CLI takes effect on the next
    // Repaint, with no per-frame cost when the value is unchanged.
    if (!opaque_) return;
    PtConsolePanel* panel = (__bridge PtConsolePanel*)opaque_;
    dispatch_async(dispatch_get_main_queue(), ^{
        [panel.consoleView applyFontScale];
        [panel.consoleView setNeedsDisplay:YES];
    });
}

void ConsoleOverlay::NotifyParentResized(int /*width*/, int /*height*/) {
    if (!opaque_) return;
    PtConsolePanel* panel = (__bridge PtConsolePanel*)opaque_;
    dispatch_async(dispatch_get_main_queue(), ^{
        [panel layoutToParent];
    });
}

void ConsoleOverlay::SetGlobalInstance(ConsoleOverlay* o) { g_instance = o; }

// Mac state-persistence stubs. The Cocoa overlay's history + scrollback
// live inside PtConsoleView's Objective-C ivars (NSMutableArray of
// NSString *); a parallel serialiser to the Win32 one would need to
// reach into that view across the C++/Obj-C boundary and dispatch_sync
// back to main to read it safely. Not required for the
// r_software_blit_recreate=prompt feature this lands alongside (that
// path is Win32-only -- DXGI flip-model lockout is Microsoft-specific
// and Mac never spawns a replacement process), so the Mac side returns
// false (no-op) for now. File a follow-up if Mac users want the same
// up-arrow / scrollback persistence across cold restarts.
bool ConsoleOverlay::SaveState(const std::string& /*path*/) const { return false; }
bool ConsoleOverlay::LoadState(const std::string& /*path*/)       { return false; }

void ConsoleOverlay::OnLog(pt::log::Level level, const std::string& body) {
    if (g_instance == nullptr || g_instance->opaque_ == nullptr) return;
    const char* lv = "info";
    switch (level) {
        case pt::log::Level::Warn:  lv = "warn";  break;
        case pt::log::Level::Error: lv = "error"; break;
        case pt::log::Level::Info:  lv = "info";  break;
    }
    NSString* line = [NSString stringWithUTF8String:body.c_str()];
    NSString* lvs  = [NSString stringWithUTF8String:lv];
    PtConsolePanel* panel = (__bridge PtConsolePanel*)g_instance->opaque_;
    PtConsoleView*  view  = panel.consoleView;
    dispatch_async(dispatch_get_main_queue(), ^{
        [view appendLine:line level:lvs];
    });
}

}  // namespace pt::app

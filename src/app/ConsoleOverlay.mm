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

#include "../console/Console.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

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

@property NSMutableArray<NSString*>*    allNames;

// Tab-completion ghost state (fish-shell autosuggestion).  First Tab on
// an ambiguous prefix extends to the longest common prefix AND shows the
// first remaining match in dim colour after the cursor.  Subsequent Tabs
// cycle (Shift+Tab back); Right-arrow at end / End commits;
// Esc / typing dismisses.
@property (strong) NSTextField*         ghostLabel;
@property (strong) NSMutableArray<NSString*>* ghostMatches;
@property NSInteger                     ghostIndex;
@property (strong) NSString*            ghostBefore;
@property (strong) NSString*            ghostPrefix;
@property BOOL                          ghostIsToken0;
@property BOOL                          ghostActive;
// Free-form cvar value-position state.  When ghostIsMeta is YES the
// matches array is [current, default] of a cvar with no allowed_values
// list, and ghostAnnotation holds "default: X" / "current: Y" describing
// the inactive one so both pieces are visible at once.
@property BOOL                          ghostIsMeta;
@property (strong) NSString*            ghostAnnotation;

- (instancetype)initWithFrame:(NSRect)frame;
- (void)appendLine:(NSString*)line level:(NSString*)level;
- (void)submitInput;
- (void)submitLine:(NSString*)line;
- (BOOL)handleTab;
- (BOOL)cycleGhost:(NSInteger)dir;
- (BOOL)commitGhost;
- (void)dismissGhost;
- (void)renderGhost;
- (void)activateValueGhost:(NSString*)cvarName;
- (void)refreshGhostAnnotation;
- (void)refreshNames;
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
@implementation PtConsoleView

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;

    self.wantsLayer = YES;
    self.layer.masksToBounds = NO;

    self.history    = [NSMutableArray array];
    self.historyPos = 0;
    self.allNames   = [NSMutableArray array];
    self.palette    = PtPaletteForTheme(@"hardcore");

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
    in.placeholderString = @"sys_info  ·  Tab completes  ·  Up/Down history";
    in.delegate = self;
    [self addSubview:in];
    self.inputField = in;

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
    [self rebuildBanner];
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
    // so an active ghost suggestion (e.g. one auto-activated by the
    // commitGhost path that calls into submitInput) would otherwise
    // stay visible with its stale text overlaid on the now-empty
    // field.  Drop the ghost explicitly here so the submit always
    // leaves a clean prompt.
    if (self.ghostActive) [self dismissGhost];
    [self submitLine:line];
    [self refreshNames];
}

// Paste-to-multiline: when the user pastes text containing newlines, run
// each complete line as its own command and leave the trailing partial
// (anything after the last newline) in the input. Fires from the field
// editor whenever stringValue changes from a user action -- programmatic
// setStringValue: does not re-trigger this, so the writeback below is safe.
- (void)controlTextDidChange:(NSNotification*)note {
    if (note.object != self.inputField) return;

    // Typing while the ghost is showing invalidates the suggestion (the
    // typed prefix has changed, so the transparent mirror would no
    // longer align).  Dismiss eagerly.
    if (self.ghostActive) [self dismissGhost];

    NSString* value = self.inputField.stringValue;

    // Auto-activate the value-position ghost when the user types
    // `<name> ` themselves (without tab + commit).  Fires when input is
    // exactly "<single token> " -- mirrors the post-commit auto-
    // activation so manual typers get the same affordance.
    if (value.length >= 2 && [value characterAtIndex:value.length - 1] == ' ') {
        NSString* trimmed = [value substringToIndex:value.length - 1];
        if (trimmed.length > 0 &&
            [trimmed rangeOfString:@" "].location == NSNotFound) {
            [self activateValueGhost:trimmed];
        }
    }

    // Backtick is the show/hide toggle; stripping it here catches the
    // case where the user presses ` to open AND starts typing fast
    // -- the original ` event arrives at the field editor before the
    // panel becomes the key window, so the NSEvent monitor that's
    // supposed to swallow it doesn't get a chance.
    if ([value rangeOfString:@"`"].location != NSNotFound) {
        value = [value stringByReplacingOccurrencesOfString:@"`" withString:@""];
        self.inputField.stringValue = value;
    }

    if ([value rangeOfString:@"\n"].location == NSNotFound) return;

    NSArray<NSString*>* parts = [value componentsSeparatedByString:@"\n"];
    NSString* trailing = parts.lastObject ? parts.lastObject : @"";
    self.inputField.stringValue = trailing;

    for (NSUInteger i = 0; i + 1 < parts.count; ++i) {
        [self submitLine:parts[i]];
    }
    [self refreshNames];
}

- (void)refreshNames {
    [self.allNames removeAllObjects];
    pt::console::Console::Get().EnumerateCVars("",
        [self](pt::console::CVar& v) {
            [self.allNames addObject:[NSString stringWithUTF8String:v.name.c_str()]];
        });
    pt::console::Console::Get().EnumerateCommands("",
        [self](pt::console::Command& c) {
            [self.allNames addObject:[NSString stringWithUTF8String:c.name.c_str()]];
        });
    [self.allNames sortUsingSelector:@selector(compare:)];
}

// NSControlTextEditingDelegate dispatch.  The field's `delegate` is set
// to this PtConsoleView, so AppKit calls the method on us (not on the
// field subclass).  Tab/Shift+Tab/Right/End/Esc are intercepted for
// ghost-mode interaction when active; otherwise they fall through to
// completion / history / cancel.
- (BOOL)control:(NSControl*)control textView:(NSTextView*)textView
      doCommandBySelector:(SEL)cmd {
    // Ghost-mode keys.  When the autosuggestion is active these
    // selectors mean cycle / commit / dismiss; everything else
    // dismisses and falls through to default behaviour.
    if (self.ghostActive) {
        if (cmd == @selector(insertTab:))     return [self cycleGhost:+1];
        if (cmd == @selector(insertBacktab:)) return [self cycleGhost:-1];
        if (cmd == @selector(moveRight:) || cmd == @selector(moveToEndOfLine:) ||
            cmd == @selector(moveToEndOfDocument:) || cmd == @selector(moveToEndOfParagraph:)) {
            return [self commitGhost];
        }
        if (cmd == @selector(insertNewline:)) {
            [self commitGhost];
            [self submitInput];
            return YES;
        }
        if (cmd == @selector(cancelOperation:)) {
            [self dismissGhost];
            return YES;
        }
        // Any other command dismisses; fall through.
        [self dismissGhost];
    }

    if (cmd == @selector(insertTab:))      return [self handleTab];
    if (cmd == @selector(insertBacktab:))  return [self handleTab];   // first Tab even with shift
    if (cmd == @selector(insertNewline:)) { [self submitInput]; return YES; }
    if (cmd == @selector(moveUp:)) {
        if (self.historyPos > 0) {
            self.historyPos -= 1;
            self.inputField.stringValue = self.history[self.historyPos];
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
        }
        return YES;
    }
    return NO;
}

- (BOOL)handleTab {
    if (self.allNames.count == 0) [self refreshNames];

    NSString* value = self.inputField.stringValue;
    NSRange lastSpace = [value rangeOfString:@" " options:NSBackwardsSearch];

    NSString* prefix;
    NSArray<NSString*>* candidates;

    if (lastSpace.location == NSNotFound) {
        // Token 0: cvar / command name.
        prefix = value;
        candidates = self.allNames;
    } else {
        // Value position. `toggle <cvar>` is special-cased: token 1 is a
        // cvar name (only those with allowed_values are useful). Otherwise
        // we complete from the named cvar's allowed_values.
        NSString* firstTok = [value substringToIndex:[value rangeOfString:@" "].location];
        prefix = [value substringFromIndex:lastSpace.location + 1];
        if ([firstTok isEqualToString:@"toggle"]) {
            NSMutableArray* toggleables = [NSMutableArray array];
            pt::console::Console::Get().EnumerateCVars("",
                [&](pt::console::CVar& v) {
                    if (!v.allowed_values.empty()) {
                        [toggleables addObject:[NSString stringWithUTF8String:v.name.c_str()]];
                    }
                });
            [toggleables sortUsingSelector:@selector(compare:)];
            candidates = toggleables;
        } else {
            auto* cv = pt::console::Console::Get().FindCVar(
                std::string_view([firstTok UTF8String]));
            if (cv == nullptr || cv->allowed_values.empty()) return YES;
            NSMutableArray* allowed = [NSMutableArray array];
            for (auto& a : cv->allowed_values) {
                [allowed addObject:[NSString stringWithUTF8String:a.c_str()]];
            }
            candidates = allowed;
        }
    }

    NSMutableArray<NSString*>* matches = [NSMutableArray array];
    for (NSString* n in candidates) {
        if ([n hasPrefix:prefix]) [matches addObject:n];
    }
    if (matches.count == 0) return YES;

    BOOL isToken0 = (lastSpace.location == NSNotFound);
    NSString* before = isToken0 ? @"" : [value substringToIndex:lastSpace.location + 1];

    if (matches.count == 1) {
        NSString* tail = isToken0 ? [matches[0] stringByAppendingString:@" "]
                                  : matches[0];
        self.inputField.stringValue = [before stringByAppendingString:tail];
        [self dismissGhost];
        return YES;
    }

    // Extend to longest common prefix (if it advances), then activate
    // ghost mode for cycling.
    NSString* common = matches[0];
    for (NSUInteger i = 1; i < matches.count; ++i) {
        NSUInteger j = 0;
        NSUInteger lim = MIN(common.length, [matches[i] length]);
        while (j < lim && [common characterAtIndex:j] == [matches[i] characterAtIndex:j]) ++j;
        common = [common substringToIndex:j];
        if (common.length == 0) break;
    }
    NSString* typedPrefix = prefix;
    if (common.length > prefix.length) {
        self.inputField.stringValue = [before stringByAppendingString:common];
        typedPrefix = common;
    }

    self.ghostMatches  = matches;
    self.ghostIndex    = 0;
    self.ghostBefore   = before;
    self.ghostPrefix   = typedPrefix;
    self.ghostIsToken0 = isToken0;
    self.ghostActive   = YES;
    [self renderGhost];
    return YES;
}

- (BOOL)cycleGhost:(NSInteger)dir {
    if (!self.ghostActive || self.ghostMatches.count == 0) return YES;
    NSInteger n = (NSInteger)self.ghostMatches.count;
    NSInteger i = ((self.ghostIndex + dir) % n + n) % n;
    self.ghostIndex = i;
    [self refreshGhostAnnotation];
    [self renderGhost];
    return YES;
}

- (BOOL)commitGhost {
    if (!self.ghostActive || self.ghostMatches.count == 0) {
        [self dismissGhost];
        return YES;
    }
    NSString* committed = self.ghostMatches[self.ghostIndex];
    BOOL      wasToken0 = self.ghostIsToken0;
    NSString* tail      = wasToken0 ? [committed stringByAppendingString:@" "] : committed;
    self.inputField.stringValue = [self.ghostBefore stringByAppendingString:tail];
    [self dismissGhost];
    // Token-0 commit just landed on a name -- if it's a cvar, chain
    // into a value-position ghost so the user immediately sees the
    // current (and default) value.
    if (wasToken0) [self activateValueGhost:committed];
    return YES;
}

- (void)dismissGhost {
    self.ghostActive      = NO;
    self.ghostMatches     = nil;
    self.ghostIndex       = 0;
    self.ghostBefore      = nil;
    self.ghostPrefix      = nil;
    self.ghostIsToken0    = NO;
    self.ghostIsMeta      = NO;
    self.ghostAnnotation  = nil;
    self.ghostLabel.attributedStringValue = [[NSAttributedString alloc] initWithString:@""];
}

- (void)activateValueGhost:(NSString*)name {
    auto& C = pt::console::Console::Get();
    NSMutableArray<NSString*>* matches = [NSMutableArray array];
    BOOL meta = NO;
    if (auto* cv = C.FindCVar(std::string_view([name UTF8String])); cv != nullptr) {
        if (!cv->allowed_values.empty()) {
            for (auto& a : cv->allowed_values) {
                [matches addObject:[NSString stringWithUTF8String:a.c_str()]];
            }
        } else {
            NSString* current = [NSString stringWithUTF8String:cv->value.c_str()];
            NSString* dflt    = [NSString stringWithUTF8String:cv->default_value.c_str()];
            [matches addObject:current];
            if (![dflt isEqualToString:current]) {
                [matches addObject:dflt];
                meta = YES;
            }
        }
    } else if (auto* cmd = C.FindCommand(std::string_view([name UTF8String]));
               cmd != nullptr && !cmd->default_args.empty()) {
        [matches addObject:[NSString stringWithUTF8String:cmd->default_args.c_str()]];
    }
    if (matches.count == 0) return;

    self.ghostMatches    = matches;
    self.ghostIndex      = 0;
    self.ghostBefore     = self.inputField.stringValue;   // "<name> "
    self.ghostPrefix     = @"";
    self.ghostIsToken0   = NO;
    self.ghostIsMeta     = meta;
    self.ghostActive     = YES;
    [self refreshGhostAnnotation];
    [self renderGhost];
}

- (void)refreshGhostAnnotation {
    if (!self.ghostIsMeta || self.ghostMatches.count < 2) {
        self.ghostAnnotation = nil;
        return;
    }
    // matches[0] = current, matches[1] = default (per activateValueGhost).
    if (self.ghostIndex == 0) {
        self.ghostAnnotation = [NSString stringWithFormat:@"  default: %@",
                                                          self.ghostMatches[1]];
    } else {
        self.ghostAnnotation = [NSString stringWithFormat:@"  current: %@",
                                                          self.ghostMatches[0]];
    }
}

- (void)renderGhost {
    if (!self.ghostActive || self.ghostMatches.count == 0) {
        self.ghostLabel.attributedStringValue = [[NSAttributedString alloc] initWithString:@""];
        return;
    }
    NSString* match = self.ghostMatches[self.ghostIndex];
    if (match.length < self.ghostPrefix.length || ![match hasPrefix:self.ghostPrefix]) {
        self.ghostLabel.attributedStringValue = [[NSAttributedString alloc] initWithString:@""];
        return;
    }
    NSString* typed = self.inputField.stringValue;
    NSString* tail  = [match substringFromIndex:self.ghostPrefix.length];

    NSFont* defaultFont = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightRegular];
    NSFont* font = self.inputField.font ? self.inputField.font : defaultFont;
    NSColor* dim = self.palette.info ? self.palette.info : [NSColor grayColor];
    NSMutableAttributedString* s = [[NSMutableAttributedString alloc] init];
    // Transparent run mirrors typed text width so the visible tail
    // lines up just past the cursor.
    [s appendAttributedString:[[NSAttributedString alloc] initWithString:typed
        attributes:@{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: [NSColor clearColor],
        }]];
    if (tail.length > 0) {
        [s appendAttributedString:[[NSAttributedString alloc] initWithString:tail
            attributes:@{
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: dim,
            }]];
    }
    if (self.ghostAnnotation.length > 0) {
        // Annotation rides in the same dim colour but italicised so it
        // reads as commentary rather than another committable token.
        NSFontDescriptor* fd = [font.fontDescriptor
            fontDescriptorWithSymbolicTraits:NSFontDescriptorTraitItalic];
        NSFont* italicCandidate = [NSFont fontWithDescriptor:fd size:font.pointSize];
        NSFont* italic = italicCandidate ? italicCandidate : font;
        [s appendAttributedString:[[NSAttributedString alloc] initWithString:self.ghostAnnotation
            attributes:@{
                NSFontAttributeName: italic,
                NSForegroundColorAttributeName: [dim colorWithAlphaComponent:0.7],
            }]];
    }
    self.ghostLabel.attributedStringValue = s;
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
    // run on the main thread per AppKit rules; cvar on_change runs on
    // the engine thread / civetweb worker thread, so dispatch async
    // to main). Win32 implementation is in ConsoleOverlay_Win32.cpp.
    if (!opaque_) return;
    PtConsolePanel* panel = (__bridge PtConsolePanel*)opaque_;
    dispatch_async(dispatch_get_main_queue(), ^{
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

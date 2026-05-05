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
@interface PtLogoView : NSView
@end

@implementation PtLogoView

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

    NSColor* cyan = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:1.0];
    [cyan setStroke];
    [cyan setFill];

    // Hex frame.
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

    // Three-bounce ray path.
    NSBezierPath* ray = [NSBezierPath bezierPath];
    [ray moveToPoint:NSMakePoint(7.5, 11.0)];
    [ray lineToPoint:NSMakePoint(13.0, 19.0)];
    [ray lineToPoint:NSMakePoint(21.5, 14.0)];
    [ray lineToPoint:NSMakePoint(24.5, 22.0)];
    ray.lineWidth = 1.6;
    ray.lineCapStyle = NSLineCapStyleRound;
    ray.lineJoinStyle = NSLineJoinStyleRound;
    [ray stroke];

    // Hit-point dots.
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

@interface PtConsoleView : NSView <NSTextFieldDelegate, NSTextViewDelegate>

@property (strong) NSVisualEffectView*  backdrop;
@property (strong) NSScrollView*        outputScroll;
@property (strong) NSTextView*          outputView;
@property (strong) PtConsoleInputField* inputField;
@property (strong) NSTextField*         promptLabel;
@property (strong) NSTextField*         statusLabel;

@property NSMutableArray<NSString*>*    history;
@property NSInteger                     historyPos;

@property NSMutableArray<NSString*>*    allNames;
@property NSString*                     lastTabPrefix;
@property BOOL                          lastTabShownList;

- (instancetype)initWithFrame:(NSRect)frame;
- (void)appendLine:(NSString*)line level:(NSString*)level;
- (void)submitInput;
- (BOOL)handleTab;
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
    bg.layer.borderColor = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:0.45].CGColor;
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
    // Electric cyan to match the web console.
    prompt.textColor = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:1.0];
    prompt.bezeled = NO;
    prompt.drawsBackground = NO;
    [self addSubview:prompt];
    self.promptLabel = prompt;

    NSTextField* status = [NSTextField labelWithString:@"DEMONT · PATHTRACER · CONSOLE"];
    status.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightSemibold];
    status.textColor = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:0.65];
    status.frame = NSMakeRect(frame.size.width - 240, frame.size.height - 24, 200, 16);
    status.alignment = NSTextAlignmentRight;
    status.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [self addSubview:status];
    self.statusLabel = status;

    // Top-left corner logo glyph -- same design as the web console.
    PtLogoView* logo = [[PtLogoView alloc] initWithFrame:NSMakeRect(14, frame.size.height - 28, 18, 18)];
    logo.autoresizingMask = NSViewMinYMargin;
    logo.wantsLayer = YES;
    [self addSubview:logo];

    PtConsoleInputField* in = [[PtConsoleInputField alloc]
        initWithFrame:NSMakeRect(32, 8, frame.size.width - 46, 22)];
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
    [self appendLine:@"DeMonT PathTracer · v0.1.0  ·  De Monte Carlo-esque Tracer"
              level:@"out"];
    [self appendLine:@"console attached. type \"list_commands\" or hit Tab."
              level:@"info"];

    return self;
}

// Special-case bannered ASCII line -- one fixed colour per role.
- (void)appendBannerLine:(NSString*)line level:(NSString*)role {
    NSColor* col = [NSColor colorWithCalibratedRed:0.0 green:0.94 blue:1.0 alpha:1.0];      // cyan frame
    if ([role isEqualToString:@"letters"]) {
        col = [NSColor colorWithCalibratedRed:0.95 green:0.97 blue:1.0 alpha:1.0];          // bright fg
    } else if ([role isEqualToString:@"ray"]) {
        // Hotter pink so the triangle edges + bounces really pop.
        col = [NSColor colorWithCalibratedRed:1.0 green:0.37 blue:0.64 alpha:1.0];
    }
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
    // Match the web console palette: cyan, magenta, amber, hot red.
    if ([level isEqualToString:@"warn"])  return [NSColor colorWithCalibratedRed:1.00 green:0.78 blue:0.23 alpha:1.0];
    if ([level isEqualToString:@"error"]) return [NSColor colorWithCalibratedRed:1.00 green:0.29 blue:0.37 alpha:1.0];
    if ([level isEqualToString:@"input"]) return [NSColor colorWithCalibratedRed:1.00 green:0.23 blue:0.55 alpha:1.0];
    if ([level isEqualToString:@"out"])   return [NSColor colorWithWhite:0.95 alpha:1.0];
    return [NSColor colorWithWhite:0.52 alpha:1.0];
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

- (void)submitInput {
    NSString* line = [self.inputField.stringValue copy];
    if (line.length == 0) return;
    self.inputField.stringValue = @"";
    [self.history addObject:line];
    if (self.history.count > 200) [self.history removeObjectAtIndex:0];
    self.historyPos = self.history.count;

    [self appendLine:line level:@"input"];

    auto result = pt::console::Console::Get().Execute(
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
// field subclass).  We map insertTab:/insertNewline:/moveUp:/moveDown:
// to our completion / submit / history hooks.
- (BOOL)control:(NSControl*)control textView:(NSTextView*)textView
      doCommandBySelector:(SEL)cmd {
    if (cmd == @selector(insertTab:))      return [self handleTab];
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
        // Value position: complete from the cvar's allowed_values.
        NSString* cvarName = [value substringToIndex:[value rangeOfString:@" "].location];
        auto* cv = pt::console::Console::Get().FindCVar(
            std::string_view([cvarName UTF8String]));
        if (cv == nullptr || cv->allowed_values.empty()) return YES;
        NSMutableArray* allowed = [NSMutableArray array];
        for (auto& a : cv->allowed_values) {
            [allowed addObject:[NSString stringWithUTF8String:a.c_str()]];
        }
        candidates = allowed;
        prefix = [value substringFromIndex:lastSpace.location + 1];
    }

    NSMutableArray<NSString*>* matches = [NSMutableArray array];
    for (NSString* n in candidates) {
        if ([n hasPrefix:prefix]) [matches addObject:n];
    }
    if (matches.count == 0) return YES;

    NSString* before = (lastSpace.location == NSNotFound)
                         ? @""
                         : [value substringToIndex:lastSpace.location + 1];

    if (matches.count == 1) {
        NSString* tail = (lastSpace.location == NSNotFound)
                            ? [matches[0] stringByAppendingString:@" "]
                            : matches[0];
        self.inputField.stringValue = [before stringByAppendingString:tail];
        self.lastTabPrefix = nil;
        return YES;
    }

    NSString* common = matches[0];
    for (NSUInteger i = 1; i < matches.count; ++i) {
        NSUInteger j = 0;
        NSUInteger lim = MIN(common.length, [matches[i] length]);
        while (j < lim && [common characterAtIndex:j] == [matches[i] characterAtIndex:j]) ++j;
        common = [common substringToIndex:j];
        if (common.length == 0) break;
    }
    if (common.length > prefix.length) {
        self.inputField.stringValue = [before stringByAppendingString:common];
        self.lastTabPrefix = common;
        self.lastTabShownList = NO;
        return YES;
    }

    if (self.lastTabPrefix && [self.lastTabPrefix isEqualToString:prefix] && !self.lastTabShownList) {
        [self appendLine:[matches componentsJoinedByString:@"  "] level:@"out"];
        self.lastTabShownList = YES;
    } else {
        self.lastTabPrefix = prefix;
        self.lastTabShownList = NO;
    }
    return YES;
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

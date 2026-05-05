// Native console overlay implementation. AppKit + CoreText.
//
// Layout (top half of the window when shown):
//   [NSVisualEffectView  -- dark vibrancy backdrop, rounded bottom corners]
//     [NSScrollView      -- output history, monospaced]
//     [NSTextField       -- input field with `>` prompt]
//   Slides in/out with Core Animation; toggled by backtick.

#include "ConsoleOverlay.h"

#include "../console/Console.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <atomic>
#include <mutex>
#include <vector>

namespace {
pt::app::ConsoleOverlay* g_instance = nullptr;
}

// ---------------------------------------------------------------------------
// PtConsoleInputField: NSTextField subclass that:
//   * styles itself for the console (mono font, accent prompt)
//   * intercepts Tab for completion
//   * intercepts Up/Down for history
@class PtConsoleView;

@interface PtConsoleInputField : NSTextField
@property (weak) PtConsoleView* owner;
@end

// ---------------------------------------------------------------------------
@interface PtConsoleView : NSView <NSTextFieldDelegate, NSTextViewDelegate>

@property (strong) NSVisualEffectView*  backdrop;
@property (strong) NSScrollView*        outputScroll;
@property (strong) NSTextView*          outputView;
@property (strong) PtConsoleInputField* inputField;
@property (strong) NSTextField*         promptLabel;
@property (strong) NSTextField*         statusLabel;

@property BOOL                          isShown;
@property NSMutableArray<NSString*>*    history;
@property NSInteger                     historyPos;

@property NSMutableArray<NSString*>*    allNames;     // cvars + commands sorted
@property NSString*                     lastTabPrefix;
@property BOOL                          lastTabShownList;

- (instancetype)initWithFrame:(NSRect)frame;
- (void)layoutForParent;
- (void)show;
- (void)hide;
- (void)toggle;
- (void)appendLine:(NSString*)line level:(NSString*)level;
- (void)submitInput;
- (BOOL)handleTab;
- (void)refreshNames;
@end

// ---------------------------------------------------------------------------
@implementation PtConsoleInputField

- (BOOL)becomeFirstResponder {
    BOOL ok = [super becomeFirstResponder];
    if (ok) {
        // Move caret to end after focus.
        NSText* editor = [self currentEditor];
        if (editor) [editor setSelectedRange:NSMakeRange(self.stringValue.length, 0)];
    }
    return ok;
}

// Field editor key handling. We catch Tab / Up / Down here.
- (BOOL)control:(NSControl*)control textView:(NSTextView*)textView
      doCommandBySelector:(SEL)cmd {
    if (cmd == @selector(insertTab:)) {
        return [self.owner handleTab];
    }
    if (cmd == @selector(moveUp:)) {
        if (self.owner.historyPos > 0) {
            self.owner.historyPos -= 1;
            self.stringValue = self.owner.history[self.owner.historyPos];
        }
        return YES;
    }
    if (cmd == @selector(moveDown:)) {
        if (self.owner.historyPos < (NSInteger)self.owner.history.count) {
            self.owner.historyPos += 1;
            if (self.owner.historyPos == (NSInteger)self.owner.history.count) {
                self.stringValue = @"";
            } else {
                self.stringValue = self.owner.history[self.owner.historyPos];
            }
        }
        return YES;
    }
    if (cmd == @selector(insertNewline:)) {
        [self.owner submitInput];
        return YES;
    }
    return NO;
}
@end

// ---------------------------------------------------------------------------
@implementation PtConsoleView

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;

    self.wantsLayer = YES;
    self.layer.masksToBounds = NO;

    self.history = [NSMutableArray array];
    self.historyPos = 0;
    self.allNames = [NSMutableArray array];

    // ---- Vibrancy backdrop ----
    NSVisualEffectView* bg = [[NSVisualEffectView alloc] initWithFrame:self.bounds];
    bg.material = NSVisualEffectMaterialHUDWindow;
    bg.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    bg.state = NSVisualEffectStateActive;
    bg.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    bg.wantsLayer = YES;
    bg.layer.cornerRadius = 8.0;
    bg.layer.maskedCorners = kCALayerMinXMinYCorner | kCALayerMaxXMinYCorner;
    bg.layer.borderWidth = 1.0;
    bg.layer.borderColor = [NSColor colorWithWhite:0.2 alpha:0.5].CGColor;
    [self addSubview:bg];
    self.backdrop = bg;

    // ---- Output scroll view + text view ----
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

    // ---- Prompt label ----
    NSTextField* prompt = [NSTextField labelWithString:@">"];
    prompt.frame = NSMakeRect(14, 8, 14, 22);
    prompt.font = [NSFont monospacedSystemFontOfSize:13 weight:NSFontWeightSemibold];
    prompt.textColor = [NSColor colorWithCalibratedRed:0.91 green:0.60 blue:0.41 alpha:1.0];
    prompt.bezeled = NO;
    prompt.drawsBackground = NO;
    [self addSubview:prompt];
    self.promptLabel = prompt;

    // ---- Status label (top-right) ----
    NSTextField* status = [NSTextField labelWithString:@"path tracer · console"];
    status.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    status.textColor = [NSColor colorWithWhite:0.55 alpha:1.0];
    status.frame = NSMakeRect(frame.size.width - 220, frame.size.height - 22, 200, 16);
    status.alignment = NSTextAlignmentRight;
    status.autoresizingMask = NSViewMinXMargin | NSViewMinYMargin;
    [self addSubview:status];
    self.statusLabel = status;

    // ---- Input field ----
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

    self.alphaValue = 0.0;
    self.hidden = YES;
    self.isShown = NO;

    [self appendLine:@"console attached. type \"list_commands\" or hit Tab."
              level:@"info"];

    return self;
}

// Lay out at top half of parent, full width.
- (void)layoutForParent {
    NSView* parent = self.superview;
    if (!parent) return;
    CGFloat w = parent.bounds.size.width;
    CGFloat h = MIN(MAX(parent.bounds.size.height * 0.45, 240), 560);
    self.frame = NSMakeRect(0, parent.bounds.size.height - h, w, h);
}

- (void)show {
    if (self.isShown) return;
    self.isShown = YES;
    self.hidden = NO;

    NSView* parent = self.superview;
    if (parent) {
        CGFloat w = parent.bounds.size.width;
        CGFloat h = MIN(MAX(parent.bounds.size.height * 0.45, 240), 560);
        self.frame = NSMakeRect(0, parent.bounds.size.height, w, h);  // start above
    }

    [NSAnimationContext runAnimationGroup:^(NSAnimationContext* ctx) {
        ctx.duration = 0.22;
        ctx.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseOut];
        self.animator.alphaValue = 1.0;
        if (self.superview) {
            CGFloat w = self.superview.bounds.size.width;
            CGFloat h = self.frame.size.height;
            [self.animator setFrame:NSMakeRect(0, self.superview.bounds.size.height - h, w, h)];
        }
    } completionHandler:^{
        [self.window makeFirstResponder:self.inputField];
    }];
}

- (void)hide {
    if (!self.isShown) return;
    self.isShown = NO;

    [NSAnimationContext runAnimationGroup:^(NSAnimationContext* ctx) {
        ctx.duration = 0.18;
        ctx.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseIn];
        self.animator.alphaValue = 0.0;
        if (self.superview) {
            CGFloat w = self.superview.bounds.size.width;
            CGFloat h = self.frame.size.height;
            [self.animator setFrame:NSMakeRect(0, self.superview.bounds.size.height, w, h)];
        }
    } completionHandler:^{
        self.hidden = YES;
        [self.window makeFirstResponder:nil];
    }];
}

- (void)toggle { if (self.isShown) [self hide]; else [self show]; }

- (NSColor*)colorForLevel:(NSString*)level {
    if ([level isEqualToString:@"warn"])  return [NSColor colorWithCalibratedRed:0.85 green:0.75 blue:0.42 alpha:1.0];
    if ([level isEqualToString:@"error"]) return [NSColor colorWithCalibratedRed:0.86 green:0.42 blue:0.34 alpha:1.0];
    if ([level isEqualToString:@"input"]) return [NSColor colorWithCalibratedRed:0.91 green:0.60 blue:0.41 alpha:1.0];
    if ([level isEqualToString:@"out"])   return [NSColor colorWithWhite:0.92 alpha:1.0];
    return [NSColor colorWithWhite:0.65 alpha:1.0];   // info default
}

- (void)appendLine:(NSString*)line level:(NSString*)level {
    NSString* prefix = @"";
    if ([level isEqualToString:@"input"]) prefix = @"> ";
    NSDictionary* attrs = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName: [self colorForLevel:level],
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
        // Strip trailing newline; appendLine adds its own.
        while (o.length > 0 && [o characterAtIndex:o.length - 1] == '\n') {
            o = [o substringToIndex:o.length - 1];
        }
        [self appendLine:o level:@"out"];
    }
    if (!result.ok) {
        NSString* e = [NSString stringWithUTF8String:result.error.c_str()];
        [self appendLine:[NSString stringWithFormat:@"error: %@", e] level:@"error"];
    }

    // After execution something might have registered a new cvar/command.
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

- (BOOL)handleTab {
    if (self.allNames.count == 0) [self refreshNames];

    NSString* value = self.inputField.stringValue;
    NSRange firstSpace = [value rangeOfString:@" "];
    if (firstSpace.location != NSNotFound) return NO;   // past first token

    NSMutableArray<NSString*>* matches = [NSMutableArray array];
    for (NSString* n in self.allNames) {
        if ([n hasPrefix:value]) [matches addObject:n];
    }
    if (matches.count == 0) return YES;
    if (matches.count == 1) {
        self.inputField.stringValue = [matches[0] stringByAppendingString:@" "];
        self.lastTabPrefix = nil;
        return YES;
    }

    // Longest common prefix.
    NSString* common = matches[0];
    for (NSUInteger i = 1; i < matches.count; ++i) {
        NSUInteger j = 0;
        NSUInteger lim = MIN(common.length, [matches[i] length]);
        while (j < lim && [common characterAtIndex:j] == [matches[i] characterAtIndex:j]) ++j;
        common = [common substringToIndex:j];
        if (common.length == 0) break;
    }
    if (common.length > value.length) {
        self.inputField.stringValue = common;
        self.lastTabPrefix = common;
        self.lastTabShownList = NO;
        return YES;
    }

    if (self.lastTabPrefix && [self.lastTabPrefix isEqualToString:value] && !self.lastTabShownList) {
        [self appendLine:[matches componentsJoinedByString:@"  "] level:@"out"];
        self.lastTabShownList = YES;
    } else {
        self.lastTabPrefix = value;
        self.lastTabShownList = NO;
    }
    return YES;
}

@end

// ===========================================================================
namespace pt::app {

ConsoleOverlay::ConsoleOverlay() = default;
ConsoleOverlay::~ConsoleOverlay() { Shutdown(); }

bool ConsoleOverlay::Init(void* ns_window) {
    if (ns_window == nullptr) return false;
    NSWindow* win = (__bridge NSWindow*)ns_window;
    NSView*   parent = win.contentView;
    if (parent == nil) return false;

    parent.wantsLayer = YES;

    PtConsoleView* view = [[PtConsoleView alloc] initWithFrame:parent.bounds];
    view.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
    [parent addSubview:view positioned:NSWindowAbove relativeTo:nil];
    [view layoutForParent];

    opaque_ = (__bridge_retained void*)view;

    SetGlobalInstance(this);
    return true;
}

void ConsoleOverlay::Shutdown() {
    if (opaque_) {
        PtConsoleView* view = (__bridge_transfer PtConsoleView*)opaque_;
        [view removeFromSuperview];
        opaque_ = nullptr;
    }
    if (g_instance == this) g_instance = nullptr;
}

void ConsoleOverlay::Show() {
    if (!opaque_) return;
    PtConsoleView* v = (__bridge PtConsoleView*)opaque_;
    [v show];
}

void ConsoleOverlay::Hide() {
    if (!opaque_) return;
    PtConsoleView* v = (__bridge PtConsoleView*)opaque_;
    [v hide];
}

void ConsoleOverlay::Toggle() {
    if (!opaque_) return;
    PtConsoleView* v = (__bridge PtConsoleView*)opaque_;
    [v toggle];
}

bool ConsoleOverlay::IsShown() const {
    if (!opaque_) return false;
    PtConsoleView* v = (__bridge PtConsoleView*)opaque_;
    return v.isShown == YES;
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
    PtConsoleView* v = (__bridge PtConsoleView*)g_instance->opaque_;
    // Marshal to main thread; AppKit is main-thread only.
    dispatch_async(dispatch_get_main_queue(), ^{
        [v appendLine:line level:lvs];
    });
}

}  // namespace pt::app

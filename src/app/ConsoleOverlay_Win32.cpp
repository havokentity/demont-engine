// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Native Win32 console overlay -- the Windows analog of the Cocoa
// implementation in ConsoleOverlay.mm.
//
// Implemented as a Win32 child window attached to the GLFW window's
// HWND, painted with GDI + a monospace font. The overlay is decoupled
// from the rendering pipeline (Vulkan / software / whatever) the same
// way the Mac NSView is decoupled from Metal: it lives in the OS's
// window hierarchy, draws independently, and the DWM compositor
// stacks it on top of the Vulkan-presented surface.
//
// The HARD rule against GUI libraries (Dear ImGui / Qt / etc.) means
// we use plain Win32 + GDI here. CreateFont, TextOutW, FillRect --
// all part of Windows itself, equivalent to using NSView+CoreText
// on the Mac side.

// WIN32_LEAN_AND_MEAN and NOMINMAX are set via the project-wide
// command line; defining them here would just produce redefinition
// warnings.
#include <Windows.h>

#include "ConsoleOverlay.h"
#include "../console/Completion.h"
#include "../console/Console.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>      // std::FILE / std::fopen / std::fwrite for SaveState
#include <cstdlib>     // std::strtof (engine builds /EHs-c-, no std::stof)
#include <cstring>     // std::strncmp for LoadState header check
#include <deque>
#include <filesystem>  // std::filesystem::rename for atomic SaveState
#include <fstream>     // std::ifstream / std::getline for LoadState
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace pt::app {

namespace {

// Theme palette. Names match the r_theme cvar's allowed values; the
// Mac overlay uses the same set so engine -> overlay theme flow Just
// Works on both platforms. logo_* colours mirror Mac's PtThemePalette
// logoFrame / logoLetters / logoRay -- so the boot banner re-colours
// on r_theme change just like NSView does there.
struct Theme {
    const char* name;
    COLORREF panel;
    COLORREF text;
    COLORREF accent;
    COLORREF prompt;
    COLORREF dim;
    COLORREF logo_frame;
    COLORREF logo_letters;
    COLORREF logo_ray;
};
// Multiply each RGB channel by `factor` clamped to [0, 1]. Used to
// derive a "dimmer accent" / "darker panel" shade for the popup's
// selected-row background and panel fill -- the Theme struct only
// stores the base palette, so we synthesise the depth shades here
// rather than expanding the struct + every theme table entry.
inline COLORREF DimColor(COLORREF c, float factor) {
    if (factor < 0.0f) factor = 0.0f;
    if (factor > 1.0f) factor = 1.0f;
    BYTE r = static_cast<BYTE>(static_cast<float>(GetRValue(c)) * factor + 0.5f);
    BYTE g = static_cast<BYTE>(static_cast<float>(GetGValue(c)) * factor + 0.5f);
    BYTE b = static_cast<BYTE>(static_cast<float>(GetBValue(c)) * factor + 0.5f);
    return RGB(r, g, b);
}

constexpr Theme kThemes[] = {
    //              panel             text              accent            prompt            dim               logo_frame        logo_letters      logo_ray
    {"hardcore",  RGB( 15, 18, 28), RGB(220,220,232), RGB(  0,240,255), RGB(  0,255,128), RGB(110,120,146), RGB(  0,130,160), RGB(231,234,242), RGB(255, 58,140)},
    {"amber",     RGB( 15, 10,  5), RGB(255,200,140), RGB(255,140, 40), RGB(255,200, 80), RGB(180,128, 80), RGB(180, 80, 20), RGB(255,220,160), RGB(255, 90, 40)},
    {"synthwave", RGB( 20,  8, 40), RGB(255,180,250), RGB(255, 80,180), RGB( 80,255,255), RGB(180,100,180), RGB( 80, 40,120), RGB(255,150,250), RGB( 80,255,255)},
    {"matrix",    RGB(  0, 15,  5), RGB( 80,255, 80), RGB(  0,200,  0), RGB(120,255,120), RGB( 40,140, 40), RGB(  0, 80, 30), RGB(120,255,120), RGB(180,255, 80)},
    {"vault",     RGB( 20, 30, 40), RGB(180,220,255), RGB( 80,160,255), RGB(255,200, 80), RGB(100,140,180), RGB( 40, 80,120), RGB(180,220,255), RGB(255,180, 80)},
    {"sakura",    RGB( 40, 15, 30), RGB(255,200,220), RGB(255,140,180), RGB(180,255,200), RGB(180,140,160), RGB(120, 80,100), RGB(255,210,230), RGB(255, 80,180)},
    {"mono",      RGB( 15, 15, 15), RGB(220,220,220), RGB(180,180,180), RGB(220,220,220), RGB(120,120,120), RGB( 80, 80, 80), RGB(220,220,220), RGB(160,160,160)},
};

// Per-line role used by Paint() to pick a palette colour from the
// active theme. Stored on every scrollback entry so theme switches
// recolour the banner without needing to rebuild it.
enum class LineRole : std::uint8_t {
    Default,
    LogoFrame,
    LogoLetters,
    LogoRay,
    Echo,    // "> command" lines for user-submitted input
};

struct ScrollLine {
    std::string text;
    LineRole    role = LineRole::Default;
};

// Panel height now matches Mac (PtConsolePanel.layoutToParent): 45% of
// parent client height, clamped to [240, 560]. Recomputed on resize.
constexpr int kFontHeight  = 14;
constexpr int kLineHeight  = 18;
constexpr int kPaddingX    = 12;
constexpr int kPaddingY    = 8;
constexpr int kScrollMax   = 4096;
constexpr int kHistoryMax  = 256;

// Slide+fade animation, mirroring NSAnimationContext on Mac
// (0.22s show / 0.18s hide, ease-out / ease-in cubic).
constexpr UINT_PTR kAnimTimerId   = 0xC04501u;
constexpr UINT     kAnimTickMs    = 16;
constexpr DWORD    kShowDurMs     = 220;
constexpr DWORD    kHideDurMs     = 180;
constexpr BYTE     kPanelAlpha    = 217;     // peak alpha for layered path

// Focus-watchdog timer.  Runs at low frequency while the console is
// shown; on each tick it checks whether keyboard focus has drifted off
// the console child (which has happened intermittently after Enter --
// something in GLFW's WndProc or the OS briefly hands focus back to
// the parent past our WM_SETFOCUS redirect).  Re-asserts SetFocus when
// drift is detected.  Gated on wants_focus_on_activate_ so a
// deliberate click into the game viewport stays out of the loop.
constexpr UINT_PTR kFocusGuardTimerId = 0xC04502u;
constexpr UINT     kFocusGuardMs      = 100;

// Custom message: marshal a repaint request from a non-UI thread
// (e.g. pt::log sinks, which fire on whichever thread emits the log)
// onto the overlay's owning thread. Cross-thread InvalidateRect is
// undefined per Win32 -- PostMessage is the documented bridge.
constexpr UINT WM_APP_REPAINT  = WM_APP + 1;
// Deferred SetFocus.  When the user presses backtick to open the
// console, the WM_KEYDOWN is dispatched to the parent (GLFW HWND);
// GLFW's WndProc fires our key callback which calls Show() ->
// SetFocus(child) deep in its own message handler.  GLFW (or the
// system) sometimes reverts focus before the call chain unwinds,
// leaving the parent focused and keystrokes routed to the camera
// (WASD).  Posting this message asks Windows to call SetFocus
// AFTER the current message dispatch completes -- by then GLFW has
// finished, and our SetFocus sticks.
constexpr UINT WM_APP_SETFOCUS = WM_APP + 2;

int ComputePanelHeight(int parent_h) {
    int h = static_cast<int>(parent_h * 0.45f);
    if (h < 240) h = 240;
    if (h > 560) h = 560;
    return h;
}

// UTF-8 -> UTF-16 helper for TextOutW.
std::vector<wchar_t> ToWide(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::vector<wchar_t> w(n);
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

class WinOverlay {
public:
    // OnLog is invoked from arbitrary log-source threads via the
    // pt::log sink chain, so reads of `g` must be atomic with respect
    // to Init/Shutdown writes on the main thread (otherwise we have a
    // C++ data race + a use-after-free if a log emit lands during
    // shutdown).  std::atomic<T*> with default seq-cst gives a clean
    // happens-before across the publish (g.store(this) in Init) and
    // every load(), and a clear poison value (g.store(nullptr) in
    // Shutdown) before HWND/GDI cleanup.
    static std::atomic<WinOverlay*> g;

    bool  Init(HWND parent);
    void  Shutdown();
    void  Show();
    void  Hide();
    void  Toggle();
    bool  IsShown() const { return shown_; }
    void  ApplyTheme(std::string_view name);
    void  NotifyParentResized(int w, int h);
    void  OnLog(pt::log::Level lvl, const std::string& body);
    // Public: forwards from ConsoleOverlay::Repaint() (PR #10) so cvar
    // on_change handlers can ping the overlay without copy-pasting
    // values across the IPC boundary.
    void  Repaint();
    bool  SaveState(const std::string& path) const;
    bool  LoadState(const std::string& path);

private:
    static LRESULT CALLBACK WndProcThunk(HWND, UINT, WPARAM, LPARAM);
    static LRESULT CALLBACK ParentWndProcThunk(HWND, UINT, WPARAM, LPARAM);
    LRESULT WndProc(HWND, UINT, WPARAM, LPARAM);
    void Paint(HDC dc);
    void SubmitInput();
    // VS Code-style completion popup. RefreshCompletions rebuilds the
    // candidate list from `input_` + `cursor_` and (re)opens the
    // popup; MovePopupSelection moves the highlight; CommitPopup
    // replaces the word-at-cursor with the highlighted match;
    // DismissPopup tears the state down. See the PopupState comment
    // below + Completion.h for the scoring + token-context model.
    void RefreshCompletions(bool force_show);
    void MovePopupSelection(int dir);
    void CommitPopup(bool chain_next);
    void DismissPopup();
    // Single source of truth for how many popup rows actually fit
    // between the log area and the input row at the current panel
    // size. Used by both Paint (to pick visible_n) and
    // MovePopupSelection (so scroll_offset advances at the same
    // threshold the user sees). Returns 0 if even one row won't fit.
    int  PopupVisibleRows() const;
    void StartAnim(bool showing);
    void TickAnim();
    int  CurrentY() const;
    // Re-creates `font_` at the new scaled height and updates
    // line_height_. Called from Paint() when the `con_font_scale`
    // cvar value differs from font_scale_. Clamps to [0.5, 3.0].
    // No-op (returns early) if scale matches the cached value, so
    // it's cheap to call every frame.
    void EnsureFontScale();

    enum class AnimState : std::uint8_t { Idle, Showing, Hiding };

    HWND      parent_      = nullptr;
    HWND      hwnd_        = nullptr;
    HFONT     font_        = nullptr;
    // True iff `font_` was created by us via CreateFontW and we own
    // its lifetime. False if it points at a Win32 stock object (from
    // GetStockObject), which the OS owns -- DeleteObject on a stock
    // GDI handle is documented as undefined behaviour and was caught
    // by Copilot review on the SYSTEM_FIXED_FONT fallback path.
    bool      font_is_owned_ = false;
    // Effective font scale as last applied by RecreateFontFromScale.
    // Driven by the `con_font_scale` cvar; polled inside Paint() so a
    // runtime change shows up on the next repaint without a separate
    // on_change wiring path. Clamped to [0.5, 3.0] at apply time.
    float     font_scale_   = 1.0f;
    // Cached pixel-space line height = round(kLineHeight * font_scale_).
    // Replaces the constexpr kLineHeight at the four use sites in
    // Paint() (input row Y, scrollback row count, scrollback Y stride,
    // caret rect height) so they track the current scale.
    int       line_height_  = kLineHeight;
    bool      shown_       = false;
    bool      is_layered_  = false;
    int       parent_w_    = 0;
    int       parent_h_    = 0;
    int       panel_h_     = 360;
    Theme     theme_       = kThemes[0];

    AnimState anim_state_     = AnimState::Idle;
    DWORD     anim_start_ms_  = 0;
    DWORD     anim_dur_ms_    = 0;
    int       anim_from_y_    = 0;
    int       anim_to_y_      = 0;
    BYTE      anim_from_alpha_= 0;
    BYTE      anim_to_alpha_  = 0;

    // Guards scrollback_ / input_ / cursor_ / history_ / scroll_lines_.
    // WARNING: callers must NOT invoke LOG_INFO / LOG_WARN / LOG_ERROR
    // while holding this mutex. The log emit path fans out to every
    // registered sink synchronously on the calling thread, and our own
    // OnLog sink (line ~689) re-locks state_mutex_ to push the line
    // onto scrollback_. Recursive lock on std::mutex is undefined
    // behaviour; MSVC's _Mtx_lock returns _RESOURCE_DEADLOCK_WOULD_OCCUR
    // and std::mutex::lock() throws std::system_error, which under our
    // /EHs-c- (-fno-exceptions) build immediately fast-fails the
    // process (caught in a crash dump from PR #12 review: r_denoiser
    // value-position popup + Ctrl+Space LOG_INFO -> ucrtbase fast-fail
    // subcode 7). WM_KEYDOWN now uses unique_lock so the two
    // diagnostic logs inside it can release/re-acquire around the
    // emit call.
    std::mutex              state_mutex_;
    std::deque<ScrollLine>  scrollback_;
    std::string             input_;
    int                     cursor_       = 0;
    int                     scroll_lines_ = 0;     // 0 = at bottom
    std::deque<std::string> history_;
    int                     history_pos_  = -1;

    // VS Code-style completion popup. Mirrors the web console's
    // `popupState` (see web/console.js) and reuses the shared scoring
    // engine in src/console/Completion.h. Auto-shows as the user
    // types; Up/Down navigate the highlight; Tab commits + chains
    // (so `r_bloom_<TAB>` lands the cvar name AND opens the
    // allowed-values popup); Enter/click commit without chaining;
    // Esc dismisses.
    //
    // popup_token_ captures the word-at-cursor at the moment of the
    // refresh -- including byte offsets within input_ so commit can
    // splice the candidate name in without re-parsing.
    struct PopupState {
        bool                                       active   = false;
        std::vector<pt::console::CompletionMatch>  items;
        int                                        selected = -1;
        pt::console::TokenInfo                     token;
        // Top-of-window index when items.size() exceeds the visible
        // row budget. Kept in sync with `selected` so the highlighted
        // row is always on screen.
        int                                        scroll_offset = 0;
    };
    PopupState popup_;

    // Parent-WndProc subclass for alt-tab focus restoration.  GLFW's
    // own WM_ACTIVATE handler calls SetFocus(parent) when the app
    // regains activation, which steals focus from our child overlay
    // and leaves the user unable to type into the open console.
    // ParentWndProcThunk re-grabs focus after GLFW has run, gated on
    // wants_focus_on_activate_ so a deliberate click into the parent
    // viewport (which sets the flag false) doesn't get overridden.
    WNDPROC original_parent_proc_     = nullptr;
    bool    wants_focus_on_activate_  = false;
};

std::atomic<WinOverlay*> WinOverlay::g{nullptr};

LRESULT CALLBACK WinOverlay::WndProcThunk(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(l);
        SetWindowLongPtrW(h, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<WinOverlay*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    if (self) return self->WndProc(h, m, w, l);
    return DefWindowProcW(h, m, w, l);
}

// Subclass of the parent (GLFW) window. Called for every message the
// parent receives. We forward to GLFW's original WndProc first, then
// post-process WM_ACTIVATE to redirect focus back to the overlay
// child if it's shown -- otherwise GLFW's SetFocus(parent) on
// activation leaves the overlay focusless after alt-tab.
//
// Routes through WinOverlay::g (the singleton instance set in Init)
// because GWLP_USERDATA on the parent belongs to GLFW.
LRESULT CALLBACK WinOverlay::ParentWndProcThunk(
    HWND h, UINT m, WPARAM w, LPARAM l) {
    WinOverlay* self = WinOverlay::g.load(std::memory_order_acquire);
    WNDPROC orig = (self != nullptr) ? self->original_parent_proc_ : nullptr;
    LRESULT r = (orig != nullptr)
        ? CallWindowProcW(orig, h, m, w, l)
        : DefWindowProcW(h, m, w, l);
    if (self != nullptr && self->hwnd_ && self->shown_) {
        // While the console is open we ALWAYS route keyboard focus to
        // the child -- mouse interactions with the parent viewport
        // (camera nav etc.) still work, but typing always goes to the
        // console input.  WM_ACTIVATE catches alt-tab return /
        // first-time app activation; WM_SETFOCUS catches every other
        // path that hands focus to the parent (e.g. some internal
        // GLFW step that fires once shortly after first show -- the
        // earlier `wants_focus_on_activate_` gate flipped permanently
        // to false on that, locking the user out of the console for
        // the rest of that show cycle).
        if ((m == WM_ACTIVATE && LOWORD(w) != WA_INACTIVE) ||
             m == WM_SETFOCUS) {
            SetFocus(self->hwnd_);
        }
    }
    return r;
}

bool WinOverlay::Init(HWND parent) {
    parent_ = parent;
    if (!parent_) {
        LOG_WARN("ConsoleOverlay_Win32: Init called with null parent HWND");
        return false;
    }

    static std::atomic<bool> registered{false};
    static const wchar_t* class_name = L"DemontConsoleOverlay";
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true)) {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(WNDCLASSEXW);
        wc.lpfnWndProc   = WndProcThunk;
        wc.hInstance     = GetModuleHandleW(nullptr);
        // IDC_IBEAM is MAKEINTRESOURCE(...) which expands to char*.
        // LoadCursorW wants wchar_t*, so cast through the generic
        // resource-id form.
        wc.hCursor       = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_IBEAM));
        wc.lpszClassName = class_name;
        ATOM atom = RegisterClassExW(&wc);
        if (atom == 0) {
            LOG_ERROR("ConsoleOverlay_Win32: RegisterClassExW failed (GLE={})",
                      GetLastError());
            registered.store(false);
            return false;
        }
    }

    RECT pr{};
    GetClientRect(parent_, &pr);
    parent_w_ = pr.right - pr.left;
    parent_h_ = pr.bottom - pr.top;
    panel_h_  = ComputePanelHeight(parent_h_);

    // Initially parked off-screen above the visible area; Show() slides
    // it down into view via WM_TIMER. Panel is anchored to the TOP of
    // the parent client rect, matching PtConsolePanel on Mac.
    int y = -panel_h_;

    SetLastError(0);
    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED,        // uniform-alpha translucency, see below
        class_name, L"",
        // WS_CLIPSIBLINGS prevents this window from painting into the
        // perf-overlay sibling's pixels (and vice versa) when both are
        // visible -- without it, the aggressive repaint pump on the
        // perf HUD leaves stale "copy" artifacts on the console.
        WS_CHILD | WS_CLIPSIBLINGS,
        0, y, parent_w_, panel_h_,
        parent_, nullptr,
        GetModuleHandleW(nullptr), this);
    DWORD gle = GetLastError();
    is_layered_ = (hwnd_ != nullptr);
    if (!hwnd_) {
        // Layered child windows are technically supported on Win8+ but
        // some configurations (especially with Vulkan parents) reject
        // the combination. Retry without WS_EX_LAYERED -- we lose
        // translucency but keep the overlay.
        LOG_WARN("ConsoleOverlay_Win32: layered child create failed (GLE={}), retrying opaque", gle);
        SetLastError(0);
        hwnd_ = CreateWindowExW(
            0, class_name, L"",
            WS_CHILD | WS_CLIPSIBLINGS,
            0, y, parent_w_, panel_h_,
            parent_, nullptr,
            GetModuleHandleW(nullptr), this);
        gle = GetLastError();
    }
    if (!hwnd_) {
        LOG_ERROR("ConsoleOverlay_Win32: CreateWindowExW failed (GLE={})", gle);
        return false;
    }
    LOG_INFO("ConsoleOverlay_Win32: child overlay attached to HWND={}",
             reinterpret_cast<void*>(parent_));

    // Confirm whether the create that succeeded was the layered one
    // (in case the EX style was stripped silently).
    is_layered_ = (GetWindowLongPtrW(hwnd_, GWL_EXSTYLE) & WS_EX_LAYERED) != 0;
    if (is_layered_) {
        // Start fully transparent; Show() ramps to kPanelAlpha.
        SetLayeredWindowAttributes(hwnd_, 0, 0, LWA_ALPHA);
    }

    font_ = CreateFontW(
        kFontHeight, 0, 0, 0, FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        FIXED_PITCH | FF_MODERN,
        L"Cascadia Mono");
    if (font_) {
        font_is_owned_ = true;
    } else {
        // CreateFontW failure -- e.g. Cascadia Mono not installed on
        // Windows Server SKUs. SYSTEM_FIXED_FONT always exists and is
        // owned by GDI; we MUST NOT DeleteObject it in Shutdown().
        font_          = (HFONT)GetStockObject(SYSTEM_FIXED_FONT);
        font_is_owned_ = false;
    }

    // Re-sync the cached scale invariant with the freshly-created
    // font. The font was just made at kFontHeight (the unscaled
    // default); font_scale_ MUST reflect that, otherwise EnsureFontScale
    // on the next Paint will compare cvar(e.g. 1.5) against a stale
    // font_scale_(1.5) carried over from a previous lifetime, decide
    // "no change needed", and leave the new font at the unscaled size.
    //
    // The bug only fires when Init runs a SECOND time on the same
    // WinOverlay object -- which is exactly what Engine::RecreateWindow
    // does on the vulkan->software gdi-mode HWND swap. On first init
    // font_scale_'s class-default (1.0f) already matches; on re-init
    // it carries whatever EnsureFontScale last applied. Reset both
    // here so the next Paint's EnsureFontScale always re-applies the
    // current cvar from a clean baseline.
    font_scale_   = 1.0f;
    line_height_  = kLineHeight;

    // Publish this instance with release semantics so log-thread loads
    // see fully-initialised state_mutex_, scrollback_, hwnd_, etc.
    g.store(this, std::memory_order_release);

    // Subclass the parent's WndProc so we can re-focus the overlay
    // when the application regains activation (alt-tab back). GLFW's
    // own WM_ACTIVATE path calls SetFocus(parent) which would
    // otherwise leave our child overlay focusless and unable to
    // receive keystrokes. Non-fatal if it fails -- the overlay still
    // works, just without the auto-restore.
    SetLastError(0);
    LONG_PTR prev_proc = SetWindowLongPtrW(parent_, GWLP_WNDPROC,
        reinterpret_cast<LONG_PTR>(ParentWndProcThunk));
    if (prev_proc == 0 && GetLastError() != 0) {
        LOG_WARN("ConsoleOverlay_Win32: parent WndProc subclass failed (GLE={})",
                 GetLastError());
    } else {
        original_parent_proc_ = reinterpret_cast<WNDPROC>(prev_proc);
    }

    // 13-line ASCII boot banner with role-tagged lines (frame /
    // letters / ray) so theme switching recolours them just like the
    // Mac overlay's NSView does. Same shape and characters as the
    // terminal banner in main.cpp::PrintBootLogo and the Cocoa
    // overlay's hex banner.
    scrollback_.push_back({"        \xe2\x96\x91\xe2\x96\x92\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x93\xe2\x96\x92\xe2\x96\x91", LineRole::LogoFrame});
    scrollback_.push_back({"     \xe2\x96\x91\xe2\x96\x92\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\xe2\x96\x93\xe2\x96\x92\xe2\x96\x91", LineRole::LogoFrame});
    scrollback_.push_back({"   \xe2\x96\x91\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x95\x94\xe2\x95\x90\xe2\x95\x9d   D \xc2\xb7 M \xc2\xb7 T   \xe2\x95\x9a\xe2\x95\x90\xe2\x95\x97\xe2\x96\x88\xe2\x96\x88\xe2\x96\x93\xe2\x96\x91", LineRole::LogoLetters});
    scrollback_.push_back({"  \xe2\x96\x92\xe2\x96\x88\xe2\x96\x91  \xe2\x95\x94\xe2\x95\x9d  \xe2\x95\xb2     \xe2\x97\x89     \xe2\x95\xb1  \xe2\x95\x9a\xe2\x95\x97  \xe2\x96\x91\xe2\x96\x88\xe2\x96\x92", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x93\xe2\x96\x88\xe2\x96\x91 \xe2\x95\x91    \xe2\x95\xb2   \xe2\x97\x89\xe2\x94\x82\xe2\x97\x89   \xe2\x95\xb1    \xe2\x95\x91 \xe2\x96\x91\xe2\x96\x88\xe2\x96\x93", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x88\xe2\x96\x91  \xe2\x95\x91     \xe2\x95\xb2  \xe2\x94\x80\xe2\x80\xa2\xe2\x94\x80  \xe2\x95\xb1     \xe2\x95\x91  \xe2\x96\x91\xe2\x96\x88", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x93\xe2\x96\x88\xe2\x96\x91 \xe2\x95\x91      \xe2\x95\xb3  \xe2\x80\xa2  \xe2\x95\xb3      \xe2\x95\x91 \xe2\x96\x91\xe2\x96\x88\xe2\x96\x93", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x88\xe2\x96\x91  \xe2\x95\x91     \xe2\x95\xb1  \xe2\x94\x80\xe2\x80\xa2\xe2\x94\x80  \xe2\x95\xb2     \xe2\x95\x91  \xe2\x96\x91\xe2\x96\x88", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x93\xe2\x96\x88\xe2\x96\x91 \xe2\x95\x91    \xe2\x95\xb1   \xe2\x97\x89\xe2\x94\x82\xe2\x97\x89   \xe2\x95\xb2    \xe2\x95\x91 \xe2\x96\x91\xe2\x96\x88\xe2\x96\x93", LineRole::LogoRay});
    scrollback_.push_back({"  \xe2\x96\x92\xe2\x96\x88\xe2\x96\x91  \xe2\x95\x9a\xe2\x95\x97  \xe2\x95\xb1     \xe2\x97\x89     \xe2\x95\xb2  \xe2\x95\x94\xe2\x95\x9d  \xe2\x96\x91\xe2\x96\x88\xe2\x96\x92", LineRole::LogoRay});
    scrollback_.push_back({"   \xe2\x96\x91\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x97   P \xc2\xb7 A \xc2\xb7 T   \xe2\x95\x94\xe2\x95\x90\xe2\x95\x9d\xe2\x96\x88\xe2\x96\x88\xe2\x96\x93\xe2\x96\x91", LineRole::LogoLetters});
    scrollback_.push_back({"     \xe2\x96\x91\xe2\x96\x92\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\xe2\x96\x93\xe2\x96\x92\xe2\x96\x91", LineRole::LogoFrame});
    scrollback_.push_back({"        \xe2\x96\x91\xe2\x96\x92\xe2\x96\x93\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x93\xe2\x96\x92\xe2\x96\x91", LineRole::LogoFrame});
    scrollback_.push_back({"", LineRole::Default});
    scrollback_.push_back({"  DeMonT Engine \xc2\xb7 v0.1.0  \xc2\xb7  non-rasterized \xc2\xb7 path-traced", LineRole::Default});
    scrollback_.push_back({"  console attached.  backtick toggles, PgUp/PgDn scrolls, Up/Down for history", LineRole::Default});
    scrollback_.push_back({"", LineRole::Default});

    return true;
}

void WinOverlay::Shutdown() {
    // Compare-and-clear under release ordering so any in-flight log
    // emit on a worker thread either gets the live `this` pointer
    // (and proceeds with the still-valid object up to the next g.load
    // observation) or sees nullptr and bails out at the OnLog guard.
    {
        WinOverlay* expected = this;
        g.compare_exchange_strong(expected, nullptr,
                                  std::memory_order_release,
                                  std::memory_order_relaxed);
    }
    // Restore the parent's original WndProc before tearing down.
    // If another subclass installed itself on top of ours, this will
    // remove it too -- acceptable given there's no other subclass
    // owner in this codebase. SetWindowSubclass (comctl32) would be
    // the correct general-purpose primitive but isn't linked here.
    if (parent_ && original_parent_proc_) {
        SetWindowLongPtrW(parent_, GWLP_WNDPROC,
            reinterpret_cast<LONG_PTR>(original_parent_proc_));
        original_parent_proc_ = nullptr;
    }
    if (hwnd_) {
        KillTimer(hwnd_, kAnimTimerId);
        KillTimer(hwnd_, kFocusGuardTimerId);
    }
    if (font_)  {
        // Stock GDI objects (GetStockObject) are owned by the OS --
        // DeleteObject on them is undefined; only delete fonts we
        // CreateFontW'd ourselves.
        if (font_is_owned_) DeleteObject(font_);
        font_          = nullptr;
        font_is_owned_ = false;
    }
    if (hwnd_)  { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    parent_     = nullptr;
    shown_      = false;
    anim_state_ = AnimState::Idle;
}

int WinOverlay::CurrentY() const {
    if (!hwnd_) return 0;
    RECT r{};
    GetWindowRect(hwnd_, &r);
    POINT p{r.left, r.top};
    ScreenToClient(parent_, &p);
    return p.y;
}

void WinOverlay::StartAnim(bool showing) {
    if (!hwnd_) return;
    panel_h_ = ComputePanelHeight(parent_h_);
    // Panel anchored to TOP. Off-screen rest position is one panel
    // height above the client area.
    int target_y = 0;
    int hidden_y = -panel_h_;

    int from_y;
    if (anim_state_ != AnimState::Idle) {
        from_y = CurrentY();   // smooth reverse mid-flight
    } else {
        from_y = showing ? hidden_y : target_y;
    }
    int to_y = showing ? target_y : hidden_y;

    anim_state_      = showing ? AnimState::Showing : AnimState::Hiding;
    anim_start_ms_   = GetTickCount();
    anim_dur_ms_     = showing ? kShowDurMs : kHideDurMs;
    anim_from_y_     = from_y;
    anim_to_y_       = to_y;
    anim_from_alpha_ = is_layered_ ? (showing ? BYTE(0) : kPanelAlpha) : 0;
    anim_to_alpha_   = is_layered_ ? (showing ? kPanelAlpha : BYTE(0)) : 0;

    SetWindowPos(hwnd_, nullptr, 0, from_y, parent_w_, panel_h_,
                 SWP_NOZORDER | SWP_NOACTIVATE);
    if (is_layered_) {
        SetLayeredWindowAttributes(hwnd_, 0, anim_from_alpha_, LWA_ALPHA);
    }

    if (showing) {
        ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
        // SetFocus directly + queue a deferred re-grab.  The direct
        // call usually works; the deferred one fires after the
        // current WM_KEYDOWN dispatch chain returns and re-asserts
        // focus in case GLFW's continued processing yanked it back.
        SetFocus(hwnd_);
        PostMessageW(hwnd_, WM_APP_SETFOCUS, 0, 0);
    }
    SetTimer(hwnd_, kAnimTimerId, kAnimTickMs, nullptr);
}

void WinOverlay::TickAnim() {
    if (anim_state_ == AnimState::Idle || !hwnd_) return;
    DWORD now = GetTickCount();
    DWORD elapsed = now - anim_start_ms_;
    float t = (anim_dur_ms_ > 0)
        ? std::min(1.0f, float(elapsed) / float(anim_dur_ms_))
        : 1.0f;
    // Cubic ease-out for show, ease-in for hide -- matches the
    // kCAMediaTimingFunctionEaseOut / EaseIn used on Mac.
    float eased = (anim_state_ == AnimState::Showing)
        ? 1.0f - std::pow(1.0f - t, 3.0f)
        : std::pow(t, 3.0f);

    int y = anim_from_y_ + int((anim_to_y_ - anim_from_y_) * eased);
    SetWindowPos(hwnd_, nullptr, 0, y, parent_w_, panel_h_,
                 SWP_NOZORDER | SWP_NOACTIVATE);
    if (is_layered_) {
        BYTE a = (BYTE)(anim_from_alpha_ +
                        int(int(anim_to_alpha_) - int(anim_from_alpha_)) * eased);
        SetLayeredWindowAttributes(hwnd_, 0, a, LWA_ALPHA);
    }

    if (t >= 1.0f) {
        KillTimer(hwnd_, kAnimTimerId);
        bool was_hiding = (anim_state_ == AnimState::Hiding);
        anim_state_ = AnimState::Idle;
        if (was_hiding) {
            ShowWindow(hwnd_, SW_HIDE);
            SetFocus(parent_);
        }
    }
}

void WinOverlay::Show() {
    if (!hwnd_) return;
    if (shown_ && anim_state_ != AnimState::Hiding) return;
    shown_ = true;
    // Newly shown -> we want the input field to be the focus target
    // when the app regains activation. A subsequent click into the
    // parent viewport will flip this back via WM_KILLFOCUS.
    wants_focus_on_activate_ = true;
    StartAnim(/*showing=*/true);
    // Arm the focus watchdog: every 100 ms while shown, re-check that
    // keyboard focus is on the child and restore it if not.
    if (hwnd_) SetTimer(hwnd_, kFocusGuardTimerId, kFocusGuardMs, nullptr);
}

void WinOverlay::Hide() {
    if (!hwnd_) return;
    if (!shown_ && anim_state_ != AnimState::Showing) return;
    shown_ = false;
    StartAnim(/*showing=*/false);
    if (hwnd_) KillTimer(hwnd_, kFocusGuardTimerId);
}

void WinOverlay::Toggle() { if (shown_) Hide(); else Show(); }

void WinOverlay::ApplyTheme(std::string_view name) {
    for (const auto& t : kThemes) {
        if (name == t.name) { theme_ = t; Repaint(); return; }
    }
}

void WinOverlay::NotifyParentResized(int w, int h) {
    if (parent_w_ == w && parent_h_ == h) return;
    parent_w_ = w; parent_h_ = h;
    if (!hwnd_) return;
    panel_h_ = ComputePanelHeight(h);

    // Refit the in-flight animation to the new client size; panel is
    // top-anchored so the visible-target Y is always 0 and the
    // off-screen rest is -panel_h_.
    if (anim_state_ == AnimState::Showing) {
        anim_to_y_ = 0;
    } else if (anim_state_ == AnimState::Hiding) {
        anim_from_y_ = 0;
        anim_to_y_   = -panel_h_;
    }

    int y;
    if (anim_state_ != AnimState::Idle) {
        // Mid-anim: keep current position, just match width/height.
        y = CurrentY();
    } else if (shown_) {
        y = 0;
    } else {
        y = -panel_h_;  // parked off-screen above
    }
    SetWindowPos(hwnd_, nullptr, 0, y, w, panel_h_,
                 SWP_NOZORDER | SWP_NOACTIVATE);
}

void WinOverlay::OnLog(pt::log::Level /*lvl*/, const std::string& msg) {
    // Snapshot the singleton with acquire ordering; if Shutdown has
    // already cleared it, OnLog becomes a no-op for the rest of this
    // process lifetime.  Re-using the snapshot below avoids a TOCTOU
    // hazard where g flips between an early null-check and a later
    // dereference.
    WinOverlay* self = g.load(std::memory_order_acquire);
    if (self == nullptr) return;
    // Strip ANSI escape sequences -- the log layer formats with them
    // for terminal use; GDI doesn't render them.
    std::string clean;
    clean.reserve(msg.size());
    bool in_esc = false;
    for (char c : msg) {
        if (c == '\x1b') { in_esc = true; continue; }
        if (in_esc) {
            if (c == 'm') in_esc = false;
            continue;
        }
        clean.push_back(c);
    }
    {
        std::lock_guard lk(self->state_mutex_);
        self->scrollback_.push_back({std::move(clean), LineRole::Default});
        while (self->scrollback_.size() > kScrollMax) self->scrollback_.pop_front();
    }
    // pt::log sinks fire on the calling thread (Log::Emit doesn't
    // reschedule), so OnLog can be invoked from any worker. Post a
    // custom repaint message instead of calling Repaint() / Invalidate
    // directly -- PostMessage is the only cross-thread Win32 UI bridge
    // that's well-defined.
    if (HWND target = self->hwnd_) {
        PostMessageW(target, WM_APP_REPAINT, 0, 0);
    }
}

void WinOverlay::Repaint() {
    if (hwnd_) InvalidateRect(hwnd_, nullptr, FALSE);
}

void WinOverlay::EnsureFontScale() {
    // Pull the cvar each call. Console::FindCVar is a flat-table
    // lookup, so the no-change path is O(1) name compare + a single
    // load -- cheaper than wiring up an on_change callback that
    // reaches across TUs from console -> app.
    auto& C = pt::console::Console::Get();
    auto* v = C.FindCVar("con_font_scale");
    if (v == nullptr) return;

    // The engine compiles with /EHs-c- (exceptions disabled) so std::stof
    // is off-limits. std::strtof never throws and signals "no chars
    // consumed" by returning end == start, which we treat as parse-fail.
    char* end = nullptr;
    float requested = std::strtof(v->value.c_str(), &end);
    // Reject "no digits" AND non-finite (NaN / +-Inf): strtof returns
    // a quiet NaN for "nan"/"NaN"/"NAN" and +-Inf for "inf"/"-inf"
    // without setting errno, and once a non-finite value reaches the
    // < / > clamps below they BOTH return false (NaN compares
    // unordered) -- so requested would stay non-finite, sneak past
    // the early-exit (NaN < 1e-3 is also false), and the
    // static_cast<int>(...) calls computing new_font_h / new_line_h
    // would be undefined behavior. Now with PR #10's con_font_scale
    // on_change -> Repaint -> Paint -> EnsureFontScale wiring,
    // typing `con_font_scale nan` in the console fires this path
    // immediately, so the guard matters even more than before.
    // Mirrors the same fix applied to PerfOverlay_Win32::EnsureScale.
    if (end == v->value.c_str() || !std::isfinite(requested)) requested = 1.0f;
    // Clamp to a sane range. Below 0.5 the prompt is unreadable;
    // above 3.0 the input row dominates the panel and history vanishes.
    if (requested < 0.5f) requested = 0.5f;
    if (requested > 3.0f) requested = 3.0f;

    // Treat tiny float deltas as no-op so a freshly-typed value that
    // round-trips through std::stof at slightly different precision
    // doesn't keep recreating the font.
    if (std::fabs(requested - font_scale_) < 1e-3f) return;

    const int new_font_h = std::max(6, static_cast<int>(kFontHeight * requested + 0.5f));
    const int new_line_h = std::max(8, static_cast<int>(kLineHeight * requested + 0.5f));

    HFONT new_font = CreateFontW(
        new_font_h, 0, 0, 0, FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        FIXED_PITCH | FF_MODERN,
        L"Cascadia Mono");
    if (new_font == nullptr) {
        LOG_WARN("ConsoleOverlay: CreateFontW failed at scale={:.2f} "
                 "(GLE={}); keeping previous font", requested, GetLastError());
        return;
    }

    // Swap then free the old font (must be in this order -- if the
    // window receives a paint between Delete and assign it'd reach a
    // dangling HFONT). Stock fonts are not owned and must not be
    // DeleteObject'd; mirror the ctor's font_is_owned_ tracking.
    HFONT      old_font  = font_;
    const bool old_owned = font_is_owned_;
    font_          = new_font;
    font_is_owned_ = true;
    font_scale_    = requested;
    line_height_   = new_line_h;
    if (old_font != nullptr && old_owned) DeleteObject(old_font);
    LOG_INFO("ConsoleOverlay: con_font_scale={:.2f} -> font_h={} line_h={}",
             requested, new_font_h, new_line_h);
    Repaint();
}

// --- State persistence (history + scrollback) -----------------------------
//
// File format (plain text, line-oriented, UTF-8):
//
//   DEMONT_CONSOLE_STATE v1
//   HISTORY <N>
//   <history line 1>
//   ...
//   <history line N>
//   SCROLLBACK <M>
//   <role>\t<text>
//   ...
//
// where <role> is a digit 0..4 mapping to LineRole::{Default,LogoFrame,
// LogoLetters,LogoRay,Echo}. Banner roles (LogoFrame/LogoLetters/LogoRay)
// are SKIPPED on save -- the next Init regenerates them, so saving them
// would just produce duplicates. We use a TAB to separate role from
// text so embedded spaces in log lines round-trip cleanly; embedded
// newlines aren't legal in scrollback entries (they're split into
// separate entries on push) so single-newline-per-record is safe.
//
// SaveState writes to <path>.tmp and renames; an in-flight crash leaves
// the previous good file intact. LoadState appends after whatever the
// current Init's banner already put in scrollback, with a "previous
// session" separator -- the user can tell where the restored block
// starts.

bool WinOverlay::SaveState(const std::string& path) const {
    // Snapshot under the state mutex. We must NOT hold the mutex while
    // doing file I/O -- the log path can call OnLog re-entrantly via
    // any LOG_* inside fwrite/fclose's failure handling, and OnLog
    // tries to acquire the same mutex. Copy out, release, then write.
    std::deque<std::string> hist_copy;
    std::deque<ScrollLine>  scroll_copy;
    {
        // const_cast around state_mutex_: the mutex is logically
        // mutable (acquired for read-only access here), but we promised
        // a `const`-qualified SaveState in the public API to keep
        // call sites tidy.
        auto& self = const_cast<WinOverlay&>(*this);
        std::lock_guard lock(self.state_mutex_);
        hist_copy   = history_;
        scroll_copy = scrollback_;
    }

    const std::string tmp_path = path + ".tmp";
    std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (f == nullptr) {
        LOG_WARN("ConsoleOverlay::SaveState: fopen('{}', 'wb') failed",
                 tmp_path);
        return false;
    }

    // Count non-banner scrollback entries up-front so the header is
    // accurate (a single pass over scroll_copy filters banner roles).
    std::size_t scrollback_count = 0;
    for (const auto& s : scroll_copy) {
        if (s.role == LineRole::LogoFrame ||
            s.role == LineRole::LogoLetters ||
            s.role == LineRole::LogoRay) continue;
        ++scrollback_count;
    }

    fmt::print(f, "DEMONT_CONSOLE_STATE v1\n");
    fmt::print(f, "HISTORY {}\n", hist_copy.size());
    for (const auto& h : hist_copy) {
        // Strip any embedded newlines defensively -- the line-oriented
        // format would otherwise split one history entry into two.
        std::string clean = h;
        std::replace(clean.begin(), clean.end(), '\n', ' ');
        std::replace(clean.begin(), clean.end(), '\r', ' ');
        fmt::print(f, "{}\n", clean);
    }
    fmt::print(f, "SCROLLBACK {}\n", scrollback_count);
    for (const auto& s : scroll_copy) {
        if (s.role == LineRole::LogoFrame ||
            s.role == LineRole::LogoLetters ||
            s.role == LineRole::LogoRay) continue;
        std::string clean = s.text;
        std::replace(clean.begin(), clean.end(), '\n', ' ');
        std::replace(clean.begin(), clean.end(), '\r', ' ');
        fmt::print(f, "{}\t{}\n",
                   static_cast<int>(s.role), clean);
    }
    std::fclose(f);

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        LOG_WARN("ConsoleOverlay::SaveState: rename('{}' -> '{}') failed: {}",
                 tmp_path, path, ec.message());
        // Leave the .tmp file behind so the next save can overwrite it;
        // removing here would race with antivirus scanners that hold
        // open handles transiently on newly-created files.
        return false;
    }
    LOG_INFO("ConsoleOverlay::SaveState: wrote {} history + {} scrollback entries to '{}'",
             hist_copy.size(), scrollback_count, path);
    return true;
}

bool WinOverlay::LoadState(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        // Missing file is not an error: first launch ever, or user
        // deleted the file to start fresh.
        return false;
    }

    std::string line;
    if (!std::getline(in, line) ||
        line.rfind("DEMONT_CONSOLE_STATE v1", 0) != 0) {
        LOG_WARN("ConsoleOverlay::LoadState: '{}' missing v1 header (got '{}'); skipping",
                 path, line);
        return false;
    }

    std::deque<std::string> hist_loaded;
    std::deque<ScrollLine>  scroll_loaded;

    // Section parser: small + paranoid. Unknown sections / malformed
    // lines are logged and skipped rather than aborting the whole load
    // (a partial restore beats no restore).
    while (std::getline(in, line)) {
        if (line.rfind("HISTORY ", 0) == 0) {
            std::size_t n = static_cast<std::size_t>(std::atoi(line.c_str() + 8));
            for (std::size_t i = 0; i < n; ++i) {
                if (!std::getline(in, line)) break;
                hist_loaded.push_back(std::move(line));
            }
            // Clamp to the same bound the live deque enforces so a
            // tampered file can't blow memory.
            while (hist_loaded.size() > kHistoryMax) hist_loaded.pop_front();
        } else if (line.rfind("SCROLLBACK ", 0) == 0) {
            std::size_t n = static_cast<std::size_t>(std::atoi(line.c_str() + 11));
            for (std::size_t i = 0; i < n; ++i) {
                if (!std::getline(in, line)) break;
                auto tab = line.find('\t');
                if (tab == std::string::npos || tab == 0) {
                    // Malformed entry; skip rather than carry partial.
                    continue;
                }
                int role_n = std::atoi(line.substr(0, tab).c_str());
                if (role_n < 0 ||
                    role_n > static_cast<int>(LineRole::Echo)) {
                    role_n = static_cast<int>(LineRole::Default);
                }
                scroll_loaded.push_back({
                    line.substr(tab + 1),
                    static_cast<LineRole>(role_n)
                });
            }
            while (scroll_loaded.size() > kScrollMax) scroll_loaded.pop_front();
        }
        // Other lines (blank, unrecognised) are just skipped.
    }

    // Capture sizes for the post-lock log line so we don't read the
    // shared deques after releasing the mutex (a worker-thread OnLog
    // could legally mutate scrollback_ between the unlock and the
    // log emit).
    const std::size_t hist_n   = hist_loaded.size();
    const std::size_t scroll_n = scroll_loaded.size();

    // Splice into the live state under the mutex. The current
    // scrollback_ already has Init's freshly-pushed banner; we append
    // a separator, then the loaded scrollback after it. History is
    // empty on a fresh Init so we just adopt the loaded deque.
    {
        std::lock_guard lock(state_mutex_);
        history_ = std::move(hist_loaded);
        history_pos_ = -1;  // up-arrow walks from the most recent

        if (!scroll_loaded.empty()) {
            scrollback_.push_back({"", LineRole::Default});
            scrollback_.push_back(
                {"---- previous session " + std::string(40, '-'), LineRole::Default});
            scrollback_.push_back({"", LineRole::Default});
            for (auto& sl : scroll_loaded) {
                scrollback_.push_back(std::move(sl));
                while (scrollback_.size() > kScrollMax) scrollback_.pop_front();
            }
        }
    }

    LOG_INFO("ConsoleOverlay::LoadState: restored {} history + {} scrollback entries from '{}'",
             hist_n, scroll_n, path);
    Repaint();
    return true;
}

LRESULT WinOverlay::WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    // Set by WM_KEYDOWN's Ctrl+Space branch so the WM_CHAR that
    // TranslateMessage queues for the SAME keystroke can be consumed;
    // otherwise WM_CHAR(' ') would insert a stray space at the cursor
    // and call RefreshCompletions again on top of the popup we just
    // summoned. Function-scope static (not lambda-static) so both the
    // WM_KEYDOWN setter and the WM_CHAR consumer can see it. atomic
    // is overkill on a single-threaded message pump but cheap, and
    // documents the cross-message-handler hand-off.
    static std::atomic<bool> s_suppress_next_wm_char{false};

    switch (m) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        Paint(dc);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;  // Background painted in WM_PAINT to avoid flicker
    case WM_APP_REPAINT:
        // Cross-thread repaint request (see OnLog). InvalidateRect runs
        // on the UI thread now -- safe per Win32.
        InvalidateRect(h, nullptr, FALSE);
        return 0;
    case WM_APP_SETFOCUS:
        // Re-grab focus after the original WM_KEYDOWN dispatch chain
        // has unwound (see WM_APP_SETFOCUS doc comment near declaration).
        // Only act if the overlay is still meant to be visible.
        if (shown_) SetFocus(h);
        return 0;
    case WM_TIMER:
        if (w == kAnimTimerId) { TickAnim(); return 0; }
        if (w == kFocusGuardTimerId) {
            // Re-assert focus on the child if it has drifted while
            // the console is open.  Cheap (one GetFocus call) and
            // self-correcting.  Unconditional while shown -- earlier
            // gating on wants_focus_on_activate_ permanently disabled
            // the watchdog after the first stray focus shift, which
            // is the exact "first-open-only" bug the user reported.
            if (shown_ && hwnd_) {
                if (GetFocus() != hwnd_) {
                    SetFocus(hwnd_);
                }
            }
            return 0;
        }
        break;
    case WM_MOUSEWHEEL: {
        // Mouse-wheel scrollback. Positive delta = wheel rotated
        // forward (away from user) = scroll up = look at older
        // content. WHEEL_DELTA (120) is one notch on a stock wheel;
        // 3 lines per notch matches Windows' default text-control
        // step. scroll_lines_ counts entries pushed up off the
        // bottom, capped at scrollback_.size()-1.
        int delta = GET_WHEEL_DELTA_WPARAM(w);
        int lines = (delta * 3) / WHEEL_DELTA;
        if (lines == 0) lines = (delta > 0) ? 1 : -1;
        std::lock_guard lk(state_mutex_);
        int n = static_cast<int>(scrollback_.size());
        scroll_lines_ = std::clamp(scroll_lines_ + lines, 0,
                                   std::max(0, n - 1));
        Repaint();
        return 0;
    }
    case WM_KILLFOCUS: {
        // Distinguish "user clicked into the game viewport" (focus
        // moves to parent_) from "app is deactivating" (focus moves
        // to NULL or another process). The former is a deliberate
        // step away from the console; the latter is alt-tab/minimize
        // and we should reclaim focus when the app comes back.
        HWND new_focus = reinterpret_cast<HWND>(w);
        wants_focus_on_activate_ = (new_focus != parent_);
        break;
    }
    case WM_CHAR: {
        // Ctrl+Space (handled in WM_KEYDOWN above) sets this flag so
        // we swallow the WM_CHAR that TranslateMessage queues for the
        // SAME keystroke. On US/most layouts ToUnicode(VK_SPACE,
        // ctrl-held) returns 0x20 (space), and without this guard the
        // WM_CHAR(' ') branch below would insert a stray space at the
        // cursor and call RefreshCompletions again -- visible to the
        // user as "Ctrl+Space silently inserts a space and the popup
        // I just summoned blinks / doesn't show stably." See WM_KEYDOWN
        // Ctrl+Space block for the matching set.
        if (s_suppress_next_wm_char.exchange(false)) {
            return 0;
        }
        // Backtick: toggle off (matches the engine's Mac path where
        // backtick on the GLFW window opens the overlay; once open
        // the Cocoa NSView's keyDown closes it).
        if (w == '`') { Hide(); return 0; }
        if (w == '\r') { SubmitInput(); return 0; }
        if (w == '\b') {
            // Hold the mutex across BOTH the input mutation AND the
            // RefreshCompletions call -- RefreshCompletions reads
            // input_ / cursor_ and writes popup_, so its caller must
            // hold state_mutex_ (matching the WM_KEYDOWN call sites).
            // Earlier rev released the lock between mutate + refresh,
            // creating a window where another thread could observe a
            // half-updated state. Repaint() is just InvalidateRect,
            // safe to call after the lock drops.
            {
                std::lock_guard lk(state_mutex_);
                if (cursor_ > 0) { input_.erase(cursor_ - 1, 1); cursor_--; }
                // Re-rank completions on every text mutation --
                // backspace narrows / widens the matching set
                // depending on whether the char being removed was
                // matching anywhere.
                RefreshCompletions(/*force_show=*/false);
            }
            Repaint();
            return 0;
        }
        if (w >= 32 && w < 127) {
            // Same locking rationale as the WM_BACK branch above:
            // RefreshCompletions touches input_ / cursor_ / popup_
            // and is contractually called with state_mutex_ held.
            {
                std::lock_guard lk(state_mutex_);
                input_.insert(cursor_, 1, static_cast<char>(w));
                cursor_++;
                // After every text mutation, re-rank completions
                // against the new cursor context. The popup auto-
                // shows when there are matches and the typed token
                // is non-empty (or the user has just hit `<name> ` --
                // ForceShow=true lets the value-position popup open
                // without a query). Mirrors the web console's `input`
                // event handler.
                const bool typed_separator = (w == ' ');
                RefreshCompletions(/*force_show=*/typed_separator);
            }
            Repaint();
            return 0;
        }
        return 0;
    }
    case WM_KEYDOWN: {
        // unique_lock (not lock_guard) so the Ctrl+V failure path and
        // the Ctrl+Space first-fire diagnostic can drop the mutex
        // before calling LOG_*; see the comment on state_mutex_'s
        // declaration for why holding it across a log emit fast-fails.
        std::unique_lock<std::mutex> lk(state_mutex_);

        // Modifier keys alone (Shift/Ctrl/Alt/Win) must not dismiss
        // the ghost -- otherwise a user pressing Shift+Tab fires
        // WM_KEYDOWN(VK_SHIFT) first, which would trip the
        // "any-other-key dismisses" branch below; by the time Tab
        // arrives the ghost is gone and Shift+Tab degenerates into a
        // fresh first Tab (cycle forward) instead of a step back.
        // Same for Caps Lock etc.  Just swallow modifier-only events.
        if (w == VK_SHIFT   || w == VK_CONTROL || w == VK_MENU    ||
            w == VK_LSHIFT  || w == VK_RSHIFT  ||
            w == VK_LCONTROL|| w == VK_RCONTROL||
            w == VK_LMENU   || w == VK_RMENU   ||
            w == VK_CAPITAL || w == VK_LWIN    || w == VK_RWIN) {
            return 0;
        }

        // Ctrl+V: paste from clipboard at cursor. Inline (rather than
        // a helper) because state_mutex_ is already held by the
        // lock_guard above; an outline call to a method that re-locked
        // would deadlock. CF_UNICODETEXT is preferred (proper
        // non-ASCII handling); fall back to CF_TEXT if absent. Multi-
        // line clipboard content is truncated at the first newline --
        // the overlay is one-line-at-a-time, and silently submitting
        // a chain of pasted commands would surprise the user.
        const bool ctrl_held = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
        if (ctrl_held && (w == 'V' || w == 'v')) {
            std::string text;
            if (OpenClipboard(hwnd_)) {
                if (HANDLE h = GetClipboardData(CF_UNICODETEXT); h != nullptr) {
                    if (auto* src = static_cast<const wchar_t*>(GlobalLock(h))) {
                        // n includes the trailing NUL because cchWideChar=-1
                        // signals "input is null-terminated, count it". The
                        // destination buffer therefore must be sized for n
                        // bytes, not n-1 -- we resize the std::string to n,
                        // write into all n bytes, then pop_back the trailing
                        // NUL so size() == strlen(). Earlier we sized to n-1
                        // and passed n as the destination capacity, which is
                        // a one-byte overflow into whatever sits past
                        // text.data()+size() (typically the std::string's
                        // own internal NUL slot, but UB to rely on).
                        const int n = WideCharToMultiByte(CP_UTF8, 0, src, -1,
                                                          nullptr, 0, nullptr, nullptr);
                        if (n > 1) {
                            text.resize(static_cast<std::size_t>(n));
                            WideCharToMultiByte(CP_UTF8, 0, src, -1,
                                                text.data(), n, nullptr, nullptr);
                            text.pop_back();  // drop trailing NUL
                        }
                        GlobalUnlock(h);
                    }
                } else if (HANDLE h = GetClipboardData(CF_TEXT); h != nullptr) {
                    if (auto* src = static_cast<const char*>(GlobalLock(h))) {
                        text.assign(src);
                        GlobalUnlock(h);
                    }
                }
                CloseClipboard();
            } else {
                // OpenClipboard failed -- log it, but only after
                // dropping state_mutex_ because LOG_WARN -> OnLog
                // re-locks it on the same thread (would fast-fail).
                const DWORD gle = GetLastError();
                lk.unlock();
                LOG_WARN("ConsoleOverlay: OpenClipboard failed (GLE={})", gle);
                lk.lock();
            }

            if (!text.empty()) {
                if (auto pos = text.find_first_of("\r\n"); pos != std::string::npos) {
                    text.resize(pos);
                }
                // Strip ASCII control chars except \t (preserve UTF-8
                // high-byte sequences -- 0x80+ unsigned).
                text.erase(std::remove_if(text.begin(), text.end(),
                    [](char c) {
                        return static_cast<unsigned char>(c) < 32 && c != '\t';
                    }), text.end());
                if (!text.empty()) {
                    input_.insert(cursor_, text);
                    cursor_ += static_cast<int>(text.size());
                    history_pos_ = -1;
                    Repaint();
                }
            }
            return 0;
        }

        // Ctrl+Space -- universal "show me my options" shortcut.
        // Matches VS Code / IntelliJ / every IDE. Force-opens the
        // popup regardless of current state so the user can summon
        // completions on demand even after dismissing them by
        // typing or clicking elsewhere. Has to run before the
        // popup-active "any other key dismisses" branch, otherwise
        // the dismissal would beat the open. Swallowing the
        // keystroke (`return 0`) prevents WM_CHAR from generating
        // a stray space character.
        //
        // CAVEATS: on Windows with Chinese / Japanese / Korean IME
        // enabled, Ctrl+Space is reserved by the OS for toggling
        // the IME and never reaches this handler. There's nothing
        // we can do about that short of binding a different shortcut
        // -- if a user reports it not working, suggest checking
        // Settings > Time & Language > Advanced keyboard settings
        // for any IME toggle bindings.
        //
        // The first-fire log is a one-shot diagnostic so users can
        // confirm the handler is being reached (e.g. by tailing the
        // engine log while pressing the shortcut).
        if (ctrl_held && w == VK_SPACE) {
            // Tell WM_CHAR to swallow the space that TranslateMessage
            // queued for THIS keystroke (US/most layouts map Ctrl+Space
            // to 0x20). Otherwise it'd insert a stray space at the
            // cursor and rebuild the popup on top of itself.
            s_suppress_next_wm_char.store(true);
            static std::atomic<bool> s_logged_once{false};
            if (!s_logged_once.exchange(true)) {
                // Snapshot then drop the lock for the emit: LOG_INFO
                // fans out to ConsoleOverlay::OnLog which re-locks
                // state_mutex_ (would fast-fail under std::mutex's
                // non-recursive contract + our /EHs-c- build).
                std::string snap_input  = input_;
                int         snap_cursor = cursor_;
                lk.unlock();
                LOG_INFO("ConsoleOverlay: Ctrl+Space force-show fired "
                         "(input='{}' cursor={})",
                         snap_input, snap_cursor);
                lk.lock();
            }
            RefreshCompletions(/*force_show=*/true);
            Repaint();
            return 0;
        }

        // Popup-active key handling. Up/Down navigate the highlight
        // (NOT command history); Tab commits + chains; Enter commits
        // without chaining (and falls through to the WM_CHAR '\r'
        // generated by TranslateMessage for the SAME Enter keystroke,
        // which runs SubmitInput on the just-completed line -- see the
        // VK_RETURN block below for the message-loop mechanics); End
        // and Right-arrow-at-EOL commit WITHOUT chaining (only Tab
        // chains -- matches VS Code and the web console's
        // popupCommit(chainNext=false) on the same keys); Esc
        // dismisses; any other key dismisses the popup and falls
        // through -- the subsequent WM_CHAR / repaint will re-rank
        // via the input handler, so the popup re-opens if the new
        // typed char still leaves us inside a completable token.
        if (popup_.active) {
            if (w == VK_DOWN)  { MovePopupSelection(+1); return 0; }
            if (w == VK_UP)    { MovePopupSelection(-1); return 0; }
            if (w == VK_TAB)   { CommitPopup(/*chain_next=*/true); return 0; }
            if (w == VK_ESCAPE){ DismissPopup(); return 0; }
            if (w == VK_END ||
                (w == VK_RIGHT && cursor_ == static_cast<int>(input_.size()))) {
                // chain_next=false: End / Right are "accept this row
                // and stop" gestures, mirroring the web console. Only
                // Tab chains (= "complete + show next suggestions").
                CommitPopup(/*chain_next=*/false);
                return 0;
            }
            if (w == VK_RETURN) {
                CommitPopup(/*chain_next=*/false);
                // Returning 0 here doesn't stop WM_CHAR from also
                // firing for the SAME Enter keystroke (TranslateMessage
                // in the message loop has already queued a WM_CHAR
                // '\r' by this point -- WM_KEYDOWN's return code only
                // gates DefWindowProc, not the WM_CHAR translation).
                // So after this return, WM_CHAR '\r' arrives and runs
                // SubmitInput() on the just-completed line. Matches
                // the prior ghost-era behaviour for typed Enter -- the
                // user's mental model on a console is "Enter = run",
                // so committing the highlighted match and then
                // submitting it is the least-surprise path.
                return 0;
            }
            // Any other key: tear the popup down and let the keystroke
            // route through the normal handlers below (so e.g. Left
            // arrow still moves the cursor, backspace still deletes).
            DismissPopup();
            // fall through
        }

        switch (w) {
        case VK_TAB:
            // Handled here (not WM_CHAR) so DefWindowProcW never sees
            // it -- WS_CHILD + VK_TAB otherwise triggers focus
            // traversal in the parent's tab order. Tab with popup
            // hidden FORCE-SHOWS the popup (even for an empty query),
            // mirroring the web console's behaviour where Tab on
            // `r_bloom_intensity ` opens the value-position popup.
            RefreshCompletions(/*force_show=*/true);
            Repaint();
            return 0;
        // Cursor-move keys also re-evaluate the popup against the
        // new caret position. Without this, navigating into the
        // trailing space of `r_denoiser ` via Home + End leaves the
        // popup stale -- the user would have to type a char to get
        // the value-position popup to open. Refreshing here keeps
        // the popup in sync with whatever word the cursor is on.
        case VK_LEFT:
            if (cursor_ > 0) cursor_--;
            RefreshCompletions(/*force_show=*/false);
            Repaint();
            return 0;
        case VK_RIGHT:
            if (cursor_ < (int)input_.size()) cursor_++;
            RefreshCompletions(/*force_show=*/false);
            Repaint();
            return 0;
        case VK_HOME:
            cursor_ = 0;
            RefreshCompletions(/*force_show=*/false);
            Repaint();
            return 0;
        case VK_END:
            cursor_ = (int)input_.size();
            RefreshCompletions(/*force_show=*/false);
            Repaint();
            return 0;
        case VK_DELETE:
            if (cursor_ < (int)input_.size()) {
                input_.erase(cursor_, 1); Repaint();
            }
            return 0;
        case VK_PRIOR: {
            int n = (int)scrollback_.size();
            scroll_lines_ = std::min(scroll_lines_ + 8, std::max(0, n - 1));
            Repaint();
            return 0;
        }
        case VK_NEXT:
            scroll_lines_ = std::max(0, scroll_lines_ - 8);
            Repaint();
            return 0;
        case VK_UP:
            if (history_pos_ + 1 < (int)history_.size()) {
                history_pos_++;
                input_ = history_[history_.size() - 1 - history_pos_];
                cursor_ = (int)input_.size();
                Repaint();
            }
            return 0;
        case VK_DOWN:
            if (history_pos_ > 0) {
                history_pos_--;
                input_ = history_[history_.size() - 1 - history_pos_];
                cursor_ = (int)input_.size();
            } else if (history_pos_ == 0) {
                history_pos_ = -1;
                input_.clear();
                cursor_ = 0;
            }
            Repaint();
            return 0;
        case VK_ESCAPE:
            // Two-stage Esc:
            //   1st press, input non-empty -> clear the line (matches
            //                                 fish/zsh/Bash readline +
            //                                 most rich console UIs)
            //   2nd press (input empty)    -> close the overlay
            //                                 (common console-game
            //                                 pattern, second way out
            //                                 besides backtick)
            // Hide() acquires state_mutex_ via its own path so it
            // must run AFTER this lock_guard releases at function exit.
            if (!input_.empty()) {
                input_.clear();
                cursor_ = 0;
                history_pos_ = -1;
                Repaint();
                return 0;
            }
            break;
        }
        if (w == VK_ESCAPE) { Hide(); return 0; }
        return 0;
    }
    }
    return DefWindowProcW(h, m, w, l);
}

void WinOverlay::SubmitInput() {
    std::string line;
    {
        std::lock_guard lk(state_mutex_);
        if (input_.empty()) { Repaint(); return; }
        line = input_;
        scrollback_.push_back({"> " + line, LineRole::Echo});
        while (scrollback_.size() > kScrollMax) scrollback_.pop_front();
        if (history_.empty() || history_.back() != line) {
            history_.push_back(line);
            while (history_.size() > kHistoryMax) history_.pop_front();
        }
        history_pos_ = -1;
        input_.clear();
        cursor_ = 0;
        scroll_lines_ = 0;
    }

    // Console::QueueExecute is thread-safe; the responder fires from
    // the engine main thread when Drain() runs.
    pt::console::Console::Get().QueueExecute(line,
        [this](const pt::console::ExecuteResult& r) {
            const std::string& out = r.ok ? r.output : r.error;
            if (out.empty()) { Repaint(); return; }
            std::lock_guard lk(state_mutex_);
            std::size_t i = 0;
            while (i < out.size()) {
                std::size_t end = out.find('\n', i);
                if (end == std::string::npos) end = out.size();
                scrollback_.push_back({out.substr(i, end - i), LineRole::Default});
                i = end + 1;
            }
            while (scrollback_.size() > kScrollMax) scrollback_.pop_front();
            Repaint();
        });

    Repaint();
}

// Rebuild the popup against the current input_ + cursor_ context.
// Caller (WM_KEYDOWN / WM_CHAR / paste / WM_CHAR backspace etc.)
// holds state_mutex_ when changes are recent, so direct reads of
// input_ / cursor_ here are safe. pt::console::BuildCompletions
// runs on the engine main thread (same thread that owns the cvar
// map) -- no Console-side locking needed.
//
// `force_show` mirrors the web console: when true (Tab + post-space
// triggers) the popup opens even for an empty query, so the user
// can browse value-position candidates without typing anything; when
// false (normal char input) the popup only shows when there's a
// non-empty query AND at least one match.
void WinOverlay::RefreshCompletions(bool force_show) {
    const std::size_t cur = static_cast<std::size_t>(std::max(0, cursor_));
    pt::console::TokenInfo token =
        pt::console::CurrentToken(input_, cur);

    // Empty-query gating:
    //   - Token 0 with empty text: don't auto-open. The popup would
    //     list every cvar + command in the engine every time the
    //     input becomes empty (e.g. fresh open, post-Submit clear) --
    //     that's noise. Token 0 needs at least one typed char.
    //   - Value position (is_token0 == false) with empty text: DO
    //     auto-open. This is the "user just typed `<cvar> ` and
    //     wants to see the value list (incl. current value)" path.
    //     Mirrors the web console's equivalent gate.
    if (!force_show && token.text.empty() && token.is_token0) {
        DismissPopup();
        return;
    }

    auto items = pt::console::BuildCompletions(token, /*max_results=*/60,
                                               /*description_clip=*/120);
    if (items.empty()) {
        DismissPopup();
        return;
    }

    popup_.active        = true;
    popup_.items         = std::move(items);
    popup_.selected      = 0;
    popup_.token         = std::move(token);
    popup_.scroll_offset = 0;
}

int WinOverlay::PopupVisibleRows() const {
    constexpr int kPopupMaxRows = 10;
    const int n = static_cast<int>(popup_.items.size());
    if (n == 0 || line_height_ <= 0) return 0;
    if (!hwnd_) return std::min(n, kPopupMaxRows);
    RECT rc{};
    GetClientRect(hwnd_, &rc);
    const int H        = rc.bottom;
    const int input_y  = H - line_height_ - kPaddingY;
    const int log_top  = kPaddingY + 1 + 24;
    // Available vertical room between log_top + 2 and input_y - 6
    // (matches the popup_y offset Paint uses). The -8 in the row
    // calc is the popup's 4 px internal padding top + bottom.
    const int available_h = (input_y - 6) - (log_top + 2);
    if (available_h < line_height_ + 8) return 0;
    const int by_height = std::max(1, (available_h - 8) / line_height_);
    return std::min({n, kPopupMaxRows, by_height});
}

void WinOverlay::MovePopupSelection(int dir) {
    if (!popup_.active || popup_.items.empty()) return;
    const int n = static_cast<int>(popup_.items.size());
    popup_.selected = ((popup_.selected + dir) % n + n) % n;
    // Keep the highlighted row in the visible window. `visible` is
    // the actual paint-able row count for the current panel size
    // (shared with Paint via PopupVisibleRows), so when the panel
    // is short and Paint can only fit e.g. 4 rows, Up/Down scrolls
    // the window through the rest instead of letting selection run
    // off-screen.
    const int visible = PopupVisibleRows();
    if (visible > 0) {
        if (popup_.selected < popup_.scroll_offset) {
            popup_.scroll_offset = popup_.selected;
        } else if (popup_.selected >= popup_.scroll_offset + visible) {
            popup_.scroll_offset = popup_.selected - visible + 1;
        }
    }
    Repaint();
}

// Commit the currently-highlighted candidate by replacing the word
// at the cursor with the candidate name (+ trailing space when we're
// completing token 0 so the user lands at the value position ready
// to keep typing). `chain_next` controls whether we auto-reopen the
// popup at the new cursor position after commit:
//   - Tab commits chain (Tab = "complete + show next suggestions")
//   - Enter / click commits DON'T chain (those are "accept + stop"
//     gestures; chaining would loop the popup forever on repeated
//     Enter)
// Mirrors the web console's popupCommit(chainNext) semantics.
void WinOverlay::CommitPopup(bool chain_next) {
    if (!popup_.active ||
        popup_.selected < 0 ||
        popup_.selected >= static_cast<int>(popup_.items.size())) {
        DismissPopup();
        return;
    }
    const pt::console::CompletionMatch& it = popup_.items[popup_.selected];
    const pt::console::TokenInfo&        t = popup_.token;
    std::string tail = it.name + (t.is_token0 ? std::string(" ") : std::string());
    input_  = input_.substr(0, t.start) + tail + input_.substr(t.end);
    cursor_ = static_cast<int>(t.start + tail.size());
    const bool was_token0 = t.is_token0;
    DismissPopup();
    if (chain_next && was_token0) {
        RefreshCompletions(/*force_show=*/true);
        Repaint();
    }
}

void WinOverlay::DismissPopup() {
    popup_.active   = false;
    popup_.items.clear();
    popup_.selected = -1;
    popup_.token    = pt::console::TokenInfo{};
    popup_.scroll_offset = 0;
    Repaint();
}

void WinOverlay::Paint(HDC dc) {
    // Re-pull con_font_scale every paint. Cheap: zero work when the
    // value hasn't changed; on change we delete + recreate the GDI
    // HFONT and recompute line_height_, then proceed to draw at the
    // new size. Polled here (not via cvar on_change) because the
    // overlay TU has no clean dependency hook into Console -- and a
    // per-frame compare-against-cached is < 1 us.
    EnsureFontScale();

    RECT rc{};
    GetClientRect(hwnd_, &rc);
    int W = rc.right;
    int H = rc.bottom;

    // Double-buffer to a memory DC so we don't flicker on every WM_PAINT.
    HDC mdc = CreateCompatibleDC(dc);
    HBITMAP bmp = CreateCompatibleBitmap(dc, W, H);
    HBITMAP old_bmp = (HBITMAP)SelectObject(mdc, bmp);

    HBRUSH bg = CreateSolidBrush(theme_.panel);
    FillRect(mdc, &rc, bg);
    DeleteObject(bg);

    // Top accent line, 1 px.
    HBRUSH acc = CreateSolidBrush(theme_.accent);
    RECT line = {0, 0, W, 1};
    FillRect(mdc, &line, acc);
    DeleteObject(acc);

    HFONT old_font = (HFONT)SelectObject(mdc, font_);
    SetBkMode(mdc, TRANSPARENT);

    // ---- Top-left hex logo glyph (mirrors PtLogoView in the Cocoa
    // overlay). Hex frame in accent, three-bounce ray + 2 dots in
    // logo_ray. Same hex+ray shape as the ASCII banner. 32x32 design
    // space scaled to a 20x20 panel-corner badge with Y flipped from
    // Mac's NSView (origin top-left in GDI vs lower-left in AppKit).
    {
        constexpr int kIconPx   = 20;
        constexpr int kIconX0   = 12;
        constexpr int kIconY0   = 8;
        const float   s         = float(kIconPx) / 32.0f;
        auto P = [&](float x, float y) -> POINT {
            return { kIconX0 + LONG(x * s + 0.5f),
                     kIconY0 + LONG((32.0f - y) * s + 0.5f) };
        };
        const POINT hex[7] = {
            P(16.0f,  2.5f), P(27.5f,  9.0f), P(27.5f, 23.0f),
            P(16.0f, 29.5f), P( 4.5f, 23.0f), P( 4.5f,  9.0f),
            P(16.0f,  2.5f),
        };
        HPEN frame_pen = CreatePen(PS_SOLID, 2, theme_.accent);
        HPEN old_pen = (HPEN)SelectObject(mdc, frame_pen);
        Polyline(mdc, hex, 7);
        SelectObject(mdc, old_pen);
        DeleteObject(frame_pen);

        const POINT ray[4] = {
            P( 7.5f, 11.0f), P(13.0f, 19.0f),
            P(21.5f, 14.0f), P(24.5f, 22.0f),
        };
        HPEN ray_pen = CreatePen(PS_SOLID, 2, theme_.logo_ray);
        old_pen = (HPEN)SelectObject(mdc, ray_pen);
        Polyline(mdc, ray, 4);
        SelectObject(mdc, old_pen);
        DeleteObject(ray_pen);

        // Two bounce dots.
        HBRUSH dot_brush = CreateSolidBrush(theme_.logo_ray);
        HBRUSH old_brush = (HBRUSH)SelectObject(mdc, dot_brush);
        HPEN no_pen = (HPEN)GetStockObject(NULL_PEN);
        old_pen = (HPEN)SelectObject(mdc, no_pen);
        const int dr = 2;
        for (auto& d : {ray[1], ray[2]}) {
            Ellipse(mdc, d.x - dr, d.y - dr, d.x + dr + 1, d.y + dr + 1);
        }
        SelectObject(mdc, old_pen);
        SelectObject(mdc, old_brush);
        DeleteObject(dot_brush);
    }

    // ---- Top-right status label (matches Mac's statusLabel:
    // "DEMONT · PATHTRACER · CONSOLE"). Palette dim tint, small font.
    {
        const wchar_t* status = L"DEMONT · PATHTRACER · CONSOLE";
        int slen = (int)wcslen(status);
        SIZE sz{};
        GetTextExtentPoint32W(mdc, status, slen, &sz);
        SetTextColor(mdc, theme_.dim);
        TextOutW(mdc, W - kPaddingX - sz.cx, kPaddingY + 4, status, slen);
    }

    int input_y   = H - line_height_ - kPaddingY;
    int log_top   = kPaddingY + 1 + 24;  // +1 accent, +24 for icon/status row
    int log_bot   = input_y - kPaddingY;
    int max_lines = std::max(1, (log_bot - log_top) / line_height_);

    {
        std::lock_guard lk(state_mutex_);

        int n = (int)scrollback_.size();
        // scroll_lines_ pushes the visible window upward; clamp so we
        // never read negative indices.
        int end_idx   = std::max(0, n - scroll_lines_);
        int start_idx = std::max(0, end_idx - max_lines);

        int y = log_top;
        for (int i = start_idx; i < end_idx; ++i) {
            const auto& sl = scrollback_[i];
            COLORREF c = theme_.text;
            switch (sl.role) {
                case LineRole::LogoFrame:   c = theme_.logo_frame;   break;
                case LineRole::LogoLetters: c = theme_.logo_letters; break;
                case LineRole::LogoRay:     c = theme_.logo_ray;     break;
                case LineRole::Echo:        c = theme_.prompt;       break;
                case LineRole::Default:     c = theme_.text;         break;
            }
            SetTextColor(mdc, c);
            auto wbuf = ToWide(sl.text);
            if (!wbuf.empty()) {
                TextOutW(mdc, kPaddingX, y, wbuf.data(), (int)wbuf.size());
            }
            y += line_height_;
        }

        // Prompt
        SetTextColor(mdc, theme_.prompt);
        TextOutW(mdc, kPaddingX, input_y, L"> ", 2);
        SIZE prompt_sz{};
        GetTextExtentPoint32W(mdc, L"> ", 2, &prompt_sz);

        // Input
        SetTextColor(mdc, theme_.text);
        if (!input_.empty()) {
            auto wbuf = ToWide(input_);
            TextOutW(mdc, kPaddingX + prompt_sz.cx, input_y,
                     wbuf.data(), (int)wbuf.size());
        }

        // Cursor (always-on; blinking via SetTimer is overkill)
        SIZE pre{};
        if (cursor_ > 0) {
            std::string_view pre_sv(input_.data(), cursor_);
            auto pre_w = ToWide(pre_sv);
            if (!pre_w.empty()) {
                GetTextExtentPoint32W(mdc, pre_w.data(), (int)pre_w.size(), &pre);
            }
        }
        int cx = kPaddingX + prompt_sz.cx + pre.cx;
        RECT cur = {cx, input_y, cx + 2, input_y + line_height_};
        HBRUSH cb = CreateSolidBrush(theme_.accent);
        FillRect(mdc, &cur, cb);
        DeleteObject(cb);

        // Inline ghost preview of the highlighted popup match. Only
        // shown when the match is a prefix-fit for the token at the
        // cursor -- substring / fuzzy hits go to the popup rows
        // (highlighted match-spans inside the name) since their tail
        // wouldn't concatenate cleanly with what the user typed.
        if (popup_.active && popup_.selected >= 0 &&
            popup_.selected < static_cast<int>(popup_.items.size()) &&
            cursor_ == static_cast<int>(input_.size())) {
            const auto& it    = popup_.items[popup_.selected];
            const auto& token = popup_.token;
            if (it.name.size() > token.text.size()) {
                // Case-insensitive prefix check (token.text might
                // differ in case from it.name).
                bool prefix = true;
                for (std::size_t i = 0; i < token.text.size(); ++i) {
                    auto lc = [](char c) {
                        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
                    };
                    if (lc(token.text[i]) != lc(it.name[i])) {
                        prefix = false; break;
                    }
                }
                if (prefix) {
                    std::string tail = it.name.substr(token.text.size());
                    // Tint the ghost yellow when the previewed candidate
                    // is the cvar's CURRENT value: cycling Up/Down
                    // through allowed_values, this is the one already
                    // set, so the user gets the same affordance as the
                    // popup row's accent name + left bar -- "this is
                    // what's active right now" -- but visible inline
                    // even when the popup is below the input. Hardcoded
                    // gold (matches amber/vault's prompt and the web
                    // theme's --warn) so it stays warm across every
                    // theme; the Theme palette doesn't carry a `warn`
                    // slot today, and synthesising one per theme would
                    // be churn for a single hint colour.
                    constexpr COLORREF kCurrentHintColor = RGB(255, 200, 80);
                    const bool is_current = (it.value == "current");
                    SetTextColor(mdc, is_current ? kCurrentHintColor
                                                 : theme_.dim);
                    int gx = cx + 3;
                    auto gbuf = ToWide(tail);
                    if (!gbuf.empty()) {
                        TextOutW(mdc, gx, input_y, gbuf.data(),
                                 static_cast<int>(gbuf.size()));
                    }
                }
            }
        }

        // Completion popup -- anchored ABOVE the input row (input
        // sits at the bottom of the panel, so a downward dropdown
        // would clip outside the overlay window). Each row shows the
        // candidate name with its match-spans highlighted in accent
        // colour, an optional kind chip ("cvar" / "cmd"), the
        // current value (or "current" / "default" for value-position
        // rows), and a truncated description appended after the
        // value column.
        if (popup_.active && !popup_.items.empty()) {
            // PopupVisibleRows() does the panel-fit math (shared with
            // MovePopupSelection so Up/Down scrolling agrees with what
            // the user can see). Returns 0 when not even one row
            // fits -- in that case skip rendering entirely. Otherwise
            // visible_n is min(items.size(), kPopupMaxRows=10,
            // by-height-of-available-space) and Up/Down scrolls the
            // window through the remaining items via popup_.scroll_offset.
            const int n = static_cast<int>(popup_.items.size());
            const int visible_n = PopupVisibleRows();
            const int popup_h = visible_n * line_height_ + 8;
            const int popup_w = W - 2 * kPaddingX;
            const int popup_x = kPaddingX;
            const int popup_y = input_y - popup_h - 6;
            if (visible_n > 0) {
                RECT pr = { popup_x, popup_y, popup_x + popup_w,
                            popup_y + popup_h };
                // Panel background -- slightly darker than the
                // overlay's main panel so the popup stands out from
                // the scrollback area behind it. DimColor() is the
                // theme-agnostic way to get that depth without
                // expanding every theme table entry with a separate
                // "popup_bg" colour.
                HBRUSH pbg = CreateSolidBrush(DimColor(theme_.panel, 0.55f));
                FillRect(mdc, &pr, pbg);
                DeleteObject(pbg);
                // 1 px accent outline.
                HPEN outline = CreatePen(PS_SOLID, 1, DimColor(theme_.accent, 0.6f));
                HPEN old_p = (HPEN)SelectObject(mdc, outline);
                HBRUSH no_brush = (HBRUSH)GetStockObject(NULL_BRUSH);
                HBRUSH old_b = (HBRUSH)SelectObject(mdc, no_brush);
                Rectangle(mdc, pr.left, pr.top, pr.right, pr.bottom);
                SelectObject(mdc, old_p);
                SelectObject(mdc, old_b);
                DeleteObject(outline);

                // Clip all subsequent TextOutW calls to the popup's
                // interior. The previous char-budget truncation in
                // the description draw was a guess at how many chars
                // would fit; with a real clip region we can paint the
                // full string and let GDI clip the overflow pixels
                // (still ugly visually, but won't bleed into the
                // scrollback below the popup). SaveDC / RestoreDC
                // bracket the clip so it doesn't affect drawing past
                // the popup.
                const int saved_dc = SaveDC(mdc);
                IntersectClipRect(mdc, pr.left + 1, pr.top + 1,
                                       pr.right - 1, pr.bottom - 1);

                const int scroll = popup_.scroll_offset;
                int row_y = popup_y + 4;
                for (int vi = 0; vi < visible_n; ++vi) {
                    const int idx = scroll + vi;
                    if (idx < 0 || idx >= n) break;
                    const auto& it = popup_.items[idx];
                    const bool selected   = (idx == popup_.selected);
                    const bool is_current = (it.value == "current");
                    const bool is_default = (it.value == "default");
                    if (selected) {
                        RECT rr = { popup_x + 2, row_y,
                                    popup_x + popup_w - 2,
                                    row_y + line_height_ };
                        // Selected row: accent-tinted but dim enough
                        // that the row text stays readable on top.
                        HBRUSH selbg = CreateSolidBrush(DimColor(theme_.accent, 0.22f));
                        FillRect(mdc, &rr, selbg);
                        DeleteObject(selbg);
                    }
                    // Left-edge accent bar marks the current value's
                    // row (independent of selection) so the user can
                    // always spot "what's set right now" while cycling
                    // Up/Down through the value list. Softer dim bar
                    // for the default value, so reset-to-default is
                    // also visible without competing.
                    if (is_current || is_default) {
                        RECT bar = { popup_x + 2, row_y,
                                     popup_x + 4,
                                     row_y + line_height_ };
                        COLORREF c = is_current
                            ? theme_.accent
                            : DimColor(theme_.accent, 0.45f);
                        HBRUSH bb = CreateSolidBrush(c);
                        FillRect(mdc, &bar, bb);
                        DeleteObject(bb);
                    }
                    // Name with match-span highlighting. Walk the
                    // spans, emitting the in-span runs in accent
                    // colour and the out-of-span runs in `text`.
                    // Current-value rows tint the WHOLE name in
                    // accent so the marker isn't just on the bar --
                    // the eye lands on the highlighted name first.
                    const std::string& name = it.name;
                    int tx = popup_x + 8;
                    const COLORREF name_fg = is_current
                        ? theme_.accent
                        : theme_.text;
                    std::size_t cursor_ch = 0;
                    auto draw_run = [&](std::size_t a, std::size_t b,
                                        COLORREF col) {
                        if (a >= b) return;
                        SetTextColor(mdc, col);
                        auto wbuf = ToWide(std::string_view(
                            name.data() + a, b - a));
                        if (wbuf.empty()) return;
                        TextOutW(mdc, tx, row_y, wbuf.data(),
                                 static_cast<int>(wbuf.size()));
                        SIZE sz{};
                        GetTextExtentPoint32W(mdc, wbuf.data(),
                            static_cast<int>(wbuf.size()), &sz);
                        tx += sz.cx;
                    };
                    for (const auto& sp : it.spans) {
                        if (sp.first > cursor_ch) {
                            draw_run(cursor_ch, sp.first, name_fg);
                        }
                        draw_run(sp.first, sp.second, theme_.accent);
                        cursor_ch = sp.second;
                    }
                    if (cursor_ch < name.size()) {
                        draw_run(cursor_ch, name.size(), name_fg);
                    }
                    // Optional kind chip ("cvar" / "cmd") for token-0
                    // rows. Mirrors the web console + Mac overlay so a
                    // user reading the popup can tell at a glance
                    // whether a candidate is a cvar (settable) or a
                    // command (callable). Drawn in dim so it doesn't
                    // compete with the name -- value position rows
                    // (kind == Value) don't get a chip since the kind
                    // is implicit from context.
                    if (it.kind == pt::console::CompletionKind::Cvar ||
                        it.kind == pt::console::CompletionKind::Command) {
                        const wchar_t* chip =
                            (it.kind == pt::console::CompletionKind::Cvar)
                                ? L"cvar" : L"cmd";
                        const int chip_len = (int)wcslen(chip);
                        SetTextColor(mdc, theme_.dim);
                        tx += 8;
                        TextOutW(mdc, tx, row_y, chip, chip_len);
                        SIZE csz{};
                        GetTextExtentPoint32W(mdc, chip, chip_len, &csz);
                        tx += csz.cx;
                    }
                    // Right-justified value chip + truncated
                    // description, separated by ` -- `. Painted in
                    // dim so they don't fight the name for the eye.
                    tx += 12;
                    if (!it.value.empty()) {
                        // Value chip colour: accent for "current",
                        // dimmed-accent for "default", plain dim for
                        // the cvar's actual current value when it's
                        // shown on a non-value-position row (cvar
                        // candidate at token 0). The is_current /
                        // is_default flags above already capture the
                        // tag-driven cases.
                        COLORREF vc = is_current
                            ? theme_.accent
                            : (is_default ? DimColor(theme_.accent, 0.6f)
                                          : theme_.dim);
                        SetTextColor(mdc, vc);
                        auto vbuf = ToWide(it.value);
                        if (!vbuf.empty()) {
                            TextOutW(mdc, tx, row_y, vbuf.data(),
                                     static_cast<int>(vbuf.size()));
                            SIZE sz{};
                            GetTextExtentPoint32W(mdc, vbuf.data(),
                                static_cast<int>(vbuf.size()), &sz);
                            tx += sz.cx + 12;
                        }
                    }
                    if (!it.description.empty()) {
                        // Clip region (IntersectClipRect above) keeps
                        // the text from bleeding past the popup right
                        // edge into the scrollback. We still cap with
                        // a char-budget truncate so the visible part
                        // ends in an ellipsis instead of a sharp pixel
                        // cut -- nicer reading.
                        const int budget = (popup_x + popup_w - 8) - tx;
                        if (budget > 24) {
                            std::string desc = it.description;
                            // Rough estimate: ~7 px/char in monospace
                            // at default line_height_; halve when at
                            // 2x scale. Just pick a sane cap.
                            const std::size_t max_chars =
                                static_cast<std::size_t>(std::max(0, budget / 7));
                            if (desc.size() > max_chars && max_chars > 3) {
                                // UTF-8 safe truncation -- naive resize
                                // by byte count can split a multibyte
                                // codepoint, and ToWide()'s
                                // MultiByteToWideChar(CP_UTF8) returns 0
                                // on invalid UTF-8, dropping the entire
                                // description from the popup row. Cvar
                                // descriptions contain non-ASCII (e.g.
                                // "≈" in Engine.cpp).
                                pt::console::Utf8SafeTruncate(desc, max_chars - 1);
                                desc.push_back('\xE2');  // unicode ellipsis "…"
                                desc.push_back('\x80');  // utf-8 bytes
                                desc.push_back('\xA6');
                            }
                            SetTextColor(mdc, theme_.dim);
                            auto dbuf = ToWide(desc);
                            if (!dbuf.empty()) {
                                TextOutW(mdc, tx, row_y, dbuf.data(),
                                         static_cast<int>(dbuf.size()));
                            }
                        }
                    }
                    row_y += line_height_;
                }

                // Drop the popup clip so subsequent draws (none today
                // but defence-in-depth for any future Paint() append)
                // aren't constrained.
                RestoreDC(mdc, saved_dc);
            }
        }
    }

    SelectObject(mdc, old_font);
    BitBlt(dc, 0, 0, W, H, mdc, 0, 0, SRCCOPY);
    SelectObject(mdc, old_bmp);
    DeleteObject(bmp);
    DeleteDC(mdc);
}

}  // namespace

// ---- Public ConsoleOverlay wrappers --------------------------------------

ConsoleOverlay::ConsoleOverlay()  : opaque_(new WinOverlay) {}
ConsoleOverlay::~ConsoleOverlay() {
    delete static_cast<WinOverlay*>(opaque_);
    opaque_ = nullptr;
}

bool ConsoleOverlay::Init(void* hwnd) {
    if (!opaque_ || !hwnd) return false;
    return static_cast<WinOverlay*>(opaque_)->Init(static_cast<HWND>(hwnd));
}
void ConsoleOverlay::Shutdown()                        { if (opaque_) static_cast<WinOverlay*>(opaque_)->Shutdown(); }
void ConsoleOverlay::Show()                            { if (opaque_) static_cast<WinOverlay*>(opaque_)->Show(); }
void ConsoleOverlay::Hide()                            { if (opaque_) static_cast<WinOverlay*>(opaque_)->Hide(); }
void ConsoleOverlay::Toggle()                          { if (opaque_) static_cast<WinOverlay*>(opaque_)->Toggle(); }
bool ConsoleOverlay::IsShown() const                   { return opaque_ && static_cast<WinOverlay*>(opaque_)->IsShown(); }
void ConsoleOverlay::ApplyTheme(std::string_view n)    { if (opaque_) static_cast<WinOverlay*>(opaque_)->ApplyTheme(n); }
void ConsoleOverlay::Repaint()                         { if (opaque_) static_cast<WinOverlay*>(opaque_)->Repaint(); }
void ConsoleOverlay::NotifyParentResized(int w, int h) { if (opaque_) static_cast<WinOverlay*>(opaque_)->NotifyParentResized(w, h); }
bool ConsoleOverlay::SaveState(const std::string& path) const {
    return opaque_ && static_cast<const WinOverlay*>(opaque_)->SaveState(path);
}
bool ConsoleOverlay::LoadState(const std::string& path) {
    return opaque_ && static_cast<WinOverlay*>(opaque_)->LoadState(path);
}

void ConsoleOverlay::OnLog(pt::log::Level lvl, const std::string& body) {
    if (WinOverlay* w = WinOverlay::g.load(std::memory_order_acquire)) {
        w->OnLog(lvl, body);
    }
}
void ConsoleOverlay::SetGlobalInstance(ConsoleOverlay* /*o*/) {
    // The Mac overlay uses this to bridge a sink from a static log
    // callback back to the singleton instance. Win32 does the same
    // bridging via WinOverlay::g, set in Init(); kept as a no-op
    // here so engine code stays uniform.
}

}  // namespace pt::app

// C-extern shims expected by Window.cpp / SoftwareDevice.cpp / MetalDevice.cpp.
// On Apple they come from Window_Cocoa.mm and MetalLayerAttach.mm. On Win32
// the Vulkan/software backends create their surface natively, so these are
// no-ops -- same as ConsoleOverlay_Stub.cpp on other non-Apple platforms.
extern "C" void* pt_window_native_cocoa(void*) { return nullptr; }
extern "C" void  pt_metal_attach_layer(void*, void*) {}

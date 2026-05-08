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
#include "../console/Console.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
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
    static WinOverlay* g;

    bool  Init(HWND parent);
    void  Shutdown();
    void  Show();
    void  Hide();
    void  Toggle();
    bool  IsShown() const { return shown_; }
    void  ApplyTheme(std::string_view name);
    void  NotifyParentResized(int w, int h);
    void  OnLog(pt::log::Level lvl, const std::string& body);

private:
    static LRESULT CALLBACK WndProcThunk(HWND, UINT, WPARAM, LPARAM);
    static LRESULT CALLBACK ParentWndProcThunk(HWND, UINT, WPARAM, LPARAM);
    LRESULT WndProc(HWND, UINT, WPARAM, LPARAM);
    void Paint(HDC dc);
    void Repaint();
    void SubmitInput();
    void HandleTab();
    void CycleGhost(int dir);
    void CommitGhost();
    void DismissGhost();
    void ActivateValueGhost(const std::string& cvar_name);
    void RefreshGhostAnnotation();
    void StartAnim(bool showing);
    void TickAnim();
    int  CurrentY() const;

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

    std::mutex              state_mutex_;
    std::deque<ScrollLine>  scrollback_;
    std::string             input_;
    int                     cursor_       = 0;
    int                     scroll_lines_ = 0;     // 0 = at bottom
    std::deque<std::string> history_;
    int                     history_pos_  = -1;

    // Tab-completion ghost state (fish-shell style autosuggestion).
    // First Tab on an ambiguous prefix extends to the longest common
    // prefix AND activates ghost mode: the first remaining match is
    // rendered after the cursor in dim color. Subsequent Tabs cycle
    // forward through matches; Shift+Tab cycles back; Right-arrow at
    // end / End commits; Esc / typing dismisses.
    //
    // Two activation modes:
    //   - Token 0 (cvar/command name): triggered by Tab on a partial
    //     name; matches list is the prefix-filtered name list.
    //   - Value position: either user typed `cvar ` then Tab, OR a
    //     token-0 commit just landed on a cvar (auto-activated by
    //     CommitGhost). Matches are allowed_values when present;
    //     otherwise [current, default] with `is_meta` set so the
    //     inactive one is shown as `annotation` next to the primary
    //     ghost (user sees both at once).
    struct GhostState {
        bool active = false;
        std::vector<std::string> matches;
        std::size_t              index = 0;
        std::string              before;     // text preserved before the token
        std::string              prefix;     // typed text of the token (already in input_)
        bool                     is_token0 = false;  // true → trailing space on commit
        bool                     is_meta   = false;  // matches = [current, default] of a cvar
        std::string              annotation;         // "default: X" or "current: Y" when is_meta
    };
    GhostState ghost_;

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

WinOverlay* WinOverlay::g = nullptr;

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
    WinOverlay* self = WinOverlay::g;
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

    g = this;

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
    if (g == this) g = nullptr;
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
    if (g == nullptr) return;
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
        std::lock_guard lk(g->state_mutex_);
        g->scrollback_.push_back({std::move(clean), LineRole::Default});
        while (g->scrollback_.size() > kScrollMax) g->scrollback_.pop_front();
    }
    // pt::log sinks fire on the calling thread (Log::Emit doesn't
    // reschedule), so OnLog can be invoked from any worker. Post a
    // custom repaint message instead of calling Repaint() / Invalidate
    // directly -- PostMessage is the only cross-thread Win32 UI bridge
    // that's well-defined.
    if (HWND target = g->hwnd_) {
        PostMessageW(target, WM_APP_REPAINT, 0, 0);
    }
}

void WinOverlay::Repaint() {
    if (hwnd_) InvalidateRect(hwnd_, nullptr, FALSE);
}

LRESULT WinOverlay::WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
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
        // Backtick: toggle off (matches the engine's Mac path where
        // backtick on the GLFW window opens the overlay; once open
        // the Cocoa NSView's keyDown closes it).
        if (w == '`') { Hide(); return 0; }
        if (w == '\r') { SubmitInput(); return 0; }
        if (w == '\b') {
            std::lock_guard lk(state_mutex_);
            if (cursor_ > 0) { input_.erase(cursor_ - 1, 1); cursor_--; }
            Repaint();
            return 0;
        }
        if (w >= 32 && w < 127) {
            std::lock_guard lk(state_mutex_);
            input_.insert(cursor_, 1, static_cast<char>(w));
            cursor_++;
            Repaint();
            // After typing space, if the input is exactly "<name> "
            // (single token + trailing space, cursor at end) and that
            // name resolves to a known cvar/command with suggestions,
            // auto-activate the value-position ghost.  Mirrors the
            // post-commit auto-activation so users who type the full
            // name themselves get the same affordance as those who
            // tab-completed it.
            if (w == ' ' && cursor_ == static_cast<int>(input_.size()) &&
                input_.size() >= 2) {
                auto first_space = input_.find(' ');
                if (first_space + 1 == input_.size()) {
                    ActivateValueGhost(input_.substr(0, first_space));
                }
            }
            return 0;
        }
        return 0;
    }
    case WM_KEYDOWN: {
        std::lock_guard lk(state_mutex_);

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

        // Ghost-mode preamble.  When the autosuggestion is active these
        // keys mean something different than their default: Tab cycles,
        // Shift+Tab cycles back, Right-arrow at end / End commits, Esc
        // dismisses, Enter commits-then-submits.  Any other key
        // dismisses the ghost and falls through to normal handling.
        if (ghost_.active) {
            const bool shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
            if (w == VK_TAB) {
                CycleGhost(shift ? -1 : +1);
                return 0;
            }
            if (w == VK_END ||
                (w == VK_RIGHT && cursor_ == static_cast<int>(input_.size()))) {
                CommitGhost();
                return 0;
            }
            if (w == VK_ESCAPE) {
                DismissGhost();
                return 0;
            }
            if (w == VK_RETURN) {
                CommitGhost();
                // Fall through; WM_CHAR '\r' will then submit.
            } else {
                DismissGhost();
                // Fall through to existing handlers below.
            }
        }

        switch (w) {
        case VK_TAB:
            // Handled here (not WM_CHAR) so DefWindowProcW never sees
            // it -- WS_CHILD + VK_TAB otherwise triggers focus
            // traversal in the parent's tab order.
            HandleTab();
            return 0;
        case VK_LEFT:
            if (cursor_ > 0) { cursor_--; Repaint(); }
            return 0;
        case VK_RIGHT:
            if (cursor_ < (int)input_.size()) { cursor_++; Repaint(); }
            return 0;
        case VK_HOME:   cursor_ = 0;                       Repaint(); return 0;
        case VK_END:    cursor_ = (int)input_.size();      Repaint(); return 0;
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
            // ESC also closes the overlay -- common console-game
            // pattern, gives a second way out besides backtick.
            // (Hide() acquires state_mutex_ via its own path.)
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

// First Tab on a fresh prefix.  Caller (WM_KEYDOWN) holds state_mutex_,
// so direct access to input_/cursor_/ghost_ is safe.  Console::Enumerate*
// /FindCVar run on the engine main thread (same thread that owns the
// cvar map) -- no Console-side locking needed.
//
// Behavior: if there's a single match, commit it inline (with trailing
// space for token 0).  If multiple, extend to the longest common prefix
// and then activate ghost mode: ghost_ holds the candidate list and
// renders the first match's tail as dim text after the cursor.
// Subsequent Tab/Shift+Tab cycle the ghost; Right/End commit it.
void WinOverlay::HandleTab() {
    std::size_t last_space = input_.rfind(' ');

    std::string prefix;
    std::vector<std::string> candidates;

    if (last_space == std::string::npos) {
        // Token 0: cvar / command name.
        prefix = input_;
        pt::console::Console::Get().EnumerateCVars("",
            [&](pt::console::CVar& v) { candidates.push_back(v.name); });
        pt::console::Console::Get().EnumerateCommands("",
            [&](pt::console::Command& c) { candidates.push_back(c.name); });
        std::sort(candidates.begin(), candidates.end());
    } else {
        // Value position. `toggle <cvar>` is special-cased: token 1 is a
        // cvar name (only those with allowed_values are useful). Otherwise
        // we complete from the named cvar's allowed_values.
        std::size_t first_space = input_.find(' ');
        std::string first_tok   = input_.substr(0, first_space);
        prefix                  = input_.substr(last_space + 1);
        if (first_tok == "toggle") {
            pt::console::Console::Get().EnumerateCVars("",
                [&](pt::console::CVar& v) {
                    if (!v.allowed_values.empty()) {
                        candidates.push_back(v.name);
                    }
                });
            std::sort(candidates.begin(), candidates.end());
        } else {
            auto* cv = pt::console::Console::Get().FindCVar(first_tok);
            if (cv == nullptr || cv->allowed_values.empty()) return;
            candidates = cv->allowed_values;
        }
    }

    std::vector<std::string> matches;
    for (const auto& c : candidates) {
        if (c.size() >= prefix.size() &&
            c.compare(0, prefix.size(), prefix) == 0) {
            matches.push_back(c);
        }
    }
    if (matches.empty()) return;

    const bool is_token0 = (last_space == std::string::npos);
    std::string before = is_token0 ? std::string()
                                   : input_.substr(0, last_space + 1);

    if (matches.size() == 1) {
        // Single match: commit immediately.  Trailing space for token 0
        // so the next argument can be typed straight away.
        std::string tail = is_token0 ? matches[0] + " " : matches[0];
        input_  = before + tail;
        cursor_ = static_cast<int>(input_.size());
        Repaint();
        return;
    }

    // Multiple matches: extend to their longest common prefix.
    std::string common = matches[0];
    for (std::size_t i = 1; i < matches.size(); ++i) {
        std::size_t lim = std::min(common.size(), matches[i].size());
        std::size_t j   = 0;
        while (j < lim && common[j] == matches[i][j]) ++j;
        common.resize(j);
        if (common.empty()) break;
    }
    if (common.size() > prefix.size()) {
        input_  = before + common;
        cursor_ = static_cast<int>(input_.size());
        prefix  = common;  // ghost rendering trims by this length
    }

    // Activate ghost mode -- subsequent Tabs cycle, Right/End commits.
    ghost_.active    = true;
    ghost_.matches   = std::move(matches);
    ghost_.index     = 0;
    ghost_.before    = std::move(before);
    ghost_.prefix    = std::move(prefix);
    ghost_.is_token0 = is_token0;
    Repaint();
}

void WinOverlay::CycleGhost(int dir) {
    if (!ghost_.active || ghost_.matches.empty()) return;
    int n = static_cast<int>(ghost_.matches.size());
    int i = (static_cast<int>(ghost_.index) + dir) % n;
    if (i < 0) i += n;
    ghost_.index = static_cast<std::size_t>(i);
    RefreshGhostAnnotation();
    Repaint();
}

void WinOverlay::CommitGhost() {
    if (!ghost_.active || ghost_.matches.empty()) {
        DismissGhost();
        return;
    }
    const std::string committed = ghost_.matches[ghost_.index];
    const bool        was_token0 = ghost_.is_token0;
    std::string tail = was_token0 ? (committed + " ") : committed;
    input_  = ghost_.before + tail;
    cursor_ = static_cast<int>(input_.size());
    DismissGhost();

    // Token-0 commit just landed on a name.  If the name is a cvar,
    // auto-activate a value-position ghost so the user immediately
    // sees the cvar's current value (and default, for free-form
    // cvars).  Commands have no value-position ghost.
    if (was_token0) ActivateValueGhost(committed);
}

void WinOverlay::DismissGhost() {
    if (!ghost_.active) return;
    ghost_.active = false;
    ghost_.matches.clear();
    ghost_.index  = 0;
    ghost_.before.clear();
    ghost_.prefix.clear();
    ghost_.is_token0 = false;
    ghost_.is_meta   = false;
    ghost_.annotation.clear();
    Repaint();
}

// After a token-0 commit OR a typed `<name> ` sequence, auto-show
// the cvar's current value (or a command's default args) as a ghost
// so the user can see a useful suggestion without typing anything
// more.  Three branches:
//
//   - CVar with allowed_values: cycle those (same as the existing
//     value-position behaviour after `cvar ` + Tab).
//   - Free-form cvar: cycle [current, default] with `is_meta` set
//     so the inactive one is rendered as an annotation.
//   - Command with default_args: single-match ghost showing the
//     default invocation (e.g. `screenshot demonte_screen.ppm`).
//
// No-op for commands without default_args, and for unknown names.
void WinOverlay::ActivateValueGhost(const std::string& name) {
    auto& C = pt::console::Console::Get();
    std::vector<std::string> matches;
    bool meta = false;

    if (auto* cv = C.FindCVar(name); cv != nullptr) {
        if (!cv->allowed_values.empty()) {
            matches = cv->allowed_values;
        } else {
            matches.push_back(cv->value);
            if (cv->default_value != cv->value) {
                matches.push_back(cv->default_value);
                meta = true;
            }
        }
    } else if (auto* cmd = C.FindCommand(name); cmd != nullptr) {
        if (!cmd->default_args.empty()) {
            matches.push_back(cmd->default_args);
        }
    }
    if (matches.empty()) return;

    ghost_.active    = true;
    ghost_.matches   = std::move(matches);
    ghost_.index     = 0;
    ghost_.before    = input_;   // input_ already ends with "<name> "
    ghost_.prefix.clear();
    ghost_.is_token0 = false;
    ghost_.is_meta   = meta;
    RefreshGhostAnnotation();
    Repaint();
}

void WinOverlay::RefreshGhostAnnotation() {
    ghost_.annotation.clear();
    if (!ghost_.is_meta || ghost_.matches.size() < 2) return;
    // matches[0] = current, matches[1] = default (per ActivateValueGhost).
    if (ghost_.index == 0) ghost_.annotation = "  default: " + ghost_.matches[1];
    else                   ghost_.annotation = "  current: " + ghost_.matches[0];
}

void WinOverlay::Paint(HDC dc) {
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

    int input_y   = H - kLineHeight - kPaddingY;
    int log_top   = kPaddingY + 1 + 24;  // +1 accent, +24 for icon/status row
    int log_bot   = input_y - kPaddingY;
    int max_lines = std::max(1, (log_bot - log_top) / kLineHeight);

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
            y += kLineHeight;
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
        RECT cur = {cx, input_y, cx + 2, input_y + kLineHeight};
        HBRUSH cb = CreateSolidBrush(theme_.accent);
        FillRect(mdc, &cur, cb);
        DeleteObject(cb);

        // Ghost text (dim, drawn just past the cursor).  Only the
        // portion of the candidate beyond the user's typed prefix is
        // shown -- typed chars are already in input_ above.  An
        // optional annotation (current/default for free-form cvars)
        // trails the primary ghost in the same dim colour so the user
        // sees both pieces of context at a glance.
        if (ghost_.active && !ghost_.matches.empty()) {
            const std::string& match = ghost_.matches[ghost_.index];
            if (match.size() > ghost_.prefix.size() &&
                match.compare(0, ghost_.prefix.size(), ghost_.prefix) == 0) {
                std::string ghost_tail = match.substr(ghost_.prefix.size());
                SetTextColor(mdc, theme_.dim);
                int gx = cx + 3;
                auto gbuf = ToWide(ghost_tail);
                if (!gbuf.empty()) {
                    TextOutW(mdc, gx, input_y, gbuf.data(),
                             static_cast<int>(gbuf.size()));
                    SIZE gsz{};
                    GetTextExtentPoint32W(mdc, gbuf.data(),
                                          static_cast<int>(gbuf.size()), &gsz);
                    gx += gsz.cx;
                }
                if (!ghost_.annotation.empty()) {
                    auto abuf = ToWide(ghost_.annotation);
                    if (!abuf.empty()) {
                        TextOutW(mdc, gx, input_y, abuf.data(),
                                 static_cast<int>(abuf.size()));
                    }
                }
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
void ConsoleOverlay::NotifyParentResized(int w, int h) { if (opaque_) static_cast<WinOverlay*>(opaque_)->NotifyParentResized(w, h); }

void ConsoleOverlay::OnLog(pt::log::Level lvl, const std::string& body) {
    if (WinOverlay::g) WinOverlay::g->OnLog(lvl, body);
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

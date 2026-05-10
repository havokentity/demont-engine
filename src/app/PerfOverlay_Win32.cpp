// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Win32 performance overlay -- child HWND in the top-right of the
// engine's GLFW window, painted with GDI.  Mirrors the architecture
// of ConsoleOverlay_Win32 but is single-purpose (read-only stats
// readout, no input handling) and always-visible while the level is
// non-zero.

#include <Windows.h>

#include "PerfOverlay.h"
#include "../console/Console.h"
#include "../core/Log.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>      // std::strtof for r_perf_overlay_scale parse
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace pt::app {

namespace {

struct Theme {
    const char* name;
    COLORREF panel;
    COLORREF text;
    COLORREF accent;
    COLORREF dim;
    COLORREF graph;
};
constexpr Theme kThemes[] = {
    {"hardcore",  RGB( 12, 14, 22), RGB(220,220,232), RGB(  0,240,255), RGB(110,120,146), RGB(  0,240,255)},
    {"amber",     RGB( 12,  8,  4), RGB(255,200,140), RGB(255,140, 40), RGB(180,128, 80), RGB(255,140, 40)},
    {"synthwave", RGB( 18,  6, 36), RGB(255,180,250), RGB(255, 80,180), RGB(180,100,180), RGB(255, 80,180)},
    {"matrix",    RGB(  0, 12,  4), RGB( 80,255, 80), RGB(  0,200,  0), RGB( 40,140, 40), RGB(  0,255, 80)},
    {"vault",     RGB( 16, 24, 32), RGB(180,220,255), RGB( 80,160,255), RGB(100,140,180), RGB( 80,160,255)},
    {"sakura",    RGB( 36, 12, 26), RGB(255,200,220), RGB(255,140,180), RGB(180,140,160), RGB(255,140,180)},
    {"mono",      RGB( 12, 12, 12), RGB(220,220,220), RGB(180,180,180), RGB(120,120,120), RGB(220,220,220)},
};

constexpr int kFontHeight  = 13;
constexpr int kLineHeight  = 16;
constexpr int kPaddingX    = 10;
constexpr int kPaddingY    = 8;
constexpr int kPanelW      = 296;
constexpr int kPanelMargin = 12;
constexpr int kGraphH      = 44;
constexpr BYTE kPanelAlpha = 200;

// Per-tier panel height (excludes graph block).  Sized to fit the
// number of text lines a tier renders.  Computed dynamically in
// CurrentPanelHeight so changes only need to land in one spot.
int LinesForLevel(int level) {
    switch (level) {
        case 1:  return 2;     // FPS, frame_ms
        case 2:  return 8;     // + backend, res, mem, spp, bounces, prims, gpu name
        case 3:  return 8;     // same text rows + graph
        default: return 0;
    }
}

std::vector<wchar_t> ToWide(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::vector<wchar_t> w(n);
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

class WinPerf {
public:
    bool Init(HWND parent);
    void Shutdown();
    void SetLevel(int level);
    int  Level() const { return level_.load(std::memory_order_relaxed); }
    void Update(const PerfStats& stats);
    void NotifyParentResized(int w, int h);
    void ApplyTheme(std::string_view name);

    // Public entry points exposed via PerfOverlay's stable handle.
    void Repaint();

private:
    static LRESULT CALLBACK WndProcThunk(HWND, UINT, WPARAM, LPARAM);
    LRESULT WndProc(HWND, UINT, WPARAM, LPARAM);
    void Paint(HDC dc);
    int  CurrentPanelHeight() const;
    void Reposition();
    void UpdateVisibility();
    // Re-pull r_perf_overlay_scale, recreate the HFONT + recompute
    // line_height_/panel_w_/graph_h_ if it changed since last frame.
    // Mirror of ConsoleOverlay_Win32's EnsureFontScale (PR #7). Cheap
    // when the cvar is unchanged (single cvar lookup + float compare).
    void EnsureScale();

    HWND  parent_         = nullptr;
    HWND  hwnd_           = nullptr;
    HFONT font_           = nullptr;
    bool  font_is_owned_  = false;
    bool  is_layered_     = false;
    int   parent_w_       = 0;
    int   parent_h_       = 0;
    Theme theme_          = kThemes[0];
    std::atomic<int> level_{0};
    // r_perf_overlay_scale state -- mirrors ConsoleOverlay_Win32's
    // con_font_scale plumbing (PR #7). Cvar IS the source of truth;
    // these are caches updated inside EnsureScale() at Paint() time.
    // EnsureScale() recreates the HFONT + recomputes line_height_ /
    // panel_w_ when the cvar value differs from font_scale_.
    float font_scale_  = 1.0f;
    int   line_height_ = kLineHeight;
    int   panel_w_     = kPanelW;
    int   graph_h_     = kGraphH;

    std::mutex     stats_mutex_;
    PerfStats      stats_;                 // history span owned by caller
    std::vector<float> history_copy_;      // own-it copy for safe paint
};

LRESULT CALLBACK WinPerf::WndProcThunk(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(l);
        SetWindowLongPtrW(h, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<WinPerf*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    if (self) return self->WndProc(h, m, w, l);
    return DefWindowProcW(h, m, w, l);
}

bool WinPerf::Init(HWND parent) {
    parent_ = parent;
    if (!parent_) {
        LOG_WARN("PerfOverlay_Win32: Init called with null parent HWND");
        return false;
    }

    static std::atomic<bool> registered{false};
    static const wchar_t* class_name = L"DemontPerfOverlay";
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true)) {
        WNDCLASSEXW wc{};
        wc.cbSize        = sizeof(WNDCLASSEXW);
        wc.lpfnWndProc   = WndProcThunk;
        wc.hInstance     = GetModuleHandleW(nullptr);
        wc.hCursor       = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
        wc.lpszClassName = class_name;
        if (RegisterClassExW(&wc) == 0) {
            LOG_ERROR("PerfOverlay_Win32: RegisterClassExW failed (GLE={})",
                      GetLastError());
            registered.store(false);
            return false;
        }
    }

    RECT pr{};
    GetClientRect(parent_, &pr);
    parent_w_ = pr.right - pr.left;
    parent_h_ = pr.bottom - pr.top;

    // WS_EX_LAYERED only: WS_EX_TRANSPARENT was redundant alongside
    // HTTRANSPARENT click-through and appeared to interact poorly with
    // DWM compositing of sibling layered children over a Vulkan
    // swapchain (panel went stale when the console hid).
    SetLastError(0);
    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED,
        class_name, L"",
        // WS_CLIPSIBLINGS pairs with the same flag on the console
        // overlay so neither sibling's repaint clobbers the other's
        // pixels -- without it, the aggressive RDW_UPDATENOW pump
        // here leaves "copy" artifacts on the console panel.
        WS_CHILD | WS_CLIPSIBLINGS,
        0, 0, kPanelW, 64,
        parent_, nullptr,
        GetModuleHandleW(nullptr), this);
    DWORD gle = GetLastError();
    is_layered_ = (hwnd_ != nullptr);
    if (!hwnd_) {
        LOG_WARN("PerfOverlay_Win32: layered child create failed (GLE={}), retrying opaque", gle);
        SetLastError(0);
        hwnd_ = CreateWindowExW(
            0,
            class_name, L"",
            WS_CHILD | WS_CLIPSIBLINGS,
            0, 0, kPanelW, 64,
            parent_, nullptr,
            GetModuleHandleW(nullptr), this);
        gle = GetLastError();
    }
    if (!hwnd_) {
        LOG_ERROR("PerfOverlay_Win32: CreateWindowExW failed (GLE={})", gle);
        return false;
    }
    is_layered_ = (GetWindowLongPtrW(hwnd_, GWL_EXSTYLE) & WS_EX_LAYERED) != 0;
    if (is_layered_) {
        SetLayeredWindowAttributes(hwnd_, 0, kPanelAlpha, LWA_ALPHA);
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
        font_          = (HFONT)GetStockObject(SYSTEM_FIXED_FONT);
        font_is_owned_ = false;
    }

    return true;
}

void WinPerf::Shutdown() {
    if (font_) {
        if (font_is_owned_) DeleteObject(font_);
        font_          = nullptr;
        font_is_owned_ = false;
    }
    if (hwnd_)  { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    parent_ = nullptr;
}

void WinPerf::Repaint() {
    if (hwnd_) InvalidateRect(hwnd_, nullptr, FALSE);
}

void WinPerf::EnsureScale() {
    auto& C = pt::console::Console::Get();
    auto* v = C.FindCVar("r_perf_overlay_scale");
    if (v == nullptr) return;
    char* end = nullptr;
    float requested = std::strtof(v->value.c_str(), &end);
    // Reject "no digits parsed" AND non-finite values (NaN / +-Inf).
    // strtof happily returns a quiet NaN for "nan"/"NaN"/"NAN" inputs
    // and +-Inf for "inf"; once non-finite slips into the comparisons
    // below, the < / > clamps return false in BOTH directions (NaN
    // compares unordered) -- so requested would stay NaN, sneak past
    // the early-exit, and the static_cast<int>(NaN * something) calls
    // computing new_font_h / new_panel_w / new_graph_h would be
    // undefined behavior. Fall back to 1.0 (engineering default)
    // before any math/casts. Copilot review pass on PR #10.
    if (end == v->value.c_str() || !std::isfinite(requested)) requested = 1.0f;
    if (requested < 0.5f) requested = 0.5f;
    if (requested > 3.0f) requested = 3.0f;
    if (std::fabs(requested - font_scale_) < 1e-3f) return;

    const int new_font_h = std::max(6, static_cast<int>(kFontHeight * requested + 0.5f));
    const int new_line_h = std::max(8, static_cast<int>(kLineHeight * requested + 0.5f));
    const int new_panel_w = std::max(64, static_cast<int>(kPanelW * requested + 0.5f));
    const int new_graph_h = std::max(16, static_cast<int>(kGraphH * requested + 0.5f));

    HFONT new_font = CreateFontW(
        new_font_h, 0, 0, 0, FW_NORMAL,
        FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        FIXED_PITCH | FF_MODERN,
        L"Cascadia Mono");
    if (new_font == nullptr) {
        LOG_WARN("PerfOverlay: CreateFontW failed at scale={:.2f} "
                 "(GLE={}); keeping previous font",
                 requested, GetLastError());
        return;
    }

    // Swap then free (mirrors ConsoleOverlay_Win32's EnsureFontScale
    // ordering after Copilot's review on PR #7 -- if the window paints
    // between Delete and assign it'd reach a dangling HFONT). Stock
    // fonts are not owned and must not be DeleteObject'd.
    HFONT      old_font  = font_;
    const bool old_owned = font_is_owned_;
    font_          = new_font;
    font_is_owned_ = true;
    font_scale_    = requested;
    line_height_   = new_line_h;
    panel_w_       = new_panel_w;
    graph_h_       = new_graph_h;
    if (old_font != nullptr && old_owned) DeleteObject(old_font);

    // Window dimensions changed -- reposition + force a fresh paint.
    Reposition();
    LOG_INFO("PerfOverlay: r_perf_overlay_scale={:.2f} -> "
             "font_h={} line_h={} panel_w={} graph_h={}",
             requested, new_font_h, new_line_h, new_panel_w, new_graph_h);
}

int WinPerf::CurrentPanelHeight() const {
    int level = level_.load(std::memory_order_relaxed);
    int lines = LinesForLevel(level);
    int h = kPaddingY * 2 + lines * line_height_;
    if (level >= 3) h += graph_h_ + kPaddingY;
    return h;
}

void WinPerf::Reposition() {
    if (!hwnd_) return;
    int h  = CurrentPanelHeight();
    int x  = std::max(0, parent_w_ - panel_w_ - kPanelMargin);
    int y  = kPanelMargin;
    SetWindowPos(hwnd_, nullptr, x, y, panel_w_, h,
                 SWP_NOZORDER | SWP_NOACTIVATE);
}

void WinPerf::UpdateVisibility() {
    if (!hwnd_) return;
    if (Level() == 0) {
        ShowWindow(hwnd_, SW_HIDE);
    } else {
        Reposition();
        ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
        InvalidateRect(hwnd_, nullptr, FALSE);
    }
}

void WinPerf::SetLevel(int level) {
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    level_.store(level, std::memory_order_relaxed);
    UpdateVisibility();
}

void WinPerf::Update(const PerfStats& stats) {
    if (!hwnd_ || Level() == 0) return;
    {
        std::lock_guard lk(stats_mutex_);
        stats_ = stats;
        // Take an owned copy of the history so Paint can read it
        // without racing the engine's ring buffer.
        history_copy_.assign(stats.frame_ms_history.begin(),
                             stats.frame_ms_history.end());
    }
    // Aggressive repaint pump.  Layered children over a Vulkan
    // swapchain don't reliably pick up sibling z-order shuffles or
    // present-time invalidations, so we belt-and-suspender every
    // Update tick:
    //   1. ShowWindow(SW_SHOWNOACTIVATE): defensive in case
    //      something hid the window.  Cheap if already visible.
    //   2. SetWindowPos(HWND_TOP, NOACTIVATE): keep above siblings
    //      (e.g. the console panel) without stealing focus.
    //   3. RedrawWindow(... RDW_INVALIDATE | RDW_UPDATENOW):
    //      schedule WM_PAINT *and* synchronously dispatch it now so
    //      DWM has fresh pixels in the layered surface.
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    SetWindowPos(hwnd_, HWND_TOP, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    RedrawWindow(hwnd_, nullptr, nullptr,
                 RDW_INVALIDATE | RDW_UPDATENOW);
}

void WinPerf::NotifyParentResized(int w, int h) {
    parent_w_ = w; parent_h_ = h;
    if (Level() != 0) Reposition();
}

void WinPerf::ApplyTheme(std::string_view name) {
    for (const auto& t : kThemes) {
        if (name == t.name) { theme_ = t; break; }
    }
    if (hwnd_ && Level() != 0) InvalidateRect(hwnd_, nullptr, FALSE);
}

LRESULT WinPerf::WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        Paint(dc);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_NCHITTEST:
        // Pass clicks through to the parent (game viewport) -- we're
        // a read-only HUD, never want focus or input.
        return HTTRANSPARENT;
    }
    return DefWindowProcW(h, m, w, l);
}

void WinPerf::Paint(HDC dc) {
    EnsureScale();  // re-pull r_perf_overlay_scale, recreate font on change
    RECT rc{};
    GetClientRect(hwnd_, &rc);
    int W = rc.right;
    int H = rc.bottom;

    HDC mdc = CreateCompatibleDC(dc);
    HBITMAP bmp = CreateCompatibleBitmap(dc, W, H);
    HBITMAP old_bmp = (HBITMAP)SelectObject(mdc, bmp);

    HBRUSH bg = CreateSolidBrush(theme_.panel);
    FillRect(mdc, &rc, bg);
    DeleteObject(bg);

    // Top hairline accent.
    HBRUSH acc = CreateSolidBrush(theme_.accent);
    RECT line = {0, 0, W, 1};
    FillRect(mdc, &line, acc);
    DeleteObject(acc);

    HFONT old_font = (HFONT)SelectObject(mdc, font_);
    SetBkMode(mdc, TRANSPARENT);

    // Snapshot stats under the lock.
    PerfStats s;
    std::vector<float> hist;
    {
        std::lock_guard lk(stats_mutex_);
        s    = stats_;
        hist = history_copy_;
    }

    int y = kPaddingY;
    auto draw_row = [&](COLORREF col, std::string_view text) {
        SetTextColor(mdc, col);
        auto wbuf = ToWide(text);
        if (!wbuf.empty()) {
            TextOutW(mdc, kPaddingX, y, wbuf.data(), (int)wbuf.size());
        }
        y += line_height_;
    };

    // Tier-1 lines (always shown when visible).
    {
        std::string fps_line = fmt::format("FPS  {:>5.0f}",  s.fps);
        std::string ms_line  = fmt::format("ms   {:>5.2f}  ({:.2f} / {:.2f})",
                                           s.frame_ms_avg,
                                           s.frame_ms_min,
                                           s.frame_ms_max);
        draw_row(theme_.accent, fps_line);
        draw_row(theme_.text,   ms_line);
    }

    int level = Level();
    if (level >= 2) {
        std::string be   = fmt::format("backend     {}",      s.backend ? s.backend : "");
        std::string res  = fmt::format("resolution  {}x{}",   s.width, s.height);
        double mem_mb    = double(s.gpu_bytes) / (1024.0 * 1024.0);
        std::string mem  = fmt::format("gpu memory  {:.1f} MB", mem_mb);
        std::string spp  = fmt::format("spp         {}",       s.spp);
        std::string bnc  = fmt::format("bounces     {}",       s.max_bounces);
        std::string prim = fmt::format("primitives  {}",       s.primitives);
        draw_row(theme_.dim,  be);
        draw_row(theme_.dim,  res);
        draw_row(theme_.dim,  mem);
        draw_row(theme_.dim,  spp);
        draw_row(theme_.dim,  bnc);
        draw_row(theme_.dim,  prim);
    }

    if (level >= 3 && !hist.empty()) {
        // Frame-time sparkline.  Y maps from ms range [0 .. peak]
        // where peak = max(history) clamped to a sane minimum so a
        // perfectly flat trace doesn't divide by zero.
        int gx = kPaddingX;
        int gy = y + 4;
        int gw = W - 2 * kPaddingX;
        int gh = graph_h_;

        // Frame the graph area with a 1px border.
        HBRUSH frame = CreateSolidBrush(theme_.dim);
        RECT r1{gx,        gy,            gx + gw, gy + 1};
        RECT r2{gx,        gy + gh - 1,   gx + gw, gy + gh};
        RECT r3{gx,        gy,            gx + 1,  gy + gh};
        RECT r4{gx + gw-1, gy,            gx + gw, gy + gh};
        FillRect(mdc, &r1, frame);
        FillRect(mdc, &r2, frame);
        FillRect(mdc, &r3, frame);
        FillRect(mdc, &r4, frame);
        DeleteObject(frame);

        float peak = 1.0f;  // 1ms minimum span
        for (float v : hist) peak = std::max(peak, v);
        peak = std::max(peak, 0.5f);

        // 60fps reference line at 16.67 ms (or scaled if peak smaller).
        float ref_ms = 16.667f;
        if (ref_ms < peak) {
            int ref_y = gy + gh - 1 - int((ref_ms / peak) * (gh - 2));
            HBRUSH ref_brush = CreateSolidBrush(theme_.dim);
            RECT rr{gx + 1, ref_y, gx + gw - 1, ref_y + 1};
            FillRect(mdc, &rr, ref_brush);
            DeleteObject(ref_brush);
        }

        // Draw the line as a polyline.  Sample count <= gw - 2
        // (one pixel per X; older samples may be subsampled).
        int avail = std::max(2, gw - 2);
        int n     = (int)hist.size();
        std::vector<POINT> pts;
        pts.reserve(std::min(n, avail));
        for (int i = 0; i < std::min(n, avail); ++i) {
            int hi = n - 1 - (avail - 1 - i);
            if (hi < 0) hi = 0;
            float v = hist[hi];
            int px = gx + 1 + i;
            int py = gy + gh - 1 - int((v / peak) * (gh - 2));
            pts.push_back(POINT{px, py});
        }
        if (pts.size() >= 2) {
            HPEN pen = CreatePen(PS_SOLID, 1, theme_.graph);
            HPEN old_pen = (HPEN)SelectObject(mdc, pen);
            Polyline(mdc, pts.data(), (int)pts.size());
            SelectObject(mdc, old_pen);
            DeleteObject(pen);
        }
    }

    SelectObject(mdc, old_font);
    BitBlt(dc, 0, 0, W, H, mdc, 0, 0, SRCCOPY);
    SelectObject(mdc, old_bmp);
    DeleteObject(bmp);
    DeleteDC(mdc);
}

}  // namespace

// ---- Public PerfOverlay wrappers ----------------------------------------

PerfOverlay::PerfOverlay()  : opaque_(new WinPerf) {}
PerfOverlay::~PerfOverlay() {
    if (opaque_) static_cast<WinPerf*>(opaque_)->Shutdown();
    delete static_cast<WinPerf*>(opaque_);
    opaque_ = nullptr;
}
bool PerfOverlay::Init(void* hwnd) {
    if (!opaque_ || !hwnd) return false;
    return static_cast<WinPerf*>(opaque_)->Init(static_cast<HWND>(hwnd));
}
void PerfOverlay::Shutdown()                        { if (opaque_) static_cast<WinPerf*>(opaque_)->Shutdown(); }
void PerfOverlay::SetLevel(int l)                   { if (opaque_) static_cast<WinPerf*>(opaque_)->SetLevel(l); }
int  PerfOverlay::Level() const                     { return opaque_ ? static_cast<WinPerf*>(opaque_)->Level() : 0; }
void PerfOverlay::Update(const PerfStats& s)        { if (opaque_) static_cast<WinPerf*>(opaque_)->Update(s); }
void PerfOverlay::NotifyParentResized(int w, int h) { if (opaque_) static_cast<WinPerf*>(opaque_)->NotifyParentResized(w, h); }
void PerfOverlay::ApplyTheme(std::string_view n)    { if (opaque_) static_cast<WinPerf*>(opaque_)->ApplyTheme(n); }
void PerfOverlay::Repaint()                         { if (opaque_) static_cast<WinPerf*>(opaque_)->Repaint(); }

}  // namespace pt::app

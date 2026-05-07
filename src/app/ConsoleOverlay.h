// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Native macOS console overlay.  Implemented as an NSView subclass that
// sits on top of the GLFW window's content view, drawing with CoreText
// over an NSVisualEffectView.  Toggled by backtick (Engine wires it).
//
// The HARD rule against "GUI libraries" rules out Dear ImGui / Qt etc.
// CoreText / CoreGraphics / NSVisualEffectView are part of macOS itself,
// not bundled libraries -- equivalent to using `sysctl` for hardware
// introspection.

#include "../core/Log.h"

#include <string_view>

namespace pt::app {

class ConsoleOverlay {
public:
    ConsoleOverlay();
    ~ConsoleOverlay();
    ConsoleOverlay(const ConsoleOverlay&) = delete;
    ConsoleOverlay& operator=(const ConsoleOverlay&) = delete;

    // ns_window is the NSWindow* obtained from glfwGetCocoaWindow.  Init
    // attaches the overlay to its content view and registers log sink
    // forwarding.  Returns false on platforms other than macOS.
    bool Init(void* ns_window);
    void Shutdown();

    void Show();
    void Hide();
    void Toggle();
    bool IsShown() const;

    // Apply one of the named themes (matches the r_theme cvar values:
    // hardcore | amber | synthwave | matrix | vault | sakura | mono).
    // Updates the panel border, prompt, status label colour, and
    // re-renders the ASCII banner with the theme's accent palette;
    // future appended lines also use the new palette.
    void ApplyTheme(std::string_view name);

    // Called by Engine on the GLFW resize callback so the floating panel
    // tracks the renderer window.
    void NotifyParentResized(int width, int height);

    // Forwarded by pt::log -> our sink so log lines appear in the overlay.
    static void OnLog(pt::log::Level level, const std::string& body);
    static void SetGlobalInstance(ConsoleOverlay* o);

private:
    void* opaque_ = nullptr;   // (PtConsoleView*) on macOS, opaque elsewhere
};

}  // namespace pt::app

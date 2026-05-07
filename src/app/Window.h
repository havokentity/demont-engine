// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>
#include <functional>
#include <string>

struct GLFWwindow;

namespace pt::app {

class Window {
public:
    Window();
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Always created with no graphics API context attached -- backends
    // attach their own CAMetalLayer (Metal, Software-via-Metal-present)
    // or VkSurface (Vulkan via MoltenVK).
    bool Create(int w, int h, std::string_view title);
    void Destroy();

    void PollEvents();
    bool ShouldClose() const;
    void RequestClose();

    int  Width()  const noexcept { return width_;  }
    int  Height() const noexcept { return height_; }

    GLFWwindow* Handle() const noexcept { return handle_; }

    // Returns the platform-native handle (NSWindow* on macOS, HWND on
    // Windows).  Currently unused outside of future RHI backends.
    void* NativeHandle() const;

    // Engine-installed handler for hotkeys. Called for press events with
    // (key, mods) using the GLFW_KEY_* enums.  Esc-to-close is handled
    // before this hook fires.
    using KeyHandler = std::function<void(int key, int mods)>;
    void SetKeyHandler(KeyHandler h);

    // Polled input state.  All key/button codes are GLFW_KEY_* / GLFW_MOUSE_*
    // constants -- see GLFW/glfw3.h.
    bool IsKeyDown(int glfw_key) const;
    bool IsMouseButtonDown(int glfw_button) const;

    // Cursor mode: GLFW_CURSOR_NORMAL, GLFW_CURSOR_HIDDEN, GLFW_CURSOR_DISABLED.
    // DISABLED locks the cursor and feeds unbounded delta -- used for
    // mouse-look while right-click is held.
    void SetCursorMode(int mode);
    int  CursorMode() const;

    // Returns (dx, dy) since the last call.  First call after mode change
    // returns 0 to avoid a one-frame jump.
    void ConsumeMouseDelta(double& dx, double& dy);

    // Returns scroll-wheel delta accumulated since the last call. +Y =
    // scroll up (toward user). Engines typically map this to camera
    // speed adjustment or zoom.
    void ConsumeScrollDelta(double& dx, double& dy);

private:
    static void OnResize(GLFWwindow* w, int width, int height);
    static void OnKey(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void OnScroll(GLFWwindow* w, double dx, double dy);

    GLFWwindow* handle_ = nullptr;
    int width_  = 0;
    int height_ = 0;
    std::string title_;
    KeyHandler  key_handler_;

    // Mouse delta accumulators
    double cursor_last_x_ = 0.0;
    double cursor_last_y_ = 0.0;
    bool   cursor_have_baseline_ = false;

    // Scroll-wheel accumulators. GLFW callback adds the per-event
    // delta; ConsumeScrollDelta returns the sum and resets.
    double scroll_accum_x_ = 0.0;
    double scroll_accum_y_ = 0.0;
};

}  // namespace pt::app

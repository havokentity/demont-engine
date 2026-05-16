// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Window.h"
#include "../core/Log.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace pt::app {

namespace {

bool g_glfw_inited = false;
int  g_window_count = 0;

void GlfwErrorCallback(int code, const char* desc) {
    LOG_ERROR("GLFW error {}: {}", code, desc ? desc : "(null)");
}

bool EnsureGlfw() {
    if (g_glfw_inited) return true;
    glfwSetErrorCallback(&GlfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
        LOG_ERROR("glfwInit failed");
        return false;
    }
    g_glfw_inited = true;
    return true;
}

}  // namespace

Window::Window() = default;
Window::~Window() { Destroy(); }

bool Window::Create(int w, int h, std::string_view title) {
    if (!EnsureGlfw()) return false;
    Destroy();
    title_.assign(title);

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CLIENT_API,               GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,                GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE,                  GLFW_TRUE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    handle_ = glfwCreateWindow(w, h, title_.c_str(), nullptr, nullptr);
    if (handle_ == nullptr) {
        LOG_ERROR("glfwCreateWindow({},{}) failed", w, h);
        return false;
    }
    width_  = w;
    height_ = h;
    glfwSetWindowUserPointer(handle_, this);
    glfwSetFramebufferSizeCallback(handle_, &Window::OnResize);
    glfwSetKeyCallback(handle_, &Window::OnKey);
    glfwSetScrollCallback(handle_, &Window::OnScroll);

    ++g_window_count;
    return true;
}

void Window::Destroy() {
    if (handle_ != nullptr) {
        glfwDestroyWindow(handle_);
        handle_ = nullptr;
        if (--g_window_count == 0 && g_glfw_inited) {
            glfwTerminate();
            g_glfw_inited = false;
        }
    }
}

void Window::PollEvents() {
    if (g_glfw_inited) glfwPollEvents();
}

// Win32 GLFW native declaration is set up later in the file; forward-
// declare here so Recreate() can call it. The macro guard mirrors the
// definition site below.
#if defined(_WIN32)
extern "C" void* pt_window_native_win32(void* glfw_window);
#endif

bool Window::Recreate() {
#if defined(_WIN32)
    if (handle_ == nullptr) {
        LOG_ERROR("Window::Recreate: called with no live handle");
        return false;
    }
    int pos_x = 0, pos_y = 0;
    glfwGetWindowPos(handle_, &pos_x, &pos_y);
    const int saved_w            = width_;
    const int saved_h            = height_;
    const int saved_cursor_mode  = glfwGetInputMode(handle_, GLFW_CURSOR);
    const std::string saved_t    = title_;
    void* old_native             = pt_window_native_win32(handle_);

    LOG_INFO("Window::Recreate: tearing down GLFW window (HWND={}); preserving {}x{} pos {},{} cursor_mode={}",
             old_native, saved_w, saved_h, pos_x, pos_y, saved_cursor_mode);

    Destroy();
    if (!Create(saved_w, saved_h, saved_t)) {
        LOG_ERROR("Window::Recreate: Create failed after Destroy; window is now in an unusable state");
        return false;
    }
    glfwSetWindowPos(handle_, pos_x, pos_y);
    glfwSetInputMode(handle_, GLFW_CURSOR, saved_cursor_mode);
    // Re-baseline polled accumulators against the fresh GLFW state so
    // the first ConsumeMouseDelta after recreate returns 0 instead of
    // a teleport jump.
    cursor_have_baseline_ = false;
    scroll_accum_x_       = 0.0;
    scroll_accum_y_       = 0.0;

    LOG_INFO("Window::Recreate: created new GLFW window (HWND={})",
             pt_window_native_win32(handle_));
    return true;
#else
    LOG_ERROR("Window::Recreate: Win32-only (called on a non-Win32 build)");
    return false;
#endif
}

bool Window::ShouldClose() const {
    return handle_ != nullptr && glfwWindowShouldClose(handle_);
}

void Window::RequestClose() {
    if (handle_ != nullptr) glfwSetWindowShouldClose(handle_, GLFW_TRUE);
}

void Window::Hide() {
    if (handle_ != nullptr) glfwHideWindow(handle_);
}

// Signature is `void*(void*)` everywhere -- the implementation in
// Window_Cocoa.mm casts back to GLFWwindow* internally.  Standardising
// on void* avoids an ODR violation across translation units that
// don't include GLFW headers (ConsoleOverlay_Win32, ConsoleOverlay_Stub,
// MetalDevice, SoftwareDevice all declare it as void*(void*)).
extern "C" void* pt_window_native_cocoa(void*);

#if defined(_WIN32)
#  define GLFW_EXPOSE_NATIVE_WIN32
#  include <GLFW/glfw3native.h>

// Win32 symmetric counterpart to pt_window_native_cocoa: takes a
// GLFWwindow* (as void*) and returns the underlying HWND (as void*).
// Used by SoftwareDevice's GDI present path -- same call shape as the
// Mac side so the backend doesn't need to drag GLFW headers in.
extern "C" void* pt_window_native_win32(void* glfw_window) {
    return glfw_window
        ? static_cast<void*>(glfwGetWin32Window(
              static_cast<GLFWwindow*>(glfw_window)))
        : nullptr;
}
#endif

void* Window::NativeHandle() const {
#if defined(__APPLE__)
    return pt_window_native_cocoa(handle_);
#elif defined(_WIN32)
    // Returns HWND. ConsoleOverlay_Win32 attaches a child window
    // here for the in-game console. Cast to HWND on the consumer
    // side; void* keeps Window.h free of <Windows.h>.
    return handle_ ? static_cast<void*>(glfwGetWin32Window(handle_)) : nullptr;
#else
    return nullptr;
#endif
}

void Window::OnResize(GLFWwindow* w, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self == nullptr) return;
    self->width_  = width;
    self->height_ = height;
}

void Window::SetKeyHandler(KeyHandler h) { key_handler_ = std::move(h); }

bool Window::IsKeyDown(int key) const {
    return handle_ != nullptr && glfwGetKey(handle_, key) == GLFW_PRESS;
}

bool Window::IsMouseButtonDown(int btn) const {
    return handle_ != nullptr && glfwGetMouseButton(handle_, btn) == GLFW_PRESS;
}

void Window::SetCursorMode(int mode) {
    if (handle_ == nullptr) return;
    int cur = glfwGetInputMode(handle_, GLFW_CURSOR);
    if (cur == mode) return;
    glfwSetInputMode(handle_, GLFW_CURSOR, mode);
    cursor_have_baseline_ = false;   // re-baseline on next ConsumeMouseDelta
}

int Window::CursorMode() const {
    if (handle_ == nullptr) return GLFW_CURSOR_NORMAL;
    return glfwGetInputMode(handle_, GLFW_CURSOR);
}

void Window::ConsumeMouseDelta(double& dx, double& dy) {
    dx = dy = 0.0;
    if (handle_ == nullptr) return;
    double x, y;
    glfwGetCursorPos(handle_, &x, &y);
    if (!cursor_have_baseline_) {
        cursor_last_x_ = x;
        cursor_last_y_ = y;
        cursor_have_baseline_ = true;
        return;
    }
    dx = x - cursor_last_x_;
    dy = y - cursor_last_y_;
    cursor_last_x_ = x;
    cursor_last_y_ = y;
}

void Window::ConsumeScrollDelta(double& dx, double& dy) {
    dx = scroll_accum_x_;
    dy = scroll_accum_y_;
    scroll_accum_x_ = 0.0;
    scroll_accum_y_ = 0.0;
}

void Window::OnScroll(GLFWwindow* w, double dx, double dy) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self == nullptr) return;
    self->scroll_accum_x_ += dx;
    self->scroll_accum_y_ += dy;
}

void Window::OnKey(GLFWwindow* w, int key, int /*scancode*/, int action, int mods) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(w, GLFW_TRUE);
        return;
    }
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->key_handler_) {
        self->key_handler_(key, mods);
    }
}

}  // namespace pt::app

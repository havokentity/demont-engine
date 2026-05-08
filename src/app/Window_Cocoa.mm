// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// macOS-only bridge from GLFWwindow* to NSWindow*. Kept in a separate .mm
// translation unit so the rest of the app can compile as plain C++.

#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// Parameter typed as void* to match the unified `extern "C"` declaration
// across translation units that don't include GLFW headers (Stub/Win32
// overlay sources, the MetalDevice / SoftwareDevice shims). Internally
// we cast back to GLFWwindow* before calling glfwGetCocoaWindow.
extern "C" void* pt_window_native_cocoa(void* w) {
    if (w == nullptr) return nullptr;
    return (__bridge void*)glfwGetCocoaWindow(static_cast<GLFWwindow*>(w));
}

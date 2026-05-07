// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// macOS-only bridge from GLFWwindow* to NSWindow*. Kept in a separate .mm
// translation unit so the rest of the app can compile as plain C++.

#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

extern "C" void* pt_window_native_cocoa(GLFWwindow* w) {
    if (w == nullptr) return nullptr;
    return (__bridge void*)glfwGetCocoaWindow(w);
}

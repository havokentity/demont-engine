// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Cross-platform no-op implementation of the Cocoa overlay + Metal
// layer attachment shims. Compiled on every non-Apple build so the
// engine can keep including ConsoleOverlay.h and calling these C
// shims unconditionally; they just do nothing on Windows / Linux.
//
// On non-Apple targets the Vulkan backend handles surface creation
// itself (vkCreateXcbSurfaceKHR / vkCreateWin32SurfaceKHR), so the
// pt_metal_attach_layer hook never has work to do. The overlay is a
// macOS-native NSView feature; on other platforms we lean on the web
// console (http://127.0.0.1:27960) for live cvar editing and use the
// engine's terminal logging instead.

#include "ConsoleOverlay.h"

namespace pt::app {

ConsoleOverlay::ConsoleOverlay()  {}
ConsoleOverlay::~ConsoleOverlay() {}

bool ConsoleOverlay::Init(void*) { return false; }
void ConsoleOverlay::Shutdown()  {}

void ConsoleOverlay::Show()                                    {}
void ConsoleOverlay::Hide()                                    {}
void ConsoleOverlay::Toggle()                                  {}
bool ConsoleOverlay::IsShown() const                           { return false; }
void ConsoleOverlay::ApplyTheme(std::string_view)              {}
void ConsoleOverlay::NotifyParentResized(int, int)             {}
void ConsoleOverlay::OnLog(pt::log::Level, const std::string&) {}
void ConsoleOverlay::SetGlobalInstance(ConsoleOverlay*)        {}

}  // namespace pt::app

// C-extern shims expected by Window.cpp / SoftwareDevice.cpp /
// MetalDevice.cpp. On Apple these come from Window_Cocoa.mm and
// MetalLayerAttach.mm. On non-Apple, no-ops.
extern "C" void* pt_window_native_cocoa(void*) { return nullptr; }
extern "C" void  pt_metal_attach_layer(void*, void*) {}

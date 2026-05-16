// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// No-op implementation of the Cocoa overlay + Metal layer attachment
// shims for non-Apple, non-Windows targets. This stub lets the engine
// keep including ConsoleOverlay.h and calling these C shims
// unconditionally where no native overlay implementation exists.
//
// On these targets the Vulkan backend handles surface creation itself
// (for example, via platform-specific vkCreate*SurfaceKHR entry
// points), so the pt_metal_attach_layer hook never has work to do.
// The overlay is a macOS-native NSView feature; elsewhere we lean on
// the web console (http://127.0.0.1:27960) for live cvar editing and
// use the engine's terminal logging instead.

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
bool ConsoleOverlay::SaveState(const std::string&) const       { return false; }
bool ConsoleOverlay::LoadState(const std::string&)             { return false; }

}  // namespace pt::app

// C-extern shims expected by Window.cpp / SoftwareDevice.cpp /
// MetalDevice.cpp. On Apple these come from Window_Cocoa.mm and
// MetalLayerAttach.mm. On non-Apple, no-ops.
extern "C" void* pt_window_native_cocoa(void*) { return nullptr; }
extern "C" void  pt_metal_attach_layer(void*, void*) {}

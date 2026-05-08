// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Native in-game performance overlay.  Tiered, gated by `r_perf_overlay`:
//
//   0 = off (hidden).
//   1 = basic   : fps + frame_ms (avg / min / max over the last second).
//   2 = detailed: + backend / resolution / GPU memory / spp / bounces /
//                   primitive count.
//   3 = graph   : + sparkline of recent frame_ms history.
//
// Implemented as a sibling child window to the engine's GLFW HWND
// (Win32) or floating NSPanel (Mac), painted with the OS's native text
// stack (GDI on Win, NSAttributedString on Mac).  Same architectural
// shape as ConsoleOverlay -- the rule against bundled GUI libraries
// (Dear ImGui / Qt / etc.) holds; we use only what the OS ships.
//
// GPU per-pass timing is intentionally NOT in tier 1-3.  Wiring
// VkQueryPool / MTLCounterSampleBuffer through every dispatch is a
// separate piece of work; once it lands it'll slot in as tier 4
// without changing the cvar's allowed_values shape used by the
// existing 0..3 levels.

#include <cstddef>
#include <span>
#include <string_view>

namespace pt::app {

struct PerfStats {
    double      fps          = 0.0;
    double      frame_ms_avg = 0.0;
    double      frame_ms_min = 0.0;
    double      frame_ms_max = 0.0;
    const char* backend      = "";       // "metal" / "vulkan" / "software" / "none"
    int         width        = 0;
    int         height       = 0;
    std::size_t gpu_bytes    = 0;
    int         spp          = 1;
    int         max_bounces  = 8;
    std::size_t primitives   = 0;
    // Recent frame_ms samples, oldest-first.  Used by tier-3 sparkline.
    std::span<const float> frame_ms_history;
};

class PerfOverlay {
public:
    PerfOverlay();
    ~PerfOverlay();
    PerfOverlay(const PerfOverlay&)            = delete;
    PerfOverlay& operator=(const PerfOverlay&) = delete;

    // native_handle is HWND on Win32, NSWindow* on Mac.  Returns false
    // on platforms without an implementation.
    bool Init(void* native_handle);
    void Shutdown();

    // Tier 0..3.  Visibility is implicit: level 0 hides, anything else
    // shows.  Out-of-range values clamp.
    void SetLevel(int level);
    int  Level() const;

    // Push the latest stats; called from the engine main loop after
    // RenderFrame.  The implementation re-reads its own visibility
    // state internally, so a no-op when level is 0.
    void Update(const PerfStats& stats);

    // Engine forwards the GLFW resize callback so the overlay can
    // re-anchor to the parent client rect.
    void NotifyParentResized(int w, int h);

    // Apply one of the named themes (matches r_theme cvar values).
    void ApplyTheme(std::string_view name);

private:
    void* opaque_ = nullptr;
};

}  // namespace pt::app

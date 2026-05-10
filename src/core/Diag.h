// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Log.h"

#include <atomic>
#include <fmt/format.h>

namespace pt::diag {

// Cached value of `r_diagnostic_level`. Engine.cpp's cvar registration
// installs an on_change handler that updates this; the default of 1
// means PT_DIAG_TIER1 sites that fire before the cvar registers (or in
// builds without the engine) still emit.
//
// Atomic because the writer (cvar on_change, main thread / Console::Drain)
// and the readers (PT_DIAG_TIERn at any callsite, including async pipeline
// build worker threads, the engine job system, render-thread barriers)
// race. Plain int reads + writes would be data-race UB in the C++ memory
// model regardless of the value range. Relaxed ordering is enough: we
// only need single-variable atomicity, no synchronisation against other
// state.
//
// Tiers:
//   0 = off            -- LOG_ERROR / LOG_WARN still flow; tiered diags suppressed
//   1 = state-trans    -- init / shutdown / fallback / cvar change / mode switch.
//                         One-shot events. Production default.
//   2 = per-frame sum  -- <=1 line per frame per category (cache hit-rate, etc).
//   3 = per-call       -- every encode / dispatch / barrier. Dev-only.
//
// `category` is a short string literal -- "denoiser", "rhi.vulkan",
// "engine.cvars", etc. Printed as a [bracket] prefix so `grep [denoiser]`
// filters down to one subsystem. Per-category tier filtering can be a
// follow-up; tier-only is enough for v1.
extern std::atomic<int> g_diag_level;

// Cheap tier-check helper for callers that want to gate expensive
// pre-format work (e.g. building a per-pipeline timing string) outside
// the macro body. Same relaxed load as the macros.
inline bool TierEnabled(int tier) noexcept {
    return g_diag_level.load(std::memory_order_relaxed) >= tier;
}

}  // namespace pt::diag

// Runtime-gated, NOT compile-time. The whole point of r_diagnostic_level
// is that the user can dial it from the console without rebuilding, so
// we keep all three tiers in the binary and branch on g_diag_level. Hot
// path when off is a single relaxed atomic load + compare; the format-
// args evaluation (which can be expensive) is short-circuited.
//
// Macros expand to a `do { ... } while (0)` so they parse correctly as
// statements in `if (...) PT_DIAG_TIER1(...);` without an else binding
// surprise. The inner fmt::format runs only when the gate passes.
#define PT_DIAG_TIER1(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::TierEnabled(1)) {                                     \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

#define PT_DIAG_TIER2(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::TierEnabled(2)) {                                     \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

#define PT_DIAG_TIER3(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::TierEnabled(3)) {                                     \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

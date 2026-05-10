// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Log.h"

#include <fmt/format.h>

namespace pt::diag {

// Cached value of `r_diagnostic_level`. Engine.cpp's cvar registration
// installs an on_change handler that updates this; the default of 1
// means PT_DIAG_TIER1 sites that fire before the cvar registers (or in
// builds without the engine) still emit.
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
extern int g_diag_level;

}  // namespace pt::diag

// Runtime-gated, NOT compile-time. The whole point of r_diagnostic_level
// is that the user can dial it from the console without rebuilding, so
// we keep all three tiers in the binary and branch on g_diag_level. Hot
// path when off is a single load + compare; the format-args evaluation
// (which can be expensive) is short-circuited.
//
// Macros expand to a `do { ... } while (0)` so they parse correctly as
// statements in `if (...) PT_DIAG_TIER1(...);` without an else binding
// surprise. The inner fmt::format runs only when the gate passes.
#define PT_DIAG_TIER1(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::g_diag_level >= 1) {                                  \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

#define PT_DIAG_TIER2(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::g_diag_level >= 2) {                                  \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

#define PT_DIAG_TIER3(category, ...)                                          \
    do {                                                                      \
        if (::pt::diag::g_diag_level >= 3) {                                  \
            ::pt::log::Info("[{}] {}", (category),                            \
                            ::fmt::format(__VA_ARGS__));                      \
        }                                                                     \
    } while (0)

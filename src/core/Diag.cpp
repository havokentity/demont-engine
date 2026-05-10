// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Diag.h"

namespace pt::diag {

// Initial value matches the cvar default ("1" -- state-transition).
// Engine.cpp's cvar registration installs an on_change that stores
// into this atomic when the user runs `r_diagnostic_level <n>`.
//
// Atomic + relaxed ordering: the on_change writer (main thread,
// Console::Drain context) and PT_DIAG_TIERn readers (any thread,
// including the Vulkan async pipeline build worker, job system
// workers, etc.) race. Per the C++ memory model any concurrent
// non-atomic read+write would be data-race UB even for a small int
// range like {0,1,2,3}. We only need single-variable atomicity, not
// synchronisation against other state, so relaxed is enough.
std::atomic<int> g_diag_level{1};

}  // namespace pt::diag

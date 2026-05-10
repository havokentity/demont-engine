// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Diag.h"

namespace pt::diag {

// Initial value matches the cvar default ("1" -- state-transition).
// Engine.cpp's cvar registration installs an on_change that mutates this
// in-place when the user runs `r_diagnostic_level <n>`. Plain int store
// is fine: the macro hot path is a load + compare, and a torn read of
// {0,1,2,3} can only ever yield another value in the same set, so the
// worst case is a single tier1 line emitted at tier 0 (or vice versa).
int g_diag_level = 1;

}  // namespace pt::diag

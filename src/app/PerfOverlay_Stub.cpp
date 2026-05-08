// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// No-op PerfOverlay for platforms without a native implementation.
// Init returns false so the engine treats the overlay as disabled
// and skips per-frame updates.

#include "PerfOverlay.h"

namespace pt::app {

PerfOverlay::PerfOverlay()  = default;
PerfOverlay::~PerfOverlay() = default;

bool PerfOverlay::Init(void*)                       { return false; }
void PerfOverlay::Shutdown()                        {}
void PerfOverlay::SetLevel(int)                     {}
int  PerfOverlay::Level() const                     { return 0; }
void PerfOverlay::Update(const PerfStats&)          {}
void PerfOverlay::NotifyParentResized(int, int)     {}
void PerfOverlay::ApplyTheme(std::string_view)      {}

}  // namespace pt::app

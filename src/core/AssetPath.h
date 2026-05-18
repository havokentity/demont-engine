// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <string>

namespace pt {

// Resolve a repo-relative asset path (e.g. "assets/hdri/sunset.hdr") to
// an absolute filesystem path. Consults DEMONT_ASSET_ROOT first; falls
// back to the directory two levels above the running executable
// (covers both packaged installs and dev builds where the wrapper sets
// the env var explicitly). The resolved root is cached on first call.
std::string ResolveAssetPath(const char* relative);

}  // namespace pt

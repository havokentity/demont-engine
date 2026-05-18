// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "AssetPath.h"

#include "Log.h"

#include <cstdlib>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <climits>
#  include <unistd.h>
#endif

namespace pt {

namespace {

// One-shot resolution of the asset root. Returns DEMONT_ASSET_ROOT when
// set and non-empty, otherwise walks up from the running exe looking for
// a directory containing one of: `assets/` (packaged install or dev tree),
// `.git/` (dev checkout root), or `CMakeLists.txt` (project root). The
// walk is necessary because the exe's depth relative to the asset root
// varies by layout:
//   packaged install:           <root>/bin/demont               (1 up)
//   IDE direct dev run:         <root>/build/<preset>/src/app/  (4 up)
//   ctest cell:                 <root>/build/<preset>/golden_workdir/<cell>/ (4 up)
//   pt_render_one_frame wrapper sets DEMONT_ASSET_ROOT explicitly so its
//   child demont never falls through here.
std::filesystem::path ResolveRootOnce(const char*& source_out) {
    namespace fs = std::filesystem;

    if (const char* env = std::getenv("DEMONT_ASSET_ROOT");
        env != nullptr && env[0] != '\0') {
        source_out = "env";
        return fs::path(env);
    }

#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        // Fall back to CWD on failure -- preserves the old behaviour
        // rather than crashing or returning a bogus path.
        LOG_WARN("AssetPath: GetModuleFileNameW failed (GLE={}, n={}); "
                 "asset root falls back to CWD", GetLastError(), n);
        source_out = "cwd-fallback";
        return fs::path();
    }
    fs::path exe(buf, buf + n);
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        LOG_WARN("AssetPath: readlink(/proc/self/exe) failed (errno={}); "
                 "asset root falls back to CWD", errno);
        source_out = "cwd-fallback";
        return fs::path();
    }
    buf[n] = '\0';
    fs::path exe(buf);
#endif

    // Walk up from the exe dir looking for a project / install marker.
    // Bounded at 8 levels; normal layouts need at most 4.
    std::error_code ec;
    fs::path dir = exe.parent_path();
    for (int depth = 0; depth < 8; ++depth) {
        if (fs::is_directory(dir / "assets", ec)) {
            source_out = "marker:assets";
            return dir;
        }
        if (fs::exists(dir / ".git", ec)) {
            source_out = "marker:.git";
            return dir;
        }
        if (fs::exists(dir / "CMakeLists.txt", ec)) {
            source_out = "marker:CMakeLists.txt";
            return dir;
        }
        const fs::path parent = dir.parent_path();
        if (parent == dir) break;  // filesystem root
        dir = parent;
    }

    // No marker found in 8 levels -- preserve the legacy exe-grandparent
    // behaviour so packaged installs that lack any marker still resolve
    // approximately as before.
    source_out = "exe-grandparent-fallback";
    return exe.parent_path().parent_path();
}

const std::filesystem::path& CachedRoot() {
    static const std::filesystem::path root = []() {
        const char* source = "unknown";
        auto r = ResolveRootOnce(source);
        LOG_INFO("AssetPath: asset root resolved to '{}' (source: {})",
                 r.string(), source);
        return r;
    }();
    return root;
}

}  // namespace

std::string ResolveAssetPath(const char* relative) {
    const auto& root = CachedRoot();
    if (root.empty()) {
        // CWD-fallback path: hand the relative string back unchanged so
        // legacy CWD-relative resolution still works.
        return std::string(relative);
    }
    return (root / relative).string();
}

}  // namespace pt

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstddef>
#include <string_view>

// Lookup table over the embedded React+Vite editor bundle. The
// implementation is auto-generated at build time by
// cmake/EmbedEditorAssets.cmake which walks web/editor/dist/ and
// emits one entry per file. ConsoleServer's HTTP handler calls
// FindAsset() to map URL paths -> byte buffers; misses fall through
// to a 404.
//
// URLs are stored exactly as Vite emits them (with the leading
// slash), e.g.
//   "/panels/scene-hierarchy/index.html"
//   "/shared/assets/theme-CbS55YeC.css"
// The ConsoleServer rewrites short panel routes
// ("/editor/scene-hierarchy" -> "/panels/scene-hierarchy/index.html")
// before calling FindAsset.

namespace pt::editor {

struct EmbeddedAsset {
    const char* uri  = nullptr;   // null-terminated; leading-slash form
    const unsigned char* data = nullptr;
    unsigned long  size = 0;
    const char* mime = nullptr;   // e.g. "text/html; charset=utf-8"
};

// Returns nullptr if `uri` doesn't match any embedded asset.
//
// The returned pointer references a thread-local view -- safe to read
// any number of fields off it before the next FindAsset call on the
// same thread.
const EmbeddedAsset* FindAsset(std::string_view uri);

// Iteration helpers (used by `panels` command + debug logging).
std::size_t AssetCount();
EmbeddedAsset AssetAt(std::size_t index);

}  // namespace pt::editor

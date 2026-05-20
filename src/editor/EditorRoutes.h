// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <string>
#include <string_view>

namespace pt::editor {

struct EmbeddedAsset;

// Resolve an incoming HTTP URI to the matching embedded editor asset.
// Implements three shapes:
//   1. `/editor`                 -> 4-panel chooser index (synthesised)
//   2. `/editor/<panel>`         -> `/panels/<panel>/index.html`
//   3. anything else             -> direct lookup in the asset table
//      (Vite-emitted assets live under `/panels/...` and `/shared/...`)
// Returns nullptr if no match. The caller (ConsoleServer::HttpHandler)
// translates that into a 404.
const EmbeddedAsset* ResolveRoute(std::string_view uri);

// The known editor panels (matches web/editor/vite.config.ts).
// Useful for the `panels` console command and the autoopen cvar.
// Exposed as a contiguous null-terminated array so the call site
// doesn't have to know the count statically.
const char* const* KnownPanels();
std::size_t        KnownPanelCount();
bool               IsKnownPanel(std::string_view name);

// Build the URL the user should open in their browser for the given
// panel. Honors the r_editor_dev_mode cvar -- when set, returns the
// Vite dev server URL (http://localhost:5173/panels/<name>/);
// otherwise the embedded engine URL (http://localhost:<net_port>/editor/<name>).
//
// Caller supplies the network bind addr + port pulled from the
// net_bind_address / net_port cvars (we don't reach into Console
// from here so this stays unit-testable).
std::string BuildPanelUrl(std::string_view panel,
                          std::string_view host,
                          int port,
                          bool dev_mode);

}  // namespace pt::editor

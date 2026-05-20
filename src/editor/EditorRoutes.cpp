// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "EditorRoutes.h"
#include "EditorAssets.h"

#include <fmt/format.h>

#include <string>

namespace pt::editor {

namespace {

constexpr const char* kPanels[] = {
    "scene-hierarchy",
    "inspector",
    "asset-browser",
    "toolbar",
    "material-editor",
    "lights",
    nullptr,
};
constexpr std::size_t kPanelCount = sizeof(kPanels) / sizeof(kPanels[0]) - 1;

// Synthesised index page for `GET /editor`. Lists every panel as a
// link so a user who hit the bare /editor route can navigate to the
// individual panels without remembering their names. Not embedded in
// the React bundle on purpose -- this is the engine's own splash and
// doesn't need a React app behind it.
constexpr const char* kEditorIndexHtml = R"PT_HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeMonT Editor</title>
<style>
  body { background:#06080d; color:#e7eaf2; font-family:-apple-system,BlinkMacSystemFont,
         'Segoe UI',sans-serif; padding:40px; max-width:720px; margin:0 auto;
         line-height:1.5; }
  h1 { color:#00f0ff; font-weight:600; margin:0 0 8px; }
  p.lead { color:#6f7892; margin:0 0 24px; }
  ul { list-style:none; padding:0; margin:0; display:grid; gap:8px;
       grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); }
  li a { display:block; padding:12px 14px; background:#0c1018; border:1px solid #182032;
         border-radius:4px; color:#e7eaf2; text-decoration:none; transition:border-color .15s; }
  li a:hover { border-color:#00f0ff; }
  li a strong { color:#00f0ff; display:block; font-weight:500; margin-bottom:2px; }
  li a span { color:#6f7892; font-size:12px; }
  footer { color:#3b4257; font-size:11px; margin-top:32px; }
  code { background:#0c1018; padding:2px 6px; border-radius:3px; color:#ff3a8c; }
</style></head><body>
<h1>DeMonT Editor</h1>
<p class="lead">React+Vite multi-panel editor shell. Each panel runs in its own
Chrome --app window when spawned via the console (<code>panel_open &lt;name&gt;</code>).</p>
<ul>
  <li><a href="/editor/scene-hierarchy"><strong>Scene Hierarchy</strong>
      <span>Tree of prims / lights / SDF / CSG / smoke</span></a></li>
  <li><a href="/editor/inspector"><strong>Inspector</strong>
      <span>Property editor for the current selection</span></a></li>
  <li><a href="/editor/asset-browser"><strong>Asset Browser</strong>
      <span>Scenes, HDRIs, glTF imports -- click to load</span></a></li>
  <li><a href="/editor/toolbar"><strong>Toolbar</strong>
      <span>Gizmo mode, undo/redo, snap, transform space</span></a></li>
  <li><a href="/editor/material-editor"><strong>Material Editor</strong>
      <span>PBR material control: albedo / roughness / textures / emission</span></a></li>
  <li><a href="/editor/lights"><strong>Lights</strong>
      <span>Analytic light list + per-light inspector (photometric)</span></a></li>
</ul>
<footer>Console: <a href="/" style="color:#00f0ff">/</a></footer>
</body></html>
)PT_HTML";

EmbeddedAsset MakeEditorIndexAsset() {
    EmbeddedAsset a;
    a.uri  = "/editor";
    a.data = reinterpret_cast<const unsigned char*>(kEditorIndexHtml);
    // Length excludes the implicit terminating NUL.
    std::size_t n = 0;
    while (kEditorIndexHtml[n] != '\0') ++n;
    a.size = static_cast<unsigned long>(n);
    a.mime = "text/html; charset=utf-8";
    return a;
}

}  // namespace

const char* const* KnownPanels() { return kPanels; }
std::size_t        KnownPanelCount() { return kPanelCount; }

bool IsKnownPanel(std::string_view name) {
    for (std::size_t i = 0; i < kPanelCount; ++i) {
        if (name == kPanels[i]) return true;
    }
    return false;
}

const EmbeddedAsset* ResolveRoute(std::string_view uri) {
    // 1. Bare `/editor` -> the synthesised index.
    if (uri == "/editor" || uri == "/editor/") {
        thread_local EmbeddedAsset cache = MakeEditorIndexAsset();
        return &cache;
    }

    // 2. `/editor/<panel>` -> embedded `/panels/<panel>/index.html`.
    constexpr std::string_view kEditorPrefix = "/editor/";
    if (uri.starts_with(kEditorPrefix)) {
        std::string_view name = uri.substr(kEditorPrefix.size());
        // Strip an optional trailing slash so /editor/scene-hierarchy/
        // and /editor/scene-hierarchy both work.
        if (!name.empty() && name.back() == '/') name.remove_suffix(1);
        if (IsKnownPanel(name)) {
            // Build the embedded URI on the fly. We keep it in a
            // thread-local string so the returned EmbeddedAsset's uri
            // pointer stays valid through the response.
            thread_local std::string buf;
            buf.assign("/panels/");
            buf.append(name.data(), name.size());
            buf.append("/index.html");
            return FindAsset(buf);
        }
        return nullptr;
    }

    // 3. Direct lookup. Anything starting with `/panels/...` or
    //    `/shared/...` (everything else under dist/) goes here.
    return FindAsset(uri);
}

std::string BuildPanelUrl(std::string_view panel,
                          std::string_view host,
                          int port,
                          bool dev_mode) {
    std::string h(host);
    if (h.empty() || h == "0.0.0.0") h = "localhost";

    if (dev_mode) {
        // Vite dev server. Pass an engine_port query so the panel can
        // still find the WebSocket on the engine's HTTP port.
        return fmt::format("http://{}:5173/panels/{}/?engine_port={}",
                           h, panel, port);
    }
    return fmt::format("http://{}:{}/editor/{}", h, port, panel);
}

}  // namespace pt::editor

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# pt_editor -- editor-mode foundation library. Two contributing
# layers live here:
#
#   1. (agent-19) Scene-graph snapshot + WebSocket-side dispatch
#      helpers. SerializeScene returns a nlohmann::json describing
#      every analytic prim, light, SDF cluster, CSG root and smoke
#      emitter; pt_engine pulls this in for the list_scene WS handler
#      and the selection_change / scene_dirty event broadcasts.
#
#   2. (agent-20) URL router + embedded React+Vite editor bundle.
#      Engine's ConsoleServer::HttpHandler delegates `/editor/...`,
#      `/panels/...`, and `/shared/...` URIs to
#      pt::editor::ResolveRoute. The bundle bytes are appended to
#      this target lazily by Editor.cmake's pt_attach_editor_build()
#      helper so a fresh `cmake --build` invocation produces a
#      self-contained demont binary.
#
# Both layers share the `pt::editor::` namespace. The library is
# pulled by pt_engine since SerializeScene operates on a
# pt::engine::Engine& and the WebSocket dispatch path also needs to
# call into it -- that's a one-way pt_engine -> pt_editor edge;
# pt_editor does not depend back on pt_engine in CMake terms (it
# consumes Engine.h through include paths only, which works because
# pt_engine is header-only at the class declaration level).
add_library(pt_editor STATIC
    # agent-19: scene-graph snapshot helpers (consumed by pt_engine).
    SceneGraph.cpp
    # agent-20: URL router + (lazily attached) embedded React bundle.
    EditorRoutes.cpp
)

target_include_directories(pt_editor PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_editor
    PUBLIC  pt_core
    PUBLIC  pt_renderer
    PUBLIC  pt_physics
    PUBLIC  nlohmann_json::nlohmann_json
    # fmt is used by EditorRoutes::BuildPanelUrl for URL formatting.
    PRIVATE fmt::fmt
)

# agent-20: pull in the editor build glue (defines
# pt_attach_editor_build()) and attach the generated
# editor_assets.cpp -- one file per byte of web/editor/dist/ -- to
# pt_editor. Idempotent: cached on the editor source tree's mtime
# via add_custom_command DEPENDS.
include(${CMAKE_SOURCE_DIR}/cmake/Editor.cmake)
pt_attach_editor_build(pt_editor)

pt_set_compiler_options(pt_editor)

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# pt_editor -- editor-mode backend foundation (agent-19).
#
# Scene-graph snapshot + (forthcoming) WebSocket-side dispatch helpers
# for the editor agents (20 React panel, 21 native overlay, 22 3D
# gizmos). Kept as a sibling of pt_engine rather than nested inside it
# so future editor-only TUs (e.g. gizmo math, command undo recorder)
# can grow here without bloating pt_engine. The library is pulled by
# pt_engine since SerializeScene operates on a pt::engine::Engine& and
# the WebSocket dispatch path also needs to call into it -- that's a
# one-way pt_engine -> pt_editor edge; pt_editor does not depend back
# on pt_engine in CMake terms (it consumes Engine.h through include
# paths only, which works because pt_engine is header-only at the
# class declaration level).
add_library(pt_editor STATIC
    SceneGraph.cpp
)

target_include_directories(pt_editor PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_editor
    PUBLIC  pt_core
    PUBLIC  pt_renderer
    PUBLIC  pt_physics
    PUBLIC  nlohmann_json::nlohmann_json
)

pt_set_compiler_options(pt_editor)

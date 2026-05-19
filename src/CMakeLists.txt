# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_subdirectory(core)
# pt_editor -- editor-mode foundation library. Holds:
#   - agent-19: scene-graph snapshot (SerializeScene) consumed by
#     pt_engine's list_scene WS handler.
#   - agent-20: URL router + embedded React/Vite editor bundle,
#     consumed by pt_console's HTTP handler to serve `/editor/<panel>`
#     and the generated panel asset tree.
# Depends only on renderer + physics + nlohmann_json; built before
# pt_console so the console's HTTP handler can call
# pt::editor::ResolveRoute(), and before pt_engine so Engine.cpp can
# call pt::editor::SerializeScene.
add_subdirectory(editor)
add_subdirectory(console)
add_subdirectory(audio)
# Software backend supports Mac (CAMetalLayer present) and Windows
# (SetDIBitsToDevice present).  Linux/Wayland/X11 would need its own
# present path -- not implemented, so we gate the subdirectory to
# the two platforms that actually have a working SoftwareDevice
# EndFrame.  Future r_software_blit cvar will add a Vulkan-blit
# alternative on Windows that shares the engine's VkDevice; until
# then Windows uses GDI, Mac uses Metal.
if(APPLE OR WIN32)
    add_subdirectory(rhi_software)
endif()
# Metal backend is Apple-only by definition.
if(APPLE)
    add_subdirectory(rhi_metal)
endif()
if(PT_ENABLE_VULKAN_BACKEND)
    add_subdirectory(rhi_vulkan)
endif()
add_subdirectory(rhi)
add_subdirectory(renderer)
# Effects layer: CPU-side particle sim MVP (#82). Pure CPU, no RHI
# deps -- the engine owns the per-frame GPU upload + dispatch in
# RenderFrame, mirroring the StarsComposite pattern.
add_subdirectory(effects)
# Physics: Verlet/XPBD on analytic primitives (#131 roadmap). Pure CPU,
# no RHI deps -- engine writes back particle positions into the
# AnalyticPrim buffer each frame.
add_subdirectory(physics)
# Destruction: voxelisation + connectivity-graph fracture (#140 roadmap).
# Pure CPU; engine spawns analytic-prim boxes for chunks.
add_subdirectory(destruction)
add_subdirectory(app)
add_subdirectory(engine)

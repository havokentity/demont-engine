# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_subdirectory(core)
add_subdirectory(console)
# Software backend is cross-platform: Mac presents via CAMetalLayer,
# Windows via GDI BitBlt.  Future r_software_blit cvar will add a
# Vulkan-blit alternative on Windows that shares the engine's
# VkDevice; until then Windows uses GDI, Mac uses Metal.
add_subdirectory(rhi_software)
# Metal backend is Apple-only by definition.
if(APPLE)
    add_subdirectory(rhi_metal)
endif()
if(PT_ENABLE_VULKAN_BACKEND)
    add_subdirectory(rhi_vulkan)
endif()
add_subdirectory(rhi)
add_subdirectory(renderer)
add_subdirectory(app)
add_subdirectory(engine)

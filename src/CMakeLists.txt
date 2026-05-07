# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_subdirectory(core)
add_subdirectory(console)
# rhi_software's present path uses Metal layers (Apple-only). On
# Windows / Linux we rely on the Vulkan backend; CPU fallback is
# Apple-only for now.
if(APPLE)
    add_subdirectory(rhi_software)
    add_subdirectory(rhi_metal)
endif()
if(PT_ENABLE_VULKAN_BACKEND)
    add_subdirectory(rhi_vulkan)
endif()
add_subdirectory(rhi)
add_subdirectory(renderer)
add_subdirectory(app)
add_subdirectory(engine)

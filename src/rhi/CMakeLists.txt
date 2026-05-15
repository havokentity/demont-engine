# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi STATIC
    Device.cpp
)

target_include_directories(pt_rhi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_rhi PUBLIC pt_core)

# Software backend currently builds on Mac (CAMetalLayer present)
# and Windows (SetDIBitsToDevice present).  Linux would need its own
# EndFrame branch; gating the link + the PT_HAS_SOFTWARE_BACKEND
# define to APPLE OR WIN32 keeps r_backend's allowed_values list and
# the BackendType::Software dispatcher in lockstep with what actually
# has a working present path -- otherwise a Linux user typing
# `r_backend software` would see a silent black window.
# Metal backend is Apple-only by definition.
if(APPLE OR WIN32)
    target_link_libraries(pt_rhi PUBLIC pt_rhi_software)
    target_compile_definitions(pt_rhi PUBLIC PT_HAS_SOFTWARE_BACKEND=1)
endif()
if(APPLE)
    target_link_libraries(pt_rhi PUBLIC pt_rhi_metal)
    target_compile_definitions(pt_rhi PUBLIC PT_HAS_METAL_BACKEND=1)
endif()

if(PT_ENABLE_VULKAN_BACKEND)
    target_link_libraries(pt_rhi PUBLIC pt_rhi_vulkan)
    target_compile_definitions(pt_rhi PUBLIC PT_HAS_VULKAN_BACKEND=1)
endif()

pt_set_compiler_options(pt_rhi)

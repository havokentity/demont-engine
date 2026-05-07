# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi_vulkan STATIC
    VulkanDevice.cpp
)

target_include_directories(pt_rhi_vulkan PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_vulkan
    PUBLIC pt_core
    PUBLIC pt_app_window           # GLFW + (later) Cocoa bridge
    PUBLIC Vulkan::Vulkan
    PUBLIC glfw                    # for glfwCreateWindowSurface
)

# Compile every compute kernel to SPIR-V so the Vulkan backend has
# the same pipeline lookup table the Metal backend does. Tonemap,
# BloomDown, BloomUp join the path tracer here -- the engine looks
# them up by name unconditionally, and on a Vulkan-only build (no
# Metal compiled in) any missing pipeline name would assert.
include(${CMAKE_SOURCE_DIR}/cmake/Slang.cmake)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/Tonemap.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/BloomDown.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/BloomUp.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/AutoExposure.slang
    STAGE compute ENTRY main TARGETS spirv)

pt_set_compiler_options(pt_rhi_vulkan)

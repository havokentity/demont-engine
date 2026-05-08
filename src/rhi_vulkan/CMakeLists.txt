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

# Compile every compute kernel to SPIR-V so when the Vulkan backend
# does start instantiating Tonemap / BloomDown / BloomUp pipelines,
# the SPIR-V is already available alongside PathTrace + AutoExposure.
#
# Current state: only `pathtrace` and `autoexpose` are actually
# registered in VulkanDevice's pipeline map (they share the 16-binding
# unified descriptor layout + the engine's PtPush split). The other
# three SPIR-V blobs are compiled but no VkPipeline object is created
# for them yet -- they'd need their own descriptor layout + 576B
# push-constant split to land cleanly. Engine looks pipelines up by
# name and gets id=0 when a kernel isn't registered; the engine treats
# id=0 dispatches as silent no-ops, and PathTrace.slang carries an
# inline ACES tonemap so the Vulkan image is still tonemapped without
# the post-pass tonemap pipeline. (Mac builds use these shaders today
# via the Metal backend.)
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
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PerfOverlay.slang
    STAGE compute ENTRY main TARGETS spirv)

pt_set_compiler_options(pt_rhi_vulkan)

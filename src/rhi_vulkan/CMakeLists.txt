# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi_vulkan STATIC
    VulkanDevice.cpp
    VulkanDenoiser.cpp
)

target_include_directories(pt_rhi_vulkan PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_vulkan
    PUBLIC pt_core
    PUBLIC pt_app_window           # GLFW + (later) Cocoa bridge
    PUBLIC Vulkan::Vulkan
    PUBLIC glfw                    # for glfwCreateWindowSurface
)

# ---- OptiX denoiser (NVIDIA only) -----------------------------------------
# Adds the OptiX denoiser strategy alongside the SVGF/NRD one. Gated by
# PT_OPTIX_ACTIVE (set by the top-level CMakeLists once CUDA Toolkit and
# OptiX SDK headers are both detected). When inactive the OptiX file isn't
# compiled, the CUDA libs aren't linked, and PT_ENABLE_OPTIX is undefined
# so the #if blocks in the source compile out.
if(PT_OPTIX_ACTIVE)
    target_sources(pt_rhi_vulkan PRIVATE VulkanOptixDenoiser.cpp)
    target_include_directories(pt_rhi_vulkan PRIVATE ${OPTIX_INCLUDE_DIR})
    target_link_libraries(pt_rhi_vulkan
        PRIVATE CUDA::cudart_static
        PRIVATE CUDA::cuda_driver
    )
    target_compile_definitions(pt_rhi_vulkan PUBLIC PT_ENABLE_OPTIX=1)
endif()

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
# Pre-compile Slang IR modules used by PathTrace.  Each module compiles
# in its own ninja job (parallel) and the resulting .slang-module is
# linked into PathTrace.spv via the entry-point compile's `import` +
# `-I <out_dir>` resolver.  Body-only iterations on PathTrace.slang
# skip the module re-parse, saving ~145ms / rebuild.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang)
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTraceCloud.slang)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE compute ENTRY main TARGETS spirv
    MODULE_DEPS PathTraceMath PathTraceCloud)
# Second PathTrace SPIR-V variant without RayQuery, for backends that
# don't expose VK_KHR_ray_query (Mac-Vulkan on pre-1.3 MoltenVK builds).
# The two variants share the source file; PT_SPIRV_NO_RAYQUERY strips
# the scene_tlas binding + the HW RayQuery branch, so the SPIR-V module
# never declares the RayQueryKHR capability and loads cleanly on those
# drivers. VulkanDevice picks one at pipeline build time based on
# rt_supported_; see the PathTrace pipeline build site for the gate.
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE compute ENTRY main TARGETS spirv
    VARIANT norq EXTRA_DEFINES -DPT_SPIRV_NO_RAYQUERY
    MODULE_DEPS PathTraceMath PathTraceCloud)
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
# SVGF/NRD denoiser kernels. Each is its own SPIR-V blob; the Vulkan
# backend's VulkanNrdDenoiser builds a dedicated VkPipelineLayout +
# 2 VkPipeline handles from these and dispatches them in sequence.
# Metal builds skip these -- the Mac path uses MetalFX instead.
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/DenoiseTemporal.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/DenoiseAtrous.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/DenoiseFinalize.slang
    STAGE compute ENTRY main TARGETS spirv)

pt_set_compiler_options(pt_rhi_vulkan)

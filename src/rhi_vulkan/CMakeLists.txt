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

# ---- NRD denoiser (NVIDIA RayTracingDenoiser, Vulkan, non-Apple) ----------
# Issue #50. Gated by PT_NRD_ACTIVE (set by the top-level Dependencies.cmake
# once the user has flipped PT_ENABLE_NRD=ON on a non-Apple build with the
# Vulkan backend on). When inactive the wrapper file isn't compiled, the
# NRD lib isn't linked, and PT_ENABLE_NRD is undefined in the source so
# the #if PT_ENABLE_NRD blocks inside VulkanNrdLibDenoiser.{h,cpp} compile
# out -- VulkanDenoiser.cpp is unrelated (it's the SVGF placeholder backing
# the existing r_denoiser nrd cvar value and has no PT_ENABLE_NRD gates).
# Today the wrapper is a scaffold -- it stubs the NRD instance/dispatch
# translation; the r_denoiser nrd cvar value still routes through the
# in-house SVGF chain. A follow-up PR will switch routing to
# VulkanNrdLibDenoiser::Encode once the per-frame Dispatch->VkDispatch
# translation lands.
if(PT_NRD_ACTIVE)
    target_sources(pt_rhi_vulkan PRIVATE VulkanNrdLibDenoiser.cpp)
    target_link_libraries(pt_rhi_vulkan PRIVATE NRD)
    target_compile_definitions(pt_rhi_vulkan PUBLIC PT_ENABLE_NRD=1)
endif()

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
# --- SDF Phase 1 (#97) -----------------------------------------------------
# Analytic SDF library; same compile-module pattern.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/SdfPrimitives.slang)
# --- end SDF Phase 1 -------------------------------------------------------
# --- SDF Phase 3 (#99) -----------------------------------------------------
# Iconic-fractal DE library (Mandelbulb / Mandelbox / Apollonian).
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/SdfFractals.slang)
# --- end SDF Phase 3 -------------------------------------------------------
# --- ReSTIR DI Phase A (issue #78) ----------------------------------------
# Shared Reservoir struct + WRS streaming primitives imported by
# PathTrace.slang. The Restir compute kernels themselves are Metal-only
# at MVP scope (Vulkan dispatch follow-up); we still need the module
# for the SPIR-V PathTrace build because the candidate-generation
# block imports it.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/RestirReservoir.slang)
# --- end ReSTIR DI Phase A ------------------------------------------------

# --- Megakernel feature gates: slangc defines -----------------------------
# Same -DPT_*_ENABLED translation as the Metal CMakeLists (both targets
# must define the macros so the same Slang source compiles cleanly to
# both). Each _pt_*_def variable accumulates into EXTRA_DEFINES on every
# pt_compile_slang() invocation that builds PathTrace.slang variants.
if(PT_WATER_ENABLED)
    set(_pt_water_def "-DPT_WATER_ENABLED=1")
else()
    set(_pt_water_def "-DPT_WATER_ENABLED=0")
endif()
if(PT_LIGHT_TREE)
    set(_pt_light_tree_def "-DPT_LIGHT_TREE_ENABLED=1")
else()
    set(_pt_light_tree_def "-DPT_LIGHT_TREE_ENABLED=0")
endif()
# --- end Megakernel feature gates -----------------------------------------
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE compute ENTRY main TARGETS spirv
    EXTRA_DEFINES ${_pt_water_def} ${_pt_light_tree_def}
    MODULE_DEPS PathTraceMath PathTraceCloud SdfPrimitives SdfFractals RestirReservoir)
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
    VARIANT norq EXTRA_DEFINES -DPT_SPIRV_NO_RAYQUERY ${_pt_water_def} ${_pt_light_tree_def}
    MODULE_DEPS PathTraceMath PathTraceCloud SdfPrimitives SdfFractals RestirReservoir)
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
# Issue #119 -- SVGF albedo remodulation pass.
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/DenoiseRemod.slang
    STAGE compute ENTRY main TARGETS spirv)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/DenoiseFinalize.slang
    STAGE compute ENTRY main TARGETS spirv)

pt_set_compiler_options(pt_rhi_vulkan)

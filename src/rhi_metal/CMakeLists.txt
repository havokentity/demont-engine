# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi_metal STATIC
    MetalDevice.cpp
    MetalFXDenoiser.mm
    MetalSvgfDenoiser.cpp
)

target_include_directories(pt_rhi_metal PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_metal
    PUBLIC pt_core
    PUBLIC pt_app_window     # for pt_metal_attach_layer + pt_window_native_cocoa
    PUBLIC metal_cpp
)

if(APPLE)
    target_link_libraries(pt_rhi_metal PUBLIC
        "-framework Metal"
        "-framework MetalKit"
        "-framework MetalFX"      # P10 TemporalDenoisedScaler (macOS 15+)
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Compute kernels: the unified path tracer + the post-denoise tonemap
# pass. Both compile Slang -> MSL source which the runtime feeds into
# Metal's newLibrary() at backend init.
include(${CMAKE_SOURCE_DIR}/cmake/Slang.cmake)
# Pre-compile Slang IR modules (target-agnostic; same .slang-module
# files link into the Metal target as the SPIR-V target). Calls are
# idempotent across rhi_vulkan + rhi_metal trees.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang)
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTraceCloud.slang)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS PathTraceMath PathTraceCloud
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/Tonemap.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/BloomDown.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/BloomUp.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/AutoExposure.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/PerfOverlay.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# SVGF denoiser (svgf_basic / svgf_atrous on r_denoiser). Same Slang
# sources the Vulkan backend uses; slangc cross-compiles them to MSL.
# DenoiseFinalize is NOT built on Metal -- the existing Tonemap pipeline
# is the authoritative swapchain writer here. Vulkan goes the other way:
# its Tonemap.slang pipeline IS built (see VulkanDevice.cpp's pipeline
# build block) but is currently un-routed -- Engine.cpp gates
# `use_engine_tonemap` to Metal only pending root-cause of a descriptor /
# layout corruption that black-screens the chain on PC. So on Vulkan
# today DenoiseFinalize.slang is the swapchain writer; on Metal the
# SVGF chain writes its result into post_denoise_hdr and the normal
# post-denoise bloom + tonemap dispatch takes it from there.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/DenoiseTemporal.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/DenoiseAtrous.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)

pt_set_compiler_options(pt_rhi_metal)

# metal-cpp uses Objective-C runtime under the hood; suppress -Wpedantic
# warnings about extensions in the metal-cpp headers, and don't apply our
# -fno-rtti to the impl TU since metal-cpp objc runtime needs basic typeid
# in some configs.
target_compile_options(pt_rhi_metal PRIVATE
    -Wno-deprecated-declarations
    -Wno-pedantic
)

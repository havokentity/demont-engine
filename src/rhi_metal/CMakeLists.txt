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
# --- SDF Phase 1 (#97) -----------------------------------------------------
# Analytic SDF library (sphere/box/torus/capsule/rounded box + smooth-CSG
# ops + sphere-trace). Same target-agnostic module pattern as
# PathTraceMath / PathTraceCloud above.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/SdfPrimitives.slang)
# --- end SDF Phase 1 -------------------------------------------------------
# --- SDF Phase 3 (#99) -----------------------------------------------------
# Iconic-fractal DE library (Mandelbulb / Mandelbox / Apollonian) +
# fractal sphere-trace with iteration-relaxed surface epsilon and
# central-difference normals. Independent module so PathTrace.slang's
# body-only iterations don't re-parse the DE code.
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/SdfFractals.slang)
# --- end SDF Phase 3 -------------------------------------------------------
# --- ReSTIR DI Phase A (issue #78) ----------------------------------------
# Shared Reservoir struct + WRS streaming primitives. Imported by
# PathTrace.slang (initial WRS pass) AND by each Restir kernel.
# Target-agnostic module pattern -- same .slang-module flows into
# both Metal and SPIR-V (when the Vulkan plumbing follow-up lands).
pt_compile_slang_module(SOURCE ${CMAKE_SOURCE_DIR}/shaders/RestirReservoir.slang)
# --- end ReSTIR DI Phase A ------------------------------------------------

# --- Megakernel feature gates: slangc defines -----------------------------
# Translate each top-level PT_* option into a slangc -D flag so the
# matching `#if PT_*_ENABLED` blocks in PathTrace.slang resolve. Each
# gate's _pt_*_def variable is then included in EXTRA_DEFINES on the
# pt_compile_slang() call below.
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
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS PathTraceMath PathTraceCloud SdfPrimitives SdfFractals RestirReservoir
    EXTRA_DEFINES ${_pt_water_def} ${_pt_light_tree_def}
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
# Editor 3D-transform gizmo overlay. Compute kernel that rasterizes
# world-space line segments (projected via the camera matrices in the
# push constants) over the post-tonemap swapchain image. Dispatched
# only when the editor has a primitive selected; otherwise the engine
# skips the binding chain entirely.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/EditorOverlay.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# Stateless stars+sun+moon composite. Runs once per frame on Metal
# between Denoise() and dispatch_bloom_pyramid, so post-denoise HDR
# gets sub-pixel celestials with their own aperture-sampled bokeh and
# bloom downsamples those highlights into halos.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/StarsComposite.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# Aurora borealis procedural overlay (issue #116).
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/AuroraComposite.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# Wave 7 (#24) -- procedural raymarched cloud pre-pass.
# Two compute kernels paired by the engine when r_clouds_mode ==
# procedural_raymarched: CloudsRaymarch writes clouds_color_tex +
# cloud_trans_tex; CloudsComposite blends clouds_color_tex into the
# path-traced HDR (after StarsComposite, before bloom). When the cvar
# is at the default "pathtraced" the engine elides BOTH dispatches so
# zero runtime cost on the legacy path. Imports PathTraceCloud (the
# shared cloud-density module) so the raymarch and the inline cloud
# march sample the SAME procedural field with the same parameters.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/CloudsRaymarch.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS PathTraceCloud
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/CloudsComposite.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# SIGMA-style sun-shadow denoiser (issue #115). Runs between Denoise()
# and StarsComposite; bilateral-filters the per-pixel sun-visibility
# G-buffer, multiplies the denoised visibility back into HDR.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/SigmaShadow.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
# --- ReSTIR DI Phase A (issue #78) ----------------------------------------
# Three compute kernels chained behind PathTrace's WRS candidate-
# generation pass: RestirTemporal (reproject + MIS-merge prev frame),
# RestirSpatial (N=4 neighbours in a 10 px disc, MIS-merge),
# RestirFinal (shadow-test the survivor + additive Lambert composite
# into denoise_color). Same module-import pattern as PathTrace.slang.
# Metal-only at MVP scope (matches SigmaShadow / StarsComposite /
# AuroraComposite); Vulkan plumbing is a follow-up.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/RestirTemporal.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS RestirReservoir
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/RestirSpatial.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS RestirReservoir
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/RestirFinal.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
    MODULE_DEPS RestirReservoir
)
# --- end ReSTIR DI Phase A ------------------------------------------------
# Screen-space particle composite (#82 MVP). Dispatched AFTER
# StarsComposite + AuroraComposite, BEFORE the bloom pyramid; reads
# the engine-uploaded particle storage buffer and additive-splats
# each particle as a soft Gaussian disc into the post-denoise HDR,
# depth-gated by the PathTrace G-buffer.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/ParticleComposite.slang
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
# Issue #119 -- SVGF albedo remodulation pass. Multiplies the
# demodulated lighting from the SVGF chain by the primary-hit
# albedo G-buffer to restore textured radiance before bloom + tonemap.
# Built on Metal alongside the temporal + atrous kernels.
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/DenoiseRemod.slang
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

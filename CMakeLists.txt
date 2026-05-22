# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
cmake_minimum_required(VERSION 3.27)

# Several FetchContent deps (enkiTS, civetweb) ship CMakeLists pinned to
# pre-3.5 minimums that CMake 4.x rejects. This forces a sane floor for
# them without us touching upstream files.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "Configure out of source. Use a preset, e.g. `cmake --preset mac-debug`.")
endif()

project(demont
    VERSION 0.3.30
    DESCRIPTION "DeMonT Engine -- non-rasterized real-time game engine. Path-traced every pixel, mesh CSG via Manifold. Metal + MetalFX denoiser on Apple Silicon, Vulkan + SVGF compute denoiser on NVIDIA RTX."
    LANGUAGES C CXX
)

# --- Language baseline (orthodox C++23) -------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Project options --------------------------------------------------------
set(_PT_ENABLE_VULKAN_BACKEND_DEFAULT ON)
if(APPLE)
    set(_PT_ENABLE_VULKAN_BACKEND_DEFAULT OFF)
endif()
option(PT_ENABLE_VULKAN_BACKEND "Build native Vulkan backend (Windows/Linux)" ${_PT_ENABLE_VULKAN_BACKEND_DEFAULT})
if(APPLE AND PT_ENABLE_VULKAN_BACKEND)
    message(STATUS "PT_ENABLE_VULKAN_BACKEND is disabled on Apple; macOS uses the Metal backend.")
    set(PT_ENABLE_VULKAN_BACKEND OFF CACHE BOOL "Build native Vulkan backend (Windows/Linux)" FORCE)
endif()
option(PT_ENABLE_OPTIX
    "Build OptiX denoiser (NVIDIA only; needs CUDA Toolkit + OptiX SDK installed)" ON)
# NVIDIA RayTracingDenoiser (NRD) library integration -- issue #50.
#
# When ON, top-level Dependencies.cmake fetches NRD v4.17.3 via FetchContent
# and the Vulkan backend compiles VulkanNrdLibDenoiser.cpp + links libNRD.
# When OFF (default), neither the fetch nor the link happens -- the engine
# still compiles + runs, the `r_denoiser nrd` cvar value keeps aliasing the
# in-house SVGF chain (its long-standing behaviour). Default OFF because:
#   1. NRD is Vulkan-only and Apple Silicon (the project's primary platform)
#      can never use it -- forcing a 4 MB+ dep on every configure would be
#      pure overhead for those users.
#   2. NRD's CMake spawns a custom_target that invokes its `ShaderMake` /
#      `dxc-spirv` toolchain to compile HLSL -> SPIR-V at configure time.
#      That's a non-trivial transitive-dep chain that platform packagers
#      may not have set up; opt-in keeps the default `cmake --preset
#      mac-release` configure path identical to before this PR.
#   3. The integration is staged -- v0.3.30 lands the scaffolding (CMake
#      gates + VulkanNrdLibDenoiser stub) so a follow-up PR can land the
#      actual per-frame Dispatch translation without re-touching CMake.
#
# Hard-gated to non-Apple in Dependencies.cmake (NRD is Vulkan-only and the
# Mac path uses Metal + MetalFX); a user passing -DPT_ENABLE_NRD=ON on Mac
# gets a single warning and the *active* gate PT_NRD_ACTIVE stays OFF for
# that configure. The PT_ENABLE_NRD cache value itself is left as the user
# set it -- every consumer in src/rhi_vulkan/ checks PT_NRD_ACTIVE rather
# than PT_ENABLE_NRD directly, so a stale =ON in the cache on Apple is
# harmless: the FetchContent of NRD doesn't run, no target is linked, and
# the source-side PT_ENABLE_NRD compile-definition stays undefined.
option(PT_ENABLE_NRD
    "Build NVIDIA NRD denoiser (Vulkan-only; non-Apple platforms; needs ShaderMake/dxc-spirv toolchain)" OFF)
option(PT_ENABLE_TRACY          "Enable Tracy profiler client"          ON)
# ASan + mimalloc do not coexist cleanly without MI_TRACK_ASAN; opt-in only.
# Driven by the linux-asan-ubsan-debug preset in CMakePresets.json on CI;
# can also be flipped on locally for any host (cmake/CompilerWarnings.cmake
# applies the flags project-wide, including doctest test binaries).
option(PT_ENABLE_SANITIZERS     "Enable -fsanitize=address,undefined (clang/gcc)" OFF)
# AVX-512 build of Embree adds the 16-wide BVH16 intersector for users
# on Zen 4+ (Ryzen 7000+), Zen 5 (Ryzen 9000+, the project owner's
# 9950X3D), Threadripper, Xeon Skylake+, Rocket Lake (i9-11900K), and
# HEDT chips.  Costs ~25% extra build time.  Default ON: AVX-512 has
# become the consumer-modern default on AMD since Zen 4 in 2022, and
# Embree's runtime ISA dispatch lets the binary fall back to AVX2 on
# any chip that lacks AVX-512 (Zen 3, Intel mainstream consumer since
# Alder Lake, etc.) -- so one shipped binary covers everyone without
# breaking anyone.  Local devs who want a faster build (no need for
# AVX-512 perf on their own machine) can flip this OFF in their cmake
# invocation.
option(PT_ENABLE_AVX512_EMBREE  "Enable AVX-512 build of Embree (runtime-dispatched; +~25% build time)" ON)
# Unit tests. Default ON for developer builds. The option exists for
# anyone wanting to trim configure time on a build that genuinely
# can't run tests (CI release-staging step, embedded systems, etc.) --
# pass -DPT_BUILD_TESTS=OFF to skip the doctest fetch + tests/ subtree.
# release.yml doesn't pass this today (the release artefact only copies
# build/<preset>/src/app/demont so the test binaries don't ship
# regardless). doctest is single-header so the configure-time cost is
# small either way.
option(PT_BUILD_TESTS           "Build unit tests via doctest"          ON)
# Developer tooling (imgdiff, future golden-image helpers, etc.).
# Each tool target is standalone -- no link deps on pt_engine /
# pt_rhi_vulkan / pt_rhi_metal, so cmake --build --target <tool>
# stays fast and doesn't pull in the GPU backends.
#
# Caveat: this gates *building* tools/, not project configure. The
# top-level configure still runs the usual find_package(Vulkan) +
# add_subdirectory(src) path, which means a tools-only consumer who
# wants to skip the Vulkan SDK install needs both
#   -DPT_BUILD_TOOLS=ON -DPT_ENABLE_VULKAN_BACKEND=OFF
# Disabling the engine subtree from configure is a separate option we
# haven't needed yet; revisit if a CI image-compare job wants a truly
# minimal configure.
#
# Default ON so contributors get the tools without extra cmake flags;
# release.yml can flip OFF if the release artefact should stay binary-
# minimal.
option(PT_BUILD_TOOLS           "Build developer tooling under tools/"  ON)

# --- Megakernel feature gates ----------------------------------------------
# Each PT_*_ENABLED / PT_*_ON toggle compiles a specific material/shader
# subsystem in or out of the unified PathTrace.slang megakernel. The
# compile-time gate matters more than the runtime cost: leaving an unused
# subsystem in the megakernel inflates per-thread register pressure of the
# dispatch, dropping wavefront occupancy across every pixel + thrashing
# Apple Silicon's SIMD instruction cache. (Sibling SDF Phase 2 gate lives
# in cmake/Slang.cmake -- it gates a Slang module rather than a megakernel
# branch, so the option is registered closer to the shader-compile glue.)
#
# PT_WATER_ENABLED (PR #168, issue #134): MAT_WATER analytic-plane BRDF
# (Schlick Fresnel + Snell refraction + Beer's-law absorption) + the
# 8-noise-tap wave-normal helper + the MAT_WATER branch in
# transmittance().
#
# Default ON -- preserve functionality. The original gate (PR #168)
# defaulted OFF because the megakernel paid +2.99 ms / frame on the
# Cornell-CSG perf-bisect baseline; but that left `prim_plane ... water`
# materials (e.g. tests/goldens/scenes/water_pool.cfg) rendering as
# pure-black opaque surfaces, since the MAT_WATER BRDF branch in
# PathTrace.slang was excised at compile time and the shader had no
# fallthrough for that material tag. Mirrors the PT_LIGHT_TREE policy
# (default ON for functionality; power users / perf-bench rigs flip OFF
# via -DPT_WATER_ENABLED=OFF to recover the megakernel hot-loop slimness
# on water-less scenes).
#
# The cvars (r_water_absorption_r/g/b, r_water_ior, r_water_wave_*)
# stay registered regardless of the gate so existing user cfgs don't
# error out on cvar-not-found; when the gate is OFF and a user sets a
# non-default value, Engine.cpp emits a single [warn] line at startup.
option(PT_WATER_ENABLED
    "Compile MAT_WATER BRDF (#134) into the PathTrace megakernel. ON (default) = water material renders correctly; OFF = -2.99 ms/frame on water-less scenes (water surfaces will render as opaque-black)."
    ON)

# PT_LIGHT_TREE (PR #170, issue #129): when ON, PathTrace.slang compiles
# the Conty Estevez & Kulla 2018 light tree traversal helpers and the
# picker if-branch. The runtime cvar r_light_tree continues to flip
# between tree-pick and uniform-pick at zero runtime cost. When OFF the
# traversal helpers + the picker if-branch are excised at compile time.
# Recovers most of the structural ICache pressure PR #154 introduced
# even when r_light_tree=0 -- the dead branch still cost ICache +
# register pressure on Apple Silicon because the Metal compiler cannot
# fully DCE a function reachable through a uniform runtime branch.
#
# Default ON: preserve functionality. Power users / perf benchmarks set
# -DPT_LIGHT_TREE=OFF to recover megakernel hot-loop slimness; the
# uniform-pick fallback (#73) still works correctly, just with higher
# NEE variance on scenes with >100 lights. The cvar `r_light_tree`
# stays registered either way; when the gate is OFF the cvar is a
# no-op (the engine forces light_tree_enabled=0 in the push constants
# regardless of cvar state).
#
# Note: the perf-bisect-reported +7.67 ms attribution from PR #154 is
# NOT actually recovered by gating the shader code alone (default scene
# has light_count=0 and Apple Silicon's Metal runtime DCEs the dead
# branch effectively). The cost lives elsewhere (CPU-side per-frame
# work in LightTree.cpp / EnsureLightTreeUploaded / descriptor binding
# overhead). This gate is structurally valuable (266 lines / 9 KB
# SPIR-V dropped when OFF) and templates the pattern for future
# feature gates, even though the perf delta on the default scene is
# within run-to-run variance. Follow-up perf investigation queued.
option(PT_LIGHT_TREE
    "Compile hierarchical light tree (#129) into the PathTrace megakernel. OFF strips the traversal helpers from SPIR-V (~9 KB) but does not measurably recover the +7.67 ms regression PR #154 introduced (see follow-up); ON = O(log N) importance-weighted light pick (Conty Estevez & Kulla 2018)."
    ON)

# --- Compiler/warning helper -----------------------------------------------
include(cmake/CompilerWarnings.cmake)
include(cmake/EmbedResource.cmake)

# --- Vulkan (native Windows/Linux only) ------------------------------------
if(PT_ENABLE_VULKAN_BACKEND)
    find_package(Vulkan REQUIRED)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan: ${Vulkan_LIBRARIES} (headers: ${Vulkan_INCLUDE_DIRS})")
    endif()
endif()

# --- OptiX denoiser (CUDA Toolkit + OptiX SDK, NVIDIA only) ----------------
# OptiX runs on CUDA, denoises VkImages via VK_KHR_external_memory + cuda
# imported memory. Requires the Vulkan backend (interop is the whole point).
#
# Detection self-heals: if PT_ENABLE_OPTIX is requested but a dep is missing
# we set the internal `PT_OPTIX_ACTIVE` flag to OFF for this configure pass,
# leaving the user's PT_ENABLE_OPTIX cache entry intact. On a subsequent
# reconfigure (after they install the missing dep / set OPTIX_INSTALL_DIR)
# CMake re-runs find_path because OPTIX_INCLUDE_DIR is NOTFOUND, picks it
# up, and the next build links OptiX in.
set(PT_OPTIX_ACTIVE OFF)

if(PT_ENABLE_OPTIX AND NOT PT_ENABLE_VULKAN_BACKEND)
    message(STATUS "PT_ENABLE_OPTIX requires PT_ENABLE_VULKAN_BACKEND; OptiX denoiser inactive.")
endif()

if(PT_ENABLE_OPTIX AND PT_ENABLE_VULKAN_BACKEND)
    find_package(CUDAToolkit 12.0 QUIET)

    # Try common install layouts so the build works without OPTIX_INSTALL_DIR
    # being set in the active shell (env vars set via System Properties only
    # propagate to processes started AFTER the change). The glob picks up any
    # 'OptiX SDK X.Y.Z' install on Windows; explicit Linux paths cover the
    # standard NVIDIA-recommended locations.
    file(GLOB _pt_optix_candidates
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
        "C:/Program Files/NVIDIA Corporation/OptiX SDK *"
        "/usr/local/optix*"
        "/opt/optix*"
        "$ENV{HOME}/optix*")

    find_path(OPTIX_INCLUDE_DIR
        NAMES optix.h
        HINTS
            $ENV{OPTIX_INSTALL_DIR}
            ${_pt_optix_candidates}
        PATH_SUFFIXES include
        DOC "OptiX SDK include directory")

    if(CUDAToolkit_FOUND AND OPTIX_INCLUDE_DIR)
        set(PT_OPTIX_ACTIVE ON)
        message(STATUS "OptiX denoiser: enabled")
        message(STATUS "  CUDA Toolkit:  ${CUDAToolkit_VERSION} (${CUDAToolkit_INCLUDE_DIRS})")
        message(STATUS "  OptiX headers: ${OPTIX_INCLUDE_DIR}")
    else()
        if(NOT CUDAToolkit_FOUND)
            message(STATUS "PT_ENABLE_OPTIX: CUDA Toolkit 12.0+ not found; "
                           "OptiX denoiser inactive. Install from "
                           "developer.nvidia.com/cuda-downloads and reconfigure.")
        endif()
        if(NOT OPTIX_INCLUDE_DIR)
            message(STATUS "PT_ENABLE_OPTIX: OptiX SDK headers not found; "
                           "OptiX denoiser inactive. Install OptiX 8+/9+ and either "
                           "set OPTIX_INSTALL_DIR (e.g. "
                           "'C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0') "
                           "in the shell that runs cmake, or accept the auto-detected "
                           "path on a fresh shell after install. See OPTIX_SETUP.md "
                           "in the repo root for full setup + driver-compat notes "
                           "(including the OptiX 9.0 fallback for older drivers).")
        endif()
    endif()
endif()

# --- Binary toolchain bits (slangc, metal-cpp) -----------------------------
include(cmake/SetupBinaries.cmake)

# --- Third party (FetchContent) --------------------------------------------
include(cmake/Dependencies.cmake)

# --- Sources ----------------------------------------------------------------
add_subdirectory(src)

# --- Tools ------------------------------------------------------------------
# Developer CLIs that don't depend on the engine -- imgdiff is the first.
# Gated separately from PT_BUILD_TESTS so a "tests off, tools on" build
# (e.g. someone wanting just imgdiff for a CI image-compare step) is
# still possible.
if(PT_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# --- Tests ------------------------------------------------------------------
# enable_testing() must be in the top-level CMakeLists so `ctest` from the
# build root finds the tests. tests/CMakeLists.txt defines per-test
# executables + their add_test() registrations.
if(PT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

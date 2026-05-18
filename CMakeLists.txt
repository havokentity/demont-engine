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
option(PT_ENABLE_VULKAN_BACKEND "Build Vulkan/MoltenVK backend (P4+)"   ON)
option(PT_ENABLE_OPTIX
    "Build OptiX denoiser (NVIDIA only; needs CUDA Toolkit + OptiX SDK installed)" ON)
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

# --- Compiler/warning helper -----------------------------------------------
include(cmake/CompilerWarnings.cmake)
include(cmake/EmbedResource.cmake)

# --- Vulkan (LunarG SDK installed system-wide on Mac) ----------------------
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

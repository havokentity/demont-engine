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
    VERSION 0.3.9
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
option(PT_ENABLE_SANITIZERS     "Enable -fsanitize=address,undefined in Debug" OFF)

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

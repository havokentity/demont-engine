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
    VERSION 0.1.0
    DESCRIPTION "DeMonT Engine -- non-rasterized real-time game engine for Apple Silicon (path-traced, mesh CSG, MetalFX denoiser)"
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

# --- Binary toolchain bits (slangc, metal-cpp) -----------------------------
include(cmake/SetupBinaries.cmake)

# --- Third party (FetchContent) --------------------------------------------
include(cmake/Dependencies.cmake)

# --- Sources ----------------------------------------------------------------
add_subdirectory(src)

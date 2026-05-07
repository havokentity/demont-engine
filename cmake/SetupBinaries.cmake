# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Auto-downloads the binary toolchain bits that don't fit in FetchContent
# (slangc is a binary release; Apple's metal-cpp is a single zip on the
# developer site).  Both extract into third_party/, gitignored, populated
# at configure time on a fresh clone.

# --- Slang (shader compiler) -----------------------------------------------
set(PT_SLANG_VERSION 2026.8 CACHE STRING "Slang release tag")
set(_slang_dir ${CMAKE_SOURCE_DIR}/third_party/slang)

if(NOT EXISTS "${_slang_dir}/bin/slangc")
    set(_slang_url "https://github.com/shader-slang/slang/releases/download/v${PT_SLANG_VERSION}/slang-${PT_SLANG_VERSION}-macos-aarch64.tar.gz")
    set(_slang_archive "${CMAKE_BINARY_DIR}/slang.tar.gz")
    message(STATUS "Downloading Slang ${PT_SLANG_VERSION} ...")
    file(DOWNLOAD "${_slang_url}" "${_slang_archive}"
         SHOW_PROGRESS STATUS _dl)
    list(GET _dl 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR "Slang download failed: ${_dl}")
    endif()
    file(MAKE_DIRECTORY "${_slang_dir}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${_slang_archive}"
                    WORKING_DIRECTORY "${_slang_dir}"
                    RESULT_VARIABLE _tar_rc)
    if(NOT _tar_rc EQUAL 0)
        message(FATAL_ERROR "Slang extract failed: rc=${_tar_rc}")
    endif()
    file(REMOVE "${_slang_archive}")
endif()

set(PT_SLANGC_BIN "${_slang_dir}/bin/slangc" CACHE FILEPATH "slangc path")

# --- metal-cpp ------------------------------------------------------------
set(_metalcpp_dir ${CMAKE_SOURCE_DIR}/third_party/metal-cpp)

if(NOT EXISTS "${_metalcpp_dir}/Metal/Metal.hpp")
    set(_metalcpp_url "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15.2_iOS18.2.zip")
    set(_metalcpp_archive "${CMAKE_BINARY_DIR}/metal-cpp.zip")
    message(STATUS "Downloading metal-cpp ...")
    file(DOWNLOAD "${_metalcpp_url}" "${_metalcpp_archive}"
         SHOW_PROGRESS STATUS _dl)
    list(GET _dl 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR "metal-cpp download failed: ${_dl}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${_metalcpp_archive}"
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party
                    RESULT_VARIABLE _tar_rc)
    if(NOT _tar_rc EQUAL 0)
        message(FATAL_ERROR "metal-cpp extract failed: rc=${_tar_rc}")
    endif()
    file(REMOVE "${_metalcpp_archive}")
endif()

# Header-only INTERFACE library for metal-cpp.
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp SYSTEM INTERFACE "${_metalcpp_dir}")

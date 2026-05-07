# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Auto-downloads the binary toolchain bits that don't fit in FetchContent.
# Platform-aware: each target builds with the right slangc binary, and
# Apple-only deps (metal-cpp) only land on Apple builds.

# --- Slang (shader compiler) -----------------------------------------------
# Release archives are published per-platform at
#   https://github.com/shader-slang/slang/releases/download/v<ver>/...
# Names follow slang-<ver>-<os>-<arch>.{tar.gz,zip}.
set(PT_SLANG_VERSION 2026.8 CACHE STRING "Slang release tag")
set(_slang_dir ${CMAKE_SOURCE_DIR}/third_party/slang)

if(APPLE)
    set(_slang_archive_name "slang-${PT_SLANG_VERSION}-macos-aarch64.tar.gz")
    set(_slang_exe_name "slangc")
elseif(WIN32)
    set(_slang_archive_name "slang-${PT_SLANG_VERSION}-windows-x86_64.zip")
    set(_slang_exe_name "slangc.exe")
elseif(UNIX)
    set(_slang_archive_name "slang-${PT_SLANG_VERSION}-linux-x86_64.tar.gz")
    set(_slang_exe_name "slangc")
else()
    message(FATAL_ERROR "Unsupported platform for Slang download")
endif()

set(_slangc_path "${_slang_dir}/bin/${_slang_exe_name}")

if(NOT EXISTS "${_slangc_path}")
    set(_slang_url
        "https://github.com/shader-slang/slang/releases/download/v${PT_SLANG_VERSION}/${_slang_archive_name}")
    set(_slang_archive "${CMAKE_BINARY_DIR}/${_slang_archive_name}")
    message(STATUS "Downloading Slang ${PT_SLANG_VERSION} (${_slang_archive_name}) ...")
    file(DOWNLOAD "${_slang_url}" "${_slang_archive}"
         SHOW_PROGRESS STATUS _dl)
    list(GET _dl 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR "Slang download failed: ${_dl}")
    endif()
    file(MAKE_DIRECTORY "${_slang_dir}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${_slang_archive}"
                    WORKING_DIRECTORY "${_slang_dir}"
                    RESULT_VARIABLE _tar_rc)
    if(NOT _tar_rc EQUAL 0)
        message(FATAL_ERROR "Slang extract failed: rc=${_tar_rc}")
    endif()
    file(REMOVE "${_slang_archive}")
endif()

set(PT_SLANGC_BIN "${_slangc_path}" CACHE FILEPATH "slangc path")

# --- metal-cpp (Apple only) ------------------------------------------------
if(APPLE)
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

    add_library(metal_cpp INTERFACE)
    target_include_directories(metal_cpp SYSTEM INTERFACE "${_metalcpp_dir}")
endif()

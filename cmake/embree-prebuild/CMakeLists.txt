# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Standalone CMake project that builds JUST Embree with demont-engine's
# exact configuration.  Driven by .github/workflows/prebuild-embree.yml
# (CI) and runnable locally via, from the project root:
#
#     cmake -S cmake/embree-prebuild -B build/embree-prebuild \
#           -G Ninja -DCMAKE_BUILD_TYPE=Release
#     cmake --build build/embree-prebuild
#     cmake --install build/embree-prebuild --prefix dist/embree
#
# Output layout (after --install):
#     dist/embree/
#       include/embree4/...
#       lib/libembree4.a    (or embree4.lib on Windows)
#
# The workflow then tars/zips dist/embree/ and uploads it as a
# GitHub Release asset under the `vendored/embree-<version>` tag,
# where cmake/EmbreeBinary.cmake picks it up on the consumer side.
#
# This project intentionally does NOT consume cmake/Dependencies.cmake
# -- it ONLY builds Embree, so all the glm / fmt / glfw / etc.
# fetches in Dependencies.cmake would be wasteful here.  Instead it
# pulls in cmake/EmbreeConfig.cmake directly, which is the single
# source of truth shared with the in-source compile fallback in
# Dependencies.cmake.

cmake_minimum_required(VERSION 3.27)

# Match the main project's CMake-min-version override for old deps.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

project(demont_embree_prebuild LANGUAGES C CXX)

# Embree itself is C++17 for the same libc++ reason documented in
# the main Dependencies.cmake -- transitive-include reliance that
# C++20+ libc++ dropped.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# PT_ENABLE_AVX512_EMBREE: same option as the main build, default ON
# so the standalone prebuild produces an AVX-512-enabled artefact by
# default.  The workflow can override via -D on the cmake invocation
# if we ever want an AVX-512-OFF variant.
option(PT_ENABLE_AVX512_EMBREE "Enable AVX-512 build of Embree" ON)

# Pull in the shared Embree config (version + URL + URL_HASH + the
# pt_apply_embree_config macro that sets every EMBREE_* flag).  Same
# file the main project's Dependencies.cmake includes.
include(${CMAKE_CURRENT_SOURCE_DIR}/../EmbreeConfig.cmake)

# Apply the config (sets EMBREE_GEOMETRY_*, EMBREE_ISA_*, etc).
pt_apply_embree_config()

# Fetch + build Embree.  The shared FETCHCONTENT_BASE_DIR honour lets
# this re-use the main project's cache when run side-by-side locally.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(embree
    URL       ${EMBREE_VENDORED_URL}
    URL_HASH  ${EMBREE_VENDORED_URL_HASH}
    SYSTEM
)
FetchContent_MakeAvailable(embree)

# Stage the install tree for the workflow to pick up.
#   include/embree4/  -- public headers
#   lib/              -- libembree4.a / embree4.lib + any companion
#                        libs Embree's CMake installs (sys, math,
#                        simd, tasking, etc, if they exist as
#                        separate static libs in this config).
if(NOT TARGET embree)
    message(FATAL_ERROR
        "Embree FetchContent_MakeAvailable did not create an `embree` target. "
        "Check Embree's source build output above for configure errors.")
endif()

# Embree's source tree ships its public headers under include/embree4/;
# install them straight from the fetched source dir.
install(DIRECTORY ${embree_SOURCE_DIR}/include/embree4
        DESTINATION include
        PATTERN "*.h"
        PATTERN "*.hpp"
)

# Install every Embree target as a static lib.  EMBREE_STATIC_LIB=ON
# already in the config so these are .a / .lib (not .dylib / .dll).
# We install whichever of these targets exist -- Embree's exact target
# set depends on the active geometry / ISA flags.
foreach(_tgt embree sys math simd lexers tasking)
    if(TARGET ${_tgt})
        install(TARGETS ${_tgt}
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)
    endif()
endforeach()

# ISA-specific Embree libs (libembree_avx2.a, libembree_avx512.a, ...).
# These are produced when Embree builds multiple ISAs -- our config
# only enables AVX2 + optionally AVX-512 on x86, so we'll have at most
# two extra archives here.  Names match what Embree's own install
# would emit if we used their install rules directly.
foreach(_isa avx avx2 avx512)
    if(TARGET embree_${_isa})
        install(TARGETS embree_${_isa}
            ARCHIVE DESTINATION lib)
    endif()
endforeach()

# Write a small manifest so the artefact carries its own build
# fingerprint.  Lets a consumer inspect the .tar.gz and confirm what
# they're linking against without reading the cmake/EmbreeConfig.cmake
# version constant.
# Normalise CMake's ON/OFF -> JSON true/false for valid syntax.
if(PT_ENABLE_AVX512_EMBREE)
    set(_avx512_json "true")
else()
    set(_avx512_json "false")
endif()

file(WRITE ${CMAKE_BINARY_DIR}/manifest.json "{\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"embree_src_version\": \"${EMBREE_SRC_VERSION}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"vendored_version\":   \"${EMBREE_VENDORED_VERSION}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"host_system\":        \"${CMAKE_SYSTEM_NAME}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"host_processor\":     \"${CMAKE_SYSTEM_PROCESSOR}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"compiler_id\":        \"${CMAKE_CXX_COMPILER_ID}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"compiler_version\":   \"${CMAKE_CXX_COMPILER_VERSION}\",\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "  \"avx512_enabled\":     ${_avx512_json}\n")
file(APPEND ${CMAKE_BINARY_DIR}/manifest.json "}\n")
install(FILES ${CMAKE_BINARY_DIR}/manifest.json DESTINATION .)

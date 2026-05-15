# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Single source of truth for Embree's version + build configuration.
# Both code paths consume this:
#   1. cmake/Dependencies.cmake -- fall-back path when a vendored
#      prebuilt isn't available for the host platform.  Fetches
#      Embree's source via FetchContent and compiles it inline.
#   2. cmake/embree-prebuild/CMakeLists.txt -- standalone project that
#      .github/workflows/prebuild-embree.yml drives to build the .a /
#      .lib artefacts that get uploaded to a GitHub Release.
#
# The two MUST stay byte-equivalent (same Embree version, same flags),
# otherwise the prebuilt artefact's headers won't match what consumer
# code expects.  Putting the config in one shared file (this one) is
# how we keep them in lockstep -- both call pt_apply_embree_config()
# to set every flag, both read EMBREE_VENDORED_URL +
# EMBREE_VENDORED_URL_HASH for the source tarball.
#
# Bump EMBREE_VENDORED_VERSION whenever ANY of:
#   - The Embree source version (EMBREE_VENDORED_URL / _URL_HASH)
#   - Any flag in pt_apply_embree_config()
#   - The C++ standard Embree itself compiles in (currently 17,
#     overridden via block() in Dependencies.cmake)
# changes.  Bumping invalidates the prebuilt artefact cache and the
# next run of prebuild-embree.yml builds a fresh one.

# --- Source version + hash --------------------------------------------------
# Embree's release tarball + content hash.  hashFiles()'s SHA256 was
# computed locally; matches what CMake's FetchContent would verify.
set(EMBREE_SRC_VERSION   "v4.4.0")
set(EMBREE_VENDORED_URL  "https://github.com/RenderKit/embree/archive/refs/tags/${EMBREE_SRC_VERSION}.tar.gz")
set(EMBREE_VENDORED_URL_HASH
    "SHA256=acb517b0ea0f4b442235d5331b69f96192c28da6aca5d5dde0cbe40799638d5c")

# --- Vendored prebuilt cache key --------------------------------------------
# Bumped whenever ANYTHING about the build config changes (above section
# OR pt_apply_embree_config() below).  Drives the GitHub Release tag
# name (vendored/embree-${EMBREE_VENDORED_VERSION}) so a config change
# triggers a fresh build instead of silently reusing a stale prebuilt.
#
# Suffix convention: -cfgN where N increments on config changes.
#   v4.4.0-cfg1: initial AVX2 + (opt) AVX-512 layout, no SSE2/SSE42/AVX,
#                EMBREE_GEOMETRY_TRIANGLE + INSTANCE only, EMBREE_STATIC_LIB ON,
#                EMBREE_TASKING_SYSTEM INTERNAL, EMBREE_RAY_PACKETS OFF.
set(EMBREE_VENDORED_VERSION "v4.4.0-cfg1")

# --- Per-host filename derivation -------------------------------------------
# Used by both the prebuild workflow (to NAME the uploaded artefact) and
# the EmbreeBinary download wrapper (to PICK the right artefact for the
# current host).  ARM Linux + Intel Mac aren't built right now -- they
# fall through to source compile.
function(pt_embree_artefact_name out_var)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        set(${out_var} "embree-${EMBREE_VENDORED_VERSION}-macos-arm64.tar.gz" PARENT_SCOPE)
    elseif(WIN32)
        # Windows prebuilt is the clang-cl build (matches release.yml's
        # win-clang-release preset).  MSVC consumers can't use AVX-512
        # anyway (Embree refuses cl.exe + AVX-512); they'd fall through
        # to source compile and skip AVX-512 there too.
        set(${out_var} "embree-${EMBREE_VENDORED_VERSION}-windows-x64.zip" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

# --- Embree CMake flags -- THE shared config --------------------------------
# Sets every Embree-* variable that affects the produced .a / .lib.  Both
# the prebuild workflow's standalone build AND Dependencies.cmake's
# fallback compile call this so the configs are byte-equivalent.  Any
# new EMBREE_* flag we want to set MUST go here (not directly in
# Dependencies.cmake) or the prebuilt artefact will silently drift.
macro(pt_apply_embree_config)
    set(EMBREE_TUTORIALS                OFF CACHE BOOL   "" FORCE)
    set(EMBREE_STATIC_LIB               ON  CACHE BOOL   "" FORCE)
    set(EMBREE_ISPC_SUPPORT             OFF CACHE BOOL   "" FORCE)
    set(EMBREE_TASKING_SYSTEM           "INTERNAL" CACHE STRING "" FORCE)
    set(EMBREE_GEOMETRY_TRIANGLE        ON  CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_QUAD            OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_CURVE           OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_SUBDIVISION     OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_USER            OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_INSTANCE        ON  CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_INSTANCE_ARRAY  OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_GRID            OFF CACHE BOOL   "" FORCE)
    set(EMBREE_GEOMETRY_POINT           OFF CACHE BOOL   "" FORCE)
    set(EMBREE_RAY_PACKETS              OFF CACHE BOOL   "" FORCE)
    set(EMBREE_FILTER_FUNCTION          OFF CACHE BOOL   "" FORCE)
    set(EMBREE_BACKFACE_CULLING         OFF CACHE BOOL   "" FORCE)
    set(EMBREE_ZIP_MODE                 OFF CACHE BOOL   "" FORCE)

    # x86 ISA matrix (gated on host arch; ARM hosts skip the block).
    # Mirrors the rationale block in cmake/Dependencies.cmake.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|amd64|AMD64")
        set(EMBREE_ISA_SSE2             OFF CACHE BOOL   "" FORCE)
        set(EMBREE_ISA_SSE42            OFF CACHE BOOL   "" FORCE)
        set(EMBREE_ISA_AVX              OFF CACHE BOOL   "" FORCE)
        set(EMBREE_ISA_AVX2             ON  CACHE BOOL   "" FORCE)
        if(PT_ENABLE_AVX512_EMBREE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(EMBREE_ISA_AVX512       ON  CACHE BOOL   "" FORCE)
        else()
            set(EMBREE_ISA_AVX512       OFF CACHE BOOL   "" FORCE)
        endif()
    endif()
endmacro()

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
#                Artefact set grew over time under the same cfg1 (which
#                stays valid as long as the *Embree compile flags* are
#                unchanged): macOS arm64 + Windows x64 Release first,
#                then Linux x64 + Linux ARM64 (Release), then a Windows
#                x64 Debug variant for build.yml's Debug PR-gate.  None
#                of those changed the Embree flags themselves, so cfg1
#                still applies -- the rule is "bump cfgN only when
#                pt_apply_embree_config() output changes," not when
#                the artefact MATRIX expands.
set(EMBREE_VENDORED_VERSION "v4.4.0-cfg1")

# --- Per-host + per-config filename derivation ------------------------------
# Used by both the prebuild workflow (to NAME the uploaded artefact) and
# the EmbreeBinary download wrapper (to PICK the right artefact for the
# current host).  Returns an empty string for unsupported platform combos
# (e.g. Intel Mac, FreeBSD, Linux RISC-V); callers fall through to
# source compile.
#
# `config` arg is "Release" or "Debug" (case-insensitive).  Used only
# on Windows -- everywhere else we ship Release-only because their
# linkers handle Release+Debug mixing cleanly.
#
# Windows is the exception: MSVC's STL stores `_ITERATOR_DEBUG_LEVEL`
# (0 in Release, 2 in Debug) and `RuntimeLibrary` (MD / MDd) as
# LIB-level markers, and the linker refuses to merge object files
# whose markers don't match.  Embree's implementation uses std::vector
# / std::string internally even though its public API is C, so the
# .lib carries those markers.  Net effect: a Debug demont built
# against a Release Embree prebuilt produces LNK2038 errors.  Fix is
# to ship BOTH Release and Debug Windows prebuilts, named:
#
#   embree-<v>-windows-x64.zip          (Release, default)
#   embree-<v>-windows-x64-debug.zip    (Debug)
#
# Mac + Linux don't need this distinction (clang's libc++ doesn't
# have a debug-level marker, no LIB-level CRT either), so their
# artefact names are config-agnostic.
function(pt_embree_artefact_name out_var config)
    string(TOLOWER "${config}" _cfg_lower)
    if(_cfg_lower MATCHES "debug")
        set(_cfg "debug")
    else()
        set(_cfg "release")
    endif()

    # Default: no -debug suffix (Release everywhere except Windows).
    set(_suffix "")

    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        set(_plat "macos-arm64")
        set(_ext "tar.gz")
    elseif(WIN32)
        # Windows prebuilt is the clang-cl build (matches release.yml's
        # win-clang-release preset).  MSVC consumers can't use AVX-512
        # anyway (Embree refuses cl.exe + AVX-512); they'd fall through
        # to source compile and skip AVX-512 there too.
        set(_plat "windows-x64")
        set(_ext "zip")
        if(_cfg STREQUAL "debug")
            set(_suffix "-debug")
        endif()
    elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64$|^arm64$")
        # Linux ARM64 (Pi 4/5, Graviton, Ampere Altra, etc.).  Same NEON
        # path Embree uses on Apple Silicon; distinct from macos-arm64
        # because libc / ABI / linker differ.  Strict aarch64/arm64
        # match: earlier `^arm` regex would have caught 32-bit Pi-Zero
        # / Pi-2-A+ hosts (armv7l, armv6l) and pointed them at the
        # ARM64 archive -- wrong ABI, linker would fail.  Those hosts
        # now fall through to the unsupported branch and source-compile
        # (which will also fail without 32-bit ARM Embree bits, but at
        # least with a coherent error).
        set(_plat "linux-arm64")
        set(_ext "tar.gz")
    elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^x86_64$|^amd64$|^AMD64$|^x64$")
        # Strict x86_64 / amd64 match -- earlier we used a looser
        # "anything not ARM" branch but Copilot pointed out it would
        # silently route RISC-V / ppc64le / s390x hosts to the x86_64
        # archive.  Those architectures now fall through to source
        # compile (which will also fail without the right Embree ISA
        # bits, but at least the failure mode is "Embree refused to
        # configure" rather than "we downloaded the wrong arch lib").
        set(_plat "linux-x64")
        set(_ext "tar.gz")
    else()
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(${out_var}
        "embree-${EMBREE_VENDORED_VERSION}-${_plat}${_suffix}.${_ext}"
        PARENT_SCOPE)
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

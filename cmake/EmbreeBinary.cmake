# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Try to consume a pre-built Embree binary from the project's vendored
# GitHub Release (produced by .github/workflows/prebuild-embree.yml).
# On success, defines an `embree` IMPORTED STATIC library that other
# CMake targets can link against, identical to what FetchContent +
# MakeAvailable would have produced -- but skipping the ~10-15 min
# Embree source compile entirely.
#
# Sets EMBREE_PREBUILT_FOUND to TRUE on success, FALSE on any failure
# (missing artefact, hash mismatch, unsupported platform).  The caller
# in Dependencies.cmake reads that flag and falls back to FetchContent
# compile-from-source when FALSE so new platforms / config bumps still
# work without a prebuilt -- they just take longer the first time, and
# the prebuild workflow's next run populates the artefact.
#
# Expected tarball/zip layout:
#   embree/
#     include/
#       embree4/
#         rtcore.h, rtcore_*.h ...
#     lib/
#       libembree4.a    (macOS / Linux)
#       embree4.lib     (Windows)
#
# Cache dir: ${FETCHCONTENT_BASE_DIR}/embree-prebuilt-<artefact_name>/
# (e.g. embree-prebuilt-embree-v4.4.0-cfg1-macos-arm64/).  Encodes
# platform + config in the path so cross-host shared caches (NFS-
# mounted $HOME, multi-stage container builds, two-devbox cloud
# drives) can't mistakenly serve the wrong-arch prebuilt -- each
# platform's tarball expands into its own directory.  Uses the same
# env-driven FETCHCONTENT_BASE_DIR as the rest of FetchContent so the
# external-drive workaround stays consistent.

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/EmbreeConfig.cmake)

set(EMBREE_PREBUILT_FOUND FALSE)

# Multi-config generators on Windows: bail to source-compile.
#
# CMAKE_BUILD_TYPE is empty at configure time for multi-config generators
# (Visual Studio, Xcode, Ninja Multi-Config) -- the consumer picks the
# config at build time via `cmake --build --config Debug|Release`.  If
# we proceeded here we'd download whichever artefact maps to "empty
# build type" (Release, per the default below), then a later `--config
# Debug` build would link the Release Embree .lib into Debug demont
# .obj files and hit the exact LNK2038 _ITERATOR_DEBUG_LEVEL /
# RuntimeLibrary mismatch this PR set out to avoid.
#
# Our CMakePresets.json only ships Ninja (single-config), so the
# project's supported configure paths all set CMAKE_BUILD_TYPE
# explicitly.  This guard is defensive: if a user runs
# `cmake -G "Visual Studio 17 2022"` off-script, they fall back to
# source-compile of Embree (slow but correct) instead of getting the
# linker error.
#
# Mac + Linux don't need this gate (libc++ doesn't have an iterator-
# debug-level marker, so mixing Release Embree into a multi-config
# Debug build is benign there), but applying it cross-platform isn't
# wrong either -- multi-config + multi-prebuilt is a hairball we just
# don't need to deal with given Ninja covers every real workflow.
if(WIN32 AND CMAKE_CONFIGURATION_TYPES)
    message(STATUS
        "Embree prebuilt: multi-config generator detected on Windows "
        "(CMAKE_CONFIGURATION_TYPES=${CMAKE_CONFIGURATION_TYPES}); "
        "falling back to compile-from-source.  Switch to a Ninja-based "
        "preset (--preset win-clang-release / win-clang-debug) to "
        "consume the prebuilt instead.")
    return()
endif()

# Pick the artefact for the current host + current build type.  Empty
# string = unsupported platform; fall through to source compile.
# CMAKE_BUILD_TYPE may be unset on single-config generators when the
# user didn't pass -DCMAKE_BUILD_TYPE explicitly; default to Release
# for the lookup, same as CMake does for unset build-type values.
#
# Build type only changes the artefact name on Windows (Release vs
# Debug Embree must be config-matched because MSVC's STL stamps
# _ITERATOR_DEBUG_LEVEL + RuntimeLibrary markers at .lib granularity;
# see EmbreeConfig.cmake's pt_embree_artefact_name for the full
# rationale).  On Mac + Linux pt_embree_artefact_name always returns
# the Release artefact regardless of build type.
if(CMAKE_BUILD_TYPE)
    set(_embree_buildtype "${CMAKE_BUILD_TYPE}")
else()
    set(_embree_buildtype "Release")
endif()
pt_embree_artefact_name(_embree_artefact "${_embree_buildtype}")
if(NOT _embree_artefact)
    message(STATUS "Embree prebuilt: no artefact mapping for this host (CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}', WIN32=${WIN32}, APPLE=${APPLE}); falling back to compile-from-source.")
    return()
endif()

# Derive the cache extract directory name from the artefact filename
# itself (minus archive extension).  Encodes platform + config (where
# applicable) directly into the cache path so that:
#
#   * macos-arm64 / windows-x64 / windows-x64-debug / linux-x64 /
#     linux-arm64 each get a fully isolated cache subdirectory.
#   * A shared FETCHCONTENT_BASE_DIR across hosts (e.g. NFS-mounted
#     $HOME between a Mac and a Linux VM, a multi-stage container
#     build sharing /root between platform stages, or just two
#     developers' devboxes accidentally pointing at the same cloud
#     drive) can't make the second-platform configure see the first-
#     platform's `embree/lib/libembree4.a` canary and skip the
#     download -- which would silently link the WRONG-ARCH .a / .lib.
#   * The Windows-Debug suffix in the artefact name automatically
#     produces a distinct cache dir from the Windows-Release one,
#     replacing the old explicit `_embree_cache_suffix` plumbing.
#
# Strip both .tar.gz and .zip extensions; the regex handles either.
set(_embree_cache_basename "${_embree_artefact}")
string(REGEX REPLACE "\\.(tar\\.gz|zip)$" "" _embree_cache_basename "${_embree_cache_basename}")

# GitHub Release URL.  Tag format: vendored/embree-<version>.  The
# forward-slash in the tag name needs URL-encoding for the download URL
# (%2F), even though gh / git APIs accept it raw.
set(_embree_release_tag "vendored/embree-${EMBREE_VENDORED_VERSION}")
set(_embree_url
    "https://github.com/havokentity/demont-engine/releases/download/vendored%2Fembree-${EMBREE_VENDORED_VERSION}/${_embree_artefact}")

# Cache + extract destination.  Reuses FETCHCONTENT_BASE_DIR when set so
# devs on external-drive setups (which caused git-pack failures on the
# in-source path historically) keep their cache in $HOME like everything
# else FetchContent does.
# Env-var check first: FetchContent.cmake's module init has already
# defaulted the CMake-variable FETCHCONTENT_BASE_DIR to
# ${CMAKE_BINARY_DIR}/_deps by the time this file is included from
# Dependencies.cmake, which would mask any FETCHCONTENT_BASE_DIR=...
# the user set in their shell.  Honour the env var explicitly so the
# prebuilt cache co-locates with whatever the user picked.
if(DEFINED ENV{FETCHCONTENT_BASE_DIR})
    set(_embree_cache_root "$ENV{FETCHCONTENT_BASE_DIR}")
elseif(DEFINED FETCHCONTENT_BASE_DIR)
    set(_embree_cache_root "${FETCHCONTENT_BASE_DIR}")
else()
    set(_embree_cache_root "${CMAKE_BINARY_DIR}/_deps")
endif()
set(_embree_extract_dir "${_embree_cache_root}/embree-prebuilt-${_embree_cache_basename}")
set(_embree_archive_path "${_embree_extract_dir}.archive")

# Short-circuit when the cache already has a usable extracted tree.
# `lib/<lib_name>` existence is the canary; if a previous run extracted
# successfully we skip both the download and the re-extract.
if(WIN32)
    set(_embree_lib_name "embree4.lib")
else()
    set(_embree_lib_name "libembree4.a")
endif()
set(_embree_lib_path     "${_embree_extract_dir}/embree/lib/${_embree_lib_name}")
set(_embree_include_path "${_embree_extract_dir}/embree/include")

if(NOT EXISTS "${_embree_lib_path}")
    # Need to (re-)download.  file(DOWNLOAD) is synchronous, no retry --
    # GitHub Releases CDN is reliable enough that this is fine; if a
    # transient HTTP error breaks one configure, the user just reconfigures.
    message(STATUS "Embree prebuilt: downloading ${_embree_artefact} ...")
    file(DOWNLOAD
        "${_embree_url}"
        "${_embree_archive_path}"
        STATUS _embree_dl_status
        TLS_VERIFY ON
        TIMEOUT 120
        SHOW_PROGRESS)
    list(GET _embree_dl_status 0 _embree_dl_code)
    list(GET _embree_dl_status 1 _embree_dl_msg)
    if(NOT _embree_dl_code EQUAL 0)
        # 22 = HTTP error (typically 404 for "release tag doesn't exist
        # yet").  Other codes: 6 = couldn't resolve host (offline), 28 =
        # timeout, etc.  All of these fall through to source-compile.
        message(STATUS
            "Embree prebuilt: download failed (curl code ${_embree_dl_code}: ${_embree_dl_msg}); "
            "falling back to compile-from-source.  "
            "If you want a prebuilt for this config, trigger "
            "prebuild-embree.yml via Actions tab or push a tag.")
        file(REMOVE "${_embree_archive_path}")
        return()
    endif()

    # Extract.  CMake 3.18+ can extract .tar.gz and .zip natively via
    # file(ARCHIVE_EXTRACT); our project's minimum is 3.27 so this is
    # always available.
    file(REMOVE_RECURSE "${_embree_extract_dir}")
    file(MAKE_DIRECTORY "${_embree_extract_dir}")
    file(ARCHIVE_EXTRACT
        INPUT "${_embree_archive_path}"
        DESTINATION "${_embree_extract_dir}")
    file(REMOVE "${_embree_archive_path}")   # drop the .tar.gz after extract

    if(NOT EXISTS "${_embree_lib_path}")
        message(WARNING
            "Embree prebuilt: archive extracted but ${_embree_lib_name} not found at expected path. "
            "Falling back to compile-from-source.  Path expected: ${_embree_lib_path}")
        file(REMOVE_RECURSE "${_embree_extract_dir}")
        return()
    endif()
endif()

# Collect every companion archive Embree installs alongside the main
# libembree4.  With EMBREE_STATIC_LIB=ON Embree splits its
# implementation across libsys.a / libmath.a / libsimd.a / liblexers.a
# / libtasking.a (low-level helpers) plus libembree_avx2.a /
# libembree_avx512.a (the ISA-dispatched intersector kernels Embree
# picks between at runtime via cpuid).  On the FetchContent path
# Embree's own CMake wires `target_link_libraries(embree PUBLIC ...)`
# to all of these so consumers pull them in transitively.  On the
# prebuilt path we have to chain them manually -- without this, every
# call into Embree's hot ray-traversal kernels would be an unresolved
# symbol at link time (the ISA archives hold _embree_dispatch_intersect1
# et al; the low-level libs hold most of the math / threading internals).
# Glob is restricted to actual archive files (.a on Unix, .lib on
# Windows) so we don't pick up Embree's exported CMake config tree
# under lib/cmake/embree-X.Y.Z/ which the install step also produces.
file(GLOB _embree_companions
    "${_embree_extract_dir}/embree/lib/*.a"
    "${_embree_extract_dir}/embree/lib/*.lib")
list(REMOVE_ITEM _embree_companions "${_embree_lib_path}")

# Wire up the imported target.  Same name as what the FetchContent path
# produces (`embree`), so consumer code (src/rhi_software/CMakeLists.txt)
# stays identical -- `target_link_libraries(... PRIVATE embree)` works
# either way.  INTERFACE_LINK_LIBRARIES chains the companion archives so
# the link is complete.  Link order:
#     consumer -> libembree4.a -> [companions in glob order]
# Glob ordering puts ISA archives before low-level helpers
# alphabetically, which matches the natural dependency direction
# (libembree4 calls libembree_avx2 calls libsys/libmath/...).  Apple
# ld64 / lld / clang-cl link.exe all handle any residual cycles via
# their own multi-pass static-lib resolution -- no --start-group
# wrappers needed on the platforms we ship.
add_library(embree STATIC IMPORTED GLOBAL)
set_target_properties(embree PROPERTIES
    IMPORTED_LOCATION             "${_embree_lib_path}"
    INTERFACE_INCLUDE_DIRECTORIES "${_embree_include_path}"
    INTERFACE_LINK_LIBRARIES      "${_embree_companions}"
)

# Mark the includes as SYSTEM so consumer code under our strict
# `-Wall -Wextra` doesn't complain about Embree's headers (same effect
# as `SYSTEM` on the FetchContent_Declare in the fallback path).
target_include_directories(embree SYSTEM INTERFACE "${_embree_include_path}")

set(EMBREE_PREBUILT_FOUND TRUE)
list(LENGTH _embree_companions _embree_n_companions)
message(STATUS
    "Embree prebuilt: linked ${_embree_lib_path} + ${_embree_n_companions} companion archive(s)")

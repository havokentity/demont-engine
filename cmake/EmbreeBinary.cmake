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
# Cache dir: ${FETCHCONTENT_BASE_DIR}/embree-prebuilt-<version>/
# (using the same env-driven path as the rest of FetchContent so the
# external-drive workaround stays consistent).

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/EmbreeConfig.cmake)

set(EMBREE_PREBUILT_FOUND FALSE)

# Figure out which artefact filename to look for, per host platform.
# Empty string = unsupported platform; fall through immediately.
pt_embree_artefact_name(_embree_artefact)
if(NOT _embree_artefact)
    message(STATUS "Embree prebuilt: no artefact mapping for this host (CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}', WIN32=${WIN32}, APPLE=${APPLE}); falling back to compile-from-source.")
    return()
endif()

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
set(_embree_extract_dir "${_embree_cache_root}/embree-prebuilt-${EMBREE_VENDORED_VERSION}")
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

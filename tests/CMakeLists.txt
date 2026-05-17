# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Unit tests via doctest. Each TEST_*_test.cpp file is a self-contained
# test executable -- doctest's DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN macro
# generates its own main() per executable. ctest runs each executable
# and reports pass/fail; on failure, doctest's output includes file +
# line + the failing expression so the diff is obvious.
#
# Add new tests by:
#   1. Drop tests/<area>/<name>_test.cpp implementing TEST_CASE("...")
#      blocks. The first line should be:
#        #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#      followed by:
#        #include <doctest/doctest.h>
#      Subsequent test TUs in the same executable would NOT define
#      _IMPLEMENT_WITH_MAIN -- that's only the entry-point TU. For
#      one-file-one-binary tests like ours, every file defines it.
#   2. Call pt_add_test(<name> <name>_test.cpp [LINK_TARGETS ...])
#      below. It wires the doctest dep + ctest registration in one shot.
#
# The doctest dep ships from cmake/Dependencies.cmake (FetchContent),
# already populated by the time this file is processed. The top-level
# CMakeLists.txt order is:
#   include(cmake/Dependencies.cmake)  <-- doctest is fetched + made
#                                          available here
#   add_subdirectory(src)
#   add_subdirectory(tests)            <-- this file
# so doctest::doctest is a valid target by the time pt_add_test() runs.
# `enable_testing()` is also in the top-level (must be there, not here,
# for ctest discovery from the build root).

# Helper: register a doctest-based test executable. Each test is its own
# binary so the link cost is bounded per test (instead of monotonically
# growing as we add more) and a failure in one test doesn't poison the
# others. Trade-off: more total link work, less per-link work; with
# ninja's parallel link this scales well.
function(pt_add_test test_name)
    set(_options)
    set(_oneval)
    set(_multival LINK_TARGETS SOURCES)
    cmake_parse_arguments(PT_TEST "${_options}" "${_oneval}" "${_multival}" ${ARGN})

    # First positional after test_name is the primary source file; any
    # additional SOURCES are appended. Most tests will be single-file.
    set(_sources)
    if(PT_TEST_UNPARSED_ARGUMENTS)
        list(APPEND _sources ${PT_TEST_UNPARSED_ARGUMENTS})
    endif()
    if(PT_TEST_SOURCES)
        list(APPEND _sources ${PT_TEST_SOURCES})
    endif()

    add_executable(${test_name} ${_sources})

    # Link doctest's header-only INTERFACE target. doctest 2.4+ exports
    # `doctest::doctest` (and `doctest_with_main`); we use the bare one
    # because each test TU brings its own main via _IMPLEMENT_WITH_MAIN.
    target_link_libraries(${test_name} PRIVATE doctest::doctest)

    # Per-test extra links (e.g. pt_console for the cvar round-trip test).
    if(PT_TEST_LINK_TARGETS)
        target_link_libraries(${test_name} PRIVATE ${PT_TEST_LINK_TARGETS})
    endif()

    # Match the project's compile-warning policy on test code -- catches
    # the same class of bugs in tests that we catch in `src/`. doctest's
    # own headers are SYSTEM so they don't leak warnings into our diag.
    if(COMMAND pt_target_warnings)
        pt_target_warnings(${test_name})
    endif()

    # Register with ctest. WORKING_DIRECTORY is the binary dir so tests
    # that emit artefacts (golden PNGs later, etc.) drop them somewhere
    # predictable instead of CWD-wherever-ctest-was-invoked-from.
    add_test(
        NAME    ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 60s timeout per test. Unit tests should be near-instant; this is a
    # backstop for runaway loops / deadlocks (which we hope to never have
    # but defending against is cheap).
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endfunction()

# --- Test registrations -----------------------------------------------------
# Phase 1 starter test: smoke-check that doctest is wired and the build
# system produces a runnable ctest binary. Trivial assertions only.
# Real tests live in subdirectories by module (math/, console/, etc.)
# and get added below as Phase 2/3 lands.
pt_add_test(harness_smoke harness_smoke_test.cpp)

# Cvar persistence round-trip. Exercises Console::SaveArchivedCvars +
# Console::ExecuteScript so demont.cfg writes are guaranteed to be
# replayable on next startup. Catches: quoting regressions, default-
# skip regressions, CVAR_ARCHIVE-flag-gate regressions.
pt_add_test(cvar_roundtrip
    cvar_roundtrip_test.cpp
    LINK_TARGETS pt_console)

# Image-diff compute kernel. Exercises pt::imgdiff::Compute / Passes
# directly with hand-crafted RGBA8 buffers -- no PNG I/O, no CLI
# parsing. Groundwork for the golden-image regression matrix (#45);
# tool driver is at tools/imgdiff/main.cpp (#69).
#
# Target name is `imgdiff_test` (not `imgdiff`) to avoid colliding with
# the tools/imgdiff/ executable target.
pt_add_test(imgdiff_test
    imgdiff_test.cpp)

# Frame-capture output-format selector + PNG round-trip (issue #94).
# Two surfaces: (1) the format parser + the `r_capture_format` cvar
# gate (`allowed_values` rejects everything other than "png"/"ppm"),
# (2) `EncodeAndWritePng` -> stbi_load round-trip on a known RGBA32F
# buffer. Links pt_capture_encoder (the stb-impl-bearing static lib)
# plus pt_console for the cvar test. stb_image's loader is compiled
# in `stb_image_impl_for_test.cpp` as a second source -- per-file
# `/w` (MSVC) / `-w` (Clang/GCC) below silences stb's warning flood
# without disabling warnings for the test code itself.
pt_add_test(capture_format
    capture_format_test.cpp
    SOURCES      stb_image_impl_for_test.cpp
    LINK_TARGETS pt_capture_encoder pt_console)
if(MSVC)
    set_source_files_properties(stb_image_impl_for_test.cpp
        PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(stb_image_impl_for_test.cpp
        PROPERTIES COMPILE_OPTIONS "-w")
endif()

# --- pt_core/AnalyticBvh unit tests (issue #67) ----------------------------
# Builder + traversal vs naive linear scan on a known input. Mirrors
# the shader-side BVH walk (shaders/PathTrace.slang:497-525) in pure
# host C++ so the data structure is testable without a GPU. Coverage:
#   - empty BVH -> no hit
#   - single primitive -> hit / miss cases
#   - ~50 primitives + ~1000 deterministic rays: BVH closest-hit ==
#     naive linear-scan closest-hit (the main coverage)
#   - many primitives sharing a split-plane coordinate (tie-break
#     stress on nth_element)
#   - AABB-degenerate primitive (zero-radius sphere)
#   - parent-AABB-contains-leaf-prim-AABBs structural invariant
# Links pt_renderer (where AnalyticBvh lives).
pt_add_test(pt_core_analytic_bvh
    pt_core_analytic_bvh_test.cpp
    LINK_TARGETS pt_renderer)

# ============================================================================
# Golden-image regression matrix (issue #45)
# ============================================================================
#
# Each cell here renders one frame headlessly via pt_render_one_frame
# (which fork+exec's demont with --smoke-frames= + --smoke-exec= +
# --smoke-capture-out=), then diff's the resulting PNG against a stored
# golden via the imgdiff CLI. Per-cell tolerance lives on the imgdiff
# invocation -- pixel-correct goldens use --max-delta 0; stochastic-
# noise-floored goldens use --max-delta / --mean-delta / --fail-percent
# sized to the cell's known variance.
#
# Layout:
#   tests/goldens/scenes/<scene>.cfg                       -- fixture
#   tests/goldens/<host>/<scene>__<backend>__<denoiser>.png-- golden
#   <build>/golden_actual/<scene>__<backend>__<denoiser>.png -- actual
#   <build>/golden_diff/<scene>__<backend>__<denoiser>.png   -- diff
#
# `<host>` is one of Darwin / Linux / Windows -- CMAKE_SYSTEM_NAME
# values. Goldens are host-specific because (a) Metal exists only on
# Darwin; (b) ULPs across architectures + driver versions add small
# host-correlated bias to floating-point reductions; (c) keeping
# host-baselines separate makes "regenerate on this host only" the
# safe default. See `Raytracer Plan/TESTING_STRATEGY.md` for the
# "how to add a cell" / "how to regenerate goldens" workflow.
#
# Per coordination note in the issue: this block is intentionally
# under its own banner so parallel agent branches appending other
# pt_add_test(...) blocks at the end of this file rebase cleanly.

# MVP cut: 1 scene x 3 backends x 1 denoiser = 3 cells. The matrix
# expands later (see "Add a cell" in TESTING_STRATEGY.md) -- the
# helper below is generic over (scene, backend, denoiser).
#
# The function name is pt_add_golden_cell -- one ctest test per call.
# Cells are skipped (NOT registered) on hosts where their backend
# isn't available: Metal cells only register on Darwin; Vulkan cells
# only register if PT_ENABLE_VULKAN_BACKEND is ON; Software is always
# available. This matches the runtime gate the engine already has.
#
# The host (Darwin/Linux/Windows) is also pulled from
# CMAKE_SYSTEM_NAME so the golden path is build-target-derived, not
# host-machine-derived. Cross-builds (rare in this repo) get the
# target's host directory, matching what their runtime would produce.
function(pt_add_golden_cell)
    set(_options)
    set(_oneval SCENE BACKEND DENOISER FRAMES SPP MAX_DELTA MEAN_DELTA FAIL_PERCENT)
    set(_multival EXTRA)
    cmake_parse_arguments(PT_CELL "${_options}" "${_oneval}" "${_multival}" ${ARGN})

    # Required args -- fail at configure time if a caller forgets one.
    # Use STREQUAL "" instead of `NOT VAR` because the denoiser
    # short-form `off` is a CMake-falsy string (same trap as the
    # boolean-coercion rules around OFF / NO / FALSE). STREQUAL "" only
    # matches a genuinely unset / empty variable.
    foreach(_req SCENE BACKEND DENOISER)
        if("${PT_CELL_${_req}}" STREQUAL "")
            message(FATAL_ERROR
                "pt_add_golden_cell: SCENE/BACKEND/DENOISER are all required "
                "(missing ${_req})")
        endif()
    endforeach()

    # Skip cells whose backend isn't compiled in. The wrapper would
    # fail at runtime anyway (engine rejects an unknown r_backend);
    # skipping at configure time keeps the ctest run clean.
    if(PT_CELL_BACKEND STREQUAL "metal" AND NOT APPLE)
        return()
    endif()
    if(PT_CELL_BACKEND STREQUAL "vulkan" AND NOT PT_ENABLE_VULKAN_BACKEND)
        return()
    endif()

    # Defaults. cmake_parse_arguments leaves variables undefined when
    # the caller doesn't pass the corresponding flag, so `NOT VAR`
    # evaluates true and the default fires.
    #
    # FRAMES default 64: at r_spp=1 the path-tracer accumulator's
    # noise floor is well below the 16-L2 per-pixel + 2-L2 mean +
    # 1%-bad-pixel imgdiff thresholds. Empirical measurement on the
    # cornell_csg scene: at 8 frames the Metal backend has ~3.4%
    # bad pixels between identical runs (HW-RT BLAS build entropy);
    # at 64 frames the noise floor settles below imgdiff's
    # quantisation. CI runners on the macos-26 image use a
    # paravirtualised GPU that's slower than a real M-series, so
    # 64 frames at 512x384 is still under a few seconds per cell.
    if(NOT PT_CELL_FRAMES)
        set(PT_CELL_FRAMES 64)
    endif()
    if(NOT PT_CELL_MAX_DELTA)
        set(PT_CELL_MAX_DELTA 16)
    endif()
    if(NOT PT_CELL_MEAN_DELTA)
        set(PT_CELL_MEAN_DELTA 2)
    endif()
    if(NOT PT_CELL_FAIL_PERCENT)
        set(PT_CELL_FAIL_PERCENT 1)
    endif()

    set(_host       ${CMAKE_SYSTEM_NAME})
    set(_cell_name  golden_${PT_CELL_SCENE}__${PT_CELL_BACKEND}__${PT_CELL_DENOISER})
    set(_scene_cfg  ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/${PT_CELL_SCENE}.cfg)
    set(_golden_png ${CMAKE_SOURCE_DIR}/tests/goldens/${_host}/${PT_CELL_SCENE}__${PT_CELL_BACKEND}__${PT_CELL_DENOISER}.png)
    set(_actual_png ${CMAKE_BINARY_DIR}/golden_actual/${PT_CELL_SCENE}__${PT_CELL_BACKEND}__${PT_CELL_DENOISER}.png)
    set(_diff_png   ${CMAKE_BINARY_DIR}/golden_diff/${PT_CELL_SCENE}__${PT_CELL_BACKEND}__${PT_CELL_DENOISER}.png)

    # Pre-create the actual / diff output directories. imgdiff writes
    # the diff PNG via stb_image_write which doesn't auto-create parent
    # dirs; the smoke-capture in pt_render_one_frame's child does
    # create_directories on its own, but doing both here keeps the
    # behaviour symmetric + lets a contributor run the imgdiff cell
    # in isolation against a stale actual from a previous run.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/golden_actual)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/golden_diff)

    # Build the pt_render_one_frame argument list. Spp / extra are
    # optional pass-throughs.
    set(_render_args
        --scene    ${_scene_cfg}
        --backend  ${PT_CELL_BACKEND}
        --denoiser ${PT_CELL_DENOISER}
        --frames   ${PT_CELL_FRAMES}
        --out      ${_actual_png})
    if(PT_CELL_SPP)
        list(APPEND _render_args --spp ${PT_CELL_SPP})
    endif()
    if(PT_CELL_EXTRA)
        string(REPLACE ";" "\n" _extra_joined "${PT_CELL_EXTRA}")
        list(APPEND _render_args --extra ${_extra_joined})
    endif()

    # Two-step ctest cell: render, then diff. Wrapped through cmake
    # -E env so we can `cd` into the binary dir + chain the commands
    # without writing a shell-specific recipe. -E true is the success
    # tail; CMake guarantees -E env, -E true, -E echo, etc. on every
    # host. The diff step's failure is what fails the cell; rendering
    # failure shows up via pt_render_one_frame's non-zero exit which
    # the chained `&&` short-circuits on.
    #
    # imgdiff thresholds: per-pixel L2 above MAX_DELTA classifies a
    # pixel as "bad"; >FAIL_PERCENT% of bad pixels OR a mean L2 above
    # MEAN_DELTA fails the cell. Default thresholds (16 / 2 / 1) leave
    # room for backend-specific FP nondeterminism (e.g. fma ordering
    # across drivers) without hiding real shader regressions, which
    # typically shift much more than 16 L2 over thousands of pixels.
    #
    # Cells skip themselves with FAIL/PASS based on whether the
    # golden PNG exists on this host. A missing golden is NOT a
    # silent pass -- ctest prints "FAILED" with a regen hint. See
    # TESTING_STRATEGY.md for the regen workflow.
    add_test(NAME ${_cell_name}_render
        COMMAND $<TARGET_FILE:pt_render_one_frame> ${_render_args}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${_cell_name}_render PROPERTIES
        TIMEOUT 120
        LABELS  "golden;golden_${PT_CELL_BACKEND}")

    if(EXISTS ${_golden_png})
        add_test(NAME ${_cell_name}_diff
            COMMAND $<TARGET_FILE:imgdiff>
                ${_actual_png}
                ${_golden_png}
                --max-delta    ${PT_CELL_MAX_DELTA}
                --mean-delta   ${PT_CELL_MEAN_DELTA}
                --fail-percent ${PT_CELL_FAIL_PERCENT}
                --diff         ${_diff_png}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(${_cell_name}_diff PROPERTIES
            TIMEOUT 30
            DEPENDS ${_cell_name}_render
            LABELS  "golden;golden_${PT_CELL_BACKEND}")
    else()
        # Missing-golden cell: tests/CMakeLists.txt knows the file's
        # supposed to exist on this host. Register a failing test
        # whose name + body tell a maintainer how to regenerate.
        add_test(NAME ${_cell_name}_diff
            COMMAND ${CMAKE_COMMAND} -E echo
                "MISSING GOLDEN: ${_golden_png}\nRegenerate with:\n  ctest -R ^${_cell_name}_render\n  cp ${_actual_png} ${_golden_png}\nSee Raytracer Plan/TESTING_STRATEGY.md for details.")
        set_tests_properties(${_cell_name}_diff PROPERTIES
            WILL_FAIL TRUE
            LABELS    "golden;golden_${PT_CELL_BACKEND};golden_missing")
    endif()
endfunction()

# --- MVP cells --------------------------------------------------------------
# 1 scene (cornell_csg) x 3 backends (software/metal/vulkan) x 1
# denoiser (off). Each host's CI job runs only the cells whose
# backend matches its primary platform (Mac -> metal, Windows ->
# vulkan, Linux -> software), filtered via the `golden_<backend>`
# ctest LABELS in .github/workflows/build.yml. Other hosts can run
# the off-platform cells too if they want; the wrapper handles the
# r_backend gate at engine runtime.
pt_add_golden_cell(SCENE cornell_csg BACKEND software DENOISER off)
pt_add_golden_cell(SCENE cornell_csg BACKEND metal    DENOISER off)
pt_add_golden_cell(SCENE cornell_csg BACKEND vulkan   DENOISER off)

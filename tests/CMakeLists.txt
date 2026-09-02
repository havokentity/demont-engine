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

# Cvar UX pass: platform-filtered listing + "0" as off-shorthand for
# enum cvars. Pins the visibility contract for cvars carrying a
# CVAR_PLATFORM_* bit (registered everywhere but hidden from
# EnumerateCVars on the wrong host) and the alias path that rewrites
# the literal "0" to a matching off / none / disabled allowed token
# before the value gate runs.
pt_add_test(cvar_ux
    cvar_ux_test.cpp
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

# --- pt_csg/CsgScene unit tests (issue #64) --------------------------------
# Manifold-backed CSG facade. Public-API coverage:
#   - empty scene -> Bake produces Empty() BakedMesh + error string
#   - single primitive leaves (box / sphere / cylinder) -> non-empty
#     mesh, AABB extents match the requested dimensions, translation
#     applied after origin-centering
#   - union of disjoint primitives preserves both surfaces (sum-of-
#     children triangle count + AABB spanning both)
#   - subtract (box - centred sphere) adds triangles, AABB still
#     bounded by the outer box
#   - intersect of disjoint primitives is empty; intersect of
#     overlapping primitives has the overlap-region AABB
#   - Reset / Dirty / AcknowledgeClean state machine
#   - id reuse via Remove+re-Add round-trip, duplicate-id rejection
#   - Remove cascades through internal nodes referencing the removed id
#   - same-process determinism: repeated Bake() is identical
#   - boolean identities: (a - a) == empty, (a intersect a) ~= a
#   - degenerate primitive rejection, SetRoot of unknown id rejected
#   - Dump emits a tree dump with the [root] marker
# CsgScene lives in pt_renderer (src/renderer/Csg/) so we link that.
# Tests assert STRUCTURAL properties (vertex count, triangle count,
# AABB extents) rather than vertex-by-vertex equality -- Manifold's
# mesh output isn't bit-stable across library versions but its
# structural properties (and same-process determinism) are.
pt_add_test(pt_csg_scene
    pt_csg_scene_test.cpp
    LINK_TARGETS pt_renderer)

# --- pt_renderer/TriangleBvh unit tests ------------------------------------
# Triangle-BVH-on-host follow-up to PR #106 (issue #44). The SW
# Möller-Trumbore linear-scan path was O(N) per ray; on a 1k-tri scene
# this crushed framerate to ~1 FPS at 1080p. The TriangleBvh class replaces
# that with O(log N) stack-based traversal of a host-built BVH that
# the engine uploads alongside vbuf / ibuf. Coverage mirrors the
# AnalyticBvh test pattern:
#   - empty BVH -> no hit
#   - malformed indices (size % 3 != 0) -> empty BVH, no crash
#   - single triangle -> hit / miss
#   - degenerate (collinear) triangle preserves traversal of nearby tris
#   - many tris on the same split-plane coord (tie stress on nth_element)
#   - ~30 random tris + 1000 deterministic rays: BVH closest-hit ==
#     naive linear-scan closest-hit
#   - parent-AABB contains all reachable leaf-tri AABBs invariant
# Links pt_renderer (where TriangleBvh lives, same lib as AnalyticBvh).
pt_add_test(pt_renderer_triangle_bvh
    pt_renderer_triangle_bvh_test.cpp
    LINK_TARGETS pt_renderer)

# --- Editor 3D-transform gizmo unit tests (issue: editor 3D gizmos) --------
# pt::renderer::EditorOverlay's hit-test + drag math + segment buffer
# layout. Pure host-side: no GPU, no device, no shader compile -- just
# the projection / ray / closest-point primitives the gizmo runs on.
pt_add_test(pt_renderer_editor_overlay
    pt_renderer_editor_overlay_test.cpp
    LINK_TARGETS pt_renderer)

# --- pt_math vec/mat/quat/transforms unit tests (issue #63) ----------------
# Phase 3 of #47. These tests pin the glm contract the engine consumes:
# RH coord system, column-major mat storage, perspectiveRH_ZO depth
# range, lookAtRH conventions, and the screen-to-world ray reconstruction
# math. Each test is a separate executable (one TU, one binary) -- same
# pattern as the other unit-test cells above. They link glm::glm only
# (no engine target), so they're pure-CPU + deterministic.
#
# pt_math_vec (target/ctest name; source: pt_math_vec_test.cpp)
#                            -- vec2/vec3/vec4: dot, cross, normalize,
#                               length, zero/near-zero pathology,
#                               randomized Cauchy-Schwarz + orthogonality
# pt_math_mat (target/ctest name; source: pt_math_mat_test.cpp)
#                            -- mat3/mat4: identity, multiplication,
#                               inverse, transpose, determinant,
#                               column-major storage contract
# pt_math_quat (target/ctest name; source: pt_math_quat_test.cpp)
#                            -- quaternion: identity, angleAxis,
#                               from Euler, multiplication, conjugate /
#                               inverse, slerp endpoints + midpoint,
#                               quat <-> mat3/mat4 round-trip,
#                               small-angle + full-circle edges
# pt_math_transforms (target/ctest name; source: pt_math_transforms_test.cpp)
#                            -- combined: TRS composition + inverse,
#                               lookAtRH places the camera at the
#                               view-space origin (clip space only
#                               exists after applying the projection),
#                               perspectiveRH_ZO near=0 / far=1 mapping,
#                               screen-to-world ray reconstruction
pt_add_test(pt_math_vec
    pt_math_vec_test.cpp
    LINK_TARGETS glm::glm)
pt_add_test(pt_math_mat
    pt_math_mat_test.cpp
    LINK_TARGETS glm::glm)
pt_add_test(pt_math_quat
    pt_math_quat_test.cpp
    LINK_TARGETS glm::glm)
pt_add_test(pt_math_transforms
    pt_math_transforms_test.cpp
    LINK_TARGETS glm::glm)

# --- pt_core_world_frame (issue #255) --------------------------------------
# The camera-anchored render frame: lattice, anchor, exactness, and the
# motion-vector reconstruction that makes a rebase invisible. Pure host
# arithmetic -- no device, no window -- so it runs everywhere in
# milliseconds. Links glm only; src/core/WorldFrame.h and
# src/renderer/Camera.h are both header-only for the parts under test,
# and the test deliberately calls the SHIPPED ViewProjRH_ZO rather than a
# copy of it so the two cannot drift apart.
pt_add_test(pt_core_world_frame
    pt_core_world_frame_test.cpp
    LINK_TARGETS glm::glm)

# --- pt_math_sphere (issue #254) -------------------------------------------
# Ray-sphere intersection at planetary radius. Mirrors the Slang in
# shaders/PathTraceMath.slang (no host entry point exists to call), pins
# the pre-#254 catastrophic cancellation it repairs, and sweeps radii
# from 1 m to 6371 km at nadir / grazing / inside / on-surface.
#
# Two non-default bits of wiring, both load-bearing:
#
#  * PT_SHADER_MATH_PATH lets the "shader mirror is still faithful" case
#    re-read the .slang and pin the literals the transcription depends
#    on, so the mirror cannot silently drift away from the shader.
#
#  * -ffast-math in Release is DELIBERATE, and the only test target that
#    asks for it. Metal compiles PathTraceMath with its default (fast)
#    math mode, which reassociates float add chains -- that is precisely
#    why the fix accumulates in int32 rather than using a Dekker /
#    Kahan style float compensation, which a reassociating compiler is
#    entitled to fold to zero (and does). Building the host mirror the
#    same way keeps the test honest about what the GPU actually runs;
#    without it the mirror would pass under arithmetic the shader never
#    sees. Note this does NOT come from pt_set_compiler_options -- test
#    binaries deliberately opt out of the library-level policy -- so it
#    is applied here explicitly.
pt_add_test(pt_math_sphere
    pt_math_sphere_test.cpp)
target_compile_definitions(pt_math_sphere PRIVATE
    PT_SHADER_MATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang")
if(NOT MSVC)
    target_compile_options(pt_math_sphere PRIVATE $<$<CONFIG:Release>:-ffast-math>)
else()
    target_compile_options(pt_math_sphere PRIVATE $<$<CONFIG:Release>:/fp:fast>)
endif()

# --- pt_math_altitude (issue #271) -----------------------------------------
# The other half of the same kernel: altitude above a sphere, which
# planetAltitude() and the spherical optical-depth march both computed as
# the naive `length(p - c) - R`. Same wiring as pt_math_sphere above and
# for the same two reasons -- the mirror re-reads the shaders so it cannot
# drift, and -ffast-math holds the host mirror to the reassociation Metal
# actually applies. This one re-reads PathTrace.slang as well, because
# unlike #254 the repair is at the CALL SITES, and a test that only pinned
# the kernel would pass with the sites left naive.
pt_add_test(pt_math_altitude
    pt_math_altitude_test.cpp)
#
# It re-reads CloudsRaymarch.slang too (issue #276). #271 was scoped to
# PathTrace.slang and left the pre-pass's own copy of planetAltitude on
# the naive form, so the engine carried two definitions of one function
# that disagreed -- and this target's substring pin could not see it,
# because it was only ever pointed at one of the two files. The added
# case COUNTS occurrences across both.
target_compile_definitions(pt_math_altitude PRIVATE
    PT_SHADER_MATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang"
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_SHADER_CLOUDSRAYMARCH_PATH="${CMAKE_SOURCE_DIR}/shaders/CloudsRaymarch.slang")
if(NOT MSVC)
    target_compile_options(pt_math_altitude PRIVATE $<$<CONFIG:Release>:-ffast-math>)
else()
    target_compile_options(pt_math_altitude PRIVATE $<$<CONFIG:Release>:/fp:fast>)
endif()

# --- pt_atmosphere (issue #257, planetary P3) ------------------------------
# The physical atmosphere model: three species, a body struct instead of
# Earth baked into four files, and the ray-versus-spherical-shell that
# replaced the cloud layer's flat slab.
#
# This target exists because two of P3's acceptance criteria are visual
# claims that a golden PNG cannot actually settle. "The twilight zenith
# reads blue, not brown" is a statement about a per-channel transmittance
# ratio along a grazing path, and "distant clouds merge with the horizon"
# is a statement about a chord length being finite. Both are computed and
# asserted here; the goldens then show that the same numbers reached the
# screen.
#
# Same wiring and the same two reasons as pt_math_sphere / pt_math_altitude
# / pt_math_epsilon: the mirror re-reads the shaders so it cannot drift,
# and -ffast-math holds the host mirror to the reassociation Metal applies.
# It re-reads FOUR shaders, because the model was extracted OUT of three of
# them and a test that pinned only the new module would pass with a stale
# private copy left behind in any of the others -- which is precisely how
# #276 stayed green for a cycle.
pt_add_test(pt_atmosphere
    pt_atmosphere_test.cpp)
target_compile_definitions(pt_atmosphere PRIVATE
    PT_SHADER_MATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang"
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_SHADER_CLOUDSRAYMARCH_PATH="${CMAKE_SOURCE_DIR}/shaders/CloudsRaymarch.slang"
    PT_SHADER_PATHTRACECLOUD_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceCloud.slang"
    PT_SHADER_STARSCOMPOSITE_PATH="${CMAKE_SOURCE_DIR}/shaders/StarsComposite.slang"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")
if(NOT MSVC)
    target_compile_options(pt_atmosphere PRIVATE $<$<CONFIG:Release>:-ffast-math>)
else()
    target_compile_options(pt_atmosphere PRIVATE $<$<CONFIG:Release>:/fp:fast>)
endif()

# --- pt_sky_units (issue #280) ---------------------------------------------
# The radiometric unit rebase, Hillaire 2020's multiple-scattering table,
# and the retirement of the compensating r_rayleigh / aerosol pair.
#
# It links pt_renderer because #280's physics has a HOST implementation --
# src/renderer/Atmosphere.cpp -- rather than only a shader one. That is the
# whole point of building the table on the CPU: the builder is also the
# software backend's mirror and the sky cook, so there is one implementation
# to test rather than a shader plus a transcription of it. This target
# therefore tests the real code, and re-reads the .slang only to pin that
# the GPU side still carries the same constants and the same
# parameterisation.
#
# The pins COUNT occurrences rather than testing find() != npos: a substring
# pin is satisfied by one correct copy however many stale ones survive,
# which is how #276 stayed green for a cycle.
pt_add_test(pt_sky_units
    pt_sky_units_test.cpp
    LINK_TARGETS pt_renderer)
target_compile_definitions(pt_sky_units PRIVATE
    PT_SHADER_MATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang"
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")

# --- pt_planet_terrain (issue #258, planetary P4) --------------------------
# The cubed-ellipsoid quadtree, the elevation field, chunk baking and the
# residency selector.
#
# Three of P4's acceptance criteria are claims a golden PNG cannot settle,
# because a PNG only shows that one camera saw no crack:
#   * watertightness is an equality between two chunks at DIFFERENT levels
#     over the WHOLE domain, including all 12 cube seams;
#   * "golden determinism" is a statement that an asynchronously baked chunk
#     is byte-identical to a synchronously baked one;
#   * the LOD rule's sub-pixel property and its hysteresis are numbers.
# All three are asserted here; the goldens then show the same surface
# reached the screen.
#
# -ffast-math in Release like the other numeric targets, and the finiteness
# checks use the volatile-barrier finiteBits() pattern rather than
# std::isfinite -- under __FINITE_MATH_ONLY__ the latter folds to `true`
# and an assertion goes quietly vacuous, which is exactly how PR #273
# shipped a check that could not fail.
pt_add_test(pt_planet_terrain
    pt_planet_terrain_test.cpp
    LINK_TARGETS pt_renderer)
if(NOT MSVC)
    target_compile_options(pt_planet_terrain PRIVATE $<$<CONFIG:Release>:-ffast-math>)
else()
    target_compile_options(pt_planet_terrain PRIVATE $<$<CONFIG:Release>:/fp:fast>)
endif()
# The chunk-bake cases walk a level-19 chunk's full subdivision hierarchy,
# which is ~22 000 field evaluations per chunk; the default 60 s backstop is
# tight for that on a loaded machine.
#
# TIMEOUT 600, measured rather than guessed, and now carrying deliberate
# slack. History: the 700 306-assertion suite took 157.9 s at -O0 on an M4
# Max, and 600 s (~3.8x) was meant to leave room for a ~2x-slower runner.
# The shared macos-26 and windows-latest runners turned out to be ~3.7x
# slower than that M4 Max, so at -O0 the test crept to ~585 s -- inside the
# budget by a hair, and every planetary change tipped a lane over 600.
#
# The budget was NOT raised (it went 180 -> 600 once already; raising it on
# the second occurrence of the same signal hides the trend) and the sweep
# was NOT trimmed (CI builds only Debug, so a Debug-only cut means the
# exhaustive half never runs in CI -- a labelled test nobody executes, the
# failure this file warns about two comments up). Two independent fixes
# bought the headroom back instead, and they compound:
#
#   * BUILD (#322): the compute-bound field TUs -- CubedSphere,
#     ElevationField, TerrainChunk -- are compiled at -O2 even in Debug (see
#     src/renderer/CMakeLists.txt). -O2 WITHOUT -ffast-math is IEEE
#     operation-by-operation exact, so the goldens do not move and the
#     intra-binary sync-vs-async and watertightness assertions hold by
#     construction. On an M4 Max this took the whole test from ~308 s to
#     ~20 s; on the Windows Debug lane it went from a >600 s timeout to
#     389 s (MSVC keeps checked-iterator STL in the TUs -O2 does not cover,
#     so the multiple is smaller than clang's).
#
#   * SOURCE (#307): the reference-slope bake this feature added ran per
#     chunk vertex, and the residency cases bake thousands of chunks, so the
#     per-bake regression was multiplied by the sweep -- +81 s (+26%) as
#     first written, every one of the nine slowest cases up 22-31% (the
#     signature of "every bake got dearer", not a new case or a hang).
#     Memoised (see ReferenceSlope01 / RockFractionFromRmsSlope, bit-exact
#     and pinned by the goldens) it is +18 s; per level-13 bake, Release, the
#     +41% became +11%. Worth doing on its own account, not just for CI: the
#     same per-bake cost sets how long a streaming hole persists on screen.
#
# A slow honest test stayed honest and got fast. If it ever needs to get
# faster again: cut redundancy, not cases.
set_tests_properties(pt_planet_terrain PROPERTIES TIMEOUT 600)

# --- pt_planet_residency (paced terrain streaming) -------------------------
# Chunk residency on the PACED streaming path.
#
# Separate from pt_planet_terrain because it needs a real RHI device: the
# claim under test is about PlanetTerrain::Update, which owns the arena, the
# BLAS builds and the wall-clock pacing, and none of those exist in
# pt_renderer. Links pt_engine (for PlanetTerrain) and pt_rhi (for the
# device).
#
# WHY IT CANNOT BE FOLDED INTO A GOLDEN. Every planet fixture calls
# PlanetTerrain::Settle(), which is a barrier and runs UNPACED -- a settled
# capture is a fully-resident one by construction, so a golden cannot
# observe a streaming transient at all. Asserting "no holes" after a settle
# would be a check that passes without testing anything, which is the exact
# shape this repo keeps rediscovering. This test never calls Settle().
#
# Software backend only, and honestly SKIPPED (125) rather than green when
# no device comes up. The residency policy is backend-independent -- it is
# CPU set arithmetic over quadtree keys -- so a second backend would
# re-measure the same thing at the cost of a GPU-contended CI cell.
#
# TIMEOUT 300: the flight is 180 ticks with a 4 ms frame emulation and a
# software BLAS build per admitted chunk; measured well under a minute on an
# M4 Max, and CI runners are slower at BVH builds by a wide margin.
if(APPLE OR WIN32)
    pt_add_test(pt_planet_residency
        pt_planet_residency_test.cpp
        LINK_TARGETS pt_engine pt_rhi)
    set_tests_properties(pt_planet_residency PROPERTIES
        TIMEOUT          300
        SKIP_RETURN_CODE 125
        LABELS           "planetary;residency")
endif()

# --- pt_planet_shader (issue #258, planetary P4) ---------------------------
# The host mirror of the shader-side terrain contract: the instance
# descriptor's wire format, the three terrain bindings across Slang AND the
# Vulkan slot table / layout / pool, the cone_spread equality, and the
# roughness fold.
#
# It re-reads THREE files, and that breadth is the point: the phase is
# tested on Metal, so the Vulkan half of the binding contract has no
# runtime check at all on this machine, and a shader binding the descriptor
# layout does not declare is a hard vkCreateComputePipelines failure.
#
# Every assertion COUNTS occurrences rather than testing for presence. A
# `find() != npos` pin passes as soon as one site matches, which is exactly
# how a mirror keeps passing after the shader grows a second copy of the
# mirrored thing -- how #276 shipped a stale duplicate of planetAltitude
# while pt_math_altitude stayed green.
pt_add_test(pt_planet_shader
    pt_planet_shader_test.cpp
    LINK_TARGETS pt_renderer)
target_compile_definitions(pt_planet_shader PRIVATE
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_VULKAN_DEVICE_PATH="${CMAKE_SOURCE_DIR}/src/rhi_vulkan/VulkanDevice.cpp"
    PT_PLANET_TERRAINCHUNK_H_PATH="${CMAKE_SOURCE_DIR}/src/renderer/Planet/TerrainChunk.h"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")

# --- pt_planet_albedo (issue #300, land cover) -----------------------------
# The surface albedo raster, the two physical models that modulate it, and
# the wire formats between the CPU sampler and the shader's.
#
# WHY IT IS NOT A GOLDEN. Three reasons, all recorded at the top of the
# source file, but the first is the one that decided it: `planet_surface`
# stands at 27.99 N with a 4.7 km horizon and the raster's texel is 19.5 km,
# so the raster contributes ONE COLOUR to that entire frame. No
# surface-level golden can tell a correct raster from a wrong one -- only
# from an absent one. And "Earth from orbit is recognisable" is a claim
# about SPECTRAL SHAPE (green where chlorophyll absorbs red and blue, tan
# where iron oxide absorbs blue), which a pixel comparison passes just as
# happily if both images are grey.
#
# It reads FIVE source files, and that breadth is deliberate: this phase
# added a Vulkan descriptor binding and grew PtPush past both backends'
# push-constant staging buffers, and neither has a runtime check on a Mac.
# The push overflow in particular shipped silently on Metal AND Vulkan at
# once -- the terrain simply ignored its new raster with no error anywhere
# -- so the pins cover the shared constant, both backends' use of it, and
# the compile-time guard that now catches the next one.
pt_add_test(pt_planet_albedo
    pt_planet_albedo_test.cpp
    LINK_TARGETS pt_renderer)
target_compile_definitions(pt_planet_albedo PRIVATE
    PT_PLANET_ALBEDO_PATH="${CMAKE_SOURCE_DIR}/assets/planet/earth_lite.ptalb"
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_VULKAN_DEVICE_PATH="${CMAKE_SOURCE_DIR}/src/rhi_vulkan/VulkanDevice.cpp"
    PT_VULKAN_DEVICE_H_PATH="${CMAKE_SOURCE_DIR}/src/rhi_vulkan/VulkanDevice.h"
    PT_METAL_DEVICE_PATH="${CMAKE_SOURCE_DIR}/src/rhi_metal/MetalDevice.h"
    PT_RHI_TYPES_PATH="${CMAKE_SOURCE_DIR}/src/rhi/Types.h"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")

# --- pt_math_epsilon (issue #256, planetary P2) ----------------------------
# Scale-relative ray epsilons: the Wachter-Binder representational offset
# and the ray-cone footprint term that replaced the absolute 1e-3 m every
# secondary ray used to carry. Mirrors shaders/PathTraceMath.slang the same
# way pt_math_sphere and pt_math_altitude do -- there is no host entry point
# to call -- and re-reads THREE shaders, because unlike #254 the repair is
# as much at the seventeen call sites as it is in the kernel, and a test
# that pinned only the kernel would pass with every site left at 1e-3.
#
# It also re-reads src/engine/Engine.cpp to pin r_origin_rebase_radius.
# That is not decoration: Wachter-Binder's constants are calibrated for
# coordinates in roughly [1e-2, 1e5] m, and P1's anchor is what bounds them
# to that band. If the rebase radius ever moves by orders of magnitude the
# offset scheme has to be re-derived, so the dependency is pinned rather
# than asserted in prose.
#
# -ffast-math in Release for the same reason the other two math targets ask
# for it: Metal compiles these modules with its default (fast) math mode,
# and the host mirror should face the same reassociation the GPU applies.
pt_add_test(pt_math_epsilon
    pt_math_epsilon_test.cpp)
target_compile_definitions(pt_math_epsilon PRIVATE
    PT_SHADER_MATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang"
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_SHADER_RESTIRFINAL_PATH="${CMAKE_SOURCE_DIR}/shaders/RestirFinal.slang"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")
if(NOT MSVC)
    target_compile_options(pt_math_epsilon PRIVATE $<$<CONFIG:Release>:-ffast-math>)
else()
    target_compile_options(pt_math_epsilon PRIVATE $<$<CONFIG:Release>:/fp:fast>)
endif()

# --- pt_physics unit tests (issue #132) ------------------------------------
# Phase 1 Verlet physics: handle invalidation (Remove+Add generation
# bump; 24-bit gen wraparound guard), free-fall trajectory matches
# closed-form Verlet, damping monotonically bleeds velocity, sphere-
# plane contact (bottom-y >= 0 after substep), sphere-sphere
# correction splits overlap by inv_mass, stacked-tower invariant
# (the post-sphere-sphere plane pass), Clear lifecycle, pool-full
# kInvalidHandle. Links pt_physics directly.
pt_add_test(pt_physics_test
    pt_physics_test.cpp
    LINK_TARGETS pt_physics)

# --- pt_sph SmokeSPH unit tests (Fluid Phase 3, #22) -----------------------
# Pins the CPU SPH smoke solver's mechanics: Clear lifecycle, particle
# cap enforcement, emitter spawn rate, gravity / buoyancy / wind /
# shockwave force responses, and the PackForGpu 2-float4-per-particle
# layout the path-tracer slot 16 SSBO consumes.
pt_add_test(pt_smoke_sph
    pt_smoke_sph_test.cpp
    LINK_TARGETS pt_physics)

# --- FFT ocean surface (Wave 8, #25) ---------------------------------------
# Pins the CPU Tessendorf ocean solver: Phillips-spectrum determinism,
# the deep-water dispersion w = sqrt(g*k) time evolution, the radix-2
# inverse FFT, unit-length upward normals, foam clamped to [0,1], and
# the wind -> RMS-height scaling (Phillips L = V^2/g). Links pt_physics
# only -- the solver is a leaf module with no RHI / engine dependency.
pt_add_test(pt_ocean_fft
    pt_ocean_fft_test.cpp
    LINK_TARGETS pt_physics)
# --- end FFT ocean surface -------------------------------------------------

# --- pt_planet_ocean (issue #259, planetary P5) ----------------------------
# The planetary ocean: the Cox & Munk endpoint, the geometry-to-BRDF
# conservation identity, the boat-to-orbit continuity refinement (with a
# hard-switch control that plateaus), the cascade band partition, the
# lattice-quantised tangent anchor, and the below-to-above crossing test
# that has been one-sided since #25.
#
# Links pt_physics for the solver + the Cox-Munk kernels and pt_renderer for
# PlanetSite / the WGS-84 ellipsoid, so the sea-level-versus-bathymetry case
# measures the real ellipsoid rather than a transcription of it.
pt_add_test(pt_planet_ocean
    pt_planet_ocean_test.cpp
    LINK_TARGETS pt_physics pt_renderer)
target_compile_definitions(pt_planet_ocean PRIVATE
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_ENGINE_H_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.h"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp")

# --- pt_ocean_gpu (#259 planetary P5 / #133 water roadmap Phase 2) ---------
#
# THE ACCEPTANCE TEST FOR THE GPU CASCADE SOLVER. Runs both solvers on the
# same seeded Phillips base spectrum -- the CPU pt::ocean::OceanFFT and the
# shaders/OceanCascades.slang compute pre-pass -- and compares the packed
# fields texel by texel against a bound DERIVED from float32 radix-2 FFT
# arithmetic (see the header of pt_ocean_gpu_test.cpp; ~4.0e-4 relative
# 2-norm at grid 256, and ~96% of it is the host's recurrence twiddle, not
# the kernel's). Also asserts the GPU field is bit-identical across eight
# consecutive runs, which is what the three P5 golden cells depend on.
#
# Needs a REAL Metal device: it creates one headlessly the way
# rhi_accel_update_test.cpp does, builds the "ocean_cascades" pipeline,
# dispatches, and reads the atlas back. Links pt_rhi (for the device) and
# pt_physics (for the reference solver + the Cox-Munk kernels); the push
# layout and the five-pass sequence come from
# src/physics/OceanCascadeDispatch.h, shared verbatim with
# Engine::DispatchOceanCascades so there is one transcription of the
# shader's wire format and not two.
#
# Apple-only and SKIP_RETURN_CODE 125: there is no Vulkan pipeline for this
# kernel yet (VulkanDevice's shared descriptor-set layout stops at binding
# 39 and the kernel declares 40..45) and the software backend has no
# compute pipelines at all, so on those the engine takes the CPU solver and
# there is nothing to compare. A host without a usable Metal device reads
# as SKIPPED, never as green -- the #252 failure mode.
#
# 300s timeout: MetalDevice compiles every MSL kernel in its constructor
# and the test constructs one per TEST_CASE.
if(APPLE)
    add_executable(pt_ocean_gpu pt_ocean_gpu_test.cpp)
    # pt_physics is listed explicitly even though pt_rhi already drags it
    # in four hops down (pt_rhi -> pt_rhi_metal -> pt_app_window ->
    # pt_console -> pt_editor -> pt_physics). Apple's ld64 then prints one
    # `ignoring duplicate libraries` note, because CMake emits the archive
    # at both depths; it is informational and the link is correct. The
    # alternative -- depending on that six-target chain to keep supplying
    # pt::ocean::OceanFFT -- is a worse trade than one linker note.
    target_link_libraries(pt_ocean_gpu PRIVATE doctest::doctest pt_rhi pt_physics)
    target_compile_definitions(pt_ocean_gpu PRIVATE
        PT_ENGINE_H_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.h")
    if(COMMAND pt_target_warnings)
        pt_target_warnings(pt_ocean_gpu)
    endif()
    add_test(NAME pt_ocean_gpu COMMAND pt_ocean_gpu
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(pt_ocean_gpu PROPERTIES
        TIMEOUT          300
        SKIP_RETURN_CODE 125
        LABELS           "ocean;planetary;metal")
endif()
# --- end planetary ocean ---------------------------------------------------

# --- pt_water_optics (issue #261, planetary P7) ----------------------------
# Underwater: the Pope & Fry 1997 absorption table and the Morel 1974
# scattering fit as published measurements rather than as a tint, the
# descent-to-200 m sequence verified four ways (measured coefficients, an
# ordering that holds at every depth with no tolerance, a Lipschitz
# refinement ladder with a hard-switch control that plateaus, and the
# march's own second-order convergence against the closed form the
# homogeneous column has), the medium stack that replaces a bit which
# could not represent nesting, and Snell's window with the refracted sun
# direction that makes it shade rather than merely sit in the right place.
#
# Links pt_physics only -- pt::water is a leaf module over glm with no RHI
# or engine dependency, so the tests exercise the real code rather than a
# transcription of it. The shader kernels ARE transcribed (there is no host
# entry point for them), and the last case re-reads all four sources to pin
# what the transcriptions depend on.
pt_add_test(pt_water_optics
    pt_water_optics_test.cpp
    LINK_TARGETS pt_physics)
target_compile_definitions(pt_water_optics PRIVATE
    PT_SHADER_PATHTRACE_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang"
    PT_SHADER_PATHTRACEMATH_PATH="${CMAKE_SOURCE_DIR}/shaders/PathTraceMath.slang"
    PT_ENGINE_CPP_PATH="${CMAKE_SOURCE_DIR}/src/engine/Engine.cpp"
    PT_WATER_OPTICS_CPP_PATH="${CMAKE_SOURCE_DIR}/src/physics/WaterOptics.cpp")
# --- end underwater --------------------------------------------------------

# --- phys_drop_* RGB arg shapes (PR #189 / issue #181 follow-up) -----------
# Pins the arg-shape contract for `phys_drop_sphere` (3, 4, 5, or 8 args)
# and `phys_drop_box` (6, 7, or 10 args) plus the rejection rules
# (partial-RGB shapes 6/7-arg sphere + 8/9-arg box rejected with usage
# hint, negative RGB rejected with rgb-non-negative error, HDR-bright
# RGB > 1.0 accepted as-is). Constructs an Engine with the physics
# subsystem manually wired, registers the phys_drop_* commands via
# the friend-struct test access, drives them through Console::Execute,
# and asserts on BOTH the success-log output string AND the underlying
# AnalyticPrim albedo so a regression where the two diverge can't
# slip through. Links pt_engine (which transitively pulls in everything
# pt_engine needs -- pt_console, pt_physics, pt_renderer, pt_audio,
# etc.; this is the same surface phys_drop_* runs against in production).
pt_add_test(phys_drop_args_test
    phys_drop_args_test.cpp
    LINK_TARGETS pt_engine)

# --- Named camera bookmarks (cam_save_named / cam_load_named) ----------------
# Pins the named-camera-bookmark commands added on
# `feat/cam-named-bookmarks`. Round-trips save -> load (camera pose
# preserved within ClampPitch tolerance), replace semantics
# (re-saving same name overwrites without error), rejection paths
# (empty / '#'-prefixed name, missing bookmark on load), listing
# (empty + populated), deletion, and the cfg-file persistence path
# (SaveCameraBookmarksToDisk -> LoadCameraBookmarksFromDisk recovers
# the same map verbatim). Friend-struct CamBookmarksTestAccess wires
# JUST RegisterCameraBookmarkCommands + a camera_ unique_ptr -- the
# full RegisterCommands path would capture `this` across audio /
# RHI / window state that isn't initialised in the test harness.
pt_add_test(cam_bookmarks_test
    cam_bookmarks_test.cpp
    LINK_TARGETS pt_engine)

# --- Light photometric decomposition (#176) --------------------------------
# Pins the photometric conversion constants the new `light_*_color` /
# `light_*_cd` / `light_*_lm` / `light_*_nits` / `light_*_exposed`
# command variants use to map artist-facing units onto the canonical
# W/sr (point/spot) and W/m^2/sr (sphere/quad) storage. Covers:
#   - 1cd = 1/683.002 W/sr at 555nm peak photopic (within 1e-6)
#   - 100cd point at 5m -> 4 lux illuminance acceptance scenario
#   - lumen -> W/sr conversion uses 4*pi denominator for isotropic
#   - nit -> W/m^2/sr equals the cd -> W/sr factor (luminance vs
#     luminous intensity, same SI definition)
#   - decomposition algebra final = color * scalar * 2^ev (EV=0,+1,
#     -1,+3, half-stop)
#   - cfg round-trip: the save path is canonical-form so the conversion
#     happens exactly once at input; reload is a pure float identity
#   - linearity: 2x cd -> 2x W/sr, no hidden non-linearity
# Pure-math test, no engine link -- the constants are mirrored from
# Engine.cpp's anonymous namespace per the test's own contract block.
pt_add_test(pt_light_decompose
    pt_light_decompose_test.cpp)

# --- pt_renderer/Astronomy unit tests (issue #65) --------------------------
# JD conversions (J2000 + Unix-epoch reference points), GMST at J2000
# vs the IAU 1982 formula, sun position at four cardinal solar events
# (J2000 + 2026 equinoxes + solstices) cross-checked against JPL
# HORIZONS / Naval Almanac references, equatorial -> horizon transform
# sanity (sun at zenith on equator/equinox, sunrise/sunset diurnal
# sign-change), plus the Moon API surface (position invariants /
# determinism, phase-angle synodic-cycle range, distance perigee..
# apogee envelope), sun distance (perihelion..aphelion envelope), and
# worldToJ2000Matrix (right-handed orthonormal rotation). All
# reference values are hardcoded -- no time() / no system clock --
# so the test is deterministic across host OS + compiler + build type.
# Links pt_renderer (where Astronomy lives).
pt_add_test(pt_renderer_astronomy
    pt_renderer_astronomy_test.cpp
    LINK_TARGETS pt_renderer)

# --- Planetarium unit tests (issue #281) ----------------------------------
# Planet positions are host-side, closed-form and deterministic, so this is
# where the planetarium is actually verified -- the golden cell downstream is
# only a smoke check that the compositing dispatch runs.
#
# Coverage:
#   - JPL approximate-element table integrity, checked structurally rather
#     than by re-reading the table: perihelion/aphelion envelopes and
#     sidereal periods against the NASA fact sheets, and the geometric
#     elongation ceilings for Mercury (27.8 deg) and Venus (47.3 deg) that
#     no mistyped element can survive
#   - geocentric astrometric RA/Dec for all seven bodies against JPL
#     HORIZONS at 2026-01-01, with per-body tolerances taken from JPL's own
#     published error table for this element set (15" Mercury .. 600" Saturn)
#   - both ends of the 1800-2050 validity window (1900 Jupiter, 2049
#     Venus + Mars), where a wrong rate column would show and a J2000-adjacent
#     epoch would not
#   - the 2020 Jupiter-Saturn great conjunction: a two-body test that only
#     passes if both element rows are right at once
#   - MOTION, not just position: Mars's 2020 retrograde loop sample-for-sample
#     against HORIZONS (sign pattern + 13-degree amplitude), and its two
#     stationary points landing on the published 2020-09-09 / 2020-11-14
#   - light-time correction present, correctly signed (lag, not lead)
#   - Mallama et al. 2017 photometry against HORIZONS APmag, plus the
#     Saturn ring-opening term that is worth a whole magnitude
#   - the shared point-source scale: a rasterised catalogue star's peak texel
#     must equal MagnitudeToFlux() of its magnitude, i.e. a planet and a star
#     of the same magnitude are the same brightness by construction
#   - B-V -> tint hue-only (unit luminance) and monotone in colour index
#   - a StarsComposite.slang mirror pin that COUNTS occurrences, so a change
#     touching only one of the kernel's two push declarations fails
#
# Built with -ffast-math in Release like the rest of the project, so the
# finiteness assertions go through the volatile-barrier finiteBits() pattern
# from pt_math_altitude_test.cpp -- std::isfinite folds to constant true
# under __FINITE_MATH_ONLY__ and would make those checks vacuous.
#
# Links pt_renderer (Astronomy + BscCatalog both live there).
pt_add_test(pt_renderer_planets
    pt_renderer_planets_test.cpp
    LINK_TARGETS pt_renderer)
target_compile_definitions(pt_renderer_planets PRIVATE
    PT_SHADER_STARSCOMPOSITE_PATH="${CMAKE_SOURCE_DIR}/shaders/StarsComposite.slang")

# --- pt_renderer/LightTree unit tests (issue #129 + #177) -----------------
# Pins the hierarchical-light-tree builder against the GPU's
# (right = left + 1) reachability invariant.  Pre-fix (PR #177) the
# builder reserved one slot for the parent then let recursive calls
# append the left subtree at parent+1 and the right subtree at
# parent + 1 + |left subtree|, so right was only at left + 1 when the
# left subtree was a single leaf.  At N >= 4 the right subtree became
# unreachable from the root and the shader's `pickLightFromTree` walk
# could only see half of the input lights.
#
# Coverage:
#   - empty input -> empty tree (no leak, no crash)
#   - N=1 -> single leaf node
#   - N=4 explicit cell (the size that exposed the bug)
#   - N=1..256 exhaustive sweep + 200/500/1000 MVP-scale samples
#   - coincident-lights fallback path (centroid AABB collapses to 0)
#   - parent-AABB-encloses-children + intensity-sums-to-children
#     structural invariants (would catch a "swapped left/right" or
#     "wrong combine" regression even on a future leaf-layout change)
# Links pt_renderer (where BuildLightTree lives).
pt_add_test(pt_renderer_light_tree
    pt_renderer_light_tree_test.cpp
    LINK_TARGETS pt_renderer)

# --- Hosek-Wilkie sky cook unit tests (real ArHosekSkyModel dataset) -------
# Pins the physics the previous fabricated coefficient tables destroyed:
#   - a strongly blue-dominant zenith on a clear day (b/r ~3.3), with exact
#     magnitude spot-checks so a corrupted/truncated embedded dataset fails
#   - the zenith blue channel never clamps to 0 as turbidity rises to 10
#     (the specific "olive/yellow-grey sky" regression the audit found), and
#     desaturation toward white is monotone (never inverts)
#   - ground albedo brightens the dome (correct sign; the old response was
#     near-inert / partly backwards)
#   - the cook stays finite at + beyond the [1,10] turbidity / [0,90deg]
#     elevation domain edges (no OOB index, no pow() of a negative base)
# Pure-header doctest against engine/HosekSkyModel.h -- no link needed (the
# cook + eval + embedded dataset are all header-only, inline constexpr). This
# is the CI-portable guard for the Hosek fix; the GPU golden only runs on Mac.
pt_add_test(pt_hosek_sky
    pt_hosek_sky_test.cpp)

# --- pt_renderer/HdrImage env-map CDF unit tests (issue #66) ---------------
# Phase 3 of #47. Pins the marginal/conditional luminance-CDF builder
# the env-map MIS sampler depends on. The CDF feeds NEE / env-map MIS
# (`r_mis`); a wrong CDF silently biases every sky importance sample.
#
# The CDF construction currently lives inlined inside
# Engine::ReloadEnvMap (src/engine/Engine.cpp ~lines 2369-2413). Until
# it's factored into a public helper, the test file re-implements the
# exact same algorithm (BuildEnvCdf reference) and asserts its
# mathematical contract:
#   - Marginal CDF is monotone non-decreasing, terminates at 1.0
#   - Each conditional row's CDF is monotone non-decreasing, terminates
#     at 1.0 (for lit rows) or stays at 0.0 (for dark rows)
#   - 4x4 known greyscale image matches a hand-computed reference
#   - Single bright pixel makes the CDF step exactly at that pixel
#   - Zero-luminance image leaves CDFs at 0.0 with no NaN/Inf
#   - Dark-row mixed with lit rows behaves correctly
#   - Uniform-white image follows sin(theta) lat-long weighting
#   - Round-trip sampling matches input distribution (chi-square, p > 0.01)
#   - Closed-form PDF (lum * W * H / (2 pi^2 * total)) matches
#     pixel-prob / lat-long Jacobian; degenerate guards return 0
#     (no NaN/Inf)
#   - Builder is deterministic across runs
# Pure-header doctest -- no link against pt_renderer (the algorithm is
# inlined in the test). When ReloadEnvMap's CDF builder is factored
# into a public function, swap this test over to call that function
# directly and delete the reference re-impl.
pt_add_test(hdrimage_cdf
    hdrimage_cdf_test.cpp
    LINK_TARGETS pt_renderer)

# --- Wave 9 scene save/load round-trip -------------------------------------
# Pins pt::editor::SceneToJson / SceneFromJson -- the lossless JSON
# serializer behind the scene_save / scene_load console commands. The
# feature's acceptance bar is a save -> clear -> load round-trip that
# reproduces prims (varied materials) + lights (all 4 types) + camera +
# SDF clusters + render cvars exactly; this test asserts that at the
# serialization layer (bit-exact float compares; no GPU, no Engine
# instance). Links pt_editor (where the serializer lives). pt_editor
# already exposes pt_renderer / pt_physics PUBLIC, so AnalyticPrim /
# AnalyticLight (Engine.h, header-only structs), pt::renderer::SdfPrim +
# ComputeSdfAabb, and pt::renderer::Camera all resolve without pulling
# in pt_engine / the GPU backends.
pt_add_test(scene_serialize
    scene_serialize_test.cpp
    LINK_TARGETS pt_editor)

# --- RHI acceleration-structure contract (issue #254 P0 / issue #251) ------
# The only test in the tree that drives a real RHI Device: it creates a
# headless device per backend, builds a BLAS + an N>1 TLAS, and
# dispatches shaders/AccelProbe.slang (or its CPU mirror) to read
# CommittedInstanceID() back per instance. Two things it pins that
# nothing else can:
#   - TLASInstance::instance_id survives the round trip on EVERY
#     backend. Vulkan always honoured it; Metal silently dropped it
#     (#251) and no test could see that, because the engine has only
#     ever built one instance so the id was always 0 and the backends
#     agreed by coincidence.
#   - Device::UpdateTLASInstances re-points a live TLAS and the change
#     is observed by the next dispatch, with the handle unchanged and
#     AccelGpuStallCount() unmoved -- i.e. no waitUntilCompleted /
#     vkQueueWaitIdle / vkDeviceWaitIdle on the update path.
# Links pt_rhi, which brings in whichever backends this build compiled
# (PT_HAS_*_BACKEND are PUBLIC defines on that target). The suite body
# is backend-agnostic and skips itself when a backend is absent, has no
# acceleration-structure support, or cannot produce the accel_probe
# pipeline -- so the same file is correct on Mac (metal + software),
# Windows (adds vulkan when a headless VkDevice comes up) and Linux.
#
# Whether a backend can actually host an acceleration structure is a
# RUNTIME device capability, so the suite queries the device
# (`SupportsHardwareRT()`, now backed by MTLDevice.supportsRaytracing)
# rather than assuming it, and records a per-backend reason when it
# cannot. The binary also prints that ledger, including an explicit
# note when Metal -- the only backend that can reproduce #251 -- was
# skipped.
#
# The executable is built once; ctest then gets ONE CELL PER BACKEND,
# mirroring pt_add_golden_cell's `golden_<scene>__<backend>__<denoiser>`
# shape and its `LABELS "golden;golden_${BACKEND}"`. Each cell runs the
# same binary filtered to that backend's doctest case (`-tc=`), so a
# backend that cannot run says so on its OWN ctest line:
#
#     rhi_accel_software .... Passed
#     rhi_accel_metal ....... ***Skipped
#
# That visibility is the entire point. A single aggregate cell would
# print `Passed` on a GH-hosted macOS runner -- true, because the
# software backend really did run the full suite -- while saying nothing
# about Metal having been skipped, and ctest suppresses a passing test's
# stdout so the in-binary coverage ledger never reaches the log. One
# cell per backend puts the skip where a reader cannot miss it.
#
# Registration follows pt_add_golden_cell's configure-time gating
# exactly: a cell is only registered where its backend is compiled in
# (metal => APPLE, software => APPLE OR WIN32, vulkan =>
# PT_ENABLE_VULKAN_BACKEND), so no host carries a cell that could only
# ever skip.
#
# SKIP_RETURN_CODE 125: the binary exits 125 when the backend it was
# filtered to could not run -- no device, no hardware ray tracing, or no
# accel_probe pipeline -- and ctest reports `***Skipped` rather than a
# green tick over zero coverage. That is the #252 failure mode and it is
# worth a return code to avoid. Verified in CI: the Linux lane, which
# builds neither Metal nor the software backend, correctly reports
# Skipped instead of passing.
#
# 120s timeout rather than the 60s default: the Metal backend compiles
# every MSL kernel in its constructor (PathTrace.slang is the long pole)
# and a cold CI runner with a paravirtualised GPU is a lot slower at it
# than an M4 Max.
add_executable(rhi_accel_update rhi_accel_update_test.cpp)
target_link_libraries(rhi_accel_update PRIVATE doctest::doctest pt_rhi)
if(COMMAND pt_target_warnings)
    pt_target_warnings(rhi_accel_update)
endif()

# _backend: doctest case suffix + ctest cell name + label.
function(pt_add_accel_cell _backend)
    set(_cell rhi_accel_${_backend})
    # The filter MUST be quoted. Unquoted, CMake treats the parentheses
    # as argument separators and expands `--test-case=*(metal)*` into
    # five arguments -- `--test-case=*`, `(`, `metal`, `)`, `*` -- so
    # doctest sees a match-everything filter and every cell silently
    # runs every backend. That reproduces exactly the failure this
    # per-backend split exists to prevent, with all cells reporting the
    # same thing.
    add_test(
        NAME    ${_cell}
        COMMAND rhi_accel_update "--test-case=*(${_backend})*"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(${_cell} PROPERTIES
        TIMEOUT          120
        SKIP_RETURN_CODE 125
        LABELS           "rhi_accel;rhi_accel_${_backend}")
endfunction()

if(APPLE OR WIN32)
    pt_add_accel_cell(software)
endif()
if(APPLE)
    pt_add_accel_cell(metal)
endif()
if(PT_ENABLE_VULKAN_BACKEND)
    # Skips by default even here -- VulkanDevice has no headless path.
    # See the Vulkan TEST_CASE in rhi_accel_update_test.cpp.
    pt_add_accel_cell(vulkan)
endif()

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

# MVP cut: one scene across the compiled backends. The matrix expands
# later (see "Add a cell" in TESTING_STRATEGY.md) -- the helper below
# is generic over (scene, backend, denoiser).
#
# The function name is pt_add_golden_cell -- each call registers
# *two* ctest tests: `${_cell_name}_render` (runs the harness) and
# `${_cell_name}_diff` (runs imgdiff vs the committed golden; the
# diff DEPENDS on the render so ctest serialises them per cell).
# Cells are skipped (NOT registered) on hosts where their backend
# isn't available: Metal cells only register on Darwin; Vulkan cells
# only register if PT_ENABLE_VULKAN_BACKEND is ON; Software cells
# only register on APPLE or WIN32 (rhi_software is gated to those
# platforms in src/rhi/CMakeLists.txt -- no SetDIBitsToDevice analog
# implemented for Linux yet). This matches the runtime gate the
# engine already has.
#
# The host (Darwin/Linux/Windows) is also pulled from
# CMAKE_SYSTEM_NAME so the golden path is build-target-derived, not
# host-machine-derived. Cross-builds (rare in this repo) get the
# target's host directory, matching what their runtime would produce.
function(pt_add_golden_cell)
    set(_options)
    # TAG (optional): a disambiguating suffix appended to the cell name and
    # the golden / actual / diff PNG basenames, e.g. TAG agx yields
    # `golden_<scene>__<backend>__<denoiser>__agx`. Lets two cells share the
    # same scene/backend/denoiser triple but differ in EXTRA (e.g. one per
    # tone-mapping operator, Wave 9 #27) without colliding ctest test names
    # or golden filenames. Empty (default) = the historical naming with no
    # suffix, so existing cells + their committed goldens are unaffected.
    set(_oneval SCENE BACKEND DENOISER FRAMES SPP MAX_DELTA MEAN_DELTA FAIL_PERCENT TAG)
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
    # rhi_software is gated to APPLE OR WIN32 in src/rhi/CMakeLists.txt
    # (no SetDIBitsToDevice analog implemented for Linux). Registering
    # a software cell on Linux would fail at runtime when the engine
    # rejects r_backend software; skip it at configure time.
    if(PT_CELL_BACKEND STREQUAL "software" AND NOT (APPLE OR WIN32))
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
    # Use `NOT DEFINED` (not `NOT VAR`) so a caller passing the
    # numeric `0` -- which CMake treats as boolean-false in the
    # `NOT VAR` form -- actually takes effect. Without this, the
    # documented `--max-delta 0` style pixel-exact cell would
    # silently get the default (16) instead of zero tolerance.
    # cmake_parse_arguments leaves the variable UNDEFINED when the
    # caller doesn't pass the corresponding flag, so DEFINED is the
    # right "did the caller provide a value?" signal.
    if(NOT DEFINED PT_CELL_FRAMES)
        set(PT_CELL_FRAMES 64)
    endif()
    if(NOT DEFINED PT_CELL_MAX_DELTA)
        set(PT_CELL_MAX_DELTA 16)
    endif()
    if(NOT DEFINED PT_CELL_MEAN_DELTA)
        set(PT_CELL_MEAN_DELTA 2)
    endif()
    if(NOT DEFINED PT_CELL_FAIL_PERCENT)
        set(PT_CELL_FAIL_PERCENT 1)
    endif()

    # Optional TAG suffix on the cell name + PNG basenames. `_tag_suffix`
    # is `__<tag>` when TAG is set, else empty -- so the unsuffixed naming
    # is preserved bit-for-bit for callers that don't pass TAG.
    set(_tag_suffix "")
    if(NOT "${PT_CELL_TAG}" STREQUAL "")
        set(_tag_suffix "__${PT_CELL_TAG}")
    endif()

    set(_host        ${CMAKE_SYSTEM_NAME})
    set(_stem        ${PT_CELL_SCENE}__${PT_CELL_BACKEND}__${PT_CELL_DENOISER}${_tag_suffix})
    set(_cell_name   golden_${_stem})
    set(_scene_cfg   ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/${PT_CELL_SCENE}.cfg)
    set(_golden_png  ${CMAKE_SOURCE_DIR}/tests/goldens/${_host}/${_stem}.png)
    set(_actual_png  ${CMAKE_BINARY_DIR}/golden_actual/${_stem}.png)
    set(_diff_png    ${CMAKE_BINARY_DIR}/golden_diff/${_stem}.png)
    # Per-cell working directory. The engine reads demont.cfg /
    # autoexec.cfg from CWD before applying the smoke-exec fixture, so
    # using a shared CWD (e.g. CMAKE_BINARY_DIR) would let an archived
    # cvar from a previous cell or session leak into the baseline.
    # Per-cell CWD isolates one cell from another within a ctest run;
    # the wrapper (`pt_render_one_frame/main.cpp`) additionally
    # removes `demont.cfg` and `autoexec.cfg` from its CWD before
    # invoking demont, so archived cvars from a previous run cannot
    # leak across runs either.
    set(_cell_workdir ${CMAKE_BINARY_DIR}/golden_workdir/${_cell_name})

    # Pre-create the actual / diff output directories. imgdiff writes
    # the diff PNG via stb_image_write which doesn't auto-create parent
    # dirs; the smoke-capture in pt_render_one_frame's child does
    # create_directories on its own, but doing both here keeps the
    # behaviour symmetric + lets a contributor run the imgdiff cell
    # in isolation against a stale actual from a previous run.
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/golden_actual)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/golden_diff)
    file(MAKE_DIRECTORY ${_cell_workdir})

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

    # Two-step ctest cell: a `_render` test runs pt_render_one_frame
    # (which fork+exec's demont) to produce the actual PNG under
    # build/golden_actual/, and a separate `_diff` test (DEPENDS on
    # `_render`) runs imgdiff against the host's stored golden. Two
    # tests instead of one chained recipe because (a) ctest's parallel
    # scheduler can serialise the diff after the render via DEPENDS,
    # and (b) on diff failure ctest shows which side failed without
    # the operator having to read a shell pipe.
    #
    # imgdiff thresholds: per-pixel L2 above MAX_DELTA classifies a
    # pixel as "bad"; >FAIL_PERCENT% of bad pixels OR a mean L2 above
    # MEAN_DELTA fails the cell. Default thresholds (16 / 2 / 1) leave
    # room for backend-specific FP nondeterminism (e.g. fma ordering
    # across drivers) without hiding real shader regressions, which
    # typically shift much more than 16 L2 over thousands of pixels.
    #
    # Missing-golden handling: when this host has no committed golden,
    # the `_diff` test fires the `tests/missing_golden.cmake` -P script
    # which exits non-zero with a regen hint. WILL_FAIL TRUE on the
    # test inverts that to PASS, so a missing-golden cell prints the
    # regen hint to the maintainer without breaking ctest as a whole.
    # See `Raytracer Plan/TESTING_STRATEGY.md` for the regen workflow.
    add_test(NAME ${_cell_name}_render
        COMMAND $<TARGET_FILE:pt_render_one_frame> ${_render_args}
        WORKING_DIRECTORY ${_cell_workdir})
    # TIMEOUT 300 (was 120): the *first* software-backend render in the
    # matrix pays a one-time cold-start tax that the rest of the cells
    # do not. On the Windows runner that tax is large and variable --
    # SoftwareVulkanPresent probes `vkCreateInstance`, which fails on the
    # headless GPU-less runner and falls back to GDI present, and the
    # process also eats first-run shader/JIT warm-up. The same cornell_csg
    # software cell that completes in ~77s on a warm runner has been
    # observed to spend the entire 120s budget just reaching "Software
    # backend online" on a loaded one (CI run 26155110144), tripping a
    # spurious ctest Timeout that is unrelated to render correctness.
    # 300s gives the cold first render comfortable headroom while still
    # bounding a genuinely-hung render. Pixel output is unaffected, so all
    # committed goldens stay bit-for-bit.
    set_tests_properties(${_cell_name}_render PROPERTIES
        TIMEOUT 300
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
        # Missing-golden cell: drive a -P CMake script that prints the
        # regen hint then exits non-zero (via FATAL_ERROR). WILL_FAIL
        # inverts the non-zero exit to PASS so the missing-golden state
        # is surfaced to the maintainer without breaking ctest.
        #
        # Configure-time STATUS message duplicates the regen hint so
        # maintainers see it in `cmake --preset ...` output too --
        # under ctest's `--output-on-failure`, a WILL_FAIL-passes test
        # does not print its captured output, so the in-test
        # `message(FATAL_ERROR ...)` text is hidden from CI logs.
        # The configure-time STATUS always appears.
        message(STATUS
            "tests: missing golden on this host -- ${_golden_png}\n"
            "       Regenerate with:\n"
            "         ctest --test-dir <build> -R ^${_cell_name}_render$\n"
            "         cmake -E make_directory \"${CMAKE_SOURCE_DIR}/tests/goldens/${_host}\"\n"
            "         cmake -E copy \"${_actual_png}\" \"${_golden_png}\"")
        add_test(NAME ${_cell_name}_diff
            COMMAND ${CMAKE_COMMAND}
                -DGOLDEN=${_golden_png}
                -DCELL=${_cell_name}
                -DACTUAL=${_actual_png}
                -P ${CMAKE_SOURCE_DIR}/tests/missing_golden.cmake)
        set_tests_properties(${_cell_name}_diff PROPERTIES
            WILL_FAIL TRUE
            DEPENDS   ${_cell_name}_render
            LABELS    "golden;golden_${PT_CELL_BACKEND};golden_missing")
    endif()
endfunction()

# --- MVP cells --------------------------------------------------------------
# 1 scene (cornell_csg) x compiled backends x 1 denoiser (off).
#
# CI mapping today (.github/workflows/build.yml uses `-L golden_software`
# on both the macOS and Windows lanes): the hosted GitHub runners do
# not have GPUs that survive Metal / Vulkan init reliably -- the macOS
# runner ships a paravirtualised Apple-GPU, and the Windows lane runs
# on a generic VM without an NVIDIA / AMD ICD -- so CI exercises the
# software backend on both. Metal cells still register on Mac (so a
# local `ctest` against a real M-series picks them up); Vulkan cells
# register on Windows/Linux builds where PT_ENABLE_VULKAN_BACKEND is
# enabled. Each host's CI step filters via the
# `golden_software` label; other ctest LABELS (`golden`,
# `golden_metal`, `golden_vulkan`, `golden_missing`) are for local /
# future use.
#
# pt_add_golden_cell expands `$<TARGET_FILE:pt_render_one_frame>` and
# `$<TARGET_FILE:imgdiff>` at generate time; both targets only exist
# when PT_BUILD_TOOLS=ON. Skip the matrix registration entirely when
# they're absent so a `PT_BUILD_TESTS=ON, PT_BUILD_TOOLS=OFF` configure
# succeeds. The doctest unit tests above don't depend on the tools.
if(TARGET pt_render_one_frame AND TARGET imgdiff)
    pt_add_golden_cell(SCENE cornell_csg BACKEND software DENOISER off)
    pt_add_golden_cell(SCENE cornell_csg BACKEND metal    DENOISER off)
    pt_add_golden_cell(SCENE cornell_csg BACKEND vulkan   DENOISER off)

    # --- SDF Phase 1 (#97) -------------------------------------------------
    # Acceptance cell: row of smin-blended SDF sphere pairs next to
    # analytic spheres in the same frame. Issue #97 acceptance bar is
    # "Blend region shows continuous normal (no facet artifact)" -- the
    # imgdiff tolerance below is set permissively because the SDF sphere-
    # trace's per-pixel iteration count is more sensitive to FP order
    # across backends than the analytic-sphere path. Tightening once
    # cross-backend variance is measured is fine.
    pt_add_golden_cell(SCENE sdf_smin_row BACKEND software DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE sdf_smin_row BACKEND metal    DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE sdf_smin_row BACKEND vulkan   DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end SDF Phase 1 ---------------------------------------------------

    # --- Previously-orphaned baselines -------------------------------------
    # These four Darwin goldens shipped for months with NO registered cell:
    # the PNGs sat in tests/goldens/Darwin/ and were dutifully refreshed by
    # regen commits as if they gated something, while ctest never rendered
    # the scenes at all. A regression in MAT_WATER medium tracking (#206),
    # the GPU light-tree traversal at 200 lights (#129), the SDF fractal
    # distance estimators (#99), or the aurora composite (#116) would have
    # shipped green. Registered here, with tolerances matching sdf_smin_row
    # (all four render bit-identically run-to-run on the M4 Max now that
    # smoke-mode dt is pinned + the sims are frozen during loading frames).
    pt_add_golden_cell(SCENE water_pool         BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE light_tree_200_pts BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE sdf_fractals       BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # aurora_smoke MUST run with a denoiser: AuroraComposite is dispatched
    # only inside the use_engine_tonemap branch, which is gated on
    # denoiser_active_ (Engine.cpp). The old aurora_smoke__metal__off.png
    # baseline therefore contained NO AURORA AT ALL (measured: 0.00%
    # green-dominant pixels in the upper sky, plain night blue) -- it was
    # captured through the wrapper's `off` override, which is written after
    # the fixture body and so beat the fixture's own `r_denoiser
    # svgf_atrous` line. That file is deleted; this cell captures the
    # feature the fixture documents (measured: 5.08% green-dominant).
    # ------------------------------------------------------------------
    # #280 HOLD -- THIS CELL IS EXPECTED TO FAIL, AND WAS NOT REGENERATED.
    #
    # #280 put the sky in physical radiometric units (the sun is 1360.8
    # W/m^2, Kopp & Lean 2011, not the literal 80) and every day and
    # twilight cell was re-pinned against it. This one was not, because
    # NIGHT is the part of the chain the rebase did not reach:
    #
    #   * procSky's night floor (0.005, 0.010, 0.030) is still a painted
    #     stand-in carried across by one uniform factor. The real number is
    #     computable and cited -- 250 R of O I 557.7 nm nightglow is
    #     7.09e-8 W/m^2/sr, five orders of magnitude below where the floor
    #     now sits -- but adopting it blacks out every night frame.
    #   * moonDisc, starsOnly and the 0.3 x phase moonlight term are all
    #     painted, with no physical anchor. Full-moon illuminance is
    #     4.8e-6 of solar (Kieffer & Stone 2005), which is the anchor they
    #     need and do not have.
    #   * sunDisc in modes 2/3 is painted at 250 x atm; only
    #     sunDiscPhysical is derived as E / Omega.
    #
    # Re-pinning now would encode a half-migrated state that has to be
    # re-pinned again the moment the night radiometry lands, so the cell is
    # left red on purpose. Regenerate it with that phase, not before.
    # See PR #301 and its remainder list.
    # ------------------------------------------------------------------
    pt_add_golden_cell(SCENE aurora_smoke       BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # ReSTIR DI final-shading coverage (issue #78). light_primitives_mixed
    # is the ONLY fixture that combines analytic lights (light_count > 0:
    # point + spot + sphere + quad) with a denoiser -- and that exact
    # combination is what dispatches RestirFinal (gated on albedo_tex_id_,
    # i.e. a denoiser being active, AND light_count > 0, with r_restir on by
    # default). Every other svgf_atrous cell above is a sky / procedural /
    # smoke scene with no analytic lights, so RestirFinal early-outs on
    # light_count == 0 and the ReSTIR final-shading path never ran under any
    # test. That gap hid a Metal RHI bug where RestirFinal's acceleration
    # structure (declared AFTER its SSBOs -> Metal buffer(3), not buffer(0))
    # was bound at the hardcoded buffer(0), clobbering the reservoir SSBO
    # and pushing the push-constant block off buffer(4) so the kernel read a
    # zero width/height and contributed NOTHING -- ReSTIR direct lighting
    # was a silent no-op on Metal whenever the denoiser was on. This cell
    # renders the analytic lights THROUGH the ReSTIR reservoir + shadow-test
    # path, so it pins the corrected output (per-pipeline buffer indices
    # parsed from the emitted MSL; see MetalDevice.cpp) and fails loudly if
    # the AS/push placement regresses. DENOISER svgf_atrous is mandatory:
    # the `off` override would disable the denoiser and skip RestirFinal
    # entirely (same reason aurora_smoke must run with a denoiser above).
    #
    # Tolerances are TIGHTER than the 32/4/2 used elsewhere because the
    # scene is mostly black (sun below horizon; the four analytic lights are
    # the only illumination). Losing ReSTIR entirely -- the exact regression
    # this cell guards -- only moves the small lit fraction of pixels, so a
    # loose FAIL_PERCENT/MEAN_DELTA would let the bug pass. Measured on the
    # M4 Max: the bug-vs-fixed diff is mean_delta 2.67 with 6.4% of pixels
    # past MAX_DELTA 16; the fixed render is bit-identical run-to-run (no
    # meshes -> no TLAS build entropy). 16/2/2 fails the bug on BOTH signals
    # (bad% 6.4 > 2 and mean 2.67 > 2) with headroom while passing the
    # deterministic golden. Widen per-cell if it ever flakes across Apple
    # GPU generations (this cell is a local M-series guard; CI runs the
    # software lane).
    pt_add_golden_cell(SCENE light_primitives_mixed BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 2)
    # sdf_phase2_procedural is NOT registered unconditionally: the fixture
    # exercises displaced / procedural SDF ops that are compiled out unless
    # PT_SDF_PROCEDURAL_OPS=ON (default OFF), and in a default build it
    # renders as empty terrain -- a cell here would pin the FEATURE-DISABLED
    # image and pass forever. Gate it on the option instead, so a build that
    # enables the feature also gates it. (Vulkan is excluded on purpose:
    # PT_SDF_PROCEDURAL_OPS=ON + PT_SDF_AUTODIFF=ON hangs the SPIR-V
    # pipeline build on both MoltenVK and NVIDIA -- the autodiff IR
    # complexity, not a driver bug.)
    if(PT_SDF_PROCEDURAL_OPS)
        pt_add_golden_cell(SCENE sdf_phase2_procedural BACKEND metal DENOISER off
            MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    endif()
    # --- end previously-orphaned baselines ---------------------------------

    # --- Sky / sun / stars / moon matrix (#112) ----------------------------
    # Follow-up to the #45 MVP: the cornell_csg cell exercises CSG +
    # analytic BVH but nothing in the procedural sky / astronomy /
    # stars / moon / clouds stack. Issue #112 lists the regressions
    # the current matrix would miss (PR #114's stars-stateless-
    # composite fix is the canonical example -- stars were invisible
    # under any denoiser and no cornell_csg cell could have caught it).
    #
    # Scene coverage:
    #   procedural_noon       sun @ +60 deg     daytime sky + bright sun disc
    #   procedural_evening    sun @ +5  deg     sunset orange, long Mie path
    #   procedural_dawn       sun @ -2  deg     civil twilight, no sun disc
    #   bsc_night             sun @ -45 deg     BSC stars, star_split path
    #   lunar_night           astronomical      moon disc, NEE from moon
    #   bsc_night_clouds      astronomical      stars + cloud occlusion
    #
    # Denoiser coverage per scene:
    #   software/off          -- CI-exercised cell (label golden_software)
    #   metal/svgf_atrous     -- the user's most common dev-box combo;
    #                            stresses the denoiser composite path
    #                            (the star_split bypass-the-a-trous
    #                            kernel that PR #114 fixed lives here).
    #
    # Tolerances: these scenes are noisier than cornell_csg because (a)
    # procedural sky has more bright HDR pixels which exaggerate FP
    # roundoff across backends; (b) star_split + clouds + ACES tonemap
    # each amplify tiny radiance deltas. Default tolerances (16 / 2 / 1)
    # held in early bring-up; if any cell flakes on cross-host runs the
    # MAX_DELTA / MEAN_DELTA can be widened per-cell here without
    # touching the harness.
    #
    # #280 SPLIT THIS LOOP. One MEAN_DELTA for six scenes was hiding a
    # vacuous cell: measured against a real ablation, deleting OZONE moves
    # procedural_noon by mean 2.06, and the loop's MEAN_DELTA of 4 is
    # larger than that -- so that cell could not fail when the feature it
    # is meant to cover was removed. That is the failure mode ocean_fft sat
    # in for months. Every day cell below now carries a threshold measured
    # against its own smallest real signal, recorded per cell.
    #
    # The METHOD, once, since it is the same for every cell #280 re-pinned:
    #   * reproducibility on this host is EXACTLY ZERO -- two full matrix
    #     renders of this branch are bit-identical on 86 of 87 cells
    #     (smoke_phase3_wind is the one documented exception), so the
    #     threshold is pure cross-host headroom and not run-to-run noise;
    #   * the SIGNAL is the smaller of two ablations actually performed --
    #     ozone off (r_ozone 0) and multiple scattering off (the shader's
    #     readiness gate plus the host builder, so it is off everywhere);
    #   * MEAN_DELTA is set at roughly a quarter of that signal, which
    #     leaves it ~8x above the largest cross-host codegen drift this
    #     project has measured (#279: mean <= 0.06 across 30 cells).
    # ------------------------------------------------------------------
    # #280 HOLD -- bsc_night_clouds__metal IS EXPECTED TO FAIL, AND WAS NOT REGENERATED.
    #
    # #280 put the sky in physical radiometric units (the sun is 1360.8
    # W/m^2, Kopp & Lean 2011, not the literal 80) and every day and
    # twilight cell was re-pinned against it. This one was not, because
    # NIGHT is the part of the chain the rebase did not reach:
    #
    #   * procSky's night floor (0.005, 0.010, 0.030) is still a painted
    #     stand-in carried across by one uniform factor. The real number is
    #     computable and cited -- 250 R of O I 557.7 nm nightglow is
    #     7.09e-8 W/m^2/sr, five orders of magnitude below where the floor
    #     now sits -- but adopting it blacks out every night frame.
    #   * moonDisc, starsOnly and the 0.3 x phase moonlight term are all
    #     painted, with no physical anchor. Full-moon illuminance is
    #     4.8e-6 of solar (Kieffer & Stone 2005), which is the anchor they
    #     need and do not have.
    #   * sunDisc in modes 2/3 is painted at 250 x atm; only
    #     sunDiscPhysical is derived as E / Omega.
    #
    # Re-pinning now would encode a half-migrated state that has to be
    # re-pinned again the moment the night radiometry lands, so the cell is
    # left red on purpose. Regenerate it with that phase, not before.
    # See PR #301 and its remainder list.
    # ------------------------------------------------------------------
    set(_sky_scenes_night
        bsc_night
        lunar_night
        bsc_night_clouds)
    foreach(_scene IN LISTS _sky_scenes_night)
        # NIGHT. Deliberately NOT re-pinned by #280 and deliberately not
        # re-measured: the airglow floor procSky adds is still a painted
        # stand-in (the real figure is 7.09e-8 W/m^2/sr of O I 557.7 nm
        # nightglow, five orders of magnitude below where the floor sits),
        # and the moon and star paths are painted with no physical anchor
        # at all. Thresholds measured against those would be measured
        # against a half-migrated state. See the hold note on
        # bsc_night_clouds below.
        pt_add_golden_cell(SCENE ${_scene} BACKEND software DENOISER off)
        pt_add_golden_cell(SCENE ${_scene} BACKEND metal    DENOISER svgf_atrous
            MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    endforeach()

    # DAY. Software cells keep the defaults (16 / 2 / 1) and keep their
    # ROLE, which is not what it looks like: the software tracer paints its
    # own sky and implements no atmospheric physics, so a software cell
    # cannot detect a sky regression at all (#257 recorded this). It is the
    # control that says a change did not leak into host code -- and under
    # #280's unit rebase alone every software cell moved by at most one
    # 8-bit level, which is what that control is for.
    foreach(_scene IN ITEMS procedural_noon procedural_evening procedural_dawn)
        pt_add_golden_cell(SCENE ${_scene} BACKEND software DENOISER off)
    endforeach()
    # Metal + SVGF a-trous, thresholds measured per cell (#280).
    #   procedural_noon    smallest signal 2.06 (ozone off) -> 4.1x clear
    #   procedural_evening smallest signal 15.70 (ozone off) -> 5.2x clear
    #   procedural_dawn    smallest signal 33.75 (ozone off) -> 8.4x clear
    # noon is the tight one because its sun sits at 60 degrees, where the
    # slant path through the ozone layer is at its shortest -- the same
    # geometry that makes ozone invisible at noon and decisive at twilight.
    pt_add_golden_cell(SCENE procedural_noon    BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 0.5 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE procedural_evening BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 3 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE procedural_dawn    BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end #112 sky/sun/stars/moon matrix --------------------------------

    # --- Planetarium (#281) ------------------------------------------------
    # Jupiter (V = -2.67) in Gemini, a few degrees from Castor and Pollux,
    # at a pinned date/observer. The cell pins the whole host-to-shader
    # hand-off: planetPosition -> equatorialToHorizon -> world direction ->
    # the two float4[7] push arrays -> planetSplat()'s Gaussian, AND it pins
    # the planet's brightness RELATIVE to two catalogue stars in the same
    # frame, which is the acceptance criterion issue #281 actually states.
    #
    # METAL ONLY, ON PURPOSE. The planets live exclusively in
    # StarsComposite.slang; SoftwareTracer.cpp does not run it (it is
    # hand-written C++, explicitly "not a Slang-to-CPU emit" -- see #263), so
    # a `software off` cell here would pin a planet-free image and pass
    # forever. That is precisely the trap the three unregistered sunset_*
    # fixtures fell into before #257. Vulkan is not built on macOS, so the
    # SPIR-V half of the push-constant change is unexercised by CI on this
    # host and is flagged as such in the PR.
    #
    # TOLERANCES ARE MEASURED, NOT COPIED FROM THE NEIGHBOURS. The sibling
    # night cells run 32 / 4 / 2, and at 2% of 196608 pixels this cell would
    # be VACUOUS: a planet is a sub-pixel point source, so deleting the
    # entire feature moves only a handful of pixels and a 2% threshold would
    # sail past it. A golden that cannot fail is worse than no golden.
    #
    # Measured on the dev M4 Max with `pt_render_one_frame` + `imgdiff`,
    # in BOTH build configurations (the first cut measured Release only,
    # and the cell then failed in Debug -- see the paragraph below):
    #
    #                        bad   bad%      mean_delta   max_delta
    #   repeat, Debug          0   0.000000  0.000000     0.0000
    #   repeat, Release        0   0.000000  0.000000     0.0000
    #   vs golden, Release     0   0.000000  0.000000     0.0000
    #   vs golden, Debug       0   0.000000  0.005341     1.0000
    #   `r_planets 0`, Debug  11   0.005595  0.010300   395.7714
    #   `r_planets 0`, Rel.   11   0.005595  0.010300   395.7714
    #
    # The scene is pure sky -- no geometry, so no path-traced stochastic
    # term -- which is why the repeat floor is exactly zero in both configs.
    #
    # BAD-PIXEL COUNT IS THIS CELL'S SIGNAL; THE MEAN IS A BACKSTOP. A
    # planet is a sub-pixel point source, so deleting it moves 11 pixels a
    # long way (max_delta 395) and the global mean barely at all. The
    # Debug/Release codegen difference is the exact opposite shape: one LSB
    # on many pixels, which moves the mean to 0.005341 while classifying
    # ZERO pixels as bad. FAIL_PERCENT therefore separates the two cases
    # perfectly -- 0.005595% when the planets vanish against 0.000000% for
    # a build-config change -- and 0.0025 (~4.9 px) sits 2.24x below the
    # signal with an infinite margin above the floor.
    #
    # MEAN_DELTA 0.0074 is the geometric midpoint of the two bounds it must
    # separate: 1.39x above the Debug/Release delta (0.005341) and 1.39x
    # below the planets-off signal (0.010300). Symmetric by construction
    # rather than picked. The planets-off signal is IDENTICAL in both
    # configs, so only the baseline moves with build type, not the thing
    # being detected.
    #
    # It was 0.004, measured in Release alone, which made the cell fail in
    # Debug at 1.34x overshoot on a max_delta of 1.0 with zero bad pixels.
    # Widened rather than made Release-only: a cell that skips half the
    # builds is most of the way to a cell that tests nothing, which is the
    # failure this file spends several comments warning about.
    #
    # MAX_DELTA stays at the house 32: it only classifies which pixels
    # count as bad, and a bright point source clears it by 12x.
    #
    # Sanity check when touching this: `ctest -N | grep planetarium` must
    # list golden_planetarium_jupiter__metal__svgf_atrous_render and _diff.
    # A fixture with no registered cell tests nothing -- three sunset_*
    # fixtures sat in this repo in exactly that state until #257.
    # ------------------------------------------------------------------
    # #280 HOLD -- THIS CELL IS EXPECTED TO FAIL, AND WAS NOT REGENERATED.
    #
    # #280 put the sky in physical radiometric units (the sun is 1360.8
    # W/m^2, Kopp & Lean 2011, not the literal 80) and every day and
    # twilight cell was re-pinned against it. This one was not, because
    # NIGHT is the part of the chain the rebase did not reach:
    #
    #   * procSky's night floor (0.005, 0.010, 0.030) is still a painted
    #     stand-in carried across by one uniform factor. The real number is
    #     computable and cited -- 250 R of O I 557.7 nm nightglow is
    #     7.09e-8 W/m^2/sr, five orders of magnitude below where the floor
    #     now sits -- but adopting it blacks out every night frame.
    #   * moonDisc, starsOnly and the 0.3 x phase moonlight term are all
    #     painted, with no physical anchor. Full-moon illuminance is
    #     4.8e-6 of solar (Kieffer & Stone 2005), which is the anchor they
    #     need and do not have.
    #   * sunDisc in modes 2/3 is painted at 250 x atm; only
    #     sunDiscPhysical is derived as E / Omega.
    #
    # Re-pinning now would encode a half-migrated state that has to be
    # re-pinned again the moment the night radiometry lands, so the cell is
    # left red on purpose. Regenerate it with that phase, not before.
    # See PR #301 and its remainder list.
    # ------------------------------------------------------------------
    pt_add_golden_cell(SCENE planetarium_jupiter BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 0.0074 FAIL_PERCENT 0.0025)
    # --- end Planetarium ---------------------------------------------------

    # --- Curved-Earth sunset fixtures (#51) --------------------------------
    # These three .cfg files have existed since the #51 spherical-Earth
    # work landed and were NEVER registered: the fixtures sat in
    # tests/goldens/scenes/ with no cell and no PNG, so #51's own
    # acceptance criteria -- "sunset / sunrise renders show realistic
    # horizon falloff at altitude" and the spherical-vs-planar A/B --
    # have never been tested by anything. Registered here, and pinned on
    # unmodified main BEFORE the P3 (#257) atmosphere work starts, so
    # that phase's change to them is a measurable delta rather than an
    # unverifiable assertion.
    #
    #   sunset_horizon          ground camera (y=1.7) at a 2 deg sun --
    #                           the longest sea-level slant path the
    #                           model supports, where curved vs. flat
    #                           airmass differs most.
    #   sunset_altitude         the same sun from 10 km (airliner
    #                           altitude), where the visible horizon dips
    #                           ~3.2 deg below the mathematical one.
    #   sunset_altitude_planar  identical to sunset_altitude except
    #                           `r_planet_radius 0`, which routes
    #                           atmosphericTransmittance through the
    #                           legacy planar-exponential integral and
    #                           disables procSky's horizon dip. This is
    #                           the A/B half: the pair only means
    #                           something if BOTH sides are pinned.
    #
    # Backends: software (the CI-exercised deterministic lane, label
    # golden_software) and metal. Both are meaningful here for a reason
    # specific to this phase -- SoftwareTracer.cpp is a hand-written CPU
    # tracer, not a Slang-to-CPU emit, and carries its own procSky mirror
    # with NO atmospheric transmittance at all. So a shader-only change
    # must move the metal cells and leave the software cells alone; a
    # software cell that moves is evidence the diff leaked into host code.
    # Pinning both is what makes that inference available.
    #
    # Measured at pin time, and worth recording because it bounds what
    # the A/B pair can prove: on METAL the spherical and planar halves
    # differ by mean 6.41 / max 21.2 L2 across 48.6% of the frame, so the
    # pair does demonstrate the spherical model doing something. On
    # SOFTWARE the two halves are BIT-IDENTICAL (max_delta 0) -- the CPU
    # tracer never consults r_planet_radius, so its half of the A/B
    # carries no spherical-vs-planar signal at all. The software cells
    # are pinned anyway, as the control: they are the cells that must NOT
    # move when a shader changes.
    #
    # Tolerances match the #112 sky cells for the same reason: bright HDR
    # sky pixels exaggerate cross-backend FP roundoff. Measured on the dev
    # M4 Max at pin time, run-to-run over three renders: max_delta 0 on
    # every one of the six cells (no meshes -> no TLAS build entropy, no
    # sim -> no dt integration), so the headroom below is entirely for
    # cross-host / cross-driver variance and none of it is masking noise.
    set(_sunset_scenes sunset_horizon sunset_altitude sunset_altitude_planar)
    foreach(_scene IN LISTS _sunset_scenes)
        pt_add_golden_cell(SCENE ${_scene} BACKEND software DENOISER off
            MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
        pt_add_golden_cell(SCENE ${_scene} BACKEND metal    DENOISER off
            MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    endforeach()
    # --- end curved-Earth sunset fixtures ----------------------------------

    # --- Planetary P3 (#257): the spherical cloud shell --------------------
    # The phase's headline acceptance criterion, and #51's founding one:
    # "distant clouds visually merge with the horizon at ground-level
    # camera positions". A ground camera looks dead level at a low
    # stratocumulus deck with the analytic planet body on.
    #
    #   planet_horizon_clouds         spherical frame + planet body. The
    #                                 cloud layer is a SHELL, the ground
    #                                 curves away beneath it, and the deck
    #                                 runs continuously down to the body's
    #                                 limb -- clouds touch the horizon.
    #   planet_horizon_clouds_planar  the same scene with the frame planar
    #                                 and the body off, so the layer is a
    #                                 pair of infinite planes a horizontal
    #                                 ray can never reach. This is the half
    #                                 that shows the defect.
    #
    # Pinning both is the point. A single golden of the fixed behaviour
    # says the renderer produces some image; the pair says the spherical
    # path produces a DIFFERENT one, and shows which difference.
    #
    # Metal only. The software backend's tracer is hand-written C++, not a
    # Slang-to-CPU emit: it has no cloud march at all and does not read
    # planet_ground, so a software cell here would pin an empty sky and
    # pass forever. That is the same reason clouds_raymarched has no
    # software cell.
    #
    # Tolerances match the other cloud cells (32/4/3): the march is a
    # stochastic estimator and its 2-spp noise floor over 64 accumulated
    # frames is wider than the analytic-geometry baseline.
    # --- Planetary P4 (#258): streamed terrain ---------------------------
    #
    # Three cells, all metal-only and all denoiser-off.
    #
    # WHY NO SOFTWARE CELL. The terrain rides the hardware ray query, and
    # while the software backend's Embree TLAS would carry the geometry,
    # SoftwareTracer.cpp is a hand-written CPU renderer rather than a
    # Slang-to-CPU emit -- it never reads the per-instance descriptor SSBO,
    # so every terrain hit would shade as the legacy gold mesh material. A
    # software golden would pin that placeholder, not the phase. Registering
    # the cell is cheap once the tracer reads descriptors; until then the
    # absence is the honest state. (#45's known-limited matrix cell.)
    #
    # WHY NO ORBIT CELL. "Fly to orbit and see a coherent globe" is P4
    # acceptance and the GEOMETRY half works -- at 3.0e7 m the streamer
    # serves the six roots, the anchor engages, the limb is clean. The
    # SHADING half does not, and not because of this phase: the P3 analytic
    # planet body renders as a black disc from outside the atmosphere on
    # UNMODIFIED main too (cam_pos 0 30000000 0, r_planet_ground 1). The
    # sky-view path is a function of the camera and does not survive
    # leaving the shell, which is what the roadmap says and what P6 owns.
    # planet_aerial.cfg goes as high as the atmosphere is currently valid.
    #
    # DETERMINISM, VERIFIED RATHER THAN ASSUMED. Each fixture sets
    # r_planet_lod_freeze 1 and pt_planet_settle_rounds, which drive the
    # selector to its residency fixed point BEFORE the capture starts. Five
    # separate bugs stood between "the freeze cvar is set" and "two runs
    # render the same pixels", and every one was found by rendering twice
    # and comparing rather than by reading the code. P4 (#258) found the
    # first four:
    #
    #   * settle frames accumulated into the same buffer while residency
    #     was still growing, and the settle FRAME COUNT is wall-clock
    #     dependent -- so the capture averaged a run-specific number of
    #     run-specific intermediate geometries. 22% bad pixels between two
    #     identical runs. Settle frames now discard the accumulator.
    #   * the settle advanced frame_index_, which seeds the sample
    #     sequence, so two runs with identical geometry still started their
    #     capture on different Monte Carlo samples. 8.4% bad pixels.
    #   * the LOD hysteresis consults the previous frame's leaf set, so the
    #     converged set depended on the order chunks arrived in, which is
    #     the order a worker pool finished them in. Hysteresis is suppressed
    #     under freeze; interactive sessions keep it.
    #   * PlanetTerrain::Update selected BEFORE draining finished bakes, so
    #     the desired set was always one drain stale and convergence could
    #     be declared on a set chosen without the metrics that had just
    #     landed. Two runs settled on genuinely different 1128- and
    #     1074-chunk sets.
    #
    # #284 found the fifth, which was FOUR faults with one shape: the
    # subsystem DETECTED convergence instead of GUARANTEEING it. Each was
    # a place where an answer the capture depends on was allowed to be a
    # function of how fast the machine ran, and then checked afterwards
    # rather than made impossible.
    #
    #   * the settle was a FRAME BUDGET raced against an asynchronous bake
    #     pool. Rendering a frame does not make a chunk bake, so "wait up to
    #     N frames" is a race between the frame rate and the bake rate. A
    #     Debug build lost it on every single run -- 400 frames spent, 149 /
    #     220 / 155 chunks resident on three consecutive captures, three
    #     different images -- and a loaded Release machine lost it
    #     occasionally, which is exactly the profile of a golden that passes
    #     by luck. It is now a BARRIER: PlanetTerrain::Settle runs one bake
    #     round per iteration, each round waiting for every requested chunk
    #     to land, so the round count is a property of the quadtree and not
    #     of the machine. These three cells converge in 14, 12 and 16
    #     rounds, against a bound of 256.
    #   * arena slots came off a free list in bake-arrival order, so the
    #     same chunk got a different slot in every run. The desired digest
    #     matched while the descriptor buffer did not, and the resulting
    #     difference in BLAS creation order flipped the occasional
    #     exactly-tied hit on a shared chunk edge. Draining a round
    #     completely rather than 64 at a time makes the slot map fall out of
    #     the key order.
    #   * TerrainQuadtree::Balance inserted the children of a 2:1 split as
    #     LEAVES without re-applying the split rule to them. A child whose
    #     metric had arrived earlier is measured, so it never appeared in
    #     Wanted() -- and Wanted() being empty is the whole convergence
    #     signal. The selector could therefore stop on a set that still
    #     wanted to split, and which one depended on how ragged the tree was
    #     when Balance ran. Measured: one camera, one binary, metrics
    #     agreeing on all 1214 common keys to the last bit, settling to 1128
    #     chunks under one bake schedule and 912 under another. Balance now
    #     descends into the children, so the converged set is closed under
    #     the split rule -- which is what Converged() has always been read
    #     as asserting. Asserted directly in pt_planet_terrain.
    #   * whether the BARRIER FRAME ITSELF contributed a sample was a race.
    #     UpdatePlanetTerrain runs before RenderFrame's loading-frame gate,
    #     so with the async pipeline build or a CSG bake still in flight
    #     that frame returns before the path-trace dispatch and contributes
    #     nothing -- and otherwise it dispatches one sample at the held
    #     frame_index_, which is then in the accumulator when the capture
    #     starts. Both take a second or two, so which wins is wall clock.
    #     This one survived the first 35 renders and appeared on the 36th,
    #     as a mean delta of 1.21 spread evenly over every lit pixel and
    #     zero in the sky -- a sample-count difference wearing the costume
    #     of noise. Reproduced on demand by holding the barrier until the
    #     pipelines were ready. The accumulator is now discarded again on
    #     the first frame that COUNTS, so the capture is exactly the counted
    #     frames whichever way the race goes.
    #
    # Two digests are logged at convergence, and the pair is the point. The
    # DESIRED digest says two runs picked the same chunks; the SCENE digest
    # (slot map, stitch masks, instance transforms, descriptor floats) says
    # they also laid them out the same way. #284 was invisible to the first
    # alone: three Release captures agreed on 3f4f417f3e74165d exactly and
    # still produced three different PNGs.
    #
    #                    rounds  chunks  desired            scene
    #   planet_surface       14     645  3958231c97fa53bd   6416c3a04da11c28
    #   planet_aerial        12    1419  d661b1fe2bb862ef   6fda81e4463ec7d6
    #   planet_lod_seam      16     468  2efaddfa21781929   924882c81d83b01b
    #
    # THESE ARE THE POST-#304 VALUES, and they replace a block that recorded
    # P4's (3f4f417f3e74165d / a3b3352959e2bf78 and the two beside it).
    # #304 broke the terrain power law at the hillslope scale, which lowers
    # small-scale relief, which drops e_L, so the selector stops descending
    # as deep: 1128 -> 645, 1806 -> 1419, 1008 -> 468 chunks. The PNGs were
    # regenerated at the time and this measurement was not, so the block
    # documented a chunk set that no longer existed -- #315, which also asks
    # the better question of whether a recorded digest nothing verifies
    # ought to be an assertion instead. Re-measured here on 6f88475 and
    # again on the #319 branch; all four columns agree on both, which is how
    # #319 shows that retaining the whole cut leaves steady-state
    # PUBLICATION alone.
    #
    # Residency stays clear of r_planet_chunk_budget 2048, and note that
    # since #319 that cvar is the LEAF budget: the arena is WholeCutSlots of
    # it (2728 slots at 2048) because residency is the whole cut and not its
    # frontier. The counts above are leaf counts and are unchanged.
    # EnforceBudget was instrumented and observed never to run on any of the
    # three, so the tau bisection plays no part in these cells. (P4
    # suspected it did; it does not. See r_planet_chunk_budget.)
    #
    # FRAMES 32 rather than the default 64: with the geometry frozen and the
    # sample sequence pinned, the accumulator is a deterministic function of
    # the frame count, so 32 is the ordinary path-tracer floor rather than a
    # noise compromise. It also keeps three ~1500-chunk captures comfortably
    # inside the cell timeout.
    #
    # ---- TOLERANCES ARE MEASURED, NOT COPIED FROM A NEIGHBOUR ----------
    #
    # A cell whose thresholds sit above its own signal reports PASS while
    # comparing nothing, which is #252. EVERY NUMBER BELOW WAS RE-MEASURED
    # BY #307 against the PNGs #307 regenerated, on this host, with each
    # render in an isolated working directory so archived cvars cannot leak
    # between runs. The #304-era figures described the #304-era PNGs and
    # are NOT carried forward: regenerating a golden invalidates every
    # threshold justification attached to it, and an inherited ablation is
    # not a measurement.
    #
    #   RUN-TO-RUN NOISE: ZERO, RE-ESTABLISHED RATHER THAN ASSUMED. A
    #   worktree at main (6f88475), configured and built from scratch,
    #   renders planet_surface / planet_aerial / planet_lod_seam /
    #   planet_land_orbit BYTE-IDENTICALLY twice over -- equal MD5s, and
    #   imgdiff at ZERO thresholds gives bad=0, max 0.0000, mean 0.000000.
    #   This branch does the same across all five planet cells. So there is
    #   no noise term to subtract from anything below, and every figure is
    #   100% signal.
    #
    #   #312 (water-leaving radiance) MOVED NONE OF THESE FOUR, which is
    #   worth stating because it was the obvious risk to a rebase: main's
    #   fresh renders are bit-identical to the PNGs #304 committed. All
    #   four fixtures pin r_planet_ocean 0, so the transport #312 rewrote
    #   is not on any of these frames. (#312 DID leave three water cells
    #   failing on main -- ocean_foam, planet_ocean_snell and
    #   water_pool_submerged -- which is a main-side matter, not this
    #   feature's; they fail with byte-identical deltas from a clean main
    #   build and from this branch.)
    #
    #   BLAST RADIUS, VERIFIED THE OTHER WAY. Across the whole 88-cell
    #   Metal + software matrix, the ONLY actual renders that differ
    #   between clean main and this branch are the three cells regenerated
    #   here -- plus smoke_phase3_wind, which differs from ITSELF by the
    #   same amount (max 3.46 / mean 0.018 / 1.5% of pixels) on two runs of
    #   ONE build at ONE commit, so its solver moves it and this change does
    #   not. Comparing failure LISTS would not have shown that; comparing
    #   every rendered PNG does.
    #
    #   DEBUG vs RELEASE, same host (the only same-host difference left).
    #   RE-MEASURED against the regenerated PNGs, which are
    #   Release-generated:
    #     planet_surface         max 0.0000  mean 0.000000  (pixel-identical)
    #     planet_aerial          max 0.0000  mean 0.000000  (pixel-identical)
    #     planet_lod_seam        max 3.1623  mean 0.000148  (22 px of 196 608)
    #     planet_land_orbit      max 0.0000  mean 0.000000  (pixel-identical)
    #     planet_hillslope_rock  max 0.0000  mean 0.000000  (pixel-identical)
    #   3.1623 is sqrt(10): a pixel differing by one LSB on one channel and
    #   three on another. At MAX_DELTA 16 not one of those 22 pixels is
    #   classified bad at all, so the cross-configuration bad-pixel rate at
    #   the cells' own thresholds is 0.000%.
    #
    #   SIGNAL, feature deleted (r_planet_terrain 0 -- confirmed to take
    #   effect by the ABSENCE of "terrain online" in the run log, because
    #   `--extra` silently losing to the fixture is a trap this file has
    #   already been caught by once):
    #     planet_surface         mean 118.22   bad 91.41%
    #     planet_aerial          mean 130.18   bad 99.29%
    #     planet_lod_seam        mean 179.24   bad 86.26%
    #     planet_hillslope_rock  mean  66.89   bad 87.03%
    #
    #   SIGNAL, real Earth deleted (r_planet_dem pointed at a path that does
    #   not exist, so the fractal fallback stands in). This is the tighter
    #   and more specific claim, because these goldens are pinned against
    #   ETOPO 2022 and not merely against "some terrain":
    #     planet_surface         mean 149.72   bad 99.70%
    #     planet_aerial          mean 146.87   bad 98.80%
    #     planet_lod_seam        mean 142.83   bad 99.89%
    #     planet_hillslope_rock  mean  38.67   bad 55.03%
    #   CONFIRMED BY the ABSENCE of "planet: loaded DEM" while "planet:
    #   terrain online" is still PRESENT -- both greps run on every one of
    #   the eight logs. The wording here used to ask for "rendering a
    #   PROCEDURAL body" in the log; NO SUCH LINE IS EMITTED, so that recipe
    #   could not be run as written -- the same vacuous-confirmation trap
    #   the paragraph above warns about, one paragraph later. Grepping the
    #   log for "procedural" alone does not work either: it matches two SDF
    #   startup notices and the moon-map line on every run, ablated or not.
    #
    #   SIGNAL, the hillslope scale break deleted (r_planet_hillslope_break
    #   0, which restores the pre-#304 unbroken power law bit-exactly). This
    #   is the most specific ablation these cells have AND the weakest of
    #   the three, so it is the one the thresholds are sized against:
    #     planet_surface         mean  46.25   bad 45.01%
    #     planet_aerial          mean  15.61   bad 27.26%
    #     planet_lod_seam        mean  20.95   bad 39.23%
    #     planet_hillslope_rock  mean  17.13   bad 27.83%
    #   This ablation has NO log line to confirm it by, so it is confirmed
    #   with a CONTROL instead: `--extra r_planet_hillslope_break 106` --
    #   the default, restated through the same channel -- renders
    #   planet_surface BYTE-IDENTICALLY to the golden. So the deltas above
    #   are attributable to the VALUE and not to the mechanism, and the
    #   green has a red beside it. Note also that the break-0 figures MOVED
    #   under #307 (planet_surface 64.49 -> 46.25, planet_lod_seam
    #   66.71 -> 20.95): #304 and #307 both act on the slope field, so the
    #   one measured before the other does not describe it afterwards.
    #   That is precisely why these were re-run rather than inherited.
    #
    #   The goldens are not blank, so "deleting a feature changes it" is not
    #   vacuous. Luminance over 0..255, re-measured on the regenerated PNGs:
    #     planet_surface         mean 106.5  min  7.2  max 195.2  189 levels
    #     planet_aerial          mean 203.6  min 41.8  max 241.3  195 levels
    #     planet_lod_seam        mean 209.2  min 109.4 max 232.8  125 levels
    #     planet_hillslope_rock  mean  75.3  min  7.0  max 210.7  193 levels
    #
    # So the thresholds are the MATRIX DEFAULTS (16 / 2 / 1) except where a
    # cell's own measurement asks for tighter, passed explicitly rather than
    # inherited so a reader can see they were chosen against a measurement:
    #
    #   MAX_DELTA 16 is 5.1x the largest same-host per-pixel delta (3.1623,
    #   planet_lod_seam Debug-vs-Release; the other four are 0.0000). It is
    #   NOT tightened towards that measurement on purpose: the only GPU
    #   available here is one M4 Max, and the headroom exists for a
    #   different driver's rounding and fma ordering -- the same reason
    #   every other Metal cell in this file uses 16. Tightening against a
    #   single-GPU sample would be less justified, not more.
    #
    #   MEAN_DELTA, against the weakest ablation (break-0) per cell:
    #     planet_surface         0.8 vs 46.25  --  57.8x
    #     planet_aerial          0.8 vs 15.61  --  19.5x
    #     planet_lod_seam        2   vs 20.95  --  10.5x
    #     planet_hillslope_rock  0.8 vs 17.13  --  21.4x
    #   Upward, the run-to-run term is exactly ZERO and the
    #   cross-configuration term is 0.000148 (planet_lod_seam; the rest are
    #   0.000000), which 2 sits 13 500x clear of. planet_lod_seam's 10.5x is
    #   the tightest downward margin of the four and still five times the
    #   4.6x/4.3x #280 was working with, so there is no case for moving any
    #   of them in either direction.
    #
    #   FAIL_PERCENT 1, against the same ablation: 45.0x / 27.3x / 39.2x /
    #   27.8x. Its whole range is headroom upward, because at MAX_DELTA 16
    #   no observed noise or cross-configuration pixel is classified bad at
    #   all (0.000%).
    #
    # When #284 pinned these cells, all six feature-deleted renders FAILED
    # these thresholds and all sixty identical-run renders PASSED them
    # against the PNGs committed at the time. #304 regenerated those PNGs
    # and re-ran the ablations; #307 regenerated them again and re-ran the
    # ablations again. The figures above are the #307 re-measurement.
    # Re-measure if a fixture's camera, resolution, exposure or frame count
    # changes: the margins are the point, not the numbers.
    #
    # ---- #307: WHAT MOVED, AND HOW MUCH -------------------------------
    #
    # #307 evaluates the slope that drives rock exposure on a FIXED level-11
    # lattice over a FIXED 180 m baseline instead of on the mesh's, so the
    # same ground stops changing colour when a chunk splits. It MOVED THREE
    # OF THESE CELLS and the PNGs were REGENERATED for it, with the
    # maintainer's sign-off covering exactly planet_surface, planet_aerial
    # and planet_lod_seam. Measured against clean main at 6f88475, whose
    # renders were bit-identical to the pre-regen goldens:
    #     planet_surface   mean 0.6731  rms  3.0623  max  54.16  bad  1.148%
    #     planet_aerial    mean 1.0579  rms  3.5936  max  70.58  bad  1.274%
    #     planet_lod_seam  mean 6.5737  rms 22.5834  max 150.83  bad  9.212%
    #     planet_land_orbit   bad 0, max 0.0000, mean 0.000000 -- IDENTICAL
    #
    # planet_lod_seam moves the most, and that is the defect leaving: it is
    # the fixture with the finest resident level (15, a 9.5 m slope
    # baseline), where the mesh-normal ramp was reading ~26 deg medians and
    # speckling rock across a snowfield.
    #
    # planet_land_orbit does not move because at 30 000 km the ray meets the
    # analytic body rather than the chunks, which the fixture's own header
    # already records.
    #
    # planet_surface BARELY FAILS, and that is recorded rather than smoothed
    # over. Its mean delta of 0.673 is BELOW its own MEAN_DELTA of 0.8, so
    # it fails on bad-pixel count ALONE, at 1.148% against 1% -- 1.15x. This
    # cell is a WEAK DETECTOR of this particular change, for the reason #307
    # measured: at 27.99 N the ground's slope at the ramp's own 180 m
    # baseline has a median of 12.04 deg against a 30 deg threshold, so the
    # ramp returns ~0 on both sides of the change and only the few steep
    # pixels move. Do not read that 1.15x as "the ramp is barely tested";
    # read it as "the ramp is barely tested HERE", which is exactly why #307
    # adds planet_hillslope_rock below, where the same change moves 14.3% of
    # pixels at mean 7.50.
    #
    # ---- WHY THESE THREE THRESHOLDS ARE THE VALUES THEY ARE ------------
    #
    # Kept from the earlier passes because it is REASONING rather than
    # measurement, and the measurements above have replaced the numbers it
    # was originally derived from:
    #
    #   * #280 moved planet_surface and planet_aerial from MEAN_DELTA 2 to
    #     0.8. Measured against ozone-off they moved by mean 3.65 and 3.43,
    #     so a MEAN_DELTA of 2 left 1.8x and 1.7x -- simultaneously too
    #     loose to be a clean signal and too tight to be safe against a
    #     cross-host run. 0.8 is the value that fixed that, and the margins
    #     re-measured above (57.8x and 19.5x) are far wider still, so it
    #     stays. planet_lod_seam keeps 2 because its cross-configuration
    #     term is the only non-zero one on this host.
    #
    #   * #303 made surface albedo a function of SLOPE. #304 changed the
    #     slope distribution below the 106 m break. #307 changed the
    #     BASELINE the slope is measured over. All three act on the same
    #     quantity, which is why each one moved these cells and why none of
    #     their measurements survives the next: anyone re-deriving any of
    #     the three has to RE-RENDER and RE-ABLATE, not re-use. The
    #     break-0 numbers above are the proof of that in miniature -- the
    #     identical ablation reads mean 46.25 on planet_surface today and
    #     read 64.49 before #307.
    #
    # ---- #318: THE RELIEF SECOND MOMENT, RESTORED ---------------------
    #
    # #318 is the FOURTH change to act on the slope field, and the same rule
    # applies: it RE-RENDERED and RE-ABLATED rather than inheriting. The
    # baker was area-averaging ETOPO to 2048x1024, which preserves each
    # texel's mean (the heights validate) and low-passes away its inter-texel
    # VARIANCE -- sigma(L_dem), the relief the continuation is anchored on.
    # relief is now measured on the full-resolution grid and carried in a
    # second .ptdem plane, so the terrain is as rough as the real data, not
    # as rough as its 19.5 km-block means. The ELEVATION plane is byte-
    # identical to before, so the DEM base heights did not move; only the
    # continuation amplitude did.
    #
    # WHAT MOVED, measured against clean main at 18ce5b3 (whose renders are
    # bit-identical to the pre-#318 goldens -- section (B) below):
    #     planet_surface        mean 45.92  rms 78.98  max 285.3  bad 98.43%
    #     planet_aerial         mean 43.02  rms 69.11  max 317.6  bad 99.19%
    #     planet_lod_seam       mean 17.44  rms 33.19  max 285.9  bad 97.64%
    #     planet_hillslope_rock mean 16.16  rms 35.15  max 252.7  bad 79.21%
    #   All four PNGs were REGENERATED and the maintainer's sign-off covers
    #   exactly these four. This is a whole-terrain change -- the default
    #   site stops being a smooth snowfield and reads as a rock-and-snow
    #   mountain -- which is the visible half of the #318 defect.
    #
    # WHAT DID NOT MOVE, and was NOT regenerated:
    #     planet_land_orbit     bad 0, max 0.0000, mean 0.000000 -- IDENTICAL
    #   At 30 000 km the ray meets the analytic body, not the chunks (the
    #   fixture's own header records this), so a sub-20 km continuation
    #   change cannot reach it. planet_horizon_clouds and _planar are also
    #   bit-identical branch-vs-main (they set r_planet_terrain to its
    #   default 0 -- no streamed terrain, no relief). The ocean cells set
    #   r_planet_terrain 0 as well. The DEM is read only by streamed
    #   terrain, so those five terrain cells are the whole blast radius; the
    #   git diff touches exactly the four PNGs above.
    #
    # NOISE FLOOR, re-established on 18ce5b3 (mac-release, this host):
    #   (A) run-to-run: a fresh main build renders all five planet cells
    #       BYTE-IDENTICALLY twice over -- imgdiff at 0/0/0 gives bad 0,
    #       max 0.0000, mean 0.000000 on every one. Terrain settling is a
    #       barrier, not a race, so there is no scheduling term to subtract.
    #   (B) main vs the committed (pre-#318) goldens: bad 0 on all five, so
    #       the goldens were current and every delta above is 100% signal.
    #
    # TOLERANCES: UNCHANGED (matrix defaults, seam keeps MEAN 2), and still
    # sized against the weakest ablation, RE-MEASURED against the #318
    # goldens. break-0 (r_planet_hillslope_break 0, the pre-#304 unbroken
    # power law) vs the new goldens:
    #     planet_surface        mean 50.22  bad 72.63%   62.8x / 72.6x
    #     planet_aerial         mean 27.81  bad 87.41%   34.8x / 87.4x
    #     planet_lod_seam       mean 39.47  bad 78.98%   19.7x / 79.0x
    #     planet_hillslope_rock mean 16.69  bad 69.72%   20.9x / 69.7x
    #   (margins are mean/MEAN_DELTA and bad%/FAIL_PERCENT.) The tightest,
    #   planet_lod_seam at 19.7x on mean, is still ~4x wider than the 4.6x
    #   #280 worked with, so no threshold moves. As with #304/#307 the
    #   break-0 mean rose (planet_surface 46.25 -> 50.22): un-suppressing
    #   relief gives the break more to tame, so deleting it costs more.
    #   CONFIRMED by a control -- r_planet_hillslope_break 106 (the default,
    #   restated through --extra) renders planet_surface BYTE-IDENTICALLY to
    #   the new golden, so the deltas are the value and not the channel.
    pt_add_golden_cell(SCENE planet_surface  BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 0.8 FAIL_PERCENT 1)
    # --- Land cover (#300) --------------------------------------------
    # The planet from full-disc range, which is the ONLY scale at which
    # "Earth is recognisable" is a claim a golden can hold.
    #
    # WHY planet_surface DOES NOT COVER IT. That cell stands at 27.99 N
    # with a 4.7 km horizon and the albedo raster's texel is 19.5 km, so
    # the raster contributes ONE COLOUR to the whole frame there. It can
    # tell an absent raster from a present one and nothing else. The
    # fixture's own header carries the rest of the reasoning.
    #
    # THRESHOLDS, MEASURED.
    #   Same-host repeat: this fixture is BIT-IDENTICAL run to run
    #   (mean 0.000000, max 0.0000, 0 bad pixels over 196 608). Terrain
    #   settling is a barrier rather than a race, so there is no scheduling
    #   noise to absorb.
    #   Signal: the r_planet_land_albedo 0 ablation -- which returns the
    #   surface to exactly the procedural classification -- moves it by
    #   mean 5.011 with 6.91% of pixels bad and a max delta of 275.5.
    #   The disc covers ~14% of the frame, so that mean is concentrated
    #   where the planet is. Unloading the raster entirely
    #   (r_planet_albedo_map "") gives the identical figures, which is
    #   what "the ablation really removes the feature" looks like.
    #
    #   MEAN_DELTA 0.8 therefore sits 6.3x below the weakest signal and
    #   infinitely above the observed noise. FAIL_PERCENT 1 sits 6.9x
    #   below the ablation's 6.91%. MAX_DELTA 16 is the matrix default and
    #   is NOT tightened toward the measured 0.0000 on purpose, for the
    #   reason the block above gives: one M4 Max is not a sample, and the
    #   headroom is for another driver's fma ordering.
    #
    #   RE-MEASURED after the sub-datum air-column fix, which moved this
    #   golden by mean 23.28 with 14.19% of pixels bad -- 4.6x the
    #   feature's own ablation, i.e. the atmosphere bug was a larger
    #   effect on this frame than the land cover it was supposed to be
    #   showing. The numbers above are that re-measurement, not the
    #   pre-fix ones; the margins moved by ~20% and the thresholds
    #   therefore did not need to.
    #
    #   RE-RUN AGAIN BY #307 rather than inherited, because a
    #   justification measured before #312 and #307 is not a
    #   justification afterwards. THIS CELL'S GOLDEN IS NOT REGENERATED
    #   -- it is bit-identical on clean main at 6f88475, on this branch,
    #   and to the committed PNG -- but the ablation was re-run against
    #   it anyway, and it lands on exactly the same figures: mean
    #   5.010744, bad 6.911214%, max 275.5086, with r_planet_albedo_map ""
    #   still giving the identical numbers to r_planet_land_albedo 0. So
    #   the margins above (6.3x and 6.9x) are re-confirmed by
    #   measurement, not carried over. Debug-vs-Release on this cell is
    #   also pixel-identical against the same PNG.
    #
    # NOT VACUOUS: the golden is not blank and not uniform. It is an
    # illuminated disc on black -- so the assertions that matter are made
    # in tests/pt_planet_albedo_test.cpp, which measures the raster's
    # SPECTRAL SHAPE (green where chlorophyll absorbs, red-over-green-over-
    # blue where iron oxide does) rather than its pixels.
    pt_add_golden_cell(SCENE planet_land_orbit BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 0.8 FAIL_PERCENT 1)
    # --- #317: the SAME disc WITH the sea -------------------------------
    # planet_land_orbit above pins r_planet_ocean 0 so the specular sea
    # cannot perturb the land-colour measurement -- which means it dodges
    # the exact configuration every cold start ships (terrain + ground +
    # ocean all on). #317 was a waterworld THERE: the sea-level shell
    # covered the whole globe and no land showed, and NO golden exercised
    # it, which is how it shipped uncaught. This cell is that golden -- the
    # A/B sibling that differs from planet_land_orbit in exactly one line,
    # r_planet_ocean, so the imgdiff between the two committed PNGs IS the
    # ocean shell's contribution and #317 is the difference between them.
    #
    # DETERMINISM. Same freeze + settle barrier as the sibling; verified
    # bit-identical run to run (equal renders, imgdiff at zero thresholds
    # bad=0, max 0.0000, mean 0.000000), so the noise floor is zero and
    # every figure below is signal. Converges in 1 settle round, 6 chunks.
    #
    # EXPOSURE RE-DERIVED, NOT INHERITED -- the ocean changes the frame's
    # luminance distribution, so planet_land_orbit's 0.05 does not carry
    # over. Swept at FRAMES 32 over the illuminated disc (~27 k px), the
    # two chroma separations the acceptance rests on move in OPPOSITE
    # directions, unlike the sibling where both rose together:
    #
    #     exposure  discmed  clip%  levels  soil R-B  sea B-R
    #       0.03      84.9    0.00    221      57.0     77.0
    #       0.05      97.3    0.00    232      51.0    102.0
    #       0.08     121.2    0.00    238      46.0    124.0
    #       0.12     146.0    0.00    241      41.0    140.0
    #       0.16     163.8    0.86    243      35.0    147.0
    #
    # Land soil chroma (iron oxide, R-B) FALLS with exposure as the bright
    # land washes out; sea blue (B-R) RISES. Normalising each to its own
    # peak, the minimum of the two is maximised at 0.08 (soil 0.81, sea
    # 0.84) -- the balance point where land AND sea are both well separated,
    # on the distinct-level plateau (238 of a 243 max) with nothing clipped.
    # That is the house method's "plateau of levels at the peak of the
    # chroma separations", re-run for a frame that now has two of them.
    #
    # NON-VACUOUS IN BOTH DIRECTIONS -- the whole point of the cell is that
    # it holds land AND sea, so both feature deletions must move it. Against
    # the committed golden, at MAX_DELTA 16:
    #     r_planet_ocean 0      (sea removed)  mean 13.47  bad 7.57%  max 289
    #     r_planet_land_albedo 0 (land colour) mean  4.38  bad 6.81%  max 267
    #   The ocean-off ablation is planet_land_orbit's own config, so the
    #   13.47 IS the sea's contribution measured directly. The land-off
    #   ablation keeps the raster loaded (so the coverage gate still runs
    #   and land still occludes the shell) and only zeroes the colour, so it
    #   is the strict "did the land cover reach the pixels" test. A third
    #   deletion -- reverting the #317 shader hunk -- returns the waterworld
    #   (all sea) and moves it far more; that red/green is in the PR.
    #
    # TOLERANCES, sized against the WEAKEST ablation (land-off, mean 4.38 /
    # bad 6.81%), noise floor zero:
    #   MEAN_DELTA 0.8  -> 5.5x below the weakest signal, infinitely above
    #                      the zero noise floor. Same value and reasoning as
    #                      the sibling family.
    #   FAIL_PERCENT 1  -> 6.8x below the weakest signal's 6.81%.
    #   MAX_DELTA 16    -> the matrix default, NOT tightened toward the
    #                      measured 0.0000 run-to-run: one M4 Max is not a
    #                      sample and the headroom is for another driver's
    #                      fma ordering, exactly as every other planet cell
    #                      here argues.
    # Re-measure if the camera, resolution, exposure or frame count moves.
    pt_add_golden_cell(SCENE planet_land_orbit_ocean BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 0.8 FAIL_PERCENT 1)
    pt_add_golden_cell(SCENE planet_aerial   BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 0.8 FAIL_PERCENT 1)
    pt_add_golden_cell(SCENE planet_lod_seam BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    # --- #307: the cell where the ROCK RAMP is the variable under test -----
    #
    # planet_surface cannot be that cell, and the reason is measured rather
    # than asserted: it stands 2.2 km above the derived snowline on ground
    # whose slope at the ramp's own 180 m measurement baseline has a median
    # of 12.04 deg against a 30 deg threshold. Rock exposure there is
    # 0.0007 and correctly so. This cell stands in the Annapurna Himal --
    # the study area of the paper the 30 deg comes from -- where earth_lite
    # carries 1 341 m of relief, the reference-scale slope has a median of
    # 42.96 deg, and the frame crosses the snowline so it holds rock, snow
    # and snow-shedding rock at once. See the fixture's header.
    #
    # THRESHOLDS, MEASURED -- and measured on THIS cell, not borrowed from
    # the three above it.
    #
    #   Same-host repeat: BIT-IDENTICAL run to run (equal MD5s; imgdiff at
    #   zero thresholds gives bad=0, max 0.0000, mean 0.000000), like every
    #   other planet cell here. Debug against Release is pixel-identical
    #   too, so this cell contributes nothing to the cross-configuration
    #   term the block above sizes MAX_DELTA against.
    #
    #   Signal, weakest first -- the same three ablations, confirmed the
    #   same three ways (log absence for the first two, the
    #   default-restated control for the third):
    #     r_planet_hillslope_break 0   mean  17.13   bad 27.83%
    #     r_planet_dem <no such file>  mean  38.67   bad 55.03%
    #     r_planet_terrain 0           mean  66.89   bad 87.03%
    #   So MEAN_DELTA 0.8 sits 21.4x below the weakest and FAIL_PERCENT 1
    #   sits 27.8x below it, against an observed noise floor of exactly
    #   zero. MAX_DELTA 16 is the matrix default, kept for another driver's
    #   fma ordering for the reason the block above gives.
    #
    #   AND THE ONE THAT MAKES THIS CELL WORTH ADDING: #307's own change --
    #   the ramp reading a fixed 180 m baseline instead of the mesh's --
    #   moves this frame by mean 7.5045, rms 18.5107, max 140.38, with
    #   14.298% of pixels bad. That is 12.4x the cell's own MEAN_DELTA and
    #   14.3x its FAIL_PERCENT. Compare planet_surface, where the same
    #   change is mean 0.6731 and 1.148% bad -- BELOW its MEAN_DELTA and
    #   only 1.15x its FAIL_PERCENT. The ramp is engaged here and it is not
    #   engaged there, which is the entire reason this cell exists.
    #
    #   NOT VACUOUS, and checked rather than assumed: the golden is neither
    #   blank nor uniform -- mean luminance 75.3 over 0..255, min 7.0, max
    #   210.7, 193 distinct luminance levels and 25 122 distinct RGB
    #   triples. A frame where the feature is invisible cannot fail for the
    #   feature's sake, and this one is not that frame.
    pt_add_golden_cell(SCENE planet_hillslope_rock BACKEND metal DENOISER off
                       FRAMES 32 MAX_DELTA 16 MEAN_DELTA 0.8 FAIL_PERCENT 1)
    # --- end Planetary P4 -------------------------------------------------
    pt_add_golden_cell(SCENE planet_horizon_clouds        BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
    pt_add_golden_cell(SCENE planet_horizon_clouds_planar BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
    # --- Planetary P3 (#257): ozone at civil twilight ----------------------
    # The phase's other acceptance criterion: the twilight sky must read
    # BLUE, not brown, and that is what ozone buys.
    #
    #   civil_twilight            sun 4 degrees below the horizon, ozone
    #                             at 1.0x the US Standard Atmosphere 1976
    #                             column, and r_sun_physical_transmittance
    #                             on so sunlight's colour comes from the
    #                             real slant path.
    #   civil_twilight_no_ozone   the same frame with ozone removed and
    #                             nothing else changed.
    #
    # r_sun_physical_transmittance is mandatory here and the reason is
    # geometric: ozone lives at 25 km, and a camera-to-surface ray at eye
    # height tops out around 200 m over its whole 50 km, so aerial
    # perspective can never see the layer. SUNLIGHT crosses it at a
    # grazing angle for a very long way at twilight. Without the physical
    # sun path the two halves of this pair would be identical, and the
    # cell would pin nothing.
    #
    # Metal only: the software tracer has no atmospheric transmittance at
    # all, so both halves would render its own procSky mirror and the
    # pair would carry no signal (the same reason the sunset_altitude
    # software A/B is bit-identical -- see the note on those cells).
    #
    # Measured on the dev M4 Max: blue-to-green 1.34 with ozone against
    # 1.23 without, blue-to-red 1.56 against 1.47. Ozone removes green
    # and red and leaves blue alone, which is the Chappuis band's
    # spectral shape. tests/pt_atmosphere_test.cpp asserts the same
    # mechanism on the numbers directly, including the ordering that
    # would catch the triple being entered in the wrong wavelength order.
    pt_add_golden_cell(SCENE civil_twilight          BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
    pt_add_golden_cell(SCENE civil_twilight_no_ozone BACKEND metal DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
    # --- end Planetary P3 --------------------------------------------------

    # --- Wave 9 hosek-sky --------------------------------------------------
    # Hosek-Wilkie physically-based sky (r_sky_mode hosek). Daytime sun
    # at +50 deg, camera yawed toward the sun azimuth so the frame shows
    # the zenith-to-horizon gradient AND the sun-side forward-scatter
    # glow. The Lambert ground plane in the fixture is lit by the same
    # hosekSky() radiance (visible-dome / env-lighting consistency). The
    # metal cell is the Wave 9 build+test target; software gives the
    # CI-exercised deterministic baseline. Tolerances mirror the #112
    # sky cells -- bright HDR sky pixels exaggerate cross-backend FP
    # roundoff, and the analytic Hosek fit has more transcendental ops
    # (exp/pow per channel) than the procedural gradient.
    pt_add_golden_cell(SCENE sky_hosek BACKEND software DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE sky_hosek BACKEND metal    DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end Wave 9 hosek-sky ----------------------------------------------

    # --- Fluid Phase 1 (#136) -- smoke emitter fixture --------------------
    # Two density-injection plumes at chimney altitude with clouds off
    # and procedural sky. Validates: (a) the emitter loop ADDS density
    # on top of the procedural cloud field; (b) plumes are sun-lit +
    # sky-lit via the existing cloud-march NEE pipeline; (c)
    # `r_smoke_enabled` gates the path tracer-side smoke loop. Software
    # backend is the CI-exercised cell; Metal + SVGF a-trous is the
    # user's primary dev combo and stresses the denoiser path alongside
    # the new smoke ssbo binding. Tolerances are widened because the
    # in-cloud single-scatter path is more sensitive to per-backend FP
    # ordering than primary-hit Lambert.
    pt_add_golden_cell(SCENE smoke_emitter_basic BACKEND software DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE smoke_emitter_basic BACKEND metal    DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end Fluid Phase 1 ------------------------------------------------

    # --- Wave 7 (#24) -- procedural-raymarched cloud mode -----------------
    # Pins the new r_clouds_mode procedural_raymarched code path against
    # a committed PNG so regressions in either the CloudsRaymarch
    # pre-pass or the CloudsComposite alpha-blend surface immediately.
    # Same cloud-bearing scene as the existing godrays fixture (sunset
    # + broken cumulus) so visual comparison vs the path-traced output
    # is direct. Tolerances widened versus cornell_csg because cloud
    # raymarch + light-march density evaluation involves more fbm
    # taps than the analytic-geometry path; cross-backend FP-order
    # variance accumulates faster.
    #
    # software backend SKIPPED -- the software RHI doesn't run compute
    # pipelines, so the raymarched cloud pre-pass collapses to its
    # gate-off path and the cell would just match the pathtraced
    # software golden (no signal). Metal + denoised is the canonical
    # raymarched-clouds capture.
    # #280: smallest signal 7.19 (ozone off; multiple scattering off moves
    # this cell by 249, because a cloud deck lit through the atmosphere is
    # mostly higher-order light). MEAN_DELTA 1.5 -> 4.8x clear.
    pt_add_golden_cell(SCENE clouds_raymarched BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 1.5 FAIL_PERCENT 3)
    # --- end Wave 7 (#24) -------------------------------------------------

    # --- Wave 9 -- god rays / crepuscular light shafts --------------------
    # Pins the dedicated screen-space god-rays pass (GodRays.slang) ON
    # against the sunset + broken-cumulus clouds_godrays fixture: the sun
    # sits low and centred, broken cumulus leaves gaps, and r_godrays 1
    # radial-blurs the sky/sun light through those gaps into visible
    # shafts. EXTRA flips r_godrays on (the fixture leaves it off so the
    # default render path stays bit-for-bit identical to its own golden).
    #
    # Metal + denoised is the canonical capture (the god-rays dispatch
    # rides the use_engine_tonemap branch, which on Metal requires a
    # denoiser; software backend doesn't run compute pipelines and would
    # collapse the pass to its gate-off path, matching the no-godrays
    # golden -- no signal). Tolerances match the clouds_raymarched cell:
    # the cloud march + radial-blur accumulation involve many fbm /
    # texture taps so cross-backend FP-order variance is wider than the
    # analytic-geometry cornell baseline.
    #
    # REGENERATED 2026-08-27 (planetary P3, #257). This fixture sets
    # `r_volumetric_density 0.008` -- a deliberately heavy god-ray haze,
    # 4x the (already unphysical) 0.002 default and roughly 16-80x the
    # real-Earth sea-level aerosol range of 1e-4 to 5e-4. P3 gives the
    # aerosol its real ABSORBING half, which the engine did not model at
    # all before (an implied single-scattering albedo of 1), so extinction
    # in this fixture roughly doubles and the frame moved by mean 4.153
    # against the MEAN_DELTA 4 registered here. That threshold was
    # calibrated against physics this phase deliberately changes, so the
    # correct response is a regeneration, not a widened tolerance -- the
    # same call #269 made for the July correctness fixes.
    #
    # The alternative reading was measured, not assumed: treating
    # r_volumetric_density as EXTINCTION rather than as scattering would
    # have preserved this fixture's aerial perspective exactly, but it
    # halves every haze glow and moves this same cell by mean 25.1 --
    # six times worse. So the fixture is not fine-with-a-broken-
    # implementation; both readings move it, and the one that moves it
    # least is also the one that keeps a user's authored haze level
    # meaning what they set. The reasoning is recorded at ptAtmoScale in
    # shaders/PathTraceMath.slang.
    #
    # The new capture is bit-identical across three consecutive renders
    # on the dev M4 Max (max_delta 0), so the 32/4/3 thresholds remain
    # pure cross-host headroom.
    # #280: smallest signal 7.19 (ozone off) -> MEAN_DELTA 1.5, 4.8x clear.
    pt_add_golden_cell(SCENE clouds_godrays BACKEND metal DENOISER svgf_atrous
        EXTRA "r_godrays 1"
        MAX_DELTA 32 MEAN_DELTA 1.5 FAIL_PERCENT 3)
    # --- end Wave 9 -------------------------------------------------------

    # --- Fluid Phase 3 (#22) -- SPH smoke wind fixture ---------------------
    # SPH-driven smoke column with horizontal wind. Acceptance bar from
    # the issue body: smoke responds to r_smoke_wind cvar; column drifts
    # sideways instead of straight up.
    #
    # Tolerances generous (similar to phys_rb_* fixtures) because the
    # SPH solver's per-frame integration is not bit-deterministic across
    # build types -- the spatial-hash bucket order depends on small
    # FP ordering differences. We're proving the gross visual shape
    # (column leans into wind), not per-pixel equivalence.
    #
    # Metal only at first; software backend goldens can be added once
    # the Metal one is committed.
    pt_add_golden_cell(SCENE smoke_phase3_wind BACKEND metal DENOISER off
        FRAMES 90
        MAX_DELTA 64 MEAN_DELTA 16 FAIL_PERCENT 10)
    # --- end Fluid Phase 3 ------------------------------------------------

    # --- FFT ocean surface (Wave 8, #25) ----------------------------------
    # Pins the CPU Tessendorf FFT ocean against a committed PNG: the
    # displacement + normal textures bind at engine slots 14/15
    # (vk::binding 32/33), the MAT_WATER branch ray-marches the displaced
    # heightfield, and choppy crests + foam render at a low sunset sun.
    #
    # FRAMES 120: the ocean animation advances by the per-frame dt
    # (ocean_time_ accumulates), so the surface needs a stretch of frames
    # to evolve from the t=0 spectrum into a developed sea before the
    # capture. 120 frames warms it well past the dominant-wave period.
    #
    # THE CAUSE WAS NAMED WRONG FOR MONTHS. This comment used to say the
    # tolerances had to absorb "cross-host / cross-build GPU FP-ordering".
    # That is not what it was. MetalDevice::WriteTexture is a CPU
    # replaceRegion and MetalDevice::Submit does not wait, so
    # Engine::EnsureOceanUploaded wrote the ocean field for frame N over
    # the one the GPU was still reading for frame N-1 -- a plain
    # write-after-read race. #259 closed it with a 3-deep texture ring
    # (Engine::kOceanUploadRing), and #295 then found that the ring's own
    # premise -- "the GPU is at most two frames behind the CPU" -- was
    # unenforced on Metal, which has no fence and whose headless captures
    # take no drawable to throttle on. Read the ring and the frame-pacing
    # bound (pt::rhi::kMaxFramesInFlight) as one fix; the measurement that
    # separated them is in the planet_ocean_* block below.
    #
    # REPRODUCED AND CLOSED IN THIS CELL'S OWN CONFIGURATION, not inferred
    # from the P5 fixtures. Six consecutive renders of THIS cell at
    # 3db2c8c -- the commit before #291 merged -- gave FIVE DISTINCT
    # IMAGES, and six of ocean_foam gave five.
    #
    # AFTER THE RING ALONE (#294, at 586f926) the floor was very low but
    # NOT zero: 39 Release renders gave 35 byte-identical and 4 outliers
    # at max 1.4142 L2 / mean <= 7.6e-3, 16 Debug renders gave 14 and 2 --
    # a one-LSB residue on a few dozen of 196 608 pixels, appearing in
    # roughly one render in ten, source unidentified because it could not
    # reach a threshold digit from four orders down.
    #
    # IT IS NOW EXACTLY ZERO, AND THE SOURCE WAS THE SAME RACE (#295).
    # The ring assumed a frames-in-flight bound Metal did not enforce, so
    # on the rare frame the CPU got four command buffers ahead the ring
    # wrapped onto a slot a live dispatch was still reading. One LSB here
    # rather than a torn frame because svgf_atrous averages 120 of them.
    # With MetalDevice::BeginFrame holding the CPU to
    # pt::rhi::kMaxFramesInFlight (the measurement is in the
    # planet_ocean_* block below), on the dev M4 Max:
    #
    #   Release  34 renders (10 serial + 24 at 12-way concurrency on one
    #            GPU): ALL 34 BYTE-IDENTICAL, one md5
    #   Debug    10 renders: ALL 10 BYTE-IDENTICAL
    #
    # So the floor these thresholds sit on is the same exact zero the three
    # planet_ocean_* cells have. The Release/Debug gap that remains is
    # optimisation level, not nondeterminism.
    #
    # THE PNG WAS THEN REGENERATED (#263), AND THAT IS WHY THESE NUMBERS
    # ARE NOT #294's. #294 measured this cell against a golden carrying a
    # mean 7.4947 standing drift -- dating to the July 2026 correctness
    # fixes (b2111dd Hosek, 7c237bc SIGMA demod, 875ff1d env-map NEE) plus
    # P3's aerosol absorption, the #269 category -- and every threshold it
    # could choose had to clear that drift first. The capture is re-pinned
    # at 586f926, so the drift is gone and the registered config now
    # measures EXACTLY ZERO against its own golden. Everything below is
    # re-measured against the new PNG; nothing is carried over.
    #
    # THRESHOLDS, MEASURED. Both criteria now discriminate, which is new.
    # Release and Debug, at the registered MAX_DELTA 16 (Debug is the
    # sensitive config and is the number quoted second):
    #
    #   config                     mean R / D       bad@16 R / D
    #   registered config        0.0000 /  0.0388   0.0000 /  0.0148 %
    #   r_ocean 0               11.6420 / 11.6588  27.6738 / 27.6738 %
    #   r_ocean_amplitude 0     11.8072 / 11.8241  26.7324 / 26.7324 %
    #   r_ocean_march_steps 8    6.0787 /  6.0958  12.7416 / 12.7406 %
    #   r_ocean_wind_speed 4     6.5308 /  6.5473  14.6942 / 14.6907 %
    #
    # THE VALUES ARE THE MATRIX DEFAULTS 16 / 2 / 1, passed explicitly
    # rather than inherited, for exactly the reason the planet_ocean_*
    # block gives: the within-host floor is zero, so what is left to
    # absorb is another Apple-silicon Mac's rounding and fma ordering,
    # which is unmeasured. These cells are Metal-only and the CI matrix
    # runs `-L golden_software`, so no runner ever renders them -- the
    # only host that does is a developer's.
    #
    #   * BOTH criteria trip on every ablation. Weakest margin over
    #     MEAN_DELTA 2 is r_ocean_march_steps 8 at 3.0x; weakest over
    #     FAIL_PERCENT 1 is the same ablation at 12.7x. The registered
    #     config clears MEAN_DELTA by 51x and FAIL_PERCENT by 68x in
    #     Debug, and by infinity in Release.
    #   * MEAN_DELTA IS NO LONGER A BACKSTOP. #294 had to document it as
    #     "a backstop that cannot discriminate", because the golden's own
    #     drift was 63% of the whole-feature-deleted mean and no value fit
    #     between them. With the drift at zero the gap is 0.0388 to 6.08 in
    #     Debug -- 157x -- and 2 sits in the middle of it.
    #   * MAX_DELTA 64 -> 16. The 64 existed to buy separation against a
    #     drifted golden ("at 16 only 1.5x"). Against a golden the
    #     registered config matches exactly, the separation is infinite at
    #     every MAX_DELTA, so the choice reverts to the house default,
    #     which is also the more sensitive one: it takes the weakest
    #     ablation's bad-pixel margin from 1.8x to 12.7x.
    #
    # r_ocean_wind_speed 4 IS NOW CAUGHT, AND THAT IS THE MEASUREMENT THAT
    # JUSTIFIES THE REGENERATION. Against the stale PNG the WRONG sea state
    # scored mean 6.128 and the CORRECT 12 m/s scored 7.4947 -- the wrong
    # wind rendered closer to the golden than the right one, and no
    # threshold can recover from an inversion like that. Against the new
    # PNG the correct wind is 0.000000 and the wrong wind is 6.5308, which
    # is 3.3x over MEAN_DELTA and 14.7x over FAIL_PERCENT. The ordering is
    # restored and the ablation trips both criteria.
    #
    # NONE OF THESE ABLATIONS IS A BLACK FRAME -- mean luminance 25.4..29.6
    # of 255 against the registered config's 26.1, and 97.9-99.4% of pixels
    # above black in every one. They change content, not remove it. And
    # each --extra landed: the archived cvar state written on exit carries
    # the overridden value in every run (r_ocean drops OUT of demont.cfg on
    # the ablation, because 0 is its registered default and the fixture's
    # `r_ocean 1` is what put it there), a misspelled cvar name aborts Init
    # with exit 3 rather than rendering the baseline silently, and every
    # ablation image differs from the registered render.
    #
    # The cell catches gross regressions: the ocean gate failing to engage
    # (flat plane), the textures not binding (black / sky-through water),
    # or the ray-march dropping rays through the surface. Metal-only;
    # software gates the ocean off (its CPU tracer ignores the textures)
    # so no software cell, and a Vulkan cell can land once a Vulkan PNG is
    # committed.
    pt_add_golden_cell(SCENE ocean_fft BACKEND metal DENOISER svgf_atrous
        FRAMES 120
        MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    # --- end FFT ocean surface --------------------------------------------

    # --- Ocean foam + whitecaps (Wave 9) ----------------------------------
    # Pins the Wave 9 foam shading on top of the FFT ocean: a Beaufort-7
    # (16 m/s) wind drives a choppy sea whose breaking crests + wind-driven
    # whitecaps shade as bright, rough, near-white Lambertian froth (the
    # ocean_foam.cfg fixture uses a daylight sun + higher exposure so the
    # foam reads as white in the PNG, unlike ocean_fft.cfg's dark sunset
    # which hides it). The cell catches regressions in:
    #   * the CPU foam model (crest-fold Jacobian foam + wind-coverage ramp
    #     + temporal persistence packed into ocean_displacement.w)
    #   * the MAT_WATER foam BRDF blend (threshold/smoothstep -> bright
    #     diffuse whitecap override of the specular water surface)
    #   * r_ocean_foam_amount and _persistence -- but NOT _coverage, see
    #     the ablation table below, where the old comment over-claimed
    #
    # SAME CAUSE, SAME FIX, MEASURED SEPARATELY. This cell shared
    # ocean_fft's write-after-read race on the ocean upload (read that
    # block for the mechanism) and shared its wrong explanation. Six
    # consecutive renders at 3db2c8c, the commit before #291 merged, gave
    # FIVE DISTINCT IMAGES.
    #
    # After the ring alone (#294, at 586f926) this cell had the same
    # one-LSB residue ocean_fft had: 16 Release renders gave 11
    # byte-identical and 5 outliers at max 1.4142 L2 / mean <= 3.4e-4,
    # 16 Debug renders gave 14 and 2 at max 1.0 L2 / mean 9.7e-5.
    #
    # #295 CLOSED THE RESIDUE TOO -- it was the ring's frames-in-flight
    # premise going unenforced on Metal, one layer under the ring itself.
    # With the bound in place, on the dev M4 Max:
    #
    #   Release  34 renders (10 serial + 24 at 12-way concurrency on one
    #            GPU): ALL 34 BYTE-IDENTICAL, one md5
    #   Debug    10 renders: ALL 10 BYTE-IDENTICAL
    #
    # That holds under the 12-way GPU contention #263 reported this cell
    # failing under, which is where the 24 concurrent renders came from.
    #
    # THE PNG WAS THEN REGENERATED (#263), AND MOST OF WHAT IT REMOVES
    # ARRIVED WITH #291. At 3db2c8c this cell rendered mean 4.109 / bad
    # 0.668% from its golden; at 586f926 it rendered 11.374 / 5.228,
    # because the upload ring changed WHICH ocean field a frame reads and
    # the old PNG encoded the RACED phase of the wave animation. #294
    # measured against that PNG and had 1.41x mean headroom and 1.53x
    # bad-pixel headroom left -- thresholds sized around a capture of a
    # bug. The capture is re-pinned at 586f926, so the registered config
    # now measures EXACTLY ZERO against its own golden and everything
    # below is re-measured; nothing is carried over.
    #
    # THRESHOLDS, MEASURED. Release and Debug, at the registered
    # MAX_DELTA 16 (Debug is the sensitive config and is quoted second):
    #
    #   config                        mean R / D       bad@16 R / D
    #   registered config            0.0000 /  0.0725   0.0000 /  0.0519 %
    #   r_ocean_foam_persistence 0  14.9881 / 14.9898  27.0569 / 27.0615 %
    #   r_ocean_foam_amount 0       46.4664 / 46.4661  44.0186 / 44.0216 %
    #   r_ocean 0                   61.3510 / 61.3511  56.4265 / 56.4265 %
    #   r_ocean_foam_coverage 0      0.7603 /  0.7939   1.0223 /  1.0462 %
    #
    # THE VALUES ARE THE MATRIX DEFAULTS 16 / 2 / 1, for the same reason
    # given in the ocean_fft block: a within-host floor of exactly zero
    # leaves only another Apple-silicon Mac's rounding to absorb, and these
    # Metal cells never run on a CI runner (`-L golden_software`).
    #
    #   * BOTH criteria trip on all three ablations the cell claims.
    #     Weakest margin over MEAN_DELTA 2 is persistence-off at 7.5x;
    #     weakest over FAIL_PERCENT 1 is persistence-off at 27.1x. The
    #     registered config clears MEAN_DELTA by 27.6x and FAIL_PERCENT by
    #     19.3x in Debug, and by infinity in Release.
    #   * FAIL_PERCENT 8 -> 1 and MEAN_DELTA 16 -> 2 are both consequences
    #     of the drift going away, not independent tightenings. #294 could
    #     not go below 16 on the mean because the registered config already
    #     sat at 11.37; it now sits at 0.07.
    #
    # r_ocean_foam_coverage 0 IS STILL NOT A CLAIM THIS CELL MAKES. It
    # lands at 1.02x (Release) / 1.05x (Debug) of the bad-pixel gate --
    # nominally over the line, but by four hundredths of a percent, which
    # is a coin flip on another host and not a margin. The mean criterion
    # does not catch it at all (0.76, i.e. 0.38x). The reason is unchanged
    # and is the fixture's, not the threshold's: at foam_amount 1.2 and
    # 16 m/s the crest-fold Jacobian foam saturates the crests, so removing
    # the wind-coverage ramp barely moves the frame. Pinning _coverage
    # needs a fixture at a wind speed where the ramp does the work, which
    # is why the bullet list above still excludes it.
    #
    # NO ABLATION IS A BLACK FRAME: mean luminance 131.9..155.1 of 255
    # against the registered config's 155.3, and 100.00% of pixels above
    # black in every one. And each --extra landed: the archived cvar state
    # written on exit carries the overridden value in every run (r_ocean
    # drops OUT of demont.cfg on the ablation because 0 is its registered
    # default), a misspelled cvar name aborts Init with exit 3 rather than
    # rendering the baseline silently, and every ablation image differs
    # from the registered render.
    #
    # Metal-only (software gates the ocean off; a Vulkan cell can land once
    # a Vulkan PNG is committed).
    pt_add_golden_cell(SCENE ocean_foam BACKEND metal DENOISER svgf_atrous
        FRAMES 120
        MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    # --- end Ocean foam + whitecaps ---------------------------------------

    # --- Planetary P5 (#259): the planetary ocean -------------------------
    #
    # THE PAIR OF CELLS IS THE PHASE. planet_ocean_boat and
    # planet_ocean_orbit are the two ends of the headline claim -- "reads as
    # water from a boat AND as a smooth specular sphere from orbit, with no
    # visible transition between them" -- rendered by the SAME code with
    # nothing different but the camera altitude. planet_ocean_snell is the
    # third camera the phase makes possible at all: one under the water,
    # which no fixture in this repo has ever had, and which is where the
    # Wave 8 above-to-below-only crossing test has been wrong since #25.
    #
    # WHAT A GOLDEN CANNOT DO HERE, AND WHAT DOES IT INSTEAD. Two committed
    # PNGs at the ends of a climb say nothing about the middle; they can both
    # be right while everything between them jumps.
    # tests/pt_planet_ocean_test.cpp carries that claim as numbers -- the
    # conservation identity sigma^2_geometry(w) + alpha^2_brdf(w) ==
    # sigma^2_CoxMunk at every footprint, and #260's refinement ladder with a
    # hard-switch control that plateaus. These cells pin that the numbers
    # reach the screen.
    #
    # NO TERRAIN, AND THEREFORE NO SETTLE BARRIER. All three fixtures set
    # r_planet_terrain 0 and r_planet_ground 0: the planet IS the ocean.
    # That isolates the phase (with the analytic backstop on it would sit at
    # the SAME radius as the shell and fight it for every pixel) and it means
    # none of #284's residency machinery is in the loop, so these cells are
    # reproducible without r_planet_lod_freeze or pt_planet_settle_rounds.
    #
    # DETERMINISM, MEASURED RATHER THAN ASSUMED -- and it did not start out
    # true. The first six identical renders of planet_ocean_boat produced
    # THREE distinct images, differing by up to 48.7 L2 on 10 111 of 196 608
    # pixels, with ocean_time_ and frame_index_ logged identical across all
    # six. The cause is a plain write-after-read race:
    # MetalDevice::WriteTexture is a CPU replaceRegion and
    # MetalDevice::Submit does not wait, so the ocean field for frame N was
    # being written over the one the GPU was still reading for frame N-1.
    # Engine::EnsureOceanUploaded now writes into a 3-deep ring
    # (kOceanUploadRing).
    #
    # That was also the real reason ocean_fft and ocean_foam were on the
    # known-nondeterministic list, and it has since been confirmed IN
    # THEIR OWN CONFIGURATION rather than inferred from here: six renders
    # of each at 3db2c8c gave five distinct images apiece. Both cells have
    # been re-characterised in place -- see their blocks above, including
    # what the #295 pacing bound below did to the 1-LSB residue #294
    # measured. What has NOT been re-measured is smoke_phase3_wind: it
    # stays on that list, with its cause still unknown, and nothing here
    # touches it.
    #
    # THAT RING WAS NECESSARY AND IT WAS NOT SUFFICIENT (#295). This comment
    # used to end "and all three cells render byte-identical, ten consecutive
    # times, in Release AND in Debug". That was measured, and on `main` it
    # was still false: planet_ocean_snell rendered a DIFFERENT image in 1-2
    # runs out of every 8-10, off by max 81.8 / mean 1.68 on 2.8% of pixels
    # against its own committed PNG, and worse under `ctest -j 4`.
    #
    # What a 3-deep ring assumes is that the GPU is at most two frames
    # behind the CPU. Vulkan enforces exactly that with a fence ring;
    # METAL ENFORCED NOTHING. Submit() does not wait, and a headless capture
    # -- which is what every cell in this file is -- takes no drawable, so
    # nextDrawable()'s implicit throttle never fired either. The CPU ran as
    # far ahead as it liked.
    #
    # Instrumented rather than argued. Over ten renders of _snell the
    # per-frame FNV-1a of the uploaded cascade atlas was IDENTICAL on all
    # ten (same ocean_time_, same ring slot, same bytes), so the solver was
    # exonerated; the two runs that produced the outlier image were exactly
    # the two in which the Metal backend reached FOUR command buffers in
    # flight, one more than the ring has slots. Perfect correlation, 10/10.
    # _snell is the fixture that catches it because it is the submerged one:
    # every pixel is a ray-march through the field and then a refraction
    # through the surface normal, so a stale slot lands on the whole frame
    # rather than on a glitter patch.
    #
    # The fix is the missing bound, not a deeper ring:
    # MetalDevice::BeginFrame now waits on the command buffer from
    # pt::rhi::kMaxFramesInFlight frames ago, the same contract
    # VulkanDevice::BeginFrame has always had, and the ring depths are
    # static_assert'd against it. AFTER IT, on the dev M4 Max: ten
    # consecutive byte-identical renders of all three cells in Release AND
    # ten in Debug, plus a clean `ctest -j 4`. Release matches the committed
    # PNG exactly (max_delta 0.0000 on all three); Debug is a different
    # optimisation level and lands where it always has -- boat 0.003% bad,
    # snell 0.032%, orbit bit-identical -- comfortably inside FAIL_PERCENT 1
    # and unchanged by this fix. Red-then-green: reverting just the
    # BeginFrame wait brings the outlier straight back.
    #
    # THRESHOLDS, MEASURED. Run-to-run on the dev M4 Max is 0.0000 max /
    # 0.000000 mean on all three -- bit-identical -- so these are the MATRIX
    # DEFAULTS (16 / 2 / 1) passed explicitly rather than inherited, kept
    # loose for a different driver's rounding and fma ordering exactly as
    # every other Metal cell in this file. The margins against the weakest
    # feature-deleted signal, all of which FAIL these thresholds:
    #
    #   planet_ocean_boat    r_ocean 0            mean  37.8  bad 53.9%
    #                        r_planet_ocean 0     mean  85.3  bad 61.1%
    #                        r_ocean_wind_speed 4 mean  32.3  bad 48.4%
    #   planet_ocean_orbit   r_planet_ocean 0     mean  66.7  bad 100.0%
    #                        r_ocean_wind_speed 4 mean  15.6  bad 28.2%
    #   planet_ocean_snell   r_ocean 0            mean  86.5  bad 87.7%
    #                        r_planet_ocean 0     mean 211.9  bad 99.9%
    #
    # The weakest of those is 7.8x above MEAN_DELTA 2 and 28x above
    # FAIL_PERCENT 1, against a noise floor of exactly zero. Re-measure if a
    # fixture's camera, resolution, exposure or frame count changes: the
    # margins are the point, not the numbers.
    #
    # AND NOT ONE OF THOSE ABLATIONS IS A BLACK FRAME. r_planet_ocean 0
    # leaves the sky and the sun where they were, r_ocean 0 leaves the shell
    # and its Cox-Munk roughness -- deleting the feature changes the image
    # by changing content, not by removing it, which is the trap P4 nearly
    # committed two goldens into.
    #
    # FRAMES 48 rather than the 120 the Wave 8 ocean cells use: those needed
    # a long warm-up because the surface had to evolve into a developed sea
    # from t = 0, and the cascades here are spectrally normalised to Cox &
    # Munk from the first frame rather than growing into an amplitude.
    pt_add_golden_cell(SCENE planet_ocean_boat  BACKEND metal DENOISER off
                       FRAMES 48 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    pt_add_golden_cell(SCENE planet_ocean_orbit BACKEND metal DENOISER off
                       FRAMES 48 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    pt_add_golden_cell(SCENE planet_ocean_snell BACKEND metal DENOISER off
                       FRAMES 48 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)

    # --- #305: WATER-LEAVING RADIANCE. THE FULL GOLDEN DECOMPOSITION -------
    #
    # NOT REGENERATED. #305 moves every water cell in the matrix, and the
    # regeneration is the user's call -- three other branches were in flight
    # over the same planet cells when this was measured. What is recorded
    # here instead is the decomposition, so whoever does regenerate can tell
    # signal from drift without re-deriving it.
    #
    # THE NOISE FLOOR IS EXACTLY ZERO. main rendered twice, per cell, same
    # frame count: max L2 0.00, mean L2 0.0000, on all eight water cells.
    # These renders are bit-deterministic on this host, so every number
    # below is signal and none of it is run-to-run variance.
    #
    #                          mine vs main      of which, with the
    #   cell                   max/mean/bad%     in-scatter gate OFF
    #   ---------------------  ----------------  -------------------
    #   ocean_fft                2.45/ 0.09/0.0   0.00/0.0000  (exact)
    #   planet_ocean_boat        1.73/ 0.13/0.0   0.00/0.0000  (exact)
    #   planet_ocean_orbit       3.00/ 1.65/0.0   0.00/0.0000  (exact)
    #   planet_underwater_col   48.01/ 1.11/0.4   0.00/0.0000  (exact)
    #   water_pool              19.44/ 1.44/0.0   1.73/0.2424
    #   ocean_foam             161.50/ 3.86/7.2   0.00/0.0000  (exact)
    #   planet_ocean_snell      39.56/ 2.18/0.8   0.00/0.0000  (exact)
    #   water_pool_submerged    82.46/12.59/23.7  4.12/1.1258
    #
    # THE SECOND COLUMN IS THE ISOLATION PROOF. With r_water_inscatter 0 --
    # which is now the master gate for BOTH the submerged march and the new
    # water-leaving term -- this branch reproduces main EXACTLY on six of
    # the eight cells. The two that do not are the only two with a lit
    # surface UNDER the water, and they move because the one part of #305
    # that is deliberately not gated is transmittance(): a shadow ray
    # carrying illumination through water now attenuates with c_eff = a +
    # b_b instead of with the bare absorption. 4.12 L2 worst case.
    #
    # WHICH CELLS ACTUALLY FAIL, AND WHY EACH ONE MOVES:
    #
    #   ocean_foam            max 161.50 mean 3.86 bad 7.22% (thr 16/2/1)
    #     A camera 3 m over a flat MAT_WATER plane at 3.3x the planetary
    #     cells' exposure. The whole delta is below the horizon and is a
    #     strict BRIGHTENING in blue: the worst pixel goes from (1, 3, 10)
    #     to (10, 67, 158). That is water that used to render as near-black
    #     because nothing put a source term in it.
    #
    #   planet_ocean_snell    max  39.56 mean 2.18 bad 0.75% (thr 16/2/1)
    #     Submerged camera, so the water-leaving term never fires here at
    #     all. This cell moves for the two corrections to the march: the
    #     mu_a/mu_s refraction Jacobian (a dimming, 0.93 at this fixture's
    #     60-degree sun) and c_eff (a brightening). Note the whole
    #     in-scatter term on this cell is 74.74/10.09/21.6%, so the change
    #     is under a third of what the feature it corrects contributes.
    #
    #   water_pool_submerged  max  82.46 mean 12.59 bad 23.7% (thr 16/2/1)
    #     The largest move in the matrix, and the fixture explains it: it
    #     is the only cell that sets r_water_scatter_particulate (0.15),
    #     and particulates are exactly where the old a + b extinction was
    #     worst. Petzold scattering is 98.3% forward, so charging the glow
    #     for all of it made this column roughly four times too dark. The
    #     brightening is the correction.
    #
    # AND WHICH PASS DESPITE MOVING -- worth knowing before someone reads a
    # green matrix as "nothing changed": planet_ocean_orbit (mean 1.65
    # against a threshold of 2.0) is the cell #305 was RAISED on, and it
    # moves in the direction the issue asks for. Measured on that frame
    # with the atmosphere removed and the tonemap linear, the off-glitter
    # water goes from (1.51, 1.90, 1.96) B/R 1.30 -- the near-black neutral
    # the issue reports -- to (1.54, 2.11, 2.96) B/R 1.92, and the ADDED
    # radiance alone is (0.027, 0.217, 0.994), i.e. 37:8:1 blue-to-red.
    # water_pool (1.44 against 4.0), planet_underwater_column (1.11 against
    # 2.0), planet_ocean_boat and ocean_fft also move and stay inside.
    #
    # VARIANCE, MEASURED RATHER THAN ASSUMED. The water-leaving term rides
    # the MAT_WATER interface's existing Fresnel roulette, so the obvious
    # worry is that it adds noise at grazing angles where the reflected
    # branch is taken most. It does not: on ocean_foam at the golden's own
    # 120 frames, spp 2 against spp 8 is mean L2 6.85 on this branch and
    # 7.36 on main -- this branch's residual noise is LOWER than main's.
    # (The same pair at 120 vs 240 FRAMES differs by mean 64 on main too:
    # the cascades are animated and the cell is pinned at a sea state, not
    # at a converged image. That comparison measures nothing.)
    #
    # COST, on the dev M4 Max at 512x384, measured as the slope between a
    # 200-frame and a 1200-frame run so process startup cancels:
    #   planet_ocean_orbit (400 km, ocean is most of the frame)
    #     main 18.03 ms/frame, this branch 17.99 ms/frame -- no measurable
    #     cost, because the term reuses the shadow ray the ocean sun NEE
    #     has already fired at that point.
    #   ocean_foam (flat plane, no planetary NEE, so the term fires its own)
    #     main 16.08, gate off 15.98, gate on 16.12 ms/frame -- +0.14 ms,
    #     0.9%, for one shadow ray per water pixel.
    # The submerged march this replaces above the surface costs 16 shadow
    # rays per pixel, which is the whole reason the closed form exists.
    # --- end #305 ---------------------------------------------------------

    # PLANETARY P7 (#261) REGENERATED planet_ocean_snell, AND ONLY IT OF THE
    # THREE. That cell is the matrix's only submerged camera before this
    # phase, so it is the only one the underwater in-scatter march and the
    # refracted sun NEE can reach. MEASURED against the P5 golden it
    # replaces, on the pre-rebase build where both were reproducible:
    # mean-delta 20.72 over 50.27% of pixels, max 140.44. The window gains
    # an in-scattered blue column and a sun that is now integrated at its
    # REFRACTED direction -- before P7 the sun could only be found by a BSDF
    # sample happening to land on a 0.53-degree disc after crossing the
    # interface, so the window was lit but had no sun in it. planet_ocean
    # _boat and planet_ocean_orbit are byte-identical and were not touched:
    # both cameras are in air, and every P7 code path is gated on the
    # medium stack saying otherwise.
    #
    # --- Planetary P7 (#261): underwater -------------------------------
    #
    # Two cells, and they are deliberately not one. #261 changes two things
    # that would be inseparable in a single fixture: it adds the in-scatter
    # march, and it reparameterises r_water_absorption_* onto Pope & Fry
    # 1997's measured pure-water coefficients. A cell that moved for both
    # reasons could attribute neither.
    #
    #   planet_underwater_column  states NO absorption at all, so it renders
    #                             on the new defaults. It is the cell that
    #                             fails if the measurement stops reaching
    #                             the screen. Pure seawater: molecular
    #                             scattering only, near-symmetric
    #                             depolarised-Rayleigh phase.
    #   water_pool_submerged      states water_pool's own (0.45, 0.15, 0.05)
    #                             verbatim, so the defaults cannot move it.
    #                             It is the only cell in the matrix with the
    #                             CAMERA inside the medium and real geometry
    #                             occluding the sun, and the only one that
    #                             turns the Petzold particulate species on.
    #
    # THE VACUITY TRAP IS UNUSUALLY LIVE HERE AND IS HANDLED EXPLICITLY. An
    # underwater frame is already dark, so "the image changed when I deleted
    # the feature" is a test a black frame passes. Both fixtures therefore
    # set an exposure that puts the water column in the MIDDLE of the range
    # (0.30 and 0.20, the latter two stops over water_pool's own 0.05), and
    # every ablation below is checked to be a change in CONTENT: none of
    # them blanks the frame. r_water_inscatter 0 leaves the transmitted sky,
    # Snell's window and every surface exactly where they were and removes
    # only the scattered light; r_water_scatter_particulate 0 leaves the
    # molecular species behind it.
    #
    # DETERMINISM, MEASURED BOTH WAYS, BECAUSE SERIAL IS NOT EVIDENCE. Both
    # cells render byte-identical over twelve consecutive runs AND over
    # twelve run 12-WAY CONCURRENTLY, in Release and in Debug -- 48 renders
    # apiece, one hash each. (Release and Debug agree with each other on
    # planet_underwater_column and not on water_pool_submerged, which is
    # just -ffast-math, and is why goldens carry thresholds rather than
    # being compared as hashes.)
    #
    # THE CONCURRENT HALF IS NOT A FORMALITY, AND THIS PHASE IS THE PROOF.
    # Before rebasing onto #296, planet_underwater_column was 23-of-24
    # identical SERIALLY and gave TEN DISTINCT IMAGES IN TWELVE under 12-way
    # contention, worst pair mean-delta 23.24 over 42.56% of pixels. #261
    # did not create that -- #295 did -- but it is what made it fire
    # reliably, because the in-scatter march is 16 extra shadow rays per
    # pixel and a slower frame lets the CPU run further ahead of the upload
    # ring. planet_ocean_snell went the same way, from 42-of-43 stable to
    # eight distinct images in eight. Post-#296 all three are one hash.
    #
    # THRESHOLDS, MEASURED. Run-to-run on the dev M4 Max is 0.0000 max /
    # 0.000000 mean / 0 bad pixels on both -- bit-identical, twice over --
    # so these are the MATRIX DEFAULTS (16 / 2 / 1) passed explicitly rather
    # than inherited, kept loose for another driver's rounding exactly as
    # every other Metal cell in this file. The feature-deleted margins:
    #
    #   planet_underwater_column
    #     r_water_inscatter 0                    mean   9.33  bad 19.81%
    #     r_water_inscatter_samples 1            mean   9.33  bad 19.83%
    #     legacy (0.45, 0.15, 0.05) absorption   mean  86.13  bad 96.84%
    #     absorption 0 0 0                       mean 145.40  bad 99.98%
    #   water_pool_submerged
    #     r_water_inscatter 0                    mean  17.12  bad 44.07%
    #     r_water_scatter_particulate 0          mean   8.24  bad 17.40%
    #     ambient gate deleted in the shader     mean   5.44  bad  0.75%
    #     r_water_inscatter_samples 1            mean   2.01  bad  0.36%
    #
    # The weakest FAILING row is 2.7x above MEAN_DELTA 2, against a noise
    # floor of exactly zero (ref-vs-ref, twice, on both cells: 0 bad
    # pixels). Re-measure if a fixture's camera, resolution,
    # exposure or frame count changes: the margins are the point, not the
    # numbers.
    #
    # THE AMBIENT-GATE ROW IS CAUGHT BY ONE CRITERION AND NOT THE OTHER,
    # and that is worth saying plainly rather than leaving to be discovered.
    # Deleting the gate raises the whole seabed by a little rather than a
    # few pixels by a lot, so mean-delta 5.44 is 2.7x over MEAN_DELTA while
    # bad-percent 0.75 is UNDER FAIL_PERCENT 1: this cell fails on the mean
    # and would pass on the per-pixel count alone. It also has no cvar --
    # the gate is a correctness fix, not a feature, and putting it behind a
    # switch would invite leaving it off -- so that row was measured by
    # editing the shader, not by --extra, and cannot be re-run from the
    # matrix. It is the reason this fixture's floor sits at 7 m and its
    # camera looks down at it; see the cfg header for that derivation.
    #
    # TWO ROWS ARE CONTROLS AND NOT ABLATIONS. Passing a cvar the fixture
    # already sets to that value -- `r_water_ior 1.33` on the column,
    # `r_water_scatter_particulate 0.15` on the pool -- reproduces the
    # reference BIT-IDENTICALLY, 0 bad pixels. And every ablation row above
    # that DOES move the image is itself the evidence that --extra reached
    # the cvar it named: a silently-dropped override would have given zero
    # there too.
    #
    # WHY samples_1 IS A FAILING ROW ON ONE CELL AND A PASSING ONE ON THE
    # OTHER, since it looks like an inconsistency and is not. The march's
    # QUADRATURE bias is negligible at any sample count -- two midpoint
    # samples hold the smooth integral inside one 8-bit level over the whole
    # 404 m black depth, measured in tests/pt_water_optics_test.cpp. What
    # the sample count buys is VARIANCE, and variance scales with how far
    # the sun's slant transmittance swings across the span. In the 2 m pool
    # it barely swings and N = 1 lands at mean 2.01 / 0.36% bad, only just
    # over MEAN_DELTA; over the column's 30 m it swings by orders of
    # magnitude in red and N = 1 is as far from the reference as deleting the
    # march altogether. That is the honest reason the default is 16 and not
    # 2.
    pt_add_golden_cell(SCENE planet_underwater_column BACKEND metal
                       DENOISER off
                       FRAMES 48 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)
    pt_add_golden_cell(SCENE water_pool_submerged BACKEND metal DENOISER off
                       FRAMES 48 MAX_DELTA 16 MEAN_DELTA 2 FAIL_PERCENT 1)

    # --- The Cox-Munk test (#259's stated acceptance criterion) -----------
    #
    # Render the ocean from 400 km with the FFT on, and again with the FFT
    # replaced by a fixed roughness, and the two must agree. That is the
    # criterion the issue writes down, and it is a real test rather than a
    # restatement because the two frames come from DIFFERENT machinery: one
    # synthesises three band-limited wave fields, ray-marches them, and folds
    # the sub-footprint remainder into the BRDF; the other has no height
    # field at all and carries the whole of Cox & Munk's measured slope
    # variance in the microfacet lobe. They agree only if the handover
    # conserves what it claims to.
    #
    # THE FIXED ROUGHNESS IS NOT A NUMBER SOMEONE TYPED. #259 quotes 0.18;
    # this build converges to 0.5038, and the difference is derived rather
    # than tuned -- Beckmann's slope distribution has TOTAL mean-square slope
    # alpha^2, so alpha = sigma = sqrt(0.003 + 0.00512 * 12) = 0.2539, and
    # PathTrace.slang's anisoAlpha squares the material roughness to get
    # alpha, so the roughness is sqrt(0.2539). 0.18 is sqrt(sigma^2 / 2),
    # the per-axis slope RMS, which is the right answer to a different
    # question. See src/physics/OceanFFT.h and the first case in
    # tests/pt_planet_ocean_test.cpp, which pins every step of it.
    #
    # There is no roughness cvar involved: r_ocean 0 turns the solver off,
    # the cascades then resolve nothing, and oceanBrdfAlpha2's conservation
    # law hands the BRDF the whole constant on its own. The control is the
    # SAME surface with its waves moved into the lobe, which is the only
    # form of this test that means anything.
    #
    # MEASURED: max 5.1962, mean 0.384632, 0.000% bad at MAX_DELTA 16 -- on
    # a frame whose own glitter patch spans 156 levels. The registered bar
    # is the matrix default, which the pair clears by 3.1x on max and 5.2x
    # on mean. The same pair with the roughness fold removed (the version of
    # this code that read Cox-Munk out of the solver instead of out of the
    # cvar, so r_ocean 0 fell back to a MIRROR) measured max 156.7 and mean
    # 63.0 -- 31x the threshold. The test has signal.
    #
    # The vacuity guard is that planet_ocean_orbit is ALSO pinned against a
    # committed PNG above: two black frames would agree too, so "they match"
    # and "they show an ocean" are separate assertions and neither stands in
    # for the other.
    if(APPLE)
        set(_p5_scenes ${CMAKE_SOURCE_DIR}/tests/goldens/scenes)
        set(_p5_cell planet_ocean_coxmunk__metal)
        foreach(_half a b)
            if(_half STREQUAL "a")
                set(_p5_scene planet_ocean_orbit)
            else()
                set(_p5_scene planet_ocean_orbit_smooth)
            endif()
            set(_p5_png ${CMAKE_BINARY_DIR}/golden_actual/${_p5_cell}__${_half}.png)
            set(_p5_wd  ${CMAKE_BINARY_DIR}/golden_workdir/${_p5_cell}__${_half})
            file(MAKE_DIRECTORY ${_p5_wd})
            add_test(NAME ${_p5_cell}_render_${_half}
                COMMAND $<TARGET_FILE:pt_render_one_frame>
                    --scene    ${_p5_scenes}/${_p5_scene}.cfg
                    --backend  metal
                    --denoiser off
                    --frames   48
                    --out      ${_p5_png}
                WORKING_DIRECTORY ${_p5_wd})
            set_tests_properties(${_p5_cell}_render_${_half} PROPERTIES
                TIMEOUT 300
                LABELS  "golden;golden_metal;planetary")
        endforeach()
        add_test(NAME ${_p5_cell}_diff
            COMMAND $<TARGET_FILE:imgdiff>
                ${CMAKE_BINARY_DIR}/golden_actual/${_p5_cell}__a.png
                ${CMAKE_BINARY_DIR}/golden_actual/${_p5_cell}__b.png
                --max-delta    16
                --mean-delta   2
                --fail-percent 1
                --diff         ${CMAKE_BINARY_DIR}/golden_diff/${_p5_cell}.png
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(${_p5_cell}_diff PROPERTIES
            TIMEOUT 30
            DEPENDS "${_p5_cell}_render_a;${_p5_cell}_render_b"
            LABELS  "golden;golden_metal;planetary")
    endif()
    # --- end Planetary P5 -------------------------------------------------

    # --- Physics Phase 2a rigid-body fixtures (#138, #181) ----------------
    # Pins the rigid-body fixture output against a committed PNG so any
    # accidental regression in:
    #   * SeedDefaultPrimitives -> phys_drop_sphere / phys_drop_box wiring
    #   * the integrator's orientation predictor + finite-diff omega
    #   * bounding-sphere collision against the ground plane
    #   * the #181 colored-material plumbing (per-body RGB args)
    #   * pt_smoke_skip_csg_seed / pt_smoke_skip_prim_seed gating
    # surfaces as an imgdiff failure rather than going unnoticed until
    # someone notices the demo screenshot is wrong.
    #
    # phys_rb_smoke:  small mix of spheres + boxes above the default
    #                 cornell_csg seed cluster -- exercises the rigid-body
    #                 path WITH the default scene present.
    # phys_rb_demo:   tower-of-spheres + row-of-boxes on a CLEAN stage
    #                 (cornell csg + analytic prim seeds disabled). This
    #                 is the user-facing "show me the rigid bodies" demo
    #                 fixture from #181, so it doubles as visual coverage
    #                 of `pt_smoke_skip_csg_seed` / `pt_smoke_skip_prim_seed`.
    #
    # Both scenes set `pt_smoke_late_exec` to a *relative* path
    # (tests/goldens/scenes/phys_rb_*_late.cfg). At engine init that path
    # is resolved against the engine's CWD -- but golden cells run each
    # render in its own per-cell workdir (`golden_workdir/<cell>/`), so a
    # relative path can't see the source tree. EXTRA below re-sets the
    # cvar to the absolute path AFTER the scene body inlines (the
    # synthesised smoke-exec cfg appends --extra below the scene body, so
    # the absolute value wins the last-writer race before Init's
    # end-of-Init late-exec reader fires). Mutating the scene cfgs to
    # hold absolute paths isn't an option because regen-time those paths
    # would differ across hosts.
    #
    # Only the metal/off cells register today -- no software/off goldens
    # were captured tonight. Add software cells here when those PNGs land
    # under tests/goldens/Darwin/.
    #
    # FRAMES 60: both fixtures' docstrings (and their PR commit messages
    # PR #184 / PR #189) describe the canonical capture as a "frame 60"
    # screenshot -- the late-exec heights are tuned to catch the fall
    # mid-flight at ~60 render frames.
    #
    # Tolerances: smoke-mode dt is pinned to 1/60 (commit 5d57b29) and the
    # sims are frozen during un-counted loading frames, so body positions
    # at frame 60 are DETERMINISTIC -- measured bit-identical run-to-run
    # (max_delta 0.0) on the dev M4 Max. The old 64/16/10 thresholds were
    # sized for wall-clock-dt drift that no longer exists, and were loose
    # enough to hide the very regression the list below claims to catch.
    # Measured by actually re-rendering the fixture with the RGB args
    # stripped (bodies fall back to their default tints): mean delta 2.97,
    # bad pixels 2.49% -- imgdiff PASSES under 64/16/10 (Passes() requires
    # mean <= MEAN_DELTA and bad% <= FAIL_PERCENT, and both cleared). The
    # five colored bodies simply cover too little of the frame to move a
    # whole-image mean. Tightened to the sdf_smin_row tolerances (32/4/2),
    # under which that same regressed render FAILS on the bad-pixel gate
    # (2.67% > 2%), while the real capture is bit-identical to the golden
    # (max_delta 0.0). Remaining headroom absorbs cross-host FP ordering.
    #
    # These cells catch:
    #   * Complete render failure (black / uniform output)
    #   * Bodies disappearing entirely
    #   * Material regressions on phys_rb_smoke (RGB args silently
    #     ignored -> bodies render in the default tint instead of the
    #     magenta/gold/green hues its late-exec specifies)
    # NOTE: phys_rb_demo_late.cfg spawns its bodies WITHOUT RGB args, so
    # the color-regression coverage above belongs to phys_rb_smoke only.
    # Neither cell catches subtle physics bugs that shift body positions
    # by a few cm; a structural diff (body-count + albedo histogram) is
    # the right tool for that and is left as future work.
    pt_add_golden_cell(SCENE phys_rb_smoke BACKEND metal DENOISER off
        FRAMES 60
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2
        EXTRA "pt_smoke_late_exec ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/phys_rb_smoke_late.cfg")
    pt_add_golden_cell(SCENE phys_rb_demo  BACKEND metal DENOISER off
        FRAMES 60
        MAX_DELTA 64 MEAN_DELTA 12 FAIL_PERCENT 5
        EXTRA "pt_smoke_late_exec ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/phys_rb_demo_late.cfg")
    # --- end Physics Phase 2a rigid-body fixtures -------------------------

    # --- Mesh motion blur (wave-7 #21, follow-up to PR #85) ---------------
    # motion_blur_mesh fixture exercises PathTrace.slang's mesh ray-shift
    # path: the CSG mesh is translated by one canonical frame of motion
    # (2 m/s × 1/60 s = 0.033 m in +X) BEFORE the render, so the shader's
    # mesh_motion_params.w gate fires and stochastic shutter-time samples
    # smear the mesh from origin to +0.033 m. The result is a streaked
    # CSG silhouette mirroring the analytic-prim streaks PR #85 shipped.
    #
    # Tolerances: mesh motion blur amplifies per-ray variance because
    # every primary ray is now sampled at a different shutter time
    # within the prev->curr window. The default cornell_csg (16/2/1)
    # is too tight at r_spp=1; tolerances here are sized empirically so
    # any non-trivial regression in the mesh shift path surfaces while
    # the stochastic noise floor stays within budget.
    #
    # software + metal cells register today on macOS. Vulkan validation
    # belongs on a Windows/Linux Vulkan host after the integration
    # branch lands.
    # The fixture's pt_smoke_late_exec value is a *relative* path
    # (tests/goldens/scenes/motion_blur_mesh_late.cfg). At engine init
    # that path is resolved against the engine's CWD -- but golden
    # cells run each render in its own per-cell workdir
    # (golden_workdir/<cell>/), so a relative path can't see the source
    # tree. EXTRA below re-sets the cvar to the absolute path AFTER the
    # scene body inlines (the synthesised smoke-exec cfg appends --extra
    # below the scene body, so the absolute value wins the last-writer
    # race before Init's end-of-Init late-exec reader fires). Mirrors
    # the phys_rb_smoke wiring.
    pt_add_golden_cell(SCENE motion_blur_mesh BACKEND software DENOISER off
        MAX_DELTA 64 MEAN_DELTA 8 FAIL_PERCENT 5
        EXTRA "pt_smoke_late_exec ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/motion_blur_mesh_late.cfg")
    pt_add_golden_cell(SCENE motion_blur_mesh BACKEND metal    DENOISER off
        MAX_DELTA 64 MEAN_DELTA 8 FAIL_PERCENT 5
        EXTRA "pt_smoke_late_exec ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/motion_blur_mesh_late.cfg")
    # --- end Mesh motion blur fixture -------------------------------------

    # --- Wave 8 spectral dispersion (#27) ---------------------------------
    # Glass-sphere prism with r_spectral 1: per-ray hero-wavelength
    # sampling refracts each wavelength by its own Cauchy IOR, producing
    # chromatic fringing on the glass + a dispersed (rainbow) caustic.
    # Metal golden cell (the acceptance target). Tolerances are widened
    # like the SDF / smoke cells because the hero-wavelength estimator
    # is a stochastic single-wavelength-per-path integrator -- its 1-spp
    # noise floor over 64 accumulated frames is higher than the
    # achromatic dielectric path, and the per-channel spectral weight
    # amplifies tiny cross-backend FP-order deltas. The cell guards the
    # dispersion pipeline end-to-end (Cauchy IOR + CIE reconstruction);
    # the achromatic-glass regression is already covered bit-for-bit by
    # the cornell_csg cells (r_spectral defaults to 0).
    pt_add_golden_cell(SCENE spectral_prism BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # --- end Wave 8 spectral dispersion -----------------------------------

    # --- Env double-count through dielectrics (2026-07-13 audit, B9) -------
    # env_glass_furnace: a Lambert floor + two furnace-reference spheres
    # under an INFINITE dielectric slab, lit by a uniform white-furnace IBL
    # (assets/hdri/uniform_gray.hdr). Pins the fix for the env-map NEE / MIS
    # double count: with caustics on, a Lambert vertex's env-NEE shadow ray
    # refracts through the slab (transmittance() tint != 0) AND its BSDF
    # bounce refracts through the same slab to the sky -- the MAT_DIELECTRIC
    # branch used to reset skip_sky_on_miss / prev_was_lambert_nee after the
    # first refraction, so the sky-on-miss was added at full weight on top of
    # the NEE. The under-slab floor converged ~1.5x (uniform env; up to ~2x
    # under a concentrated HDRI) brighter than energy conservation allows.
    # The fix persists those flags across the refraction chain (reset only on
    # reflection/TIR, the lobe transmittance() never follows).
    #
    # Energy-conservation witness: under the fix the under-slab floor is
    # ~0.98x the no-slab floor (Fresnel loss dims it slightly, as glass must)
    # instead of >1.0x. The scene is bit-deterministic (analytic prims only,
    # 0 BVH nodes), and reintroducing the bug moves ~99% of pixels by a mean
    # ~23 L2 -- far past the default 16/2/1 tolerances, so this is a tight
    # regression pin. Metal-only: the CPU SoftwareTracer is a single-bounce
    # direct-lighting renderer with no env-NEE / MIS / refractive shadow rays,
    # so it can neither exhibit nor guard this bug.
    pt_add_golden_cell(SCENE env_glass_furnace BACKEND metal DENOISER off)
    # --- end env double-count through dielectrics -------------------------

    # --- Wave 8 thin-lens depth of field (#27) ----------------------------
    # Row of emissive spheres receding from the camera; only the middle
    # sphere sits on the focus plane (sharp), the rest blur into bokeh
    # discs with strength proportional to their distance from focus.
    # Metal golden cell. The fixture sets photographic params (85mm
    # f/2.0, focus 7.23 m) and r_spp 8 because thin-lens DOF integrates
    # over the aperture disc -- each sample lands on a different lens
    # point, so the defocus regions need more samples per frame to
    # settle. Tolerances widened like the other stochastic cells; the
    # pinhole regression (r_dof 0 -> every sphere sharp) is the default
    # path already covered bit-for-bit by cornell_csg / the other cells.
    pt_add_golden_cell(SCENE dof_bokeh BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # --- end Wave 8 thin-lens depth of field ------------------------------

    # --- Wave 8 PBR (#26) -- textured-material fixture --------------------
    # pbr_textured exercises the PBR texture path: an analytic sphere +
    # ground plane wearing albedo (sRGB checkerboard) + normal (tangent-
    # space bumps) + roughness (gradient) maps from assets/textures/. The
    # fixture loads the committed test PNGs via prim_set_*_tex, so a clean
    # checkout reproduces it. Covers procedural UVs (sphere lat/long,
    # plane world-XZ), strip-atlas bilinear sampling + per-tile wrap, the
    # sRGB-decode-only-on-albedo split, and normal-map perturbation.
    #
    # Tolerances: textured surfaces add per-pixel detail that exaggerates
    # cross-backend FP-order variance vs the flat cornell_csg cell (the
    # checkerboard edges + normal-map bumps are high-frequency), so the
    # thresholds match the sdf_smin_row cell (32/4/2) rather than the
    # tight default. software + metal + vulkan all register; CI runs the
    # software cell (label golden_software). The committed flat-color
    # goldens (cornell_csg, sky, smoke, water) are UNAFFECTED -- materials
    # with no texture index render bit-for-bit as before.
    #
    # Goldens: software + metal committed 2026-07-14 (audit A7 follow-up --
    # the software backend now samples the atlas in SoftwareTracer.cpp, so
    # both the CI-run software cell and the metal cell pin real textured
    # frames instead of the previous missing-golden WILL_FAIL placeholders).
    # REGENERATED 2026-07-14 (mipmapping follow-up): the atlas now carries a
    # box-filtered mip chain and both backends select a ray-cone LOD, so the
    # receding-checkerboard horizon resolves to the averaged tile colour
    # instead of aliasing into noise. This cell is the visual regression pin
    # for that fix -- the far field should stay smooth, not sparkle.
    # VULKAN GOLDEN DEFERRED: not captured here because macOS registers no
    # Vulkan cell (this is the wave-8 dev host) and Mac-Vulkan pixel-
    # correctness is still unverified (#71). On a Vulkan CI host the cell
    # below stays a missing-golden placeholder until #71 validates the
    # MoltenVK/RTX path and pbr_textured__vulkan__off.png is captured +
    # committed from that host. See project_full_audit_paused.md (feature
    # "material-editor-and-textures").
    pt_add_golden_cell(SCENE pbr_textured BACKEND software DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE pbr_textured BACKEND metal    DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # Vulkan cell: golden deferred to #71 (Mac-Vulkan pixel correctness) --
    # registers on Vulkan hosts, no-op on macOS. See the note above.
    pt_add_golden_cell(SCENE pbr_textured BACKEND vulkan   DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end Wave 8 PBR fixture -------------------------------------------

    # --- Wave 9 volumetric height fog -------------------------------------
    # height_fog exercises the dedicated HeightFog.slang composite: a row
    # of Lambert spheres receding down -Z under a procedural low-sun sky,
    # with r_fog 1. The analytic exponential-height fog integrates along
    # each camera ray so the near spheres stay clear, the far ones fade
    # into the fog colour, the ground hazes to the horizon, and the
    # sun-side sky brightens via the HG forward-scatter lobe. The cell
    # guards the fog pipeline end-to-end (closed-form optical depth +
    # Beer-Lambert composite + camera-ray reconstruction from the depth
    # G-buffer).
    #
    # Metal-only -- Metal is the wave-9 build+test target (the fog
    # dispatch is gated to the Metal `use_engine_tonemap` post-denoise
    # chain; Vulkan descriptor wiring is a follow-up). The r_fog 0
    # default path is a true no-op (dispatch elided) and is already
    # covered bit-for-bit by every other committed golden, so no extra
    # "fog off" cell is needed for the goldens-preserved half of the
    # acceptance bar.
    #
    # Tolerances widened like the other procedural-sky cells: the fog
    # in-scatter is a smooth low-frequency term layered over the 1-spp
    # path-traced scene, so the dominant per-pixel variance is the
    # underlying radiance noise floor (same regime as procedural_evening)
    # rather than the fog math, which is deterministic. The thresholds
    # catch the fog failing to engage (haze-free frame), the depth
    # reconstruction breaking (uniform fog ignoring distance), or the
    # composite blowing out / zeroing the scene.
    pt_add_golden_cell(SCENE height_fog BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 48 MEAN_DELTA 8 FAIL_PERCENT 6)
    # --- end Wave 9 volumetric height fog ---------------------------------

    # --- Wave 9 advanced materials (#wave9) ------------------------------
    # Three fixtures exercising the new BSDF lobe extensions, all routed
    # through the push-constant material-override array (prim storage stride
    # untouched). Metal is the build + acceptance target. Tolerances match
    # the other stochastic specular cells (spectral / dof: 48/6/4) because
    # the VNDF-sampled anisotropic + clearcoat lobes and the random-walk SSS
    # have a higher 1-spp noise floor than flat cornell_csg, and the
    # highlight streak / coat gloss are high-frequency features that
    # exaggerate cross-backend FP-order deltas. The committed flat-material
    # goldens (cornell_csg, sky, smoke, water, pbr_textured) are UNAFFECTED:
    # a prim with no mat_ext_flags never appears in the override array, so
    # the per-hit lookup short-circuits and the base material renders
    # bit-for-bit.
    #
    # brushed_metal: anisotropic GGX metal sphere -- the sun's specular
    # highlight stretches into a brushed streak (vs an isotropic dot).
    pt_add_golden_cell(SCENE brushed_metal BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # car_paint: red Lambert pigment + clearcoat -- a crisp white gloss
    # highlight from the clear lacquer floats on the saturated red body.
    pt_add_golden_cell(SCENE car_paint BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # subsurface: backlit translucent wax sphere -- the bounded random-walk
    # SSS transmits warm light through the thin rim. This is the highest-
    # variance estimator of the three (a multi-step random walk per path),
    # so the tolerances are widened a notch beyond the specular cells.
    pt_add_golden_cell(SCENE subsurface BACKEND metal DENOISER off
        MAX_DELTA 64 MEAN_DELTA 8 FAIL_PERCENT 6)
    # --- end Wave 9 advanced materials -----------------------------------

    # --- Wave 9 tone-mapping operators (#27 follow-up) --------------------
    # AgX + Khronos PBR Neutral pinned on a daytime (procedural_noon) and a
    # night (bsc_night) scene so each new operator is regression-locked. The
    # ACES default needs no new cell -- the existing procedural_noon /
    # bsc_night `software off` goldens already pin op 0 and stay bit-for-bit
    # unchanged (op 0 collapses to the previous unconditional ACES path on
    # every backend). The EXTRA cvar selects the operator; the TAG suffix
    # disambiguates the cell name + golden filename from the ACES cell that
    # shares the same scene/backend/denoiser triple.
    #
    # Backend: software / off. The operator is a pure post-exposure pixel
    # transform applied identically on every backend (the engine packs
    # r_tonemap_op into PtPush::tonemap_params for the GPU paths and
    # SoftwareTracer reads the same field), so the software backend -- the
    # deterministic CI lane (label golden_software) -- is the right place to
    # pin the operator math. Default tolerances (16/2/1): the operator curve
    # is deterministic given the seed-pinned scene, so the only variance is
    # the path tracer's accumulator noise floor, same as the sibling ACES
    # cell. Metal/Vulkan apply the identical curve; a dedicated GPU cell can
    # be added later if cross-backend operator drift ever needs pinning.
    pt_add_golden_cell(SCENE procedural_noon BACKEND software DENOISER off
        TAG agx     EXTRA "r_tonemap_op agx")
    pt_add_golden_cell(SCENE procedural_noon BACKEND software DENOISER off
        TAG khronos EXTRA "r_tonemap_op khronos_pbr_neutral")
    pt_add_golden_cell(SCENE bsc_night       BACKEND software DENOISER off
        TAG agx     EXTRA "r_tonemap_op agx")
    pt_add_golden_cell(SCENE bsc_night       BACKEND software DENOISER off
        TAG khronos EXTRA "r_tonemap_op khronos_pbr_neutral")
    # --- end Wave 9 tone-mapping operators --------------------------------

    # --- Wave 10 bloom + lens dirt ----------------------------------------
    # A cluster of bright emissive spheres against a near-black sky drives
    # the energy-preserving dual-filter bloom pyramid hard; the procedural
    # lens-dirt overlay (r_lens_dirt 1) modulates the resulting halos
    # through a screen-space grime field. The cell guards the bloom
    # composite end-to-end: the pyramid build (13-tap Karis downsample +
    # tent upsample), the r_bloom_strength gain, and the dirt modulation
    # in Tonemap.slang.
    #
    # Metal-only: the bloom composite runs in the engine's Tonemap.slang
    # dispatch (the use_engine_tonemap branch), which is Metal's post-
    # process path; the bloom-off default (r_bloom 0 -> halo-free frame)
    # is already covered bit-for-bit by every other committed golden, and
    # the clean-lens default (r_lens_dirt 0) collapses the composite to
    # the pre-wave-10 `bloom_intensity * bloom` add, so no extra "dirt
    # off" cell is needed for the goldens-preserved half of the bar.
    #
    # Tolerances widened like the other stochastic Metal cells: the bloom
    # halo is a smooth low-frequency layer over the 4-spp path-traced
    # emissive cores, so the dominant per-pixel variance is the underlying
    # radiance noise floor rather than the bloom math (deterministic given
    # the seed). The thresholds catch bloom failing to engage (no halos),
    # the strength gain being dropped (dim halos), or the dirt modulation
    # breaking (uniform or absent grime).
    pt_add_golden_cell(SCENE bloom_lensdirt BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # --- end Wave 10 bloom + lens dirt ------------------------------------

    # --- Wave 10 polygonal bokeh ------------------------------------------
    # Same receding row of emissive spheres as the dof_bokeh cell, but with
    # a HEXAGONAL aperture (r_dof_blades 6) plus a blade rotation and
    # cat's-eye edge vignetting. The out-of-focus highlights render as
    # six-sided bokeh polygons (squeezed into lemons toward the frame edge)
    # instead of round discs. The cell guards the polygonal-iris path: the
    # concentric-disk-to-N-gon warp, the blade rotation, and the cat's-eye
    # off-axis squeeze in PathTrace.slang's camera-ray block.
    #
    # Metal golden cell. The circular-bokeh regression (r_dof_blades 0 +
    # r_dof_cateye 0 -> round discs) is the default aperture path already
    # pinned bit-for-bit by the dof_bokeh cell, so no extra "round" cell is
    # needed for the goldens-preserved half of the bar. Tolerances match
    # the dof_bokeh cell (the scene is identical apart from the aperture
    # shape, so the stochastic aperture-integration noise floor is the
    # same regime).
    pt_add_golden_cell(SCENE dof_bokeh_blades BACKEND metal DENOISER off
        MAX_DELTA 48 MEAN_DELTA 6 FAIL_PERCENT 4)
    # --- end Wave 10 polygonal bokeh --------------------------------------

    # --- SIGMA sun-shadow demod (issue #115, B10) -------------------------
    # A procedural-sun shadow across a Lambert ground with a POINT LIGHT
    # sitting inside the sun umbra. This is the cell the demod bug shipped
    # without: before B10, SigmaShadow.slang composited `hdr *= vis`,
    # multiplying the WHOLE post-denoise HDR -- point light + indirect GI
    # included -- by the sun visibility, so the shadow interior rendered
    # pure black. The fix demodulates only the direct sun (PathTrace
    # subtracts it from the denoiser input; SigmaShadow re-adds S *
    # vis_denoised), so the umbra keeps its point-light pool + indirect
    # fill while the sun-shadow edge stays crisp.
    #
    # Metal + svgf_atrous only: SIGMA is Metal-only and the demod path
    # only runs with a denoiser active (r_shadow_demod gate). The cell is
    # a sensitive regression guard -- reverting to the whole-image multiply
    # blacks out the entire umbra (the ~40% of the frame in shadow),
    # far past these thresholds. Deterministic (no meshes -> no TLAS build
    # entropy); default-ish tolerances with a little headroom for Apple
    # GPU FP-order variance.
    pt_add_golden_cell(SCENE shadow_umbra_point BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    # --- end SIGMA sun-shadow demod ---------------------------------------

    # --- Planetary P1 (#255): camera-anchored render frame ----------------
    # These cells are SELF-VALIDATING: they diff one render against
    # another render, not against a committed golden. That is deliberate
    # and it is what makes them meaningful.
    #
    # The claim under test is an INVARIANCE -- "the renderer's output does
    # not depend on where in canonical space the scene sits" -- and an
    # invariance is naturally stated as an equality between two runs. A
    # committed PNG could only ever pin one of the two sides, would need
    # regenerating whenever anything else in the tracer legitimately
    # moved, and would say nothing about the other side agreeing with it.
    # Diffing the pair says exactly the thing the phase promises, needs no
    # new binary in the tree, and cannot silently rot.
    #
    # Three fixtures, one arrangement of geometry:
    #   worldframe_origin         at the world origin  (anchor == 0, the
    #                             legacy path)
    #   worldframe_ecef           offset +6,370,304 m in y -- Earth radius
    #                             to within 705 m, and exactly 6221 anchor
    #                             lattice steps, so the anchor equals the
    #                             offset and the render frame coincides
    #                             with the origin fixture's
    #   worldframe_ecef_rebased   one further lattice step out (+1024 m),
    #                             so it computes a DIFFERENT anchor -- the
    #                             state a camera lands in the instant it
    #                             crosses a lattice boundary in flight
    #
    # equivalence: origin vs ecef        -- the phase's headline.
    # rebase:      ecef   vs ecef_rebased -- anchor-cell invariance, i.e.
    #              nothing was packed in one anchor and read in another.
    #
    # --max-delta 0 on the software backend: the CPU reference tracer is
    # deterministic, both sides consume the same pinned PRNG stream, and
    # with the anchor equal to the offset the two render frames are
    # bit-for-bit identical inputs -- so anything other than an exactly
    # equal image is a real leak. Metal gets the same pair at a small
    # tolerance rather than zero: its cells are known to move by ULP-scale
    # amounts from codegen alone (#263), and these fixtures carry a CSG
    # mesh whose hardware BLAS build is not bit-reproducible run to run.
    #
    # The mesh is in the picture on purpose. It is the only thing in the
    # engine that reaches the GPU through a TLAS instance transform, which
    # is where the anchor rides -- so these cells also cover the claim
    # that a rebase is one instance rewrite and touches no BLAS.
    #
    # _worldframe_pair(<name> <backend> <scene_a> <scene_b> <max_delta>)
    function(pt_add_worldframe_pair _name _backend _scene_a _scene_b _max_delta)
        if(_backend STREQUAL "metal" AND NOT APPLE)
            return()
        endif()
        if(_backend STREQUAL "software" AND NOT (APPLE OR WIN32))
            return()
        endif()
        set(_cell worldframe_${_name}__${_backend})
        set(_scenes ${CMAKE_SOURCE_DIR}/tests/goldens/scenes)
        foreach(_half a b)
            if(_half STREQUAL "a")
                set(_scene ${_scene_a})
            else()
                set(_scene ${_scene_b})
            endif()
            set(_png ${CMAKE_BINARY_DIR}/golden_actual/${_cell}__${_half}.png)
            set(_wd  ${CMAKE_BINARY_DIR}/golden_workdir/${_cell}__${_half})
            file(MAKE_DIRECTORY ${_wd})
            set(_args
                --scene    ${_scenes}/${_scene}.cfg
                --backend  ${_backend}
                --denoiser off
                --frames   32
                --out      ${_png})
            # The ECEF halves move the CSG mesh with a mesh_step, which
            # only exists after RegisterCsgCommands -- hence the late-exec
            # file, with an absolute path because each cell renders in its
            # own workdir. Mirrors the motion_blur_mesh wiring.
            if(EXISTS ${_scenes}/${_scene}_late.cfg)
                list(APPEND _args --extra
                     "pt_smoke_late_exec ${_scenes}/${_scene}_late.cfg")
            endif()
            add_test(NAME ${_cell}_render_${_half}
                COMMAND $<TARGET_FILE:pt_render_one_frame> ${_args}
                WORKING_DIRECTORY ${_wd})
            set_tests_properties(${_cell}_render_${_half} PROPERTIES
                TIMEOUT 300
                LABELS  "golden;golden_${_backend};worldframe")
        endforeach()
        add_test(NAME ${_cell}_diff
            COMMAND $<TARGET_FILE:imgdiff>
                ${CMAKE_BINARY_DIR}/golden_actual/${_cell}__a.png
                ${CMAKE_BINARY_DIR}/golden_actual/${_cell}__b.png
                --max-delta    ${_max_delta}
                --mean-delta   0
                --fail-percent 0
                --diff         ${CMAKE_BINARY_DIR}/golden_diff/${_cell}.png
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(${_cell}_diff PROPERTIES
            TIMEOUT 30
            DEPENDS "${_cell}_render_a;${_cell}_render_b"
            LABELS  "golden;golden_${_backend};worldframe")
    endfunction()

    pt_add_worldframe_pair(equivalence software
        worldframe_origin worldframe_ecef 0)
    pt_add_worldframe_pair(rebase software
        worldframe_ecef   worldframe_ecef_rebased 0)
    # Metal measures 0 bad pixels on both pairs on an M4 Max, and the
    # fixture is bit-reproducible run-to-run there (verified: same
    # fixture rendered twice, max L2 0.0). The registered bar is
    # --max-delta 2 rather than 0 purely so a stray ULP on a CI runner's
    # paravirtualised GPU cannot flake the cell; --fail-percent 0 keeps
    # it strict, since a real coordinate leak shows up as hundreds of L2
    # across a fifth of the frame, not as a 2/255 wobble.
    pt_add_worldframe_pair(equivalence metal
        worldframe_origin worldframe_ecef 2)
    pt_add_worldframe_pair(rebase metal
        worldframe_ecef   worldframe_ecef_rebased 2)

    # A/B the disable switch. r_origin_rebase_radius 0 pins the anchor to
    # the origin forever -- the permanent legacy path -- so on a scene
    # that never leaves the origin anyway it must produce the SAME IMAGE
    # as the default radius, bit for bit. That is the whole claim: the
    # cvar is an A/B switch and not a second renderer.
    #
    # Diffed against the untagged cell's own actual, not against the
    # committed golden. Both sides are then rendered by this build on
    # this host in this run, so the cell measures exactly the cvar and
    # nothing else; comparing to the committed PNG would fold in
    # whatever host-to-host drift already separates this machine from
    # the one the golden was captured on (measured: 2 pixels at L2 1 on
    # an M4 Max, present on main too), and that has nothing to do with
    # the switch under test. The committed-golden comparison is still
    # made -- by golden_cornell_csg__software__off_diff, which is the
    # cell that owns it.
    if(APPLE OR WIN32)
        set(_rb_png ${CMAKE_BINARY_DIR}/golden_actual/cornell_csg__software__off__rebaseoff.png)
        set(_rb_wd  ${CMAKE_BINARY_DIR}/golden_workdir/golden_cornell_csg__software__off__rebaseoff)
        file(MAKE_DIRECTORY ${_rb_wd})
        add_test(NAME golden_cornell_csg__software__off__rebaseoff_render
            COMMAND $<TARGET_FILE:pt_render_one_frame>
                --scene    ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/cornell_csg.cfg
                --backend  software
                --denoiser off
                --frames   64
                --out      ${_rb_png}
                --extra    "r_origin_rebase_radius 0"
            WORKING_DIRECTORY ${_rb_wd})
        set_tests_properties(golden_cornell_csg__software__off__rebaseoff_render
            PROPERTIES TIMEOUT 300 LABELS "golden;golden_software;worldframe")
        add_test(NAME golden_cornell_csg__software__off__rebaseoff_diff
            COMMAND $<TARGET_FILE:imgdiff>
                ${_rb_png}
                ${CMAKE_BINARY_DIR}/golden_actual/cornell_csg__software__off.png
                --max-delta 0 --mean-delta 0 --fail-percent 0
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(golden_cornell_csg__software__off__rebaseoff_diff
            PROPERTIES TIMEOUT 30
            DEPENDS "golden_cornell_csg__software__off__rebaseoff_render;golden_cornell_csg__software__off_render"
            LABELS  "golden;golden_software;worldframe")
    endif()
    # --- end Planetary P1 -------------------------------------------------

    # --- Planetary P6 (#260): leaving the planet ---------------------------
    #
    # The phase's headline is that one camera path runs from the surface to
    # 400 km with nothing switching over on the way. That is a claim about a
    # SEQUENCE, and the comment at the top of this file about what a golden
    # PNG can and cannot settle applies to it in full: three stored frames
    # cannot show that the frames BETWEEN them are continuous. So the cells
    # below are split by what they can each actually establish.
    #
    #   orbital_limb            what the frame that used to be a black disc
    #   orbital_terminator      looks like now, and that a lit limb and a
    #   sky_physical_ground     dark side coexist, and that the same code
    #                           with no altitude branch in it still makes a
    #                           daytime sky from eye height. Stored PNGs,
    #                           because "what it looks like" is what a
    #                           golden is for.
    #
    #   sky_continuity_shell    that the ONE altitude where the code changes
    #                           behaviour is not a step. Not a stored PNG --
    #                           an A/B against a control, see below.
    #
    # The rigorous global continuity proof is neither: it is
    # pt_atmosphere's "the sky is continuous in altitude from the ground to
    # orbit", which refines a 256-rung altitude ladder to 512 and 1024 and
    # requires the largest adjacent step to halve each time. A Lipschitz
    # function does that; a function with a jump plateaus instead. No
    # tolerance has to be chosen to tell them apart, which is why that
    # claim lives in a unit test and not here.
    #
    # ALL THREE GOLDEN CELLS ARE METAL-ONLY, DELIBERATELY.
    # src/rhi_software/SoftwareTracer.cpp is a hand-written preview tracer,
    # not a Slang-to-CPU emit -- it carries its own simple ProcSky gradient
    # with no atmospheric transmittance at all, and knows nothing about
    # r_sky_mode physical. A software cell here would pin the software
    # tracer's painted sky and would say nothing whatever about the shader
    # this phase changed. Vulkan is unregistered for the reason recorded in
    # the risks: no part of this arc has ever executed on SPIR-V, and the
    # first *__vulkan__off_render cell already sits at ~115 s against a
    # 120 s budget on a cold MoltenVK pipeline cache.
    # #280 measured the three against their own smallest ablation and
    # tightened all of them. orbital_limb is the notable one: only 31% of
    # its frame is not vacuum, so a mean over the WHOLE frame divides any
    # real signal by three before the threshold ever sees it -- 3.46 for
    # ozone off, against a MEAN_DELTA of 4 that could not have failed.
    #   orbital_limb        smallest signal 3.46 (ozone off) -> 4.3x clear
    #   orbital_terminator  smallest signal 15.25 (ozone off) -> 5.1x clear
    #   sky_physical_ground smallest signal 4.92 (ozone off) -> 4.1x clear
    pt_add_golden_cell(SCENE orbital_limb       BACKEND metal DENOISER off
        FRAMES 32 MAX_DELTA 32 MEAN_DELTA 0.8 FAIL_PERCENT 3)
    pt_add_golden_cell(SCENE orbital_terminator BACKEND metal DENOISER off
        FRAMES 32 MAX_DELTA 32 MEAN_DELTA 3 FAIL_PERCENT 3)
    pt_add_golden_cell(SCENE sky_physical_ground BACKEND metal DENOISER off
        FRAMES 32 MAX_DELTA 32 MEAN_DELTA 1.2 FAIL_PERCENT 3)

    # The shell crossing, as an A/B with its own control.
    #
    # A discontinuity is a change that survives refinement, so if the image
    # is a continuous function of altitude then two frames one metre apart
    # are the same frame -- at every altitude, however the renderer is put
    # together. R_top = r_planet_radius + 100 km is the single altitude in
    # the whole climb where the code does something different (below it the
    # shell clip returns [0, t], above it a proper sub-interval), so that is
    # where the pair is placed.
    #
    # A pair on its own would only measure "small", and small compared to
    # what is the whole question. So an identically-sized pair is rendered
    # at 90 km, two metres apart, entirely inside the shell and crossing
    # nothing, and both pairs are diffed at the SAME threshold. The control
    # is the calibration; the crossing has to be no worse than a move that
    # crosses nothing.
    #
    # Measured on an M4 Max, 320x240, 32 accumulation frames, all at
    # --max-delta 0 so every differing pixel counts:
    #
    #                                  max L2   mean L2    bad px
    #   same fixture rendered twice    0.0000   0.000000    0.000%
    #   10 km,  2 m apart              1.4142   0.018772    1.866%
    #   90 km,  2 m apart (CONTROL)    1.4142   0.001727    0.171%
    #   100 km, 2 m apart (CROSSING)   1.4142   0.001488    0.147%
    #   100 km, 20 m apart (crossing)  1.7321   0.014758    1.427%
    #   100 km vs 110 km               98.112   4.622206   41.514%
    #
    # Three things to read off it. The noise floor is exactly zero, so
    # every number below it is signal. The crossing is indistinguishable
    # from the control -- marginally SMALLER, in fact, because 100 km is
    # slightly further from the ground than 90 km is and the air the two
    # frames disagree about is correspondingly thinner. And refining the
    # separation tenfold shrinks the difference by 9.9x: linear, which is
    # what a Lipschitz function does and what a jump does not.
    #
    # The last row is the scale: a REAL 10 km move produces 3100x the mean
    # and 280x the bad-pixel count of the crossing. Anything that looked
    # like a step at the boundary would land somewhere in that range, not
    # in the fourth decimal place.
    #
    # No pixel differs by more than 1.7321 L2 = sqrt(3), i.e. one 8-bit
    # level in each of three channels, so the registered max-delta of 8 is
    # 4.6x clear of the observed maximum while still being 12x tighter
    # than the 98 a genuine step produces. The mean bar of 0.02 is 13x
    # clear of the crossing and 231x tighter than the step; the 1.0%
    # bad-pixel bar is 6.8x clear and 42x tighter. The headroom exists
    # because CI runners use a paravirtualised GPU -- the same reason the
    # worldframe A/B cells register max-delta 2 instead of 0.
    #
    # DESIGN RULE, HONOURED BY CONSTRUCTION: no planetary state transition
    # may require a history reset (#260, and #164 for the class of bug).
    # Nothing here is an engine-side mode change -- the shell clip is a
    # shader-side branch on a pushed uniform -- so crossing never sets
    # accum_dirty_ and a live fly-through does not flash.
    if(APPLE)
        set(_sc_scene ${CMAKE_SOURCE_DIR}/tests/goldens/scenes/shell_crossing.cfg)
        # name -> cam_pos altitude in metres
        set(_sc_names   below   above    ctl_lo  ctl_hi)
        set(_sc_alts    99999   100001   89999   90001)
        foreach(_i RANGE 3)
            list(GET _sc_names ${_i} _n)
            list(GET _sc_alts  ${_i} _a)
            set(_sc_png ${CMAKE_BINARY_DIR}/golden_actual/sky_continuity_shell__${_n}.png)
            set(_sc_wd  ${CMAKE_BINARY_DIR}/golden_workdir/sky_continuity_shell__${_n})
            file(MAKE_DIRECTORY ${_sc_wd})
            add_test(NAME sky_continuity_shell_render_${_n}
                COMMAND $<TARGET_FILE:pt_render_one_frame>
                    --scene    ${_sc_scene}
                    --backend  metal
                    --denoiser off
                    --frames   32
                    --extra    "cam_pos 0 ${_a} 0"
                    --out      ${_sc_png}
                WORKING_DIRECTORY ${_sc_wd})
            set_tests_properties(sky_continuity_shell_render_${_n} PROPERTIES
                TIMEOUT 300 LABELS "golden;golden_metal;planetary")
        endforeach()
        # The crossing, and its control, at one and the same bar.
        foreach(_pair cross:below:above control:ctl_lo:ctl_hi)
            string(REPLACE ":" ";" _pl ${_pair})
            list(GET _pl 0 _tag)
            list(GET _pl 1 _lo)
            list(GET _pl 2 _hi)
            add_test(NAME sky_continuity_shell_${_tag}
                COMMAND $<TARGET_FILE:imgdiff>
                    ${CMAKE_BINARY_DIR}/golden_actual/sky_continuity_shell__${_lo}.png
                    ${CMAKE_BINARY_DIR}/golden_actual/sky_continuity_shell__${_hi}.png
                    --max-delta 8 --mean-delta 0.02 --fail-percent 1.0
                    --diff ${CMAKE_BINARY_DIR}/golden_diff/sky_continuity_shell__${_tag}.png
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
            set_tests_properties(sky_continuity_shell_${_tag} PROPERTIES
                TIMEOUT 30
                DEPENDS "sky_continuity_shell_render_${_lo};sky_continuity_shell_render_${_hi}"
                LABELS  "golden;golden_metal;planetary")
        endforeach()
    endif()
    # --- end Planetary P6 -------------------------------------------------
else()
    message(STATUS
        "tests: golden-image matrix skipped -- pt_render_one_frame and/or "
        "imgdiff targets unavailable (PT_BUILD_TOOLS=OFF?)")
endif()

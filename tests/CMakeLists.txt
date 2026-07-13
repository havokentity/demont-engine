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
    set(_sky_scenes
        procedural_noon
        procedural_evening
        procedural_dawn
        bsc_night
        lunar_night
        bsc_night_clouds)
    foreach(_scene IN LISTS _sky_scenes)
        # Software backend / no denoiser -- this is the CI-exercised
        # cell. Tolerance kept at defaults so a real regression surfaces.
        pt_add_golden_cell(SCENE ${_scene} BACKEND software DENOISER off)
        # Metal backend + SVGF a-trous -- the user's primary dev combo
        # on M-series. Wider tolerance to absorb GPU FP-order variance.
        pt_add_golden_cell(SCENE ${_scene} BACKEND metal    DENOISER svgf_atrous
            MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    endforeach()
    # --- end #112 sky/sun/stars/moon matrix --------------------------------

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
    pt_add_golden_cell(SCENE clouds_raymarched BACKEND metal DENOISER svgf_atrous
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
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
    pt_add_golden_cell(SCENE clouds_godrays BACKEND metal DENOISER svgf_atrous
        EXTRA "r_godrays 1"
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 3)
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
    # FRAMES 120: the ocean animation advances by the per-frame wall-clock
    # dt (ocean_time_ accumulates), so the surface needs a stretch of
    # frames to evolve from the t=0 spectrum into a developed sea before
    # the capture. 120 frames warms it well past the dominant-wave period.
    #
    # Tolerances remain generous, but NOT for the reason this comment used
    # to give: smoke-mode dt is pinned to 1/60 (commit 5d57b29) and the
    # dt-integrating sims are frozen during un-counted loading frames, so
    # the surface phase at capture IS deterministic -- run-to-run diffs on
    # the dev M4 Max measure bit-identical (max_delta 0.0). What the
    # thresholds still have to absorb is cross-host / cross-build GPU
    # FP-ordering, which is unmeasured here. The cell catches gross
    # regressions: the ocean gate failing to engage (flat plane), the
    # textures not binding (black / sky-through water), or the ray-march
    # dropping rays through the surface. Metal-only at first; software
    # gates the ocean off (its CPU tracer ignores the textures) so no
    # software cell, and a Vulkan cell can land once the Metal PNG is
    # committed. Tighten once cross-host variance is actually measured.
    pt_add_golden_cell(SCENE ocean_fft BACKEND metal DENOISER svgf_atrous
        FRAMES 120
        MAX_DELTA 64 MEAN_DELTA 16 FAIL_PERCENT 10)
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
    #   * r_ocean_foam_amount / _persistence / _coverage plumbing
    # FRAMES 120 + generous tolerances for the same wall-clock-dt /
    # foam-fleck-flicker reasons documented on the ocean_fft cell above.
    # Metal-only (software gates the ocean off; a Vulkan cell can land once
    # this Metal PNG is committed).
    pt_add_golden_cell(SCENE ocean_foam BACKEND metal DENOISER svgf_atrous
        FRAMES 120
        MAX_DELTA 64 MEAN_DELTA 16 FAIL_PERCENT 10)
    # --- end Ocean foam + whitecaps ---------------------------------------

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
    pt_add_golden_cell(SCENE pbr_textured BACKEND software DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
    pt_add_golden_cell(SCENE pbr_textured BACKEND metal    DENOISER off
        MAX_DELTA 32 MEAN_DELTA 4 FAIL_PERCENT 2)
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
else()
    message(STATUS
        "tests: golden-image matrix skipped -- pt_render_one_frame and/or "
        "imgdiff targets unavailable (PT_BUILD_TOOLS=OFF?)")
endif()

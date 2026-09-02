# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_renderer STATIC
    AnalyticBvh.cpp
    Astronomy.cpp
    Atmosphere.cpp
    BscCatalog.cpp
    Csg/CsgScene.cpp
    EditorOverlay.cpp
    GltfImporter.cpp
    gltf_impl.cpp
    HdrImage.cpp
    LightTree.cpp
    MoonTexture.cpp
    Planet/CubedSphere.cpp
    Planet/ElevationField.cpp
    Planet/SurfaceAlbedo.cpp
    Planet/TerrainChunk.cpp
    Planet/TerrainQuadtree.cpp
    Planet/TerrainResidency.cpp
    TriangleBvh.cpp
)
target_include_directories(pt_renderer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_renderer
    PUBLIC  pt_core
    PRIVATE manifold
)

# Per-file warning suppression on gltf_impl.cpp. That TU is the sole
# definer of CGLTF_IMPLEMENTATION (third_party/cgltf, ~7k lines of C) AND
# STB_IMAGE_IMPLEMENTATION (third_party/stb, ~7k lines of C) -- both are
# vendored single-header libs with ~100+ -Wall / /W4 warnings we can't fix
# without forking. Same per-file `/w` (MSVC) / `-w` (Clang/GCC) pattern
# the engine's stb_impl.cpp + tools/imgdiff/stb_impl.cpp already use; the
# rest of pt_renderer (including GltfImporter.cpp's ~400 lines of loader
# logic) keeps its strict warning policy so future bugs there stay flagged.
if(MSVC)
    set_source_files_properties(gltf_impl.cpp PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(gltf_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

# --- Terrain baking is compute-bound: optimise it even in Debug ------------
# pt_planet_terrain walks a level-19 chunk's full subdivision hierarchy --
# ~22 000 field evaluations per chunk -- over an exhaustive sweep. At the
# Debug default (-O0 / no /O) that test reached ~585 s on the shared macOS
# and Windows CI runners against a 600 s TIMEOUT, i.e. no headroom, and every
# planetary change tipped a lane over the edge. This is exactly the speedup
# the test's own comment sanctions ("cut redundancy, not cases"): keep every
# case and assertion, remove only the -O0 penalty on the numeric grind.
#
# No fast-math. -O2 WITHOUT -ffast-math preserves IEEE operation-by-operation
# semantics, so the elevation field's exact output is unchanged and the
# terrain goldens do not move; the byte-identical sync-vs-async and
# cross-level watertightness assertions are intra-binary and hold by
# construction. The sanitizer build is skipped (it runs at its own -O1 with
# -g, see cmake/CompilerWarnings.cmake) so UBSan keeps full coverage of these
# files. On MSVC, /O2 overrides the target's /Od (a harmless D9025 command-
# line warning, not matched by the macOS-only zero-warning gate); /RTC1 is
# dropped for the whole Debug build in the top-level CMakeLists so /O2 is not
# a fatal D8016 here.
if(NOT PT_ENABLE_SANITIZERS)
    set(_pt_terrain_hot
        Planet/CubedSphere.cpp
        Planet/ElevationField.cpp
        Planet/TerrainChunk.cpp)
    if(MSVC)
        set_source_files_properties(${_pt_terrain_hot} PROPERTIES
            COMPILE_OPTIONS "$<$<CONFIG:Debug>:/O2>")
    else()
        set_source_files_properties(${_pt_terrain_hot} PROPERTIES
            COMPILE_OPTIONS "$<$<CONFIG:Debug>:-O2>")
    endif()
endif()

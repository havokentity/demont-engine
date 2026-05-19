# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_renderer STATIC
    AnalyticBvh.cpp
    Astronomy.cpp
    BscCatalog.cpp
    Csg/CsgScene.cpp
    EditorOverlay.cpp
    GltfImporter.cpp
    gltf_impl.cpp
    HdrImage.cpp
    LightTree.cpp
    MoonTexture.cpp
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

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte

# pt_destruction -- voxel destruction subsystem (issue #139 roadmap).
# Phase 1 (#140): voxelize CSG mesh / SDF / analytic primitives into a
# discrete VoxelGrid; the engine then renders each occupied voxel as
# an analytic box via the existing SDF cluster path -- no shader
# changes required.
#
# Header-only public deps on pt_renderer (SdfPrim, BakedMesh) so the
# library can be linked into tools that don't pull the full RHI stack.
add_library(pt_destruction STATIC
    VoxelGrid.cpp
    Voxelizer.cpp
)

target_include_directories(pt_destruction PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_destruction
    PUBLIC pt_core
    PUBLIC pt_renderer
)

pt_set_compiler_options(pt_destruction)

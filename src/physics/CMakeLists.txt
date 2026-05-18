# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte

# pt_physics -- Phases 1 + 2a of the physics roadmap (#131, #132, #138).
# Verlet integration on point-mass particles with constant gravity,
# sphere-plane (y=0) and pairwise sphere-sphere collision. Phase 2a
# adds XPBD rigid bodies (orientation + per-body inertia + angular
# velocity) with the same sphere-bounded collision shapes -- box-vs-box
# SAT lands in Phase 2b. CPU-only, single-threaded; the engine drives
# Step() once per frame.
#
# Architecturally a leaf library: depends only on glm (header-only,
# pulled in via pt_core's PUBLIC interface) and the C++23 standard
# library. No RHI, no renderer, no console -- the engine wires the
# cvars + console commands.
add_library(pt_physics STATIC
    PhysicsSystem.cpp
    RigidBody.cpp
)

target_include_directories(pt_physics PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_physics
    PUBLIC pt_core
)

pt_set_compiler_options(pt_physics)

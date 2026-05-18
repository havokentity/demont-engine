# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Effects layer -- CPU sim for the screen-space particle MVP (#82).
# Engine.cpp owns the per-frame GPU upload + dispatch; this lib just
# holds the integrator + emitter state. Keep dependencies minimal
# (glm only) so we don't drag the RHI surface into pure CPU code.

add_library(pt_effects STATIC
    ParticleSystem.cpp
)

target_include_directories(pt_effects PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_effects
    PUBLIC pt_core
)

pt_set_compiler_options(pt_effects)

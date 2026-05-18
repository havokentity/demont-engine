// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace pt::physics {

// Position-based Verlet particle. No explicit velocity is stored --
// the integrator infers it from (curr_pos - prev_pos) / dt each step,
// which is what makes Verlet stable under collision position-correction:
// pushing the particle out of an overlap automatically dampens its
// effective velocity along the contact normal (Jakobsen 2001).
//
// Units (per the project's metric-units rule): all positions are in
// metres, radius is in metres, mass in kilograms. inv_mass (1/kg) is
// stored instead of mass so the integrator's accel * dt * dt factor
// doesn't divide each step; inv_mass = 0 marks a kinematic particle
// (gravity doesn't accelerate it, collision doesn't move it -- not
// used in Phase 1's MVP, but the field is there for Phase 2's
// constraint pinning).
//
// Layout note: kept POD-ish so a future Phase 4 GPU port can blit
// the pool straight into a storage buffer. 16-byte fields keep the
// struct at 48 bytes -- one fits in a single cache line with room to
// spare for the engine's per-particle id metadata.
struct Particle {
    glm::vec3 prev_pos    {0.0f, 0.0f, 0.0f};
    glm::vec3 curr_pos    {0.0f, 0.0f, 0.0f};
    float     inv_mass    = 1.0f;   // 1 / kg; 0 = kinematic (Phase 2)
    float     radius      = 0.3f;   // metres; sphere collision shape
};

}  // namespace pt::physics

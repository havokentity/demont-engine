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
// metres, radius is in metres, mass in kilograms. inv_mass (1/m) is
// stored instead of mass so the integrator's accel * dt * dt factor
// doesn't divide each step; inv_mass = 0 marks a kinematic particle
// (gravity doesn't accelerate it, collision doesn't move it -- not
// used in Phase 1's MVP, but the field is there for Phase 2's
// constraint pinning).
//
// Layout note: kept POD-ish so a future Phase 4 GPU port can blit
// the pool straight into a storage buffer. With the project's default
// glm config (no GLM_FORCE_DEFAULT_ALIGNED_GENTYPES), glm::vec3 is the
// natural 12-byte packed layout, so the struct lands at 32 bytes on
// every platform we ship today (two 12-byte vec3s + two 4-byte
// floats, no trailing padding). Two particles fit per 64-byte cache
// line. NOT directly std430-compatible: a Phase 4 GPU port would need
// to either translate to an explicit vec4 layout in the storage
// buffer (12-byte vec3s in std430 round up to 16, and the float
// member alignment differs) or define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// at that point. Either path is fine; this struct is the CPU layout.
struct Particle {
    glm::vec3 prev_pos    {0.0f, 0.0f, 0.0f};
    glm::vec3 curr_pos    {0.0f, 0.0f, 0.0f};
    float     inv_mass    = 1.0f;   // 1 / kg; 0 = kinematic (Phase 2)
    float     radius      = 0.3f;   // metres; sphere collision shape
};

}  // namespace pt::physics

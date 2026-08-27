// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace pt::physics {

// Position-based dynamics particle. Keeps the Verlet position pair
// (prev_pos, curr_pos) that makes the scheme stable under collision
// position-correction -- pushing the particle out of an overlap
// automatically dampens its effective velocity along the contact
// normal (Jakobsen 2001) -- but carries velocity EXPLICITLY rather
// than leaving it implicit in (curr_pos - prev_pos). See #270 and the
// long derivation in PhysicsSystem::Substep: the implicit form fed
// gravity in as `accel * sdt^2`, an absolute increment that falls
// below the float32 ULP of the position it is added to above ~1 km
// (and above ~64 m at the phys_substeps maximum of 32), at which
// point the add is a bit-for-bit no-op and gravity silently stops
// existing. `velocity` accumulates `accel * sdt` instead, a quantity
// proportional to the motion rather than to the square of the
// timestep, so the substep count no longer decides whether gravity is
// representable at all.
//
// Units (per the project's metric-units rule): all positions are in
// metres, velocity in metres per second, radius is in metres, mass in
// kilograms. inv_mass (1/kg) is stored instead of mass so the
// integrator's accel * dt factor doesn't divide each step; inv_mass =
// 0 marks a kinematic particle (gravity doesn't accelerate it,
// collision doesn't move it -- not used in Phase 1's MVP, but the
// field is there for Phase 2's constraint pinning).
//
// Invariant for external consumers (the engine's per-frame writeback,
// the editor's inspector readout): at the end of every substep
//   curr_pos - prev_pos == velocity * PhysicsSystem::LastSubstepDt()
// to within float rounding, so Phase-1-era code that derives velocity
// from the position pair keeps reading the same number. The one
// caveat is a body so far from the origin that `velocity * sdt` sits
// below the position's ULP -- then the positions tie and the derived
// velocity reads 0 while the stored `velocity` is correct. Prefer the
// stored field in new code.
//
// Layout note: kept POD-ish so a future Phase 4 GPU port can blit
// the pool straight into a storage buffer. With the project's default
// glm config (no GLM_FORCE_DEFAULT_ALIGNED_GENTYPES), glm::vec3 is the
// natural 12-byte packed layout, so the struct lands at 44 bytes on
// every platform we ship today (three 12-byte vec3s + two 4-byte
// floats, no trailing padding). That is up from the 32 bytes of the
// velocity-less Phase 1 form, so particles no longer pack two to a
// 64-byte cache line; at kMaxParticles = 256 the whole pool is 11 KB
// and still resident in L1 on every target we ship, which is why the
// correctness win is worth the density. NOT directly
// std430-compatible: a Phase 4 GPU port would need to either
// translate to an explicit vec4 layout in the storage buffer
// (12-byte vec3s in std430 round up to 16, and the float member
// alignment differs) or define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// at that point. Either path is fine; this struct is the CPU layout.
struct Particle {
    glm::vec3 prev_pos    {0.0f, 0.0f, 0.0f};
    glm::vec3 curr_pos    {0.0f, 0.0f, 0.0f};
    glm::vec3 velocity    {0.0f, 0.0f, 0.0f};   // m/s, world frame (#270)
    float     inv_mass    = 1.0f;   // 1 / kg; 0 = kinematic (Phase 2)
    float     radius      = 0.3f;   // metres; sphere collision shape
};

}  // namespace pt::physics

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace pt::physics {

// Phase 2a of the physics roadmap (#131, #138).
//
// RigidBody extends the Phase 1 point-mass Particle with an orientation
// (quaternion), an angular velocity (omega, world frame), and a
// per-body inertia tensor. The integration scheme is XPBD as described
// in Mueller et al. 2020 ("Detailed Rigid Body Simulation with Extended
// Position Based Dynamics"): predict pos+orientation, apply
// position/orientation constraints, then finite-difference velocities
// at the end of the substep.
//
// MVP discipline (Phase 2a only; box-box collision is Phase 2b):
//   * Sphere + Box shapes only. Capsule deferred.
//   * Collision shape used by the broad-phase is always the bounding
//     SPHERE -- for actual spheres that's exact; for boxes it's a
//     conservative bounding sphere with radius = length(half_extents).
//     Box-vs-box / box-vs-plane corner contacts (SAT) land in Phase 2b.
//   * No persistent contacts, no sleeping, no joints, no CCD, no
//     friction or restitution -- same Phase 1 deferrals carry over.
//
// Units (per the project's metric-units rule): positions in metres,
// half_extents in metres, mass in kilograms. Inertia is stored as its
// inverse (inv_inertia, kg^-1 m^-2) in body-local frame, mirroring how
// Particle stores inv_mass: a zero inv inertia means "infinite inertia"
// = orientation pinned (kinematic), so the existing Particle convention
// extends cleanly.
//
// Closed-form inertia tensors (about the centre of mass, body-local):
//   * Solid sphere: I = (2/5) * m * r^2 * Identity
//   * Solid box (half_extents hx,hy,hz): I = m/3 * diag( hy^2+hz^2,
//                                                       hx^2+hz^2,
//                                                       hx^2+hy^2 )
//     (full extents in the textbook form get an extra 1/4 -- ours uses
//      half_extents, so the 1/12 textbook constant becomes 1/3.)

enum class Shape : std::uint8_t {
    Sphere = 0,
    Box    = 1,
};

struct RigidBody {
    // Translational state (mirrors Particle: prev/curr pair for the
    // implicit-velocity Verlet style, plus inv_mass and a collision
    // radius). For boxes, `radius` is the bounding-sphere radius used
    // by Phase 2a's sphere-only collision; the geometric shape itself
    // is described by `shape` + `half_extents`.
    glm::vec3 prev_pos    {0.0f, 0.0f, 0.0f};
    glm::vec3 curr_pos    {0.0f, 0.0f, 0.0f};
    float     inv_mass    = 1.0f;          // 1 / kg; 0 = kinematic
    float     radius      = 0.3f;          // metres (bounding sphere)

    // Rotational state. orientation is the body-to-world rotation
    // quaternion (w,x,y,z); prev_orientation lets us finite-difference
    // angular velocity at the end of each substep. omega is the world-
    // frame angular velocity in rad/s -- we keep it explicitly so
    // external impulses (none in Phase 2a, but the slot is here) can
    // poke it without going through a finite-diff round-trip.
    glm::quat orientation      {1.0f, 0.0f, 0.0f, 0.0f};
    glm::quat prev_orientation {1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 omega            {0.0f, 0.0f, 0.0f};   // rad/s, world frame

    // Body-local inverse inertia tensor (diagonal -- shape-aligned axes).
    // A zero component means "infinite inertia about that axis" =
    // rotation locked along that axis. For a sphere all three are
    // equal; for a box they differ. Stored as a vec3 since both
    // supported shapes have an orthotropic (diagonal) tensor in body
    // frame.
    glm::vec3 inv_inertia_local {1.0f, 1.0f, 1.0f};   // kg^-1 m^-2

    // Shape description.
    Shape     shape        = Shape::Sphere;
    glm::vec3 half_extents {0.3f, 0.3f, 0.3f};   // metres; box only
};

// Compute the body-local inverse inertia tensor for a solid sphere of
// given mass and radius. Returns vec3(0) for non-positive mass (caller
// sees a kinematic body that won't rotate).
glm::vec3 InvInertiaSphere(float mass, float radius);

// Compute the body-local inverse inertia tensor for a solid box of
// given mass and half-extents. Same kinematic semantics as
// InvInertiaSphere when mass is non-positive.
glm::vec3 InvInertiaBox(float mass, const glm::vec3& half_extents);

// Bounding-sphere radius for a box with the given half-extents. Used
// by Phase 2a's sphere-only narrowphase to bound a box conservatively
// before Phase 2b lands real SAT.
inline float BoxBoundingRadius(const glm::vec3& half_extents) {
    return glm::length(half_extents);
}

// Rotate a body-local inertia tensor (diagonal vec3) into world frame:
// I_world = R * I_local * R^T. Returns the full 3x3 matrix because the
// world-frame tensor is generally non-diagonal once the body has
// tumbled. Used by the integrator when projecting omega through the
// inertia tensor at predict time.
glm::mat3 WorldInverseInertia(const glm::quat& orientation,
                              const glm::vec3& inv_inertia_local);

// XPBD orientation predictor: integrate quaternion by world-frame
// omega for the substep duration dt. Equivalent to Equation 12 in the
// Mueller 2020 XPBD-RB paper (quaternion derivative form). Normalises
// the result so accumulated drift doesn't blow the quaternion up.
glm::quat IntegrateOrientation(const glm::quat& q,
                               const glm::vec3& omega,
                               float dt);

// Finite-difference angular velocity from two orientation samples
// taken `dt` seconds apart. Mirror of the position-Verlet implicit
// velocity (curr - prev) / dt, but on the rotation manifold via the
// quaternion-log map. Used at the end of each substep to refresh
// `omega` from the constrained orientation.
glm::vec3 OrientationDiffOmega(const glm::quat& prev,
                               const glm::quat& curr,
                               float dt);

}  // namespace pt::physics

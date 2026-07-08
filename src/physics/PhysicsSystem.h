// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Particle.h"
#include "RigidBody.h"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace pt::physics {

// PhysicsSystem -- Phase 1 of the physics roadmap (#131, #132).
//
// Owns a flat pool of point-mass Verlet particles, integrates them
// against constant gravity, and resolves sphere-plane (y=0) and
// pairwise sphere-sphere contacts with positional correction. Pure
// CPU, single-threaded, no broadphase. Architecturally a leaf
// subsystem: depends only on glm + std; the engine drives it via
// Init/Step/Shutdown and the per-frame writeback hook.
//
// Phase 1 deliberate non-goals (per issue #132 MVP discipline):
//   - No friction (Phase 5)
//   - No restitution (Phase 5)
//   - No sleeping islands (Phase 5)
//   - No box/capsule shapes (Phase 6+)
//   - No spatial hash; pairwise sphere-sphere is O(N^2) and the
//     particle cap of kMaxParticles is intentional to keep that
//     honest until Phase 4 lands a broadphase.
//
// Stack stability: explicitly NOT addressed. Verlet + position
// correction without contact-point persistence will leave a tower of
// spheres jittering forever, which is the correct behaviour for
// Phase 1 -- the damping factor (`phys_damping`, default 0.99 per
// substep) bleeds enough energy that things eventually settle into a
// loose pile, and Phase 5 will fix proper stacking with contact
// persistence + sleeping islands.
class PhysicsSystem {
public:
    // Cap chosen so the O(N^2) pairwise check stays well under 1 ms
    // per substep on Apple Silicon (~65k pairs at 256). Raise once
    // Phase 4 lands a spatial hash.
    static constexpr std::uint32_t kMaxParticles = 256u;

    // Stable per-particle handle. Returned by AddParticle, accepted
    // by RemoveParticle / GetParticle. Generation-tagged so a stale
    // handle (from before a Remove + Add that reused the slot)
    // returns nullptr from GetParticle instead of silently aliasing
    // the new particle. The high 24 bits are generation, low 8 are
    // the pool index -- keeps the pool small (max 256) and the
    // generation counter plenty wide for any realistic churn (a
    // pathological 16M remove+add cycle per slot is the wraparound
    // bound; at 60 Hz that's ~3.2 years of constant churn).
    using Handle = std::uint32_t;
    static constexpr Handle kInvalidHandle = 0u;

    PhysicsSystem()  = default;
    ~PhysicsSystem() = default;

    PhysicsSystem(const PhysicsSystem&)            = delete;
    PhysicsSystem& operator=(const PhysicsSystem&) = delete;

    // Drop all particles AND rigid bodies. Idempotent. Called on
    // Engine::Shutdown and from the `phys_clear` console command.
    void Clear();

    // Spawn a particle at world-space `pos` with the given radius
    // and (optional) inverse mass. Returns kInvalidHandle if the
    // pool is at kMaxParticles capacity.
    Handle AddParticle(const glm::vec3& pos, float radius, float inv_mass = 1.0f);

    // Drop the particle identified by `h`. Idempotent: a stale or
    // never-issued handle is a silent no-op (returns false).
    bool RemoveParticle(Handle h);

    // Read-only handle resolution. Returns nullptr if the slot is
    // free OR the generation no longer matches.
    const Particle* GetParticle(Handle h) const;

    // Outer-frame step. Drives `substeps` inner Verlet steps with
    // dt = frame_dt / substeps so stiff position-correction stays
    // stable at the cost of more sub-substep collision iterations
    // per frame. Caller clamps frame_dt to something sensible
    // (engine clamps to <= 1/30 s) before invoking.
    //
    // Forces in Phase 1: constant gravity on Y (m/s^2, signed --
    // pass -9.81 for Earth pointing down).
    void Step(float frame_dt, int substeps, float gravity_y, float damping_per_substep);

    // Iteration helpers for the engine's per-frame writeback to the
    // analytic-primitive buffer. The engine indexes by particle id
    // (the AnalyticPrim map key) which is set on AddParticle via
    // SetPrimId; the engine pairs each live particle with its
    // matching prim slot. Live particles are NOT necessarily packed
    // -- iterate via ForEach / use IsAlive.
    using IterFn = void (*)(Handle h, const Particle& p, std::uint32_t prim_id, void* user);
    void ForEach(IterFn fn, void* user) const;

    std::uint32_t AliveCount() const { return alive_count_; }
    std::uint32_t Capacity()   const { return kMaxParticles; }

    // Pair a particle with an analytic-primitive id so the engine's
    // per-frame writeback knows which sphere in `primitives_` to
    // update. Stored alongside the particle; cleared on Remove.
    // Returns false on stale handle.
    bool SetPrimId(Handle h, std::uint32_t prim_id);
    std::uint32_t GetPrimId(Handle h) const;

    // --- Phase 2a rigid bodies (#138) -------------------------------
    //
    // RigidBody pool is parallel to (and independent of) the Particle
    // pool. Same handle/generation discipline; handles from one pool
    // are NOT interchangeable with the other (they're a distinct
    // typedef so the type system flags the mistake at compile time).
    //
    // Phase 2a collision discipline: rigid bodies collide against the
    // ground plane and against each other using the body's bounding
    // sphere only (radius field on RigidBody). For Shape::Sphere
    // that's exact; for Shape::Box that's a conservative approximation
    // -- boxes will fall and roll against the plane via their bounding
    // sphere, which means a box never lies flat. Phase 2b lands real
    // SAT-based box-plane + box-box narrowphase. Per issue #138 MVP
    // discipline, boxes are explicitly allowed to clip through each
    // other in this PR.
    using RbHandle = std::uint32_t;
    static constexpr RbHandle kInvalidRbHandle = 0u;
    // Smaller cap than particles since rigid bodies are heavier per-
    // step (orientation integration + per-body inertia tensor) and the
    // O(N^2) pairwise sphere check still applies. 128 is comfortable
    // on M4 Max even with 8 substeps.
    static constexpr std::uint32_t kMaxRigidBodies = 128u;

    // Spawn a sphere rigid body. radius and mass must both be > 0;
    // mass == 0 spawns a kinematic body (zero inv_mass + inv_inertia).
    RbHandle AddRigidSphere(const glm::vec3& pos, float radius,
                            float mass = 1.0f,
                            const glm::quat& orientation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f});

    // Spawn a box rigid body. `half_extents` must all be > 0. The
    // body's bounding sphere is length(half_extents) -- that's what
    // Phase 2a uses for collision until SAT lands in Phase 2b.
    RbHandle AddRigidBox(const glm::vec3& pos,
                         const glm::vec3& half_extents,
                         float mass = 1.0f,
                         const glm::quat& orientation = glm::quat{1.0f, 0.0f, 0.0f, 0.0f});

    bool RemoveRigidBody(RbHandle h);
    const RigidBody* GetRigidBody(RbHandle h) const;
    bool SetRbPrimId(RbHandle h, std::uint32_t prim_id);
    std::uint32_t GetRbPrimId(RbHandle h) const;

    using RbIterFn = void (*)(RbHandle h, const RigidBody& b,
                              std::uint32_t prim_id, void* user);
    void ForEachRigidBody(RbIterFn fn, void* user) const;

    std::uint32_t RbAliveCount() const { return rb_alive_count_; }
    std::uint32_t RbCapacity()   const { return kMaxRigidBodies; }
    // --- end Phase 2a rigid bodies ----------------------------------

    // Substep dt (s) of the most recent Step() call. The Verlet pools
    // store implicit velocity as a per-substep displacement
    // (curr - prev); dividing by this converts it to real m/s.
    // 0 until the first Step() -- callers must guard.
    float LastSubstepDt() const { return last_substep_dt_; }

private:
    // Slot generation is the FULL 24 bits the public Handle API
    // promises, stored in a uint32_t so the bump-on-Remove path never
    // wraps after 256 churn cycles (the previous uint8_t form did, and
    // a stale handle could then alias a live particle once the gen
    // wrapped back to the original value). 24 bits gives ~16M cycles
    // per slot before wraparound -- big enough that the public API
    // can document it as "plenty wide for any realistic churn".
    struct Slot {
        Particle      p{};
        std::uint32_t generation = 1;  // 24-bit counter packed into Handle
        std::uint8_t  alive      = 0;
        std::uint8_t  pad0       = 0;
        std::uint16_t pad1       = 0;
        std::uint32_t prim_id    = 0;  // matching key in engine primitives_
    };

    // Inner-substep workhorse, called `substeps` times per Step. Each
    // call:
    //   1. Predict next position from Verlet (curr + (curr - prev) * damping + accel*sdt*sdt)
    //   2. Resolve sphere-plane (y=0) and sphere-sphere overlaps
    //   3. Shift the pos history forward (prev = old curr, curr = new pos)
    void Substep(float sdt, const glm::vec3& accel, float damping);

    // --- Phase 2a rigid bodies (#138) -------------------------------
    // generation is uint32_t for the same reason as Slot::generation
    // above: the handle API packs a 24-bit generation, and the old
    // uint8_t here truncated it to a period-255 counter -- 255
    // remove/add cycles on a slot (phys_clear + phys_drop churn) let a
    // stale RbHandle validate against the new occupant and silently
    // read / re-pair an unrelated body.
    struct RbSlot {
        RigidBody     body{};
        std::uint32_t generation = 1;  // 24-bit counter packed into RbHandle
        std::uint8_t  alive      = 0;
        std::uint8_t  pad0       = 0;
        std::uint16_t pad1       = 0;
        std::uint32_t prim_id    = 0;
    };

    // Inner-substep workhorse for the rigid body pool. Mirrors the
    // particle Substep but adds the orientation predictor + the
    // omega finite-difference at the end. Sphere collisions are shared
    // with the particle pool (rigid bodies vs rigid bodies pairwise +
    // rigid bodies vs ground plane). Per Phase 2a spec, particles and
    // rigid bodies do NOT interact across pools -- they're separate
    // worlds for now; if a user needs them to talk, Phase 2b's
    // unified solver lands that.
    void SubstepRigidBodies(float sdt, const glm::vec3& accel,
                            float linear_damping, float angular_damping);

    std::vector<RbSlot> rb_slots_{kMaxRigidBodies};
    std::uint32_t       rb_alive_count_ = 0;
    // --- end Phase 2a rigid bodies ----------------------------------

    std::vector<Slot> slots_{kMaxParticles};
    std::uint32_t     alive_count_ = 0;

    // Substep dt of the most recent Step() -- see LastSubstepDt().
    float last_substep_dt_ = 0.0f;
};

}  // namespace pt::physics

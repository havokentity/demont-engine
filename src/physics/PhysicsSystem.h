// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "Particle.h"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

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

    // Drop all particles. Idempotent. Called on Engine::Shutdown and
    // from the `phys_clear` console command.
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

    std::vector<Slot> slots_{kMaxParticles};
    std::uint32_t     alive_count_ = 0;
};

}  // namespace pt::physics

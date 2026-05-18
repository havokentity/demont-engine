// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "PhysicsSystem.h"

#include <algorithm>
#include <cmath>

namespace pt::physics {

namespace {

// Pack / unpack the 32-bit handle: low 8 bits = pool index (0..255),
// high 24 bits = generation. Index 0 + generation 0 = kInvalidHandle.
// The 24-bit generation is the contract the public Handle docs
// promise -- truncating to 8 bits here (the previous form) would
// silently wrap after 256 remove/add cycles per slot and let a stale
// handle alias a live particle. uint32_t arithmetic + a 24-bit mask
// keeps the pack/unpack consistent with the storage in Slot.
constexpr std::uint32_t kIndexBits = 8u;
constexpr std::uint32_t kIndexMask = (1u << kIndexBits) - 1u;
constexpr std::uint32_t kGenBits   = 24u;
constexpr std::uint32_t kGenMask   = (1u << kGenBits) - 1u;

inline PhysicsSystem::Handle MakeHandle(std::uint32_t index, std::uint32_t gen) {
    return ((gen & kGenMask) << kIndexBits) | (index & kIndexMask);
}
inline std::uint32_t HandleIndex(PhysicsSystem::Handle h) {
    return h & kIndexMask;
}
inline std::uint32_t HandleGen(PhysicsSystem::Handle h) {
    return (h >> kIndexBits) & kGenMask;
}

}  // namespace

void PhysicsSystem::Clear() {
    for (auto& s : slots_) {
        if (s.alive) {
            // Bump generation so any outstanding handle to this slot
            // resolves to nullptr on a subsequent GetParticle. Mod
            // 2^24 (the Handle's gen field width) and skip 0 so the
            // wraparound case can't synthesize the kInvalidHandle
            // pattern for an in-use slot.
            s.generation = (s.generation + 1u) & kGenMask;
            if (s.generation == 0u) s.generation = 1u;
        }
        s.alive   = 0;
        s.prim_id = 0;
        s.p       = Particle{};
    }
    alive_count_ = 0;
}

PhysicsSystem::Handle PhysicsSystem::AddParticle(const glm::vec3& pos,
                                                 float radius,
                                                 float inv_mass) {
    if (alive_count_ >= kMaxParticles) return kInvalidHandle;

    // Linear free-slot scan. Pool is fixed at 256 entries so this is
    // at most a 256-byte cache-friendly walk -- no need for an
    // explicit free list.
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (s.alive) continue;
        s.p.prev_pos = pos;     // zero implied velocity on spawn
        s.p.curr_pos = pos;
        s.p.radius   = (radius > 0.0f) ? radius : 0.3f;
        s.p.inv_mass = (inv_mass >= 0.0f) ? inv_mass : 1.0f;
        s.alive      = 1;
        s.prim_id    = 0;
        // Generation stays at whatever Remove last set it to (or 1
        // for a virgin slot) -- the caller's old handle (if any) was
        // already invalidated by the previous Remove's bump.
        ++alive_count_;
        return MakeHandle(i, s.generation);
    }
    return kInvalidHandle;  // unreachable while alive_count_ tracks correctly
}

bool PhysicsSystem::RemoveParticle(Handle h) {
    if (h == kInvalidHandle) return false;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxParticles) return false;
    auto& s = slots_[i];
    if (!s.alive)                       return false;
    if (HandleGen(h) != s.generation)   return false;
    s.alive      = 0;
    s.prim_id    = 0;
    s.p          = Particle{};
    // 24-bit wraparound mod the Handle gen field width; skip 0 so the
    // wrapped value can't synthesize the kInvalidHandle pattern.
    s.generation = (s.generation + 1u) & kGenMask;
    if (s.generation == 0u) s.generation = 1u;
    --alive_count_;
    return true;
}

const Particle* PhysicsSystem::GetParticle(Handle h) const {
    if (h == kInvalidHandle) return nullptr;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxParticles) return nullptr;
    const auto& s = slots_[i];
    if (!s.alive)                     return nullptr;
    if (HandleGen(h) != s.generation) return nullptr;
    return &s.p;
}

bool PhysicsSystem::SetPrimId(Handle h, std::uint32_t prim_id) {
    if (h == kInvalidHandle) return false;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxParticles) return false;
    auto& s = slots_[i];
    if (!s.alive)                     return false;
    if (HandleGen(h) != s.generation) return false;
    s.prim_id = prim_id;
    return true;
}

std::uint32_t PhysicsSystem::GetPrimId(Handle h) const {
    if (h == kInvalidHandle) return 0;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxParticles) return 0;
    const auto& s = slots_[i];
    if (!s.alive)                     return 0;
    if (HandleGen(h) != s.generation) return 0;
    return s.prim_id;
}

void PhysicsSystem::ForEach(IterFn fn, void* user) const {
    if (fn == nullptr) return;
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        const auto& s = slots_[i];
        if (!s.alive) continue;
        fn(MakeHandle(i, s.generation), s.p, s.prim_id, user);
    }
}

void PhysicsSystem::Step(float frame_dt, int substeps,
                         float gravity_y, float damping_per_substep) {
    if (alive_count_ == 0) return;
    if (frame_dt <= 0.0f)  return;

    // Clamp substep count into a sane range. 1..32 is plenty -- the
    // cvar default is 8 (the sweet spot for Phase 1).
    if (substeps < 1)  substeps = 1;
    if (substeps > 32) substeps = 32;

    const float sdt = frame_dt / static_cast<float>(substeps);
    const glm::vec3 accel{0.0f, gravity_y, 0.0f};

    // Sanitize damping. < 0 or > 1 would blow energy up; we cap at
    // [0.5, 1.0] -- below 0.5 anything moving collapses to zero in
    // one substep, which is also bad. The cvar default is 0.99.
    float damping = damping_per_substep;
    if (damping < 0.5f) damping = 0.5f;
    if (damping > 1.0f) damping = 1.0f;

    for (int step = 0; step < substeps; ++step) {
        Substep(sdt, accel, damping);
    }
}

void PhysicsSystem::Substep(float sdt, const glm::vec3& accel, float damping) {
    // 1. Predict next positions via velocity-Verlet with damping on
    //    the implicit velocity term (curr - prev). damping = 1 means
    //    "energy-conserving"; the default 0.99 bleeds 1% of velocity
    //    per substep. At 8 substeps/frame, 60 fps (480 substeps/sec)
    //    the residual velocity after one second is 0.99^480 ~= 0.008
    //    -- aggressive enough to settle a tossed pile into a loose
    //    heap in about a second of wall time without Phase 5's
    //    contact-persistence sleeper. See phys_damping cvar docs.
    const float sdt2 = sdt * sdt;
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (!s.alive) continue;
        if (s.p.inv_mass <= 0.0f) {
            // Kinematic -- skip integration. prev_pos pinned to
            // curr_pos so the implicit velocity stays 0.
            s.p.prev_pos = s.p.curr_pos;
            continue;
        }
        const glm::vec3 old_curr = s.p.curr_pos;
        const glm::vec3 disp     = (s.p.curr_pos - s.p.prev_pos) * damping;
        s.p.curr_pos             = old_curr + disp + accel * sdt2;
        s.p.prev_pos             = old_curr;
    }

    // 2. Resolve sphere-plane contact with the static y=0 ground.
    //    Penetration is the distance the sphere's bottom is below
    //    y=0; we push the centre back up by exactly that amount
    //    (kinematic plane: 100% of correction goes to the particle).
    //
    //    Phase 5 will introduce a non-zero restitution coefficient
    //    here; for Phase 1, the only "bounce" comes from Verlet's
    //    implicit-velocity reflection: pushing the centre up while
    //    leaving prev_pos below the original y reflects the downward
    //    velocity component over the contact normal, giving a 100%
    //    elastic bounce by construction. Damping (above) bleeds that
    //    away to settle the particle eventually.
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (!s.alive) continue;
        if (s.p.inv_mass <= 0.0f) continue;
        const float penetration = s.p.radius - s.p.curr_pos.y;
        if (penetration > 0.0f) {
            s.p.curr_pos.y += penetration;
        }
    }

    // 3. Resolve sphere-sphere contacts pairwise. O(N^2) until
    //    Phase 4. Each overlap is split half/half across the two
    //    particles (mass-weighted: a kinematic neighbour absorbs no
    //    correction, so the live particle eats the full overlap;
    //    sum-of-inv-masses split otherwise -- this generalises
    //    cleanly once Phase 2 introduces non-uniform masses).
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& si = slots_[i];
        if (!si.alive) continue;
        for (std::uint32_t j = i + 1; j < kMaxParticles; ++j) {
            auto& sj = slots_[j];
            if (!sj.alive) continue;
            const glm::vec3 d = sj.p.curr_pos - si.p.curr_pos;
            const float min_d = si.p.radius + sj.p.radius;
            const float d2    = glm::dot(d, d);
            if (d2 >= min_d * min_d) continue;
            // Degenerate: two particles spawned at the exact same
            // position. Pick an arbitrary normal so the correction
            // doesn't divide by zero. Any nudge along an axis is
            // fine; the system will sort itself out in a few steps.
            const float dist = std::sqrt(d2);
            glm::vec3 n;
            if (dist > 1e-6f) {
                n = d / dist;
            } else {
                n = glm::vec3{1.0f, 0.0f, 0.0f};
            }
            const float overlap = min_d - dist;
            const float inv_sum = si.p.inv_mass + sj.p.inv_mass;
            if (inv_sum <= 0.0f) continue;  // both kinematic
            // ki, kj add up to 1 -- mass-weighted split of the overlap.
            const float ki = si.p.inv_mass / inv_sum;
            const float kj = sj.p.inv_mass / inv_sum;
            si.p.curr_pos -= n * (overlap * ki);
            sj.p.curr_pos += n * (overlap * kj);
        }
    }

    // 4. Second sphere-plane pass. The sphere-sphere correction
    //    above can push a lower sphere downward (e.g. when a tower
    //    of stacked spheres compresses, the bottom one absorbs the
    //    full chain's overlap and gets pushed below y=radius). Re-
    //    running the plane projection here guarantees every
    //    particle ends the substep with bottom-y >= 0, so the
    //    render-frame sees the ground plane as inviolable instead
    //    of letting spheres render embedded in the floor.
    //
    //    This isn't a full Gauss-Seidel relaxation -- a sphere-
    //    sphere overlap created here is left to the NEXT substep
    //    to resolve. At kPhys default substeps=8/frame the residual
    //    overlap from this last sphere pass is invisible (sub-mm
    //    at typical 0.3 m radius). A Phase 5 full iterated solver
    //    is the long-term fix; this is the Phase-1-MVP-correct
    //    "the ground plane is the ground plane" guarantee.
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (!s.alive) continue;
        if (s.p.inv_mass <= 0.0f) continue;
        const float penetration = s.p.radius - s.p.curr_pos.y;
        if (penetration > 0.0f) {
            s.p.curr_pos.y += penetration;
        }
    }
}

}  // namespace pt::physics

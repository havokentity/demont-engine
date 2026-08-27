// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "PhysicsSystem.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/quaternion.hpp>

// Build-time proof that the fast-math exemption actually reached this
// translation unit (#270, and the same hazard as #263 / #267). The
// integrator below leans on two properties a reassociating compiler is
// free to break:
//
//   * `prev_pos + velocity * sdt` re-evaluates to the exact bits the
//     predictor stored in curr_pos, so the constraint-correction term
//     in step 5 is exactly zero for a particle that touched nothing.
//     Under -ffast-math the two spellings of the same expression may
//     be contracted differently (FMA vs mul+add) and no longer tie,
//     which would inject a spurious velocity correction on every
//     substep of every free-falling body.
//   * Debug and Release agree bit-for-bit, so a physics bug reproduces
//     in the build you can step through.
//
// cmake/CompilerWarnings.cmake's pt_set_strict_fp() is what turns the
// project-wide Release -ffast-math back off for pt_physics. If someone
// drops that call, this fires at compile time instead of silently
// handing the integrator back to the optimiser.
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "pt_physics must be built WITHOUT fast-math: call pt_set_strict_fp(pt_physics) -- see cmake/CompilerWarnings.cmake and issue #270"
#endif

namespace pt::physics {

namespace {

// Pack / unpack the 32-bit handle: low 8 bits = pool index (0..255),
// high 24 bits = generation. Index 0 + generation 0 = kInvalidHandle.
// Same packing is used for the rigid-body handle (kMaxRigidBodies =
// 128 fits comfortably in 8 bits) -- the two pools share the encoding
// but RbHandle is a distinct typedef in the header so the type checker
// catches a particle handle smuggled into a rigid-body call.
//
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

    // Mirror for the rigid-body pool. Same generation-bump discipline
    // so stale RbHandles can't accidentally aliase a refilled slot.
    for (auto& rb : rb_slots_) {
        if (rb.alive) {
            rb.generation = (rb.generation + 1u) & kGenMask;
            if (rb.generation == 0u) rb.generation = 1u;
        }
        rb.alive   = 0;
        rb.prim_id = 0;
        rb.body    = RigidBody{};
    }
    rb_alive_count_ = 0;
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
        // ...and zero explicit velocity to match. Remove/Clear already
        // reset the slot to a default-constructed Particle, so this is
        // belt-and-braces against a future partial-reset path leaving
        // a dead body's momentum attached to a fresh spawn.
        s.p.velocity = glm::vec3{0.0f};
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
    // Both pools are independently empty -> nothing to integrate.
    if (alive_count_ == 0 && rb_alive_count_ == 0) return;
    if (frame_dt <= 0.0f)  return;

    // Clamp substep count into a sane range. 1..32 is plenty -- the
    // cvar default is 8 (the sweet spot for Phase 1).
    if (substeps < 1)  substeps = 1;
    if (substeps > 32) substeps = 32;

    const float sdt = frame_dt / static_cast<float>(substeps);
    // Remember the substep dt so consumers of the Verlet implicit
    // velocity (curr - prev, a per-substep displacement) can convert
    // it to real m/s -- see LastSubstepDt().
    last_substep_dt_ = sdt;
    const glm::vec3 accel{0.0f, gravity_y, 0.0f};

    // Sanitize damping. < 0 or > 1 would blow energy up; we cap at
    // [0.5, 1.0] -- below 0.5 anything moving collapses to zero in
    // one substep, which is also bad. The cvar default is 0.99.
    float damping = damping_per_substep;
    if (damping < 0.5f) damping = 0.5f;
    if (damping > 1.0f) damping = 1.0f;

    // Angular damping: re-use the same `damping` value for omega in
    // Phase 2a. A spinning sphere on a frictionless ground would spin
    // forever; we bleed that off at the same rate as linear damping
    // for visual coherence. Phase 5 will replace this with a proper
    // contact-friction model.
    const float ang_damping = damping;

    for (int step = 0; step < substeps; ++step) {
        if (alive_count_ > 0)    Substep(sdt, accel, damping);
        if (rb_alive_count_ > 0) SubstepRigidBodies(sdt, accel, damping, ang_damping);
    }
}

void PhysicsSystem::Substep(float sdt, const glm::vec3& accel, float damping) {
    // Reciprocal of the substep dt, used by step 5 to convert the
    // constraint solver's position correction back into a velocity
    // correction. sdt is strictly positive here: Step() early-outs on
    // frame_dt <= 0 and clamps substeps into [1, 32].
    const float inv_sdt = 1.0f / sdt;

    // 1. Predict next positions by integrating the explicit velocity,
    //    with damping applied to the velocity term. damping = 1 means
    //    "energy-conserving"; the default 0.99 bleeds 1% of velocity
    //    per substep. At 8 substeps/frame, 60 fps (480 substeps/sec)
    //    the residual velocity after one second is 0.99^480 ~= 0.008
    //    -- aggressive enough to settle a tossed pile into a loose
    //    heap in about a second of wall time without Phase 5's
    //    contact-persistence sleeper. See phys_damping cvar docs.
    //
    //    WHY VELOCITY IS EXPLICIT NOW (#270). Phase 1 shipped this as
    //    position-Verlet, with the velocity left implicit in the
    //    position pair:
    //
    //        disp = (curr - prev) * damping
    //        curr = curr + disp + accel * sdt^2
    //
    //    which is algebraically correct and numerically fatal. The
    //    gravity term `accel * sdt^2` is an ABSOLUTE increment: at the
    //    default phys_substeps 8 of a 1/60 s frame it is
    //    9.81 * (1/480)^2 = 4.26e-5 m. float32 carries a 24-bit
    //    mantissa, so the spacing of representable values near a
    //    position y is y * 2^-23, and round-to-nearest swallows the
    //    increment whole once y * 2^-24 exceeds it -- above roughly
    //    1024 m. Past that altitude `curr + accel * sdt^2 == curr`
    //    bit for bit: the add is a complete no-op, the body coasts in
    //    uniform motion, and nothing anywhere reports that gravity has
    //    stopped existing.
    //
    //    The perverse part is that sdt^2 shrinks QUADRATICALLY in the
    //    substep count while the position magnitude does not, so
    //    raising phys_substeps -- the *accuracy* knob -- made the bug
    //    dramatically worse. Measured, float32 exact: gravity died
    //    above 65 km at 1 substep, above 1024 m at the default 8, and
    //    above 64 m at the cvar maximum of 32, where it was already
    //    7.5% wrong at ten metres.
    //
    //    Carrying velocity in m/s fixes that at the root, because the
    //    quantity we accumulate now scales with the MOTION instead of
    //    with the square of the timestep:
    //
    //        vel  = vel * damping + accel * sdt
    //        curr = curr + vel * sdt
    //
    //    `accel * sdt` at 32 substeps is 5.1e-3 m/s landing in a
    //    free-fall speed that climbs into the hundreds of m/s -- and
    //    critically, the ratio of increment to accumulator no longer
    //    depends on where the body is in the world, only on how long
    //    it has been falling (g*sdt / v = sdt / t). The substep
    //    inversion is gone with it: more substeps is once again
    //    strictly more accurate.
    //
    //    This preserves the Phase 1 damping semantics EXACTLY, not
    //    approximately. Write d_n = curr_n - prev_n for the old
    //    per-substep displacement and v_n = d_n / sdt. Then
    //
    //        d_{n+1} = damping * d_n + accel * sdt^2
    //                = sdt * (damping * v_n + accel * sdt)
    //                = sdt * v_{n+1}
    //
    //    -- literally the same recurrence divided through by sdt.
    //    `damping` is still a per-substep multiplier, 0.99 still
    //    bleeds 1% of velocity per substep, and the closed-form
    //    undamped free-fall drop after N substeps is still
    //    N*(N+1)/2 * accel * sdt^2. The single behavioural difference
    //    is what happens when sdt CHANGES between frames: the old
    //    form froze the per-substep displacement (so a frame-rate
    //    change silently rescaled the body's real velocity), the new
    //    one freezes the velocity, which is the physically correct
    //    choice. Smoke mode pins dt to 1/60 and the engine clamps it
    //    to <= 1/30 otherwise, so this only bites under a frame-time
    //    change, and then only in the direction of being right.
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (!s.alive) continue;
        if (s.p.inv_mass <= 0.0f) {
            // Kinematic -- skip integration. prev_pos pinned to
            // curr_pos and velocity pinned to 0 so both the stored
            // and the derived velocity read as motionless.
            s.p.prev_pos = s.p.curr_pos;
            s.p.velocity = glm::vec3{0.0f};
            continue;
        }
        s.p.velocity = s.p.velocity * damping + accel * sdt;
        s.p.prev_pos = s.p.curr_pos;
        s.p.curr_pos = s.p.curr_pos + s.p.velocity * sdt;
    }

    // 2. Resolve sphere-plane contact with the static y=0 ground.
    //    Penetration is the distance the sphere's bottom is below
    //    y=0; we push the centre back up by exactly that amount
    //    (kinematic plane: 100% of correction goes to the particle).
    //
    //    Phase 5 will introduce a non-zero restitution coefficient
    //    here; for Phase 1, the only "bounce" comes from the
    //    position-correction reflection: pushing the centre up while
    //    leaving prev_pos below the original y reflects the downward
    //    velocity component over the contact normal, giving a 100%
    //    elastic bounce by construction. Damping (above) bleeds that
    //    away to settle the particle eventually. This step, like
    //    steps 3 and 4, moves curr_pos only -- step 5 folds whatever
    //    the three of them did back into `velocity`, which is what
    //    keeps that reflection working now that velocity is stored
    //    rather than inferred.
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

    // 5. Fold the constraint solver's position corrections back into
    //    velocity -- the position-based-dynamics velocity update
    //    (Mueller 2007 section 3.4).
    //
    //    Steps 2-4 moved curr_pos without touching `velocity`. Left
    //    like that, contacts would stop affecting motion entirely: the
    //    Phase 1 integrator got its "bounce by construction" for free
    //    precisely BECAUSE velocity was the position difference, so
    //    projecting a particle out of an overlap reflected it
    //    (Jakobsen 2001). We reproduce that exactly by measuring how
    //    far the solver moved the particle away from where step 1
    //    predicted it would be, and adding that displacement over sdt
    //    to the velocity. Substituting the predictor back in,
    //
    //        v_new = v_pred + (pos_solved - pos_pred) / sdt
    //              = (pos_solved - prev_pos) / sdt
    //
    //    which is the old implicit velocity to the letter -- the same
    //    100% elastic reflection off the ground plane, the same
    //    contact-normal damping when a stack compresses.
    //
    //    `predicted` is recomputed here rather than cached in a
    //    scratch array because `prev_pos` is exactly the pre-predict
    //    position and `velocity` has not been touched since, so
    //    `prev_pos + velocity * sdt` re-evaluates to the very bits
    //    step 1 stored. That in turn makes the correction EXACTLY
    //    zero -- not merely small -- for every particle that hit no
    //    constraint, which is the property that keeps a free-falling
    //    body's velocity from being nibbled at by a spurious
    //    correction each substep. It holds only because pt_physics is
    //    compiled without fast-math (see the #error guard at the top
    //    of this file); a reassociating compiler could contract one
    //    copy of the expression into an FMA and not the other.
    for (std::uint32_t i = 0; i < kMaxParticles; ++i) {
        auto& s = slots_[i];
        if (!s.alive) continue;
        if (s.p.inv_mass <= 0.0f) continue;
        const glm::vec3 predicted = s.p.prev_pos + s.p.velocity * sdt;
        s.p.velocity += (s.p.curr_pos - predicted) * inv_sdt;
    }
}

// --- Phase 2a rigid bodies (#138) ------------------------------------------
//
// Pool management mirrors the Particle pool above (linear free-slot
// scan, generation-tagged 32-bit handles). Integration uses the XPBD
// scheme from Mueller 2020:
//   1. Predict next pos from the body's explicit linear velocity
//      (`pos += vel * sdt` after `vel = vel * damping + accel * sdt`).
//      This used to be implicit-velocity Verlet with an `accel*sdt^2`
//      increment; see the long #270 note in Substep above for why an
//      absolute per-substep increment cannot survive float32 at
//      altitude, and why the explicit form is the same recurrence
//      divided through by sdt.
//   2. Predict next orientation via the quaternion derivative q' =
//      0.5 * omega_quat * q, integrated by sdt.
//   3. Apply position-correction constraints (sphere-plane + pairwise
//      sphere-sphere using the body's bounding sphere). Phase 2a does
//      NOT generate orientation corrections from contacts -- for the
//      sphere collision shape that's mathematically correct (contact
//      normal aligns with the centre of mass), and for the conservative
//      box bounding sphere we explicitly accept that boxes won't get
//      angular impulses from contacts until Phase 2b lands SAT.
//   4. Fold the position correction back into the linear velocity and
//      finite-difference the angular velocity from the constrained
//      orientation pair, so any position correction automatically
//      damps the velocity along the contact normal -- the trick that
//      keeps stacks from exploding without an explicit restitution
//      model.

PhysicsSystem::RbHandle PhysicsSystem::AddRigidSphere(const glm::vec3& pos,
                                                     float radius,
                                                     float mass,
                                                     const glm::quat& orientation) {
    if (rb_alive_count_ >= kMaxRigidBodies) return kInvalidRbHandle;
    if (radius <= 0.0f) return kInvalidRbHandle;
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& s = rb_slots_[i];
        if (s.alive) continue;
        s.body            = RigidBody{};
        s.body.prev_pos   = pos;
        s.body.curr_pos   = pos;
        s.body.radius     = radius;
        s.body.shape      = Shape::Sphere;
        // Sphere body: half_extents stored as (r,r,r) for completeness
        // even though only `radius` is used at collision time. Keeps
        // the field non-zero so a later code path that introspects
        // half_extents on a Sphere doesn't trip a divide-by-zero.
        s.body.half_extents      = glm::vec3{radius, radius, radius};
        s.body.orientation       = orientation;
        s.body.prev_orientation  = orientation;
        s.body.omega             = glm::vec3{0.0f};
        if (mass > 0.0f) {
            s.body.inv_mass          = 1.0f / mass;
            s.body.inv_inertia_local = InvInertiaSphere(mass, radius);
        } else {
            // Kinematic: pinned in place, infinite mass / inertia.
            s.body.inv_mass          = 0.0f;
            s.body.inv_inertia_local = glm::vec3{0.0f};
        }
        s.alive    = 1;
        s.prim_id  = 0;
        ++rb_alive_count_;
        return MakeHandle(i, s.generation);
    }
    return kInvalidRbHandle;
}

PhysicsSystem::RbHandle PhysicsSystem::AddRigidBox(const glm::vec3& pos,
                                                  const glm::vec3& half_extents,
                                                  float mass,
                                                  const glm::quat& orientation) {
    if (rb_alive_count_ >= kMaxRigidBodies) return kInvalidRbHandle;
    if (half_extents.x <= 0.0f ||
        half_extents.y <= 0.0f ||
        half_extents.z <= 0.0f) return kInvalidRbHandle;
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& s = rb_slots_[i];
        if (s.alive) continue;
        s.body                  = RigidBody{};
        s.body.prev_pos         = pos;
        s.body.curr_pos         = pos;
        s.body.shape            = Shape::Box;
        s.body.half_extents     = half_extents;
        // Conservative bounding-sphere radius. Phase 2b SAT replaces
        // this with the real OBB at narrowphase time, but until then
        // boxes collide as their inscribing sphere -- they fall to
        // the ground but never lie flat. Per issue #138 MVP discipline.
        s.body.radius           = BoxBoundingRadius(half_extents);
        s.body.orientation      = orientation;
        s.body.prev_orientation = orientation;
        s.body.omega            = glm::vec3{0.0f};
        if (mass > 0.0f) {
            s.body.inv_mass          = 1.0f / mass;
            s.body.inv_inertia_local = InvInertiaBox(mass, half_extents);
        } else {
            s.body.inv_mass          = 0.0f;
            s.body.inv_inertia_local = glm::vec3{0.0f};
        }
        s.alive    = 1;
        s.prim_id  = 0;
        ++rb_alive_count_;
        return MakeHandle(i, s.generation);
    }
    return kInvalidRbHandle;
}

bool PhysicsSystem::RemoveRigidBody(RbHandle h) {
    if (h == kInvalidRbHandle) return false;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxRigidBodies) return false;
    auto& s = rb_slots_[i];
    if (!s.alive) return false;
    if (HandleGen(h) != s.generation) return false;
    s.alive      = 0;
    s.prim_id    = 0;
    s.body       = RigidBody{};
    s.generation = (s.generation + 1u) & kGenMask;
    if (s.generation == 0u) s.generation = 1u;
    --rb_alive_count_;
    return true;
}

const RigidBody* PhysicsSystem::GetRigidBody(RbHandle h) const {
    if (h == kInvalidRbHandle) return nullptr;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxRigidBodies) return nullptr;
    const auto& s = rb_slots_[i];
    if (!s.alive) return nullptr;
    if (HandleGen(h) != s.generation) return nullptr;
    return &s.body;
}

bool PhysicsSystem::SetRbPrimId(RbHandle h, std::uint32_t prim_id) {
    if (h == kInvalidRbHandle) return false;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxRigidBodies) return false;
    auto& s = rb_slots_[i];
    if (!s.alive) return false;
    if (HandleGen(h) != s.generation) return false;
    s.prim_id = prim_id;
    return true;
}

std::uint32_t PhysicsSystem::GetRbPrimId(RbHandle h) const {
    if (h == kInvalidRbHandle) return 0;
    const std::uint32_t i = HandleIndex(h);
    if (i >= kMaxRigidBodies) return 0;
    const auto& s = rb_slots_[i];
    if (!s.alive) return 0;
    if (HandleGen(h) != s.generation) return 0;
    return s.prim_id;
}

void PhysicsSystem::ForEachRigidBody(RbIterFn fn, void* user) const {
    if (fn == nullptr) return;
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        const auto& s = rb_slots_[i];
        if (!s.alive) continue;
        fn(MakeHandle(i, s.generation), s.body, s.prim_id, user);
    }
}

void PhysicsSystem::SubstepRigidBodies(float sdt, const glm::vec3& accel,
                                       float linear_damping,
                                       float angular_damping) {
    // See Substep(): sdt is strictly positive by the time we get here.
    const float inv_sdt = 1.0f / sdt;

    // 1. Predict next pos + orientation. Saves prev_pos / prev_orient
    //    so step 4 can refresh the linear velocity + omega.
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& s = rb_slots_[i];
        if (!s.alive) continue;
        auto& b = s.body;
        if (b.inv_mass <= 0.0f) {
            // Kinematic body: pin pos + orientation, pin the linear
            // velocity to 0, and leave omega untouched -- the user
            // might script a kinematic spin in a later phase; for
            // Phase 2a no external API writes omega, so it stays 0.
            b.prev_pos          = b.curr_pos;
            b.velocity          = glm::vec3{0.0f};
            b.prev_orientation  = b.orientation;
            continue;
        }
        // Linear: explicit-velocity integration with the same damping
        // the particle pool uses. Identical to Particle::Substep --
        // including the #270 reasoning for why the gravity term is
        // accel * sdt into a velocity rather than accel * sdt^2 into
        // a position -- so sphere-to-sphere pair contacts behave the
        // same whether the pair is two rigid bodies or two particles.
        b.velocity = b.velocity * linear_damping + accel * sdt;
        b.prev_pos = b.curr_pos;
        b.curr_pos = b.curr_pos + b.velocity * sdt;

        // Angular: damp omega then integrate the orientation by the
        // damped omega over sdt. Mueller 2020 Equation 12.
        b.omega                 *= angular_damping;
        const glm::quat old_q    = b.orientation;
        b.orientation            = IntegrateOrientation(b.orientation, b.omega, sdt);
        b.prev_orientation       = old_q;
    }

    // 2. Resolve sphere-plane (y=0) against the body's bounding sphere.
    //    Same code path as the particle pool; rigid-body bookkeeping
    //    differs only in that omega is left alone here -- the angular
    //    velocity update from the contact is handled by step 4's
    //    finite-difference (prev_orientation was saved before the
    //    constraint, so any orientation change introduced by future
    //    contact-rotational corrections shows up there).
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& s = rb_slots_[i];
        if (!s.alive) continue;
        if (s.body.inv_mass <= 0.0f) continue;
        const float pen = s.body.radius - s.body.curr_pos.y;
        if (pen > 0.0f) {
            s.body.curr_pos.y += pen;
        }
    }

    // 3. Pairwise sphere-sphere using bounding spheres. Identical math
    //    to Particle::Substep step 3, including the degenerate-overlap
    //    fallback. Per Phase 2a discipline this also handles box-box
    //    "collision" -- both boxes get pushed apart along the line
    //    between their centres as if they were the inscribing sphere
    //    of the OBB. Phase 2b replaces this with SAT.
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& si = rb_slots_[i];
        if (!si.alive) continue;
        for (std::uint32_t j = i + 1; j < kMaxRigidBodies; ++j) {
            auto& sj = rb_slots_[j];
            if (!sj.alive) continue;
            const glm::vec3 d = sj.body.curr_pos - si.body.curr_pos;
            const float min_d = si.body.radius + sj.body.radius;
            const float d2    = glm::dot(d, d);
            if (d2 >= min_d * min_d) continue;
            const float dist = std::sqrt(d2);
            glm::vec3 n;
            if (dist > 1e-6f) {
                n = d / dist;
            } else {
                n = glm::vec3{1.0f, 0.0f, 0.0f};
            }
            const float overlap = min_d - dist;
            const float inv_sum = si.body.inv_mass + sj.body.inv_mass;
            if (inv_sum <= 0.0f) continue;
            const float ki = si.body.inv_mass / inv_sum;
            const float kj = sj.body.inv_mass / inv_sum;
            si.body.curr_pos -= n * (overlap * ki);
            sj.body.curr_pos += n * (overlap * kj);
        }
    }

    // 4. Refresh both velocities from what the constraint solver did.
    //
    //    Linear: same position-based-dynamics velocity update as
    //    Particle::Substep step 5 -- add the solver's displacement
    //    away from the predicted position, over sdt, to the stored
    //    velocity. Substituting the predictor back in gives
    //    (curr_pos - prev_pos) / sdt, i.e. exactly the implicit
    //    velocity the Phase 2a form used, so a body bouncing off the
    //    ground plane reflects exactly as it did before. The
    //    correction is bit-exact zero for a body that touched nothing
    //    (see the fast-math note in Substep).
    //
    //    Angular: finite-difference omega from (prev_orientation,
    //    orientation). We materialise omega so external impulses
    //    (Phase 2b+) and the next substep's orientation integrator
    //    both have access to the up-to-date value. Phase 2a doesn't
    //    actually change orientation in steps 2-3 (sphere contacts
    //    don't generate torque), so this currently just preserves the
    //    freely-spinning omega; the round-trip is a no-op for spheres.
    //    Boxes don't pick up rotational impulses until Phase 2b lands
    //    either way.
    for (std::uint32_t i = 0; i < kMaxRigidBodies; ++i) {
        auto& s = rb_slots_[i];
        if (!s.alive) continue;
        auto& b = s.body;
        if (b.inv_mass <= 0.0f) continue;
        const glm::vec3 predicted = b.prev_pos + b.velocity * sdt;
        b.velocity += (b.curr_pos - predicted) * inv_sdt;
        b.omega = OrientationDiffOmega(b.prev_orientation, b.orientation, sdt);
    }
}
// --- end Phase 2a rigid bodies ---------------------------------------------

}  // namespace pt::physics

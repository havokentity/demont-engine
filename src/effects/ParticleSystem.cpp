// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// CPU side of the MVP particle system (issue #82). See the header for
// the scope-and-deferred-work summary; this file is the integration +
// emitter implementation.

#include "ParticleSystem.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace pt::effects {

namespace {

// Fixed-step clamp. Any dt > this cap is replaced by the cap so that a
// 200 ms stutter doesn't fling the integrator. 1/30 s matches the
// "minimum acceptable user-perceived frame" -- if the host is so
// loaded a frame takes longer than this, the particle field freezing
// for one tick reads better than catapulting.
constexpr float kMaxTimestep = 1.0f / 30.0f;

}  // namespace

ParticleSystem::ParticleSystem()
    : rng_(std::random_device{}()) {
    // Reserve at the soft cap so the contiguous storage never realloc-
    // copies under steady-state use. Vectors expand if the user raises
    // r_particles_max past 1024.
    particles_.reserve(max_particles_);
    // Only one continuous slot in MVP (snow). Reserving avoids the
    // single allocation on first StartContinuous.
    emitters_.reserve(4);
}

float ParticleSystem::Uniform11() {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    return d(rng_);
}

void ParticleSystem::Update(float dt) {
    // Step 0: clamp dt. See the kMaxTimestep comment above.
    if (dt <= 0.0f) return;
    if (dt > kMaxTimestep) dt = kMaxTimestep;

    // Step 1: pump continuous emitters. Convert (rate * dt) into
    // integer-spawn events via the accumulator so a fractional rate
    // doesn't drop particles long-term.
    for (auto& em : emitters_) {
        if (!em.active) continue;
        em.accumulator += em.rate * dt;
        if (em.accumulator < 1.0f) continue;
        const auto whole = static_cast<std::uint32_t>(em.accumulator);
        em.accumulator -= static_cast<float>(whole);
        EmitSpec one = em.spec;
        one.count = whole;
        EmitBurst(one);
    }

    // Step 2: integrate. Tight loop, no branches, no allocations.
    const float drag_eps = 1e-6f;
    for (auto& p : particles_) {
        // gravity (vector along -Y when gravity_y is negative).
        p.vel.y += p.gravity_y * dt;
        // analytic-decay drag: v(t+dt) = v(t) * exp(-drag*dt). When
        // drag is ~0 the exp(0)=1 branch is correct; we still pay the
        // exp call (~5 ns) per particle, which at the 1024-cap is ~5
        // microseconds total -- negligible against the integrator.
        if (p.drag > drag_eps) {
            const float decay = std::exp(-p.drag * dt);
            p.vel *= decay;
        }
        p.pos += p.vel * dt;
        p.age += dt;

        // Animate color + size between spawn and end values, parameter
        // t = age / lifetime in [0, 1]. We rebuild the START values
        // implicitly by storing the end and lerping the CURRENT field
        // value toward the end with weight dt/(lifetime - age) -- but
        // that drifts on accumulated dt. Cheaper + cleaner: just store
        // both endpoints and recompute the lerp from scratch each
        // frame. Spawn-time we copy color/size into BOTH the live
        // field and... we don't have spawn copies. Compromise: cache
        // the start values in spare slots? Just lerp using lifetime
        // and current t; for that we need the spawn values, which we
        // store implicitly via the lerp formula and a separate spawn
        // snapshot would double memory. The MVP-acceptable path is:
        // recompute the lerp t -> use a separate spawn-color member
        // we DO have... no we don't. Take 2: lerp incrementally each
        // frame from "current" toward "end" by alpha = dt/(remaining).
        // remaining = lifetime - age; when remaining > 0 this is
        // stable. As age -> lifetime this snaps to end.
        const float remaining = p.lifetime - p.age;
        if (remaining > 1e-4f) {
            const float a = std::clamp(dt / remaining, 0.0f, 1.0f);
            p.color    = p.color    + (p.end_color - p.color) * a;
            p.size     = p.size     + (p.end_size  - p.size)  * a;
        }
    }

    // Step 3: compact-out dead particles. Erase-if pattern keeps the
    // vector dense, which matters because the engine's per-frame
    // upload is the live region [0, size()) and a sparse vector would
    // either bloat the upload or need a separate active-mask.
    auto new_end = std::remove_if(
        particles_.begin(), particles_.end(),
        [](const Particle& p) { return p.age >= p.lifetime; });
    particles_.erase(new_end, particles_.end());
}

std::uint32_t ParticleSystem::EmitBurst(const EmitSpec& spec) {
    if (spec.count == 0) return 0;
    const std::uint32_t cap    = max_particles_;
    const std::uint32_t live   = static_cast<std::uint32_t>(particles_.size());
    if (live >= cap) return 0;
    const std::uint32_t budget = cap - live;
    const std::uint32_t want   = std::min(spec.count, budget);

    particles_.reserve(particles_.size() + want);
    for (std::uint32_t i = 0; i < want; ++i) {
        Particle p{};
        // Position: emit from a uniform-sphere offset around spec.pos
        // when pos_radius > 0. For the MVP "uniform-on-sphere" is
        // a cube-sample-then-cull, which is simple and bounded; with
        // a small radius the bias from rejecting outside-corners is
        // not visible. Cube-only fallback if radius is 0.
        glm::vec3 off{0.0f};
        if (spec.pos_radius > 0.0f) {
            for (int tries = 0; tries < 4; ++tries) {
                glm::vec3 t{Uniform11(), Uniform11(), Uniform11()};
                if (glm::dot(t, t) <= 1.0f) { off = t; break; }
            }
            // If the 4-try rejection loop never hit, off stays at the
            // last sample (worst case inside a 2x cube, still local).
            off *= spec.pos_radius;
        }
        p.pos = spec.pos + off;

        // Velocity: mean + uniform spread per axis.
        p.vel = spec.vel_mean + glm::vec3{
            Uniform11() * spec.vel_spread.x,
            Uniform11() * spec.vel_spread.y,
            Uniform11() * spec.vel_spread.z};

        // Per-particle randomized lifetime.
        const float lt_jitter = (spec.lifetime_spread > 0.0f)
                                    ? Uniform11() * spec.lifetime_spread
                                    : 0.0f;
        p.lifetime = std::max(0.05f, spec.lifetime_mean + lt_jitter);
        p.age      = 0.0f;

        p.color     = spec.color;
        p.end_color = spec.end_color;
        p.size      = spec.size;
        p.end_size  = spec.end_size;
        p.drag      = spec.drag;
        p.gravity_y = spec.gravity_y;

        particles_.push_back(p);
    }
    return want;
}

void ParticleSystem::StartContinuous(const char* name, const EmitSpec& spec,
                                     float rate) {
    // Hash on name only because the MVP has at most one continuous
    // emitter at a time. We linearly scan; with N <= 4 there's no
    // payoff in moving to a map.
    for (auto& em : emitters_) {
        if (em.spec.count == 0 && em.spec.lifetime_mean == spec.lifetime_mean &&
            em.spec.size == spec.size && em.active) {
            // This match heuristic is intentionally loose -- the MVP
            // names a continuous emitter by its preset (one snow at a
            // time). A future "named emitter" API would replace this.
            em.spec       = spec;
            em.rate       = rate;
            em.active     = true;
            em.accumulator = 0.0f;
            (void)name;
            return;
        }
    }
    ContinuousEmitter em{};
    em.spec        = spec;
    em.spec.count  = 0;   // continuous emitters spawn N from rate*dt, not count
    em.rate        = rate;
    em.active      = true;
    em.accumulator = 0.0f;
    emitters_.push_back(em);
    (void)name;
}

std::uint32_t ParticleSystem::StopContinuous(const char* /*name*/) {
    // MVP: stop every continuous emitter unconditionally. A real
    // named-slot map is a follow-up.
    std::uint32_t n = 0;
    for (auto& em : emitters_) {
        if (em.active) { em.active = false; ++n; }
    }
    return n;
}

void ParticleSystem::Clear() noexcept {
    particles_.clear();
    emitters_.clear();
}

// ---- Presets ---------------------------------------------------------------
//
// Each preset is shaped by what the issue body and PR description ask
// for: smoke = grey puff, spark = orange burst, snow = continuous area
// emitter. Numbers are physically motivated where they have a real
// analogue (gravity 9.81, smoke buoyancy ~0.5 m/s^2 upward).

EmitSpec ParticleSystem::PresetSmoke(const glm::vec3& at) {
    EmitSpec s{};
    s.pos             = at;
    s.pos_radius      = 0.15f;                          // 15 cm sphere
    s.vel_mean        = {0.0f, 0.4f, 0.0f};             // slow upward
    s.vel_spread      = {0.3f, 0.2f, 0.3f};
    s.color           = {0.65f, 0.65f, 0.7f, 0.8f};     // cool grey
    s.end_color       = {0.4f,  0.4f,  0.45f, 0.0f};    // darker + fade out
    s.size            = 0.10f;                          // 10 cm puff
    s.end_size        = 0.55f;                          // grows to ~55 cm
    s.lifetime_mean   = 2.0f;
    s.lifetime_spread = 0.4f;
    s.drag            = 0.8f;                           // air resistance
    s.gravity_y       = 0.5f;                           // buoyant smoke
    s.count           = 24;
    return s;
}

EmitSpec ParticleSystem::PresetSpark(const glm::vec3& at) {
    EmitSpec s{};
    s.pos             = at;
    s.pos_radius      = 0.02f;
    s.vel_mean        = {0.0f, 1.5f, 0.0f};
    s.vel_spread      = {2.5f, 1.5f, 2.5f};             // hot outward burst
    s.color           = {6.0f, 3.0f, 0.8f, 1.0f};       // bright orange (HDR) -- brighter so the tonemap doesn't squash them
    s.end_color       = {3.0f, 0.6f, 0.1f, 0.0f};       // deep red, fade out
    s.size            = 0.05f;                          // 5 cm splat -- big enough to clear a few pixels at typical viewing distance
    s.end_size        = 0.02f;                          // shrink
    s.lifetime_mean   = 0.6f;
    s.lifetime_spread = 0.2f;
    s.drag            = 0.4f;
    s.gravity_y       = -9.81f;                         // real gravity
    s.count           = 48;
    return s;
}

EmitSpec ParticleSystem::PresetSnow(const glm::vec3& centre) {
    EmitSpec s{};
    s.pos             = centre + glm::vec3{0.0f, 6.0f, 0.0f};   // 6 m overhead
    s.pos_radius      = 5.0f;                                    // 10 m wide disc-ish
    s.vel_mean        = {0.0f, -1.2f, 0.0f};                     // slow fall
    s.vel_spread      = {0.4f, 0.3f, 0.4f};
    s.color           = {1.0f, 1.0f, 1.05f, 0.9f};
    s.end_color       = {0.9f, 0.9f, 0.95f, 0.0f};
    s.size            = 0.04f;
    s.end_size        = 0.04f;                                  // doesn't grow
    s.lifetime_mean   = 6.0f;                                   // long fall
    s.lifetime_spread = 1.5f;
    s.drag            = 1.6f;                                   // air resistance + wind feel
    s.gravity_y       = -9.81f;
    s.count           = 0;                                       // continuous, rate-driven
    return s;
}

}  // namespace pt::effects

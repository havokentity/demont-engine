// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt::sph::SmokeSPH unit tests (Fluid Phase 3, #22).
//
// Pins the core contracts of the CPU SPH smoke solver. The path-traced
// rendering quality has its own golden cell -- these tests pin the
// SOLVER mechanics so a refactor can't silently break the physics
// without surfacing a unit failure first.
//
// Coverage:
//   * Clear() resets to an empty state
//   * SetMaxParticles caps + reallocates the pool
//   * SpawnParticle returns false at cap, true below cap
//   * Step is a no-op when nothing is alive AND no emitters
//   * Emitters spawn particles at the configured rate
//   * Gravity drives a single isolated particle downward
//   * Buoyancy (T_p > T_air) drives a hot particle upward against gravity
//   * Wind drives particles toward the configured wind speed
//   * QueueShockwave applies a radial impulse exactly once
//   * PackForGpu produces a contiguous 2-float4-per-particle layout
//   * (wave-9 sph-3b) Curl-noise perturbs particle motion deterministically
//   * (wave-9 sph-3b) High-viscosity XSPH stays stable (no explosion)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/physics/SmokeSPH.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using pt::sph::SmokeSPH;

TEST_CASE("SmokeSPH::Clear resets a fresh solver to empty") {
    SmokeSPH s;
    s.Clear();
    CHECK(s.AliveCount() == 0u);
    CHECK(s.MaxParticles() == 1024u);  // default
}

TEST_CASE("SmokeSPH::SetMaxParticles caps the pool") {
    SmokeSPH s;
    s.SetMaxParticles(32);
    CHECK(s.MaxParticles() == 32u);
    // Fill to cap.
    for (int i = 0; i < 32; ++i) {
        SmokeSPH::Particle p;
        p.pos = glm::vec3(static_cast<float>(i), 0.0f, 0.0f);
        p.radius = 0.1f;
        CHECK(s.SpawnParticle(p));
    }
    CHECK(s.AliveCount() == 32u);
    // 33rd spawn rejected.
    SmokeSPH::Particle p;
    CHECK_FALSE(s.SpawnParticle(p));
    CHECK(s.AliveCount() == 32u);
}

TEST_CASE("SmokeSPH::Step is a no-op when idle") {
    SmokeSPH s;
    s.SetMaxParticles(8);
    s.Step(1.0f / 60.0f);     // no emitters, no particles -> no-op
    CHECK(s.AliveCount() == 0u);
}

TEST_CASE("SmokeSPH emitter spawns particles at the configured rate") {
    SmokeSPH s;
    s.SetMaxParticles(256);
    std::vector<SmokeSPH::EmitterParams> ems;
    SmokeSPH::EmitterParams em;
    em.pos = glm::vec3(0.0f);
    em.velocity = glm::vec3(0.0f, 1.0f, 0.0f);
    em.radius = 0.5f;
    em.spawn_rate = 60.0f;    // 60/s -> 1/frame at 60 fps
    em.lifetime_s = 10.0f;
    em.temperature_kelvin = 600.0f;
    em.particle_radius = 0.1f;
    em.particle_mass = 0.02f;
    ems.push_back(em);
    s.SetEmitters(std::move(ems));
    // Tick for 1 second of fixed-substep simulation. With 60/s
    // spawn rate and 120 substeps/s, the emitter accumulates
    // 60 particles over the full second (60 = 60 * 1.0).
    for (int i = 0; i < 60; ++i) {
        s.Step(1.0f / 60.0f);
    }
    CHECK(s.AliveCount() >= 50u);   // allow drift for substep accumulation
    CHECK(s.AliveCount() <= 70u);
}

TEST_CASE("SmokeSPH gravity drives an isolated particle downward") {
    SmokeSPH s;
    s.SetMaxParticles(2);
    auto& cfg = s.MutableConfig();
    cfg.buoyancy_scale = 0.0f;     // pure gravity
    cfg.air_drag       = 0.0f;     // no drag
    cfg.wind           = glm::vec3(0.0f);
    cfg.curl_strength  = 0.0f;     // wave-9: isolate gravity (no turbulence)
    SmokeSPH::Particle p;
    p.pos = glm::vec3(0.0f, 10.0f, 0.0f);
    p.vel = glm::vec3(0.0f);
    p.radius = 0.1f;
    REQUIRE(s.SpawnParticle(p));
    // 0.5 s of free fall. expected: y(0.5) = 10 - 0.5 * 9.81 * 0.25 ~= 8.77
    for (int i = 0; i < 30; ++i) s.Step(1.0f / 60.0f);
    const auto& q = s.Particles()[0];
    REQUIRE(s.IsAlive(0));
    CHECK(q.pos.y < 9.5f);   // has fallen
    CHECK(q.pos.y > 8.5f);   // not too far (drag-free Verlet)
}

TEST_CASE("SmokeSPH buoyancy lifts a hot isolated particle") {
    SmokeSPH s;
    s.SetMaxParticles(2);
    auto& cfg = s.MutableConfig();
    cfg.buoyancy_scale = 1.0f;
    cfg.air_drag       = 0.0f;
    cfg.wind           = glm::vec3(0.0f);
    cfg.thermal_decay  = 0.0f;     // no cooling so buoyancy stays max
    SmokeSPH::Particle p;
    p.pos = glm::vec3(0.0f, 2.0f, 0.0f);
    p.radius = 0.1f;
    std::size_t idx = 0;
    REQUIRE(s.SpawnParticle(p, &idx));
    // SpawnParticle defaults to ambient temperature; we need to
    // emit via an emitter to get a hot particle.
    SmokeSPH s2;
    s2.SetMaxParticles(8);
    auto& cfg2 = s2.MutableConfig();
    cfg2.buoyancy_scale = 1.0f;
    cfg2.air_drag       = 0.0f;
    cfg2.wind           = glm::vec3(0.0f);
    cfg2.thermal_decay  = 0.0f;
    cfg2.curl_strength  = 0.0f;    // wave-9: isolate buoyancy
    cfg2.buoyancy_variation = 0.0f;
    std::vector<SmokeSPH::EmitterParams> ems;
    SmokeSPH::EmitterParams em;
    em.pos = glm::vec3(0.0f, 2.0f, 0.0f);
    em.velocity = glm::vec3(0.0f);
    em.radius = 0.01f;                    // tiny jitter -> reproducible
    em.spawn_rate = 240.0f;               // burst
    em.lifetime_s = 100.0f;
    em.temperature_kelvin = 800.0f;       // hot -> ~1.7x buoyancy vs gravity
    em.particle_radius = 0.1f;
    em.particle_mass = 0.02f;
    ems.push_back(em);
    s2.SetEmitters(std::move(ems));
    // 1 substep at 1/120 to spawn ~2 particles.
    s2.Step(1.0f / 120.0f);
    // Disable spawn now so we observe the spawned particles' motion only.
    for (auto& e : const_cast<std::vector<SmokeSPH::EmitterParams>&>(s2.Emitters())) {
        (void)e;
    }
    // We can't mutate emitters via the const API; just clear them
    // via SetEmitters to drain.
    std::vector<SmokeSPH::EmitterParams> none;
    s2.SetEmitters(std::move(none));
    // Now run 1 second under buoyancy.
    for (int i = 0; i < 60; ++i) s2.Step(1.0f / 60.0f);
    // Find any alive particle and check it rose. With buoyancy
    // scale 1, T_p=800, T_air=293, net upward acc = 9.81*(1.73-1)
    // = ~7.2 m/s^2. Over 1 s starting from rest at y=2:
    // y(1) ~= 2 + 0.5 * 7.2 = 5.6
    bool found = false;
    for (std::size_t i = 0; i < s2.Particles().size(); ++i) {
        if (!s2.IsAlive(i)) continue;
        const auto& q2 = s2.Particles()[i];
        if (q2.pos.y > 2.5f) {
            found = true;
            break;
        }
    }
    CHECK(found);
}

TEST_CASE("SmokeSPH wind drives a still particle toward wind speed") {
    SmokeSPH s;
    s.SetMaxParticles(2);
    auto& cfg = s.MutableConfig();
    cfg.gravity        = glm::vec3(0.0f);  // disable gravity
    cfg.buoyancy_scale = 0.0f;
    cfg.viscosity      = 0.0f;
    cfg.wind           = glm::vec3(3.0f, 0.0f, 0.0f);   // 3 m/s along +X
    cfg.air_drag       = 5.0f;                          // strong drag for quick settle
    cfg.curl_strength  = 0.0f;     // wave-9: isolate wind drag
    SmokeSPH::Particle p;
    p.pos = glm::vec3(0.0f);
    p.vel = glm::vec3(0.0f);
    p.radius = 0.1f;
    REQUIRE(s.SpawnParticle(p));
    // 0.5 s. With drag k=5, the particle's vx -> wind_x * (1 - exp(-k*t))
    // ~= 3 * (1 - exp(-2.5)) ~= 3 * 0.918 = 2.75. Position is the
    // integral. Should be moving in +X.
    for (int i = 0; i < 30; ++i) s.Step(1.0f / 60.0f);
    const auto& q = s.Particles()[0];
    REQUIRE(s.IsAlive(0));
    CHECK(q.pos.x > 0.5f);
}

TEST_CASE("SmokeSPH::QueueShockwave applies a radial impulse") {
    SmokeSPH s;
    s.SetMaxParticles(2);
    auto& cfg = s.MutableConfig();
    cfg.gravity        = glm::vec3(0.0f);
    cfg.buoyancy_scale = 0.0f;
    cfg.air_drag       = 0.0f;
    cfg.viscosity      = 0.0f;
    cfg.wind           = glm::vec3(0.0f);
    cfg.curl_strength  = 0.0f;     // wave-9: isolate shockwave impulse
    SmokeSPH::Particle p;
    p.pos = glm::vec3(1.0f, 0.0f, 0.0f);   // 1m to the right of origin
    p.vel = glm::vec3(0.0f);
    p.radius = 0.1f;
    REQUIRE(s.SpawnParticle(p));
    s.QueueShockwave(glm::vec3(0.0f), 10.0f, 5.0f);   // 10 J at origin, 5 m radius
    s.Step(1.0f / 60.0f);
    const auto& q = s.Particles()[0];
    REQUIRE(s.IsAlive(0));
    // After impulse + 1 frame, particle should have moved further
    // in +X (away from origin).
    CHECK(q.pos.x > 1.0f);
    CHECK(q.vel.x > 0.0f);
}

TEST_CASE("SmokeSPH::PackForGpu produces 2-float4-per-particle layout") {
    SmokeSPH s;
    s.SetMaxParticles(4);
    SmokeSPH::Particle p;
    p.pos = glm::vec3(1.0f, 2.0f, 3.0f);
    p.radius = 0.5f;
    p.age = 0.0f;
    REQUIRE(s.SpawnParticle(p));
    float buf[16] = {};   // 4 particles' worth of float4s
    const auto count = s.PackForGpu(buf, 16, 1.0f, 1.0f);
    CHECK(count == 1u);
    // v0.xyz = pos
    CHECK(buf[0] == doctest::Approx(1.0f));
    CHECK(buf[1] == doctest::Approx(2.0f));
    CHECK(buf[2] == doctest::Approx(3.0f));
    // v0.w = radius * scale
    CHECK(buf[3] == doctest::Approx(0.5f));
    // v1.x = density * fade (fresh particle -> fade=1) -- positive
    CHECK(buf[4] > 0.0f);
    // v1.y = normalised age (0 at spawn)
    CHECK(buf[5] == doctest::Approx(0.0f));
}

// --- Wave 9 sph-3b -----------------------------------------------------

TEST_CASE("SmokeSPH curl-noise perturbs particle motion (deterministically)") {
    // An isolated particle in zero gravity / buoyancy / wind feels ONLY
    // the curl-noise force. With curl off it stays put; with curl on it
    // is deflected. The deflection must be deterministic (bit-identical
    // across two runs) so goldens stay reproducible.
    auto run = [](float curl) {
        SmokeSPH s;
        s.SetMaxParticles(2);
        auto& cfg = s.MutableConfig();
        cfg.gravity        = glm::vec3(0.0f);
        cfg.buoyancy_scale = 0.0f;
        cfg.air_drag       = 0.0f;
        cfg.viscosity      = 0.0f;
        cfg.wind           = glm::vec3(0.0f);
        cfg.curl_strength  = curl;
        SmokeSPH::Particle p;
        p.pos = glm::vec3(3.0f, 3.0f, 3.0f);  // off-origin so the noise is non-trivial
        p.vel = glm::vec3(0.0f);
        REQUIRE(s.SpawnParticle(p));
        for (int i = 0; i < 30; ++i) s.Step(1.0f / 60.0f);
        return s.Particles()[0].pos;
    };
    const glm::vec3 still   = run(0.0f);
    const glm::vec3 curled1 = run(8.0f);
    const glm::vec3 curled2 = run(8.0f);
    // Curl off -> no force -> particle does not move.
    CHECK(glm::length(still - glm::vec3(3.0f, 3.0f, 3.0f)) < 1e-4f);
    // Curl on -> particle is deflected away from rest.
    CHECK(glm::length(curled1 - glm::vec3(3.0f, 3.0f, 3.0f)) > 0.05f);
    // Deterministic: two runs land bit-identical. Use exact float
    // equality (not doctest::Approx) -- the curl-noise path is a pure
    // function of (position, time) with no RNG or float-order ambiguity,
    // so re-running the same inputs reproduces the same bits exactly.
    // This is what goldens rely on, so the assertion matches the wording.
    CHECK(curled1.x == curled2.x);
    CHECK(curled1.y == curled2.y);
    CHECK(curled1.z == curled2.z);
}

TEST_CASE("SmokeSPH high-viscosity XSPH stays stable (no explosion)") {
    // The wave-8 explicit Müller viscosity force exploded the column at
    // high mu (particles flung to y~100 m). The wave-9 bounded XSPH blend
    // must instead SMOOTH the cloud: with a strong mu the live particles
    // stay within a sane domain (no detonation) and the population is not
    // decimated by particles flying past the cull bounds.
    SmokeSPH s;
    s.SetMaxParticles(512);
    auto& cfg = s.MutableConfig();
    cfg.viscosity         = 1.0f;     // very high -- would explode pre-fix
    cfg.pressure_stiffness = 1.0f;
    cfg.curl_strength     = 0.0f;     // isolate viscosity stability
    std::vector<SmokeSPH::EmitterParams> ems;
    SmokeSPH::EmitterParams em;
    em.pos = glm::vec3(0.0f, 2.0f, 0.0f);
    em.velocity = glm::vec3(0.0f, 3.0f, 0.0f);
    em.radius = 0.5f;
    em.spawn_rate = 300.0f;
    em.lifetime_s = 100.0f;           // long life so nothing ages out
    em.temperature_kelvin = 600.0f;
    em.particle_radius = 0.2f;
    em.particle_mass = 0.02f;
    ems.push_back(em);
    s.SetEmitters(std::move(ems));
    for (int i = 0; i < 60; ++i) s.Step(1.0f / 60.0f);
    // Population survived (not flung past the +/-1km cull bounds).
    CHECK(s.AliveCount() > 100u);
    // No particle detonated to an absurd position.
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < s.Particles().size(); ++i) {
        if (!s.IsAlive(i)) continue;
        const auto& p = s.Particles()[i].pos;
        max_abs = std::max(max_abs, std::max(std::abs(p.x), std::max(std::abs(p.y), std::abs(p.z))));
    }
    // A stable cohesive plume stays compact (well under 50 m in 1 s);
    // the pre-fix explicit viscosity sent particles past 100 m.
    CHECK(max_abs < 50.0f);
}

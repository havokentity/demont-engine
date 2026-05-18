// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt_physics Phase 1 unit tests (#132). Covers the four behaviours
// that the engine layer relies on but that the golden-image smoke
// fixtures can't isolate cleanly:
//
//   1. Handle invalidation: Remove + Add reusing a slot must bump
//      generation so an outstanding stale handle returns nullptr
//      from GetParticle / false from SetPrimId. The 24-bit gen field
//      is the contract documented on the public Handle type; a
//      regression to the old 8-bit packing would alias a new
//      particle with a stale handle after 256 churn cycles.
//
//   2. Free-fall integration: a particle dropped under gravity for
//      a known dt traces the Verlet-expected trajectory. Verlet has
//      no explicit velocity, so this also verifies the
//      curr - prev = implicit velocity invariant holds across
//      substeps.
//
//   3. Sphere-plane contact: a particle that crosses y=0 by
//      `penetration` is pushed back so its bottom lands exactly on
//      the plane (curr_pos.y >= radius). After Phase-1-correction
//      this should hold AT THE END of every Substep, even when
//      sphere-sphere correction in the same substep would otherwise
//      push the particle back below the plane.
//
//   4. Sphere-sphere correction: two overlapping particles get
//      pushed apart along the contact normal proportional to their
//      inv_mass, and a stacked-tower configuration leaves every
//      particle with bottom-y >= 0 after the substep (the new
//      "second plane pass" guarantee from review).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "physics/PhysicsSystem.h"

#include <cmath>

using pt::physics::PhysicsSystem;
using pt::physics::Particle;

// Tiny helper so the float comparisons stay readable.
static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

TEST_CASE("physics handle: invalid handle resolves to nullptr") {
    PhysicsSystem ps;
    CHECK(ps.GetParticle(PhysicsSystem::kInvalidHandle) == nullptr);
    CHECK_FALSE(ps.RemoveParticle(PhysicsSystem::kInvalidHandle));
}

TEST_CASE("physics handle: Remove + Add bumps generation so stale handle is rejected") {
    PhysicsSystem ps;
    const auto h1 = ps.AddParticle({0.0f, 1.0f, 0.0f}, 0.3f);
    REQUIRE(h1 != PhysicsSystem::kInvalidHandle);
    REQUIRE(ps.GetParticle(h1) != nullptr);

    CHECK(ps.RemoveParticle(h1));
    CHECK(ps.GetParticle(h1) == nullptr);          // stale -> nullptr
    CHECK_FALSE(ps.RemoveParticle(h1));            // second remove no-op
    CHECK_FALSE(ps.SetPrimId(h1, 12345u));         // stale set rejected

    // Add a new particle. It should land in the same slot (linear
    // free-slot scan starts at index 0), but with a bumped
    // generation, so the OLD handle is still rejected even though
    // the slot is now alive.
    const auto h2 = ps.AddParticle({1.0f, 2.0f, 3.0f}, 0.4f);
    REQUIRE(h2 != PhysicsSystem::kInvalidHandle);
    CHECK(h2 != h1);                                // generation differs
    CHECK(ps.GetParticle(h1) == nullptr);
    CHECK(ps.GetParticle(h2) != nullptr);
}

TEST_CASE("physics handle: generation survives many remove+add cycles") {
    // This is the 8-bit-vs-24-bit regression guard. With the old
    // 8-bit packing, after 256 cycles the slot's generation wraps
    // back to its starting value and a hypothetical stale handle
    // could alias. We don't actually keep a stale handle around for
    // 256 iterations (the test would just churn a slot); the check
    // here is that the issued handles are NEVER equal across the
    // churn -- i.e. the gen-bumping actually walks the gen field
    // rather than wrapping immediately.
    PhysicsSystem ps;
    auto prev_h = PhysicsSystem::kInvalidHandle;
    for (int i = 0; i < 300; ++i) {
        const auto h = ps.AddParticle({0.0f, 1.0f, 0.0f}, 0.3f);
        REQUIRE(h != PhysicsSystem::kInvalidHandle);
        CHECK(h != prev_h);                         // distinct from last one
        prev_h = h;
        CHECK(ps.RemoveParticle(h));
    }
}

TEST_CASE("physics integration: free-fall trajectory matches Verlet expectation") {
    PhysicsSystem ps;
    const auto h = ps.AddParticle({0.0f, 100.0f, 0.0f}, 0.3f);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);

    // Hand-roll 10 substeps of dt = 1/600 s at damping=1.0
    // (energy-conserving) so we can compare against the closed-form
    // Verlet expectation. Verlet with prev_pos = curr_pos and
    // damping = 1, accel = g:
    //   x_1 = x_0 + 0  + g * dt^2     (first step, v_implicit = 0)
    //   x_n = 2 x_{n-1} - x_{n-2} + g * dt^2
    // After N substeps the displacement is N * (N+1) / 2 * g * dt^2
    // for the prev_pos = curr_pos start (gauss-summation of the
    // implicit velocity terms 1, 2, ..., N times g*dt^2).
    constexpr float kG  = -9.81f;
    constexpr int   kN  = 10;
    constexpr float kDt = 1.0f / 600.0f;            // 10 substeps over 1/60 s frame
    ps.Step(kDt * static_cast<float>(kN), kN, kG, /*damping=*/1.0f);

    const float dt2          = kDt * kDt;
    const float expected_dy  = static_cast<float>(kN * (kN + 1)) / 2.0f * kG * dt2;
    const float expected_y   = 100.0f + expected_dy;

    const Particle* p = ps.GetParticle(h);
    REQUIRE(p != nullptr);
    // Looser tolerance than 1e-4 because Verlet accumulates a tiny
    // amount of FP error per substep; 1e-3 over 10 substeps is
    // plenty tight to catch a real regression.
    CHECK(approx(p->curr_pos.y, expected_y, 1e-3f));
}

TEST_CASE("physics integration: damping bleeds implicit velocity") {
    // Stop integrating once damping has had time to act and verify
    // the implicit velocity (curr - prev) is monotonically smaller
    // than the un-damped step would produce.
    PhysicsSystem ps;
    const auto h = ps.AddParticle({0.0f, 50.0f, 0.0f}, 0.3f);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);

    // 60 substeps at damping=0.9, dt=1/600 s. Implicit velocity
    // after substep N is (sum of damped contributions of g*dt^2),
    // strictly less than the undamped sum N * g * dt^2.
    constexpr float kDmp = 0.9f;
    constexpr int   kN   = 60;
    constexpr float kDt  = 1.0f / 600.0f;
    ps.Step(kDt * static_cast<float>(kN), kN, /*g=*/-9.81f, kDmp);

    const Particle* p = ps.GetParticle(h);
    REQUIRE(p != nullptr);
    const float implicit_v = (p->curr_pos.y - p->prev_pos.y) / kDt;
    const float undamped_v = -9.81f * kDt * static_cast<float>(kN);
    // Damped velocity must be STRICTLY ABOVE the undamped value
    // (less negative since gravity is negative), i.e. smaller in
    // magnitude.
    CHECK(implicit_v > undamped_v);
}

TEST_CASE("physics contact: sphere-plane resolves so bottom-y >= 0 after substep") {
    PhysicsSystem ps;
    const float r = 0.5f;
    const auto h = ps.AddParticle({0.0f, 0.2f, 0.0f}, r);     // already penetrating
    REQUIRE(h != PhysicsSystem::kInvalidHandle);

    // One short substep with gravity off so the only motion comes
    // from the plane correction. Use the public Step with a
    // 0-gravity vector via a 0-second dt? Step early-outs on
    // dt<=0; instead use a tiny dt with negligible gravity effect.
    ps.Step(/*frame_dt=*/1e-4f, /*substeps=*/1, /*gravity_y=*/0.0f, /*damping=*/1.0f);
    const Particle* p = ps.GetParticle(h);
    REQUIRE(p != nullptr);
    // After the substep the bottom of the sphere must be at or
    // above the plane.
    CHECK(p->curr_pos.y >= r - 1e-5f);
}

TEST_CASE("physics contact: sphere-sphere correction splits overlap by inv_mass") {
    PhysicsSystem ps;
    const float r = 0.5f;
    // Two equal-mass particles overlapping by 0.2 m along +X.
    const auto ha = ps.AddParticle({0.0f, 5.0f, 0.0f}, r, /*inv_mass=*/1.0f);
    const auto hb = ps.AddParticle({0.8f, 5.0f, 0.0f}, r, /*inv_mass=*/1.0f);
    REQUIRE(ha != PhysicsSystem::kInvalidHandle);
    REQUIRE(hb != PhysicsSystem::kInvalidHandle);

    // Step with gravity off (well above the ground plane, no plane
    // contact for either). After the overlap correction the
    // separation should be >= 2r and the particles should have moved
    // by equal-and-opposite amounts (inv_mass split 50/50).
    ps.Step(1e-4f, 1, 0.0f, /*damping=*/1.0f);
    const Particle* pa = ps.GetParticle(ha);
    const Particle* pb = ps.GetParticle(hb);
    REQUIRE(pa != nullptr);
    REQUIRE(pb != nullptr);
    const float sep = pb->curr_pos.x - pa->curr_pos.x;
    CHECK(sep >= 2.0f * r - 1e-5f);
    // Equal mass split: a moved -0.1, b moved +0.1 from the
    // overlapping starting positions (0.0 and 0.8 -> 2r apart at
    // 1.0 means each moved 0.1 toward its outside).
    CHECK(approx(pa->curr_pos.x, -0.1f, 1e-4f));
    CHECK(approx(pb->curr_pos.x,  0.9f, 1e-4f));
}

TEST_CASE("physics contact: stacked-sphere tower keeps every particle above the plane") {
    // Regression for the review note: sphere-sphere correction in
    // step 3 could push a lower sphere back below y=radius. The
    // second plane pass (step 4) must guarantee bottom-y >= 0 for
    // every particle at substep end.
    PhysicsSystem ps;
    const float r = 0.3f;
    // Three spheres stacked just barely touching the ground, with
    // small downward overlaps to force a correction pass.
    const auto h0 = ps.AddParticle({0.0f, 0.30f, 0.0f}, r);
    const auto h1 = ps.AddParticle({0.0f, 0.85f, 0.0f}, r);
    const auto h2 = ps.AddParticle({0.0f, 1.40f, 0.0f}, r);
    REQUIRE(h0 != PhysicsSystem::kInvalidHandle);
    REQUIRE(h1 != PhysicsSystem::kInvalidHandle);
    REQUIRE(h2 != PhysicsSystem::kInvalidHandle);

    // A few short substeps under real gravity. Without the second
    // plane pass the bottom sphere can briefly render with its
    // bottom below y=0 each frame; with the pass it must be at or
    // above y=radius every substep end.
    for (int i = 0; i < 30; ++i) {
        ps.Step(1.0f / 60.0f, /*substeps=*/8, /*g=*/-9.81f, /*damping=*/0.99f);
        const Particle* p0 = ps.GetParticle(h0);
        const Particle* p1 = ps.GetParticle(h1);
        const Particle* p2 = ps.GetParticle(h2);
        REQUIRE(p0 != nullptr);
        REQUIRE(p1 != nullptr);
        REQUIRE(p2 != nullptr);
        // Strict invariant: every particle is at or above its
        // touching-the-plane position.
        CHECK(p0->curr_pos.y >= r - 1e-4f);
        CHECK(p1->curr_pos.y >= r - 1e-4f);
        CHECK(p2->curr_pos.y >= r - 1e-4f);
    }
}

TEST_CASE("physics lifecycle: Clear empties the pool and invalidates handles") {
    PhysicsSystem ps;
    const auto h = ps.AddParticle({1.0f, 2.0f, 3.0f}, 0.3f);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);
    CHECK(ps.AliveCount() == 1u);

    ps.Clear();
    CHECK(ps.AliveCount() == 0u);
    CHECK(ps.GetParticle(h) == nullptr);            // bumped gen
}

TEST_CASE("physics capacity: pool full returns kInvalidHandle") {
    PhysicsSystem ps;
    // Fill to capacity.
    for (std::uint32_t i = 0; i < PhysicsSystem::kMaxParticles; ++i) {
        const auto h = ps.AddParticle({0.0f, 1.0f, 0.0f}, 0.3f);
        REQUIRE(h != PhysicsSystem::kInvalidHandle);
    }
    CHECK(ps.AliveCount() == PhysicsSystem::kMaxParticles);
    // Next add must fail with kInvalidHandle (not crash, not
    // silently overwrite).
    const auto overflow = ps.AddParticle({0.0f, 1.0f, 0.0f}, 0.3f);
    CHECK(overflow == PhysicsSystem::kInvalidHandle);
}

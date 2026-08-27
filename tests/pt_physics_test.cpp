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
//
//   5. Gravity at altitude (#270): a drop from an ordinary scene
//      altitude must take the physically correct time to reach the
//      ground at EVERY phys_substeps setting the cvar allows. The
//      Phase 1 position-Verlet form injected gravity as an absolute
//      `accel * sdt^2` increment, which falls below the float32 ULP
//      of the position above ~1 km at the default 8 substeps and
//      above ~64 m at the maximum of 32 -- so raising the accuracy
//      knob silently switched gravity off. See the free-fall block
//      at the bottom of this file.

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

// ---------------------------------------------------------------------------
// Issue #270 -- gravity must survive at altitude, at every substep count.
// ---------------------------------------------------------------------------
//
// The Phase 1 integrator added gravity to the position as an absolute
// `accel * sdt^2` increment. float32 has a 24-bit mantissa, so the gap
// between representable values near a position y is y * 2^-23, and
// round-to-nearest swallows whole any increment below y * 2^-24. At the
// default phys_substeps 8 of a 1/60 s frame that increment is
// 9.81 * (1/480)^2 = 4.26e-5 m, which disappears above roughly 1024 m;
// at the cvar maximum of 32 substeps it is 2.66e-6 m and disappears
// above 64 m. Because sdt^2 shrinks quadratically in the substep count
// while the position magnitude does not, the accuracy knob was what
// destroyed the accuracy.
//
// These cases pin the real physics: a body released from rest in
// vacuum falls h metres in sqrt(2h/g) seconds, and that must hold at
// every phys_substeps from 1 to 32. Damping is 1.0 throughout --
// phys_damping is an artificial settling aid rather than drag, and the
// analytic solution being compared against is the vacuum one.

namespace {

// Standard gravity as the engine ships it in the phys_gravity_y cvar
// (metric units, real constant -- not a rounded stand-in).
constexpr float kG   = 9.81f;          // m/s^2, magnitude
constexpr float kDt  = 1.0f / 60.0f;   // s; the frame dt smoke mode pins
constexpr float kRad = 0.30f;          // m; the default particle radius

// Half-ULP ratio of float32: a round-to-nearest add loses at most
// |result| * 2^-24. Written as the exact binary value rather than as
// 6e-8 so nobody mistakes it for a tuned fudge factor.
constexpr double kF32HalfUlp = 5.9604644775390625e-08;   // 2^-24

// Frame-by-frame landing detector, shared by the particle and the
// rigid-body drop cases.
//
// The pools can only be observed at frame boundaries, and the contact
// is 100% elastic (Phase 1 has no restitution model -- see
// PhysicsSystem::Substep step 2), so a landed body is already on its
// way back up by the time the frame ends and "y <= radius at frame
// end" simply never fires. Two independent signals cover the landing
// instead, because the contact can fall anywhere inside a frame:
//
//   * the height stopped decreasing -- the body bounced early enough
//     in the frame to end it no lower than it started; or
//   * the frame's drop collapsed to less than half the previous
//     frame's. Consecutive free-fall drops are g*dt^2*(n - 1/2),
//     strictly increasing, so their ratio never goes below 1 and a
//     one-half threshold carries a 2x margin against float32 noise. A
//     body that lands mid-frame spends the rest of that frame
//     travelling back up, which cuts the net drop far past half.
//
// A landing in the last sliver of a frame trips neither (the drop is
// still nearly a full frame's worth) and gets caught by the first
// signal on the NEXT frame -- which is why FallToleranceSeconds
// budgets a whole frame, not half of one, for observation.
//
// Both signals are armed only once the body is past the halfway point
// of its fall, so the "it never moved at all" failure mode reads as a
// timeout rather than as an instant landing on frame 1.
class LandingDetector {
public:
    LandingDetector(float y0, float ground_y)
        : prev_y_(y0), halfway_(ground_y + 0.5f * (y0 - ground_y)) {}

    // Feed one end-of-frame height. Returns true on the first frame
    // judged to contain the ground contact.
    bool Feed(float y) {
        const float drop = prev_y_ - y;
        const bool  hit  = (y < halfway_) &&
                           (y >= prev_y_ || drop < 0.5f * prev_drop_);
        prev_drop_ = drop;
        prev_y_    = y;
        return hit;
    }

private:
    float prev_y_;
    float halfway_;
    float prev_drop_ = 0.0f;
};

// Simulate a vacuum drop from `y0` and return the time at which the
// particle reaches the ground, or a negative value if it never does.
// The landing is somewhere inside the detected frame (or the sliver
// before it), so we report that frame's MIDPOINT, which keeps the
// observation error symmetric instead of biased late.
double MeasureFallSeconds(float y0, int substeps, int frame_cap) {
    PhysicsSystem ps;
    const auto h = ps.AddParticle({0.0f, y0, 0.0f}, kRad);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);

    LandingDetector det(y0, kRad);
    for (int frame = 1; frame <= frame_cap; ++frame) {
        ps.Step(kDt, substeps, -kG, /*damping=*/1.0f);
        const Particle* p = ps.GetParticle(h);
        REQUIRE(p != nullptr);
        if (det.Feed(p->curr_pos.y)) {
            return (static_cast<double>(frame) - 0.5) * static_cast<double>(kDt);
        }
    }
    return -1.0;
}

// Derived -- not tuned -- tolerance on a measured fall time, in
// seconds. Three independent, individually-bounded error sources:
//
//   (1) Observation quantization. The pool is only sampled once per
//       frame, and LandingDetector resolves the contact to the frame
//       that contains it or (for a contact in a frame's last sliver)
//       the one immediately after. Reporting the detected frame's
//       midpoint puts the true landing inside a +/- kDt window around
//       the reported value.
//
//   (2) The integrator's own discretization. Undamped, the discrete
//       trajectory has fallen g*sdt^2*n(n+1)/2 after n substeps,
//       against the continuum's g*t^2/2 at t = n*sdt. Solving
//       (g/2)(t^2 + sdt*t) = h gives
//       t_discrete = sqrt(t_exact^2 + sdt^2/4) - sdt/2, i.e. the
//       discrete solution lands EARLY by strictly less than sdt/2.
//       This term is identical for the old position-Verlet form --
//       it is the same recurrence -- so the #270 fix neither helps
//       nor hurts here.
//
//   (3) float32 accumulation. Each of the n = t_exact/sdt position
//       updates rounds to nearest and so can lose at most y0 * 2^-24
//       metres. The pessimistic bound assumes every one of them
//       rounds the same way: n * y0 * 2^-24 metres of drop error,
//       converted to a time error by dividing by the impact speed
//       sqrt(2*g*h). Round-to-nearest-even is unbiased, so the
//       observed drift is nearer sqrt(n) than n times the half-ULP --
//       measured, this bound comes out 20-40x pessimistic at 10 km.
//       It is still a bound rather than a fitted number, which is
//       the point.
double FallToleranceSeconds(double y0, int substeps, double radius = kRad) {
    const double g       = static_cast<double>(kG);
    const double dt      = static_cast<double>(kDt);
    const double drop    = y0 - radius;
    const double t_exact = std::sqrt(2.0 * drop / g);
    const double v_hit   = std::sqrt(2.0 * g * drop);
    const double sdt     = dt / static_cast<double>(substeps);

    const double observation    = dt;
    const double discretization = 0.5 * sdt;
    const double fp_drift       = (t_exact / sdt) * y0 * kF32HalfUlp / v_hit;
    return observation + discretization + fp_drift;
}

}  // namespace

TEST_CASE("physics #270: a 2048 m drop lands in sqrt(2h/g) at every substep count") {
    // The headline acceptance case. Before the fix this particle did
    // not fall AT ALL at the default 8 substeps (nor at 16 or 32):
    // 9.81 * (1/480)^2 = 4.26e-5 m is below the 1.22e-4 m half-ULP of
    // y = 2048, so `curr_pos + accel*sdt^2 == curr_pos` bit for bit.
    constexpr float y0 = 2048.0f;
    const double drop    = static_cast<double>(y0) - static_cast<double>(kRad);
    const double t_exact = std::sqrt(2.0 * drop / static_cast<double>(kG));

    for (int substeps : {1, 2, 4, 8, 16, 32}) {
        CAPTURE(substeps);
        // Four times the analytic fall time is a generous cap: a
        // particle that has not landed by then is not falling.
        const int cap = static_cast<int>(4.0 * t_exact / static_cast<double>(kDt)) + 60;
        const double t = MeasureFallSeconds(y0, substeps, cap);
        REQUIRE_MESSAGE(t > 0.0, "particle never reached the ground -- gravity is a no-op");
        CHECK(std::fabs(t - t_exact) <= FallToleranceSeconds(y0, substeps));
    }
}

TEST_CASE("physics #270: free-fall time matches sqrt(2h/g) from 10 m to 10 km") {
    // 10 m and 100 m are ordinary gameplay altitudes; 1 km and 10 km
    // are ordinary scene altitudes (the sunset_altitude fixture puts
    // the camera at y = 10000). All five must obey the same analytic
    // law at all six substep settings.
    for (float y0 : {10.0f, 100.0f, 1000.0f, 2048.0f, 10000.0f}) {
        for (int substeps : {1, 2, 4, 8, 16, 32}) {
            CAPTURE(y0);
            CAPTURE(substeps);
            const double drop    = static_cast<double>(y0) - static_cast<double>(kRad);
            const double t_exact = std::sqrt(2.0 * drop / static_cast<double>(kG));
            const int    cap     =
                static_cast<int>(4.0 * t_exact / static_cast<double>(kDt)) + 60;
            const double t       = MeasureFallSeconds(y0, substeps, cap);
            REQUIRE_MESSAGE(t > 0.0, "particle never reached the ground -- gravity is a no-op");
            CHECK(std::fabs(t - t_exact) <= FallToleranceSeconds(y0, substeps));
        }
    }
}

TEST_CASE("physics #270: gravity does not weaken as phys_substeps rises") {
    // The substeps INVERSION guard, stated as a bound that is itself
    // independent of the substep count.
    //
    // After t seconds the undamped discrete trajectory has fallen
    // (g/2)(t^2 + sdt*t) -- the continuum answer (g/2)t^2 plus a lead
    // term proportional to sdt. That lead is LARGEST at substeps = 1
    // and shrinks from there, so (g/2)*kDt*t bounds it for every legal
    // substep count at once. Add the float32 accumulation bound taken
    // at the finest substep, and the whole 1..32 sweep has to fit
    // inside one fixed tolerance. That is exactly the property #270
    // broke: the old form's error grew without limit as substeps rose,
    // all the way to a 100% error (no fall whatsoever) at 8 and above.
    //
    // 100 m is deliberately mundane. The issue measured the old
    // integrator as 7.5% wrong here at the default substeps and
    // completely dead at the maximum.
    constexpr float y0      = 100.0f;
    constexpr int   kFrames = 240;                       // 4 s at 1/60 s
    const double    t       = static_cast<double>(kFrames) * static_cast<double>(kDt);
    const double    g       = static_cast<double>(kG);

    const double continuum      = 0.5 * g * t * t;                    // 78.48 m
    const double discretization = 0.5 * g * static_cast<double>(kDt) * t;
    const double fp_drift       = (t / (static_cast<double>(kDt) / 32.0))
                                  * static_cast<double>(y0) * kF32HalfUlp;
    const double tol            = discretization + fp_drift;

    for (int substeps : {1, 2, 4, 8, 16, 32}) {
        CAPTURE(substeps);
        PhysicsSystem ps;
        const auto h = ps.AddParticle({0.0f, y0, 0.0f}, kRad);
        REQUIRE(h != PhysicsSystem::kInvalidHandle);
        for (int f = 0; f < kFrames; ++f) {
            ps.Step(kDt, substeps, -kG, /*damping=*/1.0f);
        }
        const Particle* p = ps.GetParticle(h);
        REQUIRE(p != nullptr);
        const double fell = static_cast<double>(y0) - static_cast<double>(p->curr_pos.y);
        CHECK(std::fabs(fell - continuum) <= tol);
    }
}

TEST_CASE("physics #270: damping semantics are preserved exactly") {
    // Moving gravity out of the position and into an explicit velocity
    // had to leave `phys_damping` meaning precisely what it meant
    // before: a per-SUBSTEP multiplier on velocity. Writing d_n for
    // the old per-substep displacement and v_n = d_n / sdt,
    //
    //     d_{n+1} = damping * d_n + accel * sdt^2
    //             = sdt * (damping * v_n + accel * sdt)
    //
    // is the same recurrence divided through by sdt, so the closed
    // form of the geometric series is unchanged:
    //
    //     v_N = accel * sdt * (1 - damping^N) / (1 - damping)
    //
    // Pin that closed form rather than the old inequality, so a future
    // change that quietly moves the damping to a per-FRAME multiplier
    // (or applies it after gravity instead of before) fails here
    // instead of surfacing as piles that settle wrong.
    //
    // kN is 32 because that is the ceiling Step() clamps `substeps`
    // to; asking for more would silently integrate a different sdt
    // than the closed form below assumes.
    constexpr float kDmp = 0.9f;
    constexpr int   kN   = 32;
    constexpr float kSdt = 1.0f / 600.0f;

    PhysicsSystem ps;
    const auto h = ps.AddParticle({0.0f, 50.0f, 0.0f}, kRad);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);
    ps.Step(kSdt * static_cast<float>(kN), kN, -kG, kDmp);

    const Particle* p = ps.GetParticle(h);
    REQUIRE(p != nullptr);

    const double a   = -static_cast<double>(kG);
    const double sdt = static_cast<double>(kSdt);
    const double d   = static_cast<double>(kDmp);
    const double expected_v = a * sdt * (1.0 - std::pow(d, kN)) / (1.0 - d);

    // kN chained multiply-adds each carry at most a 2^-24 relative
    // rounding error, so 60 * 2^-24 = 3.6e-6 relative bounds the
    // drift; 1e-5 leaves a ~3x margin on a derived figure.
    const double rel_tol = 1.0e-5;
    CHECK(std::fabs(static_cast<double>(p->velocity.y) - expected_v)
          <= rel_tol * std::fabs(expected_v));

    // ...and the Phase-1 way of reading velocity out of the position
    // pair must still agree, because the engine's writeback and the
    // editor's inspector both do exactly that.
    //
    // That readout is limited by the ULP of the POSITION it is
    // differenced out of, not by the velocity's own precision: at
    // y = 50 m one float32 ULP is 50 * 2^-23 = 6.0e-6 m, and dividing
    // by sdt turns it into 3.6e-3 m/s of irreducible readout noise --
    // 2000x the stored field's own error. That asymmetry is exactly
    // the caveat Particle.h documents (prefer the stored field), so
    // bound this comparison at the position ULP rather than at
    // rel_tol, and let the tighter invariant case below police the
    // relationship itself.
    const double implicit_v =
        static_cast<double>(p->curr_pos.y - p->prev_pos.y) / static_cast<double>(kSdt);
    const double readout_tol =
        (50.0 * std::ldexp(1.0, -23)) / static_cast<double>(kSdt);
    CHECK(std::fabs(implicit_v - expected_v) <= readout_tol);
}

TEST_CASE("physics #270: curr - prev stays equal to velocity * substep dt") {
    // The compatibility invariant Particle documents. Checked through
    // a ground bounce, because that is where the constraint solver
    // moves curr_pos behind the integrator's back -- if step 5's
    // velocity correction were dropped, the stored velocity and the
    // derived one would diverge here and nowhere else.
    PhysicsSystem ps;
    const auto h = ps.AddParticle({0.0f, 3.0f, 0.0f}, kRad);
    REQUIRE(h != PhysicsSystem::kInvalidHandle);

    for (int frame = 0; frame < 120; ++frame) {
        ps.Step(kDt, /*substeps=*/8, -kG, /*damping=*/0.99f);
        const Particle* p = ps.GetParticle(h);
        REQUIRE(p != nullptr);
        const float sdt = ps.LastSubstepDt();
        const glm::vec3 derived = (p->curr_pos - p->prev_pos) / sdt;
        // Both sides are O(10) m/s here and the bounce multiplies the
        // correction by 1/sdt = 480, so a few ULP of a ~1e1 magnitude
        // scaled by that -- call it 1e-3 m/s -- is the honest bound on
        // a difference that is zero in exact arithmetic.
        CHECK(approx(derived.x, p->velocity.x, 1e-3f));
        CHECK(approx(derived.y, p->velocity.y, 1e-3f));
        CHECK(approx(derived.z, p->velocity.z, 1e-3f));
    }
}

TEST_CASE("physics #270: a rigid body falls the same way a particle does") {
    // The rigid-body pool carries its own copy of the integrator, so
    // it needs its own altitude coverage -- phys_rb_demo and the
    // engine's phys_drop_sphere drive that path, not the particle one.
    constexpr float y0  = 2048.0f;
    constexpr float kBr = 0.4f;              // body radius
    const double drop    = static_cast<double>(y0) - static_cast<double>(kBr);
    const double t_exact = std::sqrt(2.0 * drop / static_cast<double>(kG));

    for (int substeps : {1, 8, 32}) {
        CAPTURE(substeps);
        PhysicsSystem ps;
        const auto h = ps.AddRigidSphere({0.0f, y0, 0.0f}, kBr, /*mass=*/1.0f);
        REQUIRE(h != PhysicsSystem::kInvalidRbHandle);

        const int cap =
            static_cast<int>(4.0 * t_exact / static_cast<double>(kDt)) + 60;
        LandingDetector det(y0, kBr);
        double t = -1.0;
        for (int frame = 1; frame <= cap; ++frame) {
            ps.Step(kDt, substeps, -kG, /*damping=*/1.0f);
            const auto* b = ps.GetRigidBody(h);
            REQUIRE(b != nullptr);
            if (det.Feed(b->curr_pos.y)) {
                t = (static_cast<double>(frame) - 0.5) * static_cast<double>(kDt);
                break;
            }
        }
        REQUIRE_MESSAGE(t > 0.0, "rigid body never reached the ground -- gravity is a no-op");
        CHECK(std::fabs(t - t_exact)
              <= FallToleranceSeconds(y0, substeps, static_cast<double>(kBr)));
    }
}

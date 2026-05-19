// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit-test coverage for the `phys_drop_*` RGB arg shapes added by
// PR #189 (issue #181 partial -- colored rigid-body materials) and the
// emissive-RGB extension shipped as the wave-4 polish (#181 polish:
// emissive rigid bodies via phys_drop_* emission args).
//
// PR #189 added optional per-body RGB tint args to `phys_drop_sphere`
// and `phys_drop_box`. The wave-4 polish extends each with an optional
// emission triple (W/sr per channel, units consistent with
// AnalyticLight::intensity for point/spot lights from PR #185). The
// arg-shape contract is documented in the command help strings + the
// PR bodies; this test pins it so a future refactor (e.g. a generic
// prim-material parser, a tuple-of-floats rework, or "let's just take a
// single hex code + a single emission value instead") doesn't silently
// break back-compat or relax the rejection rules.
//
// What's tested:
//
//   `phys_drop_sphere x y z [r m] [red green blue] [emit_r emit_g emit_b]`
//   (3, 4, 5, 8, or 11 args)
//
//     - 3-arg back-compat:  spawns at (x,y,z) with default radius (0.3),
//                           default mass (1.0), default warm-grey
//                           albedo (0.85, 0.80, 0.70), zero emission.
//                           Bit-for-bit identical to pre-#181 behaviour.
//     - 4-arg back-compat:  + custom radius
//     - 5-arg back-compat:  + custom mass
//     - 8-arg with RGB:     + custom albedo (the #189 surface)
//     - 11-arg with RGB + emit: + custom emission (the #181-polish surface)
//     - 6/7-arg partial RGB REJECTED with usage hint -- the parser
//       has no way to know whether arg[5] is mass or red, and
//       guessing would silently bias albedo. Rejecting at the gate
//       makes the user-facing error obvious.
//     - 9/10-arg partial emission REJECTED for the same reason as
//       6/7-arg partial RGB.
//     - Negative RGB REJECTED -- shading code assumes non-negative
//       reflectance; a negative value here would produce undefined
//       fireflies in downstream BSDF eval.
//     - Negative emission REJECTED -- would invert the radiance
//       accumulator and produce dark "anti-glow" spots instead of
//       an emissive surface.
//     - HDR-bright (R/G/B > 1.0) ACCEPTED as-is -- a physically-
//       valid use case (emitter-like surfaces, post-bloom
//       overbright sources, custom NPR shading).
//     - Bright emission (W/sr > 1.0) ACCEPTED as-is -- a 5 W/sr LED
//       bulb is the physical baseline; user-defined values can land
//       well above that.
//
//   `phys_drop_box x y z hx hy hz [m] [red green blue] [emit_r emit_g emit_b]`
//   (6, 7, 10, or 13 args)
//
//     - 6-arg back-compat:  no mass, no albedo, no emission. Default
//                           mass 1.0, default cool-blue albedo
//                           (0.60, 0.65, 0.85), zero emission.
//     - 7-arg back-compat:  + custom mass
//     - 10-arg with RGB:    + custom albedo
//     - 13-arg with RGB + emit: + custom emission
//     - 8/9-arg partial RGB REJECTED with usage hint
//     - 11/12-arg partial emission REJECTED with usage hint
//
// Test architecture:
//
//   The phys_drop_* commands are registered against Engine via
//   `Engine::RegisterPhysicsCommands()` and they dispatch into
//   Engine::physics_->AddRigidSphere / AddRigidBox plus the
//   private `primitives_` analytic-prim map. Both are private; we
//   reach them via the `PhysDropArgsTestAccess` friend struct
//   forward-declared in Engine.h. The struct intentionally lives
//   only here -- pt_engine never references it.
//
//   We bypass `Engine::Init()` (which would pull in RHI, window,
//   audio, jobs, the cfg loader, ...) and instead just wire up the
//   single sub-set the commands need: a PhysicsSystem instance and
//   the registered console commands. Each TEST_CASE calls
//   ResetState() at entry to clear primitives_ + the physics pool +
//   the prim-id counter so the test order doesn't matter and the
//   doctest singleton state doesn't bleed across cases.
//
//   The test asserts on TWO surfaces per success case:
//     1. The Console::Execute output string. The phys_drop_*
//        success path echoes back the resulting rgb=(R G B), so
//        a regression that wired up the wrong defaults or
//        dropped the user's args on the floor would show up
//        immediately as a string mismatch.
//     2. The actual `primitives_` map. Tests can read the resulting
//        AnalyticPrim.albedo[] / .radius_or_d through the friend
//        struct, so we pin BOTH the command-output surface AND
//        the GPU-upload-feeding-data structure (one without the
//        other could pass under a future refactor where they
//        diverge -- e.g. an output-format change that didn't
//        actually update the prim).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/console/Console.h"
#include "../src/engine/Engine.h"
#include "../src/physics/PhysicsSystem.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

// Friend-struct definition. The forward-declaration lives in Engine.h
// (where the `friend struct ::pt::engine::PhysDropArgsTestAccess`
// declaration on `class Engine` references it); the body lives here so
// production pt_engine never references it.
//
// LIFETIME / SINGLETON NOTE. Console::RegisterCommand uses try_emplace,
// so the first registration of a given name wins -- a subsequent
// RegisterCommand call with the same name keeps the OLD callback (and
// its captured `this` pointer). Per-TEST_CASE Engines would each
// destroy their object on scope exit, leaving the Console singleton's
// command map pointing at freed memory. The next TEST_CASE's
// Console::Execute would call into that dangling `this` (UB; in
// practice it fails the rejection-path assertions because the dead
// engine's state is unreadable).
//
// Fix: a single, function-local-static Engine that lives for the
// process lifetime. doctest TEST_CASEs run in definition order within
// one process; each TEST_CASE clears primitives_ + physics_ state via
// ResetState() so order-independence is preserved.
namespace pt::engine {

struct PhysDropArgsTestAccess {
    // Returns the single test-wide Engine instance. The first call
    // constructs the Engine, wires up physics_, and registers the
    // phys_drop_* commands. Subsequent calls just hand back the same
    // instance; the Console singleton's command map still points at
    // this engine's `this`, so dispatch is safe.
    static PhysDropArgsTestAccess& Get() {
        // Intentionally-leaked heap allocation. The owned Engine's
        // destructor would run Shutdown() at process exit -- which
        // writes demont.cfg into the test working dir, flushes log
        // sinks, etc. -- breaking hermeticity. A function-static
        // pointer that's never deleted lets the OS reap the memory
        // at exit without invoking ~Engine().
        static PhysDropArgsTestAccess* instance = new PhysDropArgsTestAccess();
        return *instance;
    }

    // Reset between tests so prim ids start fresh + the physics pool
    // is empty. Console commands stay registered (the closure still
    // captures THIS engine).
    void ResetState() {
        engine_.primitives_.clear();
        if (engine_.physics_) engine_.physics_->Clear();
        engine_.physics_next_prim_id_ = 100000u;
    }

    // Find the prim with the given id; nullptr if not present.
    const Engine::AnalyticPrim* FindPrim(std::uint32_t id) const {
        auto it = engine_.primitives_.find(id);
        return (it == engine_.primitives_.end()) ? nullptr : &it->second;
    }

    // Convenience: returns the last spawned prim's id. phys_drop_*
    // monotonically increments `physics_next_prim_id_` starting at
    // 100000, so the last id is (physics_next_prim_id_ - 1) after at
    // least one successful spawn. Returns 0 if no prim has been
    // spawned yet (the next-id counter hasn't moved off its initial
    // value).
    std::uint32_t LastSpawnedId() const {
        if (engine_.physics_next_prim_id_ == 100000u) return 0u;
        return engine_.physics_next_prim_id_ - 1u;
    }

    // Direct access to the prim map for tests that want to count
    // entries by-source (e.g. "two successful drops produced two
    // prims, not one").
    const std::map<std::uint32_t, Engine::AnalyticPrim>& Primitives() const {
        return engine_.primitives_;
    }

private:
    PhysDropArgsTestAccess() {
        engine_.physics_ = std::make_unique<pt::physics::PhysicsSystem>();
        engine_.RegisterPhysicsCommands();
    }

    Engine engine_;
};

}  // namespace pt::engine

namespace {

// Tiny float-compare helper that mirrors pt_physics_test.cpp's
// `approx` -- keeps the rgb/radius assertions readable. 1e-5 is the
// natural noise floor for a single fmt::format("{:.2f}", x) round-trip
// (which doesn't actually happen here -- the test reads the float
// directly from the prim -- but matches the resolution the user sees
// in the command-output echo).
bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// Returns true iff `s` contains `needle`. Just a readability shim --
// every assertion below would otherwise be the noisy
// `CHECK(out.find(...) != std::string::npos)` pattern.
bool contains(const std::string& s, std::string_view needle) {
    return s.find(needle) != std::string::npos;
}

}  // namespace

// =============================================================================
// phys_drop_sphere
// =============================================================================

TEST_CASE("phys_drop_sphere: 3-arg back-compat (defaults radius, mass, color)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("phys_drop_sphere 0 5 0");
    CHECK(r.ok);                                          // command found + dispatched
    CHECK(contains(r.output, "spawned rigid body"));      // success log fired
    // Default radius 0.3, default mass 1.0, default warm-grey albedo.
    // The success log line carries the values back via FormatLine so
    // this is the at-the-source documented contract.
    CHECK(contains(r.output, "r=0.300"));
    CHECK(contains(r.output, "m=1.000"));
    CHECK(contains(r.output, "rgb=(0.85 0.80 0.70)"));

    // Verify the underlying AnalyticPrim that the GPU-upload path
    // will pick up matches what the output reports. A regression where
    // the FormatLine printed one set of values but stored another
    // would fail HERE (the output-only check above would silently
    // pass).
    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(p->type == pt::engine::Engine::AnalyticPrim::Sphere);
    CHECK(p->material == pt::engine::Engine::AnalyticPrim::Lambert);
    CHECK(approx(p->radius_or_d, 0.3f));
    CHECK(approx(p->albedo[0], 0.85f));
    CHECK(approx(p->albedo[1], 0.80f));
    CHECK(approx(p->albedo[2], 0.70f));

}

TEST_CASE("phys_drop_sphere: 4-arg back-compat (radius set, defaults mass + color)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("phys_drop_sphere 0 5 0 0.75");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "r=0.750"));                 // explicit radius
    CHECK(contains(r.output, "m=1.000"));                 // default mass
    CHECK(contains(r.output, "rgb=(0.85 0.80 0.70)"));    // default warm-grey

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->radius_or_d, 0.75f));
    CHECK(approx(p->albedo[0], 0.85f));
    CHECK(approx(p->albedo[1], 0.80f));
    CHECK(approx(p->albedo[2], 0.70f));

}

TEST_CASE("phys_drop_sphere: 5-arg back-compat (radius + mass set, defaults color)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("phys_drop_sphere 0 5 0 0.4 2.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "r=0.400"));
    CHECK(contains(r.output, "m=2.500"));
    CHECK(contains(r.output, "rgb=(0.85 0.80 0.70)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->radius_or_d, 0.4f));
    CHECK(approx(p->albedo[0], 0.85f));
    CHECK(approx(p->albedo[1], 0.80f));
    CHECK(approx(p->albedo[2], 0.70f));

}

TEST_CASE("phys_drop_sphere: 8-arg with full RGB applies custom albedo") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Bright red, slightly desaturated.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.9 0.1 0.05");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "r=0.300"));
    CHECK(contains(r.output, "m=1.000"));
    CHECK(contains(r.output, "rgb=(0.90 0.10 0.05)"));
    // Emission defaults to zero on the 8-arg shape (no emit args).
    CHECK(contains(r.output, "emit=(0.00 0.00 0.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 0.9f));
    CHECK(approx(p->albedo[1], 0.1f));
    CHECK(approx(p->albedo[2], 0.05f));
    // The 8-arg path leaves emission at the AnalyticPrim default (0).
    CHECK(approx(p->emission[0], 0.0f));
    CHECK(approx(p->emission[1], 0.0f));
    CHECK(approx(p->emission[2], 0.0f));

}

TEST_CASE("phys_drop_sphere: 11-arg with RGB + emission applies both") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // White-emissive red ball: a 5 W/sr per channel emitter -- visible
    // as a glowing point light, plus a red diffuse undertone. This is
    // the headline emissive use case from issue #181 polish.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 1.0 0.0 0.0 5.0 5.0 5.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "r=0.300"));
    CHECK(contains(r.output, "m=1.000"));
    CHECK(contains(r.output, "rgb=(1.00 0.00 0.00)"));
    CHECK(contains(r.output, "emit=(5.00 5.00 5.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 1.0f));
    CHECK(approx(p->albedo[1], 0.0f));
    CHECK(approx(p->albedo[2], 0.0f));
    CHECK(approx(p->emission[0], 5.0f));
    CHECK(approx(p->emission[1], 5.0f));
    CHECK(approx(p->emission[2], 5.0f));

}

TEST_CASE("phys_drop_sphere: 11-arg with chromatic emission stores per-channel") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Per-channel emission: warm yellow LED-style emitter (high R+G,
    // low B). Pinning the per-channel ordering catches a regression
    // where someone parses args[8..10] as (b, g, r) by mistake.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 8.0 6.0 1.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "emit=(8.00 6.00 1.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->emission[0], 8.0f));
    CHECK(approx(p->emission[1], 6.0f));
    CHECK(approx(p->emission[2], 1.0f));

}

TEST_CASE("phys_drop_sphere: 11-arg zero-emission accepted (explicit non-emissive)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // A user passing 0 0 0 for emission must be allowed -- it's a
    // documenting-intent gesture ("this prim is explicitly non-
    // emissive") that produces the same result as the 8-arg shape.
    // The >= 0 gate accepts zero.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 0.0 0.0 0.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "emit=(0.00 0.00 0.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->emission[0], 0.0f));
    CHECK(approx(p->emission[1], 0.0f));
    CHECK(approx(p->emission[2], 0.0f));

}

TEST_CASE("phys_drop_sphere: 6-arg partial RGB rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 6 args == x y z radius mass red -- ambiguous, no way to know
    // whether the user meant "red without green/blue" or fat-fingered
    // an extra arg. Rejection path.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.9");
    CHECK(r.ok);                                          // command was found
    CHECK(contains(r.output, "usage"));                   // usage hint fired
    CHECK(contains(r.output, "phys_drop_sphere"));        // and it names the command
    // Crucially: no "spawned rigid body" message -- the rejection
    // gate fires BEFORE the spawn so the physics pool stays empty
    // and primitives_ never gets a new entry.
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: 7-arg partial RGB rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 7 args == x y z radius mass red green -- still ambiguous, blue
    // missing.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.9 0.1");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_sphere"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: negative RGB component rejected") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Red is fine, green is negative -- physically meaningless
    // reflectance, must be rejected. The arg-shape gate passes (8
    // tokens) so this exercises the >= 0 value gate downstream.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 -0.1 0.5");
    CHECK(r.ok);                                          // command found
    CHECK(contains(r.output, "rgb components must be >= 0"));
    // Rejection => no spawn => primitives empty.
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: 9-arg partial emission rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 9 args == xyz + r + m + albedo rgb + emit_r -- emit_g/emit_b
    // missing. The parser must not guess a default for the missing
    // channels; reject at the gate.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 2.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_sphere"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: 10-arg partial emission rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 10 args == xyz + r + m + albedo rgb + emit_r + emit_g -- emit_b
    // missing. Symmetric to the 9-arg case above.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 2.0 1.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_sphere"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: negative emission component rejected") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Negative emission would invert the radiance accumulator -- the
    // emissive prim would appear as a dark spot instead of a glow,
    // confusing the user. Reject at the value gate (arg-shape gate
    // passes with 11 tokens).
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 1.0 -0.2 1.0");
    CHECK(r.ok);                                          // command found
    CHECK(contains(r.output, "emission components must be >= 0"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_sphere: HDR-bright RGB (> 1.0) accepted as-is") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Overbright albedo is a physically valid use case: an emissive
    // surface in a custom shading mode, post-tonemap-bypass debug,
    // or an NPR / stylised material. The parser must NOT clamp.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 2.5 5.0 10.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    // Output echoes back the EXACT values (no silent clamp).
    CHECK(contains(r.output, "rgb=(2.50 5.00 10.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 2.5f));
    CHECK(approx(p->albedo[1], 5.0f));
    CHECK(approx(p->albedo[2], 10.0f));

}

TEST_CASE("phys_drop_sphere: bright emission (> 1.0 W/sr) accepted as-is") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // A 100-W/sr-equivalent emitter -- car-headlight bright. Physical
    // baseline is fine; the parser must NOT clamp. Mirrors the HDR
    // albedo case above so a future "let's clamp emission to a sane
    // dynamic range" regression catches both surfaces.
    auto r = C.Execute("phys_drop_sphere 0 5 0 0.3 1.0 0.5 0.5 0.5 100.0 80.0 50.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "emit=(100.00 80.00 50.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->emission[0], 100.0f));
    CHECK(approx(p->emission[1], 80.0f));
    CHECK(approx(p->emission[2], 50.0f));

}

// =============================================================================
// phys_drop_box
// =============================================================================

TEST_CASE("phys_drop_box: 6-arg back-compat (no mass, no color)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "half=(0.50 0.50 0.50)"));
    CHECK(contains(r.output, "m=1.000"));                 // default mass
    // Default cool-blue tint -- matches the pre-#181 hard-coded
    // appearance for `phys_drop_box` with no color args.
    CHECK(contains(r.output, "rgb=(0.60 0.65 0.85)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    // Box renders as bounding sphere in Phase 2a (#138 MVP); the prim
    // type is Sphere with radius = box_bounding_radius. We're not
    // testing that math here (it's pinned by pt_physics_test); just
    // assert the albedo wiring.
    CHECK(approx(p->albedo[0], 0.60f));
    CHECK(approx(p->albedo[1], 0.65f));
    CHECK(approx(p->albedo[2], 0.85f));

}

TEST_CASE("phys_drop_box: 7-arg back-compat (mass set, default color)") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 3.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "half=(0.50 0.50 0.50)"));
    CHECK(contains(r.output, "m=3.000"));                 // explicit mass
    CHECK(contains(r.output, "rgb=(0.60 0.65 0.85)"));    // default cool-blue

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 0.60f));
    CHECK(approx(p->albedo[1], 0.65f));
    CHECK(approx(p->albedo[2], 0.85f));

}

TEST_CASE("phys_drop_box: 10-arg with full RGB applies custom albedo") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Bright green box.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.1 0.9 0.2");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "m=1.000"));
    CHECK(contains(r.output, "rgb=(0.10 0.90 0.20)"));
    // Emission defaults to zero on the 10-arg shape.
    CHECK(contains(r.output, "emit=(0.00 0.00 0.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 0.1f));
    CHECK(approx(p->albedo[1], 0.9f));
    CHECK(approx(p->albedo[2], 0.2f));
    CHECK(approx(p->emission[0], 0.0f));
    CHECK(approx(p->emission[1], 0.0f));
    CHECK(approx(p->emission[2], 0.0f));

}

TEST_CASE("phys_drop_box: 13-arg with RGB + emission applies both") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Cyan-emissive box: cool-blue diffuse + cyan glow (high G+B).
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.2 0.5 0.8 0.0 4.0 4.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "rgb=(0.20 0.50 0.80)"));
    CHECK(contains(r.output, "emit=(0.00 4.00 4.00)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 0.2f));
    CHECK(approx(p->albedo[1], 0.5f));
    CHECK(approx(p->albedo[2], 0.8f));
    CHECK(approx(p->emission[0], 0.0f));
    CHECK(approx(p->emission[1], 4.0f));
    CHECK(approx(p->emission[2], 4.0f));

}

TEST_CASE("phys_drop_box: 11-arg partial emission rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 11 == xyz + hxhyhz + mass + albedo rgb + emit_r -- partial
    // emission, no green/blue.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5 0.5 0.5 2.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_box"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: 12-arg partial emission rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 12 == above + emit_g, emit_b still missing.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5 0.5 0.5 2.0 1.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_box"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: negative emission component rejected") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Symmetric to the sphere negative-emission case; pin the box's
    // value-gate too.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5 0.5 0.5 -1.0 2.0 1.0");
    CHECK(r.ok);
    CHECK(contains(r.output, "emission components must be >= 0"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: 8-arg partial RGB rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 8 == x y z hx hy hz mass red -- partial RGB, no green/blue.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_box"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: 9-arg partial RGB rejected with usage hint") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 9 == x y z hx hy hz mass red green -- blue still missing.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5 0.3");
    CHECK(r.ok);
    CHECK(contains(r.output, "usage"));
    CHECK(contains(r.output, "phys_drop_box"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: negative RGB component rejected") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // 10 tokens -- shape gate passes -- but blue is negative.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 0.5 0.3 -0.1");
    CHECK(r.ok);
    CHECK(contains(r.output, "rgb components must be >= 0"));
    CHECK_FALSE(contains(r.output, "spawned rigid body"));
    CHECK(acc.Primitives().empty());

}

TEST_CASE("phys_drop_box: HDR-bright RGB (> 1.0) accepted as-is") {
    auto& acc = pt::engine::PhysDropArgsTestAccess::Get();
    acc.ResetState();
    auto& C = pt::console::Console::Get();

    // Overbright -- same physical-validity argument as the sphere
    // case above. Pin the no-clamp behaviour.
    auto r = C.Execute("phys_drop_box 0 5 0 0.5 0.5 0.5 1.0 3.0 2.0 1.5");
    CHECK(r.ok);
    CHECK(contains(r.output, "spawned rigid body"));
    CHECK(contains(r.output, "rgb=(3.00 2.00 1.50)"));

    const auto id = acc.LastSpawnedId();
    REQUIRE(id != 0u);
    auto* p = acc.FindPrim(id);
    REQUIRE(p != nullptr);
    CHECK(approx(p->albedo[0], 3.0f));
    CHECK(approx(p->albedo[1], 2.0f));
    CHECK(approx(p->albedo[2], 1.5f));

}

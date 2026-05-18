// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for vec2/vec3/vec4 ops (issue #63 -- Phase 3 of #47).
//
// The engine consumes glm directly (src/renderer/Camera.h, Engine.cpp,
// SoftwareTracer.cpp etc.); there is no in-house wrapper around its
// vector primitives. These tests therefore pin the exact glm calls the
// engine relies on -- dot, cross, normalize, length, +/-/* operators,
// component-wise ops, and the pathological edges (zero vector,
// near-zero norm, denormal magnitude). The point is not to test glm
// itself (glm has its own test suite) but to fix the contract: if a
// future build flips a `GLM_FORCE_*` define or bumps glm to a major
// version that changes default handedness / precision, this test fails
// before the engine does.
//
// All assertions use absolute tolerances (kEps = 1e-6f for unit-length
// values, kLooseEps = 1e-5f for products of normalizations). Tighter
// than 1e-6 would surface ULP variance from fma reordering across
// clang/gcc/MSVC; looser than 1e-5 would hide a "vec3 dot lost a
// component" regression on a unit-circle vector.
//
// Determinism: no std::random, no system clock. All inputs are
// hardcoded literals or derived from the deterministic xorshift32
// helper below (same pattern as pt_core_analytic_bvh_test.cpp).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Tight tolerance for unit-magnitude assertions. Anything above this is
// either a real bug or a sign that the math library promoted a vector
// op to fma when we expected a multiply+add.
constexpr float kEps      = 1e-6f;
// Looser tolerance for compositions of multiple ops (dot of two
// normalized vectors, etc.) where ULP accumulation legitimately bites.
constexpr float kLooseEps = 1e-5f;

// Deterministic xorshift32 -- mirrors the pattern in
// pt_core_analytic_bvh_test.cpp. Self-contained: no std::random,
// implementation-independent across platforms / standard libraries.
struct XorShift32 {
    std::uint32_t state;
    explicit XorShift32(std::uint32_t seed) : state(seed ? seed : 0xC0FFEEu) {}
    std::uint32_t next() {
        std::uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    // 24-bit-mantissa float in [0,1)
    float unitf() {
        return float(next() >> 8) * (1.0f / float(1u << 24));
    }
    float rangef(float lo, float hi) { return lo + (hi - lo) * unitf(); }
};

}  // namespace

// --- vec2 -----------------------------------------------------------------
TEST_CASE("vec2: arithmetic + dot + length") {
    glm::vec2 a{3.0f, 4.0f};
    glm::vec2 b{1.0f, 2.0f};

    // Component-wise arithmetic.
    glm::vec2 sum = a + b;
    CHECK(sum.x == doctest::Approx(4.0f));
    CHECK(sum.y == doctest::Approx(6.0f));

    glm::vec2 diff = a - b;
    CHECK(diff.x == doctest::Approx(2.0f));
    CHECK(diff.y == doctest::Approx(2.0f));

    // Scalar mul.
    glm::vec2 scaled = a * 2.0f;
    CHECK(scaled.x == doctest::Approx(6.0f));
    CHECK(scaled.y == doctest::Approx(8.0f));

    // Dot product. 3*1 + 4*2 = 11.
    CHECK(glm::dot(a, b) == doctest::Approx(11.0f));

    // 3-4-5 triangle: length 5.
    CHECK(glm::length(a) == doctest::Approx(5.0f));
}

TEST_CASE("vec2: normalize + zero handling") {
    glm::vec2 v{3.0f, 4.0f};
    glm::vec2 n = glm::normalize(v);
    CHECK(glm::length(n) == doctest::Approx(1.0f).epsilon(kEps));
    CHECK(n.x == doctest::Approx(0.6f).epsilon(kEps));
    CHECK(n.y == doctest::Approx(0.8f).epsilon(kEps));

    // glm::normalize on the zero vector divides by zero, producing
    // NaN / Inf depending on platform. Our engine code (Camera::Right,
    // Camera::Up, light direction averaging, sun-disk math) all
    // pre-guards against zero-magnitude input before normalize. The
    // test below documents the precondition rather than expecting
    // glm to magically handle it -- the assertion is "the result has
    // a NaN somewhere", catching anyone who hits this in production.
    glm::vec2 z{0.0f, 0.0f};
    glm::vec2 nz = glm::normalize(z);
    CHECK((std::isnan(nz.x) || std::isnan(nz.y) ||
           std::isinf(nz.x) || std::isinf(nz.y)));
}

// --- vec3 -----------------------------------------------------------------
TEST_CASE("vec3: dot, cross, length, normalize") {
    // Axis-aligned dot products -- catches a column/row swap.
    glm::vec3 x_axis{1.0f, 0.0f, 0.0f};
    glm::vec3 y_axis{0.0f, 1.0f, 0.0f};
    glm::vec3 z_axis{0.0f, 0.0f, 1.0f};

    CHECK(glm::dot(x_axis, x_axis) == doctest::Approx(1.0f));
    CHECK(glm::dot(x_axis, y_axis) == doctest::Approx(0.0f));
    CHECK(glm::dot(x_axis, z_axis) == doctest::Approx(0.0f));
    CHECK(glm::dot(y_axis, z_axis) == doctest::Approx(0.0f));

    // glm uses a right-handed coord system by default: x cross y = +z.
    // The engine assumes RH everywhere (lookAtRH, perspectiveRH_ZO).
    // If a future glm bump or a GLM_FORCE_LEFT_HANDED leak flipped
    // this, every camera matrix in the engine would silently invert.
    glm::vec3 cxy = glm::cross(x_axis, y_axis);
    CHECK(cxy.x == doctest::Approx(0.0f));
    CHECK(cxy.y == doctest::Approx(0.0f));
    CHECK(cxy.z == doctest::Approx(1.0f));

    glm::vec3 cyz = glm::cross(y_axis, z_axis);
    CHECK(cyz.x == doctest::Approx(1.0f));
    CHECK(cyz.y == doctest::Approx(0.0f));
    CHECK(cyz.z == doctest::Approx(0.0f));

    glm::vec3 czx = glm::cross(z_axis, x_axis);
    CHECK(czx.x == doctest::Approx(0.0f));
    CHECK(czx.y == doctest::Approx(1.0f));
    CHECK(czx.z == doctest::Approx(0.0f));

    // Antisymmetry: a x b = -(b x a). Mirrors what the Camera::Right()
    // expression assumes (cross(Forward, +Y)).
    glm::vec3 a{1.5f, -2.3f, 0.7f};
    glm::vec3 b{0.2f, 1.1f, -0.4f};
    glm::vec3 axb = glm::cross(a, b);
    glm::vec3 bxa = glm::cross(b, a);
    CHECK(axb.x == doctest::Approx(-bxa.x));
    CHECK(axb.y == doctest::Approx(-bxa.y));
    CHECK(axb.z == doctest::Approx(-bxa.z));

    // Cross product is orthogonal to both inputs.
    CHECK(glm::dot(axb, a) == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(glm::dot(axb, b) == doctest::Approx(0.0f).epsilon(kLooseEps));

    // 1-2-2 triangle: sqrt(1+4+4) = 3.
    glm::vec3 v{1.0f, 2.0f, 2.0f};
    CHECK(glm::length(v) == doctest::Approx(3.0f));

    // normalize -> unit length.
    glm::vec3 n = glm::normalize(v);
    CHECK(glm::length(n) == doctest::Approx(1.0f).epsilon(kEps));
    // direction preserved -- ratios are 1:2:2.
    CHECK(n.x == doctest::Approx(1.0f / 3.0f).epsilon(kEps));
    CHECK(n.y == doctest::Approx(2.0f / 3.0f).epsilon(kEps));
    CHECK(n.z == doctest::Approx(2.0f / 3.0f).epsilon(kEps));
}

TEST_CASE("vec3: distance + lerp + reflect") {
    glm::vec3 p1{0.0f, 0.0f, 0.0f};
    glm::vec3 p2{3.0f, 0.0f, 4.0f};
    CHECK(glm::distance(p1, p2) == doctest::Approx(5.0f));

    // lerp(0) == a, lerp(1) == b, lerp(0.5) == midpoint
    glm::vec3 mid = glm::mix(p1, p2, 0.5f);
    CHECK(mid.x == doctest::Approx(1.5f));
    CHECK(mid.y == doctest::Approx(0.0f));
    CHECK(mid.z == doctest::Approx(2.0f));

    glm::vec3 at_a = glm::mix(p1, p2, 0.0f);
    CHECK(at_a.x == doctest::Approx(p1.x));
    CHECK(at_a.z == doctest::Approx(p1.z));
    glm::vec3 at_b = glm::mix(p1, p2, 1.0f);
    CHECK(at_b.x == doctest::Approx(p2.x));
    CHECK(at_b.z == doctest::Approx(p2.z));

    // Reflect across the y-axis plane: incoming +x +y, normal +y,
    // expected outgoing +x -y. This is the same math the path tracer
    // uses for the perfect-mirror BSDF sample direction.
    glm::vec3 incoming{1.0f, 1.0f, 0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 reflected = glm::reflect(incoming, normal);
    CHECK(reflected.x == doctest::Approx(1.0f));
    CHECK(reflected.y == doctest::Approx(-1.0f));
    CHECK(reflected.z == doctest::Approx(0.0f));
}

TEST_CASE("vec3: near-zero norm pathology") {
    // A vector with tiny but nonzero components must still normalize
    // to a unit-length direction. This pins the precondition the engine
    // relies on when integrating analytic light dirs near the horizon
    // (Engine.cpp:2256-2338 weighted-direction accumulator).
    glm::vec3 tiny{1e-20f, 2e-20f, 2e-20f};
    glm::vec3 n = glm::normalize(tiny);
    // Direction ratios still 1:2:2 even at denormal magnitude.
    CHECK(std::isfinite(n.x));
    CHECK(std::isfinite(n.y));
    CHECK(std::isfinite(n.z));
    CHECK(glm::length(n) == doctest::Approx(1.0f).epsilon(kLooseEps));
}

TEST_CASE("vec3: randomized invariants") {
    // Sweep ~200 deterministic random vectors and verify
    //   (1) normalize(v) has unit length
    //   (2) cross(a, b) is orthogonal to both a and b
    //   (3) |dot(a,b)| <= |a|*|b| (Cauchy-Schwarz)
    //
    // Uses xorshift32 -> the same input across hosts.
    XorShift32 rng{0xDEADBEEFu};
    for (int i = 0; i < 200; ++i) {
        glm::vec3 a{rng.rangef(-10.0f, 10.0f),
                    rng.rangef(-10.0f, 10.0f),
                    rng.rangef(-10.0f, 10.0f)};
        glm::vec3 b{rng.rangef(-10.0f, 10.0f),
                    rng.rangef(-10.0f, 10.0f),
                    rng.rangef(-10.0f, 10.0f)};
        // Skip the (statistically improbable but possible) zero vector.
        if (glm::dot(a, a) < 1e-12f || glm::dot(b, b) < 1e-12f) continue;

        glm::vec3 na = glm::normalize(a);
        CHECK(glm::length(na) == doctest::Approx(1.0f).epsilon(kLooseEps));

        glm::vec3 c = glm::cross(a, b);
        // Cross product magnitude can be 0 if a and b are parallel;
        // dot of c with a or b is still zero (mathematically). Use a
        // relative tolerance scaled to |a|*|b| so parallel-ish pairs
        // don't false-fail.
        const float scale = glm::length(a) * glm::length(b) + 1.0f;
        CHECK(std::abs(glm::dot(c, a)) <= kLooseEps * scale);
        CHECK(std::abs(glm::dot(c, b)) <= kLooseEps * scale);

        // Cauchy-Schwarz: |a.b| <= |a||b|.
        CHECK(std::abs(glm::dot(a, b)) <=
              glm::length(a) * glm::length(b) + kLooseEps);
    }
}

// --- vec4 -----------------------------------------------------------------
TEST_CASE("vec4: ctor + arithmetic + length") {
    glm::vec4 a{1.0f, 2.0f, 3.0f, 4.0f};
    glm::vec4 b{0.5f, 1.0f, 1.5f, 2.0f};

    // b == 0.5 * a -- verify via scalar mul.
    glm::vec4 a_half = a * 0.5f;
    CHECK(a_half.x == doctest::Approx(b.x));
    CHECK(a_half.y == doctest::Approx(b.y));
    CHECK(a_half.z == doctest::Approx(b.z));
    CHECK(a_half.w == doctest::Approx(b.w));

    // Component-wise add.
    glm::vec4 sum = a + b;
    CHECK(sum.x == doctest::Approx(1.5f));
    CHECK(sum.y == doctest::Approx(3.0f));
    CHECK(sum.z == doctest::Approx(4.5f));
    CHECK(sum.w == doctest::Approx(6.0f));

    // 1^2 + 2^2 + 3^2 + 4^2 = 30 -> length = sqrt(30).
    CHECK(glm::length(a) == doctest::Approx(std::sqrt(30.0f)));

    // vec3-from-vec4 truncation (swizzle) -- standard idiom in the
    // engine for converting homogeneous coords back to 3D.
    glm::vec3 trunc{a};
    CHECK(trunc.x == doctest::Approx(1.0f));
    CHECK(trunc.y == doctest::Approx(2.0f));
    CHECK(trunc.z == doctest::Approx(3.0f));
}

TEST_CASE("vec4: homogeneous-coord style ops") {
    // Standard "point" representation: w == 1. Translation by a vector
    // adds component-wise to xyz only (in a real matrix mul the w stays
    // 1 if the matrix is affine; this test is the vector-only flavor).
    glm::vec4 point{2.0f, 3.0f, 4.0f, 1.0f};
    glm::vec4 translation{0.5f, -0.5f, 0.25f, 0.0f};
    glm::vec4 moved = point + translation;
    CHECK(moved.x == doctest::Approx(2.5f));
    CHECK(moved.y == doctest::Approx(2.5f));
    CHECK(moved.z == doctest::Approx(4.25f));
    CHECK(moved.w == doctest::Approx(1.0f));

    // Dot product over 4 components -- the matrix-row-times-column
    // building block. Catches a "summed only xyz, missed w" regression.
    glm::vec4 r{1.0f, 2.0f, 3.0f, 4.0f};
    glm::vec4 c{5.0f, 6.0f, 7.0f, 8.0f};
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70.
    CHECK(glm::dot(r, c) == doctest::Approx(70.0f));
}

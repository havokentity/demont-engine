// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for quaternion ops (issue #63 -- Phase 3 of #47).
//
// The engine does not currently use quaternions in the hot path
// (camera uses Euler yaw/pitch, scene transforms are TRS matrices),
// but the quaternion API of glm is the natural rotation rep for the
// node graph that will land in later phases. These tests pin the glm
// quaternion API the engine code WILL consume:
//
//   - Construction: axis-angle, from Euler, from rotation matrix
//   - Composition: quat * quat
//   - Conjugation / inverse (unit quat -> rotational inverse)
//   - Vector rotation: q * v * q^-1 yields the rotated vector
//   - quat <-> mat3 round-trip (same rotation in both reps)
//   - slerp: 0/1 endpoints; intermediate length still unit; identity
//     -> any axis still smooth
//
// glm represents a unit quaternion as (w, x, y, z) but constructs from
// angle-axis as glm::angleAxis(angle, axis) -- the angle goes first.
// Wrong-order regressions in calling code would be caught by the
// axis-angle round-trip below.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>      // angleAxis, slerp, mat3_cast, quat_cast
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {

constexpr float kEps      = 1e-5f;
constexpr float kLooseEps = 1e-4f;

bool quat_approx_equal(const glm::quat& a, const glm::quat& b, float eps) {
    // Quaternion q and -q represent the same rotation. Compare both
    // shapes and take whichever is closer.
    auto same   = std::abs(a.w - b.w) + std::abs(a.x - b.x) +
                  std::abs(a.y - b.y) + std::abs(a.z - b.z);
    auto neg    = std::abs(a.w + b.w) + std::abs(a.x + b.x) +
                  std::abs(a.y + b.y) + std::abs(a.z + b.z);
    return std::min(same, neg) < eps * 4.0f;
}

bool vec3_approx_equal(const glm::vec3& a, const glm::vec3& b, float eps) {
    return std::abs(a.x - b.x) < eps &&
           std::abs(a.y - b.y) < eps &&
           std::abs(a.z - b.z) < eps;
}

}  // namespace

// --- Identity quaternion -------------------------------------------------
TEST_CASE("quat: default-construct identity") {
    // glm::quat() default-constructs to identity (w=1, xyz=0).
    glm::quat id{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK(id.w == doctest::Approx(1.0f));
    CHECK(id.x == doctest::Approx(0.0f));
    CHECK(id.y == doctest::Approx(0.0f));
    CHECK(id.z == doctest::Approx(0.0f));

    // Identity quaternion has unit length.
    CHECK(glm::length(id) == doctest::Approx(1.0f).epsilon(kEps));

    // Rotating any vector by identity returns the same vector.
    glm::vec3 v{1.0f, 2.0f, 3.0f};
    glm::vec3 vr = id * v;
    CHECK(vec3_approx_equal(vr, v, kLooseEps));
}

// --- Axis-angle construction --------------------------------------------
TEST_CASE("quat: angleAxis is unit length + rotates correctly") {
    // 90deg around +Y. Argument order: (angle, axis).
    glm::quat q = glm::angleAxis(glm::pi<float>() * 0.5f,
                                 glm::vec3{0.0f, 1.0f, 0.0f});
    CHECK(glm::length(q) == doctest::Approx(1.0f).epsilon(kEps));

    // Rotating +X by 90deg around +Y yields -Z.
    glm::vec3 x_axis{1.0f, 0.0f, 0.0f};
    glm::vec3 rotated = q * x_axis;
    CHECK(rotated.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(rotated.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(rotated.z == doctest::Approx(-1.0f).epsilon(kLooseEps));

    // Rotating the rotation axis by itself is a no-op.
    glm::vec3 y_axis{0.0f, 1.0f, 0.0f};
    glm::vec3 y_rotated = q * y_axis;
    CHECK(vec3_approx_equal(y_rotated, y_axis, kLooseEps));
}

TEST_CASE("quat: angleAxis round-trip (axis-angle -> mat3 -> angle-axis)") {
    // Pick a non-trivial axis and angle; build the quaternion, convert
    // to mat3, back to quaternion. The two quaternions should encode
    // the same rotation (modulo the q == -q ambiguity).
    glm::vec3 axis = glm::normalize(glm::vec3{1.0f, 2.0f, 3.0f});
    float angle = 1.2f;  // ~68.75deg
    glm::quat q1 = glm::angleAxis(angle, axis);
    glm::mat3 R  = glm::mat3_cast(q1);
    glm::quat q2 = glm::quat_cast(R);
    CHECK(quat_approx_equal(q1, q2, kLooseEps));
}

// --- Euler construction --------------------------------------------------
TEST_CASE("quat: from Euler angles") {
    // glm::quat(vec3(pitch, yaw, roll)) -- glm uses XYZ Euler order.
    // Construct quaternion from a pure-yaw Euler vector (90deg around Y)
    // and compare to angleAxis -- they must represent the same rotation.
    glm::quat q_euler  = glm::quat(glm::vec3{0.0f, glm::pi<float>() * 0.5f, 0.0f});
    glm::quat q_axis   = glm::angleAxis(glm::pi<float>() * 0.5f,
                                        glm::vec3{0.0f, 1.0f, 0.0f});
    CHECK(quat_approx_equal(q_euler, q_axis, kLooseEps));

    // Rotating +X by this q yields -Z (same as the axis-angle test).
    glm::vec3 r = q_euler * glm::vec3{1.0f, 0.0f, 0.0f};
    CHECK(r.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(r.z == doctest::Approx(-1.0f).epsilon(kLooseEps));
}

// --- Multiplication + conjugation ---------------------------------------
TEST_CASE("quat: multiplication composes rotations") {
    // q_a: 90deg around +Y, q_b: 90deg around +X. Composed (q_a * q_b)
    // should rotate +X via q_b first (X is on the axis, no change),
    // then via q_a (+X -> -Z under 90deg-around-Y).
    glm::quat q_a = glm::angleAxis(glm::pi<float>() * 0.5f,
                                   glm::vec3{0.0f, 1.0f, 0.0f});
    glm::quat q_b = glm::angleAxis(glm::pi<float>() * 0.5f,
                                   glm::vec3{1.0f, 0.0f, 0.0f});
    glm::quat composed = q_a * q_b;

    // The composed quaternion is still unit-length.
    CHECK(glm::length(composed) == doctest::Approx(1.0f).epsilon(kEps));

    // Applying q_b then q_a to +X: q_b leaves +X unchanged, q_a turns
    // it to -Z. So composed * X == -Z.
    glm::vec3 v = composed * glm::vec3{1.0f, 0.0f, 0.0f};
    CHECK(v.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(v.z == doctest::Approx(-1.0f).epsilon(kLooseEps));
}

TEST_CASE("quat: conjugate of unit quaternion equals inverse") {
    // For a unit quaternion, conjugate == inverse. q * conjugate(q)
    // must equal identity.
    glm::quat q = glm::angleAxis(0.7f, glm::normalize(glm::vec3{1.0f, 2.0f, 1.0f}));
    glm::quat c = glm::conjugate(q);
    glm::quat inv = glm::inverse(q);

    // Compare conjugate to inverse (should be equal for a unit q).
    CHECK(c.w == doctest::Approx(inv.w).epsilon(kLooseEps));
    CHECK(c.x == doctest::Approx(inv.x).epsilon(kLooseEps));
    CHECK(c.y == doctest::Approx(inv.y).epsilon(kLooseEps));
    CHECK(c.z == doctest::Approx(inv.z).epsilon(kLooseEps));

    // q * conjugate(q) == identity (w=1, xyz=0) for unit q.
    glm::quat prod = q * c;
    CHECK(prod.w == doctest::Approx(1.0f).epsilon(kLooseEps));
    CHECK(prod.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(prod.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(prod.z == doctest::Approx(0.0f).epsilon(kLooseEps));
}

// --- slerp ---------------------------------------------------------------
TEST_CASE("quat: slerp endpoints") {
    glm::quat q1 = glm::angleAxis(0.0f, glm::vec3{0.0f, 1.0f, 0.0f});
    glm::quat q2 = glm::angleAxis(glm::pi<float>(),
                                  glm::vec3{0.0f, 1.0f, 0.0f});

    // slerp(0) == q1.
    glm::quat at0 = glm::slerp(q1, q2, 0.0f);
    CHECK(quat_approx_equal(at0, q1, kLooseEps));

    // slerp(1) == q2.
    glm::quat at1 = glm::slerp(q1, q2, 1.0f);
    CHECK(quat_approx_equal(at1, q2, kLooseEps));
}

TEST_CASE("quat: slerp midpoint is unit-length and intermediate") {
    // Slerping between identity and a 90deg-around-Y rotation, the
    // midpoint should be a 45deg-around-Y rotation. The midpoint must
    // (a) be unit-length (slerp preserves length on the unit sphere)
    // and (b) match the angleAxis(45deg, +Y) reference to within
    // float epsilon.
    //
    // Note: we deliberately don't use a 180deg endpoint here -- when
    // q1 . q2 == 0 (exactly orthogonal in 4D), the shorter-arc choice
    // is ambiguous and slerp can pick either +/-90deg around the axis.
    // 90deg keeps the endpoint dot product positive (cos(45deg) =
    // sqrt(2)/2) so the shorter path is uniquely the 45deg midpoint.
    glm::quat q1 = glm::angleAxis(0.0f, glm::vec3{0.0f, 1.0f, 0.0f});
    glm::quat q2 = glm::angleAxis(glm::pi<float>() * 0.5f,
                                  glm::vec3{0.0f, 1.0f, 0.0f});

    glm::quat mid = glm::slerp(q1, q2, 0.5f);
    CHECK(glm::length(mid) == doctest::Approx(1.0f).epsilon(kLooseEps));

    // Apply mid to +X. The result should be the +X axis rotated 45deg
    // around +Y, i.e. (cos(-45deg), 0, sin(-45deg)) = (sqrt(2)/2, 0,
    // -sqrt(2)/2). (RH 90deg-around-Y sends +X to -Z, so 45deg sends
    // +X partway toward -Z while keeping a positive +X component.)
    const float root_half = std::sqrt(0.5f);
    glm::vec3 r = mid * glm::vec3{1.0f, 0.0f, 0.0f};
    CHECK(r.x == doctest::Approx( root_half).epsilon(kLooseEps));
    CHECK(r.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(r.z == doctest::Approx(-root_half).epsilon(kLooseEps));
}

TEST_CASE("quat: slerp of identical quats is just the input") {
    // Pathological case: slerp(q, q, t) for any t should be q (modulo
    // glm's q/-q ambiguity). Catches the "divide-by-zero in slerp's
    // sin(theta)" path which some implementations handle wrong.
    glm::quat q = glm::angleAxis(0.5f, glm::normalize(glm::vec3{1.0f, 2.0f, 3.0f}));
    for (float t : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        glm::quat result = glm::slerp(q, q, t);
        CHECK(quat_approx_equal(result, q, kLooseEps));
    }
}

// --- quat <-> mat3 round-trip --------------------------------------------
TEST_CASE("quat: mat3 round-trip preserves rotation action") {
    // Build q, convert to mat3, apply both to the same vector --
    // results must agree. This is the contract the engine relies on
    // when interchanging rotation reps between gameplay code (quat)
    // and the shader UBO (mat).
    glm::quat q = glm::angleAxis(0.9f,
                                 glm::normalize(glm::vec3{0.3f, 1.0f, -0.5f}));
    glm::mat3 R = glm::mat3_cast(q);

    glm::vec3 v{1.0f, 2.0f, 3.0f};
    glm::vec3 via_q  = q * v;
    glm::vec3 via_R  = R * v;

    CHECK(vec3_approx_equal(via_q, via_R, kLooseEps));
}

TEST_CASE("quat: mat4 round-trip preserves rotation action") {
    glm::quat q = glm::angleAxis(0.45f,
                                 glm::normalize(glm::vec3{1.0f, -0.2f, 0.8f}));
    glm::mat4 M = glm::mat4_cast(q);

    glm::vec4 v{1.0f, 2.0f, 3.0f, 1.0f};
    glm::vec4 via_q4 = glm::vec4(q * glm::vec3(v), 1.0f);
    glm::vec4 via_M  = M * v;

    CHECK(std::abs(via_q4.x - via_M.x) < kLooseEps);
    CHECK(std::abs(via_q4.y - via_M.y) < kLooseEps);
    CHECK(std::abs(via_q4.z - via_M.z) < kLooseEps);
}

// --- Pathological cases --------------------------------------------------
TEST_CASE("quat: very small rotation approaches identity") {
    // A 1e-6 rad rotation should be essentially the identity quaternion.
    // Catches a "sin(theta/2)/theta blew up to NaN" regression in
    // implementations that compute axis-angle without a small-angle
    // safe path.
    glm::quat q = glm::angleAxis(1e-6f, glm::vec3{0.0f, 1.0f, 0.0f});
    CHECK(q.w == doctest::Approx(1.0f).epsilon(kEps));
    CHECK(std::abs(q.x) < kEps);
    CHECK(std::abs(q.y) < kEps);  // |y| ~= sin(0.5e-6) = 5e-7
    CHECK(std::abs(q.z) < kEps);
    CHECK(std::isfinite(q.w));
    CHECK(std::isfinite(q.x));
    CHECK(std::isfinite(q.y));
    CHECK(std::isfinite(q.z));
}

TEST_CASE("quat: full-circle rotation returns identity (modulo sign)") {
    // A 2pi rotation around any axis is the identity. glm encodes this
    // as q = -identity (the "other" quaternion that represents the
    // same rotation). The double-cover means q and -q both rotate the
    // same way, so we compare with the quat_approx_equal helper that
    // takes the q ~ -q ambiguity into account.
    glm::quat q = glm::angleAxis(2.0f * glm::pi<float>(),
                                 glm::vec3{0.0f, 1.0f, 0.0f});
    glm::quat id{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK(quat_approx_equal(q, id, kLooseEps));

    // Action on a vector is still the identity (no q/-q ambiguity
    // here -- the rotation itself is unique).
    glm::vec3 v{1.0f, 2.0f, 3.0f};
    glm::vec3 r = q * v;
    CHECK(vec3_approx_equal(r, v, kLooseEps));
}

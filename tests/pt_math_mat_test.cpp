// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for mat3/mat4 ops (issue #63 -- Phase 3 of #47).
//
// The engine builds view + projection matrices via glm::lookAtRH /
// glm::perspectiveRH_ZO (Engine.cpp:3584-3586) and passes them straight
// into push-constants and shader uniforms. The matrix ops tested here
// are exactly the ones the engine consumes:
//
//   - Identity multiplication: I * M == M
//   - Composition: rotate then translate vs translate then rotate
//     (catches a column-major / row-major regression)
//   - Inverse: A * A^-1 == I within float epsilon
//   - Transpose: transpose(transpose(A)) == A
//   - Determinant for translation-only and rotation-only matrices
//
// glm is column-major by default, which matches GLSL and Slang's
// `column_major` storage. If a future change set `GLM_FORCE_ROW_MAJOR`
// or similar, every shader matrix upload would silently transpose --
// these tests pin that contract.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>

namespace {

constexpr float kEps      = 1e-5f;  // strict identity check tolerance
constexpr float kLooseEps = 1e-4f;  // composition / inverse tolerance

// Helper: are two matrices equal within an absolute tolerance?
// Iterating column-by-column matches glm's storage layout, but the
// test is layout-agnostic since we read by mat[col][row] index.
bool mat_approx_equal(const glm::mat4& a, const glm::mat4& b, float eps) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            if (std::abs(a[c][r] - b[c][r]) > eps) return false;
        }
    }
    return true;
}
bool mat_approx_equal(const glm::mat3& a, const glm::mat3& b, float eps) {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            if (std::abs(a[c][r] - b[c][r]) > eps) return false;
        }
    }
    return true;
}

}  // namespace

// --- mat3 -----------------------------------------------------------------
TEST_CASE("mat3: identity * M == M") {
    glm::mat3 I(1.0f);  // glm ctor with scalar makes the identity matrix
    glm::mat3 M{
        glm::vec3{1.0f, 2.0f, 3.0f},
        glm::vec3{4.0f, 5.0f, 6.0f},
        glm::vec3{7.0f, 8.0f, 9.0f}
    };
    glm::mat3 product = I * M;
    CHECK(mat_approx_equal(product, M, kEps));

    // And right-side: M * I == M.
    glm::mat3 product_r = M * I;
    CHECK(mat_approx_equal(product_r, M, kEps));
}

TEST_CASE("mat3: rotation matrix is orthonormal") {
    // 90-deg rotation about z, in 3x3 form (no translation). Equivalent
    // to the upper-left 3x3 of a glm::rotate around +z.
    const float a = glm::pi<float>() * 0.5f;
    const float c = std::cos(a);
    const float s = std::sin(a);

    // Column-major: column 0 = rotated +X, column 1 = rotated +Y.
    glm::mat3 R{
        glm::vec3{c, s, 0.0f},
        glm::vec3{-s, c, 0.0f},
        glm::vec3{0.0f, 0.0f, 1.0f}
    };

    // Orthonormal: R * R^T == I (within FP eps).
    glm::mat3 RRt = R * glm::transpose(R);
    CHECK(mat_approx_equal(RRt, glm::mat3(1.0f), kLooseEps));

    // Determinant is +1 for a proper rotation (not -1 for reflection).
    CHECK(glm::determinant(R) == doctest::Approx(1.0f).epsilon(kLooseEps));
}

TEST_CASE("mat3: transpose round-trip") {
    glm::mat3 M{
        glm::vec3{1.1f, 2.2f, 3.3f},
        glm::vec3{4.4f, 5.5f, 6.6f},
        glm::vec3{7.7f, 8.8f, 9.9f}
    };
    glm::mat3 MtT = glm::transpose(glm::transpose(M));
    CHECK(mat_approx_equal(MtT, M, kEps));
}

// --- mat4 -----------------------------------------------------------------
TEST_CASE("mat4: identity * M == M") {
    glm::mat4 I(1.0f);
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3{1.0f, 2.0f, 3.0f});
    CHECK(mat_approx_equal(I * M, M, kEps));
    CHECK(mat_approx_equal(M * I, M, kEps));
}

TEST_CASE("mat4: translation matrix is well-formed") {
    // glm::translate(I, t) emits a matrix whose 4th column is (t, 1)
    // and upper-left 3x3 is identity. Pin this -- the engine reads the
    // 4th column when extracting world translation.
    glm::vec3 t{1.5f, -2.0f, 3.25f};
    glm::mat4 T = glm::translate(glm::mat4(1.0f), t);

    // Column 3 (the translation column) -- glm is column-major.
    CHECK(T[3][0] == doctest::Approx(t.x));
    CHECK(T[3][1] == doctest::Approx(t.y));
    CHECK(T[3][2] == doctest::Approx(t.z));
    CHECK(T[3][3] == doctest::Approx(1.0f));

    // Diagonal is 1.
    CHECK(T[0][0] == doctest::Approx(1.0f));
    CHECK(T[1][1] == doctest::Approx(1.0f));
    CHECK(T[2][2] == doctest::Approx(1.0f));

    // Determinant of a pure translation is 1.
    CHECK(glm::determinant(T) == doctest::Approx(1.0f));
}

TEST_CASE("mat4: rotation matrix is orthonormal and determinant 1") {
    // 90deg rotation around +Y.
    const float a = glm::pi<float>() * 0.5f;
    glm::mat4 R = glm::rotate(glm::mat4(1.0f), a, glm::vec3{0.0f, 1.0f, 0.0f});

    // R * R^T == I for any proper rotation.
    glm::mat4 RRt = R * glm::transpose(R);
    CHECK(mat_approx_equal(RRt, glm::mat4(1.0f), kLooseEps));

    // det(R) == +1 (proper rotation, not reflection).
    CHECK(glm::determinant(R) == doctest::Approx(1.0f).epsilon(kLooseEps));

    // Apply to the +X basis: rotating +X by 90deg around Y should yield -Z.
    glm::vec4 x_axis{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 rotated = R * x_axis;
    CHECK(rotated.x == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(rotated.y == doctest::Approx(0.0f).epsilon(kLooseEps));
    CHECK(rotated.z == doctest::Approx(-1.0f).epsilon(kLooseEps));
}

TEST_CASE("mat4: composition is non-commutative (TR vs RT)") {
    // T*R applies R first then T; R*T applies T first then R.
    // Composing in opposite order produces materially different
    // matrices -- if a refactor accidentally swaps the multiplication
    // order, the camera basis flips and the test catches it.
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3{1.0f, 0.0f, 0.0f});
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              glm::pi<float>() * 0.5f,
                              glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 TR = T * R;  // rotate then translate
    glm::mat4 RT = R * T;  // translate then rotate

    // The two should NOT be approximately equal -- non-commutative
    // composition is the whole point of testing this.
    CHECK_FALSE(mat_approx_equal(TR, RT, 1e-3f));

    // But applying both to the origin gives the same point only if
    // the translation is along the rotation axis (it's not here, so
    // the points differ). Easier check: TR.[3] != RT.[3].
    //
    // Doctest's expression decomposer doesn't support `&&` inside
    // CHECK_FALSE (it errors with "Expression Too Complex"). Compute
    // the conjunction first as a plain bool, then assert on that.
    const bool tr_eq_rt_translation =
        (std::abs(TR[3][0] - RT[3][0]) <= kLooseEps) &&
        (std::abs(TR[3][2] - RT[3][2]) <= kLooseEps);
    CHECK_FALSE(tr_eq_rt_translation);
}

TEST_CASE("mat4: inverse undoes the original") {
    // Affine matrix: translation + rotation + uniform scale, the same
    // shape the engine builds for object-to-world transforms.
    glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3{2.0f, -1.5f, 0.75f});
    glm::mat4 R = glm::rotate(glm::mat4(1.0f),
                              0.7f,
                              glm::normalize(glm::vec3{1.0f, 1.0f, 1.0f}));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3{1.5f, 1.5f, 1.5f});
    glm::mat4 M = T * R * S;

    glm::mat4 Minv = glm::inverse(M);
    glm::mat4 should_be_identity = M * Minv;
    CHECK(mat_approx_equal(should_be_identity, glm::mat4(1.0f), kLooseEps));

    // Inverse on the other side too.
    glm::mat4 should_also_be_identity = Minv * M;
    CHECK(mat_approx_equal(should_also_be_identity, glm::mat4(1.0f), kLooseEps));
}

TEST_CASE("mat4: scale matrix scales components correctly") {
    glm::vec3 s{2.0f, 3.0f, 0.5f};
    glm::mat4 S = glm::scale(glm::mat4(1.0f), s);

    glm::vec4 p{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 scaled = S * p;
    CHECK(scaled.x == doctest::Approx(2.0f));
    CHECK(scaled.y == doctest::Approx(3.0f));
    CHECK(scaled.z == doctest::Approx(0.5f));
    CHECK(scaled.w == doctest::Approx(1.0f));

    // Determinant of a diagonal scale is the product of the diagonals.
    CHECK(glm::determinant(S) == doctest::Approx(s.x * s.y * s.z));
}

TEST_CASE("mat4: column-major storage is contractually fixed") {
    // glm stores matrices column-major. The engine uses
    // glm::value_ptr() to memcpy 16 floats into a uniform buffer; if
    // glm ever switched to row-major silently, every shader matrix
    // upload would transpose. This test pins the contract by reading
    // the matrix data via the public mat[col][row] interface and the
    // pointer interface and checking they agree.
    glm::mat4 M{
        glm::vec4{1.0f,  2.0f,  3.0f,  4.0f},   // column 0
        glm::vec4{5.0f,  6.0f,  7.0f,  8.0f},   // column 1
        glm::vec4{9.0f,  10.0f, 11.0f, 12.0f},  // column 2
        glm::vec4{13.0f, 14.0f, 15.0f, 16.0f}   // column 3
    };

    // mat[col] retrieves the column as a vec4.
    CHECK(M[0][0] == doctest::Approx(1.0f));
    CHECK(M[0][1] == doctest::Approx(2.0f));
    CHECK(M[0][2] == doctest::Approx(3.0f));
    CHECK(M[0][3] == doctest::Approx(4.0f));

    // Last column.
    CHECK(M[3][0] == doctest::Approx(13.0f));
    CHECK(M[3][3] == doctest::Approx(16.0f));

    // Memory layout: column-major means &M[0][0] is the first float of
    // column 0, &M[0][0]+4 is the first float of column 1, etc.
    const float* p = &M[0][0];
    CHECK(p[0]  == doctest::Approx(1.0f));   // column 0, row 0
    CHECK(p[1]  == doctest::Approx(2.0f));   // column 0, row 1
    CHECK(p[4]  == doctest::Approx(5.0f));   // column 1, row 0
    CHECK(p[15] == doctest::Approx(16.0f));  // column 3, row 3
}

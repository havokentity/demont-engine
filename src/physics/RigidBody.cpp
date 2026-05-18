// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "RigidBody.h"

#include <cmath>

#include <glm/gtc/quaternion.hpp>

namespace pt::physics {

glm::vec3 InvInertiaSphere(float mass, float radius) {
    if (mass <= 0.0f || radius <= 0.0f) return glm::vec3{0.0f};
    // Solid sphere: I = (2/5) m r^2, isotropic. Inverse = 1 / I.
    const float i = 0.4f * mass * radius * radius;
    const float inv = 1.0f / i;
    return glm::vec3{inv, inv, inv};
}

glm::vec3 InvInertiaBox(float mass, const glm::vec3& h) {
    if (mass <= 0.0f) return glm::vec3{0.0f};
    if (h.x <= 0.0f || h.y <= 0.0f || h.z <= 0.0f) return glm::vec3{0.0f};
    // Solid box (half-extents form): I = m/3 * diag(hy^2+hz^2,
    // hx^2+hz^2, hx^2+hy^2). Derivation: textbook full-extents form
    // I = m/12 * (W^2 + H^2) for each axis, and W = 2*hx so W^2 = 4 hx^2
    // -- the 4 cancels with the 1/12 to give 1/3. We use half-extents
    // consistently because they line up with the SDF / OBB conventions
    // used elsewhere in the engine.
    const float ix = (mass / 3.0f) * (h.y * h.y + h.z * h.z);
    const float iy = (mass / 3.0f) * (h.x * h.x + h.z * h.z);
    const float iz = (mass / 3.0f) * (h.x * h.x + h.y * h.y);
    return glm::vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
}

glm::mat3 WorldInverseInertia(const glm::quat& q, const glm::vec3& iinv) {
    // I_world^-1 = R * diag(iinv) * R^T. glm::mat3_cast gives us R
    // directly from the quaternion. For diagonal local inertia this
    // collapses to a column-scaled 3x3 multiply, but writing it
    // straight is honest and the compiler vectorises it fine on M4.
    const glm::mat3 R = glm::mat3_cast(q);
    glm::mat3 D{0.0f};
    D[0][0] = iinv.x;
    D[1][1] = iinv.y;
    D[2][2] = iinv.z;
    return R * D * glm::transpose(R);
}

glm::quat IntegrateOrientation(const glm::quat& q,
                               const glm::vec3& omega,
                               float dt) {
    // Mueller 2020 Equation 12: q_new = q + 0.5 * (omega * q) * dt.
    // The "omega * q" here is the quaternion product of the pure-
    // imaginary quaternion (0, omega) with q -- glm's operator* on a
    // quaternion built from (w=0, xyz=omega) does exactly that. Then
    // we normalise so floating-point drift over many substeps doesn't
    // turn the quaternion into a not-quite-unit vector.
    const glm::quat omega_q{0.0f, omega.x, omega.y, omega.z};
    glm::quat dq = omega_q * q;
    glm::quat out{
        q.w + 0.5f * dq.w * dt,
        q.x + 0.5f * dq.x * dt,
        q.y + 0.5f * dq.y * dt,
        q.z + 0.5f * dq.z * dt,
    };
    // Normalise -- skip if (numerically) zero, which can happen if
    // both q and omega were zero on a freshly-constructed body that
    // hasn't been initialised yet. Returning identity in that edge
    // case keeps callers from having to special-case it.
    const float n2 = out.w * out.w + out.x * out.x + out.y * out.y + out.z * out.z;
    if (n2 < 1e-20f) {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float inv = 1.0f / std::sqrt(n2);
    return glm::quat{out.w * inv, out.x * inv, out.y * inv, out.z * inv};
}

glm::vec3 OrientationDiffOmega(const glm::quat& prev,
                               const glm::quat& curr,
                               float dt) {
    if (dt <= 0.0f) return glm::vec3{0.0f};
    // omega = 2 * (curr * prev^-1).xyz / dt, with a hemisphere fix so
    // a 180-degree flip doesn't read as a 360-degree spin. This is the
    // standard finite-difference inverse of IntegrateOrientation, used
    // at the end of every XPBD substep to refresh body omega from the
    // constrained orientation pair. Mueller 2020 Equation 14.
    const glm::quat dq = curr * glm::conjugate(prev);
    glm::vec3 axis{dq.x, dq.y, dq.z};
    // The hemisphere fix: if dq.w < 0 the rotation is going the "long
    // way" around, and the imaginary part needs negating for the omega
    // direction to match the actual motion. Mueller's paper writes
    // this as `2 * sign(w) * axis / dt`.
    const float sgn = (dq.w < 0.0f) ? -1.0f : 1.0f;
    return (2.0f * sgn / dt) * axis;
}

}  // namespace pt::physics

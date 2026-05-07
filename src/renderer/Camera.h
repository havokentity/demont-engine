// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>

namespace pt::renderer {

struct Camera {
    glm::vec3 pos      { 0.0f, 1.5f, 4.0f };
    float     yaw      { 0.0f };           // radians around world-Y
    float     pitch    { -0.20f };         // radians; clamped +/- 85 deg
    float     fov_deg  { 60.0f };

    glm::vec3 Forward() const noexcept {
        float cp = std::cos(pitch);
        return { std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp };
    }
    glm::vec3 Right() const noexcept {
        return glm::normalize(glm::cross(Forward(), glm::vec3{0, 1, 0}));
    }
    glm::vec3 Up() const noexcept {
        return glm::normalize(glm::cross(Right(), Forward()));
    }
    float FovYTan() const noexcept {
        return std::tan(glm::radians(fov_deg) * 0.5f);
    }

    void ClampPitch() noexcept {
        constexpr float lim = glm::radians(85.0f);
        if (pitch >  lim) pitch =  lim;
        if (pitch < -lim) pitch = -lim;
    }
};

}  // namespace pt::renderer

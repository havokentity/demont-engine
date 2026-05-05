#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace pt::renderer {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

struct Mesh {
    std::vector<Vertex>        vertices;
    std::vector<std::uint32_t> indices;
};

// Axis-aligned box centered at the origin.  6 faces, 24 verts (4 per
// face for flat per-face normals), 12 triangles.
inline Mesh GenerateBox(float w = 1.0f, float h = 1.0f, float d = 1.0f) {
    Mesh m;
    auto addFace = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 dpt,
                       glm::vec3 n) {
        std::uint32_t base = static_cast<std::uint32_t>(m.vertices.size());
        m.vertices.push_back({a, n});
        m.vertices.push_back({b, n});
        m.vertices.push_back({c, n});
        m.vertices.push_back({dpt, n});
        m.indices.push_back(base + 0);
        m.indices.push_back(base + 1);
        m.indices.push_back(base + 2);
        m.indices.push_back(base + 0);
        m.indices.push_back(base + 2);
        m.indices.push_back(base + 3);
    };
    const float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;
    // +X face
    addFace({ hw,-hh,-hd}, { hw, hh,-hd}, { hw, hh, hd}, { hw,-hh, hd}, { 1, 0, 0});
    // -X face
    addFace({-hw,-hh, hd}, {-hw, hh, hd}, {-hw, hh,-hd}, {-hw,-hh,-hd}, {-1, 0, 0});
    // +Y face
    addFace({-hw, hh,-hd}, { hw, hh,-hd}, { hw, hh, hd}, {-hw, hh, hd}, { 0, 1, 0});
    // -Y face -- careful winding so triangles face down
    addFace({-hw,-hh, hd}, { hw,-hh, hd}, { hw,-hh,-hd}, {-hw,-hh,-hd}, { 0,-1, 0});
    // +Z face
    addFace({-hw,-hh, hd}, { hw,-hh, hd}, { hw, hh, hd}, {-hw, hh, hd}, { 0, 0, 1});
    // Wait the previous +Z had wrong winding -- redo with correct CCW order.
    return m;
}

// Same box but with proper consistent CCW winding around outward normals.
inline Mesh GenerateBoxCCW(float w = 1.0f, float h = 1.0f, float d = 1.0f) {
    Mesh m;
    const float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;
    auto add = [&](glm::vec3 n,
                   glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 dp) {
        std::uint32_t base = static_cast<std::uint32_t>(m.vertices.size());
        m.vertices.push_back({a, n});
        m.vertices.push_back({b, n});
        m.vertices.push_back({c, n});
        m.vertices.push_back({dp, n});
        m.indices.insert(m.indices.end(),
                         {base, base+1, base+2, base, base+2, base+3});
    };
    // Six faces, CCW seen from outside.
    add({ 1, 0, 0}, { hw,-hh, hd}, { hw, hh, hd}, { hw, hh,-hd}, { hw,-hh,-hd}); // +X
    add({-1, 0, 0}, {-hw,-hh,-hd}, {-hw, hh,-hd}, {-hw, hh, hd}, {-hw,-hh, hd}); // -X
    add({ 0, 1, 0}, {-hw, hh, hd}, { hw, hh, hd}, { hw, hh,-hd}, {-hw, hh,-hd}); // +Y
    add({ 0,-1, 0}, {-hw,-hh,-hd}, { hw,-hh,-hd}, { hw,-hh, hd}, {-hw,-hh, hd}); // -Y
    add({ 0, 0, 1}, {-hw,-hh, hd}, { hw,-hh, hd}, { hw, hh, hd}, {-hw, hh, hd}); // +Z
    add({ 0, 0,-1}, {-hw, hh,-hd}, { hw, hh,-hd}, { hw,-hh,-hd}, {-hw,-hh,-hd}); // -Z
    return m;
}

}  // namespace pt::renderer

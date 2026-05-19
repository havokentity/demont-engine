// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "SceneGraph.h"

#include "../engine/Engine.h"
#include "../physics/PhysicsSystem.h"
#include "../physics/RigidBody.h"
#include "../renderer/AnalyticBvh.h"

#include <cstring>
#include <string_view>

namespace pt::editor {

namespace {

using json = nlohmann::json;

// Stringify the host-side AnalyticPrim::Material enum into the lowercase
// names the editor JSON schema uses. Mirrors MaterialName in Engine.cpp
// but kept local so SceneGraph.cpp doesn't have to reach into an
// anonymous Engine.cpp namespace.
const char* MaterialName(pt::engine::Engine::AnalyticPrim::Material m) {
    switch (m) {
        case pt::engine::Engine::AnalyticPrim::Lambert:    return "lambert";
        case pt::engine::Engine::AnalyticPrim::Metal:      return "metal";
        case pt::engine::Engine::AnalyticPrim::Dielectric: return "dielectric";
        case pt::engine::Engine::AnalyticPrim::Water:      return "water";
    }
    return "lambert";
}

// SDF cluster material enum (0=Lambert, 1=Metal, 2=Dielectric) -- the
// SDF set has its own integer encoding distinct from AnalyticPrim. See
// Engine::RegisterSdfCommands :: ParseSdfMaterial for the matching parser.
const char* SdfMaterialName(std::uint32_t m) {
    switch (m) {
        case 0u: return "lambert";
        case 1u: return "metal";
        case 2u: return "dielectric";
    }
    return "lambert";
}

// AnalyticLight::Type enum -> lowercase string.
const char* LightTypeName(pt::engine::Engine::AnalyticLight::Type t) {
    switch (t) {
        case pt::engine::Engine::AnalyticLight::Point:  return "point";
        case pt::engine::Engine::AnalyticLight::Spot:   return "spot";
        case pt::engine::Engine::AnalyticLight::Sphere: return "sphere";
        case pt::engine::Engine::AnalyticLight::Quad:   return "quad";
    }
    return "point";
}

// Local-only rigid-body collector. ForEachRigidBody takes a C function
// pointer + opaque user data (no captures) so we shape `Collector` to
// match. Each call appends one JSON object to the user-supplied array.
struct RbCollector {
    json* arr;
};

void CollectRigidBody(pt::physics::PhysicsSystem::RbHandle h,
                      const pt::physics::RigidBody& b,
                      std::uint32_t prim_id, void* user) {
    auto* c = static_cast<RbCollector*>(user);
    const bool is_kinematic = (b.inv_mass == 0.0f);
    const float mass = is_kinematic ? 0.0f : (1.0f / b.inv_mass);
    // Verlet implicit velocity: (curr - prev) / dt. We don't have dt
    // here so the velocity reported is the per-substep delta, which is
    // what the engine itself writes back. Editor consumers should treat
    // it as an instantaneous velocity sample (already in m / substep,
    // which equals m/s at the default 60 Hz substep). Good enough for
    // the property panel; a future Phase X can promote the physics
    // system to track velocity explicitly.
    const float vx = b.curr_pos.x - b.prev_pos.x;
    const float vy = b.curr_pos.y - b.prev_pos.y;
    const float vz = b.curr_pos.z - b.prev_pos.z;
    json j = {
        {"id",           h},
        {"prim_id",      prim_id},
        {"pos",          {b.curr_pos.x, b.curr_pos.y, b.curr_pos.z}},
        {"velocity",     {vx, vy, vz}},
        {"mass",         mass},
        {"radius",       b.radius},
        {"is_kinematic", is_kinematic},
        {"shape",        (b.shape == pt::physics::Shape::Sphere) ? "sphere" : "box"},
    };
    if (b.shape == pt::physics::Shape::Box) {
        j["half_extents"] = {b.half_extents.x, b.half_extents.y, b.half_extents.z};
    }
    c->arr->push_back(std::move(j));
}

}  // namespace

const char* SelectionKindToString(int kind) {
    switch (static_cast<pt::engine::SelectionKind>(kind)) {
        case pt::engine::SelectionKind::None:         return "none";
        case pt::engine::SelectionKind::AnalyticPrim: return "prim";
        case pt::engine::SelectionKind::Light:        return "light";
        case pt::engine::SelectionKind::SdfCluster:   return "sdf";
        case pt::engine::SelectionKind::RigidBody:    return "rb";
    }
    return "none";
}

int SelectionKindFromString(std::string_view s) {
    if (s == "prim")  return static_cast<int>(pt::engine::SelectionKind::AnalyticPrim);
    if (s == "light") return static_cast<int>(pt::engine::SelectionKind::Light);
    if (s == "sdf")   return static_cast<int>(pt::engine::SelectionKind::SdfCluster);
    if (s == "rb")    return static_cast<int>(pt::engine::SelectionKind::RigidBody);
    return static_cast<int>(pt::engine::SelectionKind::None);
}

json SerializeScene(const pt::engine::Engine& engine) {
    json out;

    // Analytic primitives.
    json prims = json::array();
    for (const auto& [id, p] : engine.Primitives()) {
        json j = {
            {"id",        id},
            {"type",      (p.type == pt::engine::Engine::AnalyticPrim::Sphere) ? "sphere" : "plane"},
            {"material",  MaterialName(p.material)},
            {"albedo",    {p.albedo[0],   p.albedo[1],   p.albedo[2]}},
            {"emission",  {p.emission[0], p.emission[1], p.emission[2]}},
            {"roughness", p.roughness},
            {"ior",       p.ior},
        };
        if (p.type == pt::engine::Engine::AnalyticPrim::Sphere) {
            j["pos"]    = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["radius"] = p.radius_or_d;
        } else {
            j["normal"] = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["d"]      = p.radius_or_d;
        }
        prims.push_back(std::move(j));
    }
    out["primitives"] = std::move(prims);

    // Analytic lights.
    json lights = json::array();
    for (const auto& [id, L] : engine.Lights()) {
        json j = {
            {"id",        id},
            {"type",      LightTypeName(L.type)},
            {"pos",       {L.pos[0], L.pos[1], L.pos[2]}},
            {"intensity", {L.intensity[0], L.intensity[1], L.intensity[2]}},
        };
        if (L.type == pt::engine::Engine::AnalyticLight::Spot) {
            j["dir"]       = {L.dir[0], L.dir[1], L.dir[2]};
            j["cos_outer"] = L.cos_outer;
            j["cos_inner"] = L.cos_inner;
        }
        if (L.type == pt::engine::Engine::AnalyticLight::Sphere) {
            j["radius"]    = L.radius;
        }
        if (L.type == pt::engine::Engine::AnalyticLight::Quad) {
            j["dir"]    = {L.dir[0], L.dir[1], L.dir[2]};
            j["u_vec"]  = {L.u_vec[0], L.u_vec[1], L.u_vec[2]};
            j["v_half"] = L.v_half;
        }
        lights.push_back(std::move(j));
    }
    out["lights"] = std::move(lights);

    // SDF clusters.
    json sdfs = json::array();
    for (const auto& [id, S] : engine.SdfPrims()) {
        json j = {
            {"id",         id},
            {"node_count", S.node_count},
            {"aabb_min",   {S.aabb_min[0], S.aabb_min[1], S.aabb_min[2]}},
            {"aabb_max",   {S.aabb_max[0], S.aabb_max[1], S.aabb_max[2]}},
            {"material",   SdfMaterialName(S.material)},
            {"albedo",     {S.albedo[0], S.albedo[1], S.albedo[2]}},
            {"roughness",  S.roughness},
            {"ior",        S.ior},
        };
        sdfs.push_back(std::move(j));
    }
    out["sdf"] = std::move(sdfs);

    // Rigid bodies. ForEachRigidBody takes a static fn pointer + opaque
    // user pointer (no captures), so we wrap the destination JSON
    // array in a tiny adaptor struct.
    json rbs = json::array();
    if (auto* phys = engine.Physics(); phys != nullptr) {
        RbCollector ctx{ &rbs };
        phys->ForEachRigidBody(&CollectRigidBody, &ctx);
    }
    out["rigid_bodies"] = std::move(rbs);

    // Current selection snapshot. The kind string mirrors the
    // selection_change WebSocket event so UIs that subscribe to the
    // event can use the same parser.
    out["selection"] = {
        {"kind", SelectionKindToString(static_cast<int>(engine.GetSelectionKind()))},
        {"id",   engine.GetSelectionId()},
    };

    return out;
}

}  // namespace pt::editor

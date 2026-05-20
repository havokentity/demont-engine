// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "SceneGraph.h"

#include "../engine/Engine.h"
#include "../physics/PhysicsSystem.h"
#include "../physics/RigidBody.h"
#include "../renderer/AnalyticBvh.h"

#include <array>
#include <cstring>
#include <string>
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
            // Orientation quaternion (xyzw). Identity (0,0,0,1) for
            // non-rotated prims. Drives the rotate gizmo (#206); the
            // Property Inspector / Scene Hierarchy panels read this
            // to display the prim's rotation state.
            {"orient",    {p.orient[0], p.orient[1], p.orient[2], p.orient[3]}},
        };
        if (p.type == pt::engine::Engine::AnalyticPrim::Sphere) {
            j["pos"]    = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["radius"] = p.radius_or_d;
        } else {
            j["normal"] = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["d"]      = p.radius_or_d;
        }
        // Wave 8 PBR (#26) texture-map state for the Material Editor panel
        // (wave-9). For each of the four maps we emit a boolean "is a
        // texture assigned" plus the source path (empty when flat). The
        // path lets the panel display + dedupe the assignment; the
        // boolean keeps consumers that don't care about the path simple.
        // kNoTexTile (0xFFFFFFFF) -> flat (no texture), matching the
        // shader's per-channel gate. PbrTilePath returns "" for kNoTexTile.
        constexpr std::uint32_t kNoTex = 0xFFFFFFFFu;
        j["albedo_tex"]         = (p.albedo_tex    != kNoTex);
        j["normal_tex"]         = (p.normal_tex    != kNoTex);
        j["roughness_tex"]      = (p.roughness_tex != kNoTex);
        j["metallic_tex"]       = (p.metallic_tex  != kNoTex);
        j["albedo_tex_path"]    = engine.PbrTilePath(p.albedo_tex);
        j["normal_tex_path"]    = engine.PbrTilePath(p.normal_tex);
        j["roughness_tex_path"] = engine.PbrTilePath(p.roughness_tex);
        j["metallic_tex_path"]  = engine.PbrTilePath(p.metallic_tex);
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
            // Orientation quaternion (xyzw). Identity (0,0,0,1) for
            // non-rotated lights. Rotation gizmo follow-up to #206:
            // spot's cone axis and quad's normal + u-extent are
            // rotated through this at shader time. Point / sphere
            // lights are rotation-symmetric so the value is inert
            // but still surfaced so inspector panels read a stable
            // orientation state.
            {"orient",    {L.orient[0], L.orient[1], L.orient[2], L.orient[3]}},
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

// --- Wave 9 scene save/load ------------------------------------------------

namespace {

// Format version baked into the document so a future incompatible
// schema change can be detected. Bump only on a breaking change; the
// reader accepts any version <= kSceneFormatVersion (forward fields are
// ignored, missing fields default).
constexpr int kSceneFormatVersion = 1;

using AnalyticPrim  = pt::engine::Engine::AnalyticPrim;
using AnalyticLight = pt::engine::Engine::AnalyticLight;

// ---- enum <-> string round-trip helpers -----------------------------------

AnalyticPrim::Material PrimMaterialFromName(std::string_view s) {
    if (s == "metal")      return AnalyticPrim::Metal;
    if (s == "dielectric") return AnalyticPrim::Dielectric;
    if (s == "water")      return AnalyticPrim::Water;
    return AnalyticPrim::Lambert;
}

std::uint32_t SdfMaterialFromName(std::string_view s) {
    if (s == "metal")      return 1u;
    if (s == "dielectric") return 2u;
    return 0u;  // lambert
}

AnalyticLight::Type LightTypeFromName(std::string_view s) {
    if (s == "spot")   return AnalyticLight::Spot;
    if (s == "sphere") return AnalyticLight::Sphere;
    if (s == "quad")   return AnalyticLight::Quad;
    return AnalyticLight::Point;
}

// SDF leaf-shape <-> string. Mirrors pt::renderer::SdfShape; kept as a
// string in the file so the wire format survives an enum renumber.
const char* SdfShapeName(std::uint32_t shape) {
    switch (shape) {
        case pt::renderer::SDF_SHAPE_SPHERE:      return "sphere";
        case pt::renderer::SDF_SHAPE_BOX:         return "box";
        case pt::renderer::SDF_SHAPE_ROUNDED_BOX: return "rounded_box";
        case pt::renderer::SDF_SHAPE_TORUS:       return "torus";
        case pt::renderer::SDF_SHAPE_CAPSULE:     return "capsule";
        case pt::renderer::SDF_SHAPE_PLANE:       return "plane";
    }
    return "sphere";
}
std::uint32_t SdfShapeFromName(std::string_view s) {
    if (s == "box")         return pt::renderer::SDF_SHAPE_BOX;
    if (s == "rounded_box") return pt::renderer::SDF_SHAPE_ROUNDED_BOX;
    if (s == "torus")       return pt::renderer::SDF_SHAPE_TORUS;
    if (s == "capsule")     return pt::renderer::SDF_SHAPE_CAPSULE;
    if (s == "plane")       return pt::renderer::SDF_SHAPE_PLANE;
    return pt::renderer::SDF_SHAPE_SPHERE;
}

const char* SdfOpName(std::uint32_t op) {
    switch (op) {
        case pt::renderer::SDF_OP_LEAF:             return "leaf";
        case pt::renderer::SDF_OP_SMOOTH_UNION:     return "smooth_union";
        case pt::renderer::SDF_OP_SMOOTH_SUBTRACT:  return "smooth_subtract";
        case pt::renderer::SDF_OP_SMOOTH_INTERSECT: return "smooth_intersect";
        case pt::renderer::SDF_OP_DISPLACE:         return "displace";
    }
    return "leaf";
}
std::uint32_t SdfOpFromName(std::string_view s) {
    if (s == "smooth_union")     return pt::renderer::SDF_OP_SMOOTH_UNION;
    if (s == "smooth_subtract")  return pt::renderer::SDF_OP_SMOOTH_SUBTRACT;
    if (s == "smooth_intersect") return pt::renderer::SDF_OP_SMOOTH_INTERSECT;
    if (s == "displace")         return pt::renderer::SDF_OP_DISPLACE;
    return pt::renderer::SDF_OP_LEAF;
}

// ---- JSON array readers (fixed-length, default-preserving) -----------------
//
// Read an N-element float array from `node[key]`; leaves `dst` untouched
// if the field is absent / wrong-shaped (so the struct default stands).
template <std::size_t N>
void ReadFloatN(const json& node, const char* key, float (&dst)[N]) {
    auto it = node.find(key);
    if (it == node.end() || !it->is_array() || it->size() != N) return;
    for (std::size_t i = 0; i < N; ++i) {
        if ((*it)[i].is_number()) dst[i] = (*it)[i].get<float>();
    }
}

float ReadFloat(const json& node, const char* key, float fallback) {
    auto it = node.find(key);
    if (it != node.end() && it->is_number()) return it->get<float>();
    return fallback;
}

std::uint32_t ReadU32(const json& node, const char* key, std::uint32_t fallback) {
    auto it = node.find(key);
    if (it != node.end() && it->is_number_integer()) return it->get<std::uint32_t>();
    if (it != node.end() && it->is_number_unsigned()) return it->get<std::uint32_t>();
    return fallback;
}

std::string ReadStr(const json& node, const char* key, std::string fallback) {
    auto it = node.find(key);
    if (it != node.end() && it->is_string()) return it->get<std::string>();
    return fallback;
}

// Stable JSON id key. ids are user-supplied uint32 map keys.
std::uint32_t ReadId(const json& node, std::uint32_t fallback) {
    return ReadU32(node, "id", fallback);
}

}  // namespace

nlohmann::json SceneToJson(const SceneData& scene) {
    json out;
    out["format"]  = "demont-scene";
    out["version"] = kSceneFormatVersion;

    // Camera. Angles persisted in degrees (human-readable, and matches
    // the cam_yaw / cam_pitch cvar convention).
    if (scene.has_camera) {
        const auto& c = scene.camera;
        out["camera"] = {
            {"pos",   {c.pos.x, c.pos.y, c.pos.z}},
            {"yaw",   glm::degrees(c.yaw)},
            {"pitch", glm::degrees(c.pitch)},
            {"fov",   c.fov_deg},
        };
    }

    // Analytic primitives -- full editable state (geometry, material,
    // albedo, emission, roughness, ior, orientation, texture tiles).
    json prims = json::array();
    for (const auto& [id, p] : scene.prims) {
        json j = {
            {"id",        id},
            {"type",      (p.type == AnalyticPrim::Sphere) ? "sphere" : "plane"},
            {"material",  MaterialName(p.material)},
            {"albedo",    {p.albedo[0],   p.albedo[1],   p.albedo[2]}},
            {"emission",  {p.emission[0], p.emission[1], p.emission[2]}},
            {"roughness", p.roughness},
            {"ior",       p.ior},
            {"orient",    {p.orient[0], p.orient[1], p.orient[2], p.orient[3]}},
        };
        if (p.type == AnalyticPrim::Sphere) {
            j["pos"]    = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["radius"] = p.radius_or_d;
        } else {
            j["normal"] = {p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            j["d"]      = p.radius_or_d;
        }
        // Texture tiles only emitted when set (kNoTexTile == 0xFFFFFFFF).
        // Absent => "no texture", which is the load-time default, so a
        // texture-free prim round-trips to the same bytes.
        constexpr std::uint32_t kNoTex = 0xFFFFFFFFu;
        if (p.albedo_tex    != kNoTex) j["albedo_tex"]    = p.albedo_tex;
        if (p.normal_tex    != kNoTex) j["normal_tex"]    = p.normal_tex;
        if (p.roughness_tex != kNoTex) j["roughness_tex"] = p.roughness_tex;
        if (p.metallic_tex  != kNoTex) j["metallic_tex"]  = p.metallic_tex;
        prims.push_back(std::move(j));
    }
    out["primitives"] = std::move(prims);

    // Analytic lights -- raw struct fields (NOT the degree-based
    // authoring form), so a light round-trips byte-exact regardless of
    // which authoring command (canonical or ergonomic variant) created
    // it. dir / cos_* / u_vec / v_half are written for every type and
    // simply inert for types that don't read them.
    json lights = json::array();
    for (const auto& [id, L] : scene.lights) {
        json j = {
            {"id",        id},
            {"type",      LightTypeName(L.type)},
            {"pos",       {L.pos[0], L.pos[1], L.pos[2]}},
            {"radius",    L.radius},
            {"intensity", {L.intensity[0], L.intensity[1], L.intensity[2]}},
            {"dir",       {L.dir[0], L.dir[1], L.dir[2]}},
            {"cos_outer", L.cos_outer},
            {"cos_inner", L.cos_inner},
            {"u_vec",     {L.u_vec[0], L.u_vec[1], L.u_vec[2]}},
            {"v_half",    L.v_half},
            {"orient",    {L.orient[0], L.orient[1], L.orient[2], L.orient[3]}},
        };
        lights.push_back(std::move(j));
    }
    out["lights"] = std::move(lights);

    // SDF clusters -- full flat node tree (lossless). The AABB is NOT
    // persisted; it's recomputed on load from the nodes.
    json sdfs = json::array();
    for (const auto& [id, S] : scene.sdf) {
        json nodes = json::array();
        for (std::uint32_t i = 0; i < S.node_count && i < pt::renderer::SdfPrim::kMaxNodes; ++i) {
            const auto& n = S.nodes[i];
            nodes.push_back({
                {"op",      SdfOpName(n.op)},
                {"shape",   SdfShapeName(n.shape)},
                {"child_a", n.child_a},
                {"child_b", n.child_b},
                {"params",  {n.params[0], n.params[1], n.params[2], n.params[3]}},
                {"center",  {n.center[0], n.center[1], n.center[2]}},
            });
        }
        sdfs.push_back({
            {"id",        id},
            {"material",  SdfMaterialName(S.material)},
            {"albedo",    {S.albedo[0], S.albedo[1], S.albedo[2]}},
            {"roughness", S.roughness},
            {"ior",       S.ior},
            {"nodes",     std::move(nodes)},
        });
    }
    out["sdf"] = std::move(sdfs);

    // Render cvars (name/value pairs). Emitted as an object; insertion
    // order is preserved by nlohmann's ordered_json default only if the
    // build uses it -- to keep the file stable regardless, we emit an
    // ARRAY of {name,value} so order is guaranteed.
    json cvars = json::array();
    for (const auto& [name, value] : scene.cvars) {
        cvars.push_back({{"name", name}, {"value", value}});
    }
    out["cvars"] = std::move(cvars);

    return out;
}

bool SceneFromJson(const nlohmann::json& doc, SceneData& out, std::string& err) {
    if (!doc.is_object()) {
        err = "scene document root is not a JSON object";
        return false;
    }
    // Version gate. Accept anything we can read (<= current). A document
    // claiming a NEWER version may use fields we don't understand; we
    // still parse best-effort but warn via err being left empty (the
    // caller treats false as fatal, true as success, so we return true
    // and let unknown fields be ignored).
    if (auto it = doc.find("version"); it != doc.end() && it->is_number_integer()) {
        if (it->get<int>() > kSceneFormatVersion) {
            // Soft-accept: newer minor additions are field-additive.
            // (No hard fail -- keeps old binaries able to load files
            // written by a slightly newer one, dropping unknown fields.)
        }
    }

    out = SceneData{};

    // Camera (optional).
    if (auto it = doc.find("camera"); it != doc.end() && it->is_object()) {
        const json& c = *it;
        out.has_camera = true;
        if (auto p = c.find("pos"); p != c.end() && p->is_array() && p->size() == 3) {
            out.camera.pos = glm::vec3((*p)[0].get<float>(),
                                       (*p)[1].get<float>(),
                                       (*p)[2].get<float>());
        }
        out.camera.yaw     = glm::radians(ReadFloat(c, "yaw",   0.0f));
        out.camera.pitch   = glm::radians(ReadFloat(c, "pitch", 0.0f));
        out.camera.fov_deg = ReadFloat(c, "fov",   60.0f);
        out.camera.ClampPitch();
    }

    // Analytic primitives.
    if (auto it = doc.find("primitives"); it != doc.end() && it->is_array()) {
        std::uint32_t auto_id = 1;
        for (const json& j : *it) {
            if (!j.is_object()) continue;
            AnalyticPrim p{};
            const std::string type = ReadStr(j, "type", "sphere");
            p.type = (type == "plane") ? AnalyticPrim::Plane : AnalyticPrim::Sphere;
            if (p.type == AnalyticPrim::Sphere) {
                ReadFloatN(j, "pos", p.pos_or_n);
                p.radius_or_d = ReadFloat(j, "radius", 0.5f);
            } else {
                ReadFloatN(j, "normal", p.pos_or_n);
                p.radius_or_d = ReadFloat(j, "d", 0.0f);
            }
            // Seed prev == curr so a freshly loaded prim doesn't streak
            // from the origin on its first motion-blur frame (#85).
            p.prev_pos_or_n[0] = p.pos_or_n[0];
            p.prev_pos_or_n[1] = p.pos_or_n[1];
            p.prev_pos_or_n[2] = p.pos_or_n[2];
            p.material = PrimMaterialFromName(ReadStr(j, "material", "lambert"));
            ReadFloatN(j, "albedo",   p.albedo);
            ReadFloatN(j, "emission", p.emission);
            p.roughness = ReadFloat(j, "roughness", 0.0f);
            p.ior       = ReadFloat(j, "ior", 1.5f);
            ReadFloatN(j, "orient", p.orient);
            p.albedo_tex    = ReadU32(j, "albedo_tex",    0xFFFFFFFFu);
            p.normal_tex    = ReadU32(j, "normal_tex",    0xFFFFFFFFu);
            p.roughness_tex = ReadU32(j, "roughness_tex", 0xFFFFFFFFu);
            p.metallic_tex  = ReadU32(j, "metallic_tex",  0xFFFFFFFFu);
            out.prims[ReadId(j, auto_id)] = p;
            ++auto_id;
        }
    }

    // Analytic lights.
    if (auto it = doc.find("lights"); it != doc.end() && it->is_array()) {
        std::uint32_t auto_id = 1;
        for (const json& j : *it) {
            if (!j.is_object()) continue;
            AnalyticLight L{};
            L.type = LightTypeFromName(ReadStr(j, "type", "point"));
            ReadFloatN(j, "pos", L.pos);
            L.radius = ReadFloat(j, "radius", 0.0f);
            ReadFloatN(j, "intensity", L.intensity);
            ReadFloatN(j, "dir", L.dir);
            L.cos_outer = ReadFloat(j, "cos_outer", 0.0f);
            L.cos_inner = ReadFloat(j, "cos_inner", 0.0f);
            ReadFloatN(j, "u_vec", L.u_vec);
            L.v_half = ReadFloat(j, "v_half", 0.0f);
            ReadFloatN(j, "orient", L.orient);
            out.lights[ReadId(j, auto_id)] = L;
            ++auto_id;
        }
    }

    // SDF clusters -- rebuild the node tree, then recompute the AABB.
    if (auto it = doc.find("sdf"); it != doc.end() && it->is_array()) {
        std::uint32_t auto_id = 1;
        for (const json& j : *it) {
            if (!j.is_object()) continue;
            pt::renderer::SdfPrim S{};
            S.material  = SdfMaterialFromName(ReadStr(j, "material", "lambert"));
            ReadFloatN(j, "albedo", S.albedo);
            S.roughness = ReadFloat(j, "roughness", 0.0f);
            S.ior       = ReadFloat(j, "ior", 1.5f);
            std::uint32_t count = 0;
            if (auto nit = j.find("nodes"); nit != j.end() && nit->is_array()) {
                for (const json& nj : *nit) {
                    if (count >= pt::renderer::SdfPrim::kMaxNodes) break;
                    if (!nj.is_object()) continue;
                    pt::renderer::SdfNode& n = S.nodes[count];
                    n.op      = SdfOpFromName(ReadStr(nj, "op", "leaf"));
                    n.shape   = SdfShapeFromName(ReadStr(nj, "shape", "sphere"));
                    n.child_a = ReadU32(nj, "child_a", 0);
                    n.child_b = ReadU32(nj, "child_b", 0);
                    ReadFloatN(nj, "params", n.params);
                    ReadFloatN(nj, "center", n.center);
                    ++count;
                }
            }
            S.node_count = count;
            // Recompute the world AABB from the node tree. If the tree is
            // degenerate / empty, ComputeSdfAabb returns false -- skip
            // the cluster rather than upload a zero-extent bound that
            // silently misses every ray.
            if (count > 0 && pt::renderer::ComputeSdfAabb(S)) {
                out.sdf[ReadId(j, auto_id)] = S;
            }
            ++auto_id;
        }
    }

    // Render cvars.
    if (auto it = doc.find("cvars"); it != doc.end() && it->is_array()) {
        for (const json& j : *it) {
            if (!j.is_object()) continue;
            auto name = j.find("name");
            auto value = j.find("value");
            if (name != j.end() && name->is_string() &&
                value != j.end() && value->is_string()) {
                out.cvars.emplace_back(name->get<std::string>(),
                                       value->get<std::string>());
            }
        }
    }

    err.clear();
    return true;
}

}  // namespace pt::editor

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Wave 9 scene save/load round-trip (issue: scene serialization).
//
// Exercises pt::editor::SceneToJson / SceneFromJson -- the lossless
// JSON serializer the `scene_save` / `scene_load` console commands sit
// on. The acceptance bar for the feature is "save a scene with several
// prims (varied materials), lights (all 4 types), a camera pose -> load
// into a fresh engine -> identical scene." This test pins exactly that
// at the serialization layer (no GPU, no Engine instance), which is the
// part that has to be byte-exact; the engine glue around it is a thin
// SceneData <-> private-map copy + ApplySceneSnapshot.
//
// Coverage:
//   - camera pose (pos + yaw/pitch in radians via degree round-trip + fov)
//   - analytic prims: sphere + plane, all 4 materials, emission,
//     roughness, ior, orientation quaternion, texture tiles set + unset
//   - analytic lights: point / spot / sphere / quad with their
//     type-specific fields (cos_inner/outer, radius, u_vec/v_half)
//   - SDF clusters: a leaf + a 3-node smooth-union tree (op/shape/
//     children/params/center all survive; AABB recomputed on load)
//   - render cvars: name/value pairs preserved in order
//   - the JSON document carries a format tag + version
//   - a malformed (non-object) document is rejected
//
// Float comparisons are bit-exact (==) on purpose: JSON serializes
// the full float value and nlohmann round-trips it losslessly, so a
// load must reproduce the saved bits. Using a tolerance here would let
// a real precision-loss regression slip through.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/editor/SceneGraph.h"
#include "../src/engine/Engine.h"
#include "../src/renderer/AnalyticBvh.h"

#include <cmath>
#include <string>

using pt::editor::SceneData;
using pt::editor::SceneToJson;
using pt::editor::SceneFromJson;
using AnalyticPrim  = pt::engine::Engine::AnalyticPrim;
using AnalyticLight = pt::engine::Engine::AnalyticLight;

namespace {

// Build a representative scene touching every serialized field.
SceneData MakeRichScene() {
    SceneData s;

    // Camera. Use a non-default yaw/pitch/fov so a "left at default"
    // bug can't pass. pitch within the +/-85 deg clamp.
    s.has_camera   = true;
    s.camera.pos   = glm::vec3(1.25f, 2.5f, -3.75f);
    s.camera.yaw   = 0.6f;     // radians
    s.camera.pitch = -0.3f;    // radians (within clamp)
    s.camera.fov_deg = 55.0f;

    // --- Prims: varied materials -----------------------------------------
    // Lambert sphere with emission + orientation + a texture tile.
    {
        AnalyticPrim p{};
        p.type = AnalyticPrim::Sphere;
        p.material = AnalyticPrim::Lambert;
        p.pos_or_n[0] = -3.5f; p.pos_or_n[1] = 0.5f; p.pos_or_n[2] = 0.0f;
        p.radius_or_d = 0.45f;
        p.albedo[0] = 0.92f; p.albedo[1] = 0.30f; p.albedo[2] = 0.30f;
        p.emission[0] = 1.5f; p.emission[1] = 0.0f; p.emission[2] = 0.25f;
        p.roughness = 0.1f;
        p.ior = 1.33f;
        p.orient[0] = 0.0f; p.orient[1] = 0.3826834f; p.orient[2] = 0.0f; p.orient[3] = 0.9238795f;
        p.albedo_tex = 3u;   // texture tile set -> must survive
        s.prims[10] = p;
    }
    // Metal sphere (roughness matters for metal).
    {
        AnalyticPrim p{};
        p.type = AnalyticPrim::Sphere;
        p.material = AnalyticPrim::Metal;
        p.pos_or_n[0] = -2.3f; p.pos_or_n[1] = 0.5f; p.pos_or_n[2] = 0.0f;
        p.radius_or_d = 0.45f;
        p.albedo[0] = 1.0f; p.albedo[1] = 0.85f; p.albedo[2] = 0.45f;
        p.roughness = 0.05f;
        s.prims[11] = p;
    }
    // Dielectric sphere (ior matters).
    {
        AnalyticPrim p{};
        p.type = AnalyticPrim::Sphere;
        p.material = AnalyticPrim::Dielectric;
        p.pos_or_n[0] = -1.1f; p.pos_or_n[1] = 0.5f; p.pos_or_n[2] = 0.0f;
        p.radius_or_d = 0.45f;
        p.albedo[0] = 1.0f; p.albedo[1] = 1.0f; p.albedo[2] = 1.0f;
        p.ior = 1.5f;
        s.prims[12] = p;
    }
    // Water plane (the 4th material; plane geometry path).
    {
        AnalyticPrim p{};
        p.type = AnalyticPrim::Plane;
        p.material = AnalyticPrim::Water;
        p.pos_or_n[0] = 0.0f; p.pos_or_n[1] = 1.0f; p.pos_or_n[2] = 0.0f;  // normal
        p.radius_or_d = 0.0f;                                              // d
        p.albedo[0] = 0.1f; p.albedo[1] = 0.2f; p.albedo[2] = 0.3f;
        s.prims[1] = p;
    }

    // --- Lights: all 4 types ---------------------------------------------
    {
        AnalyticLight L{};
        L.type = AnalyticLight::Point;
        L.pos[0] = 0.0f; L.pos[1] = 4.0f; L.pos[2] = 0.0f;
        L.intensity[0] = 12.0f; L.intensity[1] = 11.0f; L.intensity[2] = 10.0f;
        s.lights[1] = L;
    }
    {
        AnalyticLight L{};
        L.type = AnalyticLight::Spot;
        L.pos[0] = 2.0f; L.pos[1] = 5.0f; L.pos[2] = 1.0f;
        L.dir[0] = 0.0f; L.dir[1] = -1.0f; L.dir[2] = 0.0f;
        L.intensity[0] = 50.0f; L.intensity[1] = 50.0f; L.intensity[2] = 40.0f;
        L.cos_outer = std::cos(30.0f * 3.14159265358979323846f / 180.0f);
        L.cos_inner = std::cos(20.0f * 3.14159265358979323846f / 180.0f);
        s.lights[2] = L;
    }
    {
        AnalyticLight L{};
        L.type = AnalyticLight::Sphere;
        L.pos[0] = -2.0f; L.pos[1] = 3.0f; L.pos[2] = -1.0f;
        L.radius = 0.35f;
        L.intensity[0] = 8.0f; L.intensity[1] = 8.0f; L.intensity[2] = 8.0f;
        s.lights[3] = L;
    }
    {
        AnalyticLight L{};
        L.type = AnalyticLight::Quad;
        L.pos[0] = 0.0f; L.pos[1] = 6.0f; L.pos[2] = 0.0f;
        L.dir[0] = 0.0f; L.dir[1] = -1.0f; L.dir[2] = 0.0f;     // normal
        L.u_vec[0] = 1.5f; L.u_vec[1] = 0.0f; L.u_vec[2] = 0.0f; // full half-extent vec
        L.v_half = 0.75f;
        L.intensity[0] = 5.0f; L.intensity[1] = 5.0f; L.intensity[2] = 5.0f;
        s.lights[4] = L;
    }

    // --- SDF clusters -----------------------------------------------------
    // A single-leaf sphere SDF.
    {
        pt::renderer::SdfPrim S{};
        S.material = 1u;  // metal
        S.albedo[0] = 0.3f; S.albedo[1] = 0.65f; S.albedo[2] = 0.95f;
        S.node_count = 1;
        auto& n = S.nodes[0];
        n.op = pt::renderer::SDF_OP_LEAF;
        n.shape = pt::renderer::SDF_SHAPE_SPHERE;
        n.params[0] = 0.4f;
        n.center[0] = 0.4f; n.center[1] = 0.35f; n.center[2] = 0.5f;
        REQUIRE(pt::renderer::ComputeSdfAabb(S));
        s.sdf[100] = S;
    }
    // A 3-node smooth-union of two spheres (the sdf_smin shape).
    {
        pt::renderer::SdfPrim S{};
        S.material = 0u;  // lambert
        S.albedo[0] = 0.95f; S.albedo[1] = 0.55f; S.albedo[2] = 0.2f;
        // node 0: leaf sphere A
        S.nodes[0].op = pt::renderer::SDF_OP_LEAF;
        S.nodes[0].shape = pt::renderer::SDF_SHAPE_SPHERE;
        S.nodes[0].params[0] = 0.4f;
        S.nodes[0].center[0] = 1.65f; S.nodes[0].center[1] = 0.5f; S.nodes[0].center[2] = 0.5f;
        // node 1: leaf sphere B
        S.nodes[1].op = pt::renderer::SDF_OP_LEAF;
        S.nodes[1].shape = pt::renderer::SDF_SHAPE_SPHERE;
        S.nodes[1].params[0] = 0.4f;
        S.nodes[1].center[0] = 2.15f; S.nodes[1].center[1] = 0.5f; S.nodes[1].center[2] = 0.5f;
        // node 2: smooth union of 0 and 1, k = 0.12
        S.nodes[2].op = pt::renderer::SDF_OP_SMOOTH_UNION;
        S.nodes[2].child_a = 0;
        S.nodes[2].child_b = 1;
        S.nodes[2].params[0] = 0.12f;
        S.node_count = 3;
        REQUIRE(pt::renderer::ComputeSdfAabb(S));
        s.sdf[210] = S;
    }

    // --- Render cvars -----------------------------------------------------
    s.cvars.emplace_back("r_sky_mode", "procedural");
    s.cvars.emplace_back("r_sun_elevation", "45");
    s.cvars.emplace_back("r_spp", "1");
    s.cvars.emplace_back("r_env_map", "assets/hdri/sunset.hdr");  // value w/ no space
    s.cvars.emplace_back("r_exposure", "1.5");

    return s;
}

void RequirePrimEqual(const AnalyticPrim& a, const AnalyticPrim& b) {
    REQUIRE(a.type == b.type);
    REQUIRE(a.material == b.material);
    for (int i = 0; i < 3; ++i) REQUIRE(a.pos_or_n[i] == b.pos_or_n[i]);
    REQUIRE(a.radius_or_d == b.radius_or_d);
    for (int i = 0; i < 3; ++i) REQUIRE(a.albedo[i] == b.albedo[i]);
    for (int i = 0; i < 3; ++i) REQUIRE(a.emission[i] == b.emission[i]);
    REQUIRE(a.roughness == b.roughness);
    REQUIRE(a.ior == b.ior);
    for (int i = 0; i < 4; ++i) REQUIRE(a.orient[i] == b.orient[i]);
    REQUIRE(a.albedo_tex == b.albedo_tex);
    REQUIRE(a.normal_tex == b.normal_tex);
    REQUIRE(a.roughness_tex == b.roughness_tex);
    REQUIRE(a.metallic_tex == b.metallic_tex);
}

void RequireLightEqual(const AnalyticLight& a, const AnalyticLight& b) {
    REQUIRE(a.type == b.type);
    for (int i = 0; i < 3; ++i) REQUIRE(a.pos[i] == b.pos[i]);
    REQUIRE(a.radius == b.radius);
    for (int i = 0; i < 3; ++i) REQUIRE(a.intensity[i] == b.intensity[i]);
    for (int i = 0; i < 3; ++i) REQUIRE(a.dir[i] == b.dir[i]);
    REQUIRE(a.cos_outer == b.cos_outer);
    REQUIRE(a.cos_inner == b.cos_inner);
    for (int i = 0; i < 3; ++i) REQUIRE(a.u_vec[i] == b.u_vec[i]);
    REQUIRE(a.v_half == b.v_half);
    for (int i = 0; i < 4; ++i) REQUIRE(a.orient[i] == b.orient[i]);
}

void RequireSdfEqual(const pt::renderer::SdfPrim& a, const pt::renderer::SdfPrim& b) {
    REQUIRE(a.node_count == b.node_count);
    REQUIRE(a.material == b.material);
    for (int i = 0; i < 3; ++i) REQUIRE(a.albedo[i] == b.albedo[i]);
    REQUIRE(a.roughness == b.roughness);
    REQUIRE(a.ior == b.ior);
    for (std::uint32_t i = 0; i < a.node_count; ++i) {
        const auto& na = a.nodes[i];
        const auto& nb = b.nodes[i];
        REQUIRE(na.op == nb.op);
        REQUIRE(na.shape == nb.shape);
        REQUIRE(na.child_a == nb.child_a);
        REQUIRE(na.child_b == nb.child_b);
        for (int k = 0; k < 4; ++k) REQUIRE(na.params[k] == nb.params[k]);
        for (int k = 0; k < 3; ++k) REQUIRE(na.center[k] == nb.center[k]);
    }
    // AABB is recomputed on load from identical nodes -> identical bits.
    for (int i = 0; i < 3; ++i) REQUIRE(a.aabb_min[i] == b.aabb_min[i]);
    for (int i = 0; i < 3; ++i) REQUIRE(a.aabb_max[i] == b.aabb_max[i]);
}

}  // namespace

TEST_CASE("scene round-trips through SceneToJson / SceneFromJson") {
    const SceneData orig = MakeRichScene();

    const nlohmann::json doc = SceneToJson(orig);

    // Document carries the format tag + version.
    REQUIRE(doc.contains("format"));
    REQUIRE(doc["format"] == "demont-scene");
    REQUIRE(doc.contains("version"));
    REQUIRE(doc["version"].is_number_integer());

    // Serialize -> string -> reparse mimics the on-disk path exactly
    // (the console command dump()s to a file then parse()s it back).
    const std::string text = doc.dump(2);
    const nlohmann::json reparsed =
        nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    REQUIRE_FALSE(reparsed.is_discarded());

    SceneData loaded;
    std::string err;
    REQUIRE(SceneFromJson(reparsed, loaded, err));
    REQUIRE(err.empty());

    // --- Camera ----------------------------------------------------------
    REQUIRE(loaded.has_camera);
    REQUIRE(loaded.camera.pos.x == orig.camera.pos.x);
    REQUIRE(loaded.camera.pos.y == orig.camera.pos.y);
    REQUIRE(loaded.camera.pos.z == orig.camera.pos.z);
    REQUIRE(loaded.camera.fov_deg == orig.camera.fov_deg);
    // yaw/pitch persisted as degrees and converted back to radians.
    // The deg<->rad round-trip is not bit-exact, so allow a tight eps.
    REQUIRE(loaded.camera.yaw   == doctest::Approx(orig.camera.yaw).epsilon(1e-5));
    REQUIRE(loaded.camera.pitch == doctest::Approx(orig.camera.pitch).epsilon(1e-5));

    // --- Prims -----------------------------------------------------------
    REQUIRE(loaded.prims.size() == orig.prims.size());
    for (const auto& [id, p] : orig.prims) {
        REQUIRE(loaded.prims.count(id) == 1);
        RequirePrimEqual(p, loaded.prims.at(id));
    }

    // --- Lights ----------------------------------------------------------
    REQUIRE(loaded.lights.size() == orig.lights.size());
    for (const auto& [id, L] : orig.lights) {
        REQUIRE(loaded.lights.count(id) == 1);
        RequireLightEqual(L, loaded.lights.at(id));
    }

    // --- SDF -------------------------------------------------------------
    REQUIRE(loaded.sdf.size() == orig.sdf.size());
    for (const auto& [id, S] : orig.sdf) {
        REQUIRE(loaded.sdf.count(id) == 1);
        RequireSdfEqual(S, loaded.sdf.at(id));
    }

    // --- Cvars (order + values preserved) --------------------------------
    REQUIRE(loaded.cvars.size() == orig.cvars.size());
    for (std::size_t i = 0; i < orig.cvars.size(); ++i) {
        REQUIRE(loaded.cvars[i].first  == orig.cvars[i].first);
        REQUIRE(loaded.cvars[i].second == orig.cvars[i].second);
    }
}

TEST_CASE("empty scene round-trips to an empty scene") {
    SceneData empty;  // no camera, no prims/lights/sdf/cvars
    const nlohmann::json doc = SceneToJson(empty);
    SceneData loaded;
    std::string err;
    REQUIRE(SceneFromJson(doc, loaded, err));
    REQUIRE_FALSE(loaded.has_camera);
    REQUIRE(loaded.prims.empty());
    REQUIRE(loaded.lights.empty());
    REQUIRE(loaded.sdf.empty());
    REQUIRE(loaded.cvars.empty());
}

TEST_CASE("a non-object document is rejected") {
    nlohmann::json arr = nlohmann::json::array();
    arr.push_back(1);
    SceneData loaded;
    std::string err;
    REQUIRE_FALSE(SceneFromJson(arr, loaded, err));
    REQUIRE_FALSE(err.empty());
}

TEST_CASE("texture-free prim round-trips with no-texture sentinels") {
    SceneData s;
    AnalyticPrim p{};  // all four *_tex default to 0xFFFFFFFF
    p.type = AnalyticPrim::Sphere;
    p.radius_or_d = 1.0f;
    s.prims[5] = p;

    SceneData loaded;
    std::string err;
    REQUIRE(SceneFromJson(SceneToJson(s), loaded, err));
    REQUIRE(loaded.prims.count(5) == 1);
    const auto& q = loaded.prims.at(5);
    REQUIRE(q.albedo_tex    == 0xFFFFFFFFu);
    REQUIRE(q.normal_tex    == 0xFFFFFFFFu);
    REQUIRE(q.roughness_tex == 0xFFFFFFFFu);
    REQUIRE(q.metallic_tex  == 0xFFFFFFFFu);
}

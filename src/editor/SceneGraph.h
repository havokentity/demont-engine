// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// Editor scene-graph snapshot API (agent-19 / editor-mode foundation).
//
// SerializeScene walks the engine's owned scene state (analytic
// primitives, analytic lights, SDF clusters, rigid bodies) and returns a
// JSON tree usable by the editor UI agents:
//   * agent-20 -- React web panel reading list_scene over WebSocket
//   * agent-21 -- native overlay panel reading the same call
//   * agent-22 -- 3D viewport gizmos that need positions / radii to
//                 anchor their transform handles
//
// The output format is intentionally flat and forward-compatible: each
// entry is an object with a stable `id` field plus type-specific fields
// matching the host-side struct layout (positions as 3-element arrays,
// material as a lowercase string, ...). Unknown fields on the consumer
// side should be ignored; the engine adds new fields when scene features
// land without bumping a schema version.
//
// This is a READ-ONLY snapshot. Mutations go through console commands
// (`prim_set_pos`, `prim_set_albedo`, ... -- see Engine::RegisterPrimCommands)
// or higher-level commands (`prim_sphere`, `prim_delete`, `prim_duplicate`).
// That keeps the cvar-undo / change-tracking pipeline in the loop -- UI
// agents must NOT write to `primitives_` etc. directly.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../engine/Engine.h"          // AnalyticPrim / AnalyticLight (nested)
#include "../renderer/AnalyticBvh.h"   // pt::renderer::SdfPrim
#include "../renderer/Camera.h"        // pt::renderer::Camera

namespace pt::engine { class Engine; }

namespace pt::editor {

// Build a JSON tree describing the engine's current scene. Safe to call
// from any thread: it only reads `const` accessors on Engine, but the
// caller is responsible for not racing a concurrent mutation (the
// recommendation is to call this from the main thread inside a console
// command handler or the WebSocket dispatch path -- both run under
// Console::Drain on the main thread).
//
// Schema:
//   {
//     "primitives":   [{"id":N,"type":"sphere"|"plane",
//                       "pos":[x,y,z],"radius":r,"material":"lambert"|...,
//                       "albedo":[r,g,b],"emission":[r,g,b],
//                       "roughness":v,"ior":v,
//                       // Wave 8 PBR (#26) texture-map state (wave-9):
//                       "albedo_tex":bool,"normal_tex":bool,
//                       "roughness_tex":bool,"metallic_tex":bool,
//                       "albedo_tex_path":str,"normal_tex_path":str,
//                       "roughness_tex_path":str,"metallic_tex_path":str},
//                      ...],
//     "lights":       [{"id":N,"type":"point"|"spot"|"sphere"|"quad",
//                       "pos":[x,y,z],"intensity":[r,g,b],
//                       "dir":[x,y,z],"cos_outer":v,"cos_inner":v,
//                       "radius":r,"u_vec":[x,y,z],"v_half":v}, ...],
//     "sdf":          [{"id":N,"node_count":k,
//                       "aabb_min":[x,y,z],"aabb_max":[x,y,z],
//                       "material":"lambert"|...,
//                       "albedo":[r,g,b],"roughness":v,"ior":v}, ...],
//     "rigid_bodies": [{"id":N,"prim_id":P,"pos":[x,y,z],
//                       "velocity":[x,y,z],"mass":v,"radius":r,
//                       "is_kinematic":bool}, ...],
//     "selection":    {"kind":"none"|"prim"|"light"|"sdf"|"rb","id":N}
//   }
//
// Empty arrays are emitted explicitly so consumers can iterate without
// nullability checks.
nlohmann::json SerializeScene(const pt::engine::Engine& engine);

// Convert a SelectionKind to / from the lowercase string the WebSocket
// protocol uses. Unknown / missing strings collapse to None.
const char*               SelectionKindToString(int kind);
int                       SelectionKindFromString(std::string_view s);

// --- Wave 9 scene save/load ------------------------------------------------
//
// Round-trippable, file-backed scene description. Where SerializeScene
// above is a one-way, display-oriented snapshot for editor panels
// (e.g. it emits only the SDF AABB header, not the node tree), SceneData
// is the lossless model the on-disk save/load uses: every field needed
// to reconstruct the editable scene byte-for-byte.
//
// Format choice (see PR): JSON via nlohmann_json. The engine already
// vendors nlohmann_json and already serializes the live scene to JSON
// for the editor scene-graph panel (SerializeScene). Reusing JSON --
// and the same flat per-entry schema -- means the save file and the
// editor's live payload share one shape, so there's no second format to
// keep in sync. JSON also round-trips float fields with full precision.
//
// The serializer operates on plain data (the engine's public POD scene
// structs + a cvar key/value list); it does NOT touch GPU state or an
// Engine instance, so it is unit-testable in isolation (see
// tests/scene_serialize_test.cpp). The Engine builds a SceneData from
// its private maps for save, and applies a parsed SceneData back through
// ApplySceneSnapshot + the SDF / camera / cvar paths for load.
struct SceneData {
    // Whether a camera block was present. On save it's always written;
    // on load a missing camera leaves the engine camera untouched.
    bool                                         has_camera = false;
    pt::renderer::Camera                         camera{};

    // Analytic primitives / lights, keyed by user id (same key space the
    // engine maps use). Reuses the engine's nested POD structs directly.
    std::map<std::uint32_t, pt::engine::Engine::AnalyticPrim>  prims;
    std::map<std::uint32_t, pt::engine::Engine::AnalyticLight> lights;

    // SDF clusters, full node tree (lossless). AABB is recomputed on
    // load via pt::renderer::ComputeSdfAabb so a stale stored AABB can't
    // desync the sphere-trace bound.
    std::map<std::uint32_t, pt::renderer::SdfPrim>             sdf;

    // Per-prim PBR texture PATHS (albedo / normal / roughness /
    // metallic; empty string = no map on that channel). The AnalyticPrim
    // structs above carry raw atlas TILE INDICES, which are session-
    // local (tiles are handed out in load order) -- persisting only the
    // indices meant textures silently vanished or swapped after a
    // save / restart / load round-trip. The engine fills these from
    // PbrTilePath() on save and re-resolves them through
    // LoadPbrTextureTile() on load, rewriting the indices.
    std::map<std::uint32_t, std::array<std::string, 4>>        prim_tex_paths;

    // Selected render cvars, name -> value strings (insertion-ordered so
    // the file is stable / diffable). Empty is legal.
    std::vector<std::pair<std::string, std::string>>          cvars;
};

// Serialize a SceneData to the canonical JSON document. Pretty-printed
// (2-space indent) so saved scenes diff cleanly under version control.
nlohmann::json SceneToJson(const SceneData& scene);

// Parse a JSON document (as produced by SceneToJson) back into a
// SceneData. Returns false and fills `err` on a structural error
// (missing/!object root). Unknown fields are ignored; missing optional
// fields fall back to the struct defaults. SDF AABBs are recomputed.
bool SceneFromJson(const nlohmann::json& doc, SceneData& out, std::string& err);

}  // namespace pt::editor

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

}  // namespace pt::editor

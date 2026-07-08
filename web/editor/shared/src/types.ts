// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// TypeScript mirrors of the C++ structs in src/engine/Engine.h. The
// agent-19 contract is `nlohmann::json SerializeScene(const Engine&)`
// in src/editor/SceneGraph.h, which emits per-object records keyed by
// `kind` ("prim", "light", "smoke", "sdf", "csg", ...). This file
// pins the field shapes the React panels read from those records.
//
// IMPORTANT: keep the field layout in lockstep with the C++ side --
// if the engine renames `pos_or_n` -> `position`, both ends must
// change. The contract is documented in web/editor/CONTRIBUTING.md.

// ---------------------------------------------------------------------------
// Material / type tag enums. Numeric values mirror the C++ enum --
// must match `enum Material` and `enum Type` in Engine.h::AnalyticPrim.

export const PrimTypeName = {
  0: 'sphere',
  1: 'plane',
} as const;
export type PrimTypeId = keyof typeof PrimTypeName;

export const MaterialName = {
  0: 'lambert',
  1: 'metal',
  2: 'dielectric',
  3: 'water',
} as const;
export type MaterialId = keyof typeof MaterialName;

export const LightTypeName = {
  0: 'point',
  1: 'spot',
  2: 'sphere',
  3: 'quad',
} as const;
export type LightTypeId = keyof typeof LightTypeName;

// ---------------------------------------------------------------------------
// Per-object records. The agent-19 SerializeScene JSON is expected to
// emit one of these per scene-object entry, with the discriminator on
// the `kind` field.

export interface AnalyticPrim {
  kind: 'prim';
  id: number;
  // SerializeScene emits the lowercase string name ('sphere' | 'plane');
  // the numeric enum id is tolerated for older snapshots.
  type: PrimTypeId | string;
  // Optional legacy alias -- the engine emits `type`, not `type_name`.
  type_name?: string;
  material: MaterialId | string;
  material_name?: string;
  // Sphere: pos = center, radius = sphere radius.
  // Plane: the engine emits `normal` + `d` instead of pos/radius.
  pos?: [number, number, number];
  radius?: number;
  normal?: [number, number, number];
  d?: number;
  albedo: [number, number, number];
  roughness: number;
  ior: number;
  emission: [number, number, number];
  // Orientation quaternion (xyzw); identity for unrotated prims.
  orient?: [number, number, number, number];
}

export interface AnalyticLight {
  kind: 'light';
  id: number;
  // Lowercase string name ('point' | 'spot' | 'sphere' | 'quad') as
  // emitted by SerializeScene; numeric id tolerated for older snapshots.
  type: LightTypeId | string;
  type_name?: string;
  pos: [number, number, number];
  // intensity in W/sr (point/spot) or W/m^2/sr (area)
  intensity: [number, number, number];
  // Orientation quaternion (xyzw); identity for unrotated lights.
  orient?: [number, number, number, number];
  // The per-type extras below are emitted CONDITIONALLY by
  // SerializeScene (src/editor/SceneGraph.cpp) -- absent for other
  // light types, so always guard before dereferencing.
  // sphere only -- emitter radius (m)
  radius?: number;
  // spot + quad only -- spot axis / quad normal (unit vec)
  dir?: [number, number, number];
  // spot only -- outer half-angle cosine
  cos_outer?: number;
  // spot only -- inner half-angle cosine
  cos_inner?: number;
  // quad only -- u half-extent vector (length = u_half)
  u_vec?: [number, number, number];
  // quad only -- v half-extent (m)
  v_half?: number;
}

// Catch-all for the "we haven't typed this yet" kinds the engine may
// emit (smoke emitters, SDF clusters, CSG roots, ...). Panels can
// downcast to the specific kind they care about; an unknown kind is
// rendered as opaque metadata in the inspector.
export interface UnknownObject {
  kind: string;
  id: number;
  [key: string]: unknown;
}

export type SceneObject = AnalyticPrim | AnalyticLight | UnknownObject;

// Whole-scene snapshot as the engine ships it back over WebSocket.
// agent-19's SerializeScene returns an object whose top-level layout
// is documented as a flat object map keyed by kind -> array of
// records. We tolerate either shape by lifting the `objects` array
// out either way in `flattenScene` (see store.ts).
export interface SceneSnapshot {
  // Discriminated by `kind`. The engine may return a flat array
  // OR a kind-keyed map -- the store normalises to a flat array on
  // ingest.
  objects?: SceneObject[];
  prims?: AnalyticPrim[];
  lights?: AnalyticLight[];
  // Free-form extras the engine may include.
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Selection. Mirrors agent-19's
//   Engine::SetSelection(SelectionKind kind, uint32_t id)
// and the `selection_change` event topic. SelectionKind::None is
// represented by `null`.

// Selection kinds the engine recognises. Mirrors SelectionKind in
// src/engine/Engine.h and the strings emitted by
// SelectionKindToString in src/editor/SceneGraph.cpp:
//   'prim'  -- analytic primitive (sphere / plane)
//   'light' -- analytic light (point / spot / sphere / quad)
//   'sdf'   -- SDF cluster
//   'rb'    -- rigid body (physics)
// 'csg' and 'smoke' are reserved for forward compat with kinds the
// engine may emit later (the Engine enum doesn't include them yet,
// but SceneGraph.cpp tolerates unknown top-level array names, so a
// future kind shows up in the hierarchy automatically).
export type SelectionKind = 'prim' | 'light' | 'sdf' | 'rb' | 'csg' | 'smoke';

export interface Selection {
  kind: SelectionKind;
  id: number;
}

// ---------------------------------------------------------------------------
// WebSocket envelope shapes. Inbound = client -> engine. Outbound =
// engine -> client. The existing console protocol (exec, get_cvar,
// set_cvar, list_cvars, list_commands, subscribe / unsubscribe) is
// preserved as-is; editor-specific additions are `list_scene` and
// `select`. agent-19's PR adds the server-side handlers; this client
// just speaks them.

export interface WsInboundExec {
  type: 'exec';
  id?: string;
  line: string;
}
export interface WsInboundGetCVar {
  type: 'get_cvar';
  id?: string;
  name: string;
}
export interface WsInboundSetCVar {
  type: 'set_cvar';
  id?: string;
  name: string;
  value: string;
}
export interface WsInboundSubscribe {
  type: 'subscribe' | 'unsubscribe';
  id?: string;
  topics: string[];
}
export interface WsInboundListScene {
  type: 'list_scene';
  id?: string;
}
export interface WsInboundSelect {
  type: 'select';
  id?: string;
  kind: SelectionKind | 'none';
  // Required for non-none selections; omitted for none.
  obj_id?: number;
}

export type WsInbound =
  | WsInboundExec
  | WsInboundGetCVar
  | WsInboundSetCVar
  | WsInboundSubscribe
  | WsInboundListScene
  | WsInboundSelect;

export type CVarValue = {
  name: string;
  value: string;
  default: string;
  description: string;
  flags: number;
  allowed_values?: string[];
  slider_min?: number;
  slider_max?: number;
  slider_step?: number;
};

export interface WsResult {
  type: 'result';
  id?: string;
  ok: boolean;
  error?: string;
  output?: string;
  // result-specific payloads
  cvar?: CVarValue;
  cvars?: CVarValue[];
  commands?: Array<{ name: string; description: string; default_args?: string }>;
  scene?: SceneSnapshot;
  // Bag for fields server adds we don't know about yet.
  [key: string]: unknown;
}

export interface WsEvent {
  type: 'event';
  topic: string;  // "log" | "selection_change" | "scene_dirty" | ...
  ts: number;
  data: unknown;
}

export type WsOutbound = WsResult | WsEvent;

export type CommandReply = WsResult;

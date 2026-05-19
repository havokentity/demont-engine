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
  type: PrimTypeId;
  type_name: string;
  material: MaterialId;
  material_name: string;
  // For Sphere: pos = sphere center; radius = sphere radius.
  // For Plane: pos = plane normal; radius = plane d (distance from origin).
  pos: [number, number, number];
  radius: number;
  albedo: [number, number, number];
  roughness: number;
  ior: number;
  emission: [number, number, number];
}

export interface AnalyticLight {
  kind: 'light';
  id: number;
  type: LightTypeId;
  type_name: string;
  pos: [number, number, number];
  // sphere only -- 0 for point/spot/quad
  radius: number;
  // intensity in W/sr (point/spot) or W/m^2/sr (area)
  intensity: [number, number, number];
  // spot axis / quad normal (unit vec); zeros for point lights
  dir: [number, number, number];
  // spot only -- outer half-angle cosine
  cos_outer: number;
  // spot only -- inner half-angle cosine
  cos_inner: number;
  // quad only -- u half-extent vector (length = u_half)
  u_vec: [number, number, number];
  // quad only -- v half-extent (m)
  v_half: number;
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

export type SelectionKind = 'prim' | 'light' | 'csg' | 'sdf' | 'smoke';

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

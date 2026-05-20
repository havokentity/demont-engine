// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure helpers for the Lights panel. Extracted so the photometric
// conversions, the cone-angle math, and the raw-snapshot -> typed
// LightRecord parse can be unit-tested without a DOM / WebSocket shim.
//
// The panel reads the engine's SerializeScene `lights[]` array directly
// (NOT the store's flattened `objects[]`, whose `kind` tag is derived
// from the top-level array name and so reads "lights" rather than the
// "light" SelectionKind string). The inspector panel uses the same
// raw-snapshot approach; we mirror it for resilience.

export type LightType = 'point' | 'spot' | 'sphere' | 'quad';

export interface LightRecord {
  id: number;
  type: LightType;
  pos: [number, number, number];
  intensity: [number, number, number];
  dir?: [number, number, number];
  cos_outer?: number;
  cos_inner?: number;
  radius?: number;
  u_vec?: [number, number, number];
  v_half?: number;
  orient?: [number, number, number, number];
}

// ---------------------------------------------------------------------------
// Photometric unit conversions. Mirror the constants in Engine.cpp's
// RegisterLightCommands anonymous namespace (the `_cd` / `_lm` / `_nits`
// / `_exposed` authoring variants). The engine stores raw radiometric
// intensity (W/sr for point/spot; W/m^2/sr for area lights); these let
// the panel author in candela / lumen / nit / EV and convert to the raw
// value that `light_set_intensity` writes.
//
// 683.002 lm/W is the luminous efficacy of monochromatic 555nm light
// (the SI candela definition). Single-wavelength approximation, same
// shortcut Arnold / RenderMan / Octane take.

export const kLuminousEfficacy555nm = 683.002;
export const kWsrPerCandela = 1.0 / kLuminousEfficacy555nm; // W/sr per cd
// W/sr per lumen for OMNIDIRECTIONAL emission: lm / (4*pi*sr) -> W/sr.
export const kWsrPerLumenOmni = 1.0 / (kLuminousEfficacy555nm * 4.0 * Math.PI);
export const kWm2srPerNit = 1.0 / kLuminousEfficacy555nm; // W/m^2/sr per nit

// The intensity-authoring "unit" the panel exposes per light type.
//   raw     -- engine-internal W/sr (point/spot) or W/m^2/sr (area)
//   cd      -- candela (point/spot), luminous intensity
//   lm      -- total lumens for an omnidirectional emitter (point/spot)
//   nits    -- cd/m^2 luminance (sphere/quad area lights)
// Every unit maps an authored scalar to a radiometric multiplier so the
// final per-channel value is color * scalar * multiplier.
export type IntensityUnit = 'raw' | 'cd' | 'lm' | 'nits';

// Units valid for a given light type. Point/spot are luminous-intensity
// emitters (cd / lm); sphere/quad are area emitters (nits). `raw` is
// always available as the engine-internal escape hatch.
export function unitsForType(type: LightType): IntensityUnit[] {
  if (type === 'point' || type === 'spot') return ['raw', 'cd', 'lm'];
  return ['raw', 'nits'];
}

// Multiplier from an authored unit to the engine-internal radiometric
// value (per unit scalar). Multiply the authored scalar by this to get
// the W/sr or W/m^2/sr value. EV is applied separately (2^ev) so the
// caller composes: raw = authored * unitMultiplier(unit) * 2^ev.
export function unitMultiplier(unit: IntensityUnit): number {
  switch (unit) {
    case 'cd':   return kWsrPerCandela;
    case 'lm':   return kWsrPerLumenOmni;
    case 'nits': return kWm2srPerNit;
    case 'raw':
    default:     return 1.0;
  }
}

// 2^ev exposure-stop multiplier. ev=0 -> 1.
export function exposure2x(ev: number): number {
  if (!Number.isFinite(ev)) return 1.0;
  return Math.pow(2, ev);
}

// ---------------------------------------------------------------------------
// Cone half-angle <-> cosine. The engine stores cos(half_angle) so the
// shader can compare cos(theta) without inverse trig; the editor exposes
// the human-friendly degrees.

export function clampCos(c: number): number {
  if (!Number.isFinite(c)) return 1;
  if (c > 1) return 1;
  if (c < 0) return 0;
  return c;
}

export function cosToDeg(c: number): number {
  return (Math.acos(clampCos(c)) * 180) / Math.PI;
}

export function vecLen(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// ---------------------------------------------------------------------------
// Raw-snapshot parsing. The engine's SerializeScene emits a top-level
// `lights[]` array of records WITHOUT a `kind` field; each record has
// id / type / pos / intensity / orient and type-specific extras
// (dir / cos_* for spot, radius for sphere, dir / u_vec / v_half for
// quad). We coerce defensively -- a malformed entry is dropped rather
// than crashing the panel.

function asVec3(v: unknown, fallback: [number, number, number]): [number, number, number] {
  if (Array.isArray(v) && v.length >= 3) {
    const a = Number(v[0]);
    const b = Number(v[1]);
    const c = Number(v[2]);
    if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c)) {
      return [a, b, c];
    }
  }
  return fallback;
}

function asVec4(
  v: unknown,
  fallback: [number, number, number, number],
): [number, number, number, number] {
  if (Array.isArray(v) && v.length >= 4) {
    const a = Number(v[0]);
    const b = Number(v[1]);
    const c = Number(v[2]);
    const d = Number(v[3]);
    if ([a, b, c, d].every(Number.isFinite)) return [a, b, c, d];
  }
  return fallback;
}

function asLightType(v: unknown): LightType | null {
  if (v === 'point' || v === 'spot' || v === 'sphere' || v === 'quad') return v;
  // Tolerate the numeric enum form (0..3) the engine could emit.
  if (v === 0) return 'point';
  if (v === 1) return 'spot';
  if (v === 2) return 'sphere';
  if (v === 3) return 'quad';
  return null;
}

// Parse one raw record into a typed LightRecord, or null if it's not a
// usable light entry.
export function parseLightRecord(raw: unknown): LightRecord | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  const id = Number(o.id);
  if (!Number.isFinite(id)) return null;
  const type = asLightType(o.type);
  if (!type) return null;

  const rec: LightRecord = {
    id,
    type,
    pos: asVec3(o.pos, [0, 0, 0]),
    intensity: asVec3(o.intensity, [1, 1, 1]),
    orient: asVec4(o.orient, [0, 0, 0, 1]),
  };
  if (type === 'spot') {
    rec.dir = asVec3(o.dir, [0, -1, 0]);
    rec.cos_outer = Number.isFinite(Number(o.cos_outer)) ? Number(o.cos_outer) : 1;
    rec.cos_inner = Number.isFinite(Number(o.cos_inner)) ? Number(o.cos_inner) : 1;
  } else if (type === 'sphere') {
    rec.radius = Number.isFinite(Number(o.radius)) ? Number(o.radius) : 0.25;
  } else if (type === 'quad') {
    rec.dir = asVec3(o.dir, [0, -1, 0]);
    rec.u_vec = asVec3(o.u_vec, [1, 0, 0]);
    rec.v_half = Number.isFinite(Number(o.v_half)) ? Number(o.v_half) : 0.5;
  }
  return rec;
}

// Pull the typed light list out of a raw scene snapshot. Returns a
// stable, id-sorted array so rows don't shuffle on every scene_dirty.
export function parseLights(snap: unknown): LightRecord[] {
  if (!snap || typeof snap !== 'object') return [];
  const arr = (snap as Record<string, unknown>).lights;
  if (!Array.isArray(arr)) return [];
  const out: LightRecord[] = [];
  for (const entry of arr) {
    const rec = parseLightRecord(entry);
    if (rec) out.push(rec);
  }
  out.sort((a, b) => a.id - b.id);
  return out;
}

// Pick the next free integer id given the existing lights. Lights are
// keyed by uint id in the engine; a fresh add reuses the lowest gap so
// ids stay compact (and predictable in the `light_*` console echo).
export function nextLightId(lights: { id: number }[]): number {
  const used = new Set(lights.map((l) => l.id));
  let id = 1;
  while (used.has(id)) id += 1;
  return id;
}

// Format a float for console-command emission: trims trailing zeros but
// keeps enough precision that a sub-millimetre drag round-trips.
export function fmt(n: number, prec = 4): string {
  if (!Number.isFinite(n)) return '0';
  return parseFloat(n.toFixed(prec)).toString();
}

// A short human label for a light type (panel headers / badges).
export function lightTypeLabel(type: LightType): string {
  switch (type) {
    case 'point':  return 'Point Light';
    case 'spot':   return 'Spot Light';
    case 'sphere': return 'Sphere Light';
    case 'quad':   return 'Quad Light';
  }
}

// A glyph for the light type (list rows). Mirrors the scene-hierarchy
// icons so the two panels read consistently.
export function lightTypeIcon(type: LightType): string {
  switch (type) {
    case 'point':  return '✴️'; // heavy asterisk
    case 'spot':   return '\u{1F526}';    // flashlight
    case 'sphere': return '\u{1F505}';    // dim button / glow
    case 'quad':   return '▢';       // white square w/ rounded corners
  }
}

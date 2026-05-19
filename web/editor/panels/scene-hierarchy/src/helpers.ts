// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure helpers used by the Scene Hierarchy panel. Extracted to a
// separate module so vitest unit tests don't need a DOM / WebSocket
// shim -- the React component still imports them via the same path,
// it just exports a thinner surface.

import type {
  SceneObject,
  AnalyticPrim,
  AnalyticLight,
} from '@demont/editor-shared';

// Auto-generate a stable display name from a scene object's kind + id.
// e.g. "sphere_100001", "point_light_1", "sdf_3", "rb_7". Falls back
// to "<kind>_<id>" if the kind-specific subtype isn't a string.
export function autoName(o: SceneObject): string {
  if (o.kind === 'prim') {
    const p = o as AnalyticPrim;
    const ty = typeof p.type === 'string'
      ? p.type
      : (p.type_name ?? 'prim');
    return `${ty}_${o.id}`;
  }
  if (o.kind === 'light') {
    const l = o as AnalyticLight;
    const ty = typeof l.type === 'string'
      ? l.type
      : (l.type_name ?? 'light');
    return `${ty}_light_${o.id}`;
  }
  if (o.kind === 'sdf') {
    return `sdf_${o.id}`;
  }
  if (o.kind === 'rb') {
    const shape = (o as Record<string, unknown>).shape;
    return typeof shape === 'string'
      ? `${shape}_rb_${o.id}`
      : `rb_${o.id}`;
  }
  return `${o.kind}_${o.id}`;
}

// Map a scene object to an icon. Per-type, not per-kind, so spheres
// and planes get different glyphs even though they share kind='prim'.
export function rowIcon(o: SceneObject): string {
  if (o.kind === 'prim') {
    const ty = (o as AnalyticPrim).type as unknown;
    if (ty === 'sphere' || ty === 0) return '\u{25EF}';
    if (ty === 'plane'  || ty === 1) return '\u{25AD}';
    return '\u{2728}';
  }
  if (o.kind === 'light') {
    const ty = (o as AnalyticLight).type as unknown;
    if (ty === 'point'  || ty === 0) return '\u{2734}\u{FE0F}';
    if (ty === 'spot'   || ty === 1) return '\u{1F526}';
    if (ty === 'sphere' || ty === 2) return '\u{1F505}';
    if (ty === 'quad'   || ty === 3) return '\u{25A2}';
    return '\u{1F4A1}';
  }
  if (o.kind === 'sdf') return '\u{1F32B}\u{FE0F}';
  if (o.kind === 'rb')  return '\u{26AB}';
  return '\u{2753}';
}

// Format an RGB triple in [0,1] (the engine's albedo / emission shape)
// as a CSS color. Anything beyond 1.0 is clamped for swatch display
// only; the actual scene value is unchanged. Returns null if the
// input isn't an array of at least 3 numbers.
export function rgbCss(rgb: unknown): string | null {
  if (!Array.isArray(rgb) || rgb.length < 3) return null;
  const clamp = (v: unknown): number => {
    const n = typeof v === 'number' ? v : Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(255, Math.round(n * 255)));
  };
  return `rgb(${clamp(rgb[0])}, ${clamp(rgb[1])}, ${clamp(rgb[2])})`;
}

// True iff any channel of `emission` is non-zero. Used to glow the
// swatch on emissive primitives.
export function isEmissive(o: SceneObject): boolean {
  const e = (o as Record<string, unknown>).emission;
  if (!Array.isArray(e)) return false;
  for (const v of e) {
    if (typeof v === 'number' && v > 0) return true;
  }
  return false;
}

// Substring filter, case-insensitive. Returns the substring's [start,end]
// range so we can highlight the match in the rendered name.
export function matchRange(name: string, q: string): [number, number] | null {
  if (!q) return null;
  const i = name.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return null;
  return [i, i + q.length];
}

// Sections the panel renders. Mirrors the kind discriminator on the
// engine's SerializeScene JSON (src/editor/SceneGraph.cpp) -- the
// other kinds the store may carry ('csg', 'smoke') are intentionally
// not rendered as their own sections in v1.
export type SectionKey = 'prim' | 'light' | 'sdf' | 'rb';

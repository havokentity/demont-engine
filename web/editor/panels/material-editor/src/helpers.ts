// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic helpers for the Material Editor panel. Extracted so the
// command-line builders + emission/intensity math can be unit-tested
// without a DOM or a live WebSocket (vitest node mode -- `npm test`).

/** Texture-map slots a primitive carries. Mirrors the `which` arg of
 *  the engine's set_prim_tex helper (0 albedo / 1 normal / 2 roughness
 *  / 3 metallic) and the prim_set_*_tex console commands. */
export type TexSlot = 'albedo' | 'normal' | 'roughness' | 'metallic';

/** Material BRDF models the engine supports (enum Material in
 *  Engine.h::AnalyticPrim, lowercase strings from SceneGraph.cpp). */
export type MaterialType = 'lambert' | 'metal' | 'dielectric' | 'water';

export const MATERIAL_TYPES: MaterialType[] = [
  'lambert',
  'metal',
  'dielectric',
  'water',
];

/** The prim_set_*_tex console command name for a texture slot. */
export function setTexCommand(slot: TexSlot): string {
  switch (slot) {
    case 'albedo':    return 'prim_set_albedo_tex';
    case 'normal':    return 'prim_set_normal_tex';
    case 'roughness': return 'prim_set_roughness_tex';
    case 'metallic':  return 'prim_set_metallic_tex';
  }
}

/** Format a float for a console command line. Trims trailing zeros but
 *  keeps the value lossless to 4 decimals (matching the inspector's
 *  prim_set_* dispatch precision). Non-finite -> "0". */
export function fmt(n: number, prec = 4): string {
  if (!Number.isFinite(n)) return '0';
  return parseFloat(n.toFixed(prec)).toString();
}

/** Last path component (filename) for display. Falls back to the whole
 *  string if there's no slash. Empty in -> empty out. */
export function basename(path: string): string {
  if (!path) return '';
  const parts = path.split('/');
  return parts[parts.length - 1] || path;
}

/** Decompose an emission RGB triple into a normalised base color plus a
 *  scalar intensity multiplier. The intensity is the largest channel
 *  (so the brightest channel maps to 1.0 in the color picker); the base
 *  color is the triple divided by that intensity. An all-zero emission
 *  returns intensity 0 and a neutral white base so the UI shows a
 *  sensible default color the user can dial up. */
export function decomposeEmission(
  rgb: readonly [number, number, number],
): { color: [number, number, number]; intensity: number } {
  const r = Number.isFinite(rgb[0]) ? Math.max(0, rgb[0]) : 0;
  const g = Number.isFinite(rgb[1]) ? Math.max(0, rgb[1]) : 0;
  const b = Number.isFinite(rgb[2]) ? Math.max(0, rgb[2]) : 0;
  const intensity = Math.max(r, g, b);
  if (intensity <= 0) {
    return { color: [1, 1, 1], intensity: 0 };
  }
  return { color: [r / intensity, g / intensity, b / intensity], intensity };
}

/** Recompose an emission RGB triple from a base color + scalar
 *  intensity. Inverse of decomposeEmission (modulo the all-zero base
 *  default). */
export function composeEmission(
  color: readonly [number, number, number],
  intensity: number,
): [number, number, number] {
  const k = Number.isFinite(intensity) && intensity > 0 ? intensity : 0;
  return [
    Math.max(0, color[0]) * k,
    Math.max(0, color[1]) * k,
    Math.max(0, color[2]) * k,
  ];
}

/** True when an emission triple is non-zero (any positive channel). */
export function isEmissive(
  rgb: readonly [number, number, number] | undefined,
): boolean {
  if (!rgb) return false;
  return (
    (Number.isFinite(rgb[0]) && rgb[0] > 0) ||
    (Number.isFinite(rgb[1]) && rgb[1] > 0) ||
    (Number.isFinite(rgb[2]) && rgb[2] > 0)
  );
}

// sRGB transfer (IEC 61966-2-1) for the live preview swatch. The engine
// stores albedo / emission linear; the swatch renders in display sRGB.
export function linearToSrgb(c: number): number {
  if (!Number.isFinite(c) || c <= 0) return 0;
  if (c >= 1) return 1;
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Format a linear-RGB triple as a CSS rgb() string in display sRGB,
 *  clamped to [0,255]. Returns null for non-triples (defensive against
 *  malformed snapshots). Used by the preview swatch. */
export function rgbCss(
  rgb: unknown,
): string | null {
  if (!Array.isArray(rgb) || rgb.length < 3) return null;
  const ch = (v: unknown): number => {
    const n = typeof v === 'number' && Number.isFinite(v) ? v : 0;
    const s = linearToSrgb(n);
    const i = Math.round(s * 255);
    return i < 0 ? 0 : i > 255 ? 255 : i;
  };
  return `rgb(${ch(rgb[0])}, ${ch(rgb[1])}, ${ch(rgb[2])})`;
}

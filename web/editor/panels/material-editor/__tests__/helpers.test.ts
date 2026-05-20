// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Material Editor helpers. No DOM, no
// WebSocket -- vitest node mode. Run with `npm test` from web/editor/.

import { describe, expect, it } from 'vitest';

import {
  setTexCommand,
  fmt,
  basename,
  decomposeEmission,
  composeEmission,
  isEmissive,
  rgbCss,
  MATERIAL_TYPES,
  anisotropyCommand,
  clearcoatCommand,
  subsurfaceCommand,
  ADV_DEFAULTS,
} from '../src/helpers';

describe('setTexCommand', () => {
  it('maps each slot to its prim_set_*_tex command', () => {
    expect(setTexCommand('albedo')).toBe('prim_set_albedo_tex');
    expect(setTexCommand('normal')).toBe('prim_set_normal_tex');
    expect(setTexCommand('roughness')).toBe('prim_set_roughness_tex');
    expect(setTexCommand('metallic')).toBe('prim_set_metallic_tex');
  });
});

describe('MATERIAL_TYPES', () => {
  it('lists the four engine BRDF models in enum order', () => {
    // Order must mirror enum Material in Engine.h (lambert=0 .. water=3)
    // so a future numeric-index consumer stays correct.
    expect(MATERIAL_TYPES).toEqual([
      'lambert',
      'metal',
      'dielectric',
      'water',
    ]);
  });
});

describe('fmt', () => {
  it('trims trailing zeros', () => {
    expect(fmt(1.0)).toBe('1');
    expect(fmt(0.5)).toBe('0.5');
    expect(fmt(0.123456)).toBe('0.1235'); // 4-decimal default
  });
  it('returns 0 for non-finite', () => {
    expect(fmt(NaN)).toBe('0');
    expect(fmt(Infinity)).toBe('0');
  });
  it('honours an explicit precision', () => {
    expect(fmt(0.123456, 2)).toBe('0.12');
  });
});

describe('basename', () => {
  it('returns the last path component', () => {
    expect(basename('assets/textures/checker_albedo.png')).toBe(
      'checker_albedo.png',
    );
  });
  it('returns the whole string when there is no slash', () => {
    expect(basename('foo.png')).toBe('foo.png');
  });
  it('returns empty for empty input', () => {
    expect(basename('')).toBe('');
  });
});

describe('decomposeEmission', () => {
  it('returns intensity 0 + white base for an unlit material', () => {
    const { color, intensity } = decomposeEmission([0, 0, 0]);
    expect(intensity).toBe(0);
    expect(color).toEqual([1, 1, 1]);
  });
  it('normalises by the brightest channel', () => {
    const { color, intensity } = decomposeEmission([2, 1, 0]);
    expect(intensity).toBe(2);
    expect(color).toEqual([1, 0.5, 0]);
  });
  it('handles a neutral white emitter', () => {
    const { color, intensity } = decomposeEmission([4, 4, 4]);
    expect(intensity).toBe(4);
    expect(color).toEqual([1, 1, 1]);
  });
  it('clamps negatives to zero', () => {
    const { color, intensity } = decomposeEmission([-1, 0, 0]);
    expect(intensity).toBe(0);
    expect(color).toEqual([1, 1, 1]);
  });
});

describe('composeEmission', () => {
  it('is the inverse of decompose for a lit material', () => {
    const src: [number, number, number] = [2, 1, 0];
    const { color, intensity } = decomposeEmission(src);
    expect(composeEmission(color, intensity)).toEqual(src);
  });
  it('returns black when intensity is zero', () => {
    expect(composeEmission([1, 0.5, 0.2], 0)).toEqual([0, 0, 0]);
  });
  it('treats non-finite intensity as zero', () => {
    expect(composeEmission([1, 1, 1], NaN)).toEqual([0, 0, 0]);
  });
  it('scales the base color by the multiplier', () => {
    expect(composeEmission([1, 0.5, 0], 3)).toEqual([3, 1.5, 0]);
  });
});

describe('isEmissive', () => {
  it('true when any channel is positive', () => {
    expect(isEmissive([0, 0, 0.1])).toBe(true);
  });
  it('false when all zero', () => {
    expect(isEmissive([0, 0, 0])).toBe(false);
  });
  it('false when undefined', () => {
    expect(isEmissive(undefined)).toBe(false);
  });
});

describe('advanced material command builders', () => {
  it('anisotropyCommand emits id + amount + rotation', () => {
    expect(anisotropyCommand(3, 0.5, 45)).toBe('prim_set_anisotropy 3 0.5 45');
    // Trims trailing zeros via fmt.
    expect(anisotropyCommand(0, -1.0, 0)).toBe('prim_set_anisotropy 0 -1 0');
  });
  it('clearcoatCommand emits id + weight + roughness', () => {
    expect(clearcoatCommand(2, 1, 0.03)).toBe('prim_set_clearcoat 2 1 0.03');
    expect(clearcoatCommand(7, 0, 0.5)).toBe('prim_set_clearcoat 7 0 0.5');
  });
  it('subsurfaceCommand emits id + radius + rgb', () => {
    expect(subsurfaceCommand(1, 0.002, [0.9, 0.5, 0.4])).toBe(
      'prim_set_subsurface 1 0.002 0.9 0.5 0.4',
    );
    expect(subsurfaceCommand(4, 0, [1, 1, 1])).toBe(
      'prim_set_subsurface 4 0 1 1 1',
    );
  });
  it('ADV_DEFAULTS match the engine prim_set_* defaults', () => {
    expect(ADV_DEFAULTS.anisoAmount).toBe(0);
    expect(ADV_DEFAULTS.clearcoatWeight).toBe(0);
    expect(ADV_DEFAULTS.clearcoatRoughness).toBeCloseTo(0.03, 6);
    expect(ADV_DEFAULTS.subsurfaceRadius).toBe(0);
    expect(ADV_DEFAULTS.subsurfaceColor).toEqual([1, 1, 1]);
  });
});

describe('rgbCss', () => {
  it('formats a linear triple in display sRGB', () => {
    // Linear 1.0 -> sRGB 1.0 -> 255.
    expect(rgbCss([1, 0, 0])).toBe('rgb(255, 0, 0)');
  });
  it('clamps super-unity (HDR emission) to 255', () => {
    expect(rgbCss([5, 5, 5])).toBe('rgb(255, 255, 255)');
  });
  it('clamps negatives to 0', () => {
    expect(rgbCss([-1, -0.5, 0])).toBe('rgb(0, 0, 0)');
  });
  it('returns null for non-triples', () => {
    expect(rgbCss(undefined)).toBe(null);
    expect(rgbCss('red')).toBe(null);
    expect(rgbCss([1, 2])).toBe(null);
  });
  it('treats non-number entries as 0', () => {
    expect(rgbCss([NaN, 1, 0])).toBe('rgb(0, 255, 0)');
  });
  it('applies the sRGB curve to mid-grey linear', () => {
    // Linear 0.5 -> sRGB ~0.735 -> ~188.
    expect(rgbCss([0.5, 0.5, 0.5])).toBe('rgb(188, 188, 188)');
  });
});

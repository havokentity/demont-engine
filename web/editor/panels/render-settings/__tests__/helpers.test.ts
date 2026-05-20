// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Render Settings helpers + descriptor table.
// No DOM, no WebSocket -- vitest node mode. Run with `npm test` from
// web/editor/.

import { describe, expect, it } from 'vitest';

import {
  SECTIONS,
  allCvarNames,
  fmt,
  parseCvarFloat,
  parseCvarBool,
  clamp,
} from '../src/helpers';

describe('fmt', () => {
  it('trims trailing zeros', () => {
    expect(fmt(1.0)).toBe('1');
    expect(fmt(0.5)).toBe('0.5');
    expect(fmt(0.123456)).toBe('0.12346'); // 5-decimal default
  });
  it('returns 0 for non-finite', () => {
    expect(fmt(NaN)).toBe('0');
    expect(fmt(Infinity)).toBe('0');
  });
  it('honours an explicit precision', () => {
    expect(fmt(0.123456, 2)).toBe('0.12');
  });
});

describe('parseCvarFloat', () => {
  it('parses a numeric string', () => {
    expect(parseCvarFloat('0.02')).toBeCloseTo(0.02, 6);
    expect(parseCvarFloat('-0.95')).toBeCloseTo(-0.95, 6);
    expect(parseCvarFloat('96')).toBe(96);
  });
  it('falls back to the default for missing / non-numeric input', () => {
    expect(parseCvarFloat(undefined, 3)).toBe(3);
    expect(parseCvarFloat('', 5)).toBe(5);
    expect(parseCvarFloat('abc', 7)).toBe(7);
  });
  it('defaults the fallback to 0', () => {
    expect(parseCvarFloat(undefined)).toBe(0);
  });
});

describe('parseCvarBool', () => {
  it('treats "1" / non-zero as true', () => {
    expect(parseCvarBool('1')).toBe(true);
    expect(parseCvarBool('2')).toBe(true);
    expect(parseCvarBool('true')).toBe(true);
    expect(parseCvarBool('on')).toBe(true);
  });
  it('treats "0" / empty / falsey as false', () => {
    expect(parseCvarBool('0')).toBe(false);
    expect(parseCvarBool('')).toBe(false);
    expect(parseCvarBool('false')).toBe(false);
    expect(parseCvarBool('off')).toBe(false);
    expect(parseCvarBool(undefined)).toBe(false);
  });
});

describe('clamp', () => {
  it('pins into the given range', () => {
    expect(clamp(5, 0, 1)).toBe(1);
    expect(clamp(-3, 0, 1)).toBe(0);
    expect(clamp(0.5, 0, 1)).toBe(0.5);
  });
  it('treats missing bounds as open', () => {
    expect(clamp(5, undefined, 10)).toBe(5);
    expect(clamp(-5, -10, undefined)).toBe(-5);
    expect(clamp(42)).toBe(42);
  });
});

describe('SECTIONS descriptor table', () => {
  it('covers the nine documented feature groups', () => {
    expect(SECTIONS.map((s) => s.id)).toEqual([
      'tonemap',
      'sky',
      'clouds',
      'fog',
      'godrays',
      'ocean',
      'camera',
      'spectral',
      'smoke',
    ]);
  });

  it('every control has a non-empty engine cvar name + label', () => {
    for (const s of SECTIONS) {
      expect(s.controls.length).toBeGreaterThan(0);
      for (const c of s.controls) {
        expect(c.name).toMatch(/^r_/);
        expect(c.label.length).toBeGreaterThan(0);
      }
    }
  });

  it('select controls carry their allowed-value options', () => {
    const selects = SECTIONS.flatMap((s) =>
      s.controls.filter((c) => c.kind === 'select'),
    );
    for (const c of selects) {
      expect(Array.isArray(c.options)).toBe(true);
      expect(c.options!.length).toBeGreaterThan(1);
    }
  });

  it('slider/int controls carry a sane range (min < max)', () => {
    const numeric = SECTIONS.flatMap((s) =>
      s.controls.filter((c) => c.kind === 'slider' || c.kind === 'int'),
    );
    for (const c of numeric) {
      expect(typeof c.min).toBe('number');
      expect(typeof c.max).toBe('number');
      expect(c.min!).toBeLessThan(c.max!);
    }
  });

  it('gate cvars are themselves a toggle in their section', () => {
    for (const s of SECTIONS) {
      if (!s.gate) continue;
      const gate = s.controls.find((c) => c.name === s.gate);
      expect(gate, `section ${s.id} gate ${s.gate} present`).toBeDefined();
      expect(gate!.kind).toBe('toggle');
    }
  });

  it('binds the headline tone-mapping operators verbatim', () => {
    const tm = SECTIONS.find((s) => s.id === 'tonemap')!.controls[0];
    expect(tm.name).toBe('r_tonemap_op');
    expect(tm.options).toEqual([
      'aces',
      'agx',
      'khronos_pbr_neutral',
      'reinhard',
      'linear',
    ]);
  });
});

describe('allCvarNames', () => {
  it('flattens every bound cvar with no duplicates', () => {
    const names = allCvarNames();
    expect(names.length).toBeGreaterThan(20);
    expect(new Set(names).size).toBe(names.length);
    // Spot-check representatives from each headline feature.
    for (const n of [
      'r_tonemap_op',
      'r_sky_mode',
      'r_clouds_mode',
      'r_fog',
      'r_godrays',
      'r_ocean',
      'r_dof',
      'r_spectral',
      'r_smoke_mode',
    ]) {
      expect(names).toContain(n);
    }
  });
});

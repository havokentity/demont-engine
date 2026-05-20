// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Lights panel helpers: photometric unit
// conversions, cone-angle math, and the raw-snapshot -> LightRecord
// parse. No DOM / WebSocket shim needed.

import { describe, expect, it } from 'vitest';
import {
  parseLights,
  parseLightRecord,
  nextLightId,
  cosToDeg,
  clampCos,
  vecLen,
  unitsForType,
  unitMultiplier,
  exposure2x,
  kWsrPerCandela,
  kWsrPerLumenOmni,
  kWm2srPerNit,
  fmt,
  type LightRecord,
} from '../src/helpers';

describe('photometric conversions', () => {
  it('match the engine constants (1cd = 1/683.002 W/sr)', () => {
    expect(kWsrPerCandela).toBeCloseTo(1 / 683.002, 10);
    expect(kWm2srPerNit).toBeCloseTo(1 / 683.002, 10);
    expect(kWsrPerLumenOmni).toBeCloseTo(1 / (683.002 * 4 * Math.PI), 12);
  });

  it('unitMultiplier(raw) is identity', () => {
    expect(unitMultiplier('raw')).toBe(1);
  });

  it('unitMultiplier maps cd/lm/nits to the right radiometric factor', () => {
    expect(unitMultiplier('cd')).toBeCloseTo(1 / 683.002, 10);
    expect(unitMultiplier('nits')).toBeCloseTo(1 / 683.002, 10);
    expect(unitMultiplier('lm')).toBeCloseTo(1 / (683.002 * 4 * Math.PI), 12);
  });

  it('exposure2x doubles per stop and is 1 at ev=0', () => {
    expect(exposure2x(0)).toBe(1);
    expect(exposure2x(1)).toBeCloseTo(2, 10);
    expect(exposure2x(-1)).toBeCloseTo(0.5, 10);
    expect(exposure2x(3)).toBeCloseTo(8, 10);
  });

  it('exposure2x is 1 for non-finite ev', () => {
    expect(exposure2x(NaN)).toBe(1);
    expect(exposure2x(Infinity)).toBe(1);
  });
});

describe('unitsForType', () => {
  it('offers cd/lm for luminous-intensity emitters', () => {
    expect(unitsForType('point')).toEqual(['raw', 'cd', 'lm']);
    expect(unitsForType('spot')).toEqual(['raw', 'cd', 'lm']);
  });
  it('offers nits for area emitters', () => {
    expect(unitsForType('sphere')).toEqual(['raw', 'nits']);
    expect(unitsForType('quad')).toEqual(['raw', 'nits']);
  });
});

describe('cone-angle math', () => {
  it('clampCos pins to [0,1]', () => {
    expect(clampCos(1.5)).toBe(1);
    expect(clampCos(-0.2)).toBe(0);
    expect(clampCos(0.5)).toBe(0.5);
    expect(clampCos(NaN)).toBe(1);
  });
  it('cosToDeg inverts cosine to degrees', () => {
    expect(cosToDeg(1)).toBeCloseTo(0, 5);
    expect(cosToDeg(0)).toBeCloseTo(90, 5);
    expect(cosToDeg(Math.cos((30 * Math.PI) / 180))).toBeCloseTo(30, 4);
  });
});

describe('vecLen', () => {
  it('computes Euclidean length', () => {
    expect(vecLen([3, 4, 0])).toBeCloseTo(5, 6);
    expect(vecLen([0, 0, 0])).toBe(0);
    expect(vecLen([1, 2, 2])).toBeCloseTo(3, 6);
  });
});

describe('fmt', () => {
  it('trims trailing zeros', () => {
    expect(fmt(1.5)).toBe('1.5');
    expect(fmt(2.0)).toBe('2');
    expect(fmt(0.1 + 0.2)).toBe('0.3');
  });
  it('returns 0 for non-finite', () => {
    expect(fmt(NaN)).toBe('0');
    expect(fmt(Infinity)).toBe('0');
  });
});

describe('parseLightRecord', () => {
  it('parses a point light', () => {
    const r = parseLightRecord({
      id: 1,
      type: 'point',
      pos: [0, 3, 0],
      intensity: [5, 5, 5],
      orient: [0, 0, 0, 1],
    });
    expect(r).not.toBeNull();
    expect(r!.type).toBe('point');
    expect(r!.pos).toEqual([0, 3, 0]);
    expect(r!.intensity).toEqual([5, 5, 5]);
    // Point lights carry no dir / radius / cone fields.
    expect(r!.dir).toBeUndefined();
    expect(r!.radius).toBeUndefined();
  });

  it('parses a spot light with cone + dir', () => {
    const r = parseLightRecord({
      id: 2,
      type: 'spot',
      pos: [0, 4, 0],
      intensity: [8, 8, 8],
      dir: [0, -1, 0],
      cos_outer: 0.5,
      cos_inner: 0.8,
    });
    expect(r!.type).toBe('spot');
    expect(r!.dir).toEqual([0, -1, 0]);
    expect(r!.cos_outer).toBe(0.5);
    expect(r!.cos_inner).toBe(0.8);
  });

  it('parses a sphere light radius', () => {
    const r = parseLightRecord({
      id: 3,
      type: 'sphere',
      pos: [0, 3, 0],
      intensity: [6, 6, 6],
      radius: 0.5,
    });
    expect(r!.type).toBe('sphere');
    expect(r!.radius).toBe(0.5);
  });

  it('parses a quad light u_vec + v_half', () => {
    const r = parseLightRecord({
      id: 4,
      type: 'quad',
      pos: [0, 4, 0],
      intensity: [4, 4, 4],
      dir: [0, -1, 0],
      u_vec: [0.5, 0, 0],
      v_half: 0.5,
    });
    expect(r!.type).toBe('quad');
    expect(r!.u_vec).toEqual([0.5, 0, 0]);
    expect(r!.v_half).toBe(0.5);
  });

  it('tolerates the numeric enum type form', () => {
    expect(parseLightRecord({ id: 9, type: 0 })!.type).toBe('point');
    expect(parseLightRecord({ id: 9, type: 1 })!.type).toBe('spot');
    expect(parseLightRecord({ id: 9, type: 2 })!.type).toBe('sphere');
    expect(parseLightRecord({ id: 9, type: 3 })!.type).toBe('quad');
  });

  it('rejects records with no id or unknown type', () => {
    expect(parseLightRecord({ type: 'point' })).toBeNull();
    expect(parseLightRecord({ id: 1, type: 'laser' })).toBeNull();
    expect(parseLightRecord(null)).toBeNull();
    expect(parseLightRecord(42)).toBeNull();
  });

  it('falls back to defaults for missing vectors', () => {
    const r = parseLightRecord({ id: 1, type: 'point' });
    expect(r!.pos).toEqual([0, 0, 0]);
    expect(r!.intensity).toEqual([1, 1, 1]);
    expect(r!.orient).toEqual([0, 0, 0, 1]);
  });
});

describe('parseLights', () => {
  it('reads the lights[] array and sorts by id', () => {
    const snap = {
      lights: [
        { id: 3, type: 'point', pos: [0, 0, 0], intensity: [1, 1, 1] },
        { id: 1, type: 'sphere', pos: [0, 0, 0], intensity: [1, 1, 1], radius: 1 },
        { id: 2, type: 'spot', pos: [0, 0, 0], intensity: [1, 1, 1] },
      ],
    };
    const out = parseLights(snap);
    expect(out.map((l) => l.id)).toEqual([1, 2, 3]);
  });

  it('drops malformed entries but keeps valid ones', () => {
    const snap = {
      lights: [
        { id: 1, type: 'point' },
        { id: 'oops', type: 'point' },
        { type: 'spot' },
        { id: 2, type: 'quad' },
      ],
    };
    const out = parseLights(snap);
    expect(out.map((l) => l.id)).toEqual([1, 2]);
  });

  it('returns [] when there is no lights array', () => {
    expect(parseLights({})).toEqual([]);
    expect(parseLights({ lights: 'nope' })).toEqual([]);
    expect(parseLights(null)).toEqual([]);
    expect(parseLights(undefined)).toEqual([]);
  });
});

describe('nextLightId', () => {
  it('returns 1 for an empty scene', () => {
    expect(nextLightId([])).toBe(1);
  });
  it('fills the lowest gap', () => {
    const ls: LightRecord[] = [
      { id: 1 } as LightRecord,
      { id: 2 } as LightRecord,
      { id: 4 } as LightRecord,
    ];
    expect(nextLightId(ls)).toBe(3);
  });
  it('appends after a contiguous range', () => {
    const ls: LightRecord[] = [
      { id: 1 } as LightRecord,
      { id: 2 } as LightRecord,
      { id: 3 } as LightRecord,
    ];
    expect(nextLightId(ls)).toBe(4);
  });
});

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Scene Hierarchy helpers. No DOM, no
// WebSocket -- vitest in node mode. Run with `npm test` from
// web/editor/.

import { describe, expect, it } from 'vitest';

import {
  autoName,
  rowIcon,
  rgbCss,
  isEmissive,
  matchRange,
} from '../src/helpers';
import type { SceneObject } from '@demont/editor-shared';

describe('autoName', () => {
  it('builds <type>_<id> for analytic primitives by string type', () => {
    const o: SceneObject = {
      kind: 'prim',
      id: 100001,
      // Engine emits "sphere" / "plane" as strings per SceneGraph.cpp.
      type: 'sphere' as unknown as 0,
      type_name: 'sphere',
      material: 0,
      material_name: 'lambert',
      pos: [0, 0, 0],
      radius: 1,
      albedo: [1, 1, 1],
      roughness: 0,
      ior: 1,
      emission: [0, 0, 0],
    };
    expect(autoName(o)).toBe('sphere_100001');
  });

  it('builds <type>_light_<id> for analytic lights by string type', () => {
    const o: SceneObject = {
      kind: 'light',
      id: 3,
      type: 'point' as unknown as 0,
      type_name: 'point',
      pos: [0, 0, 0],
      radius: 0,
      intensity: [1, 1, 1],
      dir: [0, 0, 0],
      cos_outer: 0,
      cos_inner: 0,
      u_vec: [0, 0, 0],
      v_half: 0,
    };
    expect(autoName(o)).toBe('point_light_3');
  });

  it('falls back to type_name when type is numeric', () => {
    const o: SceneObject = {
      kind: 'prim',
      id: 42,
      type: 0 as unknown as 0,        // SDK / older engine: numeric enum
      type_name: 'sphere',
      material: 0,
      material_name: 'lambert',
      pos: [0, 0, 0],
      radius: 1,
      albedo: [1, 1, 1],
      roughness: 0,
      ior: 1,
      emission: [0, 0, 0],
    };
    // numeric type => skip the string branch => use type_name fallback
    expect(autoName(o)).toBe('sphere_42');
  });

  it('builds sdf_<id> for SDF clusters', () => {
    const o: SceneObject = { kind: 'sdf', id: 7 } as SceneObject;
    expect(autoName(o)).toBe('sdf_7');
  });

  it('prefers shape for rigid bodies when the field is set', () => {
    const o: SceneObject = {
      kind: 'rb',
      id: 1,
      shape: 'sphere',
    } as unknown as SceneObject;
    expect(autoName(o)).toBe('sphere_rb_1');
  });

  it('falls back to rb_<id> when shape is missing', () => {
    const o: SceneObject = { kind: 'rb', id: 2 } as unknown as SceneObject;
    expect(autoName(o)).toBe('rb_2');
  });

  it('uses kind_id for any unknown kind', () => {
    const o: SceneObject = { kind: 'csg', id: 9 } as unknown as SceneObject;
    expect(autoName(o)).toBe('csg_9');
  });
});

describe('rowIcon', () => {
  it('returns sphere glyph for spherical primitives', () => {
    const o: SceneObject = { kind: 'prim', id: 1, type: 'sphere' } as unknown as SceneObject;
    expect(rowIcon(o)).toBe('\u{25EF}');
  });
  it('returns plane glyph for plane primitives', () => {
    const o: SceneObject = { kind: 'prim', id: 1, type: 'plane' } as unknown as SceneObject;
    expect(rowIcon(o)).toBe('\u{25AD}');
  });
  it('returns light-type glyph for spot light', () => {
    const o: SceneObject = { kind: 'light', id: 1, type: 'spot' } as unknown as SceneObject;
    expect(rowIcon(o)).toBe('\u{1F526}');
  });
  it('returns SDF glyph for sdf clusters', () => {
    const o: SceneObject = { kind: 'sdf', id: 1 } as unknown as SceneObject;
    expect(rowIcon(o)).toBe('\u{1F32B}\u{FE0F}');
  });
});

describe('rgbCss', () => {
  it('formats a 3-channel [0,1] triple as a CSS rgb()', () => {
    expect(rgbCss([1, 0, 0])).toBe('rgb(255, 0, 0)');
    expect(rgbCss([0.5, 0.5, 0.5])).toBe('rgb(128, 128, 128)');
  });
  it('clamps super-unity emission values to 255', () => {
    expect(rgbCss([5, 5, 5])).toBe('rgb(255, 255, 255)');
  });
  it('clamps negative values to 0', () => {
    expect(rgbCss([-1, -0.5, 0])).toBe('rgb(0, 0, 0)');
  });
  it('returns null for non-array input', () => {
    expect(rgbCss(undefined)).toBe(null);
    expect(rgbCss('red')).toBe(null);
    expect(rgbCss([1, 2])).toBe(null);  // < 3 channels
  });
  it('coerces stringified numbers to 0 if not finite', () => {
    expect(rgbCss([NaN, 1, 0])).toBe('rgb(0, 255, 0)');
  });
});

describe('isEmissive', () => {
  it('returns true when any channel is positive', () => {
    const o: SceneObject = { kind: 'prim', id: 1, emission: [0, 0, 5] } as unknown as SceneObject;
    expect(isEmissive(o)).toBe(true);
  });
  it('returns false when all channels are zero', () => {
    const o: SceneObject = { kind: 'prim', id: 1, emission: [0, 0, 0] } as unknown as SceneObject;
    expect(isEmissive(o)).toBe(false);
  });
  it('returns false when emission is missing', () => {
    const o: SceneObject = { kind: 'prim', id: 1 } as unknown as SceneObject;
    expect(isEmissive(o)).toBe(false);
  });
  it('treats non-number entries as zero', () => {
    const o: SceneObject = {
      kind: 'prim', id: 1, emission: ['1', 0, 0],
    } as unknown as SceneObject;
    expect(isEmissive(o)).toBe(false);
  });
});

describe('matchRange', () => {
  it('returns null on empty query (no filter active)', () => {
    expect(matchRange('sphere_100001', '')).toBe(null);
  });
  it('returns null when no match', () => {
    expect(matchRange('sphere_100001', 'plane')).toBe(null);
  });
  it('returns inclusive-start / exclusive-end indices on a match', () => {
    expect(matchRange('sphere_100001', '100')).toEqual([7, 10]);
  });
  it('is case-insensitive', () => {
    expect(matchRange('SPHERE_1', 'sphere')).toEqual([0, 6]);
    expect(matchRange('sphere_1', 'SPHERE')).toEqual([0, 6]);
  });
});

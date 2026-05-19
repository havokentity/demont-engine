// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Asset Browser helpers.

import { describe, expect, it } from 'vitest';
import { parseListAssetsOutput } from '../src/helpers';

describe('parseListAssetsOutput', () => {
  it('returns one entry per non-summary line', () => {
    const raw =
      'tests/goldens/scenes/aurora_smoke.cfg\n' +
      'tests/goldens/scenes/cornell_csg.cfg\n' +
      "(list_assets: 2 entries under 'tests/goldens/scenes')\n";
    expect(parseListAssetsOutput(raw)).toEqual([
      'tests/goldens/scenes/aurora_smoke.cfg',
      'tests/goldens/scenes/cornell_csg.cfg',
    ]);
  });

  it('drops blank lines + summary parens', () => {
    const raw = '\nassets/hdri/sunset.hdr\n\n(list_assets: 1)\n';
    expect(parseListAssetsOutput(raw)).toEqual(['assets/hdri/sunset.hdr']);
  });

  it('returns [] for empty input', () => {
    expect(parseListAssetsOutput('')).toEqual([]);
  });

  it('returns [] when output is only the summary', () => {
    expect(parseListAssetsOutput('(list_assets: 0 entries)\n')).toEqual([]);
  });

  it('drops error: lines defensively', () => {
    const raw =
      "error: 'foo' is not a directory\n(list_assets: 0 entries)\n";
    expect(parseListAssetsOutput(raw)).toEqual([]);
  });

  it('tolerates CRLF line endings', () => {
    const raw = 'assets/gltf/Box.glb\r\nassets/gltf/Box.gltf\r\n';
    expect(parseListAssetsOutput(raw)).toEqual([
      'assets/gltf/Box.glb',
      'assets/gltf/Box.gltf',
    ]);
  });
});

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic tests for the Asset Browser helpers.

import { describe, expect, it } from 'vitest';
import {
  mergeSceneEntries,
  parseListAssetsOutput,
  parseSceneFixtureOutput,
  parseSceneListOutput,
  quoteArg,
} from '../src/helpers';

describe('quoteArg', () => {
  it('wraps a plain path in double quotes', () => {
    expect(quoteArg('assets/hdri/sunset.hdr')).toBe('"assets/hdri/sunset.hdr"');
  });

  it('preserves spaces inside the quotes', () => {
    expect(quoteArg('assets/textures/wood planks.png')).toBe(
      '"assets/textures/wood planks.png"',
    );
  });

  it('escapes embedded quotes and backslashes for the tokenizer', () => {
    expect(quoteArg('we"ird\\path')).toBe('"we\\"ird\\\\path"');
  });
});

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

// --- Wave 9 scene save/load ------------------------------------------------

describe('parseSceneListOutput', () => {
  it('maps bare scene names to scene_load dispatch', () => {
    const raw =
      'lobby\n' + 'studio\n' + "(2 saved scene(s) under 'scenes/')\n";
    expect(parseSceneListOutput(raw)).toEqual([
      { name: 'lobby', dispatch: 'scene_load "lobby"', kind: 'saved' },
      { name: 'studio', dispatch: 'scene_load "studio"', kind: 'saved' },
    ]);
  });

  it('returns [] when only the summary line is present', () => {
    expect(parseSceneListOutput("(0 saved scene(s) under 'scenes/')\n")).toEqual(
      [],
    );
  });
});

describe('parseSceneFixtureOutput', () => {
  it('maps fixture paths to exec dispatch with filename display', () => {
    const raw =
      'tests/goldens/scenes/cornell_csg.cfg\n' +
      "(list_assets: 1 entries under 'tests/goldens/scenes')\n";
    expect(parseSceneFixtureOutput(raw)).toEqual([
      {
        name: 'cornell_csg.cfg',
        dispatch: 'exec "tests/goldens/scenes/cornell_csg.cfg"',
        kind: 'fixture',
      },
    ]);
  });
});

describe('mergeSceneEntries', () => {
  it('lists saved scenes first, then fixtures, each sorted by name', () => {
    const saved = parseSceneListOutput('studio\nlobby\n');
    const fixtures = parseSceneFixtureOutput(
      'tests/goldens/scenes/zebra.cfg\ntests/goldens/scenes/alpha.cfg\n',
    );
    const merged = mergeSceneEntries(saved, fixtures);
    expect(merged.map((e) => e.dispatch)).toEqual([
      'scene_load "lobby"',
      'scene_load "studio"',
      'exec "tests/goldens/scenes/alpha.cfg"',
      'exec "tests/goldens/scenes/zebra.cfg"',
    ]);
  });

  it('de-duplicates by dispatch line', () => {
    const a = parseSceneListOutput('lobby\nlobby\n');
    const merged = mergeSceneEntries(a, []);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.dispatch).toBe('scene_load "lobby"');
  });

  it('handles both sources empty', () => {
    expect(mergeSceneEntries([], [])).toEqual([]);
  });
});

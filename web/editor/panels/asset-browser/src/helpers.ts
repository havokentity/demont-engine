// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic helpers for the Asset Browser. Extracted so the
// console-output parser can be unit-tested without spinning up a
// WebSocket client.

/** Parse the output of `list_assets <subdir>` into an array of
 *  workspace-relative paths. The engine emits one path per line plus
 *  a trailing `(list_assets: N entries under '<dir>')` summary --
 *  this helper drops the summary, blank lines, and anything that
 *  doesn't look like a path. */
export function parseListAssetsOutput(out: string): string[] {
  const lines: string[] = [];
  for (const raw of out.split(/\r?\n/)) {
    const t = raw.trim();
    if (!t) continue;
    // Summary line starts with '(' -- skip.
    if (t.startsWith('(')) continue;
    // Defensive: drop any 'error:' line the engine might emit if a
    // future revision logs to stdout instead of returning ok=false.
    if (t.startsWith('error:')) continue;
    lines.push(t);
  }
  return lines;
}

// --- Wave 9 scene save/load ------------------------------------------------
// One entry in the Scenes tab. Saved scenes (scenes/*.json, written by
// `scene_save`) load via `scene_load <name>`; the shipped .cfg fixtures
// under tests/goldens/scenes/ replay via `exec <path>`. We carry the
// dispatch line per-row so the two sources can coexist in one list.
export interface SceneEntry {
  // Display name: bare scene name for saved scenes, filename for .cfg
  // fixtures.
  name: string;
  // The console line to run on click.
  dispatch: string;
  // Provenance, for the row's secondary text + styling.
  kind: 'saved' | 'fixture';
}

// Parse `scene_list` output (one bare scene name per line + a trailing
// `(N saved scene(s) ...)` summary) into load-dispatchable entries.
// Shares the line-filtering rules with parseListAssetsOutput (summary +
// error lines dropped).
export function parseSceneListOutput(out: string): SceneEntry[] {
  return parseListAssetsOutput(out).map((name) => ({
    name,
    dispatch: `scene_load ${name}`,
    kind: 'saved' as const,
  }));
}

// Parse `list_assets scenes` output (workspace-relative .cfg/.toml
// paths) into exec-dispatchable fixture entries.
export function parseSceneFixtureOutput(out: string): SceneEntry[] {
  return parseListAssetsOutput(out).map((path) => ({
    name: path.split('/').pop() || path,
    dispatch: `exec ${path}`,
    kind: 'fixture' as const,
  }));
}

// Merge saved scenes + shipped fixtures into one stable, de-duplicated
// list. Saved scenes sort first (the user's own work), then fixtures;
// each group is alphabetical by display name. De-dup is by dispatch
// line so the same entry can't appear twice.
export function mergeSceneEntries(
  saved: SceneEntry[],
  fixtures: SceneEntry[],
): SceneEntry[] {
  const seen = new Set<string>();
  const out: SceneEntry[] = [];
  const push = (e: SceneEntry) => {
    if (seen.has(e.dispatch)) return;
    seen.add(e.dispatch);
    out.push(e);
  };
  const byName = (a: SceneEntry, b: SceneEntry) => a.name.localeCompare(b.name);
  [...saved].sort(byName).forEach(push);
  [...fixtures].sort(byName).forEach(push);
  return out;
}

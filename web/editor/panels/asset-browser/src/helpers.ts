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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Asset Browser panel content. Categories surfaced as tabs:
//
//   Scenes  -> saved scenes (scenes/*.json) -> `scene_load <name>`, plus
//              shipped .cfg fixtures (tests/goldens/scenes/) -> `exec <path>`
//   HDRIs   -> *.hdr under assets/hdri/          -> `r_env_map <path>`
//   Meshes  -> .obj/.glb/.gltf under assets/...  -> `mesh_load_gltf <path>`
//
// The list is fetched on demand from the engine via the
// `list_assets <subdir>` console command (added in wave-7 #20). The
// Scenes tab additionally calls `scene_list` (wave-9 scene save/load)
// so scenes written by `scene_save` show up and load on click. Each row
// is single-click to apply + draggable; dropping inside the viewport
// reuses the same dispatch path. The asset browser doesn't need its own
// raycast picker -- the engine already turns drops into origin-spawned
// objects via the cvar / exec the browser dispatches.

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { DragEvent as ReactDragEvent } from 'react';
import type { WebSocketClient } from '@demont/editor-shared';
import {
  mergeSceneEntries,
  parseListAssetsOutput,
  parseSceneFixtureOutput,
  parseSceneListOutput,
  quoteArg,
  type SceneEntry,
} from './helpers';

type Category = 'scenes' | 'hdri' | 'meshes' | 'gltf';

interface AssetBrowserProps {
  client: WebSocketClient;
}

interface CategoryDef {
  key: Category;
  title: string;
  subdir: string;
  // Console command template; `{path}` is replaced with the asset's
  // workspace-relative path, quoted via quoteArg -- the engine's
  // tokenizer splits unquoted tokens on whitespace, so a filename
  // with a space would otherwise truncate the argument.
  dispatch: (path: string) => string;
  // User-facing description shown when the category is empty.
  emptyHint: string;
}

const CATEGORIES: CategoryDef[] = [
  {
    key: 'scenes',
    title: 'Scenes',
    subdir: 'scenes',
    // The Scenes tab computes per-entry dispatch in fetchItems (saved
    // scenes -> `scene_load`, fixtures -> `exec`), so this template is
    // unused for scenes; kept non-empty to satisfy the shared shape.
    dispatch: (path) => `exec ${quoteArg(path)}`,
    emptyHint:
      'Save the current scene with `scene_save <name>` (writes scenes/<name>.json), ' +
      'or drop .cfg fixtures under tests/goldens/scenes/.',
  },
  {
    key: 'hdri',
    title: 'HDRIs',
    subdir: 'hdri',
    dispatch: (path) => `r_env_map ${quoteArg(path)}`,
    emptyHint:
      'Drop .hdr environment maps into assets/hdri/. Click to set as the active sky.',
  },
  {
    key: 'gltf',
    title: 'glTF',
    subdir: 'gltf',
    dispatch: (path) => `mesh_load_gltf ${quoteArg(path)}`,
    emptyHint:
      'Drop .glb / .gltf files into assets/gltf/. Click to import as the current mesh.',
  },
  {
    key: 'meshes',
    title: 'Meshes',
    subdir: 'meshes',
    // No .obj importer yet; the click is a stub `mesh_load_gltf` to
    // surface the error path for the user. .obj support is tracked
    // separately. We still list the directory so users can see what
    // they have.
    dispatch: (path) => `mesh_load_gltf ${quoteArg(path)}`,
    emptyHint:
      'No .obj / .ply / .stl importer wired yet; this category is informational only.',
  },
];

/** Pull the asset list for one category. */
async function fetchAssets(
  client: WebSocketClient,
  subdir: string,
): Promise<string[]> {
  try {
    const r = await client.exec(`list_assets ${subdir}`);
    if (!r.ok) return [];
    return parseListAssetsOutput(String(r.output ?? ''));
  } catch {
    return [];
  }
}

function categoryDef(cat: Category): CategoryDef {
  return CATEGORIES.find((c) => c.key === cat)!;
}

// Uniform list item the panel renders regardless of category. For the
// generic asset categories `name` is the filename, `secondary` is the
// workspace-relative path, and `dispatch` is the cvar/exec line. For the
// Scenes tab the SceneEntry maps onto the same shape (secondary shows
// the dispatch verb so the user can tell a saved scene from a fixture).
interface AssetItem {
  name: string;
  secondary: string;
  dispatch: string;
}

interface RowProps {
  item: AssetItem;
  selected: boolean;
  onClick: () => void;
  onDragStart: (e: ReactDragEvent<HTMLDivElement>) => void;
}

function Row({ item, selected, onClick, onDragStart }: RowProps) {
  return (
    <div
      className={`assets-row${selected ? ' is-selected' : ''}`}
      title={item.dispatch}
      onClick={onClick}
      draggable
      onDragStart={onDragStart}
    >
      <span className="assets-name">{item.name}</span>
      <span className="assets-path">{item.secondary}</span>
    </div>
  );
}

// Fetch one category's items. The Scenes tab is special: it merges
// saved scenes (`scene_list` -> `scene_load <name>`) with the shipped
// .cfg fixtures (`list_assets scenes` -> `exec <path>`). Every other
// category maps `list_assets <subdir>` paths through the category's
// dispatch template.
async function fetchItems(
  client: WebSocketClient,
  cat: Category,
): Promise<AssetItem[]> {
  if (cat === 'scenes') {
    let saved: SceneEntry[] = [];
    let fixtures: SceneEntry[] = [];
    try {
      const r = await client.exec('scene_list');
      if (r.ok) saved = parseSceneListOutput(String(r.output ?? ''));
    } catch {
      /* leave saved empty */
    }
    try {
      const r = await client.exec('list_assets scenes');
      if (r.ok) fixtures = parseSceneFixtureOutput(String(r.output ?? ''));
    } catch {
      /* leave fixtures empty */
    }
    return mergeSceneEntries(saved, fixtures).map((e) => ({
      name: e.name,
      secondary: e.kind === 'saved' ? 'saved · scene_load' : 'fixture · exec',
      dispatch: e.dispatch,
    }));
  }
  const def = categoryDef(cat);
  const paths = await fetchAssets(client, def.subdir);
  return paths.map((p) => ({
    name: p.split('/').pop() || p,
    secondary: p,
    dispatch: def.dispatch(p),
  }));
}

export function AssetBrowser({ client }: AssetBrowserProps) {
  const [active, setActive] = useState<Category>('scenes');
  const [filter, setFilter] = useState<string>('');
  const [items, setItems] = useState<Record<Category, AssetItem[]>>({
    scenes: [],
    hdri: [],
    gltf: [],
    meshes: [],
  });
  const [loading, setLoading] = useState<Record<Category, boolean>>({
    scenes: false,
    hdri: false,
    gltf: false,
    meshes: false,
  });
  const [lastDispatch, setLastDispatch] = useState<string>('');
  const [lastError, setLastError] = useState<string>('');

  // Fetch the active category whenever it changes. Each category is
  // fetched once per mount + on every manual refresh; we don't poll
  // because the asset tree rarely changes at runtime.
  const refresh = useCallback(
    async (cat: Category) => {
      setLoading((s) => ({ ...s, [cat]: true }));
      const out = await fetchItems(client, cat);
      setItems((s) => ({ ...s, [cat]: out }));
      setLoading((s) => ({ ...s, [cat]: false }));
    },
    [client],
  );

  // First-load all four categories so the tab switch feels instant.
  useEffect(() => {
    for (const c of CATEGORIES) {
      void refresh(c.key);
    }
  }, [refresh]);

  const dispatchAsset = useCallback(
    async (line: string) => {
      setLastDispatch(line);
      setLastError('');
      try {
        const r = await client.exec(line);
        if (!r.ok) {
          setLastError(r.error ?? `${line} failed`);
        }
      } catch (err) {
        setLastError(err instanceof Error ? err.message : String(err));
      }
    },
    [client],
  );

  // Drag-start hooks: stuff the dispatch line into the drag payload
  // so the engine's viewport drop handler (if/when wired) can read
  // it. The viewport drop path doesn't exist yet; the same string is
  // also available via dataTransfer.getData('text/plain') for any
  // future native overlay that listens.
  const handleDragStart = useCallback(
    (line: string) => (e: ReactDragEvent<HTMLDivElement>) => {
      e.dataTransfer.setData('application/x-demont-asset', line);
      e.dataTransfer.setData('text/plain', line);
      e.dataTransfer.effectAllowed = 'copy';
    },
    [],
  );

  const visible = useMemo(() => {
    const all = items[active] ?? [];
    if (!filter) return all;
    const f = filter.toLowerCase();
    return all.filter(
      (it) =>
        it.name.toLowerCase().includes(f) ||
        it.secondary.toLowerCase().includes(f),
    );
  }, [items, active, filter]);

  return (
    <div className="assets-root">
      <header className="assets-tabs">
        {CATEGORIES.map((c) => (
          <button
            key={c.key}
            className={`assets-tab${active === c.key ? ' is-active' : ''}`}
            onClick={() => setActive(c.key)}
            type="button"
          >
            {c.title}
            <span className="assets-tab-count">
              {items[c.key]?.length ?? 0}
            </span>
          </button>
        ))}
        <span className="assets-spacer" />
        <button
          className="assets-refresh"
          type="button"
          onClick={() => void refresh(active)}
          title="Re-read the asset directory from disk"
        >
          Refresh
        </button>
      </header>
      <div className="assets-search">
        <input
          type="search"
          placeholder="filter..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <main className="assets-list">
        {loading[active] && visible.length === 0 && (
          <div className="assets-empty">
            <p className="dim">loading...</p>
          </div>
        )}
        {!loading[active] && visible.length === 0 && (
          <div className="assets-empty">
            <p>No matching entries.</p>
            <p className="dim">{categoryDef(active).emptyHint}</p>
          </div>
        )}
        {visible.map((it) => (
          <Row
            key={it.dispatch}
            item={it}
            selected={false}
            onClick={() => void dispatchAsset(it.dispatch)}
            onDragStart={handleDragStart(it.dispatch)}
          />
        ))}
      </main>
      <footer className="assets-statusline">
        {lastDispatch && !lastError && (
          <span className="ok-text">
            dispatched: <code>{lastDispatch}</code>
          </span>
        )}
        {lastError && (
          <span className="error-text">
            error: <code>{lastError}</code>
          </span>
        )}
      </footer>
    </div>
  );
}

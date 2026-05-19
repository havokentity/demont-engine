// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Asset Browser panel content. Three categories surfaced as tabs:
//
//   Scenes  -> .cfg under tests/goldens/scenes/  -> `exec <path>`
//   HDRIs   -> *.hdr under assets/hdri/          -> `r_env_map <path>`
//   Meshes  -> .obj/.glb/.gltf under assets/...  -> `mesh_load_gltf <path>`
//
// The list is fetched on demand from the engine via the new
// `list_assets <subdir>` console command (added in wave-7 #20). Each
// row is single-click to apply + draggable; dropping inside the
// viewport reuses the same dispatch path. The asset browser doesn't
// need its own raycast picker -- the engine already turns drops into
// origin-spawned objects via the cvar / exec the browser dispatches.

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { DragEvent as ReactDragEvent } from 'react';
import type { WebSocketClient } from '@demont/editor-shared';
import { parseListAssetsOutput } from './helpers';

type Category = 'scenes' | 'hdri' | 'meshes' | 'gltf';

interface AssetBrowserProps {
  client: WebSocketClient;
}

interface CategoryDef {
  key: Category;
  title: string;
  subdir: string;
  // Console command template; `{path}` is replaced with the asset's
  // workspace-relative path. We don't pre-quote -- the engine's
  // tokenizer accepts unquoted relative paths fine.
  dispatch: (path: string) => string;
  // User-facing description shown when the category is empty.
  emptyHint: string;
}

const CATEGORIES: CategoryDef[] = [
  {
    key: 'scenes',
    title: 'Scenes',
    subdir: 'scenes',
    dispatch: (path) => `exec ${path}`,
    emptyHint:
      'Place .cfg fixtures under tests/goldens/scenes/. Examples ship with the engine.',
  },
  {
    key: 'hdri',
    title: 'HDRIs',
    subdir: 'hdri',
    dispatch: (path) => `r_env_map ${path}`,
    emptyHint:
      'Drop .hdr environment maps into assets/hdri/. Click to set as the active sky.',
  },
  {
    key: 'gltf',
    title: 'glTF',
    subdir: 'gltf',
    dispatch: (path) => `mesh_load_gltf ${path}`,
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
    dispatch: (path) => `mesh_load_gltf ${path}`,
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

interface RowProps {
  path: string;
  selected: boolean;
  onClick: () => void;
  onDragStart: (e: ReactDragEvent<HTMLDivElement>) => void;
}

function Row({ path, selected, onClick, onDragStart }: RowProps) {
  // Filename for the primary display; full path is the tooltip.
  const name = path.split('/').pop() || path;
  return (
    <div
      className={`assets-row${selected ? ' is-selected' : ''}`}
      title={path}
      onClick={onClick}
      draggable
      onDragStart={onDragStart}
    >
      <span className="assets-name">{name}</span>
      <span className="assets-path">{path}</span>
    </div>
  );
}

export function AssetBrowser({ client }: AssetBrowserProps) {
  const [active, setActive] = useState<Category>('scenes');
  const [filter, setFilter] = useState<string>('');
  const [paths, setPaths] = useState<Record<Category, string[]>>({
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
      const def = categoryDef(cat);
      const out = await fetchAssets(client, def.subdir);
      setPaths((s) => ({ ...s, [cat]: out }));
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
    async (cat: Category, path: string) => {
      const def = categoryDef(cat);
      const line = def.dispatch(path);
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
    (cat: Category, path: string) =>
      (e: ReactDragEvent<HTMLDivElement>) => {
        const def = categoryDef(cat);
        const line = def.dispatch(path);
        e.dataTransfer.setData('application/x-demont-asset', line);
        e.dataTransfer.setData('text/plain', line);
        e.dataTransfer.effectAllowed = 'copy';
      },
    [],
  );

  const visible = useMemo(() => {
    const all = paths[active] ?? [];
    if (!filter) return all;
    const f = filter.toLowerCase();
    return all.filter((p) => p.toLowerCase().includes(f));
  }, [paths, active, filter]);

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
              {paths[c.key]?.length ?? 0}
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
        {visible.map((p) => (
          <Row
            key={p}
            path={p}
            selected={false}
            onClick={() => void dispatchAsset(active, p)}
            onDragStart={handleDragStart(active, p)}
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

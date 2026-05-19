// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Scene Hierarchy panel content. Renders the agent-19 SerializeScene
// payload as a collapsible, filterable tree, dispatches selection
// changes to the engine, and exposes a right-click context menu for
// per-object actions (focus camera, duplicate, delete).
//
// The data flow is:
//   - shared/Shell mounts the WS client + fetches list_scene
//   - selection_change + scene_dirty are mirrored into useSceneStore
//   - this component reads `objects` + `selection` from the store
//     and emits `select` / `exec` over the WS via the passed client
//
// The store flattens the per-kind buckets (`primitives[]`, `lights[]`,
// `sdf[]`, `rigid_bodies[]`) into a single `objects[]` array (each
// tagged with `kind`) so this component can re-group them by kind
// without caring whether the engine ships flat or per-bucket JSON.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import {
  useSceneStore,
  type SceneObject,
  type SelectionKind,
  type WebSocketClient,
} from '@demont/editor-shared';
import {
  autoName,
  rowIcon,
  rgbCss,
  isEmissive,
  matchRange,
  type SectionKey,
} from './helpers';

// ---- types ---------------------------------------------------------------

interface SectionDef {
  key: SectionKey;
  title: string;
  icon: string;  // emoji / symbol
}

const SECTIONS: SectionDef[] = [
  { key: 'prim',  title: 'Primitives',   icon: '\u{1F4E6}' },  // package
  { key: 'light', title: 'Lights',       icon: '\u{1F4A1}' },  // bulb
  { key: 'sdf',   title: 'SDF Clusters', icon: '\u{1F32B}\u{FE0F}' }, // fog
  { key: 'rb',    title: 'Rigid Bodies', icon: '\u{1F3AF}' },  // bullseye
];

interface ContextMenuState {
  x: number;
  y: number;
  kind: SectionKey;
  id: number;
}

// ---- inner components ----------------------------------------------------

interface RowProps {
  o: SceneObject;
  selected: boolean;
  filter: string;
  onSelect: (kind: SectionKey, id: number) => void;
  onContextMenu: (e: ReactMouseEvent, kind: SectionKey, id: number) => void;
  rowRef?: (el: HTMLDivElement | null) => void;
}

function Row({ o, selected, filter, onSelect, onContextMenu, rowRef }: RowProps) {
  const name = autoName(o);
  const range = matchRange(name, filter);
  const swatch = rgbCss((o as Record<string, unknown>).albedo);
  const emissive = isEmissive(o);
  const emissionColor = rgbCss((o as Record<string, unknown>).emission);
  const swatchStyle: CSSProperties = swatch
    ? { backgroundColor: swatch }
    : { backgroundColor: 'transparent', borderStyle: 'dashed' };
  if (emissive && emissionColor) {
    swatchStyle.color = emissionColor;
  }

  const renderName = () => {
    if (!range) return name;
    const [a, b] = range;
    return (
      <>
        {name.slice(0, a)}
        <mark>{name.slice(a, b)}</mark>
        {name.slice(b)}
      </>
    );
  };

  // Filter out kinds the SectionKey union doesn't cover ("csg", "smoke").
  // The store keeps them in objects[] for compatibility; rendering
  // happens via the catch-all section. This component only fires the
  // typed select when kind is one of the four sections.
  const kind = o.kind as SectionKey;

  return (
    <div
      ref={rowRef}
      className={`sh-row${selected ? ' is-selected' : ''}`}
      onClick={() => onSelect(kind, o.id)}
      onContextMenu={(e) => onContextMenu(e, kind, o.id)}
      title={`${name}  #${o.id}`}
      role="treeitem"
      aria-selected={selected}
    >
      <span className="sh-row-icon" aria-hidden="true">{rowIcon(o)}</span>
      {swatch !== null && (
        <span
          className={`sh-row-swatch${emissive ? ' is-emissive' : ''}`}
          style={swatchStyle}
          aria-hidden="true"
        />
      )}
      <span className="sh-row-name">{renderName()}</span>
      <span className="sh-row-id">#{o.id}</span>
    </div>
  );
}

interface SectionProps {
  section: SectionDef;
  rows: SceneObject[];
  open: boolean;
  filter: string;
  selection: { kind: SelectionKind; id: number } | null;
  onToggle: () => void;
  onSelect: (kind: SectionKey, id: number) => void;
  onContextMenu: (e: ReactMouseEvent, kind: SectionKey, id: number) => void;
  selectedRowRef?: (el: HTMLDivElement | null) => void;
}

function Section({
  section,
  rows,
  open,
  filter,
  selection,
  onToggle,
  onSelect,
  onContextMenu,
  selectedRowRef,
}: SectionProps) {
  return (
    <div className="sh-section">
      <div
        className="sh-section-header"
        onClick={onToggle}
        role="button"
        aria-expanded={open}
      >
        <span className={`sh-section-chevron${open ? ' is-open' : ''}`}>
          {'▸'}
        </span>
        <span className="sh-section-icon" aria-hidden="true">{section.icon}</span>
        <span className="sh-section-title">{section.title}</span>
        <span className="sh-section-count">{rows.length}</span>
      </div>
      {open && (
        <div className="sh-section-body" role="group">
          {rows.map((o) => {
            const isSel = selection != null
              && selection.kind === section.key
              && selection.id === o.id;
            return (
              <Row
                key={`${o.kind}-${o.id}`}
                o={o}
                selected={isSel}
                filter={filter}
                onSelect={onSelect}
                onContextMenu={onContextMenu}
                rowRef={isSel ? selectedRowRef : undefined}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

interface ContextMenuProps {
  state: ContextMenuState;
  onClose: () => void;
  onFocus: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}

function ContextMenu({
  state,
  onClose,
  onFocus,
  onDuplicate,
  onDelete,
}: ContextMenuProps) {
  // Close when the user clicks outside or hits Escape. We attach to
  // the document because the menu is rendered above everything; the
  // listeners are removed when the menu unmounts.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && target.closest('.sh-menu')) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  // Keep the menu on-screen if it would overflow. The menu is fixed
  // 160px wide minimum; we leave a 4px buffer on each edge.
  const style: CSSProperties = useMemo(() => {
    const margin = 4;
    const w = 200;
    const h = 160;
    const left = Math.min(state.x, window.innerWidth - w - margin);
    const top  = Math.min(state.y, window.innerHeight - h - margin);
    return {
      left:  Math.max(margin, left),
      top:   Math.max(margin, top),
    };
  }, [state.x, state.y]);

  const canFocus = state.kind === 'prim'
    || state.kind === 'light'
    || state.kind === 'sdf'
    || state.kind === 'rb';
  const canDuplicate = state.kind === 'prim';
  const canDelete    = state.kind === 'prim';

  return (
    <div
      className="sh-menu"
      style={style}
      role="menu"
      onContextMenu={(e) => e.preventDefault()}
    >
      <button
        type="button"
        className="sh-menu-item"
        onClick={onFocus}
        disabled={!canFocus}
        role="menuitem"
      >
        <span>Focus camera</span>
        <span className="sh-menu-shortcut">F</span>
      </button>
      <button
        type="button"
        className="sh-menu-item"
        onClick={onDuplicate}
        disabled={!canDuplicate}
        role="menuitem"
      >
        <span>Duplicate</span>
        <span className="sh-menu-shortcut">{'⌘'}D</span>
      </button>
      <button
        type="button"
        className="sh-menu-item"
        disabled
        role="menuitem"
        title="No name field exists yet -- not in v1"
      >
        <span>Rename</span>
        <span className="sh-menu-shortcut">F2</span>
      </button>
      <div className="sh-menu-divider" />
      <button
        type="button"
        className="sh-menu-item is-danger"
        onClick={onDelete}
        disabled={!canDelete}
        role="menuitem"
      >
        <span>Delete</span>
        <span className="sh-menu-shortcut">{'⌫'}</span>
      </button>
    </div>
  );
}

// ---- main panel ----------------------------------------------------------

export interface SceneHierarchyProps {
  client: WebSocketClient;
}

export function SceneHierarchy({ client }: SceneHierarchyProps) {
  const objects = useSceneStore((s) => s.objects);
  const selection = useSceneStore((s) => s.selection);
  const dirtyTick = useSceneStore((s) => s.sceneDirtyCounter);

  const [filter, setFilter] = useState('');
  const [menu, setMenu] = useState<ContextMenuState | null>(null);

  // Per-section collapsed/open state. Default: open if the section has
  // anything in it on first mount; user-toggled state then persists
  // for the rest of the session. We don't write this to a cvar -- the
  // session-only behaviour matches how Inspector folds work in Unity.
  const [openSections, setOpenSections] = useState<Record<SectionKey, boolean>>({
    prim: true,
    light: true,
    sdf: true,
    rb: true,
  });

  // ---- bucket objects by section ---------------------------------------
  const grouped = useMemo(() => {
    const out: Record<SectionKey, SceneObject[]> = {
      prim: [],
      light: [],
      sdf: [],
      rb: [],
    };
    const lc = filter.trim().toLowerCase();
    for (const o of objects) {
      const k = o.kind as SectionKey;
      if (!(k in out)) continue;  // skip unknown / 'csg' / 'smoke'
      if (lc) {
        const name = autoName(o).toLowerCase();
        if (!name.includes(lc)) continue;
      }
      out[k].push(o);
    }
    // Stable sort by id within each section so rows don't shuffle on
    // every scene_dirty.
    for (const k of Object.keys(out) as SectionKey[]) {
      out[k].sort((a, b) => a.id - b.id);
    }
    return out;
  }, [objects, filter]);

  const totalObjects = useMemo(
    () => objects.filter((o) => (o.kind as SectionKey) in grouped).length,
    [objects, grouped],
  );

  const totalAfterFilter = useMemo(
    () => SECTIONS.reduce((n, s) => n + grouped[s.key].length, 0),
    [grouped],
  );

  // ---- selection dispatching -------------------------------------------
  const handleSelect = useCallback((kind: SectionKey, id: number) => {
    // Don't optimistically update the store -- per CONTRIBUTING.md the
    // engine is the source of truth, and the selection_change event
    // will mirror the change back to us. Optimism only buys ~1 frame
    // and risks divergence if the engine rejects the id (e.g. the
    // user double-clicked at the moment an id was reaped).
    void client.selectObject(kind, id).catch(() => { /* ignore */ });
  }, [client]);

  // ---- right-click menu ------------------------------------------------
  const handleContextMenu = useCallback(
    (e: ReactMouseEvent, kind: SectionKey, id: number) => {
      e.preventDefault();
      e.stopPropagation();
      // Right-click also selects the row so the menu always operates
      // on what the user just pointed at, matching Blender / Unity.
      handleSelect(kind, id);
      setMenu({ x: e.clientX, y: e.clientY, kind, id });
    },
    [handleSelect],
  );

  // Suppress the panel-wide context menu in empty space too -- a
  // default browser menu over a Chrome --app window with dev tools
  // disabled would just confuse the user.
  const handlePanelContextMenu = useCallback((e: ReactMouseEvent) => {
    if ((e.target as HTMLElement | null)?.closest('.sh-row')) return;
    e.preventDefault();
    setMenu(null);
  }, []);

  // ---- context-menu actions --------------------------------------------
  const focusOnObject = useCallback((kind: SectionKey, id: number) => {
    const obj = objects.find((o) => o.kind === kind && o.id === id);
    if (!obj) return;
    const pos = (obj as Record<string, unknown>).pos;
    if (!Array.isArray(pos) || pos.length < 3) return;
    const x = Number(pos[0]);
    const y = Number(pos[1]);
    const z = Number(pos[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    // cam_focus computes the canonical 3m-back / 1m-up offset on the
    // engine side and resets the denoiser history. See
    // Engine::RegisterCommands ("cam_focus") for the math; we just
    // pass the world-space target.
    void client.exec(`cam_focus ${x} ${y} ${z}`);
  }, [objects, client]);

  const duplicateObject = useCallback((id: number) => {
    void client.exec(`prim_duplicate ${id}`);
  }, [client]);

  const deleteObject = useCallback((id: number) => {
    // Single confirm dialog; the engine has no undo yet so make the
    // user opt in. `prim_delete` is the editor alias for `prim_remove`.
    const ok = window.confirm(
      `Delete primitive #${id}?\n\n` +
      `This cannot be undone -- the engine doesn't keep a history.`,
    );
    if (!ok) return;
    void client.exec(`prim_delete ${id}`);
  }, [client]);

  // ---- auto-scroll the selected row into view --------------------------
  const treeRef = useRef<HTMLDivElement | null>(null);
  const selectedRowRef = useRef<HTMLDivElement | null>(null);
  const setSelectedRow = useCallback((el: HTMLDivElement | null) => {
    selectedRowRef.current = el;
  }, []);

  // When the *selection* changes (typically because the user clicked
  // in the viewport), scroll the matching row into view. We DON'T do
  // this on every render -- only when selection actually moves.
  useEffect(() => {
    if (!selection) return;
    // Wait one frame so the row is rendered (e.g. if the user clicked
    // a viewport prim that is currently collapsed under a folded
    // section, we'd miss the element on the synchronous pass).
    const handle = requestAnimationFrame(() => {
      const el = selectedRowRef.current;
      if (!el) return;
      const root = treeRef.current;
      // scrollIntoView with block:'nearest' is gentle -- it only
      // scrolls when the element is actually off-screen, so existing
      // viewport scroll position is preserved when the selection
      // moves between visible rows.
      if (root && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    });
    return () => cancelAnimationFrame(handle);
  }, [selection]);

  // Auto-expand the section the new selection belongs to. If the
  // user collapsed Rigid Bodies but then picked one in the viewport,
  // popping the section open is friendlier than silently leaving the
  // selection invisible.
  useEffect(() => {
    if (!selection) return;
    const k = selection.kind as SectionKey;
    if (!(k in openSections)) return;
    if (openSections[k]) return;
    setOpenSections((s) => ({ ...s, [k]: true }));
    // We intentionally don't include openSections in the deps -- we
    // only want to react to a *new* selection, not to user-driven
    // collapses. The handler reads the current openSections via the
    // closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection]);

  const toggleSection = useCallback((k: SectionKey) => {
    setOpenSections((s) => ({ ...s, [k]: !s[k] }));
  }, []);

  // ---- render ----------------------------------------------------------

  // Empty-scene state. Distinct from the filter-empty state below --
  // the engine genuinely has nothing.
  const isEmpty = totalObjects === 0;
  const isFilterEmpty = !isEmpty && totalAfterFilter === 0;

  return (
    <div className="sh-body">
      <div className="sh-toolbar">
        <input
          className="sh-search"
          type="text"
          placeholder="filter..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter scene objects"
          spellCheck={false}
        />
        {filter && (
          <button
            type="button"
            className="sh-search-clear"
            onClick={() => setFilter('')}
            title="Clear filter (Esc)"
          >
            {'×'}
          </button>
        )}
        <button
          type="button"
          className="sh-refresh"
          onClick={() => { void client.listScene().then((r) => {
            if (r.ok && r.scene && typeof r.scene === 'object') {
              useSceneStore.getState().setScene(r.scene);
            }
          }); }}
          title={`Re-fetch scene (auto-fetched on scene_dirty; tick=${dirtyTick})`}
        >
          refresh
        </button>
      </div>

      <div
        className="sh-tree"
        ref={treeRef}
        onContextMenu={handlePanelContextMenu}
        role="tree"
      >
        {isEmpty ? (
          <EmptyState client={client} />
        ) : isFilterEmpty ? (
          <div className="sh-filter-empty">
            No objects match <code>{filter}</code>.
          </div>
        ) : (
          SECTIONS.map((sec) => {
            const rows = grouped[sec.key];
            // Skip rendering a section that has zero rows -- clears
            // visual noise on a default scene that doesn't use SDFs
            // or rigid bodies.
            if (rows.length === 0) return null;
            return (
              <Section
                key={sec.key}
                section={sec}
                rows={rows}
                open={openSections[sec.key]}
                filter={filter}
                selection={selection}
                onToggle={() => toggleSection(sec.key)}
                onSelect={handleSelect}
                onContextMenu={handleContextMenu}
                selectedRowRef={setSelectedRow}
              />
            );
          })
        )}
      </div>

      {menu && (
        <ContextMenu
          state={menu}
          onClose={() => setMenu(null)}
          onFocus={() => { focusOnObject(menu.kind, menu.id); setMenu(null); }}
          onDuplicate={() => { duplicateObject(menu.id); setMenu(null); }}
          onDelete={() => { deleteObject(menu.id); setMenu(null); }}
        />
      )}
    </div>
  );
}

// ---- empty state ---------------------------------------------------------

function EmptyState({ client }: { client: WebSocketClient }) {
  const example = 'prim_sphere 100099 0 0.5 0 0.5 lambert 1 1 1';
  return (
    <div className="sh-empty">
      <h4>Empty scene</h4>
      <p>Use a console command to populate the world. Try:</p>
      <button
        type="button"
        className="sh-empty-example"
        onClick={() => void client.exec(example)}
        title="Click to run this command in the engine"
      >
        {example}
      </button>
      <p className="sh-empty-hint">
        also: <code>light_point_add</code>, <code>sdf_load</code>,
        {' '}<code>phys_drop_sphere</code>
      </p>
    </div>
  );
}

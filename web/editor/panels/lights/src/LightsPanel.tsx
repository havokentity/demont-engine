// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Lights panel root. Lists every analytic light (point / spot / sphere
// / quad), lets the user add + remove them, and inspects the selected
// one. The data flow mirrors the other editor panels:
//
//   - shared/Shell mounts the WS client + fetches list_scene
//   - selection_change + scene_dirty are mirrored into useSceneStore
//   - this component reads `scene` (the raw snapshot) + `selection` from
//     the store, parses the `lights[]` array directly for the defensive
//     per-field coercion in helpers.ts (the flattened `objects[]` also
//     carries the records -- SerializeScene stamps them kind:'light'
//     and flattenScene preserves that -- the inspector panel uses the
//     same raw-snapshot approach), and emits select / exec over the WS
//     via the passed client
//
// Writes flow through the engine's atomic `light_set_*` setters (via
// LightInspector) and the `light_point` / `light_spot` / `light_sphere`
// / `light_quad` constructors (for add). `light_remove` drops a light.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  useSceneStore,
  type WebSocketClient,
} from '@demont/editor-shared';
import { LightInspector } from './LightInspector';
import {
  parseLights,
  nextLightId,
  lightTypeLabel,
  lightTypeIcon,
  type LightRecord,
  type LightType,
  type IntensityUnit,
} from './helpers';

interface LightsPanelProps {
  client: WebSocketClient;
}

// Throttled `exec` -- buckets by the first whitespace-delimited token so
// independent commands (light_set_pos vs light_set_intensity) don't
// displace each other; the trailing call within kThrottleMs wins. Same
// shape as the inspector panel's useThrottledExec.
const kThrottleMs = 50;

function useThrottledExec(client: WebSocketClient): (line: string) => void {
  const pendingRef = useRef<Map<string, { line: string; timer: number }>>(
    new Map(),
  );
  const lastFiredRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const pending = pendingRef.current;
    return () => {
      for (const v of pending.values()) {
        if (v.timer) window.clearTimeout(v.timer);
      }
      pending.clear();
    };
  }, []);

  return useCallback(
    (line: string) => {
      const bucket = line.split(/\s+/)[0] ?? line;
      const now = performance.now();
      const last = lastFiredRef.current.get(bucket) ?? 0;
      const elapsed = now - last;

      const existing = pendingRef.current.get(bucket);
      if (existing) {
        window.clearTimeout(existing.timer);
      }

      const fire = (l: string) => {
        lastFiredRef.current.set(bucket, performance.now());
        pendingRef.current.delete(bucket);
        void client.exec(l).catch(() => { /* offline; retry on next edit */ });
      };

      if (elapsed >= kThrottleMs) {
        fire(line);
      } else {
        const wait = kThrottleMs - elapsed;
        const timer = window.setTimeout(() => fire(line), wait);
        pendingRef.current.set(bucket, { line, timer });
      }
    },
    [client],
  );
}

// Default-construction console lines for the "add" buttons. Each spawns
// a sensible starter light a couple metres above the origin so it's
// visible in a default scene.
function addLightCommand(type: LightType, id: number): string {
  switch (type) {
    case 'point':
      // point <id> x y z  r g b   (W/sr)
      return `light_point ${id} 0 3 0 5 5 5`;
    case 'spot':
      // spot <id> x y z  dx dy dz  outer inner  r g b
      return `light_spot ${id} 0 4 0 0 -1 0 35 20 8 8 8`;
    case 'sphere':
      // sphere <id> x y z  radius  r g b   (W/m^2/sr)
      return `light_sphere ${id} 0 3 0 0.25 6 6 6`;
    case 'quad':
      // quad <id> x y z  nx ny nz  ux uy uz  u_half v_half  r g b
      return `light_quad ${id} 0 4 0 0 -1 0 1 0 0 0.5 0.5 4 4 4`;
  }
}

const ADD_TYPES: LightType[] = ['point', 'spot', 'sphere', 'quad'];

export function LightsPanel({ client }: LightsPanelProps) {
  const scene = useSceneStore((s) => s.scene);
  const selection = useSceneStore((s) => s.selection);
  const dirtyTick = useSceneStore((s) => s.sceneDirtyCounter);

  const exec = useThrottledExec(client);

  // Per-light intensity authoring unit. Keyed by light id so switching
  // selection keeps each light's chosen unit. Defaults to 'raw'.
  const [units, setUnits] = useState<Record<number, IntensityUnit>>({});

  const [confirmDelete, setConfirmDelete] = useState<boolean>(false);
  useEffect(() => {
    setConfirmDelete(false);
  }, [selection?.kind, selection?.id]);

  // Parse the typed light list from the raw snapshot. Re-derived on
  // scene / dirtyTick change (Shell refetches on scene_dirty).
  const lights = useMemo<LightRecord[]>(() => {
    return parseLights(scene);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, dirtyTick]);

  const selectedLight = useMemo<LightRecord | undefined>(() => {
    if (!selection || selection.kind !== 'light') return undefined;
    return lights.find((l) => l.id === selection.id);
  }, [selection, lights]);

  const unitFor = useCallback(
    (id: number): IntensityUnit => units[id] ?? 'raw',
    [units],
  );

  const setUnitFor = useCallback((id: number, u: IntensityUnit) => {
    setUnits((prev) => ({ ...prev, [id]: u }));
  }, []);

  // ---- selection ----
  const handleSelect = useCallback(
    (id: number) => {
      // Engine is the source of truth; the selection_change event mirrors
      // the change back into the store.
      void client.selectObject('light', id).catch(() => { /* ignore */ });
    },
    [client],
  );

  // ---- add ----
  const handleAdd = useCallback(
    async (type: LightType) => {
      const id = nextLightId(lights);
      try {
        await client.exec(addLightCommand(type, id));
        // Select the freshly-added light so the inspector populates.
        await client.selectObject('light', id);
      } catch {
        // Offline; ignore. The next scene_dirty re-fetch will show it
        // if the command landed before the socket dropped.
      }
    },
    [client, lights],
  );

  // ---- delete ----
  const handleDelete = useCallback(() => {
    if (!selectedLight) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      window.setTimeout(() => setConfirmDelete(false), 2500);
      return;
    }
    void client.exec(`light_remove ${selectedLight.id}`).catch(() => { /* engine offline */ });
    setConfirmDelete(false);
  }, [client, selectedLight, confirmDelete]);

  // ---- frame in viewport ----
  // cam_focus computes the canonical back-and-up offset on the engine
  // side and resets the denoiser history. Scene-hierarchy uses the same.
  const handleGoto = useCallback(() => {
    if (!selectedLight) return;
    const [x, y, z] = selectedLight.pos;
    if (![x, y, z].every(Number.isFinite)) return;
    void client.exec(`cam_focus ${x} ${y} ${z}`).catch(() => { /* engine offline */ });
  }, [client, selectedLight]);

  // ---- auto-scroll the selected row into view ----
  const listRef = useRef<HTMLDivElement | null>(null);
  const selectedRowRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!selection || selection.kind !== 'light') return;
    const handle = requestAnimationFrame(() => {
      const el = selectedRowRef.current;
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    });
    return () => cancelAnimationFrame(handle);
  }, [selection]);

  // ---- render ----
  return (
    <div className="lp-body">
      <div className="lp-toolbar">
        <span className="lp-toolbar-label">Add</span>
        {ADD_TYPES.map((t) => (
          <button
            key={t}
            type="button"
            className="lp-add-btn"
            onClick={() => void handleAdd(t)}
            title={`Add a ${lightTypeLabel(t)} (${t})`}
          >
            <span aria-hidden="true">{lightTypeIcon(t)}</span>
            {t}
          </button>
        ))}
        <button
          type="button"
          className="lp-refresh"
          onClick={() => {
            void client.listScene().then((r) => {
              if (r.ok && r.scene && typeof r.scene === 'object') {
                useSceneStore.getState().setScene(r.scene);
              }
            });
          }}
          title={`Re-fetch scene (auto on scene_dirty; tick=${dirtyTick})`}
        >
          refresh
        </button>
      </div>

      <div className="lp-list" ref={listRef} role="list">
        {lights.length === 0 ? (
          <div className="lp-empty">
            <h4>No lights</h4>
            <p>
              Add an analytic light with the buttons above, or run a
              console command like{' '}
              <code>light_point 1 0 3 0 5 5 5</code>.
            </p>
          </div>
        ) : (
          lights.map((l) => {
            const isSel =
              selection != null &&
              selection.kind === 'light' &&
              selection.id === l.id;
            const peak = Math.max(l.intensity[0], l.intensity[1], l.intensity[2]);
            return (
              <div
                key={l.id}
                ref={isSel ? selectedRowRef : undefined}
                className={`lp-row-item${isSel ? ' is-selected' : ''}`}
                onClick={() => handleSelect(l.id)}
                role="listitem"
                aria-selected={isSel}
                title={`${lightTypeLabel(l.type)}  #${l.id}`}
              >
                <span className="lp-row-icon" aria-hidden="true">
                  {lightTypeIcon(l.type)}
                </span>
                <span className="lp-row-name">{l.type}</span>
                <span className="lp-row-id">#{l.id}</span>
                <span className="lp-row-intensity" title="Peak raw intensity">
                  {peak.toFixed(2)}
                </span>
              </div>
            );
          })
        )}
      </div>

      {selectedLight && (
        <div className="lp-inspector">
          <header className="lp-header">
            <span className="lp-badge is-light">
              {lightTypeLabel(selectedLight.type)}
            </span>
            <span className="lp-id">#{selectedLight.id}</span>
            <button
              type="button"
              className="lp-goto"
              onClick={handleGoto}
              title="Frame light in viewport (cam_focus)"
            >
              Goto
            </button>
          </header>
          <LightInspector
            light={selectedLight}
            unit={unitFor(selectedLight.id)}
            onUnitChange={(u) => setUnitFor(selectedLight.id, u)}
            exec={exec}
          />
          <div className="lp-actions">
            <button
              type="button"
              className={`lp-btn is-danger${confirmDelete ? ' is-confirm' : ''}`}
              onClick={handleDelete}
            >
              {confirmDelete ? 'Confirm Delete?' : 'Delete'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

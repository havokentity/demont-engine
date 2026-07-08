// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Property Inspector root component. Watches the shared store's
// selection + scene snapshot; on every change to either, re-derives
// the currently inspected record. Updates dispatch through Console::
// Execute via the WS client.
//
// Why pull from store.scene (raw snapshot) and not store.objects? The
// agent-20 store flattens `prims[]` and `lights[]` into store.objects
// but agent-19's SerializeScene emits top-level `primitives[]` /
// `lights[]` / `sdf[]` etc. -- mismatched key names. We read the raw
// snapshot to be resilient to that shape detail; the inspector cares
// about the typed records, not the flat normalisation.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  useSceneStore,
  type WebSocketClient,
  type SceneSnapshot,
} from '@demont/editor-shared';
import { PrimInspector, type PrimRecord } from './PrimInspector';
import { LightInspector, type LightRecord } from './LightInspector';

interface AppProps {
  client: WebSocketClient;
}

interface RawPrim {
  id: number;
  type: 'sphere' | 'plane';
  material: 'lambert' | 'metal' | 'dielectric' | 'water';
  pos?: [number, number, number];
  radius?: number;
  normal?: [number, number, number];
  d?: number;
  albedo: [number, number, number];
  emission: [number, number, number];
  roughness: number;
  ior: number;
}

interface RawLight {
  id: number;
  type: 'point' | 'spot' | 'sphere' | 'quad';
  pos: [number, number, number];
  intensity: [number, number, number];
  dir?: [number, number, number];
  cos_outer?: number;
  cos_inner?: number;
  radius?: number;
  u_vec?: [number, number, number];
  v_half?: number;
}

function findPrim(snap: SceneSnapshot | null, id: number): RawPrim | undefined {
  if (!snap) return undefined;
  const arr =
    (snap as Record<string, unknown>).primitives ??
    (snap as Record<string, unknown>).prims;
  if (!Array.isArray(arr)) return undefined;
  for (const p of arr as RawPrim[]) {
    if (p && p.id === id) return p;
  }
  return undefined;
}

function findLight(snap: SceneSnapshot | null, id: number): RawLight | undefined {
  if (!snap) return undefined;
  const arr = (snap as Record<string, unknown>).lights;
  if (!Array.isArray(arr)) return undefined;
  for (const l of arr as RawLight[]) {
    if (l && l.id === id) return l;
  }
  return undefined;
}

function primTypeBadge(type: 'sphere' | 'plane'): string {
  return type === 'sphere' ? 'Sphere' : 'Plane';
}

function lightTypeBadge(type: 'point' | 'spot' | 'sphere' | 'quad'): string {
  switch (type) {
    case 'point':  return 'Point Light';
    case 'spot':   return 'Spot Light';
    case 'sphere': return 'Sphere Light';
    case 'quad':   return 'Quad Light';
  }
}

// Throttled `exec` -- if the caller fires more than once per
// kThrottleMs for the same command prefix, the trailing call wins.
// We bucket by the first whitespace-delimited token so independent
// commands (prim_set_pos vs prim_set_albedo) don't displace each other.
const kThrottleMs = 50;

function useThrottledExec(client: WebSocketClient): (line: string) => void {
  const pendingRef = useRef<Map<string, { line: string; timer: number }>>(
    new Map(),
  );
  const lastFiredRef = useRef<Map<string, number>>(new Map());

  // Cleanup on unmount.
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
        // We deliberately don't await -- callers don't need the
        // result and a long network stall shouldn't queue here.
        void client.exec(l).catch(() => { /* offline; will retry on next edit */ });
      };

      if (elapsed >= kThrottleMs) {
        fire(line);
      } else {
        // Schedule the trailing edge.
        const wait = kThrottleMs - elapsed;
        const timer = window.setTimeout(() => fire(line), wait);
        pendingRef.current.set(bucket, { line, timer });
      }
    },
    [client],
  );
}

export function App({ client }: AppProps) {
  const selection = useSceneStore((s) => s.selection);
  const scene = useSceneStore((s) => s.scene);
  const dirtyTick = useSceneStore((s) => s.sceneDirtyCounter);

  const exec = useThrottledExec(client);

  // Delete confirmation latch -- two-click destructive UI.
  const [confirmDelete, setConfirmDelete] = useState<boolean>(false);
  useEffect(() => {
    // Reset confirmation when selection changes so a leftover
    // primed state doesn't fire on the next object.
    setConfirmDelete(false);
  }, [selection?.kind, selection?.id]);

  // Re-derive the inspected record whenever scene / selection /
  // dirtyTick changes. The dependency on dirtyTick is the throttled
  // refresh hook -- Shell already refetches scene on the event, so
  // we just need to re-render when the snapshot lands.
  const prim = useMemo<RawPrim | undefined>(() => {
    if (!selection || selection.kind !== 'prim') return undefined;
    return findPrim(scene, selection.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.kind, selection?.id, scene, dirtyTick]);

  const light = useMemo<RawLight | undefined>(() => {
    if (!selection || selection.kind !== 'light') return undefined;
    return findLight(scene, selection.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.kind, selection?.id, scene, dirtyTick]);

  // ---- Header actions ----
  // cam_focus computes the canonical back-and-up framing offset on the
  // engine side and resets the temporal denoiser history so the next
  // frame is a clean sample of the new viewpoint. Scene-hierarchy and
  // the lights panel dispatch the same command for their Goto actions.
  const onGoto = useCallback(() => {
    if (!selection) return;
    let target: [number, number, number] | null = null;
    if (selection.kind === 'prim') {
      if (prim?.type === 'sphere' && prim.pos) target = prim.pos;
      else if (prim?.type === 'plane') target = [0, 0, 0];
    } else if (selection.kind === 'light' && light) {
      target = light.pos;
    }
    if (!target) return;
    void client.exec(`cam_focus ${target[0]} ${target[1]} ${target[2]}`)
      .catch(() => { /* engine offline */ });
  }, [client, selection, prim, light]);

  const onDuplicate = useCallback(async () => {
    if (!selection || selection.kind !== 'prim') return;
    try {
      const r = await client.exec(`prim_duplicate ${selection.id}`);
      // Engine emits `prim_duplicate: cloned id <src> -> <new>` in
      // r.output. Parse and switch selection.
      const out = (r?.output ?? '') as string;
      const m = out.match(/->\s*(\d+)/);
      if (m) {
        const newId = parseInt(m[1], 10);
        if (Number.isFinite(newId)) {
          await client.selectObject('prim', newId);
        }
      }
    } catch {
      // Offline; ignore.
    }
  }, [client, selection]);

  const onDelete = useCallback(() => {
    if (!selection) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      window.setTimeout(() => setConfirmDelete(false), 2500);
      return;
    }
    if (selection.kind === 'prim') {
      void client.exec(`prim_delete ${selection.id}`).catch(() => { /* engine offline */ });
    } else if (selection.kind === 'light') {
      void client.exec(`light_remove ${selection.id}`).catch(() => { /* engine offline */ });
    }
    setConfirmDelete(false);
  }, [client, selection, confirmDelete]);

  // ---- Render ----

  // No selection -- the store maps the engine's {kind:'none'} payload
  // to null (selectionFromPayload in shared/src/store.ts).
  if (!selection) {
    return (
      <div className="insp-empty">
        <h3>No object selected.</h3>
        <p>
          Click an object in the viewport or pick from{' '}
          <code>Scene Hierarchy</code> to populate this panel.
        </p>
      </div>
    );
  }

  // Unsupported (sdf / smoke / csg / rb) -- the engine has selection
  // state for them but no editor mutation path yet.
  if (selection.kind !== 'prim' && selection.kind !== 'light') {
    return (
      <div className="insp-empty">
        <h3>{selection.kind} #{selection.id}</h3>
        <p>
          The inspector currently supports analytic primitives and
          analytic lights. Selection kind <code>{selection.kind}</code>{' '}
          isn't editable from this panel yet.
        </p>
      </div>
    );
  }

  // Selection points at a kind we support but the snapshot doesn't
  // have the record (rare race during scene_dirty refetch).
  if (
    (selection.kind === 'prim' && !prim) ||
    (selection.kind === 'light' && !light)
  ) {
    return (
      <div className="insp-empty">
        <h3>{selection.kind} #{selection.id}</h3>
        <p>
          Waiting for scene snapshot to refresh after a recent change...
        </p>
      </div>
    );
  }

  if (selection.kind === 'prim' && prim) {
    const primRecord: PrimRecord = prim;
    return (
      <div className="insp-root">
        <header className="insp-header">
          <span className="insp-badge">{primTypeBadge(prim.type)}</span>
          <span className="insp-id">#{prim.id}</span>
          <button
            className="insp-goto"
            onClick={onGoto}
            title="Frame object in viewport (cam_focus)"
          >
            Goto
          </button>
        </header>
        <PrimInspector prim={primRecord} exec={exec} />
        <div className="insp-actions">
          <button className="insp-btn" onClick={onDuplicate}>
            Duplicate
          </button>
          <button
            className={`insp-btn is-danger${confirmDelete ? ' is-confirm' : ''}`}
            onClick={onDelete}
          >
            {confirmDelete ? 'Confirm Delete?' : 'Delete'}
          </button>
        </div>
      </div>
    );
  }

  if (selection.kind === 'light' && light) {
    const lightRecord: LightRecord = light;
    return (
      <div className="insp-root">
        <header className="insp-header">
          <span className="insp-badge is-light">{lightTypeBadge(light.type)}</span>
          <span className="insp-id">#{light.id}</span>
          <button
            className="insp-goto"
            onClick={onGoto}
            title="Frame light in viewport (cam_focus)"
          >
            Goto
          </button>
        </header>
        <LightInspector light={lightRecord} exec={exec} />
        <div className="insp-actions">
          <button
            className={`insp-btn is-danger${confirmDelete ? ' is-confirm' : ''}`}
            onClick={onDelete}
          >
            {confirmDelete ? 'Confirm Delete?' : 'Delete'}
          </button>
        </div>
      </div>
    );
  }

  return null;
}

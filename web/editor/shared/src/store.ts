// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Shared scene store. Zustand chosen (over Jotai / Context) because:
//   - cross-panel iframe-style isolation isn't a goal; each panel runs
//     in its own Chrome --app window, so each window has its own store
//     instance, but the shape stays identical
//   - zustand exposes a plain getState() / setState() that lets the
//     WebSocket client push updates from outside the React tree
//     without forcing every consumer through useEffect/useState
//   - hooks-based selectors give us per-field subscription out of the
//     box (no re-render on unrelated scene_dirty events)
//
// State shape:
//   - connection status (mirrors WebSocketClient.status)
//   - the last full scene snapshot from `list_scene`
//   - selection (kind+id or null)
//   - a tiny log ring buffer so panels can drop in a toast notifier
//
// The WebSocketClient is intentionally NOT held in the store -- it's
// long-lived per panel and created once in main.tsx, then passed to
// Shell which subscribes the store to its events.

import { create } from 'zustand';
import type {
  Selection,
  SelectionKind,
  SceneSnapshot,
  SceneObject,
  AnalyticPrim,
  AnalyticLight,
  WsEvent,
  CVarValue,
} from './types';
import type { ClientStatus } from './ws-client';

export interface LogEntry {
  level: 'info' | 'warn' | 'error';
  message: string;
  ts: number;  // ms since epoch
}

export interface SceneStore {
  // ---- connection ----
  connectionStatus: ClientStatus;
  setConnectionStatus: (s: ClientStatus) => void;

  // ---- scene ----
  scene: SceneSnapshot | null;
  // Convenience view: flat objects array regardless of which top-
  // level shape the engine returned (objects[] OR prims[]+lights[]).
  objects: SceneObject[];
  setScene: (s: SceneSnapshot | null) => void;

  // Per-frame "the scene mutated, fetch a fresh snapshot" signal.
  // Bumped on every `scene_dirty` event so panels can wire a
  // useEffect dependency on it without holding the WS client.
  sceneDirtyCounter: number;
  bumpSceneDirty: () => void;

  // ---- selection ----
  selection: Selection | null;
  setSelection: (sel: Selection | null) => void;

  // Lookup helpers. Cheap O(N) scans -- scene size is tiny for the
  // editor MVP (handful of prims / lights). Will revisit if we ship a
  // 10k-object stress test.
  findPrim: (id: number) => AnalyticPrim | undefined;
  findLight: (id: number) => AnalyticLight | undefined;
  findObject: (kind: SelectionKind, id: number) => SceneObject | undefined;

  // ---- cvars cache (optional, panels can fetch on demand) ----
  cvars: Record<string, CVarValue>;
  setCVar: (cv: CVarValue) => void;
  setCVars: (list: CVarValue[]) => void;

  // ---- log ring buffer ----
  logs: LogEntry[];
  pushLog: (entry: LogEntry) => void;
  clearLogs: () => void;
}

const kLogRingSize = 200;

// `flattenScene` normalises the two shapes agent-19 might emit.
// agent-19's PR documentation says SerializeScene returns a JSON
// object; we don't pin the exact field layout here so either flat
// `objects[]` or per-kind `prims[]+lights[]` works on day one.
function flattenScene(s: SceneSnapshot | null): SceneObject[] {
  if (!s) return [];
  if (Array.isArray(s.objects)) return s.objects;
  const out: SceneObject[] = [];
  if (Array.isArray(s.prims)) {
    for (const p of s.prims) out.push(p);
  }
  if (Array.isArray(s.lights)) {
    for (const l of s.lights) out.push(l);
  }
  // Pull in any other top-level array we don't know about so
  // unknown kinds are still visible to the hierarchy panel.
  for (const [k, v] of Object.entries(s)) {
    if (k === 'objects' || k === 'prims' || k === 'lights') continue;
    if (Array.isArray(v)) {
      for (const obj of v) {
        if (obj && typeof obj === 'object') {
          // Tag the kind if the engine didn't provide one.
          const tagged = ('kind' in obj)
            ? obj as SceneObject
            : { ...obj as Record<string, unknown>, kind: k } as SceneObject;
          out.push(tagged);
        }
      }
    }
  }
  return out;
}

// Map a raw engine selection payload to the store's Selection-or-null
// shape. The engine represents SelectionKind::None as {kind:'none',
// id:0} (SelectionKindToString in src/editor/SceneGraph.cpp) both in
// the selection_change event and the list_scene snapshot's `selection`
// field -- 'none' is not part of the SelectionKind union, so it maps
// to null here.
export function selectionFromPayload(
  d: { kind?: string; id?: number } | null | undefined,
): Selection | null {
  if (!d || !d.kind || d.kind === 'none' || typeof d.id !== 'number') {
    return null;
  }
  return { kind: d.kind as SelectionKind, id: d.id };
}

export const useSceneStore = create<SceneStore>((set, get) => ({
  connectionStatus: 'connecting',
  setConnectionStatus: (s) => set({ connectionStatus: s }),

  scene: null,
  objects: [],
  setScene: (s) => set({ scene: s, objects: flattenScene(s) }),

  sceneDirtyCounter: 0,
  bumpSceneDirty: () => set((st) => ({ sceneDirtyCounter: st.sceneDirtyCounter + 1 })),

  selection: null,
  setSelection: (sel) => set({ selection: sel }),

  findPrim: (id) => {
    for (const o of get().objects) {
      if (o.kind === 'prim' && o.id === id) return o as AnalyticPrim;
    }
    return undefined;
  },
  findLight: (id) => {
    for (const o of get().objects) {
      if (o.kind === 'light' && o.id === id) return o as AnalyticLight;
    }
    return undefined;
  },
  findObject: (kind, id) => {
    for (const o of get().objects) {
      if (o.kind === kind && o.id === id) return o;
    }
    return undefined;
  },

  cvars: {},
  setCVar: (cv) => set((st) => ({ cvars: { ...st.cvars, [cv.name]: cv } })),
  setCVars: (list) => set((st) => {
    const next = { ...st.cvars };
    for (const cv of list) next[cv.name] = cv;
    return { cvars: next };
  }),

  logs: [],
  pushLog: (entry) => set((st) => {
    const logs = st.logs.length >= kLogRingSize
      ? [...st.logs.slice(st.logs.length - kLogRingSize + 1), entry]
      : [...st.logs, entry];
    return { logs };
  }),
  clearLogs: () => set({ logs: [] }),
}));

// Convenience: wire a WebSocketClient's events into the store. Call
// once from Shell after the client starts -- returns an unsubscribe
// function for clean teardown on unmount.
export function bindClientToStore(
  ws: {
    onEvent: (topic: string, fn: (e: WsEvent) => void) => () => void;
    onStatus: (fn: (s: ClientStatus) => void) => () => void;
  },
): () => void {
  const store = useSceneStore.getState;
  const unsubscribers: Array<() => void> = [];

  unsubscribers.push(ws.onStatus((s) => store().setConnectionStatus(s)));

  unsubscribers.push(ws.onEvent('selection_change', (e) => {
    const d = e.data as { kind?: string; id?: number } | null;
    store().setSelection(selectionFromPayload(d));
  }));

  unsubscribers.push(ws.onEvent('scene_dirty', () => {
    store().bumpSceneDirty();
  }));

  unsubscribers.push(ws.onEvent('log', (e) => {
    const d = e.data as { level?: string; message?: string } | null;
    if (!d || typeof d.message !== 'string') return;
    let level: LogEntry['level'] = 'info';
    if (d.level === 'warn' || d.level === 'error') level = d.level;
    store().pushLog({ level, message: d.message, ts: Date.now() });
  }));

  return () => {
    for (const u of unsubscribers) u();
  };
}

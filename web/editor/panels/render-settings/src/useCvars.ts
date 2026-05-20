// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Cvar cache hook for the Render Settings panel. Bulk-fetches the
// current value of every cvar the panel binds (via per-cvar get_cvar
// over the shared WebSocket) and exposes a throttled setter.
//
// Read path:
//   - on first open AND on every (re)connect, fire one get_cvar per
//     bound cvar in parallel and fold the string values into a map.
//   - cvars never emit `scene_dirty`, so there is no per-frame refresh
//     -- the local map is the UI's source of truth between writes, and
//     a reconnect re-syncs it.
//
// Write path:
//   - set() optimistically updates the local map (so the slider thumb
//     doesn't snap back while the round-trip is in flight) and dispatches
//     a throttled set_cvar. Bucketed by cvar name so dragging the fog
//     density slider doesn't displace a god-ray weight write.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { WebSocketClient, ClientStatus } from '@demont/editor-shared';

const kThrottleMs = 50;

export interface CvarMap {
  [name: string]: string;
}

export interface UseCvarsResult {
  /** Current cvar string values keyed by name. Missing until fetched. */
  values: CvarMap;
  /** True while the initial bulk fetch is in flight. */
  loading: boolean;
  /** Throttled write: optimistic local update + set_cvar dispatch. */
  set: (name: string, value: string) => void;
  /** Force a re-read of every cvar (e.g. a manual refresh button). */
  refresh: () => void;
}

export function useCvars(
  client: WebSocketClient,
  names: string[],
): UseCvarsResult {
  const [values, setValues] = useState<CvarMap>({});
  const [loading, setLoading] = useState<boolean>(true);

  // Stable snapshot of the name list so the fetch callback doesn't churn
  // when the parent re-renders with a fresh array identity.
  const namesRef = useRef<string[]>(names);
  namesRef.current = names;

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const results = await Promise.allSettled(
        namesRef.current.map((name) =>
          client.send({ type: 'get_cvar', name }),
        ),
      );
      const next: CvarMap = {};
      results.forEach((r, i) => {
        const name = namesRef.current[i];
        if (
          r.status === 'fulfilled' &&
          r.value.ok &&
          r.value.cvar &&
          typeof r.value.cvar.value === 'string'
        ) {
          next[name] = r.value.cvar.value;
        }
      });
      // Merge rather than replace so a partial fetch (engine restarting
      // mid-batch) doesn't blank already-known values.
      setValues((prev) => ({ ...prev, ...next }));
    } catch {
      // Socket not open; the status listener re-fires fetchAll on the
      // next 'open'.
    } finally {
      setLoading(false);
    }
  }, [client]);

  // Initial fetch + re-fetch on every reconnect.
  useEffect(() => {
    void fetchAll();
    let prev: ClientStatus = client.getStatus();
    const off = client.onStatus((s) => {
      // Re-sync whenever we transition INTO 'open' from a non-open state.
      if (s === 'open' && prev !== 'open') {
        void fetchAll();
      }
      prev = s;
    });
    return off;
  }, [client, fetchAll]);

  // Throttled, per-cvar-bucketed setter.
  const pendingRef = useRef<Map<string, { value: string; timer: number }>>(
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

  const set = useCallback(
    (name: string, value: string) => {
      // Optimistic local update so the control reflects the edit
      // immediately.
      setValues((prev) => ({ ...prev, [name]: value }));

      const now = performance.now();
      const last = lastFiredRef.current.get(name) ?? 0;
      const elapsed = now - last;
      const existing = pendingRef.current.get(name);
      if (existing) window.clearTimeout(existing.timer);

      const fire = (val: string) => {
        lastFiredRef.current.set(name, performance.now());
        pendingRef.current.delete(name);
        void client
          .send({ type: 'set_cvar', name, value: val })
          .catch(() => {
            /* offline; the next reconnect re-reads the truth */
          });
      };

      if (elapsed >= kThrottleMs) {
        fire(value);
      } else {
        const wait = kThrottleMs - elapsed;
        const timer = window.setTimeout(() => fire(value), wait);
        pendingRef.current.set(name, { value, timer });
      }
    },
    [client],
  );

  const refresh = useCallback(() => {
    void fetchAll();
  }, [fetchAll]);

  return { values, loading, set, refresh };
}

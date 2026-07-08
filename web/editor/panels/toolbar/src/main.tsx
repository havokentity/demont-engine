// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Editor Toolbar panel -- the wave-7 polish refit. Replaces the
// MVP "general console commands" demo with the gizmo + undo + snap
// + transform-space top-bar described in #20.
//
// Layout: thin (~32 px) horizontal strip styled after Blender's
// header. Buttons fall into three groups:
//   - Gizmo mode (Translate / Rotate / Scale) -- dispatches the
//     existing `gizmo_mode <mode>` cvar
//   - Undo / Redo -- dispatches `scene_undo` / `scene_redo`
//   - Snap toggle (`gizmo_snap`) + Space selector (`gizmo_space`)
//
// The panel polls the three cvars at mount and re-fetches whenever
// the user clicks one of its own buttons OR receives a relevant
// log event. The cvars are read via the existing `get_cvar`
// inbound message so we don't need a new WS opcode.

import { StrictMode, useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Shell,
  Toolbar,
  ToolbarGroup,
  ToolbarButton,
  type WebSocketClient,
} from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';

type GizmoMode = 'translate' | 'rotate' | 'scale';
type GizmoSpace = 'world' | 'local';

interface ToolbarBodyProps {
  client: WebSocketClient;
}

// Pull a single cvar's value off the engine. Returns undefined on
// any failure (socket closed, unknown cvar, parse error). The result
// is the string value, normalised to lowercase so the toolbar can
// compare against its constants without worrying about
// '1' vs '0' vs 'on' vs 'off' for the snap cvar.
async function getCvarString(
  client: WebSocketClient,
  name: string,
): Promise<string | undefined> {
  try {
    const r = await client.send({ type: 'get_cvar', name });
    if (!r.ok || !r.cvar) return undefined;
    return String(r.cvar.value).trim().toLowerCase();
  } catch {
    return undefined;
  }
}

function parseBool(v: string | undefined): boolean {
  if (!v) return false;
  return v === '1' || v === 'true' || v === 'on' || v === 'yes';
}

function ToolbarBody({ client }: ToolbarBodyProps) {
  const [mode, setMode] = useState<GizmoMode>('translate');
  const [snap, setSnap] = useState<boolean>(false);
  const [space, setSpace] = useState<GizmoSpace>('world');
  const [lastShortcut, setLastShortcut] = useState<string>('');

  // Pull the current cvar values whenever the WS reconnects or the
  // user fires a control. Each fetch is independent so a single
  // failure (e.g. gizmo_snap missing on an older engine) doesn't
  // wipe the other two.
  const refresh = useCallback(async () => {
    const [m, sn, sp] = await Promise.all([
      getCvarString(client, 'gizmo_mode'),
      getCvarString(client, 'gizmo_snap'),
      getCvarString(client, 'gizmo_space'),
    ]);
    if (m === 'translate' || m === 'rotate' || m === 'scale') setMode(m);
    setSnap(parseBool(sn));
    if (sp === 'world' || sp === 'local') setSpace(sp);
  }, [client]);

  useEffect(() => {
    // Initial fetch on mount, plus periodic re-poll so an external
    // change (engine GLFW G/R/S handler, another panel, console
    // input) reflects here without manual refresh. 1s cadence is
    // cheap on the WS and beats wiring a per-cvar event topic.
    let cancelled = false;
    void refresh();
    const t = window.setInterval(() => {
      if (!cancelled) void refresh();
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [refresh]);

  // ---- handlers ----
  const setGizmoMode = useCallback(
    (m: GizmoMode) => {
      setMode(m);  // optimistic; refresh() corrects on next tick
      void client.exec(`gizmo_mode ${m}`).catch(() => { /* engine offline */ });
    },
    [client],
  );

  const undo = useCallback(() => {
    void client.exec('scene_undo').catch(() => { /* engine offline */ });
    setLastShortcut('undo');
    window.setTimeout(() => setLastShortcut(''), 600);
  }, [client]);

  const redo = useCallback(() => {
    void client.exec('scene_redo').catch(() => { /* engine offline */ });
    setLastShortcut('redo');
    window.setTimeout(() => setLastShortcut(''), 600);
  }, [client]);

  const toggleSnap = useCallback(() => {
    const next = !snap;
    setSnap(next);
    void client.exec(`gizmo_snap ${next ? 1 : 0}`).catch(() => { /* engine offline */ });
  }, [client, snap]);

  const setGizmoSpace = useCallback(
    (sp: GizmoSpace) => {
      setSpace(sp);
      void client.exec(`gizmo_space ${sp}`).catch(() => { /* engine offline */ });
    },
    [client],
  );

  // Local visual flash hint when the global Cmd+Z keyboard
  // shortcut fires. Shell installs the handler globally; the
  // toolbar listens for the same event so the relevant button
  // blinks. Cleaner than threading a callback through Shell.
  useEffect(() => {
    const flash = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      const k = e.key.toLowerCase();
      if (meta && k === 'z') {
        setLastShortcut(e.shiftKey ? 'redo' : 'undo');
        window.setTimeout(() => setLastShortcut(''), 600);
      } else if (meta && k === 'y' && !e.shiftKey) {
        setLastShortcut('redo');
        window.setTimeout(() => setLastShortcut(''), 600);
      }
    };
    window.addEventListener('keydown', flash);
    return () => window.removeEventListener('keydown', flash);
  }, []);

  return (
    <Toolbar>
      <ToolbarGroup>
        <ToolbarButton
          label="Translate"
          active={mode === 'translate'}
          onClick={() => setGizmoMode('translate')}
          title="Translate gizmo (G) -- prim_set_pos / light_set_pos"
        />
        <ToolbarButton
          label="Rotate"
          active={mode === 'rotate'}
          onClick={() => setGizmoMode('rotate')}
          title="Rotate gizmo (R) -- prim_set_rotation"
        />
        <ToolbarButton
          label="Scale"
          active={mode === 'scale'}
          onClick={() => setGizmoMode('scale')}
          title="Scale gizmo (S) -- prim_set_radius for spheres"
        />
      </ToolbarGroup>
      <ToolbarGroup>
        <ToolbarButton
          label="Undo"
          onClick={undo}
          active={lastShortcut === 'undo'}
          title="Undo (Cmd/Ctrl+Z) -- pop the most recent scene mutation"
        />
        <ToolbarButton
          label="Redo"
          onClick={redo}
          active={lastShortcut === 'redo'}
          title="Redo (Cmd/Ctrl+Shift+Z, Ctrl+Y) -- reapply scene mutation"
        />
      </ToolbarGroup>
      <ToolbarGroup>
        <ToolbarButton
          label={snap ? 'Snap: On' : 'Snap: Off'}
          active={snap}
          onClick={toggleSnap}
          title="Snap drags to nearest grid step (cvar gizmo_snap)"
        />
        <ToolbarButton
          label="World"
          active={space === 'world'}
          onClick={() => setGizmoSpace('world')}
          title="World-space gizmo (cvar gizmo_space)"
        />
        <ToolbarButton
          label="Local"
          active={space === 'local'}
          onClick={() => setGizmoSpace('local')}
          title="Local-space gizmo (cvar gizmo_space). Object frame; reserved for rotation-aware mutators."
        />
      </ToolbarGroup>
    </Toolbar>
  );
}

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Toolbar"
      withClient={(client) => <ToolbarBody client={client} />}
    />
  </StrictMode>,
);

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// The editor panel shell. Every panel under web/editor/panels/* wraps
// its content in <Shell>: the wrapper takes care of:
//   - constructing a long-lived WebSocketClient pointed at /ws
//   - subscribing to selection_change + scene_dirty + log topics
//   - mirroring those into the SceneStore so panel components can
//     `useSceneStore(s => s.selection)` without ever touching the
//     WebSocket directly
//   - fetching the initial scene snapshot via `list_scene`
//   - re-fetching the snapshot every time `scene_dirty` ticks
//   - rendering the titlebar (with connection status) + statusbar
//   - exposing the client via a render-prop / context for the few
//     spots that need to send commands
//
// Downstream agents (21 / 22 / ...) just write the panel's body and
// drop it as Shell children:
//
//   <Shell title="Scene Hierarchy">
//     <SceneHierarchyTree />
//   </Shell>

import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import { useSceneStore, bindClientToStore } from '../store';
import { WebSocketClient } from '../ws-client';
import { wsEndpoint } from '../endpoint';
import { installEditorShortcuts } from '../keyboard';
import { TitleBar } from './TitleBar';
import { StatusBar, StatusCell } from './StatusBar';

export interface ShellProps {
  title: string;
  // Topics to subscribe to in addition to the defaults
  // (log + selection_change + scene_dirty). Useful for a future
  // perf-overlay panel that wants the existing `frame_stats` topic.
  extraTopics?: string[];
  // Render-prop access to the WS client for the small set of
  // panels that need to issue commands (e.g. the toolbar). Most
  // panels should reach for `useSceneStore(...)` instead.
  withClient?: (client: WebSocketClient) => ReactNode;
  children?: ReactNode;
  // When true, install the global keyboard shortcut handler
  // (Cmd/Ctrl+Z = scene_undo, Cmd/Ctrl+Shift+Z = scene_redo,
  // G / R / S = gizmo_mode). Default true; set to false if a panel
  // wants to install its own bindings without the shell's
  // interference.
  installShortcuts?: boolean;
}

export function Shell({
  title,
  extraTopics = [],
  withClient,
  children,
  installShortcuts = true,
}: ShellProps) {
  // The WebSocket client is created exactly once per panel. We hold
  // it in a ref so React strict mode's double-mount during dev
  // doesn't open + close + reopen the socket on every reload.
  const clientRef = useRef<WebSocketClient | null>(null);

  if (clientRef.current == null) {
    clientRef.current = new WebSocketClient({
      url: wsEndpoint(),
      topics: ['log', 'selection_change', 'scene_dirty', ...extraTopics],
    });
  }

  useEffect(() => {
    const client = clientRef.current!;
    const unbind = bindClientToStore(client);
    client.start();

    // Initial scene fetch. The engine may not be up yet (Chrome --app
    // can outlive demont); the request will reject, and the next
    // scene_dirty event re-triggers the fetch once the socket
    // reconnects.
    const fetchScene = async () => {
      try {
        const r = await client.listScene();
        if (r.ok && r.scene && typeof r.scene === 'object') {
          useSceneStore.getState().setScene(r.scene);
        }
      } catch {
        // Socket not open yet, or the engine doesn't support
        // list_scene in this build. Silently ignore -- store
        // already has scene=null.
      }
    };
    void fetchScene();

    // Re-fetch on scene_dirty. We piggy-back on bindClientToStore's
    // counter (it bumps on the event); a separate listener here keeps
    // the fetch behaviour out of the store.
    const off = client.onEvent('scene_dirty', () => {
      void fetchScene();
    });

    // Re-fetch whenever the socket (re)opens. Without this a panel
    // that outlives an engine restart -- or loses the connection --
    // shows the stale pre-disconnect snapshot until the next scene
    // mutation happens to tick scene_dirty.
    const offStatus = client.onStatus((s) => {
      if (s === 'open') void fetchScene();
    });

    // Global keyboard shortcuts (wave-7 #20): undo/redo + gizmo mode.
    // Installed on every panel so a user with the inspector focused
    // can still Cmd+Z to undo a hierarchy mutation. Opt-out via the
    // installShortcuts prop.
    const uninstallShortcuts = installShortcuts
      ? installEditorShortcuts(client)
      : () => { /* noop */ };

    return () => {
      uninstallShortcuts();
      offStatus();
      off();
      unbind();
      client.close();
      clientRef.current = null;
    };
    // The clientRef is intentionally stable; we don't want the effect
    // to re-run when extraTopics changes (we'd lose subscriptions).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const status = useSceneStore((s) => s.connectionStatus);
  const objects = useSceneStore((s) => s.objects);
  const selection = useSceneStore((s) => s.selection);
  const dirtyTick = useSceneStore((s) => s.sceneDirtyCounter);

  const childContent = useMemo(() => {
    if (withClient && clientRef.current) {
      return withClient(clientRef.current);
    }
    return children;
  }, [withClient, children]);

  return (
    <div className="editor-shell">
      <TitleBar title={title} status={status} />
      <main className="editor-body">{childContent}</main>
      <StatusBar>
        <StatusCell label="objects" value={objects.length} />
        <StatusCell
          label="selection"
          value={selection ? `${selection.kind} #${selection.id}` : 'none'}
        />
        <StatusCell label="dirty" value={dirtyTick} />
      </StatusBar>
    </div>
  );
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Scene Hierarchy panel entry point. The Shell wraps our content,
// passes the long-lived WebSocketClient via the `withClient` render
// prop, and takes care of the titlebar / statusbar + initial
// list_scene fetch. The actual tree component is in
// ./SceneHierarchy.tsx; everything here is plumbing.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './scene-hierarchy.css';
import { SceneHierarchy } from './SceneHierarchy';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Scene Hierarchy"
      withClient={(client) => <SceneHierarchy client={client} />}
    />
  </StrictMode>,
);

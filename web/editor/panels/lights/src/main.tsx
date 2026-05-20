// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Lights panel entry point. The Shell wraps our content, passes the
// long-lived WebSocketClient via the `withClient` render prop, and
// handles the titlebar / statusbar + initial list_scene fetch. The
// list + inspector live in ./LightsPanel.tsx; this is plumbing.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './lights.css';
import { LightsPanel } from './LightsPanel';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Lights"
      withClient={(client) => <LightsPanel client={client} />}
    />
  </StrictMode>,
);

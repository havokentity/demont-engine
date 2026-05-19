// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Property Inspector panel entry. The Shell sets up the long-lived
// WebSocket client, scene-store wiring, and selection-change
// subscriptions; this file just instantiates it and passes the client
// down to <App> via the withClient render prop.
//
// Implementation:
//   - main.tsx (here) -- React root + Shell
//   - App.tsx         -- selection/scene watcher, action dispatch
//   - PrimInspector   -- analytic-primitive property grid
//   - LightInspector  -- analytic-light property grid
//   - NumberField     -- scrub-drag numeric input
//   - Slider          -- slider + numeric readout combo
//   - ColorField      -- sRGB <-> linear color picker with HSV plane
//   - inspector.css   -- panel-local styles (uses theme.css vars)

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './inspector.css';
import { App } from './App';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell title="Inspector" withClient={(client) => <App client={client} />} />
  </StrictMode>,
);

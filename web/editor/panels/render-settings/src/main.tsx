// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Render Settings panel entry. The Shell sets up the long-lived
// WebSocket client, scene-store wiring, and connection status; this file
// instantiates it and passes the client down to <RenderSettings> via the
// withClient render prop.
//
// Implementation:
//   - main.tsx (here)   -- React root + Shell
//   - RenderSettings    -- collapsible sections bound to render cvars
//   - Controls          -- Section + CvarToggle / CvarSelect / CvarSlider
//   - useCvars          -- bulk get_cvar read + throttled set_cvar write
//   - Slider            -- slider + numeric readout combo (copied)
//   - NumberField       -- scrub-drag numeric input (copied)
//   - helpers.ts        -- cvar descriptor table + pure value coercion
//   - render-settings.css -- panel-local styles (uses theme.css vars)

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './render-settings.css';
import { RenderSettings } from './RenderSettings';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Render Settings"
      withClient={(client) => <RenderSettings client={client} />}
    />
  </StrictMode>,
);

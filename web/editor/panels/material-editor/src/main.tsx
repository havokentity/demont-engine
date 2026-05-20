// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Material Editor panel entry. The Shell sets up the long-lived
// WebSocket client, scene-store wiring, and selection-change
// subscriptions; this file instantiates it and passes the client down
// to <MaterialEditor> via the withClient render prop.
//
// Implementation:
//   - main.tsx (here)   -- React root + Shell
//   - MaterialEditor    -- selection/scene watcher, PBR material grid
//   - TexturePicker     -- per-slot texture assign / clear dropdown
//   - ColorField        -- sRGB <-> linear color picker (copied from
//                          the inspector; panel-local component)
//   - Slider            -- slider + numeric readout combo (copied)
//   - NumberField       -- scrub-drag numeric input (copied)
//   - helpers.ts        -- pure command-line builders + emission math
//   - material-editor.css -- panel-local styles (uses theme.css vars)

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './material-editor.css';
import { MaterialEditor } from './MaterialEditor';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Material Editor"
      withClient={(client) => <MaterialEditor client={client} />}
    />
  </StrictMode>,
);

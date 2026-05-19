// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell, Placeholder } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell title="Asset Browser">
      <Placeholder
        title="Asset Browser"
        description={
          <>
            Will surface the engine's <code>assets/</code> tree (HDRIs,
            BSC5 starmap, glTF imports, scene <code>.cfg</code> presets).
            Drag-drop into the scene hierarchy queues the matching
            console command (<code>scene_load</code>,{' '}
            <code>cubemap_load</code>, ...). Not in agent-20's scope.
          </>
        }
      />
    </Shell>
  </StrictMode>,
);

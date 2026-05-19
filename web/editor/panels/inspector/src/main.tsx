// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell, Placeholder } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell title="Inspector">
      <Placeholder
        title="Inspector"
        agent="agent-22"
        description={
          <>
            Property grid for the current selection. Subscribes to{' '}
            <code>selection_change</code>; on edit, dispatches{' '}
            <code>prim_set_pos / prim_set_albedo / prim_set_emission /
            prim_set_roughness / prim_set_ior / prim_set_material</code> (or
            the light equivalents) via WebSocket <code>exec</code>. The
            shell already wires selection state into the store -- agent-22
            replaces this placeholder with the property editor.
          </>
        }
      />
    </Shell>
  </StrictMode>,
);

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell, Placeholder } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell title="Scene Hierarchy">
      <Placeholder
        title="Scene Hierarchy"
        agent="agent-21"
        description={
          <>
            Tree of analytic prims, lights, SDF clusters, CSG roots, and smoke
            emitters. Click a node to call <code>select</code>; the engine
            highlights it in the path tracer and broadcasts{' '}
            <code>selection_change</code>. The shell is wired -- agent-21
            replaces this placeholder with the actual tree component.
          </>
        }
      />
    </Shell>
  </StrictMode>,
);

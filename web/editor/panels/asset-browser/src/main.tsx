// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Asset Browser panel entry. Mirrors the inspector / scene-hierarchy
// layout: ReactDOM root, <Shell> wrapper, the actual content lives
// in ./AssetBrowser.tsx so the entry stays one screen.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from '@demont/editor-shared';
import '@demont/editor-shared/theme.css';
import './asset-browser.css';
import { AssetBrowser } from './AssetBrowser';

const root = createRoot(document.getElementById('app')!);
root.render(
  <StrictMode>
    <Shell
      title="Asset Browser"
      withClient={(client) => <AssetBrowser client={client} />}
    />
  </StrictMode>,
);

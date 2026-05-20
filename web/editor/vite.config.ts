// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Multi-entry Vite config. Each panel under panels/* has its own
// index.html + src/main.tsx; the build emits one HTML per panel into
// dist/<panel>/. The engine's ConsoleServer routes
//   GET /editor/<panel>  ->  dist/<panel>/index.html (+ assets)
// so the panels share a single WebSocket endpoint and one Origin.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const panels = [
  'scene-hierarchy',
  'inspector',
  'asset-browser',
  'toolbar',
  'material-editor',
];

const input: Record<string, string> = {};
for (const p of panels) {
  input[p] = resolve(__dirname, 'panels', p, 'index.html');
}

export default defineConfig({
  plugins: [react()],

  // Each panel index.html lives at panels/<name>/index.html so the
  // dev server can serve it at /panels/<name>/. The Vite root is
  // the editor folder so all panel HTMLs are reachable.
  root: __dirname,

  resolve: {
    // Order matters: the more-specific `/theme.css` alias has to come
    // before the bare-package one, otherwise the bare alias rewrites
    // `@demont/editor-shared/theme.css` into
    // `shared/src/index.ts/theme.css` (a path inside a TS file).
    alias: [
      {
        find: /^@demont\/editor-shared\/theme\.css$/,
        replacement: resolve(__dirname, 'shared/src/theme.css'),
      },
      {
        find: '@demont/editor-shared',
        replacement: resolve(__dirname, 'shared/src/index.ts'),
      },
    ],
  },

  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input,
      output: {
        // Output layout the engine HTTP routes serve:
        //   dist/panels/<name>/index.html                <- entry HTML
        //   dist/panels/<name>/assets/<name>-<hash>.js   <- entry chunk
        //   dist/shared/assets/<name>-<hash>.js          <- shared chunks
        //   dist/shared/assets/<name>-<hash>.css         <- shared CSS
        //
        // The per-panel entry chunk lives under that panel's dir.
        // Shared chunks (theme, Placeholder, vendor) get hashed and
        // co-located under `shared/assets/` so they don't pollute the
        // panel namespace -- the engine just serves whatever URL it
        // sees against the embedded blob table.
        entryFileNames: 'panels/[name]/assets/[name]-[hash].js',
        chunkFileNames: 'shared/assets/[name]-[hash].js',
        assetFileNames: 'shared/assets/[name]-[hash][extname]',
      },
    },
  },

  server: {
    port: 5173,
    strictPort: true,
    // Useful while iterating: open Chrome at the index page that
    // links all four panels.
    open: false,
  },

  preview: {
    port: 5174,
    strictPort: true,
  },
});

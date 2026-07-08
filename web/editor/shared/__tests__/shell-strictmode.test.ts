// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// @vitest-environment happy-dom
//
// Regression test: React 18 StrictMode's dev-only setup -> cleanup ->
// setup effect cycle used to crash every panel at mount, because
// Shell's effect cleanup nulled clientRef.current -- the second setup
// then read a null client and bindClientToStore(null) threw. The fix
// keeps the client instance in the ref across the cycle (cleanup only
// close()s the socket; the second setup start()s it again). NODE_ENV
// under vitest is not 'production', so this exercises the same React
// development build the Vite dev server serves.

import { afterEach, describe, expect, it } from 'vitest';
import { StrictMode, act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Shell } from '../src/components/Shell';

// React's act() requires this global in non-Jest environments.
(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

describe('Shell under StrictMode', () => {
  let root: Root | null = null;
  let container: HTMLElement | null = null;

  afterEach(() => {
    if (root) act(() => root!.unmount());
    container?.remove();
    root = null;
    container = null;
  });

  it('survives the dev-mode double-mount effect cycle', () => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    // Before the fix this threw "Cannot read properties of null
    // (reading 'onStatus')" from the effect's second setup, and React
    // unmounted the whole tree leaving a blank panel.
    act(() => {
      root!.render(
        createElement(StrictMode, null, createElement(Shell, { title: 'Test' })),
      );
    });
    expect(container.querySelector('.editor-shell')).not.toBeNull();
  });
});

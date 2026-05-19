// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// @vitest-environment happy-dom
//
// Tests for the wave-7 #20 global editor shortcut handler. We mock
// the WebSocketClient surface (only .exec is consulted) and synthesise
// KeyboardEvents to verify each binding fires the matching console
// command -- and that text-entry focus suppresses the dispatch.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { installEditorShortcuts } from '../src/keyboard';
import type { WebSocketClient } from '../src/ws-client';

// Minimal stub: only .exec is consulted by the handler. Cast through
// `unknown` to avoid satisfying the entire WebSocketClient shape.
function makeClient() {
  const exec = vi.fn(() => Promise.resolve({ type: 'result' as const, ok: true }));
  const client = { exec } as unknown as WebSocketClient;
  return { client, exec };
}

// Synthesise a `keydown` matching the desired modifier set. The
// real DOM dispatch path goes through window.dispatchEvent so the
// installed listener picks it up.
function press(opts: {
  key: string;
  metaKey?: boolean;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  target?: Element;
}) {
  const e = new KeyboardEvent('keydown', {
    key: opts.key,
    metaKey: !!opts.metaKey,
    ctrlKey: !!opts.ctrlKey,
    shiftKey: !!opts.shiftKey,
    altKey: !!opts.altKey,
    bubbles: true,
    cancelable: true,
  });
  if (opts.target) {
    opts.target.dispatchEvent(e);
  } else {
    window.dispatchEvent(e);
  }
  return e;
}

describe('installEditorShortcuts', () => {
  let cleanup: () => void;

  afterEach(() => {
    cleanup?.();
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Cmd+Z dispatches scene_undo', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    press({ key: 'z', metaKey: true });
    expect(exec).toHaveBeenCalledWith('scene_undo');
  });

  it('Ctrl+Z also dispatches scene_undo', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    press({ key: 'z', ctrlKey: true });
    expect(exec).toHaveBeenCalledWith('scene_undo');
  });

  it('Cmd+Shift+Z dispatches scene_redo', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    press({ key: 'z', metaKey: true, shiftKey: true });
    expect(exec).toHaveBeenCalledWith('scene_redo');
  });

  it('Ctrl+Y dispatches scene_redo (Windows convention)', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    press({ key: 'y', ctrlKey: true });
    expect(exec).toHaveBeenCalledWith('scene_redo');
  });

  it('bare G/R/S dispatch gizmo_mode changes', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    press({ key: 'g' });
    press({ key: 'r' });
    press({ key: 's' });
    expect(exec).toHaveBeenCalledWith('gizmo_mode translate');
    expect(exec).toHaveBeenCalledWith('gizmo_mode rotate');
    expect(exec).toHaveBeenCalledWith('gizmo_mode scale');
  });

  it('shortcuts ignored when focus is in <input>', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    const input = document.createElement('input');
    document.body.appendChild(input);
    try {
      press({ key: 'z', metaKey: true, target: input });
      press({ key: 'g', target: input });
      expect(exec).not.toHaveBeenCalled();
    } finally {
      input.remove();
    }
  });

  it('shortcuts ignored when focus is in <textarea>', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    const ta = document.createElement('textarea');
    document.body.appendChild(ta);
    try {
      press({ key: 'g', target: ta });
      expect(exec).not.toHaveBeenCalled();
    } finally {
      ta.remove();
    }
  });

  it('shortcuts ignored on contenteditable element', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client);
    const div = document.createElement('div');
    div.contentEditable = 'true';
    document.body.appendChild(div);
    try {
      press({ key: 'z', metaKey: true, target: div });
      expect(exec).not.toHaveBeenCalled();
    } finally {
      div.remove();
    }
  });

  it('gizmoShortcuts: false suppresses G/R/S but keeps undo/redo', () => {
    const { client, exec } = makeClient();
    cleanup = installEditorShortcuts(client, { gizmoShortcuts: false });
    press({ key: 'g' });
    press({ key: 'r' });
    press({ key: 's' });
    expect(exec).not.toHaveBeenCalled();

    press({ key: 'z', metaKey: true });
    expect(exec).toHaveBeenCalledWith('scene_undo');
  });

  it('onAction callback fires for each binding', () => {
    const { client } = makeClient();
    const onAction = vi.fn();
    cleanup = installEditorShortcuts(client, { onAction });

    press({ key: 'z', metaKey: true });
    expect(onAction).toHaveBeenCalledWith('scene_undo');

    press({ key: 'z', metaKey: true, shiftKey: true });
    expect(onAction).toHaveBeenCalledWith('scene_redo');

    press({ key: 'g' });
    expect(onAction).toHaveBeenCalledWith('gizmo_translate');
  });

  it('returned cleanup function removes the listener', () => {
    const { client, exec } = makeClient();
    const off = installEditorShortcuts(client);
    off();
    press({ key: 'z', metaKey: true });
    expect(exec).not.toHaveBeenCalled();
  });
});

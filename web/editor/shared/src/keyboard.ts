// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Cross-panel keyboard shortcut dispatcher. Every editor panel runs in
// its own Chrome --app window with its own focused document, so each
// panel installs its own listener; this helper keeps the binding
// table consistent across panels.
//
// The wave-7 polish ships a small set of global shortcuts that should
// fire regardless of which panel has focus:
//
//   Cmd/Ctrl+Z        -> scene_undo
//   Cmd/Ctrl+Shift+Z  -> scene_redo
//   Ctrl+Y            -> scene_redo  (Windows convention; alongside the Cmd+Shift+Z one)
//   G / R / S         -> gizmo_mode translate / rotate / scale  (matches the existing engine hotkeys)
//
// Shortcuts only fire when no text-entry widget has focus -- a panel
// with an active <input> / <textarea> / contenteditable element keeps
// its native behaviour (typing Z into a number field shouldn't undo
// the scene).

import type { WebSocketClient } from './ws-client';

/** Identifier for the shortcut that fired -- the panel may want to
 *  flash an indicator on the matching toolbar button. */
export type ShortcutAction =
  | 'scene_undo'
  | 'scene_redo'
  | 'gizmo_translate'
  | 'gizmo_rotate'
  | 'gizmo_scale';

export interface InstallShortcutsOpts {
  /** Receives the action that fired, after the engine command is dispatched.
   *  Useful for visual flash / status-bar feedback. */
  onAction?: (action: ShortcutAction) => void;
  /** When true, gizmo G/R/S shortcuts are also installed. The
   *  scene-hierarchy + asset-browser leave these off because the
   *  engine's GLFW key handler already fires them when the main
   *  viewport window is focused -- only the React panels duplicate
   *  the binding so users can press G/R/S inside an editor panel
   *  too. Default true. */
  gizmoShortcuts?: boolean;
}

/** Returns true if the event target is a text-entry control where we
 *  shouldn't intercept printable / undo keystrokes. */
function isEditingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (t.isContentEditable) return true;
  return false;
}

/** Install the wave-7 global shortcuts on `window`. Returns a cleanup
 *  function that removes the listener. */
export function installEditorShortcuts(
  client: WebSocketClient,
  opts: InstallShortcutsOpts = {},
): () => void {
  const { onAction, gizmoShortcuts = true } = opts;

  const handler = (e: KeyboardEvent) => {
    if (isEditingTarget(e.target)) return;

    const meta = e.metaKey || e.ctrlKey;
    // Lower-case the printable key so we don't have to test both
    // 'z' and 'Z' (Shift uppercases the key on Mac).
    const k = e.key.toLowerCase();

    // Cmd/Ctrl+Z         -> undo
    // Cmd/Ctrl+Shift+Z   -> redo (also Ctrl+Y on Windows)
    if (meta && k === 'z') {
      e.preventDefault();
      if (e.shiftKey) {
        void client.exec('scene_redo');
        onAction?.('scene_redo');
      } else {
        void client.exec('scene_undo');
        onAction?.('scene_undo');
      }
      return;
    }
    if (meta && k === 'y' && !e.shiftKey) {
      // Windows-style redo. macOS rarely uses Ctrl+Y for redo but
      // it doesn't hurt to also bind it -- the meta-key gate filters
      // out plain 'y' typing in non-editing contexts.
      e.preventDefault();
      void client.exec('scene_redo');
      onAction?.('scene_redo');
      return;
    }

    // Gizmo mode shortcuts -- match the engine's GLFW handler.
    // These don't take meta; bare G / R / S press.
    if (gizmoShortcuts && !meta && !e.altKey && !e.shiftKey) {
      if (k === 'g') {
        void client.exec('gizmo_mode translate');
        onAction?.('gizmo_translate');
        return;
      }
      if (k === 'r') {
        void client.exec('gizmo_mode rotate');
        onAction?.('gizmo_rotate');
        return;
      }
      if (k === 's') {
        void client.exec('gizmo_mode scale');
        onAction?.('gizmo_scale');
        return;
      }
    }
  };

  window.addEventListener('keydown', handler);
  return () => {
    window.removeEventListener('keydown', handler);
  };
}

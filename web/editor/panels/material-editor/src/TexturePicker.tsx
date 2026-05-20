// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// A single PBR texture-map slot row: shows the currently-assigned
// texture (filename), an Assign dropdown listing assets/textures/, and
// a Clear button. Assigning dispatches the engine's prim_set_<slot>_tex
// command; clearing fires prim_clear_tex (which clears ALL maps -- the
// engine has no per-slot clear, so the parent re-applies the survivors;
// see MaterialEditor.handleClearTex).
//
// The asset list is fetched once on first open (lazy) via
// `list_assets textures`; the parent passes the shared list down so all
// four slots share one fetch.

import { useCallback, useEffect, useRef, useState } from 'react';
import { basename, type TexSlot } from './helpers';

export interface TexturePickerProps {
  slot: TexSlot;
  // Currently-assigned source path ('' = flat / no texture).
  path: string;
  // Available texture paths under assets/textures/ (parent-fetched).
  available: string[];
  loading: boolean;
  // Assign a texture path to this slot.
  onAssign: (slot: TexSlot, path: string) => void;
  // Clear this slot's texture.
  onClear: (slot: TexSlot) => void;
  // Ask the parent to (re)fetch the texture list.
  onRefresh: () => void;
}

export function TexturePicker({
  slot,
  path,
  available,
  loading,
  onAssign,
  onClear,
  onRefresh,
}: TexturePickerProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const assigned = path.length > 0;

  // Fetch the list the first time the dropdown opens.
  const openMenu = useCallback(() => {
    if (!open && available.length === 0 && !loading) onRefresh();
    setOpen((o) => !o);
  }, [open, available.length, loading, onRefresh]);

  // Click-outside dismissal.
  useEffect(() => {
    if (!open) return;
    const onDocDown = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && rootRef.current && rootRef.current.contains(t)) return;
      setOpen(false);
    };
    document.addEventListener('pointerdown', onDocDown, true);
    return () => document.removeEventListener('pointerdown', onDocDown, true);
  }, [open]);

  const pick = useCallback(
    (p: string) => {
      onAssign(slot, p);
      setOpen(false);
    },
    [onAssign, slot],
  );

  return (
    <div className="mtl-tex" ref={rootRef}>
      <button
        type="button"
        className={`mtl-tex-name${assigned ? ' is-assigned' : ''}`}
        onClick={openMenu}
        title={assigned ? path : 'No texture -- click to assign'}
      >
        <span className="mtl-tex-label">
          {assigned ? basename(path) : '(none)'}
        </span>
        <span className="mtl-tex-caret">{open ? '▴' : '▾'}</span>
      </button>
      <button
        type="button"
        className="mtl-tex-clear"
        onClick={() => onClear(slot)}
        disabled={!assigned}
        title="Clear this texture map"
      >
        {'✕'}
      </button>
      {open && (
        <div className="mtl-tex-menu">
          <div className="mtl-tex-menu-head">
            <span className="dim">assets/textures/</span>
            <button
              type="button"
              className="mtl-tex-refresh"
              onClick={onRefresh}
              title="Re-read assets/textures/ from disk"
            >
              {'↻'}
            </button>
          </div>
          {loading && <div className="mtl-tex-menu-empty dim">loading...</div>}
          {!loading && available.length === 0 && (
            <div className="mtl-tex-menu-empty dim">
              No images under assets/textures/ (png / jpg / tga / bmp).
            </div>
          )}
          {!loading &&
            available.map((p) => (
              <button
                type="button"
                key={p}
                className={`mtl-tex-opt${p === path ? ' is-current' : ''}`}
                onClick={() => pick(p)}
                title={p}
              >
                {basename(p)}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

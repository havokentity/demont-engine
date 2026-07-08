// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Material Editor root. Watches the shared store's selection + scene
// snapshot; when a primitive is selected it renders a rich PBR material
// grid:
//
//   - Material type (lambert / metal / dielectric / water)
//   - Albedo color picker + albedo texture assign
//   - Roughness slider + roughness texture
//   - Metallic texture (analytic prims pick metal via the type selector;
//     the metallic map modulates the metal F0 path -- there is no flat
//     metallic scalar in the engine, so we surface the map + a hint)
//   - Normal map assign (engine has no normal-strength uniform yet, so
//     the strength control is deferred -- assigning/clearing the map is
//     the supported path)
//   - Emission color + intensity (intensity is a client-side multiplier
//     over the base color; the product is written via prim_set_emission)
//   - IOR (dielectric / water)
//   - Live preview swatch
//
// All material state is READ from list_scene (the SceneGraph snapshot,
// which as of wave-9 emits albedo/roughness/ior/emission/material plus
// per-slot texture flags + paths). All edits are WRITTEN through the
// existing prim_set_* / prim_clear_tex console commands over the WS
// exec path -- the engine emits scene_dirty -> Shell refetches -> our
// props update, so the UI stays live without optimistic store writes.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  useSceneStore,
  type WebSocketClient,
  type SceneSnapshot,
} from '@demont/editor-shared';
import { ColorField } from './ColorField';
import { Slider } from './Slider';
import { NumberField } from './NumberField';
import { TexturePicker } from './TexturePicker';
import { AdvancedMaterial } from './AdvancedMaterial';
import {
  fmt,
  rgbCss,
  decomposeEmission,
  composeEmission,
  MATERIAL_TYPES,
  quoteArg,
  setTexCommand,
  type TexSlot,
  type MaterialType,
} from './helpers';

interface MaterialEditorProps {
  client: WebSocketClient;
}

// Raw primitive record as SceneGraph.cpp serialises it. Texture fields
// (booleans + *_path strings) were added in wave-9; older engines omit
// them, so they're optional and we default to "flat".
interface RawPrim {
  id: number;
  type: 'sphere' | 'plane';
  material: MaterialType;
  albedo: [number, number, number];
  emission: [number, number, number];
  roughness: number;
  ior: number;
  albedo_tex?: boolean;
  normal_tex?: boolean;
  roughness_tex?: boolean;
  metallic_tex?: boolean;
  albedo_tex_path?: string;
  normal_tex_path?: string;
  roughness_tex_path?: string;
  metallic_tex_path?: string;
}

function findPrim(snap: SceneSnapshot | null, id: number): RawPrim | undefined {
  if (!snap) return undefined;
  const arr =
    (snap as Record<string, unknown>).primitives ??
    (snap as Record<string, unknown>).prims;
  if (!Array.isArray(arr)) return undefined;
  for (const p of arr as RawPrim[]) {
    if (p && p.id === id) return p;
  }
  return undefined;
}

// Throttled exec: bucket by command prefix so independent commands
// (prim_set_albedo vs prim_set_roughness) don't displace each other;
// the trailing call wins within a 50ms window. Mirrors the inspector's
// useThrottledExec (App.tsx) so drag behaviour feels identical.
const kThrottleMs = 50;

function useThrottledExec(client: WebSocketClient): (line: string) => void {
  const pendingRef = useRef<Map<string, { line: string; timer: number }>>(
    new Map(),
  );
  const lastFiredRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const pending = pendingRef.current;
    return () => {
      for (const v of pending.values()) {
        if (v.timer) window.clearTimeout(v.timer);
      }
      pending.clear();
    };
  }, []);

  return useCallback(
    (line: string) => {
      const bucket = line.split(/\s+/)[0] ?? line;
      const now = performance.now();
      const last = lastFiredRef.current.get(bucket) ?? 0;
      const elapsed = now - last;
      const existing = pendingRef.current.get(bucket);
      if (existing) window.clearTimeout(existing.timer);

      const fire = (l: string) => {
        lastFiredRef.current.set(bucket, performance.now());
        pendingRef.current.delete(bucket);
        void client.exec(l).catch(() => { /* offline; retry on next edit */ });
      };

      if (elapsed >= kThrottleMs) {
        fire(line);
      } else {
        const wait = kThrottleMs - elapsed;
        const timer = window.setTimeout(() => fire(line), wait);
        pendingRef.current.set(bucket, { line, timer });
      }
    },
    [client],
  );
}

function typeBadge(type: 'sphere' | 'plane'): string {
  return type === 'sphere' ? 'Sphere' : 'Plane';
}

export function MaterialEditor({ client }: MaterialEditorProps) {
  const selection = useSceneStore((s) => s.selection);
  const scene = useSceneStore((s) => s.scene);
  const dirtyTick = useSceneStore((s) => s.sceneDirtyCounter);

  const exec = useThrottledExec(client);

  // Shared assets/textures/ list -- fetched lazily, shared by all four
  // texture-map slots so we don't fire four identical list_assets.
  const [textures, setTextures] = useState<string[]>([]);
  const [texLoading, setTexLoading] = useState(false);

  const refreshTextures = useCallback(async () => {
    setTexLoading(true);
    try {
      const r = await client.exec('list_assets textures');
      const out = String(r.output ?? '');
      const paths = out
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith('(') && !l.startsWith('error:'));
      setTextures(paths);
    } catch {
      // Engine offline; leave the list as-is.
    } finally {
      setTexLoading(false);
    }
  }, [client]);

  const prim = useMemo<RawPrim | undefined>(() => {
    if (!selection || selection.kind !== 'prim') return undefined;
    return findPrim(scene, selection.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.kind, selection?.id, scene, dirtyTick]);

  const id = prim?.id ?? -1;

  // ---- Material-param setters (existing console commands) -------------
  const setMaterial = useCallback(
    (m: MaterialType) => exec(`prim_set_material ${id} ${m}`),
    [exec, id],
  );
  const setAlbedo = useCallback(
    (rgb: [number, number, number]) =>
      exec(`prim_set_albedo ${id} ${fmt(rgb[0])} ${fmt(rgb[1])} ${fmt(rgb[2])}`),
    [exec, id],
  );
  const setRoughness = useCallback(
    (v: number) => exec(`prim_set_roughness ${id} ${fmt(v)}`),
    [exec, id],
  );
  const setIor = useCallback(
    (v: number) => exec(`prim_set_ior ${id} ${fmt(v)}`),
    [exec, id],
  );
  const setEmissionRgb = useCallback(
    (rgb: [number, number, number]) =>
      exec(`prim_set_emission ${id} ${fmt(rgb[0])} ${fmt(rgb[1])} ${fmt(rgb[2])}`),
    [exec, id],
  );

  // ---- Emission color + intensity (client-side decomposition) ---------
  const emission = prim?.emission ?? [0, 0, 0];
  const { color: emColor, intensity: emIntensity } = useMemo(
    () => decomposeEmission(emission),
    [emission],
  );
  const setEmissionColor = useCallback(
    (color: [number, number, number]) => {
      // Preserve the current intensity; if emission was zero, default
      // the intensity to 1 so picking a color actually emits.
      const k = emIntensity > 0 ? emIntensity : 1;
      setEmissionRgb(composeEmission(color, k));
    },
    [emIntensity, setEmissionRgb],
  );
  const setEmissionIntensity = useCallback(
    (k: number) => setEmissionRgb(composeEmission(emColor, k)),
    [emColor, setEmissionRgb],
  );

  // ---- Texture-map setters --------------------------------------------
  const handleAssignTex = useCallback(
    (slot: TexSlot, path: string) => {
      // Quote the path -- the console tokenizer splits unquoted args
      // on whitespace, truncating e.g. 'wood planks.png'.
      exec(`${setTexCommand(slot)} ${id} ${quoteArg(path)}`);
    },
    [exec, id],
  );

  // The engine has no per-slot clear -- prim_clear_tex wipes all four
  // maps. To clear just `slot`, wipe everything then re-assign the
  // survivors from the current snapshot. The re-assigns dedupe to the
  // same atlas tile host-side (pbr_tile_by_path_), so this is cheap.
  //
  // We bypass the throttled `exec` here and drive the raw client
  // directly, awaiting each command in order: this is a discrete button
  // click (not a drag), so throttling buys nothing, and serialising the
  // sends guarantees the clear lands before the survivor re-assigns
  // regardless of throttle bucketing.
  const handleClearTex = useCallback(
    async (slot: TexSlot) => {
      if (!prim) return;
      const survivors: Array<{ s: TexSlot; p: string }> = [];
      const keep = (s: TexSlot, p?: string) => {
        if (s !== slot && p) survivors.push({ s, p });
      };
      keep('albedo', prim.albedo_tex_path);
      keep('normal', prim.normal_tex_path);
      keep('roughness', prim.roughness_tex_path);
      keep('metallic', prim.metallic_tex_path);
      try {
        await client.exec(`prim_clear_tex ${id}`);
        for (const { s, p } of survivors) {
          await client.exec(`${setTexCommand(s)} ${id} ${quoteArg(p)}`);
        }
      } catch {
        // Engine offline; the next scene_dirty refetch reconciles state.
      }
    },
    [client, id, prim],
  );

  // ---- Render ----------------------------------------------------------

  // No selection -- the store maps the engine's {kind:'none'} payload
  // to null (selectionFromPayload in shared/src/store.ts).
  if (!selection) {
    return (
      <div className="mtl-empty">
        <h3>No material selected.</h3>
        <p>
          Select a primitive in the viewport or <code>Scene Hierarchy</code>{' '}
          to edit its PBR material.
        </p>
      </div>
    );
  }

  if (selection.kind !== 'prim') {
    return (
      <div className="mtl-empty">
        <h3>{selection.kind} #{selection.id}</h3>
        <p>
          The Material Editor edits analytic primitives. Selection kind{' '}
          <code>{selection.kind}</code> doesn't carry an editable PBR
          material.
        </p>
      </div>
    );
  }

  if (!prim) {
    return (
      <div className="mtl-empty">
        <h3>prim #{selection.id}</h3>
        <p>Waiting for the scene snapshot to refresh...</p>
      </div>
    );
  }

  const swatchColor = rgbCss(prim.albedo) ?? 'rgb(128,128,128)';
  const emissiveCss = rgbCss(emission);
  const showIor = prim.material === 'dielectric' || prim.material === 'water';

  return (
    <div className="mtl-root">
      <header className="mtl-header">
        <div
          className="mtl-preview"
          style={{ background: swatchColor }}
          title="Live albedo preview (display sRGB)"
        >
          {emissiveCss && (
            <div
              className="mtl-preview-glow"
              style={{ boxShadow: `0 0 16px 4px ${emissiveCss}` }}
            />
          )}
        </div>
        <div className="mtl-header-meta">
          <span className="mtl-badge">{typeBadge(prim.type)}</span>
          <span className="mtl-id">#{prim.id}</span>
          <span className="mtl-matname">{prim.material}</span>
        </div>
      </header>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Surface</h4>
        <div className="mtl-row">
          <label>Type</label>
          <select
            className="mtl-select"
            value={prim.material}
            onChange={(e) => setMaterial(e.target.value as MaterialType)}
          >
            {MATERIAL_TYPES.map((m) => (
              <option key={m} value={m}>
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Albedo</h4>
        <div className="mtl-row">
          <label>Base Color</label>
          <ColorField value={prim.albedo} onCommit={setAlbedo} onScrub={setAlbedo} />
        </div>
        <div className="mtl-row">
          <label>Texture</label>
          <TexturePicker
            slot="albedo"
            path={prim.albedo_tex_path ?? ''}
            available={textures}
            loading={texLoading}
            onAssign={handleAssignTex}
            onClear={handleClearTex}
            onRefresh={() => void refreshTextures()}
          />
        </div>
      </section>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Roughness</h4>
        <div className="mtl-row">
          <label>Value</label>
          <Slider
            value={prim.roughness}
            min={0}
            max={1}
            step={0.001}
            onCommit={setRoughness}
            onScrub={setRoughness}
          />
        </div>
        <div className="mtl-row">
          <label>Texture</label>
          <TexturePicker
            slot="roughness"
            path={prim.roughness_tex_path ?? ''}
            available={textures}
            loading={texLoading}
            onAssign={handleAssignTex}
            onClear={handleClearTex}
            onRefresh={() => void refreshTextures()}
          />
        </div>
      </section>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Metallic</h4>
        <div className="mtl-row">
          <label>Texture</label>
          <TexturePicker
            slot="metallic"
            path={prim.metallic_tex_path ?? ''}
            available={textures}
            loading={texLoading}
            onAssign={handleAssignTex}
            onClear={handleClearTex}
            onRefresh={() => void refreshTextures()}
          />
        </div>
        <p className="mtl-hint">
          Analytic prims select metal via the <strong>Type</strong> selector;
          the metallic map modulates the metal F0 where white. There's no
          flat metallic scalar on analytic prims.
        </p>
      </section>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Normal Map</h4>
        <div className="mtl-row">
          <label>Texture</label>
          <TexturePicker
            slot="normal"
            path={prim.normal_tex_path ?? ''}
            available={textures}
            loading={texLoading}
            onAssign={handleAssignTex}
            onClear={handleClearTex}
            onRefresh={() => void refreshTextures()}
          />
        </div>
        <p className="mtl-hint">
          Tangent-space, linear ([0,1]→[−1,1]). Strength is fixed at 1.0
          (the engine has no normal-strength uniform yet).
        </p>
      </section>

      <section className="mtl-section">
        <h4 className="mtl-section-title">Emission</h4>
        <div className="mtl-row">
          <label>Color</label>
          <ColorField
            value={emColor}
            onCommit={setEmissionColor}
            onScrub={setEmissionColor}
          />
        </div>
        <div className="mtl-row">
          <label>Intensity</label>
          <NumberField
            value={emIntensity}
            min={0}
            precision={3}
            onCommit={setEmissionIntensity}
            onScrub={setEmissionIntensity}
            title="W/sr per channel (scales the base color)"
          />
        </div>
      </section>

      {showIor && (
        <section className="mtl-section">
          <h4 className="mtl-section-title">Refraction</h4>
          <div className="mtl-row">
            <label>IOR</label>
            <NumberField
              value={prim.ior}
              min={1.0}
              max={2.4}
              onCommit={setIor}
              onScrub={setIor}
            />
          </div>
        </section>
      )}

      <AdvancedMaterial
        id={prim.id}
        isMetal={prim.material === 'metal'}
        exec={exec}
      />
    </div>
  );
}

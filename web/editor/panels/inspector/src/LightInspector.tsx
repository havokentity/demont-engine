// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Property inspector for analytic lights. Each field commit dispatches
// an ATOMIC per-field setter (mirroring the prim_set_* family):
//
//   light_set_pos       <id> <x> <y> <z>             (any type)
//   light_set_intensity <id> <r> <g> <b>             (raw W/sr or W/m^2/sr)
//   light_set_dir       <id> <nx> <ny> <nz>          (spot axis / quad normal)
//   light_set_cone      <id> <inner_deg> <outer_deg> (spot only)
//   light_set_size      <id> <r>                     (sphere radius / quad v_half)
//   light_set_uhalf     <id> <r>                     (quad U half-extent)
//
// Each setter validates id + type, mutates the light in place, and
// broadcasts scene_dirty -- so the shell re-fetches via list_scene and
// the edited field round-trips. (The composite light_point / light_spot
// / ... constructors broadcast scene_dirty too, but firing a single
// field per commit avoids rebuilding the whole light from a
// possibly-stale snapshot.)
//
// Two engine quirks the dispatch respects:
//   - light_set_cone takes inner THEN outer -- the OPPOSITE argument
//     order of the composite light_spot constructor (outer then inner).
//   - a quad's two half-extents use different setters: light_set_uhalf
//     drives U (the engine rescales u_vec to the new length, preserving
//     its in-plane axis) while light_set_size drives V.
//
// Intensity is authored as raw radiometric RGB via the HDR ColorField,
// unlike the dedicated Lights panel which layers photometric cd / lm /
// nit / EV authoring on top of the same light_set_intensity write path.

import { useCallback } from 'react';
import { NumberField } from './NumberField';
import { Slider } from './Slider';
import { ColorField } from './ColorField';

export interface LightRecord {
  id: number;
  type: 'point' | 'spot' | 'sphere' | 'quad';
  pos: [number, number, number];
  intensity: [number, number, number];
  dir?: [number, number, number];
  cos_outer?: number;
  cos_inner?: number;
  radius?: number;
  u_vec?: [number, number, number];
  v_half?: number;
}

export interface LightInspectorProps {
  light: LightRecord;
  exec: (line: string) => void;
}

function fmt(n: number, prec = 4): string {
  if (!Number.isFinite(n)) return '0';
  return parseFloat(n.toFixed(prec)).toString();
}

function clampCos(c: number): number {
  if (!Number.isFinite(c)) return 1;
  if (c > 1) return 1;
  if (c < 0) return 0;
  return c;
}

// Cosine half-angle <-> degrees. The engine stores cos(half_angle)
// because the shader compares cos(theta) directly without the inverse
// trig; the editor exposes the more human-friendly degrees.
function cosToDeg(c: number): number {
  const cc = clampCos(c);
  return (Math.acos(cc) * 180) / Math.PI;
}

function vecLen(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

export function LightInspector({ light, exec }: LightInspectorProps) {
  const id = light.id;
  const pos = light.pos;
  const intensity = light.intensity;
  const dir = light.dir ?? ([0, -1, 0] as [number, number, number]);
  const outerDeg = cosToDeg(light.cos_outer ?? 1);
  const innerDeg = cosToDeg(light.cos_inner ?? 1);
  const sphereRadius = light.radius ?? 0.25;
  const u_vec = light.u_vec ?? ([1, 0, 0] as [number, number, number]);
  const v_half = light.v_half ?? 0.5;

  // ---- atomic per-field dispatch ----
  // Each setter receives the full new field value as args, so the
  // closures only depend on [exec, id] -- no full-state reconstruction
  // from a snapshot that may already be one scene_dirty behind.
  const emitPos = useCallback(
    (p: [number, number, number]) => {
      exec(`light_set_pos ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])}`);
    },
    [exec, id],
  );

  const emitDir = useCallback(
    (d: [number, number, number]) => {
      // light_set_dir auto-normalises and rejects zero-length; skip the
      // degenerate case so we don't fire a command the engine warns on.
      if (vecLen(d) < 1e-6) return;
      exec(`light_set_dir ${id} ${fmt(d[0])} ${fmt(d[1])} ${fmt(d[2])}`);
    },
    [exec, id],
  );

  const emitCone = useCallback(
    (innerD: number, outerD: number) => {
      // Engine requires 0 <= inner <= outer <= 90 and takes inner first.
      const o = Math.max(0, Math.min(90, outerD));
      const i = Math.max(0, Math.min(o, innerD));
      exec(`light_set_cone ${id} ${fmt(i, 2)} ${fmt(o, 2)}`);
    },
    [exec, id],
  );

  const emitSize = useCallback(
    (r: number) => {
      exec(`light_set_size ${id} ${fmt(Math.max(0, r))}`);
    },
    [exec, id],
  );

  const emitUHalf = useCallback(
    (r: number) => {
      // light_set_uhalf rescales u_vec to length r; the engine rejects
      // negatives and warns on a degenerate (zero-length) u-axis.
      exec(`light_set_uhalf ${id} ${fmt(Math.max(0, r))}`);
    },
    [exec, id],
  );

  const emitIntensity = useCallback(
    (rgb: [number, number, number]) => {
      // light_set_intensity rejects negative channels.
      const r = Math.max(0, rgb[0]);
      const g = Math.max(0, rgb[1]);
      const b = Math.max(0, rgb[2]);
      exec(`light_set_intensity ${id} ${fmt(r)} ${fmt(g)} ${fmt(b)}`);
    },
    [exec, id],
  );

  // The stored u_vec packs axis * half-extent, so its length is the U
  // half-extent shown here; emitUHalf writes it via light_set_uhalf.
  const uHalfDisplay = vecLen(u_vec);

  return (
    <>
      <section className="insp-section">
        <h4 className="insp-section-title">Transform</h4>
        <div className="insp-row">
          <label>Position</label>
          <div className="insp-vec3">
            <NumberField
              axis="X"
              value={pos[0]}
              onCommit={(v) => emitPos([v, pos[1], pos[2]])}
              onScrub={(v) => emitPos([v, pos[1], pos[2]])}
            />
            <NumberField
              axis="Y"
              value={pos[1]}
              onCommit={(v) => emitPos([pos[0], v, pos[2]])}
              onScrub={(v) => emitPos([pos[0], v, pos[2]])}
            />
            <NumberField
              axis="Z"
              value={pos[2]}
              onCommit={(v) => emitPos([pos[0], pos[1], v])}
              onScrub={(v) => emitPos([pos[0], pos[1], v])}
            />
          </div>
        </div>
        {(light.type === 'spot' || light.type === 'quad') && (
          <div className="insp-row">
            <label>{light.type === 'spot' ? 'Direction' : 'Normal'}</label>
            <div className="insp-vec3">
              <NumberField
                axis="X"
                value={dir[0]}
                onCommit={(v) => emitDir([v, dir[1], dir[2]])}
                onScrub={(v) => emitDir([v, dir[1], dir[2]])}
              />
              <NumberField
                axis="Y"
                value={dir[1]}
                onCommit={(v) => emitDir([dir[0], v, dir[2]])}
                onScrub={(v) => emitDir([dir[0], v, dir[2]])}
              />
              <NumberField
                axis="Z"
                value={dir[2]}
                onCommit={(v) => emitDir([dir[0], dir[1], v])}
                onScrub={(v) => emitDir([dir[0], dir[1], v])}
              />
            </div>
          </div>
        )}
        {light.type === 'sphere' && (
          <div className="insp-row">
            <label>Radius</label>
            <NumberField
              value={sphereRadius}
              min={0.0001}
              onCommit={(v) => emitSize(v)}
              onScrub={(v) => emitSize(v)}
            />
          </div>
        )}
        {light.type === 'quad' && (
          <>
            <div className="insp-row">
              <label>U half</label>
              <NumberField
                value={uHalfDisplay}
                min={0.0001}
                onCommit={(v) => emitUHalf(v)}
                onScrub={(v) => emitUHalf(v)}
              />
            </div>
            <div className="insp-row">
              <label>V half</label>
              <NumberField
                value={v_half}
                min={0.0001}
                onCommit={(v) => emitSize(v)}
                onScrub={(v) => emitSize(v)}
              />
            </div>
          </>
        )}
      </section>

      <section className="insp-section">
        <h4 className="insp-section-title">Emission</h4>
        <div className="insp-row">
          <label>Type</label>
          <span className="insp-badge is-readonly">{light.type}</span>
        </div>
        <div className="insp-row">
          <label>Intensity</label>
          <ColorField
            value={intensity}
            onCommit={(rgb) => emitIntensity(rgb)}
            onScrub={(rgb) => emitIntensity(rgb)}
            allowHdr
          />
        </div>
        {light.type === 'spot' && (
          <>
            <div className="insp-row">
              <label>Outer (deg)</label>
              <Slider
                value={outerDeg}
                min={0}
                max={89.99}
                step={0.1}
                precision={2}
                onCommit={(v) => emitCone(innerDeg, v)}
                onScrub={(v) => emitCone(innerDeg, v)}
              />
            </div>
            <div className="insp-row">
              <label>Inner (deg)</label>
              <Slider
                value={innerDeg}
                min={0}
                max={89.99}
                step={0.1}
                precision={2}
                onCommit={(v) => emitCone(v, outerDeg)}
                onScrub={(v) => emitCone(v, outerDeg)}
              />
            </div>
          </>
        )}
      </section>
    </>
  );
}

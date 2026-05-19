// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Property inspector for analytic lights. The engine doesn't have
// per-field setters for lights (no `light_set_pos` etc); the only
// available editors are the composite type constructors:
//   light_point  <id> x y z r g b
//   light_spot   <id> x y z dx dy dz outer_deg inner_deg r g b
//   light_sphere <id> x y z radius r g b
//   light_quad   <id> x y z nx ny nz ux uy uz u_half v_half r g b
//
// On any field commit we re-emit the appropriate full-state command
// with the latest values. That's what the underlying engine update
// requires and it keeps the editor flow uniform with prims (one
// console line per commit).
//
// Caveat: light_* commands flag light_prims_dirty_ for the renderer
// but DON'T currently broadcast scene_dirty (unlike prim_set_*). So a
// light edit dispatches and renders, but the inspector won't refetch
// via the shell's scene_dirty path. The local field state (NumberField
// / ColorField mirrors) stays in sync with what the user typed, so
// the UX still feels live -- the round-trip just doesn't happen. A
// future engine-side patch can plug scene_dirty into the light_*
// handlers to close the loop.

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

  // Rebuild the canonical engine command from a partial override.
  // `over` carries whichever field changed; everything else falls back
  // to the latest known state from the scene snapshot.
  const emit = useCallback(
    (over: Partial<{
      pos: [number, number, number];
      intensity: [number, number, number];
      dir: [number, number, number];
      outerDeg: number;
      innerDeg: number;
      radius: number;
      uVec: [number, number, number];
      uHalf: number;
      vHalf: number;
    }>) => {
      const p = over.pos ?? pos;
      const I = over.intensity ?? intensity;
      switch (light.type) {
        case 'point':
          exec(`light_point ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])} ${fmt(I[0])} ${fmt(I[1])} ${fmt(I[2])}`);
          break;
        case 'spot': {
          const d = over.dir ?? dir;
          const oDeg = over.outerDeg ?? outerDeg;
          const iDeg = over.innerDeg ?? innerDeg;
          exec(`light_spot ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])} ${fmt(d[0])} ${fmt(d[1])} ${fmt(d[2])} ${fmt(oDeg, 2)} ${fmt(iDeg, 2)} ${fmt(I[0])} ${fmt(I[1])} ${fmt(I[2])}`);
          break;
        }
        case 'sphere': {
          const r = over.radius ?? sphereRadius;
          exec(`light_sphere ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])} ${fmt(r)} ${fmt(I[0])} ${fmt(I[1])} ${fmt(I[2])}`);
          break;
        }
        case 'quad': {
          const n = over.dir ?? dir;
          // u_vec is unit-normalised by the engine; the editor exposes
          // u_half separately as a scalar. Reconstruct via either the
          // override or the existing u_vec direction.
          let uDir: [number, number, number];
          let uHalf: number;
          if (over.uVec) {
            uDir = over.uVec;
            uHalf = over.uHalf ?? vecLen(over.uVec) ?? 0.5;
          } else {
            const baseLen = vecLen(u_vec) || 1;
            uDir = [u_vec[0] / baseLen, u_vec[1] / baseLen, u_vec[2] / baseLen];
            uHalf = over.uHalf ?? baseLen;
          }
          const vH = over.vHalf ?? v_half;
          exec(`light_quad ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])} ${fmt(n[0])} ${fmt(n[1])} ${fmt(n[2])} ${fmt(uDir[0])} ${fmt(uDir[1])} ${fmt(uDir[2])} ${fmt(uHalf)} ${fmt(vH)} ${fmt(I[0])} ${fmt(I[1])} ${fmt(I[2])}`);
          break;
        }
      }
    },
    [exec, id, light.type, pos, intensity, dir, outerDeg, innerDeg, sphereRadius, u_vec, v_half],
  );

  // Compute the u_half scalar from the stored u_vec magnitude.
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
              onCommit={(v) => emit({ pos: [v, pos[1], pos[2]] })}
              onScrub={(v) => emit({ pos: [v, pos[1], pos[2]] })}
            />
            <NumberField
              axis="Y"
              value={pos[1]}
              onCommit={(v) => emit({ pos: [pos[0], v, pos[2]] })}
              onScrub={(v) => emit({ pos: [pos[0], v, pos[2]] })}
            />
            <NumberField
              axis="Z"
              value={pos[2]}
              onCommit={(v) => emit({ pos: [pos[0], pos[1], v] })}
              onScrub={(v) => emit({ pos: [pos[0], pos[1], v] })}
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
                onCommit={(v) => emit({ dir: [v, dir[1], dir[2]] })}
                onScrub={(v) => emit({ dir: [v, dir[1], dir[2]] })}
              />
              <NumberField
                axis="Y"
                value={dir[1]}
                onCommit={(v) => emit({ dir: [dir[0], v, dir[2]] })}
                onScrub={(v) => emit({ dir: [dir[0], v, dir[2]] })}
              />
              <NumberField
                axis="Z"
                value={dir[2]}
                onCommit={(v) => emit({ dir: [dir[0], dir[1], v] })}
                onScrub={(v) => emit({ dir: [dir[0], dir[1], v] })}
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
              onCommit={(v) => emit({ radius: v })}
              onScrub={(v) => emit({ radius: v })}
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
                onCommit={(v) => emit({ uHalf: v })}
                onScrub={(v) => emit({ uHalf: v })}
              />
            </div>
            <div className="insp-row">
              <label>V half</label>
              <NumberField
                value={v_half}
                min={0.0001}
                onCommit={(v) => emit({ vHalf: v })}
                onScrub={(v) => emit({ vHalf: v })}
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
            onCommit={(rgb) => emit({ intensity: rgb })}
            onScrub={(rgb) => emit({ intensity: rgb })}
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
                onCommit={(v) => emit({ outerDeg: v })}
                onScrub={(v) => emit({ outerDeg: v })}
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
                onCommit={(v) => emit({ innerDeg: v })}
                onScrub={(v) => emit({ innerDeg: v })}
              />
            </div>
          </>
        )}
      </section>
    </>
  );
}

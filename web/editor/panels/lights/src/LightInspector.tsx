// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Per-light property editor for the Lights panel. Like the Property
// Inspector's LightInspector (which dispatches the same atomic
// per-field setters since #238), this editor drives the ATOMIC
// setters the engine grew in #176 / #211:
//
//   light_set_pos       <id> <x> <y> <z>
//   light_set_intensity <id> <r> <g> <b>          (raw W/sr or W/m^2/sr)
//   light_set_dir       <id> <nx> <ny> <nz>        (spot axis / quad normal)
//   light_set_cone      <id> <inner_deg> <outer_deg>
//   light_set_size      <id> <r>                   (sphere radius / quad v_half)
//   light_set_rotation  <id> <qx> <qy> <qz> <qw>
//
// Photometric authoring (the engine's `_cd` / `_lm` / `_nits` /
// `_exposed` constructor sugar) is surfaced here by converting the
// authored value to the raw radiometric one with the SAME constants the
// engine uses, then writing it through `light_set_intensity`. That keeps
// per-field commits flowing through the atomic setter (so the rotate /
// translate gizmo and the inspector share one write path) while still
// letting the artist think in candela / lumen / nit / EV.
//
// Intensity is decomposed as  raw = color * magnitude * unitMultiplier,
// where `color` is the chromaticity (channels normalised so max == 1)
// and `magnitude` is the scalar in the chosen unit. The EV stepper
// multiplies the current raw value by 2^(+/-1) per click (one camera
// stop) -- the engine doesn't persist an EV term, so it folds into raw.

import { useCallback, useMemo } from 'react';
import { NumberField } from './NumberField';
import { Slider } from './Slider';
import { ColorField } from './ColorField';
import {
  type LightRecord,
  type IntensityUnit,
  unitsForType,
  unitMultiplier,
  cosToDeg,
  vecLen,
  fmt,
} from './helpers';

export interface LightInspectorProps {
  light: LightRecord;
  // Authoring unit for the intensity magnitude (panel-level state so it
  // persists across scene_dirty refetches). Per light type.
  unit: IntensityUnit;
  onUnitChange: (u: IntensityUnit) => void;
  exec: (line: string) => void;
}

// Decompose a raw intensity triple into (chromaticity, magnitude). The
// chromaticity is the colour normalised so its max channel is 1; the
// magnitude is that max channel (the raw radiometric peak). A pure-black
// light (all zero) decomposes to white * 0 so editing colour still works.
function decompose(intensity: [number, number, number]): {
  color: [number, number, number];
  rawMag: number;
} {
  const max = Math.max(intensity[0], intensity[1], intensity[2]);
  if (max <= 0) {
    return { color: [1, 1, 1], rawMag: 0 };
  }
  return {
    color: [intensity[0] / max, intensity[1] / max, intensity[2] / max],
    rawMag: max,
  };
}

export function LightInspector({ light, unit, onUnitChange, exec }: LightInspectorProps) {
  const id = light.id;
  const pos = light.pos;
  const dir = light.dir ?? ([0, -1, 0] as [number, number, number]);
  const outerDeg = cosToDeg(light.cos_outer ?? 1);
  const innerDeg = cosToDeg(light.cos_inner ?? 1);
  const sphereRadius = light.radius ?? 0.25;
  const u_vec = light.u_vec ?? ([1, 0, 0] as [number, number, number]);
  const v_half = light.v_half ?? 0.5;

  const { color, rawMag } = useMemo(() => decompose(light.intensity), [light.intensity]);

  // Magnitude in the authored unit. raw = mag * unitMultiplier(unit)  =>
  // mag = raw / unitMultiplier(unit).
  const mag = rawMag / unitMultiplier(unit);

  const validUnits = unitsForType(light.type);

  // ---- intensity dispatch ----
  // raw[c] = color[c] * magInUnit * unitMultiplier(unit). Clamp to >= 0
  // (light_set_intensity rejects negatives).
  const emitIntensity = useCallback(
    (nextColor: [number, number, number], magInUnit: number) => {
      const k = unitMultiplier(unit);
      const r = Math.max(0, nextColor[0] * magInUnit * k);
      const g = Math.max(0, nextColor[1] * magInUnit * k);
      const b = Math.max(0, nextColor[2] * magInUnit * k);
      exec(`light_set_intensity ${id} ${fmt(r)} ${fmt(g)} ${fmt(b)}`);
    },
    [exec, id, unit],
  );

  const onColorChange = useCallback(
    (rgb: [number, number, number]) => {
      // Editing the swatch sets chromaticity; preserve the raw magnitude.
      // The picker already factors out HDR brightness, so `rgb` is the
      // normalised colour -- re-apply the current magnitude.
      emitIntensity(rgb, mag);
    },
    [emitIntensity, mag],
  );

  const onMagChange = useCallback(
    (nextMag: number) => {
      emitIntensity(color, nextMag);
    },
    [emitIntensity, color],
  );

  // EV stepper: multiply current RAW by 2^stops (one camera stop per
  // click). Folds into the raw value the engine stores.
  const stepEv = useCallback(
    (stops: number) => {
      const k = Math.pow(2, stops);
      const r = Math.max(0, light.intensity[0] * k);
      const g = Math.max(0, light.intensity[1] * k);
      const b = Math.max(0, light.intensity[2] * k);
      exec(`light_set_intensity ${id} ${fmt(r)} ${fmt(g)} ${fmt(b)}`);
    },
    [exec, id, light.intensity],
  );

  // ---- transform dispatch (atomic setters) ----
  const emitPos = useCallback(
    (p: [number, number, number]) => {
      exec(`light_set_pos ${id} ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])}`);
    },
    [exec, id],
  );

  const emitDir = useCallback(
    (d: [number, number, number]) => {
      // light_set_dir auto-normalises and rejects zero-length; guard the
      // degenerate case so we don't fire a no-op the engine warns about.
      if (vecLen(d) < 1e-6) return;
      exec(`light_set_dir ${id} ${fmt(d[0])} ${fmt(d[1])} ${fmt(d[2])}`);
    },
    [exec, id],
  );

  const emitCone = useCallback(
    (innerD: number, outerD: number) => {
      // Engine requires 0 <= inner <= outer <= 90.
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

  const isArea = light.type === 'sphere' || light.type === 'quad';
  const magLabel =
    unit === 'raw'
      ? isArea ? 'W/m²/sr' : 'W/sr'
      : unit === 'cd'
      ? 'cd'
      : unit === 'lm'
      ? 'lm'
      : 'nits';

  return (
    <>
      <section className="lp-section">
        <h4 className="lp-section-title">Transform</h4>
        <div className="lp-row">
          <label>Position</label>
          <div className="lp-vec3">
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
          <div className="lp-row">
            <label>{light.type === 'spot' ? 'Direction' : 'Normal'}</label>
            <div className="lp-vec3">
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
          <div className="lp-row">
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
            <div className="lp-row">
              <label title="Quad U half-extent (read-only here; set via light_quad). The atomic light_set_size drives V.">
                U half
              </label>
              <NumberField value={vecLen(u_vec)} disabled onCommit={() => {}} />
            </div>
            <div className="lp-row">
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

      <section className="lp-section">
        <h4 className="lp-section-title">Emission</h4>
        <div className="lp-row">
          <label>Type</label>
          <span className="lp-badge is-readonly">{light.type}</span>
        </div>
        <div className="lp-row">
          <label>Color</label>
          <ColorField
            value={color}
            onCommit={(rgb) => onColorChange(rgb)}
            onScrub={(rgb) => onColorChange(rgb)}
          />
        </div>
        <div className="lp-row">
          <label title="Photometric unit. cd/lm convert via 1cd = 1/683.002 W/sr (555nm peak); nits = cd/m^2 for area lights. raw is the engine-internal value.">
            Unit
          </label>
          <select
            className="lp-select"
            value={unit}
            onChange={(e) => onUnitChange(e.target.value as IntensityUnit)}
          >
            {validUnits.map((u) => (
              <option key={u} value={u}>
                {u === 'raw'
                  ? isArea ? 'raw (W/m²/sr)' : 'raw (W/sr)'
                  : u === 'cd'
                  ? 'candela (cd)'
                  : u === 'lm'
                  ? 'lumens (lm, omni)'
                  : 'nits (cd/m²)'}
              </option>
            ))}
          </select>
        </div>
        <div className="lp-row">
          <label>Intensity</label>
          <NumberField
            label={magLabel}
            value={mag}
            min={0}
            precision={4}
            onCommit={(v) => onMagChange(v)}
            onScrub={(v) => onMagChange(v)}
            title={`Magnitude in ${magLabel}. raw = color * magnitude * unit factor.`}
          />
        </div>
        <div className="lp-row">
          <label title="Exposure stops. Each click multiplies the raw intensity by 2 (or 1/2). Folds into the stored radiometric value -- the engine has no separate EV term.">
            Exposure
          </label>
          <div className="lp-ev">
            <button
              type="button"
              className="lp-ev-btn"
              onClick={() => stepEv(-1)}
              title="-1 stop (halve intensity)"
            >
              {'−'}1 EV
            </button>
            <button
              type="button"
              className="lp-ev-btn"
              onClick={() => stepEv(1)}
              title="+1 stop (double intensity)"
            >
              +1 EV
            </button>
          </div>
        </div>
        {light.type === 'spot' && (
          <>
            <div className="lp-row">
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
            <div className="lp-row">
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

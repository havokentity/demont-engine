// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Advanced material lobes (wave-9): anisotropic GGX, clearcoat, and
// subsurface scattering for the selected primitive. These are driven
// through the prim_set_anisotropy / prim_set_clearcoat / prim_set_subsurface
// console commands.
//
// READ caveat: SceneGraph.cpp does not serialise these params back into
// the list_scene snapshot, so unlike the base PBR controls there is no
// live read-back of the engine's current value. The controls therefore
// hold local UI state seeded from the documented defaults (ADV_DEFAULTS)
// and reset whenever the selected prim changes. A short hint in the
// section explains this. (Same shape as the Normal Map section's
// "assigning is the supported path" caveat.)

import { useEffect, useMemo, useState } from 'react';
import { ColorField } from './ColorField';
import { Slider } from './Slider';
import {
  ADV_DEFAULTS,
  anisotropyCommand,
  clearcoatCommand,
  subsurfaceCommand,
} from './helpers';

interface AdvancedMaterialProps {
  /** Selected prim id. */
  id: number;
  /** Whether the base material is metal -- anisotropy only affects the
   *  metal GGX specular lobe, so we surface a hint when it isn't. */
  isMetal: boolean;
  /** Throttled exec (the same one MaterialEditor uses for its setters). */
  exec: (line: string) => void;
}

export function AdvancedMaterial({ id, isMetal, exec }: AdvancedMaterialProps) {
  // Local, write-only state. Re-seeded from defaults on every prim change
  // because the snapshot doesn't carry these values back.
  const [anisoAmount, setAnisoAmount] = useState<number>(ADV_DEFAULTS.anisoAmount);
  const [anisoRot, setAnisoRot] = useState<number>(ADV_DEFAULTS.anisoRotationDeg);
  const [coatWeight, setCoatWeight] = useState<number>(ADV_DEFAULTS.clearcoatWeight);
  const [coatRough, setCoatRough] = useState<number>(ADV_DEFAULTS.clearcoatRoughness);
  const [ssRadius, setSsRadius] = useState<number>(ADV_DEFAULTS.subsurfaceRadius);
  const [ssColor, setSsColor] = useState<[number, number, number]>([
    ...ADV_DEFAULTS.subsurfaceColor,
  ]);

  useEffect(() => {
    // Reset to defaults when the selection moves to a different prim.
    setAnisoAmount(ADV_DEFAULTS.anisoAmount);
    setAnisoRot(ADV_DEFAULTS.anisoRotationDeg);
    setCoatWeight(ADV_DEFAULTS.clearcoatWeight);
    setCoatRough(ADV_DEFAULTS.clearcoatRoughness);
    setSsRadius(ADV_DEFAULTS.subsurfaceRadius);
    setSsColor([...ADV_DEFAULTS.subsurfaceColor]);
  }, [id]);

  // --- Anisotropy -------------------------------------------------------
  const sendAniso = useMemo(
    () => (amount: number, rot: number) =>
      exec(anisotropyCommand(id, amount, rot)),
    [exec, id],
  );
  const onAnisoAmount = (v: number) => {
    setAnisoAmount(v);
    sendAniso(v, anisoRot);
  };
  const onAnisoRot = (v: number) => {
    setAnisoRot(v);
    sendAniso(anisoAmount, v);
  };

  // --- Clearcoat --------------------------------------------------------
  const sendCoat = useMemo(
    () => (weight: number, rough: number) =>
      exec(clearcoatCommand(id, weight, rough)),
    [exec, id],
  );
  const onCoatWeight = (v: number) => {
    setCoatWeight(v);
    sendCoat(v, coatRough);
  };
  const onCoatRough = (v: number) => {
    setCoatRough(v);
    sendCoat(coatWeight, v);
  };

  // --- Subsurface -------------------------------------------------------
  const sendSss = useMemo(
    () => (radius: number, color: [number, number, number]) =>
      exec(subsurfaceCommand(id, radius, color)),
    [exec, id],
  );
  const onSsRadius = (v: number) => {
    setSsRadius(v);
    sendSss(v, ssColor);
  };
  const onSsColor = (c: [number, number, number]) => {
    setSsColor(c);
    sendSss(ssRadius, c);
  };

  return (
    <section className="mtl-section">
      <h4 className="mtl-section-title">Advanced</h4>

      <div className="mtl-subhead">Anisotropy</div>
      <div className="mtl-row">
        <label>Amount</label>
        <Slider
          value={anisoAmount}
          min={-1}
          max={1}
          step={0.01}
          onCommit={onAnisoAmount}
          onScrub={onAnisoAmount}
        />
      </div>
      <div className="mtl-row">
        <label>Rotation°</label>
        <Slider
          value={anisoRot}
          min={0}
          max={180}
          step={1}
          precision={0}
          onCommit={onAnisoRot}
          onScrub={onAnisoRot}
        />
      </div>
      {!isMetal && (
        <p className="mtl-hint">
          Anisotropic GGX shapes the <strong>metal</strong> specular lobe
          (brushed metal). Switch <strong>Type</strong> to Metal to see it.
        </p>
      )}

      <div className="mtl-subhead">Clearcoat</div>
      <div className="mtl-row">
        <label>Weight</label>
        <Slider
          value={coatWeight}
          min={0}
          max={1}
          step={0.01}
          onCommit={onCoatWeight}
          onScrub={onCoatWeight}
        />
      </div>
      <div className="mtl-row">
        <label>Roughness</label>
        <Slider
          value={coatRough}
          min={0}
          max={1}
          step={0.01}
          onCommit={onCoatRough}
          onScrub={onCoatRough}
        />
      </div>

      <div className="mtl-subhead">Subsurface</div>
      <div className="mtl-row">
        <label>Radius (m)</label>
        <Slider
          value={ssRadius}
          min={0}
          max={0.05}
          step={0.0005}
          precision={4}
          onCommit={onSsRadius}
          onScrub={onSsRadius}
        />
      </div>
      <div className="mtl-row">
        <label>Color</label>
        <ColorField value={ssColor} onCommit={onSsColor} onScrub={onSsColor} />
      </div>

      <p className="mtl-hint">
        Car paint = metal base + clearcoat; skin/wax/marble = lambert base +
        subsurface (radius = mean free path in metres). These lobes are
        write-only — the engine doesn't echo them back, so the controls reset
        on re-selection. <strong>0</strong> weight / radius clears a lobe.
      </p>
    </section>
  );
}

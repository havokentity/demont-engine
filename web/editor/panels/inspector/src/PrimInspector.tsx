// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Property inspector for analytic primitives (Sphere / Plane). The
// inspected snapshot comes from useSceneStore.scene.primitives -- we
// don't keep a local copy beyond the in-progress NumberField mirrors.
//
// All edits dispatch through Console::Execute via WS exec; the engine
// emits scene_dirty -> Shell refetches list_scene -> store updates ->
// our props update. That round-trip is fast (< 1 frame typically) so
// the UI feels live without us optimistically writing the store.

import { useCallback, type ChangeEvent } from 'react';
import { NumberField } from './NumberField';
import { Slider } from './Slider';
import { ColorField } from './ColorField';

export interface PrimRecord {
  id: number;
  // engine emits "sphere" | "plane" (string), see SceneGraph.cpp.
  type: 'sphere' | 'plane';
  material: 'lambert' | 'metal' | 'dielectric' | 'water';
  pos?: [number, number, number];
  radius?: number;
  normal?: [number, number, number];
  d?: number;
  albedo: [number, number, number];
  emission: [number, number, number];
  roughness: number;
  ior: number;
}

export interface PrimInspectorProps {
  prim: PrimRecord;
  exec: (line: string) => void;
}

function fmt(n: number, prec = 4): string {
  if (!Number.isFinite(n)) return '0';
  return parseFloat(n.toFixed(prec)).toString();
}

export function PrimInspector({ prim, exec }: PrimInspectorProps) {
  const id = prim.id;

  // ---- Transform ----
  const pos = prim.type === 'sphere'
    ? (prim.pos ?? [0, 0, 0])
    : ([0, 0, 0] as [number, number, number]);
  const normal = prim.type === 'plane'
    ? (prim.normal ?? [0, 1, 0])
    : ([0, 1, 0] as [number, number, number]);
  const planeD = prim.type === 'plane' ? (prim.d ?? 0) : 0;
  const sphereR = prim.type === 'sphere' ? (prim.radius ?? 1) : 1;

  // For spheres: prim_set_pos updates the centre.
  // For planes: prim_set_pos repurposes <x y z> as the normal vector.
  // The engine command is documented as "move a primitive (sphere
  // center or plane normal)".
  const setPos = useCallback(
    (x: number, y: number, z: number) => {
      exec(`prim_set_pos ${id} ${fmt(x)} ${fmt(y)} ${fmt(z)}`);
    },
    [exec, id],
  );

  const setAlbedo = useCallback(
    (rgb: [number, number, number]) => {
      exec(`prim_set_albedo ${id} ${fmt(rgb[0])} ${fmt(rgb[1])} ${fmt(rgb[2])}`);
    },
    [exec, id],
  );
  const setEmission = useCallback(
    (rgb: [number, number, number]) => {
      exec(`prim_set_emission ${id} ${fmt(rgb[0])} ${fmt(rgb[1])} ${fmt(rgb[2])}`);
    },
    [exec, id],
  );
  const setRoughness = useCallback((v: number) => {
    exec(`prim_set_roughness ${id} ${fmt(v)}`);
  }, [exec, id]);
  const setIor = useCallback((v: number) => {
    exec(`prim_set_ior ${id} ${fmt(v)}`);
  }, [exec, id]);
  const setMaterial = useCallback((m: string) => {
    exec(`prim_set_material ${id} ${m}`);
  }, [exec, id]);
  // prim_set_radius landed with the 3D-transform gizmo PR (#203). It
  // only accepts spheres (planes use `d` which is the signed plane
  // distance from origin; no dedicated setter for `d` yet).
  const setRadius = useCallback((v: number) => {
    if (v <= 0) return;  // engine rejects radius <= 0
    exec(`prim_set_radius ${id} ${fmt(v)}`);
  }, [exec, id]);

  const radiusEditable = prim.type === 'sphere';
  const planeDEditable = false;

  return (
    <>
      <section className="insp-section">
        <h4 className="insp-section-title">Transform</h4>
        {prim.type === 'sphere' && (
          <>
            <div className="insp-row">
              <label>Position</label>
              <div className="insp-vec3">
                <NumberField
                  axis="X"
                  value={pos[0]}
                  onCommit={(v) => setPos(v, pos[1], pos[2])}
                  onScrub={(v) => setPos(v, pos[1], pos[2])}
                />
                <NumberField
                  axis="Y"
                  value={pos[1]}
                  onCommit={(v) => setPos(pos[0], v, pos[2])}
                  onScrub={(v) => setPos(pos[0], v, pos[2])}
                />
                <NumberField
                  axis="Z"
                  value={pos[2]}
                  onCommit={(v) => setPos(pos[0], pos[1], v)}
                  onScrub={(v) => setPos(pos[0], pos[1], v)}
                />
              </div>
            </div>
            <div className="insp-row">
              <label>Radius</label>
              <NumberField
                value={sphereR}
                onCommit={setRadius}
                onScrub={setRadius}
                disabled={!radiusEditable}
                min={0.0001}
              />
            </div>
          </>
        )}
        {prim.type === 'plane' && (
          <>
            <div className="insp-row">
              <label>Normal</label>
              <div className="insp-vec3">
                <NumberField
                  axis="X"
                  label="Nx"
                  value={normal[0]}
                  onCommit={(v) => setPos(v, normal[1], normal[2])}
                  onScrub={(v) => setPos(v, normal[1], normal[2])}
                />
                <NumberField
                  axis="Y"
                  label="Ny"
                  value={normal[1]}
                  onCommit={(v) => setPos(normal[0], v, normal[2])}
                  onScrub={(v) => setPos(normal[0], v, normal[2])}
                />
                <NumberField
                  axis="Z"
                  label="Nz"
                  value={normal[2]}
                  onCommit={(v) => setPos(normal[0], normal[1], v)}
                  onScrub={(v) => setPos(normal[0], normal[1], v)}
                />
              </div>
            </div>
            <div className="insp-row">
              <label>D</label>
              <NumberField
                value={planeD}
                onCommit={() => { /* no engine command yet */ }}
                disabled={!planeDEditable}
                title={planeDEditable ? '' : 'Plane distance edit not yet wired in engine'}
              />
            </div>
          </>
        )}
      </section>

      <section className="insp-section">
        <h4 className="insp-section-title">Material</h4>
        <div className="insp-row">
          <label>Type</label>
          <select
            className="insp-select"
            value={prim.material}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setMaterial(e.target.value)}
          >
            <option value="lambert">Lambert</option>
            <option value="metal">Metal</option>
            <option value="dielectric">Dielectric</option>
            <option value="water">Water</option>
          </select>
        </div>
        <div className="insp-row">
          <label>Albedo</label>
          <ColorField value={prim.albedo} onCommit={setAlbedo} onScrub={setAlbedo} />
        </div>
        <div className="insp-row">
          <label>Emission</label>
          <ColorField
            value={prim.emission}
            onCommit={setEmission}
            onScrub={setEmission}
            allowHdr
          />
        </div>
        <div className="insp-row">
          <label>Roughness</label>
          <Slider
            value={prim.roughness}
            min={0}
            max={1}
            step={0.001}
            onCommit={setRoughness}
            onScrub={setRoughness}
          />
        </div>
        <div className="insp-row">
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
    </>
  );
}

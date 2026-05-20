// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-logic helpers + the cvar descriptor table that drives the Render
// Settings panel. Extracted so the value coercion + command-line
// builders can be unit-tested without a DOM or a live WebSocket (vitest
// node mode -- `npm test`).
//
// WHY a static descriptor table instead of leaning on the engine's
// allowed_values / slider_min metadata: only a handful of the render
// cvars carry that metadata server-side (r_tonemap_op + r_sky_mode get
// allowed_values; almost none of the floats get slider ranges). Rather
// than render half the panel as raw text boxes, the panel owns the
// control-type + UI range knowledge here. The engine stays the source
// of truth for the *current value* (read via get_cvar); we only pick the
// widget + its bounds. Ranges below are chosen from each cvar's
// documented real-world span (see the PT_CVAR descriptions in
// src/engine/Engine.cpp).

/** A single render-settings control bound to one engine cvar. */
export interface CvarSpec {
  /** Engine cvar name, e.g. "r_fog_density". */
  name: string;
  /** Human label shown next to the control. */
  label: string;
  /** Widget kind. */
  kind: 'toggle' | 'slider' | 'select' | 'int';
  /** select: the option values (written verbatim to the cvar). */
  options?: string[];
  /** slider / int: inclusive UI range. */
  min?: number;
  max?: number;
  /** slider: granularity (default 0.001); int: 1. */
  step?: number;
  /** Decimal places for the numeric readout (slider only). */
  precision?: number;
  /** Browser-native tooltip / one-line help. */
  hint?: string;
}

/** A collapsible group of controls. */
export interface SectionSpec {
  id: string;
  title: string;
  /** When set, the section is gated on a master toggle cvar (e.g. the
   *  Fog section is governed by r_fog). The toggle is rendered at the
   *  top of the section and the remaining controls dim when it is off
   *  -- purely a UX affordance; the engine still accepts writes. */
  gate?: string;
  controls: CvarSpec[];
}

// ---------------------------------------------------------------------------
// The descriptor table. Section + control order is the panel layout.
// Sliders use a documented real-world range; toggles map a 0/1 cvar;
// selects mirror the engine's allowed_values verbatim.

export const SECTIONS: SectionSpec[] = [
  {
    id: 'tonemap',
    title: 'Tone Mapping',
    controls: [
      {
        name: 'r_tonemap_op',
        label: 'Operator',
        kind: 'select',
        options: ['aces', 'agx', 'khronos_pbr_neutral', 'reinhard', 'linear'],
        hint: 'HDR->LDR display transform applied after exposure.',
      },
    ],
  },
  {
    id: 'sky',
    title: 'Sky',
    controls: [
      {
        name: 'r_sky_mode',
        label: 'Mode',
        kind: 'select',
        options: ['gradient', 'hdri', 'procedural', 'hosek'],
        hint: 'gradient | hdri | procedural | hosek (Hosek-Wilkie analytic sky).',
      },
      {
        name: 'r_sky_turbidity',
        label: 'Turbidity',
        kind: 'slider',
        min: 1,
        max: 10,
        step: 0.01,
        precision: 2,
        hint: 'Hosek-Wilkie atmospheric turbidity. 2-4 clear, 6-10 hazy. Hosek only.',
      },
      {
        name: 'r_sky_ground_albedo',
        label: 'Ground Albedo',
        kind: 'slider',
        min: 0,
        max: 1,
        step: 0.01,
        precision: 2,
        hint: 'Hosek-Wilkie ground hemispherical albedo. Hosek only.',
      },
    ],
  },
  {
    id: 'clouds',
    title: 'Clouds',
    controls: [
      {
        name: 'r_clouds_mode',
        label: 'Mode',
        kind: 'select',
        options: ['pathtraced', 'procedural_raymarched'],
        hint: 'pathtraced (reference) | procedural_raymarched (cheap pre-pass).',
      },
      {
        name: 'r_clouds_raymarched_resolution_scale',
        label: 'Res Scale',
        kind: 'slider',
        min: 0.25,
        max: 1,
        step: 0.05,
        precision: 2,
        hint: 'Raymarch pre-pass resolution scale (procedural_raymarched). 1 = full res.',
      },
    ],
  },
  {
    id: 'fog',
    title: 'Fog',
    gate: 'r_fog',
    controls: [
      {
        name: 'r_fog',
        label: 'Enabled',
        kind: 'toggle',
        hint: 'Volumetric exponential-height atmospheric fog.',
      },
      {
        name: 'r_fog_density',
        label: 'Density',
        kind: 'slider',
        min: 0,
        max: 0.2,
        step: 0.0005,
        precision: 4,
        hint: 'Extinction coefficient at base height (1/m). 0.01-0.05 = moderate fog.',
      },
      {
        name: 'r_fog_base_y',
        label: 'Base Y',
        kind: 'slider',
        min: -100,
        max: 500,
        step: 1,
        precision: 1,
        hint: 'Fog base altitude (m) where density = r_fog_density.',
      },
      {
        name: 'r_fog_scale_height',
        label: 'Scale Height',
        kind: 'slider',
        min: 10,
        max: 2000,
        step: 1,
        precision: 1,
        hint: 'e-folding altitude (m) of the density falloff. ~300 m typical.',
      },
      {
        name: 'r_fog_sun_anisotropy',
        label: 'Sun Anisotropy',
        kind: 'slider',
        min: -0.95,
        max: 0.95,
        step: 0.01,
        precision: 2,
        hint: 'Henyey-Greenstein g for the sun forward-scatter lobe.',
      },
    ],
  },
  {
    id: 'godrays',
    title: 'God Rays',
    gate: 'r_godrays',
    controls: [
      {
        name: 'r_godrays',
        label: 'Enabled',
        kind: 'toggle',
        hint: 'Screen-space crepuscular light shafts (radial blur from the sun).',
      },
      {
        name: 'r_godrays_density',
        label: 'Density',
        kind: 'slider',
        min: 0,
        max: 1,
        step: 0.01,
        precision: 2,
        hint: 'Radial-blur reach toward the sun screen position.',
      },
      {
        name: 'r_godrays_decay',
        label: 'Decay',
        kind: 'slider',
        min: 0,
        max: 1,
        step: 0.01,
        precision: 2,
        hint: 'Per-sample exponential attenuation along the march.',
      },
      {
        name: 'r_godrays_weight',
        label: 'Weight',
        kind: 'slider',
        min: 0,
        max: 2,
        step: 0.01,
        precision: 2,
        hint: 'Per-sample accumulation weight (shaft brightness).',
      },
      {
        name: 'r_godrays_exposure',
        label: 'Exposure',
        kind: 'slider',
        min: 0,
        max: 5,
        step: 0.05,
        precision: 2,
        hint: 'Final shaft intensity multiplier before additive composite.',
      },
      {
        name: 'r_godrays_samples',
        label: 'Samples',
        kind: 'int',
        min: 1,
        max: 256,
        step: 1,
        hint: 'Radial-blur samples marched per pixel (1..256).',
      },
    ],
  },
  {
    id: 'ocean',
    title: 'Ocean',
    gate: 'r_ocean',
    controls: [
      {
        name: 'r_ocean',
        label: 'Enabled',
        kind: 'toggle',
        hint: 'FFT Tessendorf ocean surface on MAT_WATER planes (Metal + Vulkan).',
      },
      {
        name: 'r_ocean_wind_speed',
        label: 'Wind Speed',
        kind: 'slider',
        min: 0,
        max: 40,
        step: 0.1,
        precision: 1,
        hint: 'Wind speed (m/s). Drives dominant wavelength. 12 = Beaufort-6.',
      },
      {
        name: 'r_ocean_choppiness',
        label: 'Choppiness',
        kind: 'slider',
        min: 0,
        max: 1.5,
        step: 0.01,
        precision: 2,
        hint: 'Tessendorf horizontal displacement lambda [0, 1.5]. Sharpens crests.',
      },
      {
        name: 'r_ocean_tile_size',
        label: 'Tile Size',
        kind: 'slider',
        min: 5,
        max: 500,
        step: 1,
        precision: 1,
        hint: 'FFT patch size (m) tiled across the water plane.',
      },
      {
        name: 'r_ocean_foam_amount',
        label: 'Foam Amount',
        kind: 'slider',
        min: 0,
        max: 4,
        step: 0.05,
        precision: 2,
        hint: 'Foam intensity multiplier over crest-fold + whitecap coverage.',
      },
      {
        name: 'r_ocean_foam_coverage',
        label: 'Foam Coverage',
        kind: 'slider',
        min: 0,
        max: 4,
        step: 0.05,
        precision: 2,
        hint: 'Wind-driven whitecap-coverage gain (ties to wind speed).',
      },
      {
        name: 'r_ocean_foam_persistence',
        label: 'Foam Persist',
        kind: 'slider',
        min: 0,
        max: 0.999,
        step: 0.005,
        precision: 3,
        hint: 'Foam lifetime [0, 1). Higher = longer-lingering foam trails.',
      },
    ],
  },
  {
    id: 'camera',
    title: 'Camera / DoF',
    gate: 'r_dof',
    controls: [
      {
        name: 'r_dof',
        label: 'Enabled',
        kind: 'toggle',
        hint: 'Thin-lens depth of field. Off = pinhole (everything sharp).',
      },
      {
        name: 'r_dof_fstop',
        label: 'f-stop',
        kind: 'slider',
        min: 0.7,
        max: 22,
        step: 0.1,
        precision: 1,
        hint: 'Lens f-number. Smaller = wider aperture = shallower DoF + bigger bokeh.',
      },
      {
        name: 'r_dof_focal_length_mm',
        label: 'Focal Length',
        kind: 'slider',
        min: 8,
        max: 400,
        step: 1,
        precision: 0,
        hint: 'Lens focal length (mm). 35 wide, 50 normal, 85 portrait.',
      },
      {
        name: 'r_dof_focus_distance_m',
        label: 'Focus Dist',
        kind: 'slider',
        min: 0.1,
        max: 100,
        step: 0.1,
        precision: 2,
        hint: 'Focus-plane distance from camera (m). Use dof_focus_here to pick.',
      },
    ],
  },
  {
    id: 'spectral',
    title: 'Spectral',
    controls: [
      {
        name: 'r_spectral',
        label: 'Enabled',
        kind: 'toggle',
        hint: 'Per-ray hero-wavelength dispersion for dielectrics (prism rainbows).',
      },
      {
        name: 'r_spectral_cauchy_c',
        label: 'Cauchy C',
        kind: 'slider',
        min: 0,
        max: 0.02,
        step: 0.0001,
        precision: 5,
        hint: 'Cauchy C coefficient (um^2). BK7 ~0.0042, SF11 ~0.0127. 0 = achromatic.',
      },
    ],
  },
  {
    id: 'smoke',
    title: 'Smoke',
    controls: [
      {
        name: 'r_smoke_mode',
        label: 'Mode',
        kind: 'select',
        options: ['procedural', 'sph', 'both'],
        hint: 'procedural (puff-chain) | sph (particle solver) | both (hybrid).',
      },
      {
        name: 'r_smoke_buoyancy',
        label: 'Buoyancy',
        kind: 'slider',
        min: 0,
        max: 2,
        step: 0.01,
        precision: 2,
        hint: 'SPH thermal-lift scaler. 0 = cold smoke; >=2 rockets off-screen. SPH only.',
      },
      {
        name: 'r_smoke_wind_x',
        label: 'Wind X',
        kind: 'slider',
        min: -10,
        max: 10,
        step: 0.1,
        precision: 1,
        hint: 'Global wind velocity along +X (m/s). SPH only.',
      },
      {
        name: 'r_smoke_wind_y',
        label: 'Wind Y',
        kind: 'slider',
        min: -10,
        max: 10,
        step: 0.1,
        precision: 1,
        hint: 'Global wind velocity along +Y (m/s). Positive = updraft. SPH only.',
      },
      {
        name: 'r_smoke_wind_z',
        label: 'Wind Z',
        kind: 'slider',
        min: -10,
        max: 10,
        step: 0.1,
        precision: 1,
        hint: 'Global wind velocity along +Z (m/s). SPH only.',
      },
    ],
  },
];

/** Flat list of every cvar name the panel binds -- used to bulk-fetch
 *  current values on connect. */
export function allCvarNames(sections: SectionSpec[] = SECTIONS): string[] {
  const out: string[] = [];
  for (const s of sections) {
    for (const c of s.controls) out.push(c.name);
  }
  return out;
}

/** Format a float for a console / set_cvar value. Trims trailing zeros
 *  but keeps the value lossless to `prec` decimals. Non-finite -> "0". */
export function fmt(n: number, prec = 5): string {
  if (!Number.isFinite(n)) return '0';
  return parseFloat(n.toFixed(prec)).toString();
}

/** Parse a cvar string value into a float, falling back to `dflt` when
 *  the string is missing or non-numeric. */
export function parseCvarFloat(value: string | undefined, dflt = 0): number {
  if (value == null) return dflt;
  const n = parseFloat(value);
  return Number.isFinite(n) ? n : dflt;
}

/** Parse a cvar string value into a boolean. The engine stores booleans
 *  as "0" / "1"; treat any non-zero numeric or "true"/"on" as true. */
export function parseCvarBool(value: string | undefined): boolean {
  if (value == null) return false;
  const v = value.trim().toLowerCase();
  if (v === '' || v === '0' || v === 'false' || v === 'off') return false;
  const n = parseFloat(v);
  if (Number.isFinite(n)) return n !== 0;
  return v === 'true' || v === 'on' || v === '1';
}

/** Clamp n into [lo, hi] (either bound optional). */
export function clamp(n: number, lo?: number, hi?: number): number {
  let v = n;
  if (lo != null && v < lo) v = lo;
  if (hi != null && v > hi) v = hi;
  return v;
}

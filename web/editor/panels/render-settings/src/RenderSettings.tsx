// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Render Settings panel root. Surfaces the "console-only" rendering
// features (tone mapping, sky, clouds, fog, god rays, ocean, camera/DoF,
// spectral, smoke) as live UI controls bound directly to their engine
// cvars over the shared WebSocket. No selection dependency -- these are
// global render state, not per-object.
//
// Architecture:
//   - SECTIONS (helpers.ts) is the single source of truth for the panel
//     layout + each control's widget/range.
//   - useCvars bulk-reads the current values and exposes a throttled
//     setter (set_cvar over WS); the engine applies the write live and
//     the next frame reflects it.
//   - A section with a `gate` cvar (Fog/GodRays/Ocean/DoF) renders its
//     master toggle first; the remaining controls dim while the gate is
//     off (purely a UX cue -- writes still go through).

import {
  useSceneStore,
  type WebSocketClient,
} from '@demont/editor-shared';
import { SECTIONS, allCvarNames, parseCvarBool } from './helpers';
import { useCvars } from './useCvars';
import { Section, CvarToggle, CvarSelect, CvarSlider } from './Controls';

interface RenderSettingsProps {
  client: WebSocketClient;
}

const ALL_NAMES = allCvarNames();

export function RenderSettings({ client }: RenderSettingsProps) {
  const status = useSceneStore((s) => s.connectionStatus);
  const { values, loading, set, refresh } = useCvars(client, ALL_NAMES);

  const offline = status !== 'open';

  return (
    <div className="rs-root">
      <header className="rs-header">
        <div className="rs-header-meta">
          <span className="rs-title">Render Settings</span>
          <span className="rs-sub">global render state · bound live to cvars</span>
        </div>
        <button
          type="button"
          className="rs-refresh"
          onClick={refresh}
          title="Re-read every value from the engine"
        >
          {loading ? '…' : '↻'}
        </button>
      </header>

      {offline && (
        <div className="rs-banner" role="status">
          Engine offline — controls reflect the last known values and will
          re-sync on reconnect.
        </div>
      )}

      {SECTIONS.map((section) => {
        const gateOn = section.gate ? parseCvarBool(values[section.gate]) : true;
        const badge = section.gate ? (
          <span className={`rs-gate-pill${gateOn ? ' is-on' : ''}`}>
            {gateOn ? 'on' : 'off'}
          </span>
        ) : undefined;

        return (
          <Section key={section.id} title={section.title} badge={badge}>
            {section.controls.map((spec) => {
              // The gate toggle itself always stays enabled.
              const isGate = spec.name === section.gate;
              const enabled = isGate ? true : gateOn;

              if (spec.kind === 'toggle') {
                return (
                  <CvarToggle
                    key={spec.name}
                    spec={spec}
                    value={values[spec.name]}
                    onSet={set}
                  />
                );
              }
              if (spec.kind === 'select') {
                return (
                  <CvarSelect
                    key={spec.name}
                    spec={spec}
                    value={values[spec.name]}
                    onSet={set}
                  />
                );
              }
              return (
                <CvarSlider
                  key={spec.name}
                  spec={spec}
                  value={values[spec.name]}
                  onSet={set}
                  enabled={enabled}
                />
              );
            })}
          </Section>
        );
      })}

      <footer className="rs-footer">
        <p className="rs-hint">
          Sky turbidity / ground albedo apply only when{' '}
          <code>r_sky_mode</code> = hosek. Ocean &amp; some effects are
          Metal/Vulkan-only (the Software backend ignores them). Smoke wind /
          buoyancy apply when <code>r_smoke_mode</code> is sph or both.
        </p>
      </footer>
    </div>
  );
}

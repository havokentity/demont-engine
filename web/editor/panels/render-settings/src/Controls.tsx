// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Presentational cvar-bound controls for the Render Settings panel. Each
// reads its current value from the cvar map (string) and writes back via
// the throttled `set` from useCvars. The widgets are deliberately thin:
// the heavy lifting (drag throttling, scrub) lives in the copied Slider /
// NumberField components; these just translate between the cvar string
// representation and the typed widget value.

import { useCallback, useState, type ReactNode } from 'react';
import { Slider } from './Slider';
import {
  fmt,
  parseCvarFloat,
  parseCvarBool,
  type CvarSpec,
} from './helpers';

// ---------------------------------------------------------------------------
// Collapsible section.

export interface SectionProps {
  title: string;
  /** Optional summary chip shown on the right of the header (e.g. the
   *  master-gate state "on" / "off"). */
  badge?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function Section({ title, badge, defaultOpen = true, children }: SectionProps) {
  const [open, setOpen] = useState<boolean>(defaultOpen);
  return (
    <section className={`rs-section${open ? ' is-open' : ''}`}>
      <button
        type="button"
        className="rs-section-head"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className={`rs-caret${open ? ' is-open' : ''}`} aria-hidden="true">
          {'▸'}
        </span>
        <span className="rs-section-title">{title}</span>
        {badge != null && <span className="rs-section-badge">{badge}</span>}
      </button>
      {open && <div className="rs-section-body">{children}</div>}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Toggle (0/1 cvar).

export interface CvarToggleProps {
  spec: CvarSpec;
  value: string | undefined;
  onSet: (name: string, value: string) => void;
}

export function CvarToggle({ spec, value, onSet }: CvarToggleProps) {
  const on = parseCvarBool(value);
  const toggle = useCallback(() => {
    onSet(spec.name, on ? '0' : '1');
  }, [on, onSet, spec.name]);
  return (
    <div className="rs-row" title={spec.hint}>
      <label htmlFor={`rs-${spec.name}`}>{spec.label}</label>
      <button
        id={`rs-${spec.name}`}
        type="button"
        role="switch"
        aria-checked={on}
        className={`rs-toggle${on ? ' is-on' : ''}`}
        onClick={toggle}
      >
        <span className="rs-toggle-knob" />
        <span className="rs-toggle-label">{on ? 'On' : 'Off'}</span>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Select (enum cvar with allowed values).

export interface CvarSelectProps {
  spec: CvarSpec;
  value: string | undefined;
  onSet: (name: string, value: string) => void;
}

export function CvarSelect({ spec, value, onSet }: CvarSelectProps) {
  const opts = spec.options ?? [];
  // If the engine reports a value outside the known options (older /
  // newer build), keep it selectable so we don't silently rewrite it.
  const cur = value ?? '';
  const known = opts.includes(cur);
  return (
    <div className="rs-row" title={spec.hint}>
      <label htmlFor={`rs-${spec.name}`}>{spec.label}</label>
      <select
        id={`rs-${spec.name}`}
        className="rs-select"
        value={cur}
        onChange={(e) => onSet(spec.name, e.target.value)}
      >
        {!known && cur !== '' && (
          <option value={cur}>{cur} (current)</option>
        )}
        {opts.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Slider / int (numeric cvar). `int` rounds to whole numbers on commit.

export interface CvarSliderProps {
  spec: CvarSpec;
  value: string | undefined;
  onSet: (name: string, value: string) => void;
  /** When false the control dims + disables (master gate off). */
  enabled?: boolean;
}

export function CvarSlider({ spec, value, onSet, enabled = true }: CvarSliderProps) {
  const isInt = spec.kind === 'int';
  const min = spec.min ?? 0;
  const max = spec.max ?? 1;
  const step = spec.step ?? (isInt ? 1 : 0.001);
  const precision = spec.precision ?? (isInt ? 0 : 3);
  const num = parseCvarFloat(value, min);

  const commit = useCallback(
    (n: number) => {
      const v = isInt ? Math.round(n) : n;
      onSet(spec.name, fmt(v));
    },
    [isInt, onSet, spec.name],
  );

  return (
    <div className="rs-row" title={spec.hint}>
      <label>{spec.label}</label>
      <Slider
        value={num}
        min={min}
        max={max}
        step={step}
        precision={precision}
        onCommit={commit}
        onScrub={commit}
        disabled={!enabled}
      />
    </div>
  );
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Number input with Unity / Blender-style click-and-drag scrub on the
// axis label. Commit semantics:
//   - Enter        -> commit + advance focus to next .insp-num-input
//   - Esc          -> restore the original value (the snapshot taken
//                     on focus) and blur
//   - blur         -> commit if the parsed value differs
//   - scrub-drag   -> commit on pointer release; live-throttled
//                     dispatches happen every ~50ms during the drag
//
// Why scrub on the label and not the input? Click-drag on a <input>
// would race with text selection. Locking pointer + dragging anywhere
// in the field also breaks shift-tab + paste UX. The label is a
// dedicated affordance.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FocusEvent,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';

export interface NumberFieldProps {
  // Logical value (engine-side, e.g. metres). The component reflects
  // it into the <input> on every value change unless the user is
  // mid-edit; that lets external scene_dirty updates flow through
  // without stealing the user's keystrokes.
  value: number;
  // Called on commit only -- NOT on every keystroke. Receives the
  // parsed float; non-numeric input rejects via the Esc path.
  onCommit: (next: number) => void;
  // Called every ~50ms during a scrub-drag so the engine sees live
  // motion. The component throttles internally; the caller can ignore
  // throttling.
  onScrub?: (next: number) => void;
  // Optional X/Y/Z tint accent for the axis tab.
  axis?: 'X' | 'Y' | 'Z';
  // Override the displayed axis label. If omitted falls back to `axis`.
  label?: string;
  // Decimal places when rendering (default 3).
  precision?: number;
  // Clamp the value into a range. The engine command will still
  // accept anything, but the UI keeps the field tidy.
  min?: number;
  max?: number;
  // Disable the input (no edits, no scrub).
  disabled?: boolean;
  // Title attribute (browser-native tooltip).
  title?: string;
}

const kScrubThrottleMs = 50;

function formatNumber(n: number, precision: number): string {
  if (!Number.isFinite(n)) return '0';
  // Trim trailing zeros for shorter display, but keep at least one
  // fractional digit to communicate "this is a float".
  const fixed = n.toFixed(precision);
  return fixed.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

function clamp(n: number, lo?: number, hi?: number): number {
  let v = n;
  if (lo != null && v < lo) v = lo;
  if (hi != null && v > hi) v = hi;
  return v;
}

export function NumberField({
  value,
  onCommit,
  onScrub,
  axis,
  label,
  precision = 3,
  min,
  max,
  disabled,
  title,
}: NumberFieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const snapshotRef = useRef<number>(value);
  const valueRef = useRef<number>(value);
  const editingRef = useRef<boolean>(false);
  const lastScrubTsRef = useRef<number>(0);

  // The visible string. Driven by the prop unless the user is mid-edit.
  const [text, setText] = useState<string>(() => formatNumber(value, precision));
  const [scrubbing, setScrubbing] = useState<boolean>(false);
  const [focused, setFocused] = useState<boolean>(false);

  // Keep the latest committed value in a ref so async scrub handlers
  // see the current state without re-binding event listeners. We only
  // overwrite from the prop when NOT mid-edit and NOT mid-scrub --
  // otherwise an in-flight scene_dirty round-trip would clobber the
  // value the user is actively dragging through.
  if (!editingRef.current && !scrubbing) {
    valueRef.current = value;
  }

  // Reflect prop changes into the input while it's not being edited
  // by the user. This is how scene_dirty refreshes flow back.
  useEffect(() => {
    if (!editingRef.current && !scrubbing) {
      setText(formatNumber(value, precision));
    }
  }, [value, precision, scrubbing]);

  const commit = useCallback(
    (raw: string) => {
      const parsed = parseFloat(raw);
      if (!Number.isFinite(parsed)) {
        // Reject silently -- restore the field to the current value.
        setText(formatNumber(valueRef.current, precision));
        return;
      }
      const next = clamp(parsed, min, max);
      if (Math.abs(next - valueRef.current) > 1e-9) {
        onCommit(next);
      } else {
        // Even when unchanged, re-render the formatted string in case
        // the user typed "1.0000" -> we want "1".
        setText(formatNumber(next, precision));
      }
    },
    [onCommit, precision, min, max],
  );

  const onFocus = useCallback(() => {
    snapshotRef.current = value;
    valueRef.current = value;
    editingRef.current = true;
    setFocused(true);
    // Auto-select content so typing replaces.
    queueMicrotask(() => inputRef.current?.select());
  }, [value]);

  const onBlur = useCallback(
    (e: FocusEvent<HTMLInputElement>) => {
      editingRef.current = false;
      setFocused(false);
      commit(e.target.value);
    },
    [commit],
  );

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        commit(e.currentTarget.value);
        // Advance to next .insp-num-input sibling.
        const allInputs = Array.from(
          document.querySelectorAll<HTMLInputElement>('.insp-num-input'),
        );
        const idx = allInputs.indexOf(e.currentTarget);
        if (idx >= 0 && idx < allInputs.length - 1) {
          allInputs[idx + 1].focus();
        } else {
          e.currentTarget.blur();
        }
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        editingRef.current = false;
        // Restore display, then blur so the field stops being focused.
        setText(formatNumber(snapshotRef.current, precision));
        e.currentTarget.blur();
        return;
      }
    },
    [commit, precision],
  );

  const onInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setText(e.target.value);
  }, []);

  // ---- Scrub-drag ----
  const scrubStateRef = useRef<{
    lastX: number;
    pointerId: number;
    el: HTMLElement;
  } | null>(null);

  const endScrub = useCallback(
    (commitFinal: boolean) => {
      const s = scrubStateRef.current;
      if (!s) return;
      scrubStateRef.current = null;
      try {
        s.el.releasePointerCapture(s.pointerId);
      } catch {
        // ignore
      }
      // Best-effort exitPointerLock; Chromium fires a security warning
      // in dev when no lock is held, so we gate on document.pointerLockElement.
      if (document.pointerLockElement) {
        document.exitPointerLock();
      }
      setScrubbing(false);
      if (commitFinal) {
        const final = clamp(valueRef.current, min, max);
        // We've been calling onScrub (throttled) during the drag; emit
        // the final crisp value via onCommit so the parent can do its
        // post-drag accounting (e.g. one canonical engine command).
        onCommit(final);
      }
    },
    [min, max, onCommit],
  );

  const onScrubMove = useCallback(
    (e: PointerEvent) => {
      const s = scrubStateRef.current;
      if (!s) return;
      // Prefer movementX when pointer is locked (raw pixel delta);
      // fall back to absolute delta from start otherwise.
      let dx: number;
      if (document.pointerLockElement === s.el) {
        dx = e.movementX;
        s.lastX += dx;
      } else {
        dx = e.clientX - s.lastX;
        s.lastX = e.clientX;
      }
      // Mutate the live value in the ref so successive moves accumulate.
      // 1 px = base multiplier (default 0.01).
      // Shift => precision (0.001), Ctrl/Meta => coarse (1.0).
      let multiplier = 0.01;
      if (e.shiftKey) multiplier = 0.001;
      else if (e.ctrlKey || e.metaKey) multiplier = 1.0;
      const delta = dx * multiplier;
      const candidate = clamp(valueRef.current + delta, min, max);
      // Push into ref so the next move starts from here. We DON'T
      // call onCommit on every move -- only the throttled onScrub.
      valueRef.current = candidate;
      setText(formatNumber(candidate, precision));
      const now = performance.now();
      if (onScrub && now - lastScrubTsRef.current > kScrubThrottleMs) {
        lastScrubTsRef.current = now;
        onScrub(candidate);
      }
    },
    [min, max, onScrub, precision],
  );

  const onScrubUp = useCallback(() => {
    endScrub(true);
  }, [endScrub]);

  const onAxisPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLSpanElement>) => {
      if (disabled) return;
      // Left button only.
      if (e.button !== 0) return;
      e.preventDefault();
      const el = e.currentTarget;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        // ignore -- some browsers reject this in synthetic events
      }
      // Snapshot the current prop into valueRef so subsequent moves
      // can accumulate from a known baseline without the (now gated)
      // top-of-function ref refresh stomping it.
      valueRef.current = value;
      // Best-effort pointer lock for smooth infinite drag.
      if (el.requestPointerLock) {
        try {
          el.requestPointerLock();
        } catch {
          // ignore
        }
      }
      scrubStateRef.current = {
        lastX: e.clientX,
        pointerId: e.pointerId,
        el,
      };
      setScrubbing(true);
      // Listen on document so the drag survives leaving the label.
      document.addEventListener('pointermove', onScrubMove);
      document.addEventListener('pointerup', onScrubUp, { once: true });
      document.addEventListener('pointercancel', () => endScrub(false), { once: true });
    },
    [disabled, value, onScrubMove, onScrubUp, endScrub],
  );

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      document.removeEventListener('pointermove', onScrubMove);
      document.removeEventListener('pointerup', onScrubUp);
    };
  }, [onScrubMove, onScrubUp]);

  const axisClass = useMemo(() => {
    if (!axis) return '';
    return ` is-${axis.toLowerCase()}`;
  }, [axis]);

  return (
    <div
      className={`insp-num${axisClass}${scrubbing ? ' is-scrubbing' : ''}${focused ? ' is-focus' : ''}`}
      title={title}
    >
      {(label || axis) && (
        <span
          className={`insp-num-axis${axisClass}`}
          onPointerDown={onAxisPointerDown}
          title={`Drag to scrub. Shift=fine, Ctrl=coarse`}
        >
          {label ?? axis}
        </span>
      )}
      <input
        ref={inputRef}
        className="insp-num-input"
        type="text"
        inputMode="decimal"
        value={text}
        disabled={disabled}
        onChange={onInputChange}
        onFocus={onFocus}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
      />
    </div>
  );
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Number input with Unity / Blender-style click-and-drag scrub on the
// axis label. Self-contained copy for the Lights panel (each panel is
// its own Vite entry / workspace; the inspector keeps its own copy).
// Commit semantics:
//   - Enter        -> commit + advance focus to next .lp-num-input
//   - Esc          -> restore the original (focus-time) value and blur
//   - blur         -> commit if the parsed value differs
//   - scrub-drag   -> commit on pointer release; ~50ms throttled live
//                     onScrub dispatches during the drag

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
  value: number;
  onCommit: (next: number) => void;
  onScrub?: (next: number) => void;
  axis?: 'X' | 'Y' | 'Z';
  label?: string;
  precision?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  title?: string;
}

const kScrubThrottleMs = 50;

function formatNumber(n: number, precision: number): string {
  if (!Number.isFinite(n)) return '0';
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

  const [text, setText] = useState<string>(() => formatNumber(value, precision));
  const [scrubbing, setScrubbing] = useState<boolean>(false);
  const [focused, setFocused] = useState<boolean>(false);

  if (!editingRef.current && !scrubbing) {
    valueRef.current = value;
  }

  useEffect(() => {
    if (!editingRef.current && !scrubbing) {
      setText(formatNumber(value, precision));
    }
  }, [value, precision, scrubbing]);

  const commit = useCallback(
    (raw: string) => {
      const parsed = parseFloat(raw);
      if (!Number.isFinite(parsed)) {
        setText(formatNumber(valueRef.current, precision));
        return;
      }
      const next = clamp(parsed, min, max);
      if (Math.abs(next - valueRef.current) > 1e-9) {
        onCommit(next);
      } else {
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
        const allInputs = Array.from(
          document.querySelectorAll<HTMLInputElement>('.lp-num-input'),
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
  // The document-level pointer listeners for an active scrub are bound
  // with this controller's signal, so a single controller.abort() in
  // endScrub() tears down pointermove + pointerup + pointercancel in one
  // shot (covers both the pointerup and pointercancel exits). Without
  // this the pointermove handler would leak after the first drag (it
  // early-returns on every move) and pointercancel {once:true} listeners
  // could accumulate across drags that never fire a cancel.
  const scrubStateRef = useRef<{
    lastX: number;
    pointerId: number;
    el: HTMLElement;
    abort: AbortController;
  } | null>(null);

  const endScrub = useCallback(
    (commitFinal: boolean) => {
      const s = scrubStateRef.current;
      if (!s) return;
      scrubStateRef.current = null;
      // Detach all document listeners bound for this scrub at once.
      s.abort.abort();
      try {
        s.el.releasePointerCapture(s.pointerId);
      } catch {
        // ignore
      }
      if (document.pointerLockElement) {
        document.exitPointerLock();
      }
      setScrubbing(false);
      if (commitFinal) {
        const final = clamp(valueRef.current, min, max);
        onCommit(final);
      }
    },
    [min, max, onCommit],
  );

  const onScrubMove = useCallback(
    (e: PointerEvent) => {
      const s = scrubStateRef.current;
      if (!s) return;
      let dx: number;
      if (document.pointerLockElement === s.el) {
        dx = e.movementX;
        s.lastX += dx;
      } else {
        dx = e.clientX - s.lastX;
        s.lastX = e.clientX;
      }
      let multiplier = 0.01;
      if (e.shiftKey) multiplier = 0.001;
      else if (e.ctrlKey || e.metaKey) multiplier = 1.0;
      const delta = dx * multiplier;
      const candidate = clamp(valueRef.current + delta, min, max);
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
      if (e.button !== 0) return;
      e.preventDefault();
      const el = e.currentTarget;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      valueRef.current = value;
      if (el.requestPointerLock) {
        try {
          el.requestPointerLock();
        } catch {
          // ignore
        }
      }
      const abort = new AbortController();
      scrubStateRef.current = {
        lastX: e.clientX,
        pointerId: e.pointerId,
        el,
        abort,
      };
      setScrubbing(true);
      // Listen on document so the drag survives leaving the label. All
      // three are bound with the same signal so endScrub()'s abort()
      // removes them together -- no leaked pointermove handler, no
      // accumulating pointercancel listeners across drags.
      const { signal } = abort;
      document.addEventListener('pointermove', onScrubMove, { signal });
      document.addEventListener('pointerup', onScrubUp, { signal });
      document.addEventListener('pointercancel', () => endScrub(false), { signal });
    },
    [disabled, value, onScrubMove, onScrubUp, endScrub],
  );

  // Cleanup on unmount: abort any in-flight scrub so its document
  // listeners (bound with the controller's signal) are detached.
  useEffect(() => {
    return () => {
      scrubStateRef.current?.abort.abort();
    };
  }, []);

  const axisClass = useMemo(() => {
    if (!axis) return '';
    return ` is-${axis.toLowerCase()}`;
  }, [axis]);

  return (
    <div
      className={`lp-num${axisClass}${scrubbing ? ' is-scrubbing' : ''}${focused ? ' is-focus' : ''}`}
      title={title}
    >
      {(label || axis) && (
        <span
          className={`lp-num-axis${axisClass}`}
          onPointerDown={onAxisPointerDown}
          title={`Drag to scrub. Shift=fine, Ctrl=coarse`}
        >
          {label ?? axis}
        </span>
      )}
      <input
        ref={inputRef}
        className="lp-num-input"
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

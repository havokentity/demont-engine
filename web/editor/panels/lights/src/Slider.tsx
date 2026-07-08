// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Slider + number-readout combo. Self-contained copy for the Lights
// panel. Commit semantics match NumberField:
//   - dragging the slider fires throttled onScrub (~50ms, with a
//     trailing flush so the last value of a burst is never dropped)
//   - releasing the slider fires onCommit
//   - keyboard changes (arrow keys) scrub too and commit on key
//     release / blur
//   - the number readout uses NumberField (Enter/Esc/blur commit)

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';
import { NumberField } from './NumberField';

export interface SliderProps {
  value: number;
  onCommit: (next: number) => void;
  onScrub?: (next: number) => void;
  min: number;
  max: number;
  step?: number;
  precision?: number;
  disabled?: boolean;
}

const kScrubThrottleMs = 50;

export function Slider({
  value,
  onCommit,
  onScrub,
  min,
  max,
  step = 0.001,
  precision = 3,
  disabled,
}: SliderProps) {
  const [local, setLocal] = useState<number>(value);
  const lastScrubTsRef = useRef<number>(0);
  const trailingTimerRef = useRef<number | null>(null);
  const draggingRef = useRef<boolean>(false);
  // Set when a keyboard-driven change lands (arrow keys never see
  // pointer events); commit happens on key release / blur instead.
  const pendingKeyCommitRef = useRef<boolean>(false);

  useEffect(() => {
    if (!draggingRef.current) {
      setLocal(value);
    }
  }, [value]);

  const clearTrailing = useCallback(() => {
    if (trailingTimerRef.current != null) {
      window.clearTimeout(trailingTimerRef.current);
      trailingTimerRef.current = null;
    }
  }, []);

  // Drop any pending trailing scrub on unmount.
  useEffect(() => clearTrailing, [clearTrailing]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const next = parseFloat(e.target.value);
      if (!Number.isFinite(next)) return;
      setLocal(next);
      if (!draggingRef.current) pendingKeyCommitRef.current = true;
      if (!onScrub) return;
      clearTrailing();
      const now = performance.now();
      const elapsed = now - lastScrubTsRef.current;
      if (elapsed > kScrubThrottleMs) {
        lastScrubTsRef.current = now;
        onScrub(next);
      } else {
        // Trailing edge: schedule the last value of a fast burst so
        // the engine always sees the final scrub position (a plain
        // leading-edge throttle would silently drop it).
        trailingTimerRef.current = window.setTimeout(() => {
          trailingTimerRef.current = null;
          lastScrubTsRef.current = performance.now();
          onScrub(next);
        }, kScrubThrottleMs - elapsed);
      }
    },
    [onScrub, clearTrailing],
  );

  const onPointerDown = useCallback(() => {
    draggingRef.current = true;
  }, []);

  const onPointerUp = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    pendingKeyCommitRef.current = false;
    clearTrailing();  // the commit below supersedes any pending scrub
    onCommit(local);
  }, [local, onCommit, clearTrailing]);

  // Keyboard-driven changes (arrow keys / PgUp / Home / ...) fire
  // onChange with no pointer events, so the pointerup commit never
  // runs -- commit on key release and blur instead. Gated on the
  // pending flag so unrelated keys (Tab) don't dispatch commands.
  const commitPendingKey = useCallback(() => {
    if (!pendingKeyCommitRef.current) return;
    pendingKeyCommitRef.current = false;
    clearTrailing();
    onCommit(local);
  }, [local, onCommit, clearTrailing]);

  return (
    <div className="lp-slider-row">
      <input
        className="lp-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={local}
        disabled={disabled}
        onChange={onChange}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onKeyUp={commitPendingKey}
        onBlur={commitPendingKey}
      />
      <NumberField
        value={local}
        onCommit={(n) => {
          setLocal(n);
          onCommit(n);
        }}
        precision={precision}
        min={min}
        max={max}
        disabled={disabled}
      />
    </div>
  );
}

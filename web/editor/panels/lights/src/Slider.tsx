// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Slider + number-readout combo. Self-contained copy for the Lights
// panel. Commit semantics match NumberField:
//   - dragging the slider fires throttled onScrub (~50ms)
//   - releasing the slider fires onCommit
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
  const draggingRef = useRef<boolean>(false);

  useEffect(() => {
    if (!draggingRef.current) {
      setLocal(value);
    }
  }, [value]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const next = parseFloat(e.target.value);
      if (!Number.isFinite(next)) return;
      setLocal(next);
      const now = performance.now();
      if (onScrub && now - lastScrubTsRef.current > kScrubThrottleMs) {
        lastScrubTsRef.current = now;
        onScrub(next);
      }
    },
    [onScrub],
  );

  const onPointerDown = useCallback(() => {
    draggingRef.current = true;
  }, []);

  const onPointerUp = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    onCommit(local);
  }, [local, onCommit]);

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

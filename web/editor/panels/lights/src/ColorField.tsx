// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Color picker for linear-RGB float triples. Self-contained copy for
// the Lights panel. The engine stores light intensity in linear-RGB;
// the swatch + hex live in sRGB display space, so we round-trip:
//
//   prop (linear float) --> sRGB display --> user edit --> linear float
//
// The popover is a minimal Saturation/Value plane + Hue bar + numeric
// readouts. HDR-range values (intensity can exceed [0,1]) are factored
// into a brightness multiplier so the picker operates in [0,1] while
// the underlying value is preserved.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';

export interface ColorFieldProps {
  value: [number, number, number];
  onCommit: (rgb: [number, number, number]) => void;
  onScrub?: (rgb: [number, number, number]) => void;
  allowHdr?: boolean;
  disabled?: boolean;
}

// IEC 61966-2-1 sRGB transfer functions.
function linearToSrgb(c: number): number {
  if (!Number.isFinite(c) || c <= 0) return 0;
  if (c >= 1) return 1;
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function srgbToLinear(c: number): number {
  if (!Number.isFinite(c) || c <= 0) return 0;
  if (c >= 1) return 1;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function clamp01(c: number): number {
  if (!Number.isFinite(c)) return 0;
  return c < 0 ? 0 : c > 1 ? 1 : c;
}

function toHex2(n: number): string {
  const v = Math.round(clamp01(n) * 255);
  return v.toString(16).padStart(2, '0').toUpperCase();
}

function rgbToHex(rgb: [number, number, number]): string {
  return (
    '#' +
    toHex2(linearToSrgb(rgb[0])) +
    toHex2(linearToSrgb(rgb[1])) +
    toHex2(linearToSrgb(rgb[2]))
  );
}

function hexToRgb(hex: string): [number, number, number] | null {
  const s = hex.replace(/^#/, '').trim();
  if (s.length !== 6 && s.length !== 3) return null;
  const expanded = s.length === 3 ? s.split('').map((c) => c + c).join('') : s;
  const r8 = parseInt(expanded.slice(0, 2), 16);
  const g8 = parseInt(expanded.slice(2, 4), 16);
  const b8 = parseInt(expanded.slice(4, 6), 16);
  if (!Number.isFinite(r8) || !Number.isFinite(g8) || !Number.isFinite(b8)) return null;
  return [srgbToLinear(r8 / 255), srgbToLinear(g8 / 255), srgbToLinear(b8 / 255)];
}

function hsvToSrgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const hh = (h % 360) / 60;
  const x = c * (1 - Math.abs((hh % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hh < 1) { r = c; g = x; }
  else if (hh < 2) { r = x; g = c; }
  else if (hh < 3) { g = c; b = x; }
  else if (hh < 4) { g = x; b = c; }
  else if (hh < 5) { r = x; b = c; }
  else { r = c; b = x; }
  const m = v - c;
  return [r + m, g + m, b + m];
}

function srgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  return [h, s, max];
}

export function ColorField({
  value,
  onCommit,
  onScrub,
  allowHdr,
  disabled,
}: ColorFieldProps) {
  const swatchRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState<boolean>(false);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);

  const [draft, setDraft] = useState<[number, number, number]>(value);

  const [hexText, setHexText] = useState<string>(() => rgbToHex(value));
  const editingHexRef = useRef<boolean>(false);

  const hdrMultiplier = useMemo(() => {
    if (!allowHdr) return 1;
    return Math.max(1, Math.max(draft[0], draft[1], draft[2]));
  }, [allowHdr, draft]);

  const display = useMemo<[number, number, number]>(() => {
    return [draft[0] / hdrMultiplier, draft[1] / hdrMultiplier, draft[2] / hdrMultiplier];
  }, [draft, hdrMultiplier]);

  const hsv = useMemo<[number, number, number]>(() => {
    const sR = linearToSrgb(display[0]);
    const sG = linearToSrgb(display[1]);
    const sB = linearToSrgb(display[2]);
    return srgbToHsv(sR, sG, sB);
  }, [display]);

  useEffect(() => {
    if (open) return;
    setDraft(value);
    if (!editingHexRef.current) setHexText(rgbToHex(value));
  }, [value, open]);

  useEffect(() => {
    if (!editingHexRef.current) setHexText(rgbToHex(draft));
  }, [draft]);

  const lastScrubTsRef = useRef<number>(0);
  const fireScrub = useCallback(
    (rgb: [number, number, number]) => {
      if (!onScrub) return;
      const now = performance.now();
      if (now - lastScrubTsRef.current > 50) {
        lastScrubTsRef.current = now;
        onScrub(rgb);
      }
    },
    [onScrub],
  );

  const setDraftAndScrub = useCallback(
    (rgb: [number, number, number]) => {
      setDraft(rgb);
      fireScrub(rgb);
    },
    [fireScrub],
  );

  const onSwatchClick = useCallback(() => {
    if (disabled) return;
    const rect = swatchRef.current?.getBoundingClientRect();
    if (rect) {
      const px = Math.min(rect.left, window.innerWidth - 220);
      const py = rect.bottom + 4;
      setAnchor({ x: Math.max(8, px), y: py });
    } else {
      setAnchor({ x: 20, y: 80 });
    }
    setDraft(value);
    setOpen(true);
  }, [disabled, value]);

  const commitDraft = useCallback(() => {
    onCommit(draft);
  }, [onCommit, draft]);

  const closePopover = useCallback(() => {
    if (!open) return;
    setOpen(false);
    commitDraft();
  }, [open, commitDraft]);

  useEffect(() => {
    if (!open) return;
    const onDocPointerDown = (ev: PointerEvent) => {
      const target = ev.target as HTMLElement | null;
      if (!target) return;
      if (target.closest('.lp-popover')) return;
      if (target.closest('.lp-swatch')) return;
      closePopover();
    };
    document.addEventListener('pointerdown', onDocPointerDown, true);
    return () => document.removeEventListener('pointerdown', onDocPointerDown, true);
  }, [open, closePopover]);

  const svPlaneRef = useRef<HTMLDivElement>(null);
  const onSvPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled) return;
      const el = svPlaneRef.current;
      if (!el) return;
      el.setPointerCapture(e.pointerId);
      const update = (clientX: number, clientY: number) => {
        const rect = el.getBoundingClientRect();
        const sx = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const sy = Math.max(0, Math.min(1, 1 - (clientY - rect.top) / rect.height));
        const [h] = hsv;
        const [sR, sG, sB] = hsvToSrgb(h, sx, sy);
        const lin: [number, number, number] = [
          srgbToLinear(sR) * hdrMultiplier,
          srgbToLinear(sG) * hdrMultiplier,
          srgbToLinear(sB) * hdrMultiplier,
        ];
        setDraftAndScrub(lin);
      };
      update(e.clientX, e.clientY);
      const onMove = (ev: PointerEvent) => update(ev.clientX, ev.clientY);
      const onUp = () => {
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
        try { el.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
      };
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    },
    [disabled, hsv, hdrMultiplier, setDraftAndScrub],
  );

  const hueBarRef = useRef<HTMLDivElement>(null);
  const onHuePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (disabled) return;
      const el = hueBarRef.current;
      if (!el) return;
      el.setPointerCapture(e.pointerId);
      const update = (clientX: number) => {
        const rect = el.getBoundingClientRect();
        const fr = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const h = fr * 360;
        const [, s, v] = hsv;
        const [sR, sG, sB] = hsvToSrgb(h, s, v);
        const lin: [number, number, number] = [
          srgbToLinear(sR) * hdrMultiplier,
          srgbToLinear(sG) * hdrMultiplier,
          srgbToLinear(sB) * hdrMultiplier,
        ];
        setDraftAndScrub(lin);
      };
      update(e.clientX);
      const onMove = (ev: PointerEvent) => update(ev.clientX);
      const onUp = () => {
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
        try { el.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
      };
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
    },
    [disabled, hsv, hdrMultiplier, setDraftAndScrub],
  );

  const onHexChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setHexText(e.target.value);
  }, []);
  const onHexFocus = useCallback(() => { editingHexRef.current = true; }, []);
  const onHexBlur = useCallback(() => {
    editingHexRef.current = false;
    const parsed = hexToRgb(hexText);
    if (parsed) {
      const scaled: [number, number, number] = [
        parsed[0] * hdrMultiplier,
        parsed[1] * hdrMultiplier,
        parsed[2] * hdrMultiplier,
      ];
      setDraft(scaled);
      onCommit(scaled);
    } else {
      setHexText(rgbToHex(draft));
    }
  }, [hexText, hdrMultiplier, draft, onCommit]);

  const onRgbCommit = useCallback(
    (idx: 0 | 1 | 2, raw: string) => {
      const v = parseFloat(raw);
      if (!Number.isFinite(v)) return;
      const next: [number, number, number] = [draft[0], draft[1], draft[2]];
      next[idx] = v < 0 ? 0 : v;
      setDraft(next);
      onCommit(next);
    },
    [draft, onCommit],
  );

  const svBackground = useMemo(() => {
    const [h] = hsv;
    const [sR, sG, sB] = hsvToSrgb(h, 1, 1);
    const pure = `rgb(${Math.round(sR * 255)},${Math.round(sG * 255)},${Math.round(sB * 255)})`;
    return `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, ${pure})`;
  }, [hsv]);

  const swatchFill = useMemo(() => {
    const [r, g, b] = display;
    return `rgb(${Math.round(clamp01(linearToSrgb(r)) * 255)},${Math.round(clamp01(linearToSrgb(g)) * 255)},${Math.round(clamp01(linearToSrgb(b)) * 255)})`;
  }, [display]);

  return (
    <>
      <div className="lp-color">
        <div
          ref={swatchRef}
          className="lp-swatch"
          onPointerDown={(e) => { if (e.button === 0) onSwatchClick(); }}
          title="Open color picker"
        >
          <div className="lp-swatch-fill" style={{ background: swatchFill }} />
        </div>
        <input
          className="lp-hex"
          type="text"
          value={hexText}
          disabled={disabled}
          onChange={onHexChange}
          onFocus={onHexFocus}
          onBlur={onHexBlur}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { (e.currentTarget as HTMLInputElement).blur(); }
            if (e.key === 'Escape') {
              setHexText(rgbToHex(draft));
              (e.currentTarget as HTMLInputElement).blur();
            }
          }}
          spellCheck={false}
        />
      </div>
      {open && anchor && (
        <div
          className="lp-popover"
          style={{ left: anchor.x, top: anchor.y }}
        >
          <div
            ref={svPlaneRef}
            className="lp-sv-plane"
            style={{ background: svBackground }}
            onPointerDown={onSvPointerDown}
          >
            <div
              className="lp-sv-cursor"
              style={{
                left: `${hsv[1] * 100}%`,
                top: `${(1 - hsv[2]) * 100}%`,
              }}
            />
          </div>
          <div
            ref={hueBarRef}
            className="lp-hue-bar"
            onPointerDown={onHuePointerDown}
          >
            <div
              className="lp-hue-cursor"
              style={{ left: `${(hsv[0] / 360) * 100}%` }}
            />
          </div>
          <div className="lp-rgb-grid">
            <label>
              R
              <input
                type="text"
                inputMode="decimal"
                defaultValue={draft[0].toFixed(3)}
                key={`r-${draft[0].toFixed(4)}`}
                onBlur={(e) => onRgbCommit(0, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
                }}
              />
            </label>
            <label>
              G
              <input
                type="text"
                inputMode="decimal"
                defaultValue={draft[1].toFixed(3)}
                key={`g-${draft[1].toFixed(4)}`}
                onBlur={(e) => onRgbCommit(1, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
                }}
              />
            </label>
            <label>
              B
              <input
                type="text"
                inputMode="decimal"
                defaultValue={draft[2].toFixed(3)}
                key={`b-${draft[2].toFixed(4)}`}
                onBlur={(e) => onRgbCommit(2, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
                }}
              />
            </label>
          </div>
        </div>
      )}
    </>
  );
}

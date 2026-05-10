#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
"""Drive demont.exe across multiple r_denoiser values and report
RMSE / SSIM between the captures. Closes the "OptiX looks worse than
SVGF" / "AOV looks identical to HDR" subjective debate by producing
numbers instead of squinting at screenshots.

Workflow per mode:
1. Write a one-shot autoexec.cfg pinning r_denoiser, r_camera, r_spp,
   r_capture_seed, r_capture_frame_at.
2. Launch demont.exe with --net-port / --net-line-port (PR #7's CLI
   args, so we don't collide with the user's running instance).
3. Wait for the capture PPM to appear in `<workdir>/captures/`.
4. Kill the process, move the PPM to `<output_dir>/<mode>.ppm`.

After all modes are captured, compute pairwise RMSE + (optional) SSIM
between the LDR PPMs and dump a CSV. Pure-stdlib for RMSE; SSIM uses
NumPy if it's importable, otherwise reports "n/a".

Determinism contract (see Engine.cpp r_capture_seed):
  * The path tracer's per-pixel PRNG seeds purely from
    (pixel_id, frame_index) -- see PathTrace.slang's pcgHash() seed.
  * `r_capture_seed N` resets engine `frame_index_` to N at cvar set
    time, so two cold launches with the same seed render the same
    sequence of frames.
  * `r_capture_seed` MUST be < `r_capture_frame_at` (otherwise the
    target frame is already in the past and the capture never fires).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_EXE   = REPO_ROOT / "build" / "win-clang-release" / "src" / "app" / "demont.exe"
DEFAULT_MODES = ["off", "svgf_atrous", "optix_hdr", "optix_hdr_aov"]
DEFAULT_PORTS = (27964, 27965)


def parse_ppm(path: Path) -> tuple[int, int, bytes]:
    """Minimal P6 PPM parser. Returns (width, height, raw_bytes_rgb)."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"{path}: not a P6 PPM (got {magic!r})")
        # Skip comments + read width / height / maxval. Headers can have
        # comments anywhere between tokens but the engine writes the
        # canonical "P6\n<w> <h>\n255\n" so the simple loop is fine.
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        maxval = int(line.strip())
        if maxval != 255:
            raise ValueError(f"{path}: maxval {maxval} != 255")
        data = f.read()
    expected = w * h * 3
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(data)}")
    return w, h, data


def rmse(a_bytes: bytes, b_bytes: bytes) -> float:
    """Per-pixel RMSE in 0..255 sRGB byte space. Pure-stdlib so the
    script runs on a clean Python install without numpy."""
    if len(a_bytes) != len(b_bytes):
        raise ValueError("RMSE: byte-length mismatch")
    n   = len(a_bytes)
    acc = 0
    # Walk in chunks for speed on CPython without giving up to numpy
    for ba, bb in zip(a_bytes, b_bytes):
        d = ba - bb
        acc += d * d
    return math.sqrt(acc / n) if n > 0 else 0.0


def ssim_if_available(a_bytes: bytes, b_bytes: bytes,
                      w: int, h: int) -> float | None:
    """SSIM via numpy if it's importable. Returns None otherwise.
    Single-channel (luma) approximation rather than full multi-channel
    SSIM -- adequate for sanity-checking denoiser deltas. Constants per
    Wang et al. (2004), with k1=0.01, k2=0.03, dynamic range L=255."""
    try:
        import numpy as np
    except ImportError:
        return None
    a = np.frombuffer(a_bytes, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
    b = np.frombuffer(b_bytes, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
    # ITU-R BT.601 luma -- close enough to perceptual for sanity sums.
    luma_a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    luma_b = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2])
    mu_a   = luma_a.mean()
    mu_b   = luma_b.mean()
    var_a  = luma_a.var()
    var_b  = luma_b.var()
    cov    = ((luma_a - mu_a) * (luma_b - mu_b)).mean()
    L      = 255.0
    c1     = (0.01 * L) ** 2
    c2     = (0.03 * L) ** 2
    num    = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den    = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(num / den)


def write_autoexec(path: Path, cfg_lines: list[str]) -> None:
    # UTF-8 without BOM -- the engine's tokenizer doesn't strip the BOM
    # and would silently mangle the first cvar name. Found while
    # smoke-testing PR #6 on the same engine.
    path.write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")


def run_capture(exe: Path,
                workdir: Path,
                cfg_lines: list[str],
                expected_ppm_glob: str,
                timeout_s: float,
                http_port: int,
                line_port: int) -> Path | None:
    """Launch demont.exe with the supplied autoexec.cfg and wait for a
    capture PPM matching `expected_ppm_glob` to appear. Returns the
    captured PPM path or None on timeout / failure.

    Non-destructive to the user's existing autoexec.cfg: if one is
    already present in `workdir`, we preserve its bytes in a
    `.compare_denoisers.bak` sibling and restore on exit (success or
    failure). This matters because demont.exe's working dir is the
    install / build tree where the user is likely to have a real cfg
    they care about."""
    autoexec = workdir / "autoexec.cfg"
    backup   = workdir / ".compare_denoisers.bak"
    backup_made = False
    if autoexec.exists():
        # If a stale backup is sitting around from a previous crashed
        # run, leave it -- restoring user's autoexec is more important
        # than not clobbering the backup. Bail loud if both exist.
        if backup.exists():
            print(f"[run_capture] WARNING: existing backup at {backup} "
                  f"-- not overwriting. Move/delete it manually if "
                  f"you're sure it's stale.", file=sys.stderr)
        else:
            shutil.copy2(autoexec, backup)
            backup_made = True
    write_autoexec(autoexec, cfg_lines)
    captures_dir = workdir / "captures"
    captures_dir.mkdir(exist_ok=True)
    # Snapshot pre-existing captures so we only consider new files.
    pre = {p.name for p in captures_dir.glob(expected_ppm_glob)}
    args = [
        str(exe),
        f"--net-port={http_port}",
        f"--net-line-port={line_port}",
    ]
    # Hide the console window on Windows; keep stderr piped so we can
    # debug if the engine refuses to start.
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000  # CREATE_NO_WINDOW
    else:
        creationflags = 0
    # Both pipes go to DEVNULL: an undrained PIPE will deadlock the
    # engine once Win32's pipe buffer (4 KB) fills, which it does
    # fast for the OptiX path's chatty init logs. If you need engine
    # output, redirect via the engine's own log facilities.
    proc = subprocess.Popen(
        args, cwd=str(workdir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    try:
        deadline = time.monotonic() + timeout_s
        captured: Path | None = None
        while time.monotonic() < deadline:
            current = {p.name for p in captures_dir.glob(expected_ppm_glob)}
            new = current - pre
            if new:
                # First new match wins (one-shot capture mode).
                name = sorted(new)[0]
                captured = captures_dir / name
                break
            time.sleep(0.25)
        if captured is None:
            print(f"[run_capture] TIMEOUT: no '{expected_ppm_glob}' "
                  f"appeared in {captures_dir} after {timeout_s:.1f}s",
                  file=sys.stderr)
        return captured
    finally:
        # Polite Ctrl+Break first, then hard kill if it doesn't exit.
        try:
            if os.name == "nt":
                import signal
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception as e:
            print(f"[run_capture] cleanup warning: {e}", file=sys.stderr)
        # Restore any pre-existing autoexec.cfg, then clean up the
        # backup. If we never made a backup, just delete our temporary
        # autoexec.
        try:
            if backup_made:
                shutil.move(str(backup), str(autoexec))
            else:
                autoexec.unlink(missing_ok=True)
        except Exception as e:
            print(f"[run_capture] cleanup warning: could not restore "
                  f"autoexec.cfg ({e}). User's original (if any) is "
                  f"in {backup}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                        help=f"Path to demont.exe (default: {DEFAULT_EXE})")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        help="r_denoiser values to compare. Default: "
                             f"{' '.join(DEFAULT_MODES)}")
    parser.add_argument("--frame", type=int, default=60,
                        help="r_capture_frame_at target frame "
                             "(must be > seed). Default: 60")
    parser.add_argument("--seed", type=int, default=1,
                        help="r_capture_seed: forces frame_index_ to this "
                             "value at startup so runs are bitwise "
                             "deterministic. Must be < --frame. Default: 1")
    parser.add_argument("--spp", type=int, default=1,
                        help="r_spp override per run (default: 1)")
    parser.add_argument("--bounces", type=int, default=8,
                        help="r_max_bounces override per run (default: 8)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Max seconds to wait per launch "
                             "(default: 30s -- includes pipeline build).")
    parser.add_argument("--http-port", type=int, default=DEFAULT_PORTS[0],
                        help=f"--net-port value (default: {DEFAULT_PORTS[0]})")
    parser.add_argument("--line-port", type=int, default=DEFAULT_PORTS[1],
                        help=f"--net-line-port value (default: {DEFAULT_PORTS[1]})")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "captures" / "compare",
                        help="Output directory for renamed PPMs + CSV "
                             "(default: <repo>/captures/compare)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV output path (default: <out-dir>/results.csv)")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Skip a mode if its output PPM already exists "
                             "in --out-dir. Useful for incremental dev.")
    args = parser.parse_args()

    if args.seed >= args.frame:
        print(f"ERROR: --seed ({args.seed}) must be < --frame ({args.frame}). "
              f"r_capture_seed sets frame_index_ at startup; if the seed is "
              f"already past the target, the capture will never fire.",
              file=sys.stderr)
        return 2

    if not args.exe.exists():
        print(f"ERROR: demont.exe not found at {args.exe}", file=sys.stderr)
        print("       Build with bootstrap_win_release.bat / "
              "build_win_release.bat first.", file=sys.stderr)
        return 2

    workdir = args.exe.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (args.out_dir / "results.csv")

    # ---- Phase 1: capture each mode --------------------------------------
    captured_paths: dict[str, Path] = {}
    for mode in args.modes:
        out_path = args.out_dir / f"{mode}.ppm"
        if args.keep_existing and out_path.exists():
            print(f"[{mode}] using existing {out_path}")
            captured_paths[mode] = out_path
            continue
        cfg = [
            f"r_denoiser {mode}",
            f"r_spp {args.spp}",
            f"r_max_bounces {args.bounces}",
            f"r_capture_seed {args.seed}",
            f"r_capture_frame_at {args.frame}",
        ]
        glob = f"capture_{args.frame:06d}_{mode}_*.ppm"
        print(f"[{mode}] launching demont.exe -> capture frame "
              f"{args.frame} ...")
        captured = run_capture(args.exe, workdir, cfg, glob,
                               args.timeout,
                               args.http_port, args.line_port)
        if captured is None:
            print(f"[{mode}] FAILED: no capture produced", file=sys.stderr)
            return 1
        shutil.move(str(captured), str(out_path))
        print(f"[{mode}]   -> {out_path}")
        captured_paths[mode] = out_path

    # ---- Phase 2: pairwise RMSE / SSIM -----------------------------------
    print("\nLoading PPMs ...")
    decoded: dict[str, tuple[int, int, bytes]] = {}
    for mode, path in captured_paths.items():
        decoded[mode] = parse_ppm(path)
        w, h, _ = decoded[mode]
        print(f"  {mode}: {w}x{h}")

    # Sanity: all the same size?
    sizes = {(w, h) for w, h, _ in decoded.values()}
    if len(sizes) > 1:
        print(f"ERROR: PPMs have mixed dimensions {sizes}; can't compare.",
              file=sys.stderr)
        return 1

    print("\nPairwise comparison ...")
    rows: list[dict[str, str]] = []
    modes = list(decoded.keys())
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            wa, ha, da = decoded[a]
            _, _,  db  = decoded[b]
            r = rmse(da, db)
            s = ssim_if_available(da, db, wa, ha)
            row = {
                "mode_a":   a,
                "mode_b":   b,
                "width":    str(wa),
                "height":   str(ha),
                "rmse_8bit": f"{r:.4f}",
                "ssim_luma": "n/a" if s is None else f"{s:.6f}",
            }
            rows.append(row)
            ssim_str = "n/a (numpy missing)" if s is None else f"SSIM={s:.6f}"
            print(f"  {a:>16} vs {b:<16}  RMSE={r:7.3f}  {ssim_str}")

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

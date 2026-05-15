#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
"""Probe `screenshot ... swap` on the OptiX and SVGF denoiser paths to
verify that bloom is captured in both. Spawns demont.exe on non-default
ports, drives r_denoiser / r_bloom over the line-protocol, captures
swap-target screenshots, and reports mean RGB deltas.

Regression coverage for the bug fixed on feature/optix-swap-screenshot-bloom:
pre-fix the OptiX bloom-on-vs-off mean RGB delta was ~0 because the
capture sampled the engine cb (pre-finalize, no bloom composite); post-fix
it matches the SVGF delta (~+15) because the capture moved into the OptiX
private cb after EncodeDenoiseFinalize.

Typical invocation:
    python scripts/probe_swap_bloom.py
    python scripts/probe_swap_bloom.py --exe build/win-clang-release/src/app/demont.exe

Sibling to scripts/compare_denoisers.py (which compares accum / denoise_color
RMSE between modes); this one specifically tests swap-target screenshot
correctness, which the RMSE script doesn't reach."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_ppm(path: Path) -> tuple[int, int, bytes]:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"{path}: not P6 ({magic!r})")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        if int(line.strip()) != 255:
            raise ValueError(f"{path}: maxval != 255")
        data = f.read()
    if len(data) != w * h * 3:
        raise ValueError(f"{path}: short data")
    return w, h, data


def mean_rgb(rgb: bytes) -> tuple[float, float, float]:
    n = len(rgb) // 3
    r = sum(rgb[0::3]) / n
    g = sum(rgb[1::3]) / n
    b = sum(rgb[2::3]) / n
    return r, g, b


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def send_cmd(host: str, port: int, cmd: str, recv_timeout_s: float = 2.0) -> str:
    """Send one newline-delimited command and read whatever response
    comes back within recv_timeout_s. The line-protocol replies right
    after the main-thread Drain runs (typically next frame), so a short
    timeout is enough."""
    with socket.create_connection((host, port), timeout=2.0) as s:
        s.sendall((cmd + "\n").encode("utf-8"))
        s.settimeout(recv_timeout_s)
        chunks: list[bytes] = []
        try:
            while True:
                data = s.recv(4096)
                if not data:
                    break
                chunks.append(data)
        except socket.timeout:
            pass
    return b"".join(chunks).decode("utf-8", errors="replace")


def wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            # Engine writes header+pixels in a single fwrite, but give
            # the OS a moment to flush before we read.
            time.sleep(0.05)
            return True
        time.sleep(0.1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exe", type=Path,
        default=REPO_ROOT / "build" / "win-debug" / "src" / "app" / "demont.exe")
    ap.add_argument("--http-port",  type=int, default=27997)
    ap.add_argument("--line-port",  type=int, default=27998)
    ap.add_argument("--out-dir",    type=Path,
                    default=REPO_ROOT / "captures" / "swap_bloom_probe",
                    help="Output dir for screenshot PPMs + demont stderr "
                         "log (default: <repo>/captures/swap_bloom_probe).")
    ap.add_argument("--settle-after-denoiser-s", type=float, default=2.5,
                    help="Sleep after r_denoiser <kind> to let lazy init "
                         "finish (OptiX state buffer is allocated on the "
                         "first Denoise call).")
    ap.add_argument("--settle-after-bloom-s", type=float, default=0.5,
                    help="Sleep after r_bloom toggle so the next-frame "
                         "bloom dispatch has happened before screenshot.")
    ap.add_argument("--screenshot-timeout-s", type=float, default=10.0,
                    help="Max wait for the swap PPM to appear on disk.")
    args = ap.parse_args()

    if not args.exe.exists():
        print(f"ERROR: {args.exe} not found. Build win-debug first.",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous PPMs so wait_for_file isn't fooled.
    for p in args.out_dir.glob("*.ppm"):
        p.unlink()

    workdir = args.exe.parent
    cli = [
        str(args.exe),
        f"--net-port={args.http_port}",
        f"--net-line-port={args.line_port}",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP
                         | 0x08000000)  # CREATE_NO_WINDOW
    print(f"[probe] launching: {' '.join(cli)}")
    stderr_log = args.out_dir / "demont_stderr.log"
    stderr_fp = open(stderr_log, "wb")
    proc = subprocess.Popen(
        cli, cwd=str(workdir),
        stdout=subprocess.DEVNULL, stderr=stderr_fp,
        creationflags=creationflags,
    )

    rc = 1
    try:
        if not wait_for_port("127.0.0.1", args.line_port, timeout_s=30.0):
            print(f"[probe] FAIL: line-protocol port {args.line_port} never opened",
                  file=sys.stderr)
            return 1
        print(f"[probe] line-protocol up on 127.0.0.1:{args.line_port}")
        # Headstart: even after the port is up, the renderer is still
        # building shader pipelines on the worker thread. A short pause
        # before issuing the first r_denoiser lets the path-tracer
        # pipeline finish so the SupportsDenoise() gate inside the
        # OptiX route doesn't bail.
        time.sleep(2.0)

        matrix = [
            ("svgf_atrous", 0, "svgf_off"),
            ("svgf_atrous", 1, "svgf_on"),
            ("optix_hdr",   0, "optix_off"),
            ("optix_hdr",   1, "optix_on"),
        ]
        results: dict[str, tuple[float, float, float]] = {}
        sticky_denoiser: str | None = None
        for denoiser, bloom, tag in matrix:
            if denoiser != sticky_denoiser:
                print(f"[probe] r_denoiser {denoiser}")
                resp = send_cmd("127.0.0.1", args.line_port,
                                f"r_denoiser {denoiser}")
                if resp.strip():
                    print(f"        -> {resp.strip()}")
                time.sleep(args.settle_after_denoiser_s)
                sticky_denoiser = denoiser
            print(f"[probe] r_bloom {bloom}")
            resp = send_cmd("127.0.0.1", args.line_port, f"r_bloom {bloom}")
            if resp.strip():
                print(f"        -> {resp.strip()}")
            time.sleep(args.settle_after_bloom_s)

            ppm = args.out_dir / f"{tag}.ppm"
            # Forward slashes work in Win32 fopen; avoids quoting.
            ppm_arg = str(ppm).replace("\\", "/")
            print(f"[probe] screenshot {ppm_arg} swap")
            resp = send_cmd("127.0.0.1", args.line_port,
                            f"screenshot {ppm_arg} swap")
            if resp.strip():
                print(f"        -> {resp.strip()}")
            if not wait_for_file(ppm, args.screenshot_timeout_s):
                print(f"[probe] FAIL: {ppm} never appeared", file=sys.stderr)
                return 1
            w, h, data = parse_ppm(ppm)
            rgb = mean_rgb(data)
            results[tag] = rgb
            print(f"        -> {ppm.name} {w}x{h}, "
                  f"mean RGB = ({rgb[0]:6.2f}, {rgb[1]:6.2f}, {rgb[2]:6.2f})")

        # ---- Report ------------------------------------------------------
        def delta(on_tag: str, off_tag: str) -> tuple[float, float, float]:
            on  = results[on_tag]
            off = results[off_tag]
            return (on[0] - off[0], on[1] - off[1], on[2] - off[2])

        svgf_d  = delta("svgf_on",  "svgf_off")
        optix_d = delta("optix_on", "optix_off")
        print()
        print("=" * 60)
        print("Mean RGB delta (bloom-on minus bloom-off):")
        print(f"  SVGF  : R={svgf_d[0]:+6.2f}  G={svgf_d[1]:+6.2f}  B={svgf_d[2]:+6.2f}")
        print(f"  OptiX : R={optix_d[0]:+6.2f}  G={optix_d[1]:+6.2f}  B={optix_d[2]:+6.2f}")
        print()
        # Heuristic pass: OptiX delta should be within ~3 units of SVGF
        # delta on each channel. A pre-fix OptiX read would be ~0; the
        # fixed read should be close to SVGF's value.
        worst = max(abs(svgf_d[i] - optix_d[i]) for i in range(3))
        if worst < 3.0:
            print(f"PASS: OptiX bloom matches SVGF (worst-channel diff "
                  f"{worst:.2f} < 3.0)")
            rc = 0
        else:
            print(f"FAIL: OptiX bloom does NOT match SVGF (worst-channel "
                  f"diff {worst:.2f} >= 3.0)")
            rc = 1
        return rc
    finally:
        try:
            send_cmd("127.0.0.1", args.line_port, "quit", recv_timeout_s=0.5)
        except OSError:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    proc.kill()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        try:
            stderr_fp.close()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())

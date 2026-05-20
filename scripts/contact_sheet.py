#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
"""Visual-QA contact-sheet harness for the DeMonT path tracer.

Renders every headline feature's showcase scene headlessly via
`pt_render_one_frame`, then tiles the captures into a single labeled
montage PNG (`build/contact_sheet.png`) plus individual per-feature
tiles (`build/contact_sheet_tiles/*.png`). The whole feature set becomes
eyeballable for quality at a glance -- the fastest way to spot a tile
that is washed out, broken, or noisy and is therefore not-yet-AAA.

WHY PYTHON + PILLOW (not a C++/stb tool):
  Pillow is already importable on the dev box and the CI images, the
  repo already ships sibling driver scripts under scripts/ (the
  compare_denoisers / probe_swap_bloom pattern), and tiling + caption
  text is a few lines of PIL versus a whole new CMake target linking
  stb_image_write + a bitmap font. A pure-script tool also keeps the
  engine/shader/CMake surface untouched, so this lands as new files only
  with zero risk of a sibling merge conflict.

WHAT IT DRIVES:
  Each showcase cell mirrors the canonical render parameters the
  golden-image regression matrix uses for that scene
  (tests/CMakeLists.txt: backend, denoiser, frame budget, and the
  feature-enabling --extra cvars -- e.g. clouds_godrays needs
  `r_godrays 1`, the tone-map cells need `r_tonemap_op <op>`). So the
  contact sheet shows each feature exactly as the regression suite
  expects it to look, captioned with the scene name + the key cvars
  under each tile.

TYPICAL INVOCATION:
    python3 scripts/contact_sheet.py
    python3 scripts/contact_sheet.py --backend metal --tile-width 320
    python3 scripts/contact_sheet.py --only ocean_foam,car_paint,sky_hosek
    python3 scripts/contact_sheet.py --fast        # clamp frame budgets

The default backend is metal (the dev box + the matrix's primary GPU
combo). Override with --backend software for the deterministic CPU lane
(some GPU-only features -- god rays, raymarched clouds, ocean, fog,
aurora -- fall back to their gate-off path on software and will render
flat; they are skipped automatically when --backend software unless
--force-all is passed).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Showcase table
# ---------------------------------------------------------------------------
# One Cell per headline feature. The (scene, backend, denoiser, frames,
# extra) tuple mirrors the canonical golden-matrix parameters for that
# scene (tests/CMakeLists.txt) so the tile shows the feature the way the
# regression suite expects it. `group` controls grid ordering / section
# grouping; `caption` is the human label drawn under the tile; `cvars`
# is the short "key cvars" hint line drawn beneath the caption.
#
# `gpu_only` marks features whose effect only fires on a compute-capable
# backend (Metal/Vulkan): god rays, raymarched clouds, the FFT ocean,
# height fog, and the aurora composite all collapse to a no-op gate-off
# path on the software tracer, so a software run would render them flat /
# identical to the plain scene (no signal). Those cells are skipped on
# --backend software unless --force-all is given.
#
# `late_exec` (physics fixtures) names a sibling *_late.cfg the scene's
# fixture references by a *relative* path; the golden matrix re-points it
# at an absolute path via --extra because each render runs in its own
# CWD. We do the same so the deferred drop fires.


@dataclass
class Cell:
    scene: str
    caption: str
    cvars: str = ""              # short "key cvars" hint drawn under the caption
    group: str = "misc"
    backend: str | None = None   # None => use the run's default backend
    denoiser: str = "off"
    frames: int = 64
    extra: list[str] = field(default_factory=list)
    spp: int | None = None
    gpu_only: bool = False
    late_exec: str | None = None  # sibling *_late.cfg basename (physics)

    def stem(self) -> str:
        """Filename-safe identifier for the tile PNG + cell selection."""
        return self.scene


# The curated set. Ordered by section so the montage reads top-to-bottom
# as: geometry, sky/atmosphere, volumetrics, water, materials, physics,
# night, tone-mapping. Frame budgets and --extra come straight from the
# golden-matrix cells in tests/CMakeLists.txt.
SHOWCASE: list[Cell] = [
    # --- Geometry -------------------------------------------------------
    Cell("cornell_csg", "CSG drilled cube", "metal off - 64f",
         group="geometry", denoiser="off", frames=64),
    Cell("sdf_smin_row", "SDF smooth-min blend row", "metal off - 64f",
         group="geometry", denoiser="off", frames=64),
    Cell("sdf_fractals", "SDF Mandelbulb fractal", "metal off - 64f",
         group="geometry", denoiser="off", frames=64),

    # --- Sky / atmosphere ----------------------------------------------
    Cell("sky_hosek", "Hosek-Wilkie sky (midday)", "metal off - hosek",
         group="sky", denoiser="off", frames=64),
    Cell("procedural_noon", "Procedural sky (noon)", "metal svgf - sun@60",
         group="sky", denoiser="svgf_atrous", frames=64),
    Cell("procedural_evening", "Procedural sky (sunset)", "metal svgf - sun@5",
         group="sky", denoiser="svgf_atrous", frames=64),
    Cell("procedural_dawn", "Procedural sky (dawn)", "metal svgf - sun@-2",
         group="sky", denoiser="svgf_atrous", frames=64),

    # --- Volumetrics ---------------------------------------------------
    Cell("clouds_godrays", "God rays / crepuscular shafts", "metal svgf - r_godrays 1",
         group="volumetric", denoiser="svgf_atrous", frames=64,
         extra=["r_godrays 1"], gpu_only=True),
    Cell("clouds_raymarched", "Raymarched volumetric clouds", "metal svgf - raymarched",
         group="volumetric", denoiser="svgf_atrous", frames=64, gpu_only=True),
    Cell("height_fog", "Exponential height fog", "metal svgf - r_fog 1",
         group="volumetric", denoiser="svgf_atrous", frames=64, gpu_only=True),
    Cell("smoke_phase3_wind", "SPH smoke + wind", "metal off - 90f wind",
         group="volumetric", denoiser="off", frames=90),

    # --- Water ---------------------------------------------------------
    Cell("ocean_fft", "FFT ocean (Tessendorf)", "metal svgf - 120f sunset",
         group="water", denoiser="svgf_atrous", frames=120, gpu_only=True),
    Cell("ocean_foam", "Ocean foam / whitecaps", "metal svgf - 16 m/s wind",
         group="water", denoiser="svgf_atrous", frames=120, gpu_only=True),
    Cell("water_pool", "Analytic water pool", "metal off - 64f",
         group="water", denoiser="off", frames=64),

    # --- Materials -----------------------------------------------------
    Cell("brushed_metal", "Anisotropic brushed metal", "metal off - aniso GGX",
         group="material", denoiser="off", frames=64),
    Cell("car_paint", "Car paint (clearcoat)", "metal off - coat 0.9",
         group="material", denoiser="off", frames=64),
    Cell("subsurface", "Subsurface scattering (backlit)", "metal off - SSS wax",
         group="material", denoiser="off", frames=64),
    Cell("spectral_prism", "Spectral dispersion (prism)", "metal off - r_spectral 1",
         group="material", denoiser="off", frames=64),
    Cell("dof_bokeh", "Depth of field / bokeh", "metal off - 85mm f/2",
         group="material", denoiser="off", frames=64),
    Cell("pbr_textured", "PBR textured (albedo/normal/rough)", "metal off - 64f",
         group="material", denoiser="off", frames=64),

    # --- Physics -------------------------------------------------------
    Cell("phys_rb_smoke", "Rigid bodies (mixed)", "metal off - 60f drop",
         group="physics", denoiser="off", frames=60,
         late_exec="phys_rb_smoke_late.cfg"),
    Cell("phys_rb_demo", "Rigid bodies (tower demo)", "metal off - 60f drop",
         group="physics", denoiser="off", frames=60,
         late_exec="phys_rb_demo_late.cfg"),
    Cell("motion_blur_mesh", "Mesh motion blur", "metal off - shutter smear",
         group="physics", denoiser="off", frames=64,
         late_exec="motion_blur_mesh_late.cfg"),

    # --- Night ---------------------------------------------------------
    Cell("bsc_night", "BSC star catalog night", "metal svgf - real stars",
         group="night", denoiser="svgf_atrous", frames=64),
    Cell("bsc_night_clouds", "Stars + cloud occlusion", "metal svgf - night clouds",
         group="night", denoiser="svgf_atrous", frames=64),
    Cell("lunar_night", "Lunar night (moon NEE)", "metal svgf - moon disc",
         group="night", denoiser="svgf_atrous", frames=64),
    Cell("aurora_smoke", "Aurora borealis", "metal svgf - r_aurora 1",
         group="night", denoiser="svgf_atrous", frames=32, gpu_only=True),

    # --- Tone-mapping (same scene, three operators side-by-side) -------
    Cell("procedural_noon", "Tonemap: ACES (default)", "software off - aces",
         group="tonemap", backend="software", denoiser="off", frames=64,
         extra=["r_tonemap_op aces"]),
    Cell("procedural_noon", "Tonemap: AgX", "software off - agx",
         group="tonemap", backend="software", denoiser="off", frames=64,
         extra=["r_tonemap_op agx"]),
    Cell("procedural_noon", "Tonemap: Khronos PBR Neutral", "software off - khronos",
         group="tonemap", backend="software", denoiser="off", frames=64,
         extra=["r_tonemap_op khronos_pbr_neutral"]),
]


# ---------------------------------------------------------------------------
# Tile naming
# ---------------------------------------------------------------------------
def cell_id(cell: Cell, backend: str) -> str:
    """Unique, filename-safe id per cell. Includes the resolved backend +
    denoiser, plus a TAG derived from the last r_tonemap_op in --extra (so
    the three tone-map cells that share scene/backend/denoiser don't
    collide on disk). Mirrors the golden-matrix `__<scene>__<backend>__<denoiser>[__tag]`
    stem convention."""
    eff_backend = cell.backend or backend
    parts = [cell.scene, eff_backend, cell.denoiser]
    tag = ""
    for e in cell.extra:
        toks = e.split()
        if len(toks) >= 2 and toks[0] == "r_tonemap_op":
            tag = toks[1]
    if tag:
        parts.append(tag)
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Render-tool resolution
# ---------------------------------------------------------------------------
def default_render_tool() -> Path:
    """Best-effort path to the pt_render_one_frame binary in a standard
    build tree. Prefers mac-release, then any build/*/tools tree. The
    caller can always override with --render-tool."""
    exe = "pt_render_one_frame" + (".exe" if os.name == "nt" else "")
    preferred = REPO_ROOT / "build" / "mac-release" / "tools" / "pt_render_one_frame" / exe
    if preferred.exists():
        return preferred
    # Fall back to the first match under any build/<preset>/ tree.
    build_root = REPO_ROOT / "build"
    if build_root.is_dir():
        for cand in sorted(build_root.glob(f"*/tools/pt_render_one_frame/{exe}")):
            if cand.exists():
                return cand
    return preferred  # return the most-likely path; caller errors if absent


# ---------------------------------------------------------------------------
# Render one cell
# ---------------------------------------------------------------------------
def render_cell(cell: Cell, *, render_tool: Path, backend: str, frames_cap: int | None,
                tiles_dir: Path, timeout: float, demont: Path | None) -> tuple[bool, Path, str]:
    """Run pt_render_one_frame for one cell. Returns (ok, png_path, note).
    `note` carries a short status string for the manifest / stderr."""
    eff_backend = cell.backend or backend
    out_png = tiles_dir / f"{cell_id(cell, backend)}.png"
    scene_cfg = REPO_ROOT / "tests" / "goldens" / "scenes" / f"{cell.scene}.cfg"
    if not scene_cfg.is_file():
        return False, out_png, f"scene cfg missing: {scene_cfg}"

    frames = cell.frames
    if frames_cap is not None:
        frames = min(frames, frames_cap)

    # Assemble --extra. The fixture's own body loads first (the wrapper
    # inlines it verbatim); our --extra is appended AFTER it, so a
    # last-writer cvar here wins -- which is exactly how the golden matrix
    # re-points the physics fixtures' relative late-exec at an absolute
    # path. The wrapper splits --extra on newlines, so join with '\n'.
    extra_cmds = list(cell.extra)
    if cell.late_exec:
        abs_late = REPO_ROOT / "tests" / "goldens" / "scenes" / cell.late_exec
        extra_cmds.append(f"pt_smoke_late_exec {abs_late}")
    extra_joined = "\n".join(extra_cmds)

    args: list[str] = [
        str(render_tool),
        "--scene", str(scene_cfg),
        "--backend", eff_backend,
        "--denoiser", cell.denoiser,
        "--frames", str(frames),
        "--out", str(out_png),
    ]
    if cell.spp is not None:
        args += ["--spp", str(cell.spp)]
    if extra_joined:
        args += ["--extra", extra_joined]
    if demont is not None:
        args += ["--demont", str(demont)]

    # Per-cell working directory so the engine's archived demont.cfg /
    # autoexec.cfg don't leak across cells (the wrapper also scrubs them,
    # but an isolated CWD is belt-and-braces and keeps the repo root clean
    # of the saved demont.cfg the engine writes on exit).
    cell_workdir = tiles_dir / "_workdir" / cell_id(cell, backend)
    cell_workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            args, cwd=str(cell_workdir),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, out_png, f"timeout after {timeout:.0f}s"
    dt = time.monotonic() - t0

    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail_str = tail[-1] if tail else f"exit {proc.returncode}"
        return False, out_png, f"render failed ({tail_str})"
    if not out_png.exists():
        return False, out_png, "render produced no PNG"
    return True, out_png, f"ok {dt:.1f}s {frames}f"


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def build_montage(rendered: list[tuple[Cell, Path | None, str]], *,
                  out_path: Path, tile_w: int, backend: str, columns: int) -> None:
    """Tile the rendered PNGs into a single captioned montage. Each tile
    is the scene capture scaled to tile_w (preserving aspect), with a
    caption bar below it (scene title + key-cvars hint). A failed cell
    gets a dark placeholder tile so the sheet is always complete."""
    from PIL import Image, ImageDraw, ImageFont

    # --- font ----------------------------------------------------------
    def load_font(size: int):
        # Try a few common TrueType faces so captions are crisp; fall back
        # to PIL's bundled bitmap font (always present) if none resolve.
        candidates = [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf",
        ]
        for c in candidates:
            try:
                return ImageFont.truetype(c, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    title_font = load_font(15)
    sub_font = load_font(12)
    head_font = load_font(20)

    # Caption-bar height: two text rows + padding.
    cap_h = 42
    pad = 10
    head_h = 56            # top banner
    sec_h = 26             # per-section label row

    # Group cells in showcase order, preserving section grouping.
    groups: list[tuple[str, list[tuple[Cell, Path | None, str]]]] = []
    for cell, png, note in rendered:
        if not groups or groups[-1][0] != cell.group:
            groups.append((cell.group, []))
        groups[-1][1].append((cell, png, note))

    # We render each section's tiles flowing left-to-right across `columns`
    # then wrap. Compute a uniform tile height from the widest aspect we
    # see (scenes are 512x384, 768x432, 1024x768 -> normalise to tile_w and
    # pick the tallest resulting height so every tile slot is the same).
    def scaled_size(png: Path | None) -> tuple[int, int]:
        if png is not None and png.exists():
            with Image.open(png) as im:
                w, h = im.size
            th = max(1, round(tile_w * h / w))
            return tile_w, th
        return tile_w, round(tile_w * 384 / 512)

    all_heights = [scaled_size(png)[1] for _, png, _ in rendered] or [round(tile_w * 384 / 512)]
    tile_img_h = max(all_heights)
    tile_total_h = tile_img_h + cap_h

    # Lay out: figure total rows across all sections.
    def section_rows(n: int) -> int:
        return (n + columns - 1) // columns

    total_grid_rows = sum(section_rows(len(cells)) for _, cells in groups)
    n_sections = len(groups)

    sheet_w = pad + columns * (tile_w + pad)
    sheet_h = (head_h
               + n_sections * sec_h
               + total_grid_rows * (tile_total_h + pad)
               + pad)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    # Top banner.
    n_ok = sum(1 for _, png, _ in rendered if png is not None and png.exists())
    banner = (f"DeMonT path tracer -- visual-QA contact sheet   "
              f"({n_ok}/{len(rendered)} tiles, backend={backend})")
    draw.rectangle([0, 0, sheet_w, head_h - 8], fill=(40, 42, 52))
    draw.text((pad, 14), banner, fill=(235, 235, 240), font=head_font)

    y = head_h
    for group_name, cells in groups:
        # Section label row.
        draw.rectangle([0, y, sheet_w, y + sec_h - 4], fill=(52, 54, 66))
        draw.text((pad, y + 4), group_name.upper(), fill=(180, 200, 255), font=title_font)
        y += sec_h

        # Tiles for this section.
        col = 0
        row_top = y
        for cell, png, note in cells:
            x = pad + col * (tile_w + pad)
            ty = row_top
            # Image area.
            if png is not None and png.exists():
                with Image.open(png) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    th = max(1, round(tile_w * h / w))
                    im = im.resize((tile_w, th), Image.LANCZOS)
                # Vertically center within the uniform tile_img_h slot.
                slot = Image.new("RGB", (tile_w, tile_img_h), (12, 12, 14))
                slot.paste(im, (0, max(0, (tile_img_h - th) // 2)))
                sheet.paste(slot, (x, ty))
            else:
                # Failure placeholder.
                draw.rectangle([x, ty, x + tile_w, ty + tile_img_h], fill=(60, 20, 24))
                draw.line([x, ty, x + tile_w, ty + tile_img_h], fill=(120, 40, 44), width=2)
                draw.line([x + tile_w, ty, x, ty + tile_img_h], fill=(120, 40, 44), width=2)
                msg = "RENDER FAILED"
                draw.text((x + 8, ty + tile_img_h // 2 - 8), msg,
                          fill=(255, 180, 180), font=title_font)

            # Caption bar.
            cy = ty + tile_img_h
            draw.rectangle([x, cy, x + tile_w, cy + cap_h], fill=(34, 36, 44))
            draw.text((x + 6, cy + 4), cell.caption, fill=(240, 240, 245), font=title_font)
            sub = cell.cvars if (png is not None and png.exists()) else note
            draw.text((x + 6, cy + 23), sub, fill=(150, 165, 195), font=sub_font)

            col += 1
            if col >= columns:
                col = 0
                row_top += tile_total_h + pad
        # Advance y past whatever rows this section consumed.
        rows_used = section_rows(len(cells))
        y = row_top + ((tile_total_h + pad) if col != 0 else 0)
        # If the last row was partial (col != 0) we already accounted the
        # final row_top; if it was full (col == 0) row_top already points
        # past it. Normalise:
        if col == 0:
            y = row_top
        else:
            y = row_top + tile_total_h + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"\n[contact_sheet] montage -> {out_path} ({sheet_w}x{sheet_h})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render-tool", type=Path, default=None,
                        help="Path to pt_render_one_frame (default: autodetect "
                             "under build/mac-release, then build/*/).")
    parser.add_argument("--demont", type=Path, default=None,
                        help="Override path to the demont binary handed to "
                             "pt_render_one_frame (default: tool resolves its "
                             "sibling).")
    parser.add_argument("--backend", default="metal",
                        choices=["metal", "software", "vulkan"],
                        help="Default backend for cells that don't pin their "
                             "own (default: metal).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "build" / "contact_sheet.png",
                        help="Montage PNG output path "
                             "(default: build/contact_sheet.png).")
    parser.add_argument("--tiles-dir", type=Path,
                        default=REPO_ROOT / "build" / "contact_sheet_tiles",
                        help="Directory for individual tile PNGs "
                             "(default: build/contact_sheet_tiles).")
    parser.add_argument("--tile-width", type=int, default=320,
                        help="Tile width in px (height follows scene aspect). "
                             "Default 320.")
    parser.add_argument("--columns", type=int, default=4,
                        help="Tiles per row in the montage (default: 4).")
    parser.add_argument("--fast", action="store_true",
                        help="Clamp every cell's frame budget to --fast-frames "
                             "for a quicker (noisier) preview run.")
    parser.add_argument("--fast-frames", type=int, default=24,
                        help="Frame cap when --fast is set (default: 24).")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-cell render timeout in seconds (default: 180).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated scene names to render (others "
                             "skipped). Matches Cell.scene.")
    parser.add_argument("--skip", default=None,
                        help="Comma-separated scene names to skip.")
    parser.add_argument("--force-all", action="store_true",
                        help="Render gpu_only cells even on --backend software "
                             "(they will render flat -- for debugging only).")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Reuse a tile PNG that already exists in "
                             "--tiles-dir instead of re-rendering it.")
    parser.add_argument("--montage-only", action="store_true",
                        help="Skip rendering; rebuild the montage from tiles "
                             "already present in --tiles-dir.")
    args = parser.parse_args()

    # PIL is required for compositing -- fail early with a clear hint.
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("ERROR: Pillow (PIL) is required for the montage step.\n"
              "       Install it with:  python3 -m pip install Pillow\n"
              "       (Individual tile PNGs can still be produced, but this "
              "harness composites them in one pass.)", file=sys.stderr)
        return 2

    render_tool = args.render_tool or default_render_tool()
    if not args.montage_only and not render_tool.exists():
        print(f"ERROR: pt_render_one_frame not found at {render_tool}\n"
              f"       Build it first, e.g.:\n"
              f"         cmake --preset mac-release\n"
              f"         cmake --build build/mac-release --target "
              f"demont pt_render_one_frame -j 8\n"
              f"       or pass --render-tool PATH.", file=sys.stderr)
        return 2

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    skip = set(s.strip() for s in args.skip.split(",")) if args.skip else set()

    # Filter the showcase set.
    cells: list[Cell] = []
    skipped_msgs: list[str] = []
    for cell in SHOWCASE:
        if only is not None and cell.scene not in only:
            continue
        if cell.scene in skip:
            continue
        eff_backend = cell.backend or args.backend
        if cell.gpu_only and eff_backend == "software" and not args.force_all:
            skipped_msgs.append(f"  skip {cell.scene} ({cell.caption}): "
                                f"gpu-only feature, no signal on software")
            continue
        cells.append(cell)

    if not cells:
        print("ERROR: no cells selected to render (check --only / --skip).",
              file=sys.stderr)
        return 2

    frames_cap = args.fast_frames if args.fast else None

    print(f"[contact_sheet] backend={args.backend}  cells={len(cells)}  "
          f"tile_width={args.tile_width}  fast={args.fast}")
    if skipped_msgs:
        print("\n".join(skipped_msgs))
    print(f"[contact_sheet] render tool: {render_tool}")
    args.tiles_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[Cell, Path | None, str]] = []
    t_start = time.monotonic()
    for i, cell in enumerate(cells, 1):
        cid = cell_id(cell, args.backend)
        out_png = args.tiles_dir / f"{cid}.png"
        if args.montage_only or (args.keep_existing and out_png.exists()):
            ok = out_png.exists()
            note = "reused" if ok else "missing tile"
            status = "REUSE" if ok else "MISS "
            print(f"[{i:2d}/{len(cells)}] {status} {cid}")
            rendered.append((cell, out_png if ok else None, note))
            continue
        print(f"[{i:2d}/{len(cells)}] render {cid} ...", flush=True)
        ok, png, note = render_cell(
            cell, render_tool=render_tool, backend=args.backend,
            frames_cap=frames_cap, tiles_dir=args.tiles_dir,
            timeout=args.timeout, demont=args.demont)
        flag = "OK   " if ok else "FAIL "
        print(f"          {flag} {note}")
        rendered.append((cell, png if ok else None, note))

    total_dt = time.monotonic() - t_start
    n_ok = sum(1 for _, p, _ in rendered if p is not None and p.exists())
    n_fail = len(rendered) - n_ok
    print(f"\n[contact_sheet] rendered {n_ok}/{len(rendered)} tiles in "
          f"{total_dt:.1f}s ({n_fail} failed)")

    build_montage(rendered, out_path=args.out, tile_w=args.tile_width,
                  backend=args.backend, columns=args.columns)

    # Print a compact manifest so a CI log / operator sees per-tile status.
    print("\n[contact_sheet] manifest:")
    for cell, png, note in rendered:
        mark = "  ok " if (png is not None and png.exists()) else "FAIL "
        print(f"  {mark} {cell.group:>10} | {cell.caption:<34} | {note}")

    # Exit non-zero if ANY tile failed, so the harness is CI-gateable, but
    # always after writing the montage (a partial sheet is still useful).
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

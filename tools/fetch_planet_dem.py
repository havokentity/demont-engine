#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
"""Fetch ETOPO 2022 from NOAA/NCEI and bake the engine's elevation grid.

Planetary P4 (#258). The engine ships this tool rather than the data.

WHY THIS SOURCE
---------------
ETOPO 2022 (NOAA National Centers for Environmental Information,
doi:10.25921/fd45-gt74) is chosen over the alternatives for three reasons,
each of which was decisive on its own:

  * It is a **US Government work and therefore public domain**, so no
    attribution obligation attaches to an MIT-licensed engine. GEBCO is free
    but requires attribution and carries mixed third-party provenance, which
    is a licensing question a permissively-licensed renderer should not have
    to answer.
  * It is a **single grid carrying both topography AND bathymetry**. That is
    exactly what a shoreline needs, and it is what SRTM/NASADEM cannot give
    at any resolution -- they are land-only.
  * It ships at a size that can actually be fetched: the 60-arc-second
    global grid is ~450 MB of netCDF, against ~100 GB for SRTM tiles.

WHY THE DATA IS NOT COMMITTED, AND WHY *SOMETHING* REAL IS
----------------------------------------------------------
The full grid is far too large for a source repository. But a purely
procedural planet would repeat a mistake this project has already made and
already fixed twice: commit b2111dd replaced fabricated Hosek-Wilkie sky
coefficients with the published ArHosekSkyModel 1.4a dataset, and
assets/stars/BSC5.dat is the genuine Yale Bright Star Catalogue rather than
a plausible-looking scatter of points. From orbit -- the money shot of the
whole planetary arc -- a procedural body looks like noise, because it is.

So: this tool downloads the real grid and bakes a small "Earth lite"
derivative that IS committable, and the engine says so loudly when the file
is missing rather than silently substituting a procedural body.

USAGE
-----
    python3 tools/fetch_planet_dem.py                 # 2048x1024 Earth lite
    python3 tools/fetch_planet_dem.py --width 8192    # a finer local grid
    python3 tools/fetch_planet_dem.py --source my.nc  # bake from a local file

Requires: numpy. netCDF4 (or xarray, or GDAL) for the .nc source; if none is
installed the tool says which one to install rather than failing obscurely.

OUTPUT FORMAT (mirrored by src/renderer/Planet/ElevationField.h -- the two
are a wire format and there is no generated header between them):

    offset  size  field
    0       8     magic "PTDEM001"
    8       4     uint32 width          (columns, west to east)
    12      4     uint32 height         (rows, north to south)
    16      8     float64 scale_m
    24      8     float64 offset_m
    32      4     uint32 flags          (0)
    36      4     uint32 reserved
    40      ...   width*height little-endian uint16, row-major

    height_m = value * scale_m + offset_m

THE AFFINE, DERIVED
-------------------
  offset -11 000 m clears Challenger Deep's -10 935 m (Gardner, Armstrong &
  Calder 2014, "So, How Deep Is the Mariana Trench?", Marine Geodesy 37:1)
  by 65 m.
  scale 0.303 m/count puts the top of the range at
  -11000 + 65535*0.303 = +8 857.1 m, clearing Everest's 8 848.86 m (2020
  China/Nepal joint survey) by 8.2 m.
  Quantisation is therefore 30.3 cm.

  float16 is NOT usable: its 11-bit mantissa gives an 8 m ULP at 8 000 m,
  which would terrace the Himalayas.
"""

import argparse
import os
import struct
import sys
import urllib.request

# NCEI's public distribution point for the ETOPO 2022 ice-surface global
# grid at 60 arc-seconds. The 30" and 15" grids live under the sibling
# directories named for their resolution.
ETOPO_URL = ("https://www.ngdc.noaa.gov/thredds/fileServer/global/"
             "ETOPO2022/60s/60s_surface_elev_netcdf/"
             "ETOPO_2022_v1_60s_N90W180_surface.nc")

DEM_MAGIC   = b"PTDEM001"
DEM_SCALE_M = 0.303
DEM_OFFSET_M = -11000.0

# Real bounds, for the sanity check below.
EARTH_MIN_M = -10935.0
EARTH_MAX_M = 8848.86


def download(url, dest):
    if os.path.exists(dest):
        print(f"[fetch] {dest} already present ({os.path.getsize(dest)/1e6:.0f} MB)")
        return dest
    print(f"[fetch] downloading {url}")
    print("[fetch] this is ~450 MB and NCEI is not always fast")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r[fetch] {done/1e6:7.0f} / {total/1e6:.0f} MB")
                sys.stdout.flush()
    print()
    return dest


def read_grid(path):
    """Return a 2D float array of elevations, north-to-south, west-to-east."""
    try:
        import numpy as np
    except ImportError:
        sys.exit("[bake] numpy is required: pip install numpy")

    for loader in ("netCDF4", "xarray", "gdal"):
        try:
            if loader == "netCDF4":
                import netCDF4
                ds = netCDF4.Dataset(path)
                name = next(v for v in ("z", "elevation", "Band1") if v in ds.variables)
                arr = np.asarray(ds.variables[name][:], dtype=np.float64)
                lats = np.asarray(ds.variables.get("lat", ds.variables.get("y"))[:])
            elif loader == "xarray":
                import xarray as xr
                ds = xr.open_dataset(path)
                name = next(v for v in ("z", "elevation", "Band1") if v in ds)
                arr = np.asarray(ds[name].values, dtype=np.float64)
                lats = np.asarray(ds["lat"].values if "lat" in ds else ds["y"].values)
            else:
                from osgeo import gdal
                ds = gdal.Open(path)
                arr = np.asarray(ds.GetRasterBand(1).ReadAsArray(), dtype=np.float64)
                lats = None
        except ImportError:
            continue
        except StopIteration:
            sys.exit(f"[bake] {path} has no recognised elevation variable")
        # ETOPO's netCDF is south-to-north; the engine's grid is
        # north-to-south (row 0 is the north pole), matching the
        # equirectangular convention every image-based env map uses.
        if lats is not None and len(lats) > 1 and lats[0] < lats[-1]:
            arr = arr[::-1, :]
        return arr
    sys.exit("[bake] install one of: netCDF4, xarray, or GDAL "
             "(pip install netCDF4)")


def resample(arr, out_w, out_h):
    """Area-average down to (out_h, out_w).

    AREA average, not point sampling. The engine's fractal continuation
    extrapolates from the local RMS inter-texel relief, so a point-sampled
    downsample would hand it aliased noise and it would faithfully continue
    the aliasing. Averaging is also the physically right decimation of a
    height field: the mean elevation of a cell is a real quantity.
    """
    import numpy as np
    h, w = arr.shape
    if (h, w) == (out_h, out_w):
        return arr
    # Integer block average where it divides evenly (the common case:
    # 21600x10800 -> 2048x1024 does not, so fall back to an index-map mean).
    ys = (np.arange(out_h + 1) * h // out_h)
    xs = (np.arange(out_w + 1) * w // out_w)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for j in range(out_h):
        band = arr[ys[j]:max(ys[j + 1], ys[j] + 1), :]
        for i in range(out_w):
            out[j, i] = band[:, xs[i]:max(xs[i + 1], xs[i] + 1)].mean()
    return out


def bake(arr, dest):
    import numpy as np
    h, w = arr.shape
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    print(f"[bake] source range {lo:.1f} .. {hi:.1f} m over {w}x{h}")
    if lo < DEM_OFFSET_M or hi > DEM_OFFSET_M + 65535 * DEM_SCALE_M:
        print(f"[bake] WARNING: source exceeds the uint16 affine "
              f"[{DEM_OFFSET_M:.0f}, {DEM_OFFSET_M + 65535*DEM_SCALE_M:.1f}] m "
              f"and will be clamped")
    q = np.clip((arr - DEM_OFFSET_M) / DEM_SCALE_M, 0, 65535)
    q = np.nan_to_num(q, nan=(0.0 - DEM_OFFSET_M) / DEM_SCALE_M)
    q = q.astype("<u2")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        f.write(DEM_MAGIC)
        f.write(struct.pack("<II", w, h))
        f.write(struct.pack("<dd", DEM_SCALE_M, DEM_OFFSET_M))
        f.write(struct.pack("<II", 0, 0))
        f.write(q.tobytes())
    size = os.path.getsize(dest)
    km_per_texel = 2 * 3.14159265358979 * 6371008.8 / w / 1000.0
    print(f"[bake] wrote {dest} -- {size/1e6:.2f} MB, "
          f"{km_per_texel:.1f} km/texel at the equator")
    # Round-trip check: the quantised grid must still contain a real Earth.
    back = q.astype(np.float64) * DEM_SCALE_M + DEM_OFFSET_M
    err = float(np.nanmax(np.abs(back - np.clip(arr, DEM_OFFSET_M,
                                                DEM_OFFSET_M + 65535 * DEM_SCALE_M))))
    print(f"[bake] max quantisation error {err*100:.1f} cm "
          f"(the affine's own step is {DEM_SCALE_M*100:.1f} cm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="local ETOPO netCDF/GeoTIFF instead of downloading")
    ap.add_argument("--cache", default="build/etopo/ETOPO_2022_v1_60s_surface.nc",
                    help="where to keep the downloaded grid")
    ap.add_argument("--out", default="assets/planet/earth_lite.ptdem",
                    help="output .ptdem path")
    ap.add_argument("--width", type=int, default=2048,
                    help="output columns (height is width/2)")
    args = ap.parse_args()

    src = args.source or download(ETOPO_URL, args.cache)
    arr = read_grid(src)
    out_w = args.width
    out_h = args.width // 2
    print(f"[bake] resampling {arr.shape[1]}x{arr.shape[0]} -> {out_w}x{out_h} "
          f"(area average)")
    arr = resample(arr, out_w, out_h)
    bake(arr, args.out)
    print()
    print("Done. The engine picks it up via r_planet_dem; provenance and the")
    print("processing steps are recorded in assets/planet/PROVENANCE.md.")


if __name__ == "__main__":
    main()

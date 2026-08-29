#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
"""Fetch NASA surface-reflectance data and bake the engine's albedo raster.

Planetary land cover (#300). Sibling of tools/fetch_planet_dem.py, and it
inherits that tool's three hard-won guards deliberately -- see GUARDS below.

WHICH SOURCE, AND HOW THE CHOICE WAS MADE
------------------------------------------
The right product is **MODIS MCD43 BRDF/Albedo** (Schaaf et al. 2002,
Remote Sensing of Environment 83:135-148): measured, atmospherically
corrected, per-band reflectance. It is a NASA product and therefore a US
Government work in the public domain, so no attribution obligation attaches
to an MIT-licensed engine -- the same status that let this repo ship an
ETOPO derivative.

The obstacle is distribution, and it was measured rather than assumed:

    GET https://e4ftl01.cr.usgs.gov/MOTA/MCD43C3.061/            -> 404
    GET .../ladsweb.../MCD43C3/2020/182/MCD43C3.A2020182.*.hdf   -> 302 to
                                       urs.earthdata.nasa.gov/oauth/authorize
    GET https://modis-pds.s3.amazonaws.com/                      -> AccessDenied
    GET https://data.lpdaac.earthdatacloud.nasa.gov/...          -> 401

Every archive of the HDF granule is behind a NASA Earthdata Login, and an
unattended bake cannot create an account. `--source mcd43` implements that
path anyway, for anyone who has a token.

**But the data is reachable without one.** NASA GIBS serves MCD43A4 -- the
Nadir BRDF-Adjusted Reflectance member of the same family -- anonymously,
as an RGB composite of MODIS bands 1 (620-670 nm), 4 (545-565 nm) and 3
(459-479 nm). The composite has a display stretch applied, and the decisive
property is that **the stretch is the published MODIS Rapid Response
piecewise-linear curve, which is monotone and therefore invertible
exactly** (NBAR_STRETCH_* below). Undo it and what is left is per-band
MODIS reflectance.

That is `--source nbar`, and it is the DEFAULT and what ships.

WHY NOT BLUE MARBLE
-------------------
`--source bluemarble` is kept for comparison, and it is not what ships,
for a reason worth recording. Blue Marble Next Generation is built from the
same MODIS surface reflectance -- but Stockli et al. (2005) section 2.5
states the composites were contrast enhanced "by use of a cubic spline
function". That curve is not published, so it cannot be undone; measured
against the NBAR inversion for the same month, Blue Marble's brightening
runs from 0.9x to 10.9x and varies by BAND and by SITE. It is not a gamma
anyone can divide out.

(Also: the `world.topo.bathy.*` variant has shaded relief baked into the
land, which corrupts an albedo outright. GIBS's BlueMarble_NextGeneration
layer is the plain composite, which is why this tool uses it and not the
eoimages file of that name.)

VALIDATION -- measured, not asserted
------------------------------------
`--validate` samples the bake at nine sites and applies two checks. From
the shipped bake:

    site                     baked RGB              lum   published  ratio
    Amazon      (0.0101, 0.0472, 0.0262)          0.0378     0.04     0.94  veg
    Congo       (0.0205, 0.0509, 0.0326)          0.0431     0.04     1.08  veg
    Boreal      (0.0095, 0.0296, 0.0164)          0.0244     0.04     0.61  veg
    Sahara      (0.3977, 0.2837, 0.1347)          0.2971     0.30     0.99  soil
    Arabia      (0.3478, 0.2776, 0.1713)          0.2849     0.30     0.95  soil
    Australia   (0.2168, 0.0966, 0.0665)          0.1200     0.16     0.75  soil
    Tibet       (0.2461, 0.1869, 0.1182)          0.1945     0.20     0.97  soil
    Greenland   (0.5131, 0.5331, 0.5678)          0.5313     0.55     0.97  ice
    Antarctica  (0.6922, 0.7274, 0.7694)          0.7229     0.75     0.96  ice

    median luminance ratio 0.96, range 0.61 to 1.08.
    area-weighted mean land reflectance 0.160.

The second check is the one that matters and the one a grey sphere cannot
pass: every vegetated site has GREEN above both RED and BLUE (chlorophyll
absorbs at 430 and 662 nm), every soil site has RED > GREEN > BLUE (iron
oxide absorbs toward the blue), and every ice site is neutral to within
15%. All nine pass. That is what makes this land COVER and not a brightness
map.

WHAT IS STILL APPROXIMATE, SAID HERE RATHER THAN DISCOVERED LATER
-----------------------------------------------------------------
  * NBAR is the BRDF at NADIR view, not the white-sky (bihemispherical)
    albedo a Lambertian renderer would properly want. For land surfaces the
    two differ by roughly 10%. `--source mcd43` fetches the white-sky
    product for anyone with a token.
  * GIBS's archived tiles are JPEG, so the inversion recovers the stretch
    exactly but not the compression.
  * Snow and ice saturate the stretch at DN 255, which inverts above 1.0
    and is clamped. ~1.2% of texels.
  * The bake is a MIN-composite across four dates. Cloud and seasonal snow
    only ever brighten a pixel, so the darkest valid observation rejects
    both -- which is what is wanted here, because the snowline is modelled
    from temperature at render time and baking one July's snow into the
    albedo would double-count it. The cost is that permanent ice reads as
    its melt-season bare ice (0.5-0.7) rather than fresh snow (0.9).

`kAlbFlagMeasuredAlbedo` in the baked header records which source a given
file came from, so the distinction lives in the container and not only in
this docstring.

THE LAND MASK COMES FROM THE DEM, NOT FROM THE IMAGE
-----------------------------------------------------
Coverage (the alpha byte) is 255 where assets/planet/earth_lite.ptdem has
elevation >= 0 and 0 elsewhere. Deriving it from the ENGINE'S OWN
elevation grid rather than from the imagery means the coastline in the
albedo raster and the coastline the terrain actually renders are the same
curve by construction -- they cannot drift apart, and a texel the source
never saw falls back to the procedural path instead of painting ocean
colour onto land.

GUARDS -- the three defects tools/fetch_planet_dem.py shipped with
----------------------------------------------------------------
  1. NO LENGTH VERIFICATION. The first ETOPO download truncated at
     143,997,480 of 478,290,125 bytes, `curl` exited 0, and `file` still
     reported valid HDF5 because the header survived. Every download here
     verifies its completed length, and `--source mcd43` takes the expected
     length from LAADS's own CSV manifest AND cross-checks it against a
     pinned constant, so a manifest that lies is caught too. Where the
     resource is dynamically rendered (GIBS WMS) a byte length is not
     stable, so the check is on the DECODED raster instead: exact
     dimensions, plus a content assertion that it is not blank.
  2. NO RETRY. NCEI answered HTTP 504 cold and 200 warm. Retry with
     exponential backoff, resuming by Range where the server allows it.
  3. IT DOWNLOADED 478 MB BEFORE DISCOVERING THE READER WAS MISSING.
     Dependencies are checked FIRST, per source.

USAGE
-----
    python3 tools/fetch_planet_albedo.py                  # Blue Marble, 2048x1024
    python3 tools/fetch_planet_albedo.py --width 4096     # a finer bake
    python3 tools/fetch_planet_albedo.py --source mcd43 --token $EARTHDATA_TOKEN
    python3 tools/fetch_planet_albedo.py --validate       # measure the bake
    python3 tools/fetch_planet_albedo.py --local img.png  # bake a local image

OUTPUT FORMAT (mirrored by src/renderer/Planet/SurfaceAlbedo.h -- the two
are a wire format and there is no generated header between them):

    offset  size  field
    0       8     magic "PTALB001"
    8       4     uint32 width          (columns, west to east)
    12      4     uint32 height         (rows, north to south)
    16      8     float64 scale         (1.0)
    24      8     float64 offset        (0.0)
    32      4     uint32 flags          (bit 0: 1 = measured albedo)
    36      4     uint32 reserved
    40      ...   width*height RGBA8, row-major

    reflectance = (value / 255)^2 * scale + offset      per RGB channel
    coverage    = A / 255                               1 = real land data

Gamma 2.0 rather than linear 8-bit because a linear byte's 1/255 step is a
22% error against closed-canopy forest's ~0.018 reflectance; gamma 2.0 puts
that step at 5.8% and costs the shader one multiply to undo.
"""

import argparse
import hashlib
import io
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

# --- Sources ---------------------------------------------------------------

# NASA GIBS, the anonymous route. Verified 200 with real pixels; no account,
# no key. GIBS renders the mosaic server-side, so we ask for an OVERSAMPLE
# and area-average it down ourselves rather than trusting their resampler --
# the same argument fetch_planet_dem.py makes for area-averaging ETOPO.
GIBS_WMS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
            "?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            "&LAYERS={layer}&CRS=EPSG:4326&BBOX=-90,-180,90,180"
            "&WIDTH={w}&HEIGHT={h}&FORMAT=image/png{extra}")
BLUEMARBLE_LAYER = "BlueMarble_NextGeneration"
# MCD43A4 Nadir BRDF-Adjusted Reflectance, per MODIS band, rendered by GIBS
# as an RGB composite of bands 1 / 4 / 3. THE DEFAULT, and the reason this
# tool does not ship Blue Marble: see WHICH SOURCE above.
NBAR_LAYER = "MODIS_Combined_L3_Nadir-BRDF_Daily"

# The MODIS Rapid Response true-colour enhancement, as published control
# points. GIBS applies it when it renders the NBAR composite, and because it
# is a MONOTONE piecewise-linear stretch it inverts exactly by interpolating
# the same points the other way -- which is the whole reason this layer is
# usable as data and Blue Marble's cubic-spline "contrast enhancement"
# (Stockli et al. 2005, section 2.5) is not.
#
# The -0.01 .. 1.10 byte-scale range is the MODIS corrected-reflectance
# convention: a little below zero so noise around a dark target is not
# clipped, and a little above one so a specular or snow target is not.
NBAR_STRETCH_IN  = (0, 30, 60, 120, 190, 255)
NBAR_STRETCH_OUT = (0, 110, 160, 210, 240, 255)
NBAR_SCALE_LO = -0.01
NBAR_SCALE_HI = 1.10
# GIBS renders ocean and no-data as this exact grey. Matched with a small
# tolerance because the archived tiles are JPEG and ringing moves it by a
# code or two near coastlines.
NBAR_SENTINEL = (128, 128, 128)
NBAR_SENTINEL_TOL = 4
# Four dates, one per season. TWO jobs, and a min-composite does both:
# clouds and seasonal snow only ever BRIGHTEN a pixel, so taking the
# darkest valid observation across the year rejects both and leaves the
# snow-free ground. That is the right base here precisely because the
# snowline is modelled from temperature at render time -- baking a
# particular July's snow into the albedo would double-count it.
NBAR_DATES = ("2019-01-15", "2019-04-15", "2019-07-01", "2019-10-15")

# The window the Planetary Computer mosaic composites over. MCD43A4 is a
# daily product with a 16-day retrieval window, so a week of it is one
# BRDF inversion's worth of observations and the mosaic's own most-recent-
# pixel rule fills the gaps.
PC_DATE = "2019-07-01"
PC_DATE_END = "2019-07-08"
# MCD43A3 white-sky (bihemispherical) albedo, SHORTWAVE broadband. Used only
# by --validate as an independent measured cross-check; it is not baked,
# because shortwave is the wrong band for an RGB renderer -- vegetation is
# dark in the visible and bright in the NIR, so a shortwave albedo would
# make the Amazon four times too bright and grey rather than green.
MCD43_WSA_LAYER = "MODIS_Combined_L3_White_Sky_Albedo_Daily"
MCD43_WSA_COLORMAP = ("https://gibs.earthdata.nasa.gov/colormaps/v1.3/"
                      "MODIS_Combined_Albedo_Daily.xml")

# LAADS DAAC, the credentialed route to per-band MCD43. Needs an Earthdata
# bearer token: https://ladsweb.modaps.eosdis.nasa.gov/ -> Profile -> App Keys.
LAADS_DIR = ("https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/"
             "MCD43C3/{year}/{doy:03d}")
# Pinned cross-check for the granule the shipped provenance quotes. The CSV
# manifest is authoritative for the length; this constant exists so a
# manifest that reports a short size cannot talk the tool into accepting a
# short file. Update BOTH when changing the granule.
MCD43C3_GRANULE = "MCD43C3.A2020182.061.2020349164553.hdf"
MCD43C3_SIZE_BYTES = 192_075_352

ALB_MAGIC = b"PTALB001"
ALB_FLAG_MEASURED_ALBEDO = 1 << 0
DEM_MAGIC = b"PTDEM001"

# --validate's reference set.
#
#   (name, lat, lon, published visible-band reflectance, spectral class)
#
# The reflectances are broadband VISIBLE (0.3-0.7 um) hemispherical values:
# Budyko 1974 "Climate and Life"; Ahrens "Meteorology Today" 11th ed. table
# 2.2; Trenberth, Fasullo & Kiehl 2009 BAMS 90:311 for the 0.15 land mean
# the area-weighted average must land near.
#
# The SPECTRAL CLASS is the more decisive of the two checks, and the one a
# grey sphere cannot fake. It is qualitative on purpose -- per-band
# published values for a named 2-degree box are not something this file can
# cite honestly -- but the ordering it asserts is unambiguous physics:
#
#   "veg"  chlorophyll absorbs at 430 and 662 nm, so GREEN exceeds both RED
#          and BLUE. A vegetated texel that does not satisfy this is not
#          vegetation, whatever its brightness.
#   "soil" iron oxides absorb toward the blue, so RED > GREEN > BLUE
#          monotonically. Deserts, bare rock and dry savanna.
#   "ice"  near-neutral with a slight blue lift; no channel may differ from
#          the mean by more than 15%.
#
# ICE IS EXPECTED TO READ LOW here and that is not a defect: the bake is a
# min-composite across the year (see NBAR_DATES), which strips seasonal snow
# so the temperature-derived snowline can put it back at render time without
# double-counting. What is left on an ice sheet is its melt-season bare ice,
# genuinely 0.5-0.7 rather than fresh snow's 0.9.
VALIDATION_SITES = [
    ("Amazon",     -5.0, -62.0, 0.04, "veg"),
    ("Congo",       0.0,  22.0, 0.04, "veg"),
    ("Boreal",     60.0, 100.0, 0.04, "veg"),
    ("Sahara",     23.0,  12.0, 0.30, "soil"),
    ("Arabia",     22.0,  45.0, 0.30, "soil"),
    ("Australia", -25.0, 130.0, 0.16, "soil"),
    ("Tibet",      33.0,  88.0, 0.20, "soil"),
    ("Greenland",  72.0, -40.0, 0.55, "ice"),
    ("Antarctica",-80.0,  20.0, 0.75, "ice"),
]


# --- Guard 3: dependencies first, before anyone's bandwidth --------------

def check_deps(source):
    """Fail BEFORE downloading, not after.

    fetch_planet_dem.py's first version spent 478 MB of somebody's
    bandwidth to reach an ImportError.
    """
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if source in ("bluemarble", "local", "validate"):
        try:
            import PIL.Image  # noqa: F401
        except ImportError:
            missing.append("pillow")
    if source == "pc":
        try:
            import tifffile  # noqa: F401
        except ImportError:
            missing.append("tifffile   -- the Planetary Computer tiles are "
                           "int16 GeoTIFF, which Pillow cannot read")
    if source == "mcd43":
        for mod in ("pyhdf.SD", "osgeo.gdal"):
            try:
                __import__(mod)
                break
            except ImportError:
                continue
        else:
            missing.append("pyhdf   (or GDAL) -- MCD43C3 is HDF4, not HDF5, "
                           "and h5py cannot read it")
    if missing:
        sys.exit("[deps] missing, and the download is large:\n"
                 + "".join(f"        pip install {m}\n" for m in missing))


# --- Guards 1 and 2: verified length, and retry with backoff -------------

def http_get(url, dest=None, expect_bytes=None, retries=5, headers=None,
             label="download"):
    """GET `url`, VERIFYING the completed length when one is known.

    Returns the bytes (and writes them to `dest` if given).

    Two defects this guards, both observed against real NASA/NOAA endpoints:

      1. TRUNCATION THAT LOOKS LIKE SUCCESS -- a download that stopped at
         30% exited 0 and `file` still called it valid, because the header
         was intact. Only a length comparison noticed.
      2. A COLD ENDPOINT RETURNING 504 -- NOAA's THREDDS answered 504 on
         first contact and 200 once warm. Retry, resuming by Range where
         the server supports it so a retry does not restart from zero.
    """
    hdrs = dict(headers or {})
    hdrs.setdefault("User-Agent", "demont-engine/fetch_planet_albedo")

    if dest and os.path.exists(dest) and expect_bytes is not None \
            and os.path.getsize(dest) == expect_bytes:
        print(f"[{label}] {dest} already present and the right length "
              f"({expect_bytes/1e6:.0f} MB)")
        return open(dest, "rb").read()

    delay, last_err = 2.0, None
    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if (dest and os.path.exists(dest)) else 0
        req_hdrs = dict(hdrs)
        if have > 0 and expect_bytes is not None:
            req_hdrs["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=req_hdrs)
        buf = bytearray()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # An Earthdata redirect answers 200 with an HTML login page.
                # Catch it here rather than handing HTML to an HDF reader.
                final = getattr(r, "url", url)
                if "urs.earthdata.nasa.gov" in final:
                    sys.exit(f"[{label}] {url}\n"
                             "        redirected to NASA Earthdata Login. "
                             "This product needs credentials:\n"
                             "        pass --token with an Earthdata bearer "
                             "token, or use the default --source bluemarble.")
                # A server that ignores Range answers 200; appending then
                # concatenates the whole file onto a partial one. Only a
                # genuine 206 licenses the append.
                resumed = (have > 0 and getattr(r, "status", 200) == 206)
                if not resumed:
                    have = 0
                total = int(r.headers.get("Content-Length", 0) or 0)
                if total and resumed:
                    total += have
                done = have
                mode = "ab" if resumed else "wb"
                fh = open(dest, mode) if dest else None
                try:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        if fh:
                            fh.write(chunk)
                        else:
                            buf += chunk
                        done += len(chunk)
                        if total:
                            sys.stdout.write(
                                f"\r[{label}] {done/1e6:7.1f} / {total/1e6:.1f} MB")
                            sys.stdout.flush()
                finally:
                    if fh:
                        fh.close()
                if total:
                    print()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_err = e
            code = getattr(e, "code", None)
            print(f"\n[{label}] attempt {attempt}/{retries} failed"
                  f"{f' (HTTP {code})' if code else ''}: {e}")
            if attempt < retries:
                print(f"[{label}] retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2.0
                continue
            break

        data = open(dest, "rb").read() if dest else bytes(buf)

        # --- The length verification, which is the point of this function.
        if expect_bytes is not None and len(data) != expect_bytes:
            print(f"[{label}] TRUNCATED: {len(data)} of {expect_bytes} bytes "
                  f"({100.0*len(data)/expect_bytes:.1f}%)")
            if attempt < retries:
                print(f"[{label}] retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2.0
                continue
            sys.exit(f"[{label}] giving up. Refusing to bake a truncated "
                     "source: it produces a plausible-looking planet that is "
                     "silently not Earth.")
        digest = hashlib.sha256(data).hexdigest()
        print(f"[{label}] {len(data)} bytes, sha256 {digest[:16]}...")
        return data

    sys.exit(f"[{label}] all {retries} attempts failed: {last_err}")


# --- Blue Marble ---------------------------------------------------------

def fetch_bluemarble(out_w, out_h, cache_dir):
    """Fetch Blue Marble NG oversampled, and area-average it down.

    OVERSAMPLE THEN AVERAGE, for exactly the reason fetch_planet_dem.py
    area-averages ETOPO: a point-sampled decimation of a 500 m mosaic to a
    20 km grid is aliased, and the engine's smoothstep-bilinear magnifies
    whatever it is handed rather than fixing it. Averaging is also the
    physically right decimation of a reflectance field -- the mean
    reflectance of a cell is a real quantity; a point sample of it is not.
    """
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    over = 4
    src_w, src_h = out_w * over, out_h * over
    # GIBS's WMS caps out around 8192; above that, fall back to a smaller
    # oversample rather than silently receiving a service exception.
    while src_w > 8192:
        over //= 2
        if over < 1:
            over, src_w, src_h = 1, out_w, out_h
            break
        src_w, src_h = out_w * over, out_h * over
    url = GIBS_WMS.format(layer=BLUEMARBLE_LAYER, w=src_w, h=src_h, extra="")
    dest = os.path.join(cache_dir, f"bluemarble_{src_w}x{src_h}.png")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[fetch] NASA GIBS {BLUEMARBLE_LAYER} at {src_w}x{src_h} "
          f"({over}x oversample)")
    # A WMS render has no stable byte length, so the verification is on the
    # DECODED raster instead -- see below.
    data = http_get(url, dest=dest, label="fetch")

    im = Image.open(io.BytesIO(data))
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im)
    # Verification for a dynamically-rendered resource: exact dimensions,
    # and it must not be blank. A WMS ServiceException renders as a valid
    # PNG of the requested size, uniformly transparent or black -- which is
    # this tool's version of "valid HDF5 that is 30% of a planet".
    if arr.shape[0] != src_h or arr.shape[1] != src_w:
        sys.exit(f"[fetch] GIBS returned {arr.shape[1]}x{arr.shape[0]}, "
                 f"asked for {src_w}x{src_h}")
    nonzero = float((arr.max(axis=2) > 8).mean())
    if nonzero < 0.20:
        sys.exit(f"[fetch] only {100*nonzero:.1f}% of the returned raster is "
                 "non-black. That is a WMS service exception rendered as an "
                 "image, not Earth. Refusing to bake it.")
    print(f"[fetch] decoded {src_w}x{src_h}, {100*nonzero:.1f}% non-black")

    lin = srgb_to_linear(arr.astype(np.float64) / 255.0)
    return area_average(lin, out_w, out_h)


def fetch_nbar(out_w, out_h, cache_dir):
    """MCD43A4 NBAR from GIBS: per-band measured reflectance, anonymously.

    Returns (rgb, valid) where rgb is out_h x out_w x 3 of linear
    reflectance and valid is the per-texel count of dates that contributed.

    WHY THIS AND NOT THE HDF GRANULE. MCD43A4's own distribution is behind
    an Earthdata Login (measured, see the module docstring). GIBS serves the
    same product anonymously as a rendered composite of bands 1 / 4 / 3, and
    the render applies a stretch that is MONOTONE AND PUBLISHED, so it
    inverts. What comes back is per-band MODIS reflectance, not somebody's
    idea of what Earth should look like.

    WHAT NBAR IS AND IS NOT. Nadir BRDF-Adjusted Reflectance is the MCD43
    BRDF evaluated at nadir view and the local solar noon zenith -- a
    directional reflectance. A renderer's Lambertian albedo would more
    properly be the white-sky (bihemispherical) albedo, which `--source
    mcd43` fetches. For land surfaces the two differ by roughly 10%, which
    is an order below the error in any other choice available here.
    """
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    over = 2
    src_w, src_h = out_w * over, out_h * over
    while src_w > 8192:
        over //= 2
        if over < 1:
            over, src_w, src_h = 1, out_w, out_h
            break
        src_w, src_h = out_w * over, out_h * over

    acc = None      # running per-band MINIMUM over dates
    hits = None     # how many dates were valid at each texel
    sent = np.asarray(NBAR_SENTINEL, dtype=np.int16)
    for date in NBAR_DATES:
        url = GIBS_WMS.format(layer=NBAR_LAYER, w=src_w, h=src_h,
                              extra=f"&TIME={date}")
        dest = os.path.join(cache_dir, f"nbar_{date}_{src_w}x{src_h}.png")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[fetch] NASA GIBS {NBAR_LAYER} {date} at {src_w}x{src_h}")
        data = http_get(url, dest=dest, label="fetch")
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if arr.shape[0] != src_h or arr.shape[1] != src_w:
            sys.exit(f"[fetch] GIBS returned {arr.shape[1]}x{arr.shape[0]}, "
                     f"asked for {src_w}x{src_h}")
        valid = ~np.all(np.abs(arr.astype(np.int16) - sent)
                        <= NBAR_SENTINEL_TOL, axis=2)
        frac = float(valid.mean())
        print(f"[fetch] {date}: {100*frac:.1f}% of texels carry data")
        # A WMS service exception renders as a valid PNG. Land is ~29% of
        # the globe, so anything under 15% is not a planet.
        if frac < 0.15:
            sys.exit(f"[fetch] only {100*frac:.1f}% of {date} is non-sentinel. "
                     "That is a service exception rendered as an image, not "
                     "Earth. Refusing to bake it.")
        refl = nbar_invert(arr)
        if acc is None:
            acc = np.where(valid[:, :, None], refl, np.inf)
            hits = valid.astype(np.uint8)
        else:
            acc = np.minimum(acc, np.where(valid[:, :, None], refl, np.inf))
            hits += valid.astype(np.uint8)

    # Texels no date saw (polar night that never lifts, permanent cloud)
    # become zero with a zero hit count, and the DEM land mask plus the
    # per-texel coverage byte send those to the procedural path.
    acc = np.where(np.isfinite(acc), acc, 0.0)
    # Snow and ice saturate the stretch: DN 255 inverts to 1.10, above any
    # physical reflectance. Clamp, and say so -- fresh snow's visible
    # reflectance is 0.90-0.95, so clamped ice is up to ~15% bright.
    clipped = float((acc > 1.0).mean())
    acc = np.clip(acc, 0.0, 1.0)
    print(f"[bake] {100*clipped:.2f}% of texels saturated the stretch "
          f"(snow and ice) and were clamped to 1.0")
    print(f"[bake] min-composite over {len(NBAR_DATES)} dates: "
          f"{100.0*float((hits > 0).mean()):.1f}% of texels have at least "
          f"one valid date")
    rgb = area_average(acc, out_w, out_h)
    cov = area_average((hits > 0).astype(np.float64), out_w, out_h)
    return rgb, cov


# --- MCD43A4 from Microsoft Planetary Computer, as RAW reflectance --------

PC_REGISTER = "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/register"
PC_TILE = ("https://planetarycomputer.microsoft.com/api/data/v1/mosaic/tiles/"
           "{sid}/WebMercatorQuad/{z}/{x}/{y}.tif?collection=modis-43A4-061"
           "&assets=Nadir_Reflectance_Band1"
           "&assets=Nadir_Reflectance_Band4"
           "&assets=Nadir_Reflectance_Band3")
PC_COLLECTION = "modis-43A4-061"
# MCD43A4 packing, from the product's own metadata.
PC_SCALE = 1.0e-4
PC_FILL = 32767
# WebMercatorQuad zoom. z5 = 32x32 tiles of 256 px = 8192 x 8192, four times
# the 2048-column DEM, so albedo detail outruns terrain detail instead of
# limiting it. ~540 MB, comparable to the 478 MB ETOPO grid this repo
# already fetches. z6 is four times the download for detail the 19.5 km
# terrain cannot match.
PC_ZOOM = 5
# Preconditions, measured against the live service rather than assumed:
#   * `collection=` is MANDATORY -- 422 without it.
#   * z >= 5 returns 200. z3 and z4 return 204 (no content), as does the
#     WorldCRS84Quad tile matrix set at every zoom, so WebMercator is the
#     only grid actually served.
#   * a 2048-pixel tile request returns 502. 256 is the served size.
PC_TILE_PX = 256

# WEBMERCATOR TRUNCATES THE POLES at +/-85.05113 degrees, and that is a real
# hole this tool does not paper over: it leaves the coverage byte at 0 for
# every texel poleward of that, so the shader falls back to the procedural
# path there. That is 0.4% of the globe by area, it is the interior of the
# two polar caps, and the fallback is correct for it -- the Antarctic
# plateau is ~2 800 m against a modelled snowline of ~1 200 m, so the
# procedural path paints snow, which is what is there.
PC_LAT_LIMIT_DEG = 85.05112877980659

# WHY --source pc IS NOT THE DEFAULT, THOUGH ITS NUMERICS ARE BETTER.
#
# A whole band of tiles fails DETERMINISTICALLY with HTTP 500. Measured, at
# z5, tile row ty=4 -- Mercator latitudes ~61.6 to ~66.5 N, the boreal belt:
#
#     GET .../WebMercatorQuad/5/8/4.tif?collection=modis-43A4-061&assets=...
#         -> 500 "Internal Server Error", in 0.7 s, every time
#
# It is not rate limiting and it is not transient. Five attempts with
# exponential backoff to 16 s all fail, a single serial request with no
# concurrency fails identically, and the failure is fast rather than a
# timeout. Neighbouring rows return 200 with valid data from the same
# registered mosaic, so the search is fine and the service is not down --
# something about compositing that particular row breaks server-side.
#
# The result is a mosaic missing a continuous ring of the northern
# hemisphere, which is not a raster this tool will bake. --source nbar
# fetches the SAME MCD43A4 product through GIBS, complete and including the
# poles, at the cost of inverting a display stretch instead of reading raw
# int16. Retry --source pc when the service is fixed; everything else about
# that path is verified working.
PC_KNOWN_FAILURE = "z5 tile row ty=4 returns HTTP 500 deterministically"


def fetch_pc_nbar(out_w, out_h, cache_dir, zoom=PC_ZOOM, workers=8):
    """MCD43A4 per-band NBAR as raw int16, mosaicked and reprojected.

    THE DIFFERENCE FROM --source nbar. Both fetch MCD43A4. GIBS serves it
    RENDERED, so that path recovers reflectance by inverting a display
    stretch off JPEG tiles and cannot represent anything above 1.0 (snow
    clamps). This one is the COG behind the same product: int16 at a
    1e-4 scale with an explicit 32767 fill and a per-pixel validity mask.
    No stretch to invert, no JPEG, no clamp.

    Returns (rgb, coverage) on the output equirectangular grid.
    """
    import json
    import numpy as np
    import tifffile
    from concurrent.futures import ThreadPoolExecutor

    n = 1 << zoom
    print(f"[fetch] Planetary Computer {PC_COLLECTION}, WebMercatorQuad z{zoom}"
          f" = {n}x{n} tiles of {PC_TILE_PX} px "
          f"({n*PC_TILE_PX}x{n*PC_TILE_PX})")
    body = json.dumps({
        "collections": [PC_COLLECTION],
        "datetime": f"{PC_DATE}T00:00:00Z/{PC_DATE_END}T00:00:00Z",
    }).encode()
    req = urllib.request.Request(
        PC_REGISTER, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "demont-engine/fetch_planet_albedo"})
    with urllib.request.urlopen(req, timeout=120) as r:
        sid = json.loads(r.read())["id"]
    print(f"[fetch] mosaic search {sid}")

    os.makedirs(cache_dir, exist_ok=True)
    # Accumulate straight into the OUTPUT equirectangular grid rather than
    # assembling the 8192x8192 mosaic first: the mosaic would be ~540 MB of
    # int16 and ~870 MB once widened to float, and nothing needs it whole.
    acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    wgt = np.zeros((out_h, out_w), dtype=np.float64)

    def one(ty, tx):
        dest = os.path.join(cache_dir, f"pc_z{zoom}_{tx}_{ty}.tif")
        url = PC_TILE.format(sid=sid, z=zoom, x=tx, y=ty)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest, os.path.getsize(dest)
        data = http_get(url, label=f"t{tx},{ty}")
        with open(dest, "wb") as f:
            f.write(data)
        return dest, len(data)

    done = 0
    total_bytes = 0
    for ty in range(n):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda tx: one(ty, tx), range(n)))
        for tx, (dest, nbytes) in enumerate(results):
            total_bytes += nbytes
            done += 1
            if nbytes == 0:
                continue          # 204: no imagery under this tile
            a = tifffile.imread(dest)
            if a.ndim != 3 or a.shape[0] != PC_TILE_PX \
                    or a.shape[1] != PC_TILE_PX or a.shape[2] < 4:
                sys.exit(f"[fetch] {dest}: expected "
                         f"{PC_TILE_PX}x{PC_TILE_PX}x4, got {a.shape}. "
                         "Refusing to bake a mosaic with a malformed tile.")
            accumulate_pc_tile(a, tx, ty, zoom, acc, wgt)
        sys.stdout.write(f"\r[fetch] {done}/{n*n} tiles, "
                         f"{total_bytes/1e6:.0f} MB")
        sys.stdout.flush()
    print()

    seen = wgt > 0.0
    print(f"[bake] mosaic covers {100.0*float(seen.mean()):.1f}% of the "
          f"output grid (WebMercator stops at +/-{PC_LAT_LIMIT_DEG:.2f} deg, "
          "so the polar caps are deliberately empty)")
    rgb = np.zeros_like(acc)
    np.divide(acc, wgt[:, :, None], out=rgb, where=seen[:, :, None])
    return np.clip(rgb, 0.0, 1.0), seen.astype(np.float64)


def accumulate_pc_tile(a, tx, ty, zoom, acc, wgt):
    """Bin one WebMercator tile's valid pixels into the equirect accumulator.

    AREA AVERAGE by binning, for the reason fetch_planet_dem.py area-averages
    ETOPO: a point-sampled reprojection of a 500 m product onto a 19.5 km
    grid is aliased, and the engine's smoothstep-bilinear magnifies whatever
    it is handed rather than repairing it.
    """
    import numpy as np
    n = 1 << zoom
    out_h, out_w = wgt.shape
    px = PC_TILE_PX

    # Validity: the 4th band is titiler's alpha, and the fill value is
    # explicit. Both are required -- alpha alone lets a 32767 fill through
    # where the mosaic had an observation of nothing.
    band = a[:, :, :3].astype(np.float64)
    alpha = a[:, :, 3]
    valid = (alpha > 127) & np.all((a[:, :, :3] >= 0) & (a[:, :, :3] < PC_FILL),
                                   axis=2)
    if not valid.any():
        return
    refl = band * PC_SCALE

    # Pixel centres -> lon/lat. WebMercatorQuad: x is linear in longitude,
    # y is linear in the Mercator ordinate, so latitude comes back through
    # the inverse Gudermannian.
    jj, ii = np.nonzero(valid)
    gx = (tx * px + ii + 0.5) / (n * px)          # 0..1 across the world
    gy = (ty * px + jj + 0.5) / (n * px)
    lon = gx * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * gy))))

    ox = np.clip(((lon + 180.0) / 360.0 * out_w).astype(np.int64), 0, out_w - 1)
    oy = np.clip(((90.0 - lat) / 180.0 * out_h).astype(np.int64), 0, out_h - 1)
    flat = oy * out_w + ox
    v = refl[jj, ii]
    for c in range(3):
        acc[:, :, c].ravel()[:] += np.bincount(
            flat, weights=v[:, c], minlength=out_h * out_w)
    wgt.ravel()[:] += np.bincount(flat, minlength=out_h * out_w)


def nbar_invert(dn):
    """Undo the MODIS Rapid Response stretch. See NBAR_STRETCH_* above."""
    import numpy as np
    x = np.interp(dn.astype(np.float64),
                  np.asarray(NBAR_STRETCH_OUT, dtype=np.float64),
                  np.asarray(NBAR_STRETCH_IN, dtype=np.float64))
    return NBAR_SCALE_LO + x * (NBAR_SCALE_HI - NBAR_SCALE_LO) / 255.0


def srgb_to_linear(c):
    """IEC 61966-2-1 sRGB EOTF.

    Applied because that is the transfer function the delivered image is
    encoded with. Skipping it -- treating display bytes as reflectance -- is
    a far larger error than anything in the --validate table, and it is the
    error a naive "use Blue Marble as albedo" makes.
    """
    import numpy as np
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def area_average(arr, out_w, out_h):
    import numpy as np
    h, w = arr.shape[0], arr.shape[1]
    if (h, w) == (out_h, out_w):
        return arr
    ys = (np.arange(out_h + 1) * h // out_h)
    xs = (np.arange(out_w + 1) * w // out_w)
    chans = arr.shape[2] if arr.ndim == 3 else 1
    out = np.empty((out_h, out_w, chans), dtype=np.float64)
    a3 = arr if arr.ndim == 3 else arr[:, :, None]
    for j in range(out_h):
        band = a3[ys[j]:max(ys[j + 1], ys[j] + 1), :, :]
        for i in range(out_w):
            out[j, i, :] = band[:, xs[i]:max(xs[i + 1], xs[i] + 1), :].mean(axis=(0, 1))
    return out if arr.ndim == 3 else out[:, :, 0]


# --- MCD43C3, the physically correct path --------------------------------

def fetch_mcd43(out_w, out_h, cache_dir, token):
    """Per-band white-sky albedo from MCD43C3, bands 1 / 4 / 3 -> R / G / B.

    MCD43C3.061 is the 0.05-degree climate-modelling grid of the MODIS
    BRDF/Albedo product (Schaaf et al. 2002, Remote Sensing of Environment
    83:135-148). `Albedo_WSA_Band1/4/3` are bihemispherical reflectance --
    the reflectance of the surface under isotropic diffuse illumination,
    which is the closest thing a measured product has to a Lambertian
    albedo, and unlike the black-sky albedo it carries no solar-geometry
    dependence for a renderer to have to undo.

    The band centres are why this maps to RGB at all, and they are close
    enough to a display primary set to be honest about:
        band 1  620-670 nm  -> R
        band 4  545-565 nm  -> G
        band 3  459-479 nm  -> B
    """
    import numpy as np
    if not token:
        sys.exit("[fetch] --source mcd43 needs an Earthdata bearer token.\n"
                 "        Get one at https://ladsweb.modaps.eosdis.nasa.gov/ "
                 "(Profile -> App Keys),\n"
                 "        then pass --token or set EARTHDATA_TOKEN.\n"
                 "        Without it, use the default --source bluemarble.")
    year, doy = 2020, 182
    base = LAADS_DIR.format(year=year, doy=doy)
    hdrs = {"Authorization": f"Bearer {token}"}

    # The manifest is authoritative for the length -- better than a pinned
    # constant, which goes stale silently. The pinned constant then
    # cross-checks the manifest, so a server reporting a short size cannot
    # talk this tool into accepting a short file either.
    print(f"[fetch] LAADS manifest {base}.csv")
    csv = http_get(base + ".csv", headers=hdrs, label="manifest").decode()
    row = None
    for line in csv.splitlines()[1:]:
        parts = line.split(",")
        if parts and parts[0].endswith(".hdf"):
            row = parts
            break
    if not row:
        sys.exit(f"[fetch] no .hdf granule listed at {base}.csv")
    name, size = row[0], int(row[2])
    print(f"[fetch] granule {name}, manifest says {size} bytes")
    if name == MCD43C3_GRANULE and size != MCD43C3_SIZE_BYTES:
        sys.exit(f"[fetch] manifest reports {size} bytes for {name}, but the "
                 f"pinned length is {MCD43C3_SIZE_BYTES}. One of them is "
                 "wrong; refusing to guess which.")
    dest = os.path.join(cache_dir, name)
    os.makedirs(cache_dir, exist_ok=True)
    http_get(f"{base}/{name}", dest=dest, expect_bytes=size, headers=hdrs,
             label="fetch")

    bands = read_hdf_bands(dest, ("Albedo_WSA_Band1", "Albedo_WSA_Band4",
                                  "Albedo_WSA_Band3"))
    # MCD43C3 scale factor 0.001, fill 32767, valid range 0..32766.
    rgb = np.stack(bands, axis=2).astype(np.float64)
    valid = np.all(rgb < 32767, axis=2)
    rgb = np.clip(rgb * 0.001, 0.0, 1.0)
    rgb[~valid] = 0.0
    print(f"[bake] MCD43C3 valid over {100.0*valid.mean():.1f}% of the grid")
    return area_average(rgb, out_w, out_h)


def read_hdf_bands(path, names):
    import numpy as np
    try:
        from pyhdf.SD import SD, SDC
        ds = SD(path, SDC.READ)
        return [np.asarray(ds.select(n)[:]) for n in names]
    except ImportError:
        pass
    from osgeo import gdal
    out = []
    for n in names:
        sub = gdal.Open(f'HDF4_EOS:EOS_GRID:"{path}":MOD_CMG_BRDF_0.05deg:{n}')
        if sub is None:
            sys.exit(f"[bake] {path} has no subdataset {n}")
        out.append(np.asarray(sub.GetRasterBand(1).ReadAsArray()))
    return out


# --- The land mask, from the engine's own DEM ----------------------------

def dem_land_mask(dem_path, out_w, out_h):
    """Coverage = 255 where the DEM says land, 0 where it says water.

    Taken from the ELEVATION grid rather than from the imagery so the
    coastline in the albedo raster is the same curve the terrain actually
    renders. Two independently-derived coastlines would disagree by a texel
    somewhere, and a texel of ocean colour on land is a visible artefact
    exactly where the eye is already looking.
    """
    import numpy as np
    if not os.path.exists(dem_path):
        print(f"[bake] WARNING: {dem_path} not found -- coverage will be 255 "
              "everywhere, so the raster will paint ocean colour on ocean "
              "texels instead of deferring to the water material.")
        return np.full((out_h, out_w), 255, dtype=np.uint8)
    with open(dem_path, "rb") as f:
        head = f.read(40)
        if head[:8] != DEM_MAGIC:
            sys.exit(f"[bake] {dem_path}: bad magic, expected PTDEM001")
        w, h = struct.unpack("<II", head[8:16])
        scale, offset = struct.unpack("<dd", head[16:32])
        raw = f.read(w * h * 2)
    if len(raw) != w * h * 2:
        sys.exit(f"[bake] {dem_path}: truncated ({len(raw)} of {w*h*2} bytes)")
    q = np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float64)
    land = (q * scale + offset) >= 0.0
    print(f"[bake] DEM {w}x{h}: land over {100.0*land.mean():.1f}% of the grid")
    # Area-average the BOOLEAN, then threshold at half a texel. A coastal
    # texel that is mostly land is land; the fractional value would make the
    # shader blend toward the procedural fallback along every shoreline.
    frac = area_average(land.astype(np.float64), out_w, out_h)
    return np.where(frac >= 0.5, 255, 0).astype(np.uint8)


# --- Bake ----------------------------------------------------------------

def bake(rgb, coverage, dest, flags):
    import numpy as np
    h, w, _ = rgb.shape
    lin = np.clip(rgb, 0.0, 1.0)
    enc = np.rint(np.power(lin, 1.0 / 2.0) * 255.0).astype(np.uint8)
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = enc
    out[:, :, 3] = coverage

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        f.write(ALB_MAGIC)
        f.write(struct.pack("<II", w, h))
        f.write(struct.pack("<dd", 1.0, 0.0))
        f.write(struct.pack("<II", flags, 0))
        f.write(out.tobytes())
    size = os.path.getsize(dest)
    km = 2 * 3.14159265358979 * 6371008.8 / w / 1000.0
    print(f"[bake] wrote {dest} -- {size/1e6:.2f} MB, {w}x{h}, "
          f"{km:.1f} km/texel at the equator, "
          f"flags={flags} ({'measured albedo' if flags & 1 else 'radiance-derived'})")

    # Round-trip: the quantised raster must still be the same planet.
    back = (enc.astype(np.float64) / 255.0) ** 2.0
    land = coverage > 127
    if land.any():
        err = float(np.abs(back - lin)[land].max())
        # AREA-weighted, by cos(latitude). An unweighted mean over an
        # equirectangular grid over-counts the poles by 1/cos(lat), and the
        # poles are ice -- which turned an honest 0.16 into a reported 0.35
        # the first time this line was written. The grid is a projection,
        # not a sample of the sphere.
        lat = (np.pi * 0.5 - (np.arange(h) + 0.5) * np.pi / h)
        wgt = (np.cos(lat)[:, None] * land).astype(np.float64)
        lum = (0.2126 * back[:, :, 0] + 0.7152 * back[:, :, 1]
               + 0.0722 * back[:, :, 2])
        mean = float((lum * wgt).sum() / wgt.sum())
        print(f"[bake] max gamma-2.0 quantisation error {err:.5f}; "
              f"area-weighted mean land reflectance {mean:.4f} "
              f"(Trenberth, Fasullo & Kiehl 2009 give 0.15 for the "
              f"SHORTWAVE land mean; the visible-band mean is lower, "
              f"because vegetation is dark in the visible and bright in "
              f"the near infrared)")


# --- Validation ----------------------------------------------------------

def sample_baked(path, lat, lon, half_deg=1.0):
    """Mean reflectance over a +/- half_deg box.

    A BOX, not a texel. One 19.5 km texel of the Sahara can be a dune field
    or a hamada and the published figure is a regional one; comparing a
    point sample against a regional mean measures the sampling, not the
    dataset. The box is small enough to stay inside each named region.
    """
    import numpy as np
    with open(path, "rb") as f:
        head = f.read(40)
        w, h = struct.unpack("<II", head[8:16])
        raw = f.read(w * h * 4)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).astype(np.float64) / 255.0
    y0 = int(np.clip((90.0 - (lat + half_deg)) / 180.0 * h, 0, h - 1))
    y1 = int(np.clip((90.0 - (lat - half_deg)) / 180.0 * h, 0, h - 1)) + 1
    x0 = int((lon - half_deg + 180.0) / 360.0 * w) % w
    x1 = x0 + max(int(2 * half_deg / 360.0 * w), 1)
    box = a[y0:max(y1, y0 + 1), :, :].take(range(x0, x1), axis=1, mode="wrap")
    px = box.reshape(-1, 4).mean(axis=0)
    return px[:3] ** 2.0, px[3]


def validate(path):
    """Measure the bake against published visible-band reflectance.

    This is a MEASUREMENT, printed so it can be recorded, not a tuning
    step: nothing here feeds back into the bake. The point is that
    PROVENANCE.md quotes numbers somebody can reproduce.
    """
    import numpy as np
    print(f"[validate] {path}")
    print(f"  {'site':<12s} {'baked RGB':>24s} {'lum':>8s} {'published':>10s} "
          f"{'ratio':>7s}  {'shape':>6s}")
    ratios, shape_fail = [], []
    for name, lat, lon, pub, cls in VALIDATION_SITES:
        rgb, cov = sample_baked(path, lat, lon)
        r_, g_, b_ = float(rgb[0]), float(rgb[1]), float(rgb[2])
        lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
        ratios.append(lum / pub)
        if cls == "veg":
            ok = g_ > r_ and g_ > b_
        elif cls == "soil":
            ok = r_ > g_ > b_
        else:
            m = (r_ + g_ + b_) / 3.0
            ok = m > 0.0 and max(abs(r_ - m), abs(g_ - m), abs(b_ - m)) < 0.15 * m
        if not ok:
            shape_fail.append(name)
        print(f"  {name:<12s} ({r_:.4f},{g_:.4f},{b_:.4f}) "
              f"{lum:8.4f} {pub:10.2f} {lum/pub:7.2f}  "
              f"{cls:>4s}{'  ok' if ok else ' FAIL'}")
    r = np.asarray(ratios)
    print(f"  luminance ratio: median {np.median(r):.2f}, "
          f"min {r.min():.2f}, max {r.max():.2f}")
    if shape_fail:
        print(f"  SPECTRAL SHAPE FAILED at: {', '.join(shape_fail)}")
        print("  That is the check a grey sphere cannot pass. A failure here "
              "means the raster is not carrying land cover,\n"
              "  whatever its brightness says -- check the source, the "
              "inversion and the registration before shipping it.")
    else:
        print("  spectral shape: every vegetated site has G > R and G > B, "
              "every soil site has R > G > B, every ice site is\n"
              "  near-neutral. This is the check a grey sphere cannot pass, "
              "and it is what makes the raster land COVER\n"
              "  rather than a brightness map.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="nbar",
                    choices=("nbar", "pc", "bluemarble", "mcd43"),
                    help="nbar = MCD43A4 per-band NBAR via GIBS, anonymous, "
                         "whole globe, SHIPS; "
                         "pc = the same product as RAW int16 from Microsoft "
                         "Planetary Computer -- better numerics, but "
                         "currently incomplete (see PC_KNOWN_FAILURE); "
                         "mcd43 = MCD43C3 per-band WHITE-SKY albedo, the "
                         "most correct of all, needs an Earthdata token; "
                         "bluemarble = a display composite with an "
                         "un-invertible tone curve, kept for comparison only")
    ap.add_argument("--zoom", type=int, default=PC_ZOOM,
                    help="WebMercatorQuad zoom for --source pc. 5 = 1024 "
                         "tiles / ~540 MB / 8192 px. Below 5 the service "
                         "returns 204.")
    ap.add_argument("--local", default=None,
                    help="bake a local equirectangular sRGB image instead of "
                         "downloading")
    ap.add_argument("--token", default=os.environ.get("EARTHDATA_TOKEN"),
                    help="Earthdata bearer token for --source mcd43")
    ap.add_argument("--cache", default="build/planet_albedo",
                    help="where to keep downloaded sources")
    ap.add_argument("--dem", default="assets/planet/earth_lite.ptdem",
                    help="DEM the land mask is taken from")
    ap.add_argument("--out", default="assets/planet/earth_lite.ptalb")
    ap.add_argument("--width", type=int, default=2048,
                    help="output columns (height is width/2)")
    ap.add_argument("--validate", action="store_true",
                    help="measure an existing bake against published "
                         "reflectances and exit")
    args = ap.parse_args()

    if args.validate:
        check_deps("validate")
        validate(args.out)
        return

    out_w, out_h = args.width, args.width // 2
    src = "local" if args.local else args.source
    check_deps(src)

    import numpy as np
    seen = None
    if args.local:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(args.local).convert("RGB")
        print(f"[bake] local source {im.size[0]}x{im.size[1]}")
        rgb = area_average(srgb_to_linear(np.asarray(im).astype(np.float64) / 255.0),
                           out_w, out_h)
        flags = 0
    elif args.source == "mcd43":
        rgb = fetch_mcd43(out_w, out_h, args.cache, args.token)
        flags = ALB_FLAG_MEASURED_ALBEDO
    elif args.source == "nbar":
        rgb, seen = fetch_nbar(out_w, out_h, args.cache)
        flags = ALB_FLAG_MEASURED_ALBEDO
    elif args.source == "pc":
        rgb, seen = fetch_pc_nbar(out_w, out_h, args.cache, args.zoom)
        flags = ALB_FLAG_MEASURED_ALBEDO
    else:
        rgb = fetch_bluemarble(out_w, out_h, args.cache)
        flags = 0

    # Coverage is the AND of two independent statements: the DEM says this
    # texel is land, and the source actually observed it. Either one alone
    # leaves a hole -- DEM-only would paint the sentinel grey onto a
    # permanently clouded valley, source-only would paint land colour onto
    # the sea ice the satellite happily measured.
    coverage = dem_land_mask(args.dem, out_w, out_h)
    if seen is not None:
        coverage = np.where(seen >= 0.5, coverage, 0).astype(np.uint8)
        print(f"[bake] coverage after AND with source visibility: "
              f"{100.0*float((coverage > 127).mean()):.1f}% of the grid")
    bake(rgb, coverage, args.out, flags)
    print()
    validate(args.out)
    print()
    print("Done. The engine picks it up via r_planet_albedo_map; the source,")
    print("its licence and what it is NOT are in assets/planet/PROVENANCE.md.")


if __name__ == "__main__":
    main()

# Planetary elevation data — provenance

Planetary P4 (#258).

A height field with no recorded origin is the same problem as a fabricated
physical constant, so this file records where the engine's terrain data comes
from, what was done to it, and what is *not* here.

## Source

**ETOPO 2022 Global Relief Model**, ice-surface elevation.

- Publisher: NOAA National Centers for Environmental Information (NCEI).
- DOI: `10.25921/fd45-gt74`
- Citation: NOAA National Centers for Environmental Information. 2022:
  *ETOPO 2022 15 Arc-Second Global Relief Model.* NOAA NCEI.
- Licence: a **work of the United States Government**, and therefore in the
  **public domain**. No attribution obligation attaches to code that derives
  from it, which is why an MIT-licensed engine can ship a derivative at all.

### Why this source and not another

| Candidate | Resolution | Bathymetry? | Licence | Verdict |
|---|---|---|---|---|
| **ETOPO 2022** | 15″ / 30″ / 60″ | **yes, same grid** | US Gov work, public domain | chosen |
| GEBCO 2024 | 15″ | yes | free, **attribution required**, mixed third-party provenance | attribution and provenance are a licensing question a permissive engine should not have to answer |
| NASADEM / SRTM | 1″ | **no — land only** | NASA/USGS, public domain | cannot make a shoreline; ~100 GB |

Bathymetry in the *same* grid is the decisive property. A shoreline is a
statement about where the land stops, and you cannot make one from a
land-only dataset — P5's ocean needs the sea floor as much as P4 needs the
land.

## What is committed here

**Nothing but this file, by design.** The full ETOPO grid is ~450 MB at 60″
and ~7.5 GB at 15″; neither belongs in a source repository.

`tools/fetch_planet_dem.py` downloads the grid from NCEI and bakes the small
"Earth lite" derivative the engine loads by default:

```
python3 tools/fetch_planet_dem.py
```

which writes `assets/planet/earth_lite.ptdem` — a 2048 × 1024 `uint16`
equirectangular grid, 4.0 MB, ~19.6 km per texel at the equator. At that
resolution it carries correct continent outlines, correct ocean basins, and
the coarse bathymetry P5's shoreline attenuation needs; everything finer is
the fractal continuation described in
`src/renderer/Planet/ElevationField.h`.

> **Status:** the `earth_lite.ptdem` file is committed, baked from the real
> NCEI ETOPO 2022 60″ grid (SHA-verified 478,290,125-byte source). It is a
> `PTDEM002` file: elevation plane plus the relief plane added in #318.

## Processing steps

1. **Download** `ETOPO_2022_v1_60s_N90W180_surface.nc` from NCEI.
2. **Orient** north-to-south. ETOPO's netCDF runs south-to-north; the
   engine's grid has row 0 at the north pole, matching the equirectangular
   convention every image-based environment map already uses.
3. **Measure relief on the full-resolution grid, before decimation (#318).**
   `relief` is sigma(L_dem), the local RMS inter-texel height difference the
   fractal continuation is anchored on. The area average in step 4 is a
   low-pass filter: it preserves each output texel's mean and destroys its
   inter-texel variance — precisely the quantity relief measures. So relief
   is computed here, on the full-resolution grid, at the output-texel lag,
   RMS-aggregated over each footprint, and carried in its own plane. See
   *The relief plane* below.
4. **Resample the elevation** to the target size by **area average**, not
   point sampling. This is not a quality preference: a point-sampled
   decimation would hand the continuation aliased noise and it would
   faithfully continue the aliasing. The mean elevation of a cell is also a
   real physical quantity; a point sample of it is not. (The variance the
   average discards is what step 3 captured first.)
5. **Quantise** both planes to `uint16` with a global affine.

## The encoding

```
height_m = value * 0.303 - 11000        value in [0, 65535]
```

Both ends are derived from measurements rather than rounded:

- `-11 000 m` clears **Challenger Deep** at −10 935 m (Gardner, Armstrong &
  Calder 2014, *So, How Deep Is the Mariana Trench?*, Marine Geodesy 37:1)
  by 65 m.
- `0.303 m/count` puts the top of the range at
  `-11000 + 65535 × 0.303 = +8 857.1 m`, clearing **Everest** at 8 848.86 m
  (2020 China/Nepal joint survey) by 8.2 m.
- The quantisation step is therefore **30.3 cm**.

`float16` is not usable: its 11-bit mantissa gives an 8 m ULP at 8 000 m,
which would terrace the Himalayas into visible steps.

Note that the value quoted in the phase issue, `value × 0.302 − 11000`, tops
out at +8 791.6 m and does *not* reach Everest. `tests/pt_planet_terrain_test.cpp`
pins both the working affine and that near miss, so the discrepancy stays
visible rather than being quietly re-introduced.

## The relief plane (#318)

`PTDEM002` appends a second `uint16` plane, the per-texel relief:

```
relief_m = relief_value * 0.303             relief_value in [0, 65535]
```

Same quantisation step as the heights, offset 0 (relief is a non-negative
height difference). The top of the range, `65535 × 0.303 = 19 857 m`, clears
the largest possible height difference on Earth (Everest 8 848.86 m minus
Challenger Deep −10 935 m = 19 784 m), so the plane **provably never clamps**
— a silent clamp would reintroduce exactly the second-moment suppression the
plane exists to fix.

Why a separate plane, rather than a cleverer decimation of the heights: the
area average is the right decimation for the *heights* (their mean is a real
quantity, and it is what makes land fraction, mean ocean depth, and every
landmark validate). It is the wrong decimation for the *variance* — an
average is a low-pass filter, and the variance is the whole second moment.
Carrying relief separately keeps the validated means intact and restores the
lost variance in one extra plane. Deriving relief from the decimated grid
afterwards, as `PTDEM001` did, could not: it measured the difference of block
means, not the point-to-point relief. The regenerated asset raised
area-weighted land relief from `p50 30.7 / p90 185 / p99 545 m` to
`p50 41.5 / p90 246 / p99 632 m`, and the Everest texel from 163 m to 379 m
(×2.3) — the peaks were the most averaged-away.

## Container format

40-byte header, then the planes. Mirrored by `pt::planet::DemHeader` in
`src/renderer/Planet/ElevationField.h`; the two are a wire format and there
is no generated header between them.

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `PTDEM002` (`PTDEM001` = no relief plane) |
| 8 | 4 | `uint32` width (columns, west→east) |
| 12 | 4 | `uint32` height (rows, north→south) |
| 16 | 8 | `float64` scale_m |
| 24 | 8 | `float64` offset_m |
| 32 | 4 | `uint32` flags (bit 0: relief plane present) |
| 36 | 4 | `uint32` reserved |
| 40 | … | width × height little-endian `uint16` elevation, row-major |
| … | … | width × height little-endian `uint16` relief, row-major (v2) |

Pixel-centre registered: texel *(x, y)* sits at
`lon = -π + (x + ½)·2π/W`, `lat = +π/2 − (y + ½)·π/H`.

`PTDEM001` files still load: the loader falls back to deriving relief from the
decimated grid (the suppressed path) and the engine logs one loud line
saying so. This tool only ever writes `PTDEM002`.

The file is stored uncompressed. zstd would take the 8 MB two-plane grid to a
few MB, which is the same order as `assets/stars/BSC5.dat`, but the engine has
no zstd dependency today and adding one is not a trade worth making yet.

## What is *not* modelled

- **Geoid undulation.** Real sea level departs from the WGS-84 ellipsoid by
  −106 m (Indian Ocean) to +85 m (north of Iceland). At a 4 654 m horizon
  from eye height that is invisible, and from orbit it is 100 m against a
  2 460 m pixel footprint. EGM2008 to degree ~360 would be ~1 MB if the
  project ever wants it.
- ~~**Land cover.**~~ *Closed by #300 — see the second half of this file.*
  The terrain's albedo used to come from a small elevation- and slope-driven
  biome function alone. It now comes from a measured MODIS raster, modulated
  by that function's DEM-derived parts.
- **City lights.** NASA Black Marble VIIRS (Román et al. 2018) is the real
  product; full resolution is ~250 MB. Generating them procedurally would be
  the fabricated-coefficients decision made again, so the choice is a
  downsampled real tile or nothing — and for this arc, nothing.

---

# Planetary surface albedo — provenance

Land cover (#300).

The terrain rendered with **one albedo** (`r_planet_ground_albedo`) plus a
fixed snowline. At 6 264 m the Himalayas came out uniform white, and from
orbit the planet was a well-lit uniform sphere. This file records where the
land colour now comes from, what it *is*, and — the part that matters most
here — **what it is not**.

## Source

**MODIS MCD43A4 v061**, Nadir BRDF-Adjusted Reflectance, bands 1 / 4 / 3.

- Publisher: NASA LP DAAC, from the MODIS BRDF/Albedo algorithm.
- DOI: `10.5067/MODIS/MCD43A4.061`
- Citation: Schaaf, C. and Wang, Z. 2021. *MODIS/Terra+Aqua BRDF/Albedo
  Nadir BRDF-Adjusted Ref Daily L3 Global 500m V061.* NASA EOSDIS LP DAAC.
  The algorithm is Schaaf et al. 2002, *First operational BRDF, albedo nadir
  reflectance products from MODIS*, Remote Sensing of Environment
  83:135–148.

### Bands, and why these three

| MODIS band | centre | → channel |
|---|---|---|
| 1 | 620–670 nm | red |
| 4 | 545–565 nm | green |
| 3 | 459–479 nm | blue |

These are the three MODIS land bands inside the visible, and they are close
enough to a display primary set to use directly. The rest are near- and
shortwave-infrared, where vegetation is *bright* — which is exactly why a
broadband shortwave albedo is the wrong product for an RGB renderer, and it
was measured and rejected on that ground (see below).

### Packing, in the underlying product

`int16`, **scale factor 1×10⁻⁴**, **fill value 32767**. The COG route reads
those directly and pairs them with titiler's alpha band; validity requires
*both*, because alpha alone lets a 32767 fill through where the mosaic had
an observation of nothing.

### Licence

NASA LP DAAC data carry **no copyright**; NASA requests citation but does
not require it. That is the accurate position, and it is what lets an
MIT-licensed engine ship a derivative — the same way it ships an ETOPO one.

It is not quite the same statement as "public domain", and the difference is
worth keeping straight. Note also that Microsoft Planetary Computer's STAC
record for this collection reports `"license": "proprietary"`; that is a
**placeholder in their catalogue metadata**, not a claim by NASA, and it does
not describe the underlying product.

## What NBAR is, and what it is NOT

**This is the most important paragraph in this file.**

NBAR is the MCD43 BRDF **evaluated at nadir view and the local solar-noon
zenith**. It is a *directional reflectance*, not an albedo.

The products that *are* albedo — **MCD43A3** (500 m) and **MCD43C3**
(0.05°), carrying directional-hemispherical (black-sky) and bihemispherical
(white-sky) albedo per band — are the physically correct choice for a
renderer, and `tools/fetch_planet_albedo.py --source mcd43` implements the
path to them. They are not what ships because **every distribution point for
them is behind a NASA Earthdata Login**, which an unattended bake cannot
obtain. Measured, not assumed:

```
GET https://e4ftl01.cr.usgs.gov/MOTA/MCD43C3.061/            -> 404
GET .../ladsweb.../MCD43C3/2020/182/MCD43C3.A2020182.*.hdf   -> 302 to
                                    urs.earthdata.nasa.gov/oauth/authorize
GET https://data.lpdaac.earthdatacloud.nasa.gov/...          -> 401
GET https://modis-pds.s3.amazonaws.com/                      -> AccessDenied
```

For a diffuse albedo texture NBAR is a good proxy — for land surfaces it
differs from the white-sky albedo by roughly 10% — but it is a proxy, and a
future reader should not have to rediscover that.

## Why not the other candidates

| Candidate | What it is | Verdict |
|---|---|---|
| **MCD43A4 NBAR** | per-band visible reflectance, measured, anonymous | **chosen** |
| MCD43A3 / C3 | per-band *albedo*, strictly better | Earthdata Login only; implemented, not shipped |
| MCD43A3 shortwave broadband (GIBS) | measured *albedo*, one scalar | wrong band. Measured: Amazon 0.128, Sahara 0.389, against visible values of ~0.03 and ~0.30. Vegetation is dark in the visible and bright in the NIR, so a shortwave albedo makes rainforest four times too bright *and* grey rather than green |
| MERRA-2 surface albedo | reanalysis, broadband | same band problem; measured 0.121 / 0.383 / 0.792 |
| MISR natural-colour DHR | RGB composite | heavily stretched and saturating; green reads ~8× the true DHR |
| NOAA AVHRR/VIIRS surface-reflectance CDR | anonymous, US Gov | red / NIR / SWIR only — no blue, no green |
| ESA WorldCover, Copernicus | land-cover classes | CC-BY: **attribution required**, which a permissive engine should not have to answer for |
| NASA Blue Marble Next Generation | true-colour composite | **rejected, see below** |

### Blue Marble, and why it is not here

Blue Marble is built from the same MODIS surface reflectance, is equally
public and equally anonymous, and is the obvious shortcut. It is rejected on
a specific, checkable ground: Stöckli et al. (2005) §2.5 states the
composites were contrast enhanced *"by use of a cubic spline function"*. That
curve is not published, so it cannot be undone. Measured against the NBAR
inversion for the same month, Blue Marble's brightening runs **0.9× to
10.9×** and varies by **band** and by **site** — it is not a gamma anyone can
divide out.

Separately, the widely-mirrored `world.topo.bathy.*` files have **shaded
relief baked into the land**, which corrupts an albedo outright: the terrain
would be lit twice, once by the renderer and once by whoever made the image.
(GIBS's `BlueMarble_NextGeneration` layer is the plain composite, without
that defect — but it still has the tone curve.)

This matters more now than it would have a phase ago. #280 (`6364f6e`) put
the sky in physical radiometric units, so an albedo carrying its own baked
illumination double-counts visibly instead of hiding inside a fudge factor.
That is the same failure mode that produced `r_rayleigh 30`.

## Access route

Two are implemented. The one that **ships** is `--source nbar`.

### `--source nbar` — NASA GIBS (shipped)

GIBS serves MCD43A4 anonymously as an RGB composite of bands 1 / 4 / 3, with
a display stretch applied. The decisive property is that **the stretch is the
published MODIS Rapid Response piecewise-linear curve, which is monotone and
therefore inverts exactly**:

```
IN  = [0, 30, 60, 120, 190, 255]
OUT = [0, 110, 160, 210, 240, 255]
reflectance = -0.01 + interp(dn, OUT, IN) * 1.11 / 255
```

Undo it and what is left is per-band MODIS reflectance. The whole globe is
covered, poles included.

Its costs, stated: the archived tiles are JPEG, so the inversion recovers the
stretch exactly but not the compression; and snow and ice saturate the
stretch at DN 255, which inverts above 1.0 and is clamped (1.2% of texels).

### `--source pc` — Microsoft Planetary Computer (implemented, blocked)

The same product's COGs, served anonymously as a dynamic mosaic, giving raw
`int16` at the 1×10⁻⁴ scale with an explicit fill — no stretch to invert, no
JPEG, no clamp. Strictly better numerics, and everything about the path is
verified working *except* that it cannot currently produce a complete globe.

Preconditions, measured against the live service rather than assumed, and
encoded in the tool:

- The mosaic must be **registered first** (`POST /api/data/v1/mosaic/register`
  with the collection and a datetime window), which returns a search id.
- `collection=modis-43A4-061` is **mandatory** — 422 without it. (The SAS
  collection id is also `modis-43A4-061`; `modis-061-cogs` is a storage
  container name and returns *"No collection found"*.)
- **z ≥ 5 returns 200; z3 and z4 return 204**, as does the `WorldCRS84Quad`
  tile matrix set at every zoom. `WebMercatorQuad` is the only grid served.
- A 2048-pixel tile request returns **502**. 256 is the served size.

**Why it is not the shipped source.** At z5, tile row `ty = 4` — Mercator
latitudes ~61.6 N to ~66.5 N, a continuous ring through the boreal belt —
returns **HTTP 500 deterministically**. Not rate limiting and not transient:
five attempts with exponential backoff to 16 s all fail, a single serial
request with no concurrency fails identically, and the failure returns in
0.7 s rather than timing out. Neighbouring rows return 200 with valid data
from the same registered mosaic. A mosaic missing a ring of the northern
hemisphere is not a raster this tool will bake, so it refuses.

z5 was the chosen zoom and the reasoning is worth keeping: 32 × 32 tiles of
256 px = **8192 × 8192**, four times the angular resolution of the
2048-column DEM, so albedo detail outruns terrain detail rather than
limiting it, at a download comparable to the 478 MB ETOPO grid. z6 would be
four times that for detail the 19.5 km/texel terrain cannot match.

WebMercator is also undefined past **±85.05113°**, so that route cannot
cover the polar caps. The tool does not paper over it: coverage stays 0
there and the shader falls back to the procedural classification, which is
correct for it — the Antarctic plateau is ~2 800 m against a modelled
snowline of ~1 200 m, so the fallback paints snow, which is what is there.

## Processing steps

1. **Fetch** four dates, one per season, at twice the output resolution.
2. **Invert** the Rapid Response stretch to per-band reflectance.
3. **Mask** GIBS's ocean/no-data sentinel `RGB(128,128,128)` with a ±4
   tolerance, because the archived tiles are JPEG and ringing moves it by a
   code or two near coastlines.
4. **Min-composite** across the four dates. Cloud and seasonal snow only
   ever *brighten* a pixel, so the darkest valid observation rejects both.
   That is what is wanted here precisely *because* the snowline is modelled
   from temperature at render time: baking one July's snow into the albedo
   and then adding modelled snow on top would double-count it.
5. **Area-average** down to the output grid. Not point sampling, for the
   reason the DEM bake gives: a point-sampled decimation of a 500 m product
   onto a 19.5 km grid is aliased, and the engine's smoothstep-bilinear
   magnifies whatever it is handed rather than repairing it.
6. **Coverage** is the **AND** of two independent statements: the DEM says
   this texel is land, and the source actually observed it. DEM-only would
   paint sentinel grey into a permanently clouded valley; source-only would
   paint land colour onto sea ice the satellite happily measured.
7. **Quantise** to RGBA8 with a gamma-2.0 encoding.

### The land mask comes from the DEM, not from the imagery

Coverage is 255 where `earth_lite.ptdem` has elevation ≥ 0. Deriving it from
the engine's *own* elevation grid means the coastline in the albedo raster
and the coastline the terrain renders are the same curve by construction —
they cannot drift apart by a texel, and a texel of ocean colour on land is a
visible artefact exactly where the eye is already looking.

## The encoding

```
reflectance = (value / 255)^2        value uint8, per RGB channel
coverage    = A / 255                1 = real land data
```

Gamma 2.0 rather than linear 8-bit, and that is not a style choice. A linear
byte's 1/255 = 0.0039 step sits against closed-canopy forest's ~0.018
reflectance — a **22% quantisation error on the darkest land there is**.
Gamma 2.0 puts that step at 0.0011 (5.8%), and snow's at 0.8%. The decode
costs the shader one multiply, which matters because it does four per
bilinear tap.

`float16` is not usable for the same reason it was not usable for the DEM:
three halves per texel is 6 bytes against 4, for precision the source does
not have.

## Container format

40 bytes of header, then the samples — deliberately the same shape as
`DemHeader`, so there is one wire format to learn and the two grids register
on the same convention. Mirrored by `pt::planet::AlbedoHeader` in
`src/renderer/Planet/SurfaceAlbedo.h`; the two are a wire format and there is
no generated header between them.

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `PTALB001` |
| 8 | 4 | `uint32` width (columns, west→east) |
| 12 | 4 | `uint32` height (rows, north→south) |
| 16 | 8 | `float64` scale |
| 24 | 8 | `float64` offset |
| 32 | 4 | `uint32` flags — **bit 0 = measured albedo** |
| 36 | 4 | `uint32` reserved |
| 40 | … | width × height RGBA8, row-major |

Pixel-centre registered on exactly the DEM's convention: texel *(x, y)* sits
at `lon = -π + (x + ½)·2π/W`, `lat = +π/2 − (y + ½)·π/H`.

**Flag bit 0 exists so the distinction lives in the container, not only in
this document.** The engine logs which kind it loaded, once, so a render is
never quietly attributed to a measurement it did not come from.

## What is committed here

`assets/planet/earth_lite.ptalb` — 2048 × 1024 RGBA8, **8.39 MB**, 19.5 km
per texel at the equator, the same grid as the DEM.

The shipped raster is 2048 wide although the source is fetched at 4096: the
extra resolution is spent on **anti-aliasing the bake**, not on the asset.
33.5 MB of RGBA8 does not belong in a source repository, and biome
boundaries are an order of magnitude softer than the coastlines the DEM's
2048 columns exist to resolve. `--width` bakes a finer one for anyone who
wants it.

## Validation

Measured, not asserted. `python3 tools/fetch_planet_albedo.py --validate`
reproduces this table; `tests/pt_planet_albedo_test.cpp` pins its structure.

The first check is luminance against published visible-band hemispherical
reflectance (Budyko 1974, *Climate and Life*; Ahrens, *Meteorology Today*
11th ed. table 2.2):

| site | baked RGB | lum | published | ratio |
|---|---|---|---|---|
| Amazon −5, −62 | 0.0101, 0.0472, 0.0262 | 0.0378 | 0.04 | 0.94 |
| Congo 0, 22 | 0.0205, 0.0509, 0.0326 | 0.0431 | 0.04 | 1.08 |
| Boreal 60, 100 | 0.0095, 0.0296, 0.0164 | 0.0244 | 0.04 | 0.61 |
| Sahara 23, 12 | 0.3977, 0.2837, 0.1347 | 0.2971 | 0.30 | 0.99 |
| Arabia 22, 45 | 0.3478, 0.2776, 0.1713 | 0.2849 | 0.30 | 0.95 |
| Australia −25, 130 | 0.2168, 0.0966, 0.0665 | 0.1200 | 0.16 | 0.75 |
| Tibet 33, 88 | 0.2461, 0.1869, 0.1182 | 0.1945 | 0.20 | 0.97 |
| Greenland 72, −40 | 0.5131, 0.5331, 0.5678 | 0.5313 | 0.55 | 0.97 |
| Antarctica −80, 20 | 0.6922, 0.7274, 0.7694 | 0.7229 | 0.75 | 0.96 |

Median ratio **0.96**. Area-weighted mean land reflectance **0.160** —
weighted by cos(latitude), because an unweighted mean over an
equirectangular grid over-counts the poles by 1/cos(lat) and the poles are
ice, which reports 0.35 for a raster whose honest value is 0.16.

Ice reads low on purpose: the bake is a min-composite across the year, so
what is left on an ice sheet is its melt-season bare ice (genuinely 0.5–0.7)
rather than fresh snow's 0.9, and the snowline model puts the snow back.

The second check is the decisive one, because **a grey sphere passes the
first**: every vegetated site must have GREEN above both RED and BLUE
(chlorophyll absorbs at 430 and 662 nm), every soil site must have
RED > GREEN > BLUE (iron oxide absorbs toward the blue), and every ice site
must be neutral to within 15%. **All nine pass.** That is what makes this
land *cover* rather than a brightness map.

## The two models the raster is modulated by

The raster is not the whole answer, and was never meant to be. At 19.5 km
per texel it cannot know that a 40° face sheds its soil, and it is one
year's composite so it cannot know where the snowline is on a body whose
radius the user has changed. Both models are derived and cited in
`src/renderer/Planet/SurfaceAlbedo.h`; in summary:

- **Slope → rock exposure**, ramping from the ~30° threshold hillslope
  (Burbank et al. 1996, Nature 379:505) to 45°. The ramp this replaced was
  zero below **60°** and did not saturate until 84°, so in practice it
  exposed no rock at all — which is why the Himalayas rendered uniform
  white. At the `planet_surface` camera the measured rock fraction now has a
  **median of 0.918**.
- **Temperature → snowline**, replacing `anchor × cos(site latitude)`. The
  warm-season freezing level, from the zonal-mean temperature (North,
  Cahalan & Coakley 1981, Rev. Geophys. 19:91) plus half the annual range,
  divided by the 6.5 K/km environmental lapse rate (ICAO / ISO 2533),
  anchored at the measured tropical ELA (Kaser & Osmaston 2002). Against
  published equilibrium-line altitudes at 0/45/60/80° it halves the mean
  error, 245 m against 441 m — and, unlike the cos law, it is evaluated at
  the *shaded point*, so a globe seen from orbit no longer has one snowline
  everywhere.

## What is *not* modelled

- **Seasons.** One composite. MCD43A4 is daily and twelve monthly bakes
  would be twelve times the asset; the temperature-derived snowline supplies
  the part of seasonality the eye actually reads.
- **Snow, in the raster, deliberately.** See the min-composite above.
- **Slope and snowline on the analytic body.** #300 puts the raster on the
  P3 backstop sphere too, because from orbit that is what the ray meets —
  but a sphere has no slope and the backstop's radius is not a terrain
  height, so applying either model there would be inventing a value rather
  than modulating one.
- **Specular water.** Ocean is masked out entirely and belongs to
  `MAT_WATER` (#259). A nadir-viewing satellite's measurement of the sea is
  not a reflectance a path tracer should be handed.
- **Anisotropy.** The raster is a Lambertian albedo. The full MCD43 BRDF
  kernels (isotropic, volumetric, geometric) are in the same product and
  would give real bidirectional terrain — a phase of its own.
- **A Köppen-style procedural classification.** The no-data fallback is the
  pre-existing elevation-and-slope function, improved but not replaced. A
  latitude-and-moisture classifier for bodies with no data is the part of
  #300 this change did not build.
- **City lights.** Still nothing; see the elevation half of this file.

# demont-engine: integration/parallel-batch-1 — manual test guide

This guide enumerates every feature / fix / change that landed in
`integration/parallel-batch-1` so you can smoke-test before the
consolidated `integration → main` merge. Organized by area; each entry
lists the PR number, what it does, new cvars / cfg fields, recommended
test fixture, and scope / known limitations.

**Integration HEAD when written:** `6a161b5313840f7d2c215493a29cb3f2b33cc05d`
**Branch fork point from main:** `d3df587` (`docs(agents): use mkdir-lockdir (portable) instead of flock (linux-only)`)
**Total PRs merged into batch-1:** 26 feature/fix PRs (#120-#128, #137, #141-#159, #163, #165, #166)

This is the first integration batch since the parallel-agent protocol
went live. Roughly two dozen feature branches were merged into the
shared integration branch over a single weekend; this guide is the
manual-test inventory needed before promoting integration to main.

> Caveat from project memory: Mac-Vulkan was historically untested
> before 2026-05-16 -- the user ran Metal + Software on Mac only.
> The first Vulkan launches on Mac and Win hit latent regressions
> (nullDescriptor, shaderStorageImage formats, VkResult-not-checked
> code paths). The Vulkan-RTX correctness floor is *freshly stable*
> as of THIS batch -- treat Vulkan-side smoke results as load-bearing.

## Quick smoke checklist

- [ ] Launch demont with default scene, no flags. Expect a Cornell-box
      render in <10 seconds on M4 Max / Metal.
- [ ] Run `ctest --test-dir build/mac-release` (or wherever you build).
      All unit tests pass.
- [ ] Run the golden-image regression matrix locally:
      `ctest --test-dir build/mac-release -L golden`. Expect 44/44 cells
      to pass on Mac (software + metal lanes).
- [ ] Run `pt_render_one_frame --scene tests/goldens/scenes/cornell_csg.cfg --backend vulkan --frames 16` on a Vulkan host
      (Linux or Win box; expect 16 frames, exit 0, non-black PNG).
- [ ] Each feature below has a "How to test" line; tick when verified.

---

## Rendering / Path Tracing

### PR #154 — Hierarchical light tree (closes #129)

**What:** Conty Estevez & Kulla 2018 light tree. CPU build is top-down
median split on the longest centroid axis; GPU traversal is 2-way
stochastic split weighted by combined cluster importance (intensity *
inverse-square * emission-cone cosine * receiver Lambert). Pushes the
practical light-count ceiling from ~100 (PR #145 naive uniform) to
~1000-5000 emitters at 1 spp. Author reports 48.5x mean-variance
reduction at 200 lights vs naive single-pick.

**Cvars added:** `r_light_tree` (CVAR_ARCHIVE, default `1`).
Set to `0` to fall back to PR #145's naive uniform single-pick for A/B.

**Test fixture:** `tests/goldens/scenes/light_tree_200_pts.cfg`
(200 random point lights, 1 spp, sun below horizon so only the lights matter).

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/light_tree_200_pts.cfg \
  --backend metal --frames 16 --out /tmp/lt_tree.png
# Toggle r_light_tree 0 in the cfg / via console and re-render -> /tmp/lt_naive.png
# Eyeball: tree image should be visibly smoother in the lit region.
```

**Scope / known issues:** Per-light PDF math is unchanged from PR #145;
only the pick PMF changes. PC-Vulkan dispatch built but not yet
load-bearing -- needs verification on Win/Linux box.

### PR #155 — ReSTIR DI Phase A (issue #78)

**What:** Bitterli et al. 2020 spatiotemporal reservoir resampling for
direct illumination. Bridges PR #145 naive NEE and PR #154 light tree:
one shadow ray per pixel regardless of light count. Four new compute
kernels (`RestirReservoir/Temporal/Spatial/Final`) chain after PathTrace
and feed denoised output into MetalFX/SVGF.

**Cvars added (all CVAR_ARCHIVE):**
- `r_restir` (default `1`) — master toggle
- `r_restir_temporal` (default `1`) — temporal reuse pass gate
- `r_restir_spatial` (default `1`) — spatial reuse pass gate
- `r_restir_k_candidates` (default `8`) — WRS candidates per pixel (1..64)
- `r_restir_bias` (default `biased`) — `biased` or `unbiased`

**Test fixture:** Same `light_tree_200_pts.cfg` or any scene with many
analytic lights; pair with `r_denoiser metalfx` to see the variance
reduction.

**How to test:** On Metal, render a 100+ point light scene with
`r_denoiser=metalfx`; toggle `r_restir 0` vs `r_restir 1` -- shadow
contact and noise should visibly improve.

**Scope / known issues:** DI only (no GI). Metal-only dispatch -- Vulkan
plumbing deferred (shaders compile to SPIR-V but the engine doesn't
dispatch the Restir kernels on Vulkan yet, same status as SigmaShadow).
Static + slow cameras work cleanly; fast camera motion can show known
temporal artifacts (temporal cap `M=20` is the lever).

### PR #145 — Analytic light primitives + Lambert NEE (issue #73)

**What:** First-class analytic lights independent of the env-map HDRI.
Four types: `point` (W/sr delta light), `spot` (cone falloff), `sphere`
(diffuse area, W/m^2/sr), `quad` (one-sided diffuse area). Path tracer's
Lambert NEE block uniformly picks one light per sample and shadow-tests
against the existing dielectric+cloud transmittance chain.

**Cvars added:** None. New console commands instead:
`light_point` / `light_spot` / `light_sphere` / `light_quad` /
`light_list` / `light_clear` / `light_remove`.

**Test fixtures:**
- `tests/goldens/scenes/light_primitives_smoke.cfg` (5 point lights, Lambert plane + sphere)
- `tests/goldens/scenes/light_primitives_mixed.cfg` (one of each light type)

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/light_primitives_mixed.cfg \
  --backend metal --frames 16 --out /tmp/lights_mixed.png
# Expect: 4 visibly distinct illumination patches; cast shadows.
```

**Scope / known issues:** Naive single-pick scales OK for ~100 lights;
PR #154 light tree and PR #155 ReSTIR resample to extend further.

### #176 — Industry-standard light ergonomics: color / cd / lm / nits / exposure

**What:** Ergonomic sugar variants over the canonical `light_*`
primitives. Same physics, same internal W/sr (point/spot) or W/m^2/sr
(area-light radiance) storage, but artist-facing input in
chromaticity-color-times-scalar, candela, lumens, nits, and/or
exposure stops. Conversion is the single-wavelength 555nm peak
photopic approximation (683.002 lm/W); see the `_cd` / `_lm` /
`_nits` description text in-engine for the per-variant exact factor.

**Cvars added:** None. New console commands only:
| Primitive | _color | _cd | _lm | _nits | _exposed |
|---|---|---|---|---|---|
| `light_point`  | yes | yes | yes | -- | yes |
| `light_spot`   | yes | yes | yes | -- | -- |
| `light_sphere` | yes | -- | -- | yes | -- |
| `light_quad`   | yes | -- | -- | yes | yes |

**Cfg round-trip:** Save path is canonical (`SaveArchivedCvars` writes
the W/sr-direct form), so the conversion happens exactly once at
authoring time. Reloading a saved cfg is a pure float identity.

**How to test:**
```
# All three variants below produce the same internal W/sr value:
light_point        1 0 5 0   0.14641 0.14641 0.14641     # canonical
light_point_color  2 0 5 0   1 1 1   0.14641             # color * scalar
light_point_cd     3 0 5 0   1 1 1   100                 # 100 cd -> 0.14641 W/sr
light_point_lm     4 0 5 0   1 1 1   1255.2              # 4*pi*100 lm -> 0.14641 W/sr
light_point_exposed 5 0 5 0  1 1 1   0.07321 1.0         # 0.07321 * 2^1 = 0.14641

# Sanity: light_list should report all five at identical (0.146, 0.146, 0.146).
```

**Test fixture:** `tests/pt_light_decompose_test.cpp` (pure-math pin
of the conversion constants + decomposition algebra + cfg round-trip
identity + linearity).

**Scope / known issues:** Single-wavelength (555nm) photopic
approximation only — per-channel spectral upsampling is out of scope
(deferred). Spot `_lm` uses the omnidirectional 4*pi conversion so it
under-states on-axis intensity for narrow cones; prefer `_cd` (which
maps to on-axis catalog candela directly) for tight-beam authoring.
IES profile import deferred.

### PR #147 — SDF fractals: Mandelbulb / Mandelbox / Apollonian (closes #99)

**What:** Three iconic fractal distance estimators as a new SDF
fractal-leaf shape variant. New `shaders/SdfFractals.slang` module
(separate from `SdfPrimitives.slang`); iteration-relaxed surface
epsilon, central-diff 4-tap tetrahedron normals, hard 512-step ceiling.

**Cvars added (all CVAR_ARCHIVE):**
- `r_sdf_fractal_power` (default `8`) — Mandelbulb power exponent
- `r_sdf_fractal_iters` (default `12`) — DE iteration count
- `r_sdf_de_eps_scale` (default `1.0`) — epsilon scale factor

**New console commands:** `sdf_mandelbulb`, `sdf_mandelbox`, `sdf_apollonian`.

**Test fixture:** `tests/goldens/scenes/sdf_fractals.cfg`

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/sdf_fractals.cfg \
  --backend metal --frames 8 --out /tmp/fractals.png
# Expect: Mandelbulb (orange) centre, Mandelbox (blue) left, Apollonian
# gasket (green) right. Per-pixel silhouette noise at 1 spp is the
# DE-approximation noise floor -- this is documented and accepted.
```

**Scope / known issues:** Mixed fractal+analytic CSG rejected at parse
time. No motion-vector reprojection for fractal pixels (deferred to NRD).
Cross-backend `eps`-mode golden deferred (needs golden-matrix eps-tolerance mode).

### PR #151 — SDF Phase 2: procedural / displaced + forward-mode normals (issue #98)

**What:** Procedural ops (noise displacement, twist, bend, infinite +
bounded domain repetition) and forward-mode autodiff normals
(~1.5x distance-eval cost vs central-diff's 6x). Tri-planar texture
projection helper as binding point for issue #74 (textured materials).

**Cvars added (all CVAR_ARCHIVE):**
- `r_sdf_normal_mode` (default `0`) — 0=central diff, 1=forward autodiff
- `r_sdf_displace_octaves` (default `4`)

**New console commands:** `sdf_displace_noise`, `sdf_twist`, `sdf_bend`,
`sdf_repeat`, `sdf_repeat_limited`.

**Test fixture:** `tests/goldens/scenes/sdf_phase2_procedural.cfg`

**How to test:** Render the procedural scene; sanity check that
`r_sdf_normal_mode 0` vs `1` produce visually-similar output
(<= 1e-3 mean delta per the issue acceptance test, author measured 0.029/255).

**Scope / known issues:** Phase 1 golden `sdf_smin_row.cfg` should remain
unchanged (non-procedural clusters keep the analytic-gradient path).

### PR #153 — Water plane phase 1 (closes #134)

**What:** New `MAT_WATER` (id 3) shaded plane with normal-map waves,
Schlick Fresnel (F0=0.02 for water), Snell refraction, Beer's-law
per-channel absorption. Wave normals = 2-octave value-noise + shared
cloud-wind accumulator. Caustics through the surface respect the
per-channel absorption tint via the refractive NEE chain.

**Cvars added (all CVAR_ARCHIVE):**
- `r_water_absorption_r/g/b` (defaults 0.45 / 0.15 / 0.05 — teal Caribbean)
- `r_water_ior` (default `1.33`, clamped [1.0, 2.4])
- `r_water_wave_scale` (default `0.3` cycles/m)
- `r_water_wave_amplitude` (default `0.2`)
- `r_water_wave_speed` (default `1.0`)

**Test fixture:** `tests/goldens/scenes/water_pool.cfg`
(half-submerged red sphere + fully-submerged yellow sphere + sandy floor,
wave_speed=0 for determinism).

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/water_pool.cfg \
  --backend metal --frames 16 --out /tmp/water.png
# Expect: visible refraction of submerged spheres, teal tint with
# depth, Fresnel highlight on water surface.
```

**Scope / known issues:** Reuses `r_refract_bounces` budget -- must be
`>=1` to see refracted underwater geometry. `prim_sphere` / `prim_plane`
/ `scene_load` now accept "water" as a material name.

### PR #148 — Voxel destruction Phase 1 (closes #140)

**What:** New `src/destruction/` subsystem. `VoxelGrid` (packed bitmap +
on-disk cache) + `Voxelizer` (sphere SDF, SdfPrim host evaluator
mirroring `SdfPrimitives.slang`, triangle-mesh +X ray parity inside-test).
Each occupied voxel becomes a `kSdfShapeBox` cluster under a reserved
id range (`0xF0000000+`), rendered through the existing sphere-trace path.

**Cvars added (all CVAR_ARCHIVE):**
- `r_voxel_size` (default `0.1` m)
- `r_voxelize_demo` (default `0`)

**New console commands:** `voxelize_object`, `voxelize_save`,
`voxelize_list`, `voxelize_clear`. Also a new `--smoke-late-exec=<cfg>`
CLI flag for fixtures that must run commands *after* the CSG seed.

**Test fixture:** None checked into `tests/goldens/scenes/` (the PR's
captures live in worktree-local `captures/`).

**How to test:**
```
# Drop into the console:
csg_box 1 1 1 0 0.5 0
voxelize_object <id> 0.1
cvar set r_voxelize_demo 1
# Expect: original mesh replaced by ~950 chunky voxel boxes.
# Toggle r_voxelize_demo 0 -> reverts to original mesh (pixel-identical baseline).
```

**Scope / known issues:** Phase 1 *renders* voxelised meshes; it does
NOT destroy them yet (Phase 2 depends on rigid-body Phase 2 #138).
Adaptive resolution / chunk merging is Phase 3. CPU only.

### PR #143 — Spherical-Earth atmosphere + horizon dip (issue #51)

**What:** Replaces planar exponential atmospheric transmittance with a
curved-Earth chord-integrated Mie + Rayleigh optical depth (8-step
composite Simpson's). `procSky` rebiases `cosTheta` so high-altitude
renders get the geometrically-correct visible horizon dip.

**Cvars added:** `r_planet_radius` (CVAR_ARCHIVE, default `6378137.0` =
WGS-84 Earth radius). `0` falls back to legacy planar exponential (debug A/B
or for tiny scenes). Real values for other bodies: Moon 1737400, Mars
3389500, Venus 6051800.

**Test fixtures:**
- `tests/goldens/scenes/sunset_horizon.cfg` (eye-level 1.7m sunset, 2deg sun)
- `tests/goldens/scenes/sunset_altitude.cfg` (10km altitude, 2deg sun, curved)
- `tests/goldens/scenes/sunset_altitude_planar.cfg` (same scene, `r_planet_radius 0`)

**How to test:** Render `sunset_horizon` -- should look identical to
pre-#51 baseline (~3.7 arcmin dip is sub-pixel at 640x360). Render
`sunset_altitude` vs `sunset_altitude_planar` -- the curved version
shows visible horizon dip (~3.2 deg).

**Scope / known issues:** Phases 2-4 (transmittance LUT, multi-scatter +
sky-view LUTs, aerial-perspective volumetric LUT) deferred -- they are
perf opts over per-pixel Simpson's, not quality wins, at 1-30 km scales.

---

## Volumetrics / Clouds

### PR #146 — Heterogeneous clouds + god rays (issue #100)

**What:** Phase 4 of #96. Upgrades existing cloud march from uniform
single-scatter into heterogeneous multi-scattering raymarcher.
Wrenninge-style multi-scatter octaves; stratified cloud-only NEE shadow
rays resolve god-ray contrast through cumulus gaps; cloud-specific
Henyey-Greenstein g decoupled from atmospheric haze.

**Cvars added (all CVAR_ARCHIVE):**
- `r_vol_density_scale` (default `1.0`) — linear cloud sigma_t multiplier
- `r_vol_phase_g` (default `0.8`) — cumulus forward-scatter g (canonical for cumulus silver lining)
- `r_vol_multiscatter_bounces` (default `2`, clamp 0..4) — Wrenninge octave count

**Test fixture:** `tests/goldens/scenes/clouds_godrays.cfg`
(sunset, broken cumulus, sun at 8deg elevation, layer 250-800m, 0.65 coverage,
camera below looking east).

**How to test:** Render the fixture -- god rays radiating from the sun
through cumulus gaps should be clearly visible on both Metal and Vulkan.

**Scope / known issues:** Frame budget ~6 ms/frame at 16-spp 768x432 on M4 Max.

### PR #126 — Cloud curl-noise detail + edge erosion (closes #117)

**What:** Bridson 2007 divergence-free curl-noise displacement on cloud
sample positions before density eval. Secondary high-frequency edge
erosion layered atop existing `r_clouds_detail` so dense cores stay
protected while soft margins fray into wisps.

**Cvars added (all CVAR_ARCHIVE):**
- `r_clouds_curl_amount` (default `0.0`) — curl displacement magnitude in metres
- `r_clouds_curl_scale` (default `0.01`) — curl frequency in cycles/m
- `r_clouds_erosion` (default `0.0`) — secondary erosion amount [0..1]

**Regression safety:** `r_clouds_curl_amount 0` + `r_clouds_erosion 0`
produce a bit-identical PNG to pre-#117 main (verified sha256 match in PR).

**How to test:** Render the existing clouds_godrays fixture twice --
once with the new knobs at default 0, once with `r_clouds_curl_amount
0.5 r_clouds_erosion 0.4`. The second should show coherent eddy-shaped
displacement concentrated around cloud edges.

**Scope / known issues:** Layer-relative amplitude in metres -- scale
with `r_clouds_freq` if you change bulk feature size.

### PR #156 — Smoke density emitters Phase 1 (issue #136)

**What:** Cheap chimney / campfire smoke that rides the existing
volumetric cloud march via additive density emitters. Each emitter is a
Gaussian-falloff density blob with parametric drift (`base + velocity * t`).
The cloud NEE / multi-scatter / aerial-perspective pipeline shades smoke
identically to a cloud cell, so plumes get sun + sky lighting for free.

**Cvars added (all CVAR_ARCHIVE):**
- `r_smoke_enabled` (default `0` — bit-exact vs pre-#136 main)
- `r_smoke_max_emitters` (default `8`, host-clamped 0..16)

**New console commands:** `smoke_emit`, `smoke_clear`, `smoke_list`.

**Test fixture:** `tests/goldens/scenes/smoke_emitter_basic.cfg`
(two plumes at chimney altitude).

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/smoke_emitter_basic.cfg \
  --backend metal --frames 16 --out /tmp/smoke.png
# Toggle r_smoke_enabled 0 -> uniformly pale sky.
# Set r_smoke_enabled 1 -> visible plume density near sun.
```

**Scope / known issues:** No simulator, no Eulerian grid -- Phase 2/3
deferred. Latent bug fixed: Metal `bound_buf_` was [12] entries, slot
12 (light primitives #73) silently dropped causing uniformly black
output on scenes using slot 13. Array bumped to [16] in this PR.

---

## Denoising

### PR #137 — SIGMA shadow denoiser (issue #115)

**What:** Mac-parity equivalent to NVIDIA's NRD library SIGMA. Restores
sharp sun-shadow edges that SVGF / MetalFX otherwise smudge. PathTrace
writes per-primary-hit sun-NEE visibility into a separate G-buffer;
`shaders/SigmaShadow.slang` 5x5-bilateral-filters it (depth + normal
edge-stops); the result is multiplied into post-radiance-denoise HDR.

**Cvars added:** `r_shadow_demod` (CVAR_ARCHIVE, default `1`).
Set to `0` to disable shadow demodulation.

**Test fixture:** Any scene with strong sun shadows; default `cornell_csg`
exercises it through the SVGF path.

**How to test:** Render a scene with sharp sun shadow on a flat plane,
once with `r_denoiser svgf_atrous` and `r_shadow_demod 1`, once with
`r_shadow_demod 0`. The `1` version should have crisper shadow boundary.

**Scope / known issues:** Metal-only dispatch for now. Sibling NRD-library
PR #123 is scaffolding-only on Vulkan (no per-frame dispatch walker).

### PR #123 — NRD library scaffolding (issue #50)

**What:** Stage-1 scaffolding for NVIDIA RayTracingDenoiser on Vulkan.
New CMake option `PT_ENABLE_NRD` (default **OFF**), hard-gated to
non-Apple Vulkan builds. NRD v4.17.3 wired via `FetchContent` (SHA256
pinned). `VulkanNrdLibDenoiser` class skeleton that creates the
SIGMA_SHADOW instance and logs version + dispatch count, then
*passthrough copies* output (no actual dispatch walker yet).

**Cvars added:** None new (uses existing `r_denoiser`).

**How to test:** On a Linux/Win-Vulkan box:
```
cmake -B build -DPT_ENABLE_NRD=ON
cmake --build build --target demont
# At runtime, look for "NRD v4.17.3" or similar in startup log.
```
On Mac: configure with `-DPT_ENABLE_NRD=ON` -- expect a friendly
warning and `PT_NRD_ACTIVE=OFF` fallback; build succeeds with no NRD link.

**Scope / known issues:** Actual per-frame dispatch walker (~300-500 LOC
of resource-translation boilerplate) is deferred to a follow-up PR.
Existing `r_denoiser nrd` value keeps current behaviour (alias for SVGF
a-trous). Unverified on Linux -- needs CI or manual Win/Linux build to
flush wiring snags.

### PR #150 — SVGF albedo demodulation (closes #119)

**What:** Divide noisy radiance by primary-hit albedo on SVGF input,
then multiply back in a new remod pass so depth/normal/luminance edge-stops
denoise *lighting* signal rather than fighting surface texture.

**Cvars added:** `r_svgf_albedo_demod` (CVAR_ARCHIVE, default `1`).
A/B test toggle.

**How to test:** Render `cornell_csg` with `r_denoiser svgf_atrous` and
`r_svgf_atrous_passes 5`, once with `r_svgf_albedo_demod 0`, once with
`1`. The `1` version preserves the dielectric-cube caustic ring's
R/G/B channel separation; the `0` version smears them.

**Scope / known issues:** Sky pixels (`albedo == float3(0)`) bypass
demod via a sky-eps cutoff (`1/65536`) so the multiply-by-zero would
not erase the sky. `r_svgf_albedo_demod 0` is bitwise-identical to
pre-PR baseline (verified `imgdiff max_delta = 0.0000`).

### PR #149 — MetalFX specular guidance (closes #118)

**What:** Plumbs three MetalFX `MTLFXTemporalDenoisedScaler` guidance
inputs that PR #114 left empty: `specularAlbedoTexture`,
`roughnessTexture`, `specularHitDistanceTexture`. Without them MetalFX
can't differentiate specular from diffuse response and produces 8x8
halos around bright reflections / metals (the failure mode user reported
on the diamond / glass-sphere scenes).

Three new G-buffers allocated only for MetalFX-family denoiser kinds:
- `specular_albedo_tex` (RGBA16F) — per-pixel F0 (metals: F0=albedo; dielectrics: F0=0.04)
- `roughness_tex` (R32F) — per-pixel surface roughness [0,1]
- `specular_hit_distance_tex` (R32F) — `primary_t * (1 - roughness)` MVP proxy

**Cvars added:** None new (guidance plumbed via `r_denoiser` family of values).

**How to test:** Render a scene with bright metallic / dielectric
spheres and `r_denoiser metalfx`. Specular halos around reflections /
metals should be tight, not smeared into 8x8 tiles.

**Scope / known issues:** Apple Silicon's 8-RW-textures-per-kernel cap
forced these three to use `WTexture2D` (write-only). Hit-distance is
MVP-proxy; real reflection-ray trace deferred.

---

## Physics

### PR #144 — Verlet physics Phase 1 (closes #132)

**What:** First subsystem under `src/physics/`. `Particle` POD +
`PhysicsSystem` with fixed 256-particle pool, Verlet integration,
sphere-plane (y=0) + pairwise sphere-sphere position correction.
Per-substep damping bleeds kinetic energy so falling spheres settle.

**Cvars added (all CVAR_ARCHIVE):**
- `phys_enabled` (default `0`)
- `phys_gravity_y` (default `-9.81`)
- `phys_substeps` (default `8`)
- `phys_damping` (default `0.99`)

**New console commands:** `phys_drop`, `phys_clear`, `phys_status`.

**Test fixtures:**
- `tests/goldens/scenes/phys_smoke.cfg` (4 spheres dropped, with late-exec)
- `tests/goldens/scenes/phys_smoke_late.cfg` (the late-exec partner)
- `tests/goldens/scenes/phys_smoke_off.cfg` (negative-control for `phys_enabled=0` no-op)

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/phys_smoke.cfg \
  --smoke-late-exec=tests/goldens/scenes/phys_smoke_late.cfg \
  --backend metal --frames 60 --out /tmp/phys.png
# Expect: 4 spheres in mid-air or settled on the ground depending on frame.
```

**Scope / known issues:** No friction, restitution, sleeping islands,
box / capsule shapes, CSG mesh collision, or GPU compute -- all deferred.
O(N^2) sphere-sphere intentionally capped at 256 particles.

### PR #152 — Rigid body Phase 2a (issue #138)

**What:** Extends Verlet pool with parallel rigid-body pool carrying
quaternion orientation, per-body inertia tensor, world-frame angular
velocity. XPBD per Mueller et al. 2020 ("Detailed Rigid Body Simulation
with Extended PBD"): predict pos+orientation, apply position constraints,
finite-diff both linear and angular velocities at substep end.

**Cvars added:** None new (reuses Phase 1 family).

**New console commands:** `phys_drop_sphere <x> <y> <z> [r] [m] [red green blue]`,
`phys_drop_box <x> <y> <z> <hx> <hy> <hz> [m] [red green blue]`. The
trailing RGB triplet is the #181 colored-rigid-body extension; omit it to
get the pre-#181 warm-grey sphere / cool-blue box defaults.

**Test fixtures:**
- `tests/goldens/scenes/phys_rb_smoke.cfg`
- `tests/goldens/scenes/phys_rb_smoke_late.cfg`

**How to test:** Drop several rigid bodies via the console; verify they
fall, rotate, and bounce. Boxes use bounding-sphere collision (Phase 2a
MVP) so they'll roll but not lie flat.

**Scope / known issues:** Phase 2a is rigid-body **core only** -- no
box-box / box-plane SAT (boxes collide as their bounding sphere). Real
OBB primitive renders via reused analytic-sphere at bounding radius.
No persistent contacts, no sleeping, no joints / motors / CCD / friction
/ restitution. No capsules.

### PR #186 — r_phys_debug_visualize: colour rigid bodies by velocity (issue #181)

**What:** Per-frame debug aid that overrides each physics-driven sphere's
albedo to encode linear speed magnitude on a 3-stop linear-RGB ramp.
Recovers speed from the implicit-velocity Verlet invariant
`(curr_pos - prev_pos) / sdt` at the end of `Engine::StepPhysics`. Particles
(Phase 1) and rigid bodies (Phase 2a) both participate. Angular speed on
rigid bodies is intentionally NOT folded in -- the viz is for translational
motion, so a tumbling-but-stationary body reads as blue. Original albedos
are cached on first override and restored when the cvar transitions back
to 0, so toggling on/off is non-destructive.

**Speed -> colour mapping (linear-RGB):**
- 0 m/s   -> blue   `(0.10, 0.30, 1.00)`
- 5 m/s   -> green  `(0.20, 1.00, 0.30)`
- 10+ m/s -> red    `(1.00, 0.20, 0.20)` (saturating above)

**Cvars added:**
- `r_phys_debug_visualize` (default `0`, `CVAR_NONE` -- per-invocation, never archived)

**New console commands:** None.

**Test fixtures:** Reuses `tests/goldens/scenes/phys_rb_smoke.cfg` /
`phys_rb_smoke_late.cfg` -- enable the cvar interactively after `exec`-ing
the fixture.

**How to test:**
1. `exec tests/goldens/scenes/phys_rb_smoke.cfg`
2. `r_phys_debug_visualize 1` -- spheres recolour: blue when stationary,
   green at moderate fall, red at peak descent (~10 m/s under default
   gravity + damping).
3. Watch them settle: as bodies hit the ground and lose energy they
   transition red -> green -> blue.
4. `r_phys_debug_visualize 0` -- albedos restore to whatever they were
   before the viz was enabled.

**Scope / known issues:** Pure debug aid; no perf budget claim. Mid-flight
edits via `prim_albedo` while the viz is active are overwritten next frame
and lost on toggle-off (the cache holds the albedo captured on first
override). Box rigid bodies render as bounding spheres per Phase 2a, so
the viz colours their bounding sphere; SAT-correct OBB primitive rendering
is Phase 2b.

---

## Sky / Astronomy

### PR #122 — Aurora borealis procedural overlay (issue #116)

**What:** Stateless post-denoise additive composite of curl-noise-driven
aurora ribbons. Mirrors StarsComposite architecture: same 5x5 min-depth
sky-pixel gate, runs once per frame on Metal between StarsComposite and
the bloom pyramid so green/red ribbons feed bloom + ACES.

Physics: Gaussian band centred at 24deg elevation (real 100-150 km
oxygen-line band). Pole-alignment falloff via `r_aurora_lat`.
Spectral mix: 557.7nm (O green), 630nm (O red blended toward zenith),
428nm (N blue at horizon). 1.5 cd/m^2 base — half-moon-lit sky scale.

**Cvars added (all CVAR_ARCHIVE):**
- `r_aurora` (default `0`)
- `r_aurora_intensity` (default `1.0`)
- `r_aurora_lat` (default `70` deg geomagnetic — real auroral oval)
- `r_aurora_animate` (default `1`)

**Test fixtures:**
- `tests/goldens/scenes/aurora_smoke.cfg`
- `tests/goldens/scenes/aurora_off_smoke.cfg` (control, `r_aurora 0`)

**How to test:**
```
pt_render_one_frame --scene tests/goldens/scenes/aurora_smoke.cfg \
  --backend metal --denoiser svgf_atrous --frames 32 --out /tmp/aurora.png
# Expect: green ribbons in upper sky, stars visible underneath.
```

**Scope / known issues:** Metal-only dispatch. Vulkan path leaves
`aurora_composite_pipeline_id_` at 0 so the gate cleanly collapses to
"no aurora". Aurora fully invisible above sun elevation +0.20 (day),
full strength below -0.10.

---

## Audio

### PR #141 — Audio subsystem MVP (issue #80)

**What:** Phase A skeleton of the audio plan. New `pt_audio` static lib
wrapping vendored `miniaudio` (single-header, public domain, v0.11.25).
Cross-platform: CoreAudio on macOS, WASAPI on Windows,
ALSA/PulseAudio on Linux. 3D-positioned WAV one-shots with 1/r distance
attenuation (clamped at 1 m) and equal-power stereo pan. 64-voice
lock-free pool; voice handles encode slot + generation so stale-handle
Stop() is a safe no-op.

**Cvars added:** None new in Phase A (issue's full `s_*` family deferred).

**New console commands:** `audio_play <path>`, `audio_stop`.

**Test asset:** `assets/audio/test_blip.wav` (48 KB, 0.5s 440Hz sine
decay, mono 16-bit PCM @ 48 kHz).

**How to test:**
```
# In console:
audio_play assets/audio/test_blip.wav
# Expect: audible blip from default speaker.
# Look for: [INFO] audio: device opened (48000 Hz, 2 ch, F32, miniaudio v0.11.25)
```

**Scope / known issues:** Issue title's headline (ray-traced occlusion +
reverb against TLAS) explicitly deferred. No HRTF, no music streaming
(one-shots only), no OGG/FLAC decode, no doppler. `Init()` returning
false (no device, CI lane without audio) is non-fatal — API no-ops cleanly.

---

## Particles

### PR #142 — Particle VFX system MVP (issue #82)

**What:** CPU-sim screen-space particle splatter. `ParticleSystem` with
Euler integrator, linear drag + gravity, fixed timestep clamp at 1/30 s.
New `shaders/ParticleComposite.slang` compute kernel — screen-space
billboard splats into post_denoise_hdr, additive Gaussian discs,
depth-gated. Dispatched after StarsComposite, before bloom pyramid so
HDR particle highlights downsample into halos.

**Cvars added (all CVAR_ARCHIVE):**
- `r_particles` (default `0` — master toggle)
- `r_particles_max` (default `1024`)

**New console commands:** `particle_emit smoke|spark|snow|snow_stop [x y z]`,
`particle_clear`, `particle_count`.

**Three spawn presets:** smoke (buoyant grey puff), spark (HDR orange
burst with gravity), snow (continuous area emitter with downward velocity).
Physics in metric units (positions m, velocities m/s, gravity 9.81 m/s^2).

**How to test:**
```
# In console:
cvar set r_particles 1
particle_emit smoke 0 0 -2
particle_emit spark 1 0 -2
# Expect: visible blob splats. Spark highlights downsample into bloom halos.
```

**Scope / known issues:** NOT visible in PT reflections / shadows / DOF /
motion blur — particles are a screen-space composite layer, NOT analytic
BVH primitives. NOT GPU-sim — CPU integrator. Metal-only dispatch (Vulkan
follow-up). Dispatch gate requires `denoiser_active_` (engine only
allocates `depth_tex_id_` when denoiser is on).

---

## glTF / Geometry import

### PR #128 — glTF 2.0 importer MVP (closes #79)

**What:** Single static mesh load from `.gltf` (with sidecar `.bin`) or
`.glb` (embedded buffer) via the existing mesh pipeline (triangle BVH +
optional hardware BLAS/TLAS — same upload path as CSG bake). POSITION
required; NORMAL imported if present, per-face geometric normal
synthesized otherwise. Triangle topology only.

**Cvars added:** None new.

**New console command:** `mesh_load_gltf <path>`. New CLI flag
`--smoke-late-exec=<cfg>` pairs with `pt_smoke_late_exec` cvar so smoke
fixtures can run commands requiring `csg_scene_` + `RegisterCommands`
to be live.

**Test asset:** `assets/gltf/Box.gltf` + `Box0.bin` + `Box.glb`
(Khronos canonical sample, CC BY 4.0 via Cesium, ~5.2 KB total).

**How to test:**
```
# Create /tmp/load_gltf_box.cfg:
mesh_load_gltf assets/gltf/Box.gltf
# Then:
demont --smoke-frames=8 --smoke-late-exec=/tmp/load_gltf_box.cfg \
  --smoke-capture-out=/tmp/gltf_box.png
# Expect: a shaded Box visible in the render.
```

**Scope / known issues:** Closes the static-mesh dimension only. PBR
materials, scene graphs, animation, lights, KTX2 stay open as explicit
follow-ups. Base-color factor + base-color texture (PNG/JPG via
stb_image) are parsed and recorded on the returned `GltfMesh` but
per-mesh material plumbing to the shader stays in issue #74's lane.
Non-triangle primitive types rejected with a clean console error (no crash).

---

## Console UX & tooling

### PR #157 — DEMONT_ASSET_ROOT env var + exe-parent fallback

**What:** `Engine::ReloadEnvMap()` + `EnsureStarMapUploaded()` were
opening assets via plain `std::fopen` on CWD-relative paths. Under
ctest, each golden cell sets a per-cell `WORKING_DIRECTORY` so the
relative open fails and the engine warns about missing HDR / BSC5.dat.

New `src/core/AssetPath.{h,cpp}` exposes `pt::ResolveAssetPath()` with
priority: 1) `DEMONT_ASSET_ROOT` env var, 2) exe-grandparent walk via
`GetModuleFileNameW` (Win) / `readlink("/proc/self/exe")` (POSIX),
3) cached after first resolve. `pt_render_one_frame` injects
`DEMONT_ASSET_ROOT` into env via `_putenv_s` / `setenv` so the child
inherits it.

**Cvars added:** None — pure plumbing.

**How to test:** Run ctest goldens; verify no "HDR cannot open file" /
"BSC load failed" WARN lines in the logs.

### PR #163 — Console UX: platform gating + smart resolve + dep warnings (closes #161, #162)

**What:** Bundles two console UX upgrades.
- **Per-value platform gating + dep warnings:** new `CVarValueFlag` enum
  (`CVAR_VALUE_MAC/WIN/LINUX`). Cvars carry an aligned `allowed_value_flags`
  vector so `r_denoiser` can mark `metalfx` as Mac-only and `nrd` /
  `optix_*` as Win/Linux-only. Wrong-platform value triggers a clear
  error listing available-on-this-platform values; the value is still
  written so a shared `demont.cfg` round-trips. `requires_predicate`
  for cross-cvar dependency warnings (e.g. `r_sun_elevation` only takes
  effect when `r_sky_mode procedural`).
- **Intelligent partial-match command resolution:** `deno metalfx`
  resolves to `r_denoiser metalfx`. Tie-breaks: exact > unique prefix >
  cvars over commands > alphabetical-first. Ambiguous prefixes log
  candidate list capped at 8.

**Cvars added:** `r_console_smart_resolve` (CVAR_ARCHIVE, default `1`).

**How to test:** From console:
```
> deno metalfx          # expect resolved to r_denoiser metalfx
> r_                    # expect ambiguous-prefix candidate list
> r_denoiser nrd        # on Mac: expect "is Windows/Linux-only" error
> r_sun_elevation 45    # if r_sky_mode is astronomical: expect [warn] hint
```

### PR #159 — Platform-filter cvar listing + "0" as off-alias

**What:** Two cvar UX fixes.
- Flag bits `CVAR_PLATFORM_MAC` / `CVAR_PLATFORM_WIN` hide wrong-platform
  cvars from `Console::EnumerateCVars` (used by `list_cvars` + autocomplete)
  while keeping them registered (set/get/archive still work, shared cfg
  round-trips). Applied to `r_software_blit` and `r_software_blit_recreate`
  (both `CVAR_PLATFORM_WIN`).
- `"0"` literal now treated as a synonym for the canonical off-token
  (`off` / `none` / `disabled`) when target cvar's `allowed_values`
  contains one of those tokens. Only `"0"` — `"1"` is NOT remapped.
  Bool cvars (`{"0","1"}`) and free-form cvars unaffected.

**Cvars added:** None — flag-bit and parser change.

**How to test:** From console:
```
> r_denoiser 0          # expect "r_denoiser off"
> list_cvars            # on Mac: should NOT show r_software_blit
```

### PR #158 — Drop overlay layered-child retry WARN noise (Win32)

**What:** `ConsoleOverlay_Win32` and `PerfOverlay_Win32` both attempt
`CreateWindowExW` with `WS_EX_LAYERED`; on a not-yet-DWM-realised parent
that call returns NULL with `GLE=0` (undocumented Win32 behaviour), and
both code paths fall back to a non-layered child that succeeds reliably.
The WARN was firing on every engine startup despite the retry being the
canonical working path; deleted with an inline comment.

**Cvars added:** None.

**How to test:** Launch demont on Windows; verify no `layered-child WARN`
line on startup.

### PR #120 — Gamepad input via GLFW (issue #83)

**What:** MVP gamepad support for first-person camera. Polls slot 0 via
`glfwGetGamepadState` every frame; GLFW's bundled SDL_GameControllerDB
maps Xbox / DualSense / Switch Pro out of the box. Left stick = translate,
right stick = look, triggers = sprint blend, smooth radial deadzone.

**Cvars added (all CVAR_ARCHIVE):**
- `cam_gamepad` (default `1`) — master toggle
- `cam_gamepad_deadzone` (default `0.15`) — stick inner-radius cutoff
- `cam_gamepad_look_sensitivity` (default `2.0`) — rad/sec at full deflection

**How to test:** Plug in an Xbox/DualSense/Switch Pro pad on Mac or Win
and verify camera moves on stick deflection. Connect / disconnect /
hot-swap edges should log once each.

**Scope / known issues:** No action-binding layer (`pt::input::ActionMap`),
no rumble (GLFW doesn't expose), no button-to-key remapping. The PR
landed **untested with hardware** — please verify with real pads.

### PR #127 — Tracy profiler instrumentation (closes #56)

**What:** Wires Tracy `ZoneScopedN` macros into ~25 hot sites across
engine, renderer, and both RHIs, plus `FrameMark` at the main loop
boundary. Routed through new shim header `src/core/Tracy.h` so macros
expand to `(void)0` in release builds — zero compile-time and run-time
cost when off. `PT_ENABLE_TRACY` already exists; `mac-debug` / `win-debug`
already set it ON.

**Cvars added:** None.

**How to test:**
```
cmake --preset mac-debug && cmake --build build/mac-debug -j 10
./build/mac-debug/src/app/demont  # connect tracy-profiler on another machine
# Expect: frames appear in the timeline with labelled zones.
```

**Scope / known issues:** No GPU zones (`TracyVkContext` / `TracyMtlContext`
need deeper backend refactor). No inner-loop markers (per #56's guidance).
Release builds have measurable-zero overhead — zones compile out entirely.

---

## Vulkan-RTX correctness fixes

> Caveat: Mac-Vulkan was historically untested before 2026-05-16.
> The fixes below are *the first* correctness floor for the Vulkan-RTX
> path. Treat fresh smoke results as load-bearing.

### PR #166 — Check VkResult on submit/wait/idle, fail-loud on device-lost

**What:** Audited every `vk*` call in `src/rhi_vulkan/VulkanDevice.cpp`
that returns a `VkResult` and was called without checking it.
Per-frame critical path (`vkQueueSubmit`, `vkWaitForFences`) and one-shot
init paths (`WriteBuffer`, `CreateTexture` layout transition,
`BuildAccelerationStructure`, etc.) now LOG_ERROR loud and propagate
failure. Sticky `device_lost_` flag on the device; `Engine::Run` loop
trips `smoke_test_failed_` so smoke runs exit code 2 instead of writing
16 all-zero PNGs and reporting success.

**Cvars added:** None.

**How to test:** Run smoke on a known-good Vulkan-RTX scene -- expect
exit code 0, non-black PNG. If the GPU device-losts (cornell_csg on
NVIDIA RTX 5090 still hangs as of this batch — separate investigation),
expect `vkQueueSubmit failed: -4 (VK_ERROR_DEVICE_LOST)` in log + exit 2.

### PR #165 — Explicit vk::image_format on RWTexture2D + WithoutFormat features

**What:** Adds `[[vk::image_format(...)]]` qualifiers to every
`RWTexture2D` / `WTexture2D` in the engine's Slang shaders, matching
each binding to its engine-side `VkImage` format. Replaces SPIR-V
`OpTypeImage <type> 2D 2 0 0 2 Unknown` + `StorageImageWriteWithoutFormat`
capability with explicit per-binding formats (rgba8 / r32f / rg16f /
rgba16f / rgba32f as appropriate). Also enables
`shaderStorageImageRead/WriteWithoutFormat` device features for
defense-in-depth.

**Cvars added:** None.

**How to test:** Render any scene on Vulkan; verify visually-correct
output. Check log for `Vulkan: shaderStorageImageRead/WriteWithoutFormat
features enabled (defense-in-depth)`.

**Important finding:** This fix **does NOT resolve** the cornell_csg
vulkan-RTX black-output bug on NVIDIA RTX 5090 / driver 596.36 /
Vulkan SDK 1.4.341.1. md5 of resulting PNG still matches the all-black
baseline. The hygiene fix is still valuable on its own (cleaner SPIR-V,
smaller capability set, spec-correct image typing) but the deeper
NVIDIA-compiler bug needs omega-1 (driver update) or deeper PathTrace
bisection.

### Commit `4b1345c` — Parser: treat `;` inside `#`/`//` comments as data

**What:** Console / cfg parser was splitting on `;` even inside `#`
and `//` line comments, so a cfg comment like `# turn on; turn off`
would silently inject a `turn off` statement. Now `;` inside the
comment span is treated as data; statement-splitting only happens
outside comments.

**Cvars added:** None.

**How to test:** Round-trip a cfg file with `;`-bearing comments; verify
no spurious statements are injected.

---

## Tests / infrastructure

### PR #125 — Golden-image matrix for sky / sun / stars / moon (closes #112)

**What:** Six new scene fixtures + 12 Darwin goldens wired into the
existing golden-image matrix (PR #103 infrastructure). Closes the
regression-coverage gap from PR #114 ("stars invisible under any
denoiser"): the original matrix only had `cornell_csg` (indoor, no sky).

**Fixtures added:**
| Fixture | Sun elev. | Exercises |
|---|---|---|
| `procedural_noon` | +60 deg | Daytime sky + sun disc, tonemap-on-bright-HDR |
| `procedural_evening` | +5 deg | Sunset orange, long-path Rayleigh+Mie |
| `procedural_dawn` | -2 deg | Civil twilight, sky-without-sun-disc branch |
| `bsc_night` | -45 deg | BSC starmap, `star_split` composite path |
| `lunar_night` | astronomical | Moon disc, NEE from moon as light source |
| `bsc_night_clouds` | astronomical | Stars + cloud transmittance occlusion |

Each scene gets `software/off` (CI-exercised) and `metal/svgf_atrous`
(user's dev-box combo) cells — 12 new cells, 24 ctest entries.

**How to test:**
```
ctest --test-dir build/mac-release -R "^golden_(procedural|bsc_night|lunar)"
# Expect: 24/24 pass.
```

### PR #121 — HdrImage CDF unit tests (closes #66)

**What:** 11 doctest cases, 685 assertions, pinning the env-map
luminance-CDF builder that feeds env-map MIS (`r_mis`) + NEE.
Tests cover monotone non-decreasing CDF, single-bright-pixel cases,
zero-luminance pathology (no NaN/Inf), uniform-white sin(theta) lat-long
Jacobian, 100k round-trip chi-square test (`chi^2 < 52.19` = chi^2(31)
p=0.01 bound), pole / zero-total guards, and determinism.

**How to test:**
```
ctest --test-dir build/mac-release -R "^hdrimage_cdf$" --output-on-failure
# Expect: 1/1 pass, ~10 ms.
```

### PR #124 — pt_math unit tests (closes #63)

**What:** Four doctest executables (vec/mat/quat/transforms), 42 cases,
1012 assertions. Tests pin the glm contract the engine relies on
(`lookAtRH`, `perspectiveRH_ZO`, column-major mat storage, RH `+X cross +Y = +Z`).
Goal: if a future glm bump or `GLM_FORCE_*` define leak flips
handedness / depth range / storage order, these tests fail BEFORE the
renderer does.

**How to test:**
```
ctest --test-dir build/mac-release -R pt_math --output-on-failure
# Expect: 4/4 pass.
```

---

## Known issues to be aware of while testing

- **Vulkan-RTX on NVIDIA driver 596.36 with SDF Phase 2 features ENABLED:**
  setting `-DPT_SDF_PROCEDURAL_OPS=ON` and/or `-DPT_SDF_AUTODIFF=ON`
  re-introduces a GPU hang (`VK_ERROR_DEVICE_LOST`) on RTX hardware.
  The 862 lines of Slang autodiff (`Dual3`) + procedural-SDF helpers
  added by PR #151 emit SPIR-V that NVIDIA's SASS JIT cannot compile
  even when the new code is unreachable at runtime. The default build
  (both gates OFF) is unaffected -- cornell_csg + 5 other test scenes
  verified rendering correctly on Win-RTX integration HEAD 842df2e.
  Mac-MoltenVK runs the same shader either way (Apple's translation
  layer routes around the offending construct). When SDF Phase 2 needs
  to be exercised, use Mac-Metal or Mac-Vulkan-software-RT for now;
  fixing the underlying SPIR-V emission requires an in-file bisect of
  `shaders/SdfPrimitives.slang` to find the offending Slang construct
  (suspected candidates: `Dual3` autodiff codegen or `sdfWorley3` 3x3x3
  nested-loop pattern). Deferred to a future wave.
- **cornell_csg on Vulkan-RTX FIXED** (was broken pre-PR-#169 with
  all-black DEVICE_LOST). PR #169's SDF gate (default OFF) excises
  the SPIR-V construct that hung NVIDIA's SASS JIT; all 6 sampled
  scenes (cornell_csg, procedural_evening, aurora_smoke, sdf_smin_row,
  light_primitives_smoke, light_tree_200_pts) now render with zero
  DEVICE_LOST on RTX 5090 + driver 596.36 + Vulkan SDK 1.4.341.1.
  Win bisect localised the cause to PR #151 SDF Phase 2 commit
  `d07a264`; PR #169's compile gate happened to wrap exactly that
  code and is the canonical fix for both the perf regression AND
  the Vulkan-RTX hang.
- **Mac-Vulkan is freshly stable** as of this batch (memory:
  `Mac Vulkan was untested before 2026-05-16`). Treat any Mac-Vulkan
  regression vs Mac-Metal as a load-bearing finding -- this is the
  first time the path has had real exercise.
- **Particles (PR #142), Aurora (PR #122), ReSTIR (PR #155),
  SigmaShadow (PR #137), Smoke (PR #156)** are all currently
  **Metal-only dispatch** -- their Vulkan pipelines are unbuilt and
  the dispatch sites collapse cleanly to no-op on Vulkan. Don't expect
  to see these effects on a Win/Linux Vulkan run.
- **NRD library (PR #123)** is **scaffolding-only** -- the per-frame
  dispatch walker is not yet implemented. `r_denoiser nrd` still
  aliases SVGF a-trous; `r_denoiser nrd_lib` proposal stays open until
  the follow-up lands.
- **Voxel destruction (PR #148)** renders voxelised meshes but does
  NOT destroy them yet — Phase 2 (fracture + voxel rigid bodies)
  depends on rigid-body Phase 2 (#138 / PR #152) which itself only
  ships Phase 2a in this batch.
- **Rigid-body box collision (PR #152)** uses the bounding sphere of
  the OBB — boxes fall and roll but never lie flat, and may clip into
  each other. Phase 2b will land box-box / box-plane SAT.

## All-in: work that landed after the initial guide commit

The first version of this guide was written at integration HEAD
`6a161b5`. The following items landed afterward and are part of the
final integration HEAD (`842df2e`) that goes into main:

- **Win's bake-race fix (PR #167, commit `d527383`)** — closes the
  race between the asynchronous CSG bake state machine and the first
  frame's PathTrace dispatch. The bake pipeline now gates the
  loading-screen exit on `bake_phase_` completion; frame 1 cannot
  dispatch the path tracer against uninitialised csg_vbuf / csg_ibuf
  / tri_bvh_nodes buffers anymore. Manifested as a Vulkan-RTX
  DEVICE_LOST on NVIDIA but is a latent race on all backends.
- **Parser fix (commit `4b1345c`) + `//` semantic alignment
  (commit `8a09752`)** — `ExecuteScript` no longer splits at `;`
  inside `#` / `//` comments or inside quoted strings. Fixes the
  smoke-runner rc=8 failure on the golden-image matrix where
  fixture comments containing semicolons (e.g.
  `# (golden-hour, not yet twilight; civil twilight is sun 0 to -6)`)
  were tokenised into a bogus `civil twilight ...` statement.
- **Cvar reset feature (commits `1e9f3ec` + `ccb08a4`)** — new
  `cvar_reset_all` console command + short `defaults` alias + new
  `--no-cfg` CLI flag (skips demont.cfg + autoexec.cfg load on
  startup). Lets testers running from an IDE wipe persisted cvar
  state before exec'ing a golden-fixture scene without manually
  moving demont.cfg out of the way. Recommended interactive
  workflow:
  ```
  > defaults
  > exec tests/goldens/scenes/cornell_csg.cfg
  ```
- **Three megakernel-gate PRs (commits `96a257c` + `a9e56ed` +
  `842df2e`):**
  - PR #168 (water gate, `96a257c`) — `PT_WATER_ENABLED` originally
    landed default OFF (recovers -2.99 ms/frame on water-less
    scenes via excising the MAT_WATER BRDF + 8-noise-tap
    wave-normal helper). Flipped to default ON in a follow-up
    fix because OFF rendered any `prim_plane ... water` material
    (e.g. `water_pool.cfg`) as opaque black — the megakernel had
    no MAT_WATER fallthrough. Power users / perf rigs flip it
    back to OFF via `-DPT_WATER_ENABLED=OFF` if they don't use
    water materials.
  - PR #169 (SDF Phase 2 gate, `a9e56ed`) — `PT_SDF_PROCEDURAL_OPS`
    + `PT_SDF_AUTODIFF` default OFF, excises the 862 LoC procedural
    + Dual3 autodiff helpers from `SdfPrimitives.slang`. Recovers
    -8.11 ms/frame AND closes the Vulkan-RTX hang (see Known issues
    above for the open SDF-Phase-2-ON-NVIDIA caveat).
  - PR #170 (light-tree gate, `842df2e`) — `PT_LIGHT_TREE` default
    ON; OFF strips ~9 KB of SPIR-V but no measurable perf delta on
    the default light_count=0 scene (Apple Metal already DCE's the
    dead branch; the +7.67 ms attribution per-commit walk gave PR
    #154 is a misattribution, the cost lives in CPU-side per-frame
    work or descriptor-binding overhead that this PR doesn't touch).
    Structural cleanup + templates the per-feature compile-gate
    pattern.

Total perf recovery (Mac-Metal default scene, 300 warmup + 3000
measurement, two-point):
- Main baseline: 8.18 ms/frame (122 fps)
- Integration HEAD pre-fix: 22.62 ms/frame (44 fps)
- Integration HEAD post-fix (`842df2e`, all gates at their original
  defaults): ~11.5 ms/frame (~86 fps)
- Remaining gap vs main: ~3 ms/frame (~36 fps). Cause unknown;
  follow-up investigation queued.

Note (water-gate flip): the original measurements above were taken
with `PT_WATER_ENABLED=OFF`, which has since been flipped to default
ON to fix the opaque-black water-surface regression. Builds taken
at the current default give back roughly the original water-gate
delta (`+2.99 ms/frame`) on the water-less default scene, putting
the Mac-Metal default-scene baseline at ~14.5 ms/frame (~69 fps).
Power users / perf bench rigs that don't use water can reclaim the
saving via `-DPT_WATER_ENABLED=OFF`.

## How this guide was generated

Original compile was at integration HEAD `6a161b5`. Updated for HEAD
`842df2e` (final-batch HEAD: includes Win's bake-race fix, parser
fix-ups, cvar-reset feature, and three megakernel-gate PRs that close
the Vulkan-RTX wedge + recover most of the perf regression).

Compiled from PR descriptions and cvar registrations on
`integration/parallel-batch-1`. To regenerate after new PRs land:

```
# 1. List merged PRs and their bodies:
gh pr list --state all --limit 60 \
   --json number,title,baseRefName,state,body \
   | jq '.[] | select(.baseRefName == "integration/parallel-batch-1" and .state == "MERGED")'

# 2. Walk merge commits since fork point:
git merge-base integration/parallel-batch-1 main  # fork
git log --merges --oneline <fork>..HEAD           # merge timeline

# 3. Cross-reference cvar registrations:
grep -E 'PT_CVAR|RegisterCommand' src/engine/Engine.cpp

# 4. Cross-reference test fixtures:
ls tests/goldens/scenes/
```

# Follow-ups & deferred work

Living doc. Long-form design sketches, speculative explorations, and
historical "this is how we shipped it" notes for the project's
larger arcs.

**As of 2026-05-16:** the well-defined actionable items previously
in this doc have been migrated to GitHub issues (see issues #44-#60).
What remains here:

1. **Long-term roadmap** (Engine vision: soft-body + voxel
   destruction) -- too big for an issue body, gets a thin tracker
   (#52) for GitHub visibility.
2. **Speculative / "if there's a real need"** items (More analytic
   primitives) -- not actionable today; the design sketch lives here
   so the next future-us picks it up with context. The Web UI editor
   and Analytical CSG sketches that used to sit in this bucket have
   since shipped and are annotated DONE below.
3. **DONE / landed entries** kept as design archeology -- how we
   built things, what the trade-offs were, what we measured.

The convention going forward: anything well-defined enough to have
acceptance criteria + a plan-when-picked-up goes straight to a
GitHub issue. This doc holds the longer-form thinking that doesn't
fit in an issue body.

---

## Engine vision: soft-body + voxel destruction (post-P11)

**Status:** A coherent identity for "non-rasterized real-time game engine"
that the path tracer is a perfect rendering substrate for. Three pillars:

1. **Path-traced beautiful soft balls.** XPBD (Extended Position-Based
   Dynamics, Müller et al. 2016) for cloth, ropes, jelly, soft-bodied
   characters. Modern Verlet's successor — same integration scheme +
   constraint projection (distance, volume, bending, shape-matching).
   Mesh BLAS rebuild/refit each frame; path tracing already handles the
   subsurface-y look of squishy materials beautifully.

2. **BeamNG-style continuous deformation.** Node-and-beam soft body for
   vehicles. Each car is a 3D mesh of mass nodes connected by stiff
   beams with breakage thresholds. When a beam breaks, the structure
   permanently deforms — *no scripted destruction, no pre-fractured
   meshes*. Path tracer just sees an updated triangle mesh per frame.

3. **Teardown-style voxel destruction.** Voxelized destructible objects
   ray-trace via DDA (a perfect axis-aligned grid is the cheapest
   acceleration structure). Destruction = `voxel[i] = 0`. Stress
   propagation across voxel chunks gives the "structural integrity
   collapses when too much support is gone" behaviour. Rigid chunks
   that detach become physics objects (Jolt later).

**Why no rasterizer is the headline:** all three of these techniques
benefit hugely from path tracing's natural handling of arbitrary
geometry, secondary rays, and HDR lighting. Voxels in particular are
*better* in a path tracer than a rasterizer because there's no
rasterization-cost penalty for millions of small primitives — just a
DDA traversal cost that scales with effective surface area.

**Progress (as of 2026-07-08):** step 1 has landed, and a physics
substrate grew underneath the later steps:

- **Voxel destruction Phase 1** (`src/destruction/` — VoxelGrid +
  Voxelizer, issue #140): voxelizes the CSG mesh and renders it as an
  analytic-box pile. Console surface: `voxelize_object` /
  `voxelize_save` / `voxelize_list` / `voxelize_clear` plus
  `r_voxel_size` / `r_voxelize_demo`. The dedicated DDA traversal
  shader + sparse brick/chunk layout sketched below is still open.
- **`src/physics/`**: Verlet particle core Phase 1 (#132), rigid-body
  Phase 2a — orientation + inertia + angular velocity (#138), driven
  by `phys_drop` / `phys_drop_sphere` / `phys_drop_box` /
  `phys_clear` / `phys_status` / `phys_ls`; SPH smoke sim (#216);
  OceanFFT water.
- **Still future:** XPBD soft-body, stress-graph fracture, node+beam
  vehicles, Jolt.

**Order of operations when picked up:**

1. **Voxel module + ray-trace path** (no physics yet). DDA traversal
   shader, sparse 3D grid representation, brick / chunk layout.
   *✓ Phase 1 landed (analytic-box pile; DDA + bricks still open).*
2. **XPBD soft-body solver** as a separate `pt::physics::Xpbd` module.
   Tested first against a hanging cloth, then a bouncing jelly cube.
   *(Not started — but the Verlet + rigid-body substrate in
   `src/physics/` (#132, #138) is the stepping stone.)*
3. **Voxel destruction**: stress graph over voxel chunks, fracture on
   threshold, freed chunks become rigid bodies.
4. **BeamNG-style node+beam solver** for vehicles, on top of XPBD.
5. **Jolt** for general rigid contact + raycasting. Bridges the gap
   between voxel chunks (rigid post-fracture) and the wider physics
   world. Last priority — Jolt's main strength is fast general rigid
   contact, which we don't need until we have a mature scene.

**Why Jolt over PhysX (when we get there):** Jolt is BSD, ~2× faster
on modern multicore CPUs (it's what Horizon Forbidden West uses).
PhysX 5 closed some of the gap with its newer solver but Jolt still
wins on raw throughput, plus it has cleaner licensing.

GitHub tracker: **#52**.

---

## Web UI scene editor ✓ DONE (superseded)

**Status:** Shipped — as a React 18 + Vite multi-panel editor under
`web/editor/`, not the vanilla-JS side panel sketched below. Seven
panels (scene-hierarchy, inspector, asset-browser, toolbar,
material-editor, lights, render-settings) are registered in
`src/editor/EditorRoutes.cpp` and served from the engine's HTTP
server; each opens in its own Chrome `--app` window via `panel_open`
/ `panel_close` / `panel_open_all` / `panels` (F2–F8 hotkeys
in-engine). Panel state lives in a zustand store fed by the console
WebSocket's scene protocol (`list_scene`, selection, property edits
via `src/editor/SceneGraph.cpp`), with a viewport gizmo overlay for
transforms and undo/redo. The npm build is embedded into the binary
by `cmake/Editor.cmake`. See `web/editor/CONTRIBUTING.md` for the
workspace layout and panel-authoring guide.

Live re-bake through the engine's async bake job — the one hard
requirement in the original sketch — is exactly how the shipped
panels drive edits, so nothing from the sketch remains open.

**Original sketch (kept as design archeology):**
- Add a side panel showing the current `prim_list` + CSG tree.
- Sliders for primitive transforms; each input change → throttled
  `prim_sphere <id> ...` or `csg_set_transform ...` command.
- Live re-bake at >10 Hz (P9 acceptance bar) — already supported by
  the engine's async bake job.

---

## More analytic primitives

**Status (speculative):** Sphere + plane only in `prim_*` commands.
Box, cylinder, disk are CSG-side.

**Why not filed as an issue:** CSG covers these use cases. Adding
analytic versions is mostly redundant unless someone hits a real
need.

**Plan if picked up (only if there's a use case the CSG path can't
cover):**
- AABB intersection (slab method) → `prim_box`.
- Disk = plane + radius check → `prim_disk`.
- Cylinder (analytic) = quadratic + caps → `prim_cylinder_an`
  (rename to disambiguate from the CSG one).
- Add new `PRIM_*` enum values to `PathTrace.slang` and a new branch
  in `traceScene`.

---

## Analytical CSG (SDFs / boolean of analytic primitives) ✓ DONE (SDF path)

**Status:** The **SDF path** below shipped as the SDF cluster system
(Phases 1–3, issues #97 / #98 / #99): analytic SDF leaves
(`sdf_sphere` / `sdf_box` / `sdf_round_box` / `sdf_torus` /
`sdf_capsule`) combined with smooth booleans (`sdf_smin` / `sdf_smax`
/ `sdf_sdiff`), procedural ops (`sdf_displace_noise` / `sdf_twist` /
`sdf_bend` / `sdf_repeat` / `sdf_repeat_limited`), and fractal DEs
(`sdf_mandelbulb` / `sdf_mandelbox` / `sdf_apollonian`) —
sphere-traced inside the path-trace kernel via
`shaders/SdfPrimitives.slang` and `shaders/SdfFractals.slang`, folded
into `AnalyticBvh` as the third hit type the sketch predicted. See
the SDF Phase 3 DONE entry below for the fractal-DE trade-offs.
Mesh CSG via Manifold remains the headline for exact booleans.

**Still unpicked:** the **hit-walk path** (exact, non-marched boolean
of analytic primitives by sorting leaf hits and walking the tree) and
a `pt::csg::AnalyticCsgScene` mirroring `CsgScene`'s API — worth
doing only if a use case needs exact analytic booleans that neither
the raymarched SDF clusters nor Manifold covers.

**Original sketch (kept as design archeology):**
- Maintain a parallel `pt::csg::AnalyticCsgScene` with leaf
  primitives + boolean nodes, mirroring `CsgScene`'s API.
- Two implementation paths:
  - **SDF path:** evaluate `min/max/-` of leaf SDFs in the shader,
    ray-march. *(← this is what shipped)*
  - **Hit-walk path:** intersect ray with all leaves, sort hits,
    walk the tree to determine which segment is inside the result.
- The unified renderer already supports both an analytic primitive
  list and a mesh TLAS; analytic CSG is a third hit type to fold in.

---

## P11: HDRI env maps + MIS ✓ DONE

**Status:** Both shipped.

- **HDRI environment map:** `r_env_map <path>` (CVAR_ARCHIVE) +
  `r_env_intensity`. Loads Radiance `.hdr` (RGBE), uploads as
  RGBA32F, samples by direction → spherical UV. Replaces the fixed
  `skyColor()` lerp in `PathTrace.slang` when `r_sky_mode hdri` is
  selected.

- **Importance sampling:** luminance CDF (1D rows + 2D total) built
  on CPU at HDRI load, uploaded as `marginal_cdf` / `conditional_cdf`
  storage buffers. Direct-light samples bias toward bright sky
  regions.

- **MIS (multiple importance sampling):** balance-heuristic
  combining of env-map NEE and BRDF-bounce miss contributions on
  Lambert hits. `r_mis 1` enables it (default on). Metal (incl.
  rough metal via `h.roughness`) and dielectric paths stay on their
  existing code paths -- MIS is currently a Lambert-only scope.

Both visible in the engine today via `r_env_map`, `r_mis`,
`r_env_intensity` cvars + the HDRI shading path in `PathTrace.slang`.

---

## P10: MetalFX TemporalDenoisedScaler — Mac path  ✓ DONE

**Status:** Fully wired and verified. `r_denoiser metalfx` produces
cleanly denoised 1-spp output at >30 FPS at 1080p on M4 Max, with
per-frame Halton sub-pixel jitter for proper TAA-style reconstruction.
Three real bugs uncovered + fixed along the way (Metal TLAS row→col
transpose, Slang→MSL `float3` stride mismatch, MetalCommandBuffer push
buffer overflow). Original implementation checklist preserved below
for reference (and as a template for the Vulkan denoiser).

---

### Original P10 implementation checklist (kept for reference):

1. **Texture formats** — add to `src/rhi/Types.h`:
   - `R32F` (single-channel float — view-space linear depth).
   - `RG16F` (two-channel half-float — pixel-space motion vectors).
   - Map them in `MetalDevice::CreateTexture` to `MTL::PixelFormatR32Float`
     and `MTL::PixelFormatRG16Float`.

2. **Shader G-buffer outputs** — `shaders/PathTrace.slang`:
   - Add bindings:
     - `[[vk::binding(6, 0)]] RWTexture2D<float4> denoise_color;` (RGBA16F linear, no tonemap)
     - `[[vk::binding(7, 0)]] RWTexture2D<float>  depth_tex;` (linear depth from camera)
     - `[[vk::binding(8, 0)]] RWTexture2D<float2> motion_tex;` (pixel-space motion)
   - Push: add `denoiser_enabled : uint`, plus two `float4x4` matrices
     `curr_view_proj` and `prev_view_proj` (16 floats each).
   - Inside the dispatch, after primary-ray hit:
     - `denoise_color[tid] = float4(frame_radiance, 1.0);` (per-frame, NOT
       accumulated -- denoiser does its own temporal reuse)
     - `depth_tex[tid] = h.t;` (or compute camera-space depth)
     - For motion: project the world hit point with `curr_view_proj` →
       `curr_screen`, with `prev_view_proj` → `prev_screen`. Motion =
       `(prev_screen - curr_screen) * dim * 0.5` (pixels).
   - When `denoiser_enabled == 0`, branch to skip the G-buffer writes
     and keep the existing accumulation-tonemap-to-swapchain path.

3. **Engine: previous-frame camera matrices** (`src/engine/Engine.{h,cpp}`):
   - Add `glm::mat4 prev_view_proj_` member, init to identity.
   - At top of `RenderFrame`, compute current view*proj from the camera
     basis. Push both matrices to the shader. After dispatch:
     `prev_view_proj_ = curr_view_proj`.
   - The camera's existing `Forward/Right/Up/FovYTan` is enough --
     build view from those, projection from FOV + aspect + near/far.

4. **Engine: G-buffer texture allocation** (`src/engine/Engine.{h,cpp}`):
   - Allocate `denoise_color_tex` (RGBA16F), `depth_tex` (R32F),
     `motion_tex` (RG16F) when `r_denoiser != off` and free when off.
   - Realloc on swapchain resize (mirror existing `accum_texture_` logic).
   - Bind them at slots 6/7/8 in `RenderFrame` when denoising.

5. **MetalFX shim** — new file `src/rhi_metal/MetalFXDenoiser.mm`:
   - `metal-cpp` doesn't bridge MetalFX, so this is one small ObjC++
     file. No bridging headers -- include `<MetalFX/MetalFX.h>`,
     `<Metal/Metal.h>` from inside the .mm.
   - C interface (mirroring `MetalAttach.mm` style):
     ```c
     extern "C" void* pt_metalfx_create(void* mtl_device, int w, int h);
     extern "C" void  pt_metalfx_destroy(void* scaler);
     extern "C" void  pt_metalfx_encode(void* scaler, void* mtl_cb,
                                         void* color_in, void* depth_in,
                                         void* motion_in, void* color_out,
                                         float jitter_x, float jitter_y,
                                         int reset);
     ```
   - Encode side wraps `MTLFXTemporalDenoisedScaler` (the post-WWDC
     2024 API): set inputColor/inputDepth/inputMotionVectors/outputColor
     properties, set jitter offsets, call `encode(commandBuffer:)`.

6. **Engine: Denoise call site** (`src/engine/Engine.cpp`):
   - After path-trace dispatch and before EndFrame, if `r_denoiser !=
     off`, call `device_->Denoise(...)` with the three G-buffers,
     current jitter offsets, and a reset flag (set on swapchain resize
     or major scene change).

7. **RHI Device interface** (`src/rhi/Device.h`):
   - Add `virtual void Denoise(...)` with a `DenoiseDesc` struct
     (input/output texture handles, jitter, reset, output dims).
   - Default: no-op on Software backend (logs once that it's unsupported).
   - Metal: routes to the metal-cpp shim above.
   - Vulkan: also no-op for P10 (will pick up the in-house SVGF in
     a later phase).

8. **CVar** (`src/console/Console.cpp` or wherever `RegisterCvars` lives):
   - `r_denoiser off | metalfx` (default `off` until the path looks
     right; flip default to `metalfx` once verified).

9. **Acceptance verification:**
   - Default scene at 1 spp with `r_denoiser metalfx` matches accumulated
     reference at 64 spp `r_denoiser off` (visual diff).
   - 30+ FPS at 1080p sustained while denoised; perfoverlay shows
     comparable cost to the accumulation path.
   - No corrupted halos on camera spin (motion vectors correct).
   - Resize works (G-buffers realloc + denoiser reset).

10. **Compatibility notes:**
    - Halton(2,3) is the typical low-discrepancy sub-pixel jitter
      sequence — produces sharper TAA than purely random; should
      cycle every 16 frames with offsets in `[-0.5, 0.5)`. Pass to
      the shader so the primary ray is jittered by `(jx, jy) /
      dim * 0.5`.
    - `MetalFX.framework` is macOS 13+. Guard at runtime if you ever
      ship to older OSes.

---

## Async pipeline build + loading screen ✓ DONE

**Status:** Shipped end-to-end on Vulkan.

What landed:
- **Persistent `VkPipelineCache`** at `%LOCALAPPDATA%/demont/pipeline.cache`
  (Windows) or `$XDG_CACHE_HOME/demont/pipeline.cache` (POSIX). Loaded
  at device init, written on shutdown, stale-blob rejection silent.
- **Async pipeline build** on a worker thread spawned from
  `VulkanDevice::Init`. Vulkan permits `vkCreateComputePipelines` in
  parallel with the main thread's queue submit, so the engine's main
  loop is never blocked by the 1-3s cold-cache pipeline compile.
- **`Engine::EnsurePipelineHandles`** lazily resolves cached pipeline
  ids each frame until the worker flips them non-zero. Once resolved,
  the check is a single uint compare (no mutex / map lookup overhead).
- **`RHI::CommandBuffer::ClearStorageTexture`** — minimal swapchain
  clear-to-colour primitive, used by the loading-frame path so the
  user sees a defined dark frame instead of undefined post-acquire
  pixels while the pipelines compile.
- **Per-pipeline timing log** (`Vulkan: async pipeline build done in
  Nms (pathtrace ..., autoexpose ..., perfoverlay ...)`) plus
  `engine: loading screen active` / `pipelines ready` pair for
  observability of cold-vs-warm cache behaviour.

**Mac/Metal:** Metal does the SPIR-V→native compile lazily on first
draw and is much faster than Vulkan's eager `vkCreateComputePipelines`.
The macOS port doesn't have the same freeze and isn't on this critical
path. The same async-build loading-screen scaffold could host a Metal
`MTLLibrary.makeComputePipelineState(completionHandler:)` worker if
we ever see a similar stall there, but no current evidence we need it.

**Future polish (not blocking):** the loading frame is a flat dark
colour. A subtle pulse / progress text would be nicer UX but needs a
text-rendering primitive that doesn't depend on the path-trace
pipeline (perfoverlay would work but it's also under construction
during the loading window).

---

## Slang IR module precompilation — landed (small extraction)

**Status:** Reinstated.  The two small helper modules
(`PathTraceMath`, `PathTraceCloud`) are compiled to `.slang-module`
IR by `pt_compile_slang_module` and imported into PathTrace.slang.
SPIR-V output is bit-identical to the monolithic build (490008 B
both ways), so zero runtime impact -- this is a pure compile-time
refactor.

**The numbers** (ninja `.ninja_log`, win-debug, NVIDIA RTX 5090):

| Phase | Monolithic | With 2 modules (~150 lines each) |
|---|---|---|
| `PathTrace.spv` alone | 855 ms | 710 ms (−145 ms saved on parse) |
| Module compiles (ninja-parallel) | — | 360 ms wall-clock |
| Clean rebuild critical path | **855 ms** | **1106 ms (+251 ms)** |
| Body-only incremental | **855 ms** | **710 ms (−145 ms)** |

We initially reverted on the basis that clean rebuilds got 250 ms
slower; reinstated on the realisation that the body-only iteration
case (the one that fires on every shader edit during development)
is what dominates a working day.  Clean rebuilds are rare (branch
switches / merges / fresh checkouts).

**Why the link cost is high:** Slang's IR-link cost is roughly
fixed per imported module regardless of size, while the saved
front-end work scales with module content.  Below ~500 LOC of
saved content per module, link overhead dominates the saved parse.
The two helpers we extracted are ~150 LOC each -- well below that
threshold individually -- but the body-only iteration win still
makes the trade worth it because the link runs ONCE per body
rebuild whereas the saved parse runs N times across an editing
session.

**When to expand the extraction:**
- **If we go to option 3 (wavefront path tracing).** Each pass
  becomes a separate entry point. All entry points share the same
  helper modules, so the per-module link cost amortises across N
  kernels instead of 1.  Both modules and many more chunks become
  clear wins.
- **If we extract a single large logical chunk** (e.g. the entire
  atmospheric march + cloud march block into one module, ~600 LOC).
  Doable but those blocks reference Push fields, so the module
  would need `extern` declarations or a parameter-passing refactor.
- **If a future Slang version reduces IR-link overhead.** Worth
  re-measuring on every Slang upgrade -- the link cost is the
  bottleneck and any improvement there directly raises the
  break-even point for further extraction.

---

## DONE: SDF Phase 3 fractal showcase (#99, landed 2026-05-18)

Added Mandelbulb / Mandelbox / Apollonian-gasket distance estimators
as a new fractal-leaf shape variant in the existing SDF cluster
infrastructure. Lives in `shaders/SdfFractals.slang` as a separate
Slang IR module so it links into PathTrace.spv (Vulkan) and
PathTrace.metallib (Metal) the same way `SdfPrimitives.slang` does,
but stays independent of the Phase 1 analytic SDF library.

### Design tradeoffs

**DE approximation vs exact analytic SDF.** Fractals don't have an
exact closed-form distance. The DE we ship is a *lower bound* on the
true distance, recovered by bounded iteration of the per-shape
escape / fold formula plus the standard `escape-radius / derivative`
estimator (Hart 1996, Iñigo Quilez "Distance to fractals"). Concrete
consequences:

1. **The sphere-trace `t += d` advance is still safe.** A lower
   bound never overshoots the surface, so we don't tunnel through.
2. **More iterations than analytic SDFs in deep-recursion regions.**
   Each step covers less ground than an exact-SDF step of the same
   nominal magnitude. The Phase 3 fixture bumps `r_sdf_max_iters`
   from the default 128 to 192 for the showcase scene; the hard
   shader-side ceiling is 512 in `traceSdfFractal` (vs 256 in
   `sdfClusterTrace` for the analytic library).
3. **Iteration-relaxed termination.** The issue spec calls for
   `dist < eps * scale * iter_count` -- we accept a looser hit late
   in the trace where the DE returns a vanishingly small (but
   non-zero) distance step after step ("no progress"). The
   `r_sdf_de_eps_scale` cvar controls the relaxation rate; default
   1.0 reproduces the canonical formula and 0 falls back to the
   constant-epsilon trace that matches analytic-SDF behaviour
   exactly.
4. **Central-difference normals.** Closed-form gradients don't
   exist for these iterators. The fractal trace samples the DE four
   times on a small tetrahedron around the hit point and recovers
   the normal from the gradient. ~4 extra DE evals per hit; the
   sampling step is `8 * iter_scaled_eps`, big enough to escape the
   iteration-relaxed termination band, small enough that the
   finite-difference error stays below the silhouette noise floor.

**Per-shader noise expectations.** The issue explicitly accepts a
silhouette-noise floor on fractal pixels -- it's the cost of having
the visual capability at all. The showcase fixture
(`tests/goldens/scenes/sdf_fractals.cfg`) renders at 1 spp and is
expected to show visible per-pixel noise on fractal silhouettes,
particularly the Mandelbulb's high-curvature thin regions and the
Apollonian gasket's nested-sphere boundaries. Higher spp converges
the noise per usual MC; the issue's acceptance criterion is
"converges with spp" rather than "noise-free at 1 spp."

### Architecture notes (for the next phase)

- **Single-leaf fractal clusters only.** The Phase 3 GPU dispatcher
  in `PathTrace.slang::traceSdfClusters` only handles single-leaf
  fractal clusters -- it does NOT route fractals through the
  smooth-CSG ops in `SdfPrimitives.slang`. Reason: those ops assume
  exact-distance children, and the DE's approximate distance would
  produce a wildly-overshooting blend. The host's `combine` helper
  (in `RegisterSdfCommands`) rejects fractal-as-CSG-child at parse
  time so the failure is loud.
- **Wire-format extension.** The host `SdfShape` enum gained ids
  6/7/8 (`SDF_SHAPE_MANDELBULB/MANDELBOX/APOLLONIAN`); see
  `src/renderer/AnalyticBvh.h`. The shader-side `evalLeafDist` in
  `SdfPrimitives.slang` is unmodified -- those shape ids fall
  through its `if` chain and return `1e30`, but the new dispatcher
  in `PathTrace.slang` intercepts them before that ever happens so
  the unreachable-path is harmless.
- **Push-constant lanes.** The fractal cvars consumed the reserved
  lanes documented on the Phase 1 push block:
  `sdf_params.w = r_sdf_fractal_iters`, `sdf_params_f.y =
  r_sdf_fractal_power`, `sdf_params_f.z = r_sdf_de_eps_scale`. No
  push-constant reshuffle required.
- **Denoiser hook (#50).** The issue specifies "per-material
  variance hint (REBLUR per-pixel hint)" and a motion-vector
  approximation for fractal pixels. Both of those land WITH the NRD
  integration in #50 -- the Phase 3 PR plumbs the DE-noise floor
  but doesn't yet feed a per-material hint to NRD because NRD
  itself isn't wired yet. When #50 lands, the cluster's material
  field can carry a "high-variance" flag that REBLUR consumes.
- **Wavefront PT (#76) payoff.** Fractal pixels are the worst case
  for warp divergence in the path tracer: adjacent pixels can take
  wildly different sphere-trace step counts. Wavefront PT recovers
  most of the lost throughput because the divergent step-count
  loop becomes one queue-and-drain per phase. The Phase 3 fixture
  is a good A/B benchmark when #76 lands.

### Out-of-scope for Phase 3 (queued)

- **Mixed fractal + analytic CSG.** Smooth-union of a Mandelbulb
  with a Phase 1 sphere would need a unified distance evaluator
  that knew which child's distance was a lower bound vs exact.
  Doable but not on the Phase 3 critical path.
- **Animated fractal parameters.** `r_sdf_fractal_power` and per-
  leaf `params[0]` are static today. Animating them per-frame is
  free in principle (just a cvar tween or a host-side animation
  hook) but the denoiser motion-vector path doesn't yet handle
  topology-changing geometry.
- **Menger sponge.** Listed as "possibly" in the issue scope; not
  shipped in the Phase 3 MVP. Add a SHAPE_MENGER variant when
  there's a real fixture asking for it -- the DE is a simple
  scale-and-cross-fold and would slot in alongside Apollonian.

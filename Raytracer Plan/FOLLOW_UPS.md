# Follow-ups & deferred work

Living doc. Things we knowingly skipped or postponed in earlier phases,
with enough context to pick them up later. Each item has a one-line
hook, a "why deferred", and a sketch of the work needed. Roughly ordered
by priority.

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

**Order of operations when picked up:**

1. **Voxel module + ray-trace path** (no physics yet). DDA traversal
   shader, sparse 3D grid representation, brick / chunk layout.
2. **XPBD soft-body solver** as a separate `pt::physics::Xpbd` module.
   Tested first against a hanging cloth, then a bouncing jelly cube.
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

---

## P11 remainder: HDRI env maps + MIS

**Status:** P11 partially landed (cvar persistence + scene_save/load).
The remaining items are bigger lifts that need shader work and a tiny
bit of CPU-side math.

**HDRI environment map (image-based lighting):**
- Load a `.hdr` lat-long image (Radiance RGBE format, well-documented).
- Upload as RGBA32F texture, sample by direction → spherical UV.
- Replace the fixed `skyColor()` lerp in `PathTrace.slang` with a sample
  from the env map when one is bound.
- Importance sampling: build a luminance CDF (1D rows + 2D total) on
  CPU at load time so direct-light samples bias toward bright sky regions.
- New cvar `r_env_map <path>` (CVAR_ARCHIVE) + `r_env_intensity`.

**MIS (multiple importance sampling) for direct lighting:**
- Sample two distributions per shading point: BRDF and lights. Weight
  via the balance heuristic to combine without bias.
- Lights are the env map (sampled via the CDF above) and any explicit
  area lights. Right now we have no lights — env map is the whole
  source.
- Visibility ray to the sampled direction; add `throughput * Le * BRDF *
  cos / (pdf_brdf + pdf_light)` per the heuristic.
- This noticeably reduces variance on Lambert + rough metal surfaces.

**Acceptance:** loading a Sunny Day HDRI and rendering the default
scene at 1 spp with `r_denoiser metalfx` should look photographically
clean within seconds. Try `r_env_map assets/sunny.hdr` — the gold
sphere should pick up sky reflection convincingly.

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
   - Implement using `MTLFXTemporalDenoisedScalerDescriptor` →
     `newTemporalDenoisedScalerWithDevice:`; per frame set the
     scaler's color/depth/motion/output textures and call
     `encodeToCommandBuffer:`.

6. **CMake** — `src/rhi_metal/CMakeLists.txt`:
   - Add `MetalFXDenoiser.mm` to sources.
   - Add `"-framework MetalFX"` to the Apple frameworks block.

7. **MetalDevice integration** — `src/rhi_metal/MetalDevice.{h,cpp}`:
   - Add `Denoise(...)` virtual on `pt::rhi::Device` (no-op default,
     Metal override calls `pt_metalfx_encode`).
   - Hold the scaler instance in `MetalDevice` (alloc lazily on first
     denoise call; recreate on swapchain resize).

8. **Engine flow** (`src/engine/Engine::RenderFrame`):
   - Read `r_denoiser`. If `metalfx`:
     - Force `reset_accum=1` every frame in the push (denoiser doesn't
       want pre-accumulated input).
     - Pass `denoiser_enabled=1` to shader.
     - After `device_->Submit(cb)` but before `EndFrame`, call
       `device_->Denoise(denoise_color_tex, depth_tex, motion_tex,
                          fc.swapchain_image, jitter_x, jitter_y, reset)`.
     - `reset` flag = 1 on the first frame after the camera or scene
       changes drastically (e.g. backend switch, CSG rebuild).
   - If `off`: leave existing accumulation flow untouched.

9. **Acceptance test:**
   - `r_denoiser metalfx; r_spp 1; r_max_bounces 4` should give a
     visibly clean image while moving the camera.
   - Toggle `r_denoiser off` -- should fall back to the accumulating
     1-spp pathtracer (noisy while moving, converges over a couple
     seconds when stationary).
   - No validation errors in Xcode Metal Frame Capture.
   - &gt;30 FPS at 1080p on M4 Max.

10. **Known gotchas:**
    - MetalFX wants depth in linear camera-space units (NOT clip-space
      `z/w`). Use `h.t` (the ray length) for primary hits.
    - Motion vectors are in PIXELS, not NDC. Multiply NDC delta by
      `dim * 0.5`.
    - `MetalFX.framework` is macOS 13+. Guard at runtime if you ever
      ship to older OSes.

---

## Vulkan denoiser (5090 path) — `r_denoiser svgf_basic | svgf_atrous | nrd`

**Status (in-house SVGF):** Landed with two quality tiers plus the
forward-compat `nrd` alias.

- `svgf_basic` — temporal accumulation only, no spatial filter. The
  temporal pass writes the next frame's history texture; the result
  is `vkCmdCopyImage`'d into `post_denoise_hdr` for the bloom +
  tonemap chain. ~1.5 ms at 1080p on a 5090. Best for fast camera
  motion (no a-trous lag), slightly noisier on disocclusions.
- `svgf_atrous` — `svgf_basic` plus 3 a-trous wavelet passes at step
  sizes 1/2/4 with depth + normal + luminance edge stops. The last
  pass writes directly into `post_denoise_hdr`. ~5 ms; cleaner on
  disocclusions, mild softening of micro-detail.
- `nrd` — currently aliases `svgf_atrous`; reserved for the proper
  NVIDIA library swap (see below). One-time log on transition so the
  user knows they're on the SVGF placeholder.

The path tracer writes a fourth G-buffer (world-space normal at
primary hit, RGBA16F, `vk::binding(16)`, gated by `PT_TARGET_SPIRV`
so the Metal/MetalFX path is unaffected) when any svgf*/nrd value
is selected.

**Open work — proper NRD library integration:**

NVIDIA's
[RayTracingDenoiser](https://github.com/NVIDIAGameWorks/RayTracingDenoiser)
(BSD since 2024) gives substantially better quality than our SVGF-lite:
REBLUR (recurrent blur with variance-aware step size selection),
RELAX (specifically tuned for ReSTIR), SIGMA (shadow-only).
Integration is multi-day work because:

- NRD's API hands you back ~60 internal `nrd::TextureDesc` slots and
  ~40 `nrd::PipelineDesc` shader stages per frame. The user is
  responsible for translating those to `VkImage` / `VkPipeline` /
  `VkDescriptorSet`, recording the dispatches in a user-supplied
  command buffer, and tracking pipeline-layout-vs-shader-resource
  identity.
- NRD wants normal+roughness packed together in NRD's own format
  (`NRD_FrontEnd_PackNormalAndRoughness`). The path tracer would need
  to emit one extra texture in NRD's encoded layout.
- viewZ format needs to be NRD's specific encoding (linear in clip,
  not in world units like our current depth_tex).
- Hit-distance buffer is a separate input -- requires the path tracer
  to write `path.t * lin_brdf_pdf_weight` into a new G-buffer.

When picked up, the work splits cleanly into:
1. **Vendor NRD via FetchContent** with `PT_ENABLE_NRD=OFF` default
   so the dependency is opt-in. NRD ships its CMake with embedded
   shader resources.
2. **G-buffer extension**: pack normal+roughness, switch depth to
   NRD's viewZ, add hit-distance.
3. **VulkanNrdDenoiser swap**: replace the SVGF dispatch chain with
   a real NRD `Denoise()` call inside the same class. The engine
   side stays unchanged because the cvar value is already `nrd`.
4. **Acceptance**: visibly better quality vs SVGF-lite on disocclusions
   and complex BRDFs (gold metal, dielectrics with caustics).

**Future option — DLSS Ray Reconstruction:** highest quality,
NVIDIA-only, needs a 5090-specific shim. Defer further; SVGF + the
eventual NRD path cover the bulk of cross-vendor users.

**Acceptance for the in-house path (already met):** clean 1-spp at
&gt;30 FPS at 1080p, no validation errors, edges preserved on the
default scene with both procedural and HDRI sky. Both `svgf_basic`
and `svgf_atrous` boot cleanly on RTX 5090 with no VK validation
messages.

**Quality follow-ups (all small):**

- *Eager denoiser pipeline build in the worker thread.* Today
  `VulkanNrdDenoiser::Init()` runs synchronously on the first
  `Denoise()` call -- a few-ms hitch on the first frame after
  toggling `r_denoiser` from `off` to a Vulkan kind. Moving the
  build into the existing async pipeline-build worker (alongside
  `pathtrace`/`autoexpose`/`perfoverlay`) eliminates the hitch
  cleanly; the denoiser's resource_mutex usage is already worker-
  safe.

**Future cross-platform option — SVGF on Metal:** the same two
shaders (`DenoiseTemporal.slang` / `DenoiseAtrous.slang`) compile to
MSL with no source changes (no SPIR-V-only intrinsics in the bodies).
Adding Metal support would mean: emit them as MSL targets in the
`pt_rhi_metal` CMakeLists, drop the `PT_TARGET_SPIRV` gate around
`PathTrace.slang`'s `normal_tex` declaration, allocate the normal
G-buffer for the Metal path too, and wire `MetalDevice::Denoise` to
dispatch the SVGF kernels as an alternative to MetalFX (perhaps
exposed as `r_denoiser svgf_basic` / `svgf_atrous` on Mac instead
of routing those values to off). That'd give Mac users a
G-buffer-aware option alongside MetalFX's black-box approach.

---

## Spherical-Earth atmospheric model (post-NRD)

**Status:** Deferred until after the Vulkan NRD denoiser lands. The
current rendering uses planar earth + a flat cloud-layer slab, which
leaves a visible gap between the cloud bottom and the geometric
horizon when the camera is near the ground -- "clouds never touch
the horizon" because a perfectly horizontal ray would need infinite
horizontal distance to enter a cloud slab above the camera. In real
life Earth curvature drops distant clouds below horizontal sightline
so they visually merge with the horizon.

**Why deferred:** the current planar approximation is good enough
for most paths (HDRI mode + scenes within a few km), and NRD is on
the critical path for Vulkan visual quality.

**Use case driving this:** procedural sky for **flight rendering** —
sunset / sunrise / altitude views where the horizon is the
focal area, sun aureole + multi-scatter need to look right, and
distant terrain should fade through curved-path aerial perspective.
HDRI mode benefits less because the env map provides the sky, but
still wins on the spherical cloud shell + curved aerial perspective
on far surfaces.

**The four phases (rough Hillaire 2020 implementation):**

1. **Spherical cloud shell** (~150 LOC, ~1 hour). Add
   `r_planet_radius` cvar (default 6371 km Earth). Replace cloud
   layer's flat slab intersection with ray-vs-spherical-shell. Cloud
   march cost rises by a couple of `sqrt`s per entry, runtime impact
   negligible. Self-contained -- can ship without phases 2-4 and
   still get the horizon-gap fix.

2. **Transmittance LUT + spherical aerial perspective** (~250 LOC).
   Replace the analytical `atmosphericTransmittance` (planar
   exponential integral) with a 256x64 LUT computed once at startup
   via a compute pass. Curved-path attenuation along the actual
   spherical air mass. Sample per cloud-march step + per surface
   hit; ~5 cycles + memory latency per sample (faster than the
   current `exp` calls in many cases).

3. **Multi-scatter + sky-view LUTs for procedural mode** (~350 LOC).
   This is the flight-sim sunset visual win:
   - **Multi-scatter LUT** 32x32, computed from the transmittance
     LUT. Adds the secondary-bounce contribution that makes
     overcast and sunset look saturated rather than washed out.
   - **Sky-view LUT** 192x108 RGBA16F, recomputed each frame
     because it depends on sun direction. Replaces `procSky`'s
     analytical math with a proper sampled sky.
   - Procedural sky's sun aureole + horizon gradient become correct.
   - HDRI mode still bypasses this entirely (it samples the env
     map for the sky color regardless).

4. **Aerial perspective volumetric LUT** (~150 LOC, only if needed).
   32x32x32 3D texture mapping `(screen pos, depth) ->
   {transmittance, in-scatter}` for distant geometry. Skip if the
   per-sample LUT from phase 2 already looks right -- this is just
   a perf optimisation that batches the curved-path math into a
   precompute. May not be needed at our typical scene scales
   (1-30 km).

**Total cost:** ~900 LOC, 1-2 days focused work, ~150 µs/frame for
the per-frame LUT updates (sky-view + AP), trivially small relative
to ray-tracing cost.

**No assets to ship** -- all LUTs are procedural compute, generated
from physical parameters (Mie/Rayleigh sigmas, sun direction,
planet radius). Storage cost ~5 MB GPU memory total.

---

## Per-face materials on CSG meshes

**Status:** All triangles emitted by `pt::csg::CsgScene::Bake()` get the
hardcoded gold-metal material in the unified shader. Manifold preserves
`originalID` per triangle (= which leaf the triangle came from), but we
discard that information.

**Why deferred:** Headline P9 acceptance was the geometry pipeline
working end-to-end. Materials per face are P9 polish; without them the
drilled cube still looks great as a demo.

**Plan when picked up:**
- In `CsgScene::AddBox/Sphere/Cylinder`, accept a `material_id`
  parameter and call `Manifold::SetProperties(...)` so the leaf manifold
  carries an originalID → material_id map.
- In `Bake()`, after `GetMeshGL()`, walk `runOriginalID[]` to build a
  per-triangle material index buffer alongside positions/indices.
- Upload that buffer at slot 6 (or similar) in `Engine::RebuildMeshResources`.
- Shader: read the material id via `q.CommittedPrimitiveIndex()` and
  dispatch on it in `traceScene`'s mesh branch.
- Console: extend `csg_box`/`csg_sphere`/`csg_cylinder` with a trailing
  material arg (default keeps existing gold).

---

## Smooth shading on CSG curved surfaces

**Status:** `EmitFlatTriangleSoup()` unwelds vertices and the shader
recomputes a flat per-triangle normal via cross product. Cubes look
clean; spheres / cylinders / drilled-cavity interiors show the
underlying triangulation as gem-like facets.

**Why deferred:** Flat shading is correct for verifying the CSG
pipeline. Smooth shading is a visual fidelity improvement, not a
correctness one.

**Plan when picked up:**
- After `Manifold::CalculateNormals(propIdx, ...)`, call
  `GetMeshGL(normalIdx = propIdx)` to get smoothed vertex normals as
  per-vertex properties.
- Stop unwelding — emit the welded mesh and a parallel normals buffer.
- Shader: read three vertex normals per triangle (via the same indices
  used for positions) and barycentric-interpolate at the hit point.
  `q.CommittedTriangleBarycentrics2()` gives `(u, v)`; third coord is
  `1 - u - v`.

---

## Web UI scene editor

**Status:** All scene editing is via console commands today (typed or
pasted into the web/in-game console).

**Why deferred:** The console is enough to drive the demo. A real
editor is a meaningful chunk of vanilla-JS work and not on the critical
path for the headline phases.

**Plan when picked up:**
- Add a side panel showing the current `prim_list` + CSG tree.
- Sliders for primitive transforms; each input change → throttled
  `prim_sphere <id> ...` or `csg_set_transform ...` command.
- Live re-bake at &gt;10 Hz (P9 acceptance bar) — already supported by
  the engine's async bake job.

---

## Software backend mesh path (Embree)

**Status:** Software backend has no triangle-mesh intersection. CSG
geometry is invisible there; analytic primitives still render.

**Why deferred:** Software backend is reference / portability /
correctness. Embree is the right way to give it real mesh perf, but
that's a P11+ scope.

**Plan when picked up:**
- Vendor Embree via FetchContent (CMake-friendly, BSD licensed).
- Software RHI's `CreateBLAS`/`CreateTLAS` build `RTCScene` instead of
  no-ops.
- Software path tracer dispatches `rtcIntersect1` per pixel.
- Aim for parity with Metal RayQuery output so the 3-backends-render-identically
  acceptance still holds.

---

## More analytic primitives

**Status:** Sphere + plane only in `prim_*` commands. Box, cylinder,
disk are CSG-side.

**Why deferred:** CSG covers these use cases. Adding analytic versions
is mostly redundant.

**Plan when picked up (if there's a real need):**
- AABB intersection (slab method) → `prim_box`.
- Disk = plane + radius check → `prim_disk`.
- Cylinder (analytic) = quadratic + caps → `prim_cylinder_an` (rename
  to disambiguate from the CSG one).
- Add new `PRIM_*` enum values to `PathTrace.slang` and a new branch in
  `traceScene`.

---

## Analytical CSG (SDFs / boolean of analytic primitives)

**Status:** CSG today goes through Manifold (mesh booleans). Analytic
booleans (sphere - cylinder etc.) require ray marching against an SDF
or hit-test trees.

**Why deferred:** Mesh CSG is the headline. Analytic CSG is a different
algorithm (raymarched SDFs or inside-outside hit walks); we'd want it
only when the user has a use case the current pipeline can't cover.

**Plan when picked up:**
- Maintain a parallel `pt::csg::AnalyticCsgScene` with leaf primitives
  + boolean nodes, mirroring `CsgScene`'s API.
- Two implementation paths:
  - **SDF path:** evaluate `min/max/-` of leaf SDFs in the shader, ray-march.
  - **Hit-walk path:** intersect ray with all leaves, sort hits, walk
    the tree to determine which segment is inside the result.
- The unified renderer already supports both an analytic primitive list
  and a mesh TLAS; analytic CSG is a third hit type to fold in.

---

## Tracy profiler instrumentation

**Status:** Tracy is fetched and built (`PT_ENABLE_TRACY=ON` in
`mac-debug` preset), but no `ZoneScoped` or frame markers in the code
yet.

**Why deferred:** No real perf bottleneck identified. We'll instrument
when we need to (e.g. when chasing 60 FPS at 4K).

**Plan when picked up:**
- `ZoneScopedN("Engine::Tick")`, `ZoneScopedN("RenderFrame")`,
  `ZoneScopedN("CsgScene::Bake")` — start with the obvious top-level
  scopes.
- Frame marker via `FrameMark` at top of the main loop.
- Connect with `tracy-profiler` from the Tracy repo.

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

## Single source of truth for PtPush + Frame UBO + Vulkan constants

**Status:** PtPush layout is currently defined in three places that
must agree by hand:

  1. `src/engine/Engine.cpp` -- the C++ struct used at dispatch site.
  2. `shaders/PathTrace.slang` -- the SPIR-V `cbuffer Push` (small)
     plus `cbuffer Frame` (spilled tail at vk::binding(14, 0)).
  3. `src/rhi_vulkan/VulkanDevice.h` -- `kPushSplitOffset`,
     `kFrameUboBinding`, `kFrameUboSize` constants.

A `static_assert(sizeof(PtPush) == ...)` in Engine.cpp catches one
class of drift (size change), but a field re-order with same total
bytes would silently mismatch between C++ and Slang.  The runtime
symptom of a desync is rendering corruption, not a build error.

**Why deferred:** the human-side cost of three-place edits is real
but rare (PtPush evolves once or twice per multi-week iteration),
and the static_assert + comment cross-references make drift
catchable in code review.

**Plan when picked up:**
- Option A: drive the layout from a single shared header
  (`src/rhi/PtPushLayout.h`) included by both Engine.cpp (via
  `#include`) and shaders/PathTrace.slang (via slangc's
  `-I` + `import`).  Slang's modules support `cbuffer`
  declarations now (post the PathTraceCloud/PathTraceMath split),
  so the spilled-tail Frame UBO would be a public cbuffer in a
  module that both PathTrace.slang and the engine import.
- Option B: derive the C++ side from Slang reflection
  (`slangc -reflection-json` or the Slang reflection API at
  init).  Less code coupling but adds a build-time tool.
- Option A is simpler; option B is more flexible.  Pick whichever
  feels right when the next big PtPush expansion lands.

**Acceptance:** changing one field in the shared layout source
must propagate to all three call sites with at most one edit.

---

## VulkanDevice::WriteBuffer fast-path for tiny runtime updates

**Status:** Current implementation is a synchronous staging-copy
+ vkQueueWaitIdle, fine for scene-load (mesh upload, CDF upload)
but produces a brief 5-15 ms stall for every cvar-change write
(e.g. `r_exposure` flip writing to `exposure_state`).

**Why deferred:** the user-visible cost is a one-frame hitch on
a knob change -- the user is the one initiating the change, and
the hitch is acceptable at human-scale interaction.  Not on a
hot path.

**Plan when picked up:**
- Add `vkCmdUpdateBuffer`-based fast-path for writes <=65536 B
  (Vulkan spec maximum for inline updates).  No staging buffer,
  no separate submit, no queue wait.
- Queue pending tiny writes; drain into the next frame's command
  buffer at BeginFrame, before any compute dispatch that reads
  the updated buffer.  Insert a HostWrite -> ComputeRead barrier
  for correctness.
- Keep the staging-copy fallback for big uploads (mesh / CDF) --
  vkCmdUpdateBuffer's 65 KB cap means it can't replace those.

---

## Async analytic-primitive BVH build (post-multi-thousand-prim)

**Status:** Today's analytic-prim BVH (landed in v0.3.9, `r_analytic_bvh_threshold`)
builds synchronously on the main thread inside `EnsurePrimitivesUploaded`.
That's fine because the trigger conditions are bounded — the rebuild fires
only on `primitives_dirty_` events (prim add/edit/remove, threshold cvar
change, scene load, backend switch), not per frame, and at N=1000 spheres
the build is well under a frame budget. The dirty flag is binary, so a
batch of N prim_sphere commands collapses into one rebuild at the next
frame.

**Why deferred:** No scene today hits the prim count where async pays for
itself, and async carries real complexity costs (worker thread, double-
buffered node array, frame-coherent swap-in so the renderer doesn't read
mid-build, edit-to-render latency = N frames). Sync wins for both demo
scenes and the 1000-prim stress test.

**When to pick this up:**

- **N > ~10k analytic prims** — O(N log N) median-split builds start
  approaching the 16 ms frame budget at 60 fps.
- **Per-frame BVH rebuilds** — e.g., voxel-destruction scenes where the
  set of finite primitives changes every tick. (Voxel-side itself will
  use a sparse octree + DDA, not a BVH; this point only applies if the
  user mixes voxel rigid-chunks with analytic prims that get added/
  removed each frame.)
- **Streaming worlds** with continuous primitive churn.

**Plan when picked up:**

1. **Dynamic thread-count gating.** Below a tunable prim-count threshold
   (default e.g. 2048 finite prims, exposed as `r_analytic_bvh_async_threshold`),
   keep the sync path — low-count edits stay zero-overhead. Above the
   threshold, dispatch to a worker. This avoids paying thread spawn /
   mutex acquisition costs on scenes that don't need it.
2. **Copy the CSG bake worker pattern.** `Engine::csg_scene_` already
   does the right thing: `bake_handle_` + `bake_phase_` (idle/baking/
   ready atomic) + `pending_baked_` swap-in on the main thread once the
   worker finishes. The analytic-prim BVH worker can sit alongside it —
   one bake job posted to `pt::jobs::JobSystem`, write the new node
   array into a `pending_bvh_` slot, flip the atomic, and the main
   thread swaps it in (uploading the new GPU buffer) at the start of
   the next `EnsurePrimitivesUploaded` call.
3. **Edit-coalescing.** While a build is in flight, additional
   `primitives_dirty_` events should queue (not start a second worker),
   and when the in-flight worker finishes, if dirty is still set, fire
   another. Two-state machine: idle → building → (idle | rebuild-queued).
4. **Don't stall the renderer.** While the new BVH is being built, the
   previous BVH stays bound and the renderer uses it. Lag is N frames
   after an edit — acceptable for the prim counts this targets (the
   user-visible result is "BVH updates feel slightly behind heavy
   bulk edits", same as Manifold CSG bakes today).
5. **Bench before / after.** Verify the async path's worker spawn +
   handoff isn't slower than sync at the threshold prim count. If it
   is, raise the threshold or tune the worker pool size. Goal: zero
   regression at small N, dominant speedup at large N.

**Pitfall:** double-buffering the GPU node buffer is non-trivial because
the buffer is shared between dispatch and write. Either grow the buffer
to 2x and swap halves with a fence, or stall briefly on the swap (the
existing CSG bake does the latter, accepting a small one-frame hitch on
swap-in).

---

## Smoke-test capture-frame race on Windows / PowerShell loopback TCP

**Status:** The user's `Send-DemontCmd` helper (one TCP connection per
command) lands commands ~2.3 s apart on Windows -- the .NET
`StreamWriter.Close()` teardown is the cost, not the network. On macOS
the same helper is <100 ms. At ~160 fps on a 5090 that's ~370 frames
between `r_capture_seed 1` and `r_capture_frame_at 64`, so by the time
the one-shot arms, `frame_index_` is already past 64 and the strict
equality check at `FrameCapture.cpp:365`
(`frame_index == s.one_shot_frame`) silently never fires. The capture
just doesn't happen and there's no warning.

Diagnostic that distinguished this from a stalled render loop:
`r_capture_seq "<prefix> 1 1"` fires on the *next* rendered frame
regardless of `frame_index_` (sentinel path in `MaybeCapture`), so the
seq path succeeded while the one-shot path silently did not. That
isolated it to a frame-counter race rather than a Vulkan / present
hang.

**Why deferred:** smoke tests pass with a batched helper
(single TCP connection, all commands written before `Close()`).
Not blocking any PR. Worth picking up because the silent no-op is
surprising to a fresh operator running the published smoke-test
script on Windows, and surprise > breakage on a workflow tool.

**Plan when picked up:** either is fine, pick one --
- Helper side: ship a batched `Send-DemontBatch` in the smoke-test
  guide / README and recommend it on Windows. Single TCP open,
  multiple `WriteLine`s, one `Close()`. Mac path unaffected.
- Engine side: relax the one-shot trigger to `frame_index >=
  s.one_shot_frame` in `FrameCapture::MaybeCapture`, disarming on
  first fire. Survives both command-latency misses and any genuine
  frame stutter that skips the target. Costs nothing on the happy
  path; the only behaviour change is that arming a one-shot at a
  past frame fires next frame instead of never firing -- arguably
  the better default anyway.

**Acceptance:** a fresh operator on Windows running the published
`Send-DemontCmd` loop captures all three armed shots without needing
to know about the helper's per-call latency.

---

## "engine: pipelines ready" log marker fires only on cold launches

**Status:** `Engine::Run` logs "engine: loading screen active (Vulkan
pipeline build pending)" / "engine: pipelines ready, normal rendering
resumed" only when `loading_frame_active_` flips from false -> true ->
false (`Engine.cpp:1554`, `:1568`). On a warm launch with cached
SPIR-V the pipeline build returns ready on first poll, so
`loading_frame_active_` is never set and neither message is emitted.
Documented as intentional in `Engine.h:354` (avoids per-frame log
spam during the build window).

This is correct for humans, but CI / smoke-test scripts that grep
for "pipelines ready" as a startup-sync marker will block forever
on a cached launch.

**Why deferred:** cosmetic. Today's smoke tests use a fixed
`Start-Sleep -Seconds 10` after launch, which works on both cold
and warm. Only worth fixing when someone wires a sync-on-marker
into automation.

**Plan when picked up:** emit a one-shot "engine: pipelines warm
(cached)" on the first frame when `loading_frame_active_` stays
false through the warm path, gated by the same one-shot flag so it
doesn't spam. Alternatively, always log a single
"engine: first frame rendered" marker right after the first
successful `Submit`/`EndFrame`, which gives CI scripts a uniform
sync point regardless of cold-vs-warm.

---

## SVGF a-trous smudges stars at night (post-composite fix)

**Status:** Observed on Mac at night-time HDRI / procedural-night
skies: stars become low-contrast smears that fade toward invisible
as `r_svgf_atrous_passes` increases. Root cause is structural to the
a-trous edge-stop formulation, not a tuning bug:

- Stars are sub-pixel point emitters -- one or a handful of bright
  pixels in an otherwise near-black sky.
- The luminance edge-stop `exp(-|L_center - L_sample| / sigma_L)`
  saturates against bright neighbours but treats dark neighbours
  as "consistent," so the kernel happily blends the star pixel's
  energy outward.
- Strides 1 / 2 / 4 / 8 each widen the footprint; by the last pass
  the energy is spread across a ~17x17 region and the star reads
  as a faint smudge instead of a point.
- Depth + normal edge-stops don't save us either -- sky pixels all
  share viewZ = far-plane and a synthesized "up" normal, so every
  star-vs-sky pair passes the geometric tests.

**Confirmed workarounds (both work, neither is the right answer):**

- `r_svgf_atrous_passes 1` -- one pass still spreads slightly but
  stars remain recognizably point-like.
- `r_denoiser svgf_basic` -- temporal only, no spatial filter at all.
  Stars stay sharp but the rest of the scene loses the a-trous
  cleanup on disocclusions.

Both are useful debugging escape hatches but neither lets us keep
the full a-trous quality for *non-star* pixels while protecting the
stars themselves.

**Why deferred:** the workarounds are usable today, the fix touches
multiple shaders + pipeline plumbing (~500 LOC), and the Vulkan NRD
integration is on the critical path ahead of this. Worth picking up
once NRD lands because NRD has the same class of issue with
point-like emitters and the same post-composite split is the
production answer there too.

**Plan when picked up -- Option A, post-denoise composite:**

The standard production pattern for point-like emitters (stars,
lens flare, distant headlights) is to route them around the
denoiser and composite back in *after* denoise / *before* bloom.
That preserves the denoiser quality on diffuse lighting while
keeping the points pixel-sharp, and bloom still picks them up
correctly because the composite happens upstream of the bloom
downsample chain.

1. **PathTrace.slang split.** Detect "star" hits when the primary
   miss samples an HDRI/procedural-sky texel above a configurable
   luminance threshold (`r_star_split_threshold`, default ~50
   cd/m^2 scene-referred). Write that contribution to a new
   `accum_stars` RGBA16F target instead of the main accumulator.
   Everything else (diffuse sky, sun disc, ground, geometry) keeps
   going to `accum_main` as today. Per-pixel binary classification,
   no overlap.

2. **Engine.cpp pipeline plumbing.** Allocate `accum_stars` next to
   the existing accumulator, bind it as a second storage texture
   in the path-trace pass, declare the matching binding slot, and
   wire it through `RebuildFrameResources()` resize handling.

3. **New `CompositeStars.slang` compute pass.** Reads
   `post_denoise_hdr` (denoised main) + `accum_stars` (unfiltered
   points), additive-blends them into a fresh `post_composite_hdr`
   target. Trivial kernel -- one add + one fma for the framecount
   normalisation, no edge-stops or filtering. Inserted between
   `Denoise()` and the bloom downsample in the render graph.

4. **Bloom + tonemap chain rebind.** Bloom currently reads
   `post_denoise_hdr`; switch it to `post_composite_hdr`. No
   shader change inside the bloom chain itself -- it's strictly
   an input rebind in the Vulkan / Metal pipeline setup. Bright
   stars will then drive bloom correctly (which they don't today
   because by the time the pixel reaches bloom it's already been
   blurred to mid-grey).

**DOF compatibility (already correct, documenting why):** the path
tracer's primary-ray aperture jitter is upstream of the star/main
split, so a star hit's `accum_stars` write already lands at the
DOF-jittered pixel position. Averaging across spp produces the
correct circle-of-confusion bokeh on stars, same as on geometry.
The composite pass reads `accum_stars` as-is, so DOF flows through
unchanged. No extra work needed.

**Bloom compatibility (already correct, documenting why):** the
composite pass runs *before* the bloom downsample chain reads its
input, so stars are present in the bright-pass extraction and
generate bloom halos exactly like any other bright pixel. This is
the whole reason for picking "post-denoise, pre-bloom" as the
insertion point -- it gives stars both denoiser-bypass *and*
bloom-participation, which is the production-correct behaviour.

**Threshold choice:** start with luminance-based selection because
it's the cleanest test for "point-like bright thing surrounded by
dark." If HDRIs with localized bright clouds end up false-positive
(unlikely but possible at sunset / aurora skies), switch to a
2x2 local-contrast test (this pixel >> max of 4 neighbours) which
is robust against bright *regions* but still flags bright *points*.

**Estimated scope:**

- `PathTrace.slang`: ~80 LOC for the split + new output binding.
- `Engine.cpp` + `EnsureFrameResources`: ~150 LOC for the
  `accum_stars` texture lifecycle + bind plumbing.
- `CompositeStars.slang` (new): ~50 LOC including header + IO.
- `VulkanDevice::RecordFrame` / `MetalDevice::RecordFrame`: ~100
  LOC each for the composite-pass dispatch insertion and bloom
  input rebind.
- Cvars + docs: ~30 LOC across `Cvars.cpp` + `cvars.md`.

Total ~500 LOC, single session of work.

**Acceptance:**

- At `r_denoiser svgf_atrous` with default 3 passes, stars at night
  remain pixel-sharp (no measurable smudging when comparing
  per-pixel luminance to a reference 1024-spp render).
- Bloom halos appear around stars proportional to their brightness.
- DOF defocus produces correct bokeh circles on stars when the
  focus plane is far from the camera.
- No regression on diffuse-scene denoise quality (a-trous still
  runs identically on `accum_main`).
- No measurable frame-time cost beyond the composite dispatch
  itself (~0.1 ms at 1080p; one full-screen add).

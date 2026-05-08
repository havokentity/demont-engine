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

## Vulkan denoiser (5090 path) — `r_denoiser nrd | svgf`

**Status:** P10 wires MetalFX for the Mac-only Metal backend. Vulkan
backend has no denoiser yet, so on a 5090 the path tracer runs at 1 spp
and looks noisy until accumulation converges.

**Why deferred:** MetalFX is Apple-only (`MetalFX.framework`). Implementing
denoising on Vulkan is a separate code path; the user is on M4 today,
Windows/5090 is a P12 milestone.

**Plan when picked up:**
- Pick a backend lib:
  - **NRD** (NVIDIA RealTime Denoiser) — BSD-licensed since 2024, ships
    with Vulkan bindings, works on AMD/Intel too. Recommended.
  - **SVGF / A-trous** — write our own compute shader (~150 LOC). More
    educational, fully portable.
  - **DLSS Ray Reconstruction** — highest quality, NVIDIA-only, needs
    a 5090-specific shim.
- The G-buffer the shader emits in P10 (color, depth, motion, normals)
  is reusable as-is; only the post-pass library differs.
- Add `nrd` / `svgf` as values to the existing `r_denoiser` cvar in
  `Engine.cpp`.
- Wire the denoise dispatch from `VulkanDevice::Denoise(...)` (mirror
  the Metal-side API).

**Acceptance:** Same as P10's Metal acceptance — 1 spp + denoiser at
&gt;30 FPS at 1080p, no major ghosting, edges preserved.

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

## Async pipeline build + loading screen

**Status:** Persistent `VkPipelineCache` shipped (loads from
`%LOCALAPPDATA%/demont/pipeline.cache` on Windows,
`$XDG_CACHE_HOME/demont/pipeline.cache` on POSIX). Subsequent launches
hit the cache and pipeline creation is near-instant.

**Why deferred:** First-launch freeze (≈1-3 s on NVIDIA, dominated by
PathTrace.spv → driver-native compile inside `vkCreateComputePipelines`)
is still a one-time bad UX moment that the cache can't solve. It only
hurts on first-ever run, after `pipeline.cache` is wiped, or after a
driver upgrade — but it's the difference between "the app is starting"
and "the app froze on launch".

**Plan when picked up:**
- Spin up a worker thread at `VulkanDevice::Init` that builds all
  compute pipelines (`pathtrace`, `autoexpose`, `perfoverlay`, future
  ones) using a thread-private `VkPipelineCache`. Vulkan pipeline
  creation is thread-safe; we just need to keep the device handle
  accessible to the worker.
- Block in the main loop on a "pipelines ready" atomic before the first
  `RenderFrame` that needs them — render a loading-screen frame in the
  meantime (clear-to-color + a small spinner / text in the
  `PerfOverlay_Win32` swapchain pass, which can run without the
  PathTrace pipeline).
- After the worker finishes, `vkMergePipelineCaches` the worker's cache
  into the main one before saving on shutdown so the merged cache lands
  on disk.
- The Tonemap / Bloom shaders go through a separate engine inline path,
  not `VulkanDevice::build_pipeline`. Audit when implementing.

**Mac/Metal:** Metal does the SPIR-V→native compile lazily on first
draw and is much faster than Vulkan's eager `vkCreateComputePipelines`.
The macOS port doesn't have the same freeze and isn't on this critical
path — but the same async-build loading-screen scaffold could host a
Metal `MTLLibrary.makeComputePipelineState(completionHandler:)` worker
if we ever see a similar stall there.

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

# Follow-ups & deferred work

Living doc. Things we knowingly skipped or postponed in earlier phases,
with enough context to pick them up later. Each item has a one-line
hook, a "why deferred", and a sketch of the work needed. Roughly ordered
by priority.

---

## P10: MetalFX TemporalDenoisedScaler — Mac path

**Status:** `r_denoiser` cvar exists (`off | metalfx`) but only `off`
is wired today. `r_spp` cvar exists for super-sampling-per-dispatch as
a stopgap (set `r_spp 4` for cleaner motion frames at 4x GPU cost).

**Why deferred from this session:** correct integration requires a
G-buffer (depth + motion vectors), prev-frame camera tracking, and an
ObjC++ shim around `MetalFX.framework`. Half-baked it produces ghosting
artifacts that are hard to debug. Leaving it for a focused session.

**Implementation checklist:**

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

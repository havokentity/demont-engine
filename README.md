# DeMonT Engine

> A real-time game engine for Apple Silicon that **never rasterizes**.
> Every pixel is a ray. Path tracing is the renderer; mesh CSG via
> Manifold is the headline; MetalFX (and later NRD on Vulkan) keeps
> the frame interactive at 1 sample per pixel.
>
> **DeMonT** = **De** **Mon**te Carlo-esque **T**racer — *-esque*
> because the algorithm will keep evolving (MIS, ReSTIR, bidirectional
> later) without ever falling back to a rasterizer.

C++23, three-backend RHI (Software / Metal / Vulkan), Slang shaders,
unified path-tracing kernel that intersects analytic primitives **and**
triangle meshes (via hardware ray query) in one pass.

The full design lives in [`Raytracer Plan/`](Raytracer%20Plan/) — five
hand-authored HTML documents covering architecture principles, design,
phase-by-phase implementation plan, console protocol, and a modern-C++
cheatsheet — plus [`Raytracer Plan/FOLLOW_UPS.md`](Raytracer%20Plan/FOLLOW_UPS.md)
which tracks deferred work in detail.

## Why no rasterizer

Triangle rasterization is fundamentally a different lighting model than
path tracing — every special case (shadows, GI, reflections, refraction,
caustics, proper transparency) has to be re-implemented as a separate
hack on top. By committing to rays for every pixel, the engine collapses
those special cases into a single algorithm and frees us to focus
hardware budget on the things that *only* path tracing can do well:
millions of CSG-defined primitives, mirror-perfect reflections, real
glass, soft shadows from area lights — all without an `if-rasterizer`
switch in the codebase.

The cost is the noise floor and the GPU bill. Both are addressed by
modern temporal denoisers (MetalFX TemporalDenoisedScaler today, NRD on
the Vulkan side once Windows lands) and by the path tracer's own
accumulation when the camera is stationary.

## Status

| Phase | Status |
|---|---|
| P0–P3 toolchain, RHI, Software + Metal backends | ✓ |
| P4 Vulkan via MoltenVK | ✓ |
| P5 FPS camera + analytic primitives | ✓ |
| P6 Materials (Lambert / metal / dielectric) | ✓ |
| P7 Path tracing + HDR accumulation + ACES tonemap | ✓ |
| P8 Triangle meshes + hardware ray query | ✓ |
| **P9 Mesh CSG via Manifold (headline)** | ✓ |
| Renderer unification (analytic + mesh in one shader) | ✓ |
| **P10 MetalFX TemporalDenoisedScaler (Mac)** | ✓ |
| Vulkan denoiser (NRD or SVGF) for Windows / 5090 | queued — see FOLLOW_UPS |
| P11 MIS, env maps, archived cvars, scene I/O | next |
| P12 Windows bringup with native Vulkan | later |

## Build

Requires macOS 26+ (for MetalFX TemporalDenoisedScaler), CMake ≥ 3.27,
Ninja, Apple's clang, Vulkan SDK (optional but recommended).

```sh
cmake --preset mac-debug
cmake --build build/mac-debug
./build/mac-debug/src/app/demont
```

CMake auto-downloads Slang and Apple's metal-cpp into `third_party/` on
first configure. Manifold, fmt, glm, glfw, mimalloc, enkiTS, civetweb,
nlohmann_json, tomlplusplus and (optionally) Tracy are vendored via
FetchContent.

## Run

Default scene ships with three analytic spheres (red Lambert / gold
roughened metal / glass dielectric IOR 1.5), a ground plane, **and** a
CSG drilled cube (`box - sphere`) — all rendered in one pass.

- **WASD** + **Space/Ctrl** — fly camera
- **Right-click drag** — mouse-look (release to free the cursor)
- **Shift** — sprint
- **Backtick (`** `)`** — open the in-window console (AppKit overlay,
  CoreText-rendered, blurred backdrop). Tab completes commands +
  cvar values; Up/Down for history; Esc to clear or close. Pasting
  text with newlines runs each line as its own command.

The web console is **not** auto-opened anymore (it kept stealing focus).
Type `web_console` in either console (or hit the URL directly:
<http://localhost:27960>) to launch it.

For shell scripting, the line protocol on `127.0.0.1:27961`:

```sh
echo "sys_info"          | nc localhost 27961
echo "r_denoiser metalfx" | nc localhost 27961
echo "toggle r_denoiser" | nc localhost 27961
echo "csg_dump"          | nc localhost 27961
```

## Notable cvars + commands

| | |
|---|---|
| `r_backend` | `none` / `software` / `metal` / `vulkan` |
| `r_denoiser` | `off` / `metalfx` (Mac); A/B with `toggle r_denoiser` |
| `r_spp` | samples per pixel per dispatch (1..32). Higher = cleaner motion at proportional GPU cost. |
| `r_max_bounces` | path-tracer bounce cap (default 8) |
| `csg_*` | live CSG editing — `csg_box`, `csg_sphere`, `csg_op subtract …`, `csg_set_root`, `csg_dump`, … |
| `prim_*` | analytic primitives — `prim_sphere`, `prim_plane`, `prim_remove`, `prim_list`, … |
| `screenshot <path.ppm> [accum\|denoise_color\|depth\|motion]` | in-app GPU readback to PPM, no OS Screen Recording permission needed |
| `toggle <cvar>` | cycle a cvar's allowed_values — quick A/B for any boolean / enum |
| `web_console` | open the browser UI on demand |

## Architecture

```
src/
  core/     Log, Memory (tagged allocator + arenas + pool),
            Jobs (enkiTS), Hardware (sysctl introspection)
  rhi/      Render Hardware Interface — Device / CommandBuffer /
            Handles / Resources / Denoise. Backend-agnostic.
  rhi_software/  CPU compute (used as a clear+present today).
  rhi_metal/     Metal backend (metal-cpp + Slang→MSL),
                 MetalFXDenoiser shim, hardware ray query.
  rhi_vulkan/    Vulkan backend via MoltenVK on Mac, native on Windows.
  renderer/      Camera, MeshGen, Csg/CsgScene (Manifold-backed).
  console/       CVar/Command registry + civetweb WS/HTTP server +
                 raw line-protocol TCP server.
  app/           GLFW window, Cocoa bridges, native NSPanel overlay.
  engine/        Top-level orchestrator + the unified path-trace pass.
shaders/   PathTrace.slang  ← the only kernel: analytic + mesh + materials
web/       Embedded vanilla-JS console UI (no build step)
cmake/     SetupBinaries, Slang.cmake (compile_slang),
           EmbedResource.cmake, Dependencies.cmake
```

## License

MIT. See [`LICENSE`](LICENSE).

The plan documents in `Raytracer Plan/` are © Rajesh D'Monte;
everything in `src/`, `shaders/`, `web/`, `cmake/` is the
implementation against that plan.

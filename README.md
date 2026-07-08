# DeMonT Engine

[![Build](https://github.com/havokentity/demont-engine/actions/workflows/build.yml/badge.svg)](https://github.com/havokentity/demont-engine/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)
[![Shaders: Slang](https://img.shields.io/badge/shaders-Slang-2D75A0)](https://shader-slang.org/)
[![Metal 4](https://img.shields.io/badge/Metal-4-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-AC162C?logo=vulkan&logoColor=white)](https://vulkan.org/)

> **Note on CI coverage:** CI runs on `macos-26` so the full MetalFX
> `TemporalDenoisedScaler` path is built and exercised. The shim is
> also SDK-version-gated (`__MAC_OS_X_VERSION_MAX_ALLOWED >= 260000`),
> so contributors on macOS 14/15 dev boxes can still build — the
> denoiser compiles to no-op stubs and the engine falls back to the
> un-denoised path tracer at runtime.

> A real-time game engine that **never rasterizes**. Every pixel is a
> ray. Path tracing is the renderer; mesh CSG via Manifold is the
> headline; modern temporal denoisers (MetalFX on Apple, SVGF / OptiX
> on Vulkan) keep the frame interactive at 1 sample per pixel.
>
> **DeMonT** = **De** **Mon**te Carlo-esque **T**racer — *-esque*
> because the algorithm will keep evolving (MIS, ReSTIR, bidirectional
> later) without ever falling back to a rasterizer.

## Supported platforms

| Platform | Backend | Denoiser | Status |
|---|---|---|---|
| **Apple Silicon** (M-series, macOS 26+) | Metal (hardware RT) | MetalFX TemporalDenoisedScaler | ✓ primary |
| **NVIDIA RTX** (Windows / Linux, RTX 2000-series and up) | Vulkan (`VK_KHR_ray_query` + `VK_KHR_acceleration_structure`) | SVGF (basic / à-trous) or OptiX HDR / temporal; NRD library queued | ✓ supported (since v0.3.0) |
| Software fallback (any) | CPU reference path tracer (Embree-traced meshes + analytic primitives) | none | reference / bring-up |

Built on C++23 with a platform RHI (Software / Metal on macOS, native
Vulkan on Windows/Linux), Slang shaders that cross-compile to MSL and SPIR-V, and a unified
path-tracing kernel that intersects analytic primitives **and**
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
modern temporal denoisers (MetalFX TemporalDenoisedScaler on Apple,
the in-house SVGF chain on both Metal and Vulkan, OptiX HDR / temporal
variants on NVIDIA) and by the path tracer's own accumulation when the
camera is stationary.

## Status

| Phase | Status |
|---|---|
| P0–P3 toolchain, RHI, Software + Metal backends | ✓ |
| P4 Native Vulkan (Windows/Linux) | ✓ |
| P5 FPS camera + analytic primitives | ✓ |
| P6 Materials (Lambert / metal / dielectric) | ✓ |
| P7 Path tracing + HDR accumulation + ACES tonemap | ✓ |
| P8 Triangle meshes + hardware ray query | ✓ |
| **P9 Mesh CSG via Manifold (headline)** | ✓ |
| Renderer unification (analytic + mesh in one shader) | ✓ |
| **P10 MetalFX TemporalDenoisedScaler (Mac)** | ✓ |
| Vulkan denoiser: SVGF (basic / à-trous) on both backends + OptiX on NVIDIA | ✓ (`nrd` aliases the à-trous chain until the NRD library is wired) |
| P11 MIS, env maps, archived cvars, scene I/O | ✓ |
| P12 Windows bringup with native Vulkan | ✓ (PR #1, merged 2026-05-08 → v0.3.0) |

Post-P12 work — SDF raymarching phases 1–3 (#97–#99), the React scene
editor, the physics / destruction arcs, and the integration waves — is
tracked in GitHub issues and the `v0.3.x` release tags rather than
phase rows here.

## Build

### macOS (Apple Silicon)

Requires macOS 26+ (for MetalFX TemporalDenoisedScaler), CMake ≥ 3.27,
Ninja, and Apple's clang. Vulkan/MoltenVK is intentionally not built on
macOS; the native Apple path is Metal.

```sh
cmake --preset mac-debug
cmake --build build/mac-debug
./build/mac-debug/src/app/demont
```

### Windows (NVIDIA RTX)

Requires:
- Windows 10/11 with current NVIDIA drivers (RTX 2000-series or newer for hardware ray-tracing)
- [Vulkan SDK 1.3.296+](https://vulkan.lunarg.com/sdk/home) — sets `VULKAN_SDK` env var. 1.4.x SDKs build and run cleanly; no upper bound pinned.
- CMake ≥ 3.27
- Ninja
- MSVC 2022 or MSVC 2026 (Build Tools or full IDE) — or clang-cl as an alternative. C++23 support required.

```pwsh
cmake --preset win-debug
cmake --build --preset win-debug
.\build\win-debug\src\app\demont.exe
```

(or `win-release` for an optimised build, `win-clang-debug` for clang-cl.)

The Windows build:
- Skips the Metal RHI and the macOS Cocoa overlay path (Cocoa shims link as no-ops), while using a native Win32 console overlay on Windows
- Auto-selects Vulkan as the default backend (`r_backend vulkan`)
- Uses native Vulkan (no MoltenVK portability extensions)
- Picks a discrete GPU when both iGPU + dGPU are present
- Compiles every shader to SPIR-V via Slang's prebuilt Windows binary

`VK_KHR_acceleration_structure` + `VK_KHR_ray_query` are wired — the path tracer issues hardware-traversed ray queries from a compute kernel against BLAS/TLAS built per CSG/mesh bake. The dedicated `VK_KHR_ray_tracing_pipeline` (raygen / miss / hit shader binding tables) is queued; ray-query in compute already gives hardware traversal, so the extra pipeline pathway is only worth wiring when we need dynamic hit groups.

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
| `r_backend` | macOS: `none` / `software` / `metal`; Windows/Linux: `none` / `software` / `vulkan` |
| `r_denoiser` | `off` / `metalfx` / `svgf_basic` / `svgf_atrous` / `svgf_basic_metalfx` / `svgf_atrous_metalfx` / `nrd` / `optix_hdr` / `optix_hdr_aov` / `optix_temporal_hdr` / `optix_temporal_hdr_aov`. Mac accepts the metalfx / svgf / nrd set; NVIDIA Vulkan adds the optix variants. `list_cvars r_denoiser` prints the full matrix; A/B with `toggle r_denoiser` |
| `r_spp` | samples per pixel per dispatch (1..32). Higher = cleaner motion at proportional GPU cost. |
| `r_max_bounces` | path-tracer bounce cap (default 8) |
| `csg_*` | live CSG editing — `csg_box`, `csg_sphere`, `csg_op subtract …`, `csg_set_root`, `csg_dump`, … |
| `prim_*` | analytic primitives — `prim_sphere`, `prim_plane`, `prim_remove`, `prim_list`, … |
| `screenshot <name> [accum\|denoise_color\|bloom_mip0\|swap\|depth\|motion]` | in-app GPU readback, no OS Screen Recording permission needed. Format comes from `r_capture_format` (`png`, default, or `ppm`) — the matching extension is auto-appended to `<name>`, overriding any extension you typed |
| `toggle <cvar>` | cycle a cvar's allowed_values — quick A/B for any boolean / enum |
| `web_console` | open the browser UI on demand |

## Architecture

```
src/
  core/     Log, Memory (tagged allocator + arenas + pool),
            Jobs (enkiTS), Hardware (sysctl introspection)
  rhi/      Render Hardware Interface — Device / CommandBuffer /
            Handles / Resources / Denoise. Backend-agnostic.
  rhi_software/  CPU reference path tracer — Embree-traced meshes +
                 analytic primitives, Lambert direct lighting.
  rhi_metal/     Metal backend (metal-cpp + Slang→MSL),
                 MetalFXDenoiser shim, hardware ray query.
  rhi_vulkan/    Native Vulkan backend for Windows/Linux.
  renderer/      Camera, Csg/CsgScene (Manifold-backed),
                 AnalyticBvh (incl. SDF clusters), LightTree,
                 GltfImporter, HdrImage, Astronomy/BscCatalog.
  physics/       PhysicsSystem (Verlet + rigid body), SmokeSPH,
                 OceanFFT.
  destruction/   VoxelGrid + Voxelizer (CSG mesh → voxel box pile).
  effects/       ParticleSystem.
  audio/         AudioSystem.
  editor/        EditorRoutes (panel HTTP/WS routes) + SceneGraph.
  console/       CVar/Command registry + civetweb WS/HTTP server +
                 raw line-protocol TCP server.
  app/           GLFW window, Cocoa bridges, native NSPanel overlay.
  engine/        Top-level orchestrator + the unified path-trace pass.
shaders/   PathTrace.slang megakernel + 28 more .slang kernels
           (SVGF denoise chain, ReSTIR, bloom, clouds, SDF libraries,
           tonemap / post).
web/       Vanilla-JS console UI at the top level (no build step) +
           editor/ — React 18 + Vite multi-panel scene editor whose
           npm build is embedded into the binary by cmake/Editor.cmake.
cmake/     SetupBinaries, Slang.cmake (compile_slang),
           EmbedResource.cmake, Dependencies.cmake, Editor.cmake
```

## License

MIT. See [`LICENSE`](LICENSE).

The plan documents in `Raytracer Plan/` are © Rajesh D'Monte;
everything in `src/`, `shaders/`, `web/`, `cmake/` is the
implementation against that plan.

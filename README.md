# DeMonT PathTracer

> **DeMonT** = **De** **Mon**te Carlo-esque **T**racer
> A real-time GPU path tracer for Apple Silicon. Built on Monte Carlo
> path tracing today; the *-esque* leaves room for the algorithm to
> evolve (MIS, denoising, perhaps bidirectional or ReSTIR later).

C++23, three-backend RHI (Software / Metal / Vulkan), Slang shaders.
Designed around interactive mesh CSG via
[Manifold](https://github.com/elalish/manifold) as the headline
feature.

The full design lives in [`Raytracer Plan/`](Raytracer%20Plan/) — five
hand-authored HTML documents covering architecture principles, design,
phase-by-phase implementation plan, console protocol, and a
modern-C++ cheatsheet. They drive everything in this repo.

## Status

| Phase | Status |
|---|---|
| P0 toolchain + tagged memory manager | ✓ |
| P1 GLFW window + network console + JobSystem | ✓ |
| P2 RHI interface + Software backend | ✓ |
| P3 Metal backend (metal-cpp + Slang→MSL) | ✓ |
| P4 Vulkan via MoltenVK | pending SDK install |
| P5 FPS camera + analytic primitives | ✓ |
| P6 Materials (Lambert / metal / dielectric) | ✓ |
| P7 Path tracing modes + HDR accumulation + ACES tonemap | ✓ |
| P8 Triangle meshes + hardware ray tracing | next |
| **P9 Mesh CSG via Manifold (headline)** | next |
| P10–12 Denoising, polish, Windows | later |

## Build

Requires macOS 14+, CMake ≥ 3.27, Ninja, Apple's clang.

```sh
cmake --preset mac-debug
cmake --build build/mac-debug
./build/mac-debug/src/app/demont
```

CMake auto-downloads Slang and Apple's metal-cpp headers into
`third_party/` on first configure (~52 MB Slang binary, ~200 KB metal-cpp).

## Run

The window opens with the path-traced scene on Metal. Three colored
spheres on a grey plane: red Lambert, gold roughened metal, glass
dielectric (IOR 1.5).

- **WASD** + **Space/Ctrl** — fly camera
- **Right-click drag** — mouse-look (release to free the cursor)
- **Shift** — sprint
- **Backtick (`** `)` — open the in-window console (AppKit overlay,
  CoreText-rendered, blurred backdrop). Tab completes commands and
  cvar values; Up/Down for history; Esc to clear or close.

The browser-based console at <http://localhost:27960> opens
automatically on launch — same WebSocket-driven CVar / Command system
as the in-window overlay, with a sidebar listing every cvar's live
value.

For shell scripting, the line protocol on `127.0.0.1:27961`:

```sh
echo "sys_info"          | nc localhost 27961
echo "r_clear_color 0.05 0.1 0.2" | nc localhost 27961
echo "r_max_bounces 16"  | nc localhost 27961
```

## Architecture

```
src/
  core/     Log, Memory (tagged allocator + frame arenas + pool),
            Jobs (enkiTS), Hardware (sysctl introspection)
  rhi/      Render Hardware Interface — Device / CommandBuffer /
            Handles / Resources. Backend-agnostic.
  rhi_software/  CPU compute (today: clear stash). Metal-presented.
  rhi_metal/     Metal backend via metal-cpp. Slang→MSL at build,
                 MSL→MTLLibrary at runtime.
  rhi_vulkan/    (P4 — pending SDK)
  renderer/      Camera and (later) Scene SoA.
  console/       CVar/Command registry + civetweb WS/HTTP server +
                 raw line-protocol TCP server.
  app/           GLFW window, Cocoa bridges, native overlay (NSPanel
                 with NSVisualEffectView + CoreText).
  engine/        Top-level orchestrator.
shaders/   Clear.slang, Scene.slang, PathTrace.slang
web/       Embedded vanilla-JS console UI
cmake/     SetupBinaries (Slang + metal-cpp auto-download),
           Slang.cmake (compile_slang), EmbedResource.cmake
```

## License

MIT. See [`LICENSE`](LICENSE).

The plan documents in `Raytracer Plan/` are © Rajesh D'Monte;
everything in `src/`, `shaders/`, `web/`, `cmake/` is the
implementation against that plan.

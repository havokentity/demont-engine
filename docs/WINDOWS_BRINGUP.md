# Windows / NVIDIA bringup — Claude handoff

> Self-contained context-restore for a fresh Claude session running on
> the user's Windows + NVIDIA RTX box. The previous Mac session got
> this branch through CI; runtime testing happens here.

**Branch:** `feature/windows-nvidia` (PR #1, *open*, **not merged**)
**Target tag:** `v0.3.0` once runtime is proven and the PR lands
**Phase in plan:** P12 — Windows bringup with native Vulkan
(the 13-phase plan lives in `Raytracer Plan/*.html`)

## State at handoff (post-runtime-bringup)

Engine **builds, boots, dispatches the path tracer on the RTX 5090 with zero
Vulkan validation errors.** Spheres + ground plane render. Sky brightness is
the next visual mismatch to chase. Diff against the pre-bringup tip:

- `cmake/patch_enkits.cmake` (new): portable replacement for the Mac-only
  `sed -i.bak` that fixed `std::is_pod` in enkiTS. Pure-CMake, runs on every OS.
- `cmake/Dependencies.cmake`: enkiTS PATCH_COMMAND now calls `cmake -P`.
- `cmake/Slang.cmake`: passes `-DPT_TARGET_SPIRV` / `-DPT_TARGET_METAL` to
  slangc so the shader can fork its push-constant layout per-target.
- `shaders/PathTrace.slang`: `[[vk::push_constant]] cbuffer Push` now splits
  on SPIR-V — small prefix (112B) stays in push, tail (336B) lives in a new
  `cbuffer Frame` at `vk::binding(14, 0)`. Mac path is unchanged.
- `src/rhi_vulkan/VulkanDevice.{h,cpp}`: ~1000 LoC. Vulkan 1.3, ray-query +
  acceleration-structure + deferred-host-operations + robustness2 extensions,
  features pNext chain, 16-binding shared descriptor set layout, real
  CreateBuffer / WriteBuffer / DestroyBuffer (host-visible, persistent map),
  WriteTexture (staging buffer + cmd-buffer copy), CreateBLAS / CreateTLAS
  via VK_KHR_acceleration_structure, per-frame Frame UBO ring for the
  spilled push tail, full descriptor writes in `VulkanCommandBuffer::Dispatch`
  with engine-slot → vk-binding translation tables.

Confirmed working at runtime:
- Window creation (GLFW + Vulkan surface)
- Discrete-GPU pick (RTX 5090 over any iGPU)
- `maxPushConstantsSize` reported = 256 (NVIDIA standard)
- Path-trace pipeline creation (no SPIR-V capability errors)
- Push-constant split: 112B push + 336B Frame UBO upload per dispatch
- Descriptor binding for all 16 bindings every dispatch (storage_image x8,
  AS x1, storage_buffer x5, ubo x2) with `nullDescriptor` for unbound slots
- BLAS / TLAS build for the CSG-drilled cube
- WriteTexture upload (sunset.hdr env_map, BSC starmap, procedural moon)
- Console servers (HTTP + line) bind, web UI accessible

Open visual mismatches vs Mac (next session's work):
- Sky too bright. The path-trace shader has `output[tid] =
  acesTonemap(avg * exposure_pad.x)` (PathTrace.slang:1667), so tonemap is
  inline — but something in the input data (env_map, star_map, moon_map?
  exposure?) is feeding values that overwhelm ACES rolloff. Debug live via
  the web console at <http://127.0.0.1:27960/>:
  - `r_env_intensity 0` → does sky go to procedural gradient?
  - `r_show_stars 0` → does it dim?
  - `r_exposure 0.5` → does it dial down proportionally?
  - `r_env_map 0` → falls back to procedural sky entirely
  Whichever toggle fixes brightness names the culprit.
- CSG cube possibly missing. TLAS *builds* (no errors), but whether the
  ray-query hits register correctly in the shader is not verified.

Not blocking but cosmetic:
- `Tonemap` / `BloomDown` / `BloomUp` pipelines aren't built on Vulkan yet
  (they have their own descriptor layout + 576B push-constant block that
  needs the same split treatment as PathTrace). Engine's tonemap dispatch
  silently no-ops; PathTrace's inline tonemap does the work for now.
- JobSystem reports "1 worker thread" on Windows. Functional but suboptimal
  on a 16-core 9950X3D. Worth investigating in `JobSystem.cpp`.
- Native Vulkan terminal output renders cleaner in Windows Terminal than
  CLion's bundled emulator (better ANSI / Unicode handling).

## Build & run

Prerequisites:
- Windows 10/11 with current NVIDIA drivers
- Vulkan SDK 1.3+ → `VULKAN_SDK` env var must be set
- CMake ≥ 3.27, Ninja, MSVC 2022 (Build Tools or full IDE)

```pwsh
cmake --preset win-debug
cmake --build build/win-debug --parallel
.\build\win-debug\src\app\demont.exe
```

(`win-release` for optimised, `win-clang-debug` for clang-cl.)

## Expected first-run behaviour

- A 1280×720 GLFW window opens
- The Vulkan backend auto-selects (`r_backend vulkan`) on non-Apple —
  see `src/engine/Engine.cpp` around L66
- Physical-device pick prefers `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`
  (the RTX) over any iGPU — see `src/rhi_vulkan/VulkanDevice.cpp`
  device-pick block
- The path tracer runs an **inline ray-query compute kernel** against
  hardware-built acceleration structures. `VK_KHR_ray_query` and
  `VK_KHR_acceleration_structure` are wired in this PR (BLAS/TLAS
  built on every CSG bake / mesh load). The dedicated
  `VK_KHR_ray_tracing_pipeline` (RT pipeline state object with
  raygen / miss / hit shaders) is deliberately not wired — ray-query
  in a compute shader gives us the same hardware traversal without
  the extra pipeline complexity, and the RT pipeline pathway is queued
  for a later PR if we ever need shader binding tables / dynamic
  hit groups.
- HTTP console at <http://localhost:27960>; line protocol on
  `127.0.0.1:27961`. The web console does NOT auto-open — type
  `web_console` in either console to launch.
- Default scene: red Lambert sphere + roughened gold metal sphere +
  glass IOR-1.5 dielectric + ground plane + a CSG-drilled cube,
  rendered in one pass.

## Likely failure modes (in rough order of probability)

1. **GLFW window doesn't open / surface creation fails**
   — check `vkCreateWin32SurfaceKHR` is in the extension list; should
   come automatically via `glfwGetRequiredInstanceExtensions`
2. **No discrete GPU picked / falls back to integrated**
   — inspect `PickPhysicalDevice` in `VulkanDevice.cpp`; verify the
   RTX shows up in `vkEnumeratePhysicalDevices`
3. **Validation layer crashes at startup**
   — validation is on in Debug builds; if `VK_LAYER_KHRONOS_validation`
   yells anything, fix that first
4. **Swapchain format mismatch**
   — we ask for `VK_FORMAT_B8G8R8A8_UNORM`; if the driver doesn't
   support it, fall back to whatever the surface offers
5. **Shader pipeline creation fails**
   — Slang already produced the SPIR-V; if the driver rejects it,
   dump the SPIR-V and run it through `spirv-val`
6. **Console server can't bind**
   — Winsock init + ws2_32 are in `ConsoleServer.cpp`; failures show
   up as `WSA error <n>` log lines

If anything goes wrong, capture:
- exact error / stack / validation-layer output
- `vulkaninfo --summary` so we know what the RTX is reporting
- the demont console banner (logs picked GPU + extensions)

## Commits in this PR

| commit | what |
|---|---|
| `47ac364` | cross-platform scaffolding (presets, src gating, ConsoleOverlay stub, README) |
| `dc939d9` | CI: jakoch Vulkan SDK install action |
| `364ec88` | CI: pin Vulkan 1.3.296.0 |
| `d64020b` | POSIX→MSVC shims for Log/main + Winsock for ConsoleServer |
| `5ed61b0` | `/Zc:preprocessor` for `__VA_OPT__` |
| `a8c8a5d` | `M_PI` → `std::numbers::pi` |
| `f8db291` | `__fp16` → portable IEEE-754 binary16 decode |
| `a7fdf53` | docs: this handoff document (initial draft) |
| `79dca33` | Vulkan path-tracer port + GPU auto-exposure + console UX |
| `0f8b9ae` | Win32 console overlay (Mac parity, native HWND child + slide/fade) |
| `40f5704` | Vulkan: query supported device features before enabling optional bits |
| `d6d635e` | README: clarify Win32 overlay (non-Apple Cocoa shims still no-op) |
| `9677716` | ConsoleOverlay_Stub: comment now reflects non-Apple, non-Windows targets |
| `218ab98` | README: drop redundant build dir when using `cmake --build --preset` |
| `47e54f4` | docs(vulkan): fix 16-binding comments and nullDescriptor note |

## Out of scope for this PR (queued for follow-ups)

- `VK_KHR_ray_tracing_pipeline` (the dedicated RT pipeline state object
  with raygen / miss / hit shaders) — ray-query in compute already
  gives hardware traversal, so this is only worth wiring if we ever
  need shader binding tables / dynamic hit groups
- NRD (NVIDIA Real-time Denoisers) integration on Vulkan
- Linux end-to-end CI (presets ship; no workflow yet)
- DXGI / Windows-native present path for the software RHI fallback

Tracked in `Raytracer Plan/FOLLOW_UPS.md`.

## Project rules (do not break)

- **No `Co-Authored-By: Claude` trailer on demont commits** — strip
  it from every commit message
- **Agent owns all CMake** — `CMakeLists.txt`, presets,
  `cmake/*.cmake` edits flow through the agent; user does not edit
  CMake directly
- **One phase at a time** — don't pull future-phase work into this
  v0.3.0 PR (RT extensions, NRD, Linux CI all wait)
- **Real physics + metric units** — 1 world unit = 1 metre; real
  atmospheric / optical / mechanical constants; no demo-scaled
  shortcuts or image-based hacks

## Happy path

1. demont.exe boots → window opens → red sphere + gold sphere + glass
   sphere + drilled cube render at interactive rates
2. eyeball that the image matches the Mac reference
3. report success → user merges PR #1 → tag `v0.3.0`
4. open the next PR for hardware RT extensions

## Sad path

Iterate on whatever blew up. Mac build must stay clean throughout —
treat any commit that regresses Mac CI as a bug to fix immediately.

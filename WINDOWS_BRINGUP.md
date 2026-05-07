# Windows / NVIDIA bringup — Claude handoff

> Self-contained context-restore for a fresh Claude session running on
> the user's Windows + NVIDIA RTX box. The previous Mac session got
> this branch through CI; runtime testing happens here.

**Branch:** `feature/windows-nvidia` (PR #1, *open*, **not merged**)
**Target tag:** `v0.3.0` once runtime is proven and the PR lands
**Phase in plan:** P12 — Windows bringup with native Vulkan
(the 13-phase plan lives in `Raytracer Plan/*.html`)

## State at handoff

CI is **green** on commit `f8db291`:

| job | runner | status |
|---|---|---|
| macOS (Apple Silicon) | `macos-26` | success |
| Windows (NVIDIA RTX target) | `windows-latest` | success |

What that proves:
- Every TU compiles under MSVC 14.44 (VS 2022)
- Linking against `Ws2_32.lib` works for the line-protocol TCP server
- Slang's Windows-x86_64 binary compiles all 4 shaders to SPIR-V
  (`PathTrace`, `Tonemap`, `BloomDown`, `BloomUp`)
- Vulkan SDK 1.3.296.0 resolves `Vulkan::Vulkan` cleanly
- The final `demont.exe` links

What CI does **not** prove (this is the work for this session):
- Window creation actually works (GLFW + Vulkan surface)
- Discrete-GPU selection picks the RTX over an iGPU/MUX
- Compute kernels run end-to-end on real NVIDIA silicon
- Swapchain creation + present timing are sane
- The console servers (HTTP + line) bind on Windows
- The renderer produces the expected image at runtime

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
- The path tracer runs the **manual ray-shape compute kernel**, not
  hardware RT. `VK_KHR_ray_query` / `VK_KHR_acceleration_structure` /
  `VK_KHR_ray_tracing_pipeline` are deliberately NOT wired in this PR
  (queued for the next one)
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

## Out of scope for this PR (queued for follow-ups)

- `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure`
  wiring (real hardware RT on the RTX)
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Engine.h"

#include "CaptureEncoder.h"
#include "CaptureFormat.h"
#include "FrameCapture.h"

#include "../app/ConsoleOverlay.h"
#include "../app/Gamepad.h"
#include "../app/PerfOverlay.h"
#include "../app/Window.h"
#include "../audio/AudioSystem.h"
#include "../console/Console.h"
#include "../console/ConsoleServer.h"
#include "../core/AssetPath.h"
#include "../core/Diag.h"
#include "../core/Hardware/HardwareInfo.h"
#include "../core/Jobs/JobSystem.h"
#include "../core/Log.h"
#include "../core/Memory/Memory.h"
#include "../core/Tracy.h"
#include "../destruction/VoxelGrid.h"
#include "../destruction/Voxelizer.h"
#include "../editor/EditorRoutes.h"
#include "../editor/SceneGraph.h"
#include "../renderer/Astronomy.h"
#include "../renderer/BscCatalog.h"
#include "../renderer/MoonTexture.h"
#include "../renderer/RestirReservoir.h"
#include "../renderer/Camera.h"
#include "../renderer/Csg/CsgScene.h"
#include "../renderer/GltfImporter.h"
#include "../renderer/HdrImage.h"
#include "../renderer/LightTree.h"
#include "../renderer/MeshGen.h"
#include "../effects/ParticleSystem.h"
#include "../physics/PhysicsSystem.h"
#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define TOML_EXCEPTIONS 0
#include <toml++/toml.h>
#include <cstdint>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>      // std::sqrt for cam-teleport detection
#include <cstdlib>
#include <ctime>
#include <numbers>
#include <numeric>
#include <sstream>    // std::istringstream for cam_load slot parsing
#include <thread>
#include <vector>     // std::vector<char> command-line buffer for RestartProcess

// <windows.h> is included only for RecreateWindow's restart-prompt and
// the user-facing MessageBoxW + CreateProcessA + GetModuleFileNameA
// calls. NOMINMAX prevents <windows.h>'s min/max macros from colliding
// with std::min/std::max used elsewhere in this TU.
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
// POSIX needs fork() / execl() / setsid() / access() / kill() for the
// editor panel-spawn / panel-close path. Pulled in unconditionally on
// non-Win32 so the linker doesn't have to chase undefined references
// to the wrappers used inside SpawnPanelBrowser / KillPanelPid.
#  include <signal.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace pt::engine {

namespace {

Engine* g_instance = nullptr;

// ---- CVars (registered statically) ----------------------------------------
namespace cvar {
    using namespace pt::console;
    PT_CVAR(net_port,         "27960", "WebSocket / HTTP port",          CVAR_ARCHIVE);
    PT_CVAR(net_line_port,    "27961", "Plain-text TCP port",            CVAR_ARCHIVE);
    PT_CVAR(net_bind_address, "127.0.0.1", "Bind address; loopback only by default", CVAR_ARCHIVE);
    PT_CVAR(app_window_width,  "1280", "Initial window width",           CVAR_ARCHIVE);
    PT_CVAR(app_window_height, "720",  "Initial window height",          CVAR_ARCHIVE);
    PT_CVAR(app_vsync,         "1",    "Swapchain vsync (1=on)",         CVAR_ARCHIVE);
    PT_CVAR(app_auto_open_console, "0",
            "Open the web console in the default browser at startup. Off by default; use `web_console` command to open on demand.",
            CVAR_ARCHIVE);
    PT_CVAR(app_overlay_enabled, "1",
            "Enable the in-window native console overlay (backtick toggles)", CVAR_ARCHIVE);
    // ---- Editor scaffold (agent-20) ----------------------------------------
    // CSV list of panels to auto-open on engine init, AFTER the
    // WebSocket server is up. Each entry is one of the known panel
    // names (scene-hierarchy / inspector / asset-browser / toolbar).
    // Whitespace around commas is trimmed; unknown names are warned
    // and skipped. Empty string (default) = no panels auto-open.
    PT_CVAR(r_editor_panels_autoopen, "",
            "Comma-separated list of editor panels to open on engine "
            "init. Names: scene-hierarchy, inspector, asset-browser, "
            "toolbar. Example: \"scene-hierarchy,inspector\". Empty "
            "(default) = no auto-open. Panels are spawned as Chrome "
            "--app windows; if Chrome / Edge isn't found, the engine "
            "logs a warning and continues.",
            CVAR_ARCHIVE);
    // Dev mode: when 1, panel_open targets the Vite dev server URL
    // (http://localhost:5173/panels/<name>/) instead of the
    // embedded /editor/<name>. Lets you iterate on React code with
    // HMR while the engine runs. Requires `npm run dev` in
    // web/editor/ to be alive on port 5173.
    PT_CVAR(r_editor_dev_mode, "0",
            "Editor panel dev mode. 0 = panels load from the engine's "
            "embedded /editor/<name> route. 1 = panels load from the "
            "Vite dev server at http://localhost:5173/panels/<name>/. "
            "Start the dev server with `npm run dev` in web/editor/. "
            "Useful for HMR while iterating on panel React code.",
            CVAR_ARCHIVE);
    // ---- end Editor scaffold -----------------------------------------------
    PT_CVAR(pt_smoke_frames, "0",
            "Smoke-test frame budget. >0 = render this many frames then "
            "exit cleanly (exit code 0). 0 = run normally until the user "
            "quits. Typically set via the --smoke-frames=N CLI override "
            "so a developer can run "
            "`demont --smoke-frames=8 --r-backend=metal` on real "
            "hardware to validate that a backend boots + renders + "
            "tears down without crashing -- the exit code (0 success, "
            "2 = backend init failed) plus the log output give a "
            "useful smoke signal without needing a full screenshot "
            "regression matrix. NOT wired into GitHub Actions CI "
            "because the public-tier runners have no GPU (Mac has "
            "only a paravirtualized GPU; Windows-latest has no GPU "
            "at all; the paid `gpu-t4-4-core` is Team/Enterprise-only "
            "and not available to open-source repos). When/if "
            "self-hosted runners on real M-series / RTX hardware "
            "appear, wire this back into build.yml -- the engine side "
            "is ready. NOT CVAR_ARCHIVE -- per-invocation knob, never "
            "persisted to demont.cfg.",
            CVAR_NONE);
    // Golden-image regression matrix support (issue #45). pt_smoke_exec
    // points at a console-script fixture file (e.g.
    // tests/goldens/scenes/cornell_csg.cfg) that's exec'd right after
    // demont.cfg + autoexec.cfg + --<cvar>= CLI overrides but BEFORE
    // the backend boots. Scope at this exec point: cvar writes only --
    // camera pose, sun, path-tracer knobs, denoiser, spp, exposure,
    // capture format/seed, app_window_* (consumed at window creation
    // below). CSG console commands (`csg_*`) and analytic-primitive
    // commands are out of scope here because `csg_scene_` +
    // `SeedDefaultPrimitives()` haven't run yet; see the
    // TODO(#45-followup) at the call site for a planned late-phase
    // exec to widen the supported command set.
    // pt_smoke_capture_out is the destination PNG path for the final
    // frame written at the end of the smoke run; engine does a
    // synchronous readback into this path right before tearing the
    // device down so ctest cells can diff it against a stored golden
    // via the imgdiff CLI (PR #92). Both are per-invocation knobs, not
    // archived to demont.cfg, set via the --smoke-exec= /
    // --smoke-capture-out= CLI overrides plumbed through
    // ApplyCommandLineCvarOverrides. Empty string = disabled.
    PT_CVAR(pt_smoke_exec, "",
            "Path to a console-script .cfg fixture exec'd at engine "
            "start, after demont.cfg + autoexec.cfg + --<cvar>= CLI "
            "overrides. Used by the golden-image regression matrix "
            "(issue #45) to lock down per-cell CVAR state (camera, sun, "
            "path-tracer knobs, exposure, capture seed, app_window_*). "
            "CSG console commands (csg_box / csg_sphere / csg_op / "
            "csg_reset / etc.) are NOT supported at this exec point "
            "today -- csg_scene_ + SeedDefaultPrimitives have not run "
            "yet, and several of those handlers dereference csg_scene_ "
            "before checking it. Typically set via "
            "--smoke-exec=tests/goldens/scenes/cornell_csg.cfg. "
            "Empty = no fixture. NOT CVAR_ARCHIVE -- per-invocation.",
            CVAR_NONE);
    PT_CVAR(pt_smoke_capture_out, "",
            "Destination PNG path for the final smoke-mode frame. Set "
            "via --smoke-capture-out=path. After pt_smoke_frames render "
            "iterations the engine does a synchronous ReadbackTexture "
            "of accum/denoise_color (same source-of-truth picker as "
            "FrameCapture / `screenshot`), tonemaps host-side with "
            "ACES + sRGB OETF, and writes a PNG to this path. Used by "
            "the golden-image regression matrix (issue #45). Empty = "
            "no capture. NOT CVAR_ARCHIVE -- per-invocation.",
            CVAR_NONE);
    // --- SDF Phase 1 (#97) -----------------------------------------------------
    PT_CVAR(pt_smoke_skip_csg_seed, "0",
            "1 = skip the headline drilled-cube CSG seed in Init(). Used "
            "by golden-image fixtures that don't want the default CSG "
            "mesh in frame (e.g. the SDF acceptance scene). Read once "
            "in Init() right before SeedDefaultCsgScene; cvar writes "
            "after first frame have no effect. NOT CVAR_ARCHIVE -- "
            "per-invocation.",
            CVAR_NONE);
    PT_CVAR(pt_smoke_skip_prim_seed, "0",
            "1 = skip the default 3-sphere + ground-plane analytic-prim "
            "seed in Init(). Used by golden-image fixtures that want a "
            "fully-custom scene without the canonical primitives "
            "(e.g. the SDF acceptance scene replaces them with its own "
            "row of analytic spheres). Read once in Init() right before "
            "SeedDefaultPrimitives. NOT CVAR_ARCHIVE -- per-invocation.",
            CVAR_NONE);
    PT_CVAR(pt_smoke_late_exec, "",
            "Path to a console-script .cfg fixture exec'd AFTER "
            "SeedDefaultCsgScene + SeedDefaultPrimitives + console "
            "command registration. Use this for CSG / prim / SDF / "
            "phys_drop_* commands that need csg_scene_ + the seeded "
            "sets + PhysicsSystem to exist first. Same strict-failure "
            "semantics as pt_smoke_exec (read failure / parse error "
            "aborts Init) during the Init-time read. AFTER Init "
            "completes, setting this cvar at runtime (e.g. via "
            "interactive `exec phys_rb_smoke.cfg`) fires an on_change "
            "handler that loads + executes the script immediately -- "
            "so a fixture cfg that uses pt_smoke_late_exec works both "
            "in --smoke-exec=... CLI mode and in interactive console "
            "exec. Re-entrant sets from inside a late-exec script are "
            "ignored with a warning. NOT CVAR_ARCHIVE -- per-invocation.",
            CVAR_NONE);
    // --- end SDF Phase 1 -------------------------------------------------------
    PT_CVAR(con_font_scale, "1.0",
            "Console overlay font scale. 1.0 = baseline (14 logical-unit "
            "CreateFontW height on Win32; 13/12/9 pt input/output/status "
            "NSFont on Mac). Effective range 0.5..3.0; values outside "
            "are clamped at render time. Honored by both the Win32 and "
            "Cocoa overlays.",
            CVAR_ARCHIVE);
    PT_CVAR(r_perf_overlay,    "0",
            "Tiered in-game performance overlay. 0 = off, 1 = fps + frame_ms, "
            "2 = + backend / resolution / GPU memory / spp / bounces / primitives, "
            "3 = + frame-time sparkline. (Tier 4 reserved for per-pass GPU "
            "timestamps once VkQueryPool / MTLCounterSampleBuffer is wired.)",
            CVAR_ARCHIVE);
    PT_CVAR(r_perf_overlay_scale, "1.0",
            "Perf overlay scale. 1.0 = baseline (13 logical-unit "
            "CreateFontW height on Win32; 11 pt monospaced NSFont on Mac). "
            "Both backends scale the 296px panel width, 44px sparkline "
            "height, line height and paddings by the same factor. "
            "Effective range 0.5..3.0; values outside are clamped at "
            "render time. Honored by both the Win32 and Cocoa overlays.",
            CVAR_ARCHIVE);
    PT_CVAR(r_perf_overlay_mode, "native",
            "Backend for the perf overlay. 'native' = OS-native child window "
            "(GDI on Win, NSPanel on Mac) -- has full text readout but can "
            "leave compositing artifacts over Vulkan swapchains. 'rhi' = final "
            "compute pass on the swapchain -- artifact-free, captured in "
            "screenshots, currently visual-only (panel + sparkline, no text).",
            CVAR_ARCHIVE);
    PT_CVAR(r_theme, "hardcore",
            "Web console theme: hardcore|amber|synthwave|matrix|vault|sakura|mono",
            CVAR_ARCHIVE);
    // Issue #162: console smart-resolve. When 1 (default) a typed
    // prefix that uniquely matches one cvar/command is auto-resolved
    // to the canonical name and dispatched, with one info-level log
    // line showing what we picked. Ambiguous prefixes still fall
    // back to the candidate-list error. 0 = strict exact-match
    // (legacy behaviour).
    PT_CVAR(r_console_smart_resolve, "1",
            "Console UX: 1 (default) = auto-resolve a typed prefix to "
            "the unique canonical command/cvar name (e.g. `deno metalfx` "
            "-> `r_denoiser metalfx`) when there's exactly one prefix "
            "match. Ambiguous prefixes still error with the candidate "
            "list. 0 = strict exact-match like before.",
            CVAR_ARCHIVE);
    // Default backend: Metal on Apple Silicon, Vulkan everywhere else
    // (Windows + Linux use Vulkan + RT extensions; the Metal backend
    // doesn't compile off Apple).
#if defined(__APPLE__)
    PT_CVAR(r_backend,         "metal",    "One of none|software|metal|vulkan",CVAR_ARCHIVE);
#else
    PT_CVAR(r_backend,         "vulkan",   "One of none|software|vulkan",      CVAR_ARCHIVE);
#endif
    // Software backend's present path on Windows. Default 'vulkan' uses
    // a minimal VkInstance/VkSurface/VkSwapchain owned by SoftwareDevice
    // so the window stays in DXGI flip-model presentation throughout
    // its lifetime -- the only spec-compliant way to keep displaying
    // after the Vulkan backend ever rendered to the same HWND. 'gdi'
    // uses SetDIBitsToDevice into the HWND's HDC (lower overhead, no
    // Vulkan dependency), but is permanently broken once Vulkan ever
    // touches the window (Microsoft DXGI flip-model lockout, see
    // https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/for-best-performance--use-dxgi-flip-model);
    // safe only for software-fresh-start sessions. The cvar takes
    // effect on the next backend (re)initialise. No-op on Mac/Linux.
    PT_CVAR(r_software_blit,   "vulkan",   "Software backend present path on Windows: vulkan (default; spec-compliant across all backend switches; minimal extra VkInstance) | gdi (legacy SetDIBitsToDevice; broken after vulkan -> software switch per Microsoft DXGI flip-model lockout). No-op on Mac/Linux.", CVAR_ARCHIVE | CVAR_PLATFORM_WIN);
    // r_software_blit_recreate selects the behaviour when the vulkan ->
    // software switch happens with r_software_blit=gdi (the DXGI flip-
    // model lockout scenario; see r_software_blit docs above and the
    // dispatch site in RequestBackendSwitch).
    //   auto   = (default) recreate the GLFW window / HWND in-process
    //            so the new HWND has never been a Vulkan present target
    //            and accepts GDI writes. Preserves engine state (cvars,
    //            console history, CSG scene, camera). Visible as a
    //            brief window flicker at the switch moment.
    //   prompt = pop a MessageBox asking the user whether to restart
    //            the process. Yes spawns a fresh demont.exe with the
    //            original argv and exits the current one; No falls back
    //            to the warn path.
    //   warn   = today's legacy behaviour: emit a LOG_WARN, leave the
    //            window stuck on the last Vulkan frame, require the
    //            user to quit + relaunch manually.
    // No-op on Mac/Linux -- the lockout is Microsoft-DXGI-specific.
    // Takes effect on the next such switch.
    PT_CVAR(r_software_blit_recreate, "auto",
            "Win32 only: behaviour when r_backend switches vulkan -> software with r_software_blit=gdi. auto (default) = in-process HWND recreate so GDI works on the fresh HWND | prompt = MessageBox Restart Now? (Yes spawns fresh process with original argv) | warn = legacy LOG_WARN, manual restart. No-op on Mac/Linux.",
            CVAR_ARCHIVE | CVAR_PLATFORM_WIN);
    // Predictive pipeline JIT prewarming. When 1 (default),
    // Engine::Init (initial device) and Engine::RequestBackendSwitch
    // (every subsequent r_backend change) call
    // Device::EnsurePipelineWarmed for every compute kernel the engine
    // knows about, giving the active backend a chance to start (or
    // confirm completion of) the pipeline build BEFORE the first
    // RenderFrame Dispatch. On backends with an async pipeline-build
    // worker (Vulkan today), this turns "first dispatch waits for the
    // worker" into "the worker starts in parallel with the engine's
    // RegisterCommands / Init tail" so the user-visible loading screen
    // covers the JIT cost rather than a steady-state frame hitch. On
    // backends that build pipelines synchronously at device construction
    // (Metal / software), the call is a no-op by the base class default.
    // Set 0 to disable for A/B perf testing of the prewarm path; affects
    // ONLY backend swap / scene transitions -- steady-state ms/frame is
    // identical with the cvar at 0 or 1.
    PT_CVAR(r_pipeline_prewarm, "1",
            "Predictive pipeline JIT prewarming. 1 (default) = Engine "
            "tells Device which kernels it will dispatch right after "
            "Init / backend swap so the async pipeline-build worker can "
            "race ahead; 0 = let the first Dispatch lazily resolve. No "
            "effect on steady-state perf; reduces or eliminates the "
            "first-frame hitch after r_backend changes.",
            CVAR_ARCHIVE);
    PT_CVAR(r_max_bounces,     "8",  "Max path bounces per ray",          CVAR_ARCHIVE);
    PT_CVAR(r_spp,             "1",  "Samples per pixel per dispatch (>=1). Higher = cleaner motion frames at proportional GPU cost.", CVAR_ARCHIVE);
    PT_CVAR(r_firefly_clamp,   "10",  "Per-contribution firefly clamp (per-channel ceiling on each indirect light contribution: env-NEE, ambient skylight, bounce-to-sky). Suppresses single-sample spikes from BSDF-sampled bounces hitting an HDRI sun pixel, while leaving camera-direct sky unbounded so the sun renders at full intensity. ACES saturates anything above ~5 to ~1.0 for SDR, so 10 preserves visible highlights and kills fireflies. 0 disables.", CVAR_ARCHIVE);
    PT_CVAR(r_quality,         "high",  "Master quality preset that drives r_spp, r_max_bounces, r_caustics, r_refract_bounces, etc. Options: low (fast, no caustics), medium (default-ish), high (caustics, more bounces), ultra (max). 'custom' leaves per-feature cvars as-is.", CVAR_ARCHIVE);
    PT_CVAR(r_caustics,        "1",  "Refractive shadow rays. 1 = NEE rays refract through dielectrics so glass/diamond produce caustic patterns; 0 = treat all dielectrics as opaque shadow blockers (faster, blocks any caustic). Path-tracer-correct in both modes.", CVAR_ARCHIVE);
    PT_CVAR(r_refract_bounces, "4",  "Maximum dielectric refractions a single shadow ray may chain through before giving up (returns no contribution). Higher catches more multi-facet caustics; lower is faster. NOTE: water (#134) shares this budget -- needs >=1 to see refracted underwater geometry.", CVAR_ARCHIVE);
    // ---- Editor backend (agent-19 / PR #201) -------------------------------
    // Master gate for the magenta silhouette outline drawn around the
    // currently-selected analytic primitive. The host folds this gate
    // into PtPush::editor_params.y (and forces .x to 0 when this is 0)
    // so the shader can early-out on a single uint compare. 1 = outline
    // visible when an AnalyticPrim is selected; 0 = no outline regardless
    // of selection. Bit-identical to pre-PR-#201 frames when no selection
    // is set even with the gate on.
    PT_CVAR(r_editor_outline, "1",
            "Magenta silhouette outline for the selected analytic "
            "primitive. 1 = outline drawn at dot(n,-rd) < 0.2 silhouette "
            "edges when a prim is selected; 0 = no outline. Bit-identical "
            "to pre-editor frames when no selection is set.",
            CVAR_ARCHIVE);
    // ---- end Editor backend -------------------------------------------------
    // --- Water Phase 1 (#134): shaded analytic plane with normal-map waves --
    // r_water_* cvars feed PathTrace.slang's MAT_WATER BRDF branch via the
    // water_params0 / water_params1 push fields.  All CVAR_ARCHIVE so a
    // user's tuning persists across sessions.
    PT_CVAR(r_water_absorption_r, "0.45",
            "Per-channel Beer's-law absorption coefficient (1/m) on the RED "
            "channel for MAT_WATER. Higher = red attenuates faster with "
            "depth (more teal water). Default 0.45 matches the classic "
            "Caribbean tint when combined with the green/blue defaults.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_absorption_g, "0.15",
            "Per-channel Beer's-law absorption coefficient (1/m) on the "
            "GREEN channel for MAT_WATER. Default 0.15.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_absorption_b, "0.05",
            "Per-channel Beer's-law absorption coefficient (1/m) on the "
            "BLUE channel for MAT_WATER. Default 0.05 -- blue survives "
            "longest, giving the deep-water blue tint.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_ior, "1.33",
            "Refractive index of water for MAT_WATER. Default 1.33 (real "
            "water at 20C / 589 nm sodium line). Clamped to [1.0, 2.4] in "
            "shader since lower than air is unphysical and higher than "
            "diamond would be nonsense.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_wave_scale, "0.3",
            "Normal-map wave frequency multiplier for MAT_WATER. Default "
            "0.3 (cycles / m) -- ~3 m surface wavelength, calm-lake feel. "
            "Higher = finer ripples, lower = longer swell.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_wave_amplitude, "0.2",
            "Surface-normal perturbation strength for MAT_WATER. Default "
            "0.2 -- subtle ripples without obvious noise patches. The "
            "perturbation acts only on the SHADING normal in Phase 1; "
            "the surface itself is geometrically flat. Phase 2 (FFT) "
            "displaces real geometry.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_wave_speed, "1.0",
            "Wave animation rate multiplier for MAT_WATER. 1.0 is roughly "
            "physical for a calm lake; 0 freezes the surface (useful for "
            "deterministic captures / goldens).",
            CVAR_ARCHIVE);
    // --- end Water Phase 1 ---------------------------------------------------
    // --- Water Phase 4 (#134) deep-water tint --------------------------------
    // Per-ray underwater medium tracking: when a bounce ray (TIR-reflected
    // back down, or any underwater bounce that misses scene geometry)
    // escapes to the empty "deep water" horizon, returning skyColor(rd)
    // would punch a bright sky-blue hole through the underwater rendering.
    // r_water_deep_color_r/g/b is the RGB tint that replaces sky on
    // those misses; defaults to near-black so deep-water voids read as
    // honest darkness rather than tinted-but-bright sky. Throughput
    // (per-segment Beer's law applied along the way) still modulates
    // these values, so a short underwater ray escaping picks up more
    // of the deep tint than a long-path ray that fades to true black.
    PT_CVAR(r_water_deep_color_r, "0.0",
            "Deep-water void colour (RED channel) returned by the shader "
            "when an underwater ray escapes scene geometry to infinity. "
            "Default 0.0 (true black). Multiplied by the ray's running "
            "throughput so per-segment Beer's-law attenuation along the "
            "underwater path still applies. Raise above 0 for a tunable "
            "'water glow' look that mimics deep-water ambient scattering "
            "without a full volumetric integral. Linear RGB units.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_deep_color_g, "0.0",
            "Deep-water void colour (GREEN channel). See "
            "r_water_deep_color_r for full semantics. Default 0.0.",
            CVAR_ARCHIVE);
    PT_CVAR(r_water_deep_color_b, "0.0",
            "Deep-water void colour (BLUE channel). See "
            "r_water_deep_color_r for full semantics. Default 0.0.",
            CVAR_ARCHIVE);
    // --- end Water Phase 4 ---------------------------------------------------
    PT_CVAR(r_denoiser,        "off",
            "Denoiser. off = noisy 1-spp, accumulating image only. "
            "metalfx = Mac MetalFX TemporalDenoisedScaler (Apple Silicon "
            "only). svgf_basic = in-house temporal accumulation only "
            "(no spatial filter, ~1.5 ms at 1080p; cleaner under fast "
            "motion, slightly noisier on disocclusions). Supported on "
            "Vulkan and Metal -- same Slang sources cross-compiled to "
            "each target. svgf_atrous = svgf_basic + 3-pass a-trous "
            "edge-aware spatial filter (~5 ms; cleaner on disocclusions, "
            "mild softening of micro detail). Supported on Vulkan and "
            "Metal. nrd = same dispatch chain as svgf_atrous today on "
            "both backends; reserved for the proper NVIDIA "
            "RayTracingDenoiser library integration once that's wired "
            "up on Vulkan (see Raytracer Plan/FOLLOW_UPS.md). optix_hdr "
            "= NVIDIA OptiX HDR denoiser "
            "via CUDA-Vulkan interop (gated by build-time PT_ENABLE_OPTIX, "
            "auto-detected at configure when CUDA Toolkit + OptiX SDK are "
            "found; runtime gracefully falls back to off if the GPU/driver "
            "isn't OptiX-capable). optix_hdr_aov = OptiX HDR + albedo + "
            "normal AOV guide layers (better edge fidelity on textured "
            "or diffuse-color-rich surfaces; same per-frame characteristic "
            "as optix_hdr -- non-temporal). optix_temporal_hdr = HDR model "
            "with motion-vector flow guide + 1-frame denoised-output "
            "history; closes the per-frame flicker gap vs SVGF on static "
            "scenes while keeping OptiX's no-ghosting motion behaviour. "
            "optix_temporal_hdr_aov = optix_temporal_hdr + albedo+normal "
            "AOV guides; the strongest OptiX variant. "
            "svgf_basic_metalfx / svgf_atrous_metalfx = SVGF on Metal "
            "followed by MetalFX TemporalDenoisedScaler as a finalizer "
            "(ML TAA on top of SVGF-denoised input -- cleanest edge AA "
            "available; Mac only). Mac builds accept "
            "svgf_basic / svgf_atrous / svgf_basic_metalfx / "
            "svgf_atrous_metalfx / nrd / metalfx; they ignore optix_*. "
            "Non-NVIDIA Vulkan builds ignore optix_*; Vulkan builds "
            "ignore metalfx / *_metalfx.",
            CVAR_ARCHIVE);
    PT_CVAR(r_svgf_atrous_passes, "1",
            "Number of A-Trous wavelet passes the in-house SVGF denoiser "
            "runs after the temporal accumulation. The A-Trous structure "
            "keeps the same 5x5 binomial kernel but multiplies its tap "
            "stride per pass (1, 2, 4, 8, 16) -- constant 25 taps per pass, "
            "effective footprint doubles per pass. "
            "1 = 5x5 (default, sharpest). "
            "2 = 9x9 (mild large-scale cleanup). "
            "3 = 17x17. "
            "4 = 33x33 (helps low-freq indirect / soft AO). "
            "5 = 65x65 (canonical SVGF / Schied 2017 paper config). "
            "Clamped to 1..5. Higher passes cost ~1 ms each at 1080p; "
            "only affects r_denoiser svgf_atrous / nrd (svgf_basic skips "
            "the spatial chain entirely).",
            CVAR_ARCHIVE);
    PT_CVAR(r_svgf_albedo_demod, "1",
            "SVGF albedo demodulation (issue #119). 1 = divide the noisy "
            "radiance by the primary-hit albedo on input to the SVGF "
            "chain (temporal + a-trous), denoise the lighting-only "
            "signal so the depth/normal/luminance edge-stops aren't "
            "fighting surface texture detail, and multiply albedo back "
            "on the way out. 0 = legacy pre-demod path (full radiance "
            "through the chain). Affects r_denoiser = svgf_basic / "
            "svgf_atrous / svgf_basic_metalfx / svgf_atrous_metalfx / "
            "nrd; non-SVGF denoisers ignore this flag. Sky pixels "
            "(albedo = 0 sentinel) skip the divide AND the multiply-"
            "back so emissive radiance from the path tracer's miss "
            "shader is forwarded unchanged.",
            CVAR_ARCHIVE);
    PT_CVAR(r_hdr_pipeline,    "1",  "Linear-HDR pipeline through MetalFX. 1 = path tracer writes raw HDR, MetalFX denoises in HDR, post-pass applies exposure+ACES (recommended). 0 = path tracer pre-applies exposure+ACES, MetalFX denoises LDR, tonemap pass is a passthrough copy. Only affects the denoiser-on path.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom,           "1",  "HDR bloom (downsample/upsample pyramid, additive composite before ACES). 0 disables; tonemap then samples a 1x1 zero buffer.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_threshold, "1.0","Linear-HDR luminance threshold for the bloom extract. Pixels below this value contribute nothing to the pyramid; pixels above contribute proportional to (lum - threshold). The path tracer's pixels are in tonemap-relative units (sun ~30, env ~3) so a threshold of 1.0 picks up only HDR highlights.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_intensity, "0.05","Linear blend factor of the bloom layer added on top of the HDR image before tonemap. 0 disables, 1 makes the bloom layer dominate. Realistic camera lens flare is in the 0.02-0.10 range.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_mips,      "5",  "How many mip levels the bloom pyramid uses (1..5). More mips = a softer / wider halo; fewer mips = a tighter glow. Capped to kBloomMips at compile time.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_radius,    "1.0","Per-mip upsample 'spread' multiplier. 1.0 = pixel-accurate dual-filter blur; >1 widens each upsample tap (softer, more diffuse halo); <1 tightens it (sharper core, less spread). Real range 0.5..3.0.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare,      "0",   "Lens flare. Image-based ghost reflections sampled from the bloom layer at mirror-across-centre positions. 0 disables.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_intensity, "0.15", "Linear blend strength of the flare layer. Real-camera lens flare is typically 0.1-0.3 of the bright source.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_dispersion, "0.012", "Per-channel scale offset for chromatic aberration on ghosts. 0 = achromatic (white ghosts), >0 = colourful rainbow fringe along ghost edges. Real lenses 0.01-0.03.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_count,"4",   "Number of ghost reflections to render (1..6). Each ghost has a different scale + colour tint hardcoded in the shader.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_threshold, "0.0", "(image mode only) Per-ghost luminance gate. 0 = no gate. 'sun' mode ignores this and draws clean sun-only flare.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_mode, "physical", "Lens flare algorithm. 'physical' = Hullin paraxial: ghosts come from a real 4-element 50mm lens model, with per-channel chromatic dispersion derived from each glass element's Abbe number. 'sun' = simpler explicit sun-position flare: projects the sun direction to screen and draws soft Gaussian-disc ghosts at its mirrored positions (clean, no scene mirroring). 'image' = legacy image-based flare: mirror-samples the full bloom mip at every output pixel.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_size, "0.04", "(sun mode) Ghost disc radius in fractions of screen *height*. 0.02 = ~22 px on 1080p, 0.08 = ~86 px. Default 0.04 (~43 px) gives discrete circular discs. Aspect-correct -- discs stay round on any aspect ratio.", CVAR_ARCHIVE);
    PT_CVAR(r_debug_sun_overlay, "0", "Debug: when 1, the tonemap pass overlays a green crosshair at the engine-projected sun_uv. Useful for verifying the lens-flare projection lines up with the rendered sun. Sets r_lens_flare_intensity to a sentinel internally; remember to disable when done.", 0);
    PT_CVAR(r_exposure,        "1.5","Manual HDR exposure multiplier applied before ACES tonemap. Used when r_auto_exposure = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_auto_exposure,   "1",  "Auto-exposure: 0 = use r_exposure manual value, 1 = sample accum_hdr each frame and adapt exposure toward r_exposure_target (eye-adaptation feel).", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_min,    "1e-6",  "Minimum exposure scalar that auto-exposure can settle on. Used as a floor; the geometric-mean metering in AutoExposure.slang produces values down to ~1e-5 for genuine outdoor daylight (sky luminance of ~1e4 units / 0.18 target = ~1.8e-5). 1e-6 leaves headroom and prevents NaN pathologies; bump up to ~0.05 if you want auto-exposure to refuse to dim below a certain level for stylistic reasons.", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_max,    "4.0",   "Maximum exposure scalar that auto-exposure can settle on. The reason nights stay dark instead of being boosted to look like day -- bumping this lets the eye adapt further into the dark, lowering it caps the boost.", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_target, "0.18",  "Middle-grey target for auto-exposure. 0.18 matches the Zone-V/Munsell middle-grey convention; lower values aim for a darker overall look.", CVAR_ARCHIVE);
    PT_CVAR(r_eye_adapt_speed, "0.20",  "Per-update interpolation factor for auto-exposure (0..1). Smaller = slower eye adaptation. The update fires every frame on the GPU (single-workgroup reduction over accum_hdr), so 0.20 settles ~80% in 5 frames, ~95% in 16 frames -- roughly 'eye-adapted in a quarter second at 60fps'.", CVAR_ARCHIVE);
    PT_CVAR(r_eye_model,       "human", "Preset 'iris/lens' tuning: human (default), cat (better dim-light dynamic range), owl (nocturnal -- huge max), dslr_iso100 (locked, narrow), dslr_iso6400 (locked, high gain), phone (auto, modest range), linear (no tonemap, debug). Selecting a preset writes r_exposure_min/max/target/r_eye_adapt_speed; 'custom' leaves them as-is.", CVAR_ARCHIVE);
    PT_CVAR(r_env_map,         "assets/hdri/sunset.hdr",
            "Path to a Radiance .hdr environment map. Used when r_sky_mode = hdri. "
            "Default points at the bundled CC0 sunset HDRI; resolves relative to CWD.",
            CVAR_ARCHIVE);
    PT_CVAR(r_env_intensity,   "1.0","Scalar multiplier on env-map samples. Useful for darkening/brightening the IBL without re-authoring the HDRI.", CVAR_ARCHIVE);
    PT_CVAR(r_accum_ema_alpha, "0.0",
            "Exponential moving average for the accumulator. 0 = legacy "
            "online running mean (unbounded convergence, classic path "
            "tracer behaviour). >0 = EMA blend with this history retention "
            "factor in [0, 1) -- avg = prev*alpha + sample*(1-alpha). "
            "Effective sliding window ~1/(1-alpha) frames (0.9 ~10 frames, "
            "0.95 ~20). Lets the accumulator track dynamic content (wind-"
            "drifting clouds, animated lights, slow time-of-day) without "
            "the wet-paint smear that pure running mean produces on "
            "non-stationary input, at the cost of a variance floor "
            "proportional to 1/window. When >0 the engine also skips the "
            "cloud-wind accum-dirty trigger so EMA can actually do its "
            "job. Only meaningful with r_denoiser off; with a denoiser "
            "on the denoiser does its own (smarter, motion-vector-aware) "
            "temporal reuse and this value is ignored.",
            CVAR_ARCHIVE);
    PT_CVAR(r_analytic_bvh_threshold, "16",
            "Minimum finite analytic-primitive count at which the engine "
            "switches the closest-hit search from a linear scan to a "
            "CPU-built binary BVH (uploaded as a storage buffer, traversed "
            "iteratively in the shader). Below this count the per-pixel "
            "linear loop wins on GPUs because BVH traversal overhead "
            "dominates with few primitives. Planes are always linear-"
            "tested (infinite extent excludes them from the BVH); only "
            "finite primitives (spheres today) count toward the threshold. "
            "Set high (e.g. 1024) to force linear-only for A/B; set low "
            "(e.g. 1) to force the BVH path always. Takes effect on next "
            "primitives_dirty rebuild.",
            CVAR_ARCHIVE);
    // --- SDF Phase 1 (#97) -----------------------------------------------------
    PT_CVAR(r_sdf_max_iters,          "128",
            "Maximum sphere-trace iterations per SDF cluster, per ray. "
            "Each iteration evaluates the cluster's distance field once "
            "and advances by that distance. Higher = fewer missed glancing "
            "hits on near-grazing rays at the cost of GPU time. Clamped "
            "to 1..256 in the shader to bound worst-case work. Default 128 "
            "is comfortable for analytic SDFs + smooth-CSG ops (Phase 1); "
            "fractals in Phase 3 of #96 will want this higher.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sdf_epsilon,            "0.0001",
            "Surface-distance terminator for the SDF sphere-trace, in "
            "metres. A step that lands within this distance of the "
            "surface is treated as a hit. Default 1e-4 m (0.1 mm) -- well "
            "below the angular extent of one pixel at typical camera "
            "distances. Lower = sharper silhouettes at the cost of more "
            "iterations near the surface; higher = visibly rounded "
            "silhouettes but cheaper rendering.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sdf_debug_iters,        "0",
            "1 = output iteration heat-map for each SDF cluster instead "
            "of normal shading. Useful for tuning r_sdf_max_iters per "
            "scene. Phase 1 wires the flag through the push struct; the "
            "actual heat-map visualisation lands with Phase 2 procedural "
            "noise tuning.",
            0);
    // --- end SDF Phase 1 -------------------------------------------------------
    // --- SDF Phase 3 (#99) fractal cvars ---------------------------------------
    // Defaults match the issue spec. Three knobs:
    //   r_sdf_fractal_power : default exponent for Mandelbulb (the
    //       textbook polar-power formula). 8 is the canonical bulb.
    //       Mandelbox / Apollonian reuse this slot only as a fallback
    //       when their per-leaf params[0] is zero; in practice each
    //       fractal has its own canonical scale (Mandelbox 2.5,
    //       Apollonian 1.3) and the fixture should set those
    //       explicitly via the sdf_mandelbox / sdf_apollonian
    //       commands.
    //   r_sdf_fractal_iters : per-DE bounded iteration count
    //       (4..16 practical; 12 is a good Mandelbulb default).
    //       Independent of r_sdf_max_iters (which caps sphere-trace
    //       steps, not DE iterations).
    //   r_sdf_de_eps_scale  : multiplies the iteration-relaxed
    //       surface-epsilon (`eps * (1 + scale * step_idx)`) in the
    //       fractal sphere-trace, per the issue's "dist < eps *
    //       scale * iter_count" relaxation. 0 falls back to the
    //       constant-epsilon trace (matches analytic-SDF behaviour);
    //       higher = looser termination = fewer steps wasted in
    //       deep-recursion no-progress areas at the cost of slightly
    //       softer silhouettes far from camera.
    PT_CVAR(r_sdf_fractal_power,      "8",
            "Default polar exponent for the Mandelbulb DE (textbook "
            "power-n formula). 8 is the canonical bulb; 2..16 are "
            "practically interesting. Mandelbox / Apollonian fall "
            "back to this only when their per-leaf params[0] is zero; "
            "each has its own canonical scale (Mandelbox 2.5, "
            "Apollonian 1.3) and the fixture should set those "
            "explicitly via the sdf_mandelbox / sdf_apollonian "
            "commands.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sdf_fractal_iters,      "12",
            "Per-DE bounded iteration count for fractal SDFs "
            "(Mandelbulb / Mandelbox / Apollonian). Independent of "
            "r_sdf_max_iters (which caps sphere-trace STEPS, not the "
            "per-step DE iteration). 4..16 is the practical range; "
            "12 captures fine bulb detail with sane perf. Clamped "
            "to 1..32 in the host.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sdf_de_eps_scale,       "1.0",
            "Iteration-relaxed surface-epsilon scale for fractal "
            "sphere-tracing. The fractal trace terminates when "
            "dist < r_sdf_epsilon * (1 + r_sdf_de_eps_scale * step_idx) "
            "so deep-recursion areas accept a slightly looser hit "
            "rather than burning the full step budget on micro-"
            "progress. 0 falls back to the constant-epsilon trace "
            "(matches analytic-SDF behaviour). Higher = fewer wasted "
            "steps but slightly softer silhouettes far from camera.",
            CVAR_ARCHIVE);
    // --- end SDF Phase 3 -------------------------------------------------------
    // --- Voxel destruction Phase 1 (#140) --------------------------------------
    PT_CVAR(r_voxel_size,             "0.1",
            "Voxel side length in metres for the destruction subsystem. "
            "Default 0.1 m (10 cm chunks) -- the Minecraft / Teardown "
            "reference size. Voxelization uses this value at the moment "
            "`voxelize_object` is invoked; lowering it after a voxel grid "
            "exists does NOT shrink existing voxels (re-run the command). "
            "Maps directly to VoxelGrid::voxel_size on construction; "
            "occupied voxels render as analytic boxes via the existing "
            "SDF cluster path (one cluster per voxel, kSdfShapeBox).",
            CVAR_ARCHIVE);
    PT_CVAR(r_voxelize_demo,          "0",
            "1 = render every voxelized object as its chunky-box pile "
            "INSTEAD of the source CSG mesh (A/B toggle for the "
            "destruction Phase 1 visualization). Mesh / TLAS bindings "
            "are suppressed for the frame so the path tracer sees only "
            "the voxels; analytic primitives and SDF clusters render "
            "alongside as usual. 0 reverts to original mesh rendering "
            "with zero regression -- the voxel SDF clusters are pulled "
            "out of the upload set so the shader doesn't even iterate "
            "them. Idempotent on every frame via SyncVoxelDemoState.",
            CVAR_ARCHIVE);
    // --- end Voxel destruction Phase 1 -----------------------------------------
    // --- SDF Phase 2 (#98) -----------------------------------------------------
    PT_CVAR(r_sdf_normal_mode,        "0",
            "SDF cluster normal evaluation mode: 0 = forward-mode "
            "autodiff (default; one walk of the op-tree, gradient via "
            "Dual3 chain rule, ~1.5x distance-eval cost), 1 = 6-tap "
            "central differences (legacy / sanity check; 6x distance-"
            "eval cost). Only affects clusters that contain Phase 2 "
            "procedural ops (sdf_displace_noise / sdf_twist / sdf_bend "
            "/ sdf_repeat / sdf_repeat_limited) -- Phase 1 (sphere / "
            "box / smin / ...) always uses the closed-form analytic "
            "gradient regardless of this setting. Forward-AD is "
            "continuous in space (no epsilon-jitter that re-rolls per "
            "bounce -- a variance reducer for the PT) and beats central "
            "differences once DE complexity rises.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sdf_displace_octaves,   "4",
            "Default FBM octave count for sdf_displace_noise. Clamped "
            "to 1..6 on the host -- the shader runs a dynamic loop "
            "bounded by min(octaves, 6) (deliberately not unrolled to "
            "keep Metal's XPC compiler from timing out on the chained "
            "SDF cluster walker), and 6 covers the useful inverse-"
            "frequency range from 1 m down to ~3 cm. Higher = more "
            "detail at the cost of N noise lookups per distance eval. "
            "Each octave doubles the frequency and halves the "
            "amplitude. Per-cluster overrides go via the "
            "sdf_displace_noise console command's octaves arg; this "
            "cvar is just the default the command uses when the user "
            "omits the octaves field.",
            CVAR_ARCHIVE);
    // --- end SDF Phase 2 -------------------------------------------------------
    PT_CVAR(r_mis,             "1",
            "Multiple importance sampling for direct lighting on Lambert "
            "hits under HDRI. 1 = balance-heuristic MIS between env-map "
            "NEE and BRDF-sampled bounces (lower variance on rough "
            "Lambert under HDRIs with concentrated bright regions; the "
            "BRDF-sampled bounce hitting the sun is recovered cleanly). "
            "0 = legacy NEE-only path (BRDF-bounce sky-miss is skipped "
            "to avoid double-counting). MIS currently applies only to "
            "Lambert hits; metal (including rough metal via h.roughness) "
            "and dielectric paths are left on their existing code paths "
            "either way.",
            CVAR_ARCHIVE);
    // --- Light tree (#129) ----------------------------------------------------
    PT_CVAR(r_light_tree,      "1",
            "Hierarchical light tree (Conty Estevez & Kulla 2018) for "
            "O(log N) NEE light selection. 1 = traverse the tree to pick "
            "lights weighted by combined cluster importance (intensity * "
            "geometric falloff * emission-cone cosine); 2x-10x variance "
            "reduction at 1 spp for scenes with >100 lights. 0 = fall back "
            "to #73's naive uniform single-pick (regression-protection "
            "toggle; statistically equivalent to the tree at very low light "
            "counts, but much higher variance at high counts). The tree "
            "itself is always built and uploaded; this cvar only flips the "
            "shader-side picker between tree-traversal and uniform-pick, "
            "so the runtime cost of toggling is zero.",
            CVAR_ARCHIVE);
    // --- end Light tree -------------------------------------------------------
    PT_CVAR(r_hdri_extract_percentile, "0.005",
            "Top-luminance percentile threshold (0..0.5) for HDRI light "
            "cluster extraction. Pixels above this percentile are flood-"
            "filled into directional lights for crisp shadows. Default 0.005 "
            "(top 0.5%) catches sun-class peaks on outdoor HDRIs and lamp "
            "cores on interiors. Lower (e.g. 0.002) extracts only the very "
            "brightest -- useful when an interior HDRI is over-extracting "
            "stray bright pixels into spurious lights. Higher (e.g. 0.02) "
            "catches dimmer features but risks merging the sky into the "
            "extracted set. Takes effect on next env-map (re)load: change "
            "this then re-set r_env_map to retrigger extraction.",
            CVAR_ARCHIVE);

    // Procedural sky (Preetham-lite analytic). Used when r_sky_mode is
    // "procedural". The sun position drives both the sky colour gradient
    // and the disk; positive elevation = above horizon (day), negative =
    // below (night); azimuth in degrees, 0 = -Z (north), 90 = +X (east).
    PT_CVAR(r_sky_mode,        "hdri", "Sky rendering: gradient (cheap fallback) | hdri (sample r_env_map) | procedural (analytic with sun position). Defaults to hdri so the bundled sunset.hdr is visible out of the box.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_elevation,   "30.0", "Sun elevation in degrees above horizon (-90..90). Drives day/sunset/night blend in procedural sky. Overridden when r_sky_use_astronomical = 1.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_azimuth,     "135.0","Sun azimuth in degrees (0=north, 90=east, 180=south, 270=west). Overridden when r_sky_use_astronomical = 1.", CVAR_ARCHIVE);

    // Astronomical sky parameters. r_sky_use_astronomical=1 overrides
    // the manual sun cvars with positions computed from observer
    // lat/lon and current UTC time. Default location: Chennai, India.
    PT_CVAR(r_sky_use_astronomical, "0",
            "1 = compute sun position from r_sky_lat/r_sky_lon and current UTC, ignoring r_sun_elevation/r_sun_azimuth. 0 = use manual sun cvars.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sky_lat,         "13.0827", "Observer latitude in degrees (+N). Default: Chennai, India. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_lon,         "80.2707", "Observer longitude in degrees (+E). Default: Chennai, India. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_hour,        "12.0",    "Hour of day (0..24) for the astronomical sun + starmap. Interpreted in UTC if r_sky_hour_local = 0, else in the selected city's local time. r_sky_animate = 1 advances this every frame. Ignored when r_sky_use_astronomical = 0 (manual-sun mode -- the engine emits a one-line warning if this cvar is changed in that mode).", CVAR_ARCHIVE);
    PT_CVAR(r_sky_hour_local,  "1",       "If 1, r_sky_hour is the city's local time (using r_sky_tz_offset_hours, set by r_sky_city). If 0, UTC. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    // Date selector. All three at 0 means "use current date" (default). Set
    // to a real date to back-date the sun, moon, stars, and any seasonal
    // effects. Year 1900-2100 covers any reasonable interactive use; the
    // Meeus formulas degrade slowly past that band but stay within ~1 deg.
    PT_CVAR(r_sky_year,        "0",       "Year for astronomical date (1900..2100). 0 = use today's UTC year. Combined with r_sky_month + r_sky_day to set sun/moon/star positions for any date. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_month,       "0",       "Month for astronomical date (1..12). 0 = use today's month. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_day,         "0",       "Day of month for astronomical date (1..31). 0 = use today's day. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_tz_offset_hours, "5.5", "Selected city's UTC offset in hours (positive east). Updated when r_sky_city changes; you generally don't write this directly. Ignored when r_sky_use_astronomical = 0.", CVAR_READONLY);
    PT_CVAR(r_sky_animate,     "0",       "If 1, advance r_sky_hour every frame at r_sky_animate_rate hours per real-time second. Wraps at 24. Ignored when r_sky_use_astronomical = 0 (Tick skips the per-frame advance entirely; the engine emits a one-line warning if this cvar is changed in manual-sun mode).", CVAR_ARCHIVE);
    PT_CVAR(r_sky_animate_rate,"0.5",     "Hours of sim time per real-time second when r_sky_animate = 1. 0.5 = half-hour/s (a full day in 48s). 24 = compress a day into 1s. 1/3600 ≈ live-time. Ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_city,        "chennai", "Preset observer location. Selecting one writes r_sky_lat / r_sky_lon to that city's coordinates. 'custom' leaves them as-is. lat/lon are ignored when r_sky_use_astronomical = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_show_stars,      "1",       "Render stars at night (sun below horizon). 0 disables.", CVAR_ARCHIVE);
    PT_CVAR(r_stars_mode,      "bsc",     "Star source: 'bsc' = real Yale Bright Star Catalog (J2000-frame map rotated to local horizon per frame, requires assets/stars/BSC5.dat), 'procedural' = hash-based random starfield (fast, no catalog needed). Falls back to procedural when 'bsc' is requested but the catalog failed to load.", CVAR_ARCHIVE);
    PT_CVAR(r_stars_twinkle,   "1",       "Per-star atmospheric scintillation. 0 = static field, 1 = each star modulates +/-45%% at a per-star 1-3 Hz frequency (cheap shader noise, no extra texture lookups). r_stars_twinkle_speed scales the frequency.", CVAR_ARCHIVE);
    // Tune-knob for the twinkle rate. The shader's base per-star
    // frequency is 1-3 Hz (period 0.33-1.0 s); this multiplier scales
    // it -- 0.3 gives 0.3-0.9 Hz (period 1.1-3.3 s, calm-night cadence),
    // 1.0 keeps the base range, >1 speeds it up toward the legacy 3-8 Hz
    // band. Plumbed through exposure_pad.w (the same byte the on/off
    // flag used to occupy); zero still means "twinkle off" so toggling
    // r_stars_twinkle 0 short-circuits.
    PT_CVAR(r_stars_twinkle_speed, "0.3",
            "Multiplier on per-star twinkle frequency. 0.3 (default) "
            "= calm-night cadence at 0.3-0.9 Hz; 1.0 = 1-3 Hz "
            "in-game look; >1 speeds up toward dramatic flicker.",
            CVAR_ARCHIVE);
    // Issue #46: SVGF a-trous smudges sub-pixel stars + erodes the sun
    // and moon discs. When this is on (default), PathTrace.slang
    // subtracts starsOnly + sunDisc + moonDisc from the primary-miss
    // sky term so denoise_color reaches SVGF/MetalFX celestial-free,
    // and the engine's StarsComposite kernel re-adds them post-denoise
    // with N aperture-sampled rays per pixel. Net effect: pixel-sharp
    // stars + a clean single-frame bokeh on the sun/moon (DOF
    // resolves in one frame instead of accumulating over many through
    // the denoiser's temporal stage). Set to 0 to A/B against the
    // legacy behaviour where SVGF gets to smudge celestials.
    PT_CVAR(r_star_split,      "1",
            "Composite stars+sun+moon over the post-denoise HDR "
            "(issue #46). 1 = peel celestials out of the denoiser's "
            "input and re-add them with aperture-sampled bokeh after "
            "Denoise(). 0 = legacy fold-into-sky behaviour, celestials "
            "go through the denoiser. No effect when r_denoiser = off.",
            CVAR_ARCHIVE);
    // Per-pixel aperture sample count for StarsComposite.slang. With
    // DOF off, samples vary only sub-pixel jitter so >1 just smooths
    // star aliasing slightly. With DOF on, each sample picks a
    // different point on the lens aperture, so the bokeh disc on the
    // sun / moon resolves in ONE frame -- compare to the path tracer's
    // 1-spp aperture sample which integrates the bokeh over many SVGF-
    // accumulated frames (and looks noisy in motion). Cost is linear
    // in N: 16 takes ~0.1ms on M4 Max at 1080p (small kernel, no BVH
    // traversal, just 16x starsOnly + sunDisc + moonDisc). 64 is the
    // upper clamp; the disc is over-sampled past ~32 anyway.
    PT_CVAR(r_stars_aperture_samples, "16",
            "StarsComposite per-pixel aperture sample count (1..64). "
            "Higher = smoother sun/moon bokeh with DOF on; "
            "default 16 is the sweet spot.",
            CVAR_ARCHIVE);
    // Aurora borealis procedural overlay (issue #116). Stateless
    // post-denoise composite, runs once per frame on Metal between
    // StarsComposite and the bloom pyramid. Zero performance cost when
    // r_aurora == 0 -- the dispatch is elided engine-side.
    PT_CVAR(r_aurora,           "0",
            "Aurora borealis/australis procedural overlay. 0 = off "
            "(zero cost, dispatch elided), 1 = render FBM-noise "
            "modulated aurora ribbons in the upper sky at night (sun "
            "below horizon). Procedural mode only -- no aurora when "
            "r_sky_mode = hdri / gradient. See "
            "shaders/AuroraComposite.slang for the visual construction.",
            CVAR_ARCHIVE);
    PT_CVAR(r_aurora_intensity, "1.0",
            "Aurora brightness multiplier (0..2). 1.0 = nominal "
            "kilo-rayleigh emission scaled to ~1.5 cd/m^2 in the "
            "engine's HDR-linear units (comparable to half-moon-lit "
            "sky); 2.0 = active geomagnetic-storm brightness.",
            CVAR_ARCHIVE);
    PT_CVAR(r_aurora_lat,       "70",
            "Magnetic-pole latitude in degrees the aurora oval is "
            "anchored to (positive = boreal, default 70 deg matches "
            "the auroral oval around the geomagnetic north pole). "
            "Drives the pole-alignment falloff: rays pointing toward "
            "this elevation get full aurora density, rays away from "
            "it fade smoothly to zero. Reduce toward 30-40 for "
            "lower-latitude observers (cinematic; real aurora is "
            "rarely visible that far south).",
            CVAR_ARCHIVE);
    PT_CVAR(r_aurora_animate,   "1",
            "Time-based shimmer animation. 0 = static curtain "
            "(deterministic frame-to-frame, useful for golden-image "
            "fixtures); 1 = ribbons drift and the fine-noise layer "
            "evolves at ~0.03 rad/s for the 'waves traveling along "
            "the curtain' look.",
            CVAR_ARCHIVE);
    // Issue #115: SVGF a-trous and MetalFX both smudge hard sun shadows
    // on flat surfaces because their edge-stops (depth, normal,
    // luminance) all see ~zero change across a shadow boundary on a
    // single plane. SIGMA-style fix: write per-pixel sun visibility to
    // a separate G-buffer, denoise that signal with a depth+normal
    // bilateral, multiply the denoised visibility into the post-denoise
    // HDR. NVIDIA's NRD ships this under the name SIGMA; this is our
    // hand-rolled Slang implementation, Mac-first. Set to 0 to A/B
    // against the legacy fold-into-radiance behaviour where SVGF /
    // MetalFX get to smudge sun shadows.
    PT_CVAR(r_shadow_demod,    "1",
            "SIGMA-style sun shadow demodulation pass (issue #115). "
            "1 = write per-pixel sun-NEE visibility to a separate "
            "G-buffer, denoise it with a depth+normal bilateral, "
            "multiply the denoised visibility into the post-denoise "
            "HDR for sharp sun shadows under SVGF / MetalFX. "
            "0 = legacy fold-into-radiance behaviour where the active "
            "radiance denoiser smudges sun shadow boundaries. "
            "No effect when r_denoiser = off.",
            CVAR_ARCHIVE);

    // -------------------------------------------------------------------
    // Particle / VFX system (issue #82 MVP).
    //
    // CPU-side simulation, screen-space billboard composite. Particles
    // are NOT visible in path-traced reflections / shadows; the
    // "PT-compatible" headline of #82 is a follow-up (analytic-prim
    // list + BVH refit). GPU-side simulation is also a follow-up.
    // See shaders/ParticleComposite.slang for the kernel rationale
    // and src/effects/ParticleSystem.h for the CPU sim contract.
    // -------------------------------------------------------------------
    PT_CVAR(r_particles, "0",
            "Master toggle for the particle / VFX system composite "
            "dispatch (#82 MVP). 0 disables the GPU composite kernel "
            "(no particles painted into the HDR texture). The CPU sim "
            "tick runs unconditionally so a 0->1 transition doesn't "
            "snap stale particles into view; existing particles age "
            "out naturally while the composite is off. Metal-only "
            "today; Vulkan dispatch is a follow-up.",
            CVAR_ARCHIVE);
    PT_CVAR(r_particles_max, "1024",
            "Cap on simultaneously live particles. Cheap to raise "
            "(per-frame upload is the cost); a few thousand is fine "
            "for the MVP CPU sim, GPU sim follow-up scales further. "
            "Default 1024.",
            CVAR_ARCHIVE);

    // -------------------------------------------------------------------
    // ReSTIR DI Phase A (issue #78). Reservoir-based Spatio-Temporal
    // Importance Resampling for Direct Illumination -- the modern fix
    // for "many lights at 1 spp": replaces "pick one light + shadow ray"
    // with "build a reservoir of K weighted candidates, reuse across
    // screen + time, shadow-test the single survivor". Without it,
    // light primitives (#73) cap out at the dozen-light regime before
    // naive NEE picks the wrong one and noise dominates. With it,
    // hundreds to thousands of lights run cleanly under MetalFX / SVGF
    // / NRD denoising.
    //
    // Phase A scope: DI only (no GI / indirect bounce reservoirs).
    // Architecture: PathTrace.slang's primary-hit Lambert NEE switches
    // to WRS candidate generation when r_restir = 1; three compute
    // passes follow (RestirTemporal, RestirSpatial, RestirFinal) and
    // additively blend the resampled contribution into denoise_color
    // before the radiance denoiser. Gated host-side on
    // `denoiser_active_` -- the denoiser-off path keeps single-pick
    // NEE (the accumulator handles many-light variance fine at high
    // spp; ReSTIR is primarily a 1-spp / denoiser-input quality lever).
    // -------------------------------------------------------------------
    PT_CVAR(r_restir, "1",
            "Master toggle for ReSTIR DI Phase A (issue #78). 1 = "
            "primary-hit Lambert NEE switches to WRS candidate "
            "generation + reservoir spatiotemporal reuse + shadow "
            "test on the single survivor; 0 = legacy uniform "
            "single-pick NEE from issue #73. Only takes effect when "
            "the denoiser is active (the Phase A composite writes "
            "into denoise_color). Default 1: ReSTIR is the new "
            "many-lights baseline.",
            CVAR_ARCHIVE);
    PT_CVAR(r_restir_temporal, "1",
            "ReSTIR temporal reuse pass gate. 1 = reproject the "
            "previous frame's final reservoir via motion vectors and "
            "MIS-combine with this frame's WRS reservoir (Bitterli "
            "Algorithm 4). 0 = skip temporal reuse, current-frame "
            "reservoir flows straight to the spatial pass. Turning "
            "this off increases per-frame noise but eliminates the "
            "rare temporal smear under fast camera motion.",
            CVAR_ARCHIVE);
    PT_CVAR(r_restir_spatial, "1",
            "ReSTIR spatial reuse pass gate. 1 = sample N=4 jittered "
            "neighbours in a 10 px disc and MIS-combine with the "
            "centre reservoir. 0 = skip the spatial pass; survivor "
            "depends only on the K initial candidates + temporal "
            "history. The spatial pass is the dominant variance-"
            "reduction step for static cameras; turning it off is "
            "mainly a diagnostic for chasing artifacts.",
            CVAR_ARCHIVE);
    PT_CVAR(r_restir_k_candidates, "8",
            "Number of initial WRS candidate samples per pixel "
            "(Bitterli K). 8 is the NVIDIA RTXDI 'balanced' default "
            "and trades well against temporal / spatial reuse. Higher "
            "K reduces per-frame variance at linear ALU cost; the "
            "shader caps at 64. Sub-1 values are clamped to 1.",
            CVAR_ARCHIVE);
    PT_CVAR(r_restir_bias, "biased",
            "ReSTIR bias-correction mode. 'biased' (default) = "
            "straight WRS resampling; cheaper (~30% off the spatial "
            "pass cost) and visually masked by the radiance denoiser. "
            "'unbiased' = per-pair MIS weighting in the spatial "
            "merge, Bitterli's rigorous estimator. 'unbiased' is the "
            "correct ground-truth setting for offline reference; "
            "'biased' is what AAA shipped engines use at runtime.",
            CVAR_ARCHIVE);

    // Camera controls.  Mouse-look engages while RIGHT mouse is held.
    PT_CVAR(cam_speed,         "3.0", "Movement speed (units/sec)",        CVAR_ARCHIVE);
    PT_CVAR(cam_sprint_mult,   "3.0", "Speed multiplier with shift",       CVAR_ARCHIVE);
    PT_CVAR(cam_sensitivity,   "0.12","Mouse-look sensitivity (deg/pixel)", CVAR_ARCHIVE);
    PT_CVAR(cam_fov,           "60.0","Vertical field of view (degrees)",  CVAR_ARCHIVE);
    PT_CVAR(cam_pos,           "0 1.5 4", "Camera position (x y z)",       CVAR_ARCHIVE);
    PT_CVAR(cam_yaw,           "0",       "Yaw in degrees",                CVAR_ARCHIVE);
    PT_CVAR(cam_pitch,         "-11.5",   "Pitch in degrees (clamped +/- 85)", CVAR_ARCHIVE);
    // Gamepad input (#83). MVP: left stick = translate (forward / strafe),
    // right stick = look (yaw / pitch), triggers = sprint (analog blend
    // 0..1, additive with shift-key sprint). GLFW's gamepad API wraps
    // SDL2's controller DB so Xbox / DualSense / Switch Pro all work
    // out of the box. No haptics / rumble in MVP -- GLFW doesn't expose
    // them; would need SDL2 or HIDAPI as a follow-up.
    PT_CVAR(cam_gamepad,                  "1",
            "Gamepad input master toggle. 1 = poll slot 0 every frame "
            "and feed its left/right sticks + triggers into the camera; "
            "0 = ignore the pad even if connected. Hot-pluggable -- "
            "no engine restart needed.",
            CVAR_ARCHIVE);
    PT_CVAR(cam_gamepad_deadzone,         "0.15",
            "Radial deadzone on both sticks in [0, 1] of the native "
            "stick magnitude. Sticks inside this radius read as 0; "
            "outside it the surviving magnitude is rescaled to [0, 1] "
            "so there's no snap at the boundary. 0.15 covers the "
            "typical Xbox / DualSense rest-position jitter; raise to "
            "0.25 for worn-out sticks, lower toward 0.05 for precision "
            "aiming.",
            CVAR_ARCHIVE);
    PT_CVAR(cam_gamepad_look_sensitivity, "2.0",
            "Right-stick look sensitivity in radians/second at full "
            "deflection (post-deadzone magnitude == 1). At 2.0 a full-"
            "tilt right stick yaws ~115 deg/sec; raise for snappier "
            "look, lower for cinematic / precision pans. Independent "
            "of cam_sensitivity (which is mouse degrees/pixel).",
            CVAR_ARCHIVE);
    // Camera-state slots for the cam_save / cam_load / cam_list / cam_reset
    // commands (registered in RegisterCommands). Slots 1..9 are user-
    // savable via `cam_save [N]`; slot 0 is the engineering default
    // (hardcoded inside cam_reset, no cvar). Format string is
    // "x y z yaw_deg pitch_deg fov_deg" -- same space-separated
    // convention cam_pos uses, six floats. Empty string = unsaved.
    // CVAR_ARCHIVE so saved viewpoints persist across runs.
    // Format: "x y z yaw_deg pitch_deg fov_deg" -- six floats, all
    // angles in DEGREES (Camera stores yaw/pitch as radians; the
    // cam_save / cam_load helpers handle the conversion).
    PT_CVAR(cam_slot_1, "", "Saved camera state, slot 1: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_2, "", "Saved camera state, slot 2: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_3, "", "Saved camera state, slot 3: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_4, "", "Saved camera state, slot 4: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_5, "", "Saved camera state, slot 5: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_6, "", "Saved camera state, slot 6: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_7, "", "Saved camera state, slot 7: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_8, "", "Saved camera state, slot 8: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_slot_9, "", "Saved camera state, slot 9: \"x y z yaw_deg pitch_deg fov_deg\"", CVAR_ARCHIVE);
    PT_CVAR(cam_teleport_threshold, "5.0",
            "Per-frame position-delta threshold (in world units) for "
            "auto-firing the active denoiser's reset_history flag. "
            "Larger jumps (e.g. user typed a coordinate in console, or "
            "moved through a sphere whose inside is black) clear the "
            "current denoiser's history -- consumed by SVGF/NRD's "
            "temporal accumulation path, MetalFX's history-state "
            "reset, AND the OptiX temporal denoisers' "
            "prev_output / internal-guide-layer ping-pong -- instead "
            "of bleeding stale pre-jump color into post-jump frames. "
            "Default 5 = ~33x typical WASD per-frame movement at "
            "cam_speed=3, 60fps. 0 = disabled.",
            CVAR_ARCHIVE);
    // Depth-of-field (thin-lens camera). Each primary ray originates
    // at a random sample on the aperture and aims through the focal-
    // plane intersection of the pinhole ray. Bokeh shape comes from
    // the aperture sampling distribution (round disk vs polygon).
    PT_CVAR(r_dof,             "0",   "Depth of field. 0 = pinhole camera (everything sharp). 1 = thin-lens with r_dof_aperture / r_dof_focal_distance.", CVAR_ARCHIVE);
    PT_CVAR(r_dof_aperture,    "0.05","Aperture radius in world units. Bigger = more blur on out-of-focus pixels. Real-camera analogue: focal_length / f_number; e.g. 50mm at f/2.8 ~= 0.018 (assuming the scene is in metres).", CVAR_ARCHIVE);
    PT_CVAR(r_dof_focal_distance, "5.0", "Distance from camera (world units) where the scene is in perfect focus. Closer / farther pixels get bokeh proportional to their distance from this plane.", CVAR_ARCHIVE);
    PT_CVAR(r_dof_blades,      "0",   "Aperture blade count. 0 = perfectly round disk (circular bokeh). 5/6/8 = polygonal iris (matching real lens aperture blades) -- gives polygonal bokeh on out-of-focus highlights.", CVAR_ARCHIVE);
    // Volumetrics: single-scatter ray march along primary rays, NEE
    // toward sun at each sample. Atmospheric haze + god rays through
    // gaps in geometry (sun shafts).
    PT_CVAR(r_volumetric,           "0",    "Volumetric single-scatter (atmospheric haze + sun shafts). 0 disables; otherwise march r_volumetric_samples points along each primary ray and NEE-shadow-test toward the sun at each.", CVAR_ARCHIVE);
    PT_CVAR(r_volumetric_density,   "0.002", "Mie extinction coefficient (per metre). Real Earth aerosol haze is 1e-4 to 5e-4 at sea level; 0.002 gives visible god rays paired with r_rayleigh=30 for blue sky. 0.005 = thick haze (visible god rays, partial blue-sky tint loss). 0.05 = fog (sky goes white). Use Rayleigh (r_rayleigh) for the blue sky -- this is the haze layer on top.", CVAR_ARCHIVE);
    PT_CVAR(r_volumetric_anisotropy,"0.7",  "Henyey-Greenstein phase g in [-0.95, 0.95]. +0.7 = forward-peaked atmosphere, 0 = isotropic fog, negative = back-scattering. Higher g makes the sun's halo much brighter when the camera looks near it.", CVAR_ARCHIVE);
    PT_CVAR(r_volumetric_intensity, "1.0",  "Linear scale on the volumetric contribution. Useful to dial god rays up/down without changing the underlying density.", CVAR_ARCHIVE);
    PT_CVAR(r_volumetric_samples,   "16",   "March sample count per primary ray (4..64). More = smoother shafts at proportional GPU cost. 16 is a comfortable default with the denoiser on.", CVAR_ARCHIVE);

    // Phase 4 (#100) heterogeneous volumetric raymarching cvars. These
    // affect the CLOUD march specifically (vol_density_scale, vol_phase_g,
    // vol_multiscatter_bounces); the atmospheric Mie/Rayleigh haze march
    // keeps its own r_volumetric_* cvars above. Naming with r_vol_ matches
    // the issue spec exactly and keeps the haze knobs (r_volumetric_*)
    // separate so users can tune them independently.
    PT_CVAR(r_vol_density_scale,        "1.0",  "Cloud volumetric density scale (Phase 4, #100). Linear multiplier on the per-step cloud sigma_t inside the cloud march. 0.5 = half-thickness clouds; 2.0 = double-thickness. Carries through to shadow rays so the cloud's Beer-Lambert occlusion stays consistent with the body it casts shadows from. Range: clamped to >=1e-3 shader-side (a literal 0 would render the cloud as a transmissive haze, which is what r_clouds=0 means instead).", CVAR_ARCHIVE);
    PT_CVAR(r_vol_phase_g,              "0.8",  "Cloud Henyey-Greenstein anisotropy g (Phase 4, #100). Decoupled from r_volumetric_anisotropy (which controls the atmospheric haze). +0.8 = canonical cumulus forward-scatter lobe (silver lining around the sun edge); 0 = isotropic; negative = back-scatter. Range [-0.95, 0.95] clamped shader-side; outside that the HG denominator pinches to zero on grazing directions and produces firefly spikes.", CVAR_ARCHIVE);
    PT_CVAR(r_vol_multiscatter_bounces, "2",    "Cloud multi-scatter octaves (Phase 4, #100). Wrenninge-style cheap proxy for diffuse light bouncing several times inside the cloud volume before exiting toward the camera. 0 = pure single-scatter (legacy behaviour; cloud cores under self-shadow read pitch-black); 1-4 = N additional octaves of decaying-extinction (a=0.5 per octave) decaying-anisotropy (b=0.5 per octave) fake bounces. Default 2 matches the issue spec and gives cumulus the bright diffuse fill expected. Clamped to 0..4 shader-side.", CVAR_ARCHIVE);
    PT_CVAR(r_cloud_attenuate_background, "0",
            "Apply Beer-Lambert attenuation of the sky / background "
            "radiance through the cloud + smoke volume. 0 (default, "
            "back-compat with all Wave-1 cloud goldens) = clouds and "
            "smoke only ADD in-scattering on top of an unchanged sky; "
            "dense clouds never appear darker than the sky behind. "
            "1 = the running transmittance product through the volume "
            "(trans_eye_c at line ~4747 of PathTrace.slang -- already "
            "computed for in-scatter weighting and the G-buffer write) "
            "is also multiplied into the existing radiance, so a dense "
            "cloud / smoke plume actually OCCLUDES the bright sky "
            "behind it. This is what 'real smoke darkens the sky' "
            "expects. Flipping ON requires regen of every cloud golden "
            "in the regression matrix; the default is OFF until that "
            "regen lands as part of the Wave-2 smoke/cloud visual "
            "overhaul (see issue #179).", CVAR_ARCHIVE);

    // Volumetric clouds. Layer on top of homogeneous haze: when r_clouds
    // is on, the volumetric march multiplies sigma_t by a position-
    // sampled cloud density (procedural fbm value-noise). Presets
    // (`r_clouds_preset`) snap the individual params to known-good
    // weather looks; the per-day seed (`r_clouds_seed`) shifts the
    // noise hash so the same preset can produce visually distinct
    // patterns each day.
    PT_CVAR(r_clouds,                "0",        "Volumetric clouds (real-meteorology altitudes, 1 unit = 1 metre). Activates the volumetric march even if r_volumetric is off. Cumulus 200-500m, stratus 100-300m, cirrus 6000-12000m. The march clips to the cloud altitude band, so distant low-elevation rays still hit the layer when the sky-pixel march range is large enough.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_preset,         "cumulus",  "Cloud preset name. One of clear|cumulus|stratus|cirrus|overcast|storm|custom. Setting this snaps r_clouds_coverage / _base / _top / _density / _freq / _detail to real meteorological values; 'custom' leaves them alone for manual tuning.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_coverage,       "0.45",     "Sky coverage fraction [0..1]. 0 = clear, 1 = full overcast. Subtracted from the noise threshold so only high-noise regions become cloud.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_base_height,    "200.0",    "Cloud layer bottom altitude in metres (1 unit = 1m). Cumulus 200-500m, stratus 100-300m, cirrus 6000-12000m. Real meteorology, not engine-scaled.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_top_height,     "500.0",    "Cloud layer top altitude in metres.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_density,        "0.06",     "Peak extinction inside the cloud, in per-metre sigma_t. Real meteorology: light cumulus 0.03-0.05, typical cumulus 0.05-0.10, stratus 0.04-0.08, storm/cumulonimbus 0.10-0.30. Across a 300m cumulus layer, sigma=0.05 gives optical depth ~15 -- mostly opaque core, translucent edges.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_freq,           "0.005",    "Noise frequency in cycles per metre. 0.003 -> ~330m features (slow undulating cumulus), 0.01 -> ~100m features (smaller puffy cumulus). Match to typical horizontal cloud size, not vertical layer thickness.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_detail,         "0.35",     "High-frequency detail amount [0..1]. 0 = soft blobby clouds, 1 = wispy/eroded edges.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_curl_amount,    "0.0",      "Curl-noise displacement magnitude in metres applied to the cloud sample position before density evaluation (Bridson 2007, 'Curl-Noise for Procedural Fluid Flow'). Divergence-free, so it shears the cloud body into filamentous eddies without inflating its volume. 0 disables (default; output bit-equivalent to pre-#117 main so existing CVAR_ARCHIVE scenes that turn on r_clouds keep the old look until the user opts in); 0.3 gives subtle cumulus edge wisps; 0.8+ gives heavily turbulent / sheared-streamer cirrus. Layer-relative magnitude in metres -- scale with r_clouds_freq if you change the bulk feature size.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_curl_scale,     "0.01",     "Curl-noise frequency in cycles per metre. 0.01 = ~100m turbulent eddies (cumulus edge filaments); 0.003 = ~330m large-scale shear (cirrus streamers); 0.03 = ~33m fine wisps (close-up shots, sub-cloud detail). Independent of r_clouds_freq so coarse bulk cloud bodies can still carry fine curl turbulence.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_erosion,        "0.0",      "Secondary high-frequency edge-erosion amount [0..1] layered on top of r_clouds_detail. Eats away soft cloud margins into sub-feature wisps. r_clouds_detail controls the bulk wispy boundary; r_clouds_erosion paints the filamentous fringe on top. 0 = identical to pre-#117 main; 0.3-0.6 gives the characteristic frayed look of close-up cumulus.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_wind_x,         "5.0",      "Wind speed along +X in metres/second. Drifts the cloud field over time. Light breeze 2-3, fresh wind 8-12, gale 20+.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_wind_z,         "0.0",      "Wind speed along +Z in metres/second.", CVAR_ARCHIVE);
    PT_CVAR(r_clouds_seed,           "0",        "Per-day noise seed (any float). Same preset + different seed = visually distinct cloud pattern. Use one seed per in-game day so each day has its own weather pattern.", CVAR_ARCHIVE);
    // --- Fluid Phase 1 (#136) -- smoke emitters --------------------------------
    // Cheap density-injection plumes that ride the existing cloud march.
    // r_smoke_enabled gates the shader-side loop entirely (0 = bit-exact
    // versus pre-#136 main); r_smoke_max_emitters tightens the GPU-side
    // iteration cap below kMaxSmokeEmitters for situations where the
    // user wants to dial back per-frame cost without rebuilding the
    // emitter list. Both ARCHIVE so the preferred config persists.
    PT_CVAR(r_smoke_enabled,         "0",
            "Density-injection smoke emitters (fluid Phase 1, #136). When 1, "
            "every active emitter contributes additive Gaussian-falloff "
            "density inside cloud_density_at() -- the existing cloud march "
            "scatters / shades the plume identically to cloud cells. 0 is "
            "bit-exact vs pre-#136 main (the shader-side loop elides "
            "entirely). Plumes drift parametrically as base + velocity * t; "
            "no real fluid simulation (Phase 2+).", CVAR_ARCHIVE);
    PT_CVAR(r_smoke_max_emitters,    "8",
            "Per-frame cap on smoke emitters processed inside cloud_density_at "
            "(#136). Hard-clamped on the host to [0, kMaxSmokeEmitters=16]. "
            "Even when the engine-side `smoke_emit` list is longer than this, "
            "only the first N entries get pushed to GPU each frame. Use to "
            "dial back per-pixel cost on lower-tier GPUs without dropping "
            "specific emitters from the list.", CVAR_ARCHIVE);
    // --- Fluid Phase 2 (#180) -- advected emission --------------------------
    // Phase 2 turns each emitter from a single drifting Gaussian into a
    // CHAIN of P time-staggered "puffs" -- the head sits at the emitter
    // mouth (age=0), the tail at age=r_smoke_lifetime. Puffs age, advect
    // along the emitter's velocity scaled by r_smoke_advection_strength,
    // expand radially via r_smoke_expansion_rate, and fade their density
    // linearly into the lifetime so the column dissipates rather than
    // translating rigidly. Still parametric (no solver state); the visible
    // win over Phase 1 is that smoke now looks like rising / drifting
    // smoke instead of a moving blob.
    PT_CVAR(r_smoke_lifetime,          "4.0",
            "Smoke puff lifetime in seconds (#180). Each emitter spawns a "
            "chain of r_smoke_puff_count puffs spaced evenly across this "
            "lifetime; density fades linearly to zero at age=lifetime. "
            "Larger values = taller, longer-lasting columns; tune together "
            "with r_smoke_advection_strength (which sets how fast the column "
            "stretches per second of lifetime). 0 collapses to the Phase 1 "
            "single-blob model (only the head puff renders).", CVAR_ARCHIVE);
    PT_CVAR(r_smoke_advection_strength, "1.0",
            "Multiplier on each emitter's velocity that controls how far "
            "older puffs drift from the source (#180). 1.0 = use the "
            "emitter's velocity verbatim (older puffs at age t sit at "
            "base + velocity * t). 0 = puffs stack at the source (steam-jet "
            "feel); 2.0+ = stretched plume. Phase 2 has no turbulence -- "
            "the column is a straight chain along the velocity vector.",
            CVAR_ARCHIVE);
    PT_CVAR(r_smoke_puff_count,        "12",
            "Number of staggered puffs per emitter (#180). Each puff is a "
            "Gaussian at a different age along [0, r_smoke_lifetime]; more "
            "puffs = smoother column at the cost of P density evaluations "
            "per cloud-march sample per emitter. Hard-clamped to [1, 16] in "
            "the shader so the inner loop bound is compile-time-known. "
            "Wave-6 bumped the cap from 8 to 16 and the default from 6 to "
            "12 so the chain reads as a continuous column out-of-the-box "
            "rather than a string of discrete blobs. 1 mimics Phase 1's "
            "single-blob shape.", CVAR_ARCHIVE);
    PT_CVAR(r_smoke_expansion_rate,    "0.15",
            "Per-second radial-expansion rate for an aging smoke puff "
            "(#180). At age t the effective radius is r * (1 + rate * t), "
            "modelling the turbulent-diffusion broadening of a real rising "
            "plume. 0 = constant-radius puffs (column has uniform width); "
            "0.1-0.3 reads as fluffy smoke; >0.5 puffs balloon visibly. "
            "Stacks with the per-emitter base `radius`.", CVAR_ARCHIVE);
    // --- end Fluid Phase 2 ---------------------------------------------------
    // --- Fluid wave-6 -- "looks like smoke" pass --------------------------
    // Visual fidelity knobs added so smoke_emitter_basic.cfg renders as
    // recognisable smoke (turbulence detail, exponential decay tail,
    // age-driven tint) instead of a smooth puff-chain blob. All four
    // defaults are tuned for the cigarette-column reference scene so
    // `exec smoke_emitter_basic.cfg` reads like smoke out-of-the-box.
    // 0 values collapse each contribution to a no-op, so a user can
    // dial back the wave-6 character without recompiling.
    PT_CVAR(r_smoke_turbulence_scale,   "4.0",
            "Smoke turbulence noise frequency in cycles per metre "
            "(wave-6). Modulates the per-puff Gaussian density by a "
            "3-octave fbm so the column reads as wispy turbulent smoke "
            "rather than a smooth blob. Low values (1-2) = large soft "
            "wisps; default 4 = fine high-frequency detail; 8+ = very "
            "fine sub-feature wisps (e.g. close-up cigarette smoke). "
            "Pairs with r_smoke_turbulence_amplitude.", CVAR_ARCHIVE);
    PT_CVAR(r_smoke_turbulence_amplitude, "0.6",
            "Strength of the turbulence noise modulation on smoke density "
            "(wave-6). 0 = no modulation, bit-identical to Phase 2's "
            "smooth Gaussian puff. 1 = density fully replaced by the "
            "noise pattern (very wispy, may hollow out the column). "
            "0.4-0.7 reads as natural fluffy smoke with visible internal "
            "structure. The noise is sampled in puff-local coords so the "
            "wisp pattern moves with the puff as it advects.",
            CVAR_ARCHIVE);
    PT_CVAR(r_smoke_decay_shape,         "2.0",
            "Exponent on the per-puff age-fade curve (wave-6). Density "
            "at age t = peak * pow(1 - t / lifetime, exp). Exp=1 "
            "recovers Phase 2's linear fade bit-exact (slow tail-off). "
            "Exp=2 (default) keeps the head puff denser and lets the "
            "tail wisp away faster -- the characteristic 'fresh puff, "
            "vanishing tail' shape of real smoke columns. Exp=3-4 "
            "sharpens that further for short-lived puffs (e.g. steam "
            "jets). Hard-clamped to >=1.", CVAR_ARCHIVE);
    PT_CVAR(r_smoke_age_tint_strength,   "0.2",
            "Amount of age-driven warm-to-cool tint applied to each "
            "smoke puff (wave-6). Fresh puffs get a faintly warm yellow "
            "cast, older puffs drift toward a cool grey-blue. 0 = no "
            "age tint (puff colour == per-emitter tint, Phase 2 "
            "behaviour). 0.2 (default) reads as natural cigarette/wood "
            "smoke. 1.0 maxes the warm/cool contrast for stylized "
            "smoke. Stacks multiplicatively with the per-emitter tint "
            "(v2.xyz on the smoke SSBO).", CVAR_ARCHIVE);
    // --- end Fluid wave-6 -----------------------------------------------------
    // --- end Fluid Phase 1 -----------------------------------------------------
    // --- Motion blur (#85) ----------------------------------------------------
    // Cook-1984 style stochastic shutter-time sampling. Each primary ray
    // picks a uniform random t in [0, shutter_speed]; sphere primitives
    // lerp between prev_pos_or_n and pos_or_n at t/shutter_speed. The
    // engine snapshots curr -> prev once per StepPhysics tick so the
    // delta encodes one frame's displacement.
    //
    // r_motion_blur 0 is bit-identical to pre-#85 main: the shader's
    // sphere position read collapses to a plain pos_or_n (lerp factor
    // forced to 1.0 even with motion-blur off-path, so the prev_pos
    // packed into the primitive buffer is simply ignored).
    PT_CVAR(r_motion_blur,             "0",
            "Per-ray stochastic shutter-time sampling for motion blur "
            "(Cook 1984). 0 = off, bit-identical to pre-#85 (no shutter "
            "jitter, sphere positions read straight from pos_or_n). 1 = "
            "each primary ray picks t ~ U(0, r_shutter_speed) and the "
            "scene is queried at that shutter time. Sphere primitives "
            "lerp between prev_pos_or_n (last frame's position) and "
            "pos_or_n (current) by t / shutter_speed -- visible streak "
            "length scales with per-frame body displacement. Combined "
            "with multi-spp accumulation this gives temporal anti-"
            "aliasing for free. Planes don't move so they're unaffected.",
            CVAR_ARCHIVE);
    PT_CVAR(r_shutter_speed,           "0.0167",
            "Shutter open duration in seconds (#85) for motion blur "
            "sampling. Default 1/60 s = 0.0167 s matches a 60 fps render "
            "loop with a 'full' shutter -- a body moving 5 m/s shows a "
            "streak ~8 cm long. 1/240 s = 0.0042 s gives a tighter "
            "shutter (less smear, sharper motion); 1/30 s = 0.0333 s "
            "doubles the smear for cinematic blur. Real cinema shutters "
            "are typically 180 degrees of frame time (1/48 s at 24 fps); "
            "matching that on a 60 fps render means 1/120 s = 0.0083 s. "
            "Only consulted when r_motion_blur == 1.", CVAR_ARCHIVE);
    // --- end Motion blur ------------------------------------------------------
    PT_CVAR(r_rayleigh,              "30.0",     "Atmospheric Rayleigh scattering scale on the per-channel sea-level sigma (R 5.8e-6, G 13.5e-6, B 33.1e-6 per metre). 1.0 = real Earth atmosphere -- but our typical r_volumetric_density (Mie haze) is ~30x stronger than real-Earth haze, so bumping this to 30 keeps the sky visibly blue at typical haze settings. Drop to 1.0 if you also drop r_volumetric_density to 0.0001-0.0005 (real haze). 0 disables Rayleigh.", CVAR_ARCHIVE);
    PT_CVAR(r_planet_radius,         "6378137.0", "Planet radius in metres for spherical-Earth atmospheric scattering (issue #51). Default 6,378,137 m = WGS-84 equatorial Earth radius. The path tracer's `atmosphericTransmittance` numerically integrates Mie + Rayleigh optical depth along a chord through a thin shell around a sphere of this radius (centre at world origin + offset so y=0 sits on the surface). Set to 0 to fall back to the legacy planar exponential integral (1/sin(elev) airmass) -- useful as a debug A/B or for tiny-scene tests where curvature is invisible. Real values for other bodies: Moon 1,737,400, Mars 3,389,500, Venus 6,051,800. Affects only the atmosphere integral, not collision / shadow geometry.", CVAR_ARCHIVE);
    PT_CVAR(r_moon_size,             "1.0",      "Moon angular-size multiplier. 1.0 = our default 0.55deg half-angle (already 2x the real 0.27deg, for visibility at typical 60-FOV 1080p). 5+ = dramatic 'big moon' shots; 0.5 = real lunar size (very small). Astronomical distance variation (perigee/apogee) is also applied on top -- supermoons render ~14% bigger than micro-moons.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_size,              "1.0",      "Sun angular-size multiplier. 1.0 = real ~0.55deg half-angle. Astronomical Earth-Sun distance (perihelion/aphelion) modulates this ~3.4% across the year. Bump for cinematic shots.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_horizon_flatten,   "1",        "Atmospheric refraction differentially lifts the sun's lower limb more than its upper limb, vertically squishing the disc into an oval as it nears the horizon (Saemundsson 1986). 1 = physical flatten enabled (vertical-scale ~0.78 at elev=0 / ~21% squish, ~0.87 at 1deg, ~0.97 at 5deg, ~0.99 at 10deg); 0 = render a perfect circle regardless of elevation. Horizontal radius is unchanged either way; r_sun_size stacks on top.", CVAR_ARCHIVE);
    PT_CVAR(r_moon_brightness,       "0.7",      "Moon disc brightness multiplier. 1.0 = the post-ACES-tuned default; 0.7 reads as a softer night-sky moon where surface texture stays under the ACES knee. Drop further for a darker moon, raise toward 1.5+ for an artificial bright-moon look.", CVAR_ARCHIVE);

    PT_CVAR(dev_cheats,        "0",    "Gate for CHEAT-flagged cvars",   0);
    PT_CVAR(dev_log_level,     "info", "error|warn|info|debug",          0);
    PT_CVAR(r_diagnostic_level, "1",
            "Tiered diagnostics gate for PT_DIAG_TIERn() callsites. "
            "0 = off (LOG_ERROR/LOG_WARN still flow; tiered diagnostics "
            "suppressed). 1 = state transition (init / shutdown / fallback "
            "/ cvar change / mode switch -- one-shot events; production "
            "default). 2 = per-frame summary (<=1 line per frame per "
            "category, e.g. pipeline cache hit-rate, denoiser dispatch "
            "summary). 3 = per-call detail (every encode / dispatch / "
            "barrier; dev-only, very chatty). Adoption is opt-in per "
            "callsite -- raising the level only emits at sites that have "
            "been migrated to PT_DIAG_TIERn().",
            CVAR_ARCHIVE);
    PT_CVAR(r_capture_frame_at, "0",
            "Frame-capture: write the current visible frame to "
            "captures/capture_<frame_n>_<denoiser>_<ts>.<ext> at the given "
            "absolute frame index (engine's frame_index_, monotonic from "
            "0 on cold start; <ext> picked by r_capture_format). 0 = "
            "disabled. One-shot per cvar set; auto-resets to 0 after the "
            "capture fires. ABSOLUTE on purpose so it pairs with "
            "r_capture_seed for bit-identical capture across runs -- for "
            "interactive `capture in N frames from now` use the "
            "`capture_in N` command, which writes this cvar to "
            "frame_index + N. Source texture is picked from the engine "
            "state: accum_hdr (RGBA32F) when denoiser is off, "
            "denoise_color (RGBA16F) when denoiser is on. Both go through "
            "the same on-CPU ACES + sRGB OETF as the screenshot command, "
            "so the output matches what's on screen modulo bloom / "
            "lens-flare / perf-overlay.",
            0);
    PT_CVAR(r_capture_seq, "",
            "Frame-capture sequence: capture N frames at the given "
            "interval. Format: '<prefix> <N> <interval>' (positional, "
            "space-separated). Example: 'baseline 60 4' captures 60 "
            "frames every 4th frame starting from the next render "
            "frame. The prefix tags the output filenames "
            "(captures/<prefix>_<frame_n>_<denoiser>_<ts>.ppm). Empty "
            "string disables. Useful for analyzing temporal stability "
            "(non-temporal denoisers show frame-to-frame variance; "
            "temporal ones converge).",
            0);
    PT_CVAR(r_capture_seed, "0",
            "Frame-capture deterministic seed: when non-zero, the cvar's "
            "on_change handler resets the engine's frame_index_ counter "
            "to this value and forces an accum reset. Combined with "
            "identical camera + identical scene + identical r_spp, this "
            "makes two demont.exe runs produce bitwise-identical noise "
            "patterns at the same r_capture_frame_at. The path tracer's "
            "PRNG seeds purely from (pixel_id, frame_index) so this is "
            "the single knob needed for deterministic A/B comparison "
            "across denoiser configurations. 0 = let frame_index_ run "
            "naturally from cold-start 0.",
            0);

    // Hardware info (filled at startup, READONLY).
    PT_CVAR(sys_cpu_model,    "?",  "CPU brand string",              CVAR_READONLY);
    PT_CVAR(sys_cpu_pcores,   "0",  "Performance cores",             CVAR_READONLY);
    PT_CVAR(sys_cpu_ecores,   "0",  "Efficiency cores",              CVAR_READONLY);
    PT_CVAR(sys_cpu_features, "?",  "SIMD features",                 CVAR_READONLY);
    PT_CVAR(sys_ram_total_mb, "0",  "Physical RAM in MB",            CVAR_READONLY);
    PT_CVAR(sys_os,           "?",  "Operating system",              CVAR_READONLY);
    PT_CVAR(sys_gpu_name,     "?",  "GPU model (filled by RHI)",     CVAR_READONLY);
    PT_CVAR(sys_gpu_unified,  "1",  "1 if unified memory architecture", CVAR_READONLY);
    PT_CVAR(sys_gpu_hwrt,     "0",  "1 if hardware ray tracing supported", CVAR_READONLY);

    // --- Physics Phase 1 (#132) ------------------------------------------
    // Verlet-PBD physics layer. Off by default -- the path tracer ships
    // with analytic primitives that are static-by-default; enabling
    // physics swaps them into kinematically-driven spheres without
    // changing their visual identity. See src/physics/PhysicsSystem.h
    // for the integrator details. Stable sphere stacking is a Phase 5
    // problem -- Phase 1 spheres bounce until damping kills the energy.
    PT_CVAR(phys_enabled,     "0",
            "Enable the Verlet physics layer (#132). 0 = no physics (the "
            "engine acts exactly like a non-physics build, analytic prims "
            "are static); 1 = step every frame with the current "
            "phys_gravity_y / phys_substeps / phys_damping settings, "
            "writing back sphere positions to the analytic-primitive "
            "buffer before the BVH refit. Phase 5 will add friction, "
            "restitution, and stacking stability -- expect bouncy spheres "
            "and a settling pile in Phase 1.",
            CVAR_ARCHIVE);
    PT_CVAR(phys_gravity_y,   "-9.81",
            "Gravitational acceleration along world-Y in metres per "
            "second squared. -9.81 = Earth, +9.81 = anti-gravity, 0 = "
            "weightless. 1 world unit = 1 metre per the project's "
            "metric-units convention.",
            CVAR_ARCHIVE);
    PT_CVAR(phys_substeps,    "8",
            "Inner Verlet substeps per frame (1..32). The collision "
            "position-correction is a stiff constraint, so the per-frame "
            "dt is divided into N substeps before the integrator runs. "
            "8 substeps is the sweet spot for 60 fps + the default "
            "0.3 m radius: smaller numbers let fast-falling spheres "
            "tunnel through the ground plane, larger numbers waste CPU.",
            CVAR_ARCHIVE);
    PT_CVAR(phys_damping,     "0.99",
            "Per-substep velocity multiplier (0.5..1.0). 1.0 = energy-"
            "conserving (perpetual bounce); the default 0.99 bleeds "
            "1% of the integrator's implicit velocity per substep. At "
            "phys_substeps=8 and 60 fps that's 8*60=480 substeps/sec, "
            "so velocity decays as 0.99^480 ~= 0.008 (i.e. ~99% of "
            "velocity gone after one second, or kinetic energy down "
            "to ~0.006% of its initial value). The aggressive bleed "
            "is what lets a tossed pile of spheres settle into a loose "
            "heap in roughly a second of wall time without needing "
            "Phase 5's contact-persistence sleeper. Drop closer to "
            "1.0 (e.g. 0.999) for longer-lasting motion; values below "
            "~0.95 read as syrup at 60 fps.",
            CVAR_ARCHIVE);

    // Issue #181 visibility polish: per-frame override of analytic-prim
    // albedo for any sphere driven by the physics layer, colouring it
    // by linear-speed magnitude. Pure debug aid; intentionally NOT
    // archived so it never persists into a normal session. Mapping is
    // a 3-stop linear ramp in linear-RGB, calibrated to the default
    // damped Verlet integrator's implicit-velocity range:
    //   0 m/s  -> blue   (0.10, 0.30, 1.00)
    //   1 m/s  -> green  (0.20, 1.00, 0.30)
    //   2+ m/s -> red    (1.00, 0.20, 0.20)
    // 2 m/s is the practical terminal speed under default (g=-9.81,
    // phys_substeps=8, phys_damping=0.99, 60 fps): the implicit-velocity
    // Verlet caps at |g| * sdt / (1-damping) ~= 2.04 m/s. Users who
    // raise phys_damping closer to 1.0 (or hit it with Phase 2b+
    // external impulses) will see saturated red past 2 m/s -- read as
    // "moving as fast as the system permits". When the cvar transitions
    // 1 -> 0 the engine restores each overridden prim's original albedo
    // from its per-prim cache, so toggling the viz on/off is non-
    // destructive even mid-flight.
    PT_CVAR(r_phys_debug_visualize, "0",
            "Debug: when 1, every analytic-prim sphere driven by the "
            "physics layer (particles + rigid bodies) has its albedo "
            "overridden per frame to encode linear speed magnitude on "
            "a blue->green->red ramp (0 m/s = blue, 1 m/s = green, "
            "2+ m/s = red, linearly interpolated between the stops). "
            "The 2 m/s saturation matches the implicit-velocity Verlet "
            "terminal speed under default settings (|g| * sdt / "
            "(1-damping) ~= 2.04 m/s at -9.81 gravity, 8 substeps, "
            "0.99 damping, 60 fps); bodies stay saturated above that. "
            "Original albedos are cached on first override and restored "
            "when the cvar transitions back to 0 so toggling is non-"
            "destructive. Useful for verifying Phase 1/2a rigid-body "
            "motion in the path-traced view (#181). NOT CVAR_ARCHIVE "
            "-- per-invocation knob.",
            CVAR_NONE);
}  // namespace cvar

}  // namespace

Engine::Engine()  { g_instance = this; }
Engine::~Engine() { Shutdown(); if (g_instance == this) g_instance = nullptr; }

Engine* Engine::Instance() { return g_instance; }

bool Engine::Init() {
    pt::mem::Init();

    camera_ = std::make_unique<pt::renderer::Camera>();
    // ParticleSystem (issue #82 MVP). Pure CPU sim -- safe to
    // construct before any device exists. The cvar r_particles_max
    // will be honoured below once Console::Drain has processed the
    // archived demont.cfg.
    particles_ = std::make_unique<pt::effects::ParticleSystem>();

    pt::hw::Populate();
    auto& hi = pt::hw::GetInfo();
    auto& C  = pt::console::Console::Get();
    C.SetCVarOverride("sys_cpu_model",    hi.cpu_model);
    C.SetCVarOverride("sys_cpu_pcores",   std::to_string(hi.cpu_pcores));
    C.SetCVarOverride("sys_cpu_ecores",   std::to_string(hi.cpu_ecores));
    C.SetCVarOverride("sys_cpu_features", hi.cpu_features);
    C.SetCVarOverride("sys_ram_total_mb", std::to_string(hi.ram_total_mb));
    C.SetCVarOverride("sys_os",           hi.os_name);

    RegisterCommands();

    // pt_smoke_late_exec runtime fire-and-forget. During Init() the
    // end-of-Init read block (search for "Late smoke-exec hook" below)
    // reads the cvar once after all engine init is done. Without this
    // on_change wiring, an interactive `exec phys_rb_smoke.cfg` issued
    // at the console AFTER init completes would just set the cvar's
    // value and silently no-op -- nothing would read it. Wire an
    // on_change handler that runs the script immediately, but ONLY
    // after engine_initialized_ has been latched true (so we don't
    // double-fire during the demont.cfg/autoexec.cfg/pt_smoke_exec
    // load sequence below where the end-of-Init block handles things).
    // Re-entry guard prevents infinite loops if a late-exec script
    // itself sets pt_smoke_late_exec.
    if (auto* lx = pt::console::Console::Get().FindCVar("pt_smoke_late_exec")) {
        lx->on_change = [this](const pt::console::CVar& cv) {
            if (!engine_initialized_) return;
            if (cv.value.empty())     return;
            static thread_local bool in_late_exec = false;
            if (in_late_exec) {
                LOG_WARN("pt_smoke_late_exec: re-entrant set ignored ('{}')", cv.value);
                return;
            }
            in_late_exec = true;
            struct Guard { bool& f; ~Guard(){ f = false; } } g{in_late_exec};

            FILE* lf = std::fopen(cv.value.c_str(), "rb");
            if (lf == nullptr) {
                LOG_ERROR("pt_smoke_late_exec: '{}' not found", cv.value);
                return;
            }
            std::string lbody; char lbuf[4096];
            while (auto ln = std::fread(lbuf, 1, sizeof(lbuf), lf)) lbody.append(lbuf, ln);
            const bool lread_err = (std::ferror(lf) != 0);
            std::fclose(lf);
            if (lread_err) {
                LOG_ERROR("pt_smoke_late_exec '{}': read error", cv.value);
                return;
            }
            auto& Cl = pt::console::Console::Get();
            Cl.SetSuppressDepWarnings(true);
            auto lr = Cl.ExecuteScript(lbody);
            Cl.SetSuppressDepWarnings(false);
            if (!lr.ok) {
                LOG_ERROR("pt_smoke_late_exec '{}': {}", cv.value, lr.error);
                return;
            }
            LOG_INFO("loaded late-exec fixture {} (interactive)", cv.value);
        };
    }

    // Physical lens-flare init. Trace ghost paths for the canonical
    // lens once at startup -- the matrices are independent of frame
    // state so this is amortised across the whole run. Per-frame the
    // engine just calls prepare_shader_ghosts (~24 mat-vec) to convert
    // these to viewport-relative scales for the tonemap push struct.
    lens_system_      = lensflare::make_default_lens();
    lens_ghost_count_ = lensflare::trace_ghosts(
                            lens_system_, lens_ghosts_, lensflare::kMaxGhosts);
    lens_main_path_   = lensflare::trace_main_path(lens_system_);
    LOG_INFO("Lens flare: traced {} ghost paths (main B = {:.2f} mm)",
             lens_ghost_count_, lens_main_path_.B_g);

    // P11 persistence: replay last session's archived cvars first, then
    // the user's autoexec.cfg. Both are exec'd as plain console scripts
    // so any command (not just `<cvar> <value>`) is valid in autoexec.
    auto exec_if_exists = [](const char* path) {
        FILE* f = std::fopen(path, "rb");
        if (f == nullptr) return;
        std::string body; char buf[4096];
        while (auto n = std::fread(buf, 1, sizeof(buf), f)) body.append(buf, n);
        std::fclose(f);
        auto r = pt::console::Console::Get().ExecuteScript(body);
        if (!r.ok) LOG_WARN("{}: {}", path, r.error);
        LOG_INFO("loaded {}", path);
    };
    // cfg_loading_ suppresses astronomy-only on_change warnings
    // during cfg/autoexec/CLI overrides. Cfg writes happen in
    // lex order so r_sky_* values are set before
    // r_sky_use_astronomical -- without the suppression a user with
    // astro=1 saved would see warnings get superseded a moment later.
    // The post-cfg-load audit a few lines below uses the final state.
    //
    // Also suppress the issue #161 cross-cvar dependency warnings
    // for the same reason -- a cfg might set the dependent cvar on
    // a line BEFORE the prerequisite cvar (lex / save order is not
    // semantic order), so warning per line would produce false
    // positives on every load. After cfg-load completes, the
    // interactive user-set path re-engages predicate evaluation.
    cfg_loading_ = true;
    pt::console::Console::Get().SetSuppressDepWarnings(true);

    // `--no-cfg` (bare flag, no `=value`) skips BOTH demont.cfg and
    // autoexec.cfg load. Lets a tester run a golden-fixture scene
    // against a guaranteed-default cvar state without first having to
    // `mv demont.cfg demont.cfg.bak`. Pairs naturally with the
    // `cvar_reset_all` console command for runtime resets. The flag
    // parses here -- BEFORE the cfg-load calls below -- because the
    // generic `--<cvar>=value` override pass runs AFTER cfg load and
    // can't be used to express "don't load the cfg in the first place."
    bool skip_cfg_load = false;
    if (argv_ != nullptr) {
        for (int i = 1; i < argc_; ++i) {
            if (argv_[i] == nullptr) continue;
            if (std::string_view(argv_[i]) == "--no-cfg") {
                skip_cfg_load = true;
                break;
            }
        }
    }
    if (skip_cfg_load) {
        LOG_INFO("engine: --no-cfg given -- skipping demont.cfg + autoexec.cfg + favorites.cfg + console_history.txt + camera_bookmarks.cfg load");
    } else {
        exec_if_exists("demont.cfg");      // archived cvars from last quit
        exec_if_exists("autoexec.cfg");    // user-supplied startup script (overrides above)
        // Favourites are a parallel persistence file -- NOT a console
        // script. Loaded directly into Console::favorites_ via
        // LoadFavoritesFromDisk so the f1..fN magic invocation works
        // from the first line of any session. One favourite per line;
        // blank lines + leading `#` are ignored.
        LoadFavoritesFromDisk();
        // Console up-arrow history (parallel persistence file, NOT a
        // console script). Loaded directly into Console::history_ so
        // the platform overlay can seed its native walk buffer from
        // Console::History() during Init below. One line per entry;
        // blank lines + leading `#` are ignored.
        LoadConsoleHistoryFromDisk();
        // Named camera bookmarks (cam_save_named / cam_load_named) --
        // same per-line cfg pattern, but key/value parsed into
        // camera_bookmarks_ rather than a console-favourites vector.
        LoadCameraBookmarksFromDisk();
    }

    // Command-line cvar overrides land last so they beat both archived
    // and autoexec values. Currently this is the entry point for
    // running multiple demont.exe instances on different ports
    // (--net-port / --net-line-port); other --<cvar>=<value> args can
    // be added inside ApplyCommandLineCvarOverrides as needed.
    ApplyCommandLineCvarOverrides();

    // Golden-image regression matrix (issue #45) fixture exec. Runs
    // AFTER CLI overrides so the --smoke-exec=path flag has populated
    // pt_smoke_exec, and BEFORE the device boots so the fixture's cvar
    // writes (r_spp, r_max_bounces, camera pose, sun position,
    // app_window_*, r_capture_*, etc.) take effect during window
    // creation + first frame.
    //
    // Scope at THIS exec point: cvar writes only. CSG console commands
    // (`csg_box`, `csg_sphere`, `csg_cylinder`, `csg_op`, `csg_reset`,
    // `csg_remove`, `csg_dump`) are out of scope here -- `csg_scene_`
    // is not constructed and `SeedDefaultPrimitives()` has not run yet
    // (see the section below where both happen). And worse than silent:
    // several of the csg_* handlers dereference `csg_scene_` directly
    // before checking it, so a fixture that calls them at this point
    // would *crash* the engine, not no-op. The strict failure path
    // below catches that as a fixture parse/exec failure.
    // TODO(#45-followup): add a second exec pass after
    // SeedDefaultPrimitives() so CSG-touching fixtures work properly.
    //
    // Strict failure mode (unlike demont.cfg / autoexec.cfg above):
    // a smoke-exec fixture that can't be read or that produces a
    // parse error aborts Init -- demont returns non-zero, the wrapper
    // emits kExitSpawnError, the ctest `_render` cell fails. Silently
    // rendering against a half-applied fixture would surface as a
    // confusing pixel diff at best, or get committed as a bogus
    // baseline at regen time at worst.
    //
    // Same cfg_loading_ guard as demont.cfg/autoexec.cfg so astronomy
    // on_change warnings stay suppressed during fixture load.
    if (auto* sx = C.FindCVar("pt_smoke_exec"); sx != nullptr && !sx->value.empty()) {
        FILE* sf = std::fopen(sx->value.c_str(), "rb");
        if (sf == nullptr) {
            LOG_ERROR("pt_smoke_exec: '{}' not found", sx->value);
            cfg_loading_ = false;
            pt::console::Console::Get().SetSuppressDepWarnings(false);
            return false;
        }
        std::string sbody; char sbuf[4096];
        while (auto sn = std::fread(sbuf, 1, sizeof(sbuf), sf)) sbody.append(sbuf, sn);
        // fopen succeeds on a directory on POSIX, but fread on a
        // directory returns 0 with ferror() set. Without this check we
        // would run ExecuteScript on an empty body and silently render
        // the engine default instead of the intended fixture -- the
        // exact "confusing pixel diff later" failure mode this fail-fast
        // path is meant to prevent.
        const bool sread_err = (std::ferror(sf) != 0);
        std::fclose(sf);
        if (sread_err) {
            LOG_ERROR("pt_smoke_exec '{}': read error", sx->value);
            cfg_loading_ = false;
            pt::console::Console::Get().SetSuppressDepWarnings(false);
            return false;
        }
        auto sr = pt::console::Console::Get().ExecuteScript(sbody);
        if (!sr.ok) {
            LOG_ERROR("pt_smoke_exec '{}': {}", sx->value, sr.error);
            cfg_loading_ = false;
            pt::console::Console::Get().SetSuppressDepWarnings(false);
            return false;
        }
        LOG_INFO("loaded smoke-exec fixture {}", sx->value);
    }
    cfg_loading_ = false;
    pt::console::Console::Get().SetSuppressDepWarnings(false);

    // Post-cfg-load astronomy-mode audit. If astro is off after cfg +
    // autoexec + CLI overrides have applied, emit ONE summary warning
    // pointing at the dependency. This replaces the per-cvar startup
    // warnings (which were lex-order false-positive prone) with an
    // accurate single line that fires only when the final state is
    // a no-op for the saved astro cvars. Mid-session per-cvar
    // warnings still fire on user-issued console writes.
    if (auto* a = pt::console::Console::Get().FindCVar("r_sky_use_astronomical");
        a && !a->GetBool()) {
        LOG_WARN("r_sky_use_astronomical = 0 -- astronomy-only cvars "
                 "(r_sky_hour, r_sky_lat/lon, r_sky_year/month/day, "
                 "r_sky_city, r_sky_animate*, r_sky_hour_local, "
                 "r_sky_tz_offset_hours) have no visible effect in this "
                 "mode. Set r_sky_use_astronomical 1 to engage "
                 "time-based sun.");
    }

    // Job system.
    jobs_ = std::make_unique<pt::jobs::JobSystem>();
    jobs_->Init();
    pt::jobs::JobSystem::SetInstance(jobs_.get());

    // Audio subsystem (issue #80 MVP). Constructed BEFORE the smoke-exec
    // fixtures run so a fixture can issue `audio_play` calls -- the
    // device backend (CoreAudio / WASAPI / ALSA) is independent of the
    // GPU + window so there's no ordering dependency that pushes it
    // later. Init() failure is non-fatal: the rest of the AudioSystem
    // API no-ops cleanly when IsRunning() is false so headless / CI
    // boxes that lack audio continue working.
    audio_system_ = std::make_unique<pt::audio::AudioSystem>();
    if (!audio_system_->Init()) {
        LOG_WARN("audio: subsystem init failed; engine continues without audio");
    }

    // Window. Always created with no graphics-API context; each backend
    // attaches its own CAMetalLayer (or VkSurface for Vulkan) to the
    // NSWindow content view, so no recreate is needed on backend switch.
    auto* ww = C.FindCVar("app_window_width");
    auto* wh = C.FindCVar("app_window_height");
    int win_w = ww ? ww->GetInt() : 1280;
    int win_h = wh ? wh->GetInt() : 720;
    window_ = std::make_unique<pt::app::Window>();
    if (!window_->Create(win_w, win_h, "DeMonT Engine")) {
        LOG_ERROR("Failed to open window");
        return false;
    }

    // Console server.
    server_ = std::make_unique<pt::console::ConsoleServer>();
    pt::console::ConsoleServer::SetGlobalInstance(server_.get());
    pt::log::AddSink(&pt::console::ConsoleServer::OnLog);

    pt::console::ConsoleServer::Config sc;
    if (auto* p = C.FindCVar("net_port"))         sc.http_port = static_cast<std::uint16_t>(p->GetInt());
    if (auto* p = C.FindCVar("net_line_port"))    sc.line_port = static_cast<std::uint16_t>(p->GetInt());
    if (auto* p = C.FindCVar("net_bind_address")) sc.bind_address = p->value;
    server_->Start(sc, &C);

    // Editor backend (agent-19): wire WebSocket protocol hooks. Both
    // run on the civetweb worker thread; the select handler is safe
    // because it goes through QueueExecute (which is thread-safe). The
    // scene-dump handler reads engine maps and is documented (per
    // SerializeScene's threading notes) as caller-races-mutations'-
    // problem -- in practice the only mutations come from
    // Console::Drain on the main thread, and the dump reply is built
    // off a single read of those maps, so a torn read is essentially
    // impossible during normal play.
    server_->SetSelectHandler(
        [](std::string_view kind, std::uint32_t id) {
            const int k = pt::editor::SelectionKindFromString(kind);
            // QueueExecute is the canonical main-thread crossing. We
            // funnel the selection through a console line so the same
            // command path UI buttons / scripts use also flows here --
            // future change-tracking (undo, history) plugs in once.
            //
            // Format: "select_set <kind> <id>" (a thin internal alias
            // around SetSelection; registered below in RegisterCommands).
            std::string line = fmt::format("select_set {} {}",
                                           pt::editor::SelectionKindToString(k), id);
            pt::console::Console::Get().QueueExecute(std::move(line),
                                                     pt::console::Responder{});
        });
    server_->SetSceneDumpHandler([this]() -> std::string {
        // Build the JSON tree on the worker thread. SerializeScene only
        // reads `const` accessors; the caller-races-mutations contract
        // means we trust the engine not to be tearing the maps apart
        // while the worker thread is iterating. In the live use case
        // (Drain runs on main, scene mutations land there too) this
        // never races.
        return pt::editor::SerializeScene(*this).dump();
    });

    // Physics subsystem (#132) -- Verlet integrator + sphere-plane /
    // sphere-sphere collision. Allocated unconditionally so the
    // `phys_drop` console command + `phys_enabled` toggle work
    // without an engine restart; the system is dormant when
    // `phys_enabled = 0` (StepPhysics returns immediately).
    physics_ = std::make_unique<pt::physics::PhysicsSystem>();

    // CSG scene -- the source of truth for triangle-mesh geometry from
    // P9 onward. Seeded with the headline drilled-cube scene so first-
    // frame renders something interesting. SDF Phase 1 (#97) fixtures
    // can opt out via pt_smoke_skip_csg_seed so the SDF acceptance
    // scene renders without the gold drilled-cube in frame.
    csg_scene_ = std::make_unique<pt::csg::CsgScene>();
    bool skip_csg_seed = false;
    if (auto* v = C.FindCVar("pt_smoke_skip_csg_seed")) skip_csg_seed = v->GetBool();
    if (!skip_csg_seed) {
        SeedDefaultCsgScene();
    } else {
        // CsgScene's ctor itself seeds a unit box at id=1 (see Reset()
        // in src/renderer/Csg/CsgScene.cpp), so just skipping
        // SeedDefaultCsgScene isn't enough -- we still need to wipe
        // that default leaf. Remove drops it cleanly along with any
        // root reference. Now the bake produces an empty mesh and the
        // shader's tlas_present / bvh_params.z gates kick in.
        csg_scene_->Remove(1);
        LOG_INFO("engine: skipping default CSG seed (pt_smoke_skip_csg_seed=1)");
    }

    // Analytic primitives -- spheres + planes that the unified renderer
    // intersects alongside the mesh TLAS. Seeded with the canonical
    // 3-sphere + ground-plane scene. SDF Phase 1 (#97) fixtures can
    // opt out via pt_smoke_skip_prim_seed.
    bool skip_prim_seed = false;
    if (auto* v = C.FindCVar("pt_smoke_skip_prim_seed")) skip_prim_seed = v->GetBool();
    if (!skip_prim_seed) {
        SeedDefaultPrimitives();
    } else {
        LOG_INFO("engine: skipping default analytic-prim seed (pt_smoke_skip_prim_seed=1)");
    }

    // Boot the requested backend so the window renders something on
    // startup (defaults to "software" -- the only one online today).
    if (auto* v = C.FindCVar("r_backend")) {
        BackendType t = BackendType::None;
        if      (v->value == "software") t = BackendType::Software;
        else if (v->value == "metal")    t = BackendType::Metal;
        else if (v->value == "vulkan")   t = BackendType::Vulkan;
        if (t != BackendType::None) RequestBackendSwitch(t);
    }

    // Native in-window console overlay (uses macOS AppKit -- not a GUI
    // library; AppKit ships with the OS). Backtick toggles it.
    if (window_) {
        if (auto* ov = C.FindCVar("app_overlay_enabled"); ov && ov->GetBool()) {
            overlay_ = std::make_unique<pt::app::ConsoleOverlay>();
            if (overlay_->Init(window_->NativeHandle())) {
                pt::log::AddSink(&pt::app::ConsoleOverlay::OnLog);
                // P11 persistence: restore the previous session's
                // up-arrow history + scrollback. Single-shot: call
                // exactly once per process, here at Engine::Init's
                // overlay-attach site. A repeated call would re-append
                // the "previous session" separator block on top of
                // already-loaded entries; the Win32 LoadState impl
                // doesn't guard against that. Missing file is fine
                // (first launch, deleted file). Mac + Linux overlays
                // return false (no-op).
                overlay_->LoadState("demont_console.state");
            } else {
                overlay_.reset();
            }
        }

        // Always-visible perf overlay (separate child window from the
        // console).  Visibility is gated by r_perf_overlay -- the
        // overlay starts in its initial state and the on_change
        // handler below pushes the cvar's actual value once Init
        // returns.
        perf_overlay_ = std::make_unique<pt::app::PerfOverlay>();
        if (!perf_overlay_->Init(window_->NativeHandle())) {
            perf_overlay_.reset();
        } else if (auto* tv = C.FindCVar("r_theme")) {
            perf_overlay_->ApplyTheme(tv->value);
        }
        if (perf_overlay_) {
            int level = 0;
            if (auto* lv = C.FindCVar("r_perf_overlay")) {
                // std::atoi instead of std::stoi -- the project compiles
                // with -fno-exceptions, so std::stoi's throw on parse
                // failure aborts. atoi returns 0 on failure cleanly.
                level = std::atoi(lv->value.c_str());
            }
            // Native overlay is hidden when RHI mode is selected so
            // both don't draw at once.
            bool rhi = false;
            if (auto* mv = C.FindCVar("r_perf_overlay_mode")) rhi = (mv->value == "rhi");
            perf_overlay_->SetLevel(rhi ? 0 : level);
        }

        window_->SetKeyHandler([this](int key, int /*mods*/) {
            constexpr int kGrave = 96;  // GLFW_KEY_GRAVE_ACCENT
            // GLFW function-key codes (matches GLFW3 header constants).
            constexpr int kF2 = 291;
            constexpr int kF3 = 292;
            constexpr int kF4 = 293;
            constexpr int kF5 = 294;
            // Gizmo-mode hotkeys for the editor 3D gizmo overlay.
            constexpr int kG     = 71;   // GLFW_KEY_G
            constexpr int kR     = 82;   // GLFW_KEY_R
            constexpr int kS     = 83;   // GLFW_KEY_S
            if (key == kGrave) {
                if (overlay_) {
                    overlay_->Toggle();
                } else {
                    OpenWebConsole();
                }
                return;
            }
            // Editor panel hotkeys (#agent-20). The keys map 1:1 to
            // the four known panels; firing the existing console
            // command keeps the spawn-PID tracking centralised in
            // RegisterEditorCommands.
            //
            // The native console overlay's text-input mode swallows
            // key events at the platform layer (NSEvent on macOS,
            // WM_CHAR on Win32) before they reach GLFW, so we don't
            // need a separate "console active" gate here -- a key
            // event reaching this handler implies the game window is
            // the responder, not the console field.
            const char* panel = nullptr;
            switch (key) {
                case kF2: panel = "scene-hierarchy"; break;
                case kF3: panel = "inspector";       break;
                case kF4: panel = "asset-browser";   break;
                case kF5: panel = "toolbar";         break;
                default: break;
            }
            if (panel != nullptr) {
                pt::console::Console::Get().QueueExecute(
                    fmt::format("panel_open {}", panel),
                    {});
                return;
            }
            // Gizmo-mode hotkeys (issue: editor 3D gizmos). Match
            // Blender's G/R/S convention. We don't fire them while
            // the console overlay is active (typing into the console
            // shouldn't switch modes) and we DO fire them while WASD
            // is held -- the engine's WASD lookup polls IsKeyDown(W)
            // continuously, not via the per-press handler, so the
            // press hook doesn't fight movement.
            if (overlay_ && overlay_->IsShown()) return;
            // Gate gizmo hotkeys on three predicates:
            //   1. r_editor_gizmo is on (otherwise a fat-fingered G in a
            //      screenshot run shouldn't flip mode).
            //   2. A primitive is selected. Without selection the
            //      gizmo isn't drawn -- and S in particular doubles
            //      as the camera-backward key, so we keep the
            //      pass-through path while no prim is being edited.
            //   3. RMB camera look isn't active (the user is
            //      flying the camera, not editing).
            //
            // When all three hold, G/R/S re-purpose to gizmo mode
            // toggles. Pressing S still feeds IsKeyDown(83) in the
            // same frame, so the camera will continue stepping
            // backward while the user picks a mode -- documented
            // limitation; a follow-up will introduce a `gizmo_active`
            // input-mode bracket that suppresses WASD.
            auto& C = pt::console::Console::Get();
            bool gz_enabled = true;
            if (auto* v = C.FindCVar("r_editor_gizmo")) gz_enabled = v->GetBool();
            // selected_kind_ is the canonical PR #201 selection state;
            // only analytic-prim selections drive the gizmo today.
            if (!gz_enabled ||
                selected_kind_ != SelectionKind::AnalyticPrim ||
                selected_prim_id_ == 0) return;
            if (mouse_look_active_) return;
            if (key == kG) { C.SetCVarOverride("gizmo_mode", "translate"); return; }
            if (key == kR) { C.SetCVarOverride("gizmo_mode", "rotate");    return; }
            if (key == kS) { C.SetCVarOverride("gizmo_mode", "scale");     return; }
        });
    }

    if (auto* v = C.FindCVar("app_auto_open_console"); v && v->GetBool()) {
        OpenWebConsole();
    }

    // Seed camera from cvars (so cam_pos / cam_yaw / cam_pitch persist
    // across runs once we save archived cvars in P11).
    if (auto* v = C.FindCVar("cam_pos")) {
        std::string_view sv = v->value;
        std::size_t i = 0;
        glm::vec3 p = camera_->pos;
        for (int k = 0; k < 3; ++k) {
            while (i < sv.size() && sv[i] == ' ') ++i;
            char* end = nullptr;
            p[k] = std::strtof(sv.data() + i, &end);
            if (!end || end == sv.data() + i) break;
            i = static_cast<std::size_t>(end - sv.data());
        }
        camera_->pos = p;
    }
    if (auto* v = C.FindCVar("cam_yaw"))   camera_->yaw   = glm::radians(v->GetFloat());
    if (auto* v = C.FindCVar("cam_pitch")) camera_->pitch = glm::radians(v->GetFloat());
    if (auto* v = C.FindCVar("cam_fov"))   camera_->fov_deg = v->GetFloat();
    camera_->ClampPitch();

    // --- SDF Phase 1 (#97) -----------------------------------------------------
    // Late smoke-exec hook. Runs AFTER:
    //   - csg_scene_ construction + SeedDefaultCsgScene (or its skip)
    //   - SeedDefaultPrimitives (or its skip)
    //   - RegisterCommands (csg_*, prim_*, sdf_*) all registered
    //   - cvar overrides applied
    //   - Camera state pulled from cvars
    // so a fixture can issue any console command without crashing.
    // Same strict-failure semantics as pt_smoke_exec above.
    if (auto* lx = pt::console::Console::Get().FindCVar("pt_smoke_late_exec");
        lx != nullptr && !lx->value.empty()) {
        FILE* lf = std::fopen(lx->value.c_str(), "rb");
        if (lf == nullptr) {
            LOG_ERROR("pt_smoke_late_exec: '{}' not found", lx->value);
            return false;
        }
        std::string lbody; char lbuf[4096];
        while (auto ln = std::fread(lbuf, 1, sizeof(lbuf), lf)) lbody.append(lbuf, ln);
        const bool lread_err = (std::ferror(lf) != 0);
        std::fclose(lf);
        if (lread_err) {
            LOG_ERROR("pt_smoke_late_exec '{}': read error", lx->value);
            return false;
        }
        pt::console::Console::Get().SetSuppressDepWarnings(true);
        auto lr = pt::console::Console::Get().ExecuteScript(lbody);
        pt::console::Console::Get().SetSuppressDepWarnings(false);
        if (!lr.ok) {
            LOG_ERROR("pt_smoke_late_exec '{}': {}", lx->value, lr.error);
            return false;
        }
        LOG_INFO("loaded late smoke-exec fixture {}", lx->value);
    }
    // --- end SDF Phase 1 -------------------------------------------------------

    // Latch BEFORE the success log so any subsequent runtime write to
    // pt_smoke_late_exec via interactive `exec` triggers the on_change
    // handler installed near the top of Init() (search for
    // "pt_smoke_late_exec runtime fire-and-forget").
    engine_initialized_ = true;

    // Editor scaffold (agent-20): if r_editor_panels_autoopen is non-
    // empty, spawn each listed panel now -- after the WebSocket server
    // is up + command registration is done so the spawned Chrome
    // --app window can immediately connect to /ws and serve its
    // bundle. Smoke-test mode skips this so headless CI doesn't try
    // to launch browsers.
    if (auto* sf = C.FindCVar("pt_smoke_frames"); sf == nullptr || sf->GetInt() == 0) {
        OpenAutoOpenedPanels();
    }

    LOG_INFO("Engine initialized.");
    return true;
}

void Engine::OpenWebConsole() {
    auto* C  = &pt::console::Console::Get();
    auto* p  = C->FindCVar("net_port");
    auto* a  = C->FindCVar("net_bind_address");
    int  port = p ? p->GetInt() : 27960;
    std::string host = (a && !a->value.empty() && a->value != "0.0.0.0")
                       ? a->value : std::string("localhost");
    auto url = fmt::format("http://{}:{}/", host, port);
    LOG_INFO("Opening console: {}", url);

#if defined(__APPLE__)
    auto cmd = fmt::format("/usr/bin/open '{}' >/dev/null 2>&1", url);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("`open` returned {} -- visit {} manually", rc, url);
#elif defined(_WIN32)
    auto cmd = fmt::format("cmd /c start \"\" \"{}\"", url);
    std::system(cmd.c_str());
#else
    auto cmd = fmt::format("xdg-open '{}' >/dev/null 2>&1 &", url);
    std::system(cmd.c_str());
#endif
}

void Engine::ApplyCommandLineCvarOverrides() {
    if (argv_ == nullptr || argc_ <= 1) return;
    auto& C = pt::console::Console::Get();

    // Pure pass-through table: each entry maps a CLI flag prefix
    // (with trailing `=`) to the underlying cvar name. Adding a new
    // override is one row -- no extra parsing branches.
    struct OverrideMap { std::string_view flag_prefix; std::string_view cvar_name; };
    static constexpr OverrideMap kOverrides[] = {
        { "--net-port=",      "net_port"      },  // HTTP/WebSocket UI
        { "--net-line-port=", "net_line_port" },  // TCP console
        // --smoke-frames= and --r-backend= drive the manual / local
        // smoke-test mode (see pt_smoke_frames cvar in RegisterCVars
        // for the full backstory). --smoke-frames= gets an extra
        // integer-validation pass below because the underlying cvar
        // stores everything as a string + CVar::GetInt() is forgiving
        // (returns 0 on parse failure) -- without explicit validation
        // here, --smoke-frames=abc would silently disable smoke mode
        // instead of failing loudly. --r-backend= relies on the
        // generic allowed_values check below (r_backend's set is
        // {none, software, metal, vulkan}).
        { "--smoke-frames=",  "pt_smoke_frames" },
        { "--r-backend=",     "r_backend"     },
        // Golden-image regression matrix wiring (issue #45). Both
        // flags are pure pass-through to their backing cvars; cvar
        // help text covers the semantics. Engine::Run reads them on
        // loop start, the post-budget hook performs the readback +
        // PNG write. Empty value (no `=` payload) is rejected by the
        // shared empty-check above so `--smoke-exec=` with no path
        // doesn't silently disable the override.
        { "--smoke-exec=",         "pt_smoke_exec"        },
        { "--smoke-late-exec=",    "pt_smoke_late_exec"   },
        { "--smoke-capture-out=",  "pt_smoke_capture_out" },
        // Late smoke-exec hook (issue #97, also used by #140's voxel
        // demo fixture). Runs AFTER the CSG / prim / SDF seed +
        // command registration so the fixture can issue console
        // commands like `voxelize_object` or `csg_*` that depend on
        // those being live.
        { "--smoke-late-exec=",    "pt_smoke_late_exec"   },
    };

    for (int i = 1; i < argc_; ++i) {
        if (argv_[i] == nullptr) continue;
        const std::string_view arg(argv_[i]);
        bool matched = false;
        for (const auto& o : kOverrides) {
            if (!arg.starts_with(o.flag_prefix)) continue;
            const std::string_view value = arg.substr(o.flag_prefix.size());
            matched = true;
            if (value.empty()) {
                LOG_WARN("engine: {} given with empty value -- ignored",
                         o.flag_prefix);
                cli_arg_was_rejected_ = true;
                break;
            }
            // 1. Generic allowed_values check. SetCVarOverride bypasses
            //    CVAR_READONLY (necessary for some engine-set cvars)
            //    but does NOT enforce allowed_values (the equivalent
            //    check in Console::Execute is what normally rejects
            //    bad values from user input). For CLI args we want
            //    both behaviours: bypass READONLY, but reject typos /
            //    unsupported values. Hence the manual check here.
            auto* cv = C.FindCVar(o.cvar_name);
            if (cv != nullptr && !cv->allowed_values.empty()) {
                bool in_set = false;
                for (const auto& av : cv->allowed_values) {
                    if (av == value) { in_set = true; break; }
                }
                if (!in_set) {
                    std::string allowed_csv;
                    for (std::size_t j = 0; j < cv->allowed_values.size(); ++j) {
                        if (j) allowed_csv += ", ";
                        allowed_csv += cv->allowed_values[j];
                    }
                    LOG_ERROR("engine: {} = '{}' rejected: not in allowed set "
                              "for cvar '{}' (allowed: {}). Override ignored.",
                              arg, value, o.cvar_name, allowed_csv);
                    cli_arg_was_rejected_ = true;
                    break;
                }
            }
            // 2. Special-case validation for pt_smoke_frames: must
            //    parse as a non-negative integer. CVar::GetInt() is
            //    forgiving (silent fallback to 0 on parse failure)
            //    which would silently disable smoke mode for typos
            //    like --smoke-frames=abc. Better to reject here.
            if (o.cvar_name == std::string_view("pt_smoke_frames")) {
                // Manual digit scan keeps us out of std::stoll's
                // throw-on-failure semantics (we build with
                // -fno-exceptions). A leading '+' or '-' would also
                // be rejected -- negative budgets are nonsense.
                bool numeric = !value.empty();
                for (char c : value) {
                    if (c < '0' || c > '9') { numeric = false; break; }
                }
                if (!numeric) {
                    LOG_ERROR("engine: {} = '{}' rejected: --smoke-frames "
                              "must be a non-negative integer (got non-numeric "
                              "value). Override ignored.", arg, value);
                    cli_arg_was_rejected_ = true;
                    break;
                }
            }
            if (!C.SetCVarOverride(o.cvar_name, value)) {
                LOG_WARN("engine: {} -> SetCVarOverride('{}', '{}') failed "
                         "(cvar not registered yet?)",
                         arg, o.cvar_name, value);
                cli_arg_was_rejected_ = true;
            } else {
                LOG_INFO("engine: CLI override {} = '{}' (post-cfg)",
                         o.cvar_name, value);
            }
            break;
        }
        if (!matched && arg.starts_with("--")) {
            LOG_WARN("engine: unrecognised CLI flag '{}' (ignored)", arg);
        }
    }
}

void Engine::Shutdown() {
    // Stop the audio device first so its worker thread is gone before
    // the rest of the engine state it doesn't strictly depend on
    // (overlay sinks, console server, jobs) starts unwinding. Idempotent
    // -- safe even if Init() never constructed audio_system_.
    if (audio_system_) audio_system_->Shutdown();
    audio_system_.reset();

    TearDownDevice();

    // P11 persistence: dump every CVAR_ARCHIVE cvar that's been changed
    // away from its default into demont.cfg so the next launch picks
    // back up the user's settings.
    if (int n = pt::console::Console::Get().SaveArchivedCvars("demont.cfg"); n >= 0) {
        LOG_INFO("saved {} archived cvar(s) to demont.cfg", n);
    } else {
        LOG_WARN("could not write demont.cfg");
    }

    // P11 persistence: dump up-arrow history + scrollback so the next
    // Engine startup (cold launch or prompt-restart replacement) can
    // pick them back up. Mac + Linux overlays return false (no-op);
    // Win32 actually writes. Failure here is non-fatal (logged inside
    // SaveState). Done BEFORE RemoveAllSinks + overlay->Shutdown so
    // both the in-memory deques and the log sink are still wired.
    if (overlay_) overlay_->SaveState("demont_console.state");

    // Cross-platform console-input-history persistence. The Win32
    // overlay's SaveState above writes a richer state file
    // (history + scrollback) but it's Win32-only. console_history.txt
    // is the cross-platform up-arrow-only file that every host
    // (macOS, Linux, headless test) writes -- backed by Console's own
    // history_ vector, which platform overlays mirror via
    // PushHistory on submit. Safe to call alongside overlay_->SaveState
    // -- the two files coexist (SaveState is .state, history is .txt).
    SaveConsoleHistoryToDisk();

    pt::log::RemoveAllSinks();
    if (overlay_) overlay_->Shutdown();
    overlay_.reset();
    if (server_) server_->Stop();
    server_.reset();
    pt::console::ConsoleServer::SetGlobalInstance(nullptr);

    if (window_) window_->Destroy();
    window_.reset();

    if (jobs_) jobs_->Shutdown();
    pt::jobs::JobSystem::SetInstance(nullptr);
    jobs_.reset();

    // Physics is pure CPU data -- no device resources to release;
    // drop it after jobs_ so any future async physics worker (Phase
    // 4 GPU broadphase) can be joined safely here. Today this is a
    // plain unique_ptr destruction.
    if (physics_) {
        physics_->Clear();
        physics_.reset();
    }
}

void Engine::TearDownDevice() {
    // Drain any in-flight CSG bake so its worker doesn't touch the
    // engine after the device dies.
    if (jobs_ && bake_handle_.internal != nullptr) {
        jobs_->Wait(bake_handle_);
        bake_handle_ = {};
    }
    bake_phase_.store(0, std::memory_order_release);
    pending_baked_.reset();

    accum_texture_id_ = 0;
    accum_w_ = accum_h_ = 0;
    if (device_) {
        // Drain the GPU BEFORE releasing any resources. The Metal RHI's
        // DestroyTexture/DestroyBuffer call `release()` immediately on
        // the underlying MTLTexture/MTLBuffer; if the GPU still has the
        // last frame's compute dispatch in flight when we release them,
        // the driver gets left in a degraded internal state (no hard
        // crash thanks to deferred-release semantics, but WindowServer
        // / SkyLight have to clean up at the next lock/unlock or app
        // switch -- which has been observed as a visible multi-second
        // hitch when reactivating the screen after a demont session).
        // The Vulkan RHI's DestroyDevice path joins its async pipeline
        // worker + calls vkDeviceWaitIdle in its own destructor, but
        // per-resource Destroy* calls there are also synchronous so the
        // same ordering rule applies.
        device_->WaitIdle();
        if (scene_tlas_id_        != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
        if (box_blas_id_          != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
        if (box_vbuf_id_          != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
        if (box_ibuf_id_          != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
        if (tri_bvh_nodes_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_nodes_id_});
        if (tri_bvh_permuted_ids_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_permuted_ids_id_});
        if (prim_buffer_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{prim_buffer_id_});
        if (bvh_node_buffer_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{bvh_node_buffer_id_});
        // --- SDF Phase 1 (#97) -------------------------------------------------
        if (sdf_cluster_buffer_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{sdf_cluster_buffer_id_});
        // --- end SDF Phase 1 ---------------------------------------------------
        // --- Light primitives (#73) --------------------------------------------
        if (light_buffer_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{light_buffer_id_});
        // --- end Light primitives ----------------------------------------------
        // --- Light tree (#129) -------------------------------------------------
        // Double-buffered: destroy both GPU slots. The async builder
        // (CPU-side worker + double-buffered host trees) is torn down
        // in the *engine* dtor where it can't possibly race a frame.
        for (auto& id : light_tree_buffer_ids_) {
            if (id != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{id});
        }
        // --- end Light tree ----------------------------------------------------
        // --- Fluid Phase 1 (#136) ---------------------------------------------
        if (smoke_buffer_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{smoke_buffer_id_});
        // --- end Fluid Phase 1 ------------------------------------------------
        if (denoise_color_tex_id_    != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        if (depth_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
        if (motion_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
        if (post_denoise_hdr_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
        if (cloud_trans_tex_id_      != 0) device_->DestroyTexture(pt::rhi::TextureHandle{cloud_trans_tex_id_});
        // SIGMA shadow visibility buffer (issue #115). Storage buffer
        // (not texture) -- see comment at the allocation site.
        if (shadow_vis_buf_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{shadow_vis_buf_id_});
        // --- ReSTIR DI Phase A (issue #78) -------------------------------------
        // Per-pixel reservoir SSBOs. Triple-buffered: curr (PathTrace
        // output), prev (previous-frame final), swap (spatial-pass
        // output, swapped into prev at end-of-frame).
        if (restir_reservoir_curr_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_});
        if (restir_reservoir_prev_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_prev_buf_id_});
        if (restir_reservoir_swap_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_swap_buf_id_});
        // --- end ReSTIR DI Phase A ---------------------------------------------
        // MetalFX specular guidance G-buffers (issue #118).
        if (specular_albedo_tex_id_       != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_albedo_tex_id_});
        if (roughness_tex_id_             != 0) device_->DestroyTexture(pt::rhi::TextureHandle{roughness_tex_id_});
        if (specular_hit_distance_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_hit_distance_tex_id_});
        for (auto& id : bloom_mip_tex_id_) {
            if (id != 0) device_->DestroyTexture(pt::rhi::TextureHandle{id});
            id = 0;
        }
        if (bloom_dummy_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{bloom_dummy_tex_id_});
        if (env_map_tex_id_         != 0) device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
        if (env_marginal_cdf_id_    != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_marginal_cdf_id_});
        if (env_conditional_cdf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_conditional_cdf_id_});
        if (star_map_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{star_map_tex_id_});
        if (moon_map_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{moon_map_tex_id_});
        // GPU-resident exposure scalar buffer; pipelines (incl.
        // autoexpose_pipeline_id_) are owned by the device handle and
        // released by device_.reset() below, same as tonemap / bloom.
        if (exposure_state_id_      != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{exposure_state_id_});
        if (perfoverlay_drawlist_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{perfoverlay_drawlist_id_});
        if (editor_overlay_segs_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{editor_overlay_segs_buf_id_});
        if (placeholder_storage_id_  != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{placeholder_storage_id_});
        // Particle system GPU-side storage buffer (#82 MVP). The
        // CPU-side ParticleSystem object survives the device teardown
        // -- it has no GPU resources of its own. Only the per-frame
        // upload buffer is owned by the device.
        if (particles_storage_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{particles_storage_id_});
    }
    scene_tlas_id_        = 0;
    box_blas_id_          = 0;
    box_vbuf_id_          = 0;
    box_ibuf_id_          = 0;
    mesh_tri_count_       = 0;
    tri_bvh_nodes_id_       = 0;
    tri_bvh_permuted_ids_id_ = 0;
    tri_bvh_node_count_     = 0;
    tri_bvh_                = pt::renderer::TriangleBvh{};
    prim_buffer_id_       = 0;
    prim_buffer_capacity_ = 0;
    bvh_node_buffer_id_       = 0;
    bvh_node_buffer_capacity_ = 0;
    linear_prim_count_        = 0;
    // --- SDF Phase 1 (#97) -------------------------------------------------
    sdf_cluster_buffer_id_    = 0;
    sdf_cluster_capacity_     = 0;
    sdf_cluster_count_        = 0;
    sdf_prims_dirty_          = true;
    // --- end SDF Phase 1 ---------------------------------------------------
    // --- Light primitives (#73) --------------------------------------------
    light_buffer_id_       = 0;
    light_buffer_capacity_ = 0;
    light_count_uploaded_  = 0;
    light_prims_dirty_     = true;        // re-upload on next device
    // --- end Light primitives ----------------------------------------------
    // --- Light tree (#129) -------------------------------------------------
    light_tree_buffer_ids_          = {0, 0};
    light_tree_buffer_capacity_     = 0;
    light_tree_node_count_uploaded_ = 0;
    light_tree_active_slot_         = 0;
    light_tree_first_built_         = false;
    // builder itself is kept across device tears; it's a pure-CPU
    // worker, never touched the device.
    // --- end Light tree ----------------------------------------------------
    // --- Fluid Phase 1 (#136) ---------------------------------------------
    smoke_buffer_id_       = 0;
    smoke_buffer_capacity_ = 0;
    smoke_count_uploaded_  = 0;
    // --- end Fluid Phase 1 ------------------------------------------------
    denoise_color_tex_id_    = 0;
    depth_tex_id_            = 0;
    motion_tex_id_           = 0;
    post_denoise_hdr_tex_id_ = 0;
    cloud_trans_tex_id_      = 0;
    shadow_vis_buf_id_       = 0;
    sigma_shadow_pipeline_id_ = 0;
    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    restir_reservoir_curr_buf_id_ = 0;
    restir_reservoir_prev_buf_id_ = 0;
    restir_reservoir_swap_buf_id_ = 0;
    restir_temporal_pipeline_id_  = 0;
    restir_spatial_pipeline_id_   = 0;
    restir_final_pipeline_id_     = 0;
    restir_alloc_w_               = 0;
    restir_alloc_h_               = 0;
    // --- end ReSTIR DI Phase A ---------------------------------------------
    // MetalFX specular guidance G-buffers (issue #118).
    specular_albedo_tex_id_       = 0;
    roughness_tex_id_             = 0;
    specular_hit_distance_tex_id_ = 0;
    tonemap_pipeline_id_     = 0;
    stars_composite_pipeline_id_ = 0;
    aurora_composite_pipeline_id_ = 0;
    particle_composite_pipeline_id_ = 0;
    particles_storage_id_           = 0;
    particles_storage_capacity_     = 0;
    bloom_down_pipeline_id_  = 0;
    bloom_up_pipeline_id_    = 0;
    perfoverlay_pipeline_id_       = 0;
    perfoverlay_drawlist_id_       = 0;
    perfoverlay_drawlist_capacity_ = 0;
    editor_overlay_pipeline_id_      = 0;
    editor_overlay_segs_buf_id_      = 0;
    editor_overlay_segs_buf_capacity_ = 0;
    bloom_dummy_tex_id_      = 0;
    for (auto& id : bloom_mip_tex_id_) id = 0;
    for (auto& w  : bloom_mip_w_)      w  = 0;
    for (auto& h  : bloom_mip_h_)      h  = 0;
    env_map_tex_id_       = 0;
    env_marginal_cdf_id_      = 0;
    env_conditional_cdf_id_   = 0;
    env_total_luminance_      = 0.0f;
    star_map_tex_id_      = 0;
    star_map_present_     = 0;
    moon_map_tex_id_      = 0;
    autoexpose_pipeline_id_ = 0;
    exposure_state_id_      = 0;
    placeholder_storage_id_ = 0;
    denoiser_active_      = false;
    prev_view_proj_valid_ = false;
    primitives_dirty_     = true;        // re-upload on next device

    if (device_) {
        // Already drained above before the resource destroys; this
        // WaitIdle is the belt-and-braces guard for the device's own
        // pipeline / accel-structure tail before `device_.reset()`
        // tears the queue down. Cheap when nothing's pending.
        device_->WaitIdle();
        device_.reset();
    }
    pathtrace_pipeline_id_ = 0;
}

void Engine::RequestBackendSwitch(BackendType to) {
    // Guard: cfg-driven on_change can fire from inside Console::ExecuteScript
    // during the demont.cfg load step, which RegisterCommands has already
    // sequenced by then -- but Init() runs the cfg load BEFORE it
    // creates window_.  A premature switch here would create a device
    // with no native-window handle (no CAMetalLayer attached on Mac,
    // no HWND on Windows), the present blit would silently no-op every
    // frame, and the explicit RequestBackendSwitch at the tail of Init()
    // would then short-circuit via the `to == current_backend_ && device_`
    // check below -- leaving the unrenderable device in place.  Symptom:
    // `r_backend software` in demont.cfg renders nothing on launch, but
    // toggling to a different backend and back fixes it (because by then
    // window_ exists).  Returning here defers the real setup to the
    // explicit Init() call that runs after window creation.
    if (!window_) return;

    if (to == current_backend_ && (to == BackendType::None || device_)) return;
    LOG_INFO("backend switch: {} -> {}",
             pt::rhi::BackendName(current_backend_),
             pt::rhi::BackendName(to));

#if defined(_WIN32)
    // The vulkan->software switch with r_software_blit=gdi will leave
    // the window stuck on the last Vulkan frame for the rest of the
    // session (Microsoft DXGI flip-model lockout, see Vulkan
    // VK_KHR_win32_surface spec note + Microsoft DXGI flip-model docs).
    // r_software_blit_recreate selects how we handle this:
    //   - "auto"   : RecreateWindow() -- destroy + recreate the HWND
    //                in-process so the new HWND has never been a
    //                Vulkan present target (preserves engine state)
    //   - "prompt" : MessageBox Restart Now? -- Yes spawns a fresh
    //                process; No falls back to warn
    //   - "warn"   : legacy LOG_WARN; user manually relaunches
    // Decision is recorded BEFORE TearDownDevice runs so we can short-
    // circuit the prompt path (Yes -> spawn + set wants_quit_, return
    // without touching the device -- the main loop's normal teardown
    // path runs as the loop exits).
    bool need_recreate = false;
    bool need_warn     = false;
    if (to == BackendType::Software &&
        current_backend_ == BackendType::Vulkan) {
        if (auto* v = pt::console::Console::Get().FindCVar("r_software_blit");
            v && v->value == "gdi") {
            std::string mode = "auto";
            if (auto* m = pt::console::Console::Get().FindCVar("r_software_blit_recreate")) {
                mode = m->value;
            }
            if (mode == "auto") {
                need_recreate = true;
            } else if (mode == "prompt") {
                LOG_INFO("r_software_blit_recreate=prompt: asking user whether to restart for a clean GDI session");
                HWND parent = window_ ? static_cast<HWND>(window_->NativeHandle()) : nullptr;
                int btn = MessageBoxW(
                    parent,
                    L"Switching from Vulkan to Software with r_software_blit=gdi "
                    L"will leave the window stuck on the last Vulkan frame "
                    L"(Microsoft DXGI flip-model lockout).\r\n\r\n"
                    L"Restart the application now to use GDI cleanly?\r\n\r\n"
                    L"Yes: relaunch (current session is lost)\r\n"
                    L"No: continue without restart (set r_software_blit vulkan to recover)",
                    L"demont engine: vulkan -> software (GDI)",
                    MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON1);
                if (btn == IDYES) {
                    // Persist the user's current cvar state (most
                    // importantly r_backend=software and r_software_
                    // blit=gdi -- both freshly mutated in this Tick
                    // and not yet on disk) BEFORE spawning so the
                    // child reads what the user actually intended.
                    // Otherwise CreateProcessA + child Init can race
                    // ahead of this process's Shutdown-time cfg
                    // write, and the child boots from stale on-disk
                    // values (typically the previous launch's
                    // r_backend=vulkan -- not what the user asked
                    // for). Bare LOG only; failure here is non-fatal.
                    if (int n = pt::console::Console::Get().SaveArchivedCvars("demont.cfg"); n >= 0) {
                        LOG_INFO("r_software_blit_recreate=prompt: pre-spawn cfg save ok ({} cvar(s))", n);
                    } else {
                        LOG_WARN("r_software_blit_recreate=prompt: pre-spawn cfg save failed -- child may boot with stale cvar state");
                    }
                    // Persist up-arrow history + scrollback now so the
                    // child can pick them up on Init. Shutdown also
                    // saves later, but that's AFTER the child has
                    // already started reading -- so we'd lose this
                    // session's tail. Saving twice is harmless (same
                    // atomic .tmp + rename).
                    if (overlay_) {
                        overlay_->SaveState("demont_console.state");
                    }
                    // Release our HTTP / TCP console-server ports BEFORE
                    // spawning so the child's ConsoleServer::Start can
                    // bind them cleanly. Without this the child races
                    // ahead of our Shutdown-time server reset and hits
                    // "Failed to start civetweb on 127.0.0.1:<port>" --
                    // the child still runs (bind failure is non-fatal)
                    // but loses HTTP / web-console access for the new
                    // session. Releasing here also drops any already-
                    // connected web console clients; they should
                    // reconnect to the new process which now owns the
                    // ports.
                    //
                    // server_.reset() calls ~ConsoleServer -> Stop()
                    // exactly once. We also null the singleton pointer
                    // so the static OnLog log sink (still registered
                    // for the remainder of this Tick) sees g_instance
                    // == nullptr and bails out instead of dereferencing
                    // a destroyed server_. Shutdown's `if (server_)`
                    // guard turns the later server_->Stop() into a
                    // no-op; ConsoleServer::Stop is NOT idempotent
                    // wrt WSACleanup (unconditional cleanup would
                    // double-decrement Winsock's refcount if called
                    // twice), so a single reset here is the right
                    // shape.
                    if (server_) {
                        LOG_INFO("r_software_blit_recreate=prompt: releasing console-server ports for child");
                        pt::console::ConsoleServer::SetGlobalInstance(nullptr);
                        server_.reset();
                    }
                    if (RestartProcess()) {
                        LOG_INFO("r_software_blit_recreate=prompt: spawn succeeded; tearing down GPU + hiding window so the user sees only the new process");
                        // Tear down the Vulkan device now (instead of
                        // letting Shutdown do it at end of loop) for
                        // two reasons:
                        //
                        //   1. The current Tick is mid-iteration; the
                        //      remaining RenderFrame call would dispatch
                        //      a final frame on a device whose swapchain
                        //      may have lost compatibility while the
                        //      modal MessageBox was up (DWM can
                        //      reconfigure during modal dispatch).
                        //      RenderFrame guards on (!device_) so
                        //      clearing it makes the rest of this Tick
                        //      a clean no-op.
                        //   2. The child process is already starting
                        //      up its own Vulkan instance; releasing
                        //      ours promptly avoids two processes
                        //      hammering the GPU + driver state at the
                        //      same moment (driver crashes have been
                        //      reported on this overlap window with
                        //      the previous prompt-path implementation).
                        //
                        // Hide the window so the user only sees the
                        // child's window during the brief overlap
                        // before wants_quit_ unwinds the main loop.
                        // Window stays alive (Hide, not Destroy) so
                        // the rest of this Tick + the Shutdown path
                        // can finish without window_->Handle() going
                        // null on them.
                        TearDownDevice();
                        current_backend_ = BackendType::None;
                        if (window_) window_->Hide();
                        return;
                    }
                    LOG_WARN("r_software_blit_recreate=prompt: CreateProcessA failed; falling back to warn behaviour");
                    need_warn = true;
                } else {
                    LOG_INFO("r_software_blit_recreate=prompt: user declined restart; falling back to warn behaviour");
                    need_warn = true;
                }
            } else {
                // mode == "warn" (or any unrecognised value -- the
                // allowed_values validator at registration time rejects
                // anything else, so this branch is the explicit warn
                // path).
                need_warn = true;
            }
        }
    }
#endif

    TearDownDevice();

#if defined(_WIN32)
    if (need_recreate) {
        LOG_INFO("r_software_blit_recreate=auto: vulkan -> software gdi: recreating HWND to escape DXGI flip-model lockout");
        if (!RecreateWindow()) {
            // Hard stop. Window::Recreate failure leaves window_ alive
            // but with handle_ == nullptr (Destroy succeeded, Create
            // failed), and RecreateWindow has already dropped both
            // overlays. Continuing to Device::Create below with the
            // null native handle would produce an unpresentable device:
            // every present blit would silently no-op forever (the same
            // failure mode the early `if (!window_) return;` guards
            // against on initial init).
            //
            // The old device is already torn down (TearDownDevice ran
            // above), so we can't recover the previous render either.
            // Request a clean process exit so the user sees Shutdown
            // run its course and can relaunch. wants_quit_ is the same
            // signal RestartProcess uses on the prompt path -- the
            // main loop checks it on the next iteration.
            LOG_ERROR("r_software_blit_recreate=auto: RecreateWindow failed -- window is destroyed and engine is unrenderable; requesting clean quit so the user can relaunch");
            current_backend_ = BackendType::None;
            wants_quit_      = true;
            return;
        }
    }
    if (need_warn) {
        LOG_WARN("backend switch vulkan -> software with "
                 "r_software_blit=gdi: window will be permanently "
                 "stuck on the last Vulkan frame (Microsoft DXGI "
                 "flip-model lockout). Quit + relaunch with "
                 "r_backend software (and r_software_blit gdi) "
                 "in demont.cfg for a clean GDI session, set "
                 "r_software_blit vulkan to switch live, or set "
                 "r_software_blit_recreate auto for an automatic "
                 "in-process recreate.");
    }
#endif

    if (to == BackendType::None) {
        current_backend_ = BackendType::None;
        return;
    }

    // No window recreate needed: window is always NO_API and each backend
    // attaches its own surface (CAMetalLayer / VkSurface) to it.

    pt::rhi::NativeWindowHandle nw{
        .opaque = window_ ? static_cast<void*>(window_->Handle()) : nullptr,
        .width  = window_ ? window_->Width()  : 0,
        .height = window_ ? window_->Height() : 0,
    };
    device_ = pt::rhi::Device::Create(to, nw);
    if (!device_) {
        LOG_ERROR("failed to create {} device", pt::rhi::BackendName(to));
        current_backend_ = BackendType::None;
        return;
    }

    current_backend_ = to;
    pt::console::Console::Get().SetCVarOverride(
        "sys_gpu_name", device_->DeviceName());
    pt::console::Console::Get().SetCVarOverride(
        "sys_gpu_hwrt", device_->SupportsHardwareRT() ? "1" : "0");

    // Compute kernels are looked up by name from the active device's
    // pre-built pipeline table. EnsurePipelineHandles fills the cached
    // ids; on Metal those resolve immediately, on Vulkan they may
    // resolve to 0 here while the async pipeline-build worker is
    // still in flight, in which case RenderFrame keeps re-resolving
    // each frame until they flip non-zero.
    EnsurePipelineHandles();
    // Predictive pipeline JIT prewarming. After backend swap the fresh
    // VulkanDevice's worker is already in flight (it was launched by
    // the constructor); this call hands the worker an explicit list of
    // names the engine will dispatch. Today most of those resolve to
    // "already in the worker's hardcoded build list, no-op" but the
    // handshake gives future on-demand pipelines a clean entry point
    // without each new pipeline owner having to touch this function.
    // Metal / software back-ends override this to a no-op via the
    // Device::EnsurePipelineWarmed default. See r_pipeline_prewarm.
    PrewarmPipelines();

    // GPU-resident exposure scalar (1 float). AutoExposure.slang
    // updates this each tick when r_auto_exposure=1; engine writes the
    // manual r_exposure value here when r_auto_exposure=0. PathTrace.slang
    // reads from it in the final tonemap, replacing the per-frame
    // accum_hdr readback path.
    //
    // Seed with r_exposure if manual mode is the user's current state,
    // otherwise 1.0 (auto mode kernel will overwrite on first frame).
    {
        auto buf = device_->CreateBuffer({
            .size = sizeof(float),
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "exposure_state",
        });
        exposure_state_id_ = buf.id;
        if (exposure_state_id_ == 0) {
            // exposure_state is required by PathTrace.slang (and
            // Tonemap.slang in the Metal denoiser path) -- they read
            // exposure_state[0] unconditionally. With it unallocated,
            // the binds at slot 6 are silently skipped and the kernels
            // either read the wrong slot or read garbage (especially
            // on Metal where there's no nullDescriptor safety). Log
            // and continue; further diagnosis would surface as black /
            // wrong-exposure output rather than a silent failure.
            LOG_ERROR("exposure_state buffer creation failed -- auto / manual exposure will misbehave");
        } else {
            float init_exposure = 1.0f;
            auto& Cx = pt::console::Console::Get();
            bool auto_exp = true;
            if (auto* av = Cx.FindCVar("r_auto_exposure")) auto_exp = av->GetBool();
            if (!auto_exp) {
                if (auto* ev = Cx.FindCVar("r_exposure")) init_exposure = ev->GetFloat();
            }
            device_->WriteBuffer(buf, &init_exposure, sizeof(float), 0);
        }
    }

    // Perf-overlay draw list: one DrawCmd per uint4 entry.  Sized for
    // tier-3 worst case (~256 sparkline segments + a handful of chrome
    // entries).  Uploaded each frame via WriteBuffer when the RHI-mode
    // overlay is active.
    {
        constexpr std::uint32_t kPerfDrawCapacity = 320;   // entries
        auto buf = device_->CreateBuffer({
            .size = sizeof(std::uint32_t) * 4 * kPerfDrawCapacity,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "perfoverlay_drawlist",
        });
        perfoverlay_drawlist_id_       = buf.id;
        perfoverlay_drawlist_capacity_ = kPerfDrawCapacity;
    }

    // Editor 3D-transform gizmo segment buffer (issue: editor 3D gizmos).
    // One Segment record = 3 * float4 = 48 bytes; the engine writes the
    // CPU-built list each frame via WriteBuffer and the shader iterates
    // [0, num_segments). Sized to comfortably hold the rotate-mode
    // worst case: 3 rings @ 32 segments + 24 translate-mode segments
    // for the scale gizmo's three cubes = ~120 segments. Pick 256 for
    // safety -> 12 KB.
    {
        constexpr std::uint32_t kEditorOverlayCapacity = 256;
        auto buf = device_->CreateBuffer({
            .size = sizeof(float) * 12u * kEditorOverlayCapacity,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "editor_overlay_segments",
        });
        editor_overlay_segs_buf_id_       = buf.id;
        editor_overlay_segs_buf_capacity_ = kEditorOverlayCapacity;
    }

    // Always-present placeholder storage buffer. Used as a harmless
    // fallback for optional binding slots (env CDFs at 4/5, BVH nodes
    // at 7, tri_bvh nodes/ids at 8/9, SDF clusters at 10) whose primary
    // buffer can legally be 0 (no env map; no BVH built; no SDFs
    // uploaded). Metal computes the dynamic push-constant slot from
    // max-bound + 1 of the contiguous
    // range, so leaving any slot in that range unbound shifts the push
    // slot and corrupts every push field -> all-black render. Previous
    // code reused prim_buffer_id_ for the same purpose, but with
    // pt_smoke_skip_prim_seed=1 and no analytic prims that id stays 0
    // too. Sized at 16 bytes (one float4) -- the shader never reads
    // these slots when their respective count fields are 0.
    {
        auto buf = device_->CreateBuffer({
            .size = 16,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "placeholder_storage",
        });
        placeholder_storage_id_ = buf.id;
        if (placeholder_storage_id_ == 0) {
            LOG_ERROR("placeholder_storage buffer creation failed -- "
                      "optional binding slots may fall back to slot 0 / "
                      "shift Metal push-constant slot");
        }
    }

    // Mark the analytic-primitive buffer dirty so it gets uploaded on
    // the first frame against this device.
    primitives_dirty_ = true;
    // The CSG mesh BLAS + TLAS were just destroyed in TearDownDevice
    // but csg_scene_->Dirty() only tracks user-driven topology changes
    // -- it stays clean across a backend switch where the tree is
    // unchanged.  Force EnsureMeshUpdated() to kick a re-bake on the
    // first frame against the new device, otherwise the CSG mesh
    // would silently disappear after a Software <-> Metal swap.
    force_mesh_rebuild_ = true;

    // Reload env map on the new device so its texture handle is valid.
    if (auto* v = pt::console::Console::Get().FindCVar("r_env_map");
        v && !v->value.empty()) {
        ReloadEnvMap(v->value);
    }

    // BSC starmap: load + rasterise once on this device.
    EnsureStarMapUploaded();
    EnsureMoonMapUploaded();
}

bool Engine::RecreateWindow() {
#if defined(_WIN32)
    if (!window_) {
        LOG_ERROR("Engine::RecreateWindow: no window to recreate");
        return false;
    }
    LOG_INFO("Engine::RecreateWindow: tearing down overlays before HWND swap");
    // Shut overlays down explicitly BEFORE destroying the parent
    // GLFW window. Windows would cascade-destroy the child HWNDs when
    // the parent dies, but the overlay objects still cache stale
    // hwnd_ pointers in that case -- their next Paint/Repaint/Update
    // would touch a dead handle. Explicit Shutdown nulls those
    // pointers cleanly, and the static log dispatch (ConsoleOverlay::
    // OnLog -> g_instance) goes inert until re-Init.
    if (overlay_) {
        overlay_->Shutdown();
    }
    if (perf_overlay_) {
        perf_overlay_->Shutdown();
    }

    if (!window_->Recreate()) {
        LOG_ERROR("Engine::RecreateWindow: Window::Recreate failed; engine is in an unusable window state");
        // Drop the overlays since their parent_ is dead. The engine
        // is now unrenderable (no window, both overlays gone) and
        // returning false signals the dispatch site to hard-stop:
        // it sets current_backend_=None + wants_quit_=true so the
        // main loop exits cleanly and the user can relaunch. We
        // can't continue down the device-creation path because
        // window_->Handle() is nullptr -- the new device would have
        // no native handle and present-blits would silently no-op
        // forever.
        overlay_.reset();
        perf_overlay_.reset();
        return false;
    }

    LOG_INFO("Engine::RecreateWindow: re-attaching overlays to fresh HWND");
    auto& C = pt::console::Console::Get();
    if (overlay_) {
        if (!overlay_->Init(window_->NativeHandle())) {
            LOG_WARN("Engine::RecreateWindow: ConsoleOverlay re-Init failed; in-window log overlay disabled for this session");
            overlay_.reset();
        } else if (auto* tv = C.FindCVar("r_theme")) {
            // Re-apply the current theme. r_theme's on_change handler
            // would push the theme to overlay_ if it fired, but the
            // cvar value hasn't changed across the recreate -- so the
            // on_change never fires and the freshly-Init'd overlay
            // sits at its default theme. Mirror the perf_overlay
            // branch below (which already does this).
            overlay_->ApplyTheme(tv->value);
        }
        // Sink list is unchanged: ConsoleOverlay::OnLog stays
        // registered and starts forwarding again now that the static
        // g_instance points at the live overlay.
    }
    if (perf_overlay_) {
        if (!perf_overlay_->Init(window_->NativeHandle())) {
            LOG_WARN("Engine::RecreateWindow: PerfOverlay re-Init failed; perf HUD disabled for this session");
            perf_overlay_.reset();
        } else {
            // Mirror the initial setup in Init(): re-apply theme, then
            // resolve r_perf_overlay + r_perf_overlay_mode to set the
            // tier level (or 0 if r_perf_overlay_mode=rhi).
            if (auto* tv = C.FindCVar("r_theme")) {
                perf_overlay_->ApplyTheme(tv->value);
            }
            int level = 0;
            if (auto* lv = C.FindCVar("r_perf_overlay")) {
                level = std::atoi(lv->value.c_str());
            }
            bool rhi = false;
            if (auto* mv = C.FindCVar("r_perf_overlay_mode")) rhi = (mv->value == "rhi");
            perf_overlay_->SetLevel(rhi ? 0 : level);
        }
    }
    LOG_INFO("Engine::RecreateWindow: ok; HWND swap complete");
    return true;
#else
    LOG_ERROR("Engine::RecreateWindow: Win32-only (called on a non-Win32 build)");
    return false;
#endif
}

bool Engine::RestartProcess() {
#if defined(_WIN32)
    // Resolve the running exe path via GetModuleFileNameA -- argv[0]
    // can be a relative path or a different casing than what the OS
    // actually loaded, and using the OS-reported path avoids surprises
    // when the user launched from a different cwd. ANSI is fine: paths
    // on Windows can legitimately contain non-ASCII via the active
    // code page, but the engine's existing logging + cfg load chain is
    // also ANSI, so we stay consistent. If a non-ASCII path bites,
    // upgrading to GetModuleFileNameW is a localised change.
    char exe_path[MAX_PATH] = {0};
    DWORD n = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        LOG_ERROR("Engine::RestartProcess: GetModuleFileNameA failed (GLE={}, n={})", GetLastError(), n);
        return false;
    }

    // Build the lpCommandLine string. CreateProcessA requires a single
    // writable buffer, by convention starting with the (quoted)
    // application name followed by space-separated args.
    //
    // Windows argv quoting per CommandLineToArgvW
    // (https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args
    //  #parsing-c-command-line-arguments) is unusual:
    //
    //   - 2n   backslashes followed by " -> n backslashes + begin/end quote.
    //   - 2n+1 backslashes followed by " -> n backslashes + literal quote.
    //   - n    backslashes not followed by " -> n LITERAL backslashes.
    //
    // The naive "escape every \\ and \"" approach (used in the first
    // pass of this code, caught by Copilot review) doubles every
    // backslash unconditionally. CommandLineToArgvW then sees a run of
    // backslashes not followed by a quote and treats it as literal:
    // a Windows path like C:\Program Files\demont.exe becomes
    // C:\\Program Files\\demont.exe in the child's argv, breaking any
    // arg whose value contains a backslash.
    //
    // Correct algorithm (Daniel Colascione, "Everyone quotes command
    // line arguments the wrong way"):
    //   - If arg has no whitespace / tab / quote / cmd metacharacter:
    //     emit literally.
    //   - Else: wrap in "...". Inside the quoted run:
    //     - Track runs of consecutive backslashes.
    //     - If the run is followed by " (embedded literal quote):
    //         double the run + emit a single escape backslash + emit ".
    //     - If the run is at end-of-arg (closing quote follows):
    //         double the run.
    //     - Otherwise (run followed by ordinary char):
    //         emit the run literally.
    //
    // The trigger-set is extended beyond ` \t"` to include the cmd.exe
    // metacharacters & | < > ( ) ^ -- because we ship the command
    // line through `cmd.exe /c start ...` below, an arg containing
    // any of those would be reinterpreted by cmd before the target
    // sees it. Wrapping in double quotes neutralises them inside the
    // quoted run, leaving the target's CommandLineToArgvW parse
    // unchanged. NOTE: % is NOT escaped here -- cmd expands
    // %VAR% inside double quotes too, so an arg like `--user=%USER%`
    // would still be interpreted by cmd (the canonical fix is the
    // %^% trick but it doesn't survive cmd's quoting parser). The
    // engine's own argv doesn't use % so this is a documented
    // residual limit, not a live bug.
    auto quote = [](const std::string& s) -> std::string {
        if (!s.empty() && s.find_first_of(" \t\"&|<>()^") == std::string::npos) return s;
        std::string r = "\"";
        for (std::size_t i = 0; i < s.size(); ) {
            std::size_t bs = 0;
            while (i < s.size() && s[i] == '\\') { ++bs; ++i; }
            if (i == s.size()) {
                r.append(2 * bs, '\\');               // run before closing "
            } else if (s[i] == '"') {
                r.append(2 * bs, '\\');               // run before embedded "
                r += '\\';
                r += '"';
                ++i;
            } else {
                r.append(bs, '\\');                   // run before ordinary char
                r += s[i];
                ++i;
            }
        }
        r += '"';
        return r;
    };
    // Bounce the spawn through cmd.exe's `start` builtin instead of
    // calling CreateProcessA on the exe directly. Rationale:
    //
    // Modern launcher hosts (CLion's run/debug, Visual Studio, Windows
    // Terminal, VS Code's integrated terminal) wrap their child
    // processes in a job object configured with
    // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and -- critically for the
    // CLion case -- WITHOUT JOB_OBJECT_LIMIT_BREAKAWAY_OK. A direct
    // CreateProcessA spawn inherits that job, and the previous attempt
    // (CREATE_BREAKAWAY_FROM_JOB with a fallback retry on
    // ERROR_ACCESS_DENIED) silently fell back to a non-breakaway spawn
    // when the host forbade breakaway. The new process then lived for
    // a few frames until this process exited and the host's job-close
    // killed every other process in the job, including the freshly-
    // spawned child. From the user's perspective: child window opens,
    // runs for a moment, then disappears silently.
    //
    // The shell-detach trick: spawn cmd.exe with a one-shot
    // `start "" "<exe>" <args>` command. cmd's internal `start`
    // builtin uses ShellExecuteEx, which routes through the shell
    // dispatch path (explorer.exe / sechost) and produces a process
    // tree where the target is a child of the shell, NOT of cmd or
    // us. The target is therefore OUTSIDE the host's job entirely
    // and survives this process's exit unconditionally.
    //
    // cmd.exe itself does inherit the host's job and dies with it,
    // but it's a transient shim that exits within milliseconds after
    // kicking off start -- by the time the host kills its job, cmd
    // is already gone. CREATE_NO_WINDOW hides the cmd console
    // (which would otherwise flash briefly); the spawned demont
    // gets its own fresh console window via the default
    // ShellExecuteEx SW_SHOWNORMAL behaviour for console-subsystem
    // apps, so stdio still has a destination.
    //
    // The empty "" before <exe> is start's window-title parameter
    // (required when the path is itself quoted -- otherwise start
    // interprets the quoted path AS the title and treats subsequent
    // tokens as the command).

    char sys_dir[MAX_PATH] = {0};
    UINT sd_len = GetSystemDirectoryA(sys_dir, MAX_PATH);
    if (sd_len == 0 || sd_len >= MAX_PATH) {
        LOG_ERROR("Engine::RestartProcess: GetSystemDirectoryA failed (GLE={}, len={})", GetLastError(), sd_len);
        return false;
    }
    std::string cmd_exe_path = std::string(sys_dir) + "\\cmd.exe";

    // lpCommandLine for cmd.exe. argv[0] is cmd.exe by convention.
    // The /c flag tells cmd to run the supplied command then exit.
    // We quote the start-target with the same Colascione algorithm
    // (start.exe passes its argv through CommandLineToArgvW to the
    // target), and prepend an empty "" for start's title slot.
    std::string shell_cmdline = "cmd.exe /c start \"\" " + quote(exe_path);
    for (int i = 1; i < argc_ && argv_ != nullptr; ++i) {
        if (argv_[i] == nullptr) break;
        shell_cmdline += ' ';
        shell_cmdline += quote(argv_[i]);
    }
    std::vector<char> cmdbuf(shell_cmdline.begin(), shell_cmdline.end());
    cmdbuf.push_back('\0');

    STARTUPINFOA       si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    LOG_INFO("Engine::RestartProcess: shell-detaching via '{}'", shell_cmdline);

    BOOL ok = CreateProcessA(
        cmd_exe_path.c_str(), // lpApplicationName
        cmdbuf.data(),        // lpCommandLine (writable)
        nullptr,              // lpProcessAttributes
        nullptr,              // lpThreadAttributes
        FALSE,                // bInheritHandles
        CREATE_NO_WINDOW,     // hide cmd's transient console
        nullptr,              // lpEnvironment (inherit)
        nullptr,              // lpCurrentDirectory (inherit)
        &si, &pi);
    if (!ok) {
        LOG_ERROR("Engine::RestartProcess: CreateProcessA for cmd.exe failed (GLE={})", GetLastError());
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    LOG_INFO("Engine::RestartProcess: cmd-shim PID={} spawned; target will be a child of the shell (outside host's job); setting wants_quit_", pi.dwProcessId);
    wants_quit_ = true;
    return true;
#else
    LOG_ERROR("Engine::RestartProcess: Win32-only (called on a non-Win32 build)");
    return false;
#endif
}

void Engine::SeedDefaultCsgScene() {
    if (!csg_scene_) return;
    csg_scene_->Reset();                                 // unit box at id 1, root
    csg_scene_->AddSphere(2, 0.6f, 48, 0.0f, 0.5f, 0.0f); // sphere at box centre
    csg_scene_->Combine(3, pt::csg::OpType::Subtract, 1, 2); // box - sphere
    csg_scene_->SetRoot(3);
}

void Engine::EnsureMeshUpdated() {
    if (!device_) return;
    // Mesh CSG rendering works on every backend now, not just those
    // exposing hardware ray tracing. The fast path (Metal Ray Tracing,
    // Vulkan VK_KHR_ray_query, Embree on the software backend) builds
    // a BLAS/TLAS in CreateBLAS/CreateTLAS; the slow-fallback path
    // (e.g. Mac-Vulkan on MoltenVK builds without VK_KHR_ray_query --
    // anything pre-MoltenVK 1.3) uploads the baked vertex + index
    // buffer and lets PathTrace.slang linear-scan triangles via the
    // mesh_tri_count > 0 branch. RebuildMeshResources chooses the
    // path automatically based on device_->SupportsHardwareRT().
    if (!csg_scene_) return;

    // Phase 2: the worker has a result waiting. Pull it onto the main
    // thread, free the job handle, swap in fresh GPU resources.
    if (bake_phase_.load(std::memory_order_acquire) == 2) {
        if (jobs_ && bake_handle_.internal != nullptr) {
            jobs_->Wait(bake_handle_);
        }
        bake_handle_ = {};
        std::unique_ptr<pt::csg::BakedMesh> baked = std::move(pending_baked_);
        bake_phase_.store(0, std::memory_order_release);
        if (baked && !baked->Empty()) {
            RebuildMeshResources(*baked);
            accum_dirty_ = true;
        } else {
            // Bake produced empty geometry (degenerate CSG). Drop any
            // existing mesh resources so r_mode mesh falls back to clear
            // rather than rendering a stale shape.
            if (scene_tlas_id_ != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
            if (box_blas_id_   != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
            if (box_vbuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
            if (box_ibuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
            if (tri_bvh_nodes_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_nodes_id_});
            if (tri_bvh_permuted_ids_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_permuted_ids_id_});
            scene_tlas_id_ = box_blas_id_ = box_vbuf_id_ = box_ibuf_id_ = 0;
            tri_bvh_nodes_id_ = tri_bvh_permuted_ids_id_ = 0;
            tri_bvh_node_count_ = 0;
            tri_bvh_ = pt::renderer::TriangleBvh{};
            mesh_tri_count_ = 0;
        }
    }

    // Phase 0: idle. If the scene is dirty OR the engine forced a
    // rebuild (backend switch destroyed the device-side BLAS/TLAS but
    // didn't touch the CSG tree itself), kick a fresh bake. Ack the
    // scene clean BEFORE submitting so a mutation during the bake re-marks
    // it dirty -- we'll then bake again on the next ready->idle transition.
    if (bake_phase_.load(std::memory_order_acquire) == 0 &&
        (csg_scene_->Dirty() || force_mesh_rebuild_) && jobs_) {
        force_mesh_rebuild_ = false;
        csg_scene_->AcknowledgeClean();
        bake_phase_.store(1, std::memory_order_release);
        bake_handle_ = jobs_->Submit([this] {
            std::string err;
            auto baked = std::make_unique<pt::csg::BakedMesh>(
                csg_scene_->Bake(&err));
            if (baked->Empty() && !err.empty()) {
                LOG_WARN("CSG bake: {}", err);
            }
            pending_baked_ = std::move(baked);
            bake_phase_.store(2, std::memory_order_release);
        });
    }
}

void Engine::RebuildMeshResources(const pt::csg::BakedMesh& baked) {
    PT_ZONE_SCOPED_N("Engine::RebuildMeshResources");
    if (!device_) return;

    // Drain any in-flight GPU work so it's safe to destroy old resources.
    device_->WaitIdle();

    if (scene_tlas_id_ != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
    if (box_blas_id_   != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
    if (box_vbuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
    if (box_ibuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
    // PR #106 follow-up: triangle BVH SSBOs live alongside vbuf/ibuf
    // and follow the same lifetime -- a new bake invalidates the
    // previous tree, so destroy + reallocate. Same WaitIdle() above
    // covers the prior dispatches that may still reference these.
    if (tri_bvh_nodes_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_nodes_id_});
    if (tri_bvh_permuted_ids_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{tri_bvh_permuted_ids_id_});
    scene_tlas_id_ = box_blas_id_ = box_vbuf_id_ = box_ibuf_id_ = 0;
    tri_bvh_nodes_id_ = tri_bvh_permuted_ids_id_ = 0;
    tri_bvh_node_count_ = 0;
    tri_bvh_ = pt::renderer::TriangleBvh{};
    mesh_tri_count_ = 0;

    const std::size_t vbytes = sizeof(float) * baked.positions.size();
    const std::size_t ibytes = sizeof(std::uint32_t) * baked.indices.size();
    auto vbuf = device_->CreateBuffer({
        .size = vbytes, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "csg_vbuf",
    });
    auto ibuf = device_->CreateBuffer({
        .size = ibytes, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "csg_ibuf",
    });
    if (vbuf.id == 0 || ibuf.id == 0) {
        LOG_ERROR("CSG: failed to allocate vertex/index storage buffers");
        return;
    }
    device_->WriteBuffer(vbuf, baked.positions.data(), vbytes);
    device_->WriteBuffer(ibuf, baked.indices.data(),   ibytes);
    box_vbuf_id_ = vbuf.id;
    box_ibuf_id_ = ibuf.id;
    // Set mesh_tri_count_ BEFORE the AS build below. If BLAS/TLAS fails
    // on an RT-capable backend, the early-returns leave vbuf/ibuf
    // uploaded; with mesh_tri_count_ already set, the next frame's
    // `sw_mesh_present` gate fires and the shader's SW linear-scan path
    // renders the mesh instead of silently dropping it. Without this
    // ordering, an AS failure would leave the engine in a half-built
    // state -- buffers uploaded but invisible because neither HW nor SW
    // path triggers.
    mesh_tri_count_ = baked.TriangleCount();

    // Build the triangle BVH on this thread (the same thread the CSG
    // bake job ran on, or whichever main-loop tick consumed the bake
    // result -- both cases are equally OK here, the build is pure
    // CPU work bounded by O(N log N) on triangle count). Uploaded to
    // two storage buffers; the shader's SW mesh path walks them in
    // place of the previous O(N) Möller-Trumbore loop. PR #106
    // follow-up; see comment on tri_bvh_ in Engine.h.
    tri_bvh_.Build(baked.positions, baked.indices);
    if (!tri_bvh_.Empty()) {
        const auto& nodes        = tri_bvh_.Nodes();
        const auto& permuted_ids = tri_bvh_.PermutedPrimIds();
        const std::size_t nodes_bytes    = nodes.size() * sizeof(pt::renderer::TriBvhNode);
        const std::size_t permuted_bytes = permuted_ids.size() * sizeof(std::uint32_t);
        auto nbuf = device_->CreateBuffer({
            .size       = nodes_bytes,
            .usage      = pt::rhi::BufferUsage::Storage,
            .debug_name = "tri_bvh_nodes",
        });
        auto pbuf = device_->CreateBuffer({
            .size       = permuted_bytes,
            .usage      = pt::rhi::BufferUsage::Storage,
            .debug_name = "tri_bvh_permuted_ids",
        });
        if (nbuf.id != 0 && pbuf.id != 0) {
            device_->WriteBuffer(nbuf, nodes.data(),        nodes_bytes);
            device_->WriteBuffer(pbuf, permuted_ids.data(), permuted_bytes);
            tri_bvh_nodes_id_        = nbuf.id;
            tri_bvh_permuted_ids_id_ = pbuf.id;
            tri_bvh_node_count_      = static_cast<std::uint32_t>(nodes.size());
            LOG_INFO("CSG: tri-BVH built ({} tris -> {} nodes, {} permuted ids)",
                     baked.TriangleCount(), nodes.size(), permuted_ids.size());
        } else {
            // Allocation failure: log + fall back to a no-BVH state.
            // The shader gates the SW BVH walk on bvh_params.w > 0,
            // so this is benign -- the SW mesh path drops out for this
            // frame (mesh stays invisible). Beats the previous-PR
            // O(N) linear scan still being there as a fallback because
            // we removed it.
            LOG_ERROR("CSG: tri-BVH SSBO allocation failed ({} tris); SW mesh path disabled this bake",
                      baked.TriangleCount());
            if (nbuf.id != 0) device_->DestroyBuffer(nbuf);
            if (pbuf.id != 0) device_->DestroyBuffer(pbuf);
            tri_bvh_ = pt::renderer::TriangleBvh{};
        }
    }

    // BLAS/TLAS build is conditional on hardware ray tracing. When the
    // backend can't host an AS (Mac-Vulkan on pre-1.3 MoltenVK is the
    // motivating case -- it doesn't expose VK_KHR_acceleration_structure),
    // the engine still uploads the vertex + index buffers above and lets
    // PathTrace.slang linear-scan triangles via the mesh_tri_count > 0
    // branch. Performance scales linearly with TriangleCount() on that
    // path, which is fine for sub-1k-tri scenes like the headline CSG
    // demos but should not be relied on for production-size meshes --
    // see the one-shot warning below.
    if (device_->SupportsHardwareRT()) {
        pt::rhi::BLASDesc blas_desc{
            .vertex_positions = baked.positions.data(),
            .vertex_count     = baked.VertexCount(),
            .indices          = baked.indices.data(),
            .index_count      = static_cast<std::uint32_t>(baked.indices.size()),
            .debug_name       = "csg",
        };
        auto blas = device_->CreateBLAS(blas_desc);
        if (blas.id == 0) {
            LOG_ERROR("CSG: BLAS build failed ({} verts, {} tris); "
                      "falling back to SW linear-scan path",
                      baked.VertexCount(), baked.TriangleCount());
            return;
        }
        box_blas_id_ = blas.id;

        // Single instance, identity transform -- the CSG bake already places
        // geometry in world space.
        pt::rhi::TLASInstance inst{};
        inst.blas = blas;
        inst.transform[0] = 1; inst.transform[1] = 0; inst.transform[2] = 0; inst.transform[3]  = 0.0f;
        inst.transform[4] = 0; inst.transform[5] = 1; inst.transform[6] = 0; inst.transform[7]  = 0.0f;
        inst.transform[8] = 0; inst.transform[9] = 0; inst.transform[10]= 1; inst.transform[11] = 0.0f;
        inst.instance_id = 0;
        inst.mask        = 0xFF;
        std::vector<pt::rhi::TLASInstance> insts{inst};
        pt::rhi::TLASDesc tlas_desc{ .instances = insts, .debug_name = "scene" };
        auto tlas = device_->CreateTLAS(tlas_desc);
        scene_tlas_id_ = tlas.id;
        if (scene_tlas_id_ == 0) {
            LOG_ERROR("CSG: TLAS build failed; tearing down BLAS and "
                      "falling back to SW linear-scan path");
            device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
            box_blas_id_ = 0;
            return;
        }
        LOG_INFO("CSG: rebuilt mesh ({} verts, {} tris) -- hw-rt BLAS+TLAS",
                 baked.VertexCount(), baked.TriangleCount());
    } else {
        // Software-mesh fallback: vbuf + ibuf are already uploaded above
        // and the triangle BVH was built / uploaded a few lines up.
        // PathTrace.slang walks the BVH for triangles when bvh_params.z
        // is non-zero (mesh present) and the HW RT path is off.
        // Replaces the previous O(N) Möller-Trumbore linear scan PR
        // #106 shipped -- the linear scan crushed framerate to ~1 FPS
        // at 1080p on a 1k-tri scene; the BVH walk is O(log N) and
        // brings interactivity back.
        LOG_INFO("CSG: rebuilt mesh ({} verts, {} tris) -- software path (host-built tri-BVH, no hw RT)",
                 baked.VertexCount(), baked.TriangleCount());

        // The previous perf-cliff warning (~4k tris on the SW linear
        // scan) no longer applies now that the SW path uses a real
        // BVH. The walker's runtime cost scales O(log N), so the
        // 1080p / 6-bounce / 1k-tri case that motivated PR #106 sits
        // around ~30 FPS instead of ~1 FPS on Mac-Vulkan -- the
        // empirical knee where SW becomes uncomfortable is closer to
        // ~100k tris now (mostly bounded by AABB-fetch bandwidth, not
        // intersect count). We keep the one-shot guard variable
        // declared in Engine.h so a future "warn at 100k+ tris on SW
        // backend" check can drop in here without re-plumbing.
        (void)sw_mesh_perf_warning_fired_;
    }
}

void Engine::SeedDefaultPrimitives() {
    primitives_.clear();
    auto rgb = [](float r, float g, float b) {
        return std::array<float, 3>{r, g, b};
    };
    auto add_sphere = [&](std::uint32_t id, float x, float y, float z, float r,
                          AnalyticPrim::Material mat,
                          std::array<float, 3> color, float roughness, float ior) {
        AnalyticPrim p{};
        p.type        = AnalyticPrim::Sphere;
        p.material    = mat;
        p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
        // Motion blur (#85): seed prev_pos == curr_pos so a newly-spawned
        // body produces a zero-length lerp on its first rendered frame
        // (no spurious streak from an uninitialized prev_pos).
        p.prev_pos_or_n[0] = x; p.prev_pos_or_n[1] = y; p.prev_pos_or_n[2] = z;
        p.radius_or_d = r;
        p.albedo[0]   = color[0]; p.albedo[1] = color[1]; p.albedo[2] = color[2];
        p.roughness   = roughness;
        p.ior         = ior;
        primitives_[id] = p;
    };
    auto add_plane = [&](std::uint32_t id, float nx, float ny, float nz, float d,
                         AnalyticPrim::Material mat, std::array<float, 3> color) {
        AnalyticPrim p{};
        p.type        = AnalyticPrim::Plane;
        p.material    = mat;
        p.pos_or_n[0] = nx; p.pos_or_n[1] = ny; p.pos_or_n[2] = nz;
        // Planes don't move (infinite extent); prev_pos == pos_or_n
        // keeps the GPU layout uniform without special-casing.
        p.prev_pos_or_n[0] = nx; p.prev_pos_or_n[1] = ny; p.prev_pos_or_n[2] = nz;
        p.radius_or_d = d;
        p.albedo[0]   = color[0]; p.albedo[1] = color[1]; p.albedo[2] = color[2];
        p.roughness   = 0.0f;
        p.ior         = 1.0f;
        primitives_[id] = p;
    };
    add_sphere(1, -1.2f, 0.5f,  0.2f, 0.5f, AnalyticPrim::Lambert,    rgb(0.92f, 0.30f, 0.30f), 0.0f,  1.0f);
    add_sphere(2,  0.0f, 0.5f,  0.0f, 0.5f, AnalyticPrim::Metal,      rgb(1.00f, 0.85f, 0.45f), 0.05f, 1.0f);
    add_sphere(3,  1.2f, 0.5f, -0.2f, 0.5f, AnalyticPrim::Dielectric, rgb(1.00f, 1.00f, 1.00f), 0.0f,  1.5f);
    add_plane (4,  0.0f, 1.0f,  0.0f, 0.0f, AnalyticPrim::Lambert,    rgb(0.55f, 0.55f, 0.55f));
    primitives_dirty_ = true;
    accum_dirty_      = true;
}

void Engine::ReloadEnvMap(const std::string& path) {
    PT_ZONE_SCOPED_N("Engine::ReloadEnvMap");
    if (!device_) {
        // Defer: cvar set before backend is up. Stash the path and apply
        // on the next RequestBackendSwitch.
        env_map_path_ = path;
        return;
    }

    auto destroy_env = [this]() {
        if (env_map_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
        if (env_marginal_cdf_id_      != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_marginal_cdf_id_});
        if (env_conditional_cdf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_conditional_cdf_id_});
        env_map_tex_id_         = 0;
        env_marginal_cdf_id_    = 0;
        env_conditional_cdf_id_ = 0;
        env_total_luminance_    = 0.0f;
        hdri_lights_            = {};
        hdri_lights_count_      = 0;
    };

    device_->WaitIdle();
    destroy_env();

    env_map_path_ = path;
    if (path.empty()) {
        accum_dirty_ = true;     // sky changed -> pixel values change
        return;
    }

    std::string err;
    auto img = pt::renderer::LoadRadianceHdr(pt::ResolveAssetPath(path.c_str()), &err);
    if (img.Empty()) {
        LOG_WARN("env_map: failed to load '{}': {}", path, err);
        accum_dirty_ = true;
        return;
    }

    const std::uint32_t W = img.width, H = img.height;

    // === HDRI multi-light extraction ===
    // Threshold the HDRI at the top 0.5% luminance percentile, then run
    // 4-connected flood-fill (with horizontal wrap for lat-long) on the
    // mask. Each connected component is a candidate light. Keep the top
    // kMaxHdriLights by integrated flux; mask all kept clusters out of
    // the env-map CDF so env-map NEE doesn't double-count them. The
    // path tracer then does a stochastic directional NEE picking one
    // light per sample weighted by cluster luminance -- single-sun
    // outdoor HDRI gives one big cluster, night HDRI gives one moon,
    // interior gives one cluster per lamp, all handled identically.
    std::vector<std::uint8_t> light_mask(std::size_t(W) * H, 0u);
    {
        // Luminance per pixel + percentile threshold.
        const std::size_t N_pix = std::size_t(W) * H;
        std::vector<float> lums(N_pix);
        float peak_lum = 0.0f;
        for (std::size_t pi = 0; pi < N_pix; ++pi) {
            const float r = img.rgb[pi * 3 + 0];
            const float g = img.rgb[pi * 3 + 1];
            const float b = img.rgb[pi * 3 + 2];
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            lums[pi] = lum;
            if (lum > peak_lum) peak_lum = lum;
        }

        // Skip extraction on effectively black HDRIs -- a flood-fill on
        // a uniform field would just pick noise. Anything below 0.5
        // raw nits is "image is dim, no real lights detectable".
        constexpr float kMinPeakLum = 0.5f;
        if (peak_lum > kMinPeakLum) {
            // Top-N% percentile threshold via partial-sort. The default
            // 0.5% is sun-tuned but works for indoor HDRIs too because
            // it scales with whatever the brightest content actually is
            // (vs the prior fixed 50%-of-peak rule which only fired
            // when peak luminance was sun-class). Driver: r_hdri_
            // extract_percentile cvar, clamped to (0, 0.5] so the
            // threshold stays at a meaningful "bright" tail.
            double top_frac = 0.005;
            if (auto* v = pt::console::Console::Get().FindCVar("r_hdri_extract_percentile")) {
                top_frac = double(v->GetFloat());
                if (top_frac < 1e-6) top_frac = 1e-6;
                if (top_frac > 0.5)  top_frac = 0.5;
            }
            const std::size_t k_idx = N_pix - std::max<std::size_t>(
                1, std::size_t(double(N_pix) * top_frac));
            std::vector<float> lums_sorted = lums;  // nth_element mutates
            std::nth_element(lums_sorted.begin(),
                             lums_sorted.begin() + k_idx,
                             lums_sorted.end());
            float thresh = lums_sorted[k_idx];
            // Floor to avoid clustering noise on dim HDRIs where the
            // 99.5%ile lands below background.
            if (thresh < 0.05f * peak_lum) thresh = 0.05f * peak_lum;
            if (thresh < kMinPeakLum)      thresh = kMinPeakLum;

            for (std::size_t pi = 0; pi < N_pix; ++pi) {
                if (lums[pi] >= thresh) light_mask[pi] = 1u;
            }
        }
    }

    // Connected-components on light_mask. Lat-long wraps horizontally
    // (column 0 is adjacent to column W-1) but not vertically. BFS via
    // a vector stack keeps memory bounded; for typical HDRIs the masked
    // set is < 1% of pixels so this is fast.
    constexpr double kTwoPiSq = 2.0 * std::numbers::pi * std::numbers::pi;
    const double dOmega_unit = kTwoPiSq / double(std::size_t(W) * H);
    struct ClusterAcc {
        glm::dvec3 centroid_dir_acc{0.0};
        double     centroid_w_acc = 0.0;
        glm::dvec3 flux_acc{0.0};
        double     lum_acc = 0.0;       // ∫_cluster L dΩ -- used for sort + pmf
        std::uint32_t pixels = 0;
    };
    std::vector<ClusterAcc> clusters;
    std::vector<std::uint32_t> cluster_id(std::size_t(W) * H, 0u);  // 0 = unvisited
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bfs_stack;
    bfs_stack.reserve(1024);
    auto pix_idx = [&](std::uint32_t u, std::uint32_t v) -> std::size_t {
        return std::size_t(v) * W + u;
    };
    for (std::uint32_t v0 = 0; v0 < H; ++v0) {
        for (std::uint32_t u0 = 0; u0 < W; ++u0) {
            const std::size_t pi0 = pix_idx(u0, v0);
            if (!light_mask[pi0] || cluster_id[pi0] != 0u) continue;
            const std::uint32_t cid = std::uint32_t(clusters.size()) + 1u;
            clusters.emplace_back();
            ClusterAcc& acc = clusters.back();
            bfs_stack.clear();
            bfs_stack.emplace_back(u0, v0);
            cluster_id[pi0] = cid;
            while (!bfs_stack.empty()) {
                auto [u, v] = bfs_stack.back();
                bfs_stack.pop_back();
                const std::size_t pi = pix_idx(u, v);
                const double theta   = std::numbers::pi * (double(v) + 0.5) / double(H);
                const double sint    = std::sin(theta);
                const double cost    = std::cos(theta);
                const double dOmega  = dOmega_unit * sint;
                const double phi     = 2.0 * std::numbers::pi * (double(u) + 0.5) / double(W)
                                     - std::numbers::pi;
                const float  r = img.rgb[pi * 3 + 0];
                const float  g = img.rgb[pi * 3 + 1];
                const float  b = img.rgb[pi * 3 + 2];
                const double lum  = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                const double wdir = lum * dOmega;
                acc.centroid_dir_acc += glm::dvec3(sint * std::cos(phi), cost,
                                                   sint * std::sin(phi)) * wdir;
                acc.centroid_w_acc   += wdir;
                acc.flux_acc         += glm::dvec3(double(r), double(g), double(b)) * dOmega;
                acc.lum_acc          += lum * dOmega;
                acc.pixels           += 1u;
                const std::uint32_t um = (u == 0u)     ? (W - 1u) : (u - 1u);
                const std::uint32_t up = (u + 1u >= W) ? 0u       : (u + 1u);
                auto try_push = [&](std::uint32_t uu, std::uint32_t vv) {
                    const std::size_t pp = pix_idx(uu, vv);
                    if (light_mask[pp] && cluster_id[pp] == 0u) {
                        cluster_id[pp] = cid;
                        bfs_stack.emplace_back(uu, vv);
                    }
                };
                try_push(um, v);
                try_push(up, v);
                if (v > 0)      try_push(u, v - 1u);
                if (v + 1u < H) try_push(u, v + 1u);
            }
        }
    }

    // Index-sort clusters by integrated luminance descending so we
    // know which original cluster_ids survive (cluster_id references
    // the discovery-order index, not the sorted position). Keep top
    // kMaxHdriLights and drop tiny noise clusters (< 3 px).
    std::vector<std::uint32_t> order(clusters.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            return clusters[a].lum_acc > clusters[b].lum_acc;
        });
    std::vector<std::uint8_t> keep_id(clusters.size() + 1u, 0u);   // 1-based
    hdri_lights_       = {};
    hdri_lights_count_ = 0;
    double total_lum = 0.0;
    for (std::uint32_t idx : order) {
        if (hdri_lights_count_ >= Engine::kMaxHdriLights) break;
        const ClusterAcc& c = clusters[idx];
        if (c.pixels < 3u || c.centroid_w_acc <= 0.0) continue;
        keep_id[idx + 1u] = 1u;
        HdriLight& L = hdri_lights_[hdri_lights_count_++];
        L.dir        = glm::vec3(glm::normalize(c.centroid_dir_acc));
        L.irradiance = glm::vec3(c.flux_acc);
        L.luminance  = float(c.lum_acc);
        total_lum   += c.lum_acc;
    }
    if (total_lum > 0.0) {
        const float inv_total = float(1.0 / total_lum);
        for (std::uint32_t i = 0; i < hdri_lights_count_; ++i) {
            hdri_lights_[i].pmf = hdri_lights_[i].luminance * inv_total;
        }
    }
    // Pixels in clusters we DIDN'T keep get unmasked from light_mask
    // so env-map NEE can still pick up that minor light via the
    // importance CDF (just no crisp directional shadow for those).
    for (std::size_t pi = 0; pi < std::size_t(W) * H; ++pi) {
        const std::uint32_t cid = cluster_id[pi];
        if (cid != 0u && !keep_id[cid]) light_mask[pi] = 0u;
    }
    if (hdri_lights_count_ > 0) {
        std::string msg = fmt::format(
            "env_map: extracted {} HDRI light(s) from '{}':",
            hdri_lights_count_, path);
        for (std::uint32_t i = 0; i < hdri_lights_count_; ++i) {
            const HdriLight& L = hdri_lights_[i];
            msg += fmt::format(
                "\n  [{}] dir=({:.2f},{:.2f},{:.2f}) flux=({:.1f},{:.1f},{:.1f}) pmf={:.3f}",
                i, L.dir.x, L.dir.y, L.dir.z,
                L.irradiance.r, L.irradiance.g, L.irradiance.b, L.pmf);
        }
        LOG_INFO("{}", msg);
    }

    // Repack RGB -> RGBA + compute luminance per pixel for the CDF.
    // Pixels in any kept extracted cluster are zeroed in the CDF
    // luminance only (so env-map NEE skips them -- those clusters are
    // handled by the directional NEE in the shader). The env_map
    // texture data itself keeps the bright pixels intact so camera-
    // direct rays still see the visible bright pixels in the sky.
    std::vector<float> rgba(std::size_t(W) * H * 4);
    std::vector<float> conditional(std::size_t(W) * H);     // CDF per row
    std::vector<float> marginal(H);                          // CDF over rows
    double total = 0.0;
    for (std::uint32_t v = 0; v < H; ++v) {
        const double sin_theta = std::sin(std::numbers::pi * (double(v) + 0.5) / double(H));
        double row_sum = 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            const std::size_t pi = std::size_t(v) * W + u;
            float r = img.rgb[pi * 3 + 0];
            float g = img.rgb[pi * 3 + 1];
            float b = img.rgb[pi * 3 + 2];
            rgba[pi * 4 + 0] = r;
            rgba[pi * 4 + 1] = g;
            rgba[pi * 4 + 2] = b;
            rgba[pi * 4 + 3] = 1.0f;
            const float lum_raw = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float lum = light_mask[pi] ? 0.0f : lum_raw;
            const double weight = double(lum) * sin_theta;
            row_sum += weight;
            conditional[pi] = float(row_sum);   // unnormalized prefix sum within row
        }
        // Normalize within row to [0, 1].
        const double norm = (row_sum > 0.0) ? (1.0 / row_sum) : 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            conditional[std::size_t(v) * W + u] = float(double(conditional[std::size_t(v) * W + u]) * norm);
        }
        marginal[v] = float(row_sum);
        total += row_sum;
    }
    // Marginal: prefix-sum + normalize so marginal[H-1] == 1.0.
    {
        const double norm = (total > 0.0) ? (1.0 / total) : 0.0;
        double prefix = 0.0;
        for (std::uint32_t v = 0; v < H; ++v) {
            prefix += marginal[v];
            marginal[v] = float(prefix * norm);
        }
        if (H > 0) marginal[H - 1] = 1.0f;       // guard against fp drift
    }
    env_total_luminance_ = float(total);

    auto tex = device_->CreateTexture({
        .width  = W, .height = H,
        .format = pt::rhi::TextureFormat::RGBA32F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "env_map",
    });
    if (tex.id == 0 || !device_->WriteTexture(tex, rgba.data(), rgba.size() * sizeof(float))) {
        LOG_ERROR("env_map: texture create/upload failed ({}x{})", W, H);
        if (tex.id != 0) device_->DestroyTexture(tex);
        return;
    }
    env_map_tex_id_ = tex.id;

    auto m_buf = device_->CreateBuffer({
        .size = sizeof(float) * H, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "env_marginal_cdf",
    });
    auto c_buf = device_->CreateBuffer({
        .size = sizeof(float) * std::size_t(W) * H, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "env_conditional_cdf",
    });
    if (m_buf.id == 0 || c_buf.id == 0) {
        LOG_ERROR("env_map: CDF buffer creation failed");
        device_->DestroyTexture(tex); env_map_tex_id_ = 0;
        if (m_buf.id != 0) device_->DestroyBuffer(m_buf);
        if (c_buf.id != 0) device_->DestroyBuffer(c_buf);
        return;
    }
    device_->WriteBuffer(m_buf, marginal.data(),    sizeof(float) * H);
    device_->WriteBuffer(c_buf, conditional.data(), sizeof(float) * std::size_t(W) * H);
    env_marginal_cdf_id_    = m_buf.id;
    env_conditional_cdf_id_ = c_buf.id;

    accum_dirty_ = true;
    LOG_INFO("env_map: loaded {} ({}x{} HDR, total luminance {:.2f})", path, W, H, env_total_luminance_);
}

void Engine::EnsureMoonMapUploaded() {
    if (!device_) return;
    if (moon_map_tex_id_ != 0) return;
    PT_ZONE_SCOPED_N("Engine::EnsureMoonMapUploaded");
    // 512x256 = ~1 MB at RGBA16F. The moon disc is tiny on screen
    // (~9-15 px at default r_moon_size), so anything past a 256x
    // near-side coverage is sampled-down to invisibility -- 2K was
    // overkill. Bigger sizes via r_moon_size still resolve fine
    // because the texture's mare features are LOW-frequency, well
    // above one texel of detail at this resolution.
    constexpr std::uint32_t kW = 512;
    constexpr std::uint32_t kH = 256;
    std::vector<float> rgba;
    pt::moon::generateMoonTexture(int(kW), int(kH), rgba);
    auto tex = device_->CreateTexture({
        .width  = kW, .height = kH,
        .format = pt::rhi::TextureFormat::RGBA16F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "moon_map",
    });
    if (tex.id == 0) {
        LOG_WARN("moonmap: texture create failed");
        return;
    }
    // Pack to half precision (same path as the BSC starmap upload).
    std::vector<std::uint16_t> half(rgba.size());
    auto f32_to_f16 = [](float f) -> std::uint16_t {
        std::uint32_t u; std::memcpy(&u, &f, sizeof(u));
        std::uint32_t sign = (u >> 16) & 0x8000;
        std::int32_t  exp  = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mant = u & 0x7FFFFF;
        if (exp <= 0)  return static_cast<std::uint16_t>(sign);
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00);
        return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
    };
    for (std::size_t i = 0; i < rgba.size(); ++i) half[i] = f32_to_f16(rgba[i]);
    if (!device_->WriteTexture(tex, half.data(), half.size() * sizeof(std::uint16_t))) {
        LOG_WARN("moonmap: texture upload failed");
        device_->DestroyTexture(tex);
        return;
    }
    moon_map_tex_id_ = tex.id;
    LOG_INFO("moonmap: procedural lunar surface generated, {}x{} RGBA16F", kW, kH);
}

void Engine::EnsureStarMapUploaded() {
    if (!device_) return;
    if (star_map_tex_id_ != 0) return;          // already uploaded on this device
    PT_ZONE_SCOPED_N("Engine::EnsureStarMapUploaded");

    constexpr const char*  kPath = "assets/stars/BSC5.dat";
    // 8192x4096 RGBA16F = 256MB. Trades VRAM for star crispness:
    // each star's pixel splat covers fewer screen pixels at typical
    // FOVs, so the bilinear sampling doesn't blur points into puffs.
    // Earlier 4096x2048 produced visible "filter haze" around stars.
    constexpr std::uint32_t kW   = 8192;
    constexpr std::uint32_t kH   = 4096;

    std::string err;
    auto stars = pt::stars::LoadBsc5(pt::ResolveAssetPath(kPath), &err);
    if (stars.empty()) {
        LOG_WARN("starmap: BSC load failed ({}); stars disabled", err);
        star_map_present_ = 0;
        return;
    }

    std::vector<float> rgba;
    pt::stars::RasteriseJ2000Map(stars, kW, kH, rgba);

    // GPU expects RGBA16F. The rasteriser hands us float-RGBA already in
    // half-friendly range (brightest entries ~25), so we let the RHI
    // convert on upload (Metal's WriteTexture for RGBA16F packs the
    // float32 source into half-precision). If the RHI can't accept
    // float32 source for an RGBA16F texture it will fail upload and
    // we'll fall back to no-stars.
    auto tex = device_->CreateTexture({
        .width  = kW, .height = kH,
        .format = pt::rhi::TextureFormat::RGBA16F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "star_map_j2000",
    });
    if (tex.id == 0) {
        LOG_WARN("starmap: texture create failed; stars disabled");
        star_map_present_ = 0;
        return;
    }
    // RGBA16F upload: the Metal RHI's WriteTexture path expects bytes
    // matching the texture format. Pack to half manually (cheap).
    std::vector<std::uint16_t> half(rgba.size());
    auto f32_to_f16 = [](float f) -> std::uint16_t {
        // Standard IEEE-754 half conversion. Branchless path for the
        // normalised range, with a flush-to-zero fallback for tiny
        // values; star fluxes never overflow half max (~65504).
        std::uint32_t u; std::memcpy(&u, &f, sizeof(u));
        std::uint32_t sign = (u >> 16) & 0x8000;
        std::int32_t  exp  = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mant = u & 0x7FFFFF;
        if (exp <= 0)  return static_cast<std::uint16_t>(sign);                 // underflow -> 0
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00);        // overflow -> inf
        return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
    };
    for (std::size_t i = 0; i < rgba.size(); ++i) half[i] = f32_to_f16(rgba[i]);
    if (!device_->WriteTexture(tex, half.data(), half.size() * sizeof(std::uint16_t))) {
        LOG_WARN("starmap: texture upload failed; stars disabled");
        device_->DestroyTexture(tex);
        star_map_present_ = 0;
        return;
    }
    star_map_tex_id_  = tex.id;
    star_map_present_ = 1;
    // Diagnostics: how much of the rasterised map actually has flux?
    // If nonzero count is low or peak is tiny something has gone wrong
    // in the splatter (clamps, magnitude scaling, etc.).
    {
        std::size_t nonzero = 0;
        float peak = 0.0f;
        const std::size_t N = std::size_t(kW) * kH;
        for (std::size_t i = 0; i < N; ++i) {
            float r = rgba[i * 4 + 0];
            float g = rgba[i * 4 + 1];
            float b = rgba[i * 4 + 2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (lum > 0.0f) ++nonzero;
            if (lum > peak) peak = lum;
        }
        LOG_INFO("starmap: loaded {} stars (BSC5), {}x{} J2000 RGBA16F "
                 "(nonzero texels {}/{}, peak luminance {:.3f})",
                 stars.size(), kW, kH, nonzero, N, peak);
    }
}

void Engine::EnsurePrimitivesUploaded() {
    if (!device_) return;
    if (!primitives_dirty_) return;

    const std::uint32_t count = static_cast<std::uint32_t>(primitives_.size());
    // 6 float4s (24 floats) per primitive after #85 motion blur (which
    // added v3 = prev_pos_or_n for shutter-time position lerp) AND #181
    // polish (which adds v4 = emission for self-emissive rigid bodies)
    // AND #206 rotation gizmo (which adds v5 = orient quaternion so
    // analytic planes can tilt under the editor's rotate gizmo).
    // Layout:
    //   v0 = (xyz: center / normal,  w: radius / d)
    //   v1 = (rgb: albedo,           a: roughness)
    //   v2 = (x: type,  y: material, z: ior, w: user-prim id)
    //   v3 = (xyz: prev_pos,         w: pad)   -- motion blur, #85
    //   v4 = (rgb: emission W/sr,    w: pad)   -- self-emission, #181
    //   v5 = (xyzw: orient quat)               -- rotation gizmo, #206
    // Shaders use `idx * 6u` to step prims; PathTrace.slang's
    // testAnalyticPrim, RestirFinal.slang's shadowVis, and
    // SoftwareTracer.cpp's analytic-prim scan all MUST step at this
    // stride. Off-paths (identity orient, no motion blur, no emission)
    // pay only the extra load cost; rendered output is bit-identical
    // when orient == identity (the shader gates the quat-rotate on the
    // identity check).
    constexpr std::uint32_t kFloatsPerPrim = 24;   // 6 float4s
    constexpr std::uint32_t kBytesPerPrim  = sizeof(float) * kFloatsPerPrim;

    // Allocate / grow the storage buffer when needed. We grow by powers
    // of two from a 16-prim floor so steady-state edits don't reallocate.
    if (count > prim_buffer_capacity_) {
        std::uint32_t new_cap = prim_buffer_capacity_ ? prim_buffer_capacity_ : 16u;
        while (new_cap < count) new_cap *= 2u;
        if (prim_buffer_id_ != 0) {
            device_->WaitIdle();
            device_->DestroyBuffer(pt::rhi::BufferHandle{prim_buffer_id_});
            prim_buffer_id_ = 0;
        }
        auto buf = device_->CreateBuffer({
            .size = std::size_t(new_cap) * kBytesPerPrim,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "analytic_primitives",
        });
        if (buf.id == 0) {
            LOG_ERROR("primitive buffer allocation failed (capacity {})", new_cap);
            return;
        }
        prim_buffer_id_       = buf.id;
        prim_buffer_capacity_ = new_cap;
    }

    if (count == 0) {
        linear_prim_count_ = 0;
        analytic_bvh_      = pt::renderer::AnalyticBvh{};
        primitives_dirty_  = false;
        accum_dirty_       = true;
        return;
    }

    // Partition primitives into two ordered lists:
    //   - infinite (planes, ...): linear-tested at the front of the buffer.
    //   - finite   (spheres, ...): candidates for the BVH path; if the
    //     count crosses r_analytic_bvh_threshold we build a CPU BVH over
    //     them, otherwise they're appended for linear traversal.
    // PRIM_PLANE in the shader is type 1; PRIM_SPHERE is 0. The
    // AnalyticPrim::type field matches.
    constexpr std::uint32_t kPrimPlane  = 1u;

    // Track each prim's user-facing id alongside its pointer so we can
    // bake the id into v2.w of the packed GPU layout below. The editor
    // outline shader (agent-19) compares this against the
    // selected_prim_id push field; without uploading the id the shader
    // would only see the GPU-reorder index (post-BVH permutation),
    // which has no stable user-visible meaning.
    struct PrimWithId {
        const AnalyticPrim* p;
        std::uint32_t       id;
    };
    std::vector<PrimWithId> infinite_prims;
    std::vector<PrimWithId> finite_prims;
    infinite_prims.reserve(count);
    finite_prims.reserve(count);
    for (const auto& [id, p] : primitives_) {
        if (p.type == kPrimPlane) infinite_prims.push_back({&p, id});
        else                       finite_prims.push_back({&p, id});
    }

    // Decide whether to build the BVH. Default threshold is 16 finite
    // primitives -- below that the per-pixel linear loop beats BVH
    // setup + traversal overhead on GPU. The cvar lets the user tune
    // empirically per scene.
    std::uint32_t bvh_threshold = 16u;
    if (auto* v = pt::console::Console::Get().FindCVar("r_analytic_bvh_threshold")) {
        const int t = v->GetInt();
        if (t > 0) bvh_threshold = static_cast<std::uint32_t>(t);
    }

    std::vector<std::uint32_t> finite_order;       // index into finite_prims after BVH reordering
    finite_order.resize(finite_prims.size());
    for (std::uint32_t i = 0; i < finite_prims.size(); ++i) finite_order[i] = i;

    // Motion blur (#85): the BVH AABB must encompass the SWEPT VOLUME
    // of each sphere from prev_pos to curr_pos so the shader's shutter-
    // time-sampled hit test isn't culled. Inflate the bounding radius
    // by the per-frame displacement magnitude and shift center to the
    // midpoint of prev / curr. When motion blur is off (or no body
    // moved this frame) the delta is zero and the input matches the
    // pre-#85 single-frame bound exactly -- BVH topology stays bit-
    // identical for static / unmoving scenes.
    bool motion_blur_active = false;
    if (auto* v = pt::console::Console::Get().FindCVar("r_motion_blur")) {
        motion_blur_active = v->GetBool();
    }
    const bool build_bvh = static_cast<std::uint32_t>(finite_prims.size()) >= bvh_threshold;
    if (build_bvh) {
        std::vector<pt::renderer::BvhPrim> bvh_input;
        bvh_input.reserve(finite_prims.size());
        for (std::uint32_t i = 0; i < finite_prims.size(); ++i) {
            const AnalyticPrim& p = *finite_prims[i].p;
            pt::renderer::BvhPrim bp{};
            float cx = p.pos_or_n[0];
            float cy = p.pos_or_n[1];
            float cz = p.pos_or_n[2];
            float radius_inflate = 0.0f;
            if (motion_blur_active) {
                const float dx = p.pos_or_n[0] - p.prev_pos_or_n[0];
                const float dy = p.pos_or_n[1] - p.prev_pos_or_n[1];
                const float dz = p.pos_or_n[2] - p.prev_pos_or_n[2];
                const float disp = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (disp > 0.0f) {
                    // Midpoint between prev and curr; inflate radius by
                    // half the displacement so the AABB contains BOTH
                    // sphere positions (and every interpolated position
                    // in between, since the lerp travels a straight line).
                    cx = 0.5f * (p.pos_or_n[0] + p.prev_pos_or_n[0]);
                    cy = 0.5f * (p.pos_or_n[1] + p.prev_pos_or_n[1]);
                    cz = 0.5f * (p.pos_or_n[2] + p.prev_pos_or_n[2]);
                    radius_inflate = 0.5f * disp;
                }
            }
            bp.center[0] = cx;
            bp.center[1] = cy;
            bp.center[2] = cz;
            // Sphere bounding radius = radius + half-displacement when
            // motion blur is active. Future finite types (box, disk,
            // cylinder) should compute a bounding-sphere radius here
            // from their extent.
            bp.radius    = p.radius_or_d + radius_inflate;
            bp.prim_id   = i;
            bvh_input.push_back(bp);
        }
        analytic_bvh_.Build(bvh_input);
        finite_order = analytic_bvh_.PermutedPrimIds();
    } else {
        analytic_bvh_ = pt::renderer::AnalyticBvh{};
    }

    linear_prim_count_ = static_cast<std::uint32_t>(infinite_prims.size());

    // Pack into the shader buffer in two segments:
    //   [0..linear_prim_count_)               : infinite prims (planes)
    //   [linear_prim_count_..count)           : finite prims, reordered to match
    //                                            BVH leaf order (if BVH built)
    //                                            or in arbitrary order (linear path)
    std::vector<float> packed(std::size_t(count) * kFloatsPerPrim, 0.0f);
    auto pack_prim = [&](std::size_t out_idx, const AnalyticPrim& p,
                         std::uint32_t user_id) {
        std::size_t off = out_idx * kFloatsPerPrim;
        // v0 -- xyz: center / normal,  w: radius / d
        packed[off + 0]  = p.pos_or_n[0];
        packed[off + 1]  = p.pos_or_n[1];
        packed[off + 2]  = p.pos_or_n[2];
        packed[off + 3]  = p.radius_or_d;
        // v1 -- rgb: albedo, a: roughness
        packed[off + 4]  = p.albedo[0];
        packed[off + 5]  = p.albedo[1];
        packed[off + 6]  = p.albedo[2];
        packed[off + 7]  = p.roughness;
        // v2 -- x: type, y: material, z: ior, w: user prim id (uint
        // reinterpreted as float; recovered shader-side via asuint()).
        // The id was previously an unused 0.0f pad; the editor backend
        // (agent-19) repurposed it so PathTrace.slang can compare a
        // hit's prim id against editor_params.x for the outline gate.
        // Same asuint() pattern bvh_nodes.w lanes already use.
        packed[off + 8]  = static_cast<float>(p.type);
        packed[off + 9]  = static_cast<float>(p.material);
        packed[off + 10] = p.ior;
        {
            float id_as_float;
            std::memcpy(&id_as_float, &user_id, sizeof(float));
            packed[off + 11] = id_as_float;
        }
        // v3 -- xyz: prev_pos (motion blur, #85), w: pad. The shader
        // reads this when r_motion_blur != 0 to lerp the sphere center
        // at the ray's shutter time. Layout collapses to a no-op when
        // motion blur is off (load happens, lerp factor forced to 1.0).
        packed[off + 12] = p.prev_pos_or_n[0];
        packed[off + 13] = p.prev_pos_or_n[1];
        packed[off + 14] = p.prev_pos_or_n[2];
        packed[off + 15] = 0.0f;
        // v4 -- xyz: emission (W/sr per channel, #181 polish), w: pad.
        // Default zero; PathTrace.slang's hit handler adds throughput *
        // emission to radiance BEFORE BRDF eval, matching the additive-
        // emitter pattern used for emissive sphere/quad area lights.
        packed[off + 16] = p.emission[0];
        packed[off + 17] = p.emission[1];
        packed[off + 18] = p.emission[2];
        packed[off + 19] = 0.0f;
        // v5 -- xyzw: orientation quaternion (rotation gizmo, #206).
        // Identity is (0,0,0,1) so non-rotated prims read as a no-op
        // quat-rotate on the shader side. Spheres ignore this (rotation
        // -symmetric). Planes rotate v0.xyz by this quaternion at
        // intersect time to derive the effective world-space normal.
        packed[off + 20] = p.orient[0];
        packed[off + 21] = p.orient[1];
        packed[off + 22] = p.orient[2];
        packed[off + 23] = p.orient[3];
    };
    for (std::size_t i = 0; i < infinite_prims.size(); ++i) {
        pack_prim(i, *infinite_prims[i].p, infinite_prims[i].id);
    }
    for (std::size_t i = 0; i < finite_order.size(); ++i) {
        const auto& src = finite_prims[finite_order[i]];
        pack_prim(linear_prim_count_ + i, *src.p, src.id);
    }
    device_->WriteBuffer(pt::rhi::BufferHandle{prim_buffer_id_},
                         packed.data(), packed.size() * sizeof(float));

    // Upload BVH node buffer if a tree was built. Node stride is 32B
    // (matches the shader struct's std430 layout). We grow by powers
    // of two like the primitive buffer.
    if (build_bvh && !analytic_bvh_.Empty()) {
        const std::uint32_t needed_nodes = static_cast<std::uint32_t>(analytic_bvh_.NodeCount());
        if (needed_nodes > bvh_node_buffer_capacity_) {
            std::uint32_t new_cap = bvh_node_buffer_capacity_ ? bvh_node_buffer_capacity_ : 64u;
            while (new_cap < needed_nodes) new_cap *= 2u;
            if (bvh_node_buffer_id_ != 0) {
                device_->WaitIdle();
                device_->DestroyBuffer(pt::rhi::BufferHandle{bvh_node_buffer_id_});
                bvh_node_buffer_id_ = 0;
            }
            auto buf = device_->CreateBuffer({
                .size = std::size_t(new_cap) * sizeof(pt::renderer::BvhNode),
                .usage = pt::rhi::BufferUsage::Storage,
                .debug_name = "analytic_bvh_nodes",
            });
            if (buf.id == 0) {
                LOG_ERROR("BVH node buffer allocation failed (capacity {})", new_cap);
                // Fall back to linear scan -- clear the BVH so the
                // shader path doesn't try to traverse missing data.
                analytic_bvh_ = pt::renderer::AnalyticBvh{};
            } else {
                bvh_node_buffer_id_       = buf.id;
                bvh_node_buffer_capacity_ = new_cap;
            }
        }
        if (!analytic_bvh_.Empty() && bvh_node_buffer_id_ != 0) {
            device_->WriteBuffer(pt::rhi::BufferHandle{bvh_node_buffer_id_},
                                 analytic_bvh_.Nodes().data(),
                                 analytic_bvh_.NodeCount() * sizeof(pt::renderer::BvhNode));
        }
    }

    // Dedup log spam: physics moves rigid bodies every frame, which
    // re-flags primitives_dirty_ and re-enters this function, but the
    // *count* / linear-vs-finite split / BVH topology only changes
    // when prims are added/removed/rebuilt. Log only the
    // structurally-interesting events; per-frame position-only
    // re-uploads are silent. See Engine.h bvh_last_log_* cache.
    const std::uint32_t bvh_node_count = analytic_bvh_.Empty()
        ? 0u
        : static_cast<std::uint32_t>(analytic_bvh_.NodeCount());
    const bool bvh_empty = analytic_bvh_.Empty();
    const bool log_changed =
        bvh_last_log_count_       != static_cast<std::int64_t>(count) ||
        bvh_last_log_linear_      != linear_prim_count_                ||
        bvh_last_log_node_count_  != bvh_node_count                    ||
        bvh_last_log_empty_       != bvh_empty;
    if (log_changed) {
        LOG_INFO("[bvh] uploaded {} prim(s): {} linear (planes etc), {} finite ({}, {} BVH nodes)",
                 count, linear_prim_count_, count - linear_prim_count_,
                 bvh_empty ? "linear scan" : "BVH", bvh_node_count);
        if (!bvh_empty && bvh_node_count > 0) {
            const auto& n0 = analytic_bvh_.Nodes()[0];
            LOG_INFO("[bvh] root node aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}] left_first={} count={}",
                     n0.aabb_min[0], n0.aabb_min[1], n0.aabb_min[2],
                     n0.aabb_max[0], n0.aabb_max[1], n0.aabb_max[2],
                     n0.left_first, n0.count);
        }
        bvh_last_log_count_      = static_cast<std::int64_t>(count);
        bvh_last_log_linear_     = linear_prim_count_;
        bvh_last_log_node_count_ = bvh_node_count;
        bvh_last_log_empty_      = bvh_empty;
    }
    primitives_dirty_ = false;
    accum_dirty_      = true;
}

// --- SDF Phase 1 (#97) -----------------------------------------------------
void Engine::EnsureSdfPrimsUploaded() {
    if (!device_) return;
    if (!sdf_prims_dirty_) return;

    const std::uint32_t count = static_cast<std::uint32_t>(sdf_prims_.size());

    // Per-cluster GPU layout (must match SdfPrimitives.slang's
    // loadSdfCluster):
    //   header (4 float4s = 64 B):
    //     hdr0.xyz = aabb_min        ;  asuint(hdr0.w) = node_count
    //     hdr1.xyz = aabb_max        ;  asuint(hdr1.w) = material
    //     hdr2.xyz = albedo          ;          hdr2.w = roughness
    //     hdr3.x   = ior             ;  hdr3.yzw pad
    //   nodes (kMaxNodes * 3 float4s = 384 B):
    //     n0.x=asfloat(op)  n0.y=asfloat(shape)
    //     n0.z=asfloat(child_a)  n0.w=asfloat(child_b)
    //     n1 = params (float4)
    //     n2.xyz = center           ; n2.w pad
    constexpr std::uint32_t kHeaderFloat4s = 4;
    constexpr std::uint32_t kNodeFloat4s   = 3;
    constexpr std::uint32_t kClusterFloat4s =
        kHeaderFloat4s + pt::renderer::SdfPrim::kMaxNodes * kNodeFloat4s;
    constexpr std::uint32_t kBytesPerCluster = sizeof(float) * 4u * kClusterFloat4s;

    // Grow GPU buffer by powers of two from a floor of 4 clusters --
    // matches the analytic-prim buffer's allocation pattern so
    // steady-state edits don't reallocate.
    if (count > sdf_cluster_capacity_) {
        std::uint32_t new_cap = sdf_cluster_capacity_ ? sdf_cluster_capacity_ : 4u;
        while (new_cap < count) new_cap *= 2u;
        if (sdf_cluster_buffer_id_ != 0) {
            device_->WaitIdle();
            device_->DestroyBuffer(pt::rhi::BufferHandle{sdf_cluster_buffer_id_});
            sdf_cluster_buffer_id_ = 0;
        }
        auto buf = device_->CreateBuffer({
            .size = std::size_t(new_cap) * kBytesPerCluster,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "sdf_clusters",
        });
        if (buf.id == 0) {
            LOG_ERROR("[sdf] cluster buffer allocation failed (capacity {})", new_cap);
            return;
        }
        sdf_cluster_buffer_id_ = buf.id;
        sdf_cluster_capacity_  = new_cap;
    }

    if (count == 0) {
        sdf_cluster_count_ = 0;
        sdf_prims_dirty_   = false;
        accum_dirty_       = true;
        LOG_INFO("[sdf] no clusters active");
        return;
    }

    // Allocate enough capacity for the buffer's full footprint (even
    // unused trailing clusters) so the shader's bounds checks against
    // sdf_cluster_count are the only thing that matters.
    std::vector<float> packed(std::size_t(sdf_cluster_capacity_) * 4u * kClusterFloat4s, 0.0f);

    auto pack_cluster = [&](std::size_t idx, const pt::renderer::SdfPrim& p) {
        const std::size_t base = idx * std::size_t(4u * kClusterFloat4s);
        // hdr0
        packed[base + 0]  = p.aabb_min[0];
        packed[base + 1]  = p.aabb_min[1];
        packed[base + 2]  = p.aabb_min[2];
        std::memcpy(&packed[base + 3], &p.node_count, sizeof(float));
        // hdr1
        packed[base + 4]  = p.aabb_max[0];
        packed[base + 5]  = p.aabb_max[1];
        packed[base + 6]  = p.aabb_max[2];
        std::memcpy(&packed[base + 7], &p.material, sizeof(float));
        // hdr2
        packed[base + 8]  = p.albedo[0];
        packed[base + 9]  = p.albedo[1];
        packed[base + 10] = p.albedo[2];
        packed[base + 11] = p.roughness;
        // hdr3
        packed[base + 12] = p.ior;
        // 13/14/15 pad (already zeroed)

        for (std::uint32_t i = 0; i < pt::renderer::SdfPrim::kMaxNodes; ++i) {
            const std::size_t nb = base + std::size_t(4u * (kHeaderFloat4s + i * kNodeFloat4s));
            const pt::renderer::SdfNode& n = p.nodes[i];
            // n0
            std::memcpy(&packed[nb + 0], &n.op,      sizeof(float));
            std::memcpy(&packed[nb + 1], &n.shape,   sizeof(float));
            std::memcpy(&packed[nb + 2], &n.child_a, sizeof(float));
            std::memcpy(&packed[nb + 3], &n.child_b, sizeof(float));
            // n1 (params)
            packed[nb + 4] = n.params[0];
            packed[nb + 5] = n.params[1];
            packed[nb + 6] = n.params[2];
            packed[nb + 7] = n.params[3];
            // n2 (center)
            packed[nb + 8]  = n.center[0];
            packed[nb + 9]  = n.center[1];
            packed[nb + 10] = n.center[2];
            // pad at +11 already zero
        }
    };

    std::size_t idx = 0;
    for (auto& [id, prim] : sdf_prims_) {
        // Re-derive the cluster AABB from its current node tree before
        // packing. The console commands that build / combine clusters
        // already call ComputeSdfAabb, but recomputing here closes the
        // gap if anything mutates an SdfPrim in-place between those
        // calls and the next upload -- the sphere-trace is AABB-bounded
        // so a stale AABB silently misses hits. Failure leaves the
        // existing aabb_min/max in place (best-effort; a malformed
        // tree would also fail at construction time and never reach
        // here).
        if (!pt::renderer::ComputeSdfAabb(prim)) {
            LOG_WARN("[sdf] cluster id={} AABB recompute failed at upload; "
                     "using existing AABB which may be stale", id);
        }
        pack_cluster(idx++, prim);
    }

    device_->WriteBuffer(pt::rhi::BufferHandle{sdf_cluster_buffer_id_},
                         packed.data(), packed.size() * sizeof(float));
    sdf_cluster_count_ = count;
    sdf_prims_dirty_   = false;
    accum_dirty_       = true;
    LOG_INFO("[sdf] uploaded {} cluster(s), iter budget = {} (r_sdf_max_iters), epsilon = {} m (r_sdf_epsilon)",
             count,
             [&]{ if (auto* v = pt::console::Console::Get().FindCVar("r_sdf_max_iters")) return v->GetInt(); return 128; }(),
             [&]{ if (auto* v = pt::console::Console::Get().FindCVar("r_sdf_epsilon"))  return v->GetFloat(); return 1e-4f; }());
    for (const auto& [id, p] : sdf_prims_) {
        LOG_INFO("[sdf]   id={} nodes={} aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}]",
                 id, p.node_count,
                 p.aabb_min[0], p.aabb_min[1], p.aabb_min[2],
                 p.aabb_max[0], p.aabb_max[1], p.aabb_max[2]);
    }
}
// --- end SDF Phase 1 -------------------------------------------------------

// --- Light primitives (#73) ------------------------------------------------
//
// Per-light GPU layout (must match PathTrace.slang's LightRecord +
// loadLight, kLightStrideFloat4 = 5 float4s = 80 B):
//
//   v0.xyz = pos             ;  v0.w   = radius (sphere)
//   v1.rgb = intensity       ;  v1.w   = type as float (0..3)
//   v2.xyz = dir (spot axis / quad normal)
//   v2.w   = cos_outer       (spot)
//   v3.xyz = u_vec (quad u-half-extent vector); v3.x = cos_inner (spot)
//   v3.w   = v_half          (quad)
//   v4.xyzw = orient quaternion (xyzw, unit) -- rotates dir / u_vec
//             at shader time so the gizmo can drive the cone-axis /
//             quad-frame without mutating the stored vectors.
//
// Grows by powers of two from a floor of 16 lights -- same allocation
// pattern as the analytic prim / SDF cluster / BVH-node buffers, so
// steady-state edits don't reallocate. The shader's NEE block is
// gated on `light_count > 0`; an empty set leaves the binding
// resolved to a placeholder but the shader doesn't read it.
void Engine::EnsureLightsUploaded() {
    if (!device_) return;
    if (!light_prims_dirty_) return;

    const std::uint32_t count = static_cast<std::uint32_t>(light_prims_.size());
    constexpr std::uint32_t kFloatsPerLight = 20;   // 5 float4s
    constexpr std::uint32_t kBytesPerLight  = sizeof(float) * kFloatsPerLight;

    // Allocate / grow the storage buffer when needed. 16-light floor
    // matches the analytic-prim convention; doubles up from there.
    if (count > light_buffer_capacity_) {
        std::uint32_t new_cap = light_buffer_capacity_ ? light_buffer_capacity_ : 16u;
        while (new_cap < count) new_cap *= 2u;
        if (light_buffer_id_ != 0) {
            device_->WaitIdle();
            device_->DestroyBuffer(pt::rhi::BufferHandle{light_buffer_id_});
            light_buffer_id_ = 0;
        }
        auto buf = device_->CreateBuffer({
            .size       = std::size_t(new_cap) * kBytesPerLight,
            .usage      = pt::rhi::BufferUsage::Storage,
            .debug_name = "analytic_lights",
        });
        if (buf.id == 0) {
            LOG_ERROR("light buffer allocation failed (capacity {})", new_cap);
            return;
        }
        light_buffer_id_       = buf.id;
        light_buffer_capacity_ = new_cap;
    }

    if (count == 0) {
        light_count_uploaded_ = 0;
        light_prims_dirty_    = false;
        accum_dirty_          = true;
        return;
    }

    // Allocate enough capacity for the buffer's full footprint (even
    // unused trailing entries) so the shader's bounds check against
    // light_count is the only thing that matters.
    std::vector<float> packed(std::size_t(light_buffer_capacity_) * kFloatsPerLight, 0.0f);
    std::size_t idx = 0;
    for (const auto& [id, L] : light_prims_) {
        const std::size_t off = idx * kFloatsPerLight;
        // v0: position + sphere radius
        packed[off + 0]  = L.pos[0];
        packed[off + 1]  = L.pos[1];
        packed[off + 2]  = L.pos[2];
        packed[off + 3]  = L.radius;
        // v1: intensity + type
        packed[off + 4]  = L.intensity[0];
        packed[off + 5]  = L.intensity[1];
        packed[off + 6]  = L.intensity[2];
        packed[off + 7]  = static_cast<float>(L.type);
        // v2: direction + spot cos_outer
        packed[off + 8]  = L.dir[0];
        packed[off + 9]  = L.dir[1];
        packed[off + 10] = L.dir[2];
        packed[off + 11] = L.cos_outer;
        // v3: quad u_vec (which also stores spot cos_inner in .x) + quad v_half
        packed[off + 12] = (L.type == AnalyticLight::Spot) ? L.cos_inner : L.u_vec[0];
        packed[off + 13] = L.u_vec[1];
        packed[off + 14] = L.u_vec[2];
        packed[off + 15] = L.v_half;
        // v4: orientation quaternion (xyzw). Spot lights rotate `dir`
        // through this before the cone-angle test; quad lights rotate
        // both `dir` (normal) and the u-extent vec assembled at sample
        // time. Point/sphere are rotation-symmetric so the shader
        // shortcuts the rotation; the data is still uploaded so the
        // editor/inspector reads a stable value.
        packed[off + 16] = L.orient[0];
        packed[off + 17] = L.orient[1];
        packed[off + 18] = L.orient[2];
        packed[off + 19] = L.orient[3];
        ++idx;
    }
    device_->WriteBuffer(pt::rhi::BufferHandle{light_buffer_id_},
                         packed.data(), packed.size() * sizeof(float));

    light_count_uploaded_ = count;
    light_prims_dirty_    = false;
    accum_dirty_          = true;
    LOG_INFO("[lights] uploaded {} analytic light(s)", count);
    for (const auto& [id, L] : light_prims_) {
        const char* tname = (L.type == AnalyticLight::Point)  ? "point"
                          : (L.type == AnalyticLight::Spot)   ? "spot"
                          : (L.type == AnalyticLight::Sphere) ? "sphere"
                                                              : "quad";
        LOG_INFO("[lights]   id={} type={} pos=({:.2f},{:.2f},{:.2f}) intensity=({:.2f},{:.2f},{:.2f})",
                 id, tname,
                 L.pos[0], L.pos[1], L.pos[2],
                 L.intensity[0], L.intensity[1], L.intensity[2]);
    }
}
// --- end Light primitives --------------------------------------------------

// --- Light tree (#129) -----------------------------------------------------
// ASYNC REBUILD + DOUBLE-BUFFERED UPLOAD.
//
// Old (synchronous) flow:
//   RenderFrame
//     -> EnsureLightTreeUploaded
//        -> BuildLightTree (~3ms for typical scenes; CPU-bound, single-thread)
//        -> WriteBuffer to slot 13
//   -> EnsureLightsUploaded
//   -> the rest of the frame
//
// New (async) flow:
//   RenderFrame
//     -> EnsureLightTreeUploaded
//        -> if light_prims_dirty_, snapshot + submit to async builder (~0ms)
//        -> TryAcquireResult; if a tree is ready, upload it to the
//           INACTIVE GPU slot then flip light_tree_active_slot_
//   -> EnsureLightsUploaded
//   -> the rest of the frame runs IN PARALLEL with the worker's rebuild
//
// The worker's CPU work overlaps the engine's render-record + GPU dispatch
// of frame N; the result is consumed at the top of frame N+1 (or later
// frames -- it's fine if the worker takes longer than one frame; we'll
// still keep using the prior tree until the new one lands).
//
// DIRTY-BIT GATING. We only kick the worker when light_prims_dirty_ is
// set (any light mutation -- creation, edit, deletion). Most frames have
// dirty=false and skip the dispatch entirely. The non-dirty fast path
// also skips the TryAcquireResult call -- there's nothing in flight, so
// nothing to acquire.
//
// FIRST-FRAME FALLBACK. On the very first dirty edge we run a synchronous
// build so the path tracer's NEE block has a real tree to traverse on
// frame 0 (otherwise the binding stays on the placeholder for ~1 frame
// while the worker spins up, which means worse-than-baseline variance
// for that frame). Cost: one ~3ms synchronous build on a scene change.
// All subsequent rebuilds are async.
//
// DOUBLE-BUFFERED GPU SLOTS. The path tracer reads slot N during a
// frame submit; if we want to overwrite while the GPU is still reading
// we'd need to WaitIdle() (the slow path the old code used on resize).
// Instead, we keep TWO GPU node buffers of equal capacity and write to
// the inactive one -- WriteBuffer maps + memcpys into a staging buffer
// the backend uploads on the next Submit, completely independent of
// whatever the active slot is doing. After the upload we flip the
// active-slot index; the bind site reads `light_tree_buffer_ids_[
// light_tree_active_slot_]` on the next BindBuffer(13, ...). The
// previously-active slot becomes the WRITE target for the next
// rebuild -- by the time we touch it the GPU has long since moved on.
//
// Empty light set -> node_count = 0; binding falls back to the
// placeholder buffer at the dispatch site for Metal-push-slot stability.
// The shader's `light_tree_node_count > 0` push gate is the runtime
// "actually traverse" switch -- same partial-binding pattern as
// shadow_vis_buf / light_prims.
void Engine::EnsureLightTreeUploaded() {
    if (!device_) return;

    constexpr std::uint32_t kBytesPerNode = sizeof(pt::renderer::LightTreeNode);

    auto grow_slots_to = [&](std::uint32_t needed_nodes) -> bool {
        if (needed_nodes <= light_tree_buffer_capacity_ &&
            light_tree_buffer_ids_[0] != 0 &&
            light_tree_buffer_ids_[1] != 0) {
            return true;
        }
        std::uint32_t new_cap = light_tree_buffer_capacity_
                                  ? light_tree_buffer_capacity_
                                  : 32u;
        while (new_cap < needed_nodes) new_cap *= 2u;
        // Both slots resize together; the inactive slot might have an
        // in-flight read but on a resize we *have* to drain the GPU
        // because the descriptor binding's size is changing. (The
        // common steady-state path skips this branch entirely.)
        device_->WaitIdle();
        for (auto& id : light_tree_buffer_ids_) {
            if (id != 0) {
                device_->DestroyBuffer(pt::rhi::BufferHandle{id});
                id = 0;
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            auto buf = device_->CreateBuffer({
                .size       = std::size_t(new_cap) * kBytesPerNode,
                .usage      = pt::rhi::BufferUsage::Storage,
                .debug_name = (slot == 0) ? "light_tree_nodes_0"
                                          : "light_tree_nodes_1",
            });
            if (buf.id == 0) {
                LOG_ERROR("light tree buffer allocation failed "
                          "(capacity {}, slot {})", new_cap, slot);
                return false;
            }
            light_tree_buffer_ids_[slot] = buf.id;
        }
        light_tree_buffer_capacity_ = new_cap;
        // Resize invalidates whatever was uploaded; reset the count so
        // the consumer falls back to the placeholder until the next
        // tree lands.
        light_tree_node_count_uploaded_ = 0;
        light_tree_first_built_         = false;
        light_tree_active_slot_         = 0;
        return true;
    };

    auto upload_tree_to_slot = [&](const pt::renderer::LightTree& tree,
                                   int slot) -> bool {
        const std::uint32_t node_count =
            static_cast<std::uint32_t>(tree.nodes.size());
        if (!grow_slots_to(std::max(node_count, 1u))) return false;
        if (node_count == 0) {
            // Empty-tree publish: just clear the consumer-visible count.
            // No upload needed -- the bind site already falls back to
            // the placeholder when count == 0.
            light_tree_node_count_uploaded_ = 0;
            light_tree_first_built_         = true;
            return true;
        }
        // Allocate to FULL capacity (trailing unused nodes zero-init)
        // so the shader's bounds check against node_count is the only
        // thing that matters. Same allocation shape as the
        // pre-double-buffer code.
        std::vector<pt::renderer::LightTreeNode> packed(
            light_tree_buffer_capacity_);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            packed[i] = tree.nodes[i];
        }
        device_->WriteBuffer(
            pt::rhi::BufferHandle{light_tree_buffer_ids_[slot]},
            packed.data(),
            std::size_t(light_tree_buffer_capacity_) * kBytesPerNode);
        light_tree_node_count_uploaded_ = node_count;
        light_tree_first_built_         = true;
        return true;
    };

    auto snapshot_inputs = [&]() {
        // Snapshot the light list in upload order. Engine::EnsureLightsUploaded
        // (below) packs lights into the GPU buffer in std::map iteration
        // order (id-sorted); the tree's leaf left_first values index into
        // that same packed array, so we MUST mirror that ordering exactly.
        //
        // Direction / u-extent vectors are pre-rotated through the
        // light's orientation quaternion so the tree's cone-axis
        // aggregation reflects the visible cone (the NEE shader does
        // the same rotation at sample time via L.orient). Identity
        // quat (0,0,0,1) yields the original axis unchanged.
        std::vector<pt::renderer::LightInput> inputs;
        inputs.reserve(light_prims_.size());
        for (const auto& [id, L] : light_prims_) {
            pt::renderer::LightInput li;
            li.type = static_cast<std::uint32_t>(L.type);
            // quatRotate(q, v) = v + 2 * cross(u, cross(u, v) + w * v).
            // Inlined here so the engine doesn't need to import a math
            // header just for this single helper.
            const float qx = L.orient[0], qy = L.orient[1];
            const float qz = L.orient[2], qw = L.orient[3];
            auto rotate = [qx, qy, qz, qw](const float v[3], float out[3]) {
                const float cx = qy * v[2] - qz * v[1] + qw * v[0];
                const float cy = qz * v[0] - qx * v[2] + qw * v[1];
                const float cz = qx * v[1] - qy * v[0] + qw * v[2];
                out[0] = v[0] + 2.0f * (qy * cz - qz * cy);
                out[1] = v[1] + 2.0f * (qz * cx - qx * cz);
                out[2] = v[2] + 2.0f * (qx * cy - qy * cx);
            };
            float dir_rot[3];   rotate(L.dir,   dir_rot);
            float u_vec_rot[3]; rotate(L.u_vec, u_vec_rot);
            for (int i = 0; i < 3; ++i) {
                li.pos[i]       = L.pos[i];
                li.intensity[i] = L.intensity[i];
                li.dir[i]       = dir_rot[i];
                li.u_vec[i]     = u_vec_rot[i];
            }
            li.radius    = L.radius;
            li.cos_outer = L.cos_outer;
            li.v_half    = L.v_half;
            inputs.push_back(li);
        }
        return inputs;
    };

    // Lazy-construct the async builder on first call. Engine ctor would
    // have done it but the renderer header drags in <thread> /
    // <condition_variable> -- not the kind of thing we want to pull into
    // Engine.h, so the unique_ptr is built here.
    if (!light_tree_builder_) {
        light_tree_builder_ =
            std::make_unique<pt::renderer::AsyncLightTreeBuilder>();
    }

    // First-frame fallback: if the light set is dirty AND no tree has
    // ever been published, run a synchronous build so we don't render
    // the first frame with the placeholder bound. The user mutated the
    // scene, expects the very next frame to reflect it -- we trade ~3ms
    // on that ONE frame for correct first-frame variance. Steady-state
    // (subsequent dirty edges) takes the async path below.
    if (light_prims_dirty_ && !light_tree_first_built_) {
        auto inputs = snapshot_inputs();
        pt::renderer::LightTree sync_tree;
        pt::renderer::BuildLightTree(inputs, sync_tree);
        if (!upload_tree_to_slot(sync_tree, light_tree_active_slot_)) {
            return;
        }
        // Also seed the builder so its internal slot is consistent with
        // what we just uploaded. The next dirty edge will hand it the
        // updated inputs and the worker will overwrite the inactive
        // slot's host-side tree -- that's fine.
        light_tree_builder_->SubmitInputs(std::move(inputs));
        LOG_INFO("[light_tree] first-frame sync build -- {} lights, {} nodes",
                 sync_tree.light_count,
                 static_cast<std::uint32_t>(sync_tree.nodes.size()));
        // Don't return: still drop through to the TryAcquireResult /
        // dispatch path so a same-frame submit doesn't get latched
        // again on the next call.
    } else if (light_prims_dirty_) {
        // Steady-state dirty edge: hand a fresh snapshot to the worker.
        // The previous build's result (if any) stays in the active GPU
        // slot until the new one lands.
        light_tree_builder_->SubmitInputs(snapshot_inputs());
    }

    // Drain ONE completed result from the worker (if any). Worker may
    // produce zero or one new tree per frame -- if it queued multiple
    // dispatches the worker collapsed them into the latest snapshot,
    // so we only ever publish the freshest.
    if (light_tree_builder_) {
        if (const pt::renderer::LightTree* tree =
                light_tree_builder_->TryAcquireResult()) {
            const int write_slot = 1 - light_tree_active_slot_;
            if (upload_tree_to_slot(*tree, write_slot)) {
                light_tree_active_slot_ = write_slot;
                LOG_INFO("[light_tree] async-built tree published -- "
                         "{} lights, {} nodes (slot {})",
                         tree->light_count,
                         static_cast<std::uint32_t>(tree->nodes.size()),
                         write_slot);
            }
        }
    }
}
// --- end Light tree --------------------------------------------------------

void Engine::EnsurePipelineHandles() {
    if (!device_) return;
    auto resolve = [&](std::uint64_t& cached, const char* name) {
        if (cached != 0) return;
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = name,
            .bytecode    = {},
            .debug_name  = name,
        };
        cached = device_->CreateComputePipeline(desc).id;
    };
    resolve(pathtrace_pipeline_id_,    "pathtrace");
    resolve(tonemap_pipeline_id_,      "tonemap");
    resolve(bloom_down_pipeline_id_,   "bloom_down");
    resolve(bloom_up_pipeline_id_,     "bloom_up");
    resolve(autoexpose_pipeline_id_,   "autoexpose");
    resolve(perfoverlay_pipeline_id_,  "perfoverlay");
    // Editor 3D-transform gizmo overlay (issue: editor 3D gizmos).
    // Resolves on Metal + Vulkan; Software creates the pipeline by
    // name on demand inside CreateComputePipeline so we get an id
    // back even though the kernel itself is the CPU-side
    // SoftwareDevice::RunEditorOverlay path.
    resolve(editor_overlay_pipeline_id_, "editor_overlay");
    // Metal-only today; the Vulkan backend's pipeline-build worker has
    // no entry for this name, so resolve will leave the id at 0 and the
    // dispatch site short-circuits. Vulkan stars-composite plumbing is
    // a follow-up.
    resolve(stars_composite_pipeline_id_, "stars_composite");
    // Aurora borealis composite (issue #116). Metal-only at MVP scope;
    // Vulkan id stays 0 and the gate folds correctly to "no aurora".
    resolve(aurora_composite_pipeline_id_, "aurora_composite");
    // SIGMA shadow denoiser (issue #115). Metal-only at MVP scope;
    // r_shadow_demod gate collapses to "no SIGMA" on Vulkan.
    resolve(sigma_shadow_pipeline_id_, "sigma_shadow");
    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    // Three compute kernels chained behind PathTrace's WRS
    // candidate-generation pass: temporal reuse -> spatial reuse ->
    // final shadow + Lambert composite. Metal-only at Phase A scope;
    // Vulkan pipeline-build worker has no entry for these names so
    // the resolve leaves the ids at 0 and the dispatch gates collapse
    // cleanly to "no ReSTIR, legacy single-pick NEE in PathTrace."
    // Vulkan plumbing is a follow-up alongside SigmaShadow /
    // StarsComposite / AuroraComposite -- same shape.
    resolve(restir_temporal_pipeline_id_, "restir_temporal");
    resolve(restir_spatial_pipeline_id_,  "restir_spatial");
    resolve(restir_final_pipeline_id_,    "restir_final");
    // --- end ReSTIR DI Phase A ---------------------------------------------
    // Screen-space particle composite (#82 MVP). Dispatch site below
    // additionally gates on particle_count > 0 and r_particles, so the
    // resolve cost is 1 atomic compare per frame until Vulkan grows a
    // real entry.
    resolve(particle_composite_pipeline_id_, "particle_composite");
}

void Engine::LoadFavoritesFromDisk() {
    // favorites.cfg lives next to demont.cfg (CWD). One favourite per
    // line; blank lines and lines starting with '#' are ignored. Read
    // into Console::favorites_ directly -- this is NOT a console
    // script (don't ExecuteScript it; the saved lines aren't meant to
    // run at load time, only when the user types f<N>).
    FILE* f = std::fopen("favorites.cfg", "rb");
    if (f == nullptr) return;
    std::vector<std::string> loaded;
    char buf[4096];
    std::string accum;
    while (std::size_t n = std::fread(buf, 1, sizeof(buf), f)) {
        accum.append(buf, n);
    }
    std::fclose(f);
    std::size_t i = 0;
    while (i < accum.size()) {
        std::size_t end = i;
        while (end < accum.size() && accum[end] != '\n') ++end;
        std::string line = accum.substr(i, end - i);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip blank + comment lines.
        std::size_t lead = 0;
        while (lead < line.size() && (line[lead] == ' ' || line[lead] == '\t')) ++lead;
        if (lead < line.size() && line[lead] != '#') {
            loaded.push_back(std::move(line));
        }
        i = (end < accum.size()) ? end + 1 : end;
    }
    pt::console::Console::Get().SetFavorites(std::move(loaded));
    LOG_INFO("favorites: loaded {} entry(ies) from favorites.cfg",
             pt::console::Console::Get().FavoriteCount());
}

void Engine::SaveFavoritesToDisk() {
    auto& C = pt::console::Console::Get();
    const auto favs = C.Favorites();
    // Even if empty, rewrite the file so a cleared list persists as
    // empty (otherwise an old file would resurrect on next launch).
    FILE* f = std::fopen("favorites.cfg", "wb");
    if (f == nullptr) {
        LOG_WARN("favorites: failed to open favorites.cfg for write");
        return;
    }
    const std::string header =
        "# demont console favourites. One command line per entry.\n"
        "# Saved automatically by the `fav` / `unfav` / `fav_clear`\n"
        "# commands. Lines starting with `#` and blank lines are\n"
        "# ignored on load. Invocation: type `f1` for the first\n"
        "# entry, `f2` for the second, etc.\n";
    std::fwrite(header.data(), 1, header.size(), f);
    for (const auto& line : favs) {
        std::fwrite(line.data(), 1, line.size(), f);
        std::fputc('\n', f);
    }
    std::fclose(f);
}

void Engine::LoadConsoleHistoryFromDisk() {
    // console_history.txt lives next to demont.cfg (CWD). One history
    // entry per line; blank lines + lines starting with '#' are
    // ignored (so users can hand-edit and leave comments).
    //
    // Loaded directly into Console::history_ via SetHistory -- the
    // platform overlay seeds its native walk buffer from
    // Console::History() during Init, so this MUST run BEFORE
    // overlay_->Init() to take effect on the first frame.
    //
    // Missing file is fine (first launch, deleted file). All other
    // errors are warnings -- a malformed history file should never
    // wedge engine startup.
    FILE* f = std::fopen("console_history.txt", "rb");
    if (f == nullptr) return;
    std::vector<std::string> loaded;
    char buf[4096];
    std::string accum;
    while (std::size_t n = std::fread(buf, 1, sizeof(buf), f)) {
        accum.append(buf, n);
    }
    std::fclose(f);
    std::size_t i = 0;
    while (i < accum.size()) {
        std::size_t end = i;
        while (end < accum.size() && accum[end] != '\n') ++end;
        std::string line = accum.substr(i, end - i);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip blank + comment lines (after optional leading
        // whitespace -- mirrors LoadFavoritesFromDisk).
        std::size_t lead = 0;
        while (lead < line.size() && (line[lead] == ' ' || line[lead] == '\t')) ++lead;
        if (lead < line.size() && line[lead] != '#') {
            loaded.push_back(std::move(line));
        }
        i = (end < accum.size()) ? end + 1 : end;
    }
    pt::console::Console::Get().SetHistory(std::move(loaded));
    LOG_INFO("console history: loaded {} entry(ies) from console_history.txt",
             pt::console::Console::Get().HistoryCount());
}

void Engine::SaveConsoleHistoryToDisk() {
    auto& C = pt::console::Console::Get();
    const auto hist = C.History();
    // Even if empty, rewrite the file so a cleared list persists as
    // empty across launches -- otherwise an old file would resurrect
    // history the user explicitly cleared.
    FILE* f = std::fopen("console_history.txt", "wb");
    if (f == nullptr) {
        LOG_WARN("console history: failed to open console_history.txt for write");
        return;
    }
    const std::string header =
        "# demont console input history. One command line per entry.\n"
        "# Auto-written at engine shutdown so the up-arrow scroll-back\n"
        "# survives across sessions. Lines starting with `#` and blank\n"
        "# lines are ignored on load. Capped at the most recent\n"
        "# Console::kMaxHistoryDepth entries -- safe to hand-edit.\n";
    std::fwrite(header.data(), 1, header.size(), f);
    // Cap on write too as a defence-in-depth: PushHistory already
    // trims, but SetHistory could in theory deliver an over-large
    // vector if a future caller bypasses it.
    const std::size_t kCap = pt::console::Console::kMaxHistoryDepth;
    const std::size_t start = hist.size() > kCap ? hist.size() - kCap : 0;
    for (std::size_t k = start; k < hist.size(); ++k) {
        std::fwrite(hist[k].data(), 1, hist[k].size(), f);
        std::fputc('\n', f);
    }
    std::fclose(f);
}

void Engine::LoadCameraBookmarksFromDisk() {
    // camera_bookmarks.cfg lives next to demont.cfg + favorites.cfg
    // (CWD). One bookmark per line: `<name> <x> <y> <z> <yaw_deg>
    // <pitch_deg> <fov_deg>` (seven whitespace-separated tokens; the
    // first is the user-chosen mnemonic, the remaining six are the
    // same float format the numeric cam_slot_N cvars use). Blank
    // lines and lines whose first non-space char is `#` are ignored.
    // Malformed lines are skipped with a warning rather than aborting
    // the load -- a corrupted entry shouldn't take the whole file
    // down with it. Duplicate names: last one wins (std::map insert
    // via `bookmarks[name] = state`).
    FILE* f = std::fopen("camera_bookmarks.cfg", "rb");
    if (f == nullptr) return;
    std::string accum;
    char buf[4096];
    while (std::size_t n = std::fread(buf, 1, sizeof(buf), f)) {
        accum.append(buf, n);
    }
    std::fclose(f);

    camera_bookmarks_.clear();
    std::size_t i = 0;
    std::size_t loaded = 0, skipped = 0;
    while (i < accum.size()) {
        std::size_t end = i;
        while (end < accum.size() && accum[end] != '\n') ++end;
        std::string line = accum.substr(i, end - i);
        i = (end < accum.size()) ? end + 1 : end;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Skip blanks + comment lines.
        std::size_t lead = 0;
        while (lead < line.size() && (line[lead] == ' ' || line[lead] == '\t')) ++lead;
        if (lead >= line.size()) continue;
        if (line[lead] == '#')   continue;

        // Split into name + tail-of-six-floats. We don't validate the
        // floats here; the cam_load_named parser does that on use,
        // and a malformed value is an error message rather than a
        // silent load failure.
        std::size_t name_end = lead;
        while (name_end < line.size() && line[name_end] != ' ' && line[name_end] != '\t') ++name_end;
        if (name_end == lead) { ++skipped; continue; }
        std::string name = line.substr(lead, name_end - lead);
        std::size_t state_start = name_end;
        while (state_start < line.size() && (line[state_start] == ' ' || line[state_start] == '\t')) ++state_start;
        if (state_start >= line.size()) { ++skipped; continue; }
        std::string state = line.substr(state_start);
        camera_bookmarks_[std::move(name)] = std::move(state);
        ++loaded;
    }
    if (skipped > 0) {
        LOG_WARN("camera_bookmarks: skipped {} malformed line(s) in camera_bookmarks.cfg",
                 skipped);
    }
    LOG_INFO("camera_bookmarks: loaded {} bookmark(s) from camera_bookmarks.cfg",
             loaded);
}

void Engine::SaveCameraBookmarksToDisk() {
    // Always rewrite, even if camera_bookmarks_ is empty -- otherwise
    // a cleared-all-bookmarks state would silently resurrect the
    // stale file on next launch. Same convention as SaveFavoritesToDisk.
    FILE* f = std::fopen("camera_bookmarks.cfg", "wb");
    if (f == nullptr) {
        LOG_WARN("camera_bookmarks: failed to open camera_bookmarks.cfg for write");
        return;
    }
    const std::string header =
        "# demont named camera bookmarks. One bookmark per line:\n"
        "#   <name> <x> <y> <z> <yaw_deg> <pitch_deg> <fov_deg>\n"
        "# Saved automatically by `cam_save_named` and\n"
        "# `cam_delete_bookmark`. Lines starting with `#` and blank\n"
        "# lines are ignored on load. Invocation: type\n"
        "# `cam_load_named <name>` to restore a viewpoint.\n";
    std::fwrite(header.data(), 1, header.size(), f);
    for (const auto& [name, state] : camera_bookmarks_) {
        std::fwrite(name.data(), 1, name.size(), f);
        std::fputc(' ', f);
        std::fwrite(state.data(), 1, state.size(), f);
        std::fputc('\n', f);
    }
    std::fclose(f);
}

void Engine::RegisterCameraBookmarkCommands() {
    // Named-camera-bookmark commands -- complements the numeric
    // cam_slot_1..9 system in RegisterCommands. Bookmarks are keyed
    // by user-chosen mnemonic (`overhead`, `closeup`, `cornell_3q`),
    // share the same six-float "x y z yaw_deg pitch_deg fov_deg"
    // serialization, and persist to camera_bookmarks.cfg (rewritten
    // on every save / delete).
    auto& C = pt::console::Console::Get();

    // Local copies of the camera-state format / parse helpers used by
    // the numeric cam_slot system. Kept as no-capture lambdas so
    // they're copy-constructible into the std::function commands.
    const auto fmt_cam_state = [](const pt::renderer::Camera& c) -> std::string {
        constexpr float kRadToDeg = 57.29577951308232f;
        return fmt::format("{:.6f} {:.6f} {:.6f} {:.4f} {:.4f} {:.3f}",
                           c.pos.x, c.pos.y, c.pos.z,
                           c.yaw   * kRadToDeg,
                           c.pitch * kRadToDeg,
                           c.fov_deg);
    };
    const auto parse_cam_state = [](std::string_view s, pt::renderer::Camera& out) -> bool {
        constexpr float kDegToRad = 0.017453292519943295f;
        std::istringstream iss{std::string(s)};
        float x, y, z, yaw_d, pitch_d, fov_d;
        if (!(iss >> x >> y >> z >> yaw_d >> pitch_d >> fov_d)) return false;
        out.pos     = glm::vec3(x, y, z);
        out.yaw     = yaw_d   * kDegToRad;
        out.pitch   = pitch_d * kDegToRad;
        out.fov_deg = fov_d;
        out.ClampPitch();
        return true;
    };
    // Name validation: must be a single non-empty token containing
    // only printable ASCII excluding whitespace and `#` (the cfg
    // comment prefix). Allows identifier-style chars (a-z, A-Z, 0-9,
    // underscore, hyphen, dot). The cfg file's line-per-bookmark
    // format means embedded whitespace would split the parse on
    // load; the token-level ban covers that.
    const auto is_valid_bookmark_name = [](std::string_view name) -> bool {
        if (name.empty()) return false;
        for (char c : name) {
            if (c <= ' ' || c == '#' || c == 0x7f) return false;
        }
        return true;
    };

    C.RegisterCommand("cam_save_named",
        "cam_save_named <name>: capture the current camera state into "
        "a named bookmark (e.g. `cam_save_named overhead`). Name must "
        "be a single non-empty token without whitespace or '#'. "
        "Persists to camera_bookmarks.cfg.",
        [this, fmt_cam_state, is_valid_bookmark_name](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_save_named: no camera"); return; }
            if (args.size() != 1) {
                out.PrintLine("usage: cam_save_named <name>");
                return;
            }
            if (!is_valid_bookmark_name(args[0])) {
                out.FormatLine("cam_save_named: invalid name '{}' (must be a single "
                               "non-empty token without whitespace or '#')",
                               std::string(args[0]));
                return;
            }
            std::string name(args[0]);
            const std::string state = fmt_cam_state(*camera_);
            const bool replaced = camera_bookmarks_.contains(name);
            camera_bookmarks_[name] = state;
            SaveCameraBookmarksToDisk();
            if (replaced) {
                out.FormatLine("cam_save_named: replaced '{}' = {}", name, state);
            } else {
                out.FormatLine("cam_save_named: saved '{}' = {}", name, state);
            }
        });

    C.RegisterCommand("cam_load_named",
        "cam_load_named <name>: restore the camera from a named "
        "bookmark saved with cam_save_named. Use cam_list_bookmarks "
        "to see available names. Fires the active denoiser's "
        "history-reset flag so the temporal denoise pipeline "
        "(SVGF/NRD/MetalFX/OptiX-temporal) doesn't blend pre-teleport "
        "content forward.",
        [this, parse_cam_state](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_load_named: no camera"); return; }
            if (args.size() != 1) {
                out.PrintLine("usage: cam_load_named <name>");
                return;
            }
            const std::string name(args[0]);
            auto it = camera_bookmarks_.find(name);
            if (it == camera_bookmarks_.end()) {
                out.FormatLine("cam_load_named: no bookmark named '{}' "
                               "(use cam_list_bookmarks to see saved names)",
                               name);
                return;
            }
            if (!parse_cam_state(it->second, *camera_)) {
                out.FormatLine("cam_load_named: bookmark '{}' contents malformed: '{}'",
                               name, it->second);
                return;
            }
            prev_view_proj_valid_ = false;  // active denoiser reset_history (SVGF/NRD/MetalFX/OptiX-temporal)
            out.FormatLine("cam_load_named: '{}' = {}", name, it->second);
        });

    C.RegisterCommand("cam_list_bookmarks",
        "List all named camera bookmarks (saved via cam_save_named). "
        "Output is sorted alphabetically; numeric slots 1..9 are "
        "listed by `cam_list` instead.",
        [this](auto, pt::console::Output& out) {
            if (camera_bookmarks_.empty()) {
                out.PrintLine("(no camera bookmarks yet -- use "
                              "`cam_save_named <name>` to save one)");
                return;
            }
            out.FormatLine("{} bookmark(s):", camera_bookmarks_.size());
            for (const auto& [name, state] : camera_bookmarks_) {
                out.FormatLine("  {}  {}", name, state);
            }
        });

    C.RegisterCommand("cam_delete_bookmark",
        "cam_delete_bookmark <name>: remove a named camera bookmark. "
        "Rewrites camera_bookmarks.cfg so the deletion persists.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: cam_delete_bookmark <name>");
                return;
            }
            const std::string name(args[0]);
            auto it = camera_bookmarks_.find(name);
            if (it == camera_bookmarks_.end()) {
                out.FormatLine("cam_delete_bookmark: no bookmark named '{}'",
                               name);
                return;
            }
            camera_bookmarks_.erase(it);
            SaveCameraBookmarksToDisk();
            out.FormatLine("cam_delete_bookmark: removed '{}' "
                           "({} bookmark(s) remain)",
                           name, camera_bookmarks_.size());
        });
}

void Engine::PrewarmPipelines() {
    if (!device_) return;
    // Cvar gate. Set r_pipeline_prewarm 0 for A/B perf testing of the
    // prewarm path -- with it off, the engine falls back to the
    // pre-existing behaviour (first dispatch implicitly forces a
    // CreateComputePipeline-by-name lookup; on Vulkan the async worker
    // still builds eagerly via the device constructor but no second
    // engine-side handshake fires after the constructor returns).
    auto& C = pt::console::Console::Get();
    if (auto* v = C.FindCVar("r_pipeline_prewarm"); v && !v->GetBool()) {
        return;
    }
    // Keep this list in lockstep with EnsurePipelineHandles. The two
    // lists serve different purposes (EnsurePipelineHandles caches the
    // returned id; PrewarmPipelines signals intent) but cover the same
    // set of kernel names. Adding a name to one without the other is a
    // bug: an unknown handle pipeline produces an id==0 dispatch
    // short-circuit that's correct-but-silent, and an unprewarmed
    // pipeline produces a one-shot hitch on first dispatch.
    auto warm = [&](const char* name) {
        device_->EnsurePipelineWarmed(name);
    };
    warm("pathtrace");
    warm("tonemap");
    warm("bloom_down");
    warm("bloom_up");
    warm("autoexpose");
    warm("perfoverlay");
    warm("stars_composite");
    warm("aurora_composite");
    warm("sigma_shadow");
    warm("restir_temporal");
    warm("restir_spatial");
    warm("restir_final");
    warm("particle_composite");
}

void Engine::RenderFrame() {
    PT_ZONE_SCOPED_N("Engine::RenderFrame");
    // Pipelines may still be building on the Vulkan backend's async
    // worker; re-resolve cached ids each frame until each flips
    // non-zero. Once all are cached the resolves are no-ops.
    EnsurePipelineHandles();
    if (!device_) return;
    // Drive the CSG bake state machine BEFORE the loading-screen check so
    // that on frame 1 the bake gets submitted (phase 0 -> 1), the loading
    // screen renders, and the PathTrace dispatch is NOT fired with
    // uninitialised csg_vbuf / csg_ibuf / tri_bvh_nodes / tri_bvh_permuted_ids
    // descriptors. On NVIDIA, dispatching with uninit mesh descriptors
    // page-faults the GPU at the first triangle-traversal read -> TDR ->
    // DEVICE_LOST. MoltenVK runs the same dispatch fine because Apple's
    // memory zeroing + lax OOB handling masks the race.
    EnsureMeshUpdated();
    const bool mesh_pending = (bake_phase_.load(std::memory_order_acquire) != 0);
    if (pathtrace_pipeline_id_ == 0 || mesh_pending) {
        // Loading frame. Async pipeline build still in flight, so the
        // path tracer / tonemap / overlay dispatches would all no-op
        // and leave the swapchain image in its just-acquired
        // UNDEFINED state -- visible as flickering / stale pixels.
        // Issue a minimal clear to a defined dark colour so the user
        // sees a clean frame while the pipelines finish compiling.
        // Once the worker's done, EnsurePipelineHandles() flips the
        // cached ids non-zero and this branch stops firing.
        if (!loading_frame_active_) {
            LOG_INFO("engine: loading screen active (Vulkan pipeline build pending)");
            loading_frame_active_ = true;
        }
        auto fc = device_->BeginFrame();
        auto* cb = device_->AcquireCommandBuffer();
        if (cb) {
            constexpr float kLoadingFrameRgba[4] = { 0.05f, 0.06f, 0.08f, 1.0f };
            cb->ClearStorageTexture(fc.swapchain_image, kLoadingFrameRgba);
            device_->Submit(cb);
        }
        device_->EndFrame(cb);
        return;
    }
    if (loading_frame_active_) {
        LOG_INFO("engine: pipelines ready, normal rendering resumed");
        loading_frame_active_ = false;
    }

    // EnsureMeshUpdated() now runs earlier (above the loading-screen check)
    // so the CSG bake state machine kicks in before the first dispatch.
    EnsurePrimitivesUploaded();
    // --- Voxel destruction Phase 1 (#140) ----------------------------------
    // Refresh the reserved-id voxel SDF clusters against the current
    // r_voxelize_demo cvar before SDF upload. Sets voxel_demo_active_
    // for the mesh-hide gate below.
    SyncVoxelDemoState();
    // --- end Voxel destruction Phase 1 -------------------------------------
    // --- SDF Phase 1 (#97) -------------------------------------------------
    EnsureSdfPrimsUploaded();
    // --- end SDF Phase 1 ---------------------------------------------------
    // --- Light primitives (#73) + Light tree (#129) ------------------------
    // Order: tree BEFORE the list, because the tree's "should I rebuild?"
    // edge is the same `light_prims_dirty_` flag that EnsureLightsUploaded
    // clears once it's done. The build itself just reads light_prims_;
    // both ensures consume the same snapshot.
    EnsureLightTreeUploaded();
    EnsureLightsUploaded();
    // --- end Light primitives ----------------------------------------------
    // --- Fluid Phase 1 (#136) -- smoke emitters ----------------------------
    // Re-upload every frame so parametric drift (base + velocity * t)
    // doesn't need a dirty flag. Cheap: 16 emitters * 48 B = 768 B.
    EnsureSmokeEmittersUploaded();
    // --- end Fluid Phase 1 -------------------------------------------------

    auto& C = pt::console::Console::Get();

    // P10/P12 denoiser state. Resolved before BeginFrame so we know
    // whether to allocate the G-buffer textures this frame. The cvar
    // value chooses the kind (off / metalfx / svgf / nrd / optix_*);
    // per-backend gating then drops it to off if the active device
    // doesn't support the chosen kind. metalfx is Mac-only; svgf, nrd,
    // and optix_* are Vulkan-only; optix_* additionally requires the
    // build-time PT_ENABLE_OPTIX (CUDA Toolkit + OptiX SDK detected)
    // and a runtime CUDA-capable NVIDIA GPU. The no-op "off" value
    // short-circuits all G-buffer allocation.
    DenoiserKind want_kind = DenoiserKind::Off;
    if (auto* v = C.FindCVar("r_denoiser")) {
        const auto& s = v->value;
        if (s == "metalfx" && current_backend_ == BackendType::Metal) {
            want_kind = DenoiserKind::MetalFX;
        } else if (current_backend_ == BackendType::Metal) {
            // SVGF on Metal: same in-house SVGF chain as Vulkan, same
            // Slang sources cross-compiled to MSL. svgf_basic = temporal
            // only; svgf_atrous = temporal + a-trous (passes set by
            // r_svgf_atrous_passes). The `nrd` alias falls through to
            // svgf_atrous today (placeholder for the NVIDIA RayTracingDenoiser
            // library). The *_metalfx variants chain MetalFX
            // TemporalDenoisedScaler as a finalizer (ML TAA on top of
            // SVGF-denoised input) -- highest quality but Mac-only.
            if      (s == "svgf_basic")            want_kind = DenoiserKind::SvgfBasic;
            else if (s == "svgf_atrous")           want_kind = DenoiserKind::SvgfAtrous;
            else if (s == "svgf_basic_metalfx")    want_kind = DenoiserKind::SvgfBasicMetalFx;
            else if (s == "svgf_atrous_metalfx")   want_kind = DenoiserKind::SvgfAtrousMetalFx;
            else if (s == "nrd")                   want_kind = DenoiserKind::Nrd;
        } else if (current_backend_ == BackendType::Vulkan) {
            // The *_metalfx variants chain MetalFX as a finalizer (Mac
            // only). On Vulkan we silently degrade to the corresponding
            // plain SVGF mode so a user's demont.cfg / preset value
            // works across both backends -- they just lose the MetalFX
            // finalizer on Vulkan, matching the cvar help text.
            if      (s == "svgf_basic")            want_kind = DenoiserKind::SvgfBasic;
            else if (s == "svgf_atrous")           want_kind = DenoiserKind::SvgfAtrous;
            else if (s == "svgf_basic_metalfx")    want_kind = DenoiserKind::SvgfBasic;
            else if (s == "svgf_atrous_metalfx")   want_kind = DenoiserKind::SvgfAtrous;
            else if (s == "nrd")                   want_kind = DenoiserKind::Nrd;
            else if (s == "optix_hdr")             want_kind = DenoiserKind::OptixHdr;
            else if (s == "optix_hdr_aov")         want_kind = DenoiserKind::OptixHdrAov;
            else if (s == "optix_temporal_hdr")    want_kind = DenoiserKind::OptixTemporalHdr;
            else if (s == "optix_temporal_hdr_aov") want_kind = DenoiserKind::OptixTemporalHdrAov;
        }
    }
    if (want_kind != DenoiserKind::Off && !device_->SupportsDenoise()) {
        // Backend doesn't support denoising yet (Vulkan: pipelines
        // still building on the worker thread, denoiser allocates
        // lazily on first dispatch). Stay on the noisy path until
        // SupportsDenoise flips true.
        want_kind = DenoiserKind::Off;
    }
    const bool want_denoiser = (want_kind != DenoiserKind::Off);
    if (want_denoiser != denoiser_active_ || want_kind != denoiser_kind_) {
        // Toggle (or kind switch -- e.g. svgf -> nrd): free the G-buffer
        // and force a history reset on the next allocation.
        denoiser_active_      = want_denoiser;
        denoiser_kind_        = want_kind;
        prev_view_proj_valid_ = false;
        // One-time log on transitions so the user sees which path
        // they're on (especially for the nrd-as-svgf-placeholder case).
        if (want_kind == DenoiserKind::Nrd) {
            LOG_INFO("engine: r_denoiser=nrd accepted -- routing through the in-house "
                     "SVGF kernels (atrous chain) until the NVIDIA RayTracingDenoiser "
                     "library is integrated. See Raytracer Plan/FOLLOW_UPS.md for the "
                     "integration plan.");
        } else if (want_kind == DenoiserKind::SvgfBasic) {
            LOG_INFO("engine: r_denoiser=svgf_basic -- temporal accumulation only "
                     "(no spatial filter)");
        } else if (want_kind == DenoiserKind::SvgfAtrous) {
            LOG_INFO("engine: r_denoiser=svgf_atrous -- temporal + a-trous "
                     "edge-aware filter");
        } else if (want_kind == DenoiserKind::SvgfBasicMetalFx) {
            LOG_INFO("engine: r_denoiser=svgf_basic_metalfx -- SVGF (temporal only) "
                     "-> MetalFX TemporalDenoisedScaler chain");
        } else if (want_kind == DenoiserKind::SvgfAtrousMetalFx) {
            LOG_INFO("engine: r_denoiser=svgf_atrous_metalfx -- SVGF (temporal + a-trous) "
                     "-> MetalFX TemporalDenoisedScaler chain");
        } else if (want_kind == DenoiserKind::MetalFX) {
            LOG_INFO("engine: r_denoiser=metalfx -- MetalFX TemporalDenoisedScaler active");
        } else if (want_kind == DenoiserKind::OptixHdr) {
            LOG_INFO("engine: r_denoiser=optix_hdr -- NVIDIA OptiX denoiser (HDR model) "
                     "via CUDA-Vulkan interop active");
        } else if (want_kind == DenoiserKind::OptixHdrAov) {
            LOG_INFO("engine: r_denoiser=optix_hdr_aov -- NVIDIA OptiX denoiser (HDR + "
                     "albedo + normal AOV model) via CUDA-Vulkan interop active");
        } else if (want_kind == DenoiserKind::OptixTemporalHdr) {
            LOG_INFO("engine: r_denoiser=optix_temporal_hdr -- NVIDIA OptiX denoiser "
                     "(TEMPORAL model: motion-vector flow + 1-frame output history) "
                     "via CUDA-Vulkan interop active");
        } else if (want_kind == DenoiserKind::OptixTemporalHdrAov) {
            LOG_INFO("engine: r_denoiser=optix_temporal_hdr_aov -- NVIDIA OptiX denoiser "
                     "(TEMPORAL_AOV model: motion + history + albedo + normal guides) "
                     "via CUDA-Vulkan interop active");
        }
        if (!want_denoiser && device_) {
            if (denoise_color_tex_id_    != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
            if (depth_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
            if (motion_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
            if (normal_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{normal_tex_id_});
            if (albedo_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{albedo_tex_id_});
            if (post_denoise_hdr_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
            if (cloud_trans_tex_id_      != 0) device_->DestroyTexture(pt::rhi::TextureHandle{cloud_trans_tex_id_});
            // SIGMA shadow visibility buffer (issue #115). Storage
            // buffer; mirrors the denoiser teardown rule -- the buffer
            // is allocated by the denoiser-active path and must be
            // freed when the user flips r_denoiser off.
            if (shadow_vis_buf_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{shadow_vis_buf_id_});
            // --- ReSTIR DI Phase A (issue #78) -----------------------------
            // Reservoir SSBOs live and die with the denoiser path -- the
            // Phase A composite writes into denoise_color, which only
            // exists when a denoiser is active. Freeing here mirrors the
            // shadow_vis_buf_id_ pattern.
            if (restir_reservoir_curr_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_});
            if (restir_reservoir_prev_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_prev_buf_id_});
            if (restir_reservoir_swap_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_swap_buf_id_});
            // --- end ReSTIR DI Phase A -------------------------------------
            // MetalFX specular guidance G-buffers (issue #118).
            if (specular_albedo_tex_id_       != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_albedo_tex_id_});
            if (roughness_tex_id_             != 0) device_->DestroyTexture(pt::rhi::TextureHandle{roughness_tex_id_});
            if (specular_hit_distance_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_hit_distance_tex_id_});
            for (auto& id : bloom_mip_tex_id_) {
                if (id != 0) device_->DestroyTexture(pt::rhi::TextureHandle{id});
                id = 0;
            }
            denoise_color_tex_id_ = depth_tex_id_ = motion_tex_id_ = 0;
            normal_tex_id_ = albedo_tex_id_ = post_denoise_hdr_tex_id_ = 0;
            cloud_trans_tex_id_   = 0;
            shadow_vis_buf_id_    = 0;
            restir_reservoir_curr_buf_id_ = 0;
            restir_reservoir_prev_buf_id_ = 0;
            restir_reservoir_swap_buf_id_ = 0;
            restir_alloc_w_ = 0;
            restir_alloc_h_ = 0;
            specular_albedo_tex_id_       = 0;
            roughness_tex_id_             = 0;
            specular_hit_distance_tex_id_ = 0;
        }
    }

    pt::rhi::FrameContext fc;
    {
        PT_ZONE_SCOPED_N("Device::BeginFrame");
        fc = device_->BeginFrame();
    }

    // Camera-movement detection -> reset accumulation.
    auto& cam = *camera_;
    bool cam_moved = (cam.pos.x != last_cam_pos_[0]) ||
                     (cam.pos.y != last_cam_pos_[1]) ||
                     (cam.pos.z != last_cam_pos_[2]) ||
                     (cam.yaw   != last_cam_yaw_)    ||
                     (cam.pitch != last_cam_pitch_)  ||
                     (cam.fov_deg != last_cam_fov_);
    if (cam_moved) accum_dirty_ = true;
    // Auto-fire reset_history on a "teleport" -- per-frame position
    // delta exceeds cam_teleport_threshold (default 5 units, ~33x
    // typical WASD movement). Catches cam_load (already explicitly
    // resets), cam_reset (same), AND inadvertent jumps where the user
    // typed a coordinate or moved a camera through a sphere whose
    // inside is black. Without this auto-detection, post-teleport
    // frames blend with the pre-teleport temporal history, visibly
    // bleeding stale color forward (the v0.3.4 inside-sphere
    // symptom). prev_view_proj_valid_ already gets cleared on
    // backend toggle / denoiser kind switch / first frame, so the
    // check is naturally a no-op on those paths. Only fires when the
    // engine's *been* rendering normally and the camera then jumps.
    if (prev_view_proj_valid_) {
        const float dx = cam.pos.x - last_cam_pos_[0];
        const float dy = cam.pos.y - last_cam_pos_[1];
        const float dz = cam.pos.z - last_cam_pos_[2];
        const float dist2 = dx * dx + dy * dy + dz * dz;
        float thresh = 5.0f;
        if (auto* v = pt::console::Console::Get().FindCVar("cam_teleport_threshold")) {
            thresh = v->GetFloat();
        }
        if (thresh > 0.0f && dist2 > thresh * thresh) {
            prev_view_proj_valid_ = false;
            LOG_INFO("engine: camera teleport detected ({:.2f}u > threshold {:.2f}u); "
                     "firing denoiser reset_history", std::sqrt(dist2), thresh);
        }
    }
    last_cam_pos_[0] = cam.pos.x; last_cam_pos_[1] = cam.pos.y; last_cam_pos_[2] = cam.pos.z;
    last_cam_yaw_   = cam.yaw;   last_cam_pitch_  = cam.pitch;  last_cam_fov_  = cam.fov_deg;

    // Lazily (re)create the HDR accumulation texture; reallocate if the
    // swapchain size changed.
    bool size_changed = (accum_w_ != static_cast<int>(fc.width)) ||
                        (accum_h_ != static_cast<int>(fc.height));
    if (accum_texture_id_ == 0 || size_changed) {
        if (accum_texture_id_ != 0) {
            device_->DestroyTexture(pt::rhi::TextureHandle{accum_texture_id_});
        }
        pt::rhi::TextureDesc td{
            .width  = fc.width,
            .height = fc.height,
            .format = pt::rhi::TextureFormat::RGBA32F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "accum_hdr",
        };
        auto h = device_->CreateTexture(td);
        accum_texture_id_ = h.id;
        if (accum_texture_id_ == 0) {
            LOG_ERROR("CreateTexture(RGBA32F {}x{}) failed", fc.width, fc.height);
            return;
        }
        accum_w_ = static_cast<int>(fc.width);
        accum_h_ = static_cast<int>(fc.height);
        accum_dirty_ = true;
    }

    // P10: G-buffer textures for the denoiser. Allocate (and re-allocate
    // on resize) only while r_denoiser is on -- saves ~32MB at 1080p when
    // denoising is off. The post-denoise HDR intermediate is the linear
    // RGBA16F MetalFX writes to; the `tonemap` compute kernel reads it
    // and writes the swapchain.
    // SVGF/NRD use the normal G-buffer for edge-aware spatial filtering.
    // The OptiX AOV variants (HdrAov + TemporalHdrAov) also use it as
    // the normal guide layer alongside albedo. MetalFX (macOS 26+
    // MTLFXTemporalDenoisedScaler) ALSO accepts normal + diffuse albedo
    // guidance inputs -- without them it falls back to a conservative
    // blur that doesn't converge on static cameras, looking visibly
    // worse than the accumulated no-denoiser path. Adding the
    // MetalFX kinds here so the engine allocates + fills the G-buffer
    // for them too. The plain OptiX HDR variants (Hdr + TemporalHdr)
    // don't take normals.
    const bool want_normal_gbuffer =
        (denoiser_kind_ == DenoiserKind::SvgfBasic           ||
         denoiser_kind_ == DenoiserKind::SvgfAtrous          ||
         denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx    ||
         denoiser_kind_ == DenoiserKind::SvgfAtrousMetalFx   ||
         denoiser_kind_ == DenoiserKind::MetalFX             ||
         denoiser_kind_ == DenoiserKind::Nrd                 ||
         denoiser_kind_ == DenoiserKind::OptixHdrAov         ||
         denoiser_kind_ == DenoiserKind::OptixTemporalHdrAov);
    // Albedo G-buffer: OptiX AOV variants (HdrAov + TemporalHdrAov),
    // MetalFX (and SVGF->MetalFX chained modes, since MetalFX is the
    // finalizer there too), AND the pure SVGF/NRD kinds (issue #119,
    // for albedo demodulation -- divides noisy radiance by albedo on
    // input to the SVGF chain, multiplies back on output, so the
    // depth/normal/luminance edge-stops denoise the lighting signal
    // rather than fighting texture detail).
    // Apple's MTLFXTemporalDenoisedScaler also takes diffuseAlbedoTexture
    // as a guidance input -- providing it lets MetalFX preserve surface
    // color edges instead of bleeding them.
    // Plain OptiX HDR models don't take albedo and don't run through
    // SVGF, so they stay opted out.
    const bool want_albedo_gbuffer =
        (denoiser_kind_ == DenoiserKind::OptixHdrAov         ||
         denoiser_kind_ == DenoiserKind::OptixTemporalHdrAov ||
         denoiser_kind_ == DenoiserKind::MetalFX             ||
         denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx    ||
         denoiser_kind_ == DenoiserKind::SvgfAtrousMetalFx   ||
         // SVGF / NRD albedo demod (#119): the in-house chain reads
         // albedo at the demod-divide site too, not just the MetalFX
         // family. SvgfBasic / SvgfAtrous / Nrd join the gbuffer set.
         denoiser_kind_ == DenoiserKind::SvgfBasic           ||
         denoiser_kind_ == DenoiserKind::SvgfAtrous          ||
         denoiser_kind_ == DenoiserKind::Nrd);
    // MetalFX specular-guidance G-buffers (issue #118). Apple's
    // MTLFXTemporalDenoisedScaler accepts specularAlbedo + roughness +
    // specularHitDistance as guidance inputs; with PR #114's normal +
    // diffuseAlbedo plumbing already in place, these close the gap with
    // DLSS Ray Reconstruction quality on Apple Silicon and kill the
    // 8x8 specular halos the user reported. Gated on MetalFX-family
    // kinds only -- SVGF / NRD / OptiX paths don't accept these
    // (separate issue for SVGF wiring; see #118's "Out of scope"
    // section). All three travel together: they all feed the same
    // MTLFXTemporalDenoisedScalerDescriptor so partial allocation
    // would just stall on a half-bound scaler.
    const bool want_specular_guidance_gbuffers =
        (denoiser_kind_ == DenoiserKind::MetalFX             ||
         denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx    ||
         denoiser_kind_ == DenoiserKind::SvgfAtrousMetalFx);
    // Bloom-without-denoiser path: when the user has r_bloom on but
    // no denoiser, the engine still needs `denoise_color` (as the
    // path tracer's linear-HDR output the bloom pyramid samples) and
    // the bloom mip chain so the post-process composite pass can
    // build a bloom layer. depth/motion/post_denoise_hdr/normal/
    // albedo stay unallocated -- those are denoiser-only. Backend-
    // agnostic flag, but each backend hits a different composite
    // kernel below:
    //   - Metal: engine's Tonemap.slang dispatch reads denoise_color
    //     + bloom_mip[0] directly (use_engine_tonemap branch).
    //   - Vulkan: routes through Denoise(Kind::FinalizeOnly), which
    //     dispatches VulkanNrdDenoiser::EncodeFinalizeOnly on its
    //     dedicated 4-binding layout. Tonemap.slang remains broken
    //     on Vulkan (black swapchain) so the engine deliberately
    //     skips it on this backend even for the no-denoiser case.
    // When this flag is off, the path tracer's inline tonemap to the
    // swapchain is the whole post-process -- no bloom is added.
    bool r_bloom_on_local = false;
    if (auto* v = C.FindCVar("r_bloom")) r_bloom_on_local = v->GetBool();
    bool bloom_without_denoiser =
        (!denoiser_active_) && r_bloom_on_local &&
        (tonemap_pipeline_id_ != 0);
    // Either path needs denoise_color + the bloom mip chain. Only the
    // denoiser path needs the depth/motion/post_denoise_hdr/normal/
    // albedo G-buffers. NOT const -- the allocation-failure branch
    // below clears denoiser_active_ / bloom_without_denoiser and
    // needs to drag this flag down with them so the downstream slot-2
    // bind and push.write_hdr_aux setter don't try to use a freshly-
    // failed (handle 0) denoise_color texture.
    bool need_hdr_aux = denoiser_active_ || bloom_without_denoiser;
    if (need_hdr_aux &&
        (denoise_color_tex_id_ == 0 || size_changed ||
         (denoiser_active_ &&
            (depth_tex_id_ == 0 || motion_tex_id_ == 0 ||
             post_denoise_hdr_tex_id_ == 0 ||
             (want_normal_gbuffer && normal_tex_id_ == 0) ||
             (want_albedo_gbuffer && albedo_tex_id_ == 0) ||
             (want_specular_guidance_gbuffers &&
              (specular_albedo_tex_id_ == 0 ||
               roughness_tex_id_ == 0 ||
               specular_hit_distance_tex_id_ == 0)))))) {
        if (denoise_color_tex_id_     != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        // depth/motion/normal/albedo/post_denoise_hdr only exist on the
        // denoiser path. Destroy them only when the denoiser path is
        // active -- otherwise we'd UAF them if a prior denoiser session
        // left them allocated and we're now in the bloom-only branch
        // (they stay around as harmless leftovers until denoiser turns
        // back on, which re-fires this block and reallocates them).
        if (denoiser_active_) {
            if (depth_tex_id_             != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
            if (motion_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
            if (normal_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{normal_tex_id_});
            if (albedo_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{albedo_tex_id_});
            if (post_denoise_hdr_tex_id_  != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
            if (cloud_trans_tex_id_       != 0) device_->DestroyTexture(pt::rhi::TextureHandle{cloud_trans_tex_id_});
            // SIGMA shadow visibility buffer (issue #115). Storage
            // buffer; same denoiser_active_ lifecycle as the textures
            // above. Resize teardown destroys it before the new
            // allocation runs further down.
            if (shadow_vis_buf_id_        != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{shadow_vis_buf_id_});
            // --- ReSTIR DI Phase A (issue #78) -----------------------------
            if (restir_reservoir_curr_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_});
            if (restir_reservoir_prev_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_prev_buf_id_});
            if (restir_reservoir_swap_buf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{restir_reservoir_swap_buf_id_});
            // --- end ReSTIR DI Phase A -------------------------------------
            // MetalFX specular guidance G-buffers (issue #118).
            if (specular_albedo_tex_id_       != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_albedo_tex_id_});
            if (roughness_tex_id_             != 0) device_->DestroyTexture(pt::rhi::TextureHandle{roughness_tex_id_});
            if (specular_hit_distance_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{specular_hit_distance_tex_id_});
            // Explicitly zero the IDs here. The subsequent CreateTexture
            // calls overwrite them on the success path, but a CreateTexture
            // failure below leaves the ID still pointing at a freed handle
            // unless we zero it first. Matches the pattern used at engine
            // shutdown (Shutdown() ~line 1078) and the r_denoiser-off
            // teardown above (~line 2697).
            depth_tex_id_            = 0;
            motion_tex_id_           = 0;
            normal_tex_id_           = 0;
            albedo_tex_id_           = 0;
            post_denoise_hdr_tex_id_ = 0;
            cloud_trans_tex_id_      = 0;
            shadow_vis_buf_id_       = 0;
            restir_reservoir_curr_buf_id_ = 0;
            restir_reservoir_prev_buf_id_ = 0;
            restir_reservoir_swap_buf_id_ = 0;
            restir_alloc_w_               = 0;
            restir_alloc_h_               = 0;
            specular_albedo_tex_id_       = 0;
            roughness_tex_id_             = 0;
            specular_hit_distance_tex_id_ = 0;
        }
        auto color_h = device_->CreateTexture({
            .width = fc.width, .height = fc.height,
            .format = pt::rhi::TextureFormat::RGBA16F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "denoise_color",
        });
        denoise_color_tex_id_    = color_h.id;
        // State transition log: tells future readers (and rules out
        // silent failure when the on-screen bloom doesn't match
        // expectations) WHICH consumer is about to start sampling
        // denoise_color this frame -- the denoiser's Encode (in which
        // case the G-buffer block below also fires) or the engine
        // bloom + finalize chain (this is the only place that signals
        // the bloom-without-denoiser path took over). The downstream
        // finalize is Tonemap.slang on Metal, Denoise(FinalizeOnly)
        // -> VulkanNrdDenoiser::EncodeFinalizeOnly on Vulkan.
        if (bloom_without_denoiser && !denoiser_active_) {
            LOG_INFO("engine: bloom-without-denoiser engaged -- allocated denoise_color "
                     "({}x{} RGBA16F) as bloom-pyramid + finalize HDR source",
                     fc.width, fc.height);
        }
        if (denoiser_active_) {
            auto depth_h = device_->CreateTexture({
                .width = fc.width, .height = fc.height,
                .format = pt::rhi::TextureFormat::R32F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "denoise_depth",
            });
            auto motion_h = device_->CreateTexture({
                .width = fc.width, .height = fc.height,
                .format = pt::rhi::TextureFormat::RG16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "denoise_motion",
            });
            auto post_h = device_->CreateTexture({
                .width = fc.width, .height = fc.height,
                .format = pt::rhi::TextureFormat::RGBA16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "post_denoise_hdr",
            });
            depth_tex_id_            = depth_h.id;
            motion_tex_id_           = motion_h.id;
            post_denoise_hdr_tex_id_ = post_h.id;
            // Cloud transmittance G-buffer (issue #46 follow-up).
            // R32F (matches depth_tex's format; the value is a single
            // [0,1] scalar so R8_UNORM would be ideal but isn't in
            // the RHI's TextureFormat enum -- R32F is the existing
            // single-channel format and the 4x memory cost over R8
            // is 4.5 MB / 12 MB at 1080p / 4K, trivial vs the rest
            // of the denoiser texture budget). PathTrace writes the
            // camera-ray cloud-march transmittance; StarsComposite
            // reads + multiplies the celestial composite.
            auto cloud_h = device_->CreateTexture({
                .width = fc.width, .height = fc.height,
                .format = pt::rhi::TextureFormat::R32F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "cloud_trans",
            });
            cloud_trans_tex_id_ = cloud_h.id;
            // SIGMA shadow visibility buffer (issue #115). One R32F per
            // pixel of sun-NEE shadow-ray transmittance, written by
            // PathTrace.slang and bilateral-filtered by SigmaShadow.slang
            // before being multiplied into post_denoise_hdr. A storage
            // BUFFER (not a texture) because Metal's compute pipeline
            // caps RWTexture2D bindings at 8 per kernel and PathTrace
            // is already exactly at that limit (output, accum_hdr,
            // denoise_color, depth_tex, motion_tex, normal_tex,
            // albedo_tex, cloud_trans_tex). A 9th RW texture would fail
            // pipeline build on Apple Silicon. Same workaround
            // DenoiseAtrous.slang uses for its variance + moments
            // buffers. Size: width * height * sizeof(float)
            // = ~8.3 MB at 1080p, trivial.
            auto svis_h = device_->CreateBuffer({
                .size  = static_cast<std::size_t>(fc.width) *
                         static_cast<std::size_t>(fc.height) *
                         sizeof(float),
                .usage = pt::rhi::BufferUsage::Storage,
                .debug_name = "shadow_vis",
            });
            shadow_vis_buf_id_ = svis_h.id;
            // --- ReSTIR DI Phase A (issue #78) -----------------------------
            // Allocate three per-pixel reservoir SSBOs (curr / prev /
            // swap). Each is sized W * H * sizeof(pt::renderer::Reservoir)
            // = W * H * 64 B. At 1080p: 127 MB per buffer, ~381 MB total
            // for the triple ring. Same denoiser_active_ lifecycle as
            // shadow_vis_buf_id_ above; the buffers are freed and
            // reallocated on swapchain resize.
            //
            // We allocate them unconditionally whenever the denoiser is
            // active (not gated on r_restir) because the Phase A
            // dispatch chain SHORT-CIRCUITS cleanly on r_restir=0 via
            // the per-pass push gates -- but PathTrace.slang's binding
            // 29 always needs a real buffer to keep Metal's MSL slot
            // assignment stable (same lesson the
            // shadow_vis_buf_id_ slot-11 fallback below the dispatch
            // captures). Cost: 380 MB VRAM standby on a single-buffer
            // denoiser path the user might not even toggle ReSTIR on
            // for. Acceptable for Phase A; an optimization that keys
            // allocation on r_restir = 1 only is a trivial follow-up.
            std::size_t kReservoirBytesPerPixel =
                sizeof(pt::renderer::Reservoir);
            std::size_t reservoir_total_bytes =
                static_cast<std::size_t>(fc.width) *
                static_cast<std::size_t>(fc.height) *
                kReservoirBytesPerPixel;
            auto reservoir_curr_h = device_->CreateBuffer({
                .size       = reservoir_total_bytes,
                .usage      = pt::rhi::BufferUsage::Storage,
                .debug_name = "restir_reservoir_curr",
            });
            auto reservoir_prev_h = device_->CreateBuffer({
                .size       = reservoir_total_bytes,
                .usage      = pt::rhi::BufferUsage::Storage,
                .debug_name = "restir_reservoir_prev",
            });
            auto reservoir_swap_h = device_->CreateBuffer({
                .size       = reservoir_total_bytes,
                .usage      = pt::rhi::BufferUsage::Storage,
                .debug_name = "restir_reservoir_swap",
            });
            restir_reservoir_curr_buf_id_ = reservoir_curr_h.id;
            restir_reservoir_prev_buf_id_ = reservoir_prev_h.id;
            restir_reservoir_swap_buf_id_ = reservoir_swap_h.id;
            restir_alloc_w_ = fc.width;
            restir_alloc_h_ = fc.height;
            // --- end ReSTIR DI Phase A -------------------------------------
            // Normal G-buffer: SVGF/NRD use it for edge-aware spatial
            // filtering; the OptiX AOV denoiser uses it as a guide layer.
            // MetalFX ignores normals and the path tracer's normal write is
            // gated by PT_TARGET_SPIRV (so on Metal there's no shader
            // binding expecting it either).
            if (want_normal_gbuffer) {
                auto normal_h = device_->CreateTexture({
                    .width = fc.width, .height = fc.height,
                    .format = pt::rhi::TextureFormat::RGBA16F,
                    .usage  = pt::rhi::TextureUsage::Storage,
                    .debug_name = "denoise_normal",
                });
                normal_tex_id_ = normal_h.id;
            } else {
                normal_tex_id_ = 0;
            }
            // Albedo G-buffer: OptiX AOV only. RGBA16F to keep headroom
            // for HDR-encoded materials (emissive base colors, etc.) and to
            // match the format convention of the other G-buffers.
            if (want_albedo_gbuffer) {
                auto albedo_h = device_->CreateTexture({
                    .width = fc.width, .height = fc.height,
                    .format = pt::rhi::TextureFormat::RGBA16F,
                    .usage  = pt::rhi::TextureUsage::Storage,
                    .debug_name = "denoise_albedo",
                });
                albedo_tex_id_ = albedo_h.id;
                LOG_INFO("engine: allocated denoise_albedo G-buffer ({}x{} RGBA16F) "
                         "(consumers: OptiX AOV, MetalFX, SVGF demod)", fc.width, fc.height);
            } else {
                albedo_tex_id_ = 0;
            }
            // MetalFX specular-guidance G-buffers (issue #118). All
            // three travel together: they feed the same MetalFX
            // descriptor (set in pt_metalfx_create), so partial
            // allocation would just stall the scaler on a nil binding.
            //
            // specular_albedo: RGBA16F per-pixel F0 (matches the enum's
            //   format convention and gives metals room to encode the
            //   full RGB Fresnel reflectance).
            // roughness: R32F single-channel surface roughness in [0,1].
            //   The RHI doesn't expose R16F today and the precision
            //   gap is academic for a guidance input -- R32F costs
            //   ~33MB/4K (3840*2160*4B) vs R16F's ~16.5MB and saves
            //   us a new format slot.
            // specular_hit_distance: R32F single-channel world-units
            //   distance. Same R16F-vs-R32F trade as roughness; using
            //   R32F unifies allocation code with cloud_trans_tex /
            //   depth_tex.
            if (want_specular_guidance_gbuffers) {
                auto spec_albedo_h = device_->CreateTexture({
                    .width = fc.width, .height = fc.height,
                    .format = pt::rhi::TextureFormat::RGBA16F,
                    .usage  = pt::rhi::TextureUsage::Storage,
                    .debug_name = "denoise_specular_albedo",
                });
                specular_albedo_tex_id_ = spec_albedo_h.id;
                auto roughness_h = device_->CreateTexture({
                    .width = fc.width, .height = fc.height,
                    .format = pt::rhi::TextureFormat::R32F,
                    .usage  = pt::rhi::TextureUsage::Storage,
                    .debug_name = "denoise_roughness",
                });
                roughness_tex_id_ = roughness_h.id;
                auto spec_hit_dist_h = device_->CreateTexture({
                    .width = fc.width, .height = fc.height,
                    .format = pt::rhi::TextureFormat::R32F,
                    .usage  = pt::rhi::TextureUsage::Storage,
                    .debug_name = "denoise_specular_hit_distance",
                });
                specular_hit_distance_tex_id_ = spec_hit_dist_h.id;
                LOG_INFO("engine: allocated MetalFX specular guidance G-buffers ({}x{}) "
                         "[specular_albedo RGBA16F + roughness R32F + specular_hit_distance R32F]",
                         fc.width, fc.height);
            } else {
                specular_albedo_tex_id_       = 0;
                roughness_tex_id_             = 0;
                specular_hit_distance_tex_id_ = 0;
            }
        }
        prev_view_proj_valid_ = false;        // history is invalid after resize
        if (denoise_color_tex_id_ == 0) {
            LOG_ERROR("denoise_color allocation failed at {}x{}", fc.width, fc.height);
            // Both paths need denoise_color, so fall back to the
            // inline-tonemap-to-swapchain path (path tracer writes
            // LDR straight to the swap, no bloom). need_hdr_aux drops
            // with the two source flags so the slot-2 bind and
            // push.write_hdr_aux setter downstream don't drive a
            // handle-0 / unbound texture through the shader.
            denoiser_active_ = false;
            bloom_without_denoiser = false;
            need_hdr_aux = false;
        } else if (denoiser_active_ &&
            (depth_tex_id_      == 0 || motion_tex_id_       == 0 ||
             post_denoise_hdr_tex_id_ == 0 ||
             (want_normal_gbuffer && normal_tex_id_ == 0) ||
             (want_albedo_gbuffer && albedo_tex_id_ == 0) ||
             (want_specular_guidance_gbuffers &&
              (specular_albedo_tex_id_ == 0 ||
               roughness_tex_id_ == 0 ||
               specular_hit_distance_tex_id_ == 0)))) {
            LOG_ERROR("denoiser G-buffer allocation failed at {}x{}", fc.width, fc.height);
            denoiser_active_ = false;
        }
        // Bloom mip chain. mip[0] is half-res; each subsequent mip
        // halves again. Caps at 1x1 if the swapchain is tiny. Allocated
        // alongside denoise_color whenever EITHER the denoiser path or
        // the Metal bloom-without-denoiser path is active -- both feed
        // the same engine-side Tonemap.slang composite that reads mip[0].
        for (int i = 0; i < kBloomMips; ++i) {
            if (bloom_mip_tex_id_[i] != 0) {
                device_->DestroyTexture(pt::rhi::TextureHandle{bloom_mip_tex_id_[i]});
                bloom_mip_tex_id_[i] = 0;
            }
            std::uint32_t bw = std::max(1u, fc.width  >> (i + 1));
            std::uint32_t bh = std::max(1u, fc.height >> (i + 1));
            auto bm = device_->CreateTexture({
                .width = bw, .height = bh,
                .format = pt::rhi::TextureFormat::RGBA16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "bloom_mip",
            });
            bloom_mip_tex_id_[i] = bm.id;
            bloom_mip_w_[i]      = bw;
            bloom_mip_h_[i]      = bh;
        }
        // 1x1 zero-filled placeholder bound to the tonemap pass when
        // bloom is disabled. Tonemap's bloom_intensity push is set to
        // 0 in that case, but Metal still demands the slot be bound;
        // a tiny dummy avoids leaving the slot empty.
        if (bloom_dummy_tex_id_ == 0) {
            auto dh = device_->CreateTexture({
                .width = 1, .height = 1,
                .format = pt::rhi::TextureFormat::RGBA16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "bloom_dummy",
            });
            bloom_dummy_tex_id_ = dh.id;
            std::uint16_t zero[4] {0,0,0,0};
            if (dh.id != 0) device_->WriteTexture(dh, zero, sizeof(zero));
        }
    }

    auto* cb = device_->AcquireCommandBuffer();
    cb->BindComputePipeline(pt::rhi::PipelineHandle{pathtrace_pipeline_id_});
    cb->BindStorageTexture(0, fc.swapchain_image);
    cb->BindStorageTexture(1, pt::rhi::TextureHandle{accum_texture_id_});

    // Slot mapping (matches PathTrace.slang and the Metal-buffer layout
    // Slang produces): AS at buffer(0), then storage buffers in declared
    // order. Our RHI uses separate "slot namespaces" so the user-facing
    // numbers below are 1-based for buffers (the AS, declared at vk slot
    // 2, lands at buffer(0) on Metal; mesh_positions at buffer(1), etc.).
    // Voxel destruction Phase 1 (#140) hide-mesh gate. When the demo
    // toggle is on AND we have at least one VoxelGrid uploaded, treat
    // the CSG mesh as absent for this frame so the path tracer sees
    // ONLY the voxelized form. Both the HW (TLAS) and SW (linear
    // triangle scan) branches are suppressed via local flags --
    // scene_tlas_id_ / mesh_tri_count_ themselves are NOT mutated, so
    // flipping r_voxelize_demo back to 0 instantly restores the mesh
    // without a re-bake.
    const bool voxel_hide_mesh = voxel_demo_active_;
    bool tlas_present = (scene_tlas_id_ != 0) && !voxel_hide_mesh;
    // Software-mesh fallback (mesh_tri_count_ > 0 && !tlas_present):
    // we still bind the vertex + index buffers and let PathTrace.slang
    // linear-scan triangles. Currently exercised on Mac-Vulkan when the
    // installed MoltenVK build doesn't expose VK_KHR_acceleration_structure
    // (pre-1.3); the gold path stays on top of VK_KHR_ray_query when it
    // is available. See VulkanDevice's RT extension probe log.
    const bool sw_mesh_present = (!tlas_present && mesh_tri_count_ > 0 &&
                                  box_vbuf_id_ != 0 && box_ibuf_id_ != 0 &&
                                  !voxel_hide_mesh);
    if (tlas_present) {
        cb->BindAccelStruct(2, pt::rhi::AccelStructHandle{scene_tlas_id_});
        if (box_vbuf_id_ != 0) cb->BindBuffer(1, pt::rhi::BufferHandle{box_vbuf_id_}, 0);
        if (box_ibuf_id_ != 0) cb->BindBuffer(2, pt::rhi::BufferHandle{box_ibuf_id_}, 0);
    } else if (sw_mesh_present) {
        cb->BindBuffer(1, pt::rhi::BufferHandle{box_vbuf_id_}, 0);
        cb->BindBuffer(2, pt::rhi::BufferHandle{box_ibuf_id_}, 0);
    }
    if (prim_buffer_id_ != 0) {
        cb->BindBuffer(3, pt::rhi::BufferHandle{prim_buffer_id_}, 0);
    }
    // P11 MIS CDFs at MSL slots 4/5. CRITICAL: always bind something
    // here even if no env map is loaded -- the shader declares these
    // buffers, so leaving slots 4/5 unbound shifts the dynamically-
    // assigned push-constant slot to the wrong index (Metal computes
    // it as max-bound + 1) and the shader reads garbage push fields
    // -> all-black render. When env CDFs aren't available we fall
    // back to the always-present placeholder storage buffer; the
    // shader never reads from slots 4/5 when env_map_present == 0.
    // (Previously we reused prim_buffer_id_ here, but that id is 0
    // when pt_smoke_skip_prim_seed=1 and no analytic prims exist.)
    pt::rhi::BufferHandle slot4 = (env_marginal_cdf_id_ != 0)
        ? pt::rhi::BufferHandle{env_marginal_cdf_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    pt::rhi::BufferHandle slot5 = (env_conditional_cdf_id_ != 0)
        ? pt::rhi::BufferHandle{env_conditional_cdf_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot4.id != 0) cb->BindBuffer(4, slot4, 0);
    if (slot5.id != 0) cb->BindBuffer(5, slot5, 0);
    // GPU-driven exposure scalar at slot 6. PathTrace reads it for
    // the final tonemap; AutoExposure (dispatched later this frame)
    // updates it. Replaces the per-frame readback path that stalled
    // the GPU on dGPU.
    if (exposure_state_id_ != 0) {
        cb->BindBuffer(6, pt::rhi::BufferHandle{exposure_state_id_}, 0);
    }
    // Analytic-primitive BVH nodes at slot 7. The shader only reads
    // this buffer when push.bvh_params.y > 0 (i.e. the engine built a
    // tree); but the binding must always resolve to a real buffer for
    // the same reason slot 4/5 do (Metal's dynamic slot assignment
    // depends on max-bound + 1 of the contiguous range -- leaving 7
    // unbound would shift the push slot). When no BVH is built we
    // fall back to the always-present placeholder storage buffer.
    pt::rhi::BufferHandle slot7 = (bvh_node_buffer_id_ != 0)
        ? pt::rhi::BufferHandle{bvh_node_buffer_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot7.id != 0) cb->BindBuffer(7, slot7, 0);
    // Engine slots 8 / 9 -> shader bindings 19 / 20 (tri_bvh_nodes,
    // tri_bvh_permuted_ids; PR #106 follow-up). Same "always bind
    // something" rule as slots 4/5/7 -- Metal's dynamic push-slot
    // assignment is max-bound + 1, so leaving these unbound when the
    // CSG bake hasn't run yet would shift the push slot off by two and
    // corrupt every push field. When no triangle BVH exists we re-use
    // the primitive buffer as a harmless placeholder; the shader gates
    // its reads on bvh_params.w > 0 (the runtime "is the BVH
    // populated" signal).
    pt::rhi::BufferHandle slot8 = (tri_bvh_nodes_id_ != 0)
        ? pt::rhi::BufferHandle{tri_bvh_nodes_id_}
        : pt::rhi::BufferHandle{prim_buffer_id_};
    pt::rhi::BufferHandle slot9 = (tri_bvh_permuted_ids_id_ != 0)
        ? pt::rhi::BufferHandle{tri_bvh_permuted_ids_id_}
        : pt::rhi::BufferHandle{prim_buffer_id_};
    if (slot8.id != 0) cb->BindBuffer(8, slot8, 0);
    if (slot9.id != 0) cb->BindBuffer(9, slot9, 0);
    // --- SDF Phase 1 (#97) -------------------------------------------------
    // SDF cluster buffer at engine slot 10 (MSL slot order;
    // vk::binding(21, 0)). Moved from engine slot 8 / vk::binding 19
    // to make room for PR #106's tri_bvh_* at slots 8/9 / bindings
    // 19/20. The shader only reads this when push.sdf_params.x > 0;
    // the binding still has to resolve to a real buffer for the same
    // reason 4/5/7/8/9 do (Metal computes dynamic slots from
    // max-bound + 1 of the contiguous range -- leaving 10 unbound
    // would shift the push slot by one and corrupt every field). Use
    // the always-present placeholder storage buffer when no SDF
    // clusters exist. (Earlier versions reused prim_buffer_id_, but
    // that id is 0 when pt_smoke_skip_prim_seed=1 and no analytic
    // prims have been seeded -- the binding would silently drop and
    // shift the push slot.)
    pt::rhi::BufferHandle slot10 = (sdf_cluster_buffer_id_ != 0)
        ? pt::rhi::BufferHandle{sdf_cluster_buffer_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot10.id != 0) cb->BindBuffer(10, slot10, 0);
    // --- end SDF Phase 1 ---------------------------------------------------
    // (light primitives slot bind moved to slot 12 below, since SIGMA
    // claimed slot 11 for shadow_vis_buf on the integration branch.)
    // P10 G-buffer texture binds. The shader's vk::binding numbers (6/7/8)
    // are Vulkan descriptor slots; on Metal Slang assigns texture(N) in
    // declaration order, so output/accum/denoise_color/depth/motion/env
    // become texture(0..5). The Metal RHI treats the slot arg as the MSL
    // texture index, so we bind them at 2/3/4/5 here.
    //
    // Slot 2 (denoise_color) is bound for the bloom-without-denoiser
    // Metal path too -- the path tracer writes linear HDR into it
    // (gated on push.write_hdr_aux) and the engine Tonemap.slang pass
    // samples it as the bloom-pyramid + composite source. Slots 3/4
    // (depth/motion) stay denoiser-only: the shader's G-buffer block
    // is still gated on denoiser_enabled, and we don't allocate those
    // textures in the bloom-only path.
    if (need_hdr_aux) {
        cb->BindStorageTexture(2, pt::rhi::TextureHandle{denoise_color_tex_id_});
    }
    if (denoiser_active_) {
        cb->BindStorageTexture(3, pt::rhi::TextureHandle{depth_tex_id_});
        cb->BindStorageTexture(4, pt::rhi::TextureHandle{motion_tex_id_});
    }
    // Engine slot 8 -> vk::binding 16 (normal_tex). Vulkan-only:
    // PathTrace.slang's normal_tex declaration is gated by
    // PT_TARGET_SPIRV so Metal builds don't expect anything here, and
    // the engine slot is only used by the SVGF/NRD path. We also bind
    // the placeholder env_map / star_map / moon_map at slots they
    // need; binding 16 is similarly safe-to-bind because the Metal
    // backend's bound_tex_[8] is exactly 8 entries (slot 8 silently
    // dropped) and the Vulkan backend extends to 12.
    if (denoiser_active_ && normal_tex_id_ != 0) {
        cb->BindStorageTexture(8, pt::rhi::TextureHandle{normal_tex_id_});
    }
    // Engine slot 9 -> vk::binding 17 (albedo_tex). Vulkan + OptiX-AOV
    // only. PathTrace.slang's albedo_tex is PT_TARGET_SPIRV-gated so
    // Metal builds don't expect anything; the Metal backend's 8-slot
    // bound_tex_ silently drops slot 9 anyway. When albedo_tex_id_ is
    // 0 (any non-AOV mode) we don't bind here AND the host sets
    // push.write_albedo_gbuffer = 0 so the shader elides its write --
    // required under VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, which
    // (unlike the older nullDescriptor scheme) leaves the slot absent
    // from the descriptor set rather than as a safe null pass-through.
    if (denoiser_active_ && albedo_tex_id_ != 0) {
        cb->BindStorageTexture(9, pt::rhi::TextureHandle{albedo_tex_id_});
    }
    // Engine slot 10 -> vk::binding 22 (cloud_trans_tex). R32F per-
    // pixel transmittance written by PathTrace's volumetric cloud
    // march, read by StarsComposite. Allocated alongside the rest of
    // the denoiser textures (denoiser_active_), so the bind fires
    // any time the path-tracer dispatch needs to write the G-buffer
    // for the composite chain.
    if (denoiser_active_ && cloud_trans_tex_id_ != 0) {
        cb->BindStorageTexture(10, pt::rhi::TextureHandle{cloud_trans_tex_id_});
    }
    // Engine buffer slot 11 -> vk::binding 23 (shadow_vis_buf). One
    // R32F per pixel, sun-NEE shadow-ray transmittance from the path
    // tracer's primary-hit pass. Storage BUFFER, not a texture --
    // Apple Silicon's 8-RW-texture cap is already saturated by the
    // existing PathTrace G-buffer set (see comment at the
    // shadow_vis_buf declaration in PathTrace.slang).
    //
    // ALWAYS bind something to slot 11 (same "max-bound + 1 must land
    // at the kernel's expected Push slot" rule slots 4/5/7/8/9/10
    // already follow). When the denoiser is off OR the shadow_vis_buf
    // hasn't been allocated yet, fall back to the always-present
    // placeholder storage buffer; the shader gates its reads / writes
    // on `write_shadow_vis` (set to 0 by the engine when demod is off
    // OR the buffer is missing) so this binding is dead at runtime --
    // it only exists to keep Metal's push-slot computation
    // (max-bound + 1 = slot 12, matching MSL's Push assignment)
    // stable. Without this fallback the OFF-case push slot would
    // collapse to 11 and corrupt every push field; first noticed as
    // the golden_cornell_csg__metal__off_diff regression on the
    // first build of issue #115.
    pt::rhi::BufferHandle slot11 = (denoiser_active_ && shadow_vis_buf_id_ != 0)
        ? pt::rhi::BufferHandle{shadow_vis_buf_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot11.id != 0) cb->BindBuffer(11, slot11, 0);
    // --- Light primitives (#73) --------------------------------------------
    // Analytic light list at engine slot 12 (MSL slot order;
    // vk::binding(27, 0)). Slot 12 because SIGMA's shadow_vis_buf
    // already claims slot 11 -- light_prims is declared AFTER
    // shadow_vis_buf in PathTrace.slang so it lands one MSL slot
    // higher. The shader gates its NEE-to-lights branch on
    // push.light_count > 0; the binding still has to resolve to a
    // real buffer to keep Metal's push-slot computation stable
    // (same lesson the slot11 fallback above captures). Falls back
    // to the always-present placeholder storage buffer when no
    // lights are active.
    pt::rhi::BufferHandle slot12 = (light_buffer_id_ != 0)
        ? pt::rhi::BufferHandle{light_buffer_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot12.id != 0) cb->BindBuffer(12, slot12, 0);
    // --- end Light primitives ----------------------------------------------
    // --- Light tree (#129) -------------------------------------------------
    // Hierarchical light tree at engine slot 13 (MSL slot 13;
    // vk::binding(28, 0)). Declared AFTER light_prims in PathTrace.slang
    // so MSL lands one slot higher. The shader gates traversal on
    // push.light_tree_node_count > 0 AND r_light_tree=1; binding still
    // has to resolve to a real buffer for Metal's push-slot computation
    // (same lesson slot11/slot12 fallbacks capture). Falls back to the
    // always-present placeholder storage buffer when no tree is built.
    //
    // DOUBLE-BUFFERED: the engine alternates between two GPU node
    // buffers (light_tree_buffer_ids_[0/1]) so the worker thread can
    // upload to the inactive slot while the path tracer reads the
    // active one. `light_tree_active_slot_` is flipped by
    // EnsureLightTreeUploaded once the worker publishes a new tree.
    // Until the very first tree has been built we still fall back to
    // the placeholder, same as before.
    const std::uint64_t active_lt_buf =
        light_tree_first_built_
            ? light_tree_buffer_ids_[light_tree_active_slot_]
            : 0u;
    pt::rhi::BufferHandle slot13 = (active_lt_buf != 0)
        ? pt::rhi::BufferHandle{active_lt_buf}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot13.id != 0) cb->BindBuffer(13, slot13, 0);
    // --- end Light tree ----------------------------------------------------
    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    // Reservoir SSBO at engine slot 14 -> Metal MSL slot 14 ->
    // vk::binding(29). Lands AFTER the light tree at slot 13 (#129).
    // Always bind something so Metal's push-slot computation
    // (max-bound + 1 = slot 15) stays stable -- the shader's r_restir
    // gate elides the WRITE when r_restir = 0 but the binding has to
    // resolve to a real buffer. Falls back to the always-present
    // placeholder when ReSTIR resources haven't been allocated yet
    // (denoiser-off path, or pre-first-frame). Same pattern slot 11
    // (shadow_vis) / slot 12 (light_prims) / slot 13 (light_tree) follow.
    pt::rhi::BufferHandle slot14 =
        (denoiser_active_ && restir_reservoir_curr_buf_id_ != 0)
            ? pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_}
            : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot14.id != 0) cb->BindBuffer(14, slot14, 0);
    // --- end ReSTIR DI Phase A ---------------------------------------------
    // --- Fluid Phase 1 (#136) -- smoke emitters ----------------------------
    // Smoke emitter list at engine slot 15 (MSL slot order;
    // vk::binding(30, 0)). The agent originally targeted slot 13 /
    // binding 28 but that's owned by light tree #129 after the integration
    // merge -- moved to slot 15 / binding 30 (one past ReSTIR's slot 14 /
    // binding 29). Always-bound (placeholder fallback) for the same
    // max-bound + 1 push-slot rule; the shader gates its emitter loop on
    // push.smoke_params.x > 0 AND push.smoke_params.y != 0.
    pt::rhi::BufferHandle slot15 = (smoke_buffer_id_ != 0)
        ? pt::rhi::BufferHandle{smoke_buffer_id_}
        : pt::rhi::BufferHandle{placeholder_storage_id_};
    if (slot15.id != 0) cb->BindBuffer(15, slot15, 0);
    // --- end Fluid Phase 1 -------------------------------------------------
    // Engine slots 11/12/13 -> vk::bindings 24/25/26: the MetalFX
    // specular-guidance trio (issue #118). Path tracer writes them
    // alongside the existing G-buffers when the write_specular_*_gbuffer
    // push gates are non-zero; MetalFX (and SVGF->MetalFX chained
    // kinds) consume them via MTLFXTemporalDenoisedScalerDescriptor's
    // specularAlbedoTexture / roughnessTexture / specularHitDistanceTexture.
    // Engine allocates them only when want_specular_guidance_gbuffers
    // is set (see allocation block above); the bind is gated on
    // non-zero id so non-MetalFX dispatches leave the slots unbound
    // and the matching push gates elide the shader writes.
    if (denoiser_active_ && specular_albedo_tex_id_ != 0) {
        cb->BindStorageTexture(11, pt::rhi::TextureHandle{specular_albedo_tex_id_});
    }
    if (denoiser_active_ && roughness_tex_id_ != 0) {
        cb->BindStorageTexture(12, pt::rhi::TextureHandle{roughness_tex_id_});
    }
    if (denoiser_active_ && specular_hit_distance_tex_id_ != 0) {
        cb->BindStorageTexture(13, pt::rhi::TextureHandle{specular_hit_distance_tex_id_});
    }
    if (env_map_tex_id_ != 0) {
        cb->BindStorageTexture(5, pt::rhi::TextureHandle{env_map_tex_id_});
    }
    // BSC starmap (always bind when present so the shader's binding is
    // satisfied; sampling is gated by star_map_present in the push so
    // the texture being a black 1x1 placeholder is not a problem).
    if (moon_map_tex_id_ != 0) {
        // Slot 13 in the shader; engine uses slot 7 in the local
        // bind list (slot 6 was already taken by star_map). The RHI
        // maps these to the path-tracer pipeline's binding indices.
        cb->BindStorageTexture(7, pt::rhi::TextureHandle{moon_map_tex_id_});
    }
    if (star_map_tex_id_ != 0) {
        cb->BindStorageTexture(6, pt::rhi::TextureHandle{star_map_tex_id_});
    }

    std::uint32_t bounces = 8;
    if (auto* v = C.FindCVar("r_max_bounces")) bounces = (std::uint32_t)v->GetInt();
    std::uint32_t spp = 1;
    if (auto* v = C.FindCVar("r_spp")) {
        int n = v->GetInt();
        if (n < 1)  n = 1;
        if (n > 32) n = 32;     // clamp to keep a runaway value from freezing the GPU
        spp = static_cast<std::uint32_t>(n);
    }

    const auto fwd   = cam.Forward();
    const auto right = cam.Right();
    const auto up    = cam.Up();
    const float aspect = (fc.height > 0) ? float(fc.width) / float(fc.height) : 1.0f;

    // Build current view*projection. Used by the shader's G-buffer pass
    // to reproject hit world positions into both current and previous
    // screen space (motion vectors). Identity prev_view_proj on the
    // first frame yields zero motion, which the denoiser tolerates.
    glm::mat4 view = glm::lookAtRH(cam.pos, cam.pos + fwd, glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(cam.fov_deg), aspect, 0.05f, 500.0f);
    glm::mat4 curr_view_proj = proj * view;

    struct PtPush {
        float pos_fovtan[4];
        float fwd_aspect[4];
        float right_xyz[4];
        float up_xyz[4];
        std::uint32_t frame_index;
        std::uint32_t reset_accum;
        std::uint32_t max_bounces;
        std::uint32_t tlas_present;
        std::uint32_t prim_count;
        std::uint32_t spp;
        std::uint32_t denoiser_enabled;
        std::uint32_t env_map_present;
        float halton_jitter[2];
        float env_intensity;
        float env_total_luminance;
        float curr_view_proj[16];
        float prev_view_proj[16];
        float sun_and_mode[4];            // .xyz = sun_dir, .w = float(sky_mode)
        float exposure_pad[4];            // .x = exposure, .y = procedural-stars on, .z = bsc-stars on
        // World->J2000 rotation, row-major. Each row stored as float4 so
        // Slang's std430 layout (which pads vec3 to vec4 anyway) lines up
        // with the host. .w of each row is unused.
        float w2j_row0[4];
        float w2j_row1[4];
        float w2j_row2[4];
        // .x = aperture radius (0 = pinhole, no DOF). .y = focal
        // distance (world units). .z = aperture blade count (0 =
        // round disk, 3..16 = polygonal iris). .w reserved.
        float dof_params[4];
        // .x = density (0 disables). .y = HG anisotropy. .z =
        // intensity. .w = march sample count (cast to int in shader).
        float vol_params[4];
        // Volumetric cloud parameters. See PathTrace.slang's Push for
        // the field-by-field layout. Four float4s = 64 bytes (was 48
        // before issue #117 added clouds_p4 for curl-noise + secondary
        // edge erosion plumbing).
        float clouds_p1[4];   // (base_y, top_y, coverage, peak_density)
        float clouds_p2[4];   // (wind_x, wind_z, freq, time_seconds)
        float clouds_p3[4];   // (seed_offset_x, seed_offset_z, detail_amount, rayleigh_int)
        // clouds_p4 (#117): (curl_amount_m, curl_scale_cycles_per_m,
        // erosion_amount, _reserved). curl_amount is the Bridson-2007
        // curl-noise displacement magnitude in metres applied to the
        // sample position before density evaluation; 0 disables the
        // curl branch entirely so r_clouds_curl_amount 0 is bit-exact
        // versus pre-#117 main. curl_scale is the curl-noise frequency
        // in cycles per metre (default 0.01 = 100m turbulent eddies,
        // suitable for cumulus edge wisps). erosion_amount is a
        // secondary high-frequency erosion term layered on top of
        // clouds_p3.z so cp3.z controls bulk wispy edges and cp4.z
        // controls the sub-feature filamentous fringe.
        float clouds_p4[4];
        // Moon. .xyz = unit vector toward moon (computed from astro),
        // .w = phase angle radians (0 = new, π = full).
        float moon_dir_phase[4];
        // Moon extras. .x = r_moon_size multiplier, .y = astronomical
        // distance ratio (mean / current_km, perigee ~1.06, apogee
        // ~0.95), .zw reserved. Final disc size = .x * .y * base.
        float moon_extra[4];
        // Sun extras. .x = r_sun_size multiplier, .y = Earth-Sun
        // distance ratio (mean / current_AU), .z = vertical-flatten
        // ratio for atmospheric refraction at horizon (1.0 = circle,
        // <1.0 = squished oval; computed via Saemundsson differential
        // across the real solar disc). .w reserved.
        float sun_extra[4];
        // HDRI multi-light extraction. Each entry: .xyz = unit dir to
        // a bright cluster centroid, .w = pmf (sums to 1 across kept
        // entries) used by the shader's stochastic light selector.
        // Color: .rgb = ∫_cluster L dΩ (raw env_map units, scaled by
        // env_intensity in shader). hdri_lights_count is the number
        // of valid entries (0..kMaxHdriLights). Only meaningful when
        // env_map_present == 1.
        float hdri_lights_dir[Engine::kMaxHdriLights][4];
        float hdri_lights_col[Engine::kMaxHdriLights][4];
        std::uint32_t hdri_lights_count;
        // 1 -> kernel writes normal_tex at primary hit, 0 -> skip. We
        // already gate the entire G-buffer block on denoiser_enabled,
        // but normal is only consumed by the SVGF chain (svgf_basic /
        // svgf_atrous / nrd) and the OptiX AOV variants. For metalfx /
        // optix_hdr / optix_temporal_hdr the engine doesn't allocate
        // normal_tex_id_, so writing normal_tex[tid] would either be
        // a robust-access no-op (Vulkan) or an unbound-texture write
        // (Metal -- implementation defined). This flag lets the
        // shader skip the write entirely, saving one RGBA16F write
        // per pixel per frame on non-normal-consuming denoiser modes
        // AND closing the unbound-write hole on Metal.
        std::uint32_t write_normal_gbuffer;
        // 1 -> kernel writes albedo_tex at primary hit, 0 -> skip.
        // Albedo is only consumed by the OptiX AOV variants
        // (optix_hdr_aov / optix_temporal_hdr_aov); the engine
        // doesn't allocate albedo_tex_id_ for any other denoiser.
        // Required gate after the descriptor strategy moved to
        // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT -- the unconditional
        // albedo_tex write used to rely on Vulkan robust-buffer-access
        // returning zero for the null descriptor on non-AOV modes;
        // under partially-bound the slot stays unwritten in the
        // descriptor set and the write would be UB. Mirrors the
        // write_normal_gbuffer gating exactly.
        std::uint32_t write_albedo_gbuffer;
        // 1 -> kernel writes per-pixel linear-HDR radiance to
        // denoise_color. Set when EITHER a denoiser is active OR bloom
        // is requested on a backend that runs the engine-side Tonemap
        // chain (Metal, currently). The shader's depth/motion/normal/
        // albedo G-buffer block stays gated on denoiser_enabled (those
        // textures are denoiser-only), so this flag is the narrower
        // "produce an HDR aux for post-process" signal. Lives in the
        // 4-byte slot that used to be _hdri_pad[0]; layout-equivalent.
        std::uint32_t write_hdr_aux;
        // r_mis: 1 -> apply balance-heuristic MIS weights to env-map
        // NEE and BRDF-bounce miss contributions on Lambert hits.
        // 0 -> classic NEE-only path (skip_sky_on_miss after NEE).
        // Reuses the slot that used to be `_hdri_pad[1]`; same total
        // struct size, same static_assert sum below.
        std::uint32_t mis_enabled;
        // Explicit 12-byte pad to align `accum_params` (float4) at the
        // next 16-byte boundary, matching the std140 / MSL cbuffer
        // rule the Slang shader compiler applies to `Push` /
        // `Frame` in PathTrace.slang. Originally we had 4 uints here
        // (hdri_lights_count + write_normal_gbuffer + write_hdr_aux +
        // mis_enabled) totalling 16 bytes -- already a multiple of 16,
        // so accum_params landed at the right offset without any
        // explicit pad. Adding write_albedo_gbuffer pushed the run
        // to 5 uints (20 bytes); without this pad the host writes
        // accum_params at +20 while the shader reads it at +32, which
        // shifts every subsequent field by 12 bytes -- linear_prim_count
        // reads as 0, bvh_node_count reads as garbage past struct end,
        // and primary rays sail through the analytic-plane geometry
        // because the linear-pass loop runs zero iterations. Symptom
        // was "Metal walls/floor wash to procedural-sky pink with the
        // SVGF chain active" -- TLAS-mesh rays kept working (separate
        // code path) but analytic-plane hits all missed. The offsetof
        // static_asserts at the bottom of this struct guard against
        // the same class of bug recurring if anyone slips a new uint
        // in here.
        std::uint32_t _pad_before_accum_params[3];
        // Accumulator parameters.
        //   .x = r_accum_ema_alpha — exponential-moving-average history
        //        retention factor in [0, 1). 0 = legacy online-mean
        //        accumulator (the shader path that's been here forever:
        //        avg = (prev*n + sample) / (n+1), unbounded convergence).
        //        >0 = EMA blend: avg = prev*α + sample*(1-α), effective
        //        window ~1/(1-α). Lets the accumulator track dynamic
        //        content (wind-drifting clouds, animated lights, slow
        //        time-of-day) without the wet-paint smear that the
        //        running mean produces, at the cost of a variance floor
        //        proportional to 1/window. When >0 the engine also
        //        skips the cloud-wind accum_dirty_ trigger so EMA can
        //        actually do its job. 0.9 is a sensible starting point.
        //   .y/.z/.w reserved for future accum-side knobs (e.g. variance
        //   clamp, max-sample cap, motion-vector reproject weight).
        float accum_params[4];
        // Analytic-primitive traversal params (hybrid linear / BVH).
        //   .x = linear_prim_count -- # of always-linear-tested prims
        //        at the front of `primitives` (infinite-extent prims:
        //        planes; future: explicit-skip toggles).
        //   .y = bvh_node_count    -- 0 disables the BVH path; >0
        //        enables iterative BVH traversal over the remaining
        //        prims at [linear_prim_count, prim_count). Root node
        //        always at bvh_nodes[0]. Built by AnalyticBvh on
        //        EnsurePrimitivesUploaded when the finite-prim count
        //        meets r_analytic_bvh_threshold.
        //   .z/.w reserved for future BVH-side knobs (leaf size,
        //   builder selection).
        std::uint32_t bvh_params[4];
        // --- SDF Phase 1 (#97) + Phase 2 (#98) + Phase 3 (#99) -----------------
        // SDF traversal params (Phase 1/2/3 fields, packed).
        //   .x = sdf_cluster_count -- number of SDF clusters in the
        //        sdf_clusters buffer. 0 disables the entire SDF path.
        //   .y = sdf_max_iters     -- r_sdf_max_iters cvar, clamped to
        //        1..256 on the host (the shader's hard upper bound).
        //        Controls sphere-trace STEPS (analytic SDF + fractal),
        //        NOT the per-DE-eval iteration count.
        //   .z = sdf_debug_iters   -- r_sdf_debug_iters cvar. 1 ->
        //        future iteration-count heat map; Phase 1 just plumbs
        //        the bit through so Phase 2 turns it on without a
        //        push-constant reshuffle.
        //   .w = sdf_fractal_iters -- r_sdf_fractal_iters cvar (Phase 3).
        //        Per-DE bounded iteration count for the
        //        Mandelbulb / Mandelbox / Apollonian iterators.
        //        Independent of .y. Clamped 1..32 on the host.
        //        (Phase 2's r_sdf_normal_mode lives at sdf_params_f.w
        //        because Phase 3 claimed this slot first.)
        std::uint32_t sdf_params[4];
        // .x = r_sdf_epsilon (sphere-trace surface terminator, metres).
        // .y = r_sdf_fractal_power (Phase 3) -- default Mandelbulb
        //      polar exponent; per-leaf params[0] overrides.
        // .z = r_sdf_de_eps_scale (Phase 3) -- iteration-relaxed
        //      surface-epsilon scale, `eps * (1 + scale*step_idx)`.
        // .w reserved.
        float         sdf_params_f[4];
        // --- end SDF Phase 1 + Phase 3 -----------------------------------------
        // Celestials composite gate (issue #46). 1 = peel
        // starsOnly + sunDisc + moonDisc OUT of the primary-miss sky
        // term so denoise_color stays celestial-free, then the engine's
        // StarsComposite kernel re-adds them post-denoise with N
        // aperture-sampled rays per pixel (clean bokeh, no SVGF
        // a-trous smudge on sub-pixel stars). 0 = legacy fold-into-
        // sky behaviour. See PathTrace.slang's matching gate and
        // shaders/StarsComposite.slang for the composite kernel.
        // Followed by 12 bytes of pad so the trailing block lands on
        // a 16-byte boundary, matching the std140 / MSL cbuffer rule
        // Slang applies to the shader-side `Push` / `Frame` blocks.
        //
        // Phase 4 (#100) heterogeneous volumetric raymarching repurposes
        // the original 12-byte trailing pad as three floats:
        //   vol_density_scale     -- r_vol_density_scale cvar; linear
        //                            multiplier on the cloud sigma_t
        //                            inside the cloud march body.
        //   vol_phase_g_cloud     -- r_vol_phase_g cvar; HG anisotropy
        //                            for the cloud march, decoupled
        //                            from vol_params.y (atmospheric
        //                            haze g).
        //   vol_multiscatter_oct  -- r_vol_multiscatter_bounces cvar
        //                            (cast to float; shader clamps to
        //                            0..4 ints). Number of Wrenninge-
        //                            style multi-scatter octaves added
        //                            on top of single-scatter.
        std::uint32_t composite_celestials;
        std::uint32_t _pad_star_split;
        // SIGMA shadow demodulation gate (issue #115).
        std::uint32_t write_shadow_vis;
        std::uint32_t _pad_shadow_vis;
        // Heterogeneous volumetric cloud march (#100).
        float vol_density_scale;
        float vol_phase_g_cloud;
        float vol_multiscatter_oct;
        float _pad_vol;
        // --- Light primitives (#73) + Light tree (#129) --------------------
        // light_count: number of analytic light primitives in the
        // `light_prims` storage buffer this dispatch. 0 disables the
        // Lambert NEE-to-lights branch entirely.
        // light_tree_node_count: number of nodes in the hierarchical
        // light tree SSBO at binding 28. 0 -> shader falls back to
        // #73's naive uniform pick. Set to 0 when the r_light_tree
        // cvar is off too, so the cvar acts as a runtime kill switch
        // without rebuilding shaders or reuploading the tree buffer.
        // light_tree_enabled: explicit boolean for the shader's
        // traversal branch; kept separate from node_count so a
        // future variant can pass a non-zero node_count with the
        // traversal still gated (e.g. for diagnostics).
        // Trailing 1 uint pad keeps the block 16-byte aligned for
        // the std140 / MSL cbuffer rule. Mirrors PathTrace.slang's
        // matching trailing block in both the Metal Push cbuffer and
        // the SPIR-V Frame UBO.
        std::uint32_t light_count;
        std::uint32_t light_tree_node_count;
        std::uint32_t light_tree_enabled;
        std::uint32_t _pad_light_tail;
        // --- end Light primitives + Light tree -----------------------------
        // MetalFX specular-guidance G-buffer write gates (issue #118).
        std::uint32_t write_specular_albedo_gbuffer;
        std::uint32_t write_roughness_gbuffer;
        std::uint32_t write_specular_hit_distance_gbuffer;
        std::uint32_t _pad_specular_gbuffers0;
        // --- Water Phase 1 (#134) ----------------------------------------
        // water_params0.xyz = absorption per channel (1/m, Beer's law);
        //                .w = ior (clamped to [1.0, 2.4] in shader).
        // water_params1.x  = wave_scale (cycles / m, normal-map noise freq).
        //                .y = wave_amplitude (normal perturbation strength).
        //                .z = wave_speed (animation rate multiplier).
        //                .w = time_seconds (shared accumulator with cloud
        //                                   wind; wraps on 24 h sim window).
        // Two vec4s = 32 B.
        float water_params0[4];
        float water_params1[4];
        // --- end Water Phase 1 -------------------------------------------
        // --- ReSTIR DI Phase A (issue #78) ---------------------------------
        std::uint32_t restir_enabled;
        std::uint32_t restir_k_candidates;
        std::uint32_t _pad_restir0;
        std::uint32_t _pad_restir1;
        // --- end ReSTIR DI Phase A -----------------------------------------
        // --- Fluid Phase 1 (#136) ----------------------------------------
        // smoke_params.x = smoke_count
        //             .y = smoke_enabled (0/1)
        //             .z = time_seconds (parametric drift accumulator)
        //             .w = r_cloud_attenuate_background flag (added in
        //                  57fd9ff; piggybacks the reserved slot)
        // One vec4 = 16 B.
        float smoke_params[4];
        // --- end Fluid Phase 1 -------------------------------------------
        // --- Fluid Phase 2 (#180) ----------------------------------------
        // smoke_params2.x = r_smoke_lifetime          (seconds; >=0)
        //              .y = r_smoke_advection_strength (multiplier on vel)
        //              .z = r_smoke_puff_count        (1..16, clamped)
        //              .w = r_smoke_expansion_rate    (per-second radius growth)
        // One vec4 = 16 B. Drives the inner per-emitter puff-chain loop.
        float smoke_params2[4];
        // --- end Fluid Phase 2 -------------------------------------------
        // --- Fluid wave-6 "looks like smoke" -----------------------------
        // smoke_params3.x = r_smoke_turbulence_scale     (noise freq, cycles/m)
        //              .y = r_smoke_turbulence_amplitude (0..1 modulation)
        //              .z = r_smoke_decay_shape          (>=1 exponent on age fade)
        //              .w = r_smoke_age_tint_strength    (0..1 age-driven tint)
        // One vec4 = 16 B. All four read by smoke_density_add* in PathTrace.slang.
        float smoke_params3[4];
        // --- end Fluid wave-6 --------------------------------------------
        // --- Motion blur (#85) -------------------------------------------
        // motion_blur_params.x = r_motion_blur (1 = enable shutter-time
        //                       jitter + per-prim prev_pos lerp; 0 =
        //                       legacy single-frame sampling, bit-exact
        //                       vs pre-#85 main even though prev_pos is
        //                       still packed into the primitive buffer).
        //                .y = r_shutter_speed in seconds. Currently
        //                       unused on the shader side (the lerp
        //                       factor is just U(0,1) over the shutter
        //                       window; the host snapshots prev/curr
        //                       once per StepPhysics tick so a longer
        //                       shutter visually scales the streak
        //                       through how often the user calls render
        //                       between physics steps -- a hook for
        //                       future per-ray shutter weighting if we
        //                       ever decouple physics dt from frame dt).
        //                .z/.w reserved.
        // One vec4 = 16 B.
        float motion_blur_params[4];
        // --- end Motion blur ---------------------------------------------
        // --- Underwater medium tracking (Phase 4, #134) ------------------
        // water_state.x = camera_initial_in_water flag (0.0 / 1.0). The
        //                 engine scans the analytic-primitives map for
        //                 MAT_WATER planes once per dispatch (cheap; the
        //                 scene typically has 0 or 1 water plane) and
        //                 checks whether cam.pos.y is below any of them
        //                 along the plane's normal-projected half-space.
        //                 When 1.0, PathTrace.slang's bounce loop starts
        //                 in_water = 1 so per-segment Beer's-law
        //                 attenuation fires on the camera primary ray
        //                 without needing the legacy b == 0 special case.
        //             .y/.z/.w = r_water_deep_color_r/g/b -- the RGB tint
        //                 the shader returns when an in-water ray misses
        //                 scene geometry and would otherwise sample
        //                 skyColor(rd). Defaults to (0, 0, 0) so deep-
        //                 water voids read as honest black; raise per
        //                 cvar for tunable scattering ambient.
        // One vec4 = 16 B.
        float water_state[4];
        // --- end Underwater medium tracking ------------------------------
        // --- Editor backend (agent-19 / PR #201, host wired in #210) -----
        // editor_params.x = selected_prim_id (0 = no selection / paint
        //                   nothing). Host writes the EFFECTIVE id here
        //                   (zero when r_editor_outline == 0 OR the
        //                   selected_kind_ != SelectionKind::AnalyticPrim)
        //                   so PathTrace.slang can collapse the outline
        //                   test to a single "if (.x != 0 && prim_id == .x)"
        //                   without re-reading .y / .z on every prim hit.
        //              .y = r_editor_outline gate (0/1; 0 forces the
        //                   outline off regardless of .x).
        //              .z = selected_kind (mirrors host SelectionKind
        //                   enum; only AnalyticPrim paints today).
        //              .w reserved.
        // One uvec4 = 16 B.
        std::uint32_t editor_params[4];
        // --- end Editor backend ------------------------------------------
    } push{};
    push.pos_fovtan[0] = cam.pos.x; push.pos_fovtan[1] = cam.pos.y;
    push.pos_fovtan[2] = cam.pos.z; push.pos_fovtan[3] = cam.FovYTan();
    push.fwd_aspect[0] = fwd.x; push.fwd_aspect[1] = fwd.y;
    push.fwd_aspect[2] = fwd.z; push.fwd_aspect[3] = aspect;
    push.right_xyz[0]  = right.x; push.right_xyz[1] = right.y;
    push.right_xyz[2]  = right.z; push.right_xyz[3] = 0.0f;
    push.up_xyz[0]     = up.x; push.up_xyz[1] = up.y;
    push.up_xyz[2]     = up.z; push.up_xyz[3] = 0.0f;
    push.frame_index   = frame_index_++;
    // When the denoiser is on it does its own temporal reuse, so we feed
    // it FRESH (un-accumulated) per-frame radiance every dispatch. The
    // shader still writes accum_hdr (so toggling denoiser off picks back
    // up the accumulating image) but reads it as if reset.
    push.reset_accum   = (denoiser_active_ || accum_dirty_) ? 1u : 0u;
    push.max_bounces   = bounces;
    push.tlas_present  = tlas_present ? 1u : 0u;
    push.prim_count    = static_cast<std::uint32_t>(primitives_.size());
    push.spp           = spp;
    push.denoiser_enabled = denoiser_active_ ? 1u : 0u;
    // Tell the kernel to also produce its linear-HDR aux write to
    // denoise_color whenever we need a bloom source -- i.e. either
    // a real denoiser is active (its input) or the bloom-only Metal
    // path is on (Tonemap.slang's sampling source). Separate from
    // denoiser_enabled because the depth/motion G-buffer block is
    // denoiser-only; this flag is the narrower "write HDR aux" signal.
    push.write_hdr_aux = need_hdr_aux ? 1u : 0u;
    // SVGF / NRD / OptiX-AOV / MetalFX all consume normal_tex; the plain
    // OptiX HDR variants do not, and the engine doesn't allocate
    // normal_tex_id_ for those -- so the kernel must not write to an
    // unbound slot. denoiser_active_ alone isn't enough; tie the gate
    // to whether the engine actually owns a normal G-buffer this frame.
    push.write_normal_gbuffer =
        (denoiser_active_ && normal_tex_id_ != 0) ? 1u : 0u;
    // Same gating logic as write_normal_gbuffer. OptiX-AOV, MetalFX
    // (and its SVGF-chained variants), AND the plain SVGF/NRD kinds
    // (issue #119, albedo demodulation) take a diffuse albedo guide
    // layer; everything else dispatches with binding 17 unbound under
    // partially-bound semantics. The runtime gate keys on
    // albedo_tex_id_ != 0, so it lights up automatically for any
    // denoiser_kind_ that flips want_albedo_gbuffer above.
    push.write_albedo_gbuffer =
        (denoiser_active_ && albedo_tex_id_ != 0) ? 1u : 0u;
    // MetalFX specular-guidance G-buffer write gates (issue #118). Same
    // gating logic as the normal/albedo gates: only ever set when the
    // engine actually owns the matching texture for this dispatch.
    // The host's want_specular_guidance_gbuffers flag drives allocation
    // (set only for DenoiserKind::MetalFX / SvgfBasicMetalFx /
    // SvgfAtrousMetalFx); the runtime gate here is the descriptor-
    // is-actually-bound signal. Under partially-bound semantics the
    // shader-side write MUST elide when the slot is unbound; the
    // per-texture gate is what enables that elision.
    push.write_specular_albedo_gbuffer =
        (denoiser_active_ && specular_albedo_tex_id_ != 0) ? 1u : 0u;
    push.write_roughness_gbuffer =
        (denoiser_active_ && roughness_tex_id_ != 0) ? 1u : 0u;
    push.write_specular_hit_distance_gbuffer =
        (denoiser_active_ && specular_hit_distance_tex_id_ != 0) ? 1u : 0u;
    push._pad_specular_gbuffers0 = 0u;
    push.env_map_present  = (env_map_tex_id_ != 0) ? 1u : 0u;
    {
        float intensity = 1.0f;
        if (auto* v = C.FindCVar("r_env_intensity")) intensity = v->GetFloat();
        push.env_intensity = intensity;
    }
    push.env_total_luminance = env_total_luminance_;
    {
        int mis = 1;
        if (auto* v = C.FindCVar("r_mis")) mis = v->GetInt();
        push.mis_enabled = (mis != 0) ? 1u : 0u;
    }
    {
        float ema = 0.0f;
        if (auto* v = C.FindCVar("r_accum_ema_alpha")) ema = v->GetFloat();
        // Clamp to [0, 0.999]: 1.0 would be "never accept new samples"
        // which freezes the image at whatever's already in accum_hdr.
        if (ema < 0.0f) ema = 0.0f;
        if (ema > 0.999f) ema = 0.999f;
        push.accum_params[0] = ema;
        push.accum_params[1] = 0.0f;
        push.accum_params[2] = 0.0f;
        push.accum_params[3] = 0.0f;
    }
    {
        push.bvh_params[0] = linear_prim_count_;
        push.bvh_params[1] = analytic_bvh_.Empty()
            ? 0u
            : static_cast<std::uint32_t>(analytic_bvh_.NodeCount());
        // .z carries `mesh_tri_count` for the software-mesh linear-scan
        // fallback. Set only when the engine has uploaded vbuf/ibuf for
        // a CSG mesh AND no TLAS was bound this frame (i.e. the backend
        // lacks hardware ray tracing). The shader gates the linear-scan
        // branch on this being non-zero. Zero on every other path -- the
        // shader's existing `tlas_present` test continues to drive the
        // RayQuery branch on RT-capable backends.
        push.bvh_params[2] = sw_mesh_present
            ? mesh_tri_count_
            : 0u;
        // .w carries the triangle-BVH node count. The shader's SW
        // mesh branch walks the tri-BVH (engine slots 8/9) when this
        // is non-zero AND .z is non-zero (mesh present, no TLAS). PR
        // #106 follow-up: replaces the previous O(N) Möller-Trumbore
        // linear scan with O(log N) stack-based BVH traversal -- fixes
        // the ~1 FPS @1080p perf cliff on Mac-Vulkan (MoltenVK without
        // VK_KHR_ray_query). If the build failed (allocation failure
        // -- rare but possible on memory-tight devices), .w stays at
        // 0 and the SW mesh branch short-circuits to "no hit", which
        // is a benign failure mode (mesh stays invisible) compared to
        // a per-ray full scan.
        push.bvh_params[3] = sw_mesh_present
            ? tri_bvh_node_count_
            : 0u;
    }
    // --- SDF Phase 1 (#97) + Phase 2 (#98) ---------------------------------
    {
        int    max_iters   = 128;
        float  epsilon     = 1e-4f;
        int    debug       = 0;
        int    normal_mode = 0;     // Phase 2: forward-AD (0) or central (1)
        if (auto* v = C.FindCVar("r_sdf_max_iters"))   max_iters   = v->GetInt();
        if (auto* v = C.FindCVar("r_sdf_epsilon"))     epsilon     = v->GetFloat();
        if (auto* v = C.FindCVar("r_sdf_debug_iters")) debug       = v->GetInt();
        if (auto* v = C.FindCVar("r_sdf_normal_mode")) normal_mode = v->GetInt();
        if (max_iters < 1)   max_iters = 1;
        if (max_iters > 256) max_iters = 256;     // shader clamps the same way
        if (epsilon   < 1e-7f) epsilon = 1e-7f;
        // --- SDF Phase 3 (#99) fractal cvars -------------------------------
        // Pack the three fractal knobs into the reserved push slots
        // (sdf_params.w = iters, sdf_params_f.y = power default,
        // sdf_params_f.z = de-eps scale). Phase 1's reserved lanes
        // are now consumed; further additions need a new push block.
        float  fractal_power     = 8.0f;
        int    fractal_iters     = 12;
        float  fractal_de_eps_sc = 1.0f;
        if (auto* v = C.FindCVar("r_sdf_fractal_power"))   fractal_power     = v->GetFloat();
        if (auto* v = C.FindCVar("r_sdf_fractal_iters"))   fractal_iters     = v->GetInt();
        if (auto* v = C.FindCVar("r_sdf_de_eps_scale"))    fractal_de_eps_sc = v->GetFloat();
        if (fractal_power < 2.0f)  fractal_power = 2.0f;     // power < 2 is degenerate
        if (fractal_power > 64.0f) fractal_power = 64.0f;    // soft upper cap; shader handles arbitrary
        if (fractal_iters < 1)     fractal_iters = 1;
        if (fractal_iters > 32)    fractal_iters = 32;       // shader clamps the same way
        if (fractal_de_eps_sc < 0.0f) fractal_de_eps_sc = 0.0f;
        // --- end SDF Phase 3 -----------------------------------------------
        if (normal_mode < 0) normal_mode = 0;
        // r_sdf_normal_mode is bounded to {0, 1} (allowed_values guard +
        // this clamp). 0 = forward-AD, 1 = central differences. Values
        // out of band are clamped UP to 1 (central) rather than back to
        // 0 because the host treats central-diff as the conservative
        // fallback: anything but a known good "forward-AD please" value
        // gets the predictable 6-tap path.
        if (normal_mode > 1) normal_mode = 1;
        // Disable the entire SDF path when no clusters are active --
        // the shader's gate is sdf_params.x > 0.
        push.sdf_params[0] = sdf_cluster_count_;
        push.sdf_params[1] = static_cast<std::uint32_t>(max_iters);
        push.sdf_params[2] = (debug != 0) ? 1u : 0u;
        push.sdf_params[3] = static_cast<std::uint32_t>(fractal_iters);
        push.sdf_params_f[0] = epsilon;
        push.sdf_params_f[1] = fractal_power;
        push.sdf_params_f[2] = fractal_de_eps_sc;
        // Phase 2 (#98) r_sdf_normal_mode encoded as float here because Phase 3
        // (#99) claimed sdf_params.w first. 0.0 = forward AD, 1.0 = central diff.
        push.sdf_params_f[3] = static_cast<float>(normal_mode);
    }
    // --- end SDF Phase 1 / 2 -----------------------------------------------

    // Celestials composite gate (issue #46). When set, PathTrace.slang
    // subtracts starsOnly + sunDisc + moonDisc + the procSky inline
    // sun (disk + halo) from the primary-miss sky term so denoise_color
    // carries only the procedural sky base. The engine's StarsComposite
    // kernel (dispatched after Denoise(), before bloom_pyramid) re-adds
    // the celestials with N aperture-sampled rays per pixel.
    //
    // CRITICAL: this gate must NOT fire when the composite dispatch
    // won't actually run -- otherwise PathTrace peels celestials out
    // of denoise_color and nothing adds them back. Conditions:
    //   - denoiser_active_      : composite only exists post-denoise
    //   - r_star_split on       : user opted in
    //   - sky mode procedural   : only mode the composite handles
    //                             (hdri bakes celestials into env_map,
    //                             gradient sky has none)
    //   - stars_composite_pipeline_id_ != 0 : the composite kernel
    //                             must be registered on the active
    //                             backend. Both Metal and Vulkan build
    //                             + register stars_composite. The
    //                             dispatch lives in two places:
    //                             (a) Metal: inside the
    //                                 `use_engine_tonemap` branch
    //                                 between SVGF output and Tonemap.
    //                             (b) Vulkan: inside the dual-Denoise
    //                                 sequence (SvgfNoFinalize ->
    //                                 StarsComposite -> FinalizeOnly)
    //                                 since Tonemap.slang doesn't reach
    //                                 the swap on Vulkan -- see issue
    //                                 #46 + the dual-Denoise comment
    //                                 block below.
    //   - cloud_trans_tex_id_ != 0 : composite reads this texture and
    //                             would crash without it.
    //   - depth_tex_id_ != 0   : composite reads this texture for the
    //                             sky-pixel gate.
    // sky_mode_int is shared by both the celestials-composite gate
    // (below) and the SIGMA shadow-demod gate further down -- both
    // semantically require procedural-sky mode, so resolve r_sky_mode
    // once and reuse to keep the two gates in sync.
    int sky_mode_int = 2;
    if (auto* v = C.FindCVar("r_sky_mode")) {
        const std::string& m = v->value;
        sky_mode_int = (m == "procedural") ? 2
                     : (m == "hdri")       ? 1
                                           : 0;
    }
    bool engine_composite_active = false;
    {
        bool star_split_on = true;
        if (auto* v = C.FindCVar("r_star_split")) star_split_on = v->GetBool();
        engine_composite_active =
            denoiser_active_ &&
            star_split_on &&
            (sky_mode_int == 2) &&
            stars_composite_pipeline_id_ != 0 &&
            cloud_trans_tex_id_ != 0 &&
            depth_tex_id_ != 0;
        push.composite_celestials = engine_composite_active ? 1u : 0u;
    }
    push._pad_star_split = 0u;
    push._pad_shadow_vis = 0u;
    push._pad_vol        = 0.0f;
    // SIGMA shadow demodulation gate (issue #115). Mirrors the
    // celestials gate above. PathTrace writes the per-primary-hit
    // sun-NEE shadow-ray transmittance only when:
    //   - denoiser_active_                : the smudge only matters
    //                                        under SVGF / MetalFX; the
    //                                        legacy 1-spp path already
    //                                        produces sharp shadows.
    //   - r_shadow_demod on               : user opted in.
    //   - sky_mode_int == 2 (procedural)  : the visibility traced is
    //                                        toward sun_and_mode.xyz;
    //                                        only the procedural-sun
    //                                        lighting path uses that
    //                                        direction. HDRI mode samples
    //                                        extracted HDRI clusters and
    //                                        never casts toward
    //                                        sun_and_mode -- demodulating
    //                                        HDRI scenes by procedural
    //                                        sun visibility incorrectly
    //                                        darkens them. Gradient mode
    //                                        has no NEE at all. HDRI
    //                                        demod is out of scope for
    //                                        this PR -- see issue #115.
    //   - sigma_shadow_pipeline_id_ != 0  : Metal-only today.
    //   - shadow_vis_buf_id_ != 0         : the buffer is actually
    //                                        allocated.
    //   - depth_tex_id_ != 0 + normal_tex_id_ checked at dispatch site.
    {
        bool shadow_demod_on = true;
        if (auto* v = C.FindCVar("r_shadow_demod")) {
            shadow_demod_on = v->GetBool();
        }
        const bool engine_shadow_demod_active =
            denoiser_active_ &&
            shadow_demod_on &&
            (sky_mode_int == 2) &&
            sigma_shadow_pipeline_id_ != 0 &&
            shadow_vis_buf_id_ != 0 &&
            depth_tex_id_ != 0;
        push.write_shadow_vis = engine_shadow_demod_active ? 1u : 0u;
    }
    // Heterogeneous volumetric cloud march (#100).
    {
        float dscale   = 1.0f;
        float phase_g  = 0.8f;
        int   ms_bnc   = 2;
        if (auto* v = C.FindCVar("r_vol_density_scale"))         dscale  = v->GetFloat();
        if (auto* v = C.FindCVar("r_vol_phase_g"))               phase_g = v->GetFloat();
        if (auto* v = C.FindCVar("r_vol_multiscatter_bounces"))  ms_bnc  = v->GetInt();
        push.vol_density_scale    = dscale;
        push.vol_phase_g_cloud    = phase_g;
        push.vol_multiscatter_oct = float(ms_bnc);
    }
    // Analytic light count (#73). Drives PathTrace.slang's NEE-to-lights
    // gate; 0 disables it (the binding still resolves to a placeholder
    // buffer for slot stability -- see slot12 above).
    push.light_count = light_count_uploaded_;
    // Light tree (#129). r_light_tree cvar acts as a runtime kill
    // switch (default 1; set to 0 for the naive uniform-pick fallback,
    // useful for variance-regression comparison and for fixtures that
    // exercise the original #73 sampler).
    //
    // PT_LIGHT_TREE_ENABLED (CMake option PT_LIGHT_TREE) acts as the
    // *compile-time* gate. When the megakernel was built with the gate
    // OFF (perf/megakernel-split-light-tree), the shader excises the
    // traversal helpers and the picker if-branch entirely, so we MUST
    // force light_tree_enabled=0 in PtPush regardless of cvar state --
    // otherwise the shader would skip the uniform-pick fallback branch
    // (which is now its only path) on a stale uniform read. The runtime
    // cvar is silently a no-op in that case; a one-shot startup [warn]
    // in the cvar-watch block flags it for the user.
#if PT_LIGHT_TREE_ENABLED
    bool light_tree_on = true;
    if (auto* v = C.FindCVar("r_light_tree")) light_tree_on = (v->GetInt() != 0);
    push.light_tree_node_count = light_tree_on
        ? light_tree_node_count_uploaded_
        : 0u;
    push.light_tree_enabled = (light_tree_on &&
                                light_tree_node_count_uploaded_ > 0u) ? 1u : 0u;
#else
    // PT_LIGHT_TREE=OFF compile-time gate. Force both fields to 0 so
    // the shader's uniform-pick fallback is always taken; the cvar is
    // ignored.
    push.light_tree_node_count = 0u;
    push.light_tree_enabled    = 0u;
#endif
    push._pad_light_tail = 0u;
    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    // r_restir master toggle, gated additionally on:
    //   - denoiser_active_       : Phase A composite writes into denoise_color
    //   - restir_*_pipeline_id_  : Metal-only at MVP scope; Vulkan gate
    //                              collapses to 0 cleanly
    //   - light_count > 0        : nothing to resample when the light
    //                              list is empty
    bool restir_user_on = false;
    if (auto* v = C.FindCVar("r_restir")) restir_user_on = (v->GetInt() != 0);
    bool restir_dispatch_active =
        restir_user_on &&
        denoiser_active_ &&
        restir_temporal_pipeline_id_ != 0 &&
        restir_spatial_pipeline_id_  != 0 &&
        restir_final_pipeline_id_    != 0 &&
        restir_reservoir_curr_buf_id_ != 0 &&
        light_count_uploaded_ > 0u;
    // Issue #164 -- ReSTIR + MetalFX integration. MTLFXTemporalDenoisedScaler
    // applies a TAA neighborhood color clamp that rejects bright per-pixel
    // WRS spikes as outliers, nuking the ReSTIR-supplied analytic-light
    // contribution to near-invisibility. The cleanest fix at MVP scope is
    // to force-disable ReSTIR's primary-hit replacement when MetalFX is
    // the active denoiser, so PathTrace's legacy per-spp single-light NEE
    // runs inline and MetalFX sees dense per-pixel light contribution
    // (which it handles correctly -- see the no-ReSTIR baseline showing
    // brilliant floor lighting on the same scene). SVGF / NRD / OptiX
    // keep ReSTIR engaged since their spatial filter spreads the spikes
    // correctly.
    //
    // Gated AT the dispatch flag rather than the cvar so r_restir stays
    // user-controllable -- the engine just refuses to engage the ReSTIR
    // chain on MetalFX-family kinds. Must run BEFORE the PushConstants
    // emit below (line 6837) so PathTrace.slang sees push.restir_enabled
    // == 0 and falls into the legacy NEE branch.
    const bool kind_is_metalfx_family =
        (denoiser_kind_ == DenoiserKind::MetalFX             ||
         denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx    ||
         denoiser_kind_ == DenoiserKind::SvgfAtrousMetalFx);
    if (kind_is_metalfx_family && restir_dispatch_active) {
        static bool s_restir_force_off_logged = false;
        if (!s_restir_force_off_logged) {
            LOG_INFO("engine: ReSTIR force-disabled for MetalFX-family kind "
                     "(issue #164) -- MetalFX's TAA neighborhood clamp "
                     "rejects ReSTIR's 1-survivor spike pattern, leaving "
                     "analytic lights invisible. Falling back to PathTrace's "
                     "per-spp legacy NEE for many-light variance.");
            s_restir_force_off_logged = true;
        }
        restir_dispatch_active = false;
    }
    push.restir_enabled = restir_dispatch_active ? 1u : 0u;
    // One-shot gate diagnostics: ONLY when r_restir = 1 but the gate
    // evaluates to 0 (so the user-visible "nothing happens" mismatch
    // gets a diagnostic line per session). When ReSTIR engages
    // successfully the "Phase A engaged" log inside the dispatch block
    // below covers it; the silent-success path here avoids spamming
    // the journal for the common case.
    {
        static bool s_restir_gate_failure_logged = false;
        // Skip this generic gate-failure log when the MetalFX-family
        // gate above already explained the disengagement (issue #164):
        // the dedicated "ReSTIR force-disabled for MetalFX-family kind"
        // log is the user-actionable message, and emitting the generic
        // "denoiser_active=true, pipes=truetruetrue, ..." right after
        // it would be redundant noise.
        if (restir_user_on && !restir_dispatch_active &&
            !kind_is_metalfx_family &&
            !s_restir_gate_failure_logged) {
            LOG_INFO("engine: r_restir=1 but ReSTIR NOT dispatching "
                     "(denoiser_active={}, pipes_t/s/f={}{}{}, "
                     "reservoir_buf={}, light_count={}). The dispatch "
                     "engages when ALL of those are non-zero / true.",
                     denoiser_active_,
                     restir_temporal_pipeline_id_ != 0,
                     restir_spatial_pipeline_id_  != 0,
                     restir_final_pipeline_id_    != 0,
                     restir_reservoir_curr_buf_id_ != 0,
                     light_count_uploaded_);
            s_restir_gate_failure_logged = true;
        }
    }
    std::uint32_t restir_k = 8u;
    if (auto* v = C.FindCVar("r_restir_k_candidates")) {
        int n = v->GetInt();
        if (n < 1) n = 1;
        if (n > static_cast<int>(pt::renderer::kRestirMaxK)) {
            n = static_cast<int>(pt::renderer::kRestirMaxK);
        }
        restir_k = static_cast<std::uint32_t>(n);
    }
    push.restir_k_candidates = restir_k;
    push._pad_restir0 = 0u;
    push._pad_restir1 = 0u;
    // --- end ReSTIR DI Phase A ---------------------------------------------

    // Halton(2,3) sub-pixel jitter sequence in [-0.5, 0.5] each axis.
    // 16-sample period before repeating; ample for the denoiser's
    // internal history depth. Use frame_index_ which already advanced
    // above, so each frame's color ray is a unique sub-pixel sample.
    auto halton = [](std::uint32_t i, std::uint32_t base) -> float {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= float(base); r += f * float(i % base); i /= base; }
        return r;
    };
    std::uint32_t hi = (push.frame_index % 16u) + 1u;
    push.halton_jitter[0] = halton(hi, 2) - 0.5f;
    push.halton_jitter[1] = halton(hi, 3) - 0.5f;
    last_jitter_x_ = push.halton_jitter[0];
    last_jitter_y_ = push.halton_jitter[1];

    // pad3 already 0.0f from value-init
    const glm::mat4 prev_vp = prev_view_proj_valid_ ? prev_view_proj_ : curr_view_proj;
    std::memcpy(push.curr_view_proj, glm::value_ptr(curr_view_proj), sizeof(push.curr_view_proj));
    std::memcpy(push.prev_view_proj, glm::value_ptr(prev_vp),        sizeof(push.prev_view_proj));

    // Sun direction from elevation/azimuth (degrees). Convention:
    // +Y up, -Z forward (north), +X right (east). When
    // r_sky_use_astronomical is on, override with computed positions
    // from the observer lat/lon and current UTC.
    float sun_elev_deg = 30.0f, sun_azim_deg = 135.0f;
    bool astro_on = false;
    if (auto* v = C.FindCVar("r_sky_use_astronomical")) astro_on = v->GetBool();
    // Compute JD from r_sky_hour (UTC hour of day on today's date). The
    // cvar is the only time knob -- animation just drives this value
    // forward over wall-clock seconds.
    auto compute_jd = [&]() -> double {
        float hour = 12.0f;
        if (auto* v = C.FindCVar("r_sky_hour")) hour = v->GetFloat();
        bool local = true;
        if (auto* v = C.FindCVar("r_sky_hour_local")) local = v->GetBool();
        float tz = 0.0f;
        if (local) {
            if (auto* v = C.FindCVar("r_sky_tz_offset_hours")) tz = v->GetFloat();
        }
        // r_sky_hour interpreted in local civil time -> subtract the
        // city's UTC offset to get UTC hour.
        const float hour_utc = hour - tz;
        // Date: r_sky_{year,month,day} = 0 falls back to system today.
        // Any nonzero combination overrides individually.
        int yr = 0, mo = 0, dy = 0;
        if (auto* v = C.FindCVar("r_sky_year"))  yr = v->GetInt();
        if (auto* v = C.FindCVar("r_sky_month")) mo = v->GetInt();
        if (auto* v = C.FindCVar("r_sky_day"))   dy = v->GetInt();
        if (yr == 0 || mo == 0 || dy == 0) {
            const std::time_t now = std::time(nullptr);
            std::tm gm = *std::gmtime(&now);
            if (yr == 0) yr = gm.tm_year + 1900;
            if (mo == 0) mo = gm.tm_mon + 1;
            if (dy == 0) dy = gm.tm_mday;
        }
        // Construct JD at UTC midnight of the chosen date, then add
        // the requested hour-of-day. Hour may be 24+ or negative if
        // the local-time conversion crossed a UTC day boundary; the
        // hour/24 fraction handles the rollover automatically.
        const double jd_midnight = pt::astro::julianDateFromUtc(
            yr, mo, dy, 0, 0, 0.0);
        return jd_midnight + double(hour_utc) / 24.0;
    };
    if (astro_on) {
        double lat = 13.0827, lon = 80.2707;
        if (auto* v = C.FindCVar("r_sky_lat")) lat = v->GetFloat();
        if (auto* v = C.FindCVar("r_sky_lon")) lon = v->GetFloat();
        const double jd = compute_jd();
        auto sun_eq = pt::astro::sunPosition(jd);
        auto sun_h  = pt::astro::equatorialToHorizon(sun_eq, lat, lon, jd);
        sun_elev_deg = static_cast<float>(sun_h.altitude_deg);
        sun_azim_deg = static_cast<float>(sun_h.azimuth_deg);
        // Reflect computed values back to the manual cvars so the
        // settings panel shows the real numbers (read-only feel).
        C.SetCVarOverride("r_sun_elevation", std::to_string(sun_elev_deg));
        C.SetCVarOverride("r_sun_azimuth",   std::to_string(sun_azim_deg));
    } else {
        if (auto* v = C.FindCVar("r_sun_elevation")) sun_elev_deg = v->GetFloat();
        if (auto* v = C.FindCVar("r_sun_azimuth"))   sun_azim_deg = v->GetFloat();
    }
    const float elev_r = glm::radians(sun_elev_deg);
    const float azim_r = glm::radians(sun_azim_deg);
    const float ce = std::cos(elev_r), se = std::sin(elev_r);
    push.sun_and_mode[0] =  ce * std::sin(azim_r);
    push.sun_and_mode[1] =  se;
    push.sun_and_mode[2] = -ce * std::cos(azim_r);

    // Moon. Only meaningful in astronomical mode -- the phase angle
    // requires sun + moon at the same epoch / observer, which only
    // makes physical sense when both come from the date driver. When
    // r_sky_use_astronomical = 0 the moon is hidden (.y < 0 sentinel,
    // both moonDisc and moon NEE skip it).
    if (astro_on) {
        double lat = 13.0827, lon = 80.2707;
        if (auto* v = C.FindCVar("r_sky_lat")) lat = v->GetFloat();
        if (auto* v = C.FindCVar("r_sky_lon")) lon = v->GetFloat();
        const double jd_moon = compute_jd();
        auto moon_eq = pt::astro::moonPosition(jd_moon);
        auto moon_h  = pt::astro::equatorialToHorizon(moon_eq, lat, lon, jd_moon);
        const float me_r = glm::radians(static_cast<float>(moon_h.altitude_deg));
        const float ma_r = glm::radians(static_cast<float>(moon_h.azimuth_deg));
        const float mce = std::cos(me_r), mse = std::sin(me_r);
        push.moon_dir_phase[0] =  mce * std::sin(ma_r);
        push.moon_dir_phase[1] =  mse;
        push.moon_dir_phase[2] = -mce * std::cos(ma_r);
        auto sun_eq_for_phase = pt::astro::sunPosition(jd_moon);
        push.moon_dir_phase[3] = static_cast<float>(
            pt::astro::moonPhaseAngle(sun_eq_for_phase, moon_eq));
    } else {
        push.moon_dir_phase[0] = 0.0f;
        push.moon_dir_phase[1] = -1.0f;   // below-horizon sentinel
        push.moon_dir_phase[2] = 0.0f;
        push.moon_dir_phase[3] = 0.0f;
    }
    {
        float moon_size = 1.0f;
        if (auto* v = C.FindCVar("r_moon_size")) moon_size = v->GetFloat();
        if (moon_size < 0.1f) moon_size = 0.1f;
        if (moon_size > 50.0f) moon_size = 50.0f;
        push.moon_extra[0] = moon_size;
        // Distance ratio for apparent-size scaling. Default to 1.0
        // (mean distance) when astronomical mode is off; otherwise
        // compute from Meeus radial-distance series.
        float moon_dist_ratio = 1.0f;
        if (astro_on) {
            const double jd_d = compute_jd();
            double dkm = pt::astro::moonDistanceKm(jd_d);
            if (dkm > 1.0) {
                moon_dist_ratio = static_cast<float>(
                    pt::astro::kMoonDistanceMeanKm / dkm);
            }
        }
        push.moon_extra[1] = moon_dist_ratio;
        float moon_bright = 0.7f;
        if (auto* v = C.FindCVar("r_moon_brightness")) moon_bright = v->GetFloat();
        if (moon_bright < 0.0f) moon_bright = 0.0f;
        if (moon_bright > 5.0f) moon_bright = 5.0f;
        push.moon_extra[2] = moon_bright;
        push.moon_extra[3] = 0.0f;
    }
    {
        float sun_size = 1.0f;
        if (auto* v = C.FindCVar("r_sun_size")) sun_size = v->GetFloat();
        if (sun_size < 0.1f) sun_size = 0.1f;
        if (sun_size > 50.0f) sun_size = 50.0f;
        push.sun_extra[0] = sun_size;
        // Sun distance ratio. Always astronomical (cheap to compute).
        const double jd_s = compute_jd();
        double dau = pt::astro::sunDistanceAu(jd_s);
        float sun_dist_ratio = 1.0f;
        if (dau > 1e-3) {
            sun_dist_ratio = static_cast<float>(
                pt::astro::kSunDistanceMeanAu / dau);
        }
        push.sun_extra[1] = sun_dist_ratio;
        // sun_extra[2]: vertical-flatten ratio for atmospheric refraction
        // at horizon. 1.0 = circle. The lower limb is refracted more
        // than the upper limb (Saemundsson 1986), so the apparent disc
        // compresses vertically into an oval. The differential is
        // computed from the REAL solar disc (~0.265 deg half-angle) so
        // the physical flatten ratio doesn't depend on r_sun_size.
        // Always computed when the cvar is on (one tan() pair, the
        // formula naturally trends toward 1.0 above ~10 deg elevation
        // -- earlier short-circuit gates introduced a visible
        // discontinuity at the cutoff so the unconditional path wins).
        float vflat = 1.0f;
        bool refraction_on = true;
        if (auto* v = C.FindCVar("r_sun_horizon_flatten")) {
            refraction_on = v->GetBool();
        }
        if (refraction_on) {
            constexpr float kRealSunRadiusDeg = 0.265f;
            // Saemundsson 1986: R(h) = 1 / tan((h + 7.31/(h+4.4)) deg) arcmin.
            // h is apparent altitude in degrees; clamp the lower limb
            // queries so the formula's pole at h ≈ -4.4 stays bounded.
            auto saemundsson = [](float h_deg) {
                const float h = std::max(h_deg, -1.5f);
                const float arg_deg = h + 7.31f / (h + 4.4f);
                return 1.0f / std::tan(arg_deg * 3.14159265358979323846f / 180.0f);
            };
            const float R_top = saemundsson(sun_elev_deg + kRealSunRadiusDeg);
            const float R_bot = saemundsson(sun_elev_deg - kRealSunRadiusDeg);
            const float diff_arcmin = R_bot - R_top;             // bottom lifted more
            const float diff_deg    = diff_arcmin / 60.0f;
            const float vert_diam_orig_deg = 2.0f * kRealSunRadiusDeg;
            // Floor at 0.5 (50% squish) so degenerate near-horizon queries
            // don't collapse the disc — beyond ~-1.5 deg the formula is
            // anyway leaving its physical regime. The clamp also brings
            // vflat back to ~1.0 for sun centers far below the horizon
            // (both limbs hit the same clamped h, so the differential
            // collapses to zero) -- correct fallback rather than nonsense.
            vflat = std::clamp(1.0f - diff_deg / vert_diam_orig_deg,
                               0.5f, 1.0f);
        }
        // State-transition log on enable/disable AND on first crossing
        // into the active-refraction band, per the engine's
        // log-on-state-transition convention. Note vflat returns to
        // ~1.0 in TWO regimes -- high elevation (formula naturally
        // small) and far below horizon (clamp degenerate) -- so the
        // inactive log just reports the elevation rather than naming
        // a single threshold.
        static bool s_refr_logged_on = false;
        static bool s_refr_logged_active = false;
        const bool active = (vflat < 0.999f);
        if (refraction_on != s_refr_logged_on) {
            LOG_INFO("engine: r_sun_horizon_flatten={} -- atmospheric "
                     "refraction disc-flatten {}",
                     refraction_on ? 1 : 0,
                     refraction_on ? "ENABLED" : "DISABLED");
            s_refr_logged_on = refraction_on;
            s_refr_logged_active = false;
        }
        if (refraction_on && active != s_refr_logged_active) {
            if (active) {
                LOG_INFO("engine: sun horizon-refraction active "
                         "(elev={:.2f} deg, vertical-scale={:.3f}, "
                         "{:.1f}% squish)",
                         sun_elev_deg, vflat, (1.0f - vflat) * 100.0f);
            } else {
                LOG_INFO("engine: sun horizon-refraction inactive "
                         "(elev={:.2f} deg, vertical-scale={:.3f})",
                         sun_elev_deg, vflat);
            }
            s_refr_logged_active = active;
        }
        push.sun_extra[2] = vflat;
        // sun_extra[3] = planet radius in metres for spherical-Earth
        // atmospheric transmittance (issue #51). The shader's
        // `atmosphericTransmittance` numerically integrates Mie + Rayleigh
        // optical depth along the curved chord when this is > 0;
        // 0 falls back to the legacy planar exponential integral
        // (1/sin(elev) airmass divergence at horizon).
        float planet_radius_m = 6378137.0f;
        if (auto* v = C.FindCVar("r_planet_radius")) planet_radius_m = v->GetFloat();
        // Negative values are nonsense -- clamp to 0 to mean "disable
        // curved-Earth, use planar". Anything between 0 and ~Earth radius
        // produces an unphysical-but-rendered-correctly miniature planet
        // (clouds touch the horizon sooner). We don't impose an upper
        // bound so future moonlit / Mars / gas-giant render passes can
        // pass realistic radii.
        if (planet_radius_m < 0.0f) planet_radius_m = 0.0f;
        push.sun_extra[3] = planet_radius_m;
    }

    // HDRI multi-light array (computed at HDRI load in ReloadEnvMap).
    // Each entry is a directional light extracted from a bright cluster
    // in the env map; the shader's stochastic NEE picks one per sample
    // weighted by .w (pmf). Count = 0 disables HDRI directional NEE
    // (the shader falls back to env-map-only). Only meaningful when
    // env_map_present == 1.
    for (std::uint32_t i = 0; i < Engine::kMaxHdriLights; ++i) {
        const HdriLight& L = hdri_lights_[i];
        push.hdri_lights_dir[i][0] = L.dir.x;
        push.hdri_lights_dir[i][1] = L.dir.y;
        push.hdri_lights_dir[i][2] = L.dir.z;
        push.hdri_lights_dir[i][3] = L.pmf;
        push.hdri_lights_col[i][0] = L.irradiance.r;
        push.hdri_lights_col[i][1] = L.irradiance.g;
        push.hdri_lights_col[i][2] = L.irradiance.b;
        push.hdri_lights_col[i][3] = 0.0f;
    }
    push.hdri_lights_count = (env_map_tex_id_ != 0) ? hdri_lights_count_ : 0u;
    // _hdri_pad and write_normal_gbuffer (which used to be _hdri_pad[0])
    // are already zero-initialized by the `PtPush push{};` aggregate
    // value-init at the top of the function. The defensive
    // `push._hdri_pad[0..2] = 0u` line that used to live here did an
    // out-of-bounds write when _hdri_pad shrank from [3] to [2] (since
    // write_normal_gbuffer claimed the [0] slot); it has been removed.
    // write_normal_gbuffer is set explicitly earlier in the function.

    // Sky mode resolution. "hdri" with no env map loaded falls back
    // to gradient so the shader doesn't read an unbound texture.
    std::string sky_mode_str = "procedural";
    if (auto* v = C.FindCVar("r_sky_mode")) sky_mode_str = v->value;
    std::uint32_t sky_mode_id = 2;
    if      (sky_mode_str == "gradient")   sky_mode_id = 0;
    else if (sky_mode_str == "hdri")       sky_mode_id = (env_map_tex_id_ != 0) ? 1u : 0u;
    else /* procedural (default) */        sky_mode_id = 2;
    push.sun_and_mode[3] = float(sky_mode_id);
    push.env_map_present = (sky_mode_id == 1u) ? 1u : 0u;

    bool auto_exp = false;
    if (auto* v = C.FindCVar("r_auto_exposure")) auto_exp = v->GetBool();
    // exposure_pad.x is dead in the shader -- PathTrace.slang reads the
    // live exposure scalar straight from `exposure_state[0]` (updated
    // every frame by AutoExposure.slang, or seeded by the engine on
    // r_exposure on_change when r_auto_exposure=0). Zero this slot so
    // tooling that dumps the push doesn't see a stale 1.5f.
    push.exposure_pad[0] = 0.0f;
    bool show_stars = true;
    if (auto* v = C.FindCVar("r_show_stars")) show_stars = v->GetBool();
    std::string stars_mode = "bsc";
    if (auto* v = C.FindCVar("r_stars_mode")) stars_mode = v->value;
    // BSC requested + catalog loaded -> sample the J2000 starmap (.z).
    // Otherwise procedural hash starfield (.y). r_show_stars=0 zeroes both.
    const bool want_bsc =
        show_stars && stars_mode == "bsc" && star_map_present_ != 0u;
    const bool want_procedural =
        show_stars && (stars_mode == "procedural" ||
                       (stars_mode == "bsc" && star_map_present_ == 0u));
    push.exposure_pad[1] = want_procedural ? 1.0f : 0.0f;
    push.exposure_pad[2] = want_bsc        ? 1.0f : 0.0f;
    bool twinkle = true;
    if (auto* v = C.FindCVar("r_stars_twinkle")) twinkle = v->GetBool();
    float twinkle_speed = 0.3f;
    if (auto* v = C.FindCVar("r_stars_twinkle_speed")) twinkle_speed = v->GetFloat();
    // Clamp non-negative so a typo can't run twinkle backwards (would
    // alias visually with reversed-direction wrap but is mathematically
    // valid, so guard rather than abs-clamp -- 0 unambiguously means
    // "off" downstream and that's the safer end of any user error).
    if (twinkle_speed < 0.0f) twinkle_speed = 0.0f;
    // exposure_pad.w doubles as the twinkle-active flag (>0) AND the
    // frequency multiplier. r_stars_twinkle=0 zeroes it (shader skips
    // the modulation branch entirely), r_stars_twinkle=1 forwards the
    // speed multiplier so the shader does fhz_base * exposure_pad.w.
    push.exposure_pad[3] = twinkle ? twinkle_speed : 0.0f;

    // World->J2000 rotation. Always computed; the shader scales the
    // starmap sample by exposure_pad.z so when stars are off the
    // matrix is just unused work. lat/lon come from the geographic
    // cvars even when r_sky_use_astronomical = 0 -- stars still need
    // the observer's actual location to be in the right place.
    {
        double lat = 13.0827, lon = 80.2707;
        if (auto* v = C.FindCVar("r_sky_lat")) lat = v->GetFloat();
        if (auto* v = C.FindCVar("r_sky_lon")) lon = v->GetFloat();
        const double jd = compute_jd();
        float m[9];
        pt::astro::worldToJ2000Matrix(lat, lon, jd, m);
        push.w2j_row0[0] = m[0]; push.w2j_row0[1] = m[1]; push.w2j_row0[2] = m[2];
        push.w2j_row1[0] = m[3]; push.w2j_row1[1] = m[4]; push.w2j_row1[2] = m[5];
        // .w packs r_firefly_clamp -- per-contribution radiance ceiling
        // (each indirect lighting term clamped individually: env-NEE,
        // ambient skylight, bounce-to-sky). Camera-direct sky bypasses
        // this and remains unbounded so the HDRI sun appears at full
        // intensity in the rendered sky. 0 disables.
        float firefly_clamp = 10.0f;
        if (auto* v = C.FindCVar("r_firefly_clamp")) firefly_clamp = v->GetFloat();
        if (firefly_clamp < 0.0f) firefly_clamp = 0.0f;
        push.w2j_row2[0] = m[6]; push.w2j_row2[1] = m[7]; push.w2j_row2[2] = m[8]; push.w2j_row2[3] = firefly_clamp;
        // The 3x3 rotation only fills 9 floats; the w-lanes are
        // available scratch. Pack engine flags here:
        //   row0.w = HDR-pipeline (1 = raw HDR through MetalFX).
        //   row1.w = refractive-shadow-ray budget. 0 disables caustics
        //     entirely (NEE rays treat dielectrics as opaque); positive
        //     = max number of dielectric bounces a shadow ray may chain
        //     before giving up. Stored as a float because Slang push
        //     constants are easier to align that way.
        bool hdr_pipeline = true;
        if (auto* v = C.FindCVar("r_hdr_pipeline")) hdr_pipeline = v->GetBool();
        push.w2j_row0[3] = hdr_pipeline ? 1.0f : 0.0f;
        bool caustics = true;
        if (auto* v = C.FindCVar("r_caustics")) caustics = v->GetBool();
        int refract_bounces = 4;
        if (auto* v = C.FindCVar("r_refract_bounces")) refract_bounces = v->GetInt();
        if (refract_bounces < 0)  refract_bounces = 0;
        if (refract_bounces > 16) refract_bounces = 16;
        push.w2j_row1[3] = caustics ? float(refract_bounces) : 0.0f;
    }

    {
        bool dof_on = false;
        if (auto* v = C.FindCVar("r_dof")) dof_on = v->GetBool();
        float aperture = 0.0f, focal_dist = 5.0f, blades = 0.0f;
        if (auto* v = C.FindCVar("r_dof_aperture"))        aperture   = v->GetFloat();
        if (auto* v = C.FindCVar("r_dof_focal_distance"))  focal_dist = v->GetFloat();
        if (auto* v = C.FindCVar("r_dof_blades"))          blades     = float(v->GetInt());
        push.dof_params[0] = dof_on ? aperture : 0.0f;
        push.dof_params[1] = focal_dist;
        push.dof_params[2] = blades;
        push.dof_params[3] = 0.0f;
    }

    {
        bool  vol_on  = false;
        float density = 0.02f, anisotropy = 0.7f, intensity = 1.0f;
        int   samples = 16;
        if (auto* v = C.FindCVar("r_volumetric"))            vol_on     = v->GetBool();
        if (auto* v = C.FindCVar("r_volumetric_density"))    density    = v->GetFloat();
        if (auto* v = C.FindCVar("r_volumetric_anisotropy")) anisotropy = v->GetFloat();
        if (auto* v = C.FindCVar("r_volumetric_intensity"))  intensity  = v->GetFloat();
        if (auto* v = C.FindCVar("r_volumetric_samples"))    samples    = v->GetInt();
        if (samples < 4)  samples = 4;
        if (samples > 64) samples = 64;
        push.vol_params[0] = vol_on ? density : 0.0f;
        push.vol_params[1] = anisotropy;
        push.vol_params[2] = intensity;
        push.vol_params[3] = float(samples);
    }

    {
        // Volumetric clouds: ride the same march as homogeneous haze.
        // peak_density (clouds_p1.w) of 0 disables sampling; the shader
        // collapses back to plain haze sigma_t in that case.
        bool  clouds_on   = false;
        float coverage    = 0.45f;
        float base_y      = 200.0f;
        float top_y       = 500.0f;
        float density     = 12.0f;
        float freq        = 0.005f;
        float detail      = 0.35f;
        float wind_x      = 5.0f;
        float wind_z      = 0.0f;
        float seed        = 0.0f;
        // Issue #117: curl-noise displacement + secondary edge erosion.
        // Defaults to 0 so existing scenes are bit-equivalent to main
        // until the user opts in via r_clouds_curl_amount / r_clouds_erosion.
        float curl_amt    = 0.0f;
        float curl_scale  = 0.01f;
        float erosion     = 0.0f;
        if (auto* v = C.FindCVar("r_clouds"))             clouds_on = v->GetBool();
        if (auto* v = C.FindCVar("r_clouds_coverage"))    coverage  = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_base_height")) base_y    = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_top_height"))  top_y     = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_density"))     density   = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_freq"))        freq      = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_detail"))      detail    = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_curl_amount")) curl_amt  = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_curl_scale"))  curl_scale= v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_erosion"))     erosion   = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_wind_x"))      wind_x    = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_wind_z"))      wind_z    = v->GetFloat();
        if (auto* v = C.FindCVar("r_clouds_seed"))        seed      = v->GetFloat();
        if (top_y < base_y + 1.0f) top_y = base_y + 1.0f;
        // Time accumulator for wind drift. Uses frame_index * a target
        // frame interval -- imprecise versus wall clock but stable
        // across reset_accum and reproducible per-frame for the
        // accumulator's running mean (no temporal noise from clock jitter).
        //
        // Wrap on a 24-hour sim window so the noise sample coordinate
        // (pos + wind * t) stays in fp32-precise range even after long
        // sessions.  86400 s × 60 fps = 5,184,000 frames; modulo the
        // integer frame counter first so the float conversion is exact
        // (well under the ~16M fp32-integer boundary).  Practical drift
        // ceiling: with default wind 5 m/s, 432 km -- still well below
        // anywhere fp32 loses 1-meter precision.  Wrap is invisible
        // under normal use; only matters if someone leaves the engine
        // running for >24h sim time.
        constexpr std::uint64_t kCloudPeriodFrames = 86400ull * 60ull;
        const float t_seconds = float(frame_index_ % kCloudPeriodFrames)
                              * (1.0f / 60.0f);
        push.clouds_p1[0] = base_y;
        push.clouds_p1[1] = top_y;
        push.clouds_p1[2] = clouds_on ? coverage : 0.0f;
        push.clouds_p1[3] = clouds_on ? density  : 0.0f;
        push.clouds_p2[0] = wind_x;
        push.clouds_p2[1] = wind_z;
        push.clouds_p2[2] = freq;
        push.clouds_p2[3] = t_seconds;
        // Hash the seed into two coordinate offsets so r_clouds_seed=1
        // and =2 produce visually different (not just shifted) fields.
        push.clouds_p3[0] = seed * 137.137f;
        push.clouds_p3[1] = seed * 271.951f;
        push.clouds_p3[2] = detail;
        float rayleigh = 1.0f;
        if (auto* v = C.FindCVar("r_rayleigh")) rayleigh = v->GetFloat();
        push.clouds_p3[3] = rayleigh;
        // Issue #117: curl-noise displacement + secondary edge erosion.
        // Gated on clouds_on (when r_clouds 0 the whole density field is
        // zeroed anyway, but keeping the magnitudes at 0 here keeps the
        // shader-side `if (cp4.x > 0)` branch off as well -- shaves a few
        // ALU on the haze-only path).
        push.clouds_p4[0] = clouds_on ? curl_amt   : 0.0f;
        push.clouds_p4[1] = curl_scale;
        push.clouds_p4[2] = clouds_on ? erosion    : 0.0f;
        push.clouds_p4[3] = 0.0f;   // reserved
    }

    // --- Water Phase 1 (#134) ----------------------------------------------
    // Drive PathTrace.slang's MAT_WATER BRDF branch. Defaults match the
    // cvar defaults registered above; FindCVar nullptr-tolerant in case
    // a tooling path skipped registration. Time uses the same wrapped
    // accumulator as clouds_p2.w so the wave field stays fp32-precise
    // even on long sessions; r_water_wave_speed scales the rate.
    {
        float abs_r = 0.45f, abs_g = 0.15f, abs_b = 0.05f;
        float ior = 1.33f;
        float wave_scale = 0.3f, wave_amp = 0.2f, wave_speed = 1.0f;
        if (auto* v = C.FindCVar("r_water_absorption_r"))  abs_r = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_absorption_g"))  abs_g = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_absorption_b"))  abs_b = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_ior"))           ior   = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_wave_scale"))    wave_scale = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_wave_amplitude"))wave_amp   = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_wave_speed"))    wave_speed = v->GetFloat();
        // Reuse the cloud time accumulator (already wrapped to 24h sim).
        // Re-derive locally because the cloud block scoped its t_seconds
        // out of reach. Same wrap rationale: keeps fp32-precise on long
        // sessions; identical phase across cloud-wind & water-waves.
        constexpr std::uint64_t kWaterPeriodFrames = 86400ull * 60ull;
        const float water_t = float(frame_index_ % kWaterPeriodFrames)
                            * (1.0f / 60.0f);
        push.water_params0[0] = abs_r;
        push.water_params0[1] = abs_g;
        push.water_params0[2] = abs_b;
        push.water_params0[3] = ior;
        push.water_params1[0] = wave_scale;
        push.water_params1[1] = wave_amp;
        push.water_params1[2] = wave_speed;
        push.water_params1[3] = water_t;
    }
    // --- end Water Phase 1 -------------------------------------------------

    // --- Fluid Phase 1 (#136) -- smoke emitters ----------------------------
    // Pack: (count, enabled, time, reserved). count is the actual GPU-side
    // upload count produced by EnsureSmokeEmittersUploaded above, NOT
    // the unbounded engine vector size -- already clamped to the cvar
    // r_smoke_max_emitters AND the hard cap kMaxSmokeEmitters.
    {
        bool smoke_on = false;
        if (auto* v = C.FindCVar("r_smoke_enabled")) smoke_on = v->GetBool();
        // Reuse the cloud time accumulator (already wrapped to 24h sim
        // window) so the wave / cloud / smoke drift fields share a
        // coherent phase. Re-derive locally because the cloud block
        // scoped its t_seconds out of reach.
        constexpr std::uint64_t kSmokePeriodFrames = 86400ull * 60ull;
        const float smoke_t = float(frame_index_ % kSmokePeriodFrames)
                            * (1.0f / 60.0f);
        push.smoke_params[0] = static_cast<float>(smoke_count_uploaded_);
        push.smoke_params[1] = smoke_on ? 1.0f : 0.0f;
        push.smoke_params[2] = smoke_t;
        // smoke_params.w: piggyback the r_cloud_attenuate_background
        // flag here (1.0 = apply Beer-Lambert sky attenuation through
        // the cloud-march volume, 0.0 = legacy in-scatter only). This
        // slot was reserved on the original PR #136 push struct;
        // re-using it avoids growing PtPush + bumping the static-assert
        // sum. See r_cloud_attenuate_background docstring.
        bool attenuate_bg = false;
        if (auto* v = C.FindCVar("r_cloud_attenuate_background")) {
            attenuate_bg = v->GetBool();
        }
        push.smoke_params[3] = attenuate_bg ? 1.0f : 0.0f;

        // --- Fluid Phase 2 (#180) -- advected emission --------------------
        // Pack per-frame Phase 2 knobs. Clamp to safe ranges here so the
        // shader can trust the inputs without re-validating each pixel.
        float lifetime = 4.0f;
        if (auto* v = C.FindCVar("r_smoke_lifetime")) lifetime = v->GetFloat();
        if (lifetime < 0.0f) lifetime = 0.0f;
        float adv = 1.0f;
        if (auto* v = C.FindCVar("r_smoke_advection_strength")) adv = v->GetFloat();
        int puffs = 12;
        if (auto* v = C.FindCVar("r_smoke_puff_count")) puffs = v->GetInt();
        if (puffs < 1) puffs = 1;
        // Hard upper bound mirrors the shader's compile-time loop cap
        // (kSmokePuffMax = 16 in PathTrace.slang). Wave-6 bumped from 8
        // to 16 so the chain reads as a continuous column rather than a
        // string of discrete blobs. Higher values would multiply the
        // per-pixel cost without a clean perf budget.
        if (puffs > 16) puffs = 16;
        float expansion = 0.15f;
        if (auto* v = C.FindCVar("r_smoke_expansion_rate")) expansion = v->GetFloat();
        if (expansion < 0.0f) expansion = 0.0f;
        push.smoke_params2[0] = lifetime;
        push.smoke_params2[1] = adv;
        push.smoke_params2[2] = static_cast<float>(puffs);
        push.smoke_params2[3] = expansion;
        // --- end Fluid Phase 2 --------------------------------------------
        // --- Fluid wave-6 -- "looks like smoke" ---------------------------
        // Pack the per-frame visual-fidelity knobs that drive turbulence
        // detail, age-fade shape, and age-driven tint. All defaults
        // chosen so smoke_emitter_basic.cfg renders as recognisable
        // smoke out of the box without per-scene tuning.
        float turb_scale = 4.0f;
        if (auto* v = C.FindCVar("r_smoke_turbulence_scale")) turb_scale = v->GetFloat();
        if (turb_scale < 0.0f) turb_scale = 0.0f;
        float turb_amp = 0.6f;
        if (auto* v = C.FindCVar("r_smoke_turbulence_amplitude")) turb_amp = v->GetFloat();
        if (turb_amp < 0.0f) turb_amp = 0.0f;
        if (turb_amp > 1.0f) turb_amp = 1.0f;
        float decay_shape = 2.0f;
        if (auto* v = C.FindCVar("r_smoke_decay_shape")) decay_shape = v->GetFloat();
        if (decay_shape < 1.0f) decay_shape = 1.0f;
        float age_tint = 0.2f;
        if (auto* v = C.FindCVar("r_smoke_age_tint_strength")) age_tint = v->GetFloat();
        if (age_tint < 0.0f) age_tint = 0.0f;
        if (age_tint > 1.0f) age_tint = 1.0f;
        push.smoke_params3[0] = turb_scale;
        push.smoke_params3[1] = turb_amp;
        push.smoke_params3[2] = decay_shape;
        push.smoke_params3[3] = age_tint;
        // --- end Fluid wave-6 ---------------------------------------------
    }
    // --- end Fluid Phase 1 -------------------------------------------------
    // --- Motion blur (#85) -------------------------------------------------
    // Pack r_motion_blur + r_shutter_speed for the shader. r_motion_blur
    // is the master gate; r_shutter_speed is plumbed even though the
    // shader currently lerps over U(0,1) within the per-frame
    // prev/curr window (per-ray shutter weighting is a future hook --
    // see PtPush::motion_blur_params docstring above).
    {
        bool motion_blur_on = false;
        if (auto* v = C.FindCVar("r_motion_blur")) motion_blur_on = v->GetBool();
        float shutter_s = 0.0167f;
        if (auto* v = C.FindCVar("r_shutter_speed")) shutter_s = v->GetFloat();
        if (shutter_s < 0.0f) shutter_s = 0.0f;
        push.motion_blur_params[0] = motion_blur_on ? 1.0f : 0.0f;
        push.motion_blur_params[1] = shutter_s;
        push.motion_blur_params[2] = 0.0f;
        push.motion_blur_params[3] = 0.0f;
    }
    // --- end Motion blur ---------------------------------------------------
    // --- Underwater medium tracking (Phase 4, #134) ------------------------
    // Pack:
    //   .x = 1.0 if camera position is inside any MAT_WATER plane's
    //        below-the-surface half-space, else 0.0. The shader uses
    //        this to seed `in_water = 1` at ray-gen so the camera
    //        primary ray's segment through the water medium gets the
    //        new generic per-bounce Beer's-law attenuation (replacing
    //        the b == 0 special case from b2690ac).
    //   .yzw = r_water_deep_color_r/g/b -- the RGB the shader returns
    //          when an in-water ray escapes scene geometry to infinity.
    //          Defaults to (0, 0, 0) so deep-water voids read as
    //          honest black; raise per-cvar for tunable scattering
    //          ambient that fakes deep-water glow without a volumetric
    //          integral.
    {
        // Scan analytic primitives for MAT_WATER planes. Each MAT_WATER
        // plane is defined by normal `pos_or_n` and offset `radius_or_d`,
        // following the engine's plane convention `dot(n, p) + d = 0`
        // (the same form `prim_plane`'s help text documents and that
        // intersectPlane uses on both CPU and shader paths). The camera
        // sits "below" the surface -- on the side opposite the normal --
        // when dot(n, cam) + d <= 0. For the typical horizontal water
        // surface at y = H stored as normal = (0, 1, 0) + d = -H, this
        // collapses to cam.pos.y <= H, which is what users expect.
        //
        // Orientation: PR #207 added a unit quaternion per analytic prim
        // (AnalyticPrim::orient[4]) that the shader uses to rotate plane
        // normals at intersect time via quatRotate (see PRIM_PLANE branch
        // in testAnalyticPrim). We mirror that rotation here so a rotated
        // water plane's below-the-surface half-space matches what the
        // shader actually intersects; otherwise a non-identity orient
        // would seed in_water from the stored (unrotated) half-space,
        // not the visible one. Identity orient (0, 0, 0, 1) collapses
        // to the original n_raw so unrotated planes are unchanged.
        //
        // Multiple water planes are allowed (e.g. an indoor pool above
        // an aquarium): camera is in-water if it sits below ANY of them.
        // Conservative AND-vs-OR: a configuration of two water planes
        // that's actually a thin slab would mark "below" for both, and
        // we'd want in_water = 1 between them anyway. The general N-water
        // case is rare; this OR-fold is good enough for Phase 4.
        bool cam_in_water = false;
        for (const auto& [id, p] : primitives_) {
            if (p.type != AnalyticPrim::Plane) continue;
            if (p.material != AnalyticPrim::Water) continue;
            // Rotate the stored normal by the orientation quaternion --
            // matches the shader's PRIM_PLANE quatRotate path (see
            // PathTrace.slang testAnalyticPrim). quatRotate(q, v) =
            // v + 2 * cross(u, cross(u, v) + w * v) with q = (u, w).
            const glm::vec3 n_raw{p.pos_or_n[0], p.pos_or_n[1],
                                  p.pos_or_n[2]};
            const glm::vec3 u{p.orient[0], p.orient[1], p.orient[2]};
            const float    qw = p.orient[3];
            const glm::vec3 n =
                n_raw + 2.0f * glm::cross(u, glm::cross(u, n_raw) + qw * n_raw);
            const float d  = p.radius_or_d;
            const float signed_dist =
                n.x * cam.pos.x + n.y * cam.pos.y + n.z * cam.pos.z + d;
            if (signed_dist <= 0.0f) { cam_in_water = true; break; }
        }
        float deep_r = 0.0f, deep_g = 0.0f, deep_b = 0.0f;
        if (auto* v = C.FindCVar("r_water_deep_color_r")) deep_r = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_deep_color_g")) deep_g = v->GetFloat();
        if (auto* v = C.FindCVar("r_water_deep_color_b")) deep_b = v->GetFloat();
        // Clamp to [0, +inf) -- negative deep tint would subtract energy
        // from the in-flight throughput on miss, which is unphysical and
        // would hide debugging via NaNs once the radiance accumulates.
        if (deep_r < 0.0f) deep_r = 0.0f;
        if (deep_g < 0.0f) deep_g = 0.0f;
        if (deep_b < 0.0f) deep_b = 0.0f;
        push.water_state[0] = cam_in_water ? 1.0f : 0.0f;
        push.water_state[1] = deep_r;
        push.water_state[2] = deep_g;
        push.water_state[3] = deep_b;
    }
    // --- end Underwater medium tracking ------------------------------------

    // --- Editor backend (agent-19 / PR #201, host wired in #210) ------------
    // Pack the selection-outline state for PathTrace.slang. .x is the
    // EFFECTIVE selected prim id: zero when the cvar gate is off or the
    // active selection kind isn't AnalyticPrim, otherwise selected_prim_id_.
    // Folding the gate into .x lets the shader early-out on a single uint
    // compare without re-reading .y / .z on every analytic-prim hit. .y
    // still carries the raw gate so other shader paths can branch on it
    // independently; .z mirrors the host SelectionKind enum so future
    // Light / SdfCluster / RigidBody outline variants can dispatch without
    // another push field.
    {
        bool outline_on = true;
        if (auto* v = C.FindCVar("r_editor_outline")) outline_on = v->GetBool();
        const bool paint = outline_on &&
                           selected_kind_ == SelectionKind::AnalyticPrim;
        push.editor_params[0] = paint ? selected_prim_id_ : 0u;
        push.editor_params[1] = outline_on ? 1u : 0u;
        push.editor_params[2] = static_cast<std::uint32_t>(selected_kind_);
        push.editor_params[3] = 0u;
    }
    // --- end Editor backend -------------------------------------------------

    // PtPush layout: the trailing 32 bytes here include the SDF Phase 1
    // block (sdf_params uvec4 + sdf_params_f vec4) and the celestials-
    // composite gate (issue #46 -- one uint flag plus 12 bytes of
    // explicit padding that the SPIR-V / MSL cbuffer rule would have
    // inserted anyway). Mirrored in PathTrace.slang's Push/Frame block.
    // 768 (pre-SDF; 112 B push prefix + 656 B spilled tail before SDF)
    // + 16 (sdf_params uvec4) + 16 (sdf_params_f vec4) +
    // 16 (composite_celestials + 12 B pad) = 816 B baseline. Issue #117
    // adds clouds_p4 (curl-noise plumbing) = +16 B for a total 832 B.
    // Vulkan keeps the first 112 B in push constants and spills the
    // rest into the Frame UBO (kFrameUboSize = 1024); Metal keeps the
    // whole struct in a setBytes-style slot.
    // Trailing additions on top of the #117 baseline:
    //   +16 SIGMA shadow demod block (#115)
    //   +16 hetero volumetric block (#100)
    //   +16 analytic light primitives block (#73)
    //   +16 MetalFX specular guidance write gates (#118)
    //   +32 Water Phase 1 (#134) — two vec4s (absorption + ior, wave params)
    //   +16 ReSTIR DI Phase A (#78)
    //   +16 Fluid Phase 1 (#136) — smoke_params vec4 (count, enabled, time, _pad)
    //   +16 Fluid Phase 2 (#180) — smoke_params2 vec4 (lifetime, adv_strength,
    //                              puff_count, expansion_rate)
    //   +16 Fluid wave-6      — smoke_params3 vec4 (turb_scale, turb_amp,
    //                              decay_shape, age_tint_strength)
    //   +16 Motion blur (#85) — motion_blur_params vec4 (enabled, shutter_speed,
    //                              _pad, _pad)
    //   +16 Water Phase 4 (#134) — water_state vec4 (in_water flag, deep_color_rgb)
    //   +16 Editor backend (#201/#210) — editor_params uvec4 (selected_prim_id,
    //                              r_editor_outline gate, selected_kind, _pad)
    static_assert(sizeof(PtPush) == 272 + 48 + 16 + 16 + 48 + 16 + 16 + 16 + 16 + 128 + 128 + 20 + 12 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16 + 16);
    // Alignment guards: every vec4 / uvec4 field in the host PtPush
    // must sit on a 16-byte boundary to match the std140 / MSL
    // cbuffer layout the Slang compiler applies to PathTrace.slang's
    // `Push` (Metal) and `Frame` (Vulkan) blocks. The shader compiler
    // silently inserts padding to honour that rule, so if the host
    // forgets to mirror it the GPU reads from 12 bytes ahead of where
    // the host wrote (the bug class that landed `_pad_before_accum_params`
    // above). Catch the next regression at compile time.
    // Issue #117: clouds_p4 was inserted between clouds_p3 and
    // moon_dir_phase. Both anchor the trailing block at a 16-byte
    // boundary already (all elements above are float4-sized), but
    // guard explicitly so a future field re-ordering can't silently
    // slip the std140 layout out from under the shader.
    static_assert(offsetof(PtPush, clouds_p4) % 16 == 0,
                  "PtPush::clouds_p4 must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    static_assert(offsetof(PtPush, accum_params) % 16 == 0,
                  "PtPush::accum_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    static_assert(offsetof(PtPush, bvh_params) % 16 == 0,
                  "PtPush::bvh_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // --- SDF Phase 1 (#97) -------------------------------------------------
    static_assert(offsetof(PtPush, sdf_params) % 16 == 0,
                  "PtPush::sdf_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    static_assert(offsetof(PtPush, sdf_params_f) % 16 == 0,
                  "PtPush::sdf_params_f must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // --- end SDF Phase 1 ---------------------------------------------------
    static_assert(offsetof(PtPush, composite_celestials) % 16 == 0,
                  "PtPush::composite_celestials must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Issue #118 MetalFX specular guidance write gates land in their own
    // 16-byte block right after the composite_celestials block.
    static_assert(offsetof(PtPush, write_specular_albedo_gbuffer) % 16 == 0,
                  "PtPush::write_specular_albedo_gbuffer must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // --- Water Phase 1 (#134) ------------------------------------------------
    static_assert(offsetof(PtPush, water_params0) % 16 == 0,
                  "PtPush::water_params0 must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    static_assert(offsetof(PtPush, water_params1) % 16 == 0,
                  "PtPush::water_params1 must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // --- end Water Phase 1 ---------------------------------------------------
    // Light primitives + light tree 16-byte block anchor (#73 / #129).
    static_assert(offsetof(PtPush, light_count) % 16 == 0,
                  "PtPush::light_count must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // ReSTIR DI Phase A block (#78).
    static_assert(offsetof(PtPush, restir_enabled) % 16 == 0,
                  "PtPush::restir_enabled must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Fluid Phase 1 block (#136).
    static_assert(offsetof(PtPush, smoke_params) % 16 == 0,
                  "PtPush::smoke_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Fluid Phase 2 block (#180).
    static_assert(offsetof(PtPush, smoke_params2) % 16 == 0,
                  "PtPush::smoke_params2 must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Fluid wave-6 block ("looks like smoke" turbulence + age-tint).
    static_assert(offsetof(PtPush, smoke_params3) % 16 == 0,
                  "PtPush::smoke_params3 must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Motion blur block (#85).
    static_assert(offsetof(PtPush, motion_blur_params) % 16 == 0,
                  "PtPush::motion_blur_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Water Phase 4 underwater medium tracking (#134).
    static_assert(offsetof(PtPush, water_state) % 16 == 0,
                  "PtPush::water_state must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    // Editor backend block (agent-19 / PR #201, host wired in #210).
    static_assert(offsetof(PtPush, editor_params) % 16 == 0,
                  "PtPush::editor_params must be 16-byte aligned to match "
                  "std140 / MSL cbuffer layout in PathTrace.slang");
    cb->PushConstants(&push, sizeof(push));
    accum_dirty_ = false;

    auto wg_x = (fc.width  + 7) / 8;
    auto wg_y = (fc.height + 7) / 8;
    cb->Dispatch(wg_x, wg_y, 1);

    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    // Dispatch chain (when ReSTIR is engaged this frame):
    //   1. RestirTemporal : reads reservoir_curr (just written by
    //                       PathTrace), reads reservoir_prev
    //                       (last frame's final), MIS-combines into
    //                       reservoir_curr.
    //   2. RestirSpatial  : reads reservoir_curr (post-temporal), MIS-
    //                       combines with N neighbours, writes
    //                       reservoir_swap.
    //   3. RestirFinal    : reads reservoir_swap, shadow-tests the
    //                       survivor, additively blends the Lambert
    //                       contribution into denoise_color.
    //
    // At end of frame the prev/swap ids are swapped so next frame's
    // RestirTemporal reads this frame's final reservoir as prev. The
    // ping-pong avoids per-frame buffer reallocs.
    if (push.restir_enabled != 0u) {
        // One-shot logging so operators can verify the ReSTIR chain
        // engaged for a given session. Logged at INFO once per
        // engine session to avoid spamming the journal; the same
        // one-shot pattern SIGMA / particles use to advertise
        // first-engagement.
        static bool s_restir_logged = false;
        if (!s_restir_logged) {
            LOG_INFO("engine: ReSTIR DI Phase A engaged "
                     "(K={}, light_count={}, denoiser=on)",
                     push.restir_k_candidates,
                     light_count_uploaded_);
            s_restir_logged = true;
        }
        // RAW hazard: PathTrace wrote reservoir_curr_buf in its
        // candidate-generation block; RestirTemporal is about to read
        // it. Emit a compute-write -> compute-read barrier. Metal
        // auto-barriers, but this matters on Vulkan submission ordering.
        cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                     pt::rhi::BarrierDesc::Stage::ComputeRead});

        std::uint32_t bias_mode_v = pt::renderer::kRestirBiasBiased;
        if (auto* v = C.FindCVar("r_restir_bias")) {
            // CVar.value is the canonical string store; r_restir_bias
            // is a string-valued cvar with allowed_values {biased,
            // unbiased}, so direct comparison is correct here.
            if (v->value == "unbiased") {
                bias_mode_v = pt::renderer::kRestirBiasUnbiased;
            }
        }
        std::uint32_t temporal_on = 1u;
        if (auto* v = C.FindCVar("r_restir_temporal")) {
            temporal_on = v->GetBool() ? 1u : 0u;
        }
        std::uint32_t spatial_on = 1u;
        if (auto* v = C.FindCVar("r_restir_spatial")) {
            spatial_on = v->GetBool() ? 1u : 0u;
        }

        // --- RestirTemporal -------------------------------------------------
        if (restir_temporal_pipeline_id_ != 0) {
            cb->BindComputePipeline(
                pt::rhi::PipelineHandle{restir_temporal_pipeline_id_});
            // Texture slot 0: depth_tex (camera-space Z).
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{depth_tex_id_});
            // Texture slot 1: motion_tex (reprojection delta).
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{motion_tex_id_});
            // Texture slot 2: normal_tex (disocclusion gate). When the
            // active denoiser doesn't allocate normals (rare; the
            // engine forces it on for MetalFX so the bind always has
            // a non-zero id under the dispatch gate above), bind the
            // 1x1 bloom_dummy as a placeholder to satisfy Metal's
            // pipeline validation. The shader's disocclusion code
            // tolerates a black normal -- it just falls back to "no
            // temporal reuse" for that pixel.
            const std::uint64_t restir_normal_id =
                (normal_tex_id_ != 0) ? normal_tex_id_ : bloom_dummy_tex_id_;
            cb->BindStorageTexture(2, pt::rhi::TextureHandle{restir_normal_id});
            // Buffer slot 0: reservoir_curr_in (PathTrace output).
            cb->BindBuffer(0, pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_}, 0);
            // Buffer slot 1: reservoir_prev_in.
            cb->BindBuffer(1, pt::rhi::BufferHandle{restir_reservoir_prev_buf_id_}, 0);
            // Buffer slot 2: reservoir_curr_out (we ping-pong back into
            // the curr buffer via the spatial pass; the temporal pass
            // writes a TRANSIENT result that the spatial pass reads).
            // Use restir_reservoir_swap_buf_id_ here so the temporal
            // output sits in a different buffer than its inputs --
            // necessary to avoid the WAR hazard if the spatial pass
            // reads neighbour pixels at slightly different positions.
            cb->BindBuffer(2, pt::rhi::BufferHandle{restir_reservoir_swap_buf_id_}, 0);
            // Buffer slot 3: light_prims (re-evaluate prev survivor's pdf).
            pt::rhi::BufferHandle restir_lights = (light_buffer_id_ != 0)
                ? pt::rhi::BufferHandle{light_buffer_id_}
                : pt::rhi::BufferHandle{placeholder_storage_id_};
            cb->BindBuffer(3, restir_lights, 0);

            struct RestirTemporalPush {
                std::uint32_t width;
                std::uint32_t height;
                std::uint32_t frame_index;
                std::uint32_t light_count;
                float pos_fovtan[4];
                float fwd_aspect[4];
                float right_xyz[4];
                float up_xyz[4];
                std::uint32_t bias_mode;
                std::uint32_t restir_enabled;
                std::uint32_t temporal_enabled;
                std::uint32_t _pad0;
            } rt{};
            static_assert(sizeof(RestirTemporalPush) % 16 == 0,
                          "RestirTemporalPush must be 16-byte aligned");
            rt.width            = fc.width;
            rt.height           = fc.height;
            rt.frame_index      = push.frame_index;
            rt.light_count      = light_count_uploaded_;
            std::memcpy(rt.pos_fovtan, push.pos_fovtan, sizeof(rt.pos_fovtan));
            std::memcpy(rt.fwd_aspect, push.fwd_aspect, sizeof(rt.fwd_aspect));
            std::memcpy(rt.right_xyz,  push.right_xyz,  sizeof(rt.right_xyz));
            std::memcpy(rt.up_xyz,     push.up_xyz,     sizeof(rt.up_xyz));
            rt.bias_mode        = bias_mode_v;
            rt.restir_enabled   = 1u;
            rt.temporal_enabled = temporal_on;
            rt._pad0            = 0u;
            cb->PushConstants(&rt, sizeof(rt));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            // RAW: temporal wrote restir_reservoir_swap_buf; spatial
            // about to read it.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        }

        // --- RestirSpatial --------------------------------------------------
        if (restir_spatial_pipeline_id_ != 0) {
            cb->BindComputePipeline(
                pt::rhi::PipelineHandle{restir_spatial_pipeline_id_});
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{depth_tex_id_});
            const std::uint64_t restir_normal_id =
                (normal_tex_id_ != 0) ? normal_tex_id_ : bloom_dummy_tex_id_;
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{restir_normal_id});
            // Input: post-temporal reservoir (swap buffer).
            cb->BindBuffer(0, pt::rhi::BufferHandle{restir_reservoir_swap_buf_id_}, 0);
            // Output: final reservoir for this frame, lands in curr
            // (the spatial pass writes the FINAL survivor that
            // RestirFinal reads and that becomes next frame's prev).
            cb->BindBuffer(1, pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_}, 0);
            pt::rhi::BufferHandle restir_lights = (light_buffer_id_ != 0)
                ? pt::rhi::BufferHandle{light_buffer_id_}
                : pt::rhi::BufferHandle{placeholder_storage_id_};
            cb->BindBuffer(2, restir_lights, 0);

            struct RestirSpatialPush {
                std::uint32_t width;
                std::uint32_t height;
                std::uint32_t frame_index;
                std::uint32_t light_count;
                float pos_fovtan[4];
                float fwd_aspect[4];
                float right_xyz[4];
                float up_xyz[4];
                std::uint32_t bias_mode;
                std::uint32_t restir_enabled;
                std::uint32_t spatial_enabled;
                std::uint32_t _pad0;
            } rs{};
            static_assert(sizeof(RestirSpatialPush) % 16 == 0,
                          "RestirSpatialPush must be 16-byte aligned");
            rs.width            = fc.width;
            rs.height           = fc.height;
            rs.frame_index      = push.frame_index;
            rs.light_count      = light_count_uploaded_;
            std::memcpy(rs.pos_fovtan, push.pos_fovtan, sizeof(rs.pos_fovtan));
            std::memcpy(rs.fwd_aspect, push.fwd_aspect, sizeof(rs.fwd_aspect));
            std::memcpy(rs.right_xyz,  push.right_xyz,  sizeof(rs.right_xyz));
            std::memcpy(rs.up_xyz,     push.up_xyz,     sizeof(rs.up_xyz));
            rs.bias_mode        = bias_mode_v;
            rs.restir_enabled   = 1u;
            rs.spatial_enabled  = spatial_on;
            rs._pad0            = 0u;
            cb->PushConstants(&rs, sizeof(rs));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        }

        // --- RestirFinal ----------------------------------------------------
        if (restir_final_pipeline_id_ != 0 && albedo_tex_id_ != 0) {
            cb->BindComputePipeline(
                pt::rhi::PipelineHandle{restir_final_pipeline_id_});
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{depth_tex_id_});
            const std::uint64_t restir_normal_id =
                (normal_tex_id_ != 0) ? normal_tex_id_ : bloom_dummy_tex_id_;
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{restir_normal_id});
            cb->BindStorageTexture(2, pt::rhi::TextureHandle{albedo_tex_id_});
            cb->BindStorageTexture(3, pt::rhi::TextureHandle{denoise_color_tex_id_});
            // Input reservoir (post-spatial; lives in curr buf).
            cb->BindBuffer(0, pt::rhi::BufferHandle{restir_reservoir_curr_buf_id_}, 0);
            pt::rhi::BufferHandle restir_lights = (light_buffer_id_ != 0)
                ? pt::rhi::BufferHandle{light_buffer_id_}
                : pt::rhi::BufferHandle{placeholder_storage_id_};
            cb->BindBuffer(1, restir_lights, 0);
            pt::rhi::BufferHandle restir_prims = (prim_buffer_id_ != 0)
                ? pt::rhi::BufferHandle{prim_buffer_id_}
                : pt::rhi::BufferHandle{placeholder_storage_id_};
            cb->BindBuffer(2, restir_prims, 0);
            // TLAS bind for the shadow trace. Only present on backends
            // that support RT (Metal always; Vulkan when MoltenVK has
            // VK_KHR_ray_query).
            if (tlas_present) {
                cb->BindAccelStruct(3, pt::rhi::AccelStructHandle{scene_tlas_id_});
            }

            struct RestirFinalPush {
                std::uint32_t width;
                std::uint32_t height;
                std::uint32_t frame_index;
                std::uint32_t light_count;
                float pos_fovtan[4];
                float fwd_aspect[4];
                float right_xyz[4];
                float up_xyz[4];
                std::uint32_t restir_enabled;
                std::uint32_t prim_count;
                std::uint32_t tlas_present;
                std::uint32_t linear_prim_count;
            } rf{};
            static_assert(sizeof(RestirFinalPush) % 16 == 0,
                          "RestirFinalPush must be 16-byte aligned");
            rf.width             = fc.width;
            rf.height            = fc.height;
            rf.frame_index       = push.frame_index;
            rf.light_count       = light_count_uploaded_;
            std::memcpy(rf.pos_fovtan, push.pos_fovtan, sizeof(rf.pos_fovtan));
            std::memcpy(rf.fwd_aspect, push.fwd_aspect, sizeof(rf.fwd_aspect));
            std::memcpy(rf.right_xyz,  push.right_xyz,  sizeof(rf.right_xyz));
            std::memcpy(rf.up_xyz,     push.up_xyz,     sizeof(rf.up_xyz));
            rf.restir_enabled    = 1u;
            rf.prim_count        = static_cast<std::uint32_t>(primitives_.size());
            rf.tlas_present      = tlas_present ? 1u : 0u;
            rf.linear_prim_count = linear_prim_count_;
            cb->PushConstants(&rf, sizeof(rf));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            // RestirFinal wrote denoise_color; subsequent passes
            // (denoiser / bloom / tonemap) read it. Emit the
            // standard compute-write -> compute-read barrier.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        }

        // End-of-frame ping-pong: swap curr <-> prev so next frame's
        // RestirTemporal reads THIS frame's final survivor reservoir
        // (now in curr) as last-frame state.
        std::swap(restir_reservoir_curr_buf_id_, restir_reservoir_prev_buf_id_);
    }
    // --- end ReSTIR DI Phase A ---------------------------------------------

    // GPU-side auto-exposure: tiny reduction pass that samples
    // accum_hdr (which the path tracer just wrote) and updates the
    // exposure_state buffer the next frame's PathTrace reads. One
    // frame of latency, zero CPU readback, zero queue-drain stall.
    // Replaces the legacy per-8-frames CPU readback that on dGPU
    // forced a vkQueueWaitIdle and tanked fps to ~18.
    if (auto_exp && autoexpose_pipeline_id_ != 0 && exposure_state_id_ != 0
        && accum_texture_id_ != 0) {
        struct AePush {
            std::uint32_t width;
            std::uint32_t height;
            std::uint32_t stride;
            std::uint32_t pad0;
            float key;
            float exp_min;
            float exp_max;
            float adapt_speed;
        } ae{};
        // Phase-0 defensive guard. AePush is scalar-only (4 uints + 4
        // floats = 32B, naturally aligned), but adding a vec4 or
        // reordering into something non-multiple-of-16 would silently
        // shift the shader's reads. Asserts catch it at compile time.
        static_assert(sizeof(AePush) == 32,
                      "AePush layout mismatch with AutoExposure.slang");
        static_assert(sizeof(AePush) % 16 == 0,
                      "AePush must be 16-byte aligned (cbuffer rule)");
        ae.width  = fc.width;
        ae.height = fc.height;
        ae.stride = 16;
        ae.key    = 0.18f;
        ae.exp_min = 0.05f;
        ae.exp_max = 4.0f;
        ae.adapt_speed = 0.20f;
        if (auto* v = C.FindCVar("r_exposure_target")) ae.key         = v->GetFloat();
        if (auto* v = C.FindCVar("r_exposure_min"))    ae.exp_min     = v->GetFloat();
        if (auto* v = C.FindCVar("r_exposure_max"))    ae.exp_max     = v->GetFloat();
        if (auto* v = C.FindCVar("r_eye_adapt_speed")) ae.adapt_speed = v->GetFloat();

        // Compute->compute hazard: the path tracer just wrote accum_hdr
        // and the autoexpose dispatch is about to read it. On Vulkan
        // submission order alone is insufficient per spec (the second
        // dispatch's loads can race the first's stores on some drivers);
        // emit a global compute-write -> compute-read barrier. Metal
        // inserts the equivalent barrier automatically between dispatches
        // on a shared resource, so the Metal backend's Barrier() is a
        // no-op.
        cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                     pt::rhi::BarrierDesc::Stage::ComputeRead});

        cb->BindComputePipeline(pt::rhi::PipelineHandle{autoexpose_pipeline_id_});
        // CRITICAL: re-bind accum_hdr and exposure_state AFTER
        // BindComputePipeline. On Metal, BindComputePipeline clears all
        // resource binds (see MetalCommandBuffer::BindComputePipeline)
        // -- the path-tracer's binds do NOT carry over. On Vulkan binds
        // persist across pipeline switches (the descriptor set is
        // rewritten every Dispatch from bound_tex_/bound_buf_), but
        // explicit re-binds work there too and keep the engine code
        // backend-agnostic. Without this, AutoExposure on Mac runs with
        // accum_hdr nil (reads zero), exposure_state nil (writes drop),
        // and push constants land at the wrong MSL slot (push_slot is
        // computed as max-bound-buf+1 = 0 instead of the kernel's
        // declared buffer(7)) -- exposure converges to garbage.
        cb->BindStorageTexture(1, pt::rhi::TextureHandle{accum_texture_id_});
        cb->BindBuffer(6, pt::rhi::BufferHandle{exposure_state_id_}, 0);
        cb->PushConstants(&ae, sizeof(ae));
        cb->Dispatch(1, 1, 1);  // single workgroup of 64 threads
    }

    // P10 denoise pass: encoded onto the same command buffer between
    // the path tracer dispatch and Submit. MetalFX reads the G-buffer
    // textures the shader just wrote and outputs to the swapchain
    // (overwriting the shader's tonemapped fallback).
    //
    // The outer gate also catches the bloom-without-denoiser path
    // (Metal and Vulkan): in that case the inner `if (denoiser_active_)`
    // around the DenoiseDesc setup + Denoise() call below is skipped,
    // but the engine bloom + Tonemap.slang chain that follows still
    // runs -- its HDR source becomes the path tracer's direct
    // denoise_color write rather than the denoiser's post_denoise_hdr
    // output.
    if (denoiser_active_ || bloom_without_denoiser) {
        // The engine-side Tonemap.slang post-denoise chain (bloom
        // composite + ACES + lens flare + sRGB encode -> swapchain)
        // runs only on Metal today. Tonemap.slang reaches its kernel
        // on Vulkan too, but the dispatch produces a black swapchain
        // on PC -- the descriptor-visibility / barrier root cause
        // hasn't been pinned down yet (validation-layer + RenderDoc
        // capture follow-up). Affects both the SVGF/NRD-active case
        // (writes post_denoise_hdr -> Tonemap reads it) and the
        // bloom-without-denoiser case (writes denoise_color -> Tonemap
        // reads it); the bug is in the Tonemap dispatch itself, not
        // SVGF-specific as originally suspected.
        //
        // What runs in each case:
        //   - Metal + SVGF/MetalFX (denoiser_active_): engine Tonemap.slang
        //     reads post_denoise_hdr, composites bloom pre-ACES, writes swap.
        //   - Metal + bloom-without-denoiser: engine Tonemap.slang reads
        //     denoise_color (PathTrace direct write), same composite chain.
        //   - Vulkan + SVGF/NRD: build the bloom pyramid in-engine from
        //     denoise_color (PathTrace's raw HDR write, pre-denoise),
        //     then pass bloom_mip[0] through Denoise(Kind::Svgf) so
        //     VulkanNrdDenoiser::Encode composites it during its
        //     internal DenoiseFinalize step before writing the swap.
        //     The descriptor-reuse bug PR #11 originally worked around
        //     was the same root cause fixed by PR #18's per-dispatch
        //     descriptor-set ring (kept dispatches from observing each
        //     other's UPDATE_AFTER_BIND writes).
        //   - Vulkan + bloom-without-denoiser: build the bloom pyramid
        //     in-engine, then dispatch VulkanNrdDenoiser::EncodeFinalizeOnly
        //     via Denoise(Kind::FinalizeOnly). EncodeFinalizeOnly has its
        //     own dedicated 4-binding layout / descriptor set, so it
        //     sidesteps the Tonemap.slang descriptor bug.
        //   - OptiX (Vulkan only): defers its actual denoise + copy
        //     into a private cb submitted AFTER the engine cb (CUDA
        //     sync model, see VulkanOptixDenoiser::Encode's flow
        //     comment). The engine never runs the Tonemap chain in
        //     this case (would tonemap stale post_denoise_hdr).
        const bool kind_is_optix =
            (denoiser_kind_ == DenoiserKind::OptixHdr            ||
             denoiser_kind_ == DenoiserKind::OptixHdrAov         ||
             denoiser_kind_ == DenoiserKind::OptixTemporalHdr    ||
             denoiser_kind_ == DenoiserKind::OptixTemporalHdrAov);
        const bool backend_is_metal =
            (current_backend_ == pt::rhi::BackendType::Metal);
        // Metal-only: Tonemap.slang produces black on Vulkan
        // regardless of denoiser state (see comment block above).
        const bool use_engine_tonemap =
            (tonemap_pipeline_id_ != 0 && !kind_is_optix && backend_is_metal);
        // Vulkan bloom-without-denoiser: route the swapchain write
        // through Denoise(Kind::FinalizeOnly) instead. The engine
        // builds the bloom pyramid first (so DenoiseFinalize can
        // composite bloom_mip[0]) and then issues the finalize-only
        // dispatch which writes the swap.
        const bool use_vulkan_bloom_finalize =
            (!backend_is_metal && bloom_without_denoiser &&
             device_->SupportsDenoise());

        pt::rhi::Device::DenoiseDesc dd{};
        dd.color_in      = pt::rhi::TextureHandle{denoise_color_tex_id_};
        dd.depth_in      = pt::rhi::TextureHandle{depth_tex_id_};
        dd.motion_in     = pt::rhi::TextureHandle{motion_tex_id_};
        dd.normal_in     = pt::rhi::TextureHandle{normal_tex_id_};
        // OptiX AOV + MetalFX kinds populate this; everything else has
        // albedo_tex_id_ == 0 and the backend's nil handle path
        // tolerates it. MetalFX uses the diffuse albedo as a spatial-
        // filter guidance signal.
        dd.albedo_in     = pt::rhi::TextureHandle{albedo_tex_id_};
        // MetalFX specular-guidance G-buffers (issue #118). Engine
        // allocates these only for MetalFX-family kinds; for all
        // other denoiser modes the IDs are 0 and the backend treats
        // them as "no guidance for this frame" (matching the existing
        // nil-handle convention used for albedo_in on SVGF/NRD).
        dd.specular_albedo_in       = pt::rhi::TextureHandle{specular_albedo_tex_id_};
        dd.roughness_in             = pt::rhi::TextureHandle{roughness_tex_id_};
        dd.specular_hit_distance_in = pt::rhi::TextureHandle{specular_hit_distance_tex_id_};
        // Star-split accumulator was wired here for the EMA design
        // (#108). The stateless StarsComposite rewrite eliminates the
        // accumulator; the Vulkan path's denoiser finalize no longer
        // needs a stars binding. Stars on the Vulkan denoiser route
        // are a follow-up -- the StarsComposite kernel now builds and
        // registers on Vulkan via this PR (shared 14-slot descriptor
        // layout via kSlotToTexBinding[] reuse + a 112B push + 112B
        // Frame UBO tail at binding(14, 0)), but the dispatch site
        // currently lives inside the Metal-only use_engine_tonemap
        // branch. A dedicated Vulkan dispatch path (engine-side
        // post-Denoise hook, or composite-inside-VulkanNrdDenoiser
        // between atrous and finalize) is the missing piece. Until
        // then the metal-only guard on engine_composite_active above
        // keeps push.composite_celestials = 0 on Vulkan so PathTrace
        // doesn't peel celestials the composite can't add back.
        // Leaving dd.stars_in zero tells the backend the slot is
        // unused; DenoiseFinalize.slang has correspondingly dropped
        // its stars_tex declaration in this PR.
        // MetalFX writes to the linear-HDR intermediate; the tonemap
        // dispatch below converts that to sRGB and writes the swapchain.
        dd.output        = pt::rhi::TextureHandle{post_denoise_hdr_tex_id_};
        // Vulkan SVGF/NRD finalize: reads `output` (linear HDR) and
        // writes a tonemapped LDR result directly into the swapchain.
        // ONLY when the engine's separate Tonemap.slang pipeline isn't
        // built OR we can't safely chain it (OptiX, see above) -- when
        // the engine WILL tonemap (Metal always for SVGF/MetalFX,
        // Vulkan for SVGF/NRD via the push/UBO split), we tell the
        // SVGF/NRD finalize step to skip its own swapchain write by
        // passing final_output = 0. MetalFX doesn't have an internal
        // finalize step so it ignores this field either way.
        dd.final_output    = use_engine_tonemap
                                 ? pt::rhi::TextureHandle{0}
                                 : fc.swapchain_image;
        dd.exposure_state  = pt::rhi::BufferHandle{exposure_state_id_};
        dd.jitter_x      = last_jitter_x_;
        dd.jitter_y      = last_jitter_y_;
        dd.reset_history = !prev_view_proj_valid_;
        // Map the engine's denoiser kind to the RHI quality tier. Nrd
        // and the various Atrous variants all run the full a-trous
        // chain; only SvgfBasic / SvgfBasicMetalFx skip the spatial
        // filter. MetalFX ignores the quality field entirely.
        const bool quality_is_basic =
            (denoiser_kind_ == DenoiserKind::SvgfBasic ||
             denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx);
        dd.quality = quality_is_basic
                         ? pt::rhi::Device::DenoiseDesc::Quality::Basic
                         : pt::rhi::Device::DenoiseDesc::Quality::Atrous;
        // A-Trous pass count for the in-house SVGF chain. Cvar-driven
        // so the user can A/B 1..5 at runtime; clamped to [1,5] here
        // so a typo in the console can't push us past what the backend
        // supports (steps 1, 2, 4, 8, 16 -- 5 is canonical SVGF).
        std::uint32_t atrous_passes_want = 1;
        if (auto* v = C.FindCVar("r_svgf_atrous_passes")) {
            int n = v->GetInt();
            if (n < 1) n = 1;
            if (n > 5) n = 5;
            atrous_passes_want = static_cast<std::uint32_t>(n);
        }
        dd.atrous_passes = atrous_passes_want;
        // Issue #119 -- SVGF albedo demodulation. Cvar-driven; the
        // backends ignore this for non-SVGF kinds and treat it as
        // false when albedo_in_ wasn't allocated (so a transient
        // resize race that lands an SVGF dispatch one frame before
        // the albedo texture is ready degrades cleanly).
        //
        // Toggling r_svgf_albedo_demod mid-run swaps the color space
        // the SVGF history texture stores (demod-on = lighting only,
        // demod-off = full radiance), so we have to fire reset_history
        // when the EFFECTIVE flag changes between frames -- otherwise
        // the temporal blend lerps the new frame against stale history
        // in the wrong colour space for up to the temporal window.
        // The effective flag is the cvar AND-ed with albedo availability
        // (the backends do the AND themselves, but we mirror it here so
        // a resize race that drops the albedo binding for one frame
        // doesn't fire a spurious reset). PR #150 Copilot review.
        {
            bool demod_on = true;       // matches CVAR default
            if (auto* v = C.FindCVar("r_svgf_albedo_demod")) {
                demod_on = v->GetBool();
            }
            dd.albedo_demod_enabled = demod_on;
            const bool effective_demod = demod_on && (albedo_tex_id_ != 0);
            if (effective_demod != prev_svgf_albedo_demod_active_) {
                dd.reset_history = true;
                LOG_INFO("engine: SVGF albedo demod toggled ({} -> {}); "
                         "firing reset_history to clear stale {} space",
                         prev_svgf_albedo_demod_active_ ? "on" : "off",
                         effective_demod ? "on" : "off",
                         prev_svgf_albedo_demod_active_ ? "demod" : "radiance");
            }
            prev_svgf_albedo_demod_active_ = effective_demod;
        }

        // Top-level Kind: tells the backend which denoiser
        // implementation to dispatch.
        //   Vulkan: VulkanNrdDenoiser (Svgf) vs VulkanOptixDenoiser
        //           (OptixHdr / OptixHdrAov / OptixTemporalHdr /
        //           OptixTemporalHdrAov).
        //   Metal:  MetalSvgfDenoiser (Svgf) vs MetalFX
        //           (MTLFXTemporalDenoisedScaler).
        // The svgf_basic / svgf_atrous / nrd values all collapse to
        // Kind::Svgf -- they're tiers within the same backend impl
        // and the Quality field below selects between basic (temporal
        // only) and atrous (temporal + 3-pass a-trous).
        if (denoiser_kind_ == DenoiserKind::OptixHdr) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::OptixHdr;
        } else if (denoiser_kind_ == DenoiserKind::OptixTemporalHdr) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::OptixTemporalHdr;
        } else if (denoiser_kind_ == DenoiserKind::OptixTemporalHdrAov) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::OptixTemporalHdrAov;
        } else if (denoiser_kind_ == DenoiserKind::OptixHdrAov) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::OptixHdrAov;
        } else if (denoiser_kind_ == DenoiserKind::MetalFX) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::MetalFX;
        } else if (denoiser_kind_ == DenoiserKind::SvgfBasicMetalFx ||
                   denoiser_kind_ == DenoiserKind::SvgfAtrousMetalFx) {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::SvgfMetalFx;
        } else {
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::Svgf;
        }
        // r_hdr_pipeline mirrors row0.w of the path-tracer push (set
        // earlier this frame) -- we read the cvar straight here so
        // the denoiser-finalize sRGB-only branch matches whichever
        // tonemap the path tracer applied to denoise_color. Default
        // (1) keeps tonemap+OETF in finalize; setting r_hdr_pipeline 0
        // skips the tonemap so the path tracer's pre-tonemapped
        // denoise_color isn't double-mapped.
        bool dd_hdr_pipeline = true;
        if (auto* v = C.FindCVar("r_hdr_pipeline")) dd_hdr_pipeline = v->GetBool();
        dd.hdr_pipeline = dd_hdr_pipeline;
        // MetalFX wants worldToView and viewToClip separately, so pass
        // them rather than the combined view*proj. glm matrices are
        // column-major, matching simd_float4x4 on the consumer end.
        dd.world_to_view = glm::value_ptr(view);
        dd.view_to_clip  = glm::value_ptr(proj);

        // Read bloom cvars up here so both the pre-Denoise (Vulkan)
        // and post-Denoise (Metal) bloom chains share them.
        bool  bloom_on        = false;
        float bloom_thresh    = 1.0f;
        float bloom_intensity = 0.05f;
        int   bloom_mips_in   = kBloomMips;
        if (auto* v = C.FindCVar("r_bloom"))           bloom_on        = v->GetBool();
        if (auto* v = C.FindCVar("r_bloom_threshold")) bloom_thresh    = v->GetFloat();
        if (auto* v = C.FindCVar("r_bloom_intensity")) bloom_intensity = v->GetFloat();
        if (auto* v = C.FindCVar("r_bloom_mips"))      bloom_mips_in   = v->GetInt();
        const int bloom_mips = std::clamp(bloom_mips_in, 1, kBloomMips);
        float bloom_radius   = 1.0f;
        if (auto* v = C.FindCVar("r_bloom_radius"))    bloom_radius    = v->GetFloat();

        // Bloom-pyramid dispatch helper. Reads from `source_tex_id`
        // (which differs per backend -- see call sites) and builds
        // bloom_mip[0..bloom_mips-1] via downsample (Karis-suppressed
        // on the first mip for path-tracer firefly resilience) then
        // upsample-add.
        auto dispatch_bloom_pyramid = [&](std::uint64_t source_tex_id) {
            PT_ZONE_SCOPED_N("Engine::dispatch_bloom_pyramid");
            // Downsample: source_tex -> mip0 (with threshold), then
            // mip0 -> mip1 -> ... -> mip[N-1]. Each step's read of
            // mip[i-1] is the previous step's write -- explicit
            // compute_write -> compute_read barriers thread the RAW
            // hazard chain so we don't race the prior dispatch's store.
            // Metal's MetalCommandBuffer auto-barriers on shared
            // resources so its Barrier() is a no-op; Vulkan needs the
            // explicit emit (same pattern as the PathTrace -> autoexpose
            // hand-off earlier in the frame).
            for (int i = 0; i < bloom_mips; ++i) {
                if (i > 0) {
                    cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                                 pt::rhi::BarrierDesc::Stage::ComputeRead});
                }
                cb->BindComputePipeline(pt::rhi::PipelineHandle{bloom_down_pipeline_id_});
                pt::rhi::TextureHandle src_h{
                    (i == 0) ? source_tex_id : bloom_mip_tex_id_[i - 1]};
                pt::rhi::TextureHandle dst_h{bloom_mip_tex_id_[i]};
                cb->BindStorageTexture(0, src_h);
                cb->BindStorageTexture(1, dst_h);
                struct DownPush { float threshold; float pad[3]; } dp{};
                // Phase-0 defensive guard: every push struct must be 16-byte
                // aligned to satisfy the std140 / MSL cbuffer rule the Slang
                // compiler applies on the GPU side. DownPush is naturally
                // 16B (1 float + 3 pad floats), but the assert catches a
                // future "drop the pad" edit that would silently shift the
                // shader's read offset. See the same pattern on PtPush
                // (Engine.cpp:~3002) where the regression actually bit us.
                static_assert(sizeof(DownPush) % 16 == 0,
                              "DownPush must be 16-byte aligned (cbuffer rule)");
                dp.threshold = (i == 0) ? bloom_thresh : 0.0f;
                cb->PushConstants(&dp, sizeof(dp));
                cb->Dispatch((bloom_mip_w_[i] + 7) / 8,
                             (bloom_mip_h_[i] + 7) / 8, 1);
            }
            // Upsample: mip[N-1] -> mip[N-2] (additive), ..., mip 1 -> mip 0.
            // `dst[tid] = prev + tent` is read-modify-write so we also
            // need the barrier between successive ups, plus one at the
            // end so the consumer (DenoiseFinalize / Tonemap) sees the
            // final mip[0] write.
            for (int i = bloom_mips - 1; i > 0; --i) {
                cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                             pt::rhi::BarrierDesc::Stage::ComputeRead});
                cb->BindComputePipeline(pt::rhi::PipelineHandle{bloom_up_pipeline_id_});
                cb->BindStorageTexture(0, pt::rhi::TextureHandle{bloom_mip_tex_id_[i]});
                cb->BindStorageTexture(1, pt::rhi::TextureHandle{bloom_mip_tex_id_[i - 1]});
                struct UpPush { float radius; float pad[3]; } up_push{};
                // Phase-0 defensive guard -- see DownPush above for rationale.
                static_assert(sizeof(UpPush) % 16 == 0,
                              "UpPush must be 16-byte aligned (cbuffer rule)");
                up_push.radius = bloom_radius;
                cb->PushConstants(&up_push, sizeof(up_push));
                cb->Dispatch((bloom_mip_w_[i - 1] + 7) / 8,
                             (bloom_mip_h_[i - 1] + 7) / 8, 1);
            }
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        };
        const bool bloom_can_run =
            (bloom_on && bloom_intensity > 0.0f &&
             bloom_mip_tex_id_[0] != 0 &&
             bloom_down_pipeline_id_ != 0 && bloom_up_pipeline_id_ != 0);

        // Vulkan + SVGF/NRD active: build a pre-Denoise bloom pyramid
        // from denoise_color (PathTrace's raw HDR write) and pass
        // bloom_mip[0] through Denoise() so VulkanNrdDenoiser::Encode
        // composites bloom inside its internal DenoiseFinalize step.
        // Pre-denoise pyramid source carries some path-tracer noise
        // but bloom is a low-frequency feature so it's acceptable; a
        // post-denoise pyramid would require splitting Encode() into
        // stages-only + a separate FinalizeOnly call (deferred).
        //
        // History: PR #11 originally disabled this path because the
        // shared 19-binding pipeline-layout's storage-image bindings
        // 0/1 (which bloom_down re-binds to mip views) interleaved
        // with the dispatches that bracketed it -- the swap rendered
        // only the upper-left ~quarter with the rest of the image
        // showing an earlier pass. Root cause was UPDATE_AFTER_BIND
        // descriptors: every dispatch read the LATEST host write at
        // execution time, so PathTrace's `output` binding observed
        // bloom_mip[1]'s 320x180 view rebinding and the shader's
        // GetDimensions() returned (320, 180), killing all writes
        // outside that region. PR #18 fixed it with a per-dispatch
        // descriptor-set ring (VulkanDevice::AcquireDispatchDescriptorSet),
        // so each dispatch now sees the descriptors it was recorded
        // with. The Tonemap.slang black-screen bug is unrelated and
        // doesn't reach this path -- SVGF's internal DenoiseFinalize
        // uses VulkanNrdDenoiser's dedicated layout, not Tonemap.slang.
        //
        // Vulkan + bloom-without-denoiser (use_vulkan_bloom_finalize):
        // there's no SVGF dispatch in the pipeline; the engine
        // dispatches the bloom pyramid in-engine then calls
        // Denoise(Kind::FinalizeOnly) below, which routes to
        // VulkanNrdDenoiser::EncodeFinalizeOnly -- a self-contained
        // 4-binding pipeline that also sidesteps the Tonemap.slang
        // black-swapchain bug. (Tonemap.slang is broken on Vulkan
        // regardless of upstream denoiser state, contrary to the
        // earlier hypothesis that the bug was SVGF-specific.)
        //
        // OptiX is excluded here because its denoise runs on a
        // private cb gated by a CUDA->Vulkan timeline semaphore, so
        // the engine cb can't safely pass a "current frame" bloom
        // mip in -- Branch B (feature/vulkan-bloom-with-optix) deals
        // with that handshake separately.
        const bool vulkan_pre_denoise_bloom =
            (!backend_is_metal && denoiser_active_ && !kind_is_optix &&
             bloom_can_run);
        // Edge-detect log: fire once when the path engages and once
        // when it disengages. Both transitions matter operationally --
        // engagement tells the user bloom is now being composited
        // inside SVGF's finalize (so they know which dispatch to
        // capture if they want to debug); disengagement covers the
        // case where r_bloom 0 / r_denoiser none / a kind switch
        // turned it off, so a missing bloom isn't mistaken for the
        // SVGF path silently breaking.
        if (vulkan_pre_denoise_bloom && !vulkan_pre_denoise_bloom_engaged_) {
            LOG_INFO("engine: vulkan pre-denoise bloom engaged -- pyramid built from "
                     "denoise_color ({}x{} RGBA16F), bloom_mip[0] composited inside "
                     "VulkanNrdDenoiser::Encode's DenoiseFinalize",
                     fc.width, fc.height);
            vulkan_pre_denoise_bloom_engaged_ = true;
        } else if (!vulkan_pre_denoise_bloom && vulkan_pre_denoise_bloom_engaged_) {
            LOG_INFO("engine: vulkan pre-denoise bloom disengaged");
            vulkan_pre_denoise_bloom_engaged_ = false;
        }
        dd.bloom_in        = vulkan_pre_denoise_bloom
                                 ? pt::rhi::TextureHandle{bloom_mip_tex_id_[0]}
                                 : pt::rhi::TextureHandle{0};
        dd.bloom_intensity = vulkan_pre_denoise_bloom ? bloom_intensity : 0.0f;

        // Vulkan + OptiX denoiser + r_bloom: build the bloom pyramid
        // here in the engine cb from the path tracer's noisy
        // denoise_color (Karis-suppressed first downsample shrugs off
        // the path-tracer fireflies), and feed mip[0] to the OptiX
        // path's deferred finalize. The pyramid then flows to
        // VulkanOptixDenoiser::Encode via dd.bloom_in /
        // dd.bloom_intensity, where it's wired into the private cb's
        // EncodeDenoiseFinalize call -- which composites mip[0]
        // pre-ACES and writes the swap on the same dedicated
        // 4-binding layout that PR #18's bloom-without-denoiser path
        // uses, sidestepping the SVGF-active descriptor reuse bug.
        //
        // Order: dispatch_bloom_pyramid runs BEFORE device_->Denoise(dd)
        // so the pyramid's denoise_color reads happen before the OptiX
        // input copy reads it (both are reads -- no hazard, but logical
        // order is "pyramid then input copy"). The cross-queue-submit
        // boundary between this engine cb and the OptiX private cb
        // (vkQueueWaitIdle inside SubmitPostMain) covers the
        // bloom-up-write -> finalize-read synchronisation; no
        // additional barrier needed past dispatch_bloom_pyramid's
        // internal compute_write -> compute_read chain.
        const bool use_vulkan_optix_bloom =
            (!backend_is_metal && bloom_can_run && kind_is_optix &&
             device_->SupportsDenoise());
        if (use_vulkan_optix_bloom) {
            // RAW: PathTrace wrote denoise_color and bloom_down is
            // about to read it. Same global barrier the
            // use_vulkan_bloom_finalize branch below issues for the
            // identical hazard.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
            dispatch_bloom_pyramid(denoise_color_tex_id_);
            dd.bloom_in        = pt::rhi::TextureHandle{bloom_mip_tex_id_[0]};
            dd.bloom_intensity = bloom_intensity;
        }
        // Edge-detect log for the Vulkan + OptiX bloom path. Mirrors
        // the SVGF edge-detect above: fire once when the path engages
        // and once when it disengages. Both transitions matter for
        // diagnostics -- engagement tells the user bloom is now being
        // composited inside the OptiX private-cb finalize (so e.g.
        // `screenshot foo swap` will pick it up); disengagement
        // distinguishes "bloom turned off" from "bloom path silently
        // broke" if the next screenshot comes back flat. Replaces the
        // earlier function-static one-shot which only logged engagement
        // once per process lifetime, leaving subsequent r_bloom toggles
        // invisible in the log stream.
        if (use_vulkan_optix_bloom && !vulkan_optix_bloom_engaged_) {
            LOG_INFO("engine: bloom-with-OptiX engaged -- pyramid built from "
                     "PathTrace denoise_color ({}x{}), mip[0] composited in "
                     "OptiX private-cb finalize",
                     fc.width, fc.height);
            vulkan_optix_bloom_engaged_ = true;
        } else if (!use_vulkan_optix_bloom && vulkan_optix_bloom_engaged_) {
            LOG_INFO("engine: bloom-with-OptiX disengaged");
            vulkan_optix_bloom_engaged_ = false;
        }

        // Vulkan + SVGF/NRD bloom: build the pyramid from denoise_color
        // BEFORE Denoise(). VulkanNrdDenoiser::Encode samples
        // bloom_mip[0] in its internal DenoiseFinalize step, so the
        // pyramid must be a finished resource by the time SVGF runs.
        // PathTrace wrote denoise_color earlier this frame; emit the
        // RAW barrier so bloom_down's read isn't racing the store.
        // (Autoexpose upstream emits the same barrier when r_auto_exposure
        // is on -- this is unconditional cover for the off case, matching
        // the use_vulkan_bloom_finalize branch below.)
        //
        // Mutually exclusive with the OptiX block above: vulkan_pre_denoise_bloom
        // requires !kind_is_optix, use_vulkan_optix_bloom requires kind_is_optix.
        // At most one of these dispatches runs per frame.
        if (vulkan_pre_denoise_bloom) {
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
            dispatch_bloom_pyramid(denoise_color_tex_id_);
        }

        // StarsComposite (issue #46) dispatch helper -- factored out so
        // the Metal use_engine_tonemap path (further down) and the
        // Vulkan dual-Denoise path (right below) can share the same
        // bindings + push layout without duplicating the dispatch code.
        // Performs the stateless celestial composite over `target_tex_id`
        // in place. Reads no buffer bindings (StarsComposite kernel is
        // exposure-free). The dispatch is gated on the same conditions
        // as engine_composite_active above PLUS moon_map / cloud_trans
        // availability (always non-zero in practice -- procedurally
        // generated at engine init -- but belt-and-braces against init
        // races). star_map_tex_id_ falls back to bloom_dummy_tex_id_
        // when BSC isn't loaded; the shader's procedural-stars path
        // doesn't sample star_map so the placeholder is never read,
        // and gating on star_map_tex_id_ != 0 would lose sun+moon+
        // procedural-stars in that case.
        auto composite_stars_to = [&](std::uint64_t target_tex_id) {
            const std::uint64_t composite_star_map_id =
                (star_map_tex_id_ != 0) ? star_map_tex_id_ : bloom_dummy_tex_id_;
            if (push.composite_celestials == 0u ||
                stars_composite_pipeline_id_ == 0 ||
                moon_map_tex_id_ == 0 ||
                depth_tex_id_ == 0 ||
                cloud_trans_tex_id_ == 0 ||
                composite_star_map_id == 0) {
                return;
            }
            cb->BindComputePipeline(pt::rhi::PipelineHandle{stars_composite_pipeline_id_});
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{target_tex_id});
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{composite_star_map_id});
            cb->BindStorageTexture(2, pt::rhi::TextureHandle{moon_map_tex_id_});
            cb->BindStorageTexture(3, pt::rhi::TextureHandle{depth_tex_id_});
            cb->BindStorageTexture(4, pt::rhi::TextureHandle{cloud_trans_tex_id_});

            struct StarsCompositePush {
                float        pos_fovtan[4];
                float        fwd_aspect[4];
                float        right_xyz[4];
                float        up_xyz[4];
                float        sun_and_mode[4];
                float        exposure_pad[4];
                float        w2j_row0[4];
                float        w2j_row1[4];
                float        w2j_row2[4];
                float        moon_dir_phase[4];
                float        moon_extra[4];
                float        sun_extra[4];
                float        dof_params[4];
                std::uint32_t frame_index;
                std::uint32_t ap_samples;
                std::uint32_t composite_active;
                std::uint32_t _pad0;
            } sc{};
            static_assert(sizeof(StarsCompositePush) == 224,
                          "StarsCompositePush layout must match StarsComposite.slang");
            std::memcpy(sc.pos_fovtan,     push.pos_fovtan,     sizeof(sc.pos_fovtan));
            std::memcpy(sc.fwd_aspect,     push.fwd_aspect,     sizeof(sc.fwd_aspect));
            std::memcpy(sc.right_xyz,      push.right_xyz,      sizeof(sc.right_xyz));
            std::memcpy(sc.up_xyz,         push.up_xyz,         sizeof(sc.up_xyz));
            std::memcpy(sc.sun_and_mode,   push.sun_and_mode,   sizeof(sc.sun_and_mode));
            std::memcpy(sc.exposure_pad,   push.exposure_pad,   sizeof(sc.exposure_pad));
            std::memcpy(sc.w2j_row0,       push.w2j_row0,       sizeof(sc.w2j_row0));
            std::memcpy(sc.w2j_row1,       push.w2j_row1,       sizeof(sc.w2j_row1));
            std::memcpy(sc.w2j_row2,       push.w2j_row2,       sizeof(sc.w2j_row2));
            std::memcpy(sc.moon_dir_phase, push.moon_dir_phase, sizeof(sc.moon_dir_phase));
            std::memcpy(sc.moon_extra,     push.moon_extra,     sizeof(sc.moon_extra));
            std::memcpy(sc.sun_extra,      push.sun_extra,      sizeof(sc.sun_extra));
            std::memcpy(sc.dof_params,     push.dof_params,     sizeof(sc.dof_params));
            sc.frame_index      = push.frame_index;
            sc.composite_active = 1u;
            int ap_n = 16;
            if (auto* v = C.FindCVar("r_stars_aperture_samples")) {
                ap_n = v->GetInt();
            }
            if (ap_n < 1)  ap_n = 1;
            if (ap_n > 64) ap_n = 64;
            sc.ap_samples = static_cast<std::uint32_t>(ap_n);
            sc._pad0 = 0u;
            cb->PushConstants(&sc, sizeof(sc));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            // RAW: the downstream consumer (bloom_down on Metal,
            // FinalizeOnly's bloom + ACES read on Vulkan) is about to
            // read the HDR we just wrote.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        };

        // Vulkan dual-Denoise dispatch path (issue #46). On Vulkan the
        // engine-side Tonemap.slang produces a black swapchain (the
        // descriptor-visibility bug documented in the use_engine_tonemap
        // comment block above), so we can't run the Metal-style
        // sequence of "SVGF -> StarsComposite -> Tonemap -> swap".
        // Instead split the single Denoise(Svgf) call into:
        //   1) Denoise(SvgfNoFinalize) -- runs the SVGF / NRD chain,
        //      leaves the denoised linear-HDR in dd.output
        //      (post_denoise_hdr_tex_id_), and skips the internal
        //      DenoiseFinalize dispatch.
        //   2) composite_stars_to(post_denoise_hdr_tex_id_) -- reads +
        //      writes the same texture, additively layering sun / moon /
        //      stars with aperture-sampled bokeh.
        //   3) Denoise(FinalizeOnly) -- reads post_denoise_hdr_tex_id_,
        //      composites bloom_mip[0] pre-ACES, applies sRGB OETF,
        //      writes the swap. Reuses VulkanNrdDenoiser::
        //      EncodeFinalizeOnly's dedicated 4-binding layout that the
        //      bloom-without-denoiser path already exercises, so this
        //      sidesteps the Tonemap.slang issue entirely.
        // Gated on !kind_is_optix because the OptiX path runs on a
        // private cb behind a CUDA-Vulkan timeline semaphore -- the
        // engine cb can't safely interleave a StarsComposite dispatch
        // between OptiX's input copy and its deferred finalize. OptiX +
        // StarsComposite is its own follow-up if a user needs it.
        const bool vulkan_dual_denoise =
            (!backend_is_metal && denoiser_active_ &&
             engine_composite_active && !kind_is_optix &&
             device_->SupportsDenoise() &&
             post_denoise_hdr_tex_id_ != 0);
        if (vulkan_dual_denoise && !vulkan_dual_denoise_engaged_) {
            LOG_INFO("engine: vulkan dual-Denoise + StarsComposite engaged "
                     "-- SvgfNoFinalize -> composite -> FinalizeOnly "
                     "(post_denoise_hdr id={}, swap={}x{})",
                     post_denoise_hdr_tex_id_, fc.width, fc.height);
            vulkan_dual_denoise_engaged_ = true;
        } else if (!vulkan_dual_denoise && vulkan_dual_denoise_engaged_) {
            LOG_INFO("engine: vulkan dual-Denoise + StarsComposite disengaged");
            vulkan_dual_denoise_engaged_ = false;
        }

        // Skip the denoiser dispatch in the bloom-without-denoiser
        // path (Metal or Vulkan) -- no denoiser is active, so `dd`
        // above was just unused setup work. The downstream branches
        // (use_vulkan_bloom_finalize on Vulkan, use_engine_tonemap on
        // Metal) still write the swapchain from denoise_color.
        if (denoiser_active_) {
            PT_ZONE_SCOPED_N("Device::Denoise");
            if (vulkan_dual_denoise) {
                // Step 1: SVGF chain that stops before the internal
                // DenoiseFinalize dispatch. VulkanDevice::Denoise reads
                // dd.kind and (for SvgfNoFinalize) routes through
                // VulkanNrdDenoiser::Encode with final_output=0, which
                // leaves the denoised linear-HDR in dd.output.
                dd.kind = pt::rhi::Device::DenoiseDesc::Kind::SvgfNoFinalize;
                device_->Denoise(dd);
                // Step 2: stateless celestial composite over the same
                // texture. composite_stars_to issues its own RAW barrier
                // before returning, so the FinalizeOnly read is covered.
                composite_stars_to(post_denoise_hdr_tex_id_);
                // Step 3: bloom composite + ACES + sRGB OETF + swap write.
                // Reuse the same `dd` but swap kind, color_in, and
                // final_output. bloom_in / bloom_intensity /
                // exposure_state / hdr_pipeline were set above by the
                // Svgf-path setup and apply to FinalizeOnly identically.
                dd.kind         = pt::rhi::Device::DenoiseDesc::Kind::FinalizeOnly;
                dd.color_in     = pt::rhi::TextureHandle{post_denoise_hdr_tex_id_};
                dd.final_output = fc.swapchain_image;
                device_->Denoise(dd);
            } else {
                device_->Denoise(dd);
            }
        }

        // Vulkan bloom-without-denoiser: build the bloom pyramid from
        // PathTrace's denoise_color, then call Denoise(FinalizeOnly)
        // to run VulkanNrdDenoiser::EncodeFinalizeOnly. That kernel
        // composites bloom_mip[0] pre-ACES, applies the sRGB OETF, and
        // writes the swapchain through a dedicated 4-binding pipeline
        // layout that the Tonemap.slang black-screen bug doesn't affect.
        if (use_vulkan_bloom_finalize) {
            // RAW hazard: PathTrace wrote denoise_color and bloom_down
            // is about to read it. The autoexpose dispatch upstream
            // emits the same global barrier when r_auto_exposure is
            // on; emit it unconditionally here to cover auto-exposure-
            // off too. (No-op on Metal -- but Metal never enters this
            // branch.)
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
            if (bloom_can_run) {
                dispatch_bloom_pyramid(denoise_color_tex_id_);
            }
            // Repurpose dd for the FinalizeOnly route. The
            // depth/motion/normal/albedo/output/quality/kind fields
            // are unused on this path -- VulkanDevice::Denoise reads
            // only color_in / final_output / bloom_in / bloom_intensity
            // / exposure_state / hdr_pipeline when kind == FinalizeOnly.
            dd.kind = pt::rhi::Device::DenoiseDesc::Kind::FinalizeOnly;
            dd.color_in     = pt::rhi::TextureHandle{denoise_color_tex_id_};
            dd.final_output = fc.swapchain_image;
            dd.bloom_in     = (bloom_can_run && bloom_mip_tex_id_[0] != 0)
                                  ? pt::rhi::TextureHandle{bloom_mip_tex_id_[0]}
                                  : pt::rhi::TextureHandle{0};
            dd.bloom_intensity = bloom_can_run ? bloom_intensity : 0.0f;
            PT_ZONE_SCOPED_N("Device::Denoise(FinalizeOnly)");
            device_->Denoise(dd);
        }

        // Run the engine's bloom + tonemap chain only when we've told
        // the denoiser path to defer the swapchain write to us (see
        // use_engine_tonemap above). When tonemap isn't built (asset
        // pipeline failure) or the active denoiser is OptiX (deferred
        // private cb -- engine cb would tonemap STALE post_denoise_hdr),
        // Denoise()'s own finalize wrote the swapchain and the
        // dispatches below would just no-op or worse, race the OptiX
        // private cb on the swapchain image.
        if (use_engine_tonemap) {
        // Bloom pyramid: extract bright HDR pixels into bloom_mip[0]
        // (with luminance threshold), then progressively downsample
        // through the chain, then upsample additively back up to mip
        // 0. The result in mip 0 is sampled by the tonemap kernel
        // and added pre-ACES so the bloom gets the same curve squash
        // as the rest of the image.
        //
        // Metal-only here (use_engine_tonemap is Metal-gated):
        //   - Metal + denoiser active        -> post_denoise_hdr (SVGF/MetalFX output)
        //   - Metal + bloom-without-denoiser -> denoise_color    (PathTrace direct write)
        // Vulkan never enters this branch: bloom-without-denoiser
        // is handled by the `use_vulkan_bloom_finalize` branch above,
        // and denoiser-active Vulkan stays on VulkanNrdDenoiser's
        // internal finalize.
        const std::uint64_t tonemap_hdr_source_id =
            denoiser_active_ ? post_denoise_hdr_tex_id_ : denoise_color_tex_id_;
        // Metal's MetalCommandBuffer auto-barriers on shared
        // resources between dispatches, so the explicit emit below
        // is a no-op on the only backend that reaches this branch.
        // Left in for documentation symmetry with the
        // use_vulkan_bloom_finalize branch above (which DOES need
        // the explicit barrier).
        cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                     pt::rhi::BarrierDesc::Stage::ComputeRead});

        // SIGMA-style sun-shadow demodulation (issue #115). Dispatched
        // BEFORE StarsComposite so the shadow demod multiplies into
        // the post-radiance-denoise HDR before celestials are added on
        // top (celestials should stay at full brightness; they're not
        // subject to direct-sun shadowing the same way the scene
        // surfaces are). The kernel reads the noisy per-pixel
        // shadow_vis_buf that PathTrace's primary-hit pass wrote,
        // bilateral-filters it with a 5x5 depth+normal kernel, and
        // multiplies the denoised visibility into tonemap_hdr_source_id
        // in place. Gated on push.write_shadow_vis so we only run when
        // PathTrace was actually populating the buffer -- otherwise
        // we'd multiply by uninitialised memory.
        //
        // Dispatch gate mirrors push.write_shadow_vis prerequisites
        // (denoiser_active_, shadow_vis_buf_id_, depth_tex_id_,
        // sigma_shadow_pipeline_id_) AND adds normal_tex_id_ as a soft
        // requirement -- the shader has a use_normal=0 fallback path
        // (depth-only edge stops, slightly lower selectivity) for the
        // rare denoiser mode that doesn't allocate normals.
        if (push.write_shadow_vis != 0u &&
            sigma_shadow_pipeline_id_ != 0 &&
            shadow_vis_buf_id_ != 0 &&
            depth_tex_id_ != 0) {
            cb->BindComputePipeline(pt::rhi::PipelineHandle{sigma_shadow_pipeline_id_});
            // SigmaShadow texture bindings in declaration order:
            //   slot 0 hdr_inout       -- post-radiance-denoise HDR
            //   slot 1 depth_tex       -- camera-space Z (sky gate +
            //                              depth edge stop)
            //   slot 2 normal_tex      -- world-space normal (normal
            //                              edge stop; only meaningful
            //                              when normal_tex_id_ != 0
            //                              AND use_normal=1 in push)
            // Buffer:
            //   slot 3 shadow_vis_buf  -- R32F per-pixel visibility
            //                              written by PathTrace
            //
            // When normal_tex_id_ is 0 we still need slot 2 bound to
            // SOMETHING so Metal's pipeline-validation doesn't trip
            // on a missing storage texture. depth_tex is a different
            // format (R32F vs RGBA16F), so bloom_dummy_tex_id_ (1x1
            // RGBA16F, always allocated alongside the denoiser
            // textures) is the natural placeholder -- the shader's
            // use_normal=0 path doesn't read it, but Metal validates
            // the binding regardless of usage.
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{tonemap_hdr_source_id});
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{depth_tex_id_});
            const std::uint64_t sigma_normal_id =
                (normal_tex_id_ != 0) ? normal_tex_id_ : bloom_dummy_tex_id_;
            cb->BindStorageTexture(2, pt::rhi::TextureHandle{sigma_normal_id});
            // shadow_vis_buf lands at MSL buffer(0) -- Slang assigns
            // buffer slots in declaration order starting at 0, and
            // SigmaShadow.slang declares no other buffers before Push.
            // Texture slots are a separate namespace from buffer slots
            // on Metal, so binding the storage buffer at engine slot 0
            // here doesn't clash with the hdr_inout texture binding at
            // engine slot 0 above. Push lands at buffer(1) (max-bound +
            // 1 = 0 + 1) and MetalCommandBuffer::Dispatch computes that
            // automatically.
            cb->BindBuffer(0, pt::rhi::BufferHandle{shadow_vis_buf_id_}, 0);

            struct SigmaShadowPush {
                std::uint32_t width;
                std::uint32_t height;
                std::uint32_t demod_active;
                std::uint32_t _pad0;
                float         sigma_depth;
                float         sigma_normal;
                std::uint32_t use_normal;
                std::uint32_t _pad1;
            } sp{};
            static_assert(sizeof(SigmaShadowPush) == 32,
                          "SigmaShadowPush layout must match SigmaShadow.slang");
            sp.width        = fc.width;
            sp.height       = fc.height;
            sp.demod_active = 1u;
            sp._pad0        = 0u;
            // Bilateral sigmas: depth as a fraction of local Z (5%
            // works on the cornell-style fixtures the issue describes),
            // normal as an exponent on the cosine of normal-difference
            // (16 = ~30 deg tolerance for "still considered co-planar").
            // Not yet cvar-tunable; if shadow widths look off on real
            // scenes we'll plumb r_shadow_sigma_depth / _normal as a
            // follow-up.
            sp.sigma_depth  = 0.05f;
            sp.sigma_normal = 16.0f;
            sp.use_normal   = (normal_tex_id_ != 0) ? 1u : 0u;
            sp._pad1        = 0u;
            cb->PushConstants(&sp, sizeof(sp));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            // RAW: StarsComposite about to read+write the HDR we just
            // wrote. Metal auto-barriers; emitted for documentation
            // symmetry with the rest of the post-denoise chain.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        }

        // Stateless stars+sun+moon composite (issue #46). Metal-side
        // call site -- dispatched BEFORE the bloom pyramid so the bloom
        // downsample picks up the celestial highlights and produces
        // real halos, and BEFORE Tonemap so the ACES curve squashes the
        // combined image as one piece. tonemap_hdr_source_id is
        // post_denoise_hdr when the denoiser is active, denoise_color
        // otherwise; either way it's the HDR target the engine Tonemap
        // chain will read next. The composite kernel reads + writes
        // this texture in place (additive); PathTrace.slang's primary-
        // miss subtraction (push.composite_celestials) already ensured
        // it's celestial-free coming in. composite_stars_to internally
        // gates on push.composite_celestials so calling it on the
        // non-composite-active frame is a cheap no-op. The Vulkan
        // call site is in the dual-Denoise branch above (between
        // SvgfNoFinalize output and FinalizeOnly swap write).
        composite_stars_to(tonemap_hdr_source_id);

        // Aurora borealis procedural overlay (issue #116). Dispatched
        // AFTER StarsComposite so aurora ribbons layer over the
        // celestial layer in HDR space, and BEFORE the bloom pyramid
        // so bright aurora highlights get bloom halos and ACES
        // tonemap squashes them with the rest of the image. The
        // shader is stateless, reads + writes tonemap_hdr_source_id
        // in place (additive), and only needs hdr_inout + depth_tex.
        //
        // Dispatch gate: aurora_composite_pipeline_id_ is Metal-only
        // (Vulkan stays at 0), depth_tex_id_ must exist (allocated
        // whenever denoiser_active_, which use_engine_tonemap implies
        // on Metal). The cvar gate r_aurora is encoded as
        // aurora_active below; when off we elide the dispatch entirely
        // so r_aurora 0 is a true no-op.
        bool aurora_active = false;
        {
            int aurora_on = 0;
            if (auto* v = C.FindCVar("r_aurora")) aurora_on = v->GetInt();
            int aurora_sky_mode = 2;
            if (auto* v = C.FindCVar("r_sky_mode")) {
                const std::string& m = v->value;
                aurora_sky_mode = (m == "procedural") ? 2
                                : (m == "hdri")       ? 1
                                                      : 0;
            }
            aurora_active =
                aurora_on != 0 &&
                (aurora_sky_mode == 2) &&
                aurora_composite_pipeline_id_ != 0 &&
                depth_tex_id_ != 0 &&
                cloud_trans_tex_id_ != 0;
        }
        if (aurora_active) {
            cb->BindComputePipeline(pt::rhi::PipelineHandle{aurora_composite_pipeline_id_});
            cb->BindStorageTexture(0, pt::rhi::TextureHandle{tonemap_hdr_source_id});
            // depth_tex at engine slot 1 -> Metal MSL texture(1) by
            // declaration order. PathTrace's G-buffer pass writes
            // 1e10 for sky pixels; the shader's 5x5 min-dilation
            // gate at 1e9 lets us avoid painting aurora over geometry.
            cb->BindStorageTexture(1, pt::rhi::TextureHandle{depth_tex_id_});
            // cloud_trans_tex at engine slot 2 -> Metal MSL texture(2)
            // by declaration order. PathTrace's cloud march writes the
            // camera-ray transmittance here (1.0 = no occlusion); the
            // shader multiplies the aurora contribution by it so the
            // curtain gets darkened by foreground clouds the same way
            // StarsComposite darkens stars / sun / moon. Allocated
            // alongside depth_tex_id_ whenever denoiser_active_.
            cb->BindStorageTexture(2, pt::rhi::TextureHandle{cloud_trans_tex_id_});
            // No buffer bindings: AuroraComposite is purely
            // texture-driven. BindComputePipeline cleared any prior
            // PathTrace bindings, so max_bound_buf_slot = -1 and Slang's
            // MSL emission places Push at buf(0). Engine's
            // push_slot = max+1 = 0. Match.

            struct AuroraCompositePush {
                float        pos_fovtan[4];
                float        fwd_aspect[4];
                float        right_xyz[4];
                float        up_xyz[4];
                float        sun_and_mode[4];
                float        aurora_params[4];
                std::uint32_t frame_index;
                std::uint32_t composite_active;
                std::uint32_t _pad0;
                std::uint32_t _pad1;
            } ap{};
            static_assert(sizeof(AuroraCompositePush) == 112,
                          "AuroraCompositePush layout must match AuroraComposite.slang");
            std::memcpy(ap.pos_fovtan,   push.pos_fovtan,   sizeof(ap.pos_fovtan));
            std::memcpy(ap.fwd_aspect,   push.fwd_aspect,   sizeof(ap.fwd_aspect));
            std::memcpy(ap.right_xyz,    push.right_xyz,    sizeof(ap.right_xyz));
            std::memcpy(ap.up_xyz,       push.up_xyz,       sizeof(ap.up_xyz));
            std::memcpy(ap.sun_and_mode, push.sun_and_mode, sizeof(ap.sun_and_mode));

            // Pack aurora cvars. lat is in degrees user-side, radians
            // shader-side (cheaper than letting the shader pay for the
            // conversion every pixel). animate is the flag; time is in
            // seconds since startup, sourced from frame_index/60.
            float intensity = 1.0f;
            if (auto* v = C.FindCVar("r_aurora_intensity")) {
                intensity = v->GetFloat();
            }
            if (intensity < 0.0f) intensity = 0.0f;
            if (intensity > 2.0f) intensity = 2.0f;
            float lat_deg = 70.0f;
            if (auto* v = C.FindCVar("r_aurora_lat")) {
                lat_deg = v->GetFloat();
            }
            float lat_rad = lat_deg * 0.0174532925f;
            int animate_on = 1;
            if (auto* v = C.FindCVar("r_aurora_animate")) {
                animate_on = v->GetInt();
            }
            float time_sec = static_cast<float>(push.frame_index) * (1.0f / 60.0f);
            ap.aurora_params[0] = intensity;
            ap.aurora_params[1] = lat_rad;
            ap.aurora_params[2] = animate_on != 0 ? 1.0f : 0.0f;
            ap.aurora_params[3] = time_sec;
            ap.frame_index      = push.frame_index;
            ap.composite_active = 1u;
            ap._pad0 = 0u;
            ap._pad1 = 0u;
            cb->PushConstants(&ap, sizeof(ap));
            cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);

            // RAW: bloom_down about to read the HDR we just wrote.
            // Metal auto-barriers; emitted for documentation symmetry.
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeRead});
        }
        //
        // Runs in the same use_engine_tonemap branch as StarsComposite,
        // AFTER stars (so particles overlay celestials too) and BEFORE
        // the bloom pyramid (so HDR particle highlights downsample into
        // halos). Reads + writes the SAME tonemap_hdr_source_id in
        // place; the kernel does additive splats per particle.
        //
        // Gates:
        //   * r_particles == 1 (master toggle, default 0)
        //   * particle_composite_pipeline_id_ resolved (Metal only today)
        //   * particles_ live and > 0 active particles
        //   * depth_tex_id_ allocated (we need the depth gate); this
        //     implicitly requires denoiser_active_ because the engine
        //     only allocates depth_tex when a denoiser path is on.
        //     The bloom-without-denoiser branch has depth_tex_id_ = 0
        //     so the composite naturally short-circuits. Acceptable
        //     for the MVP -- particles + bloom-without-denoiser is a
        //     follow-up.
        //
        // The per-frame upload happens here too: rebuild a 3*float4
        // packed GPU layout from particles_->Particles(), grow the
        // storage buffer if needed, WriteBuffer the live region.
        bool r_particles_on = false;
        if (auto* v = C.FindCVar("r_particles")) r_particles_on = v->GetBool();
        const bool particle_gate =
            r_particles_on &&
            particle_composite_pipeline_id_ != 0 &&
            particles_ &&
            particles_->LiveCount() > 0 &&
            depth_tex_id_ != 0;
        if (particle_gate) {
            const auto& live = particles_->Particles();
            const std::uint32_t pcount =
                static_cast<std::uint32_t>(live.size());
            // One-shot info log per process: confirms the composite is
            // being dispatched against a non-zero particle count.
            // Without it the symptom of "wrong gate" looks identical to
            // "no particles painted" -- one log line saves a half-hour
            // of debug, and the one-shot pattern (static bool) matches
            // the project's other lazy-init/first-frame log idioms.
            static bool s_logged_once = false;
            if (!s_logged_once) {
                s_logged_once = true;
                LOG_INFO("particles: first composite dispatch, {} particle(s) live", pcount);
            }
            // GPU-side packed representation. Three float4 per particle,
            // 48 bytes -- matches the ParticleComposite.slang layout in
            // the kernel's load block.
            struct GpuParticle {
                float pos_size[4];     // .xyz = world pos (m), .w = size (m)
                float color_alpha[4];  // .xyz = HDR rgb, .w = alpha [0..1]
                float pad_zeye[4];     // .w = precomputed camera-space z
            };
            static_assert(sizeof(GpuParticle) == 48,
                          "GpuParticle must be 48 bytes to match ParticleComposite.slang");

            // (Re)allocate / grow the storage buffer if needed. Pow-2
            // growth keeps the reallocs amortised; the floor of 64
            // matches the threadgroup width so a single threadgroup
            // launch can always run cleanly.
            std::uint32_t need = pcount;
            if (need < 64u) need = 64u;
            if (need > particles_storage_capacity_) {
                std::uint32_t cap = (particles_storage_capacity_ > 0)
                                        ? particles_storage_capacity_ : 64u;
                while (cap < need) cap *= 2u;
                if (particles_storage_id_ != 0) {
                    device_->DestroyBuffer(pt::rhi::BufferHandle{particles_storage_id_});
                    particles_storage_id_ = 0;
                }
                auto buf = device_->CreateBuffer({
                    .size = sizeof(GpuParticle) * cap,
                    .usage = pt::rhi::BufferUsage::Storage,
                    .debug_name = "particles_storage",
                });
                particles_storage_id_       = buf.id;
                particles_storage_capacity_ = cap;
            }
            if (particles_storage_id_ != 0) {
                // Build the packed payload into the Engine-member
                // scratch buffer (12 floats per particle = one
                // GpuParticle). resize() keeps the underlying capacity
                // when shrinking, so steady-state load reuses the same
                // allocation frame-to-frame and we only pay the alloc
                // when the live count grows past the previous peak.
                // At the MVP cap of 1024 this is ~3 us of fill on M4
                // Max -- the savings are the avoided heap traffic.
                constexpr std::size_t kFloatsPerParticle =
                    sizeof(GpuParticle) / sizeof(float);
                static_assert(kFloatsPerParticle == 12,
                              "GpuParticle layout must be 12 floats");
                particle_upload_scratch_.resize(
                    static_cast<std::size_t>(pcount) * kFloatsPerParticle);
                auto* packed = reinterpret_cast<GpuParticle*>(
                    particle_upload_scratch_.data());
                const glm::vec3 cam_pos  = glm::vec3(push.pos_fovtan[0],
                                                    push.pos_fovtan[1],
                                                    push.pos_fovtan[2]);
                const glm::vec3 cam_fwd  = glm::vec3(push.fwd_aspect[0],
                                                    push.fwd_aspect[1],
                                                    push.fwd_aspect[2]);
                for (std::uint32_t i = 0; i < pcount; ++i) {
                    const auto& p = live[i];
                    packed[i].pos_size[0] = p.pos.x;
                    packed[i].pos_size[1] = p.pos.y;
                    packed[i].pos_size[2] = p.pos.z;
                    packed[i].pos_size[3] = p.size;
                    packed[i].color_alpha[0] = p.color.r;
                    packed[i].color_alpha[1] = p.color.g;
                    packed[i].color_alpha[2] = p.color.b;
                    packed[i].color_alpha[3] = p.color.a;
                    packed[i].pad_zeye[0] = 0.0f;
                    packed[i].pad_zeye[1] = 0.0f;
                    packed[i].pad_zeye[2] = 0.0f;
                    // z_eye = dot(p.pos - cam_pos, cam_fwd). cam_fwd is
                    // already unit length (Camera::Forward()).
                    packed[i].pad_zeye[3] = glm::dot(p.pos - cam_pos, cam_fwd);
                }
                device_->WriteBuffer(pt::rhi::BufferHandle{particles_storage_id_},
                                     packed,
                                     sizeof(GpuParticle) * pcount, 0);

                cb->BindComputePipeline(pt::rhi::PipelineHandle{particle_composite_pipeline_id_});
                cb->BindStorageTexture(0, pt::rhi::TextureHandle{tonemap_hdr_source_id});
                cb->BindStorageTexture(1, pt::rhi::TextureHandle{depth_tex_id_});
                cb->BindBuffer(0, pt::rhi::BufferHandle{particles_storage_id_});

                struct ParticleCompositePush {
                    float pos_fovtan[4];
                    float fwd_aspect[4];
                    float right_xyz[4];
                    float up_xyz[4];
                    std::uint32_t width;
                    std::uint32_t height;
                    std::uint32_t particle_count;
                    std::uint32_t composite_active;
                    std::uint32_t threads_x;
                    std::uint32_t _pad0;
                    std::uint32_t _pad1;
                    std::uint32_t _pad2;
                } pp{};
                static_assert(sizeof(ParticleCompositePush) == 96,
                              "ParticleCompositePush layout must match ParticleComposite.slang");
                std::memcpy(pp.pos_fovtan, push.pos_fovtan, sizeof(pp.pos_fovtan));
                std::memcpy(pp.fwd_aspect, push.fwd_aspect, sizeof(pp.fwd_aspect));
                std::memcpy(pp.right_xyz,  push.right_xyz,  sizeof(pp.right_xyz));
                std::memcpy(pp.up_xyz,     push.up_xyz,     sizeof(pp.up_xyz));
                pp.width            = fc.width;
                pp.height           = fc.height;
                pp.particle_count   = pcount;
                pp.composite_active = 1u;
                // Threadgroups along X; we keep Y = 1 (so Y has exactly
                // 8 threads). Each threadgroup = 64 threads = 64
                // particles. groups_x = ceil(pcount / 64).
                std::uint32_t groups_x = (pcount + 63u) / 64u;
                if (groups_x == 0u) groups_x = 1u;
                pp.threads_x = groups_x * 8u;
                cb->PushConstants(&pp, sizeof(pp));
                cb->Dispatch(groups_x, 1, 1);

                // RAW: bloom_down about to read the HDR we just wrote
                // (additively). Metal auto-barriers; emitted for
                // documentation symmetry with StarsComposite above.
                cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                             pt::rhi::BarrierDesc::Stage::ComputeRead});
            }
        }
        // ----- end Particle composite -----------------------------------

        if (bloom_can_run) {
            dispatch_bloom_pyramid(tonemap_hdr_source_id);
        }

        // Post-denoise tonemap: linear HDR -> exposure -> ACES -> sRGB
        // swapchain (gamma encode is implicit on store of the BGRA8_sRGB
        // surface).
        cb->BindComputePipeline(pt::rhi::PipelineHandle{tonemap_pipeline_id_});
        cb->BindStorageTexture(0, pt::rhi::TextureHandle{tonemap_hdr_source_id});
        cb->BindStorageTexture(1, fc.swapchain_image);
        // exposure_state is already bound at engine slot 6 from the
        // path-trace dispatch earlier this frame. Tonemap.slang has
        // been padded with dummy bindings so its MSL emission puts
        // exposure_state at MSL buffer(6) and Push at buffer(7) --
        // matching PathTrace's layout and the engine's universal
        // push_slot=max_bound_slot+1=7 contract. So we don't need
        // to rebind here; the inheritance from PathTrace's
        // BindBuffer(6, exposure_state) is exactly what Tonemap
        // expects on Metal. Vulkan's descriptor sets carry the
        // same vk::binding(15) population from PathTrace's set.
        // (Defensive: if PathTrace didn't run for some reason, the
        // explicit bind ensures it's there.)
        if (exposure_state_id_ != 0) {
            cb->BindBuffer(6, pt::rhi::BufferHandle{exposure_state_id_}, 0);
        }
        // Bind bloom mip 0 (built above) when bloom is on, else a
        // 1x1 zero texture so the slot has *something* and the
        // shader's branch on bloom_intensity > 0 keeps it skipped.
        pt::rhi::TextureHandle bloom_h{
            (bloom_on && bloom_intensity > 0.0f && bloom_mip_tex_id_[0] != 0)
                ? bloom_mip_tex_id_[0]
                : bloom_dummy_tex_id_};
        if (bloom_h.id != 0) cb->BindStorageTexture(2, bloom_h);
        bool hdr_pipeline = true;
        if (auto* v = C.FindCVar("r_hdr_pipeline")) hdr_pipeline = v->GetBool();
        bool flare_on = false;
        float flare_int = 0.15f, flare_disp = 0.012f, flare_thresh = 3.0f, flare_size = 0.10f;
        int   flare_count = 4;
        std::string flare_mode = "sun";
        if (auto* v = C.FindCVar("r_lens_flare"))            flare_on     = v->GetBool();
        if (auto* v = C.FindCVar("r_lens_flare_intensity"))  flare_int    = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_dispersion")) flare_disp   = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_count"))      flare_count  = v->GetInt();
        if (auto* v = C.FindCVar("r_lens_flare_threshold"))  flare_thresh = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_mode"))       flare_mode   = v->value;
        if (auto* v = C.FindCVar("r_lens_flare_size"))       flare_size   = v->GetFloat();
        if (flare_count < 1) flare_count = 1;
        if (flare_count > 6) flare_count = 6;

        // Project sun direction to screen-space UV. Used by 'sun'
        // mode to draw discrete ghost discs at the sun's mirrored
        // screen position. Sentinel (-2, -2) means sun is behind
        // the camera or the cvars are missing -- shader skips the
        // flare loop in that case.
        float sun_uv_x = -2.0f, sun_uv_y = -2.0f;
        {
            const float sx = push.sun_and_mode[0];
            const float sy = push.sun_and_mode[1];
            const float sz = push.sun_and_mode[2];
            const glm::vec3 sun_world{sx, sy, sz};
            const float fwd_dot = glm::dot(sun_world, fwd);
            if (fwd_dot > 1e-3f) {
                const float right_dot = glm::dot(sun_world, right);
                const float up_dot    = glm::dot(sun_world, up);
                const float ft = cam.FovYTan();
                const float xs = (right_dot / fwd_dot) / (ft * aspect);
                const float ys = (up_dot    / fwd_dot) / ft;
                sun_uv_x = xs * 0.5f + 0.5f;
                sun_uv_y = -ys * 0.5f + 0.5f;
            }
        }

        // Shader-side ghost record. 32 bytes; matches `ShaderGhost` in
        // Tonemap.slang exactly so the std140 layout aligns. Each ghost
        // contributes one Gaussian splat per RGB channel at scaled sun
        // UV positions; the per-channel scales encode chromatic
        // dispersion derived from the lens elements' Abbe numbers.
        struct PtShaderGhost {
            float scale_r;
            float scale_g;
            float scale_b;
            float intensity;
            float radius_px;
            float _pad[3];
        };
        static_assert(sizeof(PtShaderGhost) == 32, "ShaderGhost layout");
        static_assert(sizeof(PtShaderGhost)
                      == sizeof(lensflare::ShaderGhost), "ShaderGhost layout");

        struct TonePush {
            float        exposure;
            std::uint32_t passthrough;
            float        bloom_intensity;
            float        flare_intensity;
            float        flare_dispersion;
            std::uint32_t flare_count;
            float        flare_threshold;
            std::uint32_t flare_mode_sun;   // 1 = explicit sun-position flare
            float        flare_size;        // sun mode: ghost disc base radius
            // CRITICAL: 4-byte alignment padding so the following
            // float2 sun_uv lands at an 8-byte boundary on Metal.
            // Without this, the MSL compiler inserts implicit padding
            // before float2 -- but the C++ side doesn't, so the
            // shader reads bytes from the wrong offset (got
            // sun_uv = (engine.y, 0), making the crosshair stick to
            // the top-left edge regardless of where the sun is).
            float        _pad_sun_align;
            float        sun_uv[2];         // 8-byte aligned; matches MSL float2
            // Physical-flare extension. flare_mode_physical = 1 routes
            // the shader to gather Gaussian splats from the ghosts[]
            // array (positions are sun_uv * scale per channel; radius
            // is in pixels). Padding aligns the ghosts[] array to a
            // 16-byte boundary for std140 / Metal struct rules.
            std::uint32_t flare_mode_physical;
            std::uint32_t physical_ghost_count;
            // Star-split present flag (issue #46). 1 -> Tonemap.slang
            // adds stars_tex to the HDR colour pre-bloom; 0 -> skip
            // the additive read entirely (the slot is bound to a
            // 1x1 placeholder for descriptor validity but the shader
            // never samples it). Mirrors `stars_present` in
            // Tonemap.slang's Push cbuffer at the same byte offset.
            std::uint32_t stars_present;
            std::uint32_t _pad_stars_align;
            // 48 bytes of padding to advance ghosts[] forward to host
            // offset 112 -- the Vulkan push-constant split boundary
            // (VulkanDevice::kPushSplitOffset). The first 112 bytes of
            // this struct go to vkCmdPushConstants on the SPIR-V path;
            // the remaining 512 bytes (the ghost array) spill into the
            // Frame UBO at vk::binding(14, 0). With the array landing
            // at exactly offset 112, the spill is bit-aligned to the
            // shader-side Frame UBO declaration -- no per-byte
            // shuffling. On Metal the whole 624-byte block goes
            // through setBytes; the padding is dead weight (~0.05% of
            // a frame's CPU push budget) but keeps a single host-side
            // TonePush shape across both backends.
            float        _pad_to_push_split[12];   // 12 * 4 = 48 bytes
            PtShaderGhost ghosts[lensflare::kMaxGhosts];
        } tp{};
        static_assert(sizeof(TonePush) % 16 == 0, "TonePush 16-byte aligned");
        static_assert(sizeof(TonePush) == 624,
                      "TonePush layout (ghosts must land at offset 112 "
                      "for the SPIR-V push/UBO split)");
        static_assert(offsetof(TonePush, ghosts) == 112,
                      "ghosts[] must start at the Vulkan push-split boundary");
        // tp.exposure is now dead in the shader -- Tonemap.slang reads
        // exposure_state[0] from the bound buffer instead (matches
        // PathTrace.slang's inline tonemap path). The field stays in
        // the struct so the C++/MSL layouts don't shift; zero it so a
        // stale value doesn't mislead anyone reading a push dump.
        tp.exposure         = 0.0f;
        tp.passthrough      = hdr_pipeline ? 0u : 1u;
        tp.bloom_intensity  = (bloom_on && bloom_h.id == bloom_mip_tex_id_[0])
                                ? bloom_intensity : 0.0f;
        // Flare also samples the bloom mip, so it depends on bloom
        // having actually run. If bloom is off the bloom_tex slot is
        // the 1x1 placeholder and the flare loop would just sample
        // zeros -- gate it explicitly to skip the loop entirely.
        const bool bloom_ran    = bloom_on && bloom_h.id == bloom_mip_tex_id_[0];
        const bool flare_sun_m  = (flare_mode == "sun");
        const bool flare_phys_m = (flare_mode == "physical");
        // sun + physical modes don't need the bloom mip; image mode does.
        const bool flare_can_run = flare_on && (flare_sun_m || flare_phys_m || bloom_ran);
        tp.flare_intensity  = flare_can_run ? flare_int : 0.0f;
        // Debug overlay sentinel: -1 in flare_intensity tells the
        // shader to skip the flare loop and instead draw a green
        // crosshair at sun_uv. Lets the user visually verify the
        // engine-side projection.
        bool debug_sun = false;
        if (auto* v = C.FindCVar("r_debug_sun_overlay")) debug_sun = v->GetBool();
        if (debug_sun) tp.flare_intensity = -1.0f;
        tp.flare_dispersion = flare_disp;
        tp.flare_count      = static_cast<std::uint32_t>(flare_count);
        tp.flare_threshold  = flare_thresh;
        tp.flare_mode_sun       = flare_sun_m  ? 1u : 0u;
        tp.flare_mode_physical  = flare_phys_m ? 1u : 0u;
        // stars_present + _pad_stars_align kept in TonePush for layout
        // stability (TonePush::ghosts must stay at offset 112 for the
        // Vulkan push/UBO split); set to zero -- the shader no longer
        // reads them since stars_tex was dropped from Tonemap.slang in
        // the StarsComposite rewrite.
        tp.stars_present        = 0u;
        tp._pad_stars_align     = 0u;
        tp.flare_size       = flare_size;
        tp.sun_uv[0]        = sun_uv_x;
        tp.sun_uv[1]        = sun_uv_y;

        // Physical-mode ghost packing. Per-frame work: viewport-scaled
        // pixel radii for each of the precomputed ghost matrices,
        // capped at the user's r_lens_flare_count (so the slider is
        // still meaningful in this mode). When physical mode is off,
        // ghost_count = 0 makes the shader skip the loop.
        if (flare_phys_m && flare_can_run) {
            lensflare::ShaderGhost shader_ghosts[lensflare::kMaxGhosts];
            const int packed = lensflare::prepare_shader_ghosts(
                lens_system_, lens_ghosts_, lens_ghost_count_,
                lens_main_path_, static_cast<int>(fc.height),
                shader_ghosts, lensflare::kMaxGhosts);
            const int n = std::min(packed, flare_count);
            tp.physical_ghost_count = static_cast<std::uint32_t>(n);
            std::memcpy(tp.ghosts, shader_ghosts,
                        n * sizeof(PtShaderGhost));
        } else {
            tp.physical_ghost_count = 0u;
        }

        cb->PushConstants(&tp, sizeof(tp));
        cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);
        }  // end: if (use_engine_tonemap)
    } else if (vulkan_pre_denoise_bloom_engaged_) {
        // Both denoiser_active_ AND bloom_without_denoiser went false
        // in the same frame (e.g. user issued `r_denoiser none;
        // r_bloom 0`, or TearDownDevice cleared denoiser_active_
        // while r_bloom was already off). The predicate's edge-detect
        // inside the outer block didn't run, so the latch is stuck.
        // Clear it here with a disengage log so the next "engaged"
        // edge still fires once bloom comes back on. (Copilot review,
        // PR #22.)
        LOG_INFO("engine: vulkan pre-denoise bloom disengaged");
        vulkan_pre_denoise_bloom_engaged_ = false;
    }
    // Same stuck-latch fallback for the OptiX bloom path. Fires when
    // the outer `if (denoiser_active_ || bloom_without_denoiser)` is
    // false (so the inner edge-detect didn't run) but the OptiX-bloom
    // latch is still hot -- e.g. user toggled `r_denoiser none` while
    // r_bloom was on, or the device tore down. Clears the latch with a
    // disengage log so the next OptiX-bloom engagement still fires.
    if (vulkan_optix_bloom_engaged_ &&
        !(denoiser_active_ || bloom_without_denoiser)) {
        LOG_INFO("engine: bloom-with-OptiX disengaged");
        vulkan_optix_bloom_engaged_ = false;
    }

    // ---- Editor 3D-transform gizmo dispatch (issue: editor 3D gizmos) ----
    //
    // Post-tonemap, pre-perf-overlay. Writing into the swapchain image
    // means the gizmo doesn't get ACES'd (so it reads cleanly at any
    // exposure -- desirable for editor handles) but DOES sit underneath
    // the perf HUD. Gated on selected_prim_id_ being non-zero AND
    // r_editor_gizmo (the cvar gate is enforced inside
    // UpdateEditorGizmoFrame so the segment buffer stays cleared on
    // disable).
    UpdateEditorGizmoFrame();
    // One-shot log so we know the dispatch is firing (very useful when
    // wiring up new backends or first-run-after-merge debugging). The
    // static gates this to exactly one print per process lifetime.
    if (editor_overlay_pipeline_id_ != 0 &&
        editor_overlay_segs_buf_id_ != 0 &&
        editor_gizmo_.SegmentCount() > 0) {
        static bool s_logged_once = false;
        if (!s_logged_once) {
            LOG_INFO("editor_overlay: first dispatch, segments={} "
                     "(selected_prim_id={}, mode={})",
                     editor_gizmo_.SegmentCount(),
                     selected_prim_id_,
                     static_cast<int>(editor_gizmo_.GetMode()));
            s_logged_once = true;
        }
        // Compute-after-compute barrier on the swapchain image. Same
        // shape as the PerfOverlay barrier below: tonemap (or Optix /
        // SVGF finalize) wrote the swapchain; this kernel reads +
        // writes it.
        cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                     pt::rhi::BarrierDesc::Stage::ComputeWrite});
        cb->BindComputePipeline(pt::rhi::PipelineHandle{editor_overlay_pipeline_id_});
        cb->BindStorageTexture(0, fc.swapchain_image);
        cb->BindBuffer(1, pt::rhi::BufferHandle{editor_overlay_segs_buf_id_}, 0);

        // Push layout mirrors shaders/EditorOverlay.slang:
        //   float4 pos_fovtan, fwd_aspect, right_xyz, up_xyz;
        //   uint   num_segments, _pad0, _pad1, _pad2;
        struct EditorOverlayPush {
            float        pos_fovtan[4];
            float        fwd_aspect[4];
            float        right_xyz[4];
            float        up_xyz[4];
            std::uint32_t num_segments;
            std::uint32_t _pad0;
            std::uint32_t _pad1;
            std::uint32_t _pad2;
        } ep{};
        static_assert(sizeof(EditorOverlayPush) == 80,
                      "EditorOverlayPush must match shaders/EditorOverlay.slang");

        const auto& C2 = *camera_;
        const glm::vec3 fwd_v   = C2.Forward();
        const glm::vec3 right_v = C2.Right();
        const glm::vec3 up_v    = C2.Up();
        const float aspect_ratio = (fc.height > 0)
                                      ? float(fc.width) / float(fc.height)
                                      : 1.0f;
        ep.pos_fovtan[0] = C2.pos.x;
        ep.pos_fovtan[1] = C2.pos.y;
        ep.pos_fovtan[2] = C2.pos.z;
        ep.pos_fovtan[3] = C2.FovYTan();
        ep.fwd_aspect[0] = fwd_v.x;
        ep.fwd_aspect[1] = fwd_v.y;
        ep.fwd_aspect[2] = fwd_v.z;
        ep.fwd_aspect[3] = aspect_ratio;
        ep.right_xyz[0] = right_v.x;
        ep.right_xyz[1] = right_v.y;
        ep.right_xyz[2] = right_v.z;
        ep.right_xyz[3] = 0.0f;
        ep.up_xyz[0] = up_v.x;
        ep.up_xyz[1] = up_v.y;
        ep.up_xyz[2] = up_v.z;
        ep.up_xyz[3] = 0.0f;
        ep.num_segments = std::min(editor_gizmo_.SegmentCount(),
                                   editor_overlay_segs_buf_capacity_);
        cb->PushConstants(&ep, sizeof(ep));
        cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);
    }

    // RHI-mode perf overlay: final compute pass that composites a panel
    // + sparkline onto the post-tonemap swapchain image.  Skipped when
    // the user picked native mode or set r_perf_overlay 0.  No text
    // yet -- bitmap-font follow-up will land tier 1+ numbers here too.
    {
        auto& Cn = pt::console::Console::Get();
        int   level    = 0;
        std::string mode = "native";
        if (auto* lv = Cn.FindCVar("r_perf_overlay"))      level = std::atoi(lv->value.c_str());
        if (auto* mv = Cn.FindCVar("r_perf_overlay_mode")) mode  = mv->value;
        if (level > 0 && mode == "rhi" &&
            perfoverlay_pipeline_id_ != 0 && perfoverlay_drawlist_id_ != 0) {
            // Compute-after-compute barrier on the swapchain image.
            // The previous dispatch (PathTrace inline tonemap on
            // Vulkan, or Tonemap.slang on Metal-with-denoiser) wrote
            // the swapchain via storage-image stores; PerfOverlay
            // now reads + writes the same image.  Without this
            // barrier the load sees stale data and the resulting
            // composite scribbles over the rendered scene.  Same
            // pattern as the PathTrace -> AutoExposure barrier
            // above (Metal's Barrier() is a no-op since dispatches
            // on a shared resource auto-barrier; Vulkan needs the
            // explicit emit).
            cb->Barrier({pt::rhi::BarrierDesc::Stage::ComputeWrite,
                         pt::rhi::BarrierDesc::Stage::ComputeWrite});
            // Panel sized for the chrome we draw: a fixed-aspect
            // rectangle in the top-right corner.  Tier 3 adds height
            // for the sparkline graph.
            constexpr std::uint32_t kPanelW    = 240;
            constexpr std::uint32_t kPanelMargin = 12;
            std::uint32_t panel_h = (level >= 3) ? 90u : 28u;
            std::uint32_t panel_w = kPanelW;
            // Skip the overlay this frame if the swapchain is smaller
            // than the margin -- unsigned subtraction would underflow
            // to a huge value and produce wild panel coords / sizes.
            // It's just a HUD; no harm in dropping it for a tiny window.
            if (fc.width <= kPanelMargin || fc.height <= kPanelMargin) {
                panel_w = panel_h = 0;
            } else {
                if (panel_w + kPanelMargin > fc.width)  panel_w = fc.width - kPanelMargin;
                if (panel_h + kPanelMargin > fc.height) panel_h = fc.height - kPanelMargin;
            }
            std::uint32_t panel_x = (panel_w == 0u || panel_w + kPanelMargin > fc.width)
                                  ? 0u : fc.width - panel_w - kPanelMargin;
            std::uint32_t panel_y = kPanelMargin;

            if (panel_w == 0u || panel_h == 0u) {
                // Window too small for the HUD -- skip the dispatch.
            } else {
            // Build the draw list in panel-local coords.  Each command
            // is uint4: kind, x, y, payload.  Buffer was sized for
            // perfoverlay_drawlist_capacity_ entries; clamp.
            std::vector<std::uint32_t> dl;
            dl.reserve(perfoverlay_drawlist_capacity_ * 4);
            auto push_cmd = [&](std::uint32_t kind, std::uint32_t x,
                                std::uint32_t y, std::uint32_t payload) {
                if (dl.size() / 4 >= perfoverlay_drawlist_capacity_) return;
                dl.push_back(kind);
                dl.push_back(x);
                dl.push_back(y);
                dl.push_back(payload);
            };

            if (level >= 3 && !perf_history_.empty()) {
                // Sparkline.  Linearize the ring (oldest-first), find
                // the peak, project samples to panel-local Y.  Ref
                // line at 16.67 ms for 60 fps.
                std::size_t n = perf_history_.size();
                std::vector<float> linear(n);
                for (std::size_t i = 0; i < n; ++i) {
                    linear[i] = perf_history_[(perf_history_pos_ + i) % n];
                }
                float peak = 1.0f;
                for (float v : linear) if (v > peak) peak = v;
                if (peak < 0.5f) peak = 0.5f;

                std::uint32_t graph_x  = 6;
                std::uint32_t graph_y  = 24;     // below the (future) text rows
                std::uint32_t graph_w  = panel_w - 12;
                std::uint32_t graph_h  = panel_h - graph_y - 6;

                // 60 fps reference (16.67 ms).
                if (16.667f < peak) {
                    std::uint32_t ref_y = graph_y + graph_h - 1
                        - static_cast<std::uint32_t>((16.667f / peak) * (graph_h - 2));
                    push_cmd(/*rule*/ 2, graph_x, ref_y, graph_w);
                }

                // Sparkline polyline: emit one segment per pixel of
                // graph width, picking samples evenly across history.
                std::uint32_t segs = (graph_w > 1) ? graph_w - 1 : 0;
                for (std::uint32_t i = 0; i < segs; ++i) {
                    std::size_t i0 = (i * n) / segs;
                    std::size_t i1 = ((i + 1) * n) / segs;
                    if (i1 >= n) i1 = n - 1;
                    float v0 = linear[i0];
                    float v1 = linear[i1];
                    std::uint32_t x0 = graph_x + i;
                    std::uint32_t x1 = graph_x + i + 1;
                    std::uint32_t y0 = graph_y + graph_h - 1
                        - static_cast<std::uint32_t>((v0 / peak) * (graph_h - 2));
                    std::uint32_t y1 = graph_y + graph_h - 1
                        - static_cast<std::uint32_t>((v1 / peak) * (graph_h - 2));
                    if (y0 >= graph_y + graph_h) y0 = graph_y + graph_h - 1;
                    if (y1 >= graph_y + graph_h) y1 = graph_y + graph_h - 1;
                    // Pack y0 in low 16, y1 in high 16, x1 in payload low 16.
                    std::uint32_t z   = (y0 & 0xFFFFu) | ((y1 & 0xFFFFu) << 16);
                    std::uint32_t pay = (x1 & 0xFFFFu);
                    push_cmd(/*segment*/ 1, x0, z, pay);
                    if (dl.size() / 4 >= perfoverlay_drawlist_capacity_) break;
                }
            }

            std::uint32_t num_cmds = static_cast<std::uint32_t>(dl.size() / 4);
            if (num_cmds > 0) {
                device_->WriteBuffer(pt::rhi::BufferHandle{perfoverlay_drawlist_id_},
                                     dl.data(), dl.size() * sizeof(std::uint32_t), 0);
            }

            // Theme-driven panel/accent/graph colours (RGBA8 packed,
            // alpha used for blend strength).  Picks a small set
            // matching the native overlay's palette.
            std::string theme = "hardcore";
            if (auto* tv = Cn.FindCVar("r_theme")) theme = tv->value;
            std::uint32_t panel_col  = 0xC8160E0Cu;  // dark, alpha ~0.78
            std::uint32_t accent_col = 0xFFFFF000u;  // hardcore cyan
            std::uint32_t graph_col  = 0xFFFFF000u;
            if (theme == "amber")     { accent_col = 0xFF2890FFu; graph_col = 0xFF2890FFu; }
            else if (theme == "synthwave") { accent_col = 0xFFB450FFu; graph_col = 0xFFB450FFu; }
            else if (theme == "matrix")    { accent_col = 0xFF00FF50u; graph_col = 0xFF00FF50u; }
            else if (theme == "vault")     { accent_col = 0xFFFFA050u; graph_col = 0xFFFFA050u; }
            else if (theme == "sakura")    { accent_col = 0xFFB48CFFu; graph_col = 0xFFB48CFFu; }
            else if (theme == "mono")      { accent_col = 0xFFE0E0E0u; graph_col = 0xFFE0E0E0u; }

            struct PerfPush {
                std::uint32_t panel_x;
                std::uint32_t panel_y;
                std::uint32_t panel_w;
                std::uint32_t panel_h;
                std::uint32_t num_cmds;
                std::uint32_t panel_color;
                std::uint32_t accent_color;
                std::uint32_t graph_color;
            } pp{
                panel_x, panel_y, panel_w, panel_h,
                num_cmds,
                panel_col, accent_col, graph_col,
            };
            // Phase-0 defensive guard -- 8 uints = 32B, naturally aligned,
            // assert catches future reorderings into a non-16B size.
            static_assert(sizeof(PerfPush) == 32,
                          "PerfPush layout mismatch with PerfOverlay.slang");
            static_assert(sizeof(PerfPush) % 16 == 0,
                          "PerfPush must be 16-byte aligned (cbuffer rule)");

            cb->BindComputePipeline(pt::rhi::PipelineHandle{perfoverlay_pipeline_id_});
            cb->BindStorageTexture(0, fc.swapchain_image);
            // Engine buffer slot 1 -> kSlotToBufBinding[1] = vk::binding(3)
            // on Vulkan, MSL buffer(1) on Metal (after the
            // _slot_dummy0 declaration in PerfOverlay.slang shifts
            // draw_list off MSL slot 0).  Slot 3 here previously
            // landed at vk::binding(5) (primitives) -- which is what
            // produced the "rhi overlay totally breaks the scene"
            // garbage: the shader read mesh_positions data as
            // DrawCmd entries.
            cb->BindBuffer(1, pt::rhi::BufferHandle{perfoverlay_drawlist_id_}, 0);
            cb->PushConstants(&pp, sizeof(pp));
            cb->Dispatch((panel_w + 7) / 8, (panel_h + 7) / 8, 1);
            }   // panel_w/panel_h > 0 -- HUD dispatched
        }
    }

    {
        PT_ZONE_SCOPED_N("Device::Submit");
        device_->Submit(cb);
    }
    {
        PT_ZONE_SCOPED_N("Device::EndFrame (present)");
        device_->EndFrame(cb);
    }

    // Post-present frame-capture hook. Driven by r_capture_frame_at /
    // r_capture_seq / r_capture_seed -- the cvars' on_change handlers
    // arm the FrameCapture state and this hook services it. Cheap
    // when nothing is requested (single bool check via IsArmed());
    // pays the readback + tonemap + PPM-write cost only on the actual
    // capture frame. Source texture is the engine's "current visible
    // HDR image" -- denoise_color when a denoiser is active, accum
    // otherwise -- so PPM RMSE comparisons across r_denoiser values
    // measure the denoiser delta directly (modulo bloom / lens flare /
    // perf overlay, which only land in the swapchain).
    if (pt::engine::capture::IsArmed()) {
        // Source texture selection: when no denoiser is active, the
        // path tracer's `accum` (RGBA32F) is what gets tonemapped to
        // the swapchain. When a denoiser is on, the denoiser's output
        // image -- `post_denoise_hdr_tex_id_` (RGBA16F) -- is what bloom
        // + tonemap read; that's the right "denoiser delta" to capture.
        // Note: `denoise_color_tex_id_` is the denoiser's *input* (the
        // path tracer's per-frame HDR write), not its output -- naming
        // historical from the SVGF G-buffer days. Capturing that would
        // make svgf_atrous / optix_hdr / optix_hdr_aov produce
        // bitwise-identical PPMs (same input goes into all three).
        const auto kind = denoiser_active_
            ? pt::engine::CaptureSourceKind::DenoiseColor
            : pt::engine::CaptureSourceKind::Accum;
        const std::uint64_t denoised_tex_id =
            denoiser_active_ ? post_denoise_hdr_tex_id_ : 0;
        std::string denoiser_label = "off";
        if (auto* dv = C.FindCVar("r_denoiser")) denoiser_label = dv->value;
        float exposure_fallback = 1.0f;
        if (auto* ev = C.FindCVar("r_exposure")) exposure_fallback = ev->GetFloat();
        const auto cap_result = pt::engine::capture::MaybeCapture(
            device_.get(),
            frame_index_,
            accum_texture_id_,
            denoised_tex_id,
            exposure_state_id_,
            accum_w_,
            accum_h_,
            kind,
            denoiser_label,
            exposure_fallback);
        // Keep the user-facing cvar surface honest. Help text says
        // r_capture_frame_at "auto-resets to 0 after the capture
        // fires" and r_capture_seq is empty when disarmed -- so
        // mirror the internal disarm into the cvar values now.
        // SetCVarOverride bypasses CVAR_READONLY but these aren't
        // marked readonly anyway.
        if (cap_result.one_shot_fired) {
            C.SetCVarOverride("r_capture_frame_at", "0");
        }
        if (cap_result.seq_completed) {
            C.SetCVarOverride("r_capture_seq", "");
        }
    }

    prev_view_proj_       = curr_view_proj;
    prev_view_proj_valid_ = true;

    // Auto-exposure was a per-frame CPU readback of accum_hdr,
    // which on dGPU stalls the GPU queue (vkQueueWaitIdle every 8
    // frames -> ~18 fps). It's now done entirely on GPU by the
    // AutoExposure compute pass dispatched right after the path
    // tracer above; the result lives in the exposure_state buffer
    // and PathTrace.slang reads it on the next frame. Manual-mode
    // (r_auto_exposure=0) value updates are done via on_change
    // cvar hooks below in BindCvars(), so no per-frame work happens
    // for that path either.
}

void Engine::UpdateCamera(double dt) {
    if (!camera_ || !window_) return;
    auto& C = pt::console::Console::Get();

    // cam_warp_smooth tween advance. Runs BEFORE the WASD / RMB-look
    // input handling so manual control inputs detected below can abort
    // the tween cleanly. Smoothstep easing (3t^2 - 2t^3) for cinematic
    // in/out feel. Lerps position linearly; lerps yaw/pitch directly
    // (the cam_warp_smooth command pre-normalised yaw to the shortest
    // path so this just interpolates a small delta).
    if (cam_tween_active_) {
        cam_tween_t_seconds_ += dt;
        double tf = cam_tween_t_seconds_ / cam_tween_duration_;
        if (tf >= 1.0) {
            tf = 1.0;
            cam_tween_active_ = false;   // snap to target on final frame
        }
        // Smoothstep s = 3t^2 - 2t^3.
        const float t = static_cast<float>(tf);
        const float s = t * t * (3.0f - 2.0f * t);
        camera_->pos = glm::vec3(
            cam_tween_start_pos_[0] + (cam_tween_target_pos_[0] - cam_tween_start_pos_[0]) * s,
            cam_tween_start_pos_[1] + (cam_tween_target_pos_[1] - cam_tween_start_pos_[1]) * s,
            cam_tween_start_pos_[2] + (cam_tween_target_pos_[2] - cam_tween_start_pos_[2]) * s);
        camera_->yaw   = cam_tween_start_yaw_   + (cam_tween_target_yaw_   - cam_tween_start_yaw_)   * s;
        camera_->pitch = cam_tween_start_pitch_ + (cam_tween_target_pitch_ - cam_tween_start_pitch_) * s;
        camera_->ClampPitch();
        accum_dirty_ = true;
    }

    // Mouse-look: hold right mouse to capture, release to free the cursor.
    bool rmb = window_->IsMouseButtonDown(1);   // GLFW_MOUSE_BUTTON_RIGHT == 1
    // Abort any active tween if the user takes manual control via
    // RMB-look or WASD (the WASD check happens further down via the
    // speed/sprint multiplier; cancel pre-emptively here on RMB and
    // we'll cancel again in the WASD branch if the user moved).
    if (rmb && cam_tween_active_) {
        cam_tween_active_ = false;
    }
    if (rmb && !mouse_look_active_) {
        window_->SetCursorMode(0x00034003);     // GLFW_CURSOR_DISABLED
        mouse_look_active_ = true;
    } else if (!rmb && mouse_look_active_) {
        window_->SetCursorMode(0x00034001);     // GLFW_CURSOR_NORMAL
        mouse_look_active_ = false;
    }

    if (mouse_look_active_) {
        double dx = 0.0, dy = 0.0;
        window_->ConsumeMouseDelta(dx, dy);
        float sens = 0.12f;
        if (auto* v = C.FindCVar("cam_sensitivity")) sens = v->GetFloat();
        camera_->yaw   += glm::radians(static_cast<float>(dx) * sens);
        camera_->pitch -= glm::radians(static_cast<float>(dy) * sens);
        camera_->ClampPitch();
    }

    // Scroll wheel adjusts cam_speed multiplicatively (1.1x per notch
    // up, /1.1x per notch down). Exponential feel makes it easy to
    // span the 0.1..30 m/s slider range without dozens of clicks.
    {
        double sx = 0.0, sy = 0.0;
        window_->ConsumeScrollDelta(sx, sy);
        if (sy != 0.0) {
            if (auto* v = C.FindCVar("cam_speed")) {
                float cur = v->GetFloat();
                cur *= std::pow(1.1f, static_cast<float>(sy));
                if (cur < 0.1f)  cur = 0.1f;
                if (cur > 30.0f) cur = 30.0f;
                v->value = std::to_string(cur);
                if (v->on_change) v->on_change(*v);
            }
        }
    }

    // WASD + Space/Ctrl movement.  Speed in units/sec; shift sprints.
    float speed   = 3.0f;
    float sprint  = 3.0f;
    if (auto* v = C.FindCVar("cam_speed"))       speed  = v->GetFloat();
    if (auto* v = C.FindCVar("cam_sprint_mult")) sprint = v->GetFloat();

    bool shift = window_->IsKeyDown(340) || window_->IsKeyDown(344);  // L/R Shift

    // Gamepad input (#83). Polled before the sprint multiplier is
    // applied so the trigger contribution can blend into `sprint_mult`
    // alongside the shift-key bool. Read deadzone + look sensitivity
    // from cvars every frame; both are CVAR_ARCHIVE so live edits via
    // the console propagate immediately.
    bool   pad_enabled = true;
    float  pad_dz      = 0.15f;
    float  pad_look_s  = 2.0f;
    if (auto* v = C.FindCVar("cam_gamepad"))                  pad_enabled = v->GetBool();
    if (auto* v = C.FindCVar("cam_gamepad_deadzone"))         pad_dz      = v->GetFloat();
    if (auto* v = C.FindCVar("cam_gamepad_look_sensitivity")) pad_look_s  = v->GetFloat();

    pt::app::GamepadState pad;
    if (pad_enabled) {
        pad = pt::app::PollGamepad(pad_dz);
        pt::app::LogConnectionTransitions(pad);
    }

    // Triggers extend the shift-key sprint. We pick the larger of the
    // two trigger values so either L2 or R2 alone works (different
    // games prefer different mappings). The result blends linearly
    // between `speed` (no trigger) and `speed * sprint` (full trigger);
    // additive with shift so holding both gives the same sprint cap.
    float trig_max = std::max(pad.left_trigger, pad.right_trigger);
    bool  sprint_on = shift || trig_max > 0.0f;
    if (sprint_on) {
        // Blend factor: shift counts as full trigger.
        float blend = shift ? 1.0f : trig_max;
        speed *= 1.0f + (sprint - 1.0f) * blend;
    }

    glm::vec3 fwd   = camera_->Forward();
    glm::vec3 right = camera_->Right();
    glm::vec3 wm{0.0f};
    if (window_->IsKeyDown(87))  wm += fwd;            // W
    if (window_->IsKeyDown(83))  wm -= fwd;            // S
    if (window_->IsKeyDown(68))  wm += right;          // D
    if (window_->IsKeyDown(65))  wm -= right;          // A
    if (window_->IsKeyDown(32))  wm += glm::vec3(0,1,0);  // Space
    if (window_->IsKeyDown(341)) wm -= glm::vec3(0,1,0);  // L Ctrl
    if (glm::dot(wm, wm) > 0.0f) {
        camera_->pos += glm::normalize(wm) * (speed * static_cast<float>(dt));
        // WASD input takes manual control -- abort any in-flight tween.
        cam_tween_active_ = false;
    }

    // Gamepad translation: left stick maps to (strafe, forward). We
    // scale by stick magnitude (already in [0, 1] post-deadzone) so
    // partial deflection slows the camera proportionally -- matches
    // analog stick UX. Sprint multiplier above already factored into
    // `speed`. Decoupled from the keyboard path so the two can be
    // used simultaneously (e.g. WASD + look with the right stick).
    if (pad.connected && (pad.left_x != 0.0f || pad.left_y != 0.0f)) {
        glm::vec3 pad_move = right * pad.left_x + fwd * pad.left_y;
        // Don't re-normalise -- stick magnitude IS the intent.
        camera_->pos += pad_move * (speed * static_cast<float>(dt));
    }

    // Gamepad look: right stick maps to (yaw, pitch). Sign convention
    // matches the mouse-look path -- +X yaws right (clockwise from
    // above), +Y pitches up. Scaled by sensitivity (radians/sec at
    // full deflection) and dt. Runs whether or not RMB is held so the
    // pad can drive the camera without engaging mouse-look mode.
    if (pad.connected && (pad.right_x != 0.0f || pad.right_y != 0.0f)) {
        float rate    = pad_look_s * static_cast<float>(dt);
        camera_->yaw   += pad.right_x * rate;
        camera_->pitch += pad.right_y * rate;
        camera_->ClampPitch();
    }

    // Pull live overrides from cvars if the user typed them.  The
    // simplest UX is "if cvar string differs from camera state, apply".
    // We push back the camera state to the cvars on every frame too so
    // the panel reflects current values.
    if (auto* v = C.FindCVar("cam_fov")) {
        float new_fov = v->GetFloat();
        if (std::abs(new_fov - camera_->fov_deg) > 1e-3f) camera_->fov_deg = new_fov;
    }
}

// --- Editor backend (agent-19) -------------------------------------------
void Engine::SetSelection(SelectionKind kind, std::uint32_t id) {
    // None always pairs with id=0 so the (kind, id) state is canonical:
    // ambiguity between "kind=None, id=42" and "kind=AnalyticPrim, id=0"
    // would let outline tests misbehave.
    if (kind == SelectionKind::None) id = 0;

    const bool changed = (kind != selected_kind_) || (id != selected_prim_id_);
    if (!changed) return;

    selected_kind_     = kind;
    selected_prim_id_  = id;
    // Outline visibility changes the rendered image even though the
    // scene geometry is unchanged -- accumulating onto the previous
    // history would leave a ghost of the OLD selection. accum_dirty_
    // resets the EMA / running-mean accumulator so the next frame is a
    // clean sample of the new outline state.
    accum_dirty_       = true;

    // Notify connected editor clients. The kind / id payload mirrors
    // the WebSocket `select` incoming message so the same parser works
    // for both directions. SerializeScene's "selection" field also
    // matches this schema. We emit even when there's no server bound
    // (no-op) so the call site doesn't need a redundant guard.
    if (server_) {
        const char* kind_str = pt::editor::SelectionKindToString(static_cast<int>(kind));
        auto data = fmt::format(R"({{"kind":"{}","id":{}}})", kind_str, id);
        server_->BroadcastEvent("selection_change", data);
    }
}

void Engine::BroadcastSceneDirty() {
    // Wave-5 polish: editor clients (scene-hierarchy, inspector, etc)
    // re-fetch via list_scene on this event. Payload is a single
    // tombstone object rather than a per-prim diff so server-side
    // state stays O(1) per mutation regardless of how many editor
    // clients are connected. No-op when no ConsoleServer is bound
    // (e.g. headless --smoke-frames builds).
    if (server_) {
        server_->BroadcastEvent("scene_dirty", "{}");
    }
}

void Engine::HandleMouseInput() {
    // Bail if any prerequisite isn't satisfied. We also reset the
    // edge-detect latch so releasing the gating condition (closing the
    // console overlay, releasing RMB) doesn't immediately fire a
    // spurious pick from a button that was held during the suppressed
    // interval.
    if (!window_ || !camera_) {
        lmb_was_down_ = false;
        return;
    }
    // RMB = mouse-look mode; suppress picks because the cursor is
    // captured and the user is panning the view.
    if (window_->IsMouseButtonDown(1)) {
        lmb_was_down_ = false;
        return;
    }
    // Console overlay focus: a click inside the overlay should be
    // consumed by the overlay's text field, not by the viewport pick
    // path. Treat "shown" as "focused" for the purposes of this gate;
    // it's a strict superset, and the cost is one extra frame of pick
    // suppression when the user opens the overlay -- not noticeable.
    if (overlay_ && overlay_->IsShown()) {
        lmb_was_down_ = false;
        return;
    }

    const bool lmb_now = window_->IsMouseButtonDown(0);
    if (!lmb_now) {
        lmb_was_down_ = false;
        return;
    }
    // Edge detect: only fire on the press, not on every frame the
    // button is held. Without this a held click would re-pick whatever
    // is under the cursor every frame, which feels broken (and burns
    // accum resets).
    if (lmb_was_down_) return;
    lmb_was_down_ = true;

    // Map window-local cursor position to a primary ray. We use the
    // window's content-area dimensions (NOT the framebuffer pixels on
    // high-DPI: GLFW's cursor is in screen units, and so is
    // Window::Width/Height). The path tracer's camera basis uses the
    // same convention so a cursor at the center of the viewport maps
    // exactly to the center primary ray. Slight DPI mismatch on
    // 2x-scale Mac displays produces a sub-pixel offset that's well
    // inside the click tolerance of the sphere intersect.
    double cx = 0.0, cy = 0.0;
    window_->GetCursorPos(cx, cy);
    const int w = window_->Width();
    const int h = window_->Height();
    if (w <= 0 || h <= 0) return;
    // Clamp to viewport so an out-of-window cursor doesn't produce a
    // wildly off-axis ray. Negative / over-max coords come from GLFW
    // briefly during window-drag operations.
    if (cx < 0.0)       cx = 0.0;
    if (cx > double(w)) cx = double(w);
    if (cy < 0.0)       cy = 0.0;
    if (cy > double(h)) cy = double(h);

    // NDC-space pixel coordinates: (-1..+1) horizontally, (+1..-1)
    // vertically (Y flipped because cursor origin is top-left while
    // ray Y points up). Pixel center offset of 0.5 keeps the math
    // consistent with the shader's gl_GlobalInvocationID.xy + 0.5 form.
    const float u = (float(cx) + 0.5f) / float(w) * 2.0f - 1.0f;
    const float v = 1.0f - (float(cy) + 0.5f) / float(h) * 2.0f;
    const float aspect = float(w) / float(h);
    const float tan_half_fov = camera_->FovYTan();

    const glm::vec3 fwd   = camera_->Forward();
    const glm::vec3 right = camera_->Right();
    const glm::vec3 up    = camera_->Up();
    const glm::vec3 ro    = camera_->pos;
    const glm::vec3 rd    = glm::normalize(
        fwd + right * (u * aspect * tan_half_fov) + up * (v * tan_half_fov));

    // Walk primitives_ for the nearest hit. Match the GPU shader's
    // intersectSphere / intersectPlane epsilons (1e-4f on the
    // positive-t check) so the CPU pick agrees with what the user
    // visually saw -- a sphere that's visibly painted on-pixel always
    // accepts the click.
    std::uint32_t best_id = 0;
    float         best_t  = 1e30f;
    bool          best_hit = false;
    for (const auto& [id, p] : primitives_) {
        float t = 0.0f;
        if (p.type == AnalyticPrim::Sphere) {
            const glm::vec3 c{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            const glm::vec3 oc = ro - c;
            const float bb = glm::dot(oc, rd);
            const float cc = glm::dot(oc, oc) - p.radius_or_d * p.radius_or_d;
            const float disc = bb * bb - cc;
            if (disc < 0.0f) continue;
            const float h_ = std::sqrt(disc);
            const float t0 = -bb - h_;
            const float t1 = -bb + h_;
            if (t0 > 1e-4f) t = t0;
            else if (t1 > 1e-4f) t = t1;
            else continue;
        } else {
            // Plane. n.p + d = 0; ray hits at t = -(n.ro + d) / (n.rd).
            const glm::vec3 n{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
            const float denom = glm::dot(n, rd);
            if (std::fabs(denom) < 1e-6f) continue;
            const float th = -(glm::dot(n, ro) + p.radius_or_d) / denom;
            if (th <= 1e-4f) continue;
            t = th;
        }
        if (t < best_t) {
            best_t   = t;
            best_id  = id;
            best_hit = true;
        }
    }

    if (best_hit) {
        SetSelection(SelectionKind::AnalyticPrim, best_id);
    } else {
        SetSelection(SelectionKind::None, 0);
    }
}
// --- end Editor backend --------------------------------------------------

void Engine::Tick(double dt) {
    PT_ZONE_SCOPED_N("Engine::Tick");
    pt::console::Console::Get().Drain();

    // Deferred swap screenshot: after Drain may have queued one in
    // pending_swap_screenshot_path_, poll Device::ReadbackSwapchain
    // until it returns true. First call (which happened inside the
    // command lambda) seeded the request; subsequent ticks block on
    // Submit consuming it and the GPU finishing the copy. Bounded
    // timeout in ticks (~5s @ 60fps).
    if (!pending_swap_screenshot_path_.empty() && device_ != nullptr) {
        std::uint32_t sw = 0, sh = 0;
        pt::rhi::Device::SwapFormat fmt = pt::rhi::Device::SwapFormat::Other;
        std::vector<std::uint8_t> raw(std::size_t(accum_w_) * accum_h_ * 4u);
        const bool ready = device_->ReadbackSwapchain(raw.data(), raw.size(),
                                                      &sw, &sh, &fmt);
        if (ready && sw > 0 && sh > 0) {
            const bool bgra = (fmt == pt::rhi::Device::SwapFormat::Bgra8Unorm ||
                               fmt == pt::rhi::Device::SwapFormat::Bgra8Srgb ||
                               fmt == pt::rhi::Device::SwapFormat::Other);
            std::vector<std::uint8_t> rgb(std::size_t(sw) * sh * 3u);
            for (std::size_t i = 0, n = std::size_t(sw) * sh; i < n; ++i) {
                std::uint8_t b0 = raw[i * 4 + 0];
                std::uint8_t b1 = raw[i * 4 + 1];
                std::uint8_t b2 = raw[i * 4 + 2];
                rgb[i * 3 + 0] = bgra ? b2 : b0;
                rgb[i * 3 + 1] =        b1;
                rgb[i * 3 + 2] = bgra ? b0 : b2;
            }
            if (pt::engine::capture::WriteRgb8(
                    pending_swap_screenshot_path_,
                    rgb.data(), sw, sh,
                    pending_swap_screenshot_fmt_)) {
                LOG_INFO("screenshot: wrote {} ({}x{} swap, format={})",
                         pending_swap_screenshot_path_, sw, sh,
                         pt::engine::capture::OutputFormatExtension(
                             pending_swap_screenshot_fmt_));
            } else {
                LOG_ERROR("screenshot: cannot write '{}' for swap capture (format={})",
                          pending_swap_screenshot_path_,
                          pt::engine::capture::OutputFormatExtension(
                              pending_swap_screenshot_fmt_));
            }
            pending_swap_screenshot_path_.clear();
            pending_swap_screenshot_ticks_ = 0;
        } else if (++pending_swap_screenshot_ticks_ > 300) {
            // ~5s at 60fps. Drop on timeout.
            LOG_WARN("screenshot: swap capture timed out (never consumed by Submit) -- giving up on '{}'",
                     pending_swap_screenshot_path_);
            pending_swap_screenshot_path_.clear();
            pending_swap_screenshot_ticks_ = 0;
        }
    }

    UpdateCamera(dt);

    // Editor mode (agent-19): LMB raycast pick. Runs after UpdateCamera
    // so the camera basis the picker uses for ray construction matches
    // exactly what the path tracer will dispatch on this frame's
    // RenderFrame. Internally suppressed when RMB is held (mouse-look
    // is active) or the console overlay is shown.
    HandleMouseInput();

    // Audio listener update: push the freshest camera pose into the
    // audio subsystem so distance attenuation + stereo pan track the
    // viewer correctly. No-op when audio_system_ is null or its
    // device init failed -- the AudioSystem guards itself.
    if (audio_system_ && camera_) {
        audio_system_->Tick(camera_->pos, camera_->Forward());
    }

    // Sky animation: advance r_sky_hour by `rate * dt` real-time
    // seconds. Wraps modulo 24. Marks accumulation dirty so the path
    // tracer drops its history each frame the time is actually moving.
    {
        auto& C = pt::console::Console::Get();
        bool animate = false;
        if (auto* v = C.FindCVar("r_sky_animate")) animate = v->GetBool();
        // Skip the animate work entirely when astro mode is off:
        // r_sky_hour drives nothing visible in manual-sun mode, so
        // advancing it would just spin a feedback loop that re-fires
        // the cvar's astro-off warning every frame. The on_change
        // handler already warned the user once when they set
        // r_sky_animate 1, so they know it's a no-op.
        bool astro_on_for_animate = false;
        if (auto* v = C.FindCVar("r_sky_use_astronomical")) astro_on_for_animate = v->GetBool();
        // EMA tracks slowly-advancing astronomical sun smoothly; only
        // the running-mean path needs the per-tick dirty bit.
        float ema_alpha_sky = 0.0f;
        if (auto* v = C.FindCVar("r_accum_ema_alpha")) ema_alpha_sky = v->GetFloat();
        if (animate && astro_on_for_animate) {
            float rate = 0.0f;
            if (auto* v = C.FindCVar("r_sky_animate_rate")) rate = v->GetFloat();
            float hour = 12.0f;
            if (auto* v = C.FindCVar("r_sky_hour")) hour = v->GetFloat();
            hour += rate * float(dt);
            // Wrap into [0, 24).
            hour = std::fmod(hour, 24.0f);
            if (hour < 0.0f) hour += 24.0f;
            // SetCVarOverride below triggers r_sky_hour's on_change handler
            // (registered in RegisterCommands) which unconditionally sets
            // accum_dirty_ = true. That's correct for the legacy running-
            // mean path but defeats EMA's whole purpose -- EMA wants the
            // accumulator to smoothly track the slowly-advancing sun via
            // exponential decay, not hard-reset every tick. Snapshot
            // accum_dirty_ before the override and restore if EMA is the
            // active mode.
            bool dirty_before = accum_dirty_;
            C.SetCVarOverride("r_sky_hour", std::to_string(hour));
            if (ema_alpha_sky > 0.0f) accum_dirty_ = dirty_before;
        }

        // Cloud wind drift uses `frame_index_ * (1/60)` as time-seconds
        // in the path tracer's cloud field (see RenderFrame), so the
        // cloud pattern advances every frame whenever wind is non-zero.
        // Without dirtying the accumulator, samples taken across many
        // frames at different cloud positions get averaged into
        // accum_hdr -> visible "stretched / smeared clouds" after the
        // engine sits idle for a while; mouse motion fixes it because
        // the camera-move dirty flag resets the accumulator.  Mirror
        // the sky_animate path: when clouds are on AND wind is set,
        // mark dirty every tick so accumulation resets per frame.
        // When r_accum_ema_alpha > 0, EMA tracks the drifting cloud
        // field smoothly via exponential decay -- no need to nuke the
        // accumulator every frame. The legacy running-mean path still
        // needs the dirty trigger to avoid wet-paint smear.
        float ema_alpha_dirty = 0.0f;
        if (auto* v = C.FindCVar("r_accum_ema_alpha")) ema_alpha_dirty = v->GetFloat();
        bool clouds_on = false;
        if (auto* v = C.FindCVar("r_clouds")) clouds_on = v->GetBool();
        if (clouds_on && ema_alpha_dirty <= 0.0f) {
            float wx = 0.0f, wz = 0.0f;
            if (auto* v = C.FindCVar("r_clouds_wind_x")) wx = v->GetFloat();
            if (auto* v = C.FindCVar("r_clouds_wind_z")) wz = v->GetFloat();
            if (wx != 0.0f || wz != 0.0f) accum_dirty_ = true;
        }
    }

    // Particle / VFX system tick (#82 MVP). CPU sim runs every frame
    // regardless of r_particles so a user toggling the master flag back
    // on doesn't see stale particles snap back into existence; instead
    // particles age naturally even while the composite kernel is off.
    //
    // Threading (perf/particles-async-tick): we kick the integrator on
    // the ParticleSystem's persistent worker thread via TickAsync and
    // let it overlap StepPhysics + the rest of the per-frame CPU work.
    // The render path waits on the result via particles_->Particles()
    // (which calls WaitForTick internally) before reading the live
    // vector for upload. The MVP integrator is short (~tens of us at
    // the 1024-particle cap) but the overlap is free correctness-wise
    // and a no-op on platforms where the worker would otherwise idle.
    if (particles_) {
        auto& Cp = pt::console::Console::Get();
        if (auto* mv = Cp.FindCVar("r_particles_max")) {
            int m = mv->GetInt();
            if (m < 0)     m = 0;
            if (m > 65536) m = 65536;
            particles_->SetMaxParticles(static_cast<std::uint32_t>(m));
        }
        particles_->TickAsync(static_cast<float>(dt));
    }

    // r_phys_debug_visualize 1 -> 0 restoration (#181). Runs BEFORE
    // StepPhysics so the restore fires unconditionally on every Tick,
    // even when physics is idle and StepPhysics' early returns would
    // otherwise swallow the falling edge.
    MaybeRestorePhysDebugColors();

    // Physics step (#132): runs after UpdateCamera and before RenderFrame
    // so the per-frame writeback into `primitives_` lands before
    // EnsurePrimitivesUploaded re-packs the analytic-prim buffer +
    // refits the BVH. No-op when phys_enabled = 0 or no particles exist.
    StepPhysics(static_cast<float>(dt));

    auto t0 = std::chrono::steady_clock::now();
    RenderFrame();
    auto frame_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // Per-frame ring buffer for the tier-3 sparkline.  Sized for ~4
    // seconds of 60-fps history; the overlay subsamples down to its
    // available pixel width when rendering. Ring is resized lazily on
    // first push when an overlay actually needs samples, so a session
    // with the HUD disabled never allocates this buffer or runs the
    // sample-write logic. Both native and rhi-mode paths feed off this
    // ring, so gate on either being on.
    bool perf_active = false;
    if (auto* lv = pt::console::Console::Get().FindCVar("r_perf_overlay")) {
        perf_active = lv->GetInt() > 0;
    }
    if (perf_active) {
        if (perf_history_.empty()) perf_history_.assign(256, 0.0f);
        perf_history_[perf_history_pos_] = static_cast<float>(frame_ms);
        perf_history_pos_ = (perf_history_pos_ + 1) % perf_history_.size();
    }

    // Aggregate at ~10 Hz so the network doesn't get spammed at
    // 240+ fps and the overlay's text doesn't visually flicker.
    // Same window also drives the avg/min/max stats consumed below.
    static double accum_s = 0.0;
    static int    accum_frames = 0;
    static double accum_render_ms = 0.0;
    static double accum_min = 1.0e30;
    static double accum_max = 0.0;
    accum_s         += dt;
    accum_frames    += 1;
    accum_render_ms += frame_ms;
    if (frame_ms < accum_min) accum_min = frame_ms;
    if (frame_ms > accum_max) accum_max = frame_ms;
    if (accum_s >= 0.1) {
        double fps        = accum_frames / accum_s;
        double avg_render = accum_render_ms / accum_frames;
        int w = window_ ? window_->Width()  : 0;
        int h = window_ ? window_->Height() : 0;

        if (server_) {
            auto data = fmt::format(
                R"({{"fps":{:.1f},"frame_ms":{:.3f},"trace_ms":{:.3f},"backend":"{}","resolution":[{},{}]}})",
                fps, avg_render, avg_render,
                pt::rhi::BackendName(current_backend_), w, h);
            server_->BroadcastEvent("frame_stats", data);
        }

        if (perf_overlay_ && perf_overlay_->Level() > 0) {
            // Linearize the ring (oldest-first) so the overlay can
            // walk it left-to-right without modular arithmetic.
            std::vector<float> linear(perf_history_.size());
            for (std::size_t i = 0; i < perf_history_.size(); ++i) {
                linear[i] = perf_history_[(perf_history_pos_ + i) % perf_history_.size()];
            }
            auto& Cn = pt::console::Console::Get();
            pt::app::PerfStats st;
            st.fps          = fps;
            st.frame_ms_avg = avg_render;
            st.frame_ms_min = accum_min;
            st.frame_ms_max = accum_max;
            st.backend      = pt::rhi::BackendName(current_backend_);
            st.width        = w;
            st.height       = h;
            st.gpu_bytes    = device_ ? device_->CurrentAllocatedBytes() : 0;
            if (auto* sv = Cn.FindCVar("r_spp"))         st.spp         = sv->GetInt();
            if (auto* mv = Cn.FindCVar("r_max_bounces")) st.max_bounces = mv->GetInt();
            st.primitives   = primitives_.size();
            st.frame_ms_history = std::span<const float>(linear.data(), linear.size());
            perf_overlay_->Update(st);
        }

        accum_s         = 0.0;
        accum_frames    = 0;
        accum_render_ms = 0.0;
        accum_min       = 1.0e30;
        accum_max       = 0.0;
    }
}

void Engine::Run() {
    using clk = std::chrono::steady_clock;
    auto last = clk::now();

    // Smoke-test mode: read the per-invocation frame budget once at
    // loop start. 0 (the default) = unbounded; >0 = render exactly
    // this many frames then exit cleanly. Set via the
    // `--smoke-frames=N` CLI override which routes to the
    // pt_smoke_frames cvar in RegisterCVars.
    //
    // Intended use is manual local validation -- e.g. after touching
    // a backend's init path, run
    //   ./demont --smoke-frames=8 --r-backend=metal
    // and confirm the process exits 0. The 2-exit-code "backend init
    // failed in 10s" path below catches the silent-init-failure case
    // that's otherwise hard to spot.
    //
    // NOT wired into GH Actions CI today -- the public-tier hosted
    // runners have no real GPU. See the comment block on
    // pt_smoke_frames in RegisterCVars for the full backstory and
    // the build.yml comment for where the smoke steps used to live.
    //
    // Read once (not per-frame) because mid-run mutations to the
    // budget would just be confusing -- the budget is a launch
    // parameter, not a runtime knob.
    std::uint32_t smoke_frame_budget = 0;
    if (auto* v = pt::console::Console::Get().FindCVar("pt_smoke_frames")) {
        const int n = v->GetInt();
        if (n > 0) smoke_frame_budget = static_cast<std::uint32_t>(n);
    }
    if (smoke_frame_budget > 0) {
        LOG_INFO("smoke-test mode: will render {} frame(s) then exit cleanly",
                 smoke_frame_budget);
        // If any CLI arg was rejected at parse time (allowed-values
        // miss, non-numeric --smoke-frames, etc.), fail the smoke
        // test immediately. The engine is in a known-misconfigured
        // state -- e.g. the user asked for `--r-backend=metalfx`,
        // got "metalfx is not in allowed set", and the override was
        // dropped, so the engine's now running with whatever
        // r_backend was previously. That's NOT what the operator
        // requested; better to surface as a smoke-test failure than
        // silently render against the wrong backend.
        if (cli_arg_was_rejected_) {
            LOG_ERROR("smoke-test: at least one CLI arg was rejected "
                      "during parse. Failing the smoke test (exit code 2) "
                      "instead of proceeding with a misconfigured engine.");
            smoke_test_failed_ = true;
            return;
        }
    }
    std::uint32_t smoke_frames_rendered = 0;

    // Smoke-test hang detector. If the device_ never becomes non-null
    // (backend init failed silently, MoltenVK refused to create on the
    // host, Metal hardware-RT requirement unmet on a virtual GPU, etc.)
    // the existing post-device frame counter would NEVER fire, and the
    // pre-device sleep loop further down would idle indefinitely. On a
    // CI runner that means the job burns until the workflow timeout
    // rather than failing promptly. Track elapsed time since Run start;
    // if we've been waiting for the device past kSmokeNoDeviceTimeoutSec
    // and smoke mode is on, give up loudly with smoke_test_failed_ = true
    // and let main() translate that into a non-zero exit code.
    //
    // 10s is well past every "normal" Init+ first-frame timeline (most
    // backends bind within ~100-500ms; the worst case is Vulkan cold
    // pipeline compile which is already bounded by the async pipeline-
    // build worker, capped at a few seconds even on cold caches). A
    // backend that hasn't bound at 10s is broken, not slow.
    constexpr double kSmokeNoDeviceTimeoutSec = 10.0;
    const auto       run_start                = clk::now();

    while (!wants_quit_ && !window_->ShouldClose()) {
        auto now = clk::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;

        window_->PollEvents();
        // Child overlay HWND on Win32 doesn't auto-resize when the
        // parent does; push the latest client size each frame (no-op
        // unless changed). Also matches Mac's NSWindow setFrame logic
        // which is driven by AppKit's resize, not the engine loop.
        if (overlay_) {
            overlay_->NotifyParentResized(window_->Width(), window_->Height());
        }
        if (perf_overlay_) {
            perf_overlay_->NotifyParentResized(window_->Width(), window_->Height());
        }
        Tick(dt);

        // If no device is bound yet we'd busy-loop; throttle.  With a
        // device, the swapchain present already paces us via vsync.
        if (!device_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }

        // Smoke-test exit accounting -- two paths:
        //
        //   1. device_ is bound AND we're past the loading window:
        //      count this iteration as a rendered frame. Once the
        //      budget is hit, request a clean exit. The
        //      `!loading_frame_active_` gate is critical: on Vulkan
        //      cold-cache launches the pipeline-build worker is in
        //      flight for the first ~1-3s, during which Tick paints
        //      a flat dark loading frame via the swapchain clear path
        //      (no PathTrace dispatch). Counting those would consume
        //      the budget on placeholder paint, not real renders --
        //      `--smoke-frames=8` could exit after only 2 real
        //      frames if 6 loading frames slipped in first.
        //
        //   2. device_ is still null: backend init may still be in
        //      flight, or may have failed silently. If we've been
        //      waiting past kSmokeNoDeviceTimeoutSec the in-flight
        //      case is implausible -- treat it as a silent failure.
        // Device-lost correctness floor (VkResult check work, parallel
        // PR series): if a vk* call inside the engine's per-frame work
        // (vkQueueSubmit / vkWaitForFences) latched IsDeviceLost,
        // there is no useful frame to count -- the GPU is dead and the
        // engine has been silently rendering against nothing. Fail the
        // smoke test loud and request a clean exit. The non-smoke
        // operator gets a single LOG_ERROR (rate-limited by the
        // already-quit branch) and the loop terminates next iteration.
        // Independent of, and ordered before, the budget accounting
        // below so a device-lost on the very last frame still produces
        // a "fail" verdict instead of a coincidental "rendered N frames"
        // success.
        if (device_ && device_->IsDeviceLost() && !device_lost_observed_) {
            device_lost_observed_ = true;
            LOG_ERROR("engine: backend device reports DEVICE_LOST "
                      "(rhi='{}'). Render output from this point is "
                      "undefined. Requesting exit.",
                      device_->DeviceName());
            if (smoke_frame_budget > 0) {
                LOG_ERROR("smoke-test: device-lost mid-run -- failing "
                          "the smoke test (exit code 2) instead of "
                          "reporting silent success against a dead GPU.");
                smoke_test_failed_ = true;
            }
            wants_quit_ = true;
        }

        if (smoke_frame_budget > 0) {
            if (device_ && !loading_frame_active_) {
                ++smoke_frames_rendered;
                if (smoke_frames_rendered >= smoke_frame_budget) {
                    LOG_INFO("smoke-test: rendered {} frame(s); "
                             "requesting clean exit",
                             smoke_frames_rendered);
                    wants_quit_ = true;
                }
            } else if (!device_) {
                const double elapsed = std::chrono::duration<double>(
                    clk::now() - run_start).count();
                if (elapsed > kSmokeNoDeviceTimeoutSec) {
                    LOG_ERROR("smoke-test: no device bound after {:.1f}s -- "
                              "backend init appears to have failed silently. "
                              "Failing the smoke test (exit code 2).",
                              elapsed);
                    smoke_test_failed_ = true;
                    wants_quit_ = true;
                }
            }
            // else: device_ bound but loading_frame_active_ -- silently
            // skip this iteration, don't count it, don't fail on it.
        }

        // Tracy frame boundary. Marks one logical frame for the
        // profiler's flame-graph view -- one tick of the main loop ==
        // one frame, regardless of whether the device was bound or the
        // loading-frame path painted. No-op when PT_ENABLE_TRACY is OFF.
        PT_FRAME_MARK;
    }
    LOG_INFO("Run loop exited.");

    // Golden-image regression matrix (issue #45) final-frame capture.
    // Fires when smoke mode hit its budget cleanly (smoke_test_failed_
    // is still false) AND the operator gave a --smoke-capture-out=path.
    // Done synchronously here, AFTER the loop exited (last RenderFrame
    // has presented) but BEFORE Shutdown -> TearDownDevice releases
    // the GPU textures we're about to read back from.
    //
    // Source-texture pick mirrors FrameCapture's MaybeCapture hook:
    // denoise_color when a denoiser is active, accum otherwise. Same
    // host-side ACES + sRGB OETF as the screenshot command / PR #94's
    // CaptureEncoder, so the PNG written here matches what a golden
    // captured via `screenshot path.png accum` would produce -- the
    // matrix can mix-and-match generation paths.
    if (smoke_frame_budget > 0 &&
        !smoke_test_failed_ &&
        smoke_frames_rendered >= smoke_frame_budget &&
        device_ != nullptr) {
        std::string out_path;
        if (auto* cv = pt::console::Console::Get().FindCVar("pt_smoke_capture_out");
            cv != nullptr) {
            out_path = cv->value;
        }
        if (!out_path.empty()) {
            // Same source-of-truth picker as the post-present FrameCapture
            // hook in RenderFrame: when a denoiser is on, capture its
            // *output* image (post_denoise_hdr_tex_id_); otherwise the
            // path tracer's accum_hdr. Reading denoise_color_tex_id_
            // when a denoiser is active would capture the denoiser's
            // INPUT (path tracer's per-frame write) and produce
            // bitwise-identical PNGs across svgf_atrous / optix_hdr /
            // optix_hdr_aov -- defeating the whole point of a
            // (backend, denoiser, scene) matrix.
            const auto kind = denoiser_active_
                ? pt::engine::CaptureSourceKind::DenoiseColor
                : pt::engine::CaptureSourceKind::Accum;
            const std::uint64_t tex_id = denoiser_active_
                ? post_denoise_hdr_tex_id_
                : accum_texture_id_;
            const int bytes_per_pixel = denoiser_active_ ? 8 : 16;  // half / float

            // Drain in-flight work so the readback sees the last frame's
            // submitted writes, not an in-progress dispatch.
            device_->WaitIdle();

            if (tex_id == 0) {
                LOG_ERROR("smoke-capture: requested source texture is not "
                          "allocated (denoiser_active={} tex_id=0). "
                          "Failing the smoke test (exit code 2).",
                          denoiser_active_);
                smoke_test_failed_ = true;
            } else if (accum_w_ <= 0 || accum_h_ <= 0) {
                LOG_ERROR("smoke-capture: invalid render extent {}x{} "
                          "-- backend never produced a real frame. "
                          "Failing the smoke test (exit code 2).",
                          accum_w_, accum_h_);
                smoke_test_failed_ = true;
            } else {
                std::vector<std::uint8_t> raw(
                    std::size_t(accum_w_) * accum_h_ * bytes_per_pixel);
                std::uint32_t rb_w = 0, rb_h = 0;
                if (!device_->ReadbackTexture(pt::rhi::TextureHandle{tex_id},
                                              raw.data(), raw.size(),
                                              &rb_w, &rb_h) ||
                    rb_w == 0 || rb_h == 0) {
                    LOG_ERROR("smoke-capture: ReadbackTexture failed "
                              "(tex_id={}, {}x{}). Failing the smoke "
                              "test (exit code 2).",
                              tex_id, accum_w_, accum_h_);
                    smoke_test_failed_ = true;
                } else {
                    // Resolve the live exposure scalar so the host-side
                    // tonemap matches what was on-screen. Same fallback
                    // path FrameCapture uses: live readback when the
                    // exposure_state buffer exists, manual r_exposure
                    // value otherwise.
                    float exposure_fallback = 1.0f;
                    if (auto* ev = pt::console::Console::Get()
                                       .FindCVar("r_exposure")) {
                        exposure_fallback = ev->GetFloat();
                    }
                    float exposure = exposure_fallback;
                    if (exposure_state_id_ != 0) {
                        float v = exposure_fallback;
                        if (device_->ReadbackBuffer(
                                pt::rhi::BufferHandle{exposure_state_id_},
                                &v, sizeof(float))) {
                            exposure = v;
                        }
                    }

                    // Parent dir is created here rather than relying on
                    // operator-supplied conventions; ctest cells write
                    // into ${CMAKE_CURRENT_BINARY_DIR}/golden_actual/...
                    // which doesn't pre-exist on a fresh build.
                    std::filesystem::path png_path(out_path);
                    if (png_path.has_parent_path()) {
                        std::error_code ec;
                        std::filesystem::create_directories(
                            png_path.parent_path(), ec);
                    }

                    const bool ok = pt::engine::capture::EncodeAndWritePng(
                        png_path, raw, rb_w, rb_h, kind, exposure);
                    if (!ok) {
                        LOG_ERROR("smoke-capture: EncodeAndWritePng failed "
                                  "for '{}'. Failing the smoke test "
                                  "(exit code 2).", png_path.string());
                        smoke_test_failed_ = true;
                    } else {
                        LOG_INFO("smoke-capture: wrote {} ({}x{}, "
                                 "source={}, exposure={:.3f})",
                                 png_path.string(), rb_w, rb_h,
                                 denoiser_active_ ? "denoise_color" : "accum_hdr",
                                 exposure);
                    }
                }
            }
        }
    }

    // Final smoke-test verdict for the "exit before budget" cancellation
    // case. wants_quit_ alone is ambiguous -- it can be set by:
    //   - This loop reaching the frame budget (success)
    //   - The user closing the window mid-smoke-test (failure)
    //   - The `quit` console command (failure -- same idea)
    //   - The no-device timeout above (already marked failure)
    // If we exited without hitting the budget AND haven't already
    // marked failure, the smoke test was cancelled prematurely --
    // record it so main() exits non-zero. Avoids the
    // "Cmd+Q-during-smoke-runs returns 0" footgun.
    if (smoke_frame_budget > 0 &&
        !smoke_test_failed_ &&
        smoke_frames_rendered < smoke_frame_budget) {
        LOG_ERROR("smoke-test: exited after {} of {} budgeted frame(s) -- "
                  "loop terminated before budget was hit (window-close, "
                  "quit command, or external signal). Failing the smoke "
                  "test (exit code 2).",
                  smoke_frames_rendered, smoke_frame_budget);
        smoke_test_failed_ = true;
    }
}

// ----- Commands -------------------------------------------------------------

namespace {
// Shared with the later anonymous namespace block above
// RegisterCsgCommands(); declared here so RegisterCommands() (which
// hosts particle_emit and other commands) can reuse it without
// duplicating a parse-float lambda. The definition lives in the
// later block below.
bool ParseFloat(std::string_view s, float& out);
}  // namespace

void Engine::RegisterCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("quit", "Exit the application cleanly.",
        [this](auto, pt::console::Output& out) {
            out.PrintLine("bye.");
            wants_quit_ = true;
            if (window_) window_->RequestClose();
        });

    C.RegisterCommand("echo", "Print arguments back.",
        [](auto args, pt::console::Output& out) {
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (i) out.Print(" ");
                out.Print(args[i]);
            }
            out.Print("\n");
        });

    // ---- cam_* dev tools ---------------------------------------------------
    // Save / restore camera viewpoints by slot number. Slot 0 is the
    // engineering default (read-only, hardcoded inside cam_reset).
    // Slots 1..9 are user-savable, persisted across runs via the
    // cam_slot_N CVAR_ARCHIVE cvars registered above. cam_load and
    // cam_reset both fire prev_view_proj_valid_ = false so the next
    // frame's dd.reset_history clears the ACTIVE temporal denoiser's
    // history -- consumed by SVGF/NRD's temporal accumulation,
    // MetalFX's history-state reset, and the OptiX temporal
    // denoisers' prev_output / internal-guide-layer ping-pong.
    // Without this the per-frame temporal blend keeps mixing
    // pre-teleport content into post-teleport frames (visible as the
    // "went inside a sphere, came out, screen still tinted black"
    // symptom v0.3.4 surfaced).
    // Helpers as no-capture lambdas (so they're copy-constructible into
    // the std::function backing each command). Constants live inside
    // each lambda rather than being captured -- captures-by-reference
    // chain trips up MSVC's _Enable_if_callable_t when the helpers get
    // captured by value into the inner command lambdas.
    const auto fmt_cam_state = [](const pt::renderer::Camera& c) -> std::string {
        constexpr float kRadToDeg = 57.29577951308232f;
        return fmt::format("{:.6f} {:.6f} {:.6f} {:.4f} {:.4f} {:.3f}",
                           c.pos.x, c.pos.y, c.pos.z,
                           c.yaw   * kRadToDeg,
                           c.pitch * kRadToDeg,
                           c.fov_deg);
    };
    const auto parse_cam_state = [](std::string_view s, pt::renderer::Camera& out) -> bool {
        constexpr float kDegToRad = 0.017453292519943295f;
        std::istringstream iss{std::string(s)};
        float x, y, z, yaw_d, pitch_d, fov_d;
        if (!(iss >> x >> y >> z >> yaw_d >> pitch_d >> fov_d)) return false;
        out.pos     = glm::vec3(x, y, z);
        out.yaw     = yaw_d   * kDegToRad;
        out.pitch   = pitch_d * kDegToRad;
        out.fov_deg = fov_d;
        out.ClampPitch();
        return true;
    };
    // /EHs-c- (exceptions disabled) -- can't use std::stoi try/catch.
    // std::from_chars is the right tool: no allocations, no exceptions,
    // and it returns BOTH a parse-failure errc AND the consumed-up-to
    // pointer so we can reject partial inputs like "1abc" (which
    // strtol would silently accept as 1). Also reports
    // std::errc::result_out_of_range explicitly instead of silently
    // truncating a long-to-int cast.
    const auto parse_slot = [](std::span<const std::string_view> args, int default_slot) -> int {
        if (args.empty()) return default_slot;
        const std::string_view sv = args[0];
        if (sv.empty()) return -1;
        int parsed = 0;
        const auto* first = sv.data();
        const auto* last  = sv.data() + sv.size();
        const auto r = std::from_chars(first, last, parsed);
        if (r.ec != std::errc{}) return -1;   // parse error or out-of-range
        if (r.ptr != last)       return -1;   // partial input -- "1abc" rejected
        return parsed;
    };

    C.RegisterCommand("cam_save",
        "Save the current camera state into a slot (1..9, default 1). "
        "Slot 0 is the engineering default and cannot be saved into.",
        [this, fmt_cam_state, parse_slot](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_save: no camera"); return; }
            const int slot = parse_slot(args, 1);
            if (slot == 0) {
                out.PrintLine("cam_save: slot 0 is the engineering default (read-only). Use 1..9.");
                return;
            }
            if (slot < 1 || slot > 9) {
                out.FormatLine("cam_save: slot {} out of range; use 1..9", slot);
                return;
            }
            const std::string s = fmt_cam_state(*camera_);
            const std::string cvar_name = fmt::format("cam_slot_{}", slot);
            pt::console::Console::Get().SetCVarOverride(cvar_name, s);
            out.FormatLine("cam_save: slot {} = {}", slot, s);
        });

    C.RegisterCommand("cam_load",
        "Load a saved camera state from a slot (1..9, default 1). "
        "Use cam_reset for slot 0 (engineering default). Loading "
        "fires the active denoiser's history-reset flag so the "
        "temporal denoise pipeline (SVGF/NRD/MetalFX/OptiX-temporal) "
        "doesn't blend pre-teleport content forward.",
        [this, parse_cam_state, parse_slot](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_load: no camera"); return; }
            const int slot = parse_slot(args, 1);
            if (slot == 0) {
                out.PrintLine("cam_load: slot 0 is the engineering default; use `cam_reset` instead.");
                return;
            }
            if (slot < 1 || slot > 9) {
                out.FormatLine("cam_load: slot {} out of range; use 1..9", slot);
                return;
            }
            const std::string cvar_name = fmt::format("cam_slot_{}", slot);
            auto* v = pt::console::Console::Get().FindCVar(cvar_name);
            if (v == nullptr || v->value.empty()) {
                out.FormatLine("cam_load: slot {} is empty (use `cam_save {}` to populate)",
                               slot, slot);
                return;
            }
            if (!parse_cam_state(v->value, *camera_)) {
                out.FormatLine("cam_load: slot {} contents malformed: '{}'",
                               slot, v->value);
                return;
            }
            prev_view_proj_valid_ = false;  // active denoiser reset_history (SVGF/NRD/MetalFX/OptiX-temporal)
            out.FormatLine("cam_load: slot {} = {}", slot, v->value);
        });

    C.RegisterCommand("cam_reset",
        "Reset camera to the engineering default (slot 0, read-only). "
        "Fires the active denoiser's history-reset flag so the "
        "temporal denoise pipeline (SVGF/NRD/MetalFX/OptiX-temporal) "
        "doesn't blend pre-reset content forward.",
        [this](auto, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_reset: no camera"); return; }
            // Hardcoded engineering default. Matches the inline defaults
            // in Camera.h (pos=(0,1.5,4), yaw=0, pitch=-0.20rad=-11.46deg,
            // fov=60). NOT read from cam_pos / cam_yaw / etc. cvars
            // because those are CVAR_ARCHIVE -- they hold the user's
            // last-quit state, not the engine default.
            camera_->pos     = glm::vec3(0.0f, 1.5f, 4.0f);
            camera_->yaw     = 0.0f;
            camera_->pitch   = -0.20f;
            camera_->fov_deg = 60.0f;
            prev_view_proj_valid_ = false;
            out.PrintLine("cam_reset: pos=(0, 1.5, 4) yaw=0 pitch=-11.46 fov=60");
        });

    // Editor "Focus camera on target" -- frame a world-space point from a
    // canonical offset. Used by the scene-hierarchy panel's right-click
    // "Focus camera" action (agent-21); the panel passes the selected
    // primitive's centre and we drop the camera into a comfortable
    // viewing distance, with yaw/pitch aimed at the target.
    //
    // Heuristic: stand 3m behind (+Z) and 1m above the target, look down
    // ~15deg. This roughly matches the Blender "frame selected"
    // shortcut for objects near the origin without needing the engine
    // to know the prim's bounding radius (which the editor panel may not
    // have for SDF clusters / rigid bodies).
    C.RegisterCommand("cam_focus",
        "cam_focus <x> <y> <z>: jump the camera so it looks at the given "
        "world-space point from a canonical 3m back / 1m up offset. "
        "Fires the active denoiser's history-reset flag so the "
        "temporal denoise pipeline doesn't blend pre-jump content "
        "forward. Used by the editor scene-hierarchy panel's "
        "right-click 'Focus camera'.",
        [this](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_focus: no camera"); return; }
            if (args.size() != 3) {
                out.PrintLine("usage: cam_focus <x> <y> <z>");
                return;
            }
            float tx = 0.0f, ty = 0.0f, tz = 0.0f;
            const auto parse_f = [](std::string_view s, float& f) -> bool {
                std::string tmp(s);
                char* end = nullptr;
                const float v = std::strtof(tmp.c_str(), &end);
                if (end == tmp.c_str() || end == nullptr) return false;
                f = v;
                return true;
            };
            if (!parse_f(args[0], tx) || !parse_f(args[1], ty) || !parse_f(args[2], tz)) {
                out.PrintLine("cam_focus: failed to parse x/y/z (expected 3 floats)");
                return;
            }
            // Canonical viewing offset: 3m back along +Z, 1m above. Then
            // recompute yaw / pitch so the camera Forward() vector points
            // at the target. yaw=0 + pitch=-0.262rad (~-15deg) is exact
            // when the camera is at (tx, ty+1, tz+3) and the target is
            // at (tx, ty, tz); we still recompute from atan2 so any
            // future offset change keeps the aim correct.
            const glm::vec3 target(tx, ty, tz);
            const glm::vec3 new_pos(tx, ty + 1.0f, tz + 3.0f);
            const glm::vec3 dir = target - new_pos;
            // Forward(yaw, pitch) = (sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch))
            // Invert: pitch = asin(dir.y / |dir|), yaw = atan2(dir.x, -dir.z).
            const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
            float yaw = 0.0f;
            float pitch = 0.0f;
            if (len > 1e-6f) {
                pitch = std::asin(dir.y / len);
                yaw   = std::atan2(dir.x, -dir.z);
            }
            camera_->pos   = new_pos;
            camera_->yaw   = yaw;
            camera_->pitch = pitch;
            camera_->ClampPitch();
            prev_view_proj_valid_ = false;
            out.FormatLine("cam_focus: target=({:.3f}, {:.3f}, {:.3f}) "
                           "pos=({:.3f}, {:.3f}, {:.3f})",
                           tx, ty, tz, new_pos.x, new_pos.y, new_pos.z);
        });

    C.RegisterCommand("cam_list",
        "List camera slots: 0 = engineering default (read-only), "
        "1..9 = user-savable. Empty slots show <empty>.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            out.PrintLine("cam slots:");
            out.PrintLine("  0: <engineering default>  (use cam_reset to load)");
            for (int slot = 1; slot <= 9; ++slot) {
                const std::string cvar_name = fmt::format("cam_slot_{}", slot);
                auto* v = C2.FindCVar(cvar_name);
                if (v != nullptr && !v->value.empty()) {
                    out.FormatLine("  {}: {}", slot, v->value);
                } else {
                    out.FormatLine("  {}: <empty>", slot);
                }
            }
        });

    // ---- Named camera bookmarks (cam_save_named / cam_load_named) -----
    // Commands live in a dedicated registration function for two
    // reasons: (1) keeps RegisterCommands shorter; (2) the unit test
    // wires JUST these commands without dragging in the rest of the
    // RegisterCommands body (which captures `this` across audio /
    // RHI / window state that isn't initialised in the test harness).
    RegisterCameraBookmarkCommands();

    C.RegisterCommand("sys_info", "Summarize hardware.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            auto get = [&](const char* n)->std::string {
                auto* v = C2.FindCVar(n); return v ? v->value : std::string("?");
            };
            out.FormatLine("CPU      {} ({}P + {}E)", get("sys_cpu_model"),
                           get("sys_cpu_pcores"), get("sys_cpu_ecores"));
            out.FormatLine("Features {}", get("sys_cpu_features"));
            out.FormatLine("RAM      {} MB", get("sys_ram_total_mb"));
            out.FormatLine("OS       {}", get("sys_os"));
            out.FormatLine("GPU      {}", get("sys_gpu_name"));
        });

    C.RegisterCommand("mem_report", "Per-tag memory breakdown.",
        [](auto, pt::console::Output& out) {
            pt::mem::TagStats s[pt::kMemTagCount];
            pt::mem::GetReport(s);
            out.FormatLine("{:<12} {:>14} {:>14} {:>10} {:>10}",
                           "tag", "live(B)", "peak(B)", "allocs", "frees");
            std::size_t tl = 0, tp = 0;
            for (int i = 0; i < pt::kMemTagCount; ++i) {
                out.FormatLine("{:<12} {:>14} {:>14} {:>10} {:>10}",
                               pt::MemTagName(static_cast<pt::MemTag>(i)),
                               s[i].live_bytes, s[i].peak_bytes,
                               s[i].alloc_count, s[i].free_count);
                tl += s[i].live_bytes;
                tp += s[i].peak_bytes;
            }
            out.FormatLine("{:<12} {:>14} {:>14}", "TOTAL", tl, tp);
        });

    C.RegisterCommand("list_cvars", "List cvars (optional prefix).",
        [](auto args, pt::console::Output& out) {
            std::string prefix;
            if (!args.empty()) prefix.assign(args[0]);
            pt::console::Console::Get().EnumerateCVars(prefix,
                [&](pt::console::CVar& v) {
                    out.FormatLine("{} = \"{}\" ({})", v.name, v.value, v.description);
                });
        });

    C.RegisterCommand("list_commands", "List commands (optional prefix).",
        [](auto args, pt::console::Output& out) {
            std::string prefix;
            if (!args.empty()) prefix.assign(args[0]);
            pt::console::Console::Get().EnumerateCommands(prefix,
                [&](pt::console::Command& c) {
                    out.FormatLine("{} -- {}", c.name, c.description);
                });
        });

    // Reset every cvar to its default value -- pair with the `--no-cfg`
    // CLI flag for fully clean-room testing, or call interactively to
    // undo a session's mutations without restarting the engine. Single
    // undo entry so a follow-up `r_undo` restores the pre-reset state.
    // Useful for testers running multiple golden-fixture scenes in a
    // row -- prevents the first scene's cvars from leaking into the
    // second's render state.
    //
    // Registered under TWO names: the verbose `cvar_reset_all` for
    // programmatic clarity (smoke scripts, automation), and the short
    // `defaults` for human use at the console -- the latter is what
    // testers type before `exec tests/goldens/scenes/<scene>.cfg`
    // when running from an IDE-launched demont that auto-loads
    // demont.cfg (and therefore needs a runtime cvar wipe each time
    // the scene changes).
    auto reset_all = [](auto, pt::console::Output& out) {
        std::size_t n = pt::console::Console::Get().ResetAllCVarsToDefaults();
        if (n == 0) {
            out.PrintLine("defaults: no cvars differed from default -- nothing to reset.");
        } else {
            out.FormatLine("defaults: reset {} cvar(s) to default. r_undo restores.", n);
        }
    };
    C.RegisterCommand("cvar_reset_all",
        "Reset every cvar to its default value. One undo entry covers the whole reset. Alias: defaults.",
        reset_all);
    C.RegisterCommand("defaults",
        "Reset every cvar to its default value. Short alias for cvar_reset_all -- handy before `exec scene.cfg`.",
        reset_all);

    // ---- Favourites ----------------------------------------------------
    // Persisted shortcut commands. `fav` with no args saves the last
    // executed line; `fav <line>` saves an explicit line. Saved
    // favourites can be invoked as `f1`, `f2`, ... `fN` -- Console's
    // Execute() intercepts that pattern before smart-resolve. The
    // favourites vector is persisted to favorites.cfg next to
    // demont.cfg; loaded at Engine::Init, saved after every mutation.
    auto save_favorites = [this]() { SaveFavoritesToDisk(); };

    C.RegisterCommand("fav",
        "fav [<command line>]: save a command as a favourite. With no args, saves the last executed line. "
        "Saved favourites can be invoked as f1, f2, ... fN. Persists across sessions via favorites.cfg.",
        [save_favorites](auto args, pt::console::Output& out) {
            auto& Cn = pt::console::Console::Get();
            std::string line;
            if (args.empty()) {
                line = Cn.LastExecutedLine();
                if (line.empty()) {
                    out.PrintLine("fav: no last-executed line to save -- run a command first, "
                                  "or pass an explicit line: `fav r_clouds 1`");
                    return;
                }
            } else {
                // Re-join the args into a single line (the tokenizer
                // already split + dequoted them).
                for (std::size_t i = 0; i < args.size(); ++i) {
                    if (i) line.push_back(' ');
                    line.append(args[i]);
                }
            }
            Cn.AddFavorite(line);
            const std::size_t idx = Cn.FavoriteCount();
            out.FormatLine("fav: saved as f{} -> `{}`", idx, line);
            save_favorites();
        });

    C.RegisterCommand("list_favs",
        "List all saved favourites with their f1, f2, ... indices.",
        [](auto, pt::console::Output& out) {
            auto& Cn = pt::console::Console::Get();
            const auto favs = Cn.Favorites();
            if (favs.empty()) {
                out.PrintLine("(no favourites yet -- run a command then type `fav` to save it)");
                return;
            }
            out.FormatLine("{} favourite(s):", favs.size());
            for (std::size_t i = 0; i < favs.size(); ++i) {
                out.FormatLine("  f{}  {}", i + 1, favs[i]);
            }
        });

    C.RegisterCommand("unfav",
        "unfav <N>: remove the favourite at 1-based index N. Subsequent f<K> indices shift down.",
        [save_favorites](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: unfav <N>"); return; }
            int n = 0;
            const auto* first = args[0].data();
            const auto* last  = args[0].data() + args[0].size();
            const auto r = std::from_chars(first, last, n);
            if (r.ec != std::errc{} || r.ptr != last || n < 1) {
                out.FormatLine("unfav: '{}' is not a positive integer", std::string(args[0]));
                return;
            }
            auto& Cn = pt::console::Console::Get();
            const std::size_t before = Cn.FavoriteCount();
            if (static_cast<std::size_t>(n) > before) {
                out.FormatLine("unfav: no favourite at index {} (have {})", n, before);
                return;
            }
            const std::string removed = Cn.Favorites()[std::size_t(n) - 1];
            Cn.RemoveFavorite(std::size_t(n));
            out.FormatLine("unfav: removed f{} (`{}`); {} favourite(s) remain",
                           n, removed, Cn.FavoriteCount());
            save_favorites();
        });

    C.RegisterCommand("fav_clear",
        "Remove all favourites. Wipes favorites.cfg on next save.",
        [save_favorites](auto, pt::console::Output& out) {
            auto& Cn = pt::console::Console::Get();
            const std::size_t before = Cn.FavoriteCount();
            Cn.ClearFavorites();
            out.FormatLine("fav_clear: removed {} favourite(s)", before);
            save_favorites();
        });
    // ---- end Favourites -----------------------------------------------

    C.RegisterCommand("scene_save",
        "scene_save <path.toml>: write camera + analytic primitives to a TOML file. (CSG state isn't saved yet -- put csg_* commands in autoexec.cfg if you want it to persist.)",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: scene_save <path.toml>"); return; }
            std::string path(args[0]);

            toml::table root;

            // Camera.
            auto& cam = *camera_;
            toml::array pos_arr{cam.pos.x, cam.pos.y, cam.pos.z};
            toml::table cam_tbl;
            cam_tbl.insert("pos",   std::move(pos_arr));
            cam_tbl.insert("yaw",   double(glm::degrees(cam.yaw)));
            cam_tbl.insert("pitch", double(glm::degrees(cam.pitch)));
            cam_tbl.insert("fov",   double(cam.fov_deg));
            root.insert("camera", std::move(cam_tbl));

            // Primitives.
            auto mat_name = [](AnalyticPrim::Material m) {
                switch (m) {
                    case AnalyticPrim::Lambert:    return "lambert";
                    case AnalyticPrim::Metal:      return "metal";
                    case AnalyticPrim::Dielectric: return "dielectric";
                    case AnalyticPrim::Water:      return "water";
                }
                return "?";
            };
            toml::array prims_arr;
            for (const auto& [id, p] : primitives_) {
                toml::table t;
                t.insert("id", int64_t(id));
                t.insert("type", p.type == AnalyticPrim::Sphere ? "sphere" : "plane");
                toml::array pn{double(p.pos_or_n[0]), double(p.pos_or_n[1]), double(p.pos_or_n[2])};
                if (p.type == AnalyticPrim::Sphere) {
                    t.insert("center", std::move(pn));
                    t.insert("radius", double(p.radius_or_d));
                } else {
                    t.insert("normal", std::move(pn));
                    t.insert("d",      double(p.radius_or_d));
                }
                t.insert("material", mat_name(p.material));
                t.insert("color", toml::array{double(p.albedo[0]), double(p.albedo[1]), double(p.albedo[2])});
                t.insert("roughness", double(p.roughness));
                t.insert("ior",       double(p.ior));
                prims_arr.push_back(std::move(t));
            }
            root.insert("primitives", std::move(prims_arr));

            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f.is_open()) { out.FormatLine("scene_save: cannot open '{}'", path); return; }
            f << "# DeMonT Engine scene -- generated by scene_save\n\n";
            f << root;
            out.FormatLine("scene: saved {} primitive(s) + camera to {}", primitives_.size(), path);
        });

    C.RegisterCommand("scene_load",
        "scene_load <path.toml>: replace current camera + analytic primitives with the contents of a TOML scene file.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: scene_load <path.toml>"); return; }
            std::string path(args[0]);

            auto parsed = toml::parse_file(path);
            if (!parsed) {
                out.FormatLine("scene_load: parse error in {}: {}",
                               path, std::string(parsed.error().description()));
                return;
            }
            const toml::table& root = parsed.table();

            // Camera (optional).
            if (auto* cam_node = root.get("camera"); cam_node && cam_node->is_table()) {
                const auto& ct = *cam_node->as_table();
                if (auto pos = ct["pos"].as_array(); pos && pos->size() == 3) {
                    camera_->pos = glm::vec3(
                        float((*pos)[0].value_or(0.0)),
                        float((*pos)[1].value_or(0.0)),
                        float((*pos)[2].value_or(0.0)));
                }
                if (auto v = ct["yaw"].value<double>())   camera_->yaw     = glm::radians(float(*v));
                if (auto v = ct["pitch"].value<double>()) camera_->pitch   = glm::radians(float(*v));
                if (auto v = ct["fov"].value<double>())   camera_->fov_deg = float(*v);
                camera_->ClampPitch();
            }

            // Primitives (replaces the entire set).
            primitives_.clear();
            if (auto* prims_node = root.get("primitives"); prims_node && prims_node->is_array()) {
                for (const auto& el : *prims_node->as_array()) {
                    if (!el.is_table()) continue;
                    const auto& t = *el.as_table();
                    AnalyticPrim p{};
                    p.type = (t["type"].value_or<std::string>("sphere") == "plane")
                                 ? AnalyticPrim::Plane : AnalyticPrim::Sphere;
                    auto pn_key = (p.type == AnalyticPrim::Plane) ? "normal" : "center";
                    if (auto pn = t[pn_key].as_array(); pn && pn->size() == 3) {
                        p.pos_or_n[0] = float((*pn)[0].value_or(0.0));
                        p.pos_or_n[1] = float((*pn)[1].value_or(0.0));
                        p.pos_or_n[2] = float((*pn)[2].value_or(0.0));
                    }
                    // Motion blur (#85): seed prev == curr for freshly
                    // loaded primitives so they don't streak from the
                    // origin on their first rendered frame.
                    p.prev_pos_or_n[0] = p.pos_or_n[0];
                    p.prev_pos_or_n[1] = p.pos_or_n[1];
                    p.prev_pos_or_n[2] = p.pos_or_n[2];
                    auto r_key = (p.type == AnalyticPrim::Plane) ? "d" : "radius";
                    p.radius_or_d = float(t[r_key].value_or<double>(0.5));
                    auto mat = t["material"].value_or<std::string>("lambert");
                    if      (mat == "metal")      p.material = AnalyticPrim::Metal;
                    else if (mat == "dielectric") p.material = AnalyticPrim::Dielectric;
                    else if (mat == "water")      p.material = AnalyticPrim::Water;
                    else                          p.material = AnalyticPrim::Lambert;
                    if (auto c = t["color"].as_array(); c && c->size() == 3) {
                        p.albedo[0] = float((*c)[0].value_or(1.0));
                        p.albedo[1] = float((*c)[1].value_or(1.0));
                        p.albedo[2] = float((*c)[2].value_or(1.0));
                    }
                    p.roughness = float(t["roughness"].value_or<double>(0.0));
                    p.ior       = float(t["ior"].value_or<double>(1.5));
                    auto id = std::uint32_t(t["id"].value_or<int64_t>(int64_t(primitives_.size() + 1)));
                    primitives_[id] = p;
                }
            }
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("scene: loaded {} primitive(s) from {}", primitives_.size(), path);
        });

    if (auto* cmd = C.RegisterCommand("screenshot",
        "screenshot <name> [accum|denoise_color|bloom_mip0|swap|depth|motion]: dump the target render texture to disk. Output format is selected by r_capture_format (png|ppm); the matching extension is auto-appended to <name>, overriding any extension you typed. ACES-tonemapped for HDR inputs (accum / denoise_color / bloom_mip0); the swap target dumps the actual presented 8-bit BGRA bytes after the engine's sRGB OETF, no host-side tonemap.",
        [this](auto args, pt::console::Output& out) {
            if (!device_) { out.PrintLine("screenshot: no device"); return; }
            if (args.empty()) { out.PrintLine("usage: screenshot <name> [accum|denoise_color|bloom_mip0|swap|depth|motion]"); return; }
            std::string path(args[0]);
            std::string target = (args.size() >= 2) ? std::string(args[1]) : std::string("accum");

            // Resolve the on-disk format + path once, up front. The cvar
            // is the single source of truth -- whatever the user typed
            // (foo, foo.png, foo.ppm, foo.bar) gets normalized to
            // foo.<png|ppm> based on r_capture_format. Both the swap
            // (deferred-write) and the synchronous targets use this.
            //
            // Console::Get() rather than the enclosing scope's `C` here:
            // the lambda only captures [this], so the outer `C` would be
            // a dangling reference inside this callback. Same pattern
            // the existing exposure read-back below uses.
            using pt::engine::capture::OutputFormat;
            using pt::engine::capture::ParseOutputFormatOr;
            using pt::engine::capture::ResolveCapturePath;
            using pt::engine::capture::WriteRgb8;
            using pt::engine::capture::OutputFormatExtension;
            using pt::engine::capture::kCaptureFormatCvar;
            auto& Csh = pt::console::Console::Get();
            OutputFormat out_fmt = OutputFormat::Png;
            if (auto* fmt_cv = Csh.FindCVar(kCaptureFormatCvar)) {
                out_fmt = ParseOutputFormatOr(fmt_cv->value, OutputFormat::Png);
            }
            const std::filesystem::path resolved_path =
                ResolveCapturePath(path, out_fmt);

            // Swap target: queue an async swapchain readback. The
            // screenshot lambda runs inside Console::Drain on the main
            // render thread, BEFORE that tick's RenderFrame; if we
            // blocked here waiting for Submit to consume our request,
            // we'd deadlock (Submit can't run while this lambda holds
            // the main thread). Instead, set a pending state, fire
            // the request, and let Engine::Tick poll across frames
            // until the device returns the data.
            if (target == "swap") {
                if (!device_->SupportsSwapchainReadback()) {
                    // No backend implementation (Metal / software /
                    // a Vulkan surface that didn't advertise
                    // TRANSFER_SRC). Bail before queueing -- otherwise
                    // the engine's Tick poll loop would spin for the
                    // 5-second timeout window and only then surface
                    // the failure, which makes the screenshot
                    // command feel broken on those backends.
                    out.PrintLine("screenshot: swap target not supported on this backend (requires Vulkan with TRANSFER_SRC on the surface)");
                    return;
                }
                if (!pending_swap_screenshot_path_.empty()) {
                    out.FormatLine("screenshot: another swap capture already pending for '{}'",
                                   pending_swap_screenshot_path_);
                    return;
                }
                std::uint32_t sw = 0, sh = 0;
                pt::rhi::Device::SwapFormat fmt = pt::rhi::Device::SwapFormat::Other;
                // Latch the request (first call sets the flag + returns
                // false; subsequent polls from Engine::Tick get the
                // data). The throwaway dst buffer here is just large
                // enough to satisfy the pre-flight extent check inside
                // ReadbackSwapchain; the real memcpy happens on Tick's
                // poll into a properly-sized buffer.
                std::vector<std::uint8_t> dummy(std::size_t(accum_w_) * accum_h_ * 4u);
                (void)device_->ReadbackSwapchain(dummy.data(), dummy.size(),
                                                 &sw, &sh, &fmt);
                // Latch the resolved path + format. The deferred writer
                // in Tick uses these so the on-disk format can't drift
                // if the user toggles r_capture_format mid-flight.
                pending_swap_screenshot_path_ = resolved_path.string();
                pending_swap_screenshot_fmt_  = out_fmt;
                out.FormatLine("screenshot: queued swap capture, will write to '{}'",
                               pending_swap_screenshot_path_);
                return;
            }

            std::uint64_t tex_id = 0;
            int channels = 4;
            int bytes_per_pixel = 0;
            const char* tag = nullptr;
            if (target == "accum") {
                tex_id = accum_texture_id_;
                bytes_per_pixel = 16;   // RGBA32F
                tag = "accum_hdr";
            } else if (target == "denoise_color") {
                // denoise_color is the path tracer's linear-HDR write
                // target. Allocated when a denoiser is active OR when
                // r_bloom is on without a denoiser (the bloom-pyramid
                // input). The `tex_id == 0` check below catches the
                // off-and-bloom-off case; no need to gate on
                // denoiser_active_ here.
                tex_id = denoise_color_tex_id_;
                bytes_per_pixel = 8;    // RGBA16F
                tag = "denoise_color";
            } else if (target == "bloom_mip0") {
                // bloom_mip0 is the half-res RGBA16F bloom pyramid
                // output (BloomDown extract -> BloomUp add). Tonemap
                // samples this and adds it pre-ACES. Dumping it is
                // the cheapest way to verify the bloom path is
                // actually computing -- non-zero values in
                // bloom-bright regions = the pyramid ran; all-zero
                // = pyramid didn't run / threshold filtered everything.
                // Decode reuses the denoise_color RGBA16F path below.
                tex_id = bloom_mip_tex_id_[0];
                bytes_per_pixel = 8;    // RGBA16F (half-res of the framebuffer)
                tag = "bloom_mip0";
            } else if (target == "depth") {
                if (!denoiser_active_) { out.PrintLine("screenshot: denoiser is off"); return; }
                tex_id = depth_tex_id_;
                bytes_per_pixel = 4;    // R32F
                channels = 1;
                tag = "depth";
            } else if (target == "motion") {
                if (!denoiser_active_) { out.PrintLine("screenshot: denoiser is off"); return; }
                tex_id = motion_tex_id_;
                bytes_per_pixel = 4;    // RG16F
                channels = 2;
                tag = "motion";
            } else {
                out.FormatLine("screenshot: unknown target '{}'", target);
                return;
            }
            if (tex_id == 0) {
                out.FormatLine("screenshot: target '{}' not allocated yet", target);
                return;
            }

            std::uint32_t w = 0, hgt = 0;
            std::vector<std::uint8_t> raw(std::size_t(accum_w_) * accum_h_ * bytes_per_pixel);
            bool ok = device_->ReadbackTexture(pt::rhi::TextureHandle{tex_id},
                                               raw.data(), raw.size(), &w, &hgt);
            if (!ok || w == 0 || hgt == 0) {
                out.PrintLine("screenshot: ReadbackTexture failed");
                return;
            }

            // Pull the live exposure scalar from the GPU-resident
            // exposure_state buffer so the host-side ACES tonemap below
            // matches what the GPU paths render on screen (PathTrace's
            // inline tonemap on Vulkan, Tonemap.slang post-pass on
            // Metal -- both multiply by exposure_state[0]).  Without
            // this the PPM has no exposure applied and only matches
            // the screen when the scalar happens to be exactly 1.0.
            // Falls back to the r_exposure cvar if the readback isn't
            // implemented (e.g. software backend).  Auto-expose mode
            // tracks the GPU value live; manual mode reads back the
            // value the engine wrote, which is exactly r_exposure.
            float screenshot_exposure = 1.0f;
            if (target == "accum" || target == "denoise_color") {
                bool got = false;
                if (exposure_state_id_ != 0) {
                    float v = 1.0f;
                    if (device_->ReadbackBuffer(pt::rhi::BufferHandle{exposure_state_id_},
                                                &v, sizeof(float))) {
                        screenshot_exposure = v;
                        got = true;
                    }
                }
                if (!got) {
                    auto& Cx = pt::console::Console::Get();
                    if (auto* ev = Cx.FindCVar("r_exposure"))
                        screenshot_exposure = ev->GetFloat();
                }
            }

            // Convert to 8-bit RGB. ACES tonemap for HDR; depth gets a
            // grayscale ramp (0=black, 1=white); motion encodes (R = +x
            // mapped to [0..1], G = +y mapped, B=0) so directionality
            // is visible. Half-float decode is a portable IEEE 754
            // binary16 unpack (see lambda below) -- works on MSVC too.
            auto tonemap = [exp = screenshot_exposure](float c) -> std::uint8_t {
                c *= exp;
                const float a = 2.51f, b = 0.03f, d = 2.43f, e = 0.59f, f = 0.14f;
                float x = (c * (a * c + b)) / (c * (d * c + e) + f);
                if (x < 0.0f) x = 0.0f;
                if (x > 1.0f) x = 1.0f;
                // sRGB OETF (linear -> nonlinear) so the stored 8-bit
                // values match what the on-screen path writes to the
                // swapchain after PathTrace.slang / Tonemap.slang's
                // srgb_oetf().  Without this the PPM is linear LDR
                // and viewers (which assume sRGB) re-apply the EOTF,
                // producing a visibly darker image than what's on
                // screen.  Same piecewise OETF as the shader.
                if (x <= 0.0031308f) {
                    x = x * 12.92f;
                } else {
                    x = 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
                }
                if (x > 1.0f) x = 1.0f;
                return static_cast<std::uint8_t>(x * 255.0f + 0.5f);
            };
            // Portable IEEE 754 binary16 -> binary32 decode. Apple Clang
            // and GCC have __fp16 as an extension and we used to lean on
            // it here, but MSVC has no equivalent (and _Float16 has
            // partial support), so do the bit-shuffle manually. Handles
            // ±0, subnormals, normals, Inf/NaN.
            auto half_to_float = [](std::uint16_t h_) -> float {
                const std::uint32_t sign = (h_ >> 15) & 0x1u;
                const std::uint32_t exp  = (h_ >> 10) & 0x1Fu;
                const std::uint32_t mant = h_ & 0x3FFu;
                std::uint32_t f;
                if (exp == 0) {
                    if (mant == 0) {
                        f = sign << 31;  // signed zero
                    } else {
                        // Subnormal: renormalise into a regular float32.
                        // Half subnormals have biased_exp = 0 and decode
                        // to (-1)^s * 2^-14 * (mant / 2^10). Walk the
                        // mantissa bits left until the implicit-1 lands
                        // at bit 10 (0x400). Each shift represents a
                        // factor of 2, lowering the effective exponent
                        // by one. Final binary32 biased exponent =
                        // 127 + (p - 24) where p is the original leading
                        // bit position. With e = 1 + (10 - p) shifts,
                        // p = 11 - e, so biased exp = 114 - e =
                        // (127 - 15 - e + 2). The previous formula used
                        // +1 instead of +2 and underestimated by one
                        // exponent (subnormals decoded 2x too small).
                        std::uint32_t e = 1;
                        std::uint32_t m = mant;
                        while ((m & 0x400u) == 0) { m <<= 1; ++e; }
                        m &= 0x3FFu;
                        f = (sign << 31)
                          | ((127u - 15u - e + 2u) << 23)
                          | (m << 13);
                    }
                } else if (exp == 31) {
                    // Inf or NaN -- propagate via mantissa.
                    f = (sign << 31) | (0xFFu << 23) | (mant << 13);
                } else {
                    // Normal: rebias exponent (15 -> 127) and shift mantissa.
                    f = (sign << 31)
                      | ((exp - 15u + 127u) << 23)
                      | (mant << 13);
                }
                float result;
                std::memcpy(&result, &f, sizeof(result));
                return result;
            };

            std::vector<std::uint8_t> rgb(std::size_t(w) * hgt * 3);
            for (std::uint32_t y = 0; y < hgt; ++y) {
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::size_t pi = std::size_t(y) * w + x;
                    float r = 0, g = 0, b = 0;
                    if (target == "accum") {
                        const float* src = reinterpret_cast<const float*>(raw.data()) + pi * 4;
                        r = src[0]; g = src[1]; b = src[2];
                    } else if (target == "denoise_color" || target == "bloom_mip0") {
                        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(raw.data()) + pi * 4;
                        r = half_to_float(src[0]);
                        g = half_to_float(src[1]);
                        b = half_to_float(src[2]);
                    } else if (target == "depth") {
                        const float* src = reinterpret_cast<const float*>(raw.data()) + pi;
                        float d_v = src[0];
                        if (d_v > 1.0f) d_v = 1.0f;   // clamp far points so depth>1 still visible
                        r = g = b = d_v;
                    } else { // motion
                        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(raw.data()) + pi * 2;
                        float mx = half_to_float(src[0]);
                        float my = half_to_float(src[1]);
                        // Map a +/-32 pixel range into [0,1] each so a
                        // grey image means "no motion".
                        r = 0.5f + mx / 64.0f;
                        g = 0.5f + my / 64.0f;
                        b = 0.0f;
                    }
                    std::uint8_t* dst = rgb.data() + pi * 3;
                    if (target == "depth" || target == "motion") {
                        // No tonemap; values already in [0,1] range.
                        auto clamp01 = [](float v) -> std::uint8_t {
                            if (v < 0) v = 0;
                            if (v > 1) v = 1;
                            return std::uint8_t(v * 255.0f + 0.5f);
                        };
                        dst[0] = clamp01(r); dst[1] = clamp01(g); dst[2] = clamp01(b);
                    } else {
                        dst[0] = tonemap(r); dst[1] = tonemap(g); dst[2] = tonemap(b);
                    }
                }
            }

            if (!WriteRgb8(resolved_path, rgb.data(), w, hgt, out_fmt)) {
                out.FormatLine("screenshot: cannot write '{}' (format={})",
                               resolved_path.string(),
                               OutputFormatExtension(out_fmt));
                return;
            }
            out.FormatLine("screenshot: wrote {} ({}x{} {}, format={})",
                           resolved_path.string(), w, hgt, tag,
                           OutputFormatExtension(out_fmt));
            (void)channels;
        })) {
        // No extension: ResolveCapturePath auto-appends .png / .ppm
        // based on r_capture_format. Keeps the ghost-suggestion in
        // sync with the active format.
        cmd->default_args = "demonte_screen";
    }

    // frame_info: read-only peek at the engine's monotonic frame counter
    // (the value r_capture_frame_at is measured against) plus the
    // backend + r_spp + current resolution + whatever's armed on the
    // capture state machine. Lets an interactive operator answer
    // "what frame am I on?" without having to peek at the perf overlay
    // (which doesn't surface frame_index anyway).
    C.RegisterCommand("frame_info",
        "Print the engine's monotonic frame index, current backend, "
        "r_spp, render extent, and any armed FrameCapture state. "
        "Use `capture_in N` to arm a one-shot relative to the current "
        "frame; r_capture_frame_at is absolute (cold-start zero) and "
        "pairs with r_capture_seed for deterministic capture across "
        "runs.",
        [this](auto, pt::console::Output& out) {
            auto& Cf = pt::console::Console::Get();
            int spp = 0;
            if (auto* sv = Cf.FindCVar("r_spp")) spp = sv->GetInt();
            std::string armed;
            if (auto* v = Cf.FindCVar("r_capture_frame_at")) {
                const int at = v->GetInt();
                if (at > 0) armed += fmt::format(" one_shot_at={}", at);
            }
            if (auto* v = Cf.FindCVar("r_capture_seq")) {
                if (!v->value.empty()) armed += fmt::format(" seq=\"{}\"", v->value);
            }
            out.FormatLine(
                "frame_index={} backend={} r_spp={} accum={}x{}{}",
                frame_index_,
                pt::rhi::BackendName(current_backend_),
                spp,
                accum_w_, accum_h_,
                armed.empty() ? std::string{} : (" armed:" + armed));
        });

    // capture_in: ergonomic relative form of r_capture_frame_at. Same
    // machinery (one-shot arm + auto-disarm + format dispatch + retry-
    // on-readback-failure) -- this command just resolves "now + N" to
    // an absolute frame_index and writes the cvar. Kept separate from
    // r_capture_frame_at so the determinism contract on the cvar
    // (absolute frame index, pairs with r_capture_seed for bitwise-
    // identical output across runs) stays intact.
    C.RegisterCommand("capture_in",
        "capture_in <N>: arm a one-shot frame capture for N frames "
        "from now (N >= 1). Convenience wrapper around r_capture_frame_at; "
        "honours r_capture_format and r_capture_seed exactly like the "
        "cvar-driven path.",
        [this](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine("usage: capture_in <N>");
                return;
            }
            // std::from_chars keeps us out of std::stoi's throw-on-fail
            // semantics (project builds with -fno-exceptions / /EHs-c-).
            // Rejects empty, signed, non-digit input.
            const std::string s_arg(args[0]);
            int n = 0;
            auto pr = std::from_chars(s_arg.data(),
                                      s_arg.data() + s_arg.size(), n);
            if (pr.ec != std::errc() ||
                pr.ptr != s_arg.data() + s_arg.size() ||
                n <= 0) {
                out.FormatLine("capture_in: N must be a positive integer "
                               "(got '{}')", args[0]);
                return;
            }
            const std::uint32_t target =
                frame_index_ + static_cast<std::uint32_t>(n);
            // SetCVarOverride writes the value AND fires on_change, so the
            // r_capture_frame_at handler installed in RegisterCommands'
            // post-cfg pass arms FrameCapture::SetOneShotFrame in one go.
            // The on_change is also what the engine watches to reset the
            // cvar surface back to "0" after the capture fires (see the
            // MaybeCapture path in Tick).
            auto& Cf = pt::console::Console::Get();
            Cf.SetCVarOverride("r_capture_frame_at", std::to_string(target));
            std::string fmt_name = "png";
            if (auto* fv = Cf.FindCVar("r_capture_format")) {
                fmt_name = fv->value;
            }
            out.FormatLine("capture_in: armed one-shot at frame {} "
                           "(current={}, +{}); format={}",
                           target, frame_index_, n, fmt_name);
        });

    C.RegisterCommand("web_console",
        "Open the web console in the default browser.",
        [this](auto, pt::console::Output& out) {
            OpenWebConsole();
            out.PrintLine("opening web console...");
        });

    // Auto-focus: trace a ray through the screen centre and write the
    // hit distance into r_dof_focal_distance. Like tap-to-focus on a
    // phone camera. CPU-side intersection against the analytic
    // primitives only -- mesh hits aren't checked yet (would need a
    // CPU BVH or a GPU depth_tex read-back). In practice the default
    // scene uses analytic primitives so this works for the common
    // case; extend later when mesh-aware focus matters.
    C.RegisterCommand("dump_sun_uv",
        "Print where the engine thinks the sun lands on screen "
        "(used by r_lens_flare_mode = sun). Sentinel like (-2, -2) "
        "means the sun is behind the camera.",
        [this](auto, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("no camera"); return; }
            const auto& cam = *camera_;
            const glm::vec3 fwd   = cam.Forward();
            const glm::vec3 right = cam.Right();
            const glm::vec3 up    = cam.Up();
            auto& C2 = pt::console::Console::Get();
            float elev = 0.0f, azim = 0.0f;
            if (auto* v = C2.FindCVar("r_sun_elevation")) elev = v->GetFloat();
            if (auto* v = C2.FindCVar("r_sun_azimuth"))   azim = v->GetFloat();
            float er = glm::radians(elev), ar = glm::radians(azim);
            glm::vec3 sun{
                std::cos(er) * std::sin(ar),
                std::sin(er),
                -std::cos(er) * std::cos(ar),
            };
            float fwd_dot = glm::dot(sun, fwd);
            float ft = cam.FovYTan();
            float aspect = 16.0f / 9.0f;
            if (auto* v = C2.FindCVar("app_window_width")) {
                int w = v->GetInt();
                if (auto* h = C2.FindCVar("app_window_height")) {
                    int hh = h->GetInt();
                    if (hh > 0) aspect = float(w) / float(hh);
                }
            }
            if (fwd_dot <= 1e-3f) {
                out.FormatLine("sun is behind/sideways: fwd_dot={:.3f} (sentinel)",
                               fwd_dot);
                return;
            }
            float xs = (glm::dot(sun, right) / fwd_dot) / (ft * aspect);
            float ys = (glm::dot(sun, up)    / fwd_dot) / ft;
            float ux = xs * 0.5f + 0.5f;
            float uy = -ys * 0.5f + 0.5f;
            out.FormatLine("sun_dir=({:.3f},{:.3f},{:.3f}) "
                           "fwd_dot={:.3f} screen_uv=({:.3f},{:.3f}) {}",
                           sun.x, sun.y, sun.z, fwd_dot, ux, uy,
                           (ux >= 0 && ux <= 1 && uy >= 0 && uy <= 1)
                               ? "(on-screen)" : "(off-screen)");
        });

    C.RegisterCommand("undo",
        "Roll back the most recent cvar change (or semicolon-bundle "
        "of changes) made via the console. Pre-transaction values "
        "are restored as a single atomic step; the rollback itself "
        "is recorded for redo. Output lists every cvar reverted with "
        "before -> after values. History stack holds the last 50 "
        "transactions.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            auto changes = C2.Undo();
            if (changes.empty()) {
                out.PrintLine("undo: history is empty");
                return;
            }
            out.FormatLine("undo: rolled back {} cvar change{}",
                           changes.size(), changes.size() == 1 ? "" : "s");
            for (const auto& c : changes) {
                out.FormatLine("  {}: \"{}\" -> \"{}\"", c.name, c.from, c.to);
            }
        });

    C.RegisterCommand("redo",
        "Reapply the most recently undone cvar change (or bundle). "
        "Output lists every cvar reapplied with before -> after "
        "values. Cleared by any new edit.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            auto changes = C2.Redo();
            if (changes.empty()) {
                out.PrintLine("redo: nothing to redo");
                return;
            }
            out.FormatLine("redo: reapplied {} cvar change{}",
                           changes.size(), changes.size() == 1 ? "" : "s");
            for (const auto& c : changes) {
                out.FormatLine("  {}: \"{}\" -> \"{}\"", c.name, c.from, c.to);
            }
        });

    C.RegisterCommand("dump_moon_pos",
        "Print the engine's computed moon position (alt/az/phase) "
        "and whether the engine thinks it should be lighting the "
        "scene. Use to debug 'no moon visible' or 'no shadow' issues.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            bool astro_on = false;
            if (auto* v = C2.FindCVar("r_sky_use_astronomical")) astro_on = v->GetBool();
            if (!astro_on) {
                out.PrintLine("r_sky_use_astronomical = 0 -> moon disabled.");
                out.PrintLine("Set r_sky_use_astronomical 1 to enable.");
                return;
            }
            // Compute moon at the engine's current effective JD.
            float hour = 12.0f;
            if (auto* v = C2.FindCVar("r_sky_hour")) hour = v->GetFloat();
            bool local = true;
            if (auto* v = C2.FindCVar("r_sky_hour_local")) local = v->GetBool();
            float tz = 0.0f;
            if (local) {
                if (auto* v = C2.FindCVar("r_sky_tz_offset_hours")) tz = v->GetFloat();
            }
            int yr = 0, mo = 0, dy = 0;
            if (auto* v = C2.FindCVar("r_sky_year"))  yr = v->GetInt();
            if (auto* v = C2.FindCVar("r_sky_month")) mo = v->GetInt();
            if (auto* v = C2.FindCVar("r_sky_day"))   dy = v->GetInt();
            if (yr == 0 || mo == 0 || dy == 0) {
                const std::time_t now = std::time(nullptr);
                std::tm gm = *std::gmtime(&now);
                if (yr == 0) yr = gm.tm_year + 1900;
                if (mo == 0) mo = gm.tm_mon + 1;
                if (dy == 0) dy = gm.tm_mday;
            }
            const float hour_utc = hour - tz;
            const double jd_midnight = pt::astro::julianDateFromUtc(yr, mo, dy, 0, 0, 0.0);
            const double jd = jd_midnight + double(hour_utc) / 24.0;
            double lat = 13.0827, lon = 80.2707;
            if (auto* v = C2.FindCVar("r_sky_lat")) lat = v->GetFloat();
            if (auto* v = C2.FindCVar("r_sky_lon")) lon = v->GetFloat();
            auto moon_eq = pt::astro::moonPosition(jd);
            auto moon_h  = pt::astro::equatorialToHorizon(moon_eq, lat, lon, jd);
            auto sun_eq  = pt::astro::sunPosition(jd);
            auto sun_h   = pt::astro::equatorialToHorizon(sun_eq, lat, lon, jd);
            double phase = pt::astro::moonPhaseAngle(sun_eq, moon_eq);
            double phase_deg = phase * 180.0 / 3.14159265358979;
            const char* lit = (moon_h.altitude_deg > 0.0) ? "ABOVE horizon"
                                                          : "below horizon";
            const char* sun_state = (sun_h.altitude_deg > 0.0) ? "above" : "below";
            out.FormatLine("Date {}-{:02}-{:02} {:.1f}h local (UTC {:.1f}h)",
                           yr, mo, dy, hour, hour_utc);
            out.FormatLine("Observer {:.4f} N, {:.4f} E", lat, lon);
            out.FormatLine("Moon: alt {:.2f} deg, az {:.2f} deg ({})",
                           moon_h.altitude_deg, moon_h.azimuth_deg, lit);
            out.FormatLine("Phase: {:.1f} deg from new (0=new, 180=full); "
                           "phase_brightness ~ {:.2f}",
                           phase_deg, 0.5 * (1.0 - std::cos(phase)));
            out.FormatLine("Sun: alt {:.2f} ({} horizon)",
                           sun_h.altitude_deg, sun_state);
            out.PrintLine("If alt > 0, phase_brightness > 0, sun below horizon, ");
            out.PrintLine("you should be seeing moonlight contribution. If not,");
            out.PrintLine("the issue is in scene/exposure, not astronomical math.");
        });

    C.RegisterCommand("dof_focus_here",
        "Auto-focus DOF on whatever's at the centre of the screen. "
        "Writes the hit distance into r_dof_focal_distance and turns "
        "r_dof on if it wasn't already.",
        [this](auto args, pt::console::Output& out) {
            (void)args;
            if (!camera_) { out.PrintLine("no camera"); return; }
            const auto& cam = *camera_;
            const glm::vec3 ro = cam.pos;
            const glm::vec3 rd = cam.Forward();   // already unit length

            float best_t = 1e30f;
            for (const auto& [id, p] : primitives_) {
                float t = best_t;
                if (p.type == AnalyticPrim::Sphere) {
                    // Standard ray-sphere intersection.
                    glm::vec3 c{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
                    glm::vec3 oc = ro - c;
                    float b = glm::dot(oc, rd);
                    float disc = b * b - (glm::dot(oc, oc) - p.radius_or_d * p.radius_or_d);
                    if (disc < 0.0f) continue;
                    float sq = std::sqrt(disc);
                    float t0 = -b - sq;
                    float t1 = -b + sq;
                    t = (t0 > 1e-3f) ? t0 : ((t1 > 1e-3f) ? t1 : best_t);
                } else { // Plane
                    glm::vec3 n{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
                    float ndotd = glm::dot(n, rd);
                    if (std::fabs(ndotd) < 1e-6f) continue;
                    float th = -(glm::dot(n, ro) + p.radius_or_d) / ndotd;
                    if (th > 1e-3f) t = th;
                }
                if (t < best_t) best_t = t;
            }

            if (best_t >= 1e29f) {
                out.PrintLine("dof_focus_here: nothing in the centre of the "
                              "frame (no analytic primitive hit). Mesh hits "
                              "aren't checked yet.");
                return;
            }

            auto& C2 = pt::console::Console::Get();
            C2.SetCVarOverride("r_dof_focal_distance", std::to_string(best_t));
            C2.SetCVarOverride("r_dof", "1");
            accum_dirty_ = true;
            out.FormatLine("dof_focus_here: focal distance set to {:.3f} (r_dof on)",
                           best_t);
        });

    C.RegisterCommand("toggle",
        "toggle <cvar>: cycle a cvar through its allowed_values (great for A/B testing).",
        [](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: toggle <cvar>"); return; }
            auto& C2 = pt::console::Console::Get();
            auto* v = C2.FindCVar(args[0]);
            if (v == nullptr) { out.FormatLine("toggle: unknown cvar '{}'", args[0]); return; }
            if (v->allowed_values.empty()) {
                out.FormatLine("toggle: '{}' has no allowed_values; can't cycle", v->name);
                return;
            }
            // Find the current value's index in allowed_values; advance one
            // (with wrap). If the current value isn't in the list (rare),
            // pick the first allowed value.
            std::size_t cur = v->allowed_values.size();   // sentinel
            for (std::size_t i = 0; i < v->allowed_values.size(); ++i) {
                if (v->allowed_values[i] == v->value) { cur = i; break; }
            }
            std::size_t next = (cur >= v->allowed_values.size()) ? 0
                                                                  : (cur + 1) % v->allowed_values.size();
            const std::string& chosen = v->allowed_values[next];
            std::string before = v->value;
            C2.SetCVarOverride(v->name, chosen);
            out.FormatLine("{} : \"{}\" -> \"{}\"", v->name, before, chosen);
        });

    C.RegisterCommand("exec", "Run a .cfg script: exec autoexec.cfg",
        [](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine("usage: exec <file>");
                return;
            }
            std::string path(args[0]);
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { out.FormatLine("cannot open: {}", path); return; }
            std::string body;
            char buf[4096];
            while (auto n = std::fread(buf, 1, sizeof(buf), f)) body.append(buf, n);
            std::fclose(f);
            auto r = pt::console::Console::Get().ExecuteScript(body);
            if (!r.output.empty()) out.Print(r.output);
            if (!r.ok) out.FormatLine("exec error: {}", r.error);
        });

    // r_perf_overlay: validate + push level changes to the overlay.
    if (auto* v = C.FindCVar("r_perf_overlay")) {
        v->allowed_values = {"0", "1", "2", "3"};
        v->on_change = [this](const pt::console::CVar& cv) {
            if (perf_overlay_) {
                int lv = std::atoi(cv.value.c_str());
                bool rhi = false;
                if (auto* mv = pt::console::Console::Get().FindCVar("r_perf_overlay_mode")) {
                    rhi = (mv->value == "rhi");
                }
                // Native overlay only takes the level when RHI mode is off;
                // otherwise it stays hidden so both don't draw at once.
                perf_overlay_->SetLevel(rhi ? 0 : lv);
            }
        };
    }
    if (auto* v = C.FindCVar("r_perf_overlay_mode")) {
        v->allowed_values = {"native", "rhi"};
        v->on_change = [this](const pt::console::CVar& cv) {
            if (perf_overlay_) {
                int lv = 0;
                if (auto* lvv = pt::console::Console::Get().FindCVar("r_perf_overlay")) {
                    lv = std::atoi(lvv->value.c_str());
                }
                perf_overlay_->SetLevel(cv.value == "rhi" ? 0 : lv);
            }
        };
    }
    // r_backend: full canonical value set with per-value platform
    // tags (issue #161). metal is Mac-only; vulkan and software are
    // cross-platform. Listing the full set means a shared demont.cfg
    // with `r_backend metal` round-trips cleanly on Windows, just no-
    // ops with a clear platform-mismatch error instead of the older
    // "invalid value" message.
    if (auto* v = C.FindCVar("r_backend")) {
        v->allowed_values        = {"none", "software", "metal", "vulkan"};
        v->allowed_value_flags   = {
            pt::console::CVAR_VALUE_ANY,    // none
            pt::console::CVAR_VALUE_ANY,    // software
            pt::console::CVAR_VALUE_MAC,    // metal
            pt::console::CVAR_VALUE_ANY,    // vulkan (cross-platform; MoltenVK on Mac)
        };
        v->on_change = [this](const pt::console::CVar& cv) {
            BackendType t = BackendType::None;
            if      (cv.value == "software") t = BackendType::Software;
            else if (cv.value == "metal")    t = BackendType::Metal;
            else if (cv.value == "vulkan")   t = BackendType::Vulkan;
            RequestBackendSwitch(t);
        };
    }
    // r_software_blit: validate the value across all platforms (so cfg
    // load doesn't bounce a saved value back to default on Mac/Linux),
    // but only emit the platform-specific WARN/INFO chatter on Win32
    // -- the cvar is documented as a no-op on Mac/Linux, so users
    // shouldn't see "selected, takes effect, beware DXGI lockout"
    // messages there.
    if (auto* v = C.FindCVar("r_software_blit")) {
        v->allowed_values      = {"vulkan", "gdi"};
        // gdi is Win-only; vulkan works everywhere (issue #161).
        v->allowed_value_flags = {
            pt::console::CVAR_VALUE_ANY,    // vulkan
            pt::console::CVAR_VALUE_WIN,    // gdi
        };
#if defined(_WIN32)
        v->on_change = [this](const pt::console::CVar& cv) {
            if (cv.value == "gdi" &&
                current_backend_ == BackendType::Vulkan) {
                LOG_WARN("r_software_blit=gdi -- if you switch r_backend "
                         "from vulkan to software in this session, the "
                         "window will be permanently stuck on the last "
                         "Vulkan frame (Microsoft DXGI flip-model lockout, "
                         "spec-defined behaviour). Restart the app to use "
                         "GDI cleanly, or set r_software_blit vulkan to "
                         "switch backends without restarting.");
            } else if (cv.value == "gdi") {
                LOG_INFO("r_software_blit=gdi -- legacy GDI present path "
                         "selected. Safe for software-fresh-start sessions; "
                         "do NOT switch from vulkan->software in this "
                         "session without restarting (Microsoft DXGI "
                         "flip-model lockout). Takes effect on next "
                         "Software backend (re)initialise.");
            } else {
                LOG_INFO("r_software_blit=vulkan -- Vulkan-blit present "
                         "path selected (default). Survives all backend "
                         "switches. Takes effect on next Software backend "
                         "(re)initialise.");
            }
        };
#endif
    }
    // r_software_blit_recreate: validate cross-platform (so cfg load
    // doesn't bounce a saved value back to default on Mac/Linux), but
    // only emit Win32-specific info chatter on_change -- the cvar is
    // documented as a no-op on Mac/Linux so users shouldn't see
    // "auto/prompt/warn behaviour" messages there.
    if (auto* v = C.FindCVar("r_software_blit_recreate")) {
        v->allowed_values = {"auto", "prompt", "warn"};
#if defined(_WIN32)
        v->on_change = [](const pt::console::CVar& cv) {
            if (cv.value == "auto") {
                LOG_INFO("r_software_blit_recreate=auto -- vulkan -> software "
                         "with r_software_blit=gdi will recreate the GLFW "
                         "window in-process (escape DXGI flip-model lockout, "
                         "preserve engine state). Brief flicker at the switch. "
                         "Takes effect on next such switch.");
            } else if (cv.value == "prompt") {
                LOG_INFO("r_software_blit_recreate=prompt -- vulkan -> software "
                         "with r_software_blit=gdi will pop a MessageBox "
                         "offering to restart the process. Yes spawns a fresh "
                         "demont.exe with the original argv; No falls back to "
                         "the warn behaviour. Takes effect on next such switch.");
            } else {
                LOG_INFO("r_software_blit_recreate=warn -- vulkan -> software "
                         "with r_software_blit=gdi will emit a LOG_WARN and "
                         "leave the window stuck on the last Vulkan frame "
                         "(legacy behaviour); user must manually quit and "
                         "relaunch. Takes effect on next such switch.");
            }
        };
#endif
    }
    // r_theme: validate + push the new theme to every WS client so the
    // browser console flips live without a page reload.
    if (auto* v = C.FindCVar("r_theme")) {
        v->allowed_values = {"hardcore", "amber", "synthwave",
                             "matrix", "vault", "sakura", "mono"};
        v->on_change = [this](const pt::console::CVar& cv) {
            if (server_) {
                auto data = fmt::format(R"({{"name":"{}"}})", cv.value);
                server_->BroadcastEvent("theme_change", data);
            }
            if (overlay_) overlay_->ApplyTheme(cv.value);
            if (perf_overlay_) perf_overlay_->ApplyTheme(cv.value);
        };
    }
    // con_font_scale: cvar IS the source of truth (the in-game Win32
    // overlay reads the live cvar value inside its EnsureFontScale()
    // poll, called from Paint()). But Paint() only fires on WM_PAINT,
    // and a web-GUI cvar set doesn't trigger one -- so without this
    // on_change wiring, the user types con_font_scale 1.5 in the web
    // console, the cvar updates, but the in-game overlay stays at the
    // old size until something else triggers a paint (typing, scroll,
    // animation timer). Pinging Repaint here closes that gap: any
    // cvar writer (web GUI, in-game console, autoexec.cfg, CLI args)
    // gets the same instant-feedback behaviour. No copy-paste of
    // values -- the overlay re-reads the cvar inside EnsureFontScale.
    if (auto* v = C.FindCVar("con_font_scale")) {
        v->on_change = [this](const pt::console::CVar&) {
            if (overlay_) overlay_->Repaint();
        };
    }
    // r_perf_overlay_scale: same web-GUI live-propagation pattern as
    // con_font_scale above. Win32 PerfOverlay's EnsureScale() polls
    // the cvar inside Paint(); without an on_change kick from the
    // setter, web GUI changes go invisible until the per-frame Update
    // tick lands a redraw. Pinging Repaint here forces the next
    // Paint immediately.
    if (auto* v = C.FindCVar("r_perf_overlay_scale")) {
        v->on_change = [this](const pt::console::CVar&) {
            if (perf_overlay_) perf_overlay_->Repaint();
        };
    }
    // dev_log_level: validate
    if (auto* v = C.FindCVar("dev_log_level")) {
        v->allowed_values = {"error", "warn", "info", "debug"};
    }
    // r_diagnostic_level: validate + mirror into pt::diag::g_diag_level so
    // every PT_DIAG_TIERn() callsite is a single relaxed atomic load +
    // compare. The on_change handler emits the transition through
    // LOG_INFO directly (NOT PT_DIAG_TIER1) so the user sees the gate
    // flip in the same console even when transitioning *to* tier 0.
    if (auto* v = C.FindCVar("r_diagnostic_level")) {
        v->allowed_values = {"0", "1", "2", "3"};
        v->slider_min  = 0.0f;
        v->slider_max  = 3.0f;
        v->slider_step = 1.0f;
        v->on_change = [](const pt::console::CVar& cv) {
            int n = std::clamp(cv.GetInt(), 0, 3);
            pt::diag::g_diag_level.store(n, std::memory_order_relaxed);
            LOG_INFO("[engine.cvars] r_diagnostic_level={} ({})", n,
                     n == 0 ? "off"
                   : n == 1 ? "state-transition"
                   : n == 2 ? "per-frame summary"
                            : "per-call detail");
        };
        // Sync the cache to whatever the cvar holds right now (e.g.
        // demont.cfg may have already overridden the default before this
        // registration block ran during Engine::Init -- although today
        // RegisterCommands runs before the cfg load, future reorders
        // shouldn't break the invariant).
        pt::diag::g_diag_level.store(std::clamp(v->GetInt(), 0, 3),
                                     std::memory_order_relaxed);
    }
    // r_capture_frame_at: arm the FrameCapture one-shot trigger. Cleared
    // back to 0 after the capture fires (FrameCapture also clears the
    // internal armed flag; this just keeps the cvar surface honest so a
    // re-set of the same value re-fires the on_change).
    if (auto* v = C.FindCVar("r_capture_frame_at")) {
        v->on_change = [](const pt::console::CVar& cv) {
            int n = cv.GetInt();
            if (n < 0) n = 0;
            pt::engine::capture::SetOneShotFrame(static_cast<std::uint32_t>(n));
        };
    }
    // r_capture_seq: parse "<prefix> <N> <interval>" (positional, space-
    // separated) and arm the sequence. Empty string disables.
    if (auto* v = C.FindCVar("r_capture_seq")) {
        v->on_change = [](const pt::console::CVar& cv) {
            const std::string& s = cv.value;
            if (s.empty()) {
                pt::engine::capture::StartSequence("", 0, 0);
                return;
            }
            // Tokenise on whitespace -- duplicate of TokenizeLine logic
            // would be overkill for three positional fields. The cvar's
            // value is already de-quoted by Console::Execute, so just
            // walk the string.
            std::vector<std::string> toks;
            std::string cur;
            for (char c : s) {
                if (c == ' ' || c == '\t') {
                    if (!cur.empty()) { toks.push_back(std::move(cur)); cur.clear(); }
                } else {
                    cur.push_back(c);
                }
            }
            if (!cur.empty()) toks.push_back(std::move(cur));
            if (toks.size() < 3) {
                LOG_WARN("r_capture_seq: expected 'prefix N interval' (got {} token(s)); "
                         "sequence disabled", toks.size());
                pt::engine::capture::StartSequence("", 0, 0);
                return;
            }
            int n        = std::atoi(toks[1].c_str());
            int interval = std::atoi(toks[2].c_str());
            if (n <= 0 || interval <= 0) {
                LOG_WARN("r_capture_seq: N and interval must be positive (got N={} interval={}); "
                         "sequence disabled", n, interval);
                pt::engine::capture::StartSequence("", 0, 0);
                return;
            }
            pt::engine::capture::StartSequence(
                toks[0],
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(interval));
        };
    }
    // r_capture_seed: when non-zero, reset frame_index_ to this value
    // and force an accum reset, so two demont.exe launches with
    // identical settings + the same seed produce bitwise-identical noise
    // patterns at the same r_capture_frame_at. Path tracer's PRNG seeds
    // purely from (pixel_id, frame_index) so this is the single knob
    // we need (see PathTrace.slang's pcgHash line ~1047). The on_change
    // handler captures `this` to mutate the engine's frame counter +
    // accum-dirty flag, which lives in the cvar registration's
    // RegisterCommands lane.
    if (auto* v = C.FindCVar("r_capture_seed")) {
        v->on_change = [this](const pt::console::CVar& cv) {
            int n = cv.GetInt();
            if (n < 0) n = 0;
            // Documented contract: 0 = let frame_index_ run naturally.
            // No-op so re-setting to 0 doesn't surprise the operator
            // by trashing accum + dropping a pending capture.
            if (n == 0) return;
            frame_index_ = static_cast<std::uint32_t>(n);
            accum_dirty_ = true;
            // Drop any pending capture state too -- a seed reset
            // implies the operator wants a clean baseline.
            pt::engine::capture::Reset();
            LOG_INFO("[capture] r_capture_seed={} -- frame_index reset, "
                     "accum cleared, pending captures dropped", n);
        };
    }
    if (auto* v = C.FindCVar("r_denoiser")) {
        // metalfx is Mac-only; svgf_basic / svgf_atrous / nrd / optix_*
        // are Vulkan-only. The RenderFrame check downgrades incompatible
        // (cvar, backend) combinations to off rather than rejecting
        // at console-set time -- so a user can have `r_denoiser
        // svgf_atrous` in their demont.cfg and still launch on a Mac
        // without the cfg getting rejected.
        //
        // optix_hdr is wired up via VulkanOptixDenoiser (gated by the
        // build-time PT_ENABLE_OPTIX flag and runtime CUDA + OptiX
        // detection -- see VulkanDevice::Denoise's OptiX branch).
        // optix_hdr_aov is currently a synonym for optix_hdr until
        // the AOV variant lands in Phase 1a step 3 alongside the path
        // tracer's primary_albedo output (the underlying OptiX path
        // logs a one-time INFO at Init noting the fallback).
        v->allowed_values = {"off", "metalfx",
                              "svgf_basic", "svgf_atrous", "nrd",
                              "svgf_basic_metalfx", "svgf_atrous_metalfx",
                              "optix_hdr", "optix_hdr_aov",
                              "optix_temporal_hdr", "optix_temporal_hdr_aov"};
        // Per-value platform tags (issue #161). Mac path is metalfx +
        // its SVGF composites; Win/Linux path is nrd + the optix
        // family. SVGF (basic/atrous) is cross-platform. A user with
        // `r_denoiser nrd` in their demont.cfg on Mac sees a clear
        // platform-mismatch error listing Mac-only values instead
        // of getting silently downgraded at RenderFrame time.
        constexpr auto kAny = pt::console::CVAR_VALUE_ANY;
        constexpr auto kMac = pt::console::CVAR_VALUE_MAC;
        constexpr auto kWL  = pt::console::CVAR_VALUE_WIN | pt::console::CVAR_VALUE_LINUX;
        v->allowed_value_flags = {
            kAny,   // off
            kMac,   // metalfx
            kAny,   // svgf_basic
            kAny,   // svgf_atrous
            kWL,    // nrd
            kMac,   // svgf_basic_metalfx
            kMac,   // svgf_atrous_metalfx
            kWL,    // optix_hdr
            kWL,    // optix_hdr_aov
            kWL,    // optix_temporal_hdr
            kWL,    // optix_temporal_hdr_aov
        };
    }
    if (auto* v = C.FindCVar("r_hdr_pipeline")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // r_star_split (issue #46): toggling at runtime changes whether
    // PathTrace subtracts the celestials (stars + sun + moon) from
    // the primary-miss sky term. The denoiser's history buffers a
    // few frames of the WITH-celestials sky on legacy and a few
    // WITHOUT on the composite path, so flipping mid-flight bleeds
    // stale samples for a moment -- reset accum_dirty_ to discard
    // the old history cleanly. (The stateless StarsComposite kernel
    // itself has no history to reset.)
    if (auto* v = C.FindCVar("r_star_split")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) {
            accum_dirty_ = true;
        };
    }
    if (auto* v = C.FindCVar("r_bloom")) {
        v->allowed_values = {"0", "1"};
    }
    if (auto* v = C.FindCVar("r_lens_flare")) {
        v->allowed_values = {"0", "1"};
    }
    // Fluid Phase 1 (#136): r_smoke_enabled is a strict bool. Matches
    // the registration pattern used by the other renderer toggles
    // above so the toggle / cvar-validation chain can treat it
    // uniformly. r_smoke_max_emitters is an integer in [0,16] and
    // stays unconstrained here (host-clamps at upload time).
    if (auto* v = C.FindCVar("r_smoke_enabled")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_lens_flare_mode")) {
        v->allowed_values = {"sun", "image"};
    }
    if (auto* v = C.FindCVar("r_debug_sun_overlay")) {
        v->allowed_values = {"0", "1"};
    }
    // Bloom + flare intensity / threshold / mips / count / dispersion
    // don't reset accumulation: they're applied in the post-tonemap
    // pass each frame, no dependency on path-tracer state.
    if (auto* v = C.FindCVar("r_caustics")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_refract_bounces")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // --- SDF Phase 2 (#98) -------------------------------------------------
    // r_sdf_normal_mode switches between forward-AD (0) and central
    // differences (1). The two paths produce subtly different normals
    // on procedural clusters (forward-AD is continuous, central is
    // tap-quantised), so flipping mid-flight bleeds the old normal
    // field's samples into the new one's accumulator until something
    // else dirties accum. Match the other renderer-affecting cvars and
    // reset accum on change.
    if (auto* v = C.FindCVar("r_sdf_normal_mode")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
        // Megakernel-perf gate (issue #98 follow-up): emit a [warn] at
        // startup if the user wants forward-AD normals (r_sdf_normal_mode
        // == 0) but the build doesn't ship the Dual3 autodiff path
        // (PT_SDF_AUTODIFF gate OFF in cmake/Slang.cmake). The shader
        // silently falls back to central differences in that case --
        // visually similar on the procedural-light defaults but worth
        // flagging so the user knows to rebuild with
        // -DPT_SDF_AUTODIFF=ON if they care about the normal-field
        // continuity that forward-AD provides.
#if !PT_SDF_AUTODIFF
        if (v->GetInt() == 0) {
            LOG_WARN("[sdf] r_sdf_normal_mode=0 (forward-AD) requested but "
                     "this build was compiled with PT_SDF_AUTODIFF=OFF; "
                     "shader falls back to 6-tap central differences. "
                     "Rebuild with -DPT_SDF_AUTODIFF=ON and "
                     "-DPT_SDF_PROCEDURAL_OPS=ON to restore the forward-AD "
                     "path. (Procedural-op presence alone costs ~6 ms/"
                     "frame on the default scene; the gate is OFF by "
                     "default.)");
        }
#endif
#if !PT_SDF_PROCEDURAL_OPS
        // Single-shot info hint: the host's sdf_displace_noise /
        // sdf_twist / etc. console commands stay registered (they're
        // valuable for tests + host-side wire-format parsing) but
        // any cluster queued via those falls into sdfClusterDist's
        // 1e30 sentinel branch on the shader side and disappears.
        // Surface that up so a user who unwraps the gate at runtime
        // (vs. the build flag) gets a clear breadcrumb.
        LOG_INFO("[sdf] PT_SDF_PROCEDURAL_OPS=OFF: procedural-op SDF "
                 "clusters (sdf_displace_noise / twist / bend / repeat / "
                 "repeat-limited) are skipped by the sphere-trace. "
                 "Rebuild with -DPT_SDF_PROCEDURAL_OPS=ON to enable.");
#endif
    }
    // r_sdf_displace_octaves changes the number of FBM octaves applied
    // by sdf_displace_noise -- the displaced surface shifts as soon as
    // a different octave count is sampled, so reset accum too.
    if (auto* v = C.FindCVar("r_sdf_displace_octaves")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // --- end SDF Phase 2 ---------------------------------------------------
    if (auto* v = C.FindCVar("r_dof")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_aperture")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_focal_distance")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_blades")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_volumetric")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    for (const char* n : {"r_volumetric_density", "r_volumetric_anisotropy",
                          "r_volumetric_intensity", "r_volumetric_samples"}) {
        if (auto* v = C.FindCVar(n)) {
            v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
        }
    }
    // Cloud cvars: every change resets accum so the temporal mean
    // doesn't smear the old field into the new one.
    if (auto* v = C.FindCVar("r_clouds")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar& cv) {
            accum_dirty_ = true;
            // One-shot hint (closed issue #178): r_clouds=1 +
            // r_accum_ema_alpha=0 produces persistent noise because
            // the cloud-wind dirty trigger keeps resetting the legacy
            // running-mean accumulator every frame. The cvar
            // docstring documents this but users still tripped on
            // it during smoke pass. Surface the hint at the moment
            // the user enables clouds so the foot-gun is signposted
            // exactly when they'd hit it.
            static bool hint_shown = false;
            if (!hint_shown && cv.GetBool()) {
                float alpha = 0.0f;
                if (auto* a = pt::console::Console::Get().FindCVar("r_accum_ema_alpha")) {
                    alpha = a->GetFloat();
                }
                if (alpha == 0.0f) {
                    LOG_INFO("[hint] r_clouds=1 + r_accum_ema_alpha=0: "
                             "cloud wind drift will reset the accumulator "
                             "each frame, persistent noise expected. Set "
                             "r_accum_ema_alpha to 0.9 for clean output "
                             "(the engine then skips the cloud-wind dirty "
                             "trigger and runs an EMA blend), or set "
                             "r_clouds_wind_x and _z to 0 to freeze wind. "
                             "Shown once per session.");
                    hint_shown = true;
                }
            }
        };
    }
    for (const char* n : {"r_clouds_coverage",  "r_clouds_base_height",
                          "r_clouds_top_height","r_clouds_density",
                          "r_clouds_freq",      "r_clouds_detail",
                          "r_clouds_curl_amount", "r_clouds_curl_scale",
                          "r_clouds_erosion",
                          "r_clouds_wind_x",    "r_clouds_wind_z",
                          "r_clouds_seed"}) {
        if (auto* v = C.FindCVar(n)) {
            v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
        }
    }
    // Phase 4 (#100) cloud-march cvars.
    for (const char* n : {"r_vol_density_scale", "r_vol_phase_g",
                          "r_vol_multiscatter_bounces"}) {
        if (auto* v = C.FindCVar(n)) {
            v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
        }
    }
    // SDF Phase 3 (#99) fractal tuning cvars.
    for (const char* n : {"r_sdf_fractal_power", "r_sdf_fractal_iters",
                          "r_sdf_de_eps_scale"}) {
        if (auto* v = C.FindCVar(n)) {
            v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
        }
    }
    // Water Phase 1 (#134): tweaking absorption / ior / wave params at
    // runtime changes the MAT_WATER BRDF output, so reset accum_dirty_
    // to avoid stale samples blending in.
    //
    // PR #164: when the megakernel was built without PT_WATER_ENABLED,
    // the cvars still register (so saved user configs don't error out
    // on cvar-not-found) but mutating them is a no-op -- the BRDF that
    // would have consumed water_params0/1 was compiled out. Emit a single
    // [warn] line per on_change call so a user who's flipped one of the
    // r_water_* values without rebuilding gets a clear pointer at why
    // their scene looks unchanged.
    for (const char* n : {"r_water_absorption_r", "r_water_absorption_g",
                          "r_water_absorption_b", "r_water_ior",
                          "r_water_wave_scale",   "r_water_wave_amplitude",
                          "r_water_wave_speed",
                          "r_water_deep_color_r", "r_water_deep_color_g",
                          "r_water_deep_color_b"}) {
        if (auto* v = C.FindCVar(n)) {
#if PT_WATER_ENABLED
            v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
#else
            v->on_change = [this](const pt::console::CVar& cv) {
                accum_dirty_ = true;
                LOG_WARN("{} = {} ignored: this build did not enable "
                         "PT_WATER_ENABLED. Reconfigure with "
                         "-DPT_WATER_ENABLED=ON to render MAT_WATER.",
                         cv.name, cv.value);
            };
#endif
        }
    }
#if !PT_WATER_ENABLED
    // One-shot startup notice: if any r_water_* cvar is non-default (e.g.
    // the user replayed a cfg that tweaks water), surface the gate state
    // ONCE before the on_change spam path takes over. Compares string
    // value to default_value since cvars are string-stored.
    {
        bool any_non_default = false;
        for (const char* n : {"r_water_absorption_r", "r_water_absorption_g",
                              "r_water_absorption_b", "r_water_ior",
                              "r_water_wave_scale",   "r_water_wave_amplitude",
                              "r_water_wave_speed",
                              "r_water_deep_color_r", "r_water_deep_color_g",
                              "r_water_deep_color_b"}) {
            if (auto* v = C.FindCVar(n);
                v && v->value != v->default_value) {
                any_non_default = true;
                break;
            }
        }
        if (any_non_default) {
            LOG_WARN("r_water_* cvar(s) set to non-default values, but this "
                     "build was compiled without PT_WATER_ENABLED -- the "
                     "MAT_WATER BRDF is not in the megakernel and water "
                     "materials will render as a no-op.  Reconfigure with "
                     "-DPT_WATER_ENABLED=ON to restore water rendering.");
        }
    }
#endif
    // Cloud preset: when set, snap the individual cloud cvars to the
    // preset's parameter set. The user can then nudge individual values
    // without changing the preset name (or set it to "custom" to lock).
    // Heights are in metres; density is the peak sigma multiplier.
    if (auto* v = C.FindCVar("r_clouds_preset")) {
        v->allowed_values = {"clear", "cumulus", "stratus", "cirrus",
                              "overcast", "storm", "custom"};
        v->on_change = [this](const pt::console::CVar& cv) {
            struct CloudPreset {
                const char* name;
                float coverage;
                float base_y;
                float top_y;
                float density;
                float freq;
                float detail;
            };
            // Real-meteorology values. Heights are metres above ground
            // (1 world unit = 1 metre). Frequencies are cycles per
            // metre, so feature_size = 1/freq metres. Densities are
            // per-metre sigma_t (extinction): light cumulus ~0.04,
            // typical cumulus 0.05-0.10, stratus 0.04-0.08, storm
            // 0.10-0.30.
            static const CloudPreset presets[] = {
                // name        cov   base    top    dens    freq      detail
                { "clear",     0.0f,  200.0f,  500.0f, 0.0f,   0.005f,  0.0f  },
                { "cumulus",   0.45f, 200.0f,  500.0f, 0.03f,  0.005f,  0.35f },
                { "stratus",   0.95f, 120.0f,  220.0f, 0.05f,  0.008f,  0.10f },
                { "cirrus",    0.55f, 8000.0f, 9500.0f, 0.005f, 0.005f,  0.15f },
                { "overcast",  0.98f, 200.0f,  700.0f, 0.05f,  0.0035f, 0.20f },
                { "storm",     0.85f, 150.0f, 2000.0f, 0.18f,  0.0045f, 0.45f },
            };
            const std::string& name = cv.value;
            if (name == "custom") return;   // leave individual cvars alone
            auto& C = pt::console::Console::Get();
            for (const auto& p : presets) {
                if (name != p.name) continue;
                auto set_f = [&C](const char* n, float v) {
                    if (auto* c = C.FindCVar(n)) {
                        c->value = std::to_string(v);
                        if (c->on_change) c->on_change(*c);
                    }
                };
                set_f("r_clouds_coverage",    p.coverage);
                set_f("r_clouds_base_height", p.base_y);
                set_f("r_clouds_top_height",  p.top_y);
                set_f("r_clouds_density",     p.density);
                set_f("r_clouds_freq",        p.freq);
                set_f("r_clouds_detail",      p.detail);
                LOG_INFO("Cloud preset '{}': cov {:.2f} base {} top {} "
                         "dens {} freq {} detail {:.2f}",
                         name, p.coverage, p.base_y, p.top_y, p.density,
                         p.freq, p.detail);
                accum_dirty_ = true;
                return;
            }
            LOG_WARN("Unknown cloud preset '{}': leaving cvars unchanged", name);
        };
    }
    // r_quality: master preset that bulk-edits the per-feature cvars.
    // Ranges chosen so 'low' is comfortably fast at 1080p on M-series
    // GPUs, 'high' is the headline-correct path, and 'ultra' bumps
    // sample counts past convergence + max refraction depth for static
    // showcase shots. 'custom' is the escape hatch that leaves the
    // sub-cvars whatever the user / autoexec last set.
    if (auto* v = C.FindCVar("r_quality")) {
        v->allowed_values = {"low", "medium", "high", "ultra", "custom"};
        v->on_change = [this](const pt::console::CVar& cv) {
            struct QPreset {
                const char* name;
                int spp, max_bounces, refract_bounces;
                bool caustics;
            };
            static const QPreset presets[] = {
                // SPP / bounces tuned for M4 Max @ 1080p; soft denoiser
                // already covers the noise on top.
                {"low",    1,  3, 1, false},  // no caustics, single shadow ray
                {"medium", 1,  6, 2, true},   // caustics on, modest depth
                {"high",   1,  8, 4, true},   // current default
                {"ultra",  4, 16, 8, true},   // showcase / stills
            };
            if (cv.value == "custom") { accum_dirty_ = true; return; }
            for (const auto& p : presets) {
                if (cv.value == p.name) {
                    auto& C2 = pt::console::Console::Get();
                    C2.SetCVarOverride("r_spp",             std::to_string(p.spp));
                    C2.SetCVarOverride("r_max_bounces",     std::to_string(p.max_bounces));
                    C2.SetCVarOverride("r_refract_bounces", std::to_string(p.refract_bounces));
                    C2.SetCVarOverride("r_caustics",        p.caustics ? "1" : "0");
                    accum_dirty_ = true;
                    LOG_INFO("r_quality: '{}' -> spp={} bounces={} refract={} caustics={}",
                             p.name, p.spp, p.max_bounces, p.refract_bounces,
                             p.caustics ? 1 : 0);
                    return;
                }
            }
        };
    }
    // r_eye_model: presets that shove sensible defaults into the
    // exposure clamp / target / adapt-speed cvars. Picked so the
    // visual differences are obvious: cats see further into the
    // dark (higher max) than humans; DSLR locks the auto loop;
    // linear is a debug pass-through.
    if (auto* v = C.FindCVar("r_eye_model")) {
        v->allowed_values = {
            "human", "cat", "owl",
            "dslr_iso100", "dslr_iso6400",
            "phone", "linear", "custom",
        };
        v->on_change = [](const pt::console::CVar& cv) {
            struct Preset {
                const char* name;
                float exp_min, exp_max, target, adapt;
                bool  auto_exposure;       // 0 forces manual r_exposure
                float manual_exp;
            };
            // exp_min for auto-exposure presets is set to 1e-6 (effectively
            // "no floor") so genuine outdoor daylight scenes can settle to
            // the ~1e-5 exposure scalar they actually need. The legacy
            // 0.05 floor was capping the dimming and burning out skies.
            // Locked-exposure presets (dslr_iso*, linear) keep min == max
            // since they're not running auto-exposure.
            static const Preset presets[] = {
                // Human eye: comfortable adaptation range, ~1s adapt time.
                {"human",        1e-6f,  4.0f,  0.18f, 0.20f, true,  1.5f},
                // Cats: rod-rich retina, ~6x dim-light sensitivity.
                {"cat",          1e-6f, 12.0f,  0.18f, 0.30f, true,  1.5f},
                // Owls: nocturnal extreme, ~100x rod density of humans.
                {"owl",          1e-6f, 30.0f,  0.18f, 0.35f, true,  1.5f},
                // DSLR locked at ISO 100: no auto, fixed exposure.
                {"dslr_iso100",  1.0f,   1.0f,  0.18f, 0.0f,  false, 1.0f},
                // DSLR locked at ISO 6400: 64x more gain.
                {"dslr_iso6400", 8.0f,   8.0f,  0.18f, 0.0f,  false, 8.0f},
                // Smartphone: auto, modest range.
                {"phone",        1e-6f,  6.0f,  0.18f, 0.25f, true,  1.5f},
                // Linear: bypass exposure entirely (debug).
                {"linear",       1.0f,   1.0f,  0.18f, 0.0f,  false, 1.0f},
            };
            if (cv.value == "custom") return;
            auto& C2 = pt::console::Console::Get();
            for (const auto& p : presets) {
                if (cv.value == p.name) {
                    C2.SetCVarOverride("r_exposure_min",    std::to_string(p.exp_min));
                    C2.SetCVarOverride("r_exposure_max",    std::to_string(p.exp_max));
                    C2.SetCVarOverride("r_exposure_target", std::to_string(p.target));
                    C2.SetCVarOverride("r_eye_adapt_speed", std::to_string(p.adapt));
                    C2.SetCVarOverride("r_auto_exposure",   p.auto_exposure ? "1" : "0");
                    if (!p.auto_exposure) {
                        C2.SetCVarOverride("r_exposure", std::to_string(p.manual_exp));
                    }
                    return;
                }
            }
        };
    }
    // r_env_map: hot-reload on change. Empty -> unload (procedural sky).
    if (auto* v = C.FindCVar("r_env_map")) {
        v->on_change = [this](const pt::console::CVar& cv) {
            ReloadEnvMap(cv.value);
        };
    }
    if (auto* v = C.FindCVar("r_env_intensity")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // r_mis changes the per-frame radiance estimator; runtime toggle
    // must invalidate accum_hdr so we don't blend pre- and post-toggle
    // samples together (same precedent as r_env_intensity above).
    if (auto* v = C.FindCVar("r_mis")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // r_light_tree (#129) flips between tree-traversal and uniform-pick.
    // Same accumulator-reset reasoning as r_mis above -- pre- and
    // post-toggle samples come from different estimators and shouldn't
    // be averaged together.
    //
    // perf/megakernel-split-light-tree: when the megakernel was built
    // without PT_LIGHT_TREE_ENABLED, the tree traversal is excised from
    // the shader entirely, so flipping the cvar is a silent no-op. Mirror
    // the PT_WATER_ENABLED pattern (PR #164) used elsewhere: emit a single
    // [warn] on each on_change call so users who flip the cvar without
    // rebuilding learn why their NEE variance didn't change.
    if (auto* v = C.FindCVar("r_light_tree")) {
#if PT_LIGHT_TREE_ENABLED
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
#else
        v->on_change = [this](const pt::console::CVar& cv) {
            accum_dirty_ = true;
            LOG_WARN("{} = {} ignored: this build did not enable "
                     "PT_LIGHT_TREE_ENABLED. Reconfigure with "
                     "-DPT_LIGHT_TREE=ON to restore the hierarchical "
                     "light tree picker.",
                     cv.name, cv.value);
        };
#endif
    }
#if !PT_LIGHT_TREE_ENABLED
    // One-shot startup notice: if r_light_tree was set to 0 in a user
    // cfg the perf characteristic is unchanged (uniform-pick is the
    // only path either way), so no warn. But if a user *expected* the
    // tree (default value 1) and the megakernel was compiled without
    // PT_LIGHT_TREE, surface it once so they know variance won't drop.
    {
        if (auto* v = C.FindCVar("r_light_tree");
            v && v->GetInt() != 0) {
            LOG_WARN("r_light_tree=1 but this build was compiled without "
                     "PT_LIGHT_TREE_ENABLED -- the megakernel uses the "
                     "uniform-pick fallback regardless. Variance will not "
                     "match the tree-pick fixtures. Reconfigure with "
                     "-DPT_LIGHT_TREE=ON to restore the hierarchical "
                     "light tree picker (recommended for scenes with "
                     ">100 lights).");
        }
    }
#endif
    // r_analytic_bvh_threshold changes the linear/BVH split decision;
    // mark primitives_dirty_ so EnsurePrimitivesUploaded re-runs the
    // partition + (re)builds or tears down the BVH accordingly.
    // accum_dirty_ follows (re-ordering the primitive buffer changes
    // hit-test order, so per-frame radiance can shift fractionally).
    if (auto* v = C.FindCVar("r_analytic_bvh_threshold")) {
        v->on_change = [this](const pt::console::CVar&) {
            primitives_dirty_ = true;
            accum_dirty_      = true;
        };
    }
    // r_accum_ema_alpha switches the accumulator update rule (online
    // running mean vs EMA blend); reusing history captured under the
    // previous algorithm would mix incompatible weights (a count-based
    // average alongside an exponential-decay average). Changing it
    // mid-render against a non-zero prev.a would also instantly mix modes
    // for a jittery frame. Start a fresh accumulation epoch on every toggle.
    if (auto* v = C.FindCVar("r_accum_ema_alpha")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // Sky cvars: changing any of them invalidates accumulation.
    if (auto* v = C.FindCVar("r_sky_mode")) {
        v->allowed_values = {"gradient", "hdri", "procedural"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sun_elevation")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sun_azimuth")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // Toggling horizon-flatten changes the sun-disc shape on the
    // procedural-sky path, so the accumulator must drop the prior
    // (mismatched) samples or the new oval/circle blends with the old
    // one until something else dirties the state.
    if (auto* v = C.FindCVar("r_sun_horizon_flatten")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // r_planet_radius (issue #51) toggles atmosphericTransmittance
    // between the planar and curved-Earth branches AND rebiases
    // procSky's horizon line. Without an accum reset, A/B-toggling
    // this against a non-zero accumulator (which the
    // sunset_altitude_planar fixture explicitly does) blends pre- and
    // post-change samples in the temporal mean until the camera moves.
    if (auto* v = C.FindCVar("r_planet_radius")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_exposure")) {
        v->on_change = [this](const pt::console::CVar& cv) {
            accum_dirty_ = true;
            // Manual mode: push the new value straight to the GPU
            // exposure_state buffer so the next frame's PathTrace
            // tonemap picks it up. In auto mode, AutoExposure.slang
            // overwrites it each frame so we don't bother.
            if (exposure_state_id_ == 0 || !device_) return;
            bool auto_exp = false;
            if (auto* av = pt::console::Console::Get().FindCVar("r_auto_exposure"))
                auto_exp = av->GetBool();
            if (!auto_exp) {
                float val = cv.GetFloat();
                device_->WriteBuffer(pt::rhi::BufferHandle{exposure_state_id_},
                                     &val, sizeof(float), 0);
            }
        };
    }
    if (auto* v = C.FindCVar("r_auto_exposure")) {
        v->on_change = [this](const pt::console::CVar& cv) {
            // Switching auto -> manual: seed the GPU buffer with the
            // current r_exposure value so the path tracer doesn't
            // briefly use whatever the auto pass last wrote.
            if (cv.GetBool()) return;
            if (exposure_state_id_ == 0 || !device_) return;
            float val = 1.5f;
            if (auto* ev = pt::console::Console::Get().FindCVar("r_exposure"))
                val = ev->GetFloat();
            device_->WriteBuffer(pt::rhi::BufferHandle{exposure_state_id_},
                                 &val, sizeof(float), 0);
        };
    }
    if (auto* v = C.FindCVar("r_sky_use_astronomical")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // Factories for astronomy-only cvars. Both emit a one-line warning
    // when r_sky_use_astronomical = 0 (the cvar has no visible effect
    // in manual-sun mode), but they differ on accumulator dirtying:
    //  - `astro_handler`            -- dirties the accumulator. Used by
    //    cvars whose value is consumed by the path tracer's per-frame
    //    state (sun position, observer location, date), where a change
    //    invalidates accumulated samples.
    //  - `astro_handler_no_dirty`   -- skips accum_dirty_. Used by
    //    r_sky_animate / r_sky_animate_rate: toggling the animation or
    //    changing its rate doesn't change the current frame's state
    //    (Tick() applies the rate next frame), and EMA-mode users
    //    don't want every cvar nudge to nuke their accumulator.
    //
    // Both suppress the warning while `cfg_loading_` is true (cfg load
    // writes cvars in lex order, so r_sky_* lines apply before
    // r_sky_use_astronomical -- false positives unless suppressed).
    // A single summary audit fires after cfg load with the correct
    // final state. They also suppress when `astro_chained_update_`
    // is set so the r_sky_city handler's writes to lat/lon/tz don't
    // re-emit per-cvar warnings -- city emits its own warn once.
    auto astro_should_warn = [this]() {
        if (cfg_loading_ || astro_chained_update_) return false;
        auto* a = pt::console::Console::Get().FindCVar("r_sky_use_astronomical");
        return a != nullptr && !a->GetBool();
    };
    auto astro_handler = [this, astro_should_warn](const char* name) {
        return [this, name, astro_should_warn](const pt::console::CVar&) {
            accum_dirty_ = true;
            if (astro_should_warn()) {
                LOG_WARN("{}: changed but r_sky_use_astronomical = 0 -- "
                         "ignored in manual-sun mode (set "
                         "r_sky_use_astronomical 1 to engage time-based sun)",
                         name);
            }
        };
    };
    auto astro_handler_no_dirty = [astro_should_warn](const char* name) {
        return [name, astro_should_warn](const pt::console::CVar&) {
            // No accum_dirty_: animate-toggle and rate-change don't
            // alter the current frame's state. EMA preservation matters.
            if (astro_should_warn()) {
                LOG_WARN("{}: changed but r_sky_use_astronomical = 0 -- "
                         "ignored in manual-sun mode (set "
                         "r_sky_use_astronomical 1 to engage time-based sun)",
                         name);
            }
        };
    };
    if (auto* v = C.FindCVar("r_sky_hour"))   v->on_change = astro_handler("r_sky_hour");
    if (auto* v = C.FindCVar("r_sky_animate")) {
        v->allowed_values = {"0", "1"};
        v->on_change = astro_handler_no_dirty("r_sky_animate");
    }
    if (auto* v = C.FindCVar("r_sky_animate_rate")) v->on_change = astro_handler_no_dirty("r_sky_animate_rate");
    if (auto* v = C.FindCVar("r_show_stars")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_stars_mode")) {
        v->allowed_values = {"bsc", "procedural"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_stars_twinkle")) {
        v->allowed_values = {"0", "1"};
        // Twinkle is per-frame in the shader; toggling doesn't need an
        // accum reset because the change is felt the next frame anyway.
    }
    if (auto* v = C.FindCVar("r_sky_lat") ) v->on_change = astro_handler("r_sky_lat");
    if (auto* v = C.FindCVar("r_sky_lon") ) v->on_change = astro_handler("r_sky_lon");
    // r_sky_time_offset isn't a registered cvar; the previous
    // FindCVar/on_change pair was dead code carried forward from an
    // earlier iteration. Removed.
    if (auto* v = C.FindCVar("r_sky_year"))  v->on_change = astro_handler("r_sky_year");
    if (auto* v = C.FindCVar("r_sky_month")) v->on_change = astro_handler("r_sky_month");
    if (auto* v = C.FindCVar("r_sky_day"))   v->on_change = astro_handler("r_sky_day");

    // City preset dropdown. Each entry is (name, lat, lon). Selecting
    // one writes lat/lon to the geographic cvars; "custom" leaves
    // them as-is so the user can dial in arbitrary coordinates.
    if (auto* v = C.FindCVar("r_sky_city")) {
        v->allowed_values = {
            "chennai", "mumbai", "delhi", "bangalore",
            "tokyo", "singapore", "beijing", "seoul",
            "london", "paris", "berlin", "moscow",
            "new_york", "san_francisco", "los_angeles", "chicago",
            "sao_paulo", "buenos_aires",
            "sydney", "auckland",
            "cairo", "cape_town",
            "reykjavik", "anchorage",
            "custom",
        };
        v->on_change = [this](const pt::console::CVar& cv) {
            // Same astronomy-only warning as the simpler astro cvars
            // (city only matters when r_sky_use_astronomical = 1) but
            // inlined because this handler also writes lat/lon/tz on
            // preset selection; we want the warn line in addition to
            // the lat/lon/tz writes, not in place of them. Suppressed
            // during cfg load (matches astro_handler convention) so a
            // user with astro=1 saved doesn't see false-positive
            // warnings before the cfg-end audit reports the truth.
            if (!cfg_loading_) {
                auto& Cw = pt::console::Console::Get();
                auto* a = Cw.FindCVar("r_sky_use_astronomical");
                if (a && !a->GetBool()) {
                    LOG_WARN("r_sky_city: changed but r_sky_use_astronomical = 0 -- "
                             "lat/lon/tz still written but no visible effect "
                             "(set r_sky_use_astronomical 1 to engage time-based sun)");
                }
            }
            // Mark chained-update before the SetCVarOverride writes
            // below so r_sky_lat / r_sky_lon's astro_handler doesn't
            // also emit per-cvar warnings -- the user issued ONE
            // command and should see ONE warn (the city one above).
            // RAII so we can't forget to clear on early return / throw.
            struct ChainGuard {
                bool& flag;
                ChainGuard(bool& f) : flag(f) { flag = true; }
                ~ChainGuard()                 { flag = false; }
            } chain_guard(astro_chained_update_);
            // tz is the city's standard-time UTC offset in hours
            // (positive east). DST is intentionally ignored: this is
            // a sky simulator, not a clock app, and DST switches add
            // complexity that would obscure the astronomy.
            struct CityRow { const char* name; double lat; double lon; double tz; };
            static const CityRow cities[] = {
                {"chennai",        13.0827,   80.2707,   5.5},
                {"mumbai",         19.0760,   72.8777,   5.5},
                {"delhi",          28.6139,   77.2090,   5.5},
                {"bangalore",      12.9716,   77.5946,   5.5},
                {"tokyo",          35.6762,  139.6503,   9.0},
                {"singapore",       1.3521,  103.8198,   8.0},
                {"beijing",        39.9042,  116.4074,   8.0},
                {"seoul",          37.5665,  126.9780,   9.0},
                {"london",         51.5074,   -0.1278,   0.0},
                {"paris",          48.8566,    2.3522,   1.0},
                {"berlin",         52.5200,   13.4050,   1.0},
                {"moscow",         55.7558,   37.6173,   3.0},
                {"new_york",       40.7128,  -74.0060,  -5.0},
                {"san_francisco",  37.7749, -122.4194,  -8.0},
                {"los_angeles",    34.0522, -118.2437,  -8.0},
                {"chicago",        41.8781,  -87.6298,  -6.0},
                {"sao_paulo",     -23.5505,  -46.6333,  -3.0},
                {"buenos_aires",  -34.6037,  -58.3816,  -3.0},
                {"sydney",        -33.8688,  151.2093,  10.0},
                {"auckland",      -36.8485,  174.7633,  12.0},
                {"cairo",          30.0444,   31.2357,   2.0},
                {"cape_town",     -33.9249,   18.4241,   2.0},
                {"reykjavik",      64.1466,  -21.9426,   0.0},
                {"anchorage",      61.2181, -149.9003,  -9.0},
            };
            if (cv.value == "custom") { accum_dirty_ = true; return; }
            for (const auto& c : cities) {
                if (cv.value == c.name) {
                    auto& C2 = pt::console::Console::Get();
                    C2.SetCVarOverride("r_sky_lat", std::to_string(c.lat));
                    C2.SetCVarOverride("r_sky_lon", std::to_string(c.lon));
                    C2.SetCVarOverride("r_sky_tz_offset_hours", std::to_string(c.tz));
                    accum_dirty_ = true;
                    LOG_INFO("r_sky_city: '{}' -> lat={:.4f} lon={:.4f} tz={:+.1f}",
                             c.name, c.lat, c.lon, c.tz);
                    return;
                }
            }
            LOG_WARN("r_sky_city: '{}' has no preset; lat/lon unchanged",
                     std::string(cv.value));
        };
    }
    if (auto* v = C.FindCVar("r_sky_hour_local")) {
        v->allowed_values = {"0", "1"};
        v->on_change = astro_handler("r_sky_hour_local");
    }
    // app_vsync / app_overlay_enabled / app_auto_open_console / dev_cheats:
    // boolean toggles -- accept 0|1. phys_enabled is the Phase 1
    // physics master switch and follows the same boolean convention
    // as r_bloom / r_hdr_pipeline (`allowed_values` so tab-completion
    // and the cvar UI treat it as a toggle, rejecting arbitrary
    // numeric values that would otherwise be silently accepted).
    // r_phys_debug_visualize (#181) is the velocity-magnitude debug
    // overlay; same on/off semantics, same toggle contract.
    for (const char* n : {"app_vsync", "app_overlay_enabled",
                          "app_auto_open_console", "dev_cheats",
                          "phys_enabled", "r_phys_debug_visualize"}) {
        if (auto* v = C.FindCVar(n)) v->allowed_values = {"0", "1"};
    }

    // Slider ranges for numeric cvars. The web UI checks slider_max >
    // slider_min and renders a draggable range input + numeric readout
    // instead of a free-form text box. Pick ranges that cover the
    // useful design space, not the full numeric domain (e.g. r_spp
    // tops out at 32 even though the renderer accepts more) -- a
    // slider that needs a microscope to land on a sane value defeats
    // the purpose. Free-form input is still possible from the command
    // line for power users.
    auto set_slider = [&C](const char* name, float lo, float hi, float step) {
        if (auto* v = C.FindCVar(name)) {
            v->slider_min  = lo;
            v->slider_max  = hi;
            v->slider_step = step;
        }
    };
    set_slider("r_spp",             1.0f,   32.0f,  1.0f);
    set_slider("r_max_bounces",     1.0f,   16.0f,  1.0f);
    set_slider("r_exposure",        0.1f,    5.0f,  0.05f);
    set_slider("r_env_intensity",   0.0f,    5.0f,  0.05f);
    set_slider("r_sun_elevation", -90.0f,   90.0f,  0.5f);
    set_slider("r_sun_azimuth",     0.0f,  360.0f,  1.0f);
    set_slider("r_sky_hour",          0.0f,  24.0f,  0.01f);
    set_slider("r_sky_animate_rate",  0.0f,  24.0f,  0.05f);
    set_slider("r_sky_year",         0.0f,  2100.0f,  1.0f);
    set_slider("r_sky_month",        0.0f,    12.0f,  1.0f);
    set_slider("r_sky_day",          0.0f,    31.0f,  1.0f);
    set_slider("r_sky_lat",         -90.0f,  90.0f,  0.1f);
    set_slider("r_sky_lon",      -180.0f,  180.0f,  0.1f);
    set_slider("cam_fov",          20.0f,  120.0f,  0.5f);
    set_slider("cam_speed",         0.1f,   30.0f,  0.1f);
    set_slider("cam_sprint_mult",   1.0f,   10.0f,  0.1f);
    set_slider("cam_sensitivity",   0.01f,   1.0f,  0.01f);
    set_slider("r_dof_aperture",        0.0f,   1.0f,  0.001f);
    set_slider("r_dof_focal_distance",  0.1f, 100.0f,  0.1f);
    set_slider("r_dof_blades",          0.0f,  16.0f,  1.0f);
    set_slider("r_volumetric_density",     0.0f,  0.20f, 0.001f);
    set_slider("r_volumetric_anisotropy", -0.95f, 0.95f, 0.01f);
    set_slider("r_volumetric_intensity",   0.0f,  4.0f,  0.05f);
    set_slider("r_volumetric_samples",     4.0f, 64.0f,  1.0f);
    set_slider("r_rayleigh",               0.0f, 100.0f,  0.5f);
    set_slider("r_moon_size",              0.5f,  20.0f,  0.1f);
    set_slider("r_sun_size",               0.5f,  20.0f,  0.1f);
    set_slider("r_moon_brightness",        0.0f,   3.0f,  0.05f);
    set_slider("r_clouds_coverage",         0.0f,    1.0f,   0.01f);
    set_slider("r_clouds_base_height",      0.0f, 12000.0f, 25.0f);
    set_slider("r_clouds_top_height",      50.0f, 14000.0f, 25.0f);
    set_slider("r_clouds_density",          0.0f,    0.5f,   0.005f);
    set_slider("r_clouds_freq",          0.0005f,   0.02f,  0.0005f);
    set_slider("r_clouds_detail",           0.0f,    1.0f,   0.01f);
    // Issue #117: curl-noise + secondary edge-erosion sliders.
    // curl_amount is the magnitude scalar on the curl displacement;
    // curl output is roughly O(5-10) per component (value-noise central
    // diff at h=0.1), so amount=0.3 -> ~2m sub-feature shear, amount=1.0
    // -> ~6-10m heavy-cumulus wisps. Spec keeps [0..1] for the UI but
    // the shader stays linear so amount>1 still works for storm shots.
    // curl_scale is cycles/metre (0.001..0.05 spans large shear
    // streamers to fine close-up wisps).
    set_slider("r_clouds_curl_amount",      0.0f,    2.0f,  0.01f);
    set_slider("r_clouds_curl_scale",     0.001f,   0.05f,  0.001f);
    set_slider("r_clouds_erosion",          0.0f,    1.0f,   0.01f);
    set_slider("r_clouds_wind_x",         -25.0f,   25.0f,   0.5f);
    set_slider("r_clouds_wind_z",         -25.0f,   25.0f,   0.5f);
    set_slider("r_clouds_seed",             0.0f,  100.0f,   1.0f);
    set_slider("r_bloom_threshold",     0.0f,  10.0f,  0.05f);
    set_slider("r_bloom_intensity",     0.0f,   1.0f,  0.005f);
    set_slider("r_bloom_mips",          1.0f,   5.0f,  1.0f);
    set_slider("r_bloom_radius",        0.5f,   3.0f,  0.05f);
    set_slider("r_lens_flare_intensity",      0.0f, 1.0f,  0.005f);
    set_slider("r_lens_flare_dispersion",     0.0f, 0.05f, 0.001f);
    set_slider("r_lens_flare_count",          1.0f, 6.0f,  1.0f);
    set_slider("r_lens_flare_threshold",      0.5f, 10.0f, 0.05f);

    RegisterCsgCommands();
    RegisterPrimCommands();
    RegisterEditorGizmoCommands();
    // --- SDF Phase 1 (#97) -----------------------------------------------------
    RegisterSdfCommands();
    // --- end SDF Phase 1 -------------------------------------------------------

    // --- Audio MVP (#80) -------------------------------------------------------
    // `audio_play <path>`  -- load + play a sound at the current camera
    //                         position (3D-attenuated stereo pan).
    // `audio_stop`         -- stop every currently-playing voice.
    // No reverb / no ray-traced occlusion / no HRTF -- those are Phase B
    // follow-ups against the renderer's TLAS.
    C.RegisterCommand("audio_play",
        "Play a sound file (WAV) at the camera position with 1/r distance "
        "attenuation and equal-power stereo pan. Usage: audio_play <path> "
        "(quote the path if it contains spaces, e.g. audio_play \"my asset.wav\")",
        [this](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine("audio_play: missing path argument");
                return;
            }
            if (!audio_system_ || !audio_system_->IsRunning()) {
                out.PrintLine("audio_play: audio subsystem not running");
                return;
            }
            if (!camera_) {
                out.PrintLine("audio_play: no camera");
                return;
            }
            const std::string path(args[0]);
            auto sound = audio_system_->LoadSound(path);
            if (sound == pt::audio::kInvalidSound) {
                out.FormatLine("audio_play: failed to load '{}'", path);
                return;
            }
            auto voice = audio_system_->PlaySound(sound, camera_->pos, 1.0f);
            if (voice == pt::audio::kInvalidVoice) {
                out.PrintLine("audio_play: voice pool full or play failed");
                return;
            }
            out.FormatLine("audio_play: '{}' voice={}", path, voice);
        });

    C.RegisterCommand("audio_stop", "Stop every currently-playing audio voice.",
        [this](auto, pt::console::Output& out) {
            if (!audio_system_) {
                out.PrintLine("audio_stop: audio subsystem not initialized");
                return;
            }
            auto n = audio_system_->StopAll();
            out.FormatLine("audio_stop: stopped {} voice(s)", n);
        });
    // --- end Audio MVP ---------------------------------------------------------
    // -----------------------------------------------------------------------
    // Particle / VFX (#82 MVP) console commands.
    //
    // Subcommands:
    //   particle_emit smoke <x> <y> <z>   - one-shot smoke puff at position
    //   particle_emit spark <x> <y> <z>   - one-shot spark burst at position
    //   particle_emit snow   [x] [y] [z]  - start continuous snow emitter
    //                                       centred at the given position
    //                                       (defaults to camera-relative
    //                                       origin if omitted)
    //   particle_emit snow_stop           - stop the snow emitter
    //   particle_clear                    - kill all live particles +
    //                                       continuous emitters
    //   particle_count                    - report live particle count
    //
    // Parsing notes:
    //   * Floats use std::strtof (deferred since console args are
    //     std::string_view) -- copy into a small char buffer first.
    //     The console splits on whitespace and semicolons before we
    //     see args, so each arg is a single token.
    //   * The first arg is the SUBCOMMAND name (smoke / spark / snow /
    //     snow_stop). The remaining 0..3 args are the optional position.
    //   * The default x/y/z when omitted is (0, 1.5, 0) -- 1.5 m above
    //     world origin, the same height as the engineering camera
    //     default. This matches "spawn at the centre of the canonical
    //     scene" intent.
    // -----------------------------------------------------------------------
    // Reuse the file-scope ParseFloat() helper (defined later in this
    // TU; forward-declared in the inner anonymous namespace above) so
    // particle_emit doesn't carry its own duplicate parse-float
    // lambda. Keeping a single canonical float parser avoids subtle
    // behaviour drift (e.g. one version trimming whitespace, another
    // not).
    C.RegisterCommand("particle_emit",
        "particle_emit <smoke|spark|snow|snow_stop> [x y z]. "
        "Spawn a preset particle burst (smoke / spark) or start/stop "
        "the continuous snow emitter at the given world position "
        "(default 0 1.5 0). Particles are screen-space billboards in "
        "the MVP -- NOT visible in path-traced reflections.",
        [this](auto args, pt::console::Output& out) {
            if (!particles_) { out.PrintLine("particle_emit: particle system not initialised"); return; }
            if (args.empty()) {
                out.PrintLine("usage: particle_emit <smoke|spark|snow|snow_stop> [x y z]");
                return;
            }
            // Reject excess args explicitly. Other commands in this
            // file (prim_sphere, etc.) bounds-check argc; matching that
            // convention here surfaces typos like
            // `particle_emit smoke 1 2 3 extra` rather than silently
            // ignoring trailing tokens.
            if (args.size() > 4u) {
                out.FormatLine(
                    "particle_emit: too many arguments ({}); expected "
                    "<smoke|spark|snow|snow_stop> [x y z]", args.size());
                return;
            }
            const std::string sub(args[0]);
            glm::vec3 at{0.0f, 1.5f, 0.0f};
            // Optional position. Accept "smoke 1 2 3" and "smoke" both;
            // partial positions ("smoke 1") error out so the user
            // doesn't accidentally spawn at a partial location.
            if (args.size() == 4u) {
                float x = 0.0f, y = 0.0f, z = 0.0f;
                if (!ParseFloat(args[1], x) ||
                    !ParseFloat(args[2], y) ||
                    !ParseFloat(args[3], z)) {
                    out.PrintLine("particle_emit: x y z must be numeric");
                    return;
                }
                at = glm::vec3(x, y, z);
            } else if (args.size() == 2u || args.size() == 3u) {
                out.PrintLine("particle_emit: provide either 0 args (default position) or 3 (x y z)");
                return;
            }

            if (sub == "smoke") {
                auto spec = pt::effects::ParticleSystem::PresetSmoke(at);
                auto n = particles_->EmitBurst(spec);
                out.FormatLine("particle_emit smoke: spawned {} particle(s) at ({:.2f}, {:.2f}, {:.2f})",
                               n, at.x, at.y, at.z);
            } else if (sub == "spark") {
                auto spec = pt::effects::ParticleSystem::PresetSpark(at);
                auto n = particles_->EmitBurst(spec);
                out.FormatLine("particle_emit spark: spawned {} particle(s) at ({:.2f}, {:.2f}, {:.2f})",
                               n, at.x, at.y, at.z);
            } else if (sub == "snow") {
                // Rate: 50 particles/sec is a comfortable indoor
                // flurry; the cap throttles it naturally on the way up.
                auto spec = pt::effects::ParticleSystem::PresetSnow(at);
                particles_->StartContinuous("snow", spec, 50.0f);
                out.FormatLine("particle_emit snow: continuous emitter started, centre ({:.2f}, {:.2f}, {:.2f})",
                               at.x, at.y, at.z);
            } else if (sub == "snow_stop") {
                auto n = particles_->StopContinuous("snow");
                out.FormatLine("particle_emit snow_stop: stopped {} continuous emitter(s)", n);
            } else {
                out.FormatLine("particle_emit: unknown preset '{}' (expected smoke|spark|snow|snow_stop)", sub);
            }
        });

    C.RegisterCommand("particle_clear",
        "Kill every live particle + stop all continuous emitters.",
        [this](auto, pt::console::Output& out) {
            if (!particles_) { out.PrintLine("particle_clear: particle system not initialised"); return; }
            particles_->Clear();
            out.PrintLine("particle_clear: ok");
        });

    C.RegisterCommand("particle_count",
        "Report the current live particle count.",
        [this](auto, pt::console::Output& out) {
            if (!particles_) { out.PrintLine("particle_count: particle system not initialised"); return; }
            out.FormatLine("particles live: {} / cap {}",
                           particles_->LiveCount(), particles_->MaxParticles());
        });
    // --- Physics Phase 1 (#132) ------------------------------------------------
    RegisterPhysicsCommands();
    // --- end Physics Phase 1 ---------------------------------------------------
    // --- Light primitives (#73) ------------------------------------------------
    RegisterLightCommands();
    // --- end Light primitives --------------------------------------------------
    // --- Fluid Phase 1 (#136) -- smoke emitters --------------------------------
    RegisterSmokeCommands();
    // --- end Fluid Phase 1 -----------------------------------------------------
    // --- Voxel destruction Phase 1 (#140) --------------------------------------
    // r_voxelize_demo is a boolean toggle so it joins the {"0","1"}
    // allowed-values pool used by `toggle` and CLI override validation
    // (mirrors r_caustics / r_dof / r_volumetric / r_clouds above). No
    // on_change here -- SyncVoxelDemoState already invalidates
    // accum_dirty_ on the toggle edge.
    if (auto* v = C.FindCVar("r_voxelize_demo")) {
        v->allowed_values = {"0", "1"};
    }
    RegisterVoxelCommands();
    // --- end Voxel destruction Phase 1 -----------------------------------------

    // --- Editor scaffold (agent-20) --------------------------------------------
    // panel_open / panel_close / panel_open_all / panel_close_all / panels.
    // Safe to register before the WebSocket server is up -- the
    // commands only consult net_port / net_bind_address at invoke
    // time, and r_editor_panels_autoopen is fired later from Init().
    RegisterEditorCommands();
    // --- end Editor scaffold ---------------------------------------------------

    // --- Cross-cvar dependency predicates (issue #161) -------------------------
    // When the user sets a cvar whose effect requires another cvar to be in
    // a specific state, evaluate the predicate after the set and emit a
    // one-line `[warn]` with an actionable hint. NEVER blocks the set:
    // a cfg-load script may set the dependency cvar later, so the warning
    // is informational only. The cfg-load suppression already lives in the
    // engine's broader `cfg_loading_` flag for astro handlers; for the
    // generic dep-warn path we just trust that the user-typed set sees
    // the latest engine state.
    //
    // Each entry: (dependent_cvar_name, predicate_lambda, hint_text).
    // The predicate captures Console by reference so it can read the
    // current value of the prerequisite cvar at the moment the
    // dependent is set.
    struct DepEntry {
        const char* name;
        std::function<bool()> pred;
        const char* hint;
    };
    auto sky_proc = []() {
        auto* m = pt::console::Console::Get().FindCVar("r_sky_mode");
        return m != nullptr && m->value == "procedural";
    };
    auto clouds_on = []() {
        auto* c = pt::console::Console::Get().FindCVar("r_clouds");
        return c != nullptr && c->GetBool();
    };
    auto denoiser_svgf = []() {
        auto* d = pt::console::Console::Get().FindCVar("r_denoiser");
        if (d == nullptr) return false;
        return d->value.rfind("svgf_", 0) == 0 || d->value == "nrd";
    };
    // denoiser_metalfx / denoiser_optix predicates would gate
    // r_metalfx_* / r_optix_* cvars once those tuning knobs land
    // (issue #161 leaves them as reserved future-work entries). Not
    // wired up today because no such cvars exist yet.
    auto smoke_on = []() {
        auto* s = pt::console::Console::Get().FindCVar("r_smoke_enabled");
        return s != nullptr && s->GetBool();
    };
    auto restir_on = []() {
        auto* r = pt::console::Console::Get().FindCVar("r_restir");
        return r != nullptr && r->GetBool();
    };
    auto lighttree_on = []() {
        auto* t = pt::console::Console::Get().FindCVar("r_light_tree");
        return t != nullptr && t->GetBool();
    };
    auto voxdemo_on = []() {
        auto* d = pt::console::Console::Get().FindCVar("r_voxelize_demo");
        return d != nullptr && d->GetBool();
    };

    const std::vector<DepEntry> deps = {
        // Sky: sun/day cvars are inert unless r_sky_mode procedural.
        // (Static hint mirrors the issue #161 spec format; the
        // "current value r_sky_mode=..." dynamic part isn't worth a
        // hint-builder API for the MVP -- the actionable
        // `cvar set r_sky_mode procedural` is the high-value bit.)
        { "r_sun_elevation", sky_proc,
            "r_sun_elevation only takes effect when r_sky_mode is procedural; "
            "hint: cvar set r_sky_mode procedural" },
        { "r_sun_azimuth", sky_proc,
            "r_sun_azimuth only takes effect when r_sky_mode is procedural; "
            "hint: cvar set r_sky_mode procedural" },
        { "r_sky_day", sky_proc,
            "r_sky_day only takes effect when r_sky_mode is procedural; "
            "hint: cvar set r_sky_mode procedural" },
        // Clouds: per-cloud cvars need r_clouds 1 to drive the march.
        { "r_clouds_coverage",   clouds_on,
            "r_clouds_coverage only takes effect when r_clouds is 1; "
            "current value r_clouds=0; "
            "hint: cvar set r_clouds 1" },
        { "r_clouds_curl_amount", clouds_on,
            "r_clouds_curl_amount only takes effect when r_clouds is 1; "
            "current value r_clouds=0; "
            "hint: cvar set r_clouds 1" },
        { "r_clouds_density", clouds_on,
            "r_clouds_density only takes effect when r_clouds is 1; "
            "current value r_clouds=0; "
            "hint: cvar set r_clouds 1" },
        { "r_clouds_detail", clouds_on,
            "r_clouds_detail only takes effect when r_clouds is 1; "
            "current value r_clouds=0; "
            "hint: cvar set r_clouds 1" },
        // SVGF tuning: needs an SVGF-family denoiser.
        { "r_svgf_atrous_passes", denoiser_svgf,
            "r_svgf_atrous_passes only takes effect when r_denoiser starts "
            "with svgf_; SVGF inactive on the current denoiser; "
            "hint: cvar set r_denoiser svgf_atrous" },
        { "r_svgf_albedo_demod", denoiser_svgf,
            "r_svgf_albedo_demod only takes effect when r_denoiser starts "
            "with svgf_; SVGF inactive on the current denoiser; "
            "hint: cvar set r_denoiser svgf_atrous" },
        // Smoke: tuning cvars only matter when smoke loop is enabled.
        { "r_smoke_max_emitters", smoke_on,
            "r_smoke_max_emitters only takes effect when r_smoke_enabled is 1; "
            "smoke loop disabled; "
            "hint: cvar set r_smoke_enabled 1" },
        // ReSTIR tuning.
        { "r_restir_k_candidates", restir_on,
            "r_restir_k_candidates only takes effect when r_restir is 1; "
            "ReSTIR dispatch skipped; "
            "hint: cvar set r_restir 1" },
        { "r_restir_temporal", restir_on,
            "r_restir_temporal only takes effect when r_restir is 1; "
            "ReSTIR dispatch skipped; "
            "hint: cvar set r_restir 1" },
        { "r_restir_spatial", restir_on,
            "r_restir_spatial only takes effect when r_restir is 1; "
            "ReSTIR dispatch skipped; "
            "hint: cvar set r_restir 1" },
        // Light tree tuning.
        { "r_light_tree_max_depth", lighttree_on,
            "r_light_tree_max_depth only takes effect when r_light_tree is 1; "
            "naive uniform NEE pick; "
            "hint: cvar set r_light_tree 1" },
        // Voxel destruction.
        { "r_voxel_size", voxdemo_on,
            "r_voxel_size only takes effect when r_voxelize_demo is 1; "
            "voxel grids hidden; "
            "hint: cvar set r_voxelize_demo 1" },
    };

    for (const auto& d : deps) {
        auto* v = C.FindCVar(d.name);
        if (v == nullptr) continue;     // cvar not registered in this build
        v->requires_predicate = d.pred;
        v->requires_hint      = d.hint;
    }
    // --- end cross-cvar dependency predicates (issue #161) ---------------------
}

namespace {

// Strip-leading parser for a positive integer node id.
bool ParseUint(std::string_view s, std::uint32_t& out) {
    if (s.empty()) return false;
    std::uint32_t v = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size()) return false;
    out = v;
    return true;
}

bool ParseFloat(std::string_view s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::string buf(s);
    float v = std::strtof(buf.c_str(), &end);
    if (end == buf.c_str()) return false;
    out = v;
    return true;
}

// Signed integer parser. Used by SDF Phase 2 (#98) `sdf_repeat_limited`
// for the half-extent triple -- semantically a non-negative cell count
// but accepting a signed type lets the host reject negative inputs at
// the parser layer with a clear error rather than silently underflowing
// the uint cast.
bool ParseInt(std::string_view s, int& out) {
    if (s.empty()) return false;
    int v = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size()) return false;
    out = v;
    return true;
}

}  // namespace

void Engine::RegisterCsgCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("csg_dump",
        "Print the current CSG tree.",
        [this](auto, pt::console::Output& out) {
            if (!csg_scene_) { out.PrintLine("CSG scene not initialised"); return; }
            std::string body;
            csg_scene_->Dump(body);
            out.Print(body);
        });

    C.RegisterCommand("csg_reset",
        "Reset the CSG scene to the default drilled-cube.",
        [this](auto, pt::console::Output& out) {
            if (!csg_scene_) { out.PrintLine("CSG scene not initialised"); return; }
            SeedDefaultCsgScene();
            out.PrintLine("csg: reset to drilled cube (root = 3, box - sphere)");
        });

    C.RegisterCommand("csg_box",
        "csg_box <id> <sx> <sy> <sz> <tx> <ty> <tz>: add a box leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 7) { out.PrintLine("usage: csg_box <id> <sx> <sy> <sz> <tx> <ty> <tz>"); return; }
            std::uint32_t id;
            float sx, sy, sz, tx, ty, tz;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], sx) || !ParseFloat(args[2], sy) || !ParseFloat(args[3], sz) ||
                !ParseFloat(args[4], tx) || !ParseFloat(args[5], ty) || !ParseFloat(args[6], tz)) {
                out.PrintLine("csg_box: arg parse failed");
                return;
            }
            if (!csg_scene_->AddBox(id, sx, sy, sz, tx, ty, tz)) {
                out.FormatLine("csg_box: id {} already exists or extents are non-positive", id);
                return;
            }
            out.FormatLine("csg: added box id={} ({}x{}x{} @ {} {} {})", id, sx, sy, sz, tx, ty, tz);
        });

    C.RegisterCommand("csg_sphere",
        "csg_sphere <id> <radius> <segments> <tx> <ty> <tz>: add a sphere leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 6) { out.PrintLine("usage: csg_sphere <id> <radius> <segments> <tx> <ty> <tz>"); return; }
            std::uint32_t id, segs;
            float radius, tx, ty, tz;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], radius) ||
                !ParseUint(args[2], segs) ||
                !ParseFloat(args[3], tx) || !ParseFloat(args[4], ty) || !ParseFloat(args[5], tz)) {
                out.PrintLine("csg_sphere: arg parse failed");
                return;
            }
            if (!csg_scene_->AddSphere(id, radius, static_cast<int>(segs), tx, ty, tz)) {
                out.FormatLine("csg_sphere: id {} already exists or radius is non-positive", id);
                return;
            }
            out.FormatLine("csg: added sphere id={} (r={} segs={} @ {} {} {})", id, radius, segs, tx, ty, tz);
        });

    C.RegisterCommand("csg_cylinder",
        "csg_cylinder <id> <radius> <height> <segments> <tx> <ty> <tz>: add a cylinder leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 7) { out.PrintLine("usage: csg_cylinder <id> <radius> <height> <segments> <tx> <ty> <tz>"); return; }
            std::uint32_t id, segs;
            float radius, height, tx, ty, tz;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], radius) || !ParseFloat(args[2], height) ||
                !ParseUint(args[3], segs) ||
                !ParseFloat(args[4], tx) || !ParseFloat(args[5], ty) || !ParseFloat(args[6], tz)) {
                out.PrintLine("csg_cylinder: arg parse failed");
                return;
            }
            if (!csg_scene_->AddCylinder(id, radius, height, static_cast<int>(segs), tx, ty, tz)) {
                out.FormatLine("csg_cylinder: id {} already exists or extents are non-positive", id);
                return;
            }
            out.FormatLine("csg: added cylinder id={} (r={} h={} segs={} @ {} {} {})",
                           id, radius, height, segs, tx, ty, tz);
        });

    C.RegisterCommand("csg_op",
        "csg_op <id> <union|subtract|intersect> <left_id> <right_id>: combine two nodes.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) { out.PrintLine("usage: csg_op <id> <union|subtract|intersect> <left> <right>"); return; }
            std::uint32_t id, l, r;
            if (!ParseUint(args[0], id) || !ParseUint(args[2], l) || !ParseUint(args[3], r)) {
                out.PrintLine("csg_op: id parse failed");
                return;
            }
            pt::csg::OpType op;
            if      (args[1] == "union")     op = pt::csg::OpType::Union;
            else if (args[1] == "subtract")  op = pt::csg::OpType::Subtract;
            else if (args[1] == "intersect") op = pt::csg::OpType::Intersect;
            else { out.FormatLine("csg_op: unknown op '{}' (want union|subtract|intersect)", args[1]); return; }
            if (!csg_scene_->Combine(id, op, l, r)) {
                out.FormatLine("csg_op: id {} already exists or operands {} / {} missing", id, l, r);
                return;
            }
            out.FormatLine("csg: combined id={} = {}({}, {})", id, pt::csg::OpName(op), l, r);
        });

    C.RegisterCommand("csg_remove",
        "csg_remove <id>: drop a node (and any internal nodes referencing it).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: csg_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("csg_remove: id parse failed"); return; }
            std::size_t n = csg_scene_->Remove(id);
            if (n == 0) out.FormatLine("csg_remove: id {} not found", id);
            else        out.FormatLine("csg: removed {} node(s) (cascaded from {})", n, id);
        });

    C.RegisterCommand("csg_set_root",
        "csg_set_root <id>: render the subtree rooted at <id>.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: csg_set_root <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("csg_set_root: id parse failed"); return; }
            if (!csg_scene_->SetRoot(id)) {
                out.FormatLine("csg_set_root: id {} not found", id);
                return;
            }
            out.FormatLine("csg: root set to {}", id);
        });

    // glTF 2.0 importer (#79, MVP). Loads a single static mesh from
    // disk and swaps it into the mesh-path resources (vbuf/ibuf + BVH +
    // optional BLAS/TLAS) -- the exact same upload path the CSG bake
    // produces, so the renderer sees no difference between a procedural
    // CSG mesh and an imported .gltf/.glb. The CSG console commands
    // (csg_*) continue to work; running `csg_box` etc after a glTF load
    // will dirty the CSG scene and the next bake overwrites the glTF
    // mesh on the next frame (intentional -- "last writer wins").
    // Backend switches (RequestBackendSwitch) force a CSG re-bake which
    // will also wipe the glTF mesh; re-run `mesh_load_gltf` after the
    // swap if needed.
    //
    // MVP scope: single mesh, single primitive, base-color factor +
    // optional base-color texture. Texture pixels are decoded and held
    // on the GltfMesh result for the textured-materials work (#74) to
    // pick up; the path tracer's shading uniform isn't plumbed for
    // per-mesh textures yet, so the texture is recorded but not yet
    // rendered (the mesh ships with whatever default color the shader
    // applies to CSG meshes).
    C.RegisterCommand("mesh_load_gltf",
        "mesh_load_gltf <path>: import a static .gltf/.glb mesh (MVP -- "
        "no animation/skins/scene-graph; replaces the current mesh).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: mesh_load_gltf <path>");
                return;
            }
            if (!device_) {
                out.PrintLine("mesh_load_gltf: device not ready");
                return;
            }

            std::string err;
            auto loaded = pt::renderer::LoadGltf(std::string(args[0]), &err);
            if (!loaded) {
                out.FormatLine("mesh_load_gltf: {}", err);
                return;
            }

            // Drain any in-flight CSG bake worker so we don't race the
            // resource swap below. EnsureMeshUpdated's main-thread
            // consume path runs on a different tick than the console
            // command lane, so if the worker is mid-execution
            // (bake_phase_ == 1) we have to wait it out before
            // RebuildMeshResources reallocates the shared
            // box_vbuf_id_ / box_ibuf_id_ / tri_bvh_* slots.
            if (bake_phase_.load(std::memory_order_acquire) == 1 &&
                jobs_ && bake_handle_.internal != nullptr) {
                jobs_->Wait(bake_handle_);
            }
            bake_handle_ = {};
            // Consume any pending bake result so the next
            // EnsureMeshUpdated tick doesn't immediately overwrite our
            // glTF mesh with the stale CSG bake we just superseded.
            pending_baked_.reset();
            bake_phase_.store(0, std::memory_order_release);
            // Also acknowledge the CSG scene as clean so its initial
            // post-seed Dirty()==true flag (set during Init()) doesn't
            // re-trigger a bake on the next frame and wipe the glTF
            // mesh. csg_box / csg_sphere / etc still re-dirty as
            // intended, so the CSG path stays usable.
            if (csg_scene_) csg_scene_->AcknowledgeClean();
            force_mesh_rebuild_ = false;

            // Convert GltfMesh -> BakedMesh (same triangle-soup layout)
            // and hand it to the existing mesh-upload pipeline.
            pt::csg::BakedMesh baked;
            baked.positions = std::move(loaded->positions);
            baked.normals   = std::move(loaded->normals);
            baked.indices   = std::move(loaded->indices);
            RebuildMeshResources(baked);
            accum_dirty_ = true;

            out.FormatLine("mesh_load_gltf: loaded '{}' ({} verts, {} tris, base color "
                           "{:.3f} {:.3f} {:.3f} {:.3f}{})",
                           loaded->source_path,
                           baked.VertexCount(),
                           baked.TriangleCount(),
                           loaded->base_color_factor[0],
                           loaded->base_color_factor[1],
                           loaded->base_color_factor[2],
                           loaded->base_color_factor[3],
                           loaded->base_color_texture.Empty()
                               ? ""
                               : ", base color tex present (not yet rendered -- see #74)");
        });
}

namespace {

bool ParseMaterial(std::string_view s, Engine::AnalyticPrim::Material& out) {
    if (s == "lambert")    { out = Engine::AnalyticPrim::Lambert;    return true; }
    if (s == "metal")      { out = Engine::AnalyticPrim::Metal;      return true; }
    if (s == "dielectric") { out = Engine::AnalyticPrim::Dielectric; return true; }
    if (s == "water")      { out = Engine::AnalyticPrim::Water;      return true; }
    return false;
}
const char* MaterialName(Engine::AnalyticPrim::Material m) {
    switch (m) {
        case Engine::AnalyticPrim::Lambert:    return "lambert";
        case Engine::AnalyticPrim::Metal:      return "metal";
        case Engine::AnalyticPrim::Dielectric: return "dielectric";
        case Engine::AnalyticPrim::Water:      return "water";
    }
    return "?";
}

}  // namespace

void Engine::RegisterPrimCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("prim_list",
        "List analytic primitives (sphere/plane).",
        [this](auto, pt::console::Output& out) {
            if (primitives_.empty()) { out.PrintLine("(no primitives)"); return; }
            out.FormatLine("{} primitive(s):", primitives_.size());
            for (const auto& [id, p] : primitives_) {
                if (p.type == AnalyticPrim::Sphere) {
                    out.FormatLine("  {:>3}  sphere  c=({:.2f} {:.2f} {:.2f}) r={:.3f}  {}  rgb=({:.2f} {:.2f} {:.2f}) rough={:.2f} ior={:.2f}",
                                   id,
                                   p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2],
                                   p.radius_or_d,
                                   MaterialName(p.material),
                                   p.albedo[0], p.albedo[1], p.albedo[2],
                                   p.roughness, p.ior);
                } else {
                    out.FormatLine("  {:>3}  plane   n=({:.2f} {:.2f} {:.2f}) d={:.3f}  {}  rgb=({:.2f} {:.2f} {:.2f})",
                                   id,
                                   p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2],
                                   p.radius_or_d,
                                   MaterialName(p.material),
                                   p.albedo[0], p.albedo[1], p.albedo[2]);
                }
            }
        });

    C.RegisterCommand("prim_clear",
        "Remove all analytic primitives.",
        [this](auto, pt::console::Output& out) {
            primitives_.clear();
            // Drop the r_phys_debug_visualize cache (#181): with every
            // underlying prim gone, the per-prim cache entries would
            // just be orphan keys nothing can reference.
            phys_debug_color_cache_.clear();
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.PrintLine("primitives: cleared");
        });

    C.RegisterCommand("prim_reset",
        "Reset analytic primitives to the default 3-sphere + ground scene.",
        [this](auto, pt::console::Output& out) {
            // SeedDefaultPrimitives replaces every prim wholesale, so the
            // old albedo cache (#181) is referentially stale -- drop it.
            phys_debug_color_cache_.clear();
            SeedDefaultPrimitives();
            out.PrintLine("primitives: reset to default (red Lambert, gold metal, glass + ground)");
        });

    C.RegisterCommand("prim_remove",
        "prim_remove <id>: drop a primitive.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: prim_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("prim_remove: id parse failed"); return; }
            if (primitives_.erase(id) == 0) {
                out.FormatLine("prim_remove: id {} not found", id);
                return;
            }
            // Drop the matching r_phys_debug_visualize cache entry (#181)
            // if any. The cached albedo was the value present BEFORE the
            // viz painted over it, but the underlying prim is gone now so
            // the cached entry would just leak. erase is a no-op when the
            // key isn't in the cache (the prim was non-phys, or the cvar
            // was never enabled).
            phys_debug_color_cache_.erase(id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: removed id {}", id);
        });

    C.RegisterCommand("prim_sphere",
        "prim_sphere <id> <x> <y> <z> <radius> <lambert|metal|dielectric|water> <r> <g> <b> [roughness] [ior]: add or replace a sphere.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() < 9 || args.size() > 11) {
                out.PrintLine("usage: prim_sphere <id> <x> <y> <z> <radius> <material> <r> <g> <b> [roughness=0] [ior=1.5]");
                return;
            }
            std::uint32_t id;
            float x, y, z, radius, r, g, b;
            float roughness = 0.0f, ior = 1.5f;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], radius) ||
                !ParseFloat(args[6], r) || !ParseFloat(args[7], g) || !ParseFloat(args[8], b)) {
                out.PrintLine("prim_sphere: arg parse failed");
                return;
            }
            AnalyticPrim::Material mat;
            if (!ParseMaterial(args[5], mat)) {
                out.FormatLine("prim_sphere: unknown material '{}' (want lambert|metal|dielectric|water)", args[5]);
                return;
            }
            if (radius <= 0.0f) { out.PrintLine("prim_sphere: radius must be > 0"); return; }
            if (args.size() >= 10 && !ParseFloat(args[9],  roughness)) { out.PrintLine("prim_sphere: bad roughness"); return; }
            if (args.size() >= 11 && !ParseFloat(args[10], ior))       { out.PrintLine("prim_sphere: bad ior");       return; }

            AnalyticPrim p{};
            p.type        = AnalyticPrim::Sphere;
            p.material    = mat;
            p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
            // Motion blur (#85): seed prev == curr so a newly added prim
            // produces no streak on the first frame.
            p.prev_pos_or_n[0] = x; p.prev_pos_or_n[1] = y; p.prev_pos_or_n[2] = z;
            p.radius_or_d = radius;
            p.albedo[0]   = r; p.albedo[1] = g; p.albedo[2] = b;
            p.roughness   = roughness;
            p.ior         = ior;
            primitives_[id] = p;
            // If the user is replacing a sphere that had a cached pre-
            // viz albedo (#181), the cached value is now stale -- drop
            // it so a later r_phys_debug_visualize 1 -> 0 transition
            // restores the NEW albedo just set here, not the old one.
            phys_debug_color_cache_.erase(id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: sphere id={} ({} @ {:.2f} {:.2f} {:.2f} r={})",
                           id, MaterialName(mat), x, y, z, radius);
        });

    C.RegisterCommand("prim_plane",
        "prim_plane <id> <nx> <ny> <nz> <d> <lambert|metal|dielectric|water> <r> <g> <b>: add or replace an infinite plane (n . p + d = 0).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: prim_plane <id> <nx> <ny> <nz> <d> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float nx, ny, nz, d, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], nx) || !ParseFloat(args[2], ny) || !ParseFloat(args[3], nz) ||
                !ParseFloat(args[4], d) ||
                !ParseFloat(args[6], r) || !ParseFloat(args[7], g) || !ParseFloat(args[8], b)) {
                out.PrintLine("prim_plane: arg parse failed");
                return;
            }
            AnalyticPrim::Material mat;
            if (!ParseMaterial(args[5], mat)) {
                out.FormatLine("prim_plane: unknown material '{}'", args[5]);
                return;
            }
            const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len < 1e-6f) { out.PrintLine("prim_plane: normal magnitude is zero"); return; }
            nx /= len; ny /= len; nz /= len;

            AnalyticPrim p{};
            p.type        = AnalyticPrim::Plane;
            p.material    = mat;
            p.pos_or_n[0] = nx; p.pos_or_n[1] = ny; p.pos_or_n[2] = nz;
            // Planes don't move; prev_pos == pos_or_n keeps the GPU layout
            // uniform without special-casing the plane branch.
            p.prev_pos_or_n[0] = nx; p.prev_pos_or_n[1] = ny; p.prev_pos_or_n[2] = nz;
            p.radius_or_d = d;
            p.albedo[0]   = r; p.albedo[1] = g; p.albedo[2] = b;
            p.roughness   = 0.0f;
            p.ior         = 1.0f;
            primitives_[id] = p;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: plane id={} ({} n=({:.2f} {:.2f} {:.2f}) d={:.3f})",
                           id, MaterialName(mat), nx, ny, nz, d);
        });

    // --- Editor backend (agent-19) property-edit dispatch -------------------
    // Fine-grained mutators for the editor agents' property panels. Each
    // sets primitives_dirty_ + accum_dirty_ so the GPU sees the update
    // (BVH refit + accum reset) on the next frame. Going through the
    // console keeps the change-tracking + WebSocket scene_dirty broadcast
    // path in the loop -- UI agents MUST NOT mutate primitives_ directly.
    //
    // Undo: skipped for prim ops in this PR (the cvar-undo snapshot
    // mechanism only tracks cvars, not the primitives map). Flagged as a
    // follow-up for the editor roadmap; once a prim-undo recorder lands
    // it will hook in here.

    auto find_prim = [this](std::string_view sv,
                            pt::console::Output& out,
                            std::uint32_t& out_id,
                            AnalyticPrim*& out_prim) -> bool {
        if (!ParseUint(sv, out_id)) {
            out.FormatLine("prim id parse failed: '{}'", sv);
            return false;
        }
        auto it = primitives_.find(out_id);
        if (it == primitives_.end()) {
            out.FormatLine("prim id {} not found", out_id);
            return false;
        }
        out_prim = &it->second;
        return true;
    };

    // Wave-5 polish: this used to be a local lambda. Promoted to an
    // Engine member (BroadcastSceneDirty) so light_*, prim_set_radius,
    // and any future mutator outside this function can also fire it.
    // Local alias kept for in-function ergonomics.
    auto broadcast_scene_dirty = [this]() { BroadcastSceneDirty(); };

    C.RegisterCommand("prim_set_pos",
        "prim_set_pos <id> <x> <y> <z>: move a primitive (sphere center or "
        "plane normal). For planes, the magnitude is normalized.",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: prim_set_pos <id> <x> <y> <z>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float x, y, z;
            if (!ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z)) {
                out.PrintLine("prim_set_pos: float parse failed");
                return;
            }
            if (p->type == AnalyticPrim::Plane) {
                // Treat (x,y,z) as the new plane normal; renormalize so a
                // typed (1,1,0) becomes a valid unit vector before
                // upload. Same magnitude check `prim_plane` uses.
                const float len = std::sqrt(x*x + y*y + z*z);
                if (len < 1e-6f) {
                    out.PrintLine("prim_set_pos: zero-magnitude plane normal");
                    return;
                }
                x /= len; y /= len; z /= len;
            }
            p->pos_or_n[0] = x; p->pos_or_n[1] = y; p->pos_or_n[2] = z;
            // Sphere prev_pos is normally the previous frame's position
            // (motion blur). A direct teleport via prim_set_pos should
            // produce a zero-length streak on the next frame -- snap
            // prev to the new pos. Plane prev mirrors pos by convention.
            p->prev_pos_or_n[0] = x; p->prev_pos_or_n[1] = y; p->prev_pos_or_n[2] = z;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_pos: id {} -> ({:.3f} {:.3f} {:.3f})", id, x, y, z);
        });

    C.RegisterCommand("prim_set_albedo",
        "prim_set_albedo <id> <r> <g> <b>: set primitive base color (linear).",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: prim_set_albedo <id> <r> <g> <b>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float r, g, b;
            if (!ParseFloat(args[1], r) || !ParseFloat(args[2], g) || !ParseFloat(args[3], b)) {
                out.PrintLine("prim_set_albedo: float parse failed");
                return;
            }
            p->albedo[0] = r; p->albedo[1] = g; p->albedo[2] = b;
            // Drop the cached pre-viz albedo (#181) -- next falling-edge
            // restore would otherwise paint the previous albedo over the
            // value the user just typed.
            phys_debug_color_cache_.erase(id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_albedo: id {} -> ({:.3f} {:.3f} {:.3f})", id, r, g, b);
        });

    C.RegisterCommand("prim_set_emission",
        "prim_set_emission <id> <r> <g> <b>: set per-prim radiant emission "
        "(W/sr per channel). Non-zero makes the prim self-luminous (#181).",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: prim_set_emission <id> <r> <g> <b>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float r, g, b;
            if (!ParseFloat(args[1], r) || !ParseFloat(args[2], g) || !ParseFloat(args[3], b)) {
                out.PrintLine("prim_set_emission: float parse failed");
                return;
            }
            p->emission[0] = r; p->emission[1] = g; p->emission[2] = b;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_emission: id {} -> ({:.3f} {:.3f} {:.3f})", id, r, g, b);
        });

    C.RegisterCommand("prim_set_roughness",
        "prim_set_roughness <id> <v>: 0 = mirror, 1 = fully rough. "
        "Mainly affects MAT_METAL surfaces.",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: prim_set_roughness <id> <v>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float v;
            if (!ParseFloat(args[1], v)) {
                out.PrintLine("prim_set_roughness: float parse failed");
                return;
            }
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            p->roughness = v;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_roughness: id {} -> {:.3f}", id, v);
        });

    C.RegisterCommand("prim_set_ior",
        "prim_set_ior <id> <v>: refractive index (dielectric / water).",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: prim_set_ior <id> <v>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float v;
            if (!ParseFloat(args[1], v)) {
                out.PrintLine("prim_set_ior: float parse failed");
                return;
            }
            // Clamp to the same [1.0, 2.4] range the water-Phase 1 shader
            // uses; outside this range the Fresnel + refraction math
            // produces visually broken results (TIR everywhere or
            // negative critical-angle).
            if (v < 1.0f) v = 1.0f;
            if (v > 2.4f) v = 2.4f;
            p->ior = v;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_ior: id {} -> {:.3f}", id, v);
        });

    C.RegisterCommand("prim_set_material",
        "prim_set_material <id> <lambert|metal|dielectric|water>: swap "
        "BRDF model. Existing albedo / roughness / ior are preserved.",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: prim_set_material <id> <lambert|metal|dielectric|water>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            AnalyticPrim::Material mat;
            if (!ParseMaterial(args[1], mat)) {
                out.FormatLine("prim_set_material: unknown material '{}' "
                               "(want lambert|metal|dielectric|water)",
                               args[1]);
                return;
            }
            p->material = mat;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_material: id {} -> {}", id, MaterialName(mat));
        });

    // --- Rotation gizmo dispatch (#206) -------------------------------------
    // Sets a primitive's orientation quaternion. Input is a raw xyzw
    // tuple which we normalize before storing (a non-unit quat would
    // alter both the rotation angle and the vector magnitude through
    // the shader's quatRotate formula). Spheres are rotation-symmetric
    // so the orientation has no visible effect but we still store it
    // (so a later change of primitive type or selection-driven UI
    // observation reads a stable value). Planes rotate their stored
    // normal by this quaternion in the shader -- see PathTrace.slang
    // testAnalyticPrim's PRIM_PLANE branch.
    C.RegisterCommand("prim_set_rotation",
        "prim_set_rotation <id> <qx> <qy> <qz> <qw>: set the primitive's "
        "orientation quaternion (xyzw). Input is normalized. Identity is "
        "(0, 0, 0, 1). Spheres are rotation-symmetric so this is visually "
        "a no-op for them; planes tilt their normal by the rotation.",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 5) {
                out.PrintLine("usage: prim_set_rotation <id> <qx> <qy> <qz> <qw>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float qx, qy, qz, qw;
            if (!ParseFloat(args[1], qx) || !ParseFloat(args[2], qy) ||
                !ParseFloat(args[3], qz) || !ParseFloat(args[4], qw)) {
                out.PrintLine("prim_set_rotation: float parse failed");
                return;
            }
            const float mag = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (mag < 1e-6f) {
                out.PrintLine("prim_set_rotation: zero-magnitude quaternion");
                return;
            }
            const float inv = 1.0f / mag;
            p->orient[0] = qx * inv;
            p->orient[1] = qy * inv;
            p->orient[2] = qz * inv;
            p->orient[3] = qw * inv;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_rotation: id {} -> ({:.4f} {:.4f} {:.4f} {:.4f})",
                           id, p->orient[0], p->orient[1], p->orient[2], p->orient[3]);
        });

    C.RegisterCommand("prim_set_rotation_euler",
        "prim_set_rotation_euler <id> <pitch_deg> <yaw_deg> <roll_deg>: set "
        "the primitive's orientation from Euler angles (XYZ order, degrees). "
        "Converts internally to the canonical quaternion stored on the prim.",
        [this, find_prim, broadcast_scene_dirty]
        (auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: prim_set_rotation_euler <id> <pitch_deg> <yaw_deg> <roll_deg>");
                return;
            }
            std::uint32_t id; AnalyticPrim* p = nullptr;
            if (!find_prim(args[0], out, id, p)) return;
            float pitch_deg, yaw_deg, roll_deg;
            if (!ParseFloat(args[1], pitch_deg) ||
                !ParseFloat(args[2], yaw_deg)   ||
                !ParseFloat(args[3], roll_deg)) {
                out.PrintLine("prim_set_rotation_euler: float parse failed");
                return;
            }
            // XYZ-intrinsic Euler -> quaternion. Half-angle formulation
            // produces a unit quaternion straight out so we don't need
            // the normalisation step that prim_set_rotation does.
            const float kDeg2Rad = 3.14159265358979323846f / 180.0f;
            const float hx = pitch_deg * kDeg2Rad * 0.5f;
            const float hy = yaw_deg   * kDeg2Rad * 0.5f;
            const float hz = roll_deg  * kDeg2Rad * 0.5f;
            const float cx = std::cos(hx), sx = std::sin(hx);
            const float cy = std::cos(hy), sy = std::sin(hy);
            const float cz = std::cos(hz), sz = std::sin(hz);
            // Quaternion q = qz * qy * qx (roll then yaw then pitch).
            // Standard XYZ-intrinsic Tait-Bryan composition.
            p->orient[0] = sx * cy * cz + cx * sy * sz;
            p->orient[1] = cx * sy * cz - sx * cy * sz;
            p->orient[2] = cx * cy * sz + sx * sy * cz;
            p->orient[3] = cx * cy * cz - sx * sy * sz;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_set_rotation_euler: id {} -> quat ({:.4f} {:.4f} {:.4f} {:.4f})",
                           id, p->orient[0], p->orient[1], p->orient[2], p->orient[3]);
        });

    C.RegisterCommand("prim_delete",
        "prim_delete <id>: remove a primitive (editor alias for prim_remove). "
        "Clears the editor selection when the deleted prim was selected.",
        [this, broadcast_scene_dirty](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: prim_delete <id>");
                return;
            }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) {
                out.PrintLine("prim_delete: id parse failed");
                return;
            }
            if (primitives_.erase(id) == 0) {
                out.FormatLine("prim_delete: id {} not found", id);
                return;
            }
            phys_debug_color_cache_.erase(id);
            // Drop the editor selection if it pointed at the body we
            // just deleted -- otherwise the outline shader would chase
            // a now-dangling id forever (a freshly-spawned prim that
            // reuses the slot would pick up the highlight).
            if (selected_kind_ == SelectionKind::AnalyticPrim &&
                selected_prim_id_ == id) {
                SetSelection(SelectionKind::None, 0);
            }
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_delete: removed id {}", id);
        });

    C.RegisterCommand("prim_duplicate",
        "prim_duplicate <id>: clone a primitive into a new id, return the "
        "new id. The new id is the next available slot above the source.",
        [this, broadcast_scene_dirty](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: prim_duplicate <id>");
                return;
            }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) {
                out.PrintLine("prim_duplicate: id parse failed");
                return;
            }
            auto it = primitives_.find(id);
            if (it == primitives_.end()) {
                out.FormatLine("prim_duplicate: id {} not found", id);
                return;
            }
            // Find the next unused id. Walk from max(source+1, 1).
            // O(N) over the primitives map -- fine for hundreds of
            // entries; the editor doesn't need a faster allocator
            // here. Start above the source so a freshly-typed
            // `prim_duplicate 1` lands at id=2 (if free) for an
            // intuitive sequential id pattern.
            std::uint32_t new_id = id + 1u;
            if (new_id == 0u) new_id = 1u;  // wraparound paranoia
            while (primitives_.count(new_id) != 0u) {
                ++new_id;
                if (new_id == 0u) {
                    out.PrintLine("prim_duplicate: id space exhausted");
                    return;
                }
            }
            AnalyticPrim copy = it->second;
            primitives_[new_id] = copy;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            broadcast_scene_dirty();
            out.FormatLine("prim_duplicate: cloned id {} -> {}", id, new_id);
        });
    // --- end Editor backend property-edit dispatch -------------------------

    // --- Editor backend (agent-19) scene-graph dump command ----------------
    // Returns the same JSON tree the WebSocket `list_scene` message
    // produces. Useful at the console prompt for ad-hoc scene
    // inspection without a UI client connected.
    C.RegisterCommand("scene_dump_json",
        "Print the editor scene-graph snapshot (primitives / lights / sdf / "
        "rigid bodies + current selection) as a JSON tree.",
        [this](auto, pt::console::Output& out) {
            const auto j = pt::editor::SerializeScene(*this);
            // dump(2) for human-readable pretty-print; the WebSocket
            // path uses dump() (compact) since the wire is parsed.
            out.PrintLine(j.dump(2));
        });
    // --- end Editor backend scene-graph dump -------------------------------

    // --- Editor backend (agent-19) selection setter ------------------------
    // Thin console wrapper around Engine::SetSelection so the
    // WebSocket select handler can route through QueueExecute -> Drain
    // and land on the main thread for the actual state mutation. Also
    // exposed to scripts so an autoexec can seed an initial selection
    // for e.g. a scripted scene tour.
    //
    // Format: `select_set <none|prim|light|sdf|rb> <id>`
    // none + id=0 clears the selection. Any other (kind, id=0) pair is
    // treated as a clear too (SetSelection normalises None to id=0
    // anyway). Unknown kind strings fall through to "none" so a
    // mistyped command doesn't silently latch a wrong-kind selection.
    C.RegisterCommand("select_set",
        "select_set <none|prim|light|sdf|rb> <id>: set the editor "
        "selection. Used by the WebSocket select message and scripts.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: select_set <none|prim|light|sdf|rb> <id>");
                return;
            }
            const int kind_int = pt::editor::SelectionKindFromString(args[0]);
            std::uint32_t id = 0;
            if (!ParseUint(args[1], id)) {
                out.PrintLine("select_set: id parse failed");
                return;
            }
            SetSelection(static_cast<SelectionKind>(kind_int), id);
            out.FormatLine("select_set: kind={} id={}",
                           pt::editor::SelectionKindToString(kind_int), id);
        });
    // --- end Editor backend selection setter -------------------------------
}

// --- Editor 3D-transform gizmo (issue: editor 3D gizmos) -------------------
//
// Registers the cvars + commands the gizmo subsystem reads / dispatches.
// Cvars:
//   r_editor_gizmo      -- master on/off switch (default 1).
//   gizmo_mode          -- "translate" | "rotate" | "scale" (default
//                          "translate"). G / R / S hotkeys flip this
//                          via SetCVarOverride from the engine's
//                          key handler.
//   r_editor_gizmo_xray -- 1 = always-on-top (default; v1 only mode),
//                          0 = z-test against depth_tex (deferred to
//                          a follow-up; see EditorOverlay.slang).
//
// Commands:
//   gizmo_select <prim_id>    -- set selected_prim_id_ to <prim_id> (or
//                                0 to deselect). Used by tests and as
//                                the manual fallback until agent-19's
//                                raycast pick path lands.
//   prim_set_pos <id> <x> <y> <z>
//                              -- mutate the position of the analytic
//                                primitive at <id>. This is the
//                                dispatch path the gizmo drag loop
//                                uses; lifting it out into a console
//                                command means undo/redo + the
//                                eventual snapshot system route the
//                                same way through Console::Execute as
//                                the user typing the line manually.
//                                Agent-19's branch introduces a
//                                richer per-kind dispatch (Sphere /
//                                Csg / Sdf); on merge our local
//                                command will defer to that.
void Engine::RegisterEditorGizmoCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCVar("r_editor_gizmo", "1",
        "Editor 3D-transform gizmo master switch. 1 = render an XYZ "
        "axis triad (translate mode) / ring set (rotate) / cube tips "
        "(scale) over the selected analytic primitive; 0 = no gizmo. "
        "Has no effect when nothing is selected (selected_prim_id_ "
        "== 0).",
        pt::console::CVAR_ARCHIVE);

    C.RegisterCVar("gizmo_mode", "translate",
        "Editor gizmo mode: translate | rotate | scale. The G / R / "
        "S hotkeys flip this via SetCVarOverride from the engine's "
        "key handler. Rotate mode does NOT currently dispatch a "
        "rotation mutation (analytic prims don't carry orientation "
        "in this codebase); it draws three rings for visual feedback "
        "only. Scale mode mutates radius_or_d for spheres.",
        pt::console::CVAR_ARCHIVE);

    C.RegisterCVar("r_editor_gizmo_xray", "1",
        "Editor gizmo z-test mode. 1 = always-on-top (Blender's "
        "default, current v1 behaviour); 0 = z-test against the "
        "scene's primary-hit depth buffer (deferred -- depth_tex is "
        "only allocated when the denoiser is active, so the kernel "
        "transparently falls back to always-on-top until that "
        "lifecycle plumbing lands).",
        pt::console::CVAR_ARCHIVE);

    // gizmo_select is a thin wrapper around the canonical
    // SetSelection(kind, id) that agent-19's PR #201 introduced. We
    // keep it as a separate command because the existing select cmd
    // takes <kind> <id> -- the editor's "I want to focus on prim N"
    // shorthand is more ergonomic and matches the hotkey workflow.
    C.RegisterCommand("gizmo_select",
        "gizmo_select <prim_id> -- focus the editor gizmo on an "
        "analytic primitive. Pass 0 to deselect (gizmo hides).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: gizmo_select <prim_id>");
                return;
            }
            std::uint32_t id = 0;
            if (!ParseUint(args[0], id)) {
                out.PrintLine("gizmo_select: id parse failed");
                return;
            }
            if (id != 0 && primitives_.find(id) == primitives_.end()) {
                out.FormatLine("gizmo_select: id {} is not in primitives_", id);
                return;
            }
            SetSelection(id == 0 ? SelectionKind::None
                                 : SelectionKind::AnalyticPrim,
                         id);
            // End any in-flight drag when the selection changes so a
            // stale drag from the previous prim doesn't bleed into the
            // new one.
            if (editor_gizmo_.IsDragging()) editor_gizmo_.EndDrag();
            if (id == 0) out.PrintLine("editor: selection cleared");
            else         out.FormatLine("editor: selected prim id={}", id);
        });

    // prim_set_pos is provided by agent-19's PR #201 with richer
    // plane-normal handling + WebSocket scene_dirty broadcast. The
    // gizmo drag loop just dispatches `prim_set_pos` via
    // Console::Execute and inherits that behaviour for free.

    C.RegisterCommand("prim_set_radius",
        "prim_set_radius <id> <r> -- set the radius of an analytic "
        "sphere primitive. Used by the editor gizmo scale-mode drag.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: prim_set_radius <id> <r>");
                return;
            }
            std::uint32_t id = 0;
            float r = 0.0f;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], r)) {
                out.PrintLine("prim_set_radius: arg parse failed");
                return;
            }
            if (r <= 0.0f) {
                out.PrintLine("prim_set_radius: radius must be > 0");
                return;
            }
            auto it = primitives_.find(id);
            if (it == primitives_.end()) {
                out.FormatLine("prim_set_radius: id {} not found", id);
                return;
            }
            if (it->second.type != AnalyticPrim::Sphere) {
                out.FormatLine("prim_set_radius: id {} is not a sphere", id);
                return;
            }
            it->second.radius_or_d = r;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            BroadcastSceneDirty();
            out.FormatLine("primitives: id {} radius -> {:.3f}", id, r);
        });

    // --- prim_set_plane_d -----------------------------------------------
    // Inspector "plane d" field stays disabled until this exists. Lets
    // the user slide a water/floor plane along its normal without
    // re-creating it. For a plane with normal n and d the plane
    // equation is n.p + d = 0, so increasing d moves the plane in the
    // -n direction (toward negative-y for a y-up floor).
    C.RegisterCommand("prim_set_plane_d",
        "prim_set_plane_d <id> <d>: set the d coefficient of an analytic "
        "plane primitive (plane eq: n.p + d = 0). Used by the editor "
        "inspector to slide a plane along its normal.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 2) {
                out.PrintLine("usage: prim_set_plane_d <id> <d>");
                return;
            }
            std::uint32_t id = 0;
            float d = 0.0f;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], d)) {
                out.PrintLine("prim_set_plane_d: arg parse failed");
                return;
            }
            auto it = primitives_.find(id);
            if (it == primitives_.end()) {
                out.FormatLine("prim_set_plane_d: id {} not found", id);
                return;
            }
            if (it->second.type != AnalyticPrim::Plane) {
                out.FormatLine("prim_set_plane_d: id {} is not a plane", id);
                return;
            }
            it->second.radius_or_d = d;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            BroadcastSceneDirty();
            out.FormatLine("primitives: id {} plane.d -> {:.3f}", id, d);
        });

    // --- cam_warp_to ------------------------------------------------------
    // Editor "Goto" button -> nice framing of the selected prim without
    // the cam_slot_9 hack agent-22 was using. Snaps the camera to a
    // pleasant viewing position: 3 units back + 1 unit up from the
    // prim, looking down ~15 degrees. Pure snap, no tween (could be
    // animated in a follow-up but snap is responsive + obvious).
    C.RegisterCommand("cam_warp_to",
        "cam_warp_to <prim_id>: snap the camera to frame an analytic "
        "primitive. Editor 'Goto' button calls this.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: cam_warp_to <prim_id>");
                return;
            }
            std::uint32_t id = 0;
            if (!ParseUint(args[0], id)) {
                out.PrintLine("cam_warp_to: id parse failed");
                return;
            }
            auto it = primitives_.find(id);
            if (it == primitives_.end()) {
                out.FormatLine("cam_warp_to: id {} not found", id);
                return;
            }
            const auto& p = it->second;
            if (!camera_) {
                out.PrintLine("cam_warp_to: camera not initialised");
                return;
            }
            // For spheres frame at radius * 4 distance; for planes use a
            // nominal 4-unit back-off (plane is infinite, framing is
            // arbitrary -- just place the camera somewhere it can see
            // the plane normal end-on).
            float dist = (p.type == AnalyticPrim::Sphere)
                       ? std::max(p.radius_or_d * 4.0f, 2.0f)
                       : 4.0f;
            camera_->pos = glm::vec3(p.pos_or_n[0],
                                     p.pos_or_n[1] + dist * 0.25f,
                                     p.pos_or_n[2] + dist);
            camera_->yaw      = 0.0f;
            camera_->pitch    = glm::radians(-15.0f);
            camera_->ClampPitch();
            accum_dirty_ = true;
            cam_tween_active_ = false;   // any manual snap aborts in-flight tween
            out.FormatLine("cam_warp_to: framed prim id {} at distance {:.2f}",
                           id, dist);
        });

    // --- cam_warp ---------------------------------------------------------
    // Generic camera teleport. Three arg shapes:
    //   cam_warp <x> <y> <z>                              (just position; keep yaw/pitch)
    //   cam_warp <x> <y> <z> <yaw_deg> <pitch_deg>        (position + explicit orientation)
    //   cam_warp <x> <y> <z> look_at <lx> <ly> <lz>       (position + auto-orient toward target)
    // Snaps immediately. Use cam_warp_smooth for tweened motion.
    C.RegisterCommand("cam_warp",
        "cam_warp <x> <y> <z> [yaw_deg pitch_deg | look_at lx ly lz]: "
        "snap the camera to a position, optionally with explicit "
        "orientation or look-at target. Aborts any in-flight smooth tween.",
        [this](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_warp: camera not initialised"); return; }
            const auto n = args.size();
            if (n != 3 && n != 5 && n != 7) {
                out.PrintLine("usage: cam_warp <x> <y> <z> [yaw_deg pitch_deg | look_at lx ly lz]");
                return;
            }
            float x, y, z;
            if (!ParseFloat(args[0], x) || !ParseFloat(args[1], y) || !ParseFloat(args[2], z)) {
                out.PrintLine("cam_warp: position parse failed");
                return;
            }
            float new_yaw   = camera_->yaw;
            float new_pitch = camera_->pitch;
            if (n == 5) {
                float yaw_deg, pitch_deg;
                if (!ParseFloat(args[3], yaw_deg) || !ParseFloat(args[4], pitch_deg)) {
                    out.PrintLine("cam_warp: yaw/pitch parse failed");
                    return;
                }
                new_yaw   = glm::radians(yaw_deg);
                new_pitch = glm::radians(pitch_deg);
            } else if (n == 7) {
                if (args[3] != "look_at") {
                    out.PrintLine("cam_warp: 7-arg form must use 'look_at <lx> <ly> <lz>'");
                    return;
                }
                float lx, ly, lz;
                if (!ParseFloat(args[4], lx) || !ParseFloat(args[5], ly) || !ParseFloat(args[6], lz)) {
                    out.PrintLine("cam_warp: look_at target parse failed");
                    return;
                }
                glm::vec3 dir = glm::vec3(lx - x, ly - y, lz - z);
                float len = glm::length(dir);
                if (len < 1e-6f) {
                    out.PrintLine("cam_warp: look_at target coincident with eye -- keeping orientation");
                } else {
                    dir /= len;
                    // Camera convention (src/renderer/Camera.h): forward =
                    // (sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch)).
                    // Invert: pitch = asin(dir.y); yaw = atan2(dir.x, -dir.z).
                    new_pitch = std::asin(std::clamp(dir.y, -1.0f, 1.0f));
                    new_yaw   = std::atan2(dir.x, -dir.z);
                }
            }
            camera_->pos      = glm::vec3(x, y, z);
            camera_->yaw      = new_yaw;
            camera_->pitch    = new_pitch;
            camera_->ClampPitch();
            accum_dirty_ = true;
            cam_tween_active_ = false;
            out.FormatLine("cam_warp: pos=({:.2f} {:.2f} {:.2f}) yaw={:.1f} pitch={:.1f}",
                           x, y, z,
                           glm::degrees(camera_->yaw),
                           glm::degrees(camera_->pitch));
        });

    // --- cam_warp_smooth --------------------------------------------------
    // Same arg shapes as cam_warp BUT prefixed with a duration in
    // seconds. UpdateCamera() lerps with smoothstep easing toward the
    // target until duration elapses. Any WASD / RMB-look input cancels
    // the tween. Useful for cinematic camera moves, scripted demos,
    // and the editor "Goto" button when you want a smooth feel.
    C.RegisterCommand("cam_warp_smooth",
        "cam_warp_smooth <duration_sec> <x> <y> <z> [yaw_deg pitch_deg | look_at lx ly lz]: "
        "tweened cam_warp with smoothstep easing over duration_sec. "
        "Manual WASD / RMB-look aborts the tween.",
        [this](auto args, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("cam_warp_smooth: camera not initialised"); return; }
            const auto n = args.size();
            if (n != 4 && n != 6 && n != 8) {
                out.PrintLine("usage: cam_warp_smooth <duration_sec> <x> <y> <z> "
                              "[yaw_deg pitch_deg | look_at lx ly lz]");
                return;
            }
            float dur, x, y, z;
            if (!ParseFloat(args[0], dur) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z)) {
                out.PrintLine("cam_warp_smooth: arg parse failed");
                return;
            }
            if (dur <= 0.0f) {
                out.PrintLine("cam_warp_smooth: duration must be > 0 (use cam_warp for instant)");
                return;
            }
            float new_yaw   = camera_->yaw;
            float new_pitch = camera_->pitch;
            if (n == 6) {
                float yaw_deg, pitch_deg;
                if (!ParseFloat(args[4], yaw_deg) || !ParseFloat(args[5], pitch_deg)) {
                    out.PrintLine("cam_warp_smooth: yaw/pitch parse failed");
                    return;
                }
                new_yaw   = glm::radians(yaw_deg);
                new_pitch = glm::radians(pitch_deg);
            } else if (n == 8) {
                if (args[4] != "look_at") {
                    out.PrintLine("cam_warp_smooth: 8-arg form must use 'look_at <lx> <ly> <lz>'");
                    return;
                }
                float lx, ly, lz;
                if (!ParseFloat(args[5], lx) || !ParseFloat(args[6], ly) || !ParseFloat(args[7], lz)) {
                    out.PrintLine("cam_warp_smooth: look_at target parse failed");
                    return;
                }
                glm::vec3 dir = glm::vec3(lx - x, ly - y, lz - z);
                float len = glm::length(dir);
                if (len >= 1e-6f) {
                    dir /= len;
                    new_pitch = std::asin(std::clamp(dir.y, -1.0f, 1.0f));
                    new_yaw   = std::atan2(dir.x, -dir.z);
                }
            }
            // Stash start state. Lerp angles via shortest-path: if the
            // target yaw differs from start by more than pi, wrap it
            // around so the camera spins the short way.
            cam_tween_start_pos_[0] = camera_->pos.x;
            cam_tween_start_pos_[1] = camera_->pos.y;
            cam_tween_start_pos_[2] = camera_->pos.z;
            cam_tween_target_pos_[0] = x;
            cam_tween_target_pos_[1] = y;
            cam_tween_target_pos_[2] = z;
            cam_tween_start_yaw_     = camera_->yaw;
            cam_tween_start_pitch_   = camera_->pitch;
            cam_tween_target_pitch_  = new_pitch;
            constexpr float kPi  = 3.14159265358979323846f;
            constexpr float kTau = kPi * 2.0f;
            float yaw_delta = new_yaw - camera_->yaw;
            // Normalise to (-pi, pi].
            while (yaw_delta >  kPi) yaw_delta -= kTau;
            while (yaw_delta < -kPi) yaw_delta += kTau;
            cam_tween_target_yaw_ = camera_->yaw + yaw_delta;
            cam_tween_t_seconds_  = 0.0;
            cam_tween_duration_   = static_cast<double>(dur);
            cam_tween_active_     = true;
            out.FormatLine("cam_warp_smooth: tweening over {:.2f}s to ({:.2f} {:.2f} {:.2f}) yaw={:.1f} pitch={:.1f}",
                           dur, x, y, z,
                           glm::degrees(new_yaw),
                           glm::degrees(new_pitch));
        });

    // --- prim_set_normal (wave-6 #25 inspector follow-up) -----------------
    // Set the normal of an analytic plane primitive. Input is auto-
    // normalised. Note that this rotates the plane about the origin's
    // projection on the new normal -- d is preserved, so the geometric
    // plane n.p + d = 0 simply uses a new n. To tilt + reposition, the
    // inspector should also dispatch prim_set_plane_d in the same batch.
    C.RegisterCommand("prim_set_normal",
        "prim_set_normal <id> <nx> <ny> <nz>: set the normal of an analytic "
        "plane primitive. Input is auto-normalised. d (plane offset) is "
        "preserved, so the plane rotates around the origin's projection "
        "on the new normal -- to tilt + reposition, also dispatch "
        "prim_set_plane_d in the same batch.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: prim_set_normal <id> <nx> <ny> <nz>");
                return;
            }
            std::uint32_t id = 0;
            float nx, ny, nz;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], nx) || !ParseFloat(args[2], ny) || !ParseFloat(args[3], nz)) {
                out.PrintLine("prim_set_normal: arg parse failed");
                return;
            }
            const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len < 1e-6f) {
                out.PrintLine("prim_set_normal: zero-length normal rejected");
                return;
            }
            auto it = primitives_.find(id);
            if (it == primitives_.end()) {
                out.FormatLine("prim_set_normal: id {} not found", id);
                return;
            }
            if (it->second.type != AnalyticPrim::Plane) {
                out.FormatLine("prim_set_normal: id {} is not a plane", id);
                return;
            }
            it->second.pos_or_n[0] = nx / len;
            it->second.pos_or_n[1] = ny / len;
            it->second.pos_or_n[2] = nz / len;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            BroadcastSceneDirty();
            out.FormatLine("primitives: id {} normal -> ({:.3f} {:.3f} {:.3f})",
                           id, it->second.pos_or_n[0], it->second.pos_or_n[1], it->second.pos_or_n[2]);
        });
}

// --- Editor 3D-transform gizmo per-frame work (issue: editor 3D gizmos) ----
//
// Called from RenderFrame BEFORE the editor_overlay compute dispatch.
// Responsibilities:
//   1. Sync editor_gizmo_'s mode from the gizmo_mode cvar.
//   2. Look up the selected primitive; bail (no upload, no dispatch) if
//      nothing is selected or the id has been removed.
//   3. Compute the per-frame gizmo size from the prim's radius.
//   4. Run hit-test against the polled cursor position to find which
//      axis (if any) the mouse is hovering -- yellow-highlight that one.
//   5. Drive the click+drag state machine via the LMB edge detect:
//        prev=false, now=true  -> begin drag if hovering an axis
//        prev=true, now=true   -> update drag (dispatch prim_set_pos)
//        prev=true, now=false  -> end drag
//      Esc while dragging restores the pre-drag position.
//   6. Rebuild the segment list and WriteBuffer it to
//      editor_overlay_segs_buf_id_.
//
// LMB hit-test is intentionally gated on r_editor_gizmo (so the user
// can keep the gizmo visible but disable interaction for screenshot
// scenarios) AND on the absence of mouse_look_active_ -- RMB camera
// look should never compete with gizmo drag.
void Engine::UpdateEditorGizmoFrame() {
    if (!device_ || !window_ || !camera_) return;

    auto& C = pt::console::Console::Get();
    // Master switch + mode pull.
    bool gizmo_enabled = true;
    if (auto* v = C.FindCVar("r_editor_gizmo")) gizmo_enabled = v->GetBool();

    {
        std::string mode = "translate";
        if (auto* v = C.FindCVar("gizmo_mode")) mode = v->value;
        if      (mode == "rotate") editor_gizmo_.SetMode(pt::renderer::EditorOverlay::Mode::Rotate);
        else if (mode == "scale")  editor_gizmo_.SetMode(pt::renderer::EditorOverlay::Mode::Scale);
        else                       editor_gizmo_.SetMode(pt::renderer::EditorOverlay::Mode::Translate);
    }

    editor_gizmo_.ClearSegments();

    // Gizmo gates on the editor's canonical selection state. Analytic
    // primitives (sphere / plane) drive prim_set_pos + prim_set_rotation;
    // analytic lights (point / spot / sphere / quad) drive
    // light_set_pos + light_set_rotation. Csg / Sdf / RigidBody gizmos
    // are a follow-up (those scopes don't expose a unified "position /
    // rotation" the gizmo can dispatch over yet).
    const bool sel_prim  = selected_kind_ == SelectionKind::AnalyticPrim &&
                           selected_prim_id_ != 0;
    const bool sel_light = selected_kind_ == SelectionKind::Light &&
                           selected_prim_id_ != 0;
    if ((!sel_prim && !sel_light) || !gizmo_enabled) {
        // Drop any in-flight drag if the user just cleared the selection
        // mid-drag (or toggled the master switch off).
        if (editor_gizmo_.IsDragging()) editor_gizmo_.EndDrag();
        return;
    }
    auto prim_it  = sel_prim  ? primitives_.find(selected_prim_id_)
                              : primitives_.end();
    auto light_it = sel_light ? light_prims_.find(selected_prim_id_)
                              : light_prims_.end();
    if ((sel_prim  && prim_it  == primitives_.end()) ||
        (sel_light && light_it == light_prims_.end())) {
        // Selection points at an entity that was removed since.
        SetSelection(SelectionKind::None, 0);
        if (editor_gizmo_.IsDragging()) editor_gizmo_.EndDrag();
        return;
    }

    // Dispatch name prefix selected from the selection kind. The gizmo
    // routes through the console so command-line parity, undo
    // bookkeeping, and the websocket scene_dirty broadcast all stay
    // single-sourced (the prim_set_* / light_set_* handlers above own
    // the writes).
    const char* set_pos_cmd      = sel_light ? "light_set_pos"
                                             : "prim_set_pos";
    const char* set_rotation_cmd = sel_light ? "light_set_rotation"
                                             : "prim_set_rotation";

    // Gizmo origin + size selection:
    //   PRIM Sphere: world-space sphere center stored in pos_or_n.
    //   PRIM Plane : no canonical world center; render at the foot of
    //                the perpendicular from the world origin to the
    //                plane (point on the plane closest to (0,0,0)) so
    //                the rotate rings sit on the surface itself rather
    //                than at infinity. Purely a placement choice --
    //                prim_set_rotation only mutates orient[].
    //   LIGHT (any): anchored at L.pos. Size derived from sphere
    //                radius / quad max(u,v) / spot 1.0 default.
    glm::vec3 origin{0.0f};
    float gizmo_size = 1.0f;
    bool  is_plane   = false;
    if (sel_prim) {
        is_plane = (prim_it->second.type == AnalyticPrim::Plane);
        if (is_plane) {
            // Planes have nothing physical to translate / scale, so we
            // only surface them for rotate mode. Translate / scale mode
            // skips planes (a "move plane along its normal" UX would
            // dispatch prim_set_pos on radius_or_d, out of scope here).
            if (editor_gizmo_.GetMode() != pt::renderer::EditorOverlay::Mode::Rotate) {
                if (editor_gizmo_.IsDragging()) editor_gizmo_.EndDrag();
                return;
            }
            // foot = -d * n (since n is unit, n . foot + d == 0 implies
            // foot . n == -d, so foot = -d * n is the point on the
            // plane nearest the world origin).
            const glm::vec3 n_raw{prim_it->second.pos_or_n[0],
                                  prim_it->second.pos_or_n[1],
                                  prim_it->second.pos_or_n[2]};
            const float d = prim_it->second.radius_or_d;
            origin     = n_raw * (-d);
            gizmo_size = 1.5f;   // fixed world-units arm/ring length
        } else {
            // Sphere path: gizmo anchored at the sphere center, sized
            // relative to the radius like the existing translate gizmo.
            origin = glm::vec3{prim_it->second.pos_or_n[0],
                               prim_it->second.pos_or_n[1],
                               prim_it->second.pos_or_n[2]};
            gizmo_size = std::max(0.5f, prim_it->second.radius_or_d * 1.5f);
        }
    } else {
        // Light path: anchored at L.pos. Sphere lights use their radius;
        // quad lights use max(|u_vec|, v_half) so the gizmo wraps the
        // emitter footprint; point/spot fall back to a sensible default
        // (1.5m) since they have no canonical extent.
        const auto& L = light_it->second;
        origin = glm::vec3{L.pos[0], L.pos[1], L.pos[2]};
        if (L.type == AnalyticLight::Sphere) {
            gizmo_size = std::max(0.5f, L.radius * 1.5f);
        } else if (L.type == AnalyticLight::Quad) {
            const float u_len = std::sqrt(L.u_vec[0]*L.u_vec[0] +
                                          L.u_vec[1]*L.u_vec[1] +
                                          L.u_vec[2]*L.u_vec[2]);
            gizmo_size = std::max(0.5f, std::max(u_len, L.v_half) * 1.5f);
        } else {
            gizmo_size = 1.5f;
        }
    }
    const float aspect = (accum_h_ > 0)
                           ? float(accum_w_) / float(accum_h_)
                           : 1.0f;

    // Cursor + LMB state.
    double mx = 0.0, my = 0.0;
    window_->GetCursorPos(mx, my);
    const bool lmb_down = window_->IsMouseButtonDown(0); // GLFW_MOUSE_BUTTON_LEFT
    // Gate clicks when the camera is being mouse-looked (RMB held). The
    // gizmo dragging an axis while the user is also yawing the camera
    // would be deeply confusing.
    const bool mouse_look = mouse_look_active_;

    pt::renderer::EditorOverlay::Axis hovered =
        pt::renderer::EditorOverlay::Axis::None;
    if (!mouse_look) {
        hovered = editor_gizmo_.HitTest(origin, gizmo_size,
                                        *camera_, aspect,
                                        accum_w_, accum_h_,
                                        mx, my, /*radius_px=*/10.0f);
    }

    // Esc-during-drag: abort and restore pre-drag state. The restored
    // command depends on which dispatch path the drag is on -- a
    // translate/scale drag rewinds the position; a rotate drag rewinds
    // the orientation quaternion. set_*_cmd resolves to the prim_ or
    // light_ prefix based on the active selection kind.
    const bool esc_pressed = window_->IsKeyDown(256); // GLFW_KEY_ESCAPE
    if (esc_pressed && editor_gizmo_.IsDragging()) {
        // Restore via the same console-routed mutation path so undo
        // / snapshot semantics line up with the normal drag.
        if (editor_gizmo_.DragMode() == pt::renderer::EditorOverlay::Mode::Rotate) {
            auto cmd = fmt::format("{} {} {} {} {} {}",
                                   set_rotation_cmd,
                                   selected_prim_id_,
                                   editor_drag_pre_orient_[0],
                                   editor_drag_pre_orient_[1],
                                   editor_drag_pre_orient_[2],
                                   editor_drag_pre_orient_[3]);
            pt::console::Console::Get().Execute(cmd);
        } else {
            const glm::vec3 pre = editor_drag_pre_pos_;
            auto cmd = fmt::format("{} {} {} {} {}",
                                   set_pos_cmd,
                                   selected_prim_id_, pre.x, pre.y, pre.z);
            pt::console::Console::Get().Execute(cmd);
        }
        editor_gizmo_.EndDrag();
        prev_lmb_down_ = false;
    }

    // LMB edge detect.
    if (!mouse_look) {
        if (lmb_down && !prev_lmb_down_) {
            // Press edge. Start drag iff hovering an axis.
            if (hovered != pt::renderer::EditorOverlay::Axis::None) {
                editor_drag_pre_pos_ = origin;
                // Snapshot the pre-drag orient so an Esc-during-drag
                // can rewind. Source depends on selection kind --
                // primitives_ for prims, light_prims_ for lights.
                const float* pre_orient =
                    sel_light ? light_it->second.orient
                              : prim_it->second.orient;
                editor_drag_pre_orient_[0] = pre_orient[0];
                editor_drag_pre_orient_[1] = pre_orient[1];
                editor_drag_pre_orient_[2] = pre_orient[2];
                editor_drag_pre_orient_[3] = pre_orient[3];
                editor_gizmo_.BeginDrag(hovered, origin,
                                        *camera_, aspect,
                                        accum_w_, accum_h_,
                                        mx, my);
            }
        } else if (lmb_down && prev_lmb_down_ && editor_gizmo_.IsDragging()) {
            // Continuing drag. Branch on the drag mode captured at
            // BeginDrag time so a mid-drag gizmo_mode toggle doesn't
            // change the dispatch path mid-motion.
            if (editor_gizmo_.DragMode() == pt::renderer::EditorOverlay::Mode::Rotate) {
                // Rotate drag: compute the signed angle around the
                // ring axis since BeginDrag, build the delta quat
                // (axis-angle), compose with the pre-drag orient,
                // dispatch prim_set_rotation. The composition is
                // new_orient = delta_q * pre_orient so the gizmo's
                // axis lives in world space (not in the prim's local
                // frame), matching user expectation.
                const float angle = editor_gizmo_.UpdateRotateDrag(
                    *camera_, aspect, accum_w_, accum_h_, mx, my);
                // Sub-half-degree threshold cuts Execute() churn when
                // the mouse is roughly still.
                constexpr float kAngleEps = 1.0e-3f;
                if (std::abs(angle) > kAngleEps) {
                    // Build delta quat from axis-angle.
                    const glm::vec3 ax = [&]() {
                        switch (editor_gizmo_.DragAxis()) {
                            case pt::renderer::EditorOverlay::Axis::X: return glm::vec3(1, 0, 0);
                            case pt::renderer::EditorOverlay::Axis::Y: return glm::vec3(0, 1, 0);
                            case pt::renderer::EditorOverlay::Axis::Z: return glm::vec3(0, 0, 1);
                            default: return glm::vec3(0, 0, 1);
                        }
                    }();
                    const float ha = angle * 0.5f;
                    const float s  = std::sin(ha);
                    const float c  = std::cos(ha);
                    // dq = (ax * s, c) in (xyz, w) order.
                    const float dqx = ax.x * s;
                    const float dqy = ax.y * s;
                    const float dqz = ax.z * s;
                    const float dqw = c;
                    // Compose: new = dq * pre. Hamilton product.
                    const float pre_x = editor_drag_pre_orient_[0];
                    const float pre_y = editor_drag_pre_orient_[1];
                    const float pre_z = editor_drag_pre_orient_[2];
                    const float pre_w = editor_drag_pre_orient_[3];
                    const float nx = dqw * pre_x + dqx * pre_w + dqy * pre_z - dqz * pre_y;
                    const float ny = dqw * pre_y - dqx * pre_z + dqy * pre_w + dqz * pre_x;
                    const float nz = dqw * pre_z + dqx * pre_y - dqy * pre_x + dqz * pre_w;
                    const float nw = dqw * pre_w - dqx * pre_x - dqy * pre_y - dqz * pre_z;
                    auto cmd = fmt::format("{} {} {} {} {} {}",
                                           set_rotation_cmd,
                                           selected_prim_id_, nx, ny, nz, nw);
                    pt::console::Console::Get().Execute(cmd);
                }
            } else {
                // Translate / scale: compute new world pos and dispatch
                // {prim,light}_set_pos via Console::Execute so the
                // engine's accum_dirty / dirty-flag handling + future
                // undo machinery route uniformly.
                const glm::vec3 new_pos =
                    editor_gizmo_.UpdateDrag(*camera_, aspect,
                                             accum_w_, accum_h_, mx, my);
                // Skip the dispatch if the delta is sub-millimetre; cuts
                // the ~60 Hz Execute() churn when the mouse is still.
                if (glm::distance(new_pos, origin) > 1e-4f) {
                    auto cmd = fmt::format("{} {} {} {} {}",
                                           set_pos_cmd,
                                           selected_prim_id_,
                                           new_pos.x, new_pos.y, new_pos.z);
                    pt::console::Console::Get().Execute(cmd);
                }
            }
        } else if (!lmb_down && prev_lmb_down_) {
            // Release edge.
            editor_gizmo_.EndDrag();
        }
    }
    prev_lmb_down_ = lmb_down;

    // Geometry build. Use the (possibly updated) origin from the live
    // entity -- it may have changed during this same frame if the drag
    // dispatched a set_pos / set_rotation above. For prim planes,
    // recompute the foot-of-perpendicular in case the normal was
    // rotated; for prim spheres and all lights, just re-read the
    // center / pos.
    glm::vec3 build_origin = origin;
    if (sel_prim) {
        auto fresh = primitives_.find(selected_prim_id_);
        if (fresh != primitives_.end()) {
            if (is_plane) {
                // Re-derive plane normal post-rotation so the rings move
                // with the visible plane. The shader applies orient at
                // intersect time, so for the gizmo we apply it here too.
                const float qx = fresh->second.orient[0];
                const float qy = fresh->second.orient[1];
                const float qz = fresh->second.orient[2];
                const float qw = fresh->second.orient[3];
                const glm::vec3 n_raw{fresh->second.pos_or_n[0],
                                      fresh->second.pos_or_n[1],
                                      fresh->second.pos_or_n[2]};
                // quatRotate(q, v) = v + 2 * cross(u, cross(u, v) + w * v)
                const glm::vec3 u{qx, qy, qz};
                const glm::vec3 n = n_raw + 2.0f *
                    glm::cross(u, glm::cross(u, n_raw) + qw * n_raw);
                build_origin = n * (-fresh->second.radius_or_d);
            } else {
                build_origin = glm::vec3{fresh->second.pos_or_n[0],
                                         fresh->second.pos_or_n[1],
                                         fresh->second.pos_or_n[2]};
            }
        }
    } else {
        auto fresh = light_prims_.find(selected_prim_id_);
        if (fresh != light_prims_.end()) {
            build_origin = glm::vec3{fresh->second.pos[0],
                                     fresh->second.pos[1],
                                     fresh->second.pos[2]};
        }
    }
    const pt::renderer::EditorOverlay::Axis hl =
        editor_gizmo_.IsDragging() ? editor_gizmo_.DragAxis() : hovered;
    editor_gizmo_.AppendGizmo(build_origin, gizmo_size, hl);

    // Upload to GPU.
    if (editor_overlay_segs_buf_id_ != 0 && editor_gizmo_.SegmentCount() > 0) {
        const std::uint32_t cap = editor_overlay_segs_buf_capacity_;
        const std::uint32_t n   = std::min(editor_gizmo_.SegmentCount(), cap);
        device_->WriteBuffer(
            pt::rhi::BufferHandle{editor_overlay_segs_buf_id_},
            editor_gizmo_.Segments().data(),
            std::size_t(n) * sizeof(pt::renderer::EditorOverlay::Segment),
            0);
    }
}

// --- SDF Phase 1 (#97) -----------------------------------------------------
//
// Console commands for the SDF primitive set. Mirrors the prim_* /
// csg_* conventions:
//   * `sdf_<shape> <id> [params...] <x y z> <material> <r g b>` adds a
//     LEAF cluster (a single-shape SDF blob). Material follows the same
//     lambert | metal | dielectric taxonomy as analytic prims.
//   * `sdf_smin / sdf_smax / sdf_sdiff <id> <child_id_a> <child_id_b>
//     <k> [material] [r g b]` combines two existing SDF leaves into a
//     smooth-CSG cluster. The two child clusters are CONSUMED -- they
//     get folded into the new cluster's flat node array and removed
//     from the active set.
//   * `sdf_list / sdf_clear / sdf_remove` round out the management API.
//
// On any successful mutation we set sdf_prims_dirty_ so the next render
// frame uploads the cluster buffer to the GPU. The shader gates the
// whole SDF path on sdf_cluster_count_ > 0 so an empty set is free.

namespace {

bool ParseSdfMaterial(std::string_view s, std::uint32_t& mat) {
    if (s == "lambert")    { mat = 0u; return true; }
    if (s == "metal")      { mat = 1u; return true; }
    if (s == "dielectric") { mat = 2u; return true; }
    return false;
}

}  // namespace

void Engine::RegisterSdfCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("sdf_list",
        "List signed-distance-field clusters.",
        [this](auto, pt::console::Output& out) {
            if (sdf_prims_.empty()) { out.PrintLine("(no SDF clusters)"); return; }
            out.FormatLine("{} SDF cluster(s):", sdf_prims_.size());
            for (const auto& [id, p] : sdf_prims_) {
                out.FormatLine("  {:>3}  nodes={}  aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}] mat={} rgb=({:.2f},{:.2f},{:.2f})",
                               id, p.node_count,
                               p.aabb_min[0], p.aabb_min[1], p.aabb_min[2],
                               p.aabb_max[0], p.aabb_max[1], p.aabb_max[2],
                               p.material,
                               p.albedo[0], p.albedo[1], p.albedo[2]);
            }
        });

    C.RegisterCommand("sdf_clear",
        "Remove all SDF clusters.",
        [this](auto, pt::console::Output& out) {
            sdf_prims_.clear();
            sdf_prims_dirty_ = true;
            accum_dirty_     = true;
            out.PrintLine("sdf: cleared");
        });

    C.RegisterCommand("sdf_remove",
        "sdf_remove <id>: drop an SDF cluster.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: sdf_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("sdf_remove: id parse failed"); return; }
            if (sdf_prims_.erase(id) == 0) {
                out.FormatLine("sdf_remove: id {} not found", id);
                return;
            }
            sdf_prims_dirty_ = true;
            accum_dirty_     = true;
            out.FormatLine("sdf: removed id {}", id);
        });

    // Helper: add a single-leaf cluster with the given shape + params.
    // Logs on success per the engine's "logging is a mantra" pattern.
    auto add_leaf = [this](std::uint32_t id,
                           pt::renderer::SdfShape shape,
                           std::array<float, 4> params,
                           float x, float y, float z,
                           std::uint32_t material,
                           float r, float g, float b,
                           pt::console::Output& out,
                           const char* tag) {
        // Reject ids in the reserved-cluster range owned by the voxel
        // demo (SyncVoxelDemoState assumes nothing else stores
        // sdf_prims_[id] for id >= kVoxelClusterIdBase and would
        // silently delete any user-issued cluster in that range on the
        // next frame). Surface the conflict to the user instead.
        if (id >= kVoxelClusterIdBase) {
            out.FormatLine("sdf_{}: id={} is in the reserved voxel-cluster range (>= 0x{:08X}); pick a smaller id",
                           tag, id, kVoxelClusterIdBase);
            return;
        }
        pt::renderer::SdfPrim prim{};
        prim.node_count   = 1;
        prim.material     = material;
        prim.albedo[0] = r; prim.albedo[1] = g; prim.albedo[2] = b;
        prim.roughness    = 0.0f;
        prim.ior          = 1.5f;
        pt::renderer::SdfNode& n = prim.nodes[0];
        n.op       = pt::renderer::SDF_OP_LEAF;
        n.shape    = shape;
        n.child_a  = 0;
        n.child_b  = 0;
        n.params[0] = params[0]; n.params[1] = params[1];
        n.params[2] = params[2]; n.params[3] = params[3];
        n.center[0] = x; n.center[1] = y; n.center[2] = z;
        if (!pt::renderer::ComputeSdfAabb(prim)) {
            out.FormatLine("sdf_{}: degenerate params", tag);
            return;
        }
        sdf_prims_[id] = prim;
        sdf_prims_dirty_ = true;
        accum_dirty_     = true;
        LOG_INFO("[sdf] add {} id={} aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}]",
                 tag, id,
                 prim.aabb_min[0], prim.aabb_min[1], prim.aabb_min[2],
                 prim.aabb_max[0], prim.aabb_max[1], prim.aabb_max[2]);
        out.FormatLine("sdf: added {} id={} @ ({:.2f} {:.2f} {:.2f})", tag, id, x, y, z);
    };

    C.RegisterCommand("sdf_sphere",
        "sdf_sphere <id> <radius> <x> <y> <z> <lambert|metal|dielectric> <r> <g> <b>: add a sphere SDF.",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 9) { out.PrintLine("usage: sdf_sphere <id> <radius> <x> <y> <z> <material> <r> <g> <b>"); return; }
            std::uint32_t id, mat;
            float radius, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], radius) ||
                !ParseFloat(args[2], x) || !ParseFloat(args[3], y) || !ParseFloat(args[4], z) ||
                !ParseSdfMaterial(args[5], mat) ||
                !ParseFloat(args[6], r) || !ParseFloat(args[7], g) || !ParseFloat(args[8], b)) {
                out.PrintLine("sdf_sphere: arg parse failed");
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_SPHERE,
                     {radius, 0.0f, 0.0f, 0.0f}, x, y, z, mat, r, g, b, out, "sphere");
        });

    C.RegisterCommand("sdf_box",
        "sdf_box <id> <hx> <hy> <hz> <x> <y> <z> <material> <r> <g> <b>: add a box SDF (half-extents).",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 11) { out.PrintLine("usage: sdf_box <id> <hx> <hy> <hz> <x> <y> <z> <material> <r> <g> <b>"); return; }
            std::uint32_t id, mat;
            float hx, hy, hz, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], hx) || !ParseFloat(args[2], hy) || !ParseFloat(args[3], hz) ||
                !ParseFloat(args[4], x)  || !ParseFloat(args[5], y)  || !ParseFloat(args[6], z) ||
                !ParseSdfMaterial(args[7], mat) ||
                !ParseFloat(args[8], r) || !ParseFloat(args[9], g) || !ParseFloat(args[10], b)) {
                out.PrintLine("sdf_box: arg parse failed");
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_BOX,
                     {hx, hy, hz, 0.0f}, x, y, z, mat, r, g, b, out, "box");
        });

    C.RegisterCommand("sdf_round_box",
        "sdf_round_box <id> <hx> <hy> <hz> <corner_r> <x> <y> <z> <material> <r> <g> <b>: add a rounded box SDF.",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 12) { out.PrintLine("usage: sdf_round_box <id> <hx> <hy> <hz> <corner_r> <x> <y> <z> <material> <r> <g> <b>"); return; }
            std::uint32_t id, mat;
            float hx, hy, hz, cr, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], hx) || !ParseFloat(args[2], hy) || !ParseFloat(args[3], hz) ||
                !ParseFloat(args[4], cr) ||
                !ParseFloat(args[5], x)  || !ParseFloat(args[6], y)  || !ParseFloat(args[7], z) ||
                !ParseSdfMaterial(args[8], mat) ||
                !ParseFloat(args[9], r) || !ParseFloat(args[10], g) || !ParseFloat(args[11], b)) {
                out.PrintLine("sdf_round_box: arg parse failed");
                return;
            }
            // Reject malformed shape parameters at parse time so a bad
            // command surfaces as a clear console error rather than a
            // silently-mis-bounded SDF (rounded box with a negative
            // corner radius would shrink the AABB inward of the actual
            // sdRoundBox surface and miss hits during the trace).
            if (hx <= 0.0f || hy <= 0.0f || hz <= 0.0f) {
                out.FormatLine("sdf_round_box: half-extents must be > 0 (got hx={} hy={} hz={})",
                               hx, hy, hz);
                return;
            }
            if (cr < 0.0f) {
                out.FormatLine("sdf_round_box: corner_r must be >= 0 (got {})", cr);
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_ROUNDED_BOX,
                     {hx, hy, hz, cr}, x, y, z, mat, r, g, b, out, "round_box");
        });

    C.RegisterCommand("sdf_torus",
        "sdf_torus <id> <R> <r> <x> <y> <z> <material> <r_col> <g_col> <b_col>: add a torus SDF (XZ plane).",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 10) { out.PrintLine("usage: sdf_torus <id> <R> <r> <x> <y> <z> <material> <r> <g> <b>"); return; }
            std::uint32_t id, mat;
            float R, r, x, y, z, cr, cg, cb;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], R) || !ParseFloat(args[2], r) ||
                !ParseFloat(args[3], x) || !ParseFloat(args[4], y) || !ParseFloat(args[5], z) ||
                !ParseSdfMaterial(args[6], mat) ||
                !ParseFloat(args[7], cr) || !ParseFloat(args[8], cg) || !ParseFloat(args[9], cb)) {
                out.PrintLine("sdf_torus: arg parse failed");
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_TORUS,
                     {R, r, 0.0f, 0.0f}, x, y, z, mat, cr, cg, cb, out, "torus");
        });

    C.RegisterCommand("sdf_capsule",
        "sdf_capsule <id> <half_height> <radius> <x> <y> <z> <material> <r> <g> <b>: add a Y-axis capsule SDF.",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 10) { out.PrintLine("usage: sdf_capsule <id> <half_h> <radius> <x> <y> <z> <material> <r> <g> <b>"); return; }
            std::uint32_t id, mat;
            float h, r, x, y, z, cr, cg, cb;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], h) || !ParseFloat(args[2], r) ||
                !ParseFloat(args[3], x) || !ParseFloat(args[4], y) || !ParseFloat(args[5], z) ||
                !ParseSdfMaterial(args[6], mat) ||
                !ParseFloat(args[7], cr) || !ParseFloat(args[8], cg) || !ParseFloat(args[9], cb)) {
                out.PrintLine("sdf_capsule: arg parse failed");
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_CAPSULE,
                     {h, r, 0.0f, 0.0f}, x, y, z, mat, cr, cg, cb, out, "capsule");
        });

    // ----- smooth-CSG combinators ------------------------------------------
    // Take two existing SDF cluster ids and fold them into a new
    // combined cluster. The children are removed from the active set
    // since their data is now inlined into the parent's node array.
    // Combined node count must not exceed SdfPrim::kMaxNodes; if it
    // would, the command fails and the children are left in place.
    auto combine = [this](std::uint32_t id,
                          std::uint32_t a_id,
                          std::uint32_t b_id,
                          float k,
                          pt::renderer::SdfOp op,
                          std::uint32_t material,
                          float r, float g, float b,
                          pt::console::Output& out,
                          const char* tag) {
        // Reject self-combine: a_id == b_id would duplicate the same
        // node stream into the output and then attempt to erase the
        // same map entry twice (the second erase is a no-op but the
        // duplicated nodes already inflated the count, possibly past
        // kMaxNodes). It's a user error -- surface it clearly.
        if (a_id == b_id) {
            out.FormatLine("sdf_{}: child_a and child_b must differ (got {} twice)",
                           tag, a_id);
            return;
        }
        // Reserved-range guard. Same as in add_leaf -- a combined
        // cluster landing at id >= kVoxelClusterIdBase would be deleted
        // by the next SyncVoxelDemoState pass.
        if (id >= kVoxelClusterIdBase) {
            out.FormatLine("sdf_{}: id={} is in the reserved voxel-cluster range (>= 0x{:08X}); pick a smaller id",
                           tag, id, kVoxelClusterIdBase);
            return;
        }
        auto it_a = sdf_prims_.find(a_id);
        auto it_b = sdf_prims_.find(b_id);
        if (it_a == sdf_prims_.end() || it_b == sdf_prims_.end()) {
            out.FormatLine("sdf_{}: child id {} or {} not found", tag, a_id, b_id);
            return;
        }
        const pt::renderer::SdfPrim& A = it_a->second;
        const pt::renderer::SdfPrim& B = it_b->second;
        // --- SDF Phase 3 (#99) guard --------------------------------------
        // Reject smooth-CSG combinations involving fractal leaves. The
        // GPU dispatcher in PathTrace.slang only handles SINGLE-LEAF
        // fractal clusters in Phase 3 (the smooth-CSG ops in
        // SdfPrimitives.slang assume exact-distance leaves and would
        // see the fractal DE's approximate distance as a wildly-
        // overshooting child, producing a corrupted blend). Lifting
        // the restriction is a future-future thing.
        auto cluster_has_fractal = [](const pt::renderer::SdfPrim& p) {
            for (std::uint32_t i = 0; i < p.node_count; ++i) {
                if (p.nodes[i].op == pt::renderer::SDF_OP_LEAF &&
                    pt::renderer::IsSdfFractalShape(p.nodes[i].shape)) {
                    return true;
                }
            }
            return false;
        };
        if (cluster_has_fractal(A) || cluster_has_fractal(B)) {
            out.FormatLine("sdf_{}: fractal SDF leaves cannot participate in "
                           "smooth-CSG ops in Phase 3 (issue #99 scope-out)",
                           tag);
            return;
        }
        // --- end SDF Phase 3 ---
        if (A.node_count + B.node_count + 1 > pt::renderer::SdfPrim::kMaxNodes) {
            out.FormatLine("sdf_{}: combined node count {} exceeds kMaxNodes={}",
                           tag, A.node_count + B.node_count + 1,
                           pt::renderer::SdfPrim::kMaxNodes);
            return;
        }
        pt::renderer::SdfPrim out_p{};
        // Copy A's nodes first (root @ 0), then B's nodes shifted by
        // A.node_count, then the op node referencing the two child
        // root indices.
        for (std::uint32_t i = 0; i < A.node_count; ++i) out_p.nodes[i] = A.nodes[i];
        for (std::uint32_t i = 0; i < B.node_count; ++i) {
            pt::renderer::SdfNode nb = B.nodes[i];
            // Rewrite child indices in B by the shift amount so they
            // still reference the right nodes inside the merged array.
            if (nb.op != pt::renderer::SDF_OP_LEAF) {
                nb.child_a += A.node_count;
                nb.child_b += A.node_count;
            }
            out_p.nodes[A.node_count + i] = nb;
        }
        std::uint32_t op_idx = A.node_count + B.node_count;
        pt::renderer::SdfNode& opn = out_p.nodes[op_idx];
        opn.op       = op;
        opn.shape    = 0;
        // child_a = root of A (always index 0 by host convention).
        // child_b = root of B (now at index A.node_count after shift).
        // BUT the host emits nodes in ascending order with children
        // earlier than parents, and the root of any sub-tree is the
        // LAST node in its block. So the root of A is at index
        // A.node_count - 1, and the root of B is at index
        // A.node_count + B.node_count - 1.
        opn.child_a  = A.node_count - 1u;
        opn.child_b  = A.node_count + B.node_count - 1u;
        opn.params[0] = k;
        // op node's center is unused -- zero by default-init.
        // The op node MUST be the LAST emitted node (the merged tree's
        // new root) so downstream walkers find the merged result.
        out_p.node_count = op_idx + 1;
        out_p.material   = material;
        out_p.albedo[0]  = r; out_p.albedo[1] = g; out_p.albedo[2] = b;
        out_p.roughness  = 0.0f;
        out_p.ior        = 1.5f;
        if (!pt::renderer::ComputeSdfAabb(out_p)) {
            out.FormatLine("sdf_{}: AABB computation failed", tag);
            return;
        }
        // Drop the consumed children. Insert the new cluster under the
        // requested id (overwrite if it collides).
        sdf_prims_.erase(it_a);
        sdf_prims_.erase(b_id);
        sdf_prims_[id]   = out_p;
        sdf_prims_dirty_ = true;
        accum_dirty_     = true;
        LOG_INFO("[sdf] {} id={} (children {}+{}, k={:.3f}) aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}]",
                 tag, id, a_id, b_id, k,
                 out_p.aabb_min[0], out_p.aabb_min[1], out_p.aabb_min[2],
                 out_p.aabb_max[0], out_p.aabb_max[1], out_p.aabb_max[2]);
        out.FormatLine("sdf: {} id={} ({} + {}, k={})", tag, id, a_id, b_id, k);
    };

    C.RegisterCommand("sdf_smin",
        "sdf_smin <id> <child_a> <child_b> <k> <material> <r> <g> <b>: smooth-union two SDF clusters (k = smoothing radius, metres).",
        [combine](auto args, pt::console::Output& out) {
            if (args.size() != 8) { out.PrintLine("usage: sdf_smin <id> <a> <b> <k> <material> <r> <g> <b>"); return; }
            std::uint32_t id, a, b, mat;
            float k, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) || !ParseUint(args[2], b) ||
                !ParseFloat(args[3], k) ||
                !ParseSdfMaterial(args[4], mat) ||
                !ParseFloat(args[5], cr) || !ParseFloat(args[6], cg) || !ParseFloat(args[7], cb)) {
                out.PrintLine("sdf_smin: arg parse failed");
                return;
            }
            combine(id, a, b, k, pt::renderer::SDF_OP_SMOOTH_UNION, mat, cr, cg, cb, out, "smin");
        });

    C.RegisterCommand("sdf_smax",
        "sdf_smax <id> <child_a> <child_b> <k> <material> <r> <g> <b>: smooth-intersect two SDF clusters.",
        [combine](auto args, pt::console::Output& out) {
            if (args.size() != 8) { out.PrintLine("usage: sdf_smax <id> <a> <b> <k> <material> <r> <g> <b>"); return; }
            std::uint32_t id, a, b, mat;
            float k, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) || !ParseUint(args[2], b) ||
                !ParseFloat(args[3], k) ||
                !ParseSdfMaterial(args[4], mat) ||
                !ParseFloat(args[5], cr) || !ParseFloat(args[6], cg) || !ParseFloat(args[7], cb)) {
                out.PrintLine("sdf_smax: arg parse failed");
                return;
            }
            combine(id, a, b, k, pt::renderer::SDF_OP_SMOOTH_INTERSECT, mat, cr, cg, cb, out, "smax");
        });

    C.RegisterCommand("sdf_sdiff",
        "sdf_sdiff <id> <child_a> <child_b> <k> <material> <r> <g> <b>: smooth-subtract child_b from child_a.",
        [combine](auto args, pt::console::Output& out) {
            if (args.size() != 8) { out.PrintLine("usage: sdf_sdiff <id> <a> <b> <k> <material> <r> <g> <b>"); return; }
            std::uint32_t id, a, b, mat;
            float k, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) || !ParseUint(args[2], b) ||
                !ParseFloat(args[3], k) ||
                !ParseSdfMaterial(args[4], mat) ||
                !ParseFloat(args[5], cr) || !ParseFloat(args[6], cg) || !ParseFloat(args[7], cb)) {
                out.PrintLine("sdf_sdiff: arg parse failed");
                return;
            }
            combine(id, a, b, k, pt::renderer::SDF_OP_SMOOTH_SUBTRACT, mat, cr, cg, cb, out, "sdiff");
        });

    // --- SDF Phase 3 (#99) fractal leaves ----------------------------------
    //
    // Three iconic-fractal commands. Each adds a SINGLE-LEAF cluster
    // whose `params[0]` is the per-shape scale knob (Mandelbulb's
    // polar exponent, Mandelbox's linear-step scale, Apollonian's
    // radial-shrink scale) and `params[1]` is the world-space effective
    // bound radius (the host AABB is a cube of side 2*params[1]).
    // Cluster shading goes through the same Lambert / Metal /
    // Dielectric material pipeline as the analytic SDF leaves above
    // -- the BSDF / NEE / denoiser don't care which side of the
    // shader the hit came from.
    //
    // Per the issue spec, fractals can't be combined into smooth-CSG
    // ops in Phase 3 (they're whole-cluster shapes; the `combine`
    // helper above rejects them at the host level because the GPU
    // dispatcher in PathTrace.slang only handles SINGLE-LEAF fractal
    // clusters). The path-of-least-surprise here is also to make the
    // host refuse: a future phase can lift the restriction once
    // mixed-CSG cluster traversal is in.

    C.RegisterCommand("sdf_mandelbulb",
        "sdf_mandelbulb <id> <power> <bound> <x> <y> <z> <material> <r> <g> <b>: "
        "add a Mandelbulb fractal SDF leaf. power=8 is canonical; bound is "
        "the cluster AABB half-extent in metres (1.2 fits the textbook bulb).",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 10) {
                out.PrintLine("usage: sdf_mandelbulb <id> <power> <bound> <x> <y> <z> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, mat;
            float power, bound, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], power) || !ParseFloat(args[2], bound) ||
                !ParseFloat(args[3], x) || !ParseFloat(args[4], y) || !ParseFloat(args[5], z) ||
                !ParseSdfMaterial(args[6], mat) ||
                !ParseFloat(args[7], r) || !ParseFloat(args[8], g) || !ParseFloat(args[9], b)) {
                out.PrintLine("sdf_mandelbulb: arg parse failed");
                return;
            }
            // power=0 is the "use r_sdf_fractal_power cvar default"
            // sentinel; below 0 is meaningless. Bound must be > 0
            // because LeafLocalAabb refuses a non-positive bound (else
            // the cube degenerates and the AABB slab test rejects
            // every ray).
            if (power < 0.0f) {
                out.FormatLine("sdf_mandelbulb: power must be >= 0 (got {})", power);
                return;
            }
            if (bound <= 0.0f) {
                out.FormatLine("sdf_mandelbulb: bound must be > 0 (got {})", bound);
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_MANDELBULB,
                     {power, bound, 0.0f, 0.0f}, x, y, z, mat, r, g, b, out, "mandelbulb");
        });

    C.RegisterCommand("sdf_mandelbox",
        "sdf_mandelbox <id> <scale> <bound> <x> <y> <z> <material> <r> <g> <b>: "
        "add a Mandelbox fractal SDF leaf. scale=2.5 is canonical; bound is "
        "the cluster AABB half-extent (4.0 typically fits the limit set).",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 10) {
                out.PrintLine("usage: sdf_mandelbox <id> <scale> <bound> <x> <y> <z> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, mat;
            float scale, bound, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], scale) || !ParseFloat(args[2], bound) ||
                !ParseFloat(args[3], x) || !ParseFloat(args[4], y) || !ParseFloat(args[5], z) ||
                !ParseSdfMaterial(args[6], mat) ||
                !ParseFloat(args[7], r) || !ParseFloat(args[8], g) || !ParseFloat(args[9], b)) {
                out.PrintLine("sdf_mandelbox: arg parse failed");
                return;
            }
            // scale=0 maps to the canonical Mandelbox scale (2.5)
            // hardcoded in the shader's evalFractalDist. The
            // r_sdf_fractal_power cvar only applies to Mandelbulb;
            // Mandelbox's "scale" is a different DE parameter (linear-
            // step magnitude, not polar power). Negative is meaningless
            // -- the linear-step magnitude has to be a positive
            // contraction for the DE to terminate.
            if (scale < 0.0f) {
                out.FormatLine("sdf_mandelbox: scale must be >= 0 (got {})", scale);
                return;
            }
            if (bound <= 0.0f) {
                out.FormatLine("sdf_mandelbox: bound must be > 0 (got {})", bound);
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_MANDELBOX,
                     {scale, bound, 0.0f, 0.0f}, x, y, z, mat, r, g, b, out, "mandelbox");
        });

    C.RegisterCommand("sdf_apollonian",
        "sdf_apollonian <id> <scale> <bound> <x> <y> <z> <material> <r> <g> <b>: "
        "add an Apollonian-gasket fractal SDF leaf. scale=1.3 is canonical; "
        "bound is the cluster AABB half-extent (2.0 typically fits the limit set).",
        [add_leaf](auto args, pt::console::Output& out) {
            if (args.size() != 10) {
                out.PrintLine("usage: sdf_apollonian <id> <scale> <bound> <x> <y> <z> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, mat;
            float scale, bound, x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], scale) || !ParseFloat(args[2], bound) ||
                !ParseFloat(args[3], x) || !ParseFloat(args[4], y) || !ParseFloat(args[5], z) ||
                !ParseSdfMaterial(args[6], mat) ||
                !ParseFloat(args[7], r) || !ParseFloat(args[8], g) || !ParseFloat(args[9], b)) {
                out.PrintLine("sdf_apollonian: arg parse failed");
                return;
            }
            if (scale < 0.0f) {
                out.FormatLine("sdf_apollonian: scale must be >= 0 (got {})", scale);
                return;
            }
            if (bound <= 0.0f) {
                out.FormatLine("sdf_apollonian: bound must be > 0 (got {})", bound);
                return;
            }
            add_leaf(id, pt::renderer::SDF_SHAPE_APOLLONIAN,
                     {scale, bound, 0.0f, 0.0f}, x, y, z, mat, r, g, b, out, "apollonian");
        });
    // --- end SDF Phase 3 ---------------------------------------------------
    // ----- SDF Phase 2 (#98) procedural / warp / domain ops ----------------
    //
    // Single-child wrap commands: take ONE existing SDF cluster id and
    // wrap it with a Phase 2 procedural op. The child is consumed (its
    // nodes are inlined into the new cluster and its id is removed from
    // the active set). Combined node count must not exceed
    // SdfPrim::kMaxNodes; if it would, the command fails and the child
    // is left in place.
    //
    // The host enforces a key invariant: warp ops (TWIST / BEND / REPEAT
    // / REPEAT_LIMITED) must be the cluster's ROOT op -- the shader's
    // sdfClusterDistWarped / sdfClusterDistDWarped look only at the
    // root's op type to pick a warp branch, and the inner walkers
    // (sdfClusterDistProcN / sdfClusterDistDN) treat non-root warps as
    // identity sentinels. So wrapping a cluster that already contains a
    // warp (chaining e.g. sdf_twist around an sdf_bend, or sdf_displace_
    // noise around an sdf_repeat) would silently drop the inner warp.
    // Reject such constructions up-front. DISPLACE / DISPLACE_NOISE are
    // pure point-displacement ops handled inline by the inner walkers,
    // so they nest cleanly and are allowed.
    auto child_has_warp = [](const pt::renderer::SdfPrim& P) {
        for (std::uint32_t i = 0; i < P.node_count; ++i) {
            auto op = P.nodes[i].op;
            if (op == pt::renderer::SDF_OP_TWIST ||
                op == pt::renderer::SDF_OP_BEND  ||
                op == pt::renderer::SDF_OP_REPEAT ||
                op == pt::renderer::SDF_OP_REPEAT_LIMITED) {
                return true;
            }
        }
        return false;
    };
    auto wrap_proc = [this, child_has_warp](std::uint32_t id,
                            std::uint32_t a_id,
                            pt::renderer::SdfOp op,
                            std::array<float, 4> op_params,
                            std::uint32_t material,
                            float r, float g, float b,
                            pt::console::Output& out,
                            const char* tag) {
        auto it_a = sdf_prims_.find(a_id);
        if (it_a == sdf_prims_.end()) {
            out.FormatLine("sdf_{}: child id {} not found", tag, a_id);
            return;
        }
        const pt::renderer::SdfPrim& A = it_a->second;
        // Enforce the "warps must be root" invariant -- see comment
        // above wrap_proc. DISPLACE_NOISE / DISPLACE-only children are
        // fine because they don't warp the point.
        bool wrap_is_warp = (op == pt::renderer::SDF_OP_TWIST ||
                              op == pt::renderer::SDF_OP_BEND  ||
                              op == pt::renderer::SDF_OP_REPEAT ||
                              op == pt::renderer::SDF_OP_REPEAT_LIMITED);
        if (wrap_is_warp && child_has_warp(A)) {
            out.FormatLine("sdf_{}: child id {} already contains a warp op "
                           "(twist / bend / repeat / repeat_limited). The "
                           "shader only applies a warp at the cluster root, "
                           "so chaining would silently drop the inner warp. "
                           "Combine the warps into a single op or apply "
                           "them to separate clusters and union via "
                           "sdf_smin.",
                           tag, a_id);
            return;
        }
        if (A.node_count + 1 > pt::renderer::SdfPrim::kMaxNodes) {
            out.FormatLine("sdf_{}: combined node count {} exceeds kMaxNodes={}",
                           tag, A.node_count + 1,
                           pt::renderer::SdfPrim::kMaxNodes);
            return;
        }
        pt::renderer::SdfPrim out_p{};
        for (std::uint32_t i = 0; i < A.node_count; ++i) out_p.nodes[i] = A.nodes[i];
        std::uint32_t op_idx = A.node_count;
        pt::renderer::SdfNode& opn = out_p.nodes[op_idx];
        opn.op       = op;
        opn.shape    = 0;
        // child_a = root of A (last node of A's sub-tree).
        opn.child_a  = A.node_count - 1u;
        opn.child_b  = 0;
        opn.params[0] = op_params[0];
        opn.params[1] = op_params[1];
        opn.params[2] = op_params[2];
        opn.params[3] = op_params[3];
        out_p.node_count = op_idx + 1;
        out_p.material   = material;
        out_p.albedo[0]  = r; out_p.albedo[1] = g; out_p.albedo[2] = b;
        out_p.roughness  = 0.0f;
        out_p.ior        = 1.5f;
        if (!pt::renderer::ComputeSdfAabb(out_p)) {
            out.FormatLine("sdf_{}: AABB computation failed", tag);
            return;
        }
        sdf_prims_.erase(it_a);
        sdf_prims_[id]   = out_p;
        sdf_prims_dirty_ = true;
        accum_dirty_     = true;
        LOG_INFO("[sdf] {} id={} (child {}) aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}]",
                 tag, id, a_id,
                 out_p.aabb_min[0], out_p.aabb_min[1], out_p.aabb_min[2],
                 out_p.aabb_max[0], out_p.aabb_max[1], out_p.aabb_max[2]);
        out.FormatLine("sdf: {} id={} (child {})", tag, id, a_id);
    };

    C.RegisterCommand("sdf_displace_noise",
        "sdf_displace_noise <id> <child_id> <amp_m> <freq_per_m> [octaves] <material> <r> <g> <b>: "
        "wrap a cluster with an FBM-noise displacement band (amp metres, "
        "freq 1/metres, octaves 1..6).",
        [wrap_proc](auto args, pt::console::Output& out) {
            // Allow octaves arg optional -- 8 args without, 9 with.
            if (args.size() != 8 && args.size() != 9) {
                out.PrintLine("usage: sdf_displace_noise <id> <child> <amp> <freq> [octaves] <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, a, mat;
            float amp, freq, cr, cg, cb;
            int   octaves = 4;
            if (auto* v = pt::console::Console::Get().FindCVar("r_sdf_displace_octaves"))
                octaves = v->GetInt();
            std::size_t base = 4;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) ||
                !ParseFloat(args[2], amp) || !ParseFloat(args[3], freq)) {
                out.PrintLine("sdf_displace_noise: arg parse failed (id/child/amp/freq)");
                return;
            }
            if (args.size() == 9) {
                if (!ParseInt(args[4], octaves)) {
                    out.PrintLine("sdf_displace_noise: octaves parse failed");
                    return;
                }
                base = 5;
            }
            if (!ParseSdfMaterial(args[base + 0], mat) ||
                !ParseFloat(args[base + 1], cr) ||
                !ParseFloat(args[base + 2], cg) ||
                !ParseFloat(args[base + 3], cb)) {
                out.PrintLine("sdf_displace_noise: arg parse failed (material / rgb)");
                return;
            }
            if (amp <= 0.0f || freq <= 0.0f) {
                out.FormatLine("sdf_displace_noise: amp and freq must be > 0 (got amp={} freq={})", amp, freq);
                return;
            }
            if (octaves < 1)  octaves = 1;
            if (octaves > 6)  octaves = 6;
            // Pack the octaves int into the params[2] float lane via
            // bit-cast (asuint on the shader side recovers it).
            std::uint32_t oct_u = static_cast<std::uint32_t>(octaves);
            float oct_f;
            std::memcpy(&oct_f, &oct_u, sizeof(float));
            wrap_proc(id, a, pt::renderer::SDF_OP_DISPLACE_NOISE,
                      {amp, freq, oct_f, 0.0f}, mat, cr, cg, cb, out, "displace_noise");
        });

    C.RegisterCommand("sdf_twist",
        "sdf_twist <id> <child_id> <rate_rad_per_m> <material> <r> <g> <b>: "
        "twist a cluster around the Y axis at `rate` radians per metre of height.",
        [wrap_proc](auto args, pt::console::Output& out) {
            if (args.size() != 7) {
                out.PrintLine("usage: sdf_twist <id> <child> <rate> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, a, mat;
            float rate, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) ||
                !ParseFloat(args[2], rate) ||
                !ParseSdfMaterial(args[3], mat) ||
                !ParseFloat(args[4], cr) || !ParseFloat(args[5], cg) || !ParseFloat(args[6], cb)) {
                out.PrintLine("sdf_twist: arg parse failed");
                return;
            }
            wrap_proc(id, a, pt::renderer::SDF_OP_TWIST,
                      {rate, 0.0f, 0.0f, 0.0f}, mat, cr, cg, cb, out, "twist");
        });

    C.RegisterCommand("sdf_bend",
        "sdf_bend <id> <child_id> <rate_rad_per_m> <material> <r> <g> <b>: "
        "bend a cluster along the X axis at `rate` radians per metre.",
        [wrap_proc](auto args, pt::console::Output& out) {
            if (args.size() != 7) {
                out.PrintLine("usage: sdf_bend <id> <child> <rate> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, a, mat;
            float rate, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) ||
                !ParseFloat(args[2], rate) ||
                !ParseSdfMaterial(args[3], mat) ||
                !ParseFloat(args[4], cr) || !ParseFloat(args[5], cg) || !ParseFloat(args[6], cb)) {
                out.PrintLine("sdf_bend: arg parse failed");
                return;
            }
            wrap_proc(id, a, pt::renderer::SDF_OP_BEND,
                      {rate, 0.0f, 0.0f, 0.0f}, mat, cr, cg, cb, out, "bend");
        });

    C.RegisterCommand("sdf_repeat",
        "sdf_repeat <id> <child_id> <px> <py> <pz> <material> <r> <g> <b>: "
        "infinite domain repetition with per-axis period (metres; 0 = no repeat on that axis). "
        "WARNS: each axis with non-zero period widens the cluster AABB to "
        "+/-1000 m. Use sdf_repeat_limited if you want a finite, BVH-"
        "friendly bound.",
        [wrap_proc](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: sdf_repeat <id> <child> <px> <py> <pz> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id, a, mat;
            float px, py, pz, cr, cg, cb;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) ||
                !ParseFloat(args[2], px) || !ParseFloat(args[3], py) || !ParseFloat(args[4], pz) ||
                !ParseSdfMaterial(args[5], mat) ||
                !ParseFloat(args[6], cr) || !ParseFloat(args[7], cg) || !ParseFloat(args[8], cb)) {
                out.PrintLine("sdf_repeat: arg parse failed");
                return;
            }
            // Infinite repeat has no finite conservative AABB; the
            // host falls back to +/-1000 m on each repeat-active axis
            // (see AnalyticBvh.cpp SDF_OP_REPEAT). Surface that so
            // the user isn't surprised by the huge cluster bound (the
            // BVH slab test still serves it, but every other primitive
            // becomes a near-miss). For a tight bound use the bounded
            // variant.
            bool repeats_any = (std::abs(px) > 1e-6f) ||
                               (std::abs(py) > 1e-6f) ||
                               (std::abs(pz) > 1e-6f);
            if (repeats_any) {
                LOG_WARN("[sdf] sdf_repeat id={} child={} : infinite repeat "
                         "has no finite AABB; cluster bound will widen to "
                         "+/-1000 m on each repeat-active axis "
                         "(px={:.3f} py={:.3f} pz={:.3f}). Use "
                         "sdf_repeat_limited for a finite bound.",
                         id, a, px, py, pz);
                out.FormatLine("sdf_repeat: warning -- infinite repeat widens "
                               "AABB to +/-1000 m on active axes "
                               "(px={:.3f} py={:.3f} pz={:.3f}). Use "
                               "sdf_repeat_limited for a finite bound.",
                               px, py, pz);
            }
            wrap_proc(id, a, pt::renderer::SDF_OP_REPEAT,
                      {px, py, pz, 0.0f}, mat, cr, cg, cb, out, "repeat");
        });

    C.RegisterCommand("sdf_repeat_limited",
        "sdf_repeat_limited <id> <child_id> <px> <py> <pz> <lx> <ly> <lz> <material> <r> <g> <b>: "
        "bounded domain repetition. <lx ly lz> are integer half-extents (cell-count either side of origin, max 1023).",
        [wrap_proc](auto args, pt::console::Output& out) {
            if (args.size() != 12) {
                out.PrintLine("usage: sdf_repeat_limited <id> <child> <px py pz> <lx ly lz> <material> <r g b>");
                return;
            }
            std::uint32_t id, a, mat;
            float px, py, pz, cr, cg, cb;
            int   lx, ly, lz;
            if (!ParseUint(args[0], id) || !ParseUint(args[1], a) ||
                !ParseFloat(args[2], px) || !ParseFloat(args[3], py) || !ParseFloat(args[4], pz) ||
                !ParseInt(args[5], lx) || !ParseInt(args[6], ly) || !ParseInt(args[7], lz) ||
                !ParseSdfMaterial(args[8], mat) ||
                !ParseFloat(args[9], cr) || !ParseFloat(args[10], cg) || !ParseFloat(args[11], cb)) {
                out.PrintLine("sdf_repeat_limited: arg parse failed");
                return;
            }
            if (lx < 0 || ly < 0 || lz < 0 || lx > 1023 || ly > 1023 || lz > 1023) {
                out.FormatLine("sdf_repeat_limited: limit out of range 0..1023 (got {},{},{})", lx, ly, lz);
                return;
            }
            // Pack the int3 half-extent into the params[3] float lane.
            // 10 bits per axis -> 0..1023, total 30 bits. Top 2 bits
            // stay zero so a future flag can move in without a wire-
            // format break.
            std::uint32_t packed = (std::uint32_t(lx) & 0x3ffu) |
                                   ((std::uint32_t(ly) & 0x3ffu) << 10) |
                                   ((std::uint32_t(lz) & 0x3ffu) << 20);
            float packed_f;
            std::memcpy(&packed_f, &packed, sizeof(float));
            wrap_proc(id, a, pt::renderer::SDF_OP_REPEAT_LIMITED,
                      {px, py, pz, packed_f}, mat, cr, cg, cb, out, "repeat_limited");
        });
    // ----- end SDF Phase 2 (#98) -------------------------------------------
}
// --- end SDF Phase 1 -------------------------------------------------------

// --- Physics Phase 1 (#132) ------------------------------------------------
//
// Console commands and per-frame tick for the Verlet physics layer.
// MVP scope is intentionally narrow (see issue #132): sphere-plane and
// sphere-sphere only, gravity-only forcing, no friction / restitution /
// stacking. Each spawned physics sphere is paired with an analytic
// primitive (the renderer's existing sphere intersector); per-frame the
// physics integrator's curr_pos is written back into the matching
// AnalyticPrim before RenderFrame's EnsurePrimitivesUploaded re-uploads
// the buffer + refits the BVH. That single writeback site is the only
// place the physics layer touches renderer-owned state.
//
// `phys_drop <x> <y> <z> [radius]` is the testing entry point: spawns a
// Lambert grey sphere at the world-space position and registers it with
// PhysicsSystem so the next StepPhysics tick drops it under gravity.
// IDs come from `physics_next_prim_id_` (starts at 100000 to keep clear
// of human-typed `prim_sphere <id>` IDs which are typically single digits).
//
// `phys_clear` removes every physics-owned analytic primitive AND clears
// the particle pool. User-authored prims (via `prim_sphere`) are
// untouched.
//
// `phys_ls` lists every physics-owned body (particles + rigid bodies)
// with prim_id, position, speed estimate, mass, and Lambert albedo --
// the introspection counterpart to `phys_drop_*` / `phys_clear`.
//
// `phys_status` dumps the integrator's current state for sanity checks.

void Engine::RegisterPhysicsCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("phys_drop",
        "phys_drop <x> <y> <z> [radius]: spawn a physics-driven sphere at the "
        "given world-space position (radius defaults to 0.3 m).",
        [this](auto args, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_drop: physics system not initialised"); return; }
            if (args.size() < 3 || args.size() > 4) {
                out.PrintLine("usage: phys_drop <x> <y> <z> [radius=0.3]");
                return;
            }
            float x, y, z, radius = 0.3f;
            if (!ParseFloat(args[0], x) || !ParseFloat(args[1], y) || !ParseFloat(args[2], z)) {
                out.PrintLine("phys_drop: arg parse failed");
                return;
            }
            if (args.size() == 4 && !ParseFloat(args[3], radius)) {
                out.PrintLine("phys_drop: bad radius");
                return;
            }
            if (radius <= 0.0f) {
                out.PrintLine("phys_drop: radius must be > 0");
                return;
            }
            const auto h = physics_->AddParticle(glm::vec3{x, y, z}, radius, 1.0f);
            if (h == pt::physics::PhysicsSystem::kInvalidHandle) {
                out.FormatLine("phys_drop: particle pool full ({} max)",
                               pt::physics::PhysicsSystem::kMaxParticles);
                return;
            }
            // Pair with a fresh analytic-prim slot. physics_next_prim_id_
            // starts at 100000 to stay clear of human-typed prim_sphere
            // ids (typically single digits), but a sufficiently
            // adventurous script can still issue `prim_sphere 100000 ...`
            // and collide. Probe forward from the candidate until we
            // hit an unused id so phys_drop NEVER overwrites a user-
            // authored primitive. The bump keeps the counter monotonic
            // so the next phys_drop starts past whatever id we landed
            // on, and a phys_clear will only erase the id we actually
            // claimed here (recorded via SetPrimId on the particle).
            std::uint32_t prim_id = physics_next_prim_id_++;
            while (primitives_.find(prim_id) != primitives_.end()) {
                prim_id = physics_next_prim_id_++;
            }
            AnalyticPrim p{};
            p.type        = AnalyticPrim::Sphere;
            p.material    = AnalyticPrim::Lambert;
            p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
            // Motion blur (#85): seed prev == curr so a freshly spawned
            // body produces no spurious streak on its first frame.
            p.prev_pos_or_n[0] = x; p.prev_pos_or_n[1] = y; p.prev_pos_or_n[2] = z;
            p.radius_or_d = radius;
            p.albedo[0]   = 0.80f; p.albedo[1] = 0.80f; p.albedo[2] = 0.85f;
            p.roughness   = 0.5f;
            p.ior         = 1.0f;
            primitives_[prim_id] = p;
            physics_->SetPrimId(h, prim_id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("phys_drop: spawned particle handle=0x{:x} prim_id={} @ ({:.2f} {:.2f} {:.2f}) r={:.3f}",
                           h, prim_id, x, y, z, radius);
        });

    // Spawn a rigid-body sphere (Phase 2a, #138). Unlike `phys_drop`
    // which spawns a Phase-1 point-mass Particle, this goes into the
    // rigid-body pool: it carries orientation + inertia + angular
    // velocity. Visually identical (sphere-plane / sphere-sphere
    // collision uses the bounding sphere), but the body is now a
    // first-class rigid body that Phase 2b's SAT narrowphase will
    // start to talk to.
    C.RegisterCommand("phys_drop_sphere",
        "phys_drop_sphere <x> <y> <z> [radius=0.3] [mass=1.0] [r g b] [emit_r emit_g emit_b]: "
        "spawn a rigid-body sphere (Phase 2a) at the world-space position. Optional RGB tint "
        "(#181), optional emission in W/sr per channel (#181 polish) -- defaults are the "
        "warm-grey Lambert from before the colored-bodies patch with zero emission.",
        [this](auto args, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_drop_sphere: physics system not initialised"); return; }
            // Accept 3..5 args (back-compat, no color, no emission),
            // 8 args (xyz + radius + mass + albedo rgb),
            // or 11 args (xyz + radius + mass + albedo rgb + emission rgb).
            // 6/7/9/10 are partial-color or partial-emission shapes --
            // rejecting them at the gate keeps the user-facing error
            // obvious (no silent bias from a guessed missing channel).
            if (args.size() < 3 || args.size() == 6 || args.size() == 7 ||
                args.size() == 9 || args.size() == 10 || args.size() > 11) {
                out.PrintLine("usage: phys_drop_sphere <x> <y> <z> [radius=0.3] [mass=1.0] [r g b] [emit_r emit_g emit_b]");
                return;
            }
            float x, y, z, radius = 0.3f, mass = 1.0f;
            // Default warm-grey tint -- matches the pre-#181 hard-coded
            // appearance so phys_drop_sphere with no color args renders
            // bit-for-bit identical to integration HEAD.
            float r_col = 0.85f, g_col = 0.80f, b_col = 0.70f;
            // Default emission is zero (non-emissive). Units W/sr per
            // channel, consistent with AnalyticLight::intensity for
            // point/spot lights (PR #185 photometric work).
            float r_emit = 0.0f, g_emit = 0.0f, b_emit = 0.0f;
            if (!ParseFloat(args[0], x) || !ParseFloat(args[1], y) || !ParseFloat(args[2], z)) {
                out.PrintLine("phys_drop_sphere: arg parse failed");
                return;
            }
            if (args.size() >= 4 && !ParseFloat(args[3], radius)) {
                out.PrintLine("phys_drop_sphere: bad radius");
                return;
            }
            if (args.size() >= 5 && !ParseFloat(args[4], mass)) {
                out.PrintLine("phys_drop_sphere: bad mass");
                return;
            }
            if (args.size() >= 8) {
                if (!ParseFloat(args[5], r_col) || !ParseFloat(args[6], g_col) || !ParseFloat(args[7], b_col)) {
                    out.PrintLine("phys_drop_sphere: bad r/g/b");
                    return;
                }
            }
            if (args.size() == 11) {
                if (!ParseFloat(args[8], r_emit) || !ParseFloat(args[9], g_emit) || !ParseFloat(args[10], b_emit)) {
                    out.PrintLine("phys_drop_sphere: bad emission r/g/b");
                    return;
                }
            }
            if (radius <= 0.0f) {
                out.PrintLine("phys_drop_sphere: radius must be > 0");
                return;
            }
            if (mass < 0.0f) {
                out.PrintLine("phys_drop_sphere: mass must be >= 0 (0 = kinematic)");
                return;
            }
            if (r_col < 0.0f || g_col < 0.0f || b_col < 0.0f) {
                out.PrintLine("phys_drop_sphere: rgb components must be >= 0");
                return;
            }
            // Negative emission would invert the radiance accumulator
            // and produce dark spots where the user expects a glow --
            // rejected at the gate same way albedo is.
            if (r_emit < 0.0f || g_emit < 0.0f || b_emit < 0.0f) {
                out.PrintLine("phys_drop_sphere: emission components must be >= 0");
                return;
            }
            const auto h = physics_->AddRigidSphere(glm::vec3{x, y, z}, radius, mass);
            if (h == pt::physics::PhysicsSystem::kInvalidRbHandle) {
                out.FormatLine("phys_drop_sphere: rigid body pool full ({} max)",
                               pt::physics::PhysicsSystem::kMaxRigidBodies);
                return;
            }
            const std::uint32_t prim_id = physics_next_prim_id_++;
            AnalyticPrim p{};
            p.type        = AnalyticPrim::Sphere;
            p.material    = AnalyticPrim::Lambert;
            p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
            // Motion blur (#85): seed prev == curr so a freshly spawned
            // body produces no spurious streak on its first frame.
            p.prev_pos_or_n[0] = x; p.prev_pos_or_n[1] = y; p.prev_pos_or_n[2] = z;
            p.radius_or_d = radius;
            p.albedo[0]   = r_col; p.albedo[1] = g_col; p.albedo[2] = b_col;
            p.roughness   = 0.4f;
            p.ior         = 1.0f;
            p.emission[0] = r_emit; p.emission[1] = g_emit; p.emission[2] = b_emit;
            primitives_[prim_id] = p;
            physics_->SetRbPrimId(h, prim_id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("phys_drop_sphere: spawned rigid body handle=0x{:x} prim_id={} @ ({:.2f} {:.2f} {:.2f}) r={:.3f} m={:.3f} rgb=({:.2f} {:.2f} {:.2f}) emit=({:.2f} {:.2f} {:.2f})",
                           h, prim_id, x, y, z, radius, mass, r_col, g_col, b_col, r_emit, g_emit, b_emit);
        });

    // Spawn a rigid-body box (Phase 2a, #138). Phase 2a does NOT
    // implement box-box / box-plane SAT collision -- per issue #138
    // MVP discipline, boxes use a bounding-sphere collision shape and
    // will fall to the ground but won't lie flat. Phase 2b lands SAT.
    // Rendering is also still sphere-only (no shader changes in this
    // PR), so the box renders as its bounding sphere too. That's
    // intentional: the visual delta from `phys_drop_sphere` is the
    // bounding-sphere RADIUS, which on a 1x1x1m cube is sqrt(3)/2 ~=
    // 0.866 m -- noticeably larger than the inscribed 0.5 m sphere.
    C.RegisterCommand("phys_drop_box",
        "phys_drop_box <x> <y> <z> <hx> <hy> <hz> [mass=1.0] [r g b] [emit_r emit_g emit_b]: "
        "spawn a rigid-body box (Phase 2a, sphere collision only -- box-box SAT lands in "
        "Phase 2b). Optional RGB tint (#181), optional emission in W/sr per channel (#181 "
        "polish) -- defaults are the cool blue Lambert from before the colored-bodies patch "
        "with zero emission.",
        [this](auto args, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_drop_box: physics system not initialised"); return; }
            // Back-compat / extended shapes:
            //   6 args  -> xyz + hxhyhz                                (mass defaults)
            //   7 args  -> xyz + hxhyhz + mass
            //   10 args -> xyz + hxhyhz + mass + albedo rgb              (#181)
            //   13 args -> xyz + hxhyhz + mass + albedo rgb + emit rgb   (#181 polish)
            // 8/9/11/12 are partial-RGB or partial-emission -- reject
            // at the gate so a fat-fingered arg doesn't silently bias
            // a missing channel to whatever stale value lands in the float.
            if (args.size() < 6 || args.size() == 8 || args.size() == 9 ||
                args.size() == 11 || args.size() == 12 || args.size() > 13) {
                out.PrintLine("usage: phys_drop_box <x> <y> <z> <hx> <hy> <hz> [mass=1.0] [r g b] [emit_r emit_g emit_b]");
                return;
            }
            float x, y, z, hx, hy, hz, mass = 1.0f;
            // Default cool-blue tint -- matches pre-#181 hard-coded
            // appearance for `phys_drop_box` with no color args.
            float r_col = 0.60f, g_col = 0.65f, b_col = 0.85f;
            // Default emission is zero (non-emissive). Units W/sr per
            // channel; see phys_drop_sphere for unit rationale.
            float r_emit = 0.0f, g_emit = 0.0f, b_emit = 0.0f;
            if (!ParseFloat(args[0], x)  || !ParseFloat(args[1], y)  || !ParseFloat(args[2], z) ||
                !ParseFloat(args[3], hx) || !ParseFloat(args[4], hy) || !ParseFloat(args[5], hz)) {
                out.PrintLine("phys_drop_box: arg parse failed");
                return;
            }
            if (args.size() >= 7 && !ParseFloat(args[6], mass)) {
                out.PrintLine("phys_drop_box: bad mass");
                return;
            }
            if (args.size() >= 10) {
                if (!ParseFloat(args[7], r_col) || !ParseFloat(args[8], g_col) || !ParseFloat(args[9], b_col)) {
                    out.PrintLine("phys_drop_box: bad r/g/b");
                    return;
                }
            }
            if (args.size() == 13) {
                if (!ParseFloat(args[10], r_emit) || !ParseFloat(args[11], g_emit) || !ParseFloat(args[12], b_emit)) {
                    out.PrintLine("phys_drop_box: bad emission r/g/b");
                    return;
                }
            }
            if (hx <= 0.0f || hy <= 0.0f || hz <= 0.0f) {
                out.PrintLine("phys_drop_box: all half-extents must be > 0");
                return;
            }
            if (mass < 0.0f) {
                out.PrintLine("phys_drop_box: mass must be >= 0 (0 = kinematic)");
                return;
            }
            if (r_col < 0.0f || g_col < 0.0f || b_col < 0.0f) {
                out.PrintLine("phys_drop_box: rgb components must be >= 0");
                return;
            }
            if (r_emit < 0.0f || g_emit < 0.0f || b_emit < 0.0f) {
                out.PrintLine("phys_drop_box: emission components must be >= 0");
                return;
            }
            const glm::vec3 h_ext{hx, hy, hz};
            const auto h = physics_->AddRigidBox(glm::vec3{x, y, z}, h_ext, mass);
            if (h == pt::physics::PhysicsSystem::kInvalidRbHandle) {
                out.FormatLine("phys_drop_box: rigid body pool full ({} max)",
                               pt::physics::PhysicsSystem::kMaxRigidBodies);
                return;
            }
            // Render as bounding sphere for Phase 2a -- shader work
            // (real box primitive in the analytic-prim shader) is
            // explicitly out of scope per issue #138.
            const float bound_r = pt::physics::BoxBoundingRadius(h_ext);
            const std::uint32_t prim_id = physics_next_prim_id_++;
            AnalyticPrim p{};
            p.type        = AnalyticPrim::Sphere;
            p.material    = AnalyticPrim::Lambert;
            p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
            // Motion blur (#85): seed prev == curr so a freshly spawned
            // body produces no spurious streak on its first frame.
            p.prev_pos_or_n[0] = x; p.prev_pos_or_n[1] = y; p.prev_pos_or_n[2] = z;
            p.radius_or_d = bound_r;
            p.albedo[0]   = r_col; p.albedo[1] = g_col; p.albedo[2] = b_col;
            p.roughness   = 0.5f;
            p.ior         = 1.0f;
            p.emission[0] = r_emit; p.emission[1] = g_emit; p.emission[2] = b_emit;
            primitives_[prim_id] = p;
            physics_->SetRbPrimId(h, prim_id);
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("phys_drop_box: spawned rigid body handle=0x{:x} prim_id={} @ ({:.2f} {:.2f} {:.2f}) half=({:.2f} {:.2f} {:.2f}) m={:.3f} bound_r={:.3f} rgb=({:.2f} {:.2f} {:.2f}) emit=({:.2f} {:.2f} {:.2f})",
                           h, prim_id, x, y, z, hx, hy, hz, mass, bound_r, r_col, g_col, b_col, r_emit, g_emit, b_emit);
        });

    C.RegisterCommand("phys_clear",
        "Remove all physics-owned analytic primitives and clear the particle + rigid-body pools.",
        [this](auto, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_clear: physics system not initialised"); return; }
            // Each live particle has a paired analytic-prim entry --
            // gather the prim ids before clearing so we don't iterate
            // a mutating map.
            std::vector<std::uint32_t> prim_ids;
            prim_ids.reserve(physics_->AliveCount() + physics_->RbAliveCount());
            physics_->ForEach(
                [](pt::physics::PhysicsSystem::Handle /*h*/,
                   const pt::physics::Particle& /*p*/,
                   std::uint32_t prim_id, void* user) {
                    auto* vec = static_cast<std::vector<std::uint32_t>*>(user);
                    if (prim_id != 0) vec->push_back(prim_id);
                },
                &prim_ids);
            // Mirror for the rigid-body pool (Phase 2a, #138). Same
            // callback signature shape; different per-element type.
            physics_->ForEachRigidBody(
                [](pt::physics::PhysicsSystem::RbHandle /*h*/,
                   const pt::physics::RigidBody& /*b*/,
                   std::uint32_t prim_id, void* user) {
                    auto* vec = static_cast<std::vector<std::uint32_t>*>(user);
                    if (prim_id != 0) vec->push_back(prim_id);
                },
                &prim_ids);
            // Count actual erases, not the gathered-id total. If the
            // user nuked some of these prims out-of-band (via
            // prim_remove or by re-issuing prim_sphere with the same
            // id), std::map::erase returns 0 for those slots and the
            // reported count would over-state what phys_clear actually
            // did.
            std::uint32_t removed = 0;
            for (auto id : prim_ids) {
                removed += static_cast<std::uint32_t>(primitives_.erase(id));
            }
            physics_->Clear();
            // Drop the r_phys_debug_visualize cache (#181). Those
            // entries are keyed by prim_id; with the underlying prims
            // gone the cache contents would just leak to no purpose.
            phys_debug_color_cache_.clear();
            if (removed > 0) {
                primitives_dirty_ = true;
                accum_dirty_      = true;
            }
            out.FormatLine("phys_clear: removed {} physics-owned prim(s); particle + rigid-body pools emptied", removed);
        });

    C.RegisterCommand("phys_status",
        "Print the current Verlet physics layer status (enabled, particle / rigid-body counts, "
        "gravity / damping / substeps, debug_viz). One-line summary intended for the chat-style "
        "console transcript.",
        [this](auto, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_status: physics system not initialised"); return; }
            auto& Cl = pt::console::Console::Get();
            bool en = false;
            int  sub = 8;
            float gy  = -9.81f;
            float dmp = 0.99f;
            // debug_viz mirrors r_phys_debug_visualize (#181). Treated as a
            // straight int so a future ramp/heat-map mode (>1) prints
            // its raw value rather than getting boolean-coerced to 1.
            int  viz = 0;
            if (auto* v = Cl.FindCVar("phys_enabled"))           en  = v->GetBool();
            if (auto* v = Cl.FindCVar("phys_substeps"))          sub = v->GetInt();
            if (auto* v = Cl.FindCVar("phys_gravity_y"))         gy  = v->GetFloat();
            if (auto* v = Cl.FindCVar("phys_damping"))           dmp = v->GetFloat();
            if (auto* v = Cl.FindCVar("r_phys_debug_visualize")) viz = v->GetInt();
            out.FormatLine("phys: enabled={} particles={}/{} rigid_bodies={}/{} gravity_y={:.3f} m/s^2 substeps={} damping={:.4f} debug_viz={}",
                           en ? 1 : 0,
                           physics_->AliveCount(),
                           physics_->Capacity(),
                           physics_->RbAliveCount(),
                           physics_->RbCapacity(),
                           gy, sub, dmp, viz);
        });

    // `phys_ls` -- list every physics-owned body (particles + rigid bodies)
    // alongside its paired analytic-prim id, position, speed estimate, mass,
    // and per-body colour. Useful for sanity-checking that a `phys_drop_*`
    // call actually landed in the pool and is being driven, and for spotting
    // run-away velocities before `phys_clear` ing the scene.
    //
    // Velocity reporting: Verlet stores prev/curr positions, so the implicit
    // velocity is `(curr - prev) / sdt`. `phys_ls` is a console command (no
    // tick context), so we derive `sdt` from `phys_substeps` plus a nominal
    // 1/60 s frame_dt -- matching what StepPhysics would compute on a typical
    // 60 Hz frame. The resulting speed is therefore an ESTIMATE; if the user
    // is running at a wildly different frame rate the absolute number will
    // be off, but the ranking and "is it moving at all?" signal stays
    // honest. This is the same pragmatic compromise the velocity debug viz
    // (#181) lives with.
    C.RegisterCommand("phys_ls",
        "List every physics-owned body (particles + rigid bodies) with prim_id, position, "
        "estimated speed (m/s @ 60 Hz), mass, and Lambert albedo.",
        [this](auto, pt::console::Output& out) {
            if (!physics_) { out.PrintLine("phys_ls: physics system not initialised"); return; }
            const std::uint32_t n_particles = physics_->AliveCount();
            const std::uint32_t n_bodies    = physics_->RbAliveCount();
            if (n_particles + n_bodies == 0) {
                out.PrintLine("phys_ls: no rigid bodies");
                return;
            }
            // Pull substep count from cvar so the (curr - prev) / sdt
            // estimate uses the same value StepPhysics will on the next
            // tick. Clamp mirrors PhysicsSystem::Step (see comments
            // there) so a phys_substeps=0 doesn't blow up the divide.
            auto& Cl = pt::console::Console::Get();
            int substeps = 8;
            if (auto* v = Cl.FindCVar("phys_substeps")) substeps = v->GetInt();
            if (substeps < 1)  substeps = 1;
            if (substeps > 32) substeps = 32;
            constexpr float kNominalFrameDt = 1.0f / 60.0f;
            const float inv_sdt = float(substeps) / kNominalFrameDt;

            // Table header. Widths picked to fit the typical Lambert
            // primitive: 12-digit prim_id (physics ids start at 100000),
            // three 7-char position columns, 7-char speed, 6-char mass,
            // RGB. Fits a standard 80-col terminal at full precision.
            out.FormatLine("phys_ls: {} particle(s) + {} rigid body(ies)", n_particles, n_bodies);
            out.PrintLine("  idx  prim_id        pos (x, y, z)                  speed   mass    rgb");

            // Particle pool first. ForEach is a C-style callback, so we
            // bounce engine state through a Ctx struct. The lambda
            // captures nothing (decays via unary +) so it can satisfy
            // the function-pointer signature without an std::function
            // allocation per call.
            struct LsCtx {
                pt::console::Output*                                 out      = nullptr;
                const std::map<std::uint32_t, Engine::AnalyticPrim>* prims    = nullptr;
                float                                                inv_sdt  = 0.0f;
                std::uint32_t                                        idx      = 0;
                const char*                                          kind     = "particle";
            };
            LsCtx ctx{};
            ctx.out      = &out;
            ctx.prims    = &primitives_;
            ctx.inv_sdt  = inv_sdt;
            ctx.kind     = "particle";

            auto particle_cb = +[](pt::physics::PhysicsSystem::Handle /*h*/,
                                   const pt::physics::Particle& p,
                                   std::uint32_t prim_id, void* user) {
                auto* c = static_cast<LsCtx*>(user);
                const glm::vec3 dv = p.curr_pos - p.prev_pos;
                const float speed  = glm::length(dv) * c->inv_sdt;
                const float mass   = (p.inv_mass > 0.0f) ? (1.0f / p.inv_mass) : 0.0f;
                float r = 0.0f, g = 0.0f, b = 0.0f;
                if (auto it = c->prims->find(prim_id); it != c->prims->end()) {
                    r = it->second.albedo[0];
                    g = it->second.albedo[1];
                    b = it->second.albedo[2];
                }
                c->out->FormatLine("  [{:>2}] {:<8} {:>12} ({:>6.2f}, {:>6.2f}, {:>6.2f}) {:>6.2f}  {:>5.2f}  ({:.2f}, {:.2f}, {:.2f})",
                                   c->idx, c->kind, prim_id,
                                   p.curr_pos.x, p.curr_pos.y, p.curr_pos.z,
                                   speed, mass, r, g, b);
                ++c->idx;
            };
            physics_->ForEach(particle_cb, &ctx);

            // Rigid bodies second. Same shape; pull mass from RigidBody's
            // inv_mass (mirrors Particle's storage convention). Speed
            // uses (curr - prev) / sdt the same way -- rotational
            // (omega) state is real but intentionally NOT surfaced here:
            // a phys_ls scan should fit one body per line; users who
            // care about angular state should reach for a future
            // phys_inspect <prim_id> command.
            ctx.kind = "rigid";
            auto rb_cb = +[](pt::physics::PhysicsSystem::RbHandle /*h*/,
                             const pt::physics::RigidBody& b,
                             std::uint32_t prim_id, void* user) {
                auto* c = static_cast<LsCtx*>(user);
                const glm::vec3 dv = b.curr_pos - b.prev_pos;
                const float speed  = glm::length(dv) * c->inv_sdt;
                const float mass   = (b.inv_mass > 0.0f) ? (1.0f / b.inv_mass) : 0.0f;
                float r = 0.0f, g = 0.0f, bl = 0.0f;
                if (auto it = c->prims->find(prim_id); it != c->prims->end()) {
                    r  = it->second.albedo[0];
                    g  = it->second.albedo[1];
                    bl = it->second.albedo[2];
                }
                c->out->FormatLine("  [{:>2}] {:<8} {:>12} ({:>6.2f}, {:>6.2f}, {:>6.2f}) {:>6.2f}  {:>5.2f}  ({:.2f}, {:.2f}, {:.2f})",
                                   c->idx, c->kind, prim_id,
                                   b.curr_pos.x, b.curr_pos.y, b.curr_pos.z,
                                   speed, mass, r, g, bl);
                ++c->idx;
            };
            physics_->ForEachRigidBody(rb_cb, &ctx);
        });
}

void Engine::MaybeRestorePhysDebugColors() {
    // Issue #181 -- the r_phys_debug_visualize 1 -> 0 falling-edge
    // restore. Called from Tick BEFORE StepPhysics every frame, so the
    // restore runs even when StepPhysics' early returns kick in
    // (physics_ null, no live bodies, phys_enabled=0, frame dt non-
    // positive). Without this hoist, turning the cvar off while the
    // simulation was idle would leave the debug-coloured spheres
    // visible indefinitely until physics started stepping again.
    auto& C = pt::console::Console::Get();
    int viz_mode = 0;
    if (auto* v = C.FindCVar("r_phys_debug_visualize")) viz_mode = v->GetInt();

    if (phys_debug_visualize_prev_ != 0 && viz_mode == 0
        && !phys_debug_color_cache_.empty()) {
        for (auto& [prim_id, orig] : phys_debug_color_cache_) {
            auto it = primitives_.find(prim_id);
            if (it == primitives_.end()) continue;
            auto& ap = it->second;
            ap.albedo[0] = orig[0];
            ap.albedo[1] = orig[1];
            ap.albedo[2] = orig[2];
        }
        phys_debug_color_cache_.clear();
        primitives_dirty_ = true;
        accum_dirty_      = true;
    }
    phys_debug_visualize_prev_ = viz_mode;
}

void Engine::StepPhysics(float dt) {
    if (!physics_) return;
    if (physics_->AliveCount() == 0 && physics_->RbAliveCount() == 0) return;

    auto& C = pt::console::Console::Get();
    bool enabled = false;
    if (auto* v = C.FindCVar("phys_enabled")) enabled = v->GetBool();
    if (!enabled) return;

    int substeps = 8;
    float gy     = -9.81f;
    float damp   = 0.99f;
    if (auto* v = C.FindCVar("phys_substeps"))  substeps = v->GetInt();
    if (auto* v = C.FindCVar("phys_gravity_y")) gy       = v->GetFloat();
    if (auto* v = C.FindCVar("phys_damping"))   damp     = v->GetFloat();

    // Clamp dt to a sane upper bound so a single huge frame (debugger
    // pause, swapchain hitch, window-drag stall) doesn't fling the
    // simulation across the world. 1/30 s = 33.3 ms is the ceiling --
    // anything beyond that is discarded (NOT accumulated into a
    // catch-up loop), so a 200 ms hitch produces 33 ms of physics
    // motion and the rest is wall-time-only. This is the intentional
    // Phase 1 behaviour: physics dilates during hitches rather than
    // burning a full hitch's worth of substeps in the next frame
    // (which could itself overrun and cascade). A proper fixed-step
    // accumulator with carry-over lands in Phase 5 alongside the
    // sleeper / contact-persistence work.
    float clamped_dt = dt;
    constexpr float kMaxStepSec = 1.0f / 30.0f;
    if (clamped_dt > kMaxStepSec) clamped_dt = kMaxStepSec;
    if (clamped_dt <= 0.0f)        return;

    physics_->Step(clamped_dt, substeps, gy, damp);

    // --- Issue #181 velocity-magnitude debug viz --------------------
    // Pull r_phys_debug_visualize once outside the iter callbacks (the
    // C-style ForEach signature can't capture state, and we want the
    // cvar GetInt cost off the per-particle path). 0 = off, 1 = colour
    // override active. The 1 -> 0 falling-edge restore lives in
    // MaybeRestorePhysDebugColors() called from Tick BEFORE this
    // function (so it runs even when the early returns above kick in);
    // here we only handle the override-while-active path.
    int viz_mode = 0;
    if (auto* v = C.FindCVar("r_phys_debug_visualize")) viz_mode = v->GetInt();
    // Substep dt is the same value PhysicsSystem::Step computed
    // internally -- substeps is clamped to [1, 32] there (see Step()),
    // so mirror that clamp here. Without the mirror, a user with
    // phys_substeps=0 or negative would see physics run at the
    // clamped sdt while the viz computed speeds at a wildly
    // different sdt (or zero, painting everything blue).
    int clamped_substeps = substeps;
    if (clamped_substeps < 1)  clamped_substeps = 1;
    if (clamped_substeps > 32) clamped_substeps = 32;
    const float inv_sdt = (clamped_dt > 0.0f)
                             ? float(clamped_substeps) / clamped_dt
                             : 0.0f;

    // Write each particle's curr_pos back into its paired analytic
    // primitive so RenderFrame's EnsurePrimitivesUploaded sees the
    // moved sphere. The callback signature is C-style by design so
    // the physics library stays free of std::function overhead --
    // we pass `this` as `user` and recover the engine pointer to
    // mutate `primitives_`. The Ctx struct also carries the debug-viz
    // state so the callback can encode speed into albedo when active
    // without a second pass over the pool.
    struct Ctx {
        std::map<std::uint32_t, Engine::AnalyticPrim>* prims        = nullptr;
        std::map<std::uint32_t, std::array<float, 3>>* color_cache  = nullptr;
        int   viz_mode                                              = 0;
        float inv_sdt                                               = 0.0f;
        void (*speed_to_rgb)(float, float*)                         = nullptr;
        bool  any_changed                                           = false;
    };
    Ctx ctx{};
    ctx.prims        = &primitives_;
    ctx.color_cache  = &phys_debug_color_cache_;
    ctx.viz_mode     = viz_mode;
    ctx.inv_sdt      = inv_sdt;
    // Speed -> linear-RGB ramp. Three stops calibrated to the default
    // damped Verlet integrator: terminal implicit speed under default
    // (g=-9.81, phys_substeps=8, phys_damping=0.99, 60 fps) is
    // |g| * sdt / (1-damping) ~= 2.04 m/s, so the ramp tops out at 2.
    // Picking the 5/10 m/s thresholds we shipped initially saturated
    // every default-config falling body to the bottom of the ramp;
    // 0/1/2 covers the full implicit-velocity range cleanly.
    //   0 m/s   -> (0.10, 0.30, 1.00)   blue   (stationary)
    //   1 m/s   -> (0.20, 1.00, 0.30)   green  (mid-fall)
    //   2+ m/s  -> (1.00, 0.20, 0.20)   red    (terminal, saturating)
    // Users who set phys_damping > 0.99 (terminal climbs) or supply
    // external impulses (Phase 2b+) past 2 m/s will see saturated red
    // -- that's the intent: "moving as fast as the system can manage".
    // Computed in linear-RGB so the ACES tonemap reads it correctly.
    // C++ lambdas-without-captures decay to function pointers via the
    // unary `+`, which lets the C-style ForEach callback dispatch
    // through Ctx without paying for std::function indirection.
    ctx.speed_to_rgb = +[](float s, float* out_rgb) {
        float v = s;
        if (v < 0.0f) v = 0.0f;
        if (v > 2.0f) v = 2.0f;
        constexpr float kStop0[3] = {0.10f, 0.30f, 1.00f};
        constexpr float kStop1[3] = {0.20f, 1.00f, 0.30f};
        constexpr float kStop2[3] = {1.00f, 0.20f, 0.20f};
        const float* a;
        const float* b;
        float t;
        if (v < 1.0f) { a = kStop0; b = kStop1; t = v; }
        else          { a = kStop1; b = kStop2; t = v - 1.0f; }
        for (int k = 0; k < 3; ++k) out_rgb[k] = a[k] + (b[k] - a[k]) * t;
    };

    physics_->ForEach(
        [](pt::physics::PhysicsSystem::Handle /*h*/,
           const pt::physics::Particle& p,
           std::uint32_t prim_id, void* user) {
            auto* c = static_cast<Ctx*>(user);
            if (prim_id == 0) return;
            auto it = c->prims->find(prim_id);
            if (it == c->prims->end()) return;
            auto& ap = it->second;
            // Only update sphere prims -- planes/etc. share the type
            // tag space but physics never targets them.
            if (ap.type != AnalyticPrim::Sphere) return;
            // Motion blur (#85): snapshot the previous curr_pos to
            // prev_pos_or_n BEFORE writing the new curr_pos. The shader
            // lerps between prev_pos_or_n and pos_or_n at the ray's
            // shutter-time sample so a falling body shows a streak
            // proportional to its per-frame displacement.
            ap.prev_pos_or_n[0] = ap.pos_or_n[0];
            ap.prev_pos_or_n[1] = ap.pos_or_n[1];
            ap.prev_pos_or_n[2] = ap.pos_or_n[2];
            ap.pos_or_n[0] = p.curr_pos.x;
            ap.pos_or_n[1] = p.curr_pos.y;
            ap.pos_or_n[2] = p.curr_pos.z;
            // Radius is fixed for now (no growing/shrinking particles
            // in Phase 1); skip writing it.
            if (c->viz_mode == 1) {
                // Cache the pre-override albedo on first touch so the
                // 1 -> 0 transition can restore it. Idempotent: the
                // cache key is prim_id and emplace is a no-op once the
                // slot exists.
                auto [cache_it, inserted] = c->color_cache->try_emplace(
                    prim_id,
                    std::array<float, 3>{ ap.albedo[0], ap.albedo[1], ap.albedo[2] });
                (void)cache_it;
                (void)inserted;
                const glm::vec3 disp = p.curr_pos - p.prev_pos;
                const float speed    = glm::length(disp) * c->inv_sdt;
                float rgb[3];
                c->speed_to_rgb(speed, rgb);
                ap.albedo[0] = rgb[0];
                ap.albedo[1] = rgb[1];
                ap.albedo[2] = rgb[2];
            }
            c->any_changed = true;
        },
        &ctx);

    // Rigid-body writeback (Phase 2a, #138). Renders as the bounding
    // sphere -- shaders/* are off-limits per issue #138 so we can't
    // upload a real OBB primitive yet. Orientation is integrated and
    // updated but doesn't visibly affect the sphere render; Phase 2b
    // adds the box analytic-prim shader path and starts using it.
    physics_->ForEachRigidBody(
        [](pt::physics::PhysicsSystem::RbHandle /*h*/,
           const pt::physics::RigidBody& b,
           std::uint32_t prim_id, void* user) {
            auto* c = static_cast<Ctx*>(user);
            if (prim_id == 0) return;
            auto it = c->prims->find(prim_id);
            if (it == c->prims->end()) return;
            auto& ap = it->second;
            if (ap.type != AnalyticPrim::Sphere) return;
            // Motion blur (#85): snapshot prev_pos before writing curr.
            // See the Particle callback above for the design note.
            ap.prev_pos_or_n[0] = ap.pos_or_n[0];
            ap.prev_pos_or_n[1] = ap.pos_or_n[1];
            ap.prev_pos_or_n[2] = ap.pos_or_n[2];
            ap.pos_or_n[0] = b.curr_pos.x;
            ap.pos_or_n[1] = b.curr_pos.y;
            ap.pos_or_n[2] = b.curr_pos.z;
            if (c->viz_mode == 1) {
                auto [cache_it, inserted] = c->color_cache->try_emplace(
                    prim_id,
                    std::array<float, 3>{ ap.albedo[0], ap.albedo[1], ap.albedo[2] });
                (void)cache_it;
                (void)inserted;
                // Rigid bodies expose linear speed via the same
                // (curr - prev) / sdt implicit-velocity scheme as
                // particles. Angular speed is intentionally NOT folded
                // in; the visualisation is for translational motion,
                // and tumbling-but-stationary bodies should read as
                // blue (matches the user's intuition of "not moving").
                const glm::vec3 disp = b.curr_pos - b.prev_pos;
                const float speed    = glm::length(disp) * c->inv_sdt;
                float rgb[3];
                c->speed_to_rgb(speed, rgb);
                ap.albedo[0] = rgb[0];
                ap.albedo[1] = rgb[1];
                ap.albedo[2] = rgb[2];
            }
            c->any_changed = true;
        },
        &ctx);

    if (ctx.any_changed) {
        primitives_dirty_ = true;
        accum_dirty_      = true;
    }
}
// --- end Physics Phase 1 ---------------------------------------------------
// --- Light primitives (#73) ------------------------------------------------
//
// Console commands for analytic light primitives. Mirrors the
// `prim_*` / `sdf_*` convention:
//   * `light_point <id> <x> <y> <z> <r> <g> <b>` -- omnidirectional
//      emitter at a point. Intensity is radiant intensity per channel
//      in W/sr (candela-equivalent).
//   * `light_spot <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg>
//      <inner_deg> <r> <g> <b>` -- point with an angular cosine
//      falloff cone. `dir` is the EMISSION axis (from the emitter
//      outward into the scene); `outer_deg` / `inner_deg` are the
//      cone half-angles in degrees (light is full inside inner,
//      zero outside outer, linear cosine ramp between).
//   * `light_sphere <id> <x> <y> <z> <radius> <r> <g> <b>` -- diffuse
//      area emitter on a sphere surface. Intensity is surface
//      radiance in W/m^2/sr; the integrator samples a uniform point
//      on the sphere each NEE call.
//   * `light_quad <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz>
//      <u_half> <v_half> <r> <g> <b>` -- diffuse one-sided rectangle.
//      `n` is the front-face normal; `u` is the in-plane direction
//      (re-orthogonalised against n); `u_half` / `v_half` are the
//      half-extent lengths in metres.
//   * `light_list` / `light_clear` / `light_remove` -- management.

namespace {

const char* LightTypeName(Engine::AnalyticLight::Type t) {
    switch (t) {
        case Engine::AnalyticLight::Point:  return "point";
        case Engine::AnalyticLight::Spot:   return "spot";
        case Engine::AnalyticLight::Sphere: return "sphere";
        case Engine::AnalyticLight::Quad:   return "quad";
    }
    return "?";
}

// --- Voxel destruction Phase 1 (#140) --------------------------------------
//
// Voxel commands. The engine voxelizes a source object on demand and
// stores the resulting VoxelGrid in voxel_grids_ keyed by source id.
// Per-frame, SyncVoxelDemoState injects the occupied voxels into
// sdf_prims_ as analytic-box SDF clusters when r_voxelize_demo=1, and
// pulls them back out when 0. No shader changes are required -- the
// path tracer treats them as ordinary SDF clusters.

// Helper: build a VoxelMaterial from an AnalyticPrim. Material kind
// mapping matches the wire format (0 Lambert, 1 Metal, 2 Dielectric).
pt::destruction::VoxelMaterial MaterialFromAnalyticPrim(const Engine::AnalyticPrim& p) {
    pt::destruction::VoxelMaterial m{};
    m.kind      = static_cast<std::uint32_t>(p.material);
    m.albedo    = { p.albedo[0], p.albedo[1], p.albedo[2] };
    m.roughness = p.roughness;
    m.ior       = p.ior;
    return m;
}

pt::destruction::VoxelMaterial MaterialFromSdfPrim(const pt::renderer::SdfPrim& p) {
    pt::destruction::VoxelMaterial m{};
    m.kind      = p.material;
    m.albedo    = { p.albedo[0], p.albedo[1], p.albedo[2] };
    m.roughness = p.roughness;
    m.ior       = p.ior;
    return m;
}

// Material for CSG meshes. CsgScene doesn't carry per-node material
// today, so Phase 1 ships with a fixed default that visually
// resembles the headline drilled cube (warm metal) -- the goal of
// `r_voxelize_demo` is the chunk-pile silhouette, not perfect
// material fidelity. Later phases (issue #139 Phase 4) attach
// per-voxel material strength.
pt::destruction::VoxelMaterial DefaultMeshVoxelMaterial() {
    pt::destruction::VoxelMaterial m{};
    m.kind      = 1u;     // Metal
    m.albedo    = { 0.85f, 0.65f, 0.30f };  // gold-ish
    m.roughness = 0.15f;
    m.ior       = 1.0f;
    return m;
}

// Build a single-LEAF SDF cluster of shape BOX for one occupied
// voxel. Half-extents are voxel_size/2 on every axis; center is the
// voxel center in world space. AABB is filled by ComputeSdfAabb so
// the BVH bound is tight.
pt::renderer::SdfPrim VoxelToSdfCluster(const std::array<float, 3>& center,
                                        float voxel_size,
                                        const pt::destruction::VoxelMaterial& mat) {
    using namespace pt::renderer;
    SdfPrim prim{};
    prim.node_count = 1u;
    prim.material   = mat.kind;
    prim.albedo[0]  = mat.albedo[0];
    prim.albedo[1]  = mat.albedo[1];
    prim.albedo[2]  = mat.albedo[2];
    prim.roughness  = mat.roughness;
    prim.ior        = mat.ior;
    SdfNode& n   = prim.nodes[0];
    n.op         = SDF_OP_LEAF;
    n.shape      = SDF_SHAPE_BOX;
    n.child_a    = 0u;
    n.child_b    = 0u;
    const float h = 0.5f * voxel_size;
    n.params[0]  = h;
    n.params[1]  = h;
    n.params[2]  = h;
    n.params[3]  = 0.0f;
    n.center[0]  = center[0];
    n.center[1]  = center[1];
    n.center[2]  = center[2];
    ComputeSdfAabb(prim);
    return prim;
}

// --- Photometric conversion helpers (#176) ---------------------------------
//
// Industry-standard light authoring lets the artist think in candela /
// lumen / chromaticity-color-times-scalar terms rather than raw W/sr.
// All conversions here are the SINGLE-WAVELENGTH (555nm peak photopic)
// approximation -- the same shortcut Arnold / RenderMan / Octane take
// when they expose a `units: lumens` enum on a point light. Per-channel
// spectral exactness would require a spectral upsampling table from
// RGB; that's deferred (see issue #176 "Out of scope").
//
// 683.002 lm/W is the luminous efficacy of monochromatic radiation at
// 555nm (peak of the CIE 1924 photopic response curve), the value the
// SI base-unit definition of the candela pins. So:
//
//   1 cd  (= 1 lm/sr) = 1/683.002 W/sr   at 555nm
//   1 nit (= 1 cd/m^2) = 1/683.002 W/m^2/sr   at 555nm
//
// For an OMNIDIRECTIONAL point emitter of luminous intensity I [cd],
// the total luminous flux is Phi = 4*pi*I [lm]. Inverting for the
// `_lm` variant gives I_cd = Phi / (4*pi), then W/sr = I_cd / 683.002.
//
// The "exposure stop" is a log2 multiplier matching camera EV: each
// stop doubles brightness, so a final scalar of intensity * 2^ev is
// applied AFTER the unit conversion. This composes cleanly with the
// engine's r_exposure tonemap stop (they multiply, but EV-on-light
// is per-light while r_exposure is global).
// Kept as a documented anchor: this is the SI value (683.002 lm/W at
// 555nm peak photopic) all conversions below derive from. Referenced
// only by comments / docs, hence the [[maybe_unused]] gate.
[[maybe_unused]] constexpr float kLuminousEfficacy555nm = 683.002f;            // lm/W
constexpr float kWsrPerCandela          = 1.0f / 683.002f;     // W/sr per cd
// W/sr per lm for OMNIDIRECTIONAL emission (lm/(4*pi*sr) -> W/sr).
constexpr float kWsrPerLumenOmni        = 1.0f / (683.002f * 4.0f * 3.14159265358979323846f);
constexpr float kWm2srPerNit            = 1.0f / 683.002f;     // W/m^2/sr per cd/m^2

// 2^ev. std::exp2f is preferred over std::pow(2,ev) (slightly faster +
// monotone). For ev=0 returns exactly 1.0f.
inline float Exposure2x(float ev) {
    return std::exp2(ev);
}

}  // namespace

void Engine::RegisterLightCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("light_list",
        "List analytic light primitives.",
        [this](auto, pt::console::Output& out) {
            if (light_prims_.empty()) { out.PrintLine("(no lights)"); return; }
            out.FormatLine("{} analytic light(s):", light_prims_.size());
            for (const auto& [id, L] : light_prims_) {
                out.FormatLine("  {:>3}  {:<6}  pos=({:.2f} {:.2f} {:.2f})  intensity=({:.2f} {:.2f} {:.2f})",
                               id, LightTypeName(L.type),
                               L.pos[0], L.pos[1], L.pos[2],
                               L.intensity[0], L.intensity[1], L.intensity[2]);
            }
        });

    C.RegisterCommand("light_clear",
        "Remove all analytic light primitives.",
        [this](auto, pt::console::Output& out) {
            light_prims_.clear();
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.PrintLine("lights: cleared");
        });

    C.RegisterCommand("light_remove",
        "light_remove <id>: drop an analytic light.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: light_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("light_remove: id parse failed"); return; }
            if (light_prims_.erase(id) == 0) {
                out.FormatLine("light_remove: id {} not found", id);
                return;
            }
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: removed id {}", id);
        });

    C.RegisterCommand("light_point",
        "light_point <id> <x> <y> <z> <r> <g> <b>: add or replace an omnidirectional point light (intensity in W/sr per channel).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 7) {
                out.PrintLine("usage: light_point <id> <x> <y> <z> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float x, y, z, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], r) || !ParseFloat(args[5], g) || !ParseFloat(args[6], b)) {
                out.PrintLine("light_point: arg parse failed");
                return;
            }
            AnalyticLight L{};
            L.type = AnalyticLight::Point;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.intensity[0] = r; L.intensity[1] = g; L.intensity[2] = b;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: point id={} pos=({:.2f} {:.2f} {:.2f})", id, x, y, z);
        });

    C.RegisterCommand("light_spot",
        "light_spot <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b>: add or replace a spot light (W/sr per channel, angular cosine falloff between inner and outer cone half-angles).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 12) {
                out.PrintLine("usage: light_spot <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], dx) || !ParseFloat(args[5], dy) || !ParseFloat(args[6], dz) ||
                !ParseFloat(args[7], outer_deg) || !ParseFloat(args[8], inner_deg) ||
                !ParseFloat(args[9], r)  || !ParseFloat(args[10], g) || !ParseFloat(args[11], b)) {
                out.PrintLine("light_spot: arg parse failed");
                return;
            }
            const float dlen = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dlen < 1e-6f) { out.PrintLine("light_spot: dir magnitude is zero"); return; }
            dx /= dlen; dy /= dlen; dz /= dlen;
            if (outer_deg < 0.0f) outer_deg = 0.0f;
            if (outer_deg > 89.99f) outer_deg = 89.99f;
            if (inner_deg < 0.0f) inner_deg = 0.0f;
            if (inner_deg > outer_deg) inner_deg = outer_deg;
            AnalyticLight L{};
            L.type = AnalyticLight::Spot;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.dir[0] = dx; L.dir[1] = dy; L.dir[2] = dz;
            L.intensity[0] = r; L.intensity[1] = g; L.intensity[2] = b;
            const float deg2rad = 3.14159265358979323846f / 180.0f;
            L.cos_outer = std::cos(outer_deg * deg2rad);
            L.cos_inner = std::cos(inner_deg * deg2rad);
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: spot id={} pos=({:.2f} {:.2f} {:.2f}) dir=({:.2f} {:.2f} {:.2f}) outer={:.1f} inner={:.1f}",
                           id, x, y, z, dx, dy, dz, outer_deg, inner_deg);
        });

    C.RegisterCommand("light_sphere",
        "light_sphere <id> <x> <y> <z> <radius> <r> <g> <b>: add or replace a spherical diffuse area light (radiance in W/m^2/sr).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 8) {
                out.PrintLine("usage: light_sphere <id> <x> <y> <z> <radius> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float x, y, z, radius, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], radius) ||
                !ParseFloat(args[5], r) || !ParseFloat(args[6], g) || !ParseFloat(args[7], b)) {
                out.PrintLine("light_sphere: arg parse failed");
                return;
            }
            if (radius <= 0.0f) { out.PrintLine("light_sphere: radius must be > 0"); return; }
            AnalyticLight L{};
            L.type = AnalyticLight::Sphere;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.radius = radius;
            L.intensity[0] = r; L.intensity[1] = g; L.intensity[2] = b;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: sphere id={} pos=({:.2f} {:.2f} {:.2f}) r={:.3f}", id, x, y, z, radius);
        });

    C.RegisterCommand("light_quad",
        "light_quad <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b>: add or replace a rectangular diffuse one-sided area light (front face = n, u-axis in-plane, both half-extents in metres).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 15) {
                out.PrintLine("usage: light_quad <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], nx) || !ParseFloat(args[5], ny) || !ParseFloat(args[6], nz) ||
                !ParseFloat(args[7], ux) || !ParseFloat(args[8], uy) || !ParseFloat(args[9], uz) ||
                !ParseFloat(args[10], uh) || !ParseFloat(args[11], vh) ||
                !ParseFloat(args[12], r) || !ParseFloat(args[13], g) || !ParseFloat(args[14], b)) {
                out.PrintLine("light_quad: arg parse failed");
                return;
            }
            const float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nlen < 1e-6f) { out.PrintLine("light_quad: normal magnitude is zero"); return; }
            nx /= nlen; ny /= nlen; nz /= nlen;
            // Re-orthogonalise u against n: u' = normalize(u - n*(n.u)).
            const float ndu = nx*ux + ny*uy + nz*uz;
            float uxo = ux - nx*ndu;
            float uyo = uy - ny*ndu;
            float uzo = uz - nz*ndu;
            const float ulen = std::sqrt(uxo*uxo + uyo*uyo + uzo*uzo);
            if (ulen < 1e-6f) { out.PrintLine("light_quad: u-axis collinear with normal"); return; }
            uxo /= ulen; uyo /= ulen; uzo /= ulen;
            if (uh <= 0.0f || vh <= 0.0f) {
                out.PrintLine("light_quad: u_half and v_half must be > 0");
                return;
            }
            AnalyticLight L{};
            L.type = AnalyticLight::Quad;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.dir[0] = nx; L.dir[1] = ny; L.dir[2] = nz;
            // u_vec is stored as the FULL half-extent vector so the
            // shader can recover both the axis (length-normalised) and
            // the half-extent (length).
            L.u_vec[0] = uxo * uh; L.u_vec[1] = uyo * uh; L.u_vec[2] = uzo * uh;
            L.v_half   = vh;
            L.intensity[0] = r; L.intensity[1] = g; L.intensity[2] = b;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: quad id={} pos=({:.2f} {:.2f} {:.2f}) n=({:.2f} {:.2f} {:.2f}) u_half={} v_half={}",
                           id, x, y, z, nx, ny, nz, uh, vh);
        });

    // --- Ergonomic variants (#176) ----------------------------------------
    // Industry-standard authoring sugar over the canonical `light_*`
    // primitives above. All variants compute the same W/sr (point/spot)
    // or W/m^2/sr (sphere/quad) value as the low-level form and populate
    // the same `AnalyticLight` struct, so internal math is unchanged.
    // The cfg-save path (`light_list`-derived SaveArchivedCvars) writes
    // the canonical W/sr form, so round-trip is preserved -- only the
    // INPUT parser converts. See the photometric-helpers block above
    // for the 1cd = 1/683.002 W/sr (555nm peak) reasoning and the
    // single-wavelength approximation caveat.

    // ---- light_point variants --------------------------------------------
    C.RegisterCommand("light_point_color",
        "light_point_color <id> <x> <y> <z> <r> <g> <b> <intensity_wsr>: "
        "omnidirectional point light authored as color (RGB chromaticity, "
        "typically in [0,1]) times scalar intensity in W/sr. Final emission "
        "per channel = color * intensity, written into the same W/sr storage "
        "the canonical `light_point` uses.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 8) {
                out.PrintLine("usage: light_point_color <id> <x> <y> <z> <r> <g> <b> <intensity_wsr>");
                return;
            }
            std::uint32_t id;
            float x, y, z, r, g, b, intens;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], r) || !ParseFloat(args[5], g) || !ParseFloat(args[6], b) ||
                !ParseFloat(args[7], intens)) {
                out.PrintLine("light_point_color: arg parse failed");
                return;
            }
            AnalyticLight L{};
            L.type = AnalyticLight::Point;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.intensity[0] = r * intens;
            L.intensity[1] = g * intens;
            L.intensity[2] = b * intens;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: point id={} pos=({:.2f} {:.2f} {:.2f}) intensity*color=({:.3f} {:.3f} {:.3f}) W/sr",
                           id, x, y, z, L.intensity[0], L.intensity[1], L.intensity[2]);
        });

    C.RegisterCommand("light_point_cd",
        "light_point_cd <id> <x> <y> <z> <r> <g> <b> <intensity_cd>: "
        "omnidirectional point light. Intensity in candela (1 cd = 1 lm/sr). "
        "Internally converted via 1 cd = 1/683.002 W/sr (555nm peak photopic, "
        "single-wavelength approximation). Final stored value: "
        "color * (intensity_cd / 683.002) W/sr per channel.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 8) {
                out.PrintLine("usage: light_point_cd <id> <x> <y> <z> <r> <g> <b> <intensity_cd>");
                return;
            }
            std::uint32_t id;
            float x, y, z, r, g, b, cd;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], r) || !ParseFloat(args[5], g) || !ParseFloat(args[6], b) ||
                !ParseFloat(args[7], cd)) {
                out.PrintLine("light_point_cd: arg parse failed");
                return;
            }
            const float wsr = cd * kWsrPerCandela;
            AnalyticLight L{};
            L.type = AnalyticLight::Point;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.intensity[0] = r * wsr;
            L.intensity[1] = g * wsr;
            L.intensity[2] = b * wsr;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: point id={} pos=({:.2f} {:.2f} {:.2f}) {:.2f} cd -> color*({:.6f}) W/sr",
                           id, x, y, z, cd, wsr);
        });

    C.RegisterCommand("light_point_lm",
        "light_point_lm <id> <x> <y> <z> <r> <g> <b> <total_lumens>: "
        "omnidirectional point light. Intensity in TOTAL luminous flux (lumens). "
        "For omnidirectional emission Phi = 4*pi*I so I_cd = Phi / (4*pi); "
        "internally converted via 1 lm omni = 1 / (683.002 * 4*pi) W/sr "
        "(555nm peak photopic approximation). Final stored value: "
        "color * (total_lumens / (683.002 * 4*pi)) W/sr per channel.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 8) {
                out.PrintLine("usage: light_point_lm <id> <x> <y> <z> <r> <g> <b> <total_lumens>");
                return;
            }
            std::uint32_t id;
            float x, y, z, r, g, b, lm;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], r) || !ParseFloat(args[5], g) || !ParseFloat(args[6], b) ||
                !ParseFloat(args[7], lm)) {
                out.PrintLine("light_point_lm: arg parse failed");
                return;
            }
            const float wsr = lm * kWsrPerLumenOmni;
            AnalyticLight L{};
            L.type = AnalyticLight::Point;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.intensity[0] = r * wsr;
            L.intensity[1] = g * wsr;
            L.intensity[2] = b * wsr;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: point id={} pos=({:.2f} {:.2f} {:.2f}) {:.2f} lm -> color*({:.6f}) W/sr",
                           id, x, y, z, lm, wsr);
        });

    C.RegisterCommand("light_point_exposed",
        "light_point_exposed <id> <x> <y> <z> <r> <g> <b> <intensity_wsr> <ev>: "
        "omnidirectional point light with explicit exposure stop. Final stored "
        "value per channel = color * intensity_wsr * 2^ev. Each EV stop doubles "
        "brightness (camera EV convention); ev=0 is identity, ev=+1 is 2x, "
        "ev=-1 is 0.5x. Composes with the global r_exposure tonemap (they "
        "multiply: per-light EV scales emission, r_exposure scales display).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: light_point_exposed <id> <x> <y> <z> <r> <g> <b> <intensity_wsr> <ev>");
                return;
            }
            std::uint32_t id;
            float x, y, z, r, g, b, intens, ev;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], r) || !ParseFloat(args[5], g) || !ParseFloat(args[6], b) ||
                !ParseFloat(args[7], intens) || !ParseFloat(args[8], ev)) {
                out.PrintLine("light_point_exposed: arg parse failed");
                return;
            }
            const float scale = intens * Exposure2x(ev);
            AnalyticLight L{};
            L.type = AnalyticLight::Point;
            L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
            L.intensity[0] = r * scale;
            L.intensity[1] = g * scale;
            L.intensity[2] = b * scale;
            light_prims_[id] = L;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: point id={} pos=({:.2f} {:.2f} {:.2f}) intensity={:.3f} W/sr ev={:+.2f} -> color*({:.6f}) W/sr",
                           id, x, y, z, intens, ev, scale);
        });

    // ---- light_spot variants ---------------------------------------------
    // The spot helper closure pulls the shared parse + dir-normalise +
    // cone-clamp logic out of every variant. unit_to_wsr is the
    // per-variant conversion factor applied to the scalar input before
    // multiplying by color.
    auto spotAddCommon = [this](std::uint32_t id,
                                float x, float y, float z,
                                float dx, float dy, float dz,
                                float outer_deg, float inner_deg,
                                float r, float g, float b,
                                float wsr,
                                pt::console::Output& out,
                                const char* tag) {
        const float dlen = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dlen < 1e-6f) { out.FormatLine("{}: dir magnitude is zero", tag); return; }
        dx /= dlen; dy /= dlen; dz /= dlen;
        if (outer_deg < 0.0f) outer_deg = 0.0f;
        if (outer_deg > 89.99f) outer_deg = 89.99f;
        if (inner_deg < 0.0f) inner_deg = 0.0f;
        if (inner_deg > outer_deg) inner_deg = outer_deg;
        AnalyticLight L{};
        L.type = AnalyticLight::Spot;
        L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
        L.dir[0] = dx; L.dir[1] = dy; L.dir[2] = dz;
        L.intensity[0] = r * wsr;
        L.intensity[1] = g * wsr;
        L.intensity[2] = b * wsr;
        const float deg2rad = 3.14159265358979323846f / 180.0f;
        L.cos_outer = std::cos(outer_deg * deg2rad);
        L.cos_inner = std::cos(inner_deg * deg2rad);
        light_prims_[id] = L;
        light_prims_dirty_ = true;
        accum_dirty_       = true;
        BroadcastSceneDirty();
        out.FormatLine("lights: spot id={} pos=({:.2f} {:.2f} {:.2f}) dir=({:.2f} {:.2f} {:.2f}) outer={:.1f} inner={:.1f} intensity*color/sr=({:.3f} {:.3f} {:.3f})",
                       id, x, y, z, dx, dy, dz, outer_deg, inner_deg,
                       L.intensity[0], L.intensity[1], L.intensity[2]);
    };

    C.RegisterCommand("light_spot_color",
        "light_spot_color <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <intensity_wsr>: "
        "spot light authored as color (RGB chromaticity) times scalar W/sr. "
        "Cone falloff identical to canonical `light_spot`.",
        [spotAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 13) {
                out.PrintLine("usage: light_spot_color <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <intensity_wsr>");
                return;
            }
            std::uint32_t id;
            float x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b, intens;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], dx) || !ParseFloat(args[5], dy) || !ParseFloat(args[6], dz) ||
                !ParseFloat(args[7], outer_deg) || !ParseFloat(args[8], inner_deg) ||
                !ParseFloat(args[9], r)  || !ParseFloat(args[10], g) || !ParseFloat(args[11], b) ||
                !ParseFloat(args[12], intens)) {
                out.PrintLine("light_spot_color: arg parse failed");
                return;
            }
            spotAddCommon(id, x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b, intens, out, "light_spot_color");
        });

    C.RegisterCommand("light_spot_cd",
        "light_spot_cd <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <intensity_cd>: "
        "spot light in candela (1 cd = 1/683.002 W/sr at 555nm peak). On-axis "
        "intensity equals the cd value; cone falloff scales it down between "
        "outer and inner half-angles.",
        [spotAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 13) {
                out.PrintLine("usage: light_spot_cd <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <intensity_cd>");
                return;
            }
            std::uint32_t id;
            float x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b, cd;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], dx) || !ParseFloat(args[5], dy) || !ParseFloat(args[6], dz) ||
                !ParseFloat(args[7], outer_deg) || !ParseFloat(args[8], inner_deg) ||
                !ParseFloat(args[9], r)  || !ParseFloat(args[10], g) || !ParseFloat(args[11], b) ||
                !ParseFloat(args[12], cd)) {
                out.PrintLine("light_spot_cd: arg parse failed");
                return;
            }
            spotAddCommon(id, x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b,
                          cd * kWsrPerCandela, out, "light_spot_cd");
        });

    C.RegisterCommand("light_spot_lm",
        "light_spot_lm <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <total_lumens>: "
        "spot light authored as total emitted luminous flux (lumens) over the "
        "cone. Uses the omnidirectional 4*pi conversion (1 lm = 1/(683.002*4*pi) W/sr) "
        "as a single-wavelength approximation -- this UNDER-states the per-steradian "
        "intensity for narrow cones (treating the lm as if it spread over the full "
        "sphere). For tighter calibration use `light_spot_cd` with the catalog "
        "on-axis candela value instead.",
        [spotAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 13) {
                out.PrintLine("usage: light_spot_lm <id> <x> <y> <z> <dx> <dy> <dz> <outer_deg> <inner_deg> <r> <g> <b> <total_lumens>");
                return;
            }
            std::uint32_t id;
            float x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b, lm;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], dx) || !ParseFloat(args[5], dy) || !ParseFloat(args[6], dz) ||
                !ParseFloat(args[7], outer_deg) || !ParseFloat(args[8], inner_deg) ||
                !ParseFloat(args[9], r)  || !ParseFloat(args[10], g) || !ParseFloat(args[11], b) ||
                !ParseFloat(args[12], lm)) {
                out.PrintLine("light_spot_lm: arg parse failed");
                return;
            }
            spotAddCommon(id, x, y, z, dx, dy, dz, outer_deg, inner_deg, r, g, b,
                          lm * kWsrPerLumenOmni, out, "light_spot_lm");
        });

    // ---- light_sphere variants -------------------------------------------
    // Sphere area light authoring helper. unit_to_radiance is the
    // per-variant conversion factor that turns the scalar input into
    // W/m^2/sr radiance.
    auto sphereAddCommon = [this](std::uint32_t id,
                                  float x, float y, float z,
                                  float radius,
                                  float r, float g, float b,
                                  float radiance,
                                  pt::console::Output& out,
                                  const char* tag) {
        if (radius <= 0.0f) { out.FormatLine("{}: radius must be > 0", tag); return; }
        AnalyticLight L{};
        L.type = AnalyticLight::Sphere;
        L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
        L.radius = radius;
        L.intensity[0] = r * radiance;
        L.intensity[1] = g * radiance;
        L.intensity[2] = b * radiance;
        light_prims_[id] = L;
        light_prims_dirty_ = true;
        accum_dirty_       = true;
        BroadcastSceneDirty();
        out.FormatLine("lights: sphere id={} pos=({:.2f} {:.2f} {:.2f}) r={:.3f} radiance*color=({:.3f} {:.3f} {:.3f}) W/m^2/sr",
                       id, x, y, z, radius,
                       L.intensity[0], L.intensity[1], L.intensity[2]);
    };

    C.RegisterCommand("light_sphere_color",
        "light_sphere_color <id> <x> <y> <z> <radius> <r> <g> <b> <radiance_wm2sr>: "
        "spherical diffuse area light authored as color (RGB chromaticity) times "
        "scalar surface radiance in W/m^2/sr. Same physics as canonical "
        "`light_sphere`, just decomposed.",
        [sphereAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: light_sphere_color <id> <x> <y> <z> <radius> <r> <g> <b> <radiance_wm2sr>");
                return;
            }
            std::uint32_t id;
            float x, y, z, radius, r, g, b, rad;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], radius) ||
                !ParseFloat(args[5], r) || !ParseFloat(args[6], g) || !ParseFloat(args[7], b) ||
                !ParseFloat(args[8], rad)) {
                out.PrintLine("light_sphere_color: arg parse failed");
                return;
            }
            sphereAddCommon(id, x, y, z, radius, r, g, b, rad, out, "light_sphere_color");
        });

    C.RegisterCommand("light_sphere_nits",
        "light_sphere_nits <id> <x> <y> <z> <radius> <r> <g> <b> <luminance_nits>: "
        "spherical diffuse area light. Luminance in nits (1 nit = 1 cd/m^2). "
        "Converted to surface radiance via 1 nit = 1/683.002 W/m^2/sr (555nm "
        "peak photopic, single-wavelength approximation).",
        [sphereAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: light_sphere_nits <id> <x> <y> <z> <radius> <r> <g> <b> <luminance_nits>");
                return;
            }
            std::uint32_t id;
            float x, y, z, radius, r, g, b, nits;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], radius) ||
                !ParseFloat(args[5], r) || !ParseFloat(args[6], g) || !ParseFloat(args[7], b) ||
                !ParseFloat(args[8], nits)) {
                out.PrintLine("light_sphere_nits: arg parse failed");
                return;
            }
            sphereAddCommon(id, x, y, z, radius, r, g, b,
                            nits * kWm2srPerNit, out, "light_sphere_nits");
        });

    // ---- light_quad variants ---------------------------------------------
    // Quad area light authoring helper. unit_to_radiance is the
    // per-variant conversion factor that turns the scalar input into
    // W/m^2/sr radiance.
    auto quadAddCommon = [this](std::uint32_t id,
                                float x, float y, float z,
                                float nx, float ny, float nz,
                                float ux, float uy, float uz,
                                float uh, float vh,
                                float r, float g, float b,
                                float radiance,
                                pt::console::Output& out,
                                const char* tag) {
        const float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (nlen < 1e-6f) { out.FormatLine("{}: normal magnitude is zero", tag); return; }
        nx /= nlen; ny /= nlen; nz /= nlen;
        const float ndu = nx*ux + ny*uy + nz*uz;
        float uxo = ux - nx*ndu;
        float uyo = uy - ny*ndu;
        float uzo = uz - nz*ndu;
        const float ulen = std::sqrt(uxo*uxo + uyo*uyo + uzo*uzo);
        if (ulen < 1e-6f) { out.FormatLine("{}: u-axis collinear with normal", tag); return; }
        uxo /= ulen; uyo /= ulen; uzo /= ulen;
        if (uh <= 0.0f || vh <= 0.0f) {
            out.FormatLine("{}: u_half and v_half must be > 0", tag);
            return;
        }
        AnalyticLight L{};
        L.type = AnalyticLight::Quad;
        L.pos[0] = x; L.pos[1] = y; L.pos[2] = z;
        L.dir[0] = nx; L.dir[1] = ny; L.dir[2] = nz;
        L.u_vec[0] = uxo * uh; L.u_vec[1] = uyo * uh; L.u_vec[2] = uzo * uh;
        L.v_half   = vh;
        L.intensity[0] = r * radiance;
        L.intensity[1] = g * radiance;
        L.intensity[2] = b * radiance;
        light_prims_[id] = L;
        light_prims_dirty_ = true;
        accum_dirty_       = true;
        BroadcastSceneDirty();
        out.FormatLine("lights: quad id={} pos=({:.2f} {:.2f} {:.2f}) n=({:.2f} {:.2f} {:.2f}) u_half={} v_half={} radiance*color=({:.3f} {:.3f} {:.3f}) W/m^2/sr",
                       id, x, y, z, nx, ny, nz, uh, vh,
                       L.intensity[0], L.intensity[1], L.intensity[2]);
    };

    C.RegisterCommand("light_quad_color",
        "light_quad_color <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <radiance_wm2sr>: "
        "rectangular diffuse one-sided area light authored as color (RGB chromaticity) "
        "times scalar surface radiance in W/m^2/sr. Same physics as canonical "
        "`light_quad`, just decomposed.",
        [quadAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 16) {
                out.PrintLine("usage: light_quad_color <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <radiance_wm2sr>");
                return;
            }
            std::uint32_t id;
            float x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b, rad;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], nx) || !ParseFloat(args[5], ny) || !ParseFloat(args[6], nz) ||
                !ParseFloat(args[7], ux) || !ParseFloat(args[8], uy) || !ParseFloat(args[9], uz) ||
                !ParseFloat(args[10], uh) || !ParseFloat(args[11], vh) ||
                !ParseFloat(args[12], r) || !ParseFloat(args[13], g) || !ParseFloat(args[14], b) ||
                !ParseFloat(args[15], rad)) {
                out.PrintLine("light_quad_color: arg parse failed");
                return;
            }
            quadAddCommon(id, x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b,
                          rad, out, "light_quad_color");
        });

    C.RegisterCommand("light_quad_nits",
        "light_quad_nits <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <luminance_nits>: "
        "rectangular diffuse one-sided area light. Luminance in nits (1 nit = 1 cd/m^2). "
        "Converted to surface radiance via 1 nit = 1/683.002 W/m^2/sr (555nm peak "
        "photopic, single-wavelength approximation). Useful for matching display "
        "spec sheets directly (e.g. \"500 nit OLED panel\").",
        [quadAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 16) {
                out.PrintLine("usage: light_quad_nits <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <luminance_nits>");
                return;
            }
            std::uint32_t id;
            float x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b, nits;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], nx) || !ParseFloat(args[5], ny) || !ParseFloat(args[6], nz) ||
                !ParseFloat(args[7], ux) || !ParseFloat(args[8], uy) || !ParseFloat(args[9], uz) ||
                !ParseFloat(args[10], uh) || !ParseFloat(args[11], vh) ||
                !ParseFloat(args[12], r) || !ParseFloat(args[13], g) || !ParseFloat(args[14], b) ||
                !ParseFloat(args[15], nits)) {
                out.PrintLine("light_quad_nits: arg parse failed");
                return;
            }
            quadAddCommon(id, x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b,
                          nits * kWm2srPerNit, out, "light_quad_nits");
        });

    C.RegisterCommand("light_quad_exposed",
        "light_quad_exposed <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <radiance_wm2sr> <ev>: "
        "rectangular diffuse one-sided area light with explicit exposure stop. "
        "Final stored radiance per channel = color * radiance_wm2sr * 2^ev "
        "(camera EV convention: each stop doubles brightness).",
        [quadAddCommon](auto args, pt::console::Output& out) {
            if (args.size() != 17) {
                out.PrintLine("usage: light_quad_exposed <id> <x> <y> <z> <nx> <ny> <nz> <ux> <uy> <uz> <u_half> <v_half> <r> <g> <b> <radiance_wm2sr> <ev>");
                return;
            }
            std::uint32_t id;
            float x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b, rad, ev;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x)  || !ParseFloat(args[2], y)  || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], nx) || !ParseFloat(args[5], ny) || !ParseFloat(args[6], nz) ||
                !ParseFloat(args[7], ux) || !ParseFloat(args[8], uy) || !ParseFloat(args[9], uz) ||
                !ParseFloat(args[10], uh) || !ParseFloat(args[11], vh) ||
                !ParseFloat(args[12], r) || !ParseFloat(args[13], g) || !ParseFloat(args[14], b) ||
                !ParseFloat(args[15], rad) || !ParseFloat(args[16], ev)) {
                out.PrintLine("light_quad_exposed: arg parse failed");
                return;
            }
            quadAddCommon(id, x, y, z, nx, ny, nz, ux, uy, uz, uh, vh, r, g, b,
                          rad * Exposure2x(ev), out, "light_quad_exposed");
        });
    // --- end #176 ergonomic variants --------------------------------------

    // --- Inspector setter family (wave-6 #25 follow-up) -------------------
    // Atomic per-field setters so the property inspector dispatches one
    // command per field commit instead of re-emitting the full composite
    // light_point / light_spot / light_sphere / light_quad constructor
    // every time a user nudges a single axis. Each setter validates id +
    // type, mutates in place, marks dirty, and broadcasts scene_dirty so
    // editor panels re-fetch via list_scene. Mirrors the prim_set_* family
    // shape introduced by agent-19 + wave-5 polish (8cd8164).

    C.RegisterCommand("light_set_pos",
        "light_set_pos <id> <x> <y> <z>: move any analytic light (point/spot/sphere/quad).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) { out.PrintLine("usage: light_set_pos <id> <x> <y> <z>"); return; }
            std::uint32_t id; float x, y, z;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z)) {
                out.PrintLine("light_set_pos: arg parse failed"); return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) { out.FormatLine("light_set_pos: id {} not found", id); return; }
            it->second.pos[0] = x; it->second.pos[1] = y; it->second.pos[2] = z;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: id {} pos -> ({:.2f} {:.2f} {:.2f})", id, x, y, z);
        });

    C.RegisterCommand("light_set_intensity",
        "light_set_intensity <id> <r> <g> <b>: set raw intensity per channel "
        "(W/sr for point/spot, W/m^2/sr for sphere/quad). Bypasses the "
        "photometric cd/lm/nits conversion -- writes the engine-internal "
        "value directly.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) { out.PrintLine("usage: light_set_intensity <id> <r> <g> <b>"); return; }
            std::uint32_t id; float r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], r) || !ParseFloat(args[2], g) || !ParseFloat(args[3], b)) {
                out.PrintLine("light_set_intensity: arg parse failed"); return;
            }
            if (r < 0.0f || g < 0.0f || b < 0.0f) {
                out.PrintLine("light_set_intensity: components must be >= 0"); return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) { out.FormatLine("light_set_intensity: id {} not found", id); return; }
            it->second.intensity[0] = r; it->second.intensity[1] = g; it->second.intensity[2] = b;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: id {} intensity -> ({:.3f} {:.3f} {:.3f})", id, r, g, b);
        });

    C.RegisterCommand("light_set_dir",
        "light_set_dir <id> <nx> <ny> <nz>: set spot-light axis OR quad "
        "normal. Input is auto-normalised. No-op (warns) for point/sphere "
        "lights which have no direction.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) { out.PrintLine("usage: light_set_dir <id> <nx> <ny> <nz>"); return; }
            std::uint32_t id; float nx, ny, nz;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], nx) || !ParseFloat(args[2], ny) || !ParseFloat(args[3], nz)) {
                out.PrintLine("light_set_dir: arg parse failed"); return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) { out.FormatLine("light_set_dir: id {} not found", id); return; }
            if (it->second.type != AnalyticLight::Spot && it->second.type != AnalyticLight::Quad) {
                out.FormatLine("light_set_dir: id {} is type {} -- no direction to set",
                               id, LightTypeName(it->second.type));
                return;
            }
            const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len < 1e-6f) { out.PrintLine("light_set_dir: zero-length direction rejected"); return; }
            it->second.dir[0] = nx / len; it->second.dir[1] = ny / len; it->second.dir[2] = nz / len;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: id {} dir -> ({:.3f} {:.3f} {:.3f})",
                           id, it->second.dir[0], it->second.dir[1], it->second.dir[2]);
        });

    C.RegisterCommand("light_set_cone",
        "light_set_cone <id> <inner_deg> <outer_deg>: set spot-light half-angles "
        "(inner <= outer; both 0..90). No-op (warns) for non-spot types.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 3) { out.PrintLine("usage: light_set_cone <id> <inner_deg> <outer_deg>"); return; }
            std::uint32_t id; float inner_deg, outer_deg;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], inner_deg) || !ParseFloat(args[2], outer_deg)) {
                out.PrintLine("light_set_cone: arg parse failed"); return;
            }
            if (inner_deg < 0.0f || outer_deg > 90.0f || inner_deg > outer_deg) {
                out.PrintLine("light_set_cone: require 0 <= inner <= outer <= 90 (degrees)"); return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) { out.FormatLine("light_set_cone: id {} not found", id); return; }
            if (it->second.type != AnalyticLight::Spot) {
                out.FormatLine("light_set_cone: id {} is not a spot light", id); return;
            }
            const float deg2rad = 3.14159265358979323846f / 180.0f;
            it->second.cos_inner = std::cos(inner_deg * deg2rad);
            it->second.cos_outer = std::cos(outer_deg * deg2rad);
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: id {} cone -> inner={:.2f}deg outer={:.2f}deg", id, inner_deg, outer_deg);
        });

    C.RegisterCommand("light_set_size",
        "light_set_size <id> <r>: set sphere-light radius OR quad-light v_half "
        "(for quad, use sister command light_set_uhalf for the U axis). "
        "No-op (warns) for point/spot types.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 2) { out.PrintLine("usage: light_set_size <id> <r>"); return; }
            std::uint32_t id; float r;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], r)) {
                out.PrintLine("light_set_size: arg parse failed"); return;
            }
            if (r < 0.0f) { out.PrintLine("light_set_size: size must be >= 0"); return; }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) { out.FormatLine("light_set_size: id {} not found", id); return; }
            if (it->second.type == AnalyticLight::Sphere) {
                it->second.radius = r;
            } else if (it->second.type == AnalyticLight::Quad) {
                it->second.v_half = r;
            } else {
                out.FormatLine("light_set_size: id {} is {} -- no size to set",
                               id, LightTypeName(it->second.type));
                return;
            }
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("lights: id {} size -> {:.3f}", id, r);
        });

    // --- Rotation gizmo dispatch (mirrors prim_set_rotation) -----------------
    // Sets a light's orientation quaternion. Input is a raw xyzw tuple
    // which we normalize before storing (a non-unit quat would alter the
    // rotation angle through the shader's quatRotate formula). Point and
    // sphere lights are rotation-symmetric so the orient has no visible
    // effect but we still store it (selection-driven UI observation
    // reads a stable value). Spot rotates its axis at shader time; quad
    // rotates normal + u-extent.
    C.RegisterCommand("light_set_rotation",
        "light_set_rotation <id> <qx> <qy> <qz> <qw>: set the light's "
        "orientation quaternion (xyzw). Input is normalized. Identity is "
        "(0, 0, 0, 1). Point / sphere lights are rotation-symmetric so "
        "this is visually a no-op for them; spot tilts its cone axis, "
        "quad tilts its normal + u-extent in place.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 5) {
                out.PrintLine("usage: light_set_rotation <id> <qx> <qy> <qz> <qw>");
                return;
            }
            std::uint32_t id; float qx, qy, qz, qw;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], qx) || !ParseFloat(args[2], qy) ||
                !ParseFloat(args[3], qz) || !ParseFloat(args[4], qw)) {
                out.PrintLine("light_set_rotation: arg parse failed");
                return;
            }
            const float mag = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
            if (mag < 1e-6f) {
                out.PrintLine("light_set_rotation: zero-magnitude quaternion");
                return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) {
                out.FormatLine("light_set_rotation: id {} not found", id);
                return;
            }
            const float inv = 1.0f / mag;
            it->second.orient[0] = qx * inv;
            it->second.orient[1] = qy * inv;
            it->second.orient[2] = qz * inv;
            it->second.orient[3] = qw * inv;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("light_set_rotation: id {} -> ({:.4f} {:.4f} {:.4f} {:.4f})",
                           id, it->second.orient[0], it->second.orient[1],
                           it->second.orient[2], it->second.orient[3]);
        });

    C.RegisterCommand("light_set_rotation_euler",
        "light_set_rotation_euler <id> <pitch_deg> <yaw_deg> <roll_deg>: "
        "set the light's orientation from Euler angles (XYZ order, "
        "degrees). Converts internally to the canonical quaternion "
        "stored on the light.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) {
                out.PrintLine("usage: light_set_rotation_euler <id> <pitch_deg> <yaw_deg> <roll_deg>");
                return;
            }
            std::uint32_t id; float pitch_deg, yaw_deg, roll_deg;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], pitch_deg) ||
                !ParseFloat(args[2], yaw_deg)   ||
                !ParseFloat(args[3], roll_deg)) {
                out.PrintLine("light_set_rotation_euler: arg parse failed");
                return;
            }
            auto it = light_prims_.find(id);
            if (it == light_prims_.end()) {
                out.FormatLine("light_set_rotation_euler: id {} not found", id);
                return;
            }
            // XYZ-intrinsic Euler -> quaternion. Half-angle formulation
            // produces a unit quaternion straight out so we don't need
            // the normalisation step. Matches prim_set_rotation_euler
            // verbatim so light + prim rotations compose identically.
            const float kDeg2Rad = 3.14159265358979323846f / 180.0f;
            const float hx = pitch_deg * kDeg2Rad * 0.5f;
            const float hy = yaw_deg   * kDeg2Rad * 0.5f;
            const float hz = roll_deg  * kDeg2Rad * 0.5f;
            const float cx = std::cos(hx), sx = std::sin(hx);
            const float cy = std::cos(hy), sy = std::sin(hy);
            const float cz = std::cos(hz), sz = std::sin(hz);
            // q = qz * qy * qx (roll then yaw then pitch).
            it->second.orient[0] = sx * cy * cz + cx * sy * sz;
            it->second.orient[1] = cx * sy * cz - sx * cy * sz;
            it->second.orient[2] = cx * cy * sz + sx * sy * cz;
            it->second.orient[3] = cx * cy * cz - sx * sy * sz;
            light_prims_dirty_ = true;
            accum_dirty_       = true;
            BroadcastSceneDirty();
            out.FormatLine("light_set_rotation_euler: id {} -> quat ({:.4f} {:.4f} {:.4f} {:.4f})",
                           id, it->second.orient[0], it->second.orient[1],
                           it->second.orient[2], it->second.orient[3]);
        });
    // --- end inspector setter family --------------------------------------
}
// --- end Light primitives --------------------------------------------------

// --- Fluid Phase 1 (#136) -- smoke emitters --------------------------------
// Console commands + GPU upload for the density-injection smoke
// emitter list. Each emitter packs into one float4 of position+radius
// (lane v0), one float4 of velocity+density (lane v1), and one float4
// of tint+falloff (lane v2) -- 3 float4s = 48 B per emitter, mirroring
// the shader-side SmokeEmitterRecord layout in PathTraceCloud.slang.
// The buffer is always sized to kMaxSmokeEmitters so re-uploads don't
// reallocate. EnsureSmokeEmittersUploaded re-uploads every frame so the
// parametric drift (base + velocity * t) doesn't need a dirty flag;
// the cost is one bounded WriteBuffer on a tiny (16-emitter) buffer.

void Engine::EnsureSmokeEmittersUploaded() {
    if (!device_) return;
    constexpr std::uint32_t kFloatsPerEmitter = 12;   // 3 float4s
    constexpr std::uint32_t kBytesPerEmitter  = sizeof(float) * kFloatsPerEmitter;

    // Buffer is always sized for the hard cap so a re-upload doesn't
    // reallocate or stutter the steady-state frame loop.
    if (smoke_buffer_id_ == 0) {
        auto buf = device_->CreateBuffer({
            .size       = std::size_t(kMaxSmokeEmitters) * kBytesPerEmitter,
            .usage      = pt::rhi::BufferUsage::Storage,
            .debug_name = "smoke_emitters",
        });
        if (buf.id == 0) {
            LOG_ERROR("smoke emitter buffer allocation failed");
            return;
        }
        smoke_buffer_id_       = buf.id;
        smoke_buffer_capacity_ = kMaxSmokeEmitters;
    }

    // Honour the runtime cvar cap. ParseInt-style: clamp to [0, hard cap].
    auto& C = pt::console::Console::Get();
    std::uint32_t soft_cap = kMaxSmokeEmitters;
    if (auto* v = C.FindCVar("r_smoke_max_emitters")) {
        int n = v->GetInt();
        if (n < 0) n = 0;
        if (n > static_cast<int>(kMaxSmokeEmitters)) n = static_cast<int>(kMaxSmokeEmitters);
        soft_cap = static_cast<std::uint32_t>(n);
    }
    const std::uint32_t count = std::min<std::uint32_t>(
        soft_cap, static_cast<std::uint32_t>(smoke_emitters_.size()));

    // Pack the full buffer footprint (trailing entries zeroed). The
    // shader's per-frame `smoke_count` push gate is the only thing
    // that bounds iteration; unused records read as zero density so
    // even if the shader misses the bound nothing wrong renders.
    std::vector<float> packed(std::size_t(kMaxSmokeEmitters) * kFloatsPerEmitter, 0.0f);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& E = smoke_emitters_[i];
        const std::size_t off = std::size_t(i) * kFloatsPerEmitter;
        // v0: position + radius
        packed[off + 0]  = E.pos[0];
        packed[off + 1]  = E.pos[1];
        packed[off + 2]  = E.pos[2];
        packed[off + 3]  = E.radius;
        // v1: velocity + density
        packed[off + 4]  = E.velocity[0];
        packed[off + 5]  = E.velocity[1];
        packed[off + 6]  = E.velocity[2];
        packed[off + 7]  = E.density;
        // v2: tint + falloff exponent
        packed[off + 8]  = E.tint[0];
        packed[off + 9]  = E.tint[1];
        packed[off + 10] = E.tint[2];
        packed[off + 11] = E.falloff;
    }
    device_->WriteBuffer(pt::rhi::BufferHandle{smoke_buffer_id_},
                         packed.data(), packed.size() * sizeof(float));
    smoke_count_uploaded_ = count;
}

void Engine::RegisterSmokeCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("smoke_emit",
        "smoke_emit <x> <y> <z> [radius=1.0] [density=1.0] [vx=0] [vy=0.2] [vz=0]: "
        "add a smoke plume to the cloud density field (#136). Radius is the "
        "Gaussian falloff sigma in metres; density is the additive sigma_t peak "
        "(per metre, same units as r_clouds_density); the optional vx/vy/vz "
        "trio is the drift velocity in metres/second (defaults to a gentle "
        "upward 0.2 m/s). Phase 2 (#180) uses this velocity as the chain axis "
        "for the puff stream -- bump vy to 1-3 for a visible rising column. "
        "Cap is kMaxSmokeEmitters = 16 -- further emits are rejected.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() < 3 || args.size() == 6 || args.size() == 7 || args.size() > 8) {
                out.PrintLine("usage: smoke_emit <x> <y> <z> [radius] [density] [vx vy vz]");
                return;
            }
            float x, y, z;
            float radius  = 1.0f;
            float density = 1.0f;
            // Default drift: slight upward bias so plumes look "rising
            // smoke" out of the box. Falloff exponent 2 = Gaussian-ish.
            float vx = 0.0f, vy = 0.2f, vz = 0.0f;
            if (!ParseFloat(args[0], x) || !ParseFloat(args[1], y) ||
                !ParseFloat(args[2], z)) {
                out.PrintLine("smoke_emit: x y z must be numeric");
                return;
            }
            if (args.size() >= 4 && !ParseFloat(args[3], radius)) {
                out.PrintLine("smoke_emit: radius must be numeric");
                return;
            }
            if (args.size() >= 5 && !ParseFloat(args[4], density)) {
                out.PrintLine("smoke_emit: density must be numeric");
                return;
            }
            if (args.size() == 8) {
                if (!ParseFloat(args[5], vx) || !ParseFloat(args[6], vy) ||
                    !ParseFloat(args[7], vz)) {
                    out.PrintLine("smoke_emit: vx vy vz must be numeric");
                    return;
                }
            }
            if (radius <= 0.0f) {
                out.PrintLine("smoke_emit: radius must be > 0");
                return;
            }
            if (density <= 0.0f) {
                out.PrintLine("smoke_emit: density must be > 0");
                return;
            }
            if (smoke_emitters_.size() >= kMaxSmokeEmitters) {
                out.FormatLine("smoke_emit: cap reached ({} emitters); run smoke_clear first",
                               kMaxSmokeEmitters);
                return;
            }
            SmokeEmitter E{};
            E.pos[0] = x; E.pos[1] = y; E.pos[2] = z;
            E.radius = radius;
            E.density = density;
            E.velocity[0] = vx;
            E.velocity[1] = vy;
            E.velocity[2] = vz;
            E.falloff = 2.0f;
            E.tint[0] = E.tint[1] = E.tint[2] = 1.0f;
            const std::uint32_t id = smoke_next_id_++;
            smoke_emitters_.push_back(E);
            accum_dirty_ = true;
            out.FormatLine("smoke: emit id={} at ({:.2f} {:.2f} {:.2f}) radius={:.2f} density={:.2f} vel=({:.2f} {:.2f} {:.2f})",
                           id, x, y, z, radius, density, vx, vy, vz);
        });

    C.RegisterCommand("smoke_clear",
        "Remove all smoke emitters spawned via smoke_emit (#136).",
        [this](auto, pt::console::Output& out) {
            const std::size_t n = smoke_emitters_.size();
            smoke_emitters_.clear();
            accum_dirty_ = true;
            out.FormatLine("smoke: cleared {} emitter(s)", n);
        });

    C.RegisterCommand("smoke_list",
        "List active smoke emitters (#136).",
        [this](auto, pt::console::Output& out) {
            if (smoke_emitters_.empty()) { out.PrintLine("(no smoke emitters)"); return; }
            out.FormatLine("{} smoke emitter(s):", smoke_emitters_.size());
            for (std::size_t i = 0; i < smoke_emitters_.size(); ++i) {
                const auto& E = smoke_emitters_[i];
                out.FormatLine("  [{}] pos=({:.2f} {:.2f} {:.2f}) radius={:.2f} density={:.2f} vel=({:.2f} {:.2f} {:.2f})",
                               i, E.pos[0], E.pos[1], E.pos[2],
                               E.radius, E.density,
                               E.velocity[0], E.velocity[1], E.velocity[2]);
            }
        });
}
// --- end Fluid Phase 1 -----------------------------------------------------

bool Engine::VoxelizeSourceObject(std::uint32_t source_id, float voxel_size,
                                  std::string* out_summary) {
    if (voxel_size <= 0.0f) {
        if (out_summary) *out_summary = "voxelize_object: r_voxel_size must be > 0";
        return false;
    }
    auto grid = std::make_unique<pt::destruction::VoxelGrid>();

    // Resolution order: CSG mesh root -> SDF cluster -> analytic
    // primitive. The first match wins.
    if (csg_scene_ && csg_scene_->HasRoot() && source_id == csg_scene_->RootId()) {
        std::string err;
        pt::csg::BakedMesh baked = csg_scene_->Bake(&err);
        if (baked.Empty()) {
            const std::string msg = fmt::format(
                "voxelize_object: CSG root id={} bake failed ({})",
                source_id, err.empty() ? "empty mesh" : err);
            if (out_summary) *out_summary = msg;
            LOG_ERROR("[voxel] {}", msg);
            return false;
        }
        const auto mat = DefaultMeshVoxelMaterial();
        if (!pt::destruction::VoxelizeBakedMesh(source_id, baked, voxel_size, mat, *grid)) {
            const std::string msg = fmt::format(
                "voxelize_object: CSG root id={} produced 0 voxels", source_id);
            if (out_summary) *out_summary = msg;
            LOG_WARN("[voxel] {}", msg);
            return false;
        }
    } else if (auto it = sdf_prims_.find(source_id); it != sdf_prims_.end()) {
        const auto mat = MaterialFromSdfPrim(it->second);
        if (!pt::destruction::VoxelizeSdf(source_id, it->second, voxel_size, mat, *grid)) {
            const std::string msg = fmt::format(
                "voxelize_object: SDF cluster id={} produced 0 voxels", source_id);
            if (out_summary) *out_summary = msg;
            LOG_WARN("[voxel] {}", msg);
            return false;
        }
    } else if (auto pit = primitives_.find(source_id); pit != primitives_.end()) {
        const auto& p = pit->second;
        if (p.type != AnalyticPrim::Sphere) {
            const std::string msg = fmt::format(
                "voxelize_object: prim id={} is not a finite primitive (plane has infinite extent)",
                source_id);
            if (out_summary) *out_summary = msg;
            LOG_ERROR("[voxel] {}", msg);
            return false;
        }
        const auto mat = MaterialFromAnalyticPrim(p);
        const float center[3] = { p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2] };
        if (!pt::destruction::VoxelizeSphere(source_id, center, p.radius_or_d,
                                             voxel_size, mat, *grid)) {
            const std::string msg = fmt::format(
                "voxelize_object: sphere id={} produced 0 voxels", source_id);
            if (out_summary) *out_summary = msg;
            LOG_WARN("[voxel] {}", msg);
            return false;
        }
    } else {
        const std::string msg = fmt::format(
            "voxelize_object: id={} not found (CSG root, SDF cluster, or analytic prim)",
            source_id);
        if (out_summary) *out_summary = msg;
        LOG_ERROR("[voxel] {}", msg);
        return false;
    }

    // Replace any prior grid for this source. Push to map.
    voxel_grids_[source_id] = std::move(grid);
    // Bump the mutation generation so the next SyncVoxelDemoState
    // observes "voxel content changed" and rebuilds the reserved-id
    // cluster set. Without the bump the sync's idempotency early-out
    // would keep the old (stale) cluster set in sdf_prims_.
    ++voxel_grids_generation_;

    // Force a sync next frame so the live demo state matches the new
    // voxel set.
    accum_dirty_ = true;
    sdf_prims_dirty_ = true;
    if (out_summary) {
        const auto& g = *voxel_grids_[source_id];
        *out_summary = fmt::format(
            "voxelize_object id={} dims={}x{}x{} occupied={} voxel_size={}m",
            source_id, g.DimX(), g.DimY(), g.DimZ(), g.Occupied(), voxel_size);
    }
    return true;
}

void Engine::SyncVoxelDemoState() {
    // Read the cvar each frame so a toggle takes effect immediately.
    bool want_demo = false;
    if (auto* v = pt::console::Console::Get().FindCVar("r_voxelize_demo")) {
        want_demo = v->GetBool();
    }
    voxel_demo_active_ = want_demo && !voxel_grids_.empty();

    // Idempotent early-out: if (a) the demo's applied state matches the
    // current want_demo AND (b) the voxel grids haven't mutated since
    // the last successful apply, the reserved-id clusters in
    // sdf_prims_ already match what we'd compute below. Doing the full
    // erase-and-rebuild every frame (the previous behaviour) burned
    // ~N voxel inserts per frame AND pinned accum_dirty_ on, which
    // prevented the accumulator from ever converging while
    // r_voxelize_demo=1 -- the path tracer would render at 1-spp forever.
    if (want_demo == voxel_demo_applied_state_ &&
        voxel_grids_generation_ == voxel_demo_applied_generation_) {
        return;
    }

    // Compute the desired reserved-id span based on want_demo.
    //
    // When the demo is off (want_demo=0) the desired set is empty:
    // we rip every kVoxelClusterIdBase+* entry out of sdf_prims_.
    // When on, we emit one cluster per occupied voxel across every
    // VoxelGrid in voxel_grids_, in object-id order. Reserved ids are
    // contiguous so the next sync's diff is cheap.
    //
    // The voxel renderer slot is sdf_prims_ -- reusing the existing
    // SDF cluster path means the path tracer's loop already handles
    // them with no shader change. The cost is N sphere-trace loops
    // per ray (one per voxel cluster); for the 1m^3 / 0.1 m acceptance
    // scene that's ~1000 trivial box traces which the GPU handles fine.

    // Step 1: erase any existing voxel-reserved clusters from
    // sdf_prims_.
    bool changed = false;
    for (auto it = sdf_prims_.begin(); it != sdf_prims_.end(); ) {
        if (it->first >= kVoxelClusterIdBase) {
            it = sdf_prims_.erase(it);
            changed = true;
        } else {
            ++it;
        }
    }

    // Step 2: if demo on, repopulate from voxel_grids_.
    if (want_demo) {
        std::uint32_t next_id = kVoxelClusterIdBase;
        for (const auto& [src_id, grid_ptr] : voxel_grids_) {
            if (!grid_ptr || !grid_ptr->Valid()) continue;
            const auto& g = *grid_ptr;
            const float voxel_size = g.VoxelSize();
            const auto& mat = g.Material();
            for (std::uint32_t k = 0; k < g.DimZ(); ++k)
            for (std::uint32_t j = 0; j < g.DimY(); ++j)
            for (std::uint32_t i = 0; i < g.DimX(); ++i) {
                if (!g.Get(i, j, k)) continue;
                const auto c = g.VoxelCenter(i, j, k);
                sdf_prims_[next_id] = VoxelToSdfCluster(c, voxel_size, mat);
                changed = true;
                ++next_id;
                if (next_id == 0u) {  // wraparound guard (practical never)
                    LOG_ERROR("[voxel] reserved cluster id space exhausted");
                    break;
                }
            }
        }
    }

    // Latch the new applied state so the next sync's early-out works.
    // Done regardless of `changed` because the only way to reach here
    // is one of (toggle edge, grid mutation), and both should clear the
    // dirty bits cleanly.
    voxel_demo_applied_state_      = want_demo;
    voxel_demo_applied_generation_ = voxel_grids_generation_;

    if (changed) {
        sdf_prims_dirty_ = true;
        accum_dirty_     = true;
    }
}

void Engine::RegisterVoxelCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("voxelize_list",
        "List active voxel grids.",
        [this](auto, pt::console::Output& out) {
            if (voxel_grids_.empty()) { out.PrintLine("(no voxel grids)"); return; }
            out.FormatLine("{} voxel grid(s):", voxel_grids_.size());
            for (const auto& [id, g] : voxel_grids_) {
                if (!g) continue;
                out.FormatLine("  src={:>3}  dims={:>3}x{:>3}x{:>3}  occupied={:>6}  voxel={:.3f}m  aabb=[{:.2f},{:.2f},{:.2f} .. {:.2f},{:.2f},{:.2f}]",
                               id,
                               g->DimX(), g->DimY(), g->DimZ(),
                               g->Occupied(), g->VoxelSize(),
                               g->Header().aabb_min[0], g->Header().aabb_min[1], g->Header().aabb_min[2],
                               g->Header().aabb_max[0], g->Header().aabb_max[1], g->Header().aabb_max[2]);
            }
        });

    C.RegisterCommand("voxelize_clear",
        "Drop all voxel grids and remove their reserved SDF clusters.",
        [this](auto, pt::console::Output& out) {
            voxel_grids_.clear();
            // Bump the generation so Sync notices the grids went away
            // and rips any stale reserved clusters out of sdf_prims_.
            ++voxel_grids_generation_;
            // Force the next render to remove any reserved-id clusters.
            SyncVoxelDemoState();
            out.PrintLine("voxelize: cleared");
        });

    C.RegisterCommand("voxelize_object",
        "voxelize_object <id>: voxelize a source object (CSG root, SDF cluster, or sphere prim) into a voxel grid at the current r_voxel_size.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: voxelize_object <id>");
                return;
            }
            std::uint32_t id = 0u;
            if (!ParseUint(args[0], id)) {
                out.PrintLine("voxelize_object: id parse failed");
                return;
            }
            float voxel_size = 0.1f;
            if (auto* v = pt::console::Console::Get().FindCVar("r_voxel_size")) {
                voxel_size = v->GetFloat();
            }
            std::string summary;
            const bool ok = VoxelizeSourceObject(id, voxel_size, &summary);
            out.PrintLine(summary);
            if (ok) {
                // Force an immediate sync so the next render reflects
                // the new voxel set without waiting for the cvar mirror.
                SyncVoxelDemoState();
            }
        });

    C.RegisterCommand("voxelize_save",
        "voxelize_save <dir>: write every active voxel grid to <dir>/voxel_cache_*.bin.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) {
                out.PrintLine("usage: voxelize_save <dir>");
                return;
            }
            if (voxel_grids_.empty()) {
                out.PrintLine("voxelize_save: nothing to save (no voxel grids)");
                return;
            }
            const std::string dir{args[0]};
            std::size_t saved = 0u;
            for (const auto& [id, g] : voxel_grids_) {
                if (!g) continue;
                const std::string path = pt::destruction::MakeCachePath(
                    dir, id, g->VoxelSize(), 0ull);
                if (g->SaveToFile(path)) {
                    out.FormatLine("voxelize_save: id={} -> {}", id, path);
                    ++saved;
                } else {
                    out.FormatLine("voxelize_save: id={} write failed", id);
                }
            }
            out.FormatLine("voxelize_save: {} grid(s) written", saved);
        });
}
// --- end Voxel destruction Phase 1 -----------------------------------------

// =============================================================================
// Editor scaffold (agent-20) -- React+Vite multi-panel commands.
// =============================================================================

namespace {

// Helper: pull the active engine HTTP host + port + dev-mode from the
// cvars. Centralised so panel_open / OpenAutoOpenedPanels don't both
// re-implement the lookup.
struct EditorEndpoint {
    std::string host = "localhost";
    int         port = 27960;
    bool        dev_mode = false;
};

EditorEndpoint ReadEditorEndpoint() {
    EditorEndpoint e;
    auto& C = pt::console::Console::Get();
    if (auto* v = C.FindCVar("net_bind_address");
        v != nullptr && !v->value.empty() && v->value != "0.0.0.0") {
        e.host = v->value;
    }
    if (auto* v = C.FindCVar("net_port")) e.port = v->GetInt();
    if (auto* v = C.FindCVar("r_editor_dev_mode")) e.dev_mode = v->GetBool();
    return e;
}

// Split a CSV cvar value into trimmed, non-empty tokens. Whitespace
// around each comma is stripped (so "a , b" parses as ["a","b"]).
std::vector<std::string> SplitCsv(std::string_view s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = i;
        while (j < s.size() && s[j] != ',') ++j;
        std::string_view tok = s.substr(i, j - i);
        // ltrim / rtrim
        std::size_t a = 0, b = tok.size();
        while (a < b && (tok[a] == ' ' || tok[a] == '\t')) ++a;
        while (b > a && (tok[b-1] == ' ' || tok[b-1] == '\t')) --b;
        if (b > a) out.emplace_back(tok.substr(a, b - a));
        i = (j == s.size()) ? j : j + 1;
    }
    return out;
}

}  // namespace

int Engine::SpawnPanelBrowser(const std::string& panel_name,
                              const std::string& url,
                              std::string* out_diag) {
#if defined(__APPLE__)
    // macOS: `open -na 'Google Chrome' --args --app=<url>` spawns a
    // detached Chrome --app window. On chrome-missing, fall back to
    // Edge then Safari (Safari can't do --app mode but at least opens
    // the URL).
    //
    // We need a PID to track for later panel_close. `open` itself
    // exits as soon as it asks LaunchServices to spawn the target,
    // so its PID isn't useful. Use posix_spawn -> /usr/bin/open with
    // a wait-for-the-target hint? No -- the cleanest path on macOS is
    // to run Chrome directly:
    //   `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --app=<url>`
    // because then we own the child PID directly.
    const char* kChromiumPaths[] = {
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        nullptr,
    };
    const char* chosen = nullptr;
    for (int i = 0; kChromiumPaths[i] != nullptr; ++i) {
        if (::access(kChromiumPaths[i], X_OK) == 0) {
            chosen = kChromiumPaths[i];
            break;
        }
    }
    if (chosen != nullptr) {
        pid_t pid = ::fork();
        if (pid == 0) {
            // Child: detach from terminal + replace with the browser.
            // setsid() so the browser survives if the parent dies.
            ::setsid();
            std::string app_arg = "--app=" + url;
            // Pass a per-panel user-data-dir argument so multiple
            // `panel_open` invocations don't fall through to a single
            // existing Chrome window (Chrome refuses to spawn a new
            // PID when an existing instance can satisfy the URL). The
            // dir is ephemeral but persistent across runs so the
            // window remembers its size.
            const char* home_env = std::getenv("HOME");
            std::string udd_arg = "--user-data-dir=" +
                std::string(home_env != nullptr ? home_env : "/tmp") +
                "/.demont_editor/" + panel_name;
            execl(chosen, chosen,
                  app_arg.c_str(),
                  udd_arg.c_str(),
                  "--no-first-run",
                  "--no-default-browser-check",
                  nullptr);
            // execl returned -- failure; bail loud.
            std::_Exit(127);
        }
        if (pid > 0) {
            if (out_diag) {
                *out_diag = fmt::format("spawned {} via {} (pid {})",
                                        panel_name, chosen, pid);
            }
            return static_cast<int>(pid);
        }
        // fork() failed; fall through to the `open` fallback.
    }

    // Final fallback: `open` the URL in the default browser. We can't
    // track the resulting PID -- so panel_close on this panel will
    // print a "no tracked PID" diagnostic. Better than nothing.
    auto cmd = fmt::format("/usr/bin/open '{}' >/dev/null 2>&1", url);
    int rc = std::system(cmd.c_str());
    if (out_diag) {
        if (rc == 0) {
            *out_diag = fmt::format(
                "opened {} in default browser (no Chromium found; close "
                "manually -- panel_close won't track this window)",
                panel_name);
        } else {
            *out_diag = fmt::format(
                "failed to open {} ({}). No Chromium browser detected "
                "and `/usr/bin/open` returned {}.",
                panel_name, url, rc);
        }
    }
    return 0;
#elif defined(_WIN32)
    // Windows: try chrome.exe / msedge.exe in standard install paths.
    // CreateProcessA gives us a PROCESS_INFORMATION with a real PID
    // we can hold onto for panel_close (TerminateProcess on the
    // matching HANDLE).
    const char* kChromiumExes[] = {
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        nullptr,
    };
    const char* chosen = nullptr;
    for (int i = 0; kChromiumExes[i] != nullptr; ++i) {
        DWORD attr = GetFileAttributesA(kChromiumExes[i]);
        if (attr != INVALID_FILE_ATTRIBUTES &&
            !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
            chosen = kChromiumExes[i];
            break;
        }
    }
    if (chosen != nullptr) {
        const char* profile_env = std::getenv("USERPROFILE");
        std::string udd = std::string(profile_env != nullptr ? profile_env : "C:\\Temp")
                          + "\\.demont_editor\\" + panel_name;
        std::string cmdline = fmt::format(
            "\"{}\" --app={} --user-data-dir=\"{}\" --no-first-run --no-default-browser-check",
            chosen, url, udd);
        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        // mutable buffer for CreateProcessA
        std::vector<char> buf(cmdline.begin(), cmdline.end());
        buf.push_back('\0');
        BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE,
                                 DETACHED_PROCESS, nullptr, nullptr, &si, &pi);
        if (ok) {
            CloseHandle(pi.hThread);
            // We keep the process handle on close indirectly by PID,
            // re-opening with OpenProcess in KillPanelPid.
            CloseHandle(pi.hProcess);
            if (out_diag) {
                *out_diag = fmt::format("spawned {} via {} (pid {})",
                                        panel_name, chosen,
                                        static_cast<int>(pi.dwProcessId));
            }
            return static_cast<int>(pi.dwProcessId);
        }
    }
    // Fallback: `start` via cmd.exe.
    auto cmd = fmt::format("cmd /c start \"\" \"{}\"", url);
    std::system(cmd.c_str());
    if (out_diag) {
        *out_diag = fmt::format(
            "opened {} in default browser (Chromium not found; "
            "panel_close won't track this window)", panel_name);
    }
    return 0;
#else
    // Other Unix: try xdg-open. PID tracking not implemented for now.
    auto cmd = fmt::format("xdg-open '{}' >/dev/null 2>&1 &", url);
    std::system(cmd.c_str());
    if (out_diag) {
        *out_diag = fmt::format(
            "opened {} via xdg-open (no PID tracked)", panel_name);
    }
    return 0;
#endif
}

void Engine::KillPanelPid(int pid) {
    if (pid <= 0) return;
#if defined(_WIN32)
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE,
                           static_cast<DWORD>(pid));
    if (h != nullptr) {
        TerminateProcess(h, 0);
        CloseHandle(h);
    }
#else
    // SIGTERM first; Chrome's app-mode windows exit cleanly on this.
    ::kill(static_cast<pid_t>(pid), SIGTERM);
#endif
}

void Engine::OpenAutoOpenedPanels() {
    auto& C = pt::console::Console::Get();
    auto* cv = C.FindCVar("r_editor_panels_autoopen");
    if (cv == nullptr || cv->value.empty()) return;
    auto endpoint = ReadEditorEndpoint();
    auto panels = SplitCsv(cv->value);
    for (const auto& name : panels) {
        if (!pt::editor::IsKnownPanel(name)) {
            LOG_WARN("r_editor_panels_autoopen: unknown panel '{}' (skipped)", name);
            continue;
        }
        if (auto it = open_panels_pids_.find(name);
            it != open_panels_pids_.end() && it->second > 0) {
            // Already tracked open (e.g. user toggled the cvar at
            // runtime and we re-entered here). Skip silently.
            continue;
        }
        auto url = pt::editor::BuildPanelUrl(name, endpoint.host,
                                             endpoint.port,
                                             endpoint.dev_mode);
        std::string diag;
        int pid = SpawnPanelBrowser(name, url, &diag);
        if (pid > 0) open_panels_pids_[name] = pid;
        LOG_INFO("editor: autoopen {} -> {}", name, diag);
    }
}

void Engine::RegisterEditorCommands() {
    auto& C = pt::console::Console::Get();

    // `panel_open <name>` -- spawn the panel in a new Chrome --app
    // window. Defaults to scene-hierarchy if no arg given so the
    // F2 hotkey can just dispatch `panel_open` with the panel name.
    auto* cmd_open = C.RegisterCommand("panel_open",
        "Open an editor panel in a Chrome --app window. Usage: "
        "panel_open <name>. Names: scene-hierarchy, inspector, "
        "asset-browser, toolbar. Use `panels` to list state. "
        "Honors r_editor_dev_mode (Vite dev server URL when 1).",
        [this](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine(
                    "panel_open: name required. Try: panel_open scene-hierarchy");
                return;
            }
            std::string name(args[0]);
            if (!pt::editor::IsKnownPanel(name)) {
                out.FormatLine(
                    "panel_open: unknown panel '{}'. Known: "
                    "scene-hierarchy, inspector, asset-browser, toolbar",
                    name);
                return;
            }
            // If we already have a tracked PID, the user probably
            // wants to re-focus -- but we can't programmatically
            // raise a Chrome --app window from another process. Just
            // spawn another window; Chrome will refuse the duplicate
            // user-data-dir lock and bail. Better behaviour: tell
            // the user to use panel_close first.
            if (auto it = open_panels_pids_.find(name);
                it != open_panels_pids_.end() && it->second > 0) {
                out.FormatLine(
                    "panel_open: {} already open (pid {}). "
                    "Use `panel_close {}` first to spawn a fresh window.",
                    name, it->second, name);
                return;
            }
            auto endpoint = ReadEditorEndpoint();
            auto url = pt::editor::BuildPanelUrl(name, endpoint.host,
                                                 endpoint.port,
                                                 endpoint.dev_mode);
            std::string diag;
            int pid = SpawnPanelBrowser(name, url, &diag);
            if (pid > 0) open_panels_pids_[name] = pid;
            out.PrintLine(diag);
            out.FormatLine("panel_open: url = {}", url);
        });
    if (cmd_open != nullptr) cmd_open->default_args = "scene-hierarchy";

    C.RegisterCommand("panel_close",
        "Close an editor panel previously opened via panel_open. "
        "Usage: panel_close <name>. Only panels with a tracked PID "
        "can be closed -- if the panel was opened in the default "
        "browser (no Chromium found at panel_open time), close the "
        "window manually.",
        [this](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine("panel_close: name required");
                return;
            }
            std::string name(args[0]);
            auto it = open_panels_pids_.find(name);
            if (it == open_panels_pids_.end() || it->second <= 0) {
                out.FormatLine(
                    "panel_close: no tracked PID for '{}' "
                    "(never opened or opened in default browser)", name);
                return;
            }
            KillPanelPid(it->second);
            int pid = it->second;
            open_panels_pids_.erase(it);
            out.FormatLine("panel_close: killed {} (pid {})", name, pid);
        });

    C.RegisterCommand("panel_open_all",
        "Open every known editor panel in Chrome --app windows. "
        "Skips panels already tracked as open.",
        [this](auto /*args*/, pt::console::Output& out) {
            auto endpoint = ReadEditorEndpoint();
            int opened = 0, skipped = 0;
            for (std::size_t i = 0; i < pt::editor::KnownPanelCount(); ++i) {
                std::string name = pt::editor::KnownPanels()[i];
                if (auto it = open_panels_pids_.find(name);
                    it != open_panels_pids_.end() && it->second > 0) {
                    ++skipped;
                    out.FormatLine("  skip {} (pid {})", name, it->second);
                    continue;
                }
                auto url = pt::editor::BuildPanelUrl(name, endpoint.host,
                                                     endpoint.port,
                                                     endpoint.dev_mode);
                std::string diag;
                int pid = SpawnPanelBrowser(name, url, &diag);
                if (pid > 0) open_panels_pids_[name] = pid;
                ++opened;
                out.FormatLine("  open {} -> {}", name, diag);
            }
            out.FormatLine("panel_open_all: {} opened, {} skipped",
                           opened, skipped);
        });

    C.RegisterCommand("panel_close_all",
        "Close every editor panel with a tracked PID.",
        [this](auto /*args*/, pt::console::Output& out) {
            int closed = 0;
            for (auto& [name, pid] : open_panels_pids_) {
                if (pid > 0) {
                    KillPanelPid(pid);
                    out.FormatLine("  close {} (pid {})", name, pid);
                    pid = 0;
                    ++closed;
                }
            }
            open_panels_pids_.clear();
            out.FormatLine("panel_close_all: {} closed", closed);
        });

    C.RegisterCommand("panels",
        "List editor panels and their open/closed state.",
        [this](auto /*args*/, pt::console::Output& out) {
            auto endpoint = ReadEditorEndpoint();
            out.FormatLine("editor panels (engine endpoint http://{}:{}, dev_mode={})",
                           endpoint.host, endpoint.port,
                           endpoint.dev_mode ? "on" : "off");
            for (std::size_t i = 0; i < pt::editor::KnownPanelCount(); ++i) {
                std::string name = pt::editor::KnownPanels()[i];
                auto it = open_panels_pids_.find(name);
                int pid = (it != open_panels_pids_.end()) ? it->second : 0;
                auto url = pt::editor::BuildPanelUrl(name, endpoint.host,
                                                     endpoint.port,
                                                     endpoint.dev_mode);
                if (pid > 0) {
                    out.FormatLine("  {:<18} OPEN (pid {})  {}", name, pid, url);
                } else {
                    out.FormatLine("  {:<18} closed         {}", name, url);
                }
            }
        });
}

}  // namespace pt::engine

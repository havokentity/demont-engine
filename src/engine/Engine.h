// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../app/Window.h"
#include "../core/Jobs/JobSystem.h"
#include "../renderer/AnalyticBvh.h"
#include "../renderer/EditorOverlay.h"
#include "../renderer/TriangleBvh.h"
#include "../rhi/Types.h"
#include "CaptureFormat.h"
#include "LensFlare.h"

#include <glm/glm.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pt::app      { class Window; class ConsoleOverlay; class PerfOverlay; }
namespace pt::audio    { class AudioSystem; }
namespace pt::jobs     { class JobSystem; }
namespace pt::console  { class ConsoleServer; }
namespace pt::rhi      { class Device; struct PipelineHandle; }
namespace pt::renderer { struct Camera; class AsyncLightTreeBuilder; }
namespace pt::csg      { class CsgScene; struct BakedMesh; }
namespace pt::effects  { class ParticleSystem; }
namespace pt::physics  { class PhysicsSystem; }
// Voxel destruction Phase 1 (issue #140). VoxelGrid lives in
// src/destruction/; the engine keeps a small map of voxelized
// objects + a reserved SDF cluster id range used to render them.
namespace pt::destruction { class VoxelGrid; }

namespace pt::engine {

using BackendType = pt::rhi::BackendType;

// Test-only access hook for tests/phys_drop_args_test.cpp. The
// `phys_drop_*` RGB arg shapes added in PR #189 are pinned by
// instantiating an Engine, manually wiring up the physics subsystem
// (without the full Init() path), and driving the registered console
// commands via Console::Execute. The struct is forward-declared here so
// Engine can grant it friend access to its private register-commands
// path + primitives_ map; the definition lives in the test TU and
// never ships in production binaries -- pt_engine doesn't reference it.
struct PhysDropArgsTestAccess;
// Test-only access for tests/cam_bookmarks_test.cpp -- pins the
// cam_save_named / cam_load_named / cam_list_bookmarks /
// cam_delete_bookmark contract end-to-end via Console::Execute.
// Defined in the test TU only; pt_engine never references it.
struct CamBookmarksTestAccess;

// --- Editor backend foundation (agent-19 / editor-mode) -------------------
// Selection target kind. Mirrored on the WebSocket protocol (the
// "selection_change" event's "kind" field is one of the lowercase
// stringified names). Distinguishes which scene-side map a selected id
// refers to: analytic primitives live in Engine::primitives_, lights in
// light_prims_, SDF clusters in sdf_prims_, rigid bodies in the physics
// system. The kind matters because all four namespaces use independent
// 32-bit ids; the same numeric value can refer to four different things.
//
// None is the "no selection" sentinel paired with id == 0 -- the editor
// uses this to clear the highlight when the user clicks empty space.
enum class SelectionKind : std::uint8_t {
    None = 0,
    AnalyticPrim,
    Light,
    SdfCluster,
    RigidBody,
};
// --- end Editor backend foundation ----------------------------------------

class Engine {
public:
    Engine();
    ~Engine();

    // Stores argv pointers for ApplyCommandLineCvarOverrides (called
    // from inside Init() AFTER the cfg load so command-line values
    // win over archived ones). Pass argc/argv unmodified from main();
    // the engine doesn't own or copy them so the caller must keep
    // them alive at least until Init() returns. No-op if Init runs
    // without args being set.
    void SetCommandLineArgs(int argc, char** argv) {
        argc_ = argc;
        argv_ = argv;
    }

    bool Init();
    void Shutdown();

    // Launches the user's default browser at the running web console URL.
    // No-op (with a warning) if the platform launcher fails.
    void OpenWebConsole();

    // Main loop.  Returns when the window is closed or `quit` runs.
    void Run();

    // True if the engine was launched in smoke-test mode
    // (`pt_smoke_frames > 0`) and the Run loop hit a fatal condition
    // that ended it BEFORE the user-frames-rendered budget was met --
    // most importantly, "backend init never produced a usable device_
    // inside the smoke-device timeout". Callers (specifically main())
    // use this to translate the smoke-test outcome into a process exit
    // code: 0 if the budget was met cleanly, non-zero if this returns
    // true. Always false in non-smoke-test runs (no budget set -> no
    // way to fail this check).
    bool SmokeTestFailed() const { return smoke_test_failed_; }

    // Vid_restart-style: drain the current device, destroy it, recreate the
    // window if the new backend requires a different graphics API hint,
    // construct the new device, set the active backend.
    void RequestBackendSwitch(BackendType to);

    // Process one frame's work after Window::PollEvents.
    void Tick(double dt);

    static Engine* Instance();

    // --- Editor backend (agent-19) ----------------------------------------
    // Selection state. Read by the path tracer (for the magenta silhouette
    // outline gated by r_editor_outline), by the WebSocket protocol's
    // selection_change event, and by future gizmo / property-panel agents
    // (20 React, 21 native overlay, 22 3D gizmos). SetSelection is the
    // canonical mutation point: it updates both fields atomically (within
    // the main-thread Drain/Tick pattern -- no cross-thread writes), marks
    // accum_dirty_ so the outline change takes effect on the next frame,
    // and broadcasts a `selection_change` WebSocket event when a
    // ConsoleServer is live so connected clients can sync their UI.
    //
    // Threading: these are mutated only from the main thread (Engine::Tick
    // -> HandleMouseInput, or Console::Drain -> registered console
    // commands). WebSocket worker threads must go through the
    // QueueExecute path or hit them via the public SetSelection from a
    // command handler -- never write directly.
    SelectionKind  GetSelectionKind() const noexcept { return selected_kind_; }
    std::uint32_t  GetSelectionId()   const noexcept { return selected_prim_id_; }
    void SetSelection(SelectionKind kind, std::uint32_t id);
    // Broadcast a scene_dirty WebSocket event so editor clients
    // re-fetch via list_scene. Any console command that mutates
    // primitives_ / light_prims_ / sdf_prims_ / rigid_bodies_ MUST
    // call this after the mutation so editor panels stay in sync.
    // No-op when no ConsoleServer is bound.
    void BroadcastSceneDirty();
    // Read-only accessors live below the AnalyticPrim / AnalyticLight
    // struct definitions (their types are nested in Engine and not
    // visible here yet). See the second editor-backend block further
    // down in this class.
    // --- end Editor backend ------------------------------------------------

public:
    // Analytic primitive description, mirrors the GPU layout (3 float4s per
    // primitive). The engine maintains a map of these and rebuilds the GPU
    // buffer whenever the set changes.
    struct AnalyticPrim {
        enum Type     : std::uint8_t { Sphere = 0, Plane = 1 };
        // Material enum mirrors PathTrace.slang MAT_* constants. Water is
        // the Phase 1 (#134) addition: shaded analytic plane with normal-
        // map waves + Beer's-law absorption + Schlick Fresnel + Snell
        // refraction. New entries MUST be appended at the end to preserve
        // the host->shader integer mapping pack_prim writes into the
        // primitives storage buffer.
        enum Material : std::uint8_t { Lambert = 0, Metal = 1, Dielectric = 2, Water = 3 };
        Type     type      = Sphere;
        Material material  = Lambert;
        float    pos_or_n[3] {0, 0, 0};   // sphere center / plane normal
        float    radius_or_d = 0.5f;      // sphere radius / plane d
        float    albedo[3]   {1, 1, 1};
        float    roughness   = 0.0f;
        float    ior         = 1.5f;
        // Motion blur (#85). Previous frame's sphere center, used by
        // PathTrace.slang to lerp the sphere position at the ray's
        // shutter-time sample. Initialized to pos_or_n so static prims
        // produce a zero-length lerp (no motion → bit-identical to the
        // motion-blur-off path). The engine snapshots curr → prev once
        // per frame in StepPhysics before writing the new curr_pos.
        // Unused for plane primitives (infinite extent never moves).
        float    prev_pos_or_n[3] {0, 0, 0};
        // Per-prim radiant emission (#181 polish). Units: W/sr per channel,
        // consistent with AnalyticLight::intensity for point/spot lights
        // (PR #185 photometric work). Default (0,0,0) -- non-emissive,
        // bit-for-bit identical to the pre-emission HEAD. When non-zero,
        // PathTrace.slang's hit handler adds throughput * emission to
        // radiance BEFORE the BRDF eval, matching the additive-emitter
        // pattern used for emissive sphere/quad area lights.
        float    emission[3] {0, 0, 0};
        // Per-prim orientation (rotation gizmo / #206). Unit quaternion
        // stored as xyzw, identity is (0,0,0,1). Spheres ignore this
        // (rotation-symmetric). Planes rotate their normal at intersect
        // time: effective_normal = quatRotate(orient, v0.xyz). Drives
        // the editor rotate-mode gizmo dispatch (prim_set_rotation).
        float    orient[4] {0, 0, 0, 1};
    };

    // Analytic light primitive (#73). First-class scene light source for
    // direct-lighting NEE, parallel to (but independent of) AnalyticPrim.
    // The engine maintains a map keyed by user id; the set is uploaded
    // to a single storage buffer when dirty (EnsureLightsUploaded).
    //
    // Units mirror real-world emitter physics:
    //   POINT  intensity in W/sr (candela-equivalent, per channel)
    //   SPOT   intensity in W/sr at the axis center; multiplied by an
    //          angular cosine falloff between cos_inner and cos_outer
    //   SPHERE/QUAD  intensity in W/m^2/sr (radiance of a diffuse
    //          area emitter). Apparent power scales with the sampled
    //          area (4*pi*r^2 for sphere, 4*u_half*v_half for quad).
    //
    // Phase 1 sampling is naive uniform single-pick; Phase 2 (#129)
    // wraps a light tree over this list with O(log N) importance
    // selection. The GPU representation packs into 5 float4s per light
    // (kFloatsPerLight below) -- see PathTrace.slang's LightRecord for
    // the lane-by-lane layout.
    struct AnalyticLight {
        enum Type : std::uint8_t { Point = 0, Spot = 1, Sphere = 2, Quad = 3 };
        Type      type        = Point;
        float     pos[3]      {0, 0, 0};
        float     radius      = 0.0f;       // sphere radius (m); 0 for point/spot/quad
        float     intensity[3]{1, 1, 1};    // W/sr (point/spot) or W/m^2/sr (area)
        float     dir[3]      {0,-1, 0};    // spot axis / quad normal (unit vec)
        float     cos_outer   = 0.0f;       // spot only -- outer half-angle cosine
        float     cos_inner   = 0.0f;       // spot only -- inner half-angle cosine
        float     u_vec[3]    {1, 0, 0};    // quad u-half-extent vector (length = u_half)
        float     v_half      = 0.0f;       // quad v-axis half-extent (m)
        // Per-light orientation (rotation gizmo follow-up to #206).
        // Unit quaternion stored as xyzw; identity (0,0,0,1) leaves the
        // light's stored dir / u_vec untouched. Point and sphere lights
        // are rotation-symmetric so their orient is visually inert;
        // spot lights rotate `dir` through `orient` before the cone-
        // angle test; quad lights rotate both `dir` (normal) and
        // `u_vec` before the area sample. Driven by light_set_rotation
        // (mirror of prim_set_rotation) + the editor rotate gizmo.
        float     orient[4]   {0, 0, 0, 1};
    };

    // --- Fluid Phase 1 (#136) -- density-injection smoke emitters ----------
    // Static / parametrically-drifting smoke plumes that ride the existing
    // volumetric cloud march via additive density contributions inside
    // cloud_density_at(). Phase 1 has NO solver state, NO advection, NO
    // grid -- each emitter is a Gaussian density blob whose "current
    // position" is base + velocity * t (parametric drift). The cloud
    // march's existing extinction / scatter / NEE pipeline handles
    // shading; smoke is rim-lit by the sun and ambient-lit by the sky
    // exactly like clouds, for free.
    //
    // Cap of kMaxSmokeEmitters at 16 is plenty for the MVP (chimney +
    // campfire + a few scattered plumes). The runtime cvar
    // r_smoke_max_emitters tightens this further so users can dial back
    // the per-frame cost without recompiling. GPU buffer is always sized
    // to kMaxSmokeEmitters so a re-upload doesn't reallocate.
    struct SmokeEmitter {
        float pos[3]      {0, 1, 0};      // base position in world space (m)
        float radius      = 1.0f;         // Gaussian falloff radius (m, sigma-equivalent)
        float density     = 1.0f;         // peak sigma_t contribution (per metre)
        float velocity[3] {0, 0.2f, 0};   // parametric drift velocity (m/s)
        float falloff     = 2.0f;         // exponent on (r/radius); 2 = Gaussian-ish
        float tint[3]     {1, 1, 1};      // optional color tint (multiplier on cloud RGB)
    };
    static constexpr std::uint32_t kMaxSmokeEmitters = 16;

    // --- Editor backend (agent-19) accessors -----------------------------
    // Read-only views into the engine's owned scene state for
    // pt::editor::SerializeScene. The nested AnalyticPrim /
    // AnalyticLight types aren't visible at the top of this class
    // (they're declared above in this same public block), so the
    // accessors have to live here -- below the struct definitions but
    // still inside the public area. Returning const refs keeps the
    // hot-path GPU writeback fast and lets the serializer iterate
    // without owning storage. Callers must not race a concurrent
    // mutation; in practice the only mutators are RegisterPrimCommands
    // / phys writeback / RegisterLightCommands etc., all of which run
    // on the main thread under Console::Drain.
    const std::map<std::uint32_t, AnalyticPrim>&     Primitives()   const noexcept { return primitives_; }
    const std::map<std::uint32_t, AnalyticLight>&    Lights()       const noexcept { return light_prims_; }
    const std::map<std::uint32_t, pt::renderer::SdfPrim>& SdfPrims() const noexcept { return sdf_prims_; }
    const pt::physics::PhysicsSystem*                Physics()      const noexcept { return physics_.get(); }
    // --- end Editor backend accessors ------------------------------------

private:
    // Test-only access (PR #181 follow-up, see forward-declaration above).
    friend struct ::pt::engine::PhysDropArgsTestAccess;
    // Test-only access for the named-camera-bookmark commands.
    friend struct ::pt::engine::CamBookmarksTestAccess;

    void RegisterCommands();
    void RegisterCsgCommands();
    void RegisterPrimCommands();
    // Editor 3D-transform gizmo console hooks (gizmo_mode, gizmo_select,
    // r_editor_gizmo*, prim_set_radius). Registered AFTER PR #201's
    // existing RegisterEditorCommands so both run on the same Init
    // sweep; the two sets of commands don't collide. Distinct from
    // RegisterEditorCommands() (which agent-20 owns for the React
    // shell's panel_* commands) so the dependency direction stays
    // explicit -- this method lives in Engine.cpp's editor-gizmo
    // block, agent-20's RegisterEditorCommands stays in its block.
    void RegisterEditorGizmoCommands();
    // Per-frame work for the editor gizmo: read selection, build the
    // segment list, run hit-test + drag math from polled input, upload
    // to the GPU buffer. Called from RenderFrame BEFORE the gizmo
    // dispatch so the buffer is populated by the time the kernel
    // binds it. No-op when selected_kind_ != AnalyticPrim.
    void UpdateEditorGizmoFrame();
    // Named camera bookmark commands (cam_save_named / cam_load_named /
    // cam_list_bookmarks / cam_delete_bookmark). Split out from
    // RegisterCommands so the unit test can wire just these (the full
    // RegisterCommands path pulls in audio / RHI / window / etc. via
    // its captured `this`). Idempotent in the same try_emplace sense
    // as the rest of Console::RegisterCommand.
    void RegisterCameraBookmarkCommands();
    // --- Physics Phase 1 (#132) -------------------------------------------
    // Console commands for the Verlet physics layer (`phys_drop`,
    // `phys_clear`, `phys_status`). Cvar registration lives alongside
    // the other PT_CVAR blocks; this just wires the commands. No-op
    // when `phys_enabled = 0`.
    void RegisterPhysicsCommands();
    // Tick the Verlet integrator + collision pass once per frame, then
    // write each particle's curr_pos back into the matching analytic
    // primitive (sphere) in primitives_. Called from Tick BEFORE
    // RenderFrame's EnsurePrimitivesUploaded so the BVH refit picks
    // up the new positions. No-op when `phys_enabled = 0` or no
    // particles exist.
    void StepPhysics(float dt);
    // Issue #181 -- restore cached pre-viz albedos when the user
    // toggles r_phys_debug_visualize from 1 back to 0. Lives outside
    // StepPhysics so the restore runs even when physics is idle
    // (phys_enabled=0, no live bodies, frame dt non-positive); called
    // from Tick BEFORE StepPhysics every frame.
    void MaybeRestorePhysDebugColors();
    // --- end Physics Phase 1 ----------------------------------------------
    // Console commands for the analytic light primitive set (#73):
    // `light_point`, `light_spot`, `light_sphere`, `light_quad`, plus
    // `light_list` / `light_clear` / `light_remove`. Mirrors the
    // prim_*/sdf_* registration style; the populated map is uploaded
    // to GPU on the next render frame via EnsureLightsUploaded.
    void RegisterLightCommands();
    // Console commands for the fluid Phase 1 smoke emitter list (#136):
    // `smoke_emit <x> <y> <z> [radius] [density]` and `smoke_clear`. The
    // list is uploaded to GPU on the next render frame via
    // EnsureSmokeEmittersUploaded.
    void RegisterSmokeCommands();
    // Re-upload the smoke emitter storage buffer from `smoke_emitters_`
    // (#136). Called from RenderFrame every frame so the parametric
    // drift (base + velocity * t) doesn't require a CPU-side dirty
    // flag -- the GPU sees the current sim time via the per-frame
    // smoke_params push. Buffer is always sized to kMaxSmokeEmitters so
    // re-uploads don't reallocate.
    void EnsureSmokeEmittersUploaded();
    // --- SDF Phase 1 (#97) -------------------------------------------------
    // Console commands for the SDF primitive set (`sdf_sphere`,
    // `sdf_box`, `sdf_smin`, ...). Mirrors the prim_*/csg_* registration
    // style; populated map is uploaded to GPU on next render frame via
    // EnsureSdfPrimsUploaded.
    void RegisterSdfCommands();
    // --- end SDF Phase 1 ---------------------------------------------------

    // --- Voxel destruction Phase 1 (#140) ----------------------------------
    // Console commands for the voxelization subsystem (`voxelize_object`,
    // `voxelize_save`, `voxelize_list`, `voxelize_clear`). Mirrors the
    // sdf_*/prim_*/csg_* style; voxelized objects are stored in
    // voxel_grids_ keyed by source object id. The render-time injection
    // into sdf_prims_ happens in SyncVoxelDemoState() once per frame.
    void RegisterVoxelCommands();

    // Voxelize the source object identified by `source_id`. The
    // resolution order is CSG mesh root -> SDF cluster id -> analytic
    // primitive id. The first match wins; subsequent ids are ignored.
    // Returns true on a successful voxelization (occupied count > 0).
    // out_summary, if non-null, receives a one-line human summary for
    // command-output echo. voxel_size in metres (caller supplies the
    // r_voxel_size cvar value).
    bool VoxelizeSourceObject(std::uint32_t source_id, float voxel_size,
                              std::string* out_summary);

    // Mirrors the r_voxelize_demo cvar to the SDF cluster set. When
    // demo=1 and at least one VoxelGrid is present, each occupied
    // voxel is injected into sdf_prims_ under a reserved id (range
    // [kVoxelClusterIdBase, kVoxelClusterIdBase + N)) so the path
    // tracer renders it as a box via the existing sphere-trace path.
    // When demo=0, those clusters are removed. Idempotent -- safe to
    // call every frame; it computes the desired cluster set then
    // diff-applies. Also flips voxel_demo_active_ for the hide-mesh
    // gate in RenderFrame.
    void SyncVoxelDemoState();
    // --- end Voxel destruction Phase 1 -------------------------------------

    // --- Editor scaffold (agent-20) ----------------------------------------
    // Console commands for the React+Vite editor shell:
    //   `panel_open <name>`       spawns Chrome --app at the panel URL
    //   `panel_close <name>`      kills the tracked PID
    //   `panel_open_all`          opens every known panel
    //   `panel_close_all`         closes every tracked panel
    //   `panels`                  prints the known panels + per-panel state
    // The spawned-PID table lives in open_panels_pids_ keyed by panel
    // name. Hotkeys F2/F3/F4/F5 fire panel_open via the existing key
    // handler in Init().
    void RegisterEditorCommands();
    // Internal: spawn the panel in a fresh Chrome --app window. Tries
    // Chrome -> Edge -> default browser fallback (which won't be
    // app-mode but at least opens). Returns the spawned PID (0 on
    // failure or if no Chromium browser was found). out_diag, if
    // non-null, gets a one-line summary for the console echo.
    int  SpawnPanelBrowser(const std::string& panel_name,
                           const std::string& url,
                           std::string* out_diag);
    // Internal: kill a previously-spawned panel PID. Tolerates a
    // missing/exited PID without throwing.
    void KillPanelPid(int pid);
    // Re-read r_editor_panels_autoopen and open every listed panel.
    // Called once at the tail of Init() (after the WebSocket server
    // is up and command registration is done). No-op if the cvar is
    // empty.
    void OpenAutoOpenedPanels();
    // --- end Editor scaffold ----------------------------------------------

    void TearDownDevice();
    void RenderFrame();

    // Win32 only: destroy + recreate the GLFW window (and its HWND)
    // and re-attach the console + perf overlays. Called by
    // RequestBackendSwitch when transitioning vulkan->software with
    // r_software_blit=gdi and r_software_blit_recreate=auto, to
    // escape the DXGI flip-model lockout that poisons an HWND once
    // Vulkan has presented to it. Engine-level state (cvars, console
    // history, CSG scene, camera, primitives) all persist -- none of
    // it is HWND-attached. Returns true on success; on false the
    // engine is unrenderable (window destroyed, both overlays
    // dropped, no recovery path) and the dispatch site treats it as
    // a hard stop: sets current_backend_=None, wants_quit_=true, and
    // returns so the main loop exits cleanly and the user can
    // relaunch. Earlier revisions of this comment said "falls back
    // to warn" -- that's stale, the implementation is hard-quit.
    bool RecreateWindow();

    // Win32 only: spawn a fresh copy of this process via CreateProcessA
    // with the original argv_ and set wants_quit_ so the current main
    // loop exits cleanly. Used by r_software_blit_recreate=prompt when
    // the user clicks Yes. Returns true on successful spawn (caller
    // should bail out of the current operation and let the loop exit);
    // false on CreateProcessA failure (caller falls back to warn).
    bool RestartProcess();

    // Scans argv_ for `--<cvar-name>=<value>` overrides and applies
    // them via Console::SetCVarOverride. Currently recognised:
    //   --net-port=N         -> net_port (HTTP/WebSocket UI)
    //   --net-line-port=N    -> net_line_port (TCP console)
    // Called from Init() after cfg load so CLI args beat archived
    // values. Adding new pass-through args is a 4-line append.
    void ApplyCommandLineCvarOverrides();

    void UpdateCamera(double dt);

    // --- Editor backend (agent-19) ----------------------------------------
    // LMB-press-edge raycast pick. Reads window cursor position, casts a
    // primary ray from the camera through that pixel using the same FOV /
    // basis the path tracer uses, walks `primitives_` for the nearest
    // analytic-prim hit, then calls SetSelection with the result. Mouse
    // is checked with edge-detect (rising-edge only, not held) via
    // lmb_was_down_ so a single press fires exactly one pick. Gated to
    // skip when RMB is held (mouse-look mode) or the console overlay is
    // shown -- both override "viewport click = pick" semantics. Called
    // every frame from Tick before RenderFrame. Cheap: O(prim_count) ray
    // tests on the CPU, ~hundreds of prims handled trivially on M4 Max.
    void HandleMouseInput();

    // Editor-side LMB-press latch for the rising-edge detector. Tracks
    // the previous frame's IsMouseButtonDown(0) so we can distinguish
    // a fresh press from a held-down state. Reset whenever the gate
    // suppresses the pick path (RMB held / console focused / no window)
    // so the user can't accidentally fire a pick by releasing RMB while
    // still holding LMB.
    bool                                        lmb_was_down_ = false;
    // --- end Editor backend ------------------------------------------------

    // Drains any in-flight CSG bake job, keeps the engine's vertex/index
    // buffers + BLAS + TLAS in sync with the current scene, and -- if the
    // scene has been mutated since the last bake -- fires a new bake on a
    // worker thread. Called every frame before pipeline binding.
    void EnsureMeshUpdated();

    // Re-upload the analytic-primitive storage buffer from the in-memory
    // map. Called from RenderFrame whenever primitives_dirty_ is set.
    void EnsurePrimitivesUploaded();

    // Re-upload the analytic-light storage buffer from `light_prims_`
    // (#73). Called from RenderFrame whenever light_prims_dirty_ is
    // set. Grows by powers of two from a small floor (16 lights) and
    // packs each entry into 4 float4s; the path tracer's Lambert NEE
    // block reads up to `light_prims_count_` records per sample
    // (uniform single-pick over the count).
    void EnsureLightsUploaded();

    // --- Light tree (#129) -------------------------------------------------
    // Rebuild + re-upload the hierarchical light tree SSBO whenever
    // the light set has been mutated since the last frame. Builds via
    // pt::renderer::BuildLightTree on the host (CPU; <1ms for 1000
    // lights). The shader gates traversal on push.light_tree_node_count
    // > 0 AND r_light_tree cvar; when either condition is false the
    // shader falls back to #73's naive uniform pick. Sized the GPU
    // buffer in powers of two starting at 32 nodes -- a 16-light scene
    // (smallest LightTree.h leaf-count granularity) produces 31 nodes
    // (2N-1), so 32 is the smallest power-of-two floor that fits without
    // an immediate realloc.
    void EnsureLightTreeUploaded();
    // --- end Light tree ----------------------------------------------------

    // --- SDF Phase 1 (#97) -------------------------------------------------
    // Re-upload the SDF cluster storage buffer from `sdf_prims_`. Called
    // from RenderFrame whenever sdf_prims_dirty_ is set. Re-runs
    // pt::renderer::ComputeSdfAabb on each cluster immediately before
    // packing so the GPU buffer's AABB always matches the current node
    // tree (the sphere-trace is AABB-bounded, so a stale AABB silently
    // misses hits).
    void EnsureSdfPrimsUploaded();
    // --- end SDF Phase 1 ---------------------------------------------------

    // Load (or unload) the env map from disk. Called from the r_env_map
    // cvar's on_change. Sets env_map_tex_id_ and env_map_path_.
    void ReloadEnvMap(const std::string& path);

    // Load BSC5 + rasterise the J2000 starmap GPU texture once. Idempotent
    // (safe to call after a backend switch -- it skips if the texture
    // already exists). Failures are non-fatal: star_map_tex_id_ stays 0
    // and the path tracer's r_show_stars output silently drops to nothing.
    void EnsureStarMapUploaded();
    void EnsureMoonMapUploaded();

    // Lazy pipeline-id (re-)resolution. The Vulkan backend builds its
    // compute pipelines on a worker thread so the window can come up
    // immediately, which means CreateComputePipeline-by-name returns
    // id=0 for any kernel still under construction. We re-resolve
    // every frame until each cached id flips non-zero, then the
    // resolve becomes a single uint compare per pipeline (no mutex,
    // no map lookup).  Idempotent: safe to call from RenderFrame.
    void EnsurePipelineHandles();

    // Predictive pipeline JIT prewarming. Signals every compute kernel
    // the engine knows about to Device::EnsurePipelineWarmed so the
    // active backend can start (or confirm completion of) the pipeline
    // build BEFORE the first RenderFrame Dispatch. Called once at the
    // tail of Init() and once after every RequestBackendSwitch path
    // creates a fresh device. Gated by the r_pipeline_prewarm cvar
    // (default 1). On backends that build pipelines synchronously at
    // device construction (Metal / software) the calls are no-ops by
    // the base-class default; on Vulkan the worker is already in
    // flight, so this is a documented handshake + forward-compat hook
    // for future on-demand pipelines. Idempotent and cheap.
    void PrewarmPipelines();

    // Favourites persistence. LoadFavoritesFromDisk reads
    // `favorites.cfg` (one line per favourite, `#` comments and blank
    // lines ignored) into Console::favorites_. SaveFavoritesToDisk
    // writes the current Console::Favorites() vector back to the same
    // file. Path is resolved relative to CWD, same as demont.cfg.
    // Both are no-ops if Console::Favorites() is unavailable.
    void LoadFavoritesFromDisk();
    void SaveFavoritesToDisk();

    // Console-history persistence. LoadConsoleHistoryFromDisk reads
    // `console_history.txt` (one line per history entry; blank lines
    // and `#` comments ignored) into Console::history_ so the
    // up-arrow scroll-back survives across demont sessions.
    // SaveConsoleHistoryToDisk writes the current Console::History()
    // back, capped to the most recent Console::kMaxHistoryDepth
    // entries. Path is resolved relative to CWD, same as demont.cfg
    // and favorites.cfg. --no-cfg skips the load (consistent with the
    // demont.cfg / autoexec.cfg / favorites.cfg skip).
    void LoadConsoleHistoryFromDisk();
    void SaveConsoleHistoryToDisk();

    // Named-camera-bookmark persistence. Mirrors the favorites.cfg
    // pattern: `camera_bookmarks.cfg` lives next to demont.cfg, one
    // bookmark per line as `<name> <x> <y> <z> <yaw_deg> <pitch_deg>
    // <fov_deg>` (seven whitespace-separated tokens). `#` comments and
    // blank lines are ignored on load. The bookmarks themselves are
    // held in `camera_bookmarks_` (name -> serialized state string,
    // same six-float format the numeric cam_save 1..9 slots use).
    // Loaded at Init(), saved after every mutating command
    // (cam_save_named / cam_delete_bookmark).
    void LoadCameraBookmarksFromDisk();
    void SaveCameraBookmarksToDisk();

    // Replace the current mesh-path resources (vertex/index buffers,
    // BLAS, TLAS) with one built from `baked`. Called from EnsureMesh*
    // on the main thread once a worker bake has completed.
    void RebuildMeshResources(const pt::csg::BakedMesh& baked);

    // Seed CsgScene with the headline drilled-cube scene so first-frame
    // mesh-mode renders something interesting. Idempotent.
    void SeedDefaultCsgScene();

    // Seed the analytic-primitive set with the canonical 3-sphere +
    // ground-plane scene (Lambert red, gold metal, glass dielectric).
    void SeedDefaultPrimitives();

    // Command-line args captured from main() via SetCommandLineArgs.
    // argv_ is a borrowed pointer; argc_ is 0 if SetCommandLineArgs
    // never ran (e.g. unit tests instantiating Engine directly).
    int                                         argc_ = 0;
    char**                                      argv_ = nullptr;

    std::unique_ptr<pt::app::Window>            window_;
    std::unique_ptr<pt::app::ConsoleOverlay>    overlay_;
    std::unique_ptr<pt::app::PerfOverlay>       perf_overlay_;
    // Ring buffer of recent frame_ms samples for the tier-3 sparkline.
    // Sized to comfortably fill the overlay's graph width at any
    // typical viewport DPI; oldest sample evicted on push.
    std::vector<float>                          perf_history_;
    std::size_t                                 perf_history_pos_ = 0;
    std::unique_ptr<pt::jobs::JobSystem>        jobs_;
    std::unique_ptr<pt::console::ConsoleServer> server_;
    // Editor scaffold (agent-20): tracks the OS pids of Chrome --app
    // windows the engine has spawned via panel_open. Indexed by panel
    // name (e.g. "scene-hierarchy"). A non-zero value means the
    // engine successfully forked a process and we own its lifecycle
    // (panel_close kills it). 0 / missing entries mean either the
    // panel has never been opened or the spawned process exited
    // independently. Cleared on engine shutdown (spawned processes
    // are NOT killed -- they outlive the engine on purpose).
    std::map<std::string, int>                  open_panels_pids_;
    std::unique_ptr<pt::rhi::Device>            device_;
    std::unique_ptr<pt::renderer::Camera>       camera_;
    // Named camera bookmarks. Parallel persistence to the numeric
    // cam_slot_1..9 cvars, keyed by user-supplied token (e.g.
    // "overhead", "closeup", "cornell_3q") instead of a 1..9 index.
    // Value is the same serialized "x y z yaw_deg pitch_deg fov_deg"
    // six-float string the slot system uses, so the parse/format
    // helpers in RegisterCommands are shared. Persisted to
    // `camera_bookmarks.cfg` next to demont.cfg / favorites.cfg --
    // loaded at Init, rewritten on every cam_save_named /
    // cam_delete_bookmark. Empty by default; no compile-time entries.
    std::map<std::string, std::string>          camera_bookmarks_;
    // Audio subsystem (issue #80 MVP -- miniaudio-backed 3D playback).
    // Init() opens the default device + voice pool; Shutdown() releases
    // both. Tick(camera_pos, camera_fwd) pushes a listener snapshot
    // into the audio thread for distance attenuation + stereo panning.
    // Ray-traced occlusion / reverb / HRTF (the issue's headline) are
    // deferred to a follow-up that consumes the renderer's TLAS.
    std::unique_ptr<pt::audio::AudioSystem>     audio_system_;
    std::unique_ptr<pt::csg::CsgScene>          csg_scene_;
    std::unique_ptr<pt::csg::BakedMesh>         pending_baked_;
    // --- Physics Phase 1 (#132) -------------------------------------------
    // Owned by the engine; ticked once per frame from Tick() between
    // UpdateCamera and RenderFrame so the analytic primitive buffer's
    // position writeback happens before EnsurePrimitivesUploaded /
    // BVH refit. Allocated in Init(); destroyed in Shutdown(). Active
    // only when the `phys_enabled` cvar is non-zero -- otherwise
    // StepPhysics returns immediately and the engine behaves exactly
    // like a no-physics build.
    std::unique_ptr<pt::physics::PhysicsSystem> physics_;
    // Auto-incrementing id used by `phys_drop` when allocating a new
    // analytic-prim slot for a spawned sphere. Starts well above the
    // user-facing prim_sphere id range (which is human-typed and
    // typically single digits) to keep the namespaces visually
    // distinct.
    std::uint32_t                               physics_next_prim_id_ = 100000u;
    // --- end Physics Phase 1 ----------------------------------------------
    pt::jobs::JobSystem::Handle                 bake_handle_{};
    std::atomic<int>                            bake_phase_{0};   // 0 idle, 1 baking, 2 ready
    // Set on backend switch (RequestBackendSwitch).  TearDownDevice
    // destroys the per-device BLAS/TLAS resources, but csg_scene_'s
    // own Dirty() flag tracks topology changes -- it stays clean
    // across switches if the user didn't mutate the CSG tree.  Without
    // this flag the next EnsureMeshUpdated tick wouldn't kick a bake
    // on the new device, so the CSG mesh would silently disappear
    // (visible only as "where did my drilled cube go?" on a Software
    // -> Metal swap).  Consumed (cleared) by EnsureMeshUpdated when
    // it enqueues a fresh bake.
    bool                                        force_mesh_rebuild_ = false;

    // Analytic primitives (sphere/plane) -- ordered by user id, uploaded
    // to a storage buffer when dirty. Mesh CSG and analytic primitives
    // are independent; the unified renderer takes the closest hit.
    //
    // Upload-time partition: primitives with infinite extent (planes)
    // go to the front of the buffer and are always linearly scanned in
    // the shader. Finite-extent primitives (spheres today, future
    // box/disk/cylinder) follow and are either scanned linearly
    // (sphere count below r_analytic_bvh_threshold) or traversed via
    // the CPU-built BVH in `analytic_bvh_` / `bvh_node_buffer_id_`
    // (count at or above threshold). linear_prim_count_ records the
    // boundary so the shader can dispatch the two paths.
    std::map<std::uint32_t, AnalyticPrim>       primitives_;
    bool                                        primitives_dirty_      = true;
    std::uint64_t                               prim_buffer_id_        = 0;
    std::uint32_t                               prim_buffer_capacity_  = 0;  // primitives that fit
    std::uint32_t                               linear_prim_count_     = 0;
    pt::renderer::AnalyticBvh                   analytic_bvh_;
    std::uint64_t                               bvh_node_buffer_id_       = 0;
    std::uint32_t                               bvh_node_buffer_capacity_ = 0;  // nodes that fit

    // Issue #181 -- per-prim albedo cache for the r_phys_debug_visualize
    // debug aid. Keyed by AnalyticPrim id (the same key as `primitives_`).
    // Populated lazily on first override (so a user who flips the cvar
    // partway through a session captures whatever albedo was current at
    // that moment, including subsequent prim_albedo edits made before
    // the cvar was enabled). Cleared after restoration on the 1 -> 0
    // transition so stale entries don't leak across sessions. The
    // previous-frame cvar value is held in `phys_debug_visualize_prev_`
    // so we can detect the falling edge.
    std::map<std::uint32_t, std::array<float, 3>> phys_debug_color_cache_;
    int                                           phys_debug_visualize_prev_ = 0;

    // --- SDF Phase 1 (#97) -------------------------------------------------
    // Signed-distance-field primitives. Independent of `primitives_` --
    // SDFs are a separate shader path (sphere-tracing inside a tight
    // AABB) and live in their own GPU storage buffer. Per-cluster
    // expression (sphere/box/torus/capsule/rounded box + smooth-CSG
    // ops). Each cluster's AABB is computed by the host via
    // pt::renderer::ComputeSdfAabb at upload time.
    //
    // The map keys are user-supplied ids (mirrors the analytic-prim
    // / CSG-node convention). sdf_prims_dirty_ triggers a re-upload
    // on the next render frame; the GPU buffer grows by powers of two
    // from a small floor (4 clusters) just like primitive_ /
    // bvh_node_buffer_ to keep steady-state edits allocation-free.
    std::map<std::uint32_t, pt::renderer::SdfPrim> sdf_prims_;
    bool                                        sdf_prims_dirty_         = true;
    std::uint64_t                               sdf_cluster_buffer_id_   = 0;
    std::uint32_t                               sdf_cluster_capacity_    = 0;  // clusters that fit
    std::uint32_t                               sdf_cluster_count_       = 0;  // clusters last uploaded
    // --- end SDF Phase 1 ---------------------------------------------------

    // Analytic light primitives (#73) -- point / spot / sphere area /
    // quad area. Independent of `primitives_`: a separate storage
    // buffer feeds PathTrace.slang's Lambert NEE-to-lights block. The
    // map is keyed by user id (same convention as analytic / SDF /
    // CSG); light_prims_dirty_ schedules a re-upload on the next
    // render frame. Phase 1 caps the active set at a soft 256 lights
    // -- naive uniform single-pick variance dominates beyond that and
    // the light tree (#129) is the answer.
    std::map<std::uint32_t, AnalyticLight>      light_prims_;
    bool                                        light_prims_dirty_       = true;
    std::uint64_t                               light_buffer_id_         = 0;
    std::uint32_t                               light_buffer_capacity_   = 0;  // lights that fit
    std::uint32_t                               light_count_uploaded_    = 0;  // lights last uploaded

    // --- Light tree (#129) -------------------------------------------------
    // Double-buffered GPU node buffer. Worker thread runs Rebuild() on
    // the CPU while main records render commands; once the new tree is
    // ready the engine uploads it to the inactive slot and atomically
    // swaps `light_tree_active_slot_`. The path-tracer dispatch binds
    // whichever slot is active when RenderFrame runs (the previous slot
    // is left alone -- the GPU may still be sampling it from an
    // in-flight frame; it can be reused on a later frame, at which
    // point either WaitIdle on the resize path or just-overwrite-with-
    // current-data handles the synchronisation).
    //
    // SLOT BOOK-KEEPING. `light_tree_buffer_capacity_` is the shared
    // capacity (both slots are sized identically; we grow them together
    // so a swap never lands on a too-small slot). `node_count_uploaded_`
    // is the consumer-visible count -- updated when the engine flips
    // the active slot. The dispatch path keys on this count + the
    // active-slot buffer id.
    std::array<std::uint64_t, 2>                light_tree_buffer_ids_ {0, 0};
    std::uint32_t                               light_tree_buffer_capacity_ = 0;
    std::uint32_t                               light_tree_node_count_uploaded_ = 0;
    int                                         light_tree_active_slot_     = 0;
    // First-tree-ever flag. Until set, the dispatch falls back to the
    // placeholder (the binding has to resolve to *something* for Metal
    // push-slot stability; same convention as the rest of the slot-13
    // path documented in Engine.cpp's bind-site block).
    bool                                        light_tree_first_built_     = false;
    // Async builder owning the worker thread + double-buffered CPU
    // result slots. unique_ptr-of-pimpl so Engine.h doesn't have to
    // include LightTree.h (the renderer header pulls in <thread>,
    // <mutex>, <condition_variable> -- not free to drag in here).
    std::unique_ptr<pt::renderer::AsyncLightTreeBuilder> light_tree_builder_;
    // --- end Light tree ----------------------------------------------------

    // --- Fluid Phase 1 (#136) -- smoke emitters ----------------------------
    // Density-injection plumes consumed by cloud_density_at(). Each emitter
    // costs one Gaussian-falloff per cloud-march sample, so cap kept tight.
    // Engine slot 15 / shader binding 30 (slot 14 taken by ReSTIR
    // reservoir #78, slot 13 by light tree #129).
    std::vector<SmokeEmitter>                   smoke_emitters_;
    std::uint64_t                               smoke_buffer_id_         = 0;
    std::uint32_t                               smoke_buffer_capacity_   = 0;
    std::uint32_t                               smoke_count_uploaded_    = 0;
    std::uint32_t                               smoke_next_id_           = 1;
    // --- end Fluid Phase 1 -------------------------------------------------

    // --- Voxel destruction Phase 1 (#140) ----------------------------------
    // VoxelGrids produced by `voxelize_object`, keyed by source object
    // id (CSG root, SDF cluster id, or analytic-prim id). Renderer
    // injection happens via SyncVoxelDemoState which spawns one
    // reserved-id SDF cluster per occupied voxel under each grid.
    // Reserved cluster id range starts at kVoxelClusterIdBase so it
    // can't collide with user-issued sdf_* ids.
    static constexpr std::uint32_t              kVoxelClusterIdBase = 0xF0000000u;
    std::map<std::uint32_t, std::unique_ptr<pt::destruction::VoxelGrid>> voxel_grids_;
    // Mirrors r_voxelize_demo for the RenderFrame hide-mesh gate.
    // Pulled at the start of each frame in SyncVoxelDemoState. When
    // true AND at least one voxel grid exists, the CSG mesh / TLAS
    // bindings are suppressed locally so the path tracer renders ONLY
    // the voxelized form (matches the A/B-toggle requirement in #140).
    bool                                        voxel_demo_active_       = false;
    // Mutation generation for the voxel grid set. Bumped by anything
    // that changes the rendered occupancy: a new VoxelizeSourceObject
    // landing, voxelize_clear, or a future destruction step that
    // toggles voxels. SyncVoxelDemoState compares
    // voxel_grids_generation_ against voxel_demo_applied_generation_
    // to decide whether the desired reserved-cluster set already
    // matches what's in sdf_prims_ -- when they match AND the demo
    // toggle hasn't changed since the last sync, the function early-
    // outs (no rebuild, no accum_dirty_, no sdf_prims_dirty_). Without
    // this the demo path would assert accum_dirty_ every frame and
    // pin the path tracer to 1-spp forever while r_voxelize_demo=1.
    std::uint64_t                               voxel_grids_generation_         = 0u;
    std::uint64_t                               voxel_demo_applied_generation_  = 0u;
    bool                                        voxel_demo_applied_state_       = false;
    // --- end Voxel destruction Phase 1 -------------------------------------

    // Always-allocated 16-byte storage buffer used as a harmless
    // placeholder for any optional binding slot whose primary buffer is
    // 0 (e.g. SDF cluster slot 10 when no SDFs exist AND analytic prims
    // are also off via pt_smoke_skip_prim_seed=1). Metal computes the
    // dynamic push-constant slot from max-bound + 1 of the contiguous
    // binding range, so leaving any slot in that range unbound shifts
    // the push slot and corrupts every field. Allocated once at device
    // init alongside exposure_state; destroyed in TearDownDevice.
    std::uint64_t                               placeholder_storage_id_  = 0;

    std::uint64_t                               pathtrace_pipeline_id_ = 0;
    std::uint64_t                               tonemap_pipeline_id_   = 0;
    std::uint64_t                               bloom_down_pipeline_id_ = 0;
    std::uint64_t                               bloom_up_pipeline_id_   = 0;
    // Stateless stars+sun+moon composite (issue #46). Dispatched on
    // Metal between Denoise() and the bloom pyramid so post-denoise
    // HDR receives sub-pixel celestials and bloom downsamples those
    // highlights into halos. Replaces the EMA accum_stars architecture
    // from #108 -- see shaders/StarsComposite.slang for the rationale.
    std::uint64_t                               stars_composite_pipeline_id_ = 0;
    // Stateless aurora borealis composite (issue #116). Dispatched on
    // Metal AFTER stars_composite (so aurora layers on top of celestials
    // in HDR space) and BEFORE the bloom pyramid (so bright aurora
    // ribbons get bloom halos and ACES tonemap squashes them with the
    // rest of the image). Curl-noise driven procedural ribbon density
    // at 100-150 km altitude band; see shaders/AuroraComposite.slang.
    // Zero on Vulkan today -- the Vulkan dispatch is a follow-up, the
    // engine gate folds correctly to "no aurora" when the id is 0.
    std::uint64_t                               aurora_composite_pipeline_id_ = 0;
    // Screen-space particle composite (#82 MVP). Same dispatch context
    // as stars/aurora -- runs in the use_engine_tonemap branch AFTER
    // those composites and BEFORE the bloom pyramid so HDR particle
    // highlights downsample into halos. CPU-side sim lives in
    // particles_; the per-frame GPU buffer (particles_storage_id_) is
    // rebuilt each frame from the live particle list. Metal-only
    // today; Vulkan dispatch + descriptor wiring is a follow-up.
    std::uint64_t                               particle_composite_pipeline_id_ = 0;
    std::uint64_t                               particles_storage_id_  = 0;   // float4[3*N]
    std::uint32_t                               particles_storage_capacity_ = 0; // particle slots that fit
    // Reusable scratch buffer for the per-frame particle upload payload
    // (the packed 12-float-per-particle GPU layout). Keeping this as an
    // Engine member rather than a `std::vector` local in RenderFrame()
    // avoids one alloc + one free per frame at the MVP particle count
    // (~48 KB at 1024 particles -- not large in isolation, but it's
    // every frame). Resize-without-shrink semantics mean the
    // allocation tracks the high-water mark of live particles.
    std::vector<float>                          particle_upload_scratch_;
    std::uint64_t                               perfoverlay_pipeline_id_ = 0;
    std::uint64_t                               perfoverlay_drawlist_id_ = 0;
    std::uint32_t                               perfoverlay_drawlist_capacity_ = 0;

    // ---- Editor 3D-transform gizmo (issue: editor 3D gizmos) ----------------
    // Pipeline + storage buffer for the world-space gizmo line list the
    // EditorOverlay.slang kernel rasterizes onto the swapchain. The
    // segment buffer is sized to a generous worst case (the rotate
    // gizmo's 3 rings @ 32 segments + safety margin -- ~256 segments
    // max -> 12 KB upload); only the prefix of `editor_gizmo_segs_count`
    // entries is consumed by the shader, so dispatch with selection off
    // is free apart from the cleared-segment write.
    //
    // editor_gizmo_ owns the geometry-build + hit-test math; the engine
    // just shovels its segment list into the GPU buffer each frame the
    // gizmo is active. Selection state lives on `selected_prim_id_` /
    // `selected_kind_` (declared below in the editor block contributed
    // by PR #201 / agent-19); the gizmo gates on selected_kind_ ==
    // SelectionKind::AnalyticPrim so future Csg/Sdf selections cleanly
    // don't fire the analytic-only gizmo path.
    pt::renderer::EditorOverlay                 editor_gizmo_;
    std::uint64_t                               editor_overlay_pipeline_id_ = 0;
    std::uint64_t                               editor_overlay_segs_buf_id_ = 0;
    std::uint32_t                               editor_overlay_segs_buf_capacity_ = 0;
    // Mouse-button-1 edge detect for click + drag. True one frame
    // after LMB transitions down->up; lets us start drags on the
    // press edge, not on every frame LMB stays down.
    bool                                        prev_lmb_down_ = false;
    // Pre-drag origin captured in BeginDrag; restored on Esc-during-drag.
    glm::vec3                                   editor_drag_pre_pos_ { 0.0f };
    // Pre-drag orientation quaternion captured in BeginDrag for
    // rotate-mode (#206). Stored xyzw; identity is (0,0,0,1). The
    // rotate dispatch builds a delta quaternion from the drag angle
    // and multiplies it onto this stored pre-drag orient so the
    // composition keeps the pre-drag rotation as a fixed origin (no
    // accumulating-error problem from re-reading the prim's current
    // orient every frame). Esc-during-drag restores via
    // `prim_set_rotation <id> <qx qy qz qw>`.
    float                                       editor_drag_pre_orient_[4] {0.0f, 0.0f, 0.0f, 1.0f};
    // ---- end Editor gizmo ---------------------------------------------------

    std::uint64_t                               autoexpose_pipeline_id_ = 0;
    std::uint64_t                               accum_texture_id_      = 0;
    // GPU-side exposure scalar: AutoExposure.slang updates this when
    // r_auto_exposure=1; engine WriteBuffer's the manual r_exposure
    // value when r_auto_exposure=0. PathTrace.slang reads this in
    // its final tonemap, replacing the per-frame readback path that
    // stalled the GPU on dGPU.
    std::uint64_t                               exposure_state_id_     = 0;
    std::uint64_t                               box_blas_id_           = 0;
    std::uint64_t                               scene_tlas_id_         = 0;
    std::uint64_t                               box_vbuf_id_           = 0;
    std::uint64_t                               box_ibuf_id_           = 0;
    // Triangle count of the currently-uploaded CSG mesh. Non-zero iff
    // box_vbuf_id_ / box_ibuf_id_ are populated (whether or not
    // scene_tlas_id_ is). Drives the software linear-scan branch in
    // PathTrace.slang via bvh_params.z when scene_tlas_id_ == 0 -- which
    // happens on backends that lack hardware ray tracing (notably
    // Mac-Vulkan on pre-1.3 MoltenVK builds that don't expose
    // VK_KHR_acceleration_structure / VK_KHR_ray_query).
    std::uint32_t                               mesh_tri_count_        = 0;
    // Triangle BVH for the SW mesh path (PR #106 follow-up). Built
    // alongside the vbuf/ibuf upload in RebuildMeshResources from the
    // BakedMesh's positions + indices; uploaded to two storage buffers
    // and bound on every dispatch as long as a CSG mesh is present.
    // Driven from `bvh_params.w = tri_bvh_node_count_` in the push
    // constants; the shader's SW mesh branch walks the tree there
    // instead of the previous O(N) Möller-Trumbore linear scan, fixing
    // the ~1 FPS @1080p perf cliff that PR #106 introduced on
    // Mac-Vulkan (MoltenVK without VK_KHR_ray_query).
    //
    // The build is unconditional (runs on both RT-capable and SW-only
    // backends) so the dispatch site is dispatch-uniform; the HW path
    // simply doesn't read these buffers. A memory-conscious follow-up
    // could gate the build on `!SupportsHardwareRT()` to avoid the
    // upload on RT-capable backends, but that's separate work.
    pt::renderer::TriangleBvh                   tri_bvh_;
    std::uint64_t                               tri_bvh_nodes_id_       = 0;
    std::uint64_t                               tri_bvh_permuted_ids_id_ = 0;
    std::uint32_t                               tri_bvh_node_count_     = 0;
    // One-shot guard so the "SW linear-scan path with N>threshold tris"
    // perf-cliff warning fires once per process, not on every CSG bake.
    bool                                        sw_mesh_perf_warning_fired_ = false;

    // P10 denoiser G-buffer textures. Allocated lazily when r_denoiser
    // moves off "off" and freed on backend teardown / when denoiser is
    // disabled. Re-allocated on swapchain resize alongside accum_hdr.
    std::uint64_t                               denoise_color_tex_id_  = 0;
    std::uint64_t                               depth_tex_id_          = 0;
    std::uint64_t                               motion_tex_id_         = 0;
    // Vulkan SVGF/NRD denoiser only: world-space surface normal at
    // primary hit, written by PathTrace.slang's G-buffer pass and read
    // by DenoiseTemporal/DenoiseAtrous for edge-aware filtering. Also
    // allocated for the OptiX AOV denoiser (optix_hdr_aov) which uses
    // the same image as its normal guide layer. Not allocated for the
    // metalfx path -- MetalFX doesn't take normals.
    std::uint64_t                               normal_tex_id_         = 0;
    // Vulkan OptiX AOV denoiser only: linear-RGB albedo at primary
    // hit, written by PathTrace.slang's G-buffer pass and consumed by
    // OPTIX_DENOISER_MODEL_KIND_AOV as the albedo guide layer for
    // diffuse-color-edge-aware denoising. Not allocated for SVGF/NRD
    // (those don't take albedo) or MetalFX -- non-zero only when
    // r_denoiser is optix_hdr_aov.
    std::uint64_t                               albedo_tex_id_         = 0;
    // Linear HDR intermediate that MetalFX writes to instead of the
    // swapchain. The `tonemap` compute kernel reads this and writes
    // exposure+ACES-encoded sRGB into the swapchain. Co-allocated +
    // resized with the other denoiser-related textures.
    std::uint64_t                               post_denoise_hdr_tex_id_ = 0;
    // Cloud transmittance G-buffer (issue #46 follow-up). R32F
    // per-pixel, written by PathTrace's volumetric cloud march at the
    // end of main() and read by StarsComposite to attenuate the
    // additive celestial contribution. 1.0 = no occlusion (no cloud,
    // ray misses the layer, or clouds disabled); values < 1.0
    // darken the stars / sun / moon proportionally so clouds
    // occlude them in the final composited image. Allocated alongside
    // the rest of the denoiser-related textures (same denoiser_active_
    // lifecycle as depth_tex / motion_tex / post_denoise_hdr).
    std::uint64_t                               cloud_trans_tex_id_ = 0;
    // SIGMA shadow visibility G-buffer (issue #115). One-float-per-pixel
    // storage buffer (NOT a texture: Apple Silicon's 8-RW-texture cap
    // on PathTrace is already saturated). PathTrace.slang writes the
    // per-primary-hit sun-NEE shadow-ray transmittance into this buffer
    // at the end of main(); the engine's SigmaShadow.slang dispatch
    // (between Denoise() and StarsComposite) does a 5x5 bilateral
    // filter of that visibility signal and multiplies the denoised
    // visibility into the post-radiance-denoise HDR, restoring sharp
    // sun-shadow boundaries that SVGF / MetalFX otherwise smudge across.
    // 1.0 = unoccluded; 0.0 = fully shadowed; sky pixels write 1.0 so
    // the SigmaShadow pass never darkens the sky composite. Allocated
    // alongside the rest of the denoiser-related resources whenever
    // denoiser_active_ AND r_shadow_demod is on (same lifecycle as
    // depth_tex / motion_tex / cloud_trans_tex). Sized
    // width * height * sizeof(float) -> ~8.3 MB at 1080p, trivial vs
    // the rest of the denoiser texture budget.
    std::uint64_t                               shadow_vis_buf_id_ = 0;
    // SigmaShadow compute pipeline (issue #115). Built once at backend
    // init; same lifecycle as stars_composite_pipeline_id_. Metal-only
    // today (Vulkan plumbing is a follow-up); on Vulkan this stays
    // zero so the engine's r_shadow_demod gate collapses to "no
    // SIGMA dispatch, legacy denoiser shadow behaviour" until the
    // Vulkan compositor lands.
    std::uint64_t                               sigma_shadow_pipeline_id_ = 0;
    // --- ReSTIR DI Phase A (issue #78) -------------------------------------
    // Per-pixel Reservoir SSBOs for the spatiotemporal resampling chain.
    // Three buffers form a ping-pong pipeline:
    //   restir_reservoir_curr_buf_id_ : PathTrace writes the WRS
    //       candidate-generation output here; RestirTemporal reads it.
    //   restir_reservoir_prev_buf_id_ : the PREVIOUS frame's final
    //       reservoir (the buffer that was `curr` last frame is `prev`
    //       this frame -- we swap ids at end-of-frame).
    //   restir_reservoir_swap_buf_id_ : scratch for the spatial pass
    //       output (and serves as next frame's `prev`).
    //
    // Each buffer is `width * height * 64 B`. At 1920x1080 that's ~127
    // MB per buffer; 3 buffers = ~380 MB. Acceptable on M4 Max's 64 GB
    // unified memory; on dGPU this is also under the 1 GB typical
    // VRAM budget for the path tracer.
    //
    // Allocated whenever `denoiser_active_` AND `r_restir` is on (same
    // lifecycle gate as shadow_vis_buf_id_). Freed in TearDownDevice
    // and re-created on swapchain resize -- pixel-strided buffers are
    // tied to the framebuffer dimensions.
    std::uint64_t                               restir_reservoir_curr_buf_id_ = 0;
    std::uint64_t                               restir_reservoir_prev_buf_id_ = 0;
    std::uint64_t                               restir_reservoir_swap_buf_id_ = 0;
    // Pipeline ids for the three Restir compute kernels. Built once at
    // device init (Metal target only at Phase A scope; Vulkan plumbing
    // is a follow-up, matches the SigmaShadow / StarsComposite
    // pattern). Zero on backends without these pipelines -> the
    // dispatch gates short-circuit cleanly.
    std::uint64_t                               restir_temporal_pipeline_id_ = 0;
    std::uint64_t                               restir_spatial_pipeline_id_  = 0;
    std::uint64_t                               restir_final_pipeline_id_    = 0;
    // Tracked frame width/height we last allocated the reservoir
    // buffers at. Mismatch -> realloc on the next frame's resize gate.
    std::uint32_t                               restir_alloc_w_ = 0;
    std::uint32_t                               restir_alloc_h_ = 0;
    // --- end ReSTIR DI Phase A ---------------------------------------------
    // MetalFX specular-guidance G-buffers (issue #118). Three textures
    // fed to MTLFXTemporalDenoisedScaler so it can tell specular from
    // diffuse response; without them MetalFX produces 8x8 halos on
    // bright reflections. Allocated only for MetalFX-family denoiser
    // kinds (DenoiserKind::MetalFX / SvgfBasicMetalFx / SvgfAtrousMetalFx);
    // SVGF / NRD / OptiX paths leave them at 0 since they don't accept
    // these guidance inputs. Same allocation lifecycle as
    // normal_tex_id_ / albedo_tex_id_.
    //
    //   specular_albedo_tex_id_       -- RGBA16F per-pixel F0 (Fresnel
    //                                    reflectance at normal incidence).
    //                                    Metals: F0 = albedo; dielectrics:
    //                                    F0 = float3(0.04); Lambert: 0.
    //   roughness_tex_id_             -- R32F single-channel surface
    //                                    roughness in [0, 1]. 0 = mirror,
    //                                    1 = fully rough. (R16F would be
    //                                    plenty for the precision but the
    //                                    RHI doesn't expose it today; see
    //                                    Engine.cpp allocation for the
    //                                    R16F-vs-R32F trade.)
    //   specular_hit_distance_tex_id_ -- R32F distance from camera to the
    //                                    specularly-reflected hit (MVP:
    //                                    primary_t * smoothness; a future
    //                                    PR can swap in a real reflection-
    //                                    ray trace). Same R32F-as-fallback
    //                                    rationale as roughness_tex_id_.
    std::uint64_t                               specular_albedo_tex_id_       = 0;
    std::uint64_t                               roughness_tex_id_             = 0;
    std::uint64_t                               specular_hit_distance_tex_id_ = 0;
    // Bloom mip chain. mip 0 is half-res of the swapchain; each
    // subsequent mip halves again. Built every frame from
    // post_denoise_hdr_tex_ via threshold + downsample + upsample
    // passes; sampled by the tonemap kernel before ACES so bloom
    // gets the same curve compression as the rest of the image.
    static constexpr int                        kBloomMips = 5;
    std::uint64_t                               bloom_mip_tex_id_[kBloomMips] {};
    std::uint32_t                               bloom_mip_w_[kBloomMips] {};
    std::uint32_t                               bloom_mip_h_[kBloomMips] {};
    std::uint64_t                               bloom_dummy_tex_id_ = 0;   // 1x1 RGBA16F when bloom off

    // Physical lens flare (Hullin paraxial). LensSystem + traced
    // ghost matrices live for the engine's lifetime; per-frame we
    // compute screen-UV scales from the current viewport via
    // prepare_shader_ghosts and pack into the tonemap push struct.
    // r_lens_flare_mode = "physical" routes the shader to gather
    // Gaussian splats at those positions; sun + image modes remain
    // available as fallbacks.
    lensflare::LensSystem                       lens_system_{};
    lensflare::Ghost                            lens_ghosts_[lensflare::kMaxGhosts] {};
    int                                         lens_ghost_count_ = 0;
    lensflare::MainPath                         lens_main_path_{};

    // P11 environment map. Allocated when r_env_map cvar points at a
    // valid .hdr file; freed on cvar change or device teardown. The
    // size matches the HDR file (typical 2048x1024 lat-long).
    std::uint64_t                               env_map_tex_id_        = 0;
    std::string                                 env_map_path_;

    // P11 env-map MIS / NEE: precomputed CDFs over luminance. Marginal
    // is a 1D CDF over rows (length H); conditional is a 2D CDF over
    // columns within each row (length H*W). Both built when an HDR is
    // loaded so the path tracer can importance-sample bright regions.
    std::uint64_t                               env_marginal_cdf_id_   = 0;
    std::uint64_t                               env_conditional_cdf_id_ = 0;
    float                                       env_total_luminance_   = 0.0f;

    // HDRI multi-light extraction. ReloadEnvMap thresholds the HDRI at
    // the top 0.5% luminance percentile, runs 4-connected flood-fill
    // (with horizontal wrap for lat-long) on the resulting mask, and
    // stores the top kMaxHdriLights clusters by integrated flux. Each
    // cluster's pixels are masked out of the env-map CDF (so env-map
    // NEE skips them) and replaced by a stochastic directional NEE
    // weighted by cluster luminance -- sharp shadows for any HDRI:
    // single-sun outdoor, lone moon at night, multi-lamp interior,
    // 3-point studio rig. The env_map texture itself is unchanged so
    // camera-direct rays still see the visible bright pixels.
    static constexpr std::uint32_t              kMaxHdriLights = 8;
    struct HdriLight {
        glm::vec3 dir        = {0.0f, 1.0f, 0.0f};  // unit vec to centroid
        float     pmf        = 0.0f;                 // p(this light), sums to 1 across lights
        glm::vec3 irradiance = {0.0f, 0.0f, 0.0f};   // ∫_cluster L dΩ per channel
        float     luminance  = 0.0f;                 // luminance(irradiance), used for sorting + pmf
    };
    std::array<HdriLight, kMaxHdriLights>       hdri_lights_{};
    std::uint32_t                               hdri_lights_count_       = 0;

    // P11 BSC starmap. RGBA16F equirectangular in J2000, rasterised once
    // at startup from assets/stars/BSC5.dat. The shader rotates incoming
    // ray directions into J2000 (using the per-frame world->J2000
    // matrix the engine pushes) and samples this texture additively
    // when the sun is below the horizon. Created lazily on first
    // backend init; reused across vid_restarts. star_map_present_ is
    // 0 if the load + rasterise failed (so the shader skips sampling).
    std::uint64_t                               star_map_tex_id_       = 0;
    std::uint32_t                               star_map_present_      = 0;

    // CPU-side particle system for issue #82 MVP. Owns the live
    // particle vector + continuous-emitter slots. Engine::Tick steps
    // it once per frame; Engine::RenderFrame rebuilds the GPU storage
    // buffer (particles_storage_id_) from its current state and
    // dispatches the ParticleComposite kernel.
    //
    // unique_ptr instead of inline storage so we can forward-declare
    // pt::effects::ParticleSystem above and keep Engine.h's include
    // surface minimal. The fwd-decl needs the pointer to be heap-owned.
    std::unique_ptr<pt::effects::ParticleSystem> particles_;

    // Moon surface texture (procedural, generated at engine init).
    // Equirectangular RGBA16F, 512x256. Sampled in moonDisc() when the
    // ray hits the moon's angular footprint; combined with
    // moon_dir_phase to give the disc actual mare + crater detail and
    // a true terminator curve from the lit-hemisphere check.
    std::uint64_t                               moon_map_tex_id_       = 0;
    glm::mat4                                   prev_view_proj_        { 1.0f };  // identity
    bool                                        prev_view_proj_valid_  = false;
    bool                                        denoiser_active_       = false;
    // Issue #119 -- previous-frame SVGF albedo-demod effective state.
    // Tracks what we actually told the backend last frame (i.e. cvar
    // gated by albedo-texture availability). When this changes
    // mid-run, the SVGF history texture holds data in the OPPOSITE
    // colour space (demod-on stores lighting / albedo; demod-off
    // stores full radiance), so we have to fire reset_history before
    // the temporal blend lerps stale history against the new
    // input. Initially false so the first-frame "valid history"
    // check (prev_view_proj_valid_) drives reset, not this gate.
    bool                                        prev_svgf_albedo_demod_active_ = false;
    // One-shot state-transition latch for the Vulkan + SVGF/NRD bloom
    // path (built in PR for feature/vulkan-bloom-svgf-nrd). True from
    // the frame the predicate first goes hot until it goes cold; lets
    // us log "engaged"/"disengaged" exactly once per toggle instead of
    // every frame. The predicate itself is recomputed in Tick(), this
    // is purely the edge-detect memory.
    bool                                        vulkan_pre_denoise_bloom_engaged_ = false;
    // Same edge-detect for the Vulkan + OptiX bloom path. Earlier code
    // used a function-static one-shot bool here, which logged the
    // initial r_bloom-on transition but stayed silent on subsequent
    // r_bloom toggles -- mismatching the SVGF path's "engaged on, log
    // each toggle" pattern and making "did bloom actually disengage?"
    // ungreppable in user logs. Member-based latch matches SVGF.
    bool                                        vulkan_optix_bloom_engaged_       = false;
    // Which denoiser kind the active backend is running. metalfx is
    // Mac-only; svgf_basic/svgf_atrous/nrd route through the in-house
    // SVGF chain on EITHER backend -- VulkanNrdDenoiser on Vulkan,
    // MetalSvgfDenoiser on Metal, both built from the same Slang
    // sources. Real NRD library integration is deferred (see
    // Raytracer Plan/FOLLOW_UPS.md). Stored so the engine knows
    // whether to allocate the normal G-buffer (both Vulkan and Metal
    // SVGF need it) and which one-time log to print.
    //
    //   SvgfBasic  = temporal accumulation only (no spatial filter)
    //   SvgfAtrous = temporal + a-trous edge-aware filter
    //   SvgfBasicMetalFx / SvgfAtrousMetalFx = chain the corresponding
    //                SVGF mode through MetalFX TemporalDenoisedScaler
    //                as a finalizer (Mac only). SVGF kills the path-
    //                tracing noise, MetalFX then ML-TAAs the edges /
    //                cleans up sub-pixel aliasing the SVGF spatial
    //                filter can't preserve. Falls back to plain SVGF
    //                on Vulkan / when MetalFX isn't available.
    //   Nrd        = currently aliases SvgfAtrous; reserved for the
    //                proper NVIDIA RayTracingDenoiser library later.
    //   OptixHdr   = NVIDIA OptiX denoiser, HDR model. CUDA-Vulkan
    //                interop; available only when PT_ENABLE_OPTIX is
    //                compiled in AND the runtime detects an OptiX-
    //                capable NVIDIA GPU. Falls back to off otherwise.
    //   OptixHdrAov= OptiX HDR + albedo + normal AOV model. Better
    //                quality (especially in shadowed regions) at small
    //                additional cost. Adds a primary_albedo G-buffer
    //                output to the path tracer (Vulkan/SPIR-V only).
    //   OptixTemporalHdr     = HDR model + motion-vector flow guide +
    //                          1-frame denoised-output history. Closes
    //                          the per-frame flicker gap vs SVGF on
    //                          static scenes while keeping OptiX's
    //                          motion-handling advantage (no temporal
    //                          ghosting on fast motion).
    //   OptixTemporalHdrAov  = OptixTemporalHdr + albedo + normal guide
    //                          layers. The strongest OptiX variant --
    //                          temporal smoothing AND AOV edge fidelity.
    enum class DenoiserKind : std::uint8_t {
        Off, MetalFX, SvgfBasic, SvgfAtrous, Nrd,
        SvgfBasicMetalFx, SvgfAtrousMetalFx,
        OptixHdr, OptixHdrAov,
        OptixTemporalHdr, OptixTemporalHdrAov,
    };
    DenoiserKind                                denoiser_kind_         = DenoiserKind::Off;
    float                                       last_jitter_x_         = 0.0f;
    float                                       last_jitter_y_         = 0.0f;

    // Auto-exposure now lives entirely on the GPU (see exposure_state_id_
    // above + AutoExposure.slang). The legacy CPU-side `current_exposure_`
    // / `autoexpose_counter_` fields drove a per-N-frames readback path
    // that's been replaced; nothing on the host needs to mirror the value
    // anymore -- PathTrace.slang reads `exposure_state[0]` directly.
    int                                         accum_w_               = 0;
    int                                         accum_h_               = 0;
    std::uint32_t                               frame_index_           = 0;

    // `screenshot ... swap` deferred-capture state. The screenshot
    // lambda runs in Console::Drain on the main render thread BEFORE
    // RenderFrame; blocking inside it would deadlock Submit, so we
    // queue the request, store the output path here, and let
    // Engine::Tick poll Device::ReadbackSwapchain across frames until
    // it returns true (= consume + WaitIdle done, staging bytes
    // ready). Empty string = no capture pending.
    std::string                                 pending_swap_screenshot_path_;
    // Output format latched at queue time (read from r_capture_format
    // then). The deferred writer in Tick uses this rather than re-reading
    // the cvar, so a user toggling r_capture_format while a swap capture
    // is in flight doesn't change the format mid-flight. The path stored
    // above already carries the matching .png / .ppm extension applied
    // via ResolveCapturePath.
    pt::engine::capture::OutputFormat           pending_swap_screenshot_fmt_ =
        pt::engine::capture::OutputFormat::Png;
    // Ticks elapsed since the screenshot was queued. Used to bound
    // the wait: if Submit never consumes the request (device down,
    // suspended loop, etc.), give up after a few seconds rather than
    // leaving the pending state forever stuck.
    int                                         pending_swap_screenshot_ticks_ = 0;

    // Tracks whether the loading-frame branch in RenderFrame has
    // ever fired (i.e. the Vulkan async pipeline build was still in
    // flight when we hit our first frame). Used to log a single
    // "loading screen active" message on entry and a single
    // "pipelines ready" message on exit -- avoids a per-frame log
    // spam during the 1-3s build window.
    bool                                        loading_frame_active_  = false;
    // Set true while Init() is sourcing demont.cfg / autoexec.cfg /
    // command-line cvar overrides. Astronomy-only cvar on_change
    // handlers consult this so they don't fire false-positive
    // "your cvar is ignored" warnings during cfg load: cfg writes
    // happen in std::map / lexicographic order, so r_sky_* lines get
    // applied before r_sky_use_astronomical, and a user with astro=1
    // saved would otherwise see warnings that get superseded a
    // millisecond later. A single summary audit fires after cfg load
    // completes (Engine::Init), with the final astro state correct.
    bool                                        cfg_loading_           = false;
    // Set true inside the r_sky_city on_change while it cascades into
    // r_sky_lat / r_sky_lon / r_sky_tz_offset_hours via SetCVarOverride.
    // The astronomy-only warning suppresses itself when this is set so
    // a single user-issued `r_sky_city` change emits one warn (for the
    // city itself), not three (city + lat + lon).
    bool                                        astro_chained_update_  = false;
    // Set true by Init() right before it returns. Used by the
    // pt_smoke_late_exec on_change handler to know whether the
    // existing end-of-Init read at line ~1508 has already handled the
    // current value (init path) or whether the cvar set is a runtime
    // interactive write that needs immediate execution (post-init
    // path -- e.g. user typed `exec tests/goldens/scenes/phys_rb_smoke.cfg`
    // at the console after init was done).
    bool                                        engine_initialized_    = false;
    bool                                        accum_dirty_           = true;
    // BVH-upload log dedup state. EnsurePrimitivesUploaded fires every
    // frame that any analytic prim's position changed (e.g. physics
    // step moved a rigid body), which previously spammed the
    // "[bvh] uploaded N prim(s)" log line at frame rate. Cache the
    // last-logged (count, linear_count, bvh_node_count, bvh_empty)
    // tuple and only emit when something actually changed (add /
    // remove / topology rebuild). Sentinel -1 forces a log on the
    // very first upload after init.
    std::int64_t                                bvh_last_log_count_       = -1;
    std::uint32_t                               bvh_last_log_linear_      = 0;
    std::uint32_t                               bvh_last_log_node_count_  = 0;
    bool                                        bvh_last_log_empty_       = true;
    BackendType                                 current_backend_       = BackendType::None;
    bool                                        mouse_look_active_     = false;
    // cam_warp_smooth tween state. While cam_tween_active_ is true,
    // UpdateCamera() lerps the camera state from (start_*) toward
    // (target_*) over cam_tween_duration_ seconds. Smoothstep easing
    // gives a cinematic in/out feel without needing a full keyframe
    // system. Cleared when the lerp completes OR when the user
    // touches WASD / RMB-look (manual control aborts the tween).
    bool                                        cam_tween_active_      = false;
    double                                      cam_tween_t_seconds_   = 0.0;
    double                                      cam_tween_duration_    = 0.0;
    float                                       cam_tween_start_pos_[3]    {0, 0, 0};
    float                                       cam_tween_target_pos_[3]   {0, 0, 0};
    float                                       cam_tween_start_yaw_       = 0.0f;
    float                                       cam_tween_target_yaw_      = 0.0f;
    float                                       cam_tween_start_pitch_     = 0.0f;
    float                                       cam_tween_target_pitch_    = 0.0f;
    std::atomic<bool>                           wants_quit_{false};
    // Smoke-test mode: set true by Run() when the smoke-test outcome is
    // a failure. Three failure modes feed it:
    //   1. No device bound within kSmokeNoDeviceTimeoutSec (~10s) --
    //      backend init failed silently.
    //   2. ApplyCommandLineCvarOverrides rejected a CLI arg
    //      (allowed_values miss, non-numeric --smoke-frames, etc.) --
    //      smoke mode shouldn't proceed against a misconfigured engine.
    //   3. Run loop exited (window-close, `quit` command, ShouldClose
    //      from any other path) before the frame budget was hit --
    //      smoke test cancelled, not completed.
    // main() reads this via SmokeTestFailed() to set the process exit
    // code. Default false; the budget=0 case never sets it.
    bool                                        smoke_test_failed_     = false;
    // Sticky flag set by ApplyCommandLineCvarOverrides when any CLI
    // arg is rejected (unknown allowed-value, non-numeric integer,
    // etc.). Engine::Run inspects it at loop start: in smoke mode,
    // a rejected arg is a hard fail (mode #2 above). In interactive
    // mode, the LOG_ERROR is enough -- no behavioural change.
    bool                                        cli_arg_was_rejected_  = false;
    // Latched once the backend reports IsDeviceLost(). The smoke-test
    // loop polls device_->IsDeviceLost() each tick and on the first
    // observation: logs a LOG_ERROR, requests exit, and (in smoke
    // mode) trips smoke_test_failed_. This flag is just so the log
    // line fires once instead of every frame between detection and
    // the loop actually exiting on wants_quit_.
    bool                                        device_lost_observed_  = false;

    // Snapshot of last camera state, used to detect movement and reset
    // accumulation. (vec4 to keep it trivially copyable.)
    float                                       last_cam_pos_[3]   { 0, 0, 0 };
    float                                       last_cam_yaw_      = 0.0f;
    float                                       last_cam_pitch_    = 0.0f;
    float                                       last_cam_fov_      = 0.0f;

    // --- Editor backend (agent-19) ----------------------------------------
    // Currently-selected scene entity. The shader's outline highlight reads
    // selected_prim_id_ via PtPush::selected_prim_id and emits a magenta
    // silhouette around any analytic-prim hit whose id matches AND
    // selected_kind_ == AnalyticPrim. Light / SdfCluster / RigidBody kinds
    // are reserved for future agents (20/22) and currently don't paint
    // an outline. id == 0 + kind == None is the "no selection" state.
    // Both mutated together via SetSelection.
    std::uint32_t                               selected_prim_id_  = 0;
    SelectionKind                               selected_kind_     = SelectionKind::None;
    // --- end Editor backend ------------------------------------------------
};

}  // namespace pt::engine

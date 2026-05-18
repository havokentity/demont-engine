// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../app/Window.h"
#include "../core/Jobs/JobSystem.h"
#include "../renderer/AnalyticBvh.h"
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
#include <vector>

namespace pt::app      { class Window; class ConsoleOverlay; class PerfOverlay; }
namespace pt::audio    { class AudioSystem; }
namespace pt::jobs     { class JobSystem; }
namespace pt::console  { class ConsoleServer; }
namespace pt::rhi      { class Device; struct PipelineHandle; }
namespace pt::renderer { struct Camera; }
namespace pt::csg      { class CsgScene; struct BakedMesh; }

namespace pt::engine {

using BackendType = pt::rhi::BackendType;

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

public:
    // Analytic primitive description, mirrors the GPU layout (3 float4s per
    // primitive). The engine maintains a map of these and rebuilds the GPU
    // buffer whenever the set changes.
    struct AnalyticPrim {
        enum Type     : std::uint8_t { Sphere = 0, Plane = 1 };
        enum Material : std::uint8_t { Lambert = 0, Metal = 1, Dielectric = 2 };
        Type     type      = Sphere;
        Material material  = Lambert;
        float    pos_or_n[3] {0, 0, 0};   // sphere center / plane normal
        float    radius_or_d = 0.5f;      // sphere radius / plane d
        float    albedo[3]   {1, 1, 1};
        float    roughness   = 0.0f;
        float    ior         = 1.5f;
    };

private:
    void RegisterCommands();
    void RegisterCsgCommands();
    void RegisterPrimCommands();
    // --- SDF Phase 1 (#97) -------------------------------------------------
    // Console commands for the SDF primitive set (`sdf_sphere`,
    // `sdf_box`, `sdf_smin`, ...). Mirrors the prim_*/csg_* registration
    // style; populated map is uploaded to GPU on next render frame via
    // EnsureSdfPrimsUploaded.
    void RegisterSdfCommands();
    // --- end SDF Phase 1 ---------------------------------------------------
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

    // Drains any in-flight CSG bake job, keeps the engine's vertex/index
    // buffers + BLAS + TLAS in sync with the current scene, and -- if the
    // scene has been mutated since the last bake -- fires a new bake on a
    // worker thread. Called every frame before pipeline binding.
    void EnsureMeshUpdated();

    // Re-upload the analytic-primitive storage buffer from the in-memory
    // map. Called from RenderFrame whenever primitives_dirty_ is set.
    void EnsurePrimitivesUploaded();

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
    std::unique_ptr<pt::rhi::Device>            device_;
    std::unique_ptr<pt::renderer::Camera>       camera_;
    // Audio subsystem (issue #80 MVP -- miniaudio-backed 3D playback).
    // Init() opens the default device + voice pool; Shutdown() releases
    // both. Tick(camera_pos, camera_fwd) pushes a listener snapshot
    // into the audio thread for distance attenuation + stereo panning.
    // Ray-traced occlusion / reverb / HRTF (the issue's headline) are
    // deferred to a follow-up that consumes the renderer's TLAS.
    std::unique_ptr<pt::audio::AudioSystem>     audio_system_;
    std::unique_ptr<pt::csg::CsgScene>          csg_scene_;
    std::unique_ptr<pt::csg::BakedMesh>         pending_baked_;
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
    std::uint64_t                               perfoverlay_pipeline_id_ = 0;
    std::uint64_t                               perfoverlay_drawlist_id_ = 0;
    std::uint32_t                               perfoverlay_drawlist_capacity_ = 0;
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

    // Moon surface texture (procedural, generated at engine init).
    // Equirectangular RGBA16F, 512x256. Sampled in moonDisc() when the
    // ray hits the moon's angular footprint; combined with
    // moon_dir_phase to give the disc actual mare + crater detail and
    // a true terminator curve from the lit-hemisphere check.
    std::uint64_t                               moon_map_tex_id_       = 0;
    glm::mat4                                   prev_view_proj_        { 1.0f };  // identity
    bool                                        prev_view_proj_valid_  = false;
    bool                                        denoiser_active_       = false;
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
    bool                                        accum_dirty_           = true;
    BackendType                                 current_backend_       = BackendType::None;
    bool                                        mouse_look_active_     = false;
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

    // Snapshot of last camera state, used to detect movement and reset
    // accumulation. (vec4 to keep it trivially copyable.)
    float                                       last_cam_pos_[3]   { 0, 0, 0 };
    float                                       last_cam_yaw_      = 0.0f;
    float                                       last_cam_pitch_    = 0.0f;
    float                                       last_cam_fov_      = 0.0f;
};

}  // namespace pt::engine

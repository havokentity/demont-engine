// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include "../app/Window.h"
#include "../core/Jobs/JobSystem.h"
#include "../rhi/Types.h"
#include "LensFlare.h"

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace pt::app      { class Window; class ConsoleOverlay; class PerfOverlay; }
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

    bool Init();
    void Shutdown();

    // Launches the user's default browser at the running web console URL.
    // No-op (with a warning) if the platform launcher fails.
    void OpenWebConsole();

    // Main loop.  Returns when the window is closed or `quit` runs.
    void Run();

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
    void TearDownDevice();
    void RenderFrame();

    void UpdateCamera(double dt);

    // Drains any in-flight CSG bake job, keeps the engine's vertex/index
    // buffers + BLAS + TLAS in sync with the current scene, and -- if the
    // scene has been mutated since the last bake -- fires a new bake on a
    // worker thread. Called every frame before pipeline binding.
    void EnsureMeshUpdated();

    // Re-upload the analytic-primitive storage buffer from the in-memory
    // map. Called from RenderFrame whenever primitives_dirty_ is set.
    void EnsurePrimitivesUploaded();

    // Load (or unload) the env map from disk. Called from the r_env_map
    // cvar's on_change. Sets env_map_tex_id_ and env_map_path_.
    void ReloadEnvMap(const std::string& path);

    // Load BSC5 + rasterise the J2000 starmap GPU texture once. Idempotent
    // (safe to call after a backend switch -- it skips if the texture
    // already exists). Failures are non-fatal: star_map_tex_id_ stays 0
    // and the path tracer's r_show_stars output silently drops to nothing.
    void EnsureStarMapUploaded();
    void EnsureMoonMapUploaded();

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
    std::unique_ptr<pt::csg::CsgScene>          csg_scene_;
    std::unique_ptr<pt::csg::BakedMesh>         pending_baked_;
    pt::jobs::JobSystem::Handle                 bake_handle_{};
    std::atomic<int>                            bake_phase_{0};   // 0 idle, 1 baking, 2 ready

    // Analytic primitives (sphere/plane) -- ordered by user id, uploaded
    // to a storage buffer when dirty. Mesh CSG and analytic primitives
    // are independent; the unified renderer takes the closest hit.
    std::map<std::uint32_t, AnalyticPrim>       primitives_;
    bool                                        primitives_dirty_      = true;
    std::uint64_t                               prim_buffer_id_        = 0;
    std::uint32_t                               prim_buffer_capacity_  = 0;  // primitives that fit

    std::uint64_t                               pathtrace_pipeline_id_ = 0;
    std::uint64_t                               tonemap_pipeline_id_   = 0;
    std::uint64_t                               bloom_down_pipeline_id_ = 0;
    std::uint64_t                               bloom_up_pipeline_id_   = 0;
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

    // P10 denoiser G-buffer textures. Allocated lazily when r_denoiser
    // moves off "off" and freed on backend teardown / when denoiser is
    // disabled. Re-allocated on swapchain resize alongside accum_hdr.
    std::uint64_t                               denoise_color_tex_id_  = 0;
    std::uint64_t                               depth_tex_id_          = 0;
    std::uint64_t                               motion_tex_id_         = 0;
    // Linear HDR intermediate that MetalFX writes to instead of the
    // swapchain. The `tonemap` compute kernel reads this and writes
    // exposure+ACES-encoded sRGB into the swapchain. Co-allocated +
    // resized with the other denoiser-related textures.
    std::uint64_t                               post_denoise_hdr_tex_id_ = 0;

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
    bool                                        accum_dirty_           = true;
    BackendType                                 current_backend_       = BackendType::None;
    bool                                        mouse_look_active_     = false;
    std::atomic<bool>                           wants_quit_{false};

    // Snapshot of last camera state, used to detect movement and reset
    // accumulation. (vec4 to keep it trivially copyable.)
    float                                       last_cam_pos_[3]   { 0, 0, 0 };
    float                                       last_cam_yaw_      = 0.0f;
    float                                       last_cam_pitch_    = 0.0f;
    float                                       last_cam_fov_      = 0.0f;
};

}  // namespace pt::engine

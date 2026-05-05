#pragma once

#include "../app/Window.h"
#include "../rhi/Types.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace pt::app      { class Window; class ConsoleOverlay; }
namespace pt::jobs     { class JobSystem; }
namespace pt::console  { class ConsoleServer; }
namespace pt::rhi      { class Device; struct PipelineHandle; }
namespace pt::renderer { struct Camera; }

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

private:
    void RegisterCommands();
    void TearDownDevice();
    void RenderFrame();

    void UpdateCamera(double dt);

    std::unique_ptr<pt::app::Window>            window_;
    std::unique_ptr<pt::app::ConsoleOverlay>    overlay_;
    std::unique_ptr<pt::jobs::JobSystem>        jobs_;
    std::unique_ptr<pt::console::ConsoleServer> server_;
    std::unique_ptr<pt::rhi::Device>            device_;
    std::unique_ptr<pt::renderer::Camera>       camera_;
    std::uint64_t                               clear_pipeline_id_     = 0;
    std::uint64_t                               scene_pipeline_id_     = 0;
    std::uint64_t                               pathtrace_pipeline_id_ = 0;
    std::uint64_t                               mesh_pipeline_id_      = 0;
    std::uint64_t                               accum_texture_id_      = 0;
    std::uint64_t                               box_blas_id_           = 0;
    std::uint64_t                               scene_tlas_id_         = 0;
    std::uint64_t                               box_vbuf_id_           = 0;
    std::uint64_t                               box_ibuf_id_           = 0;
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

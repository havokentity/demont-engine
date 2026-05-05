#pragma once

#include "../app/Window.h"
#include "../rhi/Types.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace pt::app    { class Window; class ConsoleOverlay; }
namespace pt::jobs   { class JobSystem; }
namespace pt::console{ class ConsoleServer; }
namespace pt::rhi    { class Device; struct PipelineHandle; }

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
    pt::app::GraphicsApi ApiFor(BackendType t);

    std::unique_ptr<pt::app::Window>            window_;
    std::unique_ptr<pt::app::ConsoleOverlay>    overlay_;
    std::unique_ptr<pt::jobs::JobSystem>        jobs_;
    std::unique_ptr<pt::console::ConsoleServer> server_;
    std::unique_ptr<pt::rhi::Device>            device_;
    std::uint64_t                               clear_pipeline_id_ = 0;
    BackendType                                 current_backend_   = BackendType::None;
    std::atomic<bool>                           wants_quit_{false};
};

}  // namespace pt::engine

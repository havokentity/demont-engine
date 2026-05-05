#pragma once

#include <atomic>
#include <memory>

namespace pt::app    { class Window; }
namespace pt::jobs   { class JobSystem; }
namespace pt::console{ class ConsoleServer; }

namespace pt::engine {

enum class BackendType { None, Software, Metal, Vulkan };

class Engine {
public:
    Engine();
    ~Engine();

    bool Init();
    void Shutdown();

    // Main loop.  Returns when the window is closed or `quit` runs.
    void Run();

    // Stub for Phase 1; actually destroys/creates the device starting in P2.
    void RequestBackendSwitch(BackendType to);

    // Process one frame's work after Window::PollEvents.
    void Tick(double dt);

    static Engine* Instance();

private:
    void RegisterCommands();

    std::unique_ptr<pt::app::Window>            window_;
    std::unique_ptr<pt::jobs::JobSystem>        jobs_;
    std::unique_ptr<pt::console::ConsoleServer> server_;
    std::atomic<bool>                           wants_quit_{false};
};

}  // namespace pt::engine

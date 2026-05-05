#include "core/Jobs/JobSystem.h"
#include "core/Log.h"
#include "core/Memory/Memory.h"
#include "engine/Engine.h"

int main() {
    LOG_INFO("DeMonT PathTracer -- De Monte Carlo-esque Tracer");
    pt::engine::Engine engine;
    if (!engine.Init()) {
        LOG_ERROR("Engine init failed");
        return 1;
    }

    // Smoke-test the job system (Phase 1 acceptance).
    if (auto* js = pt::jobs::JobSystem::Instance(); js != nullptr) {
        bool ran = false;
        js->Run([&ran]{ ran = true; });
        LOG_INFO("JobSystem smoke test: ran = {}", ran);
    }

    engine.Run();

    pt::mem::PrintReport();
    return 0;
}

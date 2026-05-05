#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>

namespace enki {
class TaskScheduler;
}  // namespace enki

namespace pt::jobs {

// Thin wrapper over enkiTS. One worker per physical performance core unless
// overridden. Submit returns a Handle that can be Waited on; if you don't
// need to wait, fire-and-forget by ignoring the handle.
class JobSystem {
public:
    JobSystem();
    ~JobSystem();
    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    // worker_count = 0 -> auto (P-cores - 1, leave one for main thread).
    void Init(int worker_count = 0);
    void Shutdown();

    // Number of worker threads in the pool (excludes the main thread).
    int  WorkerCount() const noexcept;

    // Returns once `fn` has finished executing on a worker. Currently
    // implemented as Submit + Wait; convenient for one-off jobs.
    void Run(std::function<void()> fn);

    // Submit a job and return a wait-handle. Caller must Wait or drop.
    struct Handle { void* internal = nullptr; };
    Handle Submit(std::function<void()> fn);
    void   Wait(Handle handle);

    // Parallel for: call fn(i) for i in [begin, end), spread across workers.
    void ParallelFor(std::size_t begin, std::size_t end,
                     std::function<void(std::size_t)> fn);

    // Singleton-style accessor for code that doesn't want to thread the
    // pointer (the engine owns the lifetime).
    static JobSystem* Instance();
    static void       SetInstance(JobSystem* s);

private:
    std::unique_ptr<enki::TaskScheduler> sched_;
    int worker_count_ = 0;
};

}  // namespace pt::jobs

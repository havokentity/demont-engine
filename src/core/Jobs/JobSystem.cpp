// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "JobSystem.h"
#include "../Diag.h"
#include "../Log.h"
#include "../Hardware/HardwareInfo.h"

#include <TaskScheduler.h>

#include <atomic>

namespace pt::jobs {

namespace {

JobSystem* g_instance = nullptr;

// Wraps a std::function and self-deletes after one invocation.
class FnTask : public enki::ITaskSet {
public:
    explicit FnTask(std::function<void()> fn) : fn_(std::move(fn)) {
        m_SetSize = 1;
    }
    void ExecuteRange(enki::TaskSetPartition, uint32_t) override {
        fn_();
    }
private:
    std::function<void()> fn_;
};

class FnRangeTask : public enki::ITaskSet {
public:
    FnRangeTask(std::size_t begin, std::size_t end,
                std::function<void(std::size_t)> fn)
        : begin_(begin), fn_(std::move(fn)) {
        m_SetSize = static_cast<uint32_t>(end - begin);
    }
    void ExecuteRange(enki::TaskSetPartition r, uint32_t) override {
        for (uint32_t i = r.start; i < r.end; ++i) {
            fn_(begin_ + i);
        }
    }
private:
    std::size_t begin_;
    std::function<void(std::size_t)> fn_;
};

}  // namespace

JobSystem::JobSystem() = default;
JobSystem::~JobSystem() { Shutdown(); }

void JobSystem::Init(int worker_count) {
    if (sched_) return;

    if (worker_count <= 0) {
        worker_count = pt::hw::GetInfo().cpu_pcores - 1;
        if (worker_count < 1) worker_count = 1;
    }
    worker_count_ = worker_count;

    sched_ = std::make_unique<enki::TaskScheduler>();
    enki::TaskSchedulerConfig cfg;
    cfg.numTaskThreadsToCreate = static_cast<uint32_t>(worker_count_);
    sched_->Initialize(cfg);

    PT_DIAG_TIER1("jobs", "JobSystem online: {} worker thread(s)", worker_count_);
}

void JobSystem::Shutdown() {
    if (sched_) {
        sched_->WaitforAllAndShutdown();
        sched_.reset();
    }
}

int JobSystem::WorkerCount() const noexcept { return worker_count_; }

JobSystem::Handle JobSystem::Submit(std::function<void()> fn) {
    auto* task = new FnTask(std::move(fn));
    sched_->AddTaskSetToPipe(task);
    return {task};
}

void JobSystem::Wait(Handle handle) {
    if (handle.internal == nullptr) return;
    auto* task = static_cast<FnTask*>(handle.internal);
    sched_->WaitforTask(task);
    delete task;
}

void JobSystem::Run(std::function<void()> fn) {
    Wait(Submit(std::move(fn)));
}

void JobSystem::ParallelFor(std::size_t begin, std::size_t end,
                            std::function<void(std::size_t)> fn) {
    if (begin >= end) return;
    FnRangeTask task(begin, end, std::move(fn));
    sched_->AddTaskSetToPipe(&task);
    sched_->WaitforTask(&task);
}

JobSystem* JobSystem::Instance() { return g_instance; }
void       JobSystem::SetInstance(JobSystem* s) { g_instance = s; }

}  // namespace pt::jobs

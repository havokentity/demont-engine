// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// MVP CPU-side particle simulator for issue #82.
//
// Scope notes (read first before extending):
//   - This is the MVP slice of issue #82. The issue's headline is "GPU
//     particles, PT-compatible"; this MVP is intentionally neither.
//     Both are explicit follow-ups -- see the PR for the split.
//   - Sim runs on the CPU each frame, into a flat std::vector<Particle>.
//     A capped live count (`r_particles_max`, default 1024) keeps the
//     update O(N) on a known tiny N so we don't need any of the
//     spatial-hash / free-list machinery that a GPU sim would want.
//   - The engine uploads the live subset to a Metal storage buffer once
//     per frame (see Engine::RenderFrame -- the dispatch site mirrors
//     StarsComposite). The GPU side never SIMULATES; it only reads the
//     CPU-uploaded state and SPLATS each particle as a soft Gaussian
//     billboard into the post-denoise HDR texture, depth-gated by the
//     PathTrace G-buffer so foreground geometry occludes particles.
//   - Positions are in WORLD-SPACE METRES (1 world unit = 1 m, per the
//     project's metric units convention). Velocities are m/s. Gravity
//     is the real value, 9.81 m/s^2, applied along world -Y.
//   - dt is clamped to a 1/30 s fixed step inside Update(). A 4x burst
//     of frame-rate variation (e.g. a window drag pause) won't catapult
//     particles across the world; the integrator simply advances by
//     the cap and the time beyond the cap is DROPPED (no sub-stepping,
//     no accumulator). This is a quick MVP-grade Euler loop -- the
//     timestep cap keeps the integration stable for the soft-ballistic
//     motion these particles do, and dropping the over-budget time
//     reads as a brief slow-mo rather than a catapult or a stutter
//     burst. Implementing remainder accumulation / sub-stepping is a
//     follow-up if a future content scene wants smooth visuals through
//     hitches.
//
// What's deferred to follow-up issues (mentioned for the next reader's
// sanity, not for action here):
//   1. PT integration (particles visible in reflections / refractions /
//      shadows). Today's particles are a SCREEN-SPACE composite layer.
//      The "PT-compatible" headline is the analytic-primitive list +
//      BVH refit path (issue #82 Phase B).
//   2. GPU simulation. Today the CPU runs the integrator and the GPU
//      only paints billboards. The "GPU particles" headline is the
//      compute-shader sim + free-list / compaction path (issue #82
//      Phase A).
//   3. Vulkan dispatch. The composite kernel runs Metal-only. The
//      Vulkan plumbing is identical in spirit to StarsComposite's
//      "Metal-only today" comment block.

#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace pt::effects {

// One simulated particle. Packed into 48 bytes on the host; the GPU
// side uses a parallel 48-byte struct (see ParticleSystem::GpuParticle
// in the .cpp). Keep the host layout free of GPU-specific padding so
// the simulator stays portable -- the GPU-facing struct is built
// per-frame in the upload step.
struct Particle {
    glm::vec3 pos        {0.0f, 0.0f, 0.0f};   // world-space metres
    glm::vec3 vel        {0.0f, 0.0f, 0.0f};   // m/s
    glm::vec4 color      {1.0f, 1.0f, 1.0f, 1.0f};   // RGBA, lerped per frame from spawn `color` toward `end_color`. The shader uses .a as the per-particle alpha multiplier on the splat.
    float     age        = 0.0f;                // seconds
    float     lifetime   = 1.0f;                // seconds
    float     size       = 0.05f;               // world-space metres (Gaussian sigma scaled per pixel by the shader)
    float     drag       = 0.0f;                // 1/s linear-drag coefficient (vel *= exp(-drag*dt))
    glm::vec4 end_color  {1.0f, 1.0f, 1.0f, 1.0f}; // color (incl. alpha) at age = lifetime (lerped per frame)
    float     end_size   = 0.05f;                  // size at age = lifetime
    float     gravity_y  = -9.81f;                 // m/s^2, world-Y component (per-particle so smoke can float)
    float     _pad0      = 0.0f;
    float     _pad1      = 0.0f;
};

// Spawn-preset descriptor used by EmitBurst / EmitContinuous. Mirrors
// the parameters available from the console commands; the preset
// helpers below build one of these for "smoke", "spark", "snow".
struct EmitSpec {
    glm::vec3 pos             {0.0f, 0.0f, 0.0f};    // emission origin (metres)
    glm::vec3 vel_mean        {0.0f, 0.0f, 0.0f};    // mean initial velocity (m/s)
    glm::vec3 vel_spread      {0.0f, 0.0f, 0.0f};    // +/- uniform spread per axis
    float     pos_radius      = 0.0f;                 // uniform sphere radius around pos (metres)
    glm::vec4 color           {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 end_color       {1.0f, 1.0f, 1.0f, 0.0f};
    float     size            = 0.05f;                // initial size (metres)
    float     end_size        = 0.05f;
    float     lifetime_mean   = 1.0f;                 // seconds
    float     lifetime_spread = 0.0f;                 // +/- uniform spread
    float     drag            = 0.0f;                 // 1/s
    float     gravity_y       = -9.81f;               // m/s^2
    std::uint32_t count       = 0;                    // for EmitBurst
};

// Continuous emitter -- one of these per active "continuous" preset
// (e.g. snow). Engine pumps each frame in Update(dt). The accumulator
// converts the float rate (particles / second) into integer-spawn
// events without long-term drift.
struct ContinuousEmitter {
    EmitSpec spec;
    float    rate            = 0.0f;   // particles per second
    float    accumulator     = 0.0f;   // fractional remainder carried frame-to-frame
    bool     active          = false;
};

class ParticleSystem {
public:
    ParticleSystem();
    ~ParticleSystem();

    // No copies (RNG state + vector are owned). Move is disabled because
    // the worker thread (`worker_`) holds a pointer back to *this; moving
    // the system out from under its own thread would dangle that pointer.
    // The engine owns ParticleSystem via std::unique_ptr so no copy/move
    // is actually exercised in practice.
    ParticleSystem(const ParticleSystem&)            = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;
    ParticleSystem(ParticleSystem&&)                 = delete;
    ParticleSystem& operator=(ParticleSystem&&)      = delete;

    // Set the cap on live particles. Anything past the cap is dropped
    // on spawn. The default (1024) is set in the constructor; the
    // engine pushes the r_particles_max cvar value here at init and on
    // change. Going BELOW the current live count doesn't truncate
    // existing particles -- new spawns just stop until naturals age
    // out, keeping the cap a soft ceiling under load.
    //
    // Threading: blocks until the async tick (if any) finishes before
    // mutating max_particles_, so mid-frame cvar changes can't race the
    // worker thread reading the cap inside EmitBurst().
    void SetMaxParticles(std::uint32_t max);
    std::uint32_t MaxParticles() const noexcept { return max_particles_; }

    // Live particle accessors. Used by the engine to build the GPU
    // upload payload each frame. The vector is contiguous (no holes
    // mid-array); dead particles are compacted-out inside Update().
    //
    // Threading: both block until the async tick finishes so the caller
    // sees a coherent particle field. The engine calls Particles() once
    // per frame from the render path, which is exactly the join point
    // we want; console commands also use these accessors and block the
    // same way (their cost is invisible -- the worker has long finished
    // by the time a user types a command).
    const std::vector<Particle>& Particles() const {
        WaitForTick();
        return particles_;
    }
    std::uint32_t LiveCount() const {
        WaitForTick();
        return static_cast<std::uint32_t>(particles_.size());
    }

    // Synchronous tick. Kicks the worker and blocks until it finishes.
    // Kept for tests + any caller that wants the old single-threaded
    // semantics; the engine's hot path uses TickAsync + WaitForTick
    // instead so the sim overlaps the rest of the frame's CPU work.
    //
    // Internally clamps dt to <= 1/30 s so a hitched main loop doesn't
    // fling particles across the world (per the header comment block).
    // Steps:
    //   1. Tick continuous emitters; spawn N = floor(rate*dt + acc).
    //   2. Integrate positions: vel += gravity*dt; pos += vel*dt;
    //      vel *= exp(-drag*dt) (analytic-decay linear drag).
    //   3. Lerp color + size from spawn -> end values by t = age/lifetime.
    //   4. Compact -- erase particles with age >= lifetime.
    void Update(float dt) {
        TickAsync(dt);
        WaitForTick();
    }

    // Post an integrator step to the persistent worker thread and
    // return immediately. The engine kicks this at frame-start so the
    // CPU sim overlaps physics + render-command-building. The dt is
    // snapshotted inside the call so the caller is free to mutate the
    // float afterward; the worker reads the snapshot, not the caller's
    // variable.
    //
    // PRE-CONDITION: any prior TickAsync must have been waited on (the
    // engine's WaitForTick before reading Particles() satisfies this).
    // Calling TickAsync twice without a wait will block on the prior
    // tick to make the misuse harmless rather than corrupt the vector.
    void TickAsync(float dt);

    // Block until the most recently posted TickAsync finishes. Idempotent
    // and cheap to call after the worker is already idle -- the implementation
    // reads an atomic flag and only enters the condition-variable wait
    // when the tick is in flight. Called automatically by Particles(),
    // LiveCount(), SetMaxParticles(), EmitBurst(), StartContinuous(),
    // StopContinuous(), and Clear() so external callers usually don't
    // need to touch it.
    void WaitForTick() const;

    // One-shot burst spawn. Used by `particle_emit smoke <x> <y> <z>`
    // and `particle_emit spark <x> <y> <z>` console commands. Up to
    // `spec.count` particles are created subject to the live cap;
    // returns how many actually landed.
    //
    // Threading: waits for the async tick before mutating particles_.
    std::uint32_t EmitBurst(const EmitSpec& spec);

    // Start a continuous emitter. The MVP supports one slot per preset
    // name; calling Start with the same name updates that slot rather
    // than appending. Today only "snow" uses this; smoke / spark are
    // burst-only.
    void StartContinuous(const char* name, const EmitSpec& spec, float rate);
    // Stop ALL continuous emitters matching `name` (one in the MVP,
    // but the API leaves room for multiple if a future preset grows
    // into a layered emitter). Returns the number stopped.
    std::uint32_t StopContinuous(const char* name);
    // Clear all live particles and stop every continuous emitter.
    // Used by the `particle_clear` console command.
    void Clear();

    // Preset builders. Each returns an EmitSpec the caller can hand to
    // EmitBurst (smoke / spark) or StartContinuous (snow). Centralized
    // so the console command and any future programmatic spawner share
    // exactly one definition.
    //
    //   Smoke: warm-gray puff, low initial velocity, slight upward
    //     bias, almost-zero gravity (smoke is buoyant), color decays
    //     to transparent over ~2 s.
    //   Spark: bright-orange burst, fast outward velocity, gravity is
    //     -9.81 m/s^2 like a real spark, color decays orange -> red
    //     -> dim over ~0.6 s.
    //   Snow: continuous area emitter, downward velocity ~1.5 m/s,
    //     gravity is -9.81 m/s^2 but with drag so flakes ride the air.
    //     pos is the centre of a 10 m horizontal disc 6 m above origin
    //     (or above `centre` if supplied).
    static EmitSpec PresetSmoke(const glm::vec3& at);
    static EmitSpec PresetSpark(const glm::vec3& at);
    static EmitSpec PresetSnow(const glm::vec3& centre);

private:
    // The actual integrator. Runs on `worker_` when posted via TickAsync
    // and inline when called from EmitBurst (because the spawn path needs
    // the same emitter pump). All callers are serialised by `tick_in_flight_`
    // and the start/done condition variable so there's no need for a mutex
    // around particles_ / emitters_ themselves.
    void UpdateLocked(float dt);

    void WorkerLoop();

    std::vector<Particle>             particles_;
    std::vector<ContinuousEmitter>    emitters_;
    std::uint32_t                     max_particles_ = 1024;

    // Per-system RNG, seeded once in the ctor. mt19937 is overkill for
    // visual-noise jitter; chose it because it's std-library-only (no
    // extra dep) and easy to verify deterministic-with-seed in tests.
    // Per-particle randomness is sampled per spawn so the same EmitSpec
    // produces a different cloud each call without us threading a seed
    // through the API.
    std::mt19937                      rng_;
    // Uniform [-1, 1) helper. Internal use only for jitter; not
    // exposed because callers would want a different distribution
    // (Gaussian, anisotropic ...) in nearly every other use.
    float Uniform11();

    // ---- Worker thread plumbing -----------------------------------------
    //
    // One persistent thread per ParticleSystem. The engine creates a
    // single ParticleSystem so there's exactly one worker for the
    // lifetime of the process. Sized for "one CPU job per frame": when
    // the worker is idle it parks on the cv; when the main thread calls
    // TickAsync() it sets pending_dt_, flips tick_in_flight_ = true,
    // and notifies; the worker wakes, runs UpdateLocked(pending_dt_),
    // flips tick_in_flight_ = false, notifies waiters.
    //
    // shutdown_ is the worker's poison pill; dtor flips it true and
    // notifies so the worker can fall out of the cv wait and exit.
    mutable std::mutex              tick_mtx_;
    mutable std::condition_variable tick_cv_;
    std::atomic<bool>               tick_in_flight_{false};
    bool                            tick_requested_  = false;  // guarded by tick_mtx_
    float                           pending_dt_      = 0.0f;   // guarded by tick_mtx_
    bool                            shutdown_        = false;  // guarded by tick_mtx_
    std::thread                     worker_;
};

}  // namespace pt::effects

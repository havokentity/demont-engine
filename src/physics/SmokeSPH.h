// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace pt::sph {

// SmokeSPH -- Fluid Phase 3 (#22): Smoothed Particle Hydrodynamics
// fluid simulation for smoke. Replaces (or augments) the procedural
// puff-chain density field from Phase 2 with a particle-based sim.
//
// Müller 2003 formulation:
//   * Density via kernel sum over neighbours (poly6).
//   * Pressure via stiff Tait/EOS: p = k * (rho - rho0).
//   * Forces: pressure gradient (spiky kernel), gravity, thermal
//     buoyancy, optional global wind + curl-noise turbulence
//     (wave-9), and impulse shockwaves for explosions.
//   * Viscosity: bounded XSPH velocity smoothing (wave-9; replaced the
//     explicit Müller viscosity-laplacian force, which was numerically
//     unstable at the high mu values smoke wants). The Müller laplacian
//     kernel is retained as a documented primitive.
//
// Spatial hash grid (uniform-cell, bucket-per-cell) for O(N)
// neighbour lookup. Cell size == kernel radius h. Particles update
// against a fixed substep dt (1/120 s by default) so the integrator
// stays stable even when the frame dt jitters.
//
// Particle count: parameterised. The standalone class default cap is
// 1024; the engine raises it via r_smoke_sph_max_particles (default 2048
// since Wave 8 #28 -- a denser pool reads as a coherent plume). Per-frame
// cost stays well below 1 ms at 120 substeps; raise toward the 4096 cap
// when more visual fidelity is needed.
//
// Particles are spawned by SmokeEmitter records in the engine (each
// emitter pushes ~r_smoke_sph_emit_rate particles per second from
// its mouth with velocity == emitter.velocity + jitter). Particles
// age and despawn after lifetime; the renderer reads particle
// (pos, radius, density) from the GPU SSBO and splat-rasterizes them
// into the smoke density field (see smoke_sph_density_add in
// PathTrace.slang).
class SmokeSPH {
public:
    // Particle state. 32 B exactly so two float4s per particle pack
    // into the GPU SSBO layout 1:1.
    struct Particle {
        glm::vec3 pos    {0.0f};  // world space, metres
        float     radius {0.1f};  // splat radius (kernel-scaled); also drives density falloff
        glm::vec3 vel    {0.0f};  // m/s
        float     age    {0.0f};  // seconds since spawn (>= lifetime -> removed)
    };

    // Per-emitter SPH spawn record. Mirrors the engine's SmokeEmitter
    // pos/velocity/radius but adds thermal/lifetime parameters that
    // drive buoyancy. Pure data; the solver owns no engine types.
    struct EmitterParams {
        glm::vec3 pos        {0.0f};
        glm::vec3 velocity   {0.0f, 1.0f, 0.0f};  // initial velocity, m/s
        float     radius     {1.0f};               // emitter mouth radius (sampling jitter)
        float     temperature_kelvin {600.0f};     // hot smoke; ambient = 293 K
        float     spawn_rate {0.0f};               // particles/sec emitted; 0 = idle
        float     lifetime_s {6.0f};               // particle lifespan; 0 = no decay
        // kg per particle. Real-air smoke particles are micrograms;
        // we run at ~0.02 kg/particle to get visible inter-particle
        // pressure forces inside the kernel support radius. This is
        // the canonical "visual fluid SPH" trick -- particles
        // represent macroscopic fluid blobs, not real-air dust.
        float     particle_mass {0.02f};
        float     particle_radius {0.1f};          // initial splat radius (m)
        float     initial_density_offset {0.0f};   // density at spawn (>=0). 0 = ambient.
    };

    // Solver tuning. All defaults map roughly onto real air-smoke
    // parameters (rho_air ~= 1.2 kg/m^3 + soot perturbation) but
    // dialled for visual stability over physical exactness.
    struct Config {
        // Kernel radius (m). Drives both the SPH neighbour cutoff and
        // the spatial hash cell size. 0.4 m is a good compromise: big
        // enough that 12-24 neighbours fit at the target density,
        // small enough that the column reads as a column rather than
        // a single smeared cloud.
        float kernel_radius_h = 0.4f;

        // Rest density rho0 (kg/m^3). Air at 1 atm @ 20 C is 1.2;
        // we set rest density slightly higher so freshly emitted hot
        // particles see a small positive pressure deviation, giving
        // them a gentle puff-out at the source.
        float rest_density = 1.2f;

        // Pressure stiffness (Tait k). Stiff EOS keeps the column
        // close to rho0 without resorting to a true incompressible
        // projection. Higher = stiffer = more spring-like = a WIDER,
        // more diffuse column (the pressure gradient pushes particles
        // apart harder). Wave 9 sph-3b lowered the default from 5 to
        // 1.5: at 5 the inter-particle repulsion over-dispersed the
        // plume into a thin ~30 m-wide haze; 1.5 keeps it a coherent
        // ~10 m column that reads as a recognizable dense plume. The
        // engine overrides via r_smoke_pressure_stiffness.
        float pressure_stiffness = 1.5f;

        // Viscosity mu (kg/m/s). Air mu ~= 1.8e-5; we run several
        // orders higher because (a) the SPH integrator at 120 Hz is
        // way too coarse to resolve Re~real-air turbulence; (b) we
        // WANT visible smoothing so the column reads as a coherent
        // plume rather than as a cloud of independent particles.
        float viscosity = 0.05f;

        // Gravity (m/s^2). Earth: -9.81 along -Y.
        glm::vec3 gravity {0.0f, -9.81f, 0.0f};

        // Ambient air temperature (K). 293 K = 20 C. Buoyancy force
        // is proportional to (T_p - T_air) / T_air per the Boussinesq
        // approximation -- hot smoke rises because it's less dense
        // than the air it displaces.
        float ambient_temperature_kelvin = 293.0f;

        // Buoyancy strength scaler (dimensionless). 1.0 = the
        // Boussinesq formula's natural strength (which is gentle and
        // looks anaemic against the gravity force); 4.0 gives a
        // visible rise speed without overshooting realism. The user
        // can override via r_smoke_buoyancy.
        float buoyancy_scale = 4.0f;

        // Thermal decay rate (K/s). Hot particles cool exponentially
        // toward ambient with this time constant. A 600 K particle
        // cools to 1/e of its excess (600 - 293 = 307 K above
        // ambient -> 307 / e ~= 113 K above ambient) in 1/this
        // seconds. 0.3 = ~3.3 s e-folding time, slower than real
        // smoke but tuned so the buoyant rise is visible across the
        // particle lifetime.
        float thermal_decay = 0.3f;

        // Air drag coefficient (1/s). Each particle bleeds velocity
        // proportional to its own speed each step. Tames the
        // explosion-impulse path so a strong shockwave doesn't
        // catapult particles off-screen permanently.
        float air_drag = 0.4f;

        // Global wind (m/s). Drives all particles horizontally; the
        // engine pushes this from r_smoke_wind. Defaults to 0.
        glm::vec3 wind {0.0f};

        // Substep dt (s). Fixed 1/120 s for stability under the
        // Müller-stiff EOS. The Step() entry-point clamps the frame
        // dt and runs floor(frame_dt / substep_dt) substeps + 1
        // half-substep for the remainder.
        float substep_dt = 1.0f / 120.0f;

        // Hard cap on substeps per frame. A 30-fps frame dt at the
        // default substep_dt would request 4 substeps; we cap at 8
        // to avoid the death spiral when frames spike.
        int   max_substeps_per_frame = 8;

        // --- Wave 9 sph-3b: turbulence / structure -----------------------
        // Curl-noise force perturbation (m/s^2). A divergence-free curl of
        // a procedural value-noise potential field, sampled at the
        // particle position, is added as a per-particle acceleration. Curl
        // noise is the canonical way to give particle smoke the rising +
        // curling structure of real turbulence WITHOUT a full fluid
        // pressure-projection grid: because curl(F) is divergence-free the
        // particles swirl and fold but don't spuriously compress or
        // disperse, so the column keeps its mass while gaining internal
        // structure. 0 disables (legacy Wave-8 behaviour).
        //
        // Strength is in m/s^2 -- it's an acceleration, mass-independent
        // (like buoyancy/wind), so all particles see the same curl field
        // regardless of their mass. ~6 m/s^2 reads as visible smoke
        // turbulence at the spatial scale below without overpowering the
        // buoyant rise.
        float curl_strength = 6.0f;

        // Curl-noise spatial frequency (1/m). Sets the size of the
        // turbulent eddies: 1/freq metres per swirl. Real smoke shows
        // structure at a range of scales; we use a single dominant scale
        // ~2-3 m (freq ~0.35) which matches the eddy size that reads as
        // "smoke curl" at the smoke_phase3 column width. Higher = finer
        // (busier) curl; lower = broad lazy swirls.
        float curl_frequency = 0.35f;

        // Curl-noise time evolution rate (1/s). The potential field
        // animates over time so the turbulence isn't a frozen static
        // pattern -- eddies are born, drift, and dissipate. 0.5 = the
        // field morphs on a ~2 s timescale, matching the visual life of a
        // smoke eddy.
        float curl_time_rate = 0.5f;

        // Curl lateral gain (dimensionless). The curl field's
        // horizontal (xz) components are amplified relative to the
        // vertical (y) component by this factor so the turbulence
        // preferentially folds the column SIDEWAYS (lateral billowing)
        // while the buoyant rise stays coherent vertically -- real
        // chimney plumes billow outward as they rise rather than churning
        // randomly in y. 1.0 = isotropic curl.
        float curl_lateral_gain = 1.6f;

        // Per-particle buoyancy variation (fractional, 0..1). Each
        // particle's buoyancy_scale is multiplied by (1 +/- this), seeded
        // deterministically per particle, so the column doesn't rise as a
        // single rigid slug -- some parcels loft faster, producing the
        // billowing front of a real plume. 0 = uniform rise (legacy).
        float buoyancy_variation = 0.35f;
        // --- end Wave 9 sph-3b -------------------------------------------
    };

    // Active shockwave event. Queued via QueueShockwave, applied
    // ONCE on the next Step (then cleared). Each shockwave adds a
    // radial impulse: per-particle dv proportional to strength /
    // dist^2 along the (particle - centre) direction.
    struct Shockwave {
        glm::vec3 centre   {0.0f};
        float     strength {0.0f};  // J of total energy; divided across affected particles
        float     radius   {0.0f};  // effective radius (m); zero distance falls back to a small epsilon
    };

    SmokeSPH()  = default;
    ~SmokeSPH() = default;

    SmokeSPH(const SmokeSPH&)            = delete;
    SmokeSPH& operator=(const SmokeSPH&) = delete;

    // Drop all particles + emitters + queued shockwaves. Resets the
    // solver to a fresh state. Idempotent.
    void Clear();

    // Cap the live particle count. Particles above the cap are
    // dropped from the tail. Default 1024.
    void SetMaxParticles(std::uint32_t cap);
    std::uint32_t MaxParticles() const { return max_particles_; }

    // Replace the emitter set. The engine maintains the
    // SmokeEmitter list independently (so persistent emitters
    // survive Clear) and pushes the snapshot into the solver each
    // frame. Order is irrelevant; the solver iterates.
    void SetEmitters(std::vector<EmitterParams> emitters);
    const std::vector<EmitterParams>& Emitters() const { return emitters_; }

    // Mutable config knobs (wind, buoyancy, gravity, etc.). The
    // engine writes from cvars each frame.
    Config&       MutableConfig()       { return cfg_; }
    const Config& GetConfig()     const { return cfg_; }

    // Queue a radial impulse to fire on the next Step. The shock
    // applies once and is cleared. Multiple shocks per frame stack.
    void QueueShockwave(const glm::vec3& centre, float strength_joules, float radius);

    // Step the simulation by frame_dt (s). Internally runs N
    // substeps at fixed substep_dt. Each substep:
    //   1. Spawn new particles from each emitter (Poisson-disc
    //      jittered around the mouth)
    //   2. Update spatial hash grid
    //   3. Compute density (poly6 kernel sum)
    //   4. Compute pressure forces (spiky gradient) + viscosity
    //      (Müller laplacian) + gravity + buoyancy + wind + drag
    //   5. Apply queued shockwaves (cleared at the end of the call)
    //   6. Integrate velocity + position (semi-implicit Euler)
    //   7. Cool particles toward ambient + age them
    //   8. Despawn aged-out / off-domain particles
    //
    // No-op if no emitters AND no live particles (the buffer is
    // empty -> nothing to integrate).
    void Step(float frame_dt);

    // Read-only particle view for the GPU upload path. Live
    // particles are NOT necessarily packed; iterate via Particles()
    // + IsAlive(i).
    const std::vector<Particle>& Particles() const { return particles_; }
    bool                          IsAlive(std::size_t i)   const { return i < alive_.size() && alive_[i] != 0; }
    std::uint32_t                 AliveCount() const { return alive_count_; }

    // Convenience: pack live particles into a contiguous (pos+radius,
    // density+_pad+_pad+_pad) float4 array for the GPU SSBO. Caller
    // owns the destination buffer. Each particle becomes 2 float4s
    // = 32 B. Returns the number of particles packed (== AliveCount).
    //
    // Density mapping: the splat density returned to the shader is
    // an approximation of the SPH density rho mapped onto a per-
    // particle Gaussian splat magnitude. Specifically, the splat
    // density = (rho / rho0) * config.particle_density_scale_for_render
    // -- so a particle at rest density renders at the user-tunable
    // baseline, and a particle in a dense pocket renders darker.
    // Tuning lives on the engine cvar r_smoke_sph_render_scale.
    //
    // Each particle occupies 2 float4s = 32 B (matches the GPU
    // layout in PathTrace.slang's sph_particles[] binding):
    //   v0.xyz = pos
    //   v0.w   = render radius (particle.radius scaled by host)
    //   v1.x   = render density (peak per-particle sigma_t for the
    //            ray-march; not the SPH rho)
    //   v1.y   = age normalised [0..1] (0 = fresh, 1 = lifetime)
    //   v1.zw  = reserved
    std::uint32_t PackForGpu(float* dst_floats,
                             std::uint32_t max_floats,
                             float render_radius_scale,
                             float render_density_scale) const;

    // Spawn one particle directly. Returns true if the particle was
    // added, false if the pool was at cap. The engine uses this for
    // manual emitter commands; emitters typically push via Step's
    // spawn loop. If `out_index` is non-null, the index of the
    // newly added slot is written there (only meaningful on success).
    bool SpawnParticle(const Particle& p, std::size_t* out_index = nullptr);

    // For tests + the editor inspector: read the spatial hash stats
    // (occupancy, max bucket size). Costs an O(N) scan each call so
    // not for hot paths.
    struct GridStats {
        std::uint32_t live_particles = 0;
        std::uint32_t buckets        = 0;
        std::uint32_t max_bucket_pop = 0;
        float         mean_density   = 0.0f;
    };
    GridStats ComputeGridStats() const;

    // Diagnostic: return the centre-of-mass of all live particles.
    // Used by the engine status command to validate wind / buoyancy.
    glm::vec3 ComputeCentreOfMass() const;

private:
    // Allocate / reset internal storage to max_particles_. Called
    // on Clear() and SetMaxParticles().
    void Reallocate();

    // Single fixed-substep integration step. Caller drives substep
    // count + emitter spawning.
    void Substep(float sdt);

    // Emit new particles for the current substep. Each emitter
    // accumulates a fractional emit_debt; once it crosses 1.0 a
    // particle is spawned at the emitter mouth.
    void RunEmitters(float sdt);

    // Rebuild the spatial hash grid against the current alive set.
    void RebuildGrid();

    // For a particle index, append the indices of all particles
    // within kernel_radius_h of it (including itself). Uses the
    // hash grid; iterates the 27-cell neighbourhood.
    void GatherNeighbours(std::size_t i, std::vector<std::size_t>& out) const;

    // Müller 2003 poly6 (density), spiky (pressure gradient), and
    // viscosity-laplacian kernels. All take r = distance from
    // particle, h = kernel radius. Defined inline in the .cpp.
    static float kernel_poly6(float r2, float h2, float h9);
    static float kernel_spiky_grad_coef(float r, float h, float h6);
    static float kernel_visc_laplacian(float r, float h, float h6);

    // --- Wave 9 sph-3b ---
    // Divergence-free curl-noise sample at world position p (m) and sim
    // time t (s). Returns an acceleration direction (unit-ish, scaled by
    // cfg_.curl_strength + lateral gain by the caller). Built as the
    // analytic curl of a 3-component value-noise potential field so the
    // result is incompressible (no spurious smoke compression/dispersal).
    glm::vec3 CurlNoise(const glm::vec3& p, float t) const;
    // --- end Wave 9 sph-3b ---

    // Hash a discrete grid cell (gx, gy, gz) into the bucket table.
    // Uses the standard Müller large-prime hash; bucket count must
    // be coprime with the primes (we round up to next prime).
    std::uint32_t HashCell(int gx, int gy, int gz) const;

    Config                                  cfg_;
    std::vector<Particle>                   particles_;
    std::vector<std::uint8_t>               alive_;
    std::uint32_t                           alive_count_ = 0;
    std::uint32_t                           max_particles_ = 1024;

    // Per-particle scratch: SPH density (kg/m^3) + pressure (Pa).
    std::vector<float>                      density_;
    std::vector<float>                      pressure_;
    std::vector<float>                      temperature_;
    std::vector<glm::vec3>                  force_;
    // --- Wave 9 sph-3b ---
    // Per-particle buoyancy-bias multiplier (around 1.0), seeded once at
    // spawn from the emitter RNG so the column rises with parcel-to-parcel
    // variation rather than as a rigid slug. Lives in a parallel scratch
    // array (kept out of the 32 B GPU Particle layout, which is full).
    std::vector<float>                      buoy_bias_;
    // Per-particle XSPH velocity correction (normalised weighted-mean
    // neighbour velocity minus own velocity), computed in the force pass
    // and applied as a bounded convex blend at integration. Stable
    // replacement for the explicit Müller viscosity force.
    std::vector<glm::vec3>                  xsph_corr_;
    // --- end Wave 9 sph-3b ---

    // Spatial hash grid. Stored as parallel arrays: bucket_first_[h]
    // is the first index into bucket_next_ for hash h; bucket_next_[i]
    // is the next particle index, or kEnd. Compact CSR-ish layout
    // that doesn't dynamically allocate per bucket. Bucket count
    // is rounded up from 2 * max_particles to the next prime.
    static constexpr std::uint32_t kEnd = 0xffffffffu;
    std::vector<std::uint32_t>              bucket_first_;
    std::vector<std::uint32_t>              bucket_next_;
    std::uint32_t                           bucket_count_ = 0;

    // Emitter set + per-emitter spawn debt. Debt accumulates each
    // substep at rate * sdt; particles spawn when debt >= 1.
    std::vector<EmitterParams>              emitters_;
    std::vector<float>                      emit_debt_;

    // Queued shockwaves; consumed + cleared on each Step.
    std::vector<Shockwave>                  pending_shocks_;

    // Deterministic per-emitter RNG for jitter. Stored alongside
    // emit_debt_ so re-tested fixtures land bit-exact across runs
    // (no system entropy). 32-bit LCG state; we don't need high-
    // quality randomness, just deterministic spatter.
    std::vector<std::uint32_t>              rng_state_;

    // Sim time accumulator (seconds since Clear). Used for thermal
    // decay phase + diagnostics.
    float                                   time_seconds_ = 0.0f;
};

}  // namespace pt::sph

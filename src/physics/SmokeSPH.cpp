// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "SmokeSPH.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pt::sph {

namespace {

// Standard Müller 2003 hash prime constants for the spatial grid.
// Coprime with any reasonable bucket count (we round bucket count up
// to the next prime so the modulo distributes well).
constexpr std::uint32_t kHashPrimeX = 73856093u;
constexpr std::uint32_t kHashPrimeY = 19349663u;
constexpr std::uint32_t kHashPrimeZ = 83492791u;

// Pi as a single-precision constant (matches the M_PI conversion
// from <math.h> without forcing _USE_MATH_DEFINES on MSVC).
constexpr float kPi = 3.14159265358979323846f;

// Next prime helper for the bucket count. We round up so the
// modulo h % bucket_count distributes uniformly across the bucket
// table. The smoke solver caps particles in the low thousands so
// the bucket count tops out around 4-8k and the prime walk is fast.
bool IsPrime(std::uint32_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if ((n & 1u) == 0u) return false;
    for (std::uint32_t i = 3; i * i <= n; i += 2) {
        if ((n % i) == 0u) return false;
    }
    return true;
}

std::uint32_t NextPrime(std::uint32_t n) {
    if (n <= 2) return 2;
    if ((n & 1u) == 0u) ++n;
    while (!IsPrime(n)) n += 2;
    return n;
}

// Cheap floor-to-int for negative values that doesn't UB on
// overflow. The spatial hash uses signed cell coords so values
// near the world origin land in the same cell on both sides of 0.
int FloorToInt(float x) {
    return static_cast<int>(std::floor(x));
}

// Wrap a 32-bit LCG step (Numerical Recipes). Deterministic per
// emitter, cheap, returns [0..1].
float Lcg01(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f;  // 2^24
}

}  // namespace

// --- Kernel definitions (Müller 2003) --------------------------------
//
// poly6:       W(r, h) = (315 / (64*pi*h^9)) * (h^2 - r^2)^3, r <= h
// spiky grad:  dW/dr   = -(45 / (pi*h^6)) * (h - r)^2 (toward j)
// viscosity:   d2W/dr2 = (45 / (pi*h^6)) * (h - r)
//
// Each returns the SCALAR coefficient; the caller multiplies by
// the relevant direction vector / density-mass ratio.

float SmokeSPH::kernel_poly6(float r2, float h2, float h9) {
    if (r2 >= h2) return 0.0f;
    const float a = (h2 - r2);
    return (315.0f / (64.0f * kPi * h9)) * a * a * a;
}

float SmokeSPH::kernel_spiky_grad_coef(float r, float h, float h6) {
    if (r >= h || r <= 0.0f) return 0.0f;
    const float a = (h - r);
    return -(45.0f / (kPi * h6)) * a * a;
}

float SmokeSPH::kernel_visc_laplacian(float r, float h, float h6) {
    if (r >= h) return 0.0f;
    return (45.0f / (kPi * h6)) * (h - r);
}

// --- Public API -------------------------------------------------------

void SmokeSPH::Clear() {
    particles_.clear();
    alive_.clear();
    alive_count_ = 0;
    density_.clear();
    pressure_.clear();
    temperature_.clear();
    force_.clear();
    bucket_first_.clear();
    bucket_next_.clear();
    bucket_count_ = 0;
    emitters_.clear();
    emit_debt_.clear();
    rng_state_.clear();
    pending_shocks_.clear();
    time_seconds_ = 0.0f;
    Reallocate();
}

void SmokeSPH::SetMaxParticles(std::uint32_t cap) {
    if (cap == max_particles_) return;
    max_particles_ = cap;
    // Drop overflowing particles (tail) if we shrank the pool.
    if (particles_.size() > cap) {
        particles_.resize(cap);
        alive_.resize(cap, 0);
        density_.resize(cap, 0.0f);
        pressure_.resize(cap, 0.0f);
        temperature_.resize(cap, cfg_.ambient_temperature_kelvin);
        force_.resize(cap, glm::vec3(0.0f));
        alive_count_ = 0;
        for (std::uint8_t a : alive_) alive_count_ += (a != 0) ? 1u : 0u;
    }
    Reallocate();
}

void SmokeSPH::SetEmitters(std::vector<EmitterParams> emitters) {
    emitters_ = std::move(emitters);
    // Preserve emit debt across frames where possible (size matches).
    // When the count changes, reset the debt so a freshly added
    // emitter doesn't fire a backlog of particles in the first
    // substep.
    if (emit_debt_.size() != emitters_.size()) {
        emit_debt_.assign(emitters_.size(), 0.0f);
        rng_state_.resize(emitters_.size());
        for (std::size_t i = 0; i < rng_state_.size(); ++i) {
            // Seed each emitter's RNG from its slot index so the
            // spatter pattern is deterministic across runs but
            // distinct per emitter.
            rng_state_[i] = static_cast<std::uint32_t>(0x9e3779b9u + i * 0x85ebca6bu);
        }
    }
}

void SmokeSPH::QueueShockwave(const glm::vec3& centre, float strength_joules, float radius) {
    Shockwave s;
    s.centre   = centre;
    s.strength = strength_joules;
    s.radius   = (radius > 1e-3f) ? radius : 1.0f;
    pending_shocks_.push_back(s);
}

void SmokeSPH::Step(float frame_dt) {
    // Initial allocation if Clear() wasn't called yet (e.g. a
    // freshly constructed solver).
    if (particles_.size() != max_particles_) {
        Reallocate();
    }
    if (frame_dt <= 0.0f) return;
    // No-op when there's nothing to integrate AND no emitters to
    // spawn new particles. Saves the spatial-hash rebuild in the
    // hot "smoke off" engine path.
    if (alive_count_ == 0 && emitters_.empty() && pending_shocks_.empty()) {
        return;
    }

    // Cap frame dt to avoid the death-spiral when frames spike.
    // 30 fps -> 1/30 s -> 4 substeps at the default 1/120 sdt.
    if (frame_dt > 1.0f / 30.0f) frame_dt = 1.0f / 30.0f;

    const float sdt = cfg_.substep_dt;
    int substeps    = static_cast<int>(std::ceil(frame_dt / sdt));
    if (substeps > cfg_.max_substeps_per_frame) {
        substeps = cfg_.max_substeps_per_frame;
    }
    if (substeps < 1) substeps = 1;

    for (int s = 0; s < substeps; ++s) {
        Substep(sdt);
    }
    // Shockwaves were applied + drained inside the first Substep call;
    // outer-level clear here is a defensive no-op for safety in case
    // a future substep early-out path leaves the queue populated.
    pending_shocks_.clear();
}

bool SmokeSPH::SpawnParticle(const Particle& p, std::size_t* out_index) {
    if (alive_count_ >= max_particles_) return false;
    // Find a free slot. The free-list logic would be a tighter loop
    // here but the particle count is in the low thousands; linear
    // scan is fine and keeps the data structure trivially mappable.
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) {
            particles_[i]   = p;
            alive_[i]       = 1;
            density_[i]     = cfg_.rest_density;
            pressure_[i]    = 0.0f;
            temperature_[i] = cfg_.ambient_temperature_kelvin;
            force_[i]       = glm::vec3(0.0f);
            ++alive_count_;
            if (out_index) *out_index = i;
            return true;
        }
    }
    return false;
}

std::uint32_t SmokeSPH::PackForGpu(float* dst_floats,
                                   std::uint32_t max_floats,
                                   float render_radius_scale,
                                   float render_density_scale) const {
    if (!dst_floats) return 0;
    std::uint32_t count = 0;
    constexpr std::uint32_t kFloatsPerParticle = 8;  // 2 float4s
    const std::uint32_t cap = max_floats / kFloatsPerParticle;
    // Reference lifetime for the per-particle age fade. Use the
    // first emitter's lifetime; falls back to 6 s when there are no
    // emitters.
    float lifetime_ref = 6.0f;
    if (!emitters_.empty() && emitters_[0].lifetime_s > 0.0f) {
        lifetime_ref = emitters_[0].lifetime_s;
    }
    for (std::size_t i = 0; i < particles_.size() && count < cap; ++i) {
        if (!alive_[i]) continue;
        const Particle& p = particles_[i];
        float* o = dst_floats + std::size_t(count) * kFloatsPerParticle;
        // v0.xyz = pos, v0.w = render radius
        o[0] = p.pos.x;
        o[1] = p.pos.y;
        o[2] = p.pos.z;
        o[3] = p.radius * render_radius_scale;
        // v1.x = render density: constant base scale * age-fade. The
        // SPH density ratio (rho/rho0) is dropped from the render
        // path: in practice the rest-density-clamped SPH only gives
        // ratio == 1 unless particles are highly compressed, which
        // we don't see in low-Re smoke. Pure constant-per-particle
        // density gives a more controllable, predictable splat.
        // Age fade applies a (1 - age/life)^1.5 cubic-ish curve so
        // particles wisp out at the end of their life rather than
        // popping off the buffer with full opacity.
        const float a_norm  = (lifetime_ref > 0.0f)
                              ? std::min(p.age / lifetime_ref, 1.0f) : 0.0f;
        const float fade    = std::pow(std::max(1.0f - a_norm, 0.0f), 1.5f);
        o[4] = render_density_scale * fade;
        o[5] = a_norm;
        o[6] = 0.0f;
        o[7] = 0.0f;
        ++count;
    }
    return count;
}

glm::vec3 SmokeSPH::ComputeCentreOfMass() const {
    glm::vec3 sum(0.0f);
    std::uint32_t n = 0;
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        sum += particles_[i].pos;
        ++n;
    }
    return (n > 0) ? (sum / static_cast<float>(n)) : glm::vec3(0.0f);
}

SmokeSPH::GridStats SmokeSPH::ComputeGridStats() const {
    GridStats s;
    s.live_particles = alive_count_;
    s.buckets        = bucket_count_;
    std::vector<std::uint32_t> pop(bucket_count_, 0);
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        const auto& p = particles_[i];
        const int gx  = FloorToInt(p.pos.x / cfg_.kernel_radius_h);
        const int gy  = FloorToInt(p.pos.y / cfg_.kernel_radius_h);
        const int gz  = FloorToInt(p.pos.z / cfg_.kernel_radius_h);
        ++pop[HashCell(gx, gy, gz)];
    }
    for (std::uint32_t p : pop) s.max_bucket_pop = std::max(s.max_bucket_pop, p);
    float sum_dens = 0.0f;
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (alive_[i]) sum_dens += density_[i];
    }
    s.mean_density = (alive_count_ > 0) ? (sum_dens / alive_count_) : 0.0f;
    return s;
}

// --- Internals --------------------------------------------------------

void SmokeSPH::Reallocate() {
    particles_.assign(max_particles_, Particle{});
    alive_.assign(max_particles_, 0);
    alive_count_ = 0;
    density_.assign(max_particles_, 0.0f);
    pressure_.assign(max_particles_, 0.0f);
    temperature_.assign(max_particles_, cfg_.ambient_temperature_kelvin);
    force_.assign(max_particles_, glm::vec3(0.0f));
    // Bucket count = next-prime( 2 * max_particles ). 2x oversample
    // keeps the average bucket population around 0.5 -> low collision
    // rate. Cap to a sane minimum so a 0-particle pool still hashes
    // without modulo-by-zero.
    const std::uint32_t target = std::max<std::uint32_t>(max_particles_ * 2u, 17u);
    bucket_count_ = NextPrime(target);
    bucket_first_.assign(bucket_count_, kEnd);
    bucket_next_.assign(max_particles_, kEnd);
}

void SmokeSPH::Substep(float sdt) {
    // 1. Spawn new particles from each emitter. Done first so
    //    freshly spawned particles see the SAME sub-step's pressure
    //    field on integration (they'll be at the rest density's
    //    pressure floor since they're isolated).
    RunEmitters(sdt);

    // 2. Build spatial hash.
    RebuildGrid();

    const float h   = cfg_.kernel_radius_h;
    const float h2  = h * h;
    float h_pow = h;
    for (int i = 1; i < 9; ++i) h_pow *= h;
    const float h9 = h_pow;
    const float h6 = h * h * h * h * h * h;

    // 3. Density pass. Each particle sums kernel contributions from
    //    its neighbours (including itself). Müller's poly6 kernel.
    std::vector<std::size_t> scratch;
    scratch.reserve(64);
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        scratch.clear();
        GatherNeighbours(i, scratch);
        float rho = 0.0f;
        for (std::size_t j : scratch) {
            const auto d = particles_[j].pos - particles_[i].pos;
            const float r2 = glm::dot(d, d);
            // Use a representative particle mass (per-emitter masses
            // differ but the difference is small; we approximate
            // with the lookup-time emitter mass if available).
            // particle_mass is stored on emitter; we cache the mean
            // mass on the particle by re-using temperature_-style
            // arrays -- but to keep the data layout flat we share
            // a single mean particle mass per emitter. Picking the
            // first emitter's mass is fine for the visual goal.
            float mass = 0.001f;
            if (!emitters_.empty()) mass = emitters_[0].particle_mass;
            rho += mass * kernel_poly6(r2, h2, h9);
        }
        density_[i] = std::max(rho, cfg_.rest_density);
        // Tait-style stiff EOS: p = k * (rho - rho0). Negative
        // pressures clamped at 0 to prevent attractive forces (the
        // classic SPH instability for low-density regions).
        const float dp = density_[i] - cfg_.rest_density;
        pressure_[i]   = (dp > 0.0f) ? (cfg_.pressure_stiffness * dp) : 0.0f;
    }

    // 4. Force pass. For each particle, accumulate:
    //   * pressure gradient toward neighbours (repulsion when over-
    //     compressed -> pushes particles apart).
    //   * viscosity laplacian (smoothing of velocity field; Müller
    //     formulation: F_visc = mu * sum_j (m_j / rho_j) * (v_j - v_i)
    //     * laplacian_W).
    //   * gravity (constant accel * mass; expressed as a force here).
    //   * Buoyancy: F_b = -rho_air * V * g * (T_air - T_p) / T_p,
    //     simplified as buoyancy_scale * (T_p - T_air) / T_air * (-g)
    //     so a hot particle gets a force opposite gravity proportional
    //     to its thermal excess. Per-particle so cool particles
    //     stop rising.
    //   * Wind drag: F_w = drag * (wind - v) -- drives velocity
    //     toward wind speed, the simplest way to make wind affect
    //     smoke without numerical instability from a direct force
    //     impulse.
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        scratch.clear();
        GatherNeighbours(i, scratch);
        glm::vec3 f_pressure(0.0f);
        glm::vec3 f_visc(0.0f);
        const float mass_i = (!emitters_.empty()) ? emitters_[0].particle_mass : 0.001f;
        for (std::size_t j : scratch) {
            if (j == i) continue;
            const auto d   = particles_[i].pos - particles_[j].pos;
            const float r2 = glm::dot(d, d);
            if (r2 >= h2 || r2 < 1e-8f) continue;
            const float r = std::sqrt(r2);
            // Pressure force (Müller-symmetric): per-j contribution
            // F = -m_j * (p_i + p_j) / (2 * rho_j) * grad_W(r,h).
            const float coef = kernel_spiky_grad_coef(r, h, h6);
            const glm::vec3 dir = d / r;
            const float rho_j = std::max(density_[j], cfg_.rest_density);
            f_pressure += -mass_i * ((pressure_[i] + pressure_[j]) / (2.0f * rho_j))
                           * coef * dir;
            // Viscosity (Müller): F = mu * m_j / rho_j * (v_j - v_i)
            // * laplacian_W.
            const float lap = kernel_visc_laplacian(r, h, h6);
            f_visc += cfg_.viscosity * (mass_i / rho_j) * (particles_[j].vel - particles_[i].vel) * lap;
        }
        // Gravity acts on the particle's mass.
        glm::vec3 f_gravity = cfg_.gravity * mass_i;
        // Buoyancy via simplified Boussinesq. F = -m * g * scale *
        // (T_p - T_air) / T_air. The sign flips so hot particles
        // rise against gravity; cool particles stop rising.
        const float t_p   = temperature_[i];
        const float t_air = std::max(cfg_.ambient_temperature_kelvin, 1.0f);
        const float t_rel = (t_p - t_air) / t_air;
        glm::vec3 f_buoy  = -cfg_.gravity * mass_i * cfg_.buoyancy_scale * t_rel;
        // Wind drag. The user-set wind drives particles toward the
        // wind speed via an aerodynamic drag formulation:
        // F = mass * k * (wind - v). The mass-proportional form means
        // dv/dt = k * (wind - v) is mass-independent, which matches
        // physical drag at low Reynolds numbers (Stokes drag has the
        // mass-canceling property and is the right limit for sub-mm
        // smoke particles in air).
        //
        // air_drag is an inverse time constant in 1/s. With the default
        // 0.4 a still particle reaches wind speed in ~2.5 s of e-fold,
        // matching observed real-air smoke response.
        glm::vec3 f_wind  = (cfg_.wind - particles_[i].vel) * cfg_.air_drag * mass_i;
        force_[i] = f_pressure + f_visc + f_gravity + f_buoy + f_wind;
    }

    // 5. Apply queued shockwaves. Single-shot impulse applied once
    //    per Step() (i.e. once across all substeps for the current
    //    frame). We drain the queue inside the first substep iteration
    //    by clearing pending_shocks_ here -- subsequent substeps in
    //    the same Step() see an empty queue and skip.
    //
    //    Energy-based impulse: dv = sqrt(2 * E / m) * (1 - dist/r_eff)
    //    falling off linearly to zero at the radius edge. Direction
    //    radial outward from the shock centre.
    if (!pending_shocks_.empty()) {
        const float mass_p = (!emitters_.empty()) ? emitters_[0].particle_mass : 0.001f;
        for (const auto& s : pending_shocks_) {
            for (std::size_t i = 0; i < particles_.size(); ++i) {
                if (!alive_[i]) continue;
                const auto d      = particles_[i].pos - s.centre;
                const float dist2 = glm::dot(d, d);
                const float r_eff = std::max(s.radius, 1e-3f);
                if (dist2 > r_eff * r_eff) continue;
                const float dist  = std::sqrt(std::max(dist2, 1e-6f));
                const glm::vec3 dir = d / dist;
                const float falloff = 1.0f - (dist / r_eff);
                const float speed   = std::sqrt(2.0f * std::max(s.strength, 0.0f)
                                                / std::max(mass_p, 1e-6f))
                                       * falloff;
                particles_[i].vel += dir * speed;
            }
        }
        pending_shocks_.clear();
    }

    // 6. Integrate. Semi-implicit Euler. v += a * dt; x += v * dt.
    //    Cap velocity magnitude so a bad pressure spike can't
    //    catapult particles to infinity -- 50 m/s is more than
    //    enough for any reasonable smoke fluid.
    const float v_cap = 50.0f;
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        const float mass_i = (!emitters_.empty()) ? emitters_[0].particle_mass : 0.001f;
        glm::vec3 a = force_[i] / std::max(mass_i, 1e-6f);
        particles_[i].vel += a * sdt;
        const float vlen = glm::length(particles_[i].vel);
        if (vlen > v_cap) {
            particles_[i].vel *= (v_cap / vlen);
        }
        particles_[i].pos += particles_[i].vel * sdt;
        // Age particle. Cool toward ambient.
        particles_[i].age += sdt;
        const float t_target = cfg_.ambient_temperature_kelvin;
        const float dT       = (temperature_[i] - t_target) * (1.0f - std::exp(-cfg_.thermal_decay * sdt));
        temperature_[i] -= dT;
    }

    // 7. Despawn aged particles. Use a single life check per
    //    emitter: a particle is despawned when its age exceeds the
    //    LARGEST emitter lifetime. The mapping particle->emitter is
    //    not stored (saves 4 B/particle) so we use the broad max.
    float max_life = 6.0f;
    for (const auto& e : emitters_) {
        max_life = std::max(max_life, e.lifetime_s);
    }
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        // Despawn on age, on far below-ground (y < -10 in metres), or
        // on far-from-origin (>1km). The world bounds keep stray
        // particles from blowing the spatial hash on a runaway wind
        // cvar.
        bool kill = false;
        if (particles_[i].age >= max_life) kill = true;
        if (particles_[i].pos.y < -10.0f) kill = true;
        const glm::vec3 absp = glm::abs(particles_[i].pos);
        if (absp.x > 1000.0f || absp.z > 1000.0f || absp.y > 1000.0f) kill = true;
        if (kill) {
            alive_[i] = 0;
            --alive_count_;
            particles_[i] = Particle{};
        }
    }

    time_seconds_ += sdt;
}

void SmokeSPH::RunEmitters(float sdt) {
    // Each emitter has a spawn rate (particles/sec). Accumulate
    // debt at rate * sdt; spawn whole particles when debt >= 1.
    // The fractional debt persists across substeps so emit timing
    // is consistent regardless of substep count.
    for (std::size_t e = 0; e < emitters_.size(); ++e) {
        EmitterParams& em = emitters_[e];
        if (em.spawn_rate <= 0.0f) continue;
        emit_debt_[e] += em.spawn_rate * sdt;
        while (emit_debt_[e] >= 1.0f && alive_count_ < max_particles_) {
            emit_debt_[e] -= 1.0f;
            // Spawn one particle at the emitter mouth. Sample inside
            // a sphere of radius em.radius * 0.5 with Poisson-disc
            // jitter (cheap approx: rejection sampling within unit
            // ball, scaled, capped to a few rejects).
            glm::vec3 jitter(0.0f);
            for (int t = 0; t < 4; ++t) {
                const float u1 = Lcg01(rng_state_[e]) * 2.0f - 1.0f;
                const float u2 = Lcg01(rng_state_[e]) * 2.0f - 1.0f;
                const float u3 = Lcg01(rng_state_[e]) * 2.0f - 1.0f;
                jitter = glm::vec3(u1, u2, u3);
                if (glm::dot(jitter, jitter) < 1.0f) break;
            }
            jitter *= 0.25f * em.radius;
            Particle p;
            p.pos    = em.pos + jitter;
            p.radius = em.particle_radius;
            // Initial velocity = emitter velocity + small radial
            // jitter so the column doesn't collapse to a perfect line.
            // Jitter kept small (0.05 m/s) so the column geometry stays
            // recognisable; viscosity + pressure broaden it from there.
            const float v_jit = 0.05f;
            glm::vec3 vjit(
                (Lcg01(rng_state_[e]) - 0.5f) * v_jit,
                (Lcg01(rng_state_[e]) - 0.5f) * v_jit,
                (Lcg01(rng_state_[e]) - 0.5f) * v_jit);
            p.vel = em.velocity + vjit;
            p.age = 0.0f;
            std::size_t idx = 0;
            if (!SpawnParticle(p, &idx)) break;
            // Override the SpawnParticle defaults: ambient -> emitter
            // emission temperature, plus optional initial density
            // offset for "thick smoke at source" emitters.
            temperature_[idx] = em.temperature_kelvin;
            density_[idx]     = cfg_.rest_density + em.initial_density_offset;
        }
        // If we hit the cap, drain the debt -- otherwise it just
        // monotonically grows when the pool is full.
        if (alive_count_ >= max_particles_) {
            emit_debt_[e] = std::min(emit_debt_[e], 1.0f);
        }
    }
}

void SmokeSPH::RebuildGrid() {
    if (bucket_first_.size() != bucket_count_) {
        bucket_first_.assign(bucket_count_, kEnd);
    } else {
        std::fill(bucket_first_.begin(), bucket_first_.end(), kEnd);
    }
    if (bucket_next_.size() != particles_.size()) {
        bucket_next_.assign(particles_.size(), kEnd);
    } else {
        std::fill(bucket_next_.begin(), bucket_next_.end(), kEnd);
    }
    const float h = cfg_.kernel_radius_h;
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        if (!alive_[i]) continue;
        const auto& p = particles_[i];
        const int gx  = FloorToInt(p.pos.x / h);
        const int gy  = FloorToInt(p.pos.y / h);
        const int gz  = FloorToInt(p.pos.z / h);
        const std::uint32_t hk = HashCell(gx, gy, gz);
        bucket_next_[i]    = bucket_first_[hk];
        bucket_first_[hk]  = static_cast<std::uint32_t>(i);
    }
}

void SmokeSPH::GatherNeighbours(std::size_t i, std::vector<std::size_t>& out) const {
    out.clear();
    const auto& p = particles_[i];
    const float h = cfg_.kernel_radius_h;
    const int gx0 = FloorToInt(p.pos.x / h);
    const int gy0 = FloorToInt(p.pos.y / h);
    const int gz0 = FloorToInt(p.pos.z / h);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const std::uint32_t hk = HashCell(gx0 + dx, gy0 + dy, gz0 + dz);
                std::uint32_t k = bucket_first_[hk];
                while (k != kEnd) {
                    out.push_back(k);
                    k = bucket_next_[k];
                }
            }
        }
    }
}

std::uint32_t SmokeSPH::HashCell(int gx, int gy, int gz) const {
    // The Müller hash mixes the three signed cell coords via large
    // primes XOR'd, then mods into the bucket table. The
    // static_cast<uint32_t>(signed_int) wrap is well-defined since
    // C++20 (two's-complement guarantee).
    const std::uint32_t ux = static_cast<std::uint32_t>(gx) * kHashPrimeX;
    const std::uint32_t uy = static_cast<std::uint32_t>(gy) * kHashPrimeY;
    const std::uint32_t uz = static_cast<std::uint32_t>(gz) * kHashPrimeZ;
    return (ux ^ uy ^ uz) % std::max<std::uint32_t>(bucket_count_, 1u);
}

}  // namespace pt::sph

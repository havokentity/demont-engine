// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// CPU path-tracer kernel for the Software backend. This is the C++
// equivalent of `PathTrace.slang::main` -- not a Slang-to-CPU emit
// (the project doesn't compile shaders to a CPU target), just a
// hand-written port covering the subset documented in
// SoftwareTracer.h. Embree handles the triangle-mesh intersection
// (mirroring `RayQuery::TraceRayInline` on GPU); analytic primitives
// are intersected with a linear loop.

#include "SoftwareTracer.h"
#include "SoftwareDevice.h"

#include "../core/Log.h"

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>    // std::max, std::min, std::clamp
#include <atomic>
#include <cmath>        // std::sqrt, std::fabs, std::pow, std::abs
#include <cstring>
#include <thread>
#include <vector>

namespace pt::rhi::sw {

namespace {

// Layout of the engine's PtPush struct (Engine.cpp::RenderFrame). Only
// the fields used by this kernel are listed; the trailing engine-only
// fields (curr/prev view-proj, sun direction, exposure pad, w2j
// rotation, DOF / vol / cloud parameters, etc.) are skipped via raw
// byte offsets when needed. Mirrors the host layout exactly --
// CMAKE_CXX_STANDARD 23 + this matching struct is byte-equivalent.
struct PtPushHead {
    float pos_fovtan[4];                // 16
    float fwd_aspect[4];                // 16
    float right_xyz[4];                 // 16
    float up_xyz[4];                    // 16
    std::uint32_t frame_index;          // 4
    std::uint32_t reset_accum;          // 4
    std::uint32_t max_bounces;          // 4
    std::uint32_t tlas_present;         // 4
    std::uint32_t prim_count;           // 4
    std::uint32_t spp;                  // 4
    std::uint32_t denoiser_enabled;     // 4
    std::uint32_t env_map_present;      // 4
    float halton_jitter[2];             // 8
    float env_intensity;                // 4
    float env_total_luminance;          // 4
    // Total: 112 bytes. Matches kPushSplitOffset on Vulkan.
};
static_assert(sizeof(PtPushHead) == 112, "PtPushHead must mirror engine's first 112B");

// Sun direction offset inside the engine's full PtPush. `sun_and_mode`
// sits at byte offset 240 (after the two 4x4 view-proj matrices that
// follow the head). Read defensively in case the engine pushes a
// smaller struct than expected (e.g. the Vulkan-SPIRV head-only push).
constexpr std::size_t kSunAndModeOffset = 240;

// Accumulator parameters offset inside the engine's full PtPush. The
// `accum_params` field sits at byte 720 (after the 4-uint header tail
// at 704: hdri_lights_count / write_normal_gbuffer / write_hdr_aux /
// mis_enabled).  .x is the r_accum_ema_alpha EMA-history-retention
// factor; .y/.z/.w reserved.  See PathTrace.slang's Push.accum_params
// + Engine.cpp's PtPush.accum_params for the canonical doc.
constexpr std::size_t kAccumParamsOffset = 720;

// Analytic prim layout (matches PathTrace.slang's `primitives` SSBO).
// Motion blur (#85): bumped from 3 float4 to 4 float4 to carry prev_pos
// alongside the existing per-prim fields. Software-backend renders here
// always use the CURRENT-frame position (v0.xyz); the host packs prev
// into v3.xyz for the GPU shader's shutter-time lerp but the SW path
// doesn't implement motion blur (no per-pixel shutter sampling -- the
// SW backend is the fallback for headless / no-GPU envs).
//
//   v0.xyz = sphere center or plane normal; v0.w = sphere radius or plane d
//   v1.rgb = albedo; v1.a = roughness
//   v2.x   = type (0 sphere / 1 plane); v2.y = material
//   v2.z   = ior; v2.w = pad
//   v3.xyz = prev_pos_or_n (sphere center at previous frame; same as
//            v0.xyz for planes). Ignored by the SW tracer (which always
//            samples at the current frame), but reserved here so the
//            stride matches PathTrace.slang.
// 4 float4 per prim = 16 floats = 64 bytes.
constexpr std::uint32_t kPrimSphere = 0u;
constexpr std::uint32_t kPrimPlane  = 1u;
constexpr std::uint32_t kMatLambert = 0u;
constexpr std::uint32_t kFloatsPerPrim = 20u;  // 5 float4s (motion blur v3 + emission v4)

struct HitInfo {
    bool      hit = false;
    float     t   = 1e30f;
    glm::vec3 normal{0, 1, 0};
    glm::vec3 albedo{0.5f};
    std::uint32_t mat = kMatLambert;
};

// Ray-sphere intersection (closest positive root). Returns hit-t in t
// and true if hit. Matches the GPU `intersectSphere` semantics.
inline bool IntersectSphere(const glm::vec3& ro, const glm::vec3& rd,
                            const glm::vec3& center, float radius, float& t) {
    glm::vec3 oc = ro - center;
    float b = glm::dot(oc, rd);
    float c = glm::dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return false;
    h = std::sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;
    if (t0 > 1e-4f) { t = t0; return true; }
    if (t1 > 1e-4f) { t = t1; return true; }
    return false;
}

// Ray-plane (n.p + d = 0).
inline bool IntersectPlane(const glm::vec3& ro, const glm::vec3& rd,
                           const glm::vec3& n, float d, float& t) {
    float denom = glm::dot(n, rd);
    if (std::fabs(denom) < 1e-6f) return false;
    float t_ = -(glm::dot(n, ro) + d) / denom;
    if (t_ > 1e-4f) { t = t_; return true; }
    return false;
}

// Procedural sky: a Preetham-lite analytic gradient. Lerps from
// horizon haze to zenith blue, plus a warm tint along the sun
// direction. Matches the rough look of PathTrace.slang's
// `procSky` fallback when no HDRI is bound.
inline glm::vec3 ProcSky(const glm::vec3& dir, const glm::vec3& sun_dir) {
    float t_y = glm::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    glm::vec3 horizon{0.55f, 0.62f, 0.70f};
    glm::vec3 zenith {0.18f, 0.32f, 0.55f};
    glm::vec3 sky    = glm::mix(horizon, zenith, std::pow(t_y, 0.7f));
    // Soft sun glow + small disc. Disc clamps so we don't blow out
    // the present blit's [0,1] clamp -- the present passes through
    // RGBA32F -> u8 conversion which clips highlights.
    float cos_sun = glm::clamp(glm::dot(dir, sun_dir), 0.0f, 1.0f);
    float halo = std::pow(cos_sun, 6.0f) * 0.5f;
    float disc = (cos_sun > 0.9998f) ? 2.5f : 0.0f;
    glm::vec3 sun_col{1.0f, 0.92f, 0.78f};
    return sky + sun_col * (halo + disc);
}

// ACES Narkowicz approximation, well-matched to the engine's
// shader-side tonemap. Followed by an sRGB OETF so the framebuffer
// values end up close to perceptually linear when the BGRA8 present
// blit truncates.
inline glm::vec3 AcesNarkowicz(glm::vec3 c) {
    constexpr float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
    return glm::clamp((c * (a * c + b)) / (c * (cc * c + d) + e),
                      glm::vec3(0.0f), glm::vec3(1.0f));
}
inline glm::vec3 SrgbOetf(glm::vec3 c) {
    auto enc = [](float x) {
        if (x <= 0.0031308f) return x * 12.92f;
        return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    };
    return glm::vec3{enc(c.r), enc(c.g), enc(c.b)};
}

// Single hit test against analytic prims + (optional) Embree TLAS.
// Closest-hit semantics.
void TraceScene(const glm::vec3& ro, const glm::vec3& rd,
                const float* prim_data, std::uint32_t prim_count,
                RTCScene tlas_scene,
                HitInfo& out) {
    out.hit = false;
    out.t   = 1e30f;

    // Analytic primitives.
    for (std::uint32_t i = 0; i < prim_count; ++i) {
        // Stride bumped from 12 floats (3 float4) to 16 floats (4 float4)
        // for motion blur (#85) -- v3 holds prev_pos for the GPU shader
        // but the SW tracer ignores it (no shutter sampling here).
        const float* v0 = prim_data + i * kFloatsPerPrim + 0u;
        const float* v1 = prim_data + i * kFloatsPerPrim + 4u;
        const float* v2 = prim_data + i * kFloatsPerPrim + 8u;
        std::uint32_t type = static_cast<std::uint32_t>(v2[0]);
        glm::vec3 albedo{v1[0], v1[1], v1[2]};
        std::uint32_t mat = static_cast<std::uint32_t>(v2[1]);
        float t = 0.0f;
        if (type == kPrimSphere) {
            glm::vec3 center{v0[0], v0[1], v0[2]};
            float r = v0[3];
            if (IntersectSphere(ro, rd, center, r, t) && t < out.t) {
                out.hit = true;
                out.t = t;
                glm::vec3 hit_pt = ro + rd * t;
                out.normal = glm::normalize(hit_pt - center);
                out.albedo = albedo;
                out.mat = mat;
            }
        } else if (type == kPrimPlane) {
            glm::vec3 n{v0[0], v0[1], v0[2]};
            float d = v0[3];
            if (IntersectPlane(ro, rd, n, d, t) && t < out.t) {
                out.hit = true;
                out.t = t;
                out.normal = n;
                out.albedo = albedo;
                out.mat = mat;
            }
        }
    }

    // Triangle meshes via Embree.
    if (tlas_scene != nullptr) {
        RTCRayHit rh{};
        rh.ray.org_x = ro.x; rh.ray.org_y = ro.y; rh.ray.org_z = ro.z;
        rh.ray.dir_x = rd.x; rh.ray.dir_y = rd.y; rh.ray.dir_z = rd.z;
        rh.ray.tnear = 1e-4f;
        rh.ray.tfar  = out.t;
        rh.ray.mask  = 0xFFFFFFFFu;
        rh.ray.flags = 0;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);
        rtcIntersect1(tlas_scene, &rh, &args);
        if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID && rh.ray.tfar < out.t) {
            out.hit = true;
            out.t   = rh.ray.tfar;
            glm::vec3 Ng{rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z};
            if (glm::length(Ng) > 1e-6f) {
                out.normal = glm::normalize(Ng);
            }
            // Default mesh material: gold metal (matches engine
            // default MESH_ALBEDO). The renderer treats it as a
            // bright Lambert in v1 since metal BRDF isn't wired
            // yet.
            out.albedo = glm::vec3{1.00f, 0.85f, 0.45f};
            out.mat    = kMatLambert;
        }
    }
}

}  // namespace

void RunPathTraceKernel(SoftwareDevice& device,
                        const SoftwareCommandBuffer& cmd) {
    if (cmd.push_constants_size < sizeof(PtPushHead)) {
        LOG_WARN("Software pathtrace: push size {} < expected head {}",
                 cmd.push_constants_size, sizeof(PtPushHead));
        return;
    }
    const PtPushHead* push = reinterpret_cast<const PtPushHead*>(cmd.push_constants_buf);

    // Sun direction lives past the SPIRV-prefix split. Read it iff
    // the engine pushed the full struct (which it does on both Metal
    // and the Vulkan-with-spilled-UBO path -- the byte layout
    // matches end-to-end at this offset).
    glm::vec3 sun_dir{0.4f, 0.8f, 0.3f};   // sensible default if push is too short
    if (cmd.push_constants_size >= kSunAndModeOffset + 16) {
        const float* sm = reinterpret_cast<const float*>(
            cmd.push_constants_buf + kSunAndModeOffset);
        glm::vec3 sd{sm[0], sm[1], sm[2]};
        if (glm::length(sd) > 1e-3f) sun_dir = glm::normalize(sd);
    } else {
        sun_dir = glm::normalize(sun_dir);
    }

    // Resolve bound resources. Output texture must exist; the accum
    // texture is optional but allocated by the engine in steady state
    // (binding 1). Writing to both lets the engine's screenshot
    // command read back the pre-tonemap HDR values from accum while
    // the present blit reads the LDR sRGB-encoded values from output.
    BackedTexture* output = device.GetTexture(TextureHandle{cmd.binds.textures[0]});
    if (output == nullptr || output->data.empty()) return;
    BackedTexture* accum = device.GetTexture(TextureHandle{cmd.binds.textures[1]});

    const std::uint32_t w = output->width;
    const std::uint32_t h = output->height;
    if (w == 0 || h == 0) return;

    const float* prim_data  = nullptr;
    std::uint32_t prim_count = 0;
    if (cmd.binds.buffers[3] != 0) {
        BackedBuffer* pb = device.GetBuffer(BufferHandle{cmd.binds.buffers[3]});
        if (pb != nullptr) {
            prim_data  = reinterpret_cast<const float*>(pb->data.data());
            prim_count = push->prim_count;
            // Don't trust prim_count past the buffer's capacity.
            // Motion blur (#85): prim stride is now 64 bytes (4 float4)
            // instead of 48 (3 float4) -- v3 carries prev_pos.
            constexpr std::size_t kBytesPerPrim = kFloatsPerPrim * sizeof(float);
            std::uint32_t max_prims = static_cast<std::uint32_t>(pb->size / kBytesPerPrim);
            if (prim_count > max_prims) prim_count = max_prims;
        }
    }

    RTCScene tlas_scene = nullptr;
    if (push->tlas_present != 0u && cmd.binds.accel_structs[2] != 0) {
        BackedAccel* a = device.GetAccel(AccelStructHandle{cmd.binds.accel_structs[2]});
        if (a != nullptr && a->is_tlas) tlas_scene = a->scene;
    }

    // Match the GPU paths' exposure handling. The engine writes the
    // current exposure scalar to exposure_state[0] (seeded by
    // r_exposure on_change when r_auto_exposure=0; the AutoExposure
    // dispatch updates it each frame when =1, but that dispatch is
    // a no-op on Software so manual mode is what works today). The
    // GPU shaders multiply HDR radiance by this value before
    // tonemapping; we mirror it so r_exposure changes track on
    // screen here too.
    float exposure = 1.0f;
    if (cmd.binds.buffers[6] != 0) {
        BackedBuffer* eb = device.GetBuffer(BufferHandle{cmd.binds.buffers[6]});
        if (eb != nullptr && eb->size >= sizeof(float)) {
            std::memcpy(&exposure, eb->data.data(), sizeof(float));
            if (!std::isfinite(exposure) || exposure <= 0.0f) exposure = 1.0f;
        }
    }

    // EMA history-retention factor (r_accum_ema_alpha).  0 = legacy
    // online running mean; >0 = EMA blend with effective window
    // ~1/(1-alpha).  Read defensively -- engine may push less than the
    // full struct on the Vulkan-SPIRV split-push path, in which case
    // we fall back to the running-mean default which is correct.
    float ema_alpha = 0.0f;
    if (cmd.push_constants_size >= kAccumParamsOffset + 16) {
        const float* ap = reinterpret_cast<const float*>(
            cmd.push_constants_buf + kAccumParamsOffset);
        ema_alpha = glm::clamp(ap[0], 0.0f, 0.999f);
    }
    const std::uint32_t reset_accum = push->reset_accum;
    const bool accum_active = (accum != nullptr
                               && accum->width  == w
                               && accum->height == h);
    // One-shot diagnostic so the user can see whether the engine
    // bound a TLAS at all. Cheap because it only fires on the first
    // dispatch and reads atomics; subsequent frames skip the log.
    {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            LOG_INFO("Software tracer first dispatch: tlas_present={} bound_accel[2]={} resolved_scene={} prims={}",
                     push->tlas_present, cmd.binds.accel_structs[2],
                     static_cast<const void*>(tlas_scene), prim_count);
        }
    }

    // Camera basis from the push struct.
    const glm::vec3 cam_pos {push->pos_fovtan[0], push->pos_fovtan[1], push->pos_fovtan[2]};
    const float fov_tan     = push->pos_fovtan[3];
    const glm::vec3 cam_fwd  {push->fwd_aspect[0], push->fwd_aspect[1], push->fwd_aspect[2]};
    const float aspect      = push->fwd_aspect[3];
    const glm::vec3 cam_right{push->right_xyz[0], push->right_xyz[1], push->right_xyz[2]};
    const glm::vec3 cam_up   {push->up_xyz[0],    push->up_xyz[1],    push->up_xyz[2]};

    const float inv_w = 1.0f / static_cast<float>(w);
    const float inv_h = 1.0f / static_cast<float>(h);

    // Parallelise across rows. The CPU work per pixel is dominated by
    // Embree's TLAS traversal; std::thread + naive row chunking is
    // simpler than enkiTS for this v1, and the work-per-row is large
    // enough that thread overhead is negligible.
    // Leave a couple of cores free for the engine's JobSystem (the
    // CSG bake worker, mesh upload paths, etc.). Otherwise the path
    // tracer pegs every core every frame and async work in the engine
    // never gets a chance to run -- visible as "CSG: rebuilt mesh"
    // never firing, so the mesh BLAS / TLAS never gets built and the
    // Embree path stays silent. Cap at hw - 4 with a min of 1.
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned thread_count = (hw > 4u) ? (hw - 4u) : 1u;
    std::atomic<std::uint32_t> next_row{0};
    auto worker = [&]() {
        while (true) {
            std::uint32_t y = next_row.fetch_add(1u, std::memory_order_relaxed);
            if (y >= h) break;
            float* row = &output->data[std::size_t(y) * w * 4];
            for (std::uint32_t x = 0; x < w; ++x) {
                // NDC -> view-space (Y flipped so screen-up = world-up).
                float ndc_x = (static_cast<float>(x) + 0.5f) * inv_w * 2.0f - 1.0f;
                float ndc_y = 1.0f - (static_cast<float>(y) + 0.5f) * inv_h * 2.0f;
                float vx = ndc_x * aspect * fov_tan;
                float vy = ndc_y * fov_tan;
                glm::vec3 rd = glm::normalize(cam_fwd + cam_right * vx + cam_up * vy);
                glm::vec3 ro = cam_pos;

                HitInfo h0;
                TraceScene(ro, rd, prim_data, prim_count, tlas_scene, h0);

                glm::vec3 col{0.0f};
                if (!h0.hit) {
                    col = ProcSky(rd, sun_dir);
                } else {
                    // Lambert direct lighting from the procedural sun.
                    // Shadow test: trace from a slightly-offset hit
                    // point along sun_dir against the same scene
                    // (re-using TraceScene closest-hit). t < tnear
                    // means clear sky to the sun.
                    glm::vec3 hit_pt = ro + rd * h0.t;
                    // Normal facing the incoming ray.
                    glm::vec3 nf = (glm::dot(h0.normal, rd) < 0.0f)
                        ? h0.normal : -h0.normal;
                    float n_dot_l = std::max(0.0f, glm::dot(nf, sun_dir));
                    glm::vec3 ambient = h0.albedo * ProcSky(nf, sun_dir) * 0.30f;

                    glm::vec3 lit{0.0f};
                    if (n_dot_l > 0.0f) {
                        HitInfo shadow;
                        TraceScene(hit_pt + nf * 1e-3f, sun_dir,
                                    prim_data, prim_count, tlas_scene, shadow);
                        if (!shadow.hit) {
                            glm::vec3 sun_rad{2.4f, 2.2f, 1.9f};
                            lit = h0.albedo * sun_rad * n_dot_l;
                        }
                    }
                    col = ambient + lit;
                }
                // Sanitize the new sample: NaN / Inf / negative
                // channels become 0.  Without this a single bad sample
                // (env.pdf underflow, sqrt of slightly-negative
                // numerical residue, etc.) injects NaN into accum and
                // poisons that pixel forever -- the running-mean update
                // `avg = (prev*n + new) / n_new` propagates NaN through
                // prev across every subsequent frame.  Matches
                // PathTrace.slang's sanitize-before-accum block.
                glm::vec3 frame_rad = col;
                if (!std::isfinite(frame_rad.r) || frame_rad.r < 0.0f) frame_rad.r = 0.0f;
                if (!std::isfinite(frame_rad.g) || frame_rad.g < 0.0f) frame_rad.g = 0.0f;
                if (!std::isfinite(frame_rad.b) || frame_rad.b < 0.0f) frame_rad.b = 0.0f;

                // Accumulator update.  When the engine bound an accum
                // texture (steady state on every frame), do a proper
                // running-mean / EMA update matching the GPU shader's
                // logic in PathTrace.slang.  When no accum is bound
                // (very first frame before EnsureAccumTexture, or a
                // legacy code path), skip the accum and tonemap the
                // per-frame radiance directly -- same as session 1+2
                // behaviour.
                glm::vec3 avg = frame_rad;
                if (accum_active) {
                    float* arow = &accum->data[std::size_t(y) * w * 4];
                    // Read prev (or zero out for fresh epoch).
                    glm::vec4 prev{0.0f};
                    if (reset_accum == 0u) {
                        prev = glm::vec4(arow[x * 4 + 0], arow[x * 4 + 1],
                                          arow[x * 4 + 2], arow[x * 4 + 3]);
                        // Sanitize prev (handle legacy-NaN imports).
                        if (!std::isfinite(prev.x) || !std::isfinite(prev.y) ||
                            !std::isfinite(prev.z) || !std::isfinite(prev.w)) {
                            prev = glm::vec4{0.0f};
                        }
                    }
                    float n_new;
                    if (prev.w < 0.5f || ema_alpha <= 0.0f) {
                        // First-sample-of-epoch OR legacy running mean
                        // (ema_alpha == 0).  prev.a == 0 seeds cleanly;
                        // ema_alpha == 0 keeps unbounded online avg.
                        n_new = prev.w + 1.0f;
                        avg = (glm::vec3(prev) * prev.w + frame_rad) / n_new;
                    } else {
                        // EMA blend with retention factor ema_alpha.
                        // Count is informational only (capped to avoid
                        // float-precision drift over long runs).
                        avg = glm::vec3(prev) * ema_alpha
                            + frame_rad   * (1.0f - ema_alpha);
                        n_new = std::min(prev.w + 1.0f,
                                          1.0f / std::max(1.0f - ema_alpha, 1e-3f));
                    }
                    arow[x * 4 + 0] = avg.r;
                    arow[x * 4 + 1] = avg.g;
                    arow[x * 4 + 2] = avg.b;
                    arow[x * 4 + 3] = n_new;
                }

                // Tonemap from the accumulated mean, not the raw
                // per-frame radiance.  Static-camera convergence
                // matches the GPU backends: noise falls as 1/sqrt(N)
                // for running-mean mode, floor at the EMA variance
                // floor for EMA mode.
                glm::vec3 tonemapped = SrgbOetf(AcesNarkowicz(avg * exposure));
                row[x * 4 + 0] = tonemapped.r;
                row[x * 4 + 1] = tonemapped.g;
                row[x * 4 + 2] = tonemapped.b;
                row[x * 4 + 3] = 1.0f;
            }
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (unsigned t = 0; t < thread_count; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
}

}  // namespace pt::rhi::sw

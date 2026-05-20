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
// Stride grew over time: 3 -> 4 in #85 (motion blur, added prev_pos)
// -> 5 in #181 polish (added per-prim emission). The software backend
// reads v0 (current-frame center), v1 (albedo), v2 (type/material),
// and v4 (emission). v3 (prev_pos) is reserved for the shader's
// shutter-time lerp; SW path doesn't implement motion blur (no per-
// pixel shutter sampling -- SW is the fallback for headless / no-GPU
// envs) so it skips v3.
//
//   v0.xyz = sphere center or plane normal; v0.w = sphere radius or plane d
//   v1.rgb = albedo; v1.a = roughness
//   v2.x   = type (0 sphere / 1 plane); v2.y = material
//   v2.z   = ior; v2.w = pad
//   v3.xyz = prev_pos_or_n (motion blur, #85; ignored here)
//   v4.xyz = emission (W/sr per channel, #181 polish); v4.w = pad
//   v5.xyzw = orient quaternion (rotation gizmo, #206)
//   v6.xyzw = PBR texture tile indices (albedo/normal/rough/metallic,
//             Wave 8 #26; ignored here -- the SW fallback doesn't sample
//             textures, so textured prims render with their flat v1
//             albedo on this path)
// 7 float4 per prim = 28 floats = 112 bytes. The stride MUST stay in
// sync with Engine.cpp's kFloatsPerPrim (28) and PathTrace.slang's
// testAnalyticPrim (reads at idx*7 stride; Wave 8 #26 bumped 6->7).
constexpr std::uint32_t kPrimSphere = 0u;
constexpr std::uint32_t kPrimPlane  = 1u;
constexpr std::uint32_t kMatLambert = 0u;
constexpr std::uint32_t kFloatsPerPrim = 28u;

struct HitInfo {
    bool      hit = false;
    float     t   = 1e30f;
    glm::vec3 normal{0, 1, 0};
    glm::vec3 albedo{0.5f};
    std::uint32_t mat = kMatLambert;
    // Per-prim emission (#181 polish). Zero on non-emissive prims;
    // testScene-equivalent adds this to the radiance accumulator before
    // BRDF eval, same as the GPU shader path. Mesh / non-analytic hits
    // overwrite to zero so a previous prim's emission doesn't leak.
    glm::vec3 emission{0.0f};
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

// Rotate a vector by a unit quaternion (xyzw). Mirrors the
// PathTrace.slang quatRotate helper -- standard q*(0,v)*q^-1 in
// cross-product form. Identity (0,0,0,1) is a no-op so untouched
// planes render bit-identically through the SW tracer.
inline glm::vec3 QuatRotateVec(const float* q4, const glm::vec3& v) {
    glm::vec3 u{q4[0], q4[1], q4[2]};
    float w = q4[3];
    return v + 2.0f * glm::cross(u, glm::cross(u, v) + w * v);
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

// --- Wave 9 tonemap (#27 follow-up) ----------------------------------------
// Host-side mirrors of the AgX + Khronos PBR Neutral operators the GPU
// paths apply (shaders/PathTrace.slang + shaders/Tonemap.slang). The
// software backend is a crude preview tracer, so its absolute pixels never
// matched the GPU paths -- but it MUST still honour r_tonemap_op so the
// cvar isn't silently a no-op on this backend (the engine's "no demo
// shortcuts" rule). Constants are kept identical to the shader copies.
inline glm::vec3 AgxTonemap(glm::vec3 c) {
    auto contrast = [](glm::vec3 x) {
        glm::vec3 x2 = x * x;
        glm::vec3 x4 = x2 * x2;
        return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4
             - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
    };
    const glm::vec3 ir0{0.842479062253094f,  0.0784335999999992f, 0.0792237451477643f};
    const glm::vec3 ir1{0.0423282422610123f, 0.878468636469772f,  0.0791661274605434f};
    const glm::vec3 ir2{0.0423756549057051f, 0.0784336f,          0.879142973793104f};
    glm::vec3 v{glm::dot(ir0, c), glm::dot(ir1, c), glm::dot(ir2, c)};

    const float kMinEv = -12.47393f, kMaxEv = 4.026069f;
    v = glm::clamp(glm::vec3{std::log2(std::max(v.x, 1e-10f)),
                             std::log2(std::max(v.y, 1e-10f)),
                             std::log2(std::max(v.z, 1e-10f))},
                   glm::vec3(kMinEv), glm::vec3(kMaxEv));
    v = (v - kMinEv) / (kMaxEv - kMinEv);
    v = contrast(v);

    const glm::vec3 kLuma{0.2126f, 0.7152f, 0.0722f};
    float luma = glm::dot(v, kLuma);
    v = luma + (v - luma);  // sat = 1, slope = 1, power = 1, offset = 0

    const glm::vec3 or0{ 1.19687900512017f,  -0.0980208811401368f, -0.0990297440797205f};
    const glm::vec3 or1{-0.0528968517574562f, 1.15190312990417f,   -0.0989611768448433f};
    const glm::vec3 or2{-0.0529716355144438f,-0.0980434501171241f,  1.15107367264116f};
    v = glm::vec3{glm::dot(or0, v), glm::dot(or1, v), glm::dot(or2, v)};
    return glm::clamp(v, glm::vec3(0.0f), glm::vec3(1.0f));
}
inline glm::vec3 KhronosPbrNeutralTonemap(glm::vec3 c) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation     = 0.15f;
    float x = std::min(c.r, std::min(c.g, c.b));
    float offset = (x < 0.08f) ? (x - 6.25f * x * x) : 0.04f;
    c -= offset;
    float peak = std::max(c.r, std::max(c.g, c.b));
    if (peak < startCompression) return c;
    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    c *= newPeak / peak;
    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return glm::mix(c, glm::vec3(newPeak), g);
}
// Dispatch on the r_tonemap_op enum (0 aces / 1 agx / 2 khronos_pbr_neutral
// / 3 reinhard / 4 linear). op 0 returns AcesNarkowicz(c), byte-identical
// to the previous unconditional ACES path.
inline glm::vec3 TonemapDispatch(glm::vec3 c, std::uint32_t op) {
    switch (op) {
        case 1u: return AgxTonemap(c);
        case 2u: return KhronosPbrNeutralTonemap(c);
        case 3u: return glm::clamp(c / (1.0f + c), glm::vec3(0.0f), glm::vec3(1.0f));
        case 4u: return glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
        default: return AcesNarkowicz(c);
    }
}
// --- end Wave 9 tonemap ----------------------------------------------------

// Single hit test against analytic prims + (optional) Embree TLAS.
// Closest-hit semantics.
void TraceScene(const glm::vec3& ro, const glm::vec3& rd,
                const float* prim_data, std::uint32_t prim_count,
                RTCScene tlas_scene,
                HitInfo& out) {
    out.hit = false;
    out.t   = 1e30f;

    // Analytic primitives. Stride is `kFloatsPerPrim` = 28 (7 float4s
    // per prim) -- see the layout comment at the top of this file.
    // v3 = prev_pos (motion blur, #85; SW path skips it). v4 carries
    // per-prim emission (#181 polish). v5 carries the orientation
    // quaternion (#206); spheres ignore it, planes use it to rotate
    // the stored normal at intersect time. v6 carries PBR texture tile
    // indices (Wave 8 #26); the SW fallback doesn't sample textures.
    for (std::uint32_t i = 0; i < prim_count; ++i) {
        const float* v0 = prim_data + i * kFloatsPerPrim + 0u;
        const float* v1 = prim_data + i * kFloatsPerPrim + 4u;
        const float* v2 = prim_data + i * kFloatsPerPrim + 8u;
        // v3 (prev_pos for motion blur) intentionally unread -- SW
        // tracer always samples at the current frame's position.
        const float* v4 = prim_data + i * kFloatsPerPrim + 16u;
        const float* v5 = prim_data + i * kFloatsPerPrim + 20u;
        std::uint32_t type = static_cast<std::uint32_t>(v2[0]);
        glm::vec3 albedo{v1[0], v1[1], v1[2]};
        glm::vec3 emission{v4[0], v4[1], v4[2]};
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
                out.emission = emission;
            }
        } else if (type == kPrimPlane) {
            // Rotate the stored normal by the orientation quaternion.
            // With the default identity quat (0,0,0,1) QuatRotateVec
            // returns v0.xyz exactly, so untouched planes are bit-
            // identical to the pre-#206 SW tracer output.
            glm::vec3 n_raw{v0[0], v0[1], v0[2]};
            glm::vec3 n = QuatRotateVec(v5, n_raw);
            float d = v0[3];
            if (IntersectPlane(ro, rd, n, d, t) && t < out.t) {
                out.hit = true;
                out.t = t;
                out.normal = n;
                out.albedo = albedo;
                out.mat = mat;
                out.emission = emission;
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
            // Mesh path doesn't source emission -- clear so a prior
            // analytic-prim hit's emission doesn't leak through when
            // this mesh hit is closer. Mirrors the GPU testAnalyticPrim
            // -> mesh-hit chain in PathTrace.slang.
            out.emission = glm::vec3{0.0f};
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
            // Don't trust prim_count past the buffer's capacity. The
            // per-prim byte size MUST match Engine.cpp's kBytesPerPrim:
            //   kFloatsPerPrim * sizeof(float) = 28 * 4 = 112 bytes,
            // after #85 (motion blur, v3=prev_pos) + #181 polish
            // (emission, v4) + #206 (orient quat, v5) + Wave 8 #26 (PBR
            // texture tiles, v6) grew the stride from 48 -> 64 -> 80 ->
            // 96 -> 112.
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

    // --- Wave 9 tonemap (#27 follow-up) ---
    // r_tonemap_op enum, packed by the engine into PtPush::tonemap_params --
    // the LAST 16-byte block of the struct. The software backend always
    // receives the full Metal-style push (no Vulkan split), so tonemap_params
    // .x sits at push_constants_size - 16. Read defensively: if the push is
    // somehow short, fall back to op 0 (aces) so a malformed dispatch can't
    // reach into out-of-range bytes. Mirrors the GPU paths' tonemap_params.x.
    std::uint32_t tonemap_op = 0u;
    if (cmd.push_constants_size >= 16) {
        std::memcpy(&tonemap_op,
                    cmd.push_constants_buf + cmd.push_constants_size - 16,
                    sizeof(std::uint32_t));
        if (tonemap_op > 4u) tonemap_op = 0u;
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
                    // Per-prim emission (#181 polish). Mirrors the GPU
                    // hit-handler at PathTrace.slang:~3850: emission adds
                    // directly to the radiance accumulator before the
                    // BRDF eval. Zero on non-emissive prims and on mesh
                    // hits (TraceScene resets emission on the mesh
                    // branch by overwrite-from-default, which is 0).
                    col = ambient + lit + h0.emission;
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
                // --- Wave 9 tonemap --- operator from r_tonemap_op
                // (tonemap_op == 0 -> AcesNarkowicz, unchanged default).
                glm::vec3 tonemapped =
                    SrgbOetf(TonemapDispatch(avg * exposure, tonemap_op));
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

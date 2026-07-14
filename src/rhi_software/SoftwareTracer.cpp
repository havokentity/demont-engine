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
#include "engine/HosekSkyModel.h"   // Wave 9 hosek-sky: real ArHosekSkyModel cook

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

// Accumulator parameters offset inside the engine's full PtPush.
// `accum_params` sits at byte 784 in the current layout -- it moved
// from 720 as clouds_p4 (#117), spectral_params (#27),
// dof_bokeh_params (wave 10) and the write_albedo/pad churn were each
// inserted ahead of it, and this raw-byte mirror silently kept reading
// hdri_lights_col[6] until the offset was re-derived (offsetof probe
// against the PtPush definition, cross-checked by the static_assert
// beside PtPush in Engine.cpp).  .x is the r_accum_ema_alpha EMA-
// history-retention factor; .y/.z/.w reserved.  See PathTrace.slang's
// Push.accum_params + Engine.cpp's PtPush for the canonical doc.
constexpr std::size_t kAccumParamsOffset = 784;

// r_tonemap_op offset inside the engine's full PtPush. tonemap_params
// is NOT the last 16-byte block anymore -- sun_extra2 (#239) was
// appended after it, so the old `push_constants_size - 16` read
// decoded sun_extra2.x (a float, default 1.0f = 0x3F800000) as the
// tonemap enum and the >4 clamp silently forced ACES for every
// operator on the software backend. Fixed offset, asserted in
// lockstep with the struct by Engine.cpp's static_assert block.
constexpr std::size_t kTonemapParamsOffset = 1600;

// Sky mode is packed into sun_and_mode.w (float), i.e. the 4th float of
// the sun_and_mode vec4. 3 == Hosek-Wilkie. See PtPush::sun_and_mode.
constexpr std::size_t kSkyModeOffset = kSunAndModeOffset + 12;

// Hosek-Wilkie params offset inside the engine's full PtPush. hosek_params
// sits at byte 1184 (asserted in lockstep by Engine.cpp's static_assert
// block). .x = turbidity [1,10], .y = ground albedo [0,1], .z = solar
// zenith angle (rad). The software backend re-cooks the real Hosek RGB
// dataset from these three uniforms (see engine/HosekSkyModel.h) so the
// CI-exercised software golden actually renders the Hosek dome instead of
// the generic ProcSky fallback.
constexpr std::size_t kHosekParamsOffset = 1184;

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
//             Wave 8 #26). uint reinterpreted as float; kPbrNoTexTile
//             (0xFFFFFFFF) on a channel means "no map, use the flat v1
//             value". Sampled by ApplyPbrTextures against the material
//             atlas bound at engine texture slot 16 -- see that helper.
// 7 float4 per prim = 28 floats = 112 bytes. The stride MUST stay in
// sync with Engine.cpp's kFloatsPerPrim (28) and PathTrace.slang's
// testAnalyticPrim (reads at idx*7 stride; Wave 8 #26 bumped 6->7).
constexpr std::uint32_t kPrimSphere = 0u;
constexpr std::uint32_t kPrimPlane  = 1u;
constexpr std::uint32_t kMatLambert = 0u;
constexpr std::uint32_t kMatMetal   = 1u;   // MAT_METAL in PathTrace.slang
constexpr std::uint32_t kFloatsPerPrim = 28u;

// Wave 8 PBR (#26) material texture atlas geometry. kPbrTileSize MUST match
// Engine.h's kPbrTileSize and PathTrace.slang's copy: the atlas is a
// vertical strip of kPbrTileSize-square tiles (kPbrTileSize wide, a whole
// number of tiles tall). A per-map tile index selects the vertical band;
// kPbrNoTexTile marks an unassigned map. The tile *count* is derived from
// the bound atlas' height at sample time (see ApplyPbrTextures) rather than
// hard-coded, so it tracks whatever the host allocated (Engine's
// kPbrAtlasTiles).
constexpr std::uint32_t kPbrTileSize  = 256u;
constexpr std::uint32_t kPbrNoTexTile = 0xFFFFFFFFu;

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
    // Wave 8 PBR (#26): base roughness + procedural surface UV + the four
    // per-map atlas tile indices, populated on analytic-prim hits so
    // ApplyPbrTextures can sample the material maps. tex_tiles default to
    // kPbrNoTexTile so an untextured hit (and every mesh hit) short-
    // circuits the sampler to a no-op, keeping flat-material renders
    // bit-identical.
    float         roughness = 1.0f;
    glm::vec2     uv{0.0f};
    std::uint32_t tex_tiles[4] = {kPbrNoTexTile, kPbrNoTexTile,
                                  kPbrNoTexTile, kPbrNoTexTile};
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

// Hosek-Wilkie physically-based sky dome (r_sky_mode = hosek). The CPU
// mirror of PathTrace.slang::hosekSky: evaluates the per-direction shape
// function F(theta, gamma) from the cooked RGB coefficients and multiplies
// by the channel radiance L_M (already scaled). theta = angle from zenith,
// gamma = angle from the sun. Returns HDR linear radiance (the caller
// applies r_exposure + tonemap, same as every other sky path). No BSC
// starmap here -- the software backend is a headless preview tracer and
// binds no star texture; the night floor stands in for the star field.
inline glm::vec3 HosekSky(const glm::vec3& dir, const glm::vec3& sun_dir,
                          const pt::hosek::Cooked& ck) {
    const float cos_theta = glm::clamp(dir.y, 0.0f, 1.0f);
    const float cos_gamma = glm::clamp(glm::dot(dir, sun_dir), -1.0f, 1.0f);
    const float gamma     = std::acos(cos_gamma);
    glm::vec3 sky{0.0f};
    for (int c = 0; c < 3; ++c) {
        const double F = pt::hosek::RadianceInternal(ck.cfg[c], cos_theta,
                                                     cos_gamma, gamma);
        sky[c] = static_cast<float>(std::max(F, 0.0) * ck.rad[c]
                                    * pt::hosek::kHosekRadianceScale);
    }
    // Fade the model out just under the horizon (F rings negative below it),
    // then blend to a deep-navy night floor as the sun drops -- both mirror
    // the shader so the software preview tracks the GPU look.
    sky *= glm::smoothstep(-0.05f, 0.02f, dir.y);
    const float day = glm::smoothstep(-0.10f, 0.20f, sun_dir.y);
    sky = glm::mix(glm::vec3{0.005f, 0.010f, 0.030f}, sky, day);
    return glm::max(sky, glm::vec3(0.0f));
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
    // Linearize out of AgX's base encoding (pow 2.2, per Wrensch's
    // agxEotf / Filament) -- kept byte-identical with the Slang copies.
    v = glm::pow(glm::max(v, glm::vec3(0.0f)), glm::vec3(2.2f));
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

// --- Wave 8 PBR (#26) -- material texture atlas sampling -------------------
// CPU mirror of PathTrace.slang's samplePbrTileLinear / samplePbrTileSrgb /
// applyNormalMap / applyPbrTextures. The atlas is a vertical strip of
// kPbrTileSize-square tiles stored RGBA8 on the GPU; on this backend
// SoftwareDevice::WriteTexture normalises those bytes to [0,1] in the
// BackedTexture float backing, so sampling here reads the exact same values
// the GPU's rgba8 fetch returns. Filtering is manual bilinear over integer
// texel fetches with a per-tile frac() wrap so tiling maps repeat
// seamlessly. The sRGB EOTF is applied ONLY to albedo (SamplePbrTileSrgb);
// normal / roughness / metallic are linear data read raw -- so we never
// double-decode.

// sRGB (IEC 61966-2-1) EOTF: encoded -> linear. Matches PathTrace.slang's
// srgbToLinear exactly (the inverse of the OETF the present blit applies).
inline glm::vec3 SrgbToLinear(glm::vec3 c) {
    auto dec = [](float x) {
        return (x <= 0.04045f)
                 ? (x / 12.92f)
                 : std::pow(std::max((x + 0.055f) / 1.055f, 0.0f), 2.4f);
    };
    return glm::vec3{dec(c.r), dec(c.g), dec(c.b)};
}

// Fetch one atlas texel (row-major RGBA32F backing). Callers pass already-
// wrapped, in-range integer coordinates.
inline glm::vec4 AtlasTexel(const BackedTexture* atlas, int x, int y) {
    const std::size_t idx =
        (std::size_t(y) * atlas->width + std::size_t(x)) * 4u;
    return glm::vec4{atlas->data[idx + 0], atlas->data[idx + 1],
                     atlas->data[idx + 2], atlas->data[idx + 3]};
}

// Bilinear-fetch one atlas tile at the (wrapped) tile-local UV. Returns the
// raw RGBA (no colour-space conversion). `tile` MUST be valid (caller gates
// on kPbrNoTexTile). Mirrors samplePbrTileLinear including the modulo wrap
// on the +1 tap so the tile edge samples its opposite edge (seamless tile).
inline glm::vec4 SamplePbrTileLinear(const BackedTexture* atlas,
                                     std::uint32_t tile, glm::vec2 uv) {
    const glm::vec2 fuv = glm::fract(uv);   // wrap into [0,1); handles < 0
    const int W = static_cast<int>(kPbrTileSize);
    const float fx  = fuv.x * static_cast<float>(kPbrTileSize) - 0.5f;
    const float fy  = fuv.y * static_cast<float>(kPbrTileSize) - 0.5f;
    const float fx0 = std::floor(fx);
    const float fy0 = std::floor(fy);
    const float tx  = fx - fx0;
    const float ty  = fy - fy0;
    int ix0 = static_cast<int>(fx0);
    int iy0 = static_cast<int>(fy0);
    int ix1 = ((ix0 + 1) % W + W) % W;
    int iy1 = ((iy0 + 1) % W + W) % W;
    ix0 = (ix0 % W + W) % W;
    iy0 = (iy0 % W + W) % W;
    const int row_base = static_cast<int>(tile) * static_cast<int>(kPbrTileSize);
    const glm::vec4 c00 = AtlasTexel(atlas, ix0, row_base + iy0);
    const glm::vec4 c10 = AtlasTexel(atlas, ix1, row_base + iy0);
    const glm::vec4 c01 = AtlasTexel(atlas, ix0, row_base + iy1);
    const glm::vec4 c11 = AtlasTexel(atlas, ix1, row_base + iy1);
    const glm::vec4 cx0 = glm::mix(c00, c10, tx);
    const glm::vec4 cx1 = glm::mix(c01, c11, tx);
    return glm::mix(cx0, cx1, ty);
}

// Albedo sample: bilinear fetch + sRGB->linear decode.
inline glm::vec3 SamplePbrTileSrgb(const BackedTexture* atlas,
                                   std::uint32_t tile, glm::vec2 uv) {
    return SrgbToLinear(glm::vec3(SamplePbrTileLinear(atlas, tile, uv)));
}

// Duff et al. 2017 branchless orthonormal basis: perturb the geometric
// normal by a tangent-space normal-map sample in [-1,1]. Mirrors
// applyNormalMap (the analytic-prim path has no per-vertex tangents, so
// this arbitrary-but-stable frame stands in, same as the GPU).
inline glm::vec3 ApplyNormalMap(glm::vec3 n, glm::vec3 ts) {
    const float s = (n.z >= 0.0f) ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float b = n.x * n.y * a;
    const glm::vec3 t {1.0f + s * n.x * n.x * a, s * b,               -s * n.x};
    const glm::vec3 bt{b,                        s + n.y * n.y * a,   -n.y};
    const glm::vec3 perturbed = ts.x * t + ts.y * bt + ts.z * n;
    return glm::normalize(perturbed);
}

// Apply a hit's PBR material maps in place, mirroring
// PathTrace.slang::applyPbrTextures. Albedo is sRGB-decoded and multiplies
// the flat tint; roughness / metallic are linear. The SW tracer's Lambert
// shading consumes albedo + the (perturbed) shading normal -- so those two
// maps drive this backend's textured render -- while roughness and the
// metallic->metal switch are sampled + stored for parity with the GPU
// HitInfo but do not alter the diffuse output today (the same parity gap
// as the SW tracer treating mesh "gold metal" as bright Lambert). No-op
// when the atlas is unbound or every tile is kPbrNoTexTile, so untextured
// hits stay bit-identical.
inline void ApplyPbrTextures(HitInfo& h, const BackedTexture* atlas) {
    if (atlas == nullptr || atlas->data.empty() ||
        atlas->width != kPbrTileSize ||
        atlas->height < kPbrTileSize) {
        return;
    }
    // Number of whole tiles that fit the bound atlas. A per-channel gate of
    // `tile < n_tiles` both skips the kPbrNoTexTile sentinel (0xFFFFFFFF is
    // never < n_tiles) and clamps any out-of-range index -- the GPU fetch
    // wraps/clamps in hardware, but a raw CPU index would walk off the
    // atlas backing, so we must bound it here. For valid host data (every
    // assigned tile within the allocated atlas) this is identical to the
    // shader's `!= kPbrNoTexTile` gate.
    const std::uint32_t n_tiles = atlas->height / kPbrTileSize;
    // Albedo (sRGB -> linear, multiplies the flat tint so a white flat
    // albedo passes the texture through unchanged).
    if (h.tex_tiles[0] < n_tiles) {
        h.albedo *= SamplePbrTileSrgb(atlas, h.tex_tiles[0], h.uv);
    }
    // Roughness (linear; .r scales the flat roughness).
    if (h.tex_tiles[2] < n_tiles) {
        const float r = SamplePbrTileLinear(atlas, h.tex_tiles[2], h.uv).r;
        h.roughness = glm::clamp(h.roughness * r, 0.0f, 1.0f);
    }
    // Metallic (linear; .r > 0.5 flips a Lambert material to metal, parity
    // with the GPU path -- inert for the SW diffuse shading today).
    if (h.tex_tiles[3] < n_tiles) {
        const float m = SamplePbrTileLinear(atlas, h.tex_tiles[3], h.uv).r;
        if (m > 0.5f && h.mat == kMatLambert) h.mat = kMatMetal;
    }
    // Normal map LAST (tangent-space [0,1] -> [-1,1]). Perturb after the
    // metallic / roughness reads so those used the geometric-normal UV (the
    // UV is normal-independent anyway).
    if (h.tex_tiles[1] < n_tiles) {
        const glm::vec3 ts =
            glm::vec3(SamplePbrTileLinear(atlas, h.tex_tiles[1], h.uv)) * 2.0f - 1.0f;
        h.normal = ApplyNormalMap(h.normal, ts);
    }
}
// --- end Wave 8 PBR --------------------------------------------------------

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
    // indices (Wave 8 #26), read below and sampled by ApplyPbrTextures.
    //
    // uint tile index recovered from its float-bit pattern (the host packs
    // it via memcpy in EnsurePrimitivesUploaded; the shader recovers it via
    // asuint()). kPbrNoTexTile (0xFFFFFFFF) on a channel means "no map".
    auto f2u = [](float f) {
        std::uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    for (std::uint32_t i = 0; i < prim_count; ++i) {
        const float* v0 = prim_data + i * kFloatsPerPrim + 0u;
        const float* v1 = prim_data + i * kFloatsPerPrim + 4u;
        const float* v2 = prim_data + i * kFloatsPerPrim + 8u;
        // v3 (prev_pos for motion blur) intentionally unread -- SW
        // tracer always samples at the current frame's position.
        const float* v4 = prim_data + i * kFloatsPerPrim + 16u;
        const float* v5 = prim_data + i * kFloatsPerPrim + 20u;
        const float* v6 = prim_data + i * kFloatsPerPrim + 24u;
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
                out.roughness = v1[3];
                // Wave 8 PBR (#26): procedural equirectangular lat/long UV
                // (u from atan2, seam at -x; v from asin, 0 south -> 1
                // north), matching testAnalyticPrim's sphere mapping.
                const glm::vec3 sn = out.normal;
                out.uv = glm::vec2(
                    0.5f + std::atan2(sn.z, sn.x) * (0.5f / glm::pi<float>()),
                    0.5f - std::asin(glm::clamp(sn.y, -1.0f, 1.0f)) / glm::pi<float>());
                out.tex_tiles[0] = f2u(v6[0]);
                out.tex_tiles[1] = f2u(v6[1]);
                out.tex_tiles[2] = f2u(v6[2]);
                out.tex_tiles[3] = f2u(v6[3]);
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
                out.roughness = v1[3];
                // Wave 8 PBR (#26): procedural planar UV from the world hit
                // point, projected onto the two axes most orthogonal to the
                // plane normal (1 UV unit = 1 metre; the atlas frac()-wraps
                // per tile). Matches testAnalyticPrim; +y floor -> (x, z).
                const glm::vec3 hp = ro + rd * t;
                const glm::vec3 an = glm::abs(n);
                glm::vec2 puv;
                if (an.y >= an.x && an.y >= an.z)  puv = glm::vec2(hp.x, hp.z);
                else if (an.x >= an.z)             puv = glm::vec2(hp.z, hp.y);
                else                               puv = glm::vec2(hp.x, hp.y);
                out.uv = puv;
                out.tex_tiles[0] = f2u(v6[0]);
                out.tex_tiles[1] = f2u(v6[1]);
                out.tex_tiles[2] = f2u(v6[2]);
                out.tex_tiles[3] = f2u(v6[3]);
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
            // No per-triangle material maps on the SW mesh path -- clear
            // the PBR tile indices / UV / roughness so a nearer mesh hit
            // doesn't inherit a previously-tested analytic prim's texture
            // state (ApplyPbrTextures then no-ops on this hit).
            out.roughness = 1.0f;
            out.uv        = glm::vec2(0.0f);
            out.tex_tiles[0] = out.tex_tiles[1] =
            out.tex_tiles[2] = out.tex_tiles[3] = kPbrNoTexTile;
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

    // Sky mode + Hosek cook. Decode sun_and_mode.w (== 3 for Hosek-Wilkie)
    // and, when active, re-cook the real ArHosekSkyModel RGB dataset from
    // (turbidity, ground albedo, solar zenith) in hosek_params -- ONCE here,
    // before the per-pixel loop, since the cook is a per-frame uniform. This
    // is what makes the CI-exercised software golden actually exercise the
    // Hosek path (it previously ignored r_sky_mode and pinned a near-black
    // ProcSky gradient). Cook clamps below-horizon suns; the night floor in
    // HosekSky handles the rest.
    bool hosek_active = false;
    pt::hosek::Cooked hosek_ck{};
    if (cmd.push_constants_size >= kSkyModeOffset + 4) {
        float sky_mode_f = 0.0f;
        std::memcpy(&sky_mode_f, cmd.push_constants_buf + kSkyModeOffset, sizeof(float));
        if (static_cast<int>(sky_mode_f + 0.5f) == 3 &&
            cmd.push_constants_size >= kHosekParamsOffset + 16) {
            const float* hp = reinterpret_cast<const float*>(
                cmd.push_constants_buf + kHosekParamsOffset);
            const float turbidity = hp[0];
            const float albedo    = hp[1];
            const float zenith    = hp[2];
            const float sun_elev  = 1.5707963267948966f - zenith;
            hosek_ck = pt::hosek::Cook(turbidity, albedo, sun_elev);
            hosek_active = true;
        }
    }
    // Unified sky lookup used at primary miss AND for the sky-lit ambient
    // term, so the Lambert floor under a Hosek sky is lit by Hosek skylight
    // (the env-lighting consistency the fixture checks).
    auto sky_radiance = [&](const glm::vec3& d) -> glm::vec3 {
        return hosek_active ? HosekSky(d, sun_dir, hosek_ck)
                            : ProcSky(d, sun_dir);
    };

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

    // Wave 8 PBR (#26): the material texture atlas, bound by the engine at
    // texture slot 16 (BindStorageTexture(16, pbr_atlas)) only once the
    // scene loads a texture. Null until then; ApplyPbrTextures no-ops when
    // it's unbound or a hit carries no tile indices, so untextured scenes
    // are unaffected. Read-only during the render -- worker threads sample
    // it concurrently.
    const BackedTexture* pbr_atlas =
        device.GetTexture(TextureHandle{cmd.binds.textures[16]});

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
    // r_tonemap_op enum, packed by the engine into PtPush::tonemap_params
    // at a fixed offset (see kTonemapParamsOffset -- it stopped being the
    // last block when sun_extra2 landed). Read defensively: if the push is
    // somehow short, fall back to op 0 (aces) so a malformed dispatch can't
    // reach into out-of-range bytes. Mirrors the GPU paths' tonemap_params.x.
    std::uint32_t tonemap_op = 0u;
    if (cmd.push_constants_size >= kTonemapParamsOffset + 16) {
        std::memcpy(&tonemap_op,
                    cmd.push_constants_buf + kTonemapParamsOffset,
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
                    col = sky_radiance(rd);
                } else {
                    // Wave 8 PBR (#26): apply the hit's material texture
                    // maps (albedo sRGB-decode + normal-map perturbation
                    // drive this diffuse backend) before shading, so all
                    // lighting below sees the textured albedo / normal.
                    // No-op for untextured hits and when the atlas is
                    // unbound. Primary hit only -- shadow rays need just
                    // occlusion, so they skip this like the GPU path.
                    ApplyPbrTextures(h0, pbr_atlas);
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
                    glm::vec3 ambient = h0.albedo * sky_radiance(nf) * 0.30f;

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

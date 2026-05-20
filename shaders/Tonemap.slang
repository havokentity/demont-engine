// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Post-denoise tonemap.
//
// Reads a per-pixel linear HDR colour (the MetalFX output, RGBA16F)
// and writes the ACES-tonemapped sRGB result to the swapchain. Sits
// between MetalFX and present in the denoiser-on path; the
// denoiser-off path bypasses this kernel and tonemaps inline in
// PathTrace.slang's swapchain write (the tradeoff: a one-pass write
// when not denoising, two passes when denoising in HDR).
//
// Bindings (Slang assigns Metal texture(0..1) in declaration order):
//   slot 0 (texture) : hdr_in -- RGBA16F linear HDR from MetalFX
//   slot 1 (texture) : ldr_out -- swapchain.
//                       - On Metal: BGRA8Unorm_sRGB, linear->sRGB
//                         encode is IMPLICIT on store.
//                       - On Vulkan/SPIR-V: BGRA8_UNORM, the encode is
//                         EXPLICIT via PT_SRGB_ENCODE() applied just
//                         before store (see the swapchain-write site
//                         inside the kernel).  Both branches land at
//                         the same on-screen colour space.

// MSL SLOT ALIGNMENT NOTICE -- mirrors AutoExposure.slang's notice.
// Slang's MSL backend assigns Metal buffer/texture slots in
// DECLARATION order. The engine's MetalCommandBuffer::Dispatch
// computes push_slot = max(bound_buf_slot)+1, which on every dispatch
// after PathTrace lands at buf(7) (PathTrace binds slots 1..6, AS at
// buffer slot 0). For the engine's setBytes(push, push_slot=7) call
// to actually reach the kernel's Push cbuffer, every kernel's MSL
// Push must be at buffer(7). Without padding, Tonemap.slang's
// declaration order would emit Push at buffer(1) -- engine's setBytes
// goes to buf(7), Tonemap's kernel reads buf(1) which is whatever
// PathTrace had bound there (box_vbuf), and the entire push struct
// is garbage. exposure_state was also reading buf(0) which gets
// silently overwritten by setAccelerationStructure(0).
//
// Fix: pad declaration order to match PathTrace's emission so Push
// lands at buffer(7) for ALL kernels. Dummies referenced in a
// runtime-conditional dead branch so Slang's optimizer keeps them.

// vk::image_format qualifiers: hdr_in is the RGBA16F denoise_color
// (the path tracer's HDR output texture for the post-denoise chain);
// ldr_out is the BGRA8_UNORM swapchain (rgba8 in SPIR-V image-format
// terms -- Slang/SPIR-V don't distinguish RGBA vs BGRA channel order
// for storage-image writes). Without the qualifier Slang emits
// Format=Unknown which NVIDIA's compiler treats as UB; see the fix-up
// PR for the chain of upstream bugs (DXC #6153 / Slang #5969 /
// NVIDIA driver 596.x).
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_in;
[[vk::binding(1, 0), vk::image_format("rgba8")]]   RWTexture2D<float4> ldr_out;
// Half-res linear HDR bloom layer. The engine builds this each frame
// via a downsample/upsample pyramid (BloomDown.slang + BloomUp.slang)
// from the bright pixels of hdr_in. Sampled here with bilinear and
// added pre-tonemap so it gets ACES-curve-compressed alongside the
// rest of the image. Always bound; the engine zero-fills it (or
// passes a 1x1 dummy) when r_bloom = 0 and bloom_intensity = 0.
//
// Bound at engine slot 2 -> shader vk::binding 6 on Vulkan (a storage
// image slot in the shared 17-binding pipeline layout) -- so this
// pipeline can re-use the same descriptor set layout the path tracer
// builds. On Metal, Slang assigns texture indices in declaration
// order regardless of the vk::binding number, so the kernel still
// sees bloom_tex at [[texture(2)]] which matches the engine's
// BindStorageTexture(2, ...) call.
[[vk::binding(6, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> bloom_tex;

// Star-split accumulator was bound here for the EMA-based design in
// #108. The stateless StarsComposite kernel (issue #46) runs UPSTREAM
// of this kernel (between Denoise() and the bloom pyramid), writing
// stars + sun + moon into the same HDR source bloom_tex's downsample
// chain reads from. So by the time Tonemap.slang runs, hdr_in already
// carries celestials and bloom_tex carries halos around them -- no
// separate stars binding needed here.

// --- buffer slot padding to push exposure_state to MSL buf(6) and
//     Push (cbuffer) to MSL buf(7) ----------------------------------
//
// Gated to Metal only: on Vulkan these dummy bindings would collide
// with the shared 17-binding pipeline layout. MSL slot alignment is
// the only reason these exist; SPIR-V doesn't need them.
//
// vk::binding numbers picked from the 18..23 range -- intentionally
// out-of-band vs the shared layout (bindings 0..17) so even if Slang
// ever starts cross-checking vk::binding on the MSL path, none of
// these collide with a real Vulkan binding. The attribute itself is
// ignored by Slang's MSL backend (Metal slot assignment is by
// declaration order: _slot_tlas -> buf(0), _slot_mesh_positions ->
// buf(1), ..., _slot_conditional_cdf -> buf(5), exposure_state below
// -> buf(6), Push -> buf(7)).
#ifdef PT_TARGET_METAL
[[vk::binding(18, 0)]] RaytracingAccelerationStructure _slot_tlas;        // buf(0)
[[vk::binding(19, 0)]] StructuredBuffer<float>  _slot_mesh_positions;     // buf(1)
[[vk::binding(20, 0)]] StructuredBuffer<uint>   _slot_mesh_indices;       // buf(2)
[[vk::binding(21, 0)]] StructuredBuffer<float4> _slot_primitives;         // buf(3)
[[vk::binding(22, 0)]] StructuredBuffer<float>  _slot_marginal_cdf;       // buf(4)
[[vk::binding(23, 0)]] StructuredBuffer<float>  _slot_conditional_cdf;    // buf(5)
#endif

// GPU-resident exposure scalar. Same buffer PathTrace.slang reads in
// its inline tonemap path, written every frame by AutoExposure.slang
// (or seeded by the engine's r_exposure on_change hook in manual
// mode). Reading here means Mac's post-pass tonemap tracks GPU-side
// auto-exposure exactly the way the inline tonemap does on Vulkan,
// without an exposure value round-tripping through the host. Lands
// at MSL buf(6) thanks to the padding above.
[[vk::binding(15, 0)]] RWStructuredBuffer<float> exposure_state;

// Physical-flare ghost record. 32 bytes -- matches the C++ struct in
// Engine.cpp (TonePush::PtShaderGhost). Per-channel scale factor
// (positions are sun_uv_centred * scale per channel), 0..1 base
// intensity, and the disc radius in pixels.
struct GhostShader {
    // (scale_r, scale_g, scale_b, intensity)
    float4 sci;
    // (radius_px, _, _, _)
    float4 rpad;
};

// Maximum ghosts the shader array can hold. MUST match
// pt::engine::lensflare::kMaxGhosts in LensFlare.h.
static const uint kPhysGhostMax = 16;

[[vk::push_constant]]
cbuffer Push {
    float exposure;
    uint  passthrough;
    float bloom_intensity;
    // Lens flare. Samples the bloom mip at mirrored-across-centre
    // screen positions to fake internal lens reflections. 0 = no
    // flare (skip the loop).
    float flare_intensity;
    // Per-channel offset between R/G/B sample scales -- creates the
    // chromatic-aberration rainbow fringe along ghost edges. 0 = no
    // dispersion (achromatic ghosts). Real range 0.0..0.05.
    float flare_dispersion;
    // Number of ghost reflections (1..16, capped at min(physical_ghost_count, flare_count) in physical mode).
    uint  flare_count;
    // (image mode only) Per-ghost luminance gate against the
    // bloom-mip sample. 0 = no gate.
    float flare_threshold;
    // 1 = 'sun' mode: project the sun's screen UV (passed in
    // sun_uv) and draw discrete soft Gaussian-disc ghosts at
    // its mirrored positions. Clean, no scene-mirror artefacts.
    // 0 = 'image' mode: mirror-sample the bloom mip at every
    // output pixel (the legacy code path).
    uint  flare_mode_sun;
    // (sun mode) base Gaussian disc radius in normalised screen
    // units. Each ghost scales this by its own size factor so
    // mirrors-near-centre are tighter than mirrors-far-out.
    float flare_size;
    // Sun's screen UV (computed engine-side from sun_dir + camera
    // basis). Sentinel (-2, -2) means sun is behind / off-screen
    // so the flare loop is skipped.
    float _pad_sun_align;
    float2 sun_uv;
    // 'physical' flare mode: gather per-channel Gaussian splats from
    // ghosts[]. Each ghost's screen UV is sun_uv_centred * scale
    // (offset from screen centre); the per-channel scales encode
    // chromatic dispersion derived from the lens elements' Abbe
    // numbers. Set when r_lens_flare_mode = "physical".
    uint  flare_mode_physical;
    // Number of valid entries in ghosts[]. 0 = skip the physical loop.
    uint  physical_ghost_count;
    // --- Wave 9 tonemap (#27 follow-up) ---
    // tonemap_op: r_tonemap_op enum (0 aces / 1 agx / 2 khronos_pbr_neutral
    // / 3 reinhard / 4 linear). Reclaimed from `_reserved_stars_present`
    // (one of the two reserved uints left behind by the EMA-based star_split
    // design #108). Stays a uint at the SAME byte offset so TonePush::ghosts
    // remains at offset 112 -- the Vulkan kPushSplitOffset boundary that
    // splits push constants (first 112 B) from the Frame UBO ghost-array
    // tail. The post-denoise pass reads this at all three tonemap call
    // sites below; op 0 (aces) is byte-identical to the previous behaviour.
    uint  tonemap_op;
    // Reserved (4 bytes). The other half of the old #108 star_split pad;
    // kept so ghosts[] stays at offset 112 across both backends.
    uint  _reserved_stars_align;
    // --- end Wave 9 tonemap ---
    // --- Wave 10 bloom/bokeh ---
    // Mirrors the three reclaimed scalars at the head of the host
    // TonePush's old pad run (Engine.cpp). bloom_strength is a final
    // gain on the bloom composite (1.0 == the pre-wave-10 look);
    // lens_dirt gates the procedural dirt overlay; lens_dirt_strength
    // is its 0..1 mix. These three scalars occupy host bytes [64..75]
    // (the back half of cbuffer register 4) -- one float slot (offset
    // 76) plus the two float4s below pad ghosts[] back to offset 112.
    float bloom_strength;       // host offset 64
    uint  lens_dirt;            // host offset 68
    float lens_dirt_strength;   // host offset 72
    float _pad_w10_tail;        // host offset 76 -- fills cbuffer register 4
    // --- end Wave 10 bloom/bokeh ---
    // 32 bytes of padding (two float4s at host offsets 80 + 96) to push
    // the ghost array forward to host offset 112 (the Vulkan
    // kPushSplitOffset boundary). On Vulkan only the first 112 bytes hit
    // vkCmdPushConstants; bytes [112..624] (the ghost array) spill into
    // the Frame UBO declared at vk::binding(14, 0) below. On Metal the
    // whole 624-byte block goes through setBytes -- the padding is wasted
    // but keeps a single host-side TonePush layout shared across both
    // backends.
    float4 _pad_to_push_split[2];
#ifndef PT_TARGET_SPIRV
    // Metal path: ghosts live in the push struct.
    GhostShader ghosts[kPhysGhostMax];
#endif
};

// SPIR-V path: ghosts live in the Frame UBO at binding 14. The
// VulkanCommandBuffer's per-dispatch ring-UBO write copies the host
// TonePush's tail (bytes [kPushSplitOffset..push_size]) into this
// buffer, so `ghosts[i]` here reads the same data the Metal path
// reads from the push struct's tail.
#ifdef PT_TARGET_SPIRV
[[vk::binding(14, 0)]]
cbuffer Frame {
    GhostShader ghosts[kPhysGhostMax];
};
#endif

float3 acesTonemap(float3 c) {
    // Narkowicz fit. Same constants as PathTrace.slang -- duplicated
    // rather than shared so this kernel stays a single self-contained
    // TU. Drift between the two would manifest as a visible diff
    // between denoiser-on and denoiser-off, so keep the magic numbers
    // identical to the path-tracer copy.
    const float a = 2.51, b = 0.03, ck = 2.43, d = 0.59, e = 0.14;
    return saturate((c * (a * c + b)) / (c * (ck * c + d) + e));
}

// --- Wave 9 tonemap (#27 follow-up) ---------------------------------------
// Modern display-transform operators, selectable via the r_tonemap_op cvar
// (host packs the enum into TonePush::tonemap_op). VERBATIM duplicate of the
// helpers in PathTrace.slang's inline tonemap -- duplicated rather than
// shared for the same reason acesTonemap above is: this kernel is a single
// self-contained TU, and any drift between the two copies would show as a
// visible A/B diff between the denoiser-on (this pass) and denoiser-off
// (inline) paths. Keep the constants identical to the PathTrace.slang copy.
//
// Operator enum (mirror of pt::engine ParseTonemapOp + the kTonemap*
// constants in PathTrace.slang):
//   0 = aces / 1 = agx / 2 = khronos_pbr_neutral / 3 = reinhard / 4 = linear
static const uint kTonemapAces     = 0u;
static const uint kTonemapAgx      = 1u;
static const uint kTonemapKhronos  = 2u;
static const uint kTonemapReinhard = 3u;
static const uint kTonemapLinear   = 4u;

// AgX -- "Minimal AgX" (Benjamin Wrensch fit of Troy Sobotka's AgX), the
// same approximation Filament ships. See PathTrace.slang for the per-stage
// derivation. Input scene-linear (post-exposure); output display-linear
// sRGB (the OETF is applied at the swapchain-write site, as for ACES).
float3 agxDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return  + 15.5     * x4 * x2
            - 40.14    * x4 * x
            + 31.96    * x4
            - 6.868    * x2 * x
            + 0.4298   * x2
            + 0.1191   * x
            - 0.00232;
}
float3 agxTonemap(float3 c) {
    const float3 ir0 = float3(0.842479062253094,  0.0784335999999992, 0.0792237451477643);
    const float3 ir1 = float3(0.0423282422610123, 0.878468636469772,  0.0791661274605434);
    const float3 ir2 = float3(0.0423756549057051, 0.0784336,          0.879142973793104);
    float3 v = float3(dot(ir0, c), dot(ir1, c), dot(ir2, c));

    const float kMinEv = -12.47393;
    const float kMaxEv =   4.026069;
    v = clamp(log2(max(v, 1e-10)), kMinEv, kMaxEv);
    v = (v - kMinEv) / (kMaxEv - kMinEv);

    v = agxDefaultContrastApprox(v);

    const float3 kLuma = float3(0.2126, 0.7152, 0.0722);
    float luma = dot(v, kLuma);
    const float3 offset = float3(0.0);
    const float3 slope  = float3(1.0);
    const float3 power  = float3(1.0);
    const float  sat    = 1.0;
    v = pow(max(v * slope + offset, float3(0.0)), power);
    v = luma + sat * (v - luma);

    const float3 or0 = float3( 1.19687900512017,  -0.0980208811401368, -0.0990297440797205);
    const float3 or1 = float3(-0.0528968517574562, 1.15190312990417,   -0.0989611768448433);
    const float3 or2 = float3(-0.0529716355144438,-0.0980434501171241,  1.15107367264116);
    v = float3(dot(or0, v), dot(or1, v), dot(or2, v));
    return saturate(v);
}

// Khronos PBR Neutral -- the official glTF-Sample-Viewer neutral tone
// mapper. Preserves material albedo + hue; compresses only the bright tail
// and desaturates toward white near the ceiling. See PathTrace.slang copy.
float3 khronosPbrNeutralTonemap(float3 c) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation     = 0.15;

    float x = min(c.r, min(c.g, c.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    c -= offset;

    float peak = max(c.r, max(c.g, c.b));
    if (peak < startCompression) return c;

    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    c *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(c, float3(newPeak), g);
}

// Dispatch on the operator enum. op == 0 returns exactly acesTonemap(c) so
// the default post-denoise path stays byte-identical to pre-Wave-9.
float3 tonemapDispatch(float3 c, uint op) {
    switch (op) {
        case kTonemapAgx:      return agxTonemap(c);
        case kTonemapKhronos:  return khronosPbrNeutralTonemap(c);
        case kTonemapReinhard: return saturate(c / (1.0 + c));
        case kTonemapLinear:   return saturate(c);
        default:               return acesTonemap(c);   // kTonemapAces
    }
}
// --- end Wave 9 tonemap ---------------------------------------------------

// sRGB OETF (linear -> nonlinear) applied at swapchain-write sites on
// the SPIR-V path. Mirror of PathTrace.slang's helper -- see that
// file's long comment for the full reasoning. Short version: Vulkan
// storage-image writes do NOT auto-encode sRGB even with
// VK_FORMAT_B8G8R8A8_SRGB (only color-attachment writes do per spec),
// so the encode happens here in the shader. Metal stays unchanged:
// Apple silicon's BGRA8Unorm_sRGB storage write encodes implicitly.
#if defined(PT_TARGET_SPIRV)
float3 srgb_oetf(float3 c) {
    c = max(c, float3(0.0, 0.0, 0.0));
    float3 lin  = c * 12.92;
    float3 expo = 1.055 * pow(c, float3(1.0/2.4)) - 0.055;
    return select(c < 0.0031308, lin, expo);
}
#define PT_SRGB_ENCODE(x) srgb_oetf(x)
#else
#define PT_SRGB_ENCODE(x) (x)
#endif

// --- Wave 10 bloom/bokeh: procedural lens dirt ----------------------------
// A screen-space grime field standing in for the smudges, dust and oil
// haze on a real camera's front element. Returns a per-pixel MULTIPLIER
// centred near 1.0: <1 where the glass is relatively clean, >1 on bright
// smudges that catch and smear the bloom. The host gates this entirely
// behind r_lens_dirt (lens_dirt == 1), so when it's off the bloom add is
// byte-identical to the pre-wave-10 composite -- no goldens move.
//
// Built from three layers, all analytic (no texture binding -> no churn
// to the shared Vulkan descriptor layout, and a clean checkout
// reproduces it bit-for-bit):
//   * value-noise smudges  -- low-frequency soft oily blobs
//   * dust specks          -- high-frequency thresholded bright dots
//   * radial edge sheen     -- dirt pools toward the frame edge, and
//                              grazing light there scatters more
// uv is normalised [0,1] screen space; aspect corrects the noise lattice
// so smudges stay round on non-square framebuffers.
float lens_dirt_hash(float2 p) {
    // Cheap 2D value hash (same family as PathTrace.slang's pcg-ish
    // float hashes). Deterministic across backends.
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}
float lens_dirt_value_noise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);   // smoothstep interpolant
    float a = lens_dirt_hash(i + float2(0.0, 0.0));
    float b = lens_dirt_hash(i + float2(1.0, 0.0));
    float c = lens_dirt_hash(i + float2(0.0, 1.0));
    float d = lens_dirt_hash(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}
float lens_dirt_mask(float2 uv, float aspect) {
    // Aspect-correct lattice coords so blobs/specks stay round.
    float2 p = float2(uv.x * aspect, uv.y);

    // Smudge layer: two octaves of value noise. Centred on 0 by the
    // -0.5 bias so it pushes the multiplier both up and down around 1.
    float smudge = lens_dirt_value_noise(p * 6.0) * 0.6
                 + lens_dirt_value_noise(p * 13.0) * 0.4;
    smudge = smudge - 0.5;                       // [-0.5, 0.5]

    // Dust specks: sparse bright pinpoints. Threshold a high-frequency
    // hash so only the top few percent of cells light up, then sharpen.
    float speck_cells = lens_dirt_hash(floor(p * 60.0));
    float speck = saturate((speck_cells - 0.90) / 0.10);
    speck = speck * speck;                        // tighten the dots

    // Radial edge sheen: grime accumulates toward the frame edge and
    // grazing rays there scatter more, so bias the multiplier up with
    // the squared distance from screen centre.
    float2 d = uv - 0.5;
    float r2 = dot(d, d) * 4.0;                    // ~1 at the frame edge
    float edge = r2 * r2;                          // soft, edge-weighted

    // Compose into a multiplier around 1.0. Smudges modulate +/-, specks
    // and edge sheen only add. Clamp so a dark smudge can dim bloom
    // toward (but not past) zero and a bright speck can roughly double it.
    // The smudge gain (1.3) is the dominant character -- it gives the
    // patchy "wiped but not clean" look on the bloom halos.
    float m = 1.0 + smudge * 1.3 + speck * 1.6 + edge * 0.6;
    return clamp(m, 0.0, 2.5);
}
// --- end Wave 10 bloom/bokeh: procedural lens dirt ------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    ldr_out.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;

    // Force Slang to emit the slot-padding dummy bindings (the
    // optimizer would otherwise strip unused declarations and
    // exposure_state would land at MSL buf(0) again, breaking the
    // engine's push_slot=7 contract -- see the slot-alignment
    // notice at the top of the file).  The condition `dim.x ==
    // 0xDEADBEEFu` is uniform (every thread reads the same
    // swapchain width) so the branch is either taken by the whole
    // dispatch or by none of it; in practice never, because dim.x
    // is the real texture width.  Slang can't statically prove
    // that at compile time, so the body survives optimisation and
    // every referenced declaration gets a real slot assignment.
    // Metal-only: the dummy bindings are only declared on Metal,
    // and SPIR-V doesn't need the alignment trick anyway.
#ifdef PT_TARGET_METAL
    if (dim.x == 0xDEADBEEFu) {
        let p = _slot_mesh_positions[0];
        let i = _slot_mesh_indices[0];
        let q = _slot_primitives[0];
        let m = _slot_marginal_cdf[0];
        let c2 = _slot_conditional_cdf[0];
        ldr_out[uint2(0, 0)] = float4(p, float(i), q.x, m + c2);
        exposure_state[0] = p;
    }
#endif

    float3 c = hdr_in[tid.xy].rgb;

    // Stars+sun+moon composite landed upstream this frame via
    // StarsComposite.slang (issue #46) -- hdr_in already includes
    // celestials and the bloom pyramid (built before this dispatch
    // from the same texture) carries halos around them. Nothing to
    // composite here.

    // Bloom + lens flare both bilinear-sample the bloom pyramid;
    // factor the bilinear sampler into a helper so the flare loop
    // doesn't repeat the math six times.
    uint2 bdim;
    bloom_tex.GetDimensions(bdim.x, bdim.y);
    int2  bd = int2(bdim);

    // sampleBloomUV(uv): bilinear bloom_tex sample at normalised UV.
    // (out-of-range UVs clamp to edges -- ghost positions can land
    // off-screen and should fade rather than wrap or sample garbage.)
    // No anonymous helpers in Slang/HLSL profile we ship, so keep
    // this inlined per call site.

    // Bloom (linear, pre-tonemap) so highlights get the same
    // ACES curve squash as the rest of the image.
    if (bloom_intensity > 0.0) {
        float2 fb = (float2(tid.xy) + 0.5) * float2(bdim) / float2(dim) - 0.5;
        int2  i0  = int2(floor(fb));
        float2 t  = frac(fb);
        int2 p00 = clamp(i0,                  int2(0,0), bd - int2(1,1));
        int2 p10 = clamp(i0 + int2(1,0),      int2(0,0), bd - int2(1,1));
        int2 p01 = clamp(i0 + int2(0,1),      int2(0,0), bd - int2(1,1));
        int2 p11 = clamp(i0 + int2(1,1),      int2(0,0), bd - int2(1,1));
        float3 b00 = bloom_tex[uint2(p00)].rgb;
        float3 b10 = bloom_tex[uint2(p10)].rgb;
        float3 b01 = bloom_tex[uint2(p01)].rgb;
        float3 b11 = bloom_tex[uint2(p11)].rgb;
        float3 b0  = lerp(b00, b10, t.x);
        float3 b1  = lerp(b01, b11, t.x);
        float3 bloom = lerp(b0, b1, t.y);
        // --- Wave 10 bloom/bokeh ---
        // r_bloom_strength gain on the energy-preserving pyramid (1.0 ==
        // the pre-wave-10 composite). Lens dirt, when enabled, modulates
        // the bloom by the procedural grime field so bright sources smear
        // through the dirt; lens_dirt_strength lerps between a clean lens
        // (mask 1.0) and the full dirt mask. Both branches collapse to the
        // original `bloom_intensity * bloom` at defaults (strength 1.0,
        // dirt off), keeping bloom goldens bit-for-bit.
        float3 add = (bloom_intensity * bloom_strength) * bloom;
        if (lens_dirt != 0u) {
            float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
            float aspect = float(dim.x) / float(max(dim.y, 1u));
            float mask = lens_dirt_mask(uv, aspect);
            add *= lerp(1.0, mask, lens_dirt_strength);
        }
        c += add;
        // --- end Wave 10 bloom/bokeh ---
    }

    // Lens flare. Reflections off internal lens elements appear in a
    // real camera as ghost copies of bright sources placed
    // approximately mirrored across the lens centre, with various
    // sub-/super-magnifications. We fake the effect by sampling the
    // already-extracted bloom layer at N "mirror scale" positions:
    //   sample_uv = -(uv - 0.5) * scale + 0.5
    // and adding the result back, tinted per ghost so the artefact
    // reads as colourful flare rather than a duplicate sun. Per-
    // channel scale offsets give the rainbow chromatic aberration
    // you see on cheap lenses.
    if (flare_intensity > 0.0) {
        // 6 ghost scales + tints shared between modes.
        const float kScales[6] = {0.55, 0.78, 1.20, 1.45, 1.85, 2.40};
        const float3 kTints[6] = {
            float3(1.00, 0.55, 0.55),  // red
            float3(0.55, 1.00, 0.55),  // green
            float3(0.55, 0.55, 1.00),  // blue
            float3(1.00, 0.75, 0.40),  // amber
            float3(0.65, 0.40, 1.00),  // purple
            float3(0.40, 0.95, 0.80),  // teal
        };
        float2 uv = (float2(tid.xy) + 0.5) / float2(dim);
        float2 cv = uv - 0.5;
        uint   n  = min(flare_count, 6u);
        float3 flare = float3(0.0);

        // 'physical' mode: gather per-channel Gaussian splats from the
        // engine-traced ghost array. ghost_uv per channel = sun_uv
        // centred and scaled by ghost.scale_*. Each channel is splatted
        // independently so chromatic dispersion (different scale per
        // wavelength, derived from glass Abbe numbers) shows up as a
        // rainbow fringe along ghost edges. No bloom sampling.
        //
        // Three physical effects layered on top:
        //   - sun-visibility gate from sampling hdr_in at sun_uv (so
        //     the flare disappears when the sun is below the horizon
        //     or occluded by scene geometry -- no explicit night-time
        //     gate needed)
        //   - hexagonal ghost silhouettes from approximating the
        //     aperture stop's blade shape (real lenses have blade
        //     count = 5..8; we hardcode 6)
        //   - procedural aperture diffraction starburst at the sun
        //     itself (Fraunhofer pattern of a 6-blade aperture is a
        //     6-pointed star)
        if (flare_mode_physical != 0u && physical_ghost_count > 0u) {
            // Skip when sun is off-screen / behind camera (sentinel uv).
            if (sun_uv.x > -1.0 && sun_uv.x < 2.0 &&
                sun_uv.y > -1.0 && sun_uv.y < 2.0) {
                float2 dimf = float2(dim);
                float2 sun_c = sun_uv - 0.5;

                // Sun-visibility gate. Sample hdr_in around sun_uv with
                // a 3x3 max so we don't miss bright sub-pixel sources.
                // The path tracer's sky integrator writes night-sky
                // luminance (~0.001) when the sun is below the horizon
                // and writes building/terrain luminance when occluded;
                // both naturally collapse the gate.
                float sun_lum = 0.0;
                if (sun_uv.x >= 0.0 && sun_uv.x <= 1.0 &&
                    sun_uv.y >= 0.0 && sun_uv.y <= 1.0) {
                    int2 sc = int2(sun_uv * dimf);
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int2 p = clamp(sc + int2(dx, dy),
                                           int2(0, 0), int2(dim) - int2(1, 1));
                            float3 s = hdr_in[uint2(p)].rgb;
                            float lm = dot(s, float3(0.2126, 0.7152, 0.0722));
                            sun_lum = max(sun_lum, lm);
                        }
                    }
                }
                // Higher-order Reinhard centred on sun_lum = 5 (between
                // a bright wall ~1 and the un-occluded sun ~50). Smooth
                // transition: ~0 below threshold, ~1 above.
                float L = sun_lum / 5.0;
                L = L * L; L = L * L; L = L * L; L = L * L;   // ^16
                float vis = L / (L + 1.0);

                // Cap by both the engine-side ghost count and the user-
                // controlled r_lens_flare_count slider so the slider
                // still works in physical mode.
                uint pn = min(physical_ghost_count, flare_count);
                pn = min(pn, kPhysGhostMax);
                for (uint gi = 0u; gi < pn; ++gi) {
                    GhostShader g = ghosts[gi];
                    float s_r = g.sci.x;
                    float s_g = g.sci.y;
                    float s_b = g.sci.z;
                    float gi_int = g.sci.w;
                    float r_px   = g.rpad.x;
                    // Per-channel screen positions. Negative scale
                    // puts the ghost on the opposite side of centre
                    // from the sun (the canonical mirrored ghost);
                    // |scale|>1 pushes it past centre.
                    float2 c_R = 0.5 + sun_c * s_r;
                    float2 c_G = 0.5 + sun_c * s_g;
                    float2 c_B = 0.5 + sun_c * s_b;
                    float2 d_R = (uv - c_R) * dimf;
                    float2 d_G = (uv - c_G) * dimf;
                    float2 d_B = (uv - c_B) * dimf;
                    // Hexagonal aperture: warp distance by the boundary
                    // distance of a regular hexagon at the ray angle.
                    // For a hex with circumradius r, boundary at angle
                    // theta is r * cos(30deg) / cos((theta mod 60deg) -
                    // 30deg). We divide d by hex_factor so the splat
                    // falls off faster along the flat edges of the
                    // hexagon, giving a 6-pointed silhouette.
                    const float kPi3 = 1.0471975512;   // 60deg
                    const float kPi6 = 0.5235987756;   // 30deg
                    const float kCos30 = 0.8660254038;
                    float a_R = atan2(d_R.y, d_R.x);
                    float a_G = atan2(d_G.y, d_G.x);
                    float a_B = atan2(d_B.y, d_B.x);
                    float h_R = kCos30 / cos((a_R - floor(a_R / kPi3) * kPi3) - kPi6);
                    float h_G = kCos30 / cos((a_G - floor(a_G / kPi3) * kPi3) - kPi6);
                    float h_B = kCos30 / cos((a_B - floor(a_B / kPi3) * kPi3) - kPi6);
                    float r2  = max(r_px * r_px, 1e-6);
                    float qR = dot(d_R, d_R) / (r2 * h_R * h_R);
                    float qG = dot(d_G, d_G) / (r2 * h_G * h_G);
                    float qB = dot(d_B, d_B) / (r2 * h_B * h_B);
                    // Super-Gaussian for a clean disc edge.
                    float iR = exp(-qR * qR);
                    float iG = exp(-qG * qG);
                    float iB = exp(-qB * qB);
                    // Brightness scales with ghost intensity (Fresnel
                    // product) and a uniform 4x boost so the ghosts
                    // are visible at typical r_lens_flare_intensity.
                    flare += float3(iR, iG, iB) * (gi_int * 4.0);
                }

                // Aperture diffraction starburst: 6-pointed star at the
                // sun position. Real Fraunhofer diffraction of a
                // hexagonal aperture has 2*N_blades = 12 spikes for a
                // 6-blade iris (because each blade's edge contributes
                // two opposite spikes), but visually 6 dominate -- we
                // get those by using cos(N*theta/2). The pattern fades
                // exponentially with distance from the sun.
                if (sun_uv.x >= 0.0 && sun_uv.x <= 1.0 &&
                    sun_uv.y >= 0.0 && sun_uv.y <= 1.0) {
                    float2 d_sun  = (uv - sun_uv) * dimf;
                    float dist_s  = length(d_sun);
                    float ang_s   = atan2(d_sun.y, d_sun.x);
                    // 6 spikes (cos(3*theta) is 6-fold symmetric).
                    float spike   = pow(abs(cos(3.0 * ang_s)), 12.0);
                    float fall    = exp(-dist_s / (0.10 * dimf.y));
                    // Slight blue tint: Rayleigh-scattered diffraction
                    // is wavelength dependent, blue dominates short-r
                    // and fades faster than red at longer distances.
                    float3 starburst = float3(0.95, 0.97, 1.05)
                                     * spike * fall * 1.8;
                    flare += starburst;
                }

                // Radial fade so ghosts wash off at the screen edges.
                // Diagonal-normalised pixel distance from centre, with
                // a 1.2 multiplier to fade slightly inside the corners.
                float diag = 0.5 * length(dimf);
                float fade = saturate(1.0 - length(cv * dimf) / diag * 1.2);
                // On-axis fade: when the sun is at screen centre, every
                // ghost lands at sun_uv too (sum of ghost = sun_c *
                // scale, sun_c = 0 means all ghosts overlap with the
                // source). Real lens flare separation is proportional
                // to off-axis angle. Without this fade the ghosts pile
                // up into one giant blob centred on the sun.
                float sun_off = saturate(length(sun_uv - 0.5) * 4.0);
                c += flare_intensity * flare * fade * vis * sun_off;
            }
            // Passthrough path still needs sRGB OETF on SPIR-V swapchains
            // (BGRA8_UNORM) -- without it the linear LDR `c` reaches the
            // surface unencoded and the OS's EOTF interprets it too dark.
            // On Metal PT_SRGB_ENCODE is identity; the swapchain is sRGB-
            // formatted and Apple silicon auto-encodes on store.
            float3 ldr_phys = PT_SRGB_ENCODE(
                // --- Wave 9 tonemap --- op from TonePush::tonemap_op (0
                // == acesTonemap, so the default post-denoise path is
                // byte-identical to pre-Wave-9).
                (passthrough != 0u) ? c : tonemapDispatch(c * exposure_state[0], tonemap_op));
            ldr_out[tid.xy] = float4(ldr_phys, 1.0);
            return;
        }

        // 'sun' mode: explicit sun-position flare. Project sun's
        // screen UV (engine-side); draw discrete soft Gaussian-disc
        // ghosts at its mirrored positions. No bloom sampling --
        // this is what people mean by lens flare.
        if (flare_mode_sun != 0u) {
            // Skip if sun is behind camera or off-screen -- engine
            // sentinels with (-2, -2) so any out-of-[-0.5, 1.5]
            // value means "no flare".
            if (sun_uv.x > -1.0 && sun_uv.x < 2.0 &&
                sun_uv.y > -1.0 && sun_uv.y < 2.0) {
                // Distance in PIXEL space so ghosts read as real
                // circles regardless of aspect ratio. UV is
                // [0,1] x [0,1] but a 16:9 screen has 1.78x more
                // pixels along X than Y -- using raw UV distance
                // gave horizontally-stretched ellipses. Multiplying
                // d_uv by float2(dim) puts both axes in pixels;
                // r is then a pixel radius derived from flare_size
                // (which is fraction-of-screen-height, the more
                // intuitive author-time scale).
                float2 dimf = float2(dim);
                float r_h = flare_size * float(dim.y);   // in pixels
                for (uint gi = 0u; gi < n; ++gi) {
                    float scale = kScales[gi];
                    // Ghost centre = mirror sun across screen centre,
                    // scaled. scale=1 places it exactly opposite the
                    // sun; <1 closer to centre; >1 past centre.
                    float2 ghost_c = 0.5 - (sun_uv - 0.5) * scale;
                    float2 d_px = (uv - ghost_c) * dimf;
                    // Slight per-ghost size variation. Far-side
                    // ghosts a touch larger than near-side.
                    float r = r_h * (0.7 + 0.3 * scale);
                    // Chromatic aberration: per-channel radius
                    // offset gives the rainbow fringe on edges.
                    float r_R = r * (1.0 + flare_dispersion * 8.0);
                    float r_G = r;
                    float r_B = r * (1.0 - flare_dispersion * 8.0);
                    float d2 = dot(d_px, d_px);
                    // Super-Gaussian (exp(-(d/r)^4)) for a clean
                    // disc edge instead of a soft halo bleed.
                    float qR = d2 / max(r_R * r_R, 1e-6);
                    float qG = d2 / max(r_G * r_G, 1e-6);
                    float qB = d2 / max(r_B * r_B, 1e-6);
                    float i_R = exp(-qR * qR);
                    float i_G = exp(-qG * qG);
                    float i_B = exp(-qB * qB);
                    float3 ghost = float3(i_R, i_G, i_B) * kTints[gi] * 4.0;
                    // Radial fade in pixel space too, so the
                    // vignette is circular.
                    float2 cd_px = (uv - 0.5) * dimf;
                    float diag   = 0.5 * length(dimf);
                    float fade   = saturate(1.0 - length(cd_px) / diag * 1.2);
                    flare += ghost * fade;
                }
                c += flare_intensity * flare;
            }
            float3 ldr_sun = PT_SRGB_ENCODE(
                // --- Wave 9 tonemap --- op from TonePush::tonemap_op (0
                // == acesTonemap, so the default post-denoise path is
                // byte-identical to pre-Wave-9).
                (passthrough != 0u) ? c : tonemapDispatch(c * exposure_state[0], tonemap_op));
            ldr_out[tid.xy] = float4(ldr_sun, 1.0);
            return;
        }

        // 'image' mode (legacy): mirror-sample the bloom mip at
        // every output pixel. Looks cinematic but reflects the
        // whole scene -- not what a real lens does.
        for (uint gi = 0u; gi < n; ++gi) {
            float scale = kScales[gi];
            // Per-channel scale offset for chromatic dispersion.
            // Flipping sign for B vs R puts blue and red on opposite
            // sides of the green ghost -- canonical lens-flare look.
            float disp = flare_dispersion;
            float2 g_uv_r = -cv * (scale + disp) + 0.5;
            float2 g_uv_g = -cv *  scale         + 0.5;
            float2 g_uv_b = -cv * (scale - disp) + 0.5;
            // Bilinear sample on each channel's UV. Ghosts that
            // land off-screen clamp to the edge tint -- visually
            // they fade because the bloom tex is mostly zero there.
            float3 r3, g3, b3;
            {
                float2 fb = g_uv_r * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                r3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            {
                float2 fb = g_uv_g * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                g3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            {
                float2 fb = g_uv_b * float2(bdim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), bd - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), bd - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), bd - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), bd - int2(1,1));
                float3 c00 = bloom_tex[uint2(p00)].rgb;
                float3 c10 = bloom_tex[uint2(p10)].rgb;
                float3 c01 = bloom_tex[uint2(p01)].rgb;
                float3 c11 = bloom_tex[uint2(p11)].rgb;
                b3 = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
            }
            float3 ghost = float3(r3.r, g3.g, b3.b);
            // Optional source-luminance gate. CRUCIAL: we measure
            // luminance against the *original* hdr_in at the ghost's
            // position, not the post-pyramid bloom sample. The
            // bloom-chain upsample stage amplifies values (each mip
            // adds the upper mip's halo into the lower one) so a
            // moderately bright source ends up at higher luminance
            // in bloom mip 0 than its original brightness -- gating
            // against the bloom sample lets things slip through that
            // shouldn't. Sampling hdr_in at the same ghost UV gives
            // the truthful "is this position actually the sun?" gate.
            float gate = 1.0;
            if (flare_threshold > 0.0) {
                int2 idim = int2(dim);
                float2 fb = g_uv_g * float2(dim) - 0.5;
                int2 i0 = int2(floor(fb));
                float2 t = frac(fb);
                int2 p00 = clamp(i0,             int2(0,0), idim - int2(1,1));
                int2 p10 = clamp(i0 + int2(1,0), int2(0,0), idim - int2(1,1));
                int2 p01 = clamp(i0 + int2(0,1), int2(0,0), idim - int2(1,1));
                int2 p11 = clamp(i0 + int2(1,1), int2(0,0), idim - int2(1,1));
                float3 s00 = hdr_in[uint2(p00)].rgb;
                float3 s10 = hdr_in[uint2(p10)].rgb;
                float3 s01 = hdr_in[uint2(p01)].rgb;
                float3 s11 = hdr_in[uint2(p11)].rgb;
                float3 src = lerp(lerp(s00, s10, t.x),
                                  lerp(s01, s11, t.x), t.y);
                float src_lum = dot(src, float3(0.2126, 0.7152, 0.0722));
                // Higher-order Reinhard: lum^16 / (lum^16 + t^16).
                // Steep cliff right at `t`, smooth math (no branch),
                // and -- unlike linear Reinhard -- doesn't dim the
                // *passing* samples. Sun at lum=50 with t=30 ->
                // ratio=1.67, ratio^16 ~= 3600, gate ~= 1.0 (full
                // brightness). Wall at lum=25 with t=30 -> ratio=0.83,
                // ratio^16 ~= 0.05, gate ~= 0.05 (killed). The previous
                // linear gate forced thresholds in the 100s to filter
                // walls and shredded the sun; this one separates them
                // cleanly at threshold values near the actual luminance
                // boundary you want.
                float L = src_lum / max(flare_threshold, 1e-4);
                L = L * L;  // ^2
                L = L * L;  // ^4
                L = L * L;  // ^8
                L = L * L;  // ^16
                gate = L / (L + 1.0);
            }
            // Radial fade: ghosts fade out as they near the screen
            // edge so we don't get a hard rectangle of flare on
            // wide-aperture shots.
            float fade = saturate(1.0 - length(uv - 0.5) * 1.5);
            flare += ghost * kTints[gi] * fade * gate;
        }
        c += flare_intensity * flare;
    }

    // Debug: when r_lens_flare_intensity is exactly negative,
    // overlay a green crosshair at sun_uv so the user can see
    // where the engine projects the sun. (Normal mode never sets
    // negative intensity; engine sets a sentinel of -1.0 when
    // dump_sun_uv_overlay = 1.)
    if (flare_intensity < 0.0 &&
        sun_uv.x > -1.0 && sun_uv.x < 2.0 &&
        sun_uv.y > -1.0 && sun_uv.y < 2.0) {
        float2 uvN = (float2(tid.xy) + 0.5) / float2(dim);
        float2 d = (uvN - sun_uv) * float2(dim) / 2.0;
        // 24-px arms, 2-px thick.
        bool on_h = abs(d.y) < 1.0 && abs(d.x) < 24.0;
        bool on_v = abs(d.x) < 1.0 && abs(d.y) < 24.0;
        if (on_h || on_v) {
            ldr_out[tid.xy] = float4(0.0, 1.0, 0.0, 1.0);
            return;
        }
    }
    // --- Wave 9 tonemap --- op from TonePush::tonemap_op (0 == acesTonemap,
    // so the default post-denoise path is byte-identical to pre-Wave-9).
    float3 ldr = PT_SRGB_ENCODE(
        (passthrough != 0u) ? c : tonemapDispatch(c * exposure_state[0], tonemap_op));
    ldr_out[tid.xy] = float4(ldr, 1.0);
}

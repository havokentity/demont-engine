// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// God rays / crepuscular light shafts as a dedicated screen-space pass
// (Wave 9). This is the classic "radial blur from the sun's screen
// position through occluders" technique (Mitchell 2007 / GPU Gems 3
// chapter 13, "Volumetric Light Scattering as a Post-Process"), NOT a
// full volumetric march -- it's the cheap, great-looking shaft effect
// you see when the sun peeks through cloud gaps.
//
// Two passes share this single entry point, selected by gr_params0.x
// (the host dispatches it twice in sequence):
//
//   pass 0 -- BUILD MASK. Reads depth_tex + hdr_inout, writes an
//             occlusion/light mask into mask_tex (binding 37, the
//             engine's reserved Wave 9 scratch slot). Bright where the
//             sky/sun is visible (far depth) and dark where scene
//             geometry occludes (near depth). The brightness comes from
//             the already-rendered HDR luminance so the shafts pick up
//             the actual colour of the sky / sun disc / cloud silver
//             lining showing through each gap.
//
//   pass 1 -- RADIAL BLUR + COMPOSITE. For each pixel, marches N samples
//             from the pixel toward the sun's screen position, summing
//             mask_tex with the standard scattering recurrence
//             (exponential decay x density x weight). The accumulated
//             shaft energy is tinted by the sun colour and added onto
//             hdr_inout in place. Reading mask_tex (written in pass 0,
//             never written here) and writing hdr_inout keeps the two
//             resources disjoint, so there's no intra-dispatch read /
//             write hazard on the radial gather.
//
// The pass is gated host-side by r_godrays; when off the engine elides
// BOTH dispatches so the existing render path is bit-for-bit unchanged.
//
// Bindings. Same scheme as AuroraComposite.slang: on Metal, Slang's
// MSL backend assigns texture indices in DECLARATION order regardless
// of the vk::binding tags --
//   hdr_inout -> texture(0), mask_tex -> texture(1), depth_tex ->
//   texture(2)
// -- so Engine.cpp binds at engine slots 0 / 1 / 2 (BindStorageTexture
// in that order). The vk::binding numbers below are the SPIR-V
// identities for when the Vulkan dispatch is wired (a follow-up; the
// god-rays pipeline id is 0 on Vulkan today, so this kernel only ever
// runs on Metal and the tags are presently SPIR-V-only):
//   binding 0  -- hdr_inout  (the engine's output / accum_hdr target slot)
//   binding 37 -- mask_tex   (engine texture slot 18, reserved for the
//                 Wave 9 god-rays scratch; mirrored in VulkanDevice.cpp's
//                 kSlotToTexBinding[] + kNumTexSlots + pool sizes. A
//                 sibling fog agent owns binding 36 -- left untouched.)
//   binding 7  -- depth_tex  (same binding PathTrace writes its primary-
//                 hit camera-space depth G-buffer to)
// See PathTrace.slang for the NVIDIA Format=Unknown rationale behind the
// vk::image_format qualifiers.
[[vk::binding(0, 0),  vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_inout;
[[vk::binding(37, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> mask_tex;
[[vk::binding(7, 0),  vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;

// MSL SLOT NOTICE: like AuroraComposite / StarsComposite, this kernel
// binds NO buffers and NO acceleration structures. Slang's MSL backend
// therefore assigns Push (the only buffer-class declaration) to MSL
// buf(0). The engine's MetalCommandBuffer::Dispatch computes
// push_slot = max_bound_buf_slot + 1; with nothing else bound and
// BindComputePipeline clearing prior bindings on Metal, max_slot = -1,
// push_slot = 0. Match.

// Per-field semantics (single contiguous push, identical on both
// backends -- 112 bytes, no Vulkan UBO-spill needed since we stay under
// the 256 B / kPushSplitOffset budget):
//   pos_fovtan   -- .xyz camera origin, .w fovYTan
//   fwd_aspect   -- .xyz forward, .w aspect ratio (width/height)
//   right_xyz    -- .xyz camera right basis
//   up_xyz       -- .xyz camera up basis
//   sun_and_mode -- .xyz sun direction (world), .w sky mode (unused here)
//   sun_uv_px    -- .xy sun screen UV [0,1] (engine-projected; radially
//                   clamped to the screen edge when the sun is off-screen
//                   so shafts still converge toward the correct azimuth),
//                   .z sun-visible flag (1 = in front of camera, 0 = behind
//                   -> pass 1 early-outs and adds nothing), .w decay
//   gr_params0   -- .x pass (0 = build mask, 1 = radial blur),
//                   .y density, .z weight, .w exposure
//   gr_params1   -- .x sample count (float, floored to int), .y sun tint
//                   strength (reserved; 1.0), .z/.w pad
[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    float4 sun_and_mode;
    float4 sun_uv_px;
    float4 gr_params0;
    float4 gr_params1;
};

// Sun-tint colour as a function of sun elevation. Mirrors the warm
// sunset reddening the rest of the engine uses (StarsComposite::sunDisc
// and PathTrace's procedural sky both lerp toward (1.55, 0.55, 0.20) at
// the horizon). Shafts at a low sun read warm-orange; a high sun stays
// near-white. Cheap closed form so pass 1 can call it once per pixel.
float3 sunTint(float3 sun_dir) {
    float am  = 1.0 / max(sun_dir.y, 0.04);
    float warmth = saturate((am - 1.0) * 0.18);
    return lerp(float3(1.00, 0.97, 0.92), float3(1.45, 0.55, 0.22), warmth);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;

    int pass = int(gr_params0.x + 0.5);

    // ---- pass 0: build the occlusion / light mask --------------------
    if (pass == 0) {
        // Sky-pixel gate, same convention as StarsComposite /
        // AuroraComposite: PathTrace's G-buffer pass writes a 1e10
        // sentinel for primary-miss (sky) pixels and the real
        // camera-space Z for surface hits. depth >= 1e9 == "sky"; below
        // that == geometry that should occlude the shafts. We do NOT
        // dilate here -- a crisp occluder edge gives crisper shafts, and
        // the radial blur in pass 1 softens the result anyway.
        float depth = depth_tex[tid.xy];
        float3 hdr  = hdr_inout[tid.xy].rgb;

        float mask = 0.0;
        if (depth >= 1.0e9) {
            // Sky / sun / cloud-gap pixel. Use its scalar luminance as
            // the mask weight so a bright sun disc or silver-lined cloud
            // edge drives strong shafts while dim blue sky contributes
            // little. Clamping the upper end keeps a single blown-out
            // sun texel from producing a hard banded streak; the radial
            // accumulation in pass 1 still builds plenty of intensity
            // from the surrounding bright region.
            float lum = dot(hdr, float3(0.2126, 0.7152, 0.0722));
            mask = min(lum, 8.0);
        }
        // Store the scalar mask in all channels (RGBA16F scratch). Pass 1
        // reads .r; the others are written for free and aid debugging.
        mask_tex[tid.xy] = float4(mask, mask, mask, 1.0);
        return;
    }

    // ---- pass 1: radial blur + additive composite --------------------
    // Behind-camera / off-screen gate. The engine sets sun_uv_px.z = 0
    // when the sun is behind the camera; in that case there's no screen
    // origin to march toward, so add nothing.
    if (sun_uv_px.z < 0.5) return;

    float2 uv      = (float2(tid.xy) + 0.5) / float2(dim);
    float2 sun_uv  = sun_uv_px.xy;

    float density  = gr_params0.y;
    float weight   = gr_params0.z;
    float exposure = gr_params0.w;
    float decay    = sun_uv_px.w;
    int   samples  = int(gr_params1.x + 0.5);
    if (samples < 1)  samples = 1;
    if (samples > 256) samples = 256;

    // Standard GPU Gems 3 setup. The per-step vector walks from the
    // current pixel toward the sun's screen position; dividing by the
    // sample count and scaling by `density` controls how far along the
    // sun-ward direction the blur reaches (density < 1 keeps the shafts
    // short / tight, density ~1 lets them sweep the full distance).
    float2 delta = (uv - sun_uv) * (density / float(samples));
    float2 coord = uv;
    float  illum = 1.0;
    float  accum = 0.0;

    // March. Sampling mask_tex with nearest-texel fetch (round to the
    // texel centre) is plenty for this effect; the radial accumulation
    // itself is the smoothing. Stepping in normalised UV keeps the math
    // aspect-correct as long as we convert to texel coords each tap.
    for (int i = 0; i < samples; ++i) {
        coord -= delta;
        int2 px = int2(coord * float2(dim));
        px = clamp(px, int2(0, 0), int2(dim) - int2(1, 1));
        float s = mask_tex[uint2(px)].r;
        accum += s * illum * weight;
        illum *= decay;
    }
    accum /= float(samples);

    // Tint by the warm sun colour and scale by exposure. The shafts are
    // additive HDR energy layered on top of the already-shaded sky /
    // clouds, so they bloom + tonemap with the rest of the frame.
    float3 shaft = sunTint(sun_and_mode.xyz) * (accum * exposure);

    // Sanitize: a NaN in the HDR source (degenerate prior pass) must not
    // propagate. Snap non-finite to zero and clamp negatives.
    bool3 bad = isnan(shaft) || isinf(shaft);
    shaft = select(bad, float3(0.0), max(shaft, float3(0.0)));

    float3 base = hdr_inout[tid.xy].rgb;
    hdr_inout[tid.xy] = float4(base + shaft, 1.0);
}

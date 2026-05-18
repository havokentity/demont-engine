// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SIGMA-style shadow denoiser (issue #115). A NRD-SIGMA-inspired
// post-denoise re-application of sharp sun-shadow boundaries.
//
// Why this pass exists
// --------------------
// When the radiance denoiser (SVGF a-trous or MetalFX) is on, hard
// sun-shadow edges on flat ground render visibly soft. Root cause is
// structural: SVGF's edge-stops weight neighbours by depth + normal +
// luminance. A shadow boundary on a flat plane has ZERO depth change,
// ZERO normal change, and a small luminance change (indirect light
// fills the shadow side), so the filter happily blurs across the
// boundary. MetalFX is even more aggressive about smoothing without
// specific shadow guidance.
//
// AAA fix (NVIDIA's NRD SIGMA library, DLSS-RR's implicit shadow
// pipeline): demodulate the direct-light visibility from the radiance,
// denoise the visibility signal with a shadow-tuned bilateral kernel,
// multiply back at the end. NRD's SIGMA library is Vulkan / D3D12
// only; for Mac and software-backend parity we hand-roll a
// SIGMA-style implementation here. This is conceptually the same
// algorithm NVIDIA ships -- our own tuning, our own Slang code.
//
// MVP architecture (issue #115)
// -----------------------------
// 1. PathTrace.slang writes per-primary-hit sun-NEE shadow-ray
//    transmittance into `shadow_vis_buf` (R32F storage buffer, one
//    float per pixel, row-major). 1.0 = unoccluded; 0.0 = fully
//    shadowed. Sky pixels (no primary hit) write 1.0 so they remain
//    untouched here.
// 2. This kernel runs POST-radiance-denoise, BEFORE StarsComposite +
//    bloom + tonemap. It bilateral-filters `shadow_vis_buf` with a
//    5x5 kernel, using depth + normal edge-stops to avoid leaking
//    across geometry seams, then multiplies the denoised visibility
//    into the post-denoise HDR.
// 3. Because the input HDR already had its shadow boundary smudged by
//    the radiance denoiser, multiplying by the SHARP visibility
//    re-imposes a crisp shadow line on top of the smudged radiance.
//    This is the "simpler MVP" composite the issue calls out -- the
//    full demod path (PathTrace divides direct light out, we
//    multiply it back) is a follow-up. The MVP cost is a slight
//    over-darkening near the boundary's lit side where the denoiser
//    bled a little shadow brightness; users still see "sharp shadow
//    edge", which is the acceptance criterion.
//
// Bilateral kernel design
// -----------------------
// 5x5 box footprint -> 25 reads per pixel. Each neighbour's
// contribution to the centre's denoised visibility is weighted by
// three Gaussian-style stops:
//
//   w_depth(c, n)  = exp(-|z_c - z_n| / sigma_z)
//                    (sigma_z proportional to depth gradient so far-away
//                     surfaces don't all fold together)
//   w_normal(c, n) = pow(max(dot(N_c, N_n), 0), sigma_n)
//                    (high exponent => only near-parallel surfaces blend)
//   w_lum          = uniform (no luminance edge stop on purpose -- the
//                    shadow boundary IS a luminance edge in radiance and
//                    that's exactly what we don't want stopping us;
//                    visibility has its own much-cleaner signal)
//
// Sky pixels write 1.0 into the buffer (no occlusion) so blending into
// the sky just biases vis slightly upward -- harmless. Sky pixels on
// the OUTPUT side are not multiplied: the per-thread sky-pixel gate
// (depth >= 1e9) skips the in-place HDR write so the procedural sky
// stays untouched and StarsComposite still gets a clean sky pass.
//
// Why post-denoise (not in front of the denoiser)
// -----------------------------------------------
// Issue #115 specifies "between PathTrace and Denoise()" for the
// denoise-the-visibility pass. The MVP composite ("hdr *
// shadow_vis_denoised post-denoise") combines them into one shader
// for engineering simplicity: a single kernel that filters visibility
// AND multiplies it into HDR, dispatched once after the radiance
// denoiser. Splitting filter + composite into two kernels is a
// follow-up if/when we want the full demod path (PathTrace divides
// direct light out, composite multiplies back -- requires PathTrace
// surgery and a separate "direct light only" buffer).
//
// Bindings (Slang's MSL backend assigns texture(N) and buffer(N) in
// declaration order; the engine's BindStorageTexture / BindBuffer
// slots in Engine.cpp match these positions):
//   texture(0)  hdr_inout       -- post-denoise HDR; we multiply in place
//   texture(1)  depth_tex       -- linear camera-space Z (PathTrace G-buffer)
//   texture(2)  normal_tex      -- world-space normal (PathTrace G-buffer)
//   buffer(0)   shadow_vis_buf  -- R32F per-pixel sun-visibility (PathTrace G-buf)
//
// Storage buffer (not texture) for shadow_vis is deliberate: Metal's
// compute kernel cap of 8 read_write textures was already saturated
// by PathTrace's existing G-buffer set (output, accum_hdr,
// denoise_color, depth_tex, motion_tex, normal_tex, albedo_tex,
// cloud_trans_tex). Same trick DenoiseAtrous.slang uses for its
// variance + moments buffers; storage buffers don't count against
// the 8-RW-texture limit on Apple Silicon compute.

// vk::image_format: hdr_inout RGBA16F, depth_tex R32F, normal_tex RGBA16F.
// See PathTrace.slang for the NVIDIA Format=Unknown rationale.
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_inout;
[[vk::binding(1, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;
[[vk::binding(2, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_tex;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> shadow_vis_buf;

[[vk::push_constant]]
cbuffer Push {
    // .x = image width, .y = image height (both in pixels). Used for
    // bounds clamping and the row-major shadow_vis_buf index.
    uint width;
    uint height;
    // 1 -> run the composite (engine's gate for r_shadow_demod &&
    // procedural sun NEE active); 0 -> early-out. Cheaper than skipping
    // the dispatch entirely because the BindComputePipeline state is
    // already paid for, but the engine still elides the dispatch when
    // this is 0 to avoid burning GPU on an early-return-everywhere
    // kernel.
    uint demod_active;
    uint _pad0;
    // Depth edge-stop sigma scale. Multiplied by the local depth
    // magnitude inside the kernel so far-away surfaces don't all
    // collapse into one weight tier. Default 0.05 (5% of local Z is
    // "same surface"); raise for thicker bilateral blur, lower for
    // crisper edges.
    float sigma_depth;
    // Normal edge-stop exponent. Higher = stricter "must be co-planar
    // to blend" weighting. Default 16: at 30 deg of normal difference,
    // weight is ~0.4; at 60 deg, ~0.01. Tunable per scene.
    float sigma_normal;
    // 1 -> use normal_tex for edge stops; 0 -> depth-only. The engine
    // sets this to 0 when normal_tex_id_ == 0 (e.g. denoiser without
    // a normal G-buffer; not the common case but kept as a fallback).
    // When 0 the kernel skips the normal sample + dot product entirely.
    uint use_normal;
    uint _pad1;
};

// 5x5 bilateral filter on the noisy visibility buffer + composite into
// the post-denoise HDR. 8x8 threadgroups match the rest of the engine's
// compute kernels for Apple Silicon (32-wide SIMD groups, 2 SIMD per
// threadgroup).
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    if (demod_active == 0u) return;

    // Sky pixel gate. PathTrace.slang's G-buffer writes a 1e10 sentinel
    // for primary-miss pixels; geometry sits well below 1e9 in
    // metric-unit world space (camera-to-mountain-horizon is at most
    // ~100 km, comfortably under 1e9 m). Sky pixels skip the entire
    // demod -- the sky already has no direct-light shadow to denoise,
    // and skipping leaves StarsComposite's downstream celestials
    // untouched.
    float z_c = depth_tex[tid.xy];
    if (z_c >= 1.0e9) return;

    // Centre-pixel normal (skipped when use_normal == 0 -- fallback
    // path with depth-only edge stops).
    float3 N_c = float3(0.0, 1.0, 0.0);
    if (use_normal != 0u) {
        N_c = normal_tex[tid.xy].xyz;
        float n_len = length(N_c);
        if (n_len > 1e-5) N_c = N_c / n_len;
        else N_c = float3(0.0, 1.0, 0.0);
    }

    // Depth-gradient-scaled sigma: surfaces at z=10m and z=100m read
    // VERY different depths but the same plane, so a fixed sigma would
    // collapse the far one to one blob. Scaling by local z (with a
    // minimum to avoid div-by-near-zero at the camera) keeps the
    // bilateral's selectivity roughly constant in screen-relative
    // terms.
    float local_z = max(z_c, 0.1);
    float sigma_z = sigma_depth * local_z;
    // Reciprocal computed once outside the loop -- 25 neighbours * 2
    // divisions per neighbour would otherwise dominate the kernel.
    float inv_sigma_z = (sigma_z > 1e-6) ? (1.0 / sigma_z) : 0.0;

    // 5x5 bilateral. We collect a weighted sum of visibility samples
    // and divide at the end. Sky neighbours (depth >= 1e9) contribute
    // vis=1.0 -- this matches PathTrace's sky write so we never see a
    // sudden jump at the geometry/sky boundary, just a slight
    // brightening of the visibility near horizon lines (where the
    // depth edge-stop will downweight them sharply anyway).
    float sum_w   = 0.0;
    float sum_vis = 0.0;
    int W = int(width) - 1;
    int H = int(height) - 1;
    int2 px = int2(tid.xy);

    for (int dy = -2; dy <= 2; ++dy) {
        int yy = clamp(px.y + dy, 0, H);
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = clamp(px.x + dx, 0, W);

            // Per-neighbour weight starts at 1 (uniform spatial kernel
            // -- the EDGE-STOP weights do the heavy lifting; the
            // bilateral here is more "selectively averaged neighbours"
            // than "smoothed").
            float w = 1.0;

            // Depth edge-stop. exp(-|dz| / sigma_z) gives 1.0 at
            // identical depth, drops smoothly toward 0 as depths
            // diverge. sigma_z is depth-magnitude-scaled (above) so the
            // weight curve has roughly the same shape at z=10 and
            // z=100.
            float z_n = depth_tex[uint2(uint(xx), uint(yy))];
            float dz = abs(z_n - z_c);
            w *= exp(-dz * inv_sigma_z);

            // Normal edge-stop. pow(max(dot(N_c, N_n), 0), sigma_normal)
            // gives 1.0 on perfectly co-planar surfaces and drops
            // rapidly off-axis. sigma_normal=16 is a tight bilateral
            // (only nearby co-planar surfaces blend); raise for
            // smoother filtering across slightly-curved surfaces.
            // Skipped entirely when use_normal == 0 (engine fallback
            // path for denoisers without normal G-buffers).
            if (use_normal != 0u) {
                float3 N_n = normal_tex[uint2(uint(xx), uint(yy))].xyz;
                float n_len = length(N_n);
                if (n_len > 1e-5) {
                    N_n = N_n / n_len;
                    float ndot = max(dot(N_c, N_n), 0.0);
                    w *= pow(ndot, sigma_normal);
                } else {
                    // Neighbour is a sky pixel (or otherwise lacks a
                    // valid normal -- PathTrace writes float3(0) for
                    // primary-miss). Drop the weight to zero so sky
                    // neighbours don't bias the visibility upward.
                    w = 0.0;
                }
            }

            // Read this neighbour's noisy visibility. Buffer is
            // row-major width*height floats; PathTrace.slang writes
            // [yy * width + xx] in the G-buffer pass.
            uint idx = uint(yy) * width + uint(xx);
            float vis_n = shadow_vis_buf[idx];

            sum_w   += w;
            sum_vis += w * vis_n;
        }
    }

    // Numerical fallback: if every neighbour got downweighted to zero
    // (catastrophic depth or normal discontinuities all around -- can
    // happen at hard geometry corners), fall back to the centre pixel's
    // own visibility so we never write NaN / Inf into the HDR.
    float vis_denoised;
    if (sum_w > 1e-6) {
        vis_denoised = sum_vis / sum_w;
    } else {
        uint idx_c = uint(px.y) * width + uint(px.x);
        vis_denoised = shadow_vis_buf[idx_c];
    }
    // Clamp to [0, 1] -- visibility is by definition in that range; a
    // slight numerical excursion outside (e.g. -1e-7 from a
    // sum-of-positives that rounded under) would otherwise show as
    // negative HDR multiply.
    vis_denoised = saturate(vis_denoised);

    // Composite: multiply the post-denoise HDR by the denoised
    // visibility, in place. Alpha kept at 1.0 (the radiance denoiser
    // wrote 1.0 there; the StarsComposite kernel that runs after us
    // also writes 1.0). This is the MVP composite the issue calls
    // out: the input HDR already had its shadow boundary smudged by
    // the radiance denoiser, so multiplying by SHARP visibility
    // re-imposes a crisp edge over the smudged radiance. Cost is a
    // slight over-darkening near the boundary's lit side, accepted
    // for MVP scope.
    float3 hdr = hdr_inout[tid.xy].rgb;
    hdr *= vis_denoised;

    // Sanitize: a NaN in the visibility buffer (extreme corner case --
    // PathTrace clamps writes, but the buffer is shared memory and
    // could see garbage if a future code path lands a partial write)
    // would persist into HDR and propagate through bloom + tonemap.
    bool3 bad = isnan(hdr) || isinf(hdr);
    hdr = select(bad, float3(0.0), max(hdr, float3(0.0)));

    hdr_inout[tid.xy] = float4(hdr, 1.0);
}

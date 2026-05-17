// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Unified path tracer.
//
// One kernel renders everything: a dynamic list of analytic primitives
// (spheres, planes) AND triangle meshes via TLAS. Each path bounce
// finds the nearest hit across all surface types, then dispatches on
// the surface's material (Lambert / metal / dielectric).
//
// Bindings (Slang/MSL maps these to MTL buffer slots starting at 0 for
// the AS, then sequential storage buffers, with push constants last):
//   slot 0 (texture) : swapchain (BGRA8Unorm) -- tonemapped LDR
//   slot 1 (texture) : accum_hdr (RGBA32F)    -- linear average + sample count in .a
//   slot 2 (accel)   : scene_tlas             -- mesh BLAS instances
//   slot 3 (buffer)  : mesh_positions         -- raw float[] xyz packed
//   slot 4 (buffer)  : mesh_indices           -- uint[] indices, 3 per tri
//   slot 5 (buffer)  : primitives             -- 3 float4 per primitive (see CPU side)
//
// `mesh_positions` is `float`-typed (not `float3`) because Slang emits
// StructuredBuffer<float3> with a 16-byte stride per element on MSL,
// while the buffer we upload is tightly packed 12-byte float3.

[[vk::binding(0, 0)]] RWTexture2D<float4> output;
[[vk::binding(1, 0)]] RWTexture2D<float4> accum_hdr;
// scene_tlas is omitted from the SPIR-V variant when the build defines
// PT_SPIRV_NO_RAYQUERY (issue #44 fallback for backends without
// VK_KHR_ray_query, e.g. Mac-Vulkan on pre-1.3 MoltenVK). Without the
// guard the SPIR-V module declares the RayQueryKHR capability up-front,
// which fails vkCreateShaderModule on those drivers and breaks the
// pipeline build entirely. Metal target always includes it -- Metal Ray
// Tracing is unconditionally present on Apple Silicon.
#if !defined(PT_TARGET_SPIRV) || !defined(PT_SPIRV_NO_RAYQUERY)
[[vk::binding(2, 0)]] RaytracingAccelerationStructure scene_tlas;
#endif
[[vk::binding(3, 0)]] StructuredBuffer<float>  mesh_positions;
[[vk::binding(4, 0)]] StructuredBuffer<uint>   mesh_indices;
[[vk::binding(5, 0)]] StructuredBuffer<float4> primitives;
// P10 denoiser G-buffer outputs. Only written when denoiser_enabled != 0.
[[vk::binding(6, 0)]] RWTexture2D<float4> denoise_color;   // RGBA16F per-frame HDR (NOT accumulated)
[[vk::binding(7, 0)]] RWTexture2D<float>  depth_tex;       // R32F  linear camera-space depth
[[vk::binding(8, 0)]] RWTexture2D<float2> motion_tex;      // RG16F pixel-space (prev - curr) delta
// P11 environment map. RGBA32F lat-long HDR; sampled by ray direction
// when env_map_present != 0. Used for both skybox visibility and
// indirect lighting in one place (path tracing collapses skybox + IBL).
//
// Declared as Texture2D (read-only) on Metal and RWTexture2D on
// Vulkan. The kernel only ever READS these three sample-only textures
// (env_map / star_map / moon_map are CPU-uploaded once via
// WriteTexture and never written from a kernel), so a read-only view
// is the natural fit. On Metal the read-only view also matters: the
// Apple Silicon compute pipeline caps read_write textures per kernel
// at 8, and PathTrace would otherwise sit right at that limit
// (output, accum_hdr, denoise_color, depth_tex, motion_tex, env_map,
// star_map, moon_map). Moving env_map / star_map / moon_map off the
// read_write quota frees room for normal_tex at binding 16, which the
// SVGF denoiser needs. Vulkan keeps RWTexture2D because its descriptor
// layout already wires these as STORAGE_IMAGE bindings -- switching to
// SAMPLED_IMAGE there would cascade through VulkanDevice's pipeline
// layout for no actual gain.
#ifdef PT_TARGET_METAL
Texture2D<float4> env_map;
#else
[[vk::binding(9, 0)]] RWTexture2D<float4> env_map;
#endif

// P11 MIS / NEE: precomputed CDFs over env_map luminance. marginal_cdf
// is length H (one entry per row); conditional_cdf is length H*W
// (per-row CDF over columns). Both monotonic, ending at 1.0. Used for
// importance-sampled direct-light samples at Lambert hits.
[[vk::binding(10, 0)]] StructuredBuffer<float> marginal_cdf;
[[vk::binding(11, 0)]] StructuredBuffer<float> conditional_cdf;

// Yale BSC starmap: an RGBA16F equirectangular texture in the J2000
// equatorial frame, rasterised once at startup from BSC5.dat. The host
// pushes a per-frame world->J2000 rotation matrix; the shader samples
// this texture additively when the sun is below the horizon.
#ifdef PT_TARGET_METAL
Texture2D<float4> star_map;
#else
[[vk::binding(12, 0)]] RWTexture2D<float4> star_map;
#endif

// Procedural moon surface texture, equirectangular RGBA16F generated
// once at engine init by pt::moon::generateMoonTexture. Sampled in
// moonDisc() to give the disc real mare + crater detail.
#ifdef PT_TARGET_METAL
Texture2D<float4> moon_map;
#else
[[vk::binding(13, 0)]] RWTexture2D<float4> moon_map;
#endif

// G-buffer: world-space normal at primary hit. Written when
// denoiser_enabled != 0 AND write_normal_gbuffer != 0. Consumed by:
//   - Vulkan SVGF/NRD chain for edge-aware temporal + spatial filtering
//     (DenoiseTemporal disocclusion rejection, DenoiseAtrous normal
//     weight in the wavelet kernel).
//   - Metal SVGF chain (svgf_basic / svgf_atrous) -- same shaders
//     cross-compiled to MSL, same edge-aware behaviour.
//   - OptiX AOV models (HdrAov / TemporalHdrAov) as the normal guide
//     layer.
// MetalFX / plain OptiX HDR don't read it -- the engine drops
// write_normal_gbuffer to 0 for those modes (and doesn't allocate
// normal_tex_id_), so the per-pixel write below skips entirely. Sky
// pixels (no hit) write float3(0) so the SVGF / atrous edge-stops
// treat them as "no surface" and skip the normal weight.
[[vk::binding(16, 0)]] RWTexture2D<float4> normal_tex;

// G-buffer: linear-RGB albedo at primary hit. Written when
// denoiser_enabled != 0 by the SPIR-V/Vulkan path; consumed by the
// OptiX AOV denoiser (OPTIX_DENOISER_MODEL_KIND_AOV) as the albedo
// guide layer alongside the world-space normal. The AOV model uses
// these to preserve diffuse-color edges and surface orientation
// detail under aggressive denoising. Gated to SPIR-V only --
// MetalFX doesn't take an albedo input. Sky pixels (no hit) write
// float3(0) as a "no surface" sentinel matching the normal G-buffer
// convention. The engine only allocates and binds this image when
// r_denoiser is optix_hdr_aov; in any other mode binding 17 stays
// at the default null descriptor and the write below is a no-op via
// Vulkan's robust-buffer-access rules.
#ifdef PT_TARGET_SPIRV
[[vk::binding(17, 0)]] RWTexture2D<float4> albedo_tex;
#endif

// GPU-resident exposure scalar. Updated each frame by AutoExposure.slang
// (when r_auto_exposure = 1) or written once by the engine via
// WriteBuffer (when r_auto_exposure = 0, holding the manual r_exposure
// value). Replaces the per-frame CPU readback of accum_hdr that
// stalled the GPU on dGPU paths -- now read directly here in the
// final tonemap, no host round-trip.
[[vk::binding(15, 0)]] RWStructuredBuffer<float> exposure_state;

// Analytic-primitive BVH (binary, CPU-built in renderer::AnalyticBvh,
// uploaded as one StructuredBuffer<float4>). Root at index 0. Used by
// traceScene's analytic-prim path when push.bvh_params.y > 0; otherwise
// the linear-scan fallback runs over the full primitive list.
//
// Stored as two float4s per node (32 bytes total) so the stride is
// identical on Metal (where Slang would otherwise pad a struct's
// nested float3 to 16 bytes -> 64-byte stride) and on Vulkan std430.
// The two uint fields are stored in the .w lanes and reinterpreted via
// asuint().
//
//   lo.xyz = aabb_min                  lo.w (as uint) = left_first
//   hi.xyz = aabb_max                  hi.w (as uint) = count
//                                                       (0  -> internal,
//                                                        >0 -> leaf with
//                                                        this many prims
//                                                        starting at
//                                                        left_first)
//
// IMPORTANT: declared AFTER exposure_state so the Metal slot
// allocation (which is by declaration order) puts bvh_nodes at slot 7,
// matching the engine's BindBuffer(7, ...) call. Moving this earlier
// would shift exposure_state's slot and break the engine binding.
[[vk::binding(18, 0)]] StructuredBuffer<float4> bvh_nodes;

// Triangle BVH (PR #106 follow-up; issue #44). Host-built mirror of
// renderer::TriangleBvh -- the same flat 32B node layout AnalyticBvh
// uses (two float4s per node, uints recovered via asuint() so the GPU
// stride stays 32B on both std430 and Metal). `tri_bvh_permuted_ids`
// is a flat list of triangle indices; each leaf's range
// [left_first, left_first + count) selects which entries to test. The
// shader reads `mesh_indices[3 * tri + 0..2]` for vertex indices and
// then `mesh_positions[3 * vid + 0..2]` for positions exactly the same
// way the (now removed) SW linear-scan path did.
//
// Always declared (PARTIALLY_BOUND), even on HW-RT builds. The runtime
// gate is `bvh_params.w > 0` -- non-zero means the engine populated
// the buffer and the shader's SW mesh path should walk it; zero means
// no SW mesh path is active for this dispatch.
[[vk::binding(19, 0)]] StructuredBuffer<float4> tri_bvh_nodes;
[[vk::binding(20, 0)]] StructuredBuffer<uint>   tri_bvh_permuted_ids;

// Star-split accumulator (issue #46). When the denoiser is active and
// `star_split_enabled` is non-zero, the primary-miss path writes
// star-only radiance to this texture instead of folding it into
// frame_radiance / denoise_color. Stars then bypass the SVGF a-trous
// kernel (which would smudge their sub-pixel energy across a ~17x17
// region by the third pass) and get composited back in at the
// post-denoise finalize / tonemap stage. Same RGBA16F format and
// dimensions as denoise_color; written with the same EMA blend as
// accum_hdr so 1-spp temporal stability matches the rest of the
// linear-HDR pipeline. When star-split is off (cvar r_star_split 0 or
// denoiser_enabled == 0) the engine binds a 1x1 placeholder texture
// here purely for descriptor-set validity; the shader's
// `star_split_enabled` gate ensures this slot is never written to in
// that case, so the placeholder's contents don't matter.
//
// Moved from binding 19 to binding 21 to coexist with triangle BVH
// (which #107 placed at 19/20 before #108 landed).
[[vk::binding(21, 0)]] RWTexture2D<float4> accum_stars;

struct BvhNode {
    float3 aabb_min;
    uint   left_first;
    float3 aabb_max;
    uint   count;
};

BvhNode loadBvhNode(uint idx) {
    float4 lo = bvh_nodes[idx * 2u + 0u];
    float4 hi = bvh_nodes[idx * 2u + 1u];
    BvhNode n;
    n.aabb_min  = lo.xyz;
    n.left_first = asuint(lo.w);
    n.aabb_max  = hi.xyz;
    n.count     = asuint(hi.w);
    return n;
}

// Same node shape as the analytic BVH; the host (renderer::TriangleBvh)
// guarantees the layout matches. Pulled out as a separate function so
// the traversal walker reads naturally and so future tri-BVH-only
// loaders (e.g. a 2x-quantised version) can be swapped without
// touching the analytic walker.
BvhNode loadTriBvhNode(uint idx) {
    float4 lo = tri_bvh_nodes[idx * 2u + 0u];
    float4 hi = tri_bvh_nodes[idx * 2u + 1u];
    BvhNode n;
    n.aabb_min  = lo.xyz;
    n.left_first = asuint(lo.w);
    n.aabb_max  = hi.xyz;
    n.count     = asuint(hi.w);
    return n;
}

// The full PtPush is 736 bytes today (112 B head + 624 B tail; see the
// static_assert at the dispatch site in Engine.cpp). Native Vulkan / NVIDIA
// caps push constants at 256B, so on SPIR-V we keep only the small
// per-dispatch prefix (camera basis + flags + jitter, 112B) in
// [[vk::push_constant]] and spill the rest into a UBO at binding(14, 0).
// On Metal, [[vk::push_constant]] lowers to a regular setBytes buffer slot
// (no hw size cap), so we keep the full struct there for simplicity.
// VulkanCommandBuffer::Dispatch byte-splits the contiguous host PtPush
// struct: bytes [0..112] -> vkCmdPushConstants, bytes [112..sizeof(PtPush))
// -> the per-frame Frame UBO upload (sized by kFrameUboSize in
// VulkanDevice.h; the dispatch path LOG_ERRORs once if the tail overflows).
[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    uint   frame_index;
    uint   reset_accum;
    uint   max_bounces;
    uint   tlas_present;     // 1 if a CSG mesh is bound, 0 -> skip RayQuery
    uint   prim_count;       // number of analytic primitives in `primitives`
    uint   spp;              // samples per pixel this dispatch (>=1)
    uint   denoiser_enabled; // 1 -> write G-buffer + denoise_color for MetalFX
    uint   env_map_present;  // 1 -> sample env_map for sky; 0 -> procedural gradient
    float2 halton_jitter;    // per-frame sub-pixel offset in [-0.5,0.5]
    float  env_intensity;    // scalar multiplier on env_map samples
    float  env_total_luminance;  // sum of luminance(u,v)*sin(theta) -- PDF normalizer
#ifndef PT_TARGET_SPIRV
    float4x4 curr_view_proj;
    float4x4 prev_view_proj;
    // .xyz = unit vector toward sun, .w = sky mode encoded as a float
    // (0 = gradient, 1 = hdri, 2 = procedural). Packed as float4 so the
    // host's uint sky_mode follows the 16-byte alignment Slang puts
    // after a float3 in MSL.
    float4 sun_and_mode;
    // .x = manual HDR exposure multiplier applied before ACES tonemap.
    // .y = 1.0 if legacy procedural starfield is active.
    // .z = 1.0 if BSC starmap should be sampled.
    // .w = 1.0 if stars should twinkle (atmospheric scintillation fake).
    float4 exposure_pad;
    // World->J2000 rotation, row-major. The 3x3 rotation only uses
    // 9 floats; the .w lanes are repurposed as engine flags:
    //   row0.w = HDR pipeline flag (1 = path tracer writes raw HDR
    //            to denoise_color, post-pass applies exposure+ACES;
    //            0 = legacy pre-tonemap).
    //   row1.w = refractive-shadow-ray budget for caustics
    //            (0 = treat dielectrics as opaque blockers,
    //            positive = max dielectric bounces a NEE shadow ray
    //            may chain through). Capped to 16 by the engine.
    //   row2.w = unused / reserved.
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
    // Thin-lens DOF parameters.
    //   .x = aperture radius (0 = pinhole, no DOF).
    //   .y = focal distance (world units).
    //   .z = aperture blade count (0 = round disk; 3..16 = polygonal
    //        iris -> polygonal bokeh).
    //   .w reserved.
    float4 dof_params;
    // Volumetric single-scatter parameters.
    //   .x = extinction coefficient (per world unit). 0 disables.
    //   .y = Henyey-Greenstein anisotropy g in [-1, 1]. +0.7 gives
    //        forward-scattering atmosphere; 0 = isotropic (fog);
    //        -0.5 = back-scattering (rare materials).
    //   .z = intensity multiplier on the final scatter contribution.
    //   .w = march sample count (cast to int; 4..64 typical).
    float4 vol_params;
    // Volumetric cloud parameters and atmospheric Rayleigh intensity
    // (clouds_p3.w). When clouds_p1.w > 0 the volumetric march
    // samples a position-dependent cloud density.
    //   clouds_p1: (base_y, top_y, coverage[0..1], peak_density)
    //              base_y / top_y are world-space heights bracketing
    //              the cloud layer; pos.y outside this range = 0.
    //              coverage = sky-fill fraction; peak_density is the
    //              extinction at the densest interior point.
    //   clouds_p2: (wind_x, wind_z, freq, time_seconds)
    //              wind translates the noise input by wind*time so
    //              the cloud field drifts. freq is cycles per world
    //              unit (~0.005 = ~200m features). time advances per
    //              frame to animate the field.
    //   clouds_p3: (seed_offset_x, seed_offset_z, detail_amount,
    //               rayleigh_intensity)
    //              seed shifts the noise hash so different seeds
    //              produce visually distinct cloud patterns at the
    //              same preset.
    //              detail_amount in [0,1] blends in a higher-frequency
    //              octave for wispy edges.
    //              rayleigh_intensity is a uniform scalar on the real
    //              per-channel Rayleigh sigma (5.8/13.5/33.1 ×1e-6 m^-1
    //              for R/G/B at sea level). 1.0 = real atmosphere; >1
    //              boosts the blue tint, 0 disables Rayleigh.
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    // Moon. .xyz = unit vector toward the moon (engine computes via
    // Meeus simplified moon-position formulas + observer lat/lon).
    // .w = phase angle in radians (0 = new, π = full); used for both
    // the moon disc's brightness in skyColor and for moonlight NEE
    // contribution amplitude on cloud / surface BRDF rays.
    float4 moon_dir_phase;
    // Moon extras. .x = r_moon_size, .y = astronomical distance ratio
    // (mean/current; perigee ~1.06, apogee ~0.95), .z = r_moon_brightness
    // (disc-multiplier scale, 1.0 = post-ACES-tuned baseline). .w reserved.
    float4 moon_extra;
    // Sun extras. .x = r_sun_size, .y = Earth-Sun distance ratio
    // (mean/current; perihelion ~1.017, aphelion ~0.984).
    // .z = vertical-flatten ratio for atmospheric refraction at horizon
    // (1.0 = render a perfect circle; <1.0 = vertically-squished oval
    // computed via Saemundsson differential across the real solar disc;
    // CPU-clamped to [0.5, 1.0]). Typical values produced by the
    // formula: ~0.78 at elev=0 (4:5 oval, peak squish), ~0.87 at 1deg,
    // ~0.95 at 3deg, ~0.97 at 5deg, ~0.99 at 10deg, indistinguishable
    // from 1.0 above ~30deg. .w reserved.
    float4 sun_extra;
    // HDRI multi-light extraction (computed at HDRI load in
    // ReloadEnvMap). Each entry is a directional light derived from a
    // bright cluster in the env map. Stochastic single-pick NEE per
    // sample; pmf is in dir.w, sums to 1 across kHdriLightsMax. .rgb
    // of color is ∫_cluster L dΩ in raw env_map units (shader scales
    // by env_intensity to match env-map NEE).
    float4 hdri_lights_dir[8];
    float4 hdri_lights_col[8];
    uint   hdri_lights_count;
    // 1 -> write normal_tex at primary hit, 0 -> skip. The engine
    // sets this based on whether the active denoiser kind consumes
    // normals (svgf_basic / svgf_atrous / nrd / optix_*_aov) AND has
    // allocated a normal G-buffer this frame. On modes that don't
    // need it (metalfx, plain optix_hdr / optix_temporal_hdr) the
    // texture isn't bound, so the kernel MUST skip the write.
    uint   write_normal_gbuffer;
    // 1 -> write albedo_tex at primary hit, 0 -> skip. The engine
    // sets this only for the OptiX AOV variants (optix_hdr_aov /
    // optix_temporal_hdr_aov), the only modes that own an albedo
    // G-buffer. Required because under
    // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT the unbound slot
    // doesn't get a null-descriptor pass-through anymore -- the
    // shader must elide the write itself.
    uint   write_albedo_gbuffer;
    // 1 -> kernel writes its per-pixel linear-HDR (or pre-tonemapped LDR
    // when w2j_row0.w = 0) radiance to `denoise_color`. Set by the engine
    // when EITHER a denoiser is active OR `r_bloom` is on without a
    // denoiser on a backend that runs the engine-side Tonemap.slang
    // chain. Distinct from `denoiser_enabled`, which still gates the
    // depth/motion/normal/albedo G-buffer writes + Halton jitter (only
    // meaningful when a real denoiser consumes them). Reuses what used
    // to be `_hdri_pad1` -- same wire offset, same Slang struct end.
    uint   write_hdr_aux;
    // 1 -> apply balance-heuristic MIS weights to env-map NEE and
    // BRDF-bounce miss contributions on Lambert hits. 0 -> classic
    // NEE-only (existing `skip_sky_on_miss` path). Metal (including
    // rough metal via h.roughness) and dielectric paths stay on their
    // existing code paths either way -- MIS is currently a Lambert-only
    // scope, not a property of those materials. Reuses what was
    // `_hdri_pad1`; layout-equivalent.
    uint   mis_enabled;
    // Accumulator parameters. See Engine.cpp's PtPush::accum_params for
    // full docs.
    //   .x = ema_alpha in [0, 1). 0 = legacy online running mean;
    //        >0 = EMA blend with this history retention factor (effective
    //        sliding window ~1/(1-alpha) frames). Lets the accumulator
    //        track dynamic content without wet-paint smear, at the cost
    //        of a variance floor proportional to 1/window.
    //   .y/.z/.w reserved for future accum-side knobs.
    float4 accum_params;
    // Analytic-primitive + triangle BVH traversal params. See Engine.cpp's
    // PtPush::bvh_params for full docs.
    //   .x = linear_prim_count — number of always-linear-tested prims
    //        at the front of `primitives` (planes today).
    //   .y = bvh_node_count    — 0 disables analytic-prim BVH path
    //        (linear over all remaining prims); >0 enables iterative
    //        traversal of `bvh_nodes` for the trailing range
    //        [linear_prim_count, prim_count).
    //   .z = mesh_tri_count    — total CSG triangle count uploaded to
    //        mesh_positions / mesh_indices this dispatch. Non-zero on
    //        the SW-mesh fallback path (no TLAS bound, e.g. Mac-Vulkan
    //        on pre-1.3 MoltenVK); the shader checks this to decide
    //        whether the SW mesh branch fires at all.
    //   .w = tri_bvh_node_count — when non-zero, the SW mesh branch
    //        traverses the host-built triangle BVH at `tri_bvh_nodes`
    //        / `tri_bvh_permuted_ids` instead of brute-force linear
    //        scanning every triangle. PR #106 follow-up: the linear
    //        scan was O(N) per ray and crushed framerate on Mac-Vulkan;
    //        the BVH walk is O(log N).
    uint4 bvh_params;
    // Star-split flag (issue #46). 1 = route star contribution out
    // of frame_radiance and into accum_stars; the post-denoise
    // finalize stage adds it back pre-tonemap so SVGF's a-trous
    // kernel never sees the sub-pixel star energy and can't smudge
    // it across neighbouring sky texels. 0 = legacy behaviour,
    // stars fold into frame_radiance like any other sky term and
    // get filtered with the rest. Engine fills this from the
    // r_star_split cvar gated by denoiser_enabled and a real
    // accum_stars allocation. The trailing 12 bytes of pad keep the
    // cbuffer trailing block 16-byte aligned for std140 / MSL.
    uint star_split_enabled;
    uint _pad_star_split0;
    uint _pad_star_split1;
    uint _pad_star_split2;
#endif  // !PT_TARGET_SPIRV
};

#ifdef PT_TARGET_SPIRV
// Spilled tail of PtPush — see the comments in the (Metal) Push cbuffer
// above for per-field semantics. Layout is byte-equivalent to bytes
// [112..736] of the host PtPush struct in Engine.cpp. Bound by
// VulkanCommandBuffer::Dispatch from a per-frame ring-buffer UBO.
[[vk::binding(14, 0)]]
cbuffer Frame {
    float4x4 curr_view_proj;
    float4x4 prev_view_proj;
    float4 sun_and_mode;
    float4 exposure_pad;
    float4 w2j_row0;
    float4 w2j_row1;
    float4 w2j_row2;
    float4 dof_params;
    float4 vol_params;
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 moon_dir_phase;
    float4 moon_extra;
    float4 sun_extra;
    float4 hdri_lights_dir[8];
    float4 hdri_lights_col[8];
    uint   hdri_lights_count;
    // See the matching field in the Metal Push cbuffer above for
    // semantics. 1 -> write normal_tex, 0 -> skip.
    uint   write_normal_gbuffer;
    // See the matching field in the Metal Push cbuffer above for
    // semantics. 1 -> write albedo_tex, 0 -> skip.
    uint   write_albedo_gbuffer;
    // See `write_hdr_aux` in the Metal Push cbuffer above.
    uint   write_hdr_aux;
    // See `mis_enabled` in the Metal Push cbuffer above. Reuses
    // `_hdri_pad1`; layout-equivalent across both targets.
    uint   mis_enabled;
    // See `accum_params` in the Metal Push cbuffer above.
    float4 accum_params;
    // See `bvh_params` in the Metal Push cbuffer above.
    uint4 bvh_params;
    // See `star_split_enabled` in the Metal Push cbuffer above.
    uint star_split_enabled;
    uint _pad_star_split0;
    uint _pad_star_split1;
    uint _pad_star_split2;
};
#endif  // PT_TARGET_SPIRV

// Pre-compiled Slang IR modules.  Cloud noise + density helpers and
// the pure-math RNG / disc-sample / intersection helpers live in
// separate .slang-module files that ninja builds in parallel and
// caches as IR -- ~145 ms off PathTrace.spv on body-only iterations
// because the front-end work for these helpers is paid once per
// module change instead of on every PathTrace rebuild.  Both modules
// are target-agnostic (no Push refs, no resource bindings) so the
// same .slang-module file links into both SPIR-V and Metal targets
// of this entry-point shader.
import PathTraceCloud;
import PathTraceMath;

static const uint MAT_LAMBERT    = 0u;
static const uint MAT_METAL      = 1u;
static const uint MAT_DIELECTRIC = 2u;

static const uint PRIM_SPHERE = 0u;
static const uint PRIM_PLANE  = 1u;

static const float3 LIGHT_DIR    = float3(0.4, 1.0, 0.3);
static const float3 MESH_ALBEDO  = float3(1.00, 0.85, 0.45);   // gold metal default

// pcgHash, randf, sampleCosineHemisphere, sampleSunDisc,
// kSunAngularRadius, fresnelSchlick, intersectSphere, intersectPlane:
// imported from PathTraceMath at the top of this file.

struct HitInfo {
    bool   hit;
    float  t;
    float3 normal;
    uint   mat;
    float3 albedo;
    float  roughness;
    float  ior;
};

// Test a single analytic primitive at `primitives[idx*3 + 0..2]` against
// (ro, rd) and update `h` if it's a closer hit. Centralizes the per-prim
// dispatch so the linear-loop and BVH-leaf paths share one body.
void testAnalyticPrim(uint idx, float3 ro, float3 rd, inout HitInfo h) {
    float4 v0 = primitives[idx * 3u + 0u];
    float4 v1 = primitives[idx * 3u + 1u];
    float4 v2 = primitives[idx * 3u + 2u];
    uint type = uint(v2.x);
    float t;
    if (type == PRIM_SPHERE) {
        if (intersectSphere(ro, rd, v0.xyz, v0.w, t) && t < h.t) {
            h.hit = true; h.t = t;
            h.normal = normalize(ro + rd * t - v0.xyz);
            h.mat = uint(v2.y);
            h.albedo = v1.rgb;
            h.roughness = v1.a;
            h.ior = v2.z;
        }
    } else if (type == PRIM_PLANE) {
        if (intersectPlane(ro, rd, v0.xyz, v0.w, t) && t < h.t) {
            h.hit = true; h.t = t;
            h.normal = v0.xyz;
            h.mat = uint(v2.y);
            h.albedo = v1.rgb;
            h.roughness = v1.a;
            h.ior = v2.z;
        }
    }
}

// Slab ray-AABB intersection. Returns true if the ray (ro, rd) hits the
// box [mn, mx] within [0, t_max]. Used to decide whether to descend a
// BVH node. rd_inv must be 1.0 / rd componentwise; we precompute it
// once per ray and pass through to avoid divides per node.
bool intersectAabb(float3 ro, float3 rd_inv, float3 mn, float3 mx, float t_max) {
    float3 t0 = (mn - ro) * rd_inv;
    float3 t1 = (mx - ro) * rd_inv;
    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);
    float tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    return tmax >= max(tmin, 0.0) && tmin <= t_max;
}

HitInfo traceScene(float3 ro, float3 rd) {
    HitInfo h;
    h.hit = false; h.t = 1e30;
    h.normal = float3(0,1,0); h.mat = MAT_LAMBERT;
    h.albedo = float3(0.5); h.roughness = 0.0; h.ior = 1.0;

    // Analytic primitives. Each occupies 3 consecutive float4s in
    // `primitives`:
    //   v0 = (xyz: center / normal,  w: radius / d)
    //   v1 = (rgb: albedo,           a: roughness)
    //   v2 = (x: type,  y: material, z: ior, w: pad)
    //
    // The host partitions the buffer at upload time so the first
    // `linear_prim_count` entries are infinite-extent prims (planes)
    // that must always be linear-tested -- they can't fit in a finite
    // AABB. The remainder is either linear-scanned (when bvh_params.y
    // == 0, i.e. count below r_analytic_bvh_threshold) or traversed
    // via the BVH whose flat node array lives in `bvh_nodes`.
    uint linear_prim_count = bvh_params.x;
    uint bvh_node_count    = bvh_params.y;
    // Linear pass over the infinite-extent segment first.
    for (uint i = 0u; i < linear_prim_count && i < prim_count; ++i) {
        testAnalyticPrim(i, ro, rd, h);
    }
    if (bvh_node_count == 0u) {
        // BVH inactive: linear-scan the rest of the buffer too.
        for (uint i = linear_prim_count; i < prim_count; ++i) {
            testAnalyticPrim(i, ro, rd, h);
        }
    } else {
        // Iterative BVH traversal with a small local stack. Depth is
        // log2(node_count); 32 slots covers ~4 billion leaves with
        // headroom. Pre-reciprocal rd to skip per-node divides.
        float3 rd_safe = select(abs(rd) > float3(1e-30), rd, float3(1e-30));
        float3 rd_inv = 1.0 / rd_safe;
        uint stack[32];
        int  sp = 0;
        stack[sp++] = 0u;
        while (sp > 0) {
            uint idx = stack[--sp];
            BvhNode node = loadBvhNode(idx);
            if (!intersectAabb(ro, rd_inv, node.aabb_min, node.aabb_max, h.t)) continue;
            if (node.count > 0u) {
                // Leaf: test the contiguous prim range.
                uint base = node.left_first + linear_prim_count;
                for (uint k = 0u; k < node.count; ++k) {
                    testAnalyticPrim(base + k, ro, rd, h);
                }
            } else {
                // Internal: push both children. Stack-depth guard.
                if (sp + 2 <= 32) {
                    stack[sp++] = node.left_first;
                    stack[sp++] = node.left_first + 1u;
                }
            }
        }
    }

    // Triangle meshes. Two paths exist:
    //   tlas_present != 0                 -> HW RayQuery (fast)
    //   tlas_present == 0 && tri_count > 0 -> SW linear-scan fallback
    // The SW path is the workaround for backends that lack
    // VK_KHR_acceleration_structure -- specifically Mac-Vulkan on
    // pre-1.3 MoltenVK installations. Accessing scene_tlas when no TLAS
    // is bound is undefined; the tlas_present guard avoids that on
    // backends that don't expose the RT extensions at all. Builds with
    // PT_SPIRV_NO_RAYQUERY remove the HW branch entirely so the SPIR-V
    // module doesn't declare the RayQueryKHR capability and can load on
    // drivers without VK_KHR_ray_query (e.g. older MoltenVK).
#if !defined(PT_TARGET_SPIRV) || !defined(PT_SPIRV_NO_RAYQUERY)
    if (tlas_present != 0u) {
        RayDesc r;
        r.Origin    = ro;
        r.Direction = rd;
        r.TMin      = 1e-3;
        r.TMax      = h.t;     // limit search to current closest

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, r);
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float t_mesh = q.CommittedRayT();
            if (t_mesh < h.t) {
                uint prim = q.CommittedPrimitiveIndex();
                uint i0 = mesh_indices[prim * 3 + 0];
                uint i1 = mesh_indices[prim * 3 + 1];
                uint i2 = mesh_indices[prim * 3 + 2];
                float3 v0 = float3(mesh_positions[i0 * 3 + 0],
                                   mesh_positions[i0 * 3 + 1],
                                   mesh_positions[i0 * 3 + 2]);
                float3 v1 = float3(mesh_positions[i1 * 3 + 0],
                                   mesh_positions[i1 * 3 + 1],
                                   mesh_positions[i1 * 3 + 2]);
                float3 v2 = float3(mesh_positions[i2 * 3 + 0],
                                   mesh_positions[i2 * 3 + 1],
                                   mesh_positions[i2 * 3 + 2]);
                float3 obj_n = normalize(cross(v1 - v0, v2 - v0));
                float3x3 wfm = (float3x3)q.CommittedObjectToWorld4x3();
                h.hit = true; h.t = t_mesh;
                h.normal = normalize(mul(wfm, obj_n));
                h.mat = MAT_METAL;
                h.albedo = MESH_ALBEDO;
                h.roughness = 0.05;
                h.ior = 1.0;
            }
        }
    } else
#endif
    if (bvh_params.z != 0u) {
        // Software fallback: traverse the host-built triangle BVH
        // (renderer::TriangleBvh -- PR #106 follow-up, replaces the
        // O(N) Möller-Trumbore linear scan #106 shipped). The CSG bake
        // places geometry in world space and the HW path uses an
        // identity-transform TLAS instance (see
        // Engine::RebuildMeshResources), so no ObjectToWorld is needed
        // here -- when per-mesh transforms land both paths gain one.
        // On builds with PT_SPIRV_NO_RAYQUERY this is the ONLY mesh
        // path (the `else` above attaches to the `if (tlas_present)`
        // block, which is preprocessed out for that variant).
        //
        // Walker mirrors the analytic-BVH walker above: iterative stack
        // (depth 64 covers >1B triangles via balanced log2 splits with
        // headroom), per-node AABB-slab test culls subtrees beyond
        // closest-t, per-leaf loop over `count` triangles via the
        // permuted-id indirection. Möller-Trumbore intersect is
        // identical to what the linear-scan loop had -- two-sided,
        // reject |det| < 1e-8 only. If the engine somehow leaves the
        // BVH unpopulated (.w == 0) while still flagging the SW mesh
        // path (.z != 0), the walker short-circuits to "no hit" and
        // the SW mesh contribution drops out for that frame; this is
        // a benign failure mode (mesh stays black) rather than a hang
        // or per-ray full scan.
        uint tri_bvh_node_count = bvh_params.w;
        float t_mesh = h.t;
        uint  prim_hit = 0u;
        bool  found    = false;
        if (tri_bvh_node_count > 0u) {
            float3 rd_safe = select(abs(rd) > float3(1e-30), rd, float3(1e-30));
            float3 rd_inv  = 1.0 / rd_safe;
            uint stack[64];
            int  sp = 0;
            stack[sp++] = 0u;
            while (sp > 0) {
                uint idx = stack[--sp];
                BvhNode node = loadTriBvhNode(idx);
                if (!intersectAabb(ro, rd_inv, node.aabb_min, node.aabb_max, t_mesh)) continue;
                if (node.count > 0u) {
                    // Leaf: test the contiguous range of permuted tri ids.
                    for (uint k = 0u; k < node.count; ++k) {
                        uint tri = tri_bvh_permuted_ids[node.left_first + k];
                        uint i0 = mesh_indices[tri * 3u + 0u];
                        uint i1 = mesh_indices[tri * 3u + 1u];
                        uint i2 = mesh_indices[tri * 3u + 2u];
                        float3 v0 = float3(mesh_positions[i0 * 3u + 0u],
                                           mesh_positions[i0 * 3u + 1u],
                                           mesh_positions[i0 * 3u + 2u]);
                        float3 v1 = float3(mesh_positions[i1 * 3u + 0u],
                                           mesh_positions[i1 * 3u + 1u],
                                           mesh_positions[i1 * 3u + 2u]);
                        float3 v2 = float3(mesh_positions[i2 * 3u + 0u],
                                           mesh_positions[i2 * 3u + 1u],
                                           mesh_positions[i2 * 3u + 2u]);
                        float3 e1 = v1 - v0;
                        float3 e2 = v2 - v0;
                        float3 pv = cross(rd, e2);
                        float  det = dot(e1, pv);
                        if (abs(det) < 1e-8) continue;
                        float inv_det = 1.0 / det;
                        float3 tvec = ro - v0;
                        float u = dot(tvec, pv) * inv_det;
                        if (u < 0.0 || u > 1.0) continue;
                        float3 qv = cross(tvec, e1);
                        float v = dot(rd, qv) * inv_det;
                        if (v < 0.0 || u + v > 1.0) continue;
                        float t_tri = dot(e2, qv) * inv_det;
                        if (t_tri > 1e-3 && t_tri < t_mesh) {
                            found    = true;
                            t_mesh   = t_tri;
                            prim_hit = tri;
                        }
                    }
                } else {
                    if (sp + 2 <= 64) {
                        stack[sp++] = node.left_first;
                        stack[sp++] = node.left_first + 1u;
                    }
                }
            }
        }
        if (found) {
            uint i0 = mesh_indices[prim_hit * 3u + 0u];
            uint i1 = mesh_indices[prim_hit * 3u + 1u];
            uint i2 = mesh_indices[prim_hit * 3u + 2u];
            float3 v0 = float3(mesh_positions[i0 * 3u + 0u],
                               mesh_positions[i0 * 3u + 1u],
                               mesh_positions[i0 * 3u + 2u]);
            float3 v1 = float3(mesh_positions[i1 * 3u + 0u],
                               mesh_positions[i1 * 3u + 1u],
                               mesh_positions[i1 * 3u + 2u]);
            float3 v2 = float3(mesh_positions[i2 * 3u + 0u],
                               mesh_positions[i2 * 3u + 1u],
                               mesh_positions[i2 * 3u + 2u]);
            float3 obj_n = normalize(cross(v1 - v0, v2 - v0));
            h.hit = true; h.t = t_mesh;
            h.normal = obj_n;
            h.mat = MAT_METAL;
            h.albedo = MESH_ALBEDO;
            h.roughness = 0.05;
            h.ior = 1.0;
        }
    }
    return h;
}

// Hand-tuned analytic sky inspired by Preetham/Hosek-Wilkie but ~30
// lines instead of ~150. Drives day/sunset/night transitions from
// `sun_dir.y` (sun's vertical component above horizon). Result is HDR
// linear; the sun disk is around 50x normal sky brightness so it
// punches through ACES tonemap and reads as "the sun" rather than a
// bright dot.
// Compute the night-sky star contribution (BSC starmap or procedural
// cell-hash fallback) for a primary ray direction. Returns the raw
// star radiance with day-fade already applied; callers add it
// additively onto whatever sky base they're using.
//
// Extracted from procSky() for issue #46: SVGF a-trous smudges these
// sub-pixel features into surrounding dark sky texels (~17x17 spread
// by the third pass). When star-split is enabled the engine routes
// this value out of frame_radiance and into accum_stars so the
// denoiser never sees it -- composited back in pre-tonemap. The
// helper is also called from procSky() itself for the legacy
// (star-split off / denoiser off) path, keeping a single source of
// truth for the star sampling logic.
float3 starsOnly(float3 rd, float3 sun) {
    float cosTheta = clamp(rd.y, -1.0, 1.0);
    float sun_elev = sun.y;
    float day      = smoothstep(-0.10, 0.20, sun_elev);
    // BSC starmap (real Yale Bright Star Catalog, J2000 frame). Sampled
    // additively whenever the host has uploaded the texture and we're
    // looking above the horizon. Falls back to the procedural cell-hash
    // scatter below when the starmap isn't available -- the host gates
    // exposure_pad.z (BSC) and .y (procedural) to be mutually exclusive.
    if (cosTheta > -0.05 && exposure_pad.z > 0.5 && day < 0.6) {
        // World ray -> J2000 unit vector via the per-frame rotation.
        float3 j;
        j.x = dot(w2j_row0.xyz, rd);
        j.y = dot(w2j_row1.xyz, rd);
        j.z = dot(w2j_row2.xyz, rd);
        // Equirectangular: u from atan2(y, x) (RA), v from asin(z) (dec).
        // Match the host rasteriser: u = ra/360 wraps, v = (90-dec)/180.
        float u = atan2(j.y, j.x) * (1.0 / 6.28318530718);
        if (u < 0.0) u += 1.0;
        float v = (1.5707963267948966 - asin(clamp(j.z, -1.0, 1.0))) * (1.0 / 3.14159265358979);
        uint2 dim;
        star_map.GetDimensions(dim.x, dim.y);
        int xi = int(floor(u * float(dim.x))) % int(dim.x);
        if (xi < 0) xi += int(dim.x);
        int yi = clamp(int(floor(v * float(dim.y))), 0, int(dim.y) - 1);
        float3 stars = star_map[uint2(uint(xi), uint(yi))].rgb;
        // Twinkle. Hash on a coarse-quantised J2000 direction (~1.5
        // deg bins) so every texel covering a single star agrees on
        // phase + frequency, and the whole star modulates as one
        // unit. (Hashing on raw texel coords made adjacent texels of
        // a multi-texel bright star twinkle out-of-phase, which
        // averaged flat to the eye.) +/-45%% amplitude over a 3-8 Hz
        // sinusoid; reads as broadband atmospheric scintillation.
        if (exposure_pad.w > 0.5) {
            int3 q  = int3(floor(j * 38.0));   // ~1.5 deg bins
            uint qh = uint(q.x + 128) * 65536u
                    + uint(q.y + 128) * 256u
                    + uint(q.z + 128);
            uint h    = pcgHash(qh + 0xb7e1u);
            float ph  = float(h & 0xFFFFu) * (1.0 / 65536.0);
            float fhz = 3.0 + float((h >> 16) & 0xFFu) * (5.0 / 256.0);
            float t   = float(frame_index) * (1.0 / 60.0);
            float w   = 0.55 + 0.45 * sin(6.28318530718 * (ph + t * fhz));
            stars *= w;
        }
        // Fade with day so stars vanish at sunrise.
        return stars * (1.0 - day);
    } else if (cosTheta > -0.05 && exposure_pad.y > 0.5 && day < 0.6) {
        // Cells sized so each one is ~3-5 pixels wide on a 1080p frame
        // -- smaller cells make stars sub-pixel and they vanish.
        float2 ll = float2(atan2(rd.z, rd.x) * (1.0 / 6.28318530718) + 0.5,
                           acos(clamp(rd.y, -1.0, 1.0)) * (1.0 / 3.14159265359));
        ll *= float2(700.0, 350.0);
        float2 cellI = floor(ll);
        float2 cellF = ll - cellI;
        float h  = frac(sin(dot(cellI, float2(127.1, 311.7))) * 43758.5453);
        float h2 = frac(sin(dot(cellI, float2(269.5, 183.3)))  * 91842.1417);
        if (h > 0.96) {
            float2 starPos = float2(0.30 + h2 * 0.40,
                                    0.30 + frac(h * 7.91) * 0.40);
            float dist = length(cellF - starPos);
            float brightness = pow(frac(h * 13.7), 4.0) * 4.0 + 0.5;
            float intensity = brightness * exp(-dist * dist * 100.0);
            float colorMix = frac(h * 5.31);
            float3 hot  = float3(0.85, 0.92, 1.10);
            float3 mid  = float3(1.00, 0.98, 0.95);
            float3 cool = float3(1.05, 0.85, 0.70);
            float3 starColor = (colorMix < 0.33) ? hot
                              : (colorMix < 0.80) ? mid : cool;
            return intensity * starColor * (1.0 - day);
        }
    }
    return float3(0.0);
}

// Star contribution at a primary-miss direction, regardless of which
// sky path is active. HDRI mode bakes stars into the env_map so we
// return 0 there (the env_map is sampled at primary miss anyway; we
// can't separate the stars from the rest of the texture without a
// brightness threshold, which is a separate piece of work). Procedural
// + gradient modes route through starsOnly() above, which mirrors the
// per-mode gates inside procSky().
float3 starsAtPrimaryMiss(float3 rd) {
    uint mode = uint(sun_and_mode.w);
    // procSky (mode == 2) is the only path that injects stars at
    // primary miss; gradient sky has no stars and HDRI mode bakes
    // them into the env_map (extracting requires a brightness
    // threshold -- deferred to bloom-on-stars follow-up, since the
    // issue acceptance criterion "bloom halos proportional to
    // brightness" implies a real threshold to know what counts as
    // a "star"). All other modes return 0 here.
    if (mode == 2u) {
        return starsOnly(rd, sun_and_mode.xyz);
    }
    return float3(0.0);
}

float3 procSky(float3 rd, float3 sun) {
    float cosTheta = clamp(rd.y, -1.0, 1.0);    // 1=zenith, 0=horizon, -1=nadir
    float cosGamma = clamp(dot(rd, sun), -1.0, 1.0);
    float sun_elev = sun.y;                     // sin(elevation)

    // Day vs night blend: smoothstep on sun elevation across the
    // civil-twilight band.
    float day = smoothstep(-0.10, 0.20, sun_elev);

    // Sunset glow: peaks when sun is just at the horizon.
    float sunset = smoothstep(0.30, -0.05, sun_elev) *
                   smoothstep(-0.20, 0.05, sun_elev);

    // Day sky palette.
    float3 zenith_day  = float3(0.32, 0.55, 0.92);
    float3 horizon_day = float3(0.78, 0.88, 1.00);
    // Sunset palette pulls toward orange/red on the side of the sun.
    float side_factor  = smoothstep(0.0, 0.7, dot(normalize(float3(rd.x, 0, rd.z)),
                                                  normalize(float3(sun.x, 0, sun.z))));
    float3 horizon_sunset = lerp(float3(0.95, 0.45, 0.20),
                                 float3(1.00, 0.65, 0.30), side_factor);
    float3 horizon_col    = lerp(horizon_day, horizon_sunset, sunset);
    float3 zenith_col     = lerp(zenith_day,  float3(0.18, 0.18, 0.30), sunset * 0.6);

    // Vertical gradient from horizon to zenith.
    float t = smoothstep(0.0, 0.40, cosTheta);
    float3 sky = lerp(horizon_col, zenith_col, t);

    // Apply day/night brightness curve. Night is a deep navy with
    // a faint horizon line, no sun.
    float3 night = float3(0.005, 0.010, 0.030) +
                   smoothstep(0.0, 0.05, abs(cosTheta - 0.0)) * 0.0;
    sky = lerp(night, sky, day);

    // Star contribution. Routed through starsOnly() so the same logic
    // is reusable from the primary-miss star-split path (issue #46).
    // When that split is active, the engine subtracts this term from
    // frame_radiance and writes it to accum_stars instead, so the
    // SVGF a-trous kernel never sees sub-pixel star energy and can't
    // smudge it across neighbouring sky texels.
    sky += starsOnly(rd, sun);

    // Sun disk: very tight angular radius (~0.25 deg ~= cos > 0.99996).
    float disk = smoothstep(0.99996, 0.99999, cosGamma);
    float3 sun_color_high = float3(1.00, 0.95, 0.85);
    float3 sun_color_low  = float3(1.00, 0.55, 0.20);
    float3 sun_col = lerp(sun_color_low, sun_color_high, day);
    sky += disk * 50.0 * sun_col * day;

    // Soft halo around the sun (Mie-ish, falls off rapidly).
    float halo = pow(max(0.0, cosGamma), 80.0);
    sky += halo * sun_col * 0.30 * day;
    // Wider, dimmer secondary scatter (Rayleigh-tinted) extends the
    // sunset orange across more of the sky.
    float wide = pow(max(0.0, cosGamma), 6.0);
    sky += wide * sun_col * sunset * 0.50;

    return max(sky, float3(0.0));
}

// Sun disc + atmospheric halo. Real solar angular diameter is 0.53deg
// (almost identical to the moon -- the reason solar eclipses work).
// Render with an actual disc + a halo whose extent grows at low sun
// elevation, matching the bright bloom you see around the sun at
// sunset/sunrise (more atmospheric thickness + dust = more scatter).
//
// The disc itself stays the same physical 0.55deg half-angle; what
// changes between noon and sunset is the colour (warm/red at low
// elev) and the halo size (much bigger at low elev, fading toward
// a tight halo at zenith). This matches real photos: sunset sun
// SEEMS bigger because the bright halo reads as part of the disc.
float3 sunDisc(float3 rd, float3 sun_dir) {
    if (sun_dir.y < -0.10) return float3(0.0);
    float ang = dot(rd, sun_dir);
    // r_sun_size * astronomical-distance ratio. perihelion ~1.017
    // (sun looks bigger), aphelion ~0.984. Half-angle scales linearly.
    float sun_mult = max(sun_extra.x * sun_extra.y, 0.1);
    float r_deg    = 0.55 * sun_mult;
    float kCosR    = cos(r_deg * 0.0174533);
    // Halo extent grows from cos(2deg) at zenith to cos(8deg) near
    // horizon. The horizon halo is what you see as the "bigger" sun
    // at sunset -- it's atmospheric scattering, not the disc itself.
    float halo_deg   = lerp(2.0, 8.0, saturate(1.0 - sun_dir.y));
    float halo_rad   = halo_deg * 0.0174533;
    float kCosHalo   = cos(halo_rad);
    if (ang <= kCosHalo) return float3(0.0);

    // Atmospheric horizon refraction differentially lifts the lower
    // limb more than the upper limb (Saemundsson 1986), squashing the
    // apparent disc into a vertical oval. CPU computes the ratio in
    // sun_extra.z; 1.0 means circular. We test ellipticity in the
    // sun's local horizon frame so only the disc squishes -- the halo
    // is at-observer scattering (kept circular by design; gradual
    // falloff hides any anisotropy that does exist).
    float vflat = sun_extra.z;
    bool oval = vflat < 0.999;

    // Disc-membership test. Circular fast path when no flatten;
    // elliptical test in sun-local frame otherwise. The squashed oval
    // is fully contained inside the original circular cone (vertical
    // radius shrinks; horizontal unchanged), so `ang > kCosR` is a
    // valid superset gate for both branches -- pixels failing it
    // can't be inside the oval either.
    bool inside_disc = false;
    if (ang > kCosR) {
        if (!oval) {
            inside_disc = true;
        } else {
            // Build sun-local horizon frame. Safe at non-zenith sun
            // (vflat < 0.999 only fires below ~6deg elev, far from
            // zenith singularity).
            float3 world_up    = float3(0.0, 1.0, 0.0);
            float3 local_right = normalize(cross(world_up, sun_dir));
            float3 local_up    = cross(sun_dir, local_right);
            float3 delta       = rd - sun_dir * ang;            // perp to sun
            float h_off        = dot(delta, local_right);
            float v_off        = dot(delta, local_up);
            float r_h          = r_deg * 0.0174533;
            float r_v          = r_h * vflat;
            float ellipse      = (h_off * h_off) / (r_h * r_h)
                               + (v_off * v_off) / (r_v * r_v);
            inside_disc = ellipse <= 1.0;
        }
    }

    // Atmospheric attenuation + warmth, matching sun_rad in NEE.
    float am  = 1.0 / max(sun_dir.y, 0.04);
    float atm = exp(-0.30 * am);
    float warmth = saturate((am - 1.0) * 0.18);
    float3 col = lerp(float3(1.00, 0.98, 0.94),
                      float3(1.55, 0.55, 0.20), warmth);

    // Halo: soft bloom around the disc.
    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        // Steeper falloff at zenith (tight halo), wider at horizon.
        float falloff = lerp(8.0, 3.0, saturate(1.0 - sun_dir.y));
        halo = pow(t_h, falloff);
    }

    if (inside_disc) {
        // Bright disc -- sun is enormously brighter than anything
        // else; ACES will clip the centre to white but the limb +
        // halo carry the warmth.
        return col * (250.0 * atm)
             + col * halo * atm * 4.0;
    }
    return col * halo * atm * 4.0;
}

// Moon disc + halo. Real lunar angular radius is 0.00465 rad (~0.27deg
// half-angle = 0.54deg full).
//   - terminator: per-pixel lit/unlit check via dot(surface_normal,
//     sun_dir). Replaces the previous flat phase-brightness scalar so
//     half-moon shows a curved terminator splitting the disc instead
//     of dimming the whole thing uniformly. New moon -> all unlit.
//   - moon_map texture: real surface (mare + crater) sampled in the
//     moon's tangent frame. Tidally locked, so tangent UV maps
//     directly to near-side surface coordinates.
//   - limb darkening: edge of disc dimmer than centre.
//   - atmospheric attenuation: same model as sun_rad (dims with airmass
//     when moon is near horizon, warms toward red).
//   - subtle halo: soft ring extending to ~1.5deg, simulating
//     atmospheric scattering around the moon.
float3 moonDisc(float3 rd, float4 mph, float3 sun_dir_world) {
    if (mph.y < -0.10) return float3(0.0);
    float3 m_dir = mph.xyz;
    float ang = dot(rd, m_dir);
    // Real lunar angular radius is 0.27 deg; we render at 0.55 deg
    // (~2x real) for visibility at typical 60-FOV 1080p, then scale
    // by r_moon_size (moon_extra.x) AND astronomical distance ratio
    // (moon_extra.y; perigee ~1.06, apogee ~0.95) so supermoons
    // really do render bigger than micro-moons.
    float size_mult = max(moon_extra.x * moon_extra.y, 0.1);
    float r_deg     = 0.55 * size_mult;
    float halo_deg  = 1.5  * size_mult;
    float kCosR     = cos(r_deg    * 0.0174533);
    float kCosHalo  = cos(halo_deg * 0.0174533);
    if (ang <= kCosHalo) return float3(0.0);

    float3 col_tint = float3(0.92, 0.94, 1.00);

    // Atmospheric attenuation matching sun_rad.
    float moon_am   = 1.0 / max(mph.y, 0.04);
    float moon_atm  = exp(-0.30 * moon_am);
    float warmth_m  = saturate((moon_am - 1.0) * 0.18);
    float3 col_atm  = lerp(float3(1.00, 1.00, 1.00),
                           float3(1.20, 0.65, 0.40), warmth_m);

    // Halo ring extends beyond the disc.
    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        halo = pow(t_h, 4.0) * 0.6;
    }

    if (ang > kCosR) {
        float t_d = (ang - kCosR) / (1.0 - kCosR);
        // Tangent frame around the moon. (du, dv) project the view ray
        // onto the disc plane in moon-local coords.
        float3 ax = abs(m_dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 t_a = normalize(cross(ax, m_dir));
        float3 t_b = cross(m_dir, t_a);
        // Scale factor 1/sin(disc half-angle) to map disc-tangent
        // projection to [-1, 1] across the visible diameter. Includes
        // the moon-size multiplier so "big moon" debug mode still
        // keeps the texture mapped onto the disc.
        float disc_scale = 1.0 / max(sin(r_deg * 0.0174533), 1e-5);
        float du = dot(rd, t_a) * disc_scale;
        float dv = dot(rd, t_b) * disc_scale;
        float r2 = du * du + dv * dv;
        float zz = sqrt(max(1.0 - r2, 0.0));

        // Outward surface normal in WORLD frame for this visible point.
        // The visible hemisphere of a sphere at distance D in direction
        // m_dir, viewed from origin, has surface normals pointing back
        // TOWARD the observer (i.e. the -m_dir half), with tangential
        // offsets parameterised by (du, dv). At the disc centre the
        // normal is -m_dir (closest point); at the limb it lies in the
        // tangent plane. Previous + sign on the m_dir term picked the
        // far hemisphere -- caused the terminator to invert and full
        // moon to read as fully unlit.
        float3 n_world = du * t_a + dv * t_b - zz * m_dir;

        // Terminator: lit-fraction at this surface point. Smooth
        // transition over a few percent for an aliasing-free edge.
        float lit = saturate(dot(n_world, sun_dir_world) * 10.0 + 0.05);
        if (lit <= 0.0) return col_tint * col_atm * halo * moon_atm * 0.05;

        // Lunar eclipse / blood moon. Earth's umbra at the moon's
        // distance is ~0.7deg radius (penumbra ~1.25deg). When the
        // moon's centre falls within this cone -- i.e. within ~0.7deg
        // of the antisolar point (-sun_dir) -- direct sunlight is
        // blocked. Atmospheric refraction through Earth's limb
        // smears red light into the umbra (the same Rayleigh
        // attenuation that reddens our own sunsets), so the moon
        // glows deep red instead of going pitch dark. Smooth
        // penumbra transition so partial eclipses blend.
        float anti_sun_dot = -dot(m_dir, sun_dir_world);
        float anti_sun_ang = acos(saturate(anti_sun_dot));
        const float kUmbraRad    = 0.0122;     // ~0.7 deg
        const float kPenumbraRad = 0.0218;     // ~1.25 deg
        float eclipse = 1.0 - smoothstep(kUmbraRad, kPenumbraRad,
                                         anti_sun_ang);

        // Sample the procedural moon map. Equirectangular UV from
        // local (du, dv, zz). Latitude from dv (north/south of moon
        // equator); longitude from du / zz (east/west of moon center).
        const float kHalfPi = 1.57079632679;
        float lon = atan2(du, zz);                              // [-pi/2, pi/2]
        float lat = asin(clamp(dv, -1.0, 1.0));                 // [-pi/2, pi/2]
        // Moon texture is full sphere (2pi wide, pi tall); near side
        // maps to roughly the centre half. u in [0.25, 0.75] keeps
        // texture variation across the visible disc.
        float u_tex = 0.5 + lon / 6.28318530718;
        float v_tex = 0.5 - lat / 3.14159265358979;
        u_tex = saturate(u_tex);
        v_tex = saturate(v_tex);
        uint2 mdim;
        moon_map.GetDimensions(mdim.x, mdim.y);
        float2 fpix = float2(u_tex * float(mdim.x), v_tex * float(mdim.y)) - 0.5;
        int2 ipix = int2(floor(fpix));
        float2 f = fpix - float2(ipix);
        int x0 = clamp(ipix.x,     0, int(mdim.x) - 1);
        int x1 = clamp(ipix.x + 1, 0, int(mdim.x) - 1);
        int y0 = clamp(ipix.y,     0, int(mdim.y) - 1);
        int y1 = clamp(ipix.y + 1, 0, int(mdim.y) - 1);
        float3 c00 = moon_map[uint2(x0, y0)].rgb;
        float3 c10 = moon_map[uint2(x1, y0)].rgb;
        float3 c01 = moon_map[uint2(x0, y1)].rgb;
        float3 c11 = moon_map[uint2(x1, y1)].rgb;
        float3 surface = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);

        // Limb darkening (real lunar limb is darker due to roughness).
        float limb = 0.55 + 0.45 * sqrt(saturate(t_d));

        // Disc multiplier 3.5 baseline scaled by user-controlled
        // r_moon_brightness (moon_extra.z). 1.0 = baseline; 0.7
        // (default) reads as a softer night moon where surface texture
        // stays well below the ACES knee. Halo scales with brightness
        // so the surrounding bloom stays proportionate.
        float bright_mult = max(moon_extra.z, 0.0);
        float disc_scale_b = 3.5 * bright_mult;
        float3 lit_col = col_tint * col_atm * surface
                       * (disc_scale_b * lit * limb * moon_atm)
                       + col_tint * col_atm * halo * lit * moon_atm
                       * (0.6 * bright_mult);

        // Blood-moon colour during eclipse: deep red-orange (the
        // refracted-through-Earth-atmosphere light tone). Surface
        // texture still shows through, just heavily red-tinted.
        // Brightness drops a lot -- a typical totality has the moon
        // ~1/10000 of normal full-moon brightness.
        float3 eclipse_col = float3(0.55, 0.15, 0.04) * surface
                           * (limb * moon_atm * 1.2);
        return lerp(lit_col, eclipse_col, eclipse);
    }
    // Halo only -- no surface here. Halo dims toward new-moon by the
    // average disc lit fraction, approximated by phase brightness.
    float phase_b = 0.5 * (1.0 - cos(mph.w));
    return col_tint * col_atm * halo * phase_b * moon_atm * 0.3;
}

// Per-contribution firefly clamp. Bounds a single indirect light
// contribution (NEE off the env map, ambient skylight, bounce-to-sky
// from b>0) so a single-sample spike from a tiny PDF or BSDF-sampled
// bounce hitting an HDRI sun pixel doesn't blow the running mean.
// Camera-direct sky paths (b == 0 sky-on-miss) bypass this so the
// sun still appears at full intensity in the rendered sky.
//
// Hue-preserving: scale by max-channel ratio rather than per-channel
// min(). A naive per-channel clamp would turn an orange contribution
// (5, 3, 1) into pure grey (1, 1, 1) at ceiling=1 -- the sunset's
// warm bounce light loses its color. Max-channel scaling maps the
// same input to (1, 0.6, 0.2), preserving hue and just bounding
// magnitude. This lets a ceiling of ~10 actually remain on without
// neutralising the indirect colour cast.
//
// Engine plumbs the ceiling via w2j_row2.w; 0 disables.
float3 clampIndirect(float3 c) {
    if (w2j_row2.w <= 0.0) return c;
    float m = max(c.r, max(c.g, c.b));
    return (m > w2j_row2.w) ? c * (w2j_row2.w / m) : c;
}

float3 skyColor(float3 rd) {
    float3 base;
    uint mode = uint(sun_and_mode.w);
    if (mode == 2u) {
        base = procSky(rd, sun_and_mode.xyz);
    } else if (env_map_present != 0u) {
        // Original env_map path follows.
        base = float3(0.0);   // filled below
    } else {
        float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
        base = lerp(float3(0.50, 0.55, 0.62), float3(1.0, 1.0, 1.0), t);
    }
    if (mode == 2u) {
        return base + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDisc(rd, sun_and_mode.xyz);
    }
    if (env_map_present != 0u) {
        // Lat-long lookup: u from atan2 of horizontal plane, v from
        // arccos of vertical component. (u, v) in [0,1].
        uint2 dim;
        env_map.GetDimensions(dim.x, dim.y);
        float u = atan2(rd.z, rd.x) * (1.0 / 6.28318530718) + 0.5;
        float v = acos(clamp(rd.y, -1.0, 1.0)) * (1.0 / 3.14159265359);
        // Bilinear by hand: nearest-neighbour gives visible quantisation
        // around the sun where the env-map is brightest.
        float2 fpix = float2(u * float(dim.x), v * float(dim.y)) - 0.5;
        int2  ipix = int2(floor(fpix));
        float2 f   = fpix - float2(ipix);
        int x0 = (ipix.x % int(dim.x) + int(dim.x)) % int(dim.x);
        int x1 = (x0 + 1) % int(dim.x);
        int y0 = clamp(ipix.y, 0, int(dim.y) - 1);
        int y1 = clamp(ipix.y + 1, 0, int(dim.y) - 1);
        float3 c00 = env_map[uint2(x0, y0)].rgb;
        float3 c10 = env_map[uint2(x1, y0)].rgb;
        float3 c01 = env_map[uint2(x0, y1)].rgb;
        float3 c11 = env_map[uint2(x1, y1)].rgb;
        float3 sky = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
        // P12: HDRI mode is the canonical "all info from the HDRI"
        // path. The HDRI texture itself contains the sun, moon, stars
        // and any other celestial bodies baked in. We deliberately do
        // NOT add a procedural sunDisc / moonDisc overlay -- the user's
        // expectation is that the rendered sky matches the HDRI source.
        // Crisp directional shadows come from the stochastic NEE over
        // hdri_lights[] (extracted bright clusters) in the path
        // tracer's Lambert hit handling.
        return sky * env_intensity;
    }
    return base + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDisc(rd, sun_and_mode.xyz);
}

float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c*(a*c+b))/(c*(d*c+e)+f));
}

#if defined(PT_TARGET_SPIRV)
// Standard sRGB OETF (linear -> nonlinear, IEC 61966-2-1: linear seg
// near zero + gamma ~2.4 above). Vulkan storage-image writes do NOT
// auto-encode for sRGB swapchain formats per spec (only color-
// attachment writes do). The Mac/Metal path uses Apple silicon's
// implicit storage-write encoding for BGRA8Unorm_sRGB and skips this.
float3 srgb_oetf(float3 c) {
    c = max(c, float3(0.0, 0.0, 0.0));
    float3 lin  = c * 12.92;
    float3 expo = 1.055 * pow(c, float3(1.0/2.4)) - 0.055;
    return select(c < 0.0031308, lin, expo);
}
#endif

// ---------- env-map importance sampling (NEE for Lambert) ----------------
struct EnvSample {
    float3 dir;
    float3 color;     // env_map sample value times env_intensity
    float  pdf;       // solid-angle PDF for this direction
};

// Binary search a monotonically-non-decreasing CDF in [base, base+count) for
// the smallest index i such that cdf[base+i] >= u. CDF expected to end at 1.0.
uint cdfSearch(uint base, uint count, float u) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (conditional_cdf[base + mid] < u) lo = mid + 1;
        else                                  hi = mid;
    }
    return min(lo, count - 1);
}
uint cdfSearchMarginal(uint count, float u) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (marginal_cdf[mid] < u) lo = mid + 1;
        else                        hi = mid;
    }
    return min(lo, count - 1);
}

EnvSample sampleEnvMap(float2 xi) {
    EnvSample s;
    s.dir = float3(0,1,0); s.color = float3(0); s.pdf = 0.0;
    if (env_map_present == 0u || env_total_luminance <= 0.0) return s;

    uint W, H;
    env_map.GetDimensions(W, H);

    uint v_idx = cdfSearchMarginal(H, xi.y);
    uint u_idx = cdfSearch(v_idx * W, W, xi.x);

    float u = (float(u_idx) + 0.5) / float(W);
    float v = (float(v_idx) + 0.5) / float(H);

    float theta = v * 3.14159265358979;
    float phi   = u * 6.28318530717958 - 3.14159265358979;
    float sin_theta = sin(theta);

    s.dir = float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));

    float3 raw  = env_map[uint2(u_idx, v_idx)].rgb;
    s.color     = raw * env_intensity;

    float lum   = 0.2126 * raw.r + 0.7152 * raw.g + 0.0722 * raw.b;
    // p(u,v) in pixel-space = lum*sin / total. Solid-angle Jacobian for
    // lat-long: dω = (2π² / (W*H)) * sin(θ) du dv. So p(ω) = p(u,v) /
    // jacobian = (lum * W * H) / (2π² * total).
    if (sin_theta > 1e-6) {
        s.pdf = (float(lum) * float(W * H)) /
                (2.0 * 3.14159265358979 * 3.14159265358979 * env_total_luminance);
    }
    return s;
}

// Returns true iff the direction maps to an HDRI extracted-cluster
// pixel -- one where multi-light extraction zeroed the env-map CDF's
// contribution. The CDF prefix sum doesn't advance at those pixels
// (0.0 is added), so conditional_cdf[base+u] == conditional_cdf[base+u-1]
// bit-identically. Used by both pdfEnvMapDir (for early-return) and
// the BRDF-side MIS miss handler (to avoid double-counting with the
// extracted-lights NEE that handles cluster flux on its side).
bool isExtractedClusterDir(float3 dir) {
    if (env_map_present == 0u || hdri_lights_count == 0u) return false;
    float ct = clamp(dir.y, -1.0, 1.0);
    float th = acos(ct);
    float st = sin(th);
    if (st < 1e-6) return false;
    float ph = atan2(dir.z, dir.x);
    uint W, H;
    env_map.GetDimensions(W, H);
    float u = (ph + 3.14159265358979) * (1.0 / 6.28318530717958);
    float v = th * (1.0 / 3.14159265358979);
    uint u_idx = min(uint(u * float(W)), W - 1u);
    uint v_idx = min(uint(v * float(H)), H - 1u);
    uint base = v_idx * W;
    float c_curr = conditional_cdf[base + u_idx];
    float c_prev = (u_idx > 0u) ? conditional_cdf[base + u_idx - 1u] : 0.0;
    return c_curr <= c_prev;
}

// Evaluate the env-map importance-sampling PDF (solid-angle) for an
// arbitrary world-space direction. Mirrors sampleEnvMap's PDF formula
// for the texel the direction projects to. Returns 0 when the env map
// isn't bound, the total luminance integral is degenerate, or the
// direction lands at a pole (sin theta -> 0 -> infinite Jacobian).
//
// Caller contract: must NOT pass extracted-cluster directions. The
// cluster check is gated at the call site (the MIS sky-miss block in
// the path-trace loop) so we don't perform it twice on every miss.
// pdfEnvMapDir does not re-check, and would return a spurious positive
// value for a cluster direction (it reads the raw env_map texel, which
// keeps cluster pixels intact).
float pdfEnvMapDir(float3 dir) {
    if (env_map_present == 0u || env_total_luminance <= 0.0) return 0.0;
    float cos_theta = clamp(dir.y, -1.0, 1.0);
    float theta     = acos(cos_theta);
    float sin_theta = sin(theta);
    if (sin_theta < 1e-6) return 0.0;
    float phi = atan2(dir.z, dir.x);
    uint W, H;
    env_map.GetDimensions(W, H);
    float u = (phi + 3.14159265358979) * (1.0 / 6.28318530717958);
    float v = theta * (1.0 / 3.14159265358979);
    uint u_idx = min(uint(u * float(W)), W - 1u);
    uint v_idx = min(uint(v * float(H)), H - 1u);
    float3 raw = env_map[uint2(u_idx, v_idx)].rgb;
    float lum  = 0.2126 * raw.r + 0.7152 * raw.g + 0.0722 * raw.b;
    return (lum * float(W * H)) /
           (2.0 * 3.14159265358979 * 3.14159265358979 * env_total_luminance);
}

// Balance heuristic between two single-sample strategies. Returns the
// MIS weight to apply to a contribution drawn from `pdf_self` given the
// other strategy would sample the same direction with `pdf_other`.
// Sums to 1 across the two strategies, so unbiased combined estimator.
float misBalance(float pdf_self, float pdf_other) {
    return pdf_self / max(pdf_self + pdf_other, 1e-12);
}

// ---------- HDRI extracted-light stochastic NEE -------------------------
// Single-pick over the up-to-8 directional lights extracted from the
// HDRI at load. Selection is luminance-weighted via the pmf stored in
// hdri_lights_dir[i].w (sums to 1 across lights). The contribution
// estimator is (color / pmf), which converges over spp + temporal to
// the integral over all lights -- replaces "evaluate every light"
// (cap × shadow-ray cost per sample) with one shadow ray per sample.
struct HdriLightSample {
    float3 dir;
    float3 color;     // raw cluster flux times env_intensity
    float  pdf;       // probability of selecting this light
};
HdriLightSample sampleHdriLight(inout uint seed) {
    HdriLightSample s;
    s.dir = float3(0,1,0); s.color = float3(0); s.pdf = 0.0;
    if (hdri_lights_count == 0u) return s;
    float u = randf(seed);
    float acc = 0.0;
    uint sel = 0u;
    [unroll(8)]
    for (uint i = 0u; i < 8u; ++i) {
        if (i >= hdri_lights_count) break;
        float p = hdri_lights_dir[i].w;
        acc += p;
        if (u <= acc) { sel = i; break; }
        sel = i;   // fallback to last if numerical drift makes acc < 1
    }
    s.dir   = hdri_lights_dir[sel].xyz;
    s.color = hdri_lights_col[sel].rgb * env_intensity;
    s.pdf   = max(hdri_lights_dir[sel].w, 1e-6);
    return s;
}

// RGB transmittance along a shadow ray. 1.0 = unoccluded, 0.0 = blocked
// by an opaque hit. Dielectric hits don't block: the ray refracts
// through (in -> out), continuing toward the light, multiplied by the
// dielectric's tint (cheap stand-in for Beer's-law absorption). This
// is what turns "shadow under a glass diamond" into actual caustic
// patterns -- the focal lensing emerges from the geometry the
// refracted shadow ray traces, not from any fancy sampling.
//
// Optical depth through the cloud layer along [ro, ro + rd*t_max].
// Returns 0 when clouds are off or the ray misses the layer; otherwise
// integrates cloud_density_at across the ray-slab intersection. Used
// by transmittance() to give clouds genuine shadow-casting behaviour:
// any NEE shadow ray that grazes a dense cloud loses light.
float cloud_optical_depth(float3 ro, float3 rd, float t_max) {
    if (clouds_p1.w <= 0.0) return 0.0;
    float base_y = clouds_p1.x;
    float top_y  = clouds_p1.y;
    float ry = rd.y;
    float t_in, t_out;
    if (abs(ry) > 1e-6) {
        float t_b = (base_y - ro.y) / ry;
        float t_t = (top_y  - ro.y) / ry;
        t_in  = max(0.0, min(t_b, t_t));
        t_out = max(t_b, t_t);
    } else {
        if (ro.y >= base_y && ro.y <= top_y) {
            t_in  = 0.0;
            t_out = t_max;
        } else {
            return 0.0;
        }
    }
    t_out = min(t_out, t_max);
    // Cap shadow march distance: a near-horizon ray crossing the cloud
    // layer at very shallow elevation can span hundreds of kilometres,
    // far more than the cloud field's coherence length. 20km is plenty
    // to capture the shadow of any reasonable cloud bank in front of
    // the sun, and limits the sample budget.
    if (t_out - t_in > 20000.0) t_in = t_out - 20000.0;
    if (t_out <= t_in) return 0.0;
    // 4 coarse samples + 3 fbm octaves for the shadow ray. Beer-Lambert
    // exp(-OD) only needs a rough density estimate to look right.
    const int Ns = 4;
    float dt = (t_out - t_in) / float(Ns);
    float od = 0.0;
    [unroll] for (int i = 0; i < Ns; ++i) {
        float t = t_in + (float(i) + 0.5) * dt;
        float3 sp = ro + rd * t;
        od += cloud_density_at(sp, clouds_p1, clouds_p2, clouds_p3, 3) * dt;
    }
    return od;
}

// Analytical atmospheric transmittance (Mie + Rayleigh) along the
// straight ray segment [ro, ro + rd*t]. Used to give cloud-march
// samples and (in future) primary-ray surface hits aerial-perspective
// fade with distance, regardless of sky mode -- the HDRI is the
// canonical sky source but doesn't model the air mass between the
// camera and a far cloud, so distant clouds need this attenuation
// applied on top.
//
// Both Mie and Rayleigh have an exponential altitude profile:
//     sigma(h) = sigma0 * exp(-h / H_scale)
// On a straight slanted ray h(s) = ro.y + rd.y * s, so the optical
// depth ∫[0,t] sigma(h(s)) ds is closed-form:
//     rd.y != 0:  H/rd.y * (exp(-ro.y/H) - exp(-(ro.y + rd.y*t)/H))
//     rd.y == 0:  t * exp(-ro.y/H)   (horizontal ray at constant alt)
// Multiplied by sigma0 for each species.  Returns the per-channel
// transmittance exp(-tau).
//
// vol_params.x is the user-controlled Mie sigma at sea level; if it's
// 0 the user has disabled Mie haze and we return Rayleigh-only.
// clouds_p3.w is the Rayleigh intensity multiplier (0 = no Rayleigh,
// 1 = real-Earth sigma).  Both gates honoured so a debug "no
// atmosphere" mode still works.
float3 atmosphericTransmittance(float3 ro, float3 rd, float t) {
    if (t <= 0.0) return float3(1.0);
    const float kHmie      = 1500.0;
    const float kHrayleigh = 8000.0;
    const float3 kSigmaR   = float3(5.8e-6, 13.5e-6, 33.1e-6);
    float  sigma_mie0 = vol_params.x;
    float  rayleigh_i = clouds_p3.w;
    bool   any_mie    = sigma_mie0 > 0.0;
    bool   any_ray    = rayleigh_i > 0.0;
    if (!any_mie && !any_ray) return float3(1.0);
    // Vertical path-length factor (analytical altitude integral).
    float gv_mie, gv_ray;
    if (abs(rd.y) > 1e-6) {
        float a_m = exp(-ro.y / kHmie);
        float b_m = exp(-(ro.y + rd.y * t) / kHmie);
        gv_mie = (a_m - b_m) * (kHmie / rd.y);
        float a_r = exp(-ro.y / kHrayleigh);
        float b_r = exp(-(ro.y + rd.y * t) / kHrayleigh);
        gv_ray = (a_r - b_r) * (kHrayleigh / rd.y);
    } else {
        gv_mie = t * exp(-max(ro.y, 0.0) / kHmie);
        gv_ray = t * exp(-max(ro.y, 0.0) / kHrayleigh);
    }
    float  tau_mie = any_mie ? sigma_mie0 * gv_mie : 0.0;
    float3 tau_ray = any_ray ? kSigmaR * rayleigh_i * gv_ray : float3(0.0);
    return exp(-(tau_ray + float3(tau_mie)));
}

// Up to kMaxRefract dielectric bounces; beyond that the ray gives up
// (returns 0) so a pathological all-glass scene can't loop forever.
// TIR also returns 0 (light can't pass that interface).
float3 transmittance(float3 ro, float3 rd, float t_max) {
    // Cloud optical depth along the original (un-refracted) ray. Real
    // clouds live in a layer high above the scene -- approximating the
    // shadow as the straight-line cloud path is fine because no glass
    // sits inside the cloud layer for the refracted segment to matter.
    float3 cloud_t = float3(exp(-cloud_optical_depth(ro, rd, t_max)));
    // Engine packs the per-frame refraction budget into w2j_row1.w
    // (0 = caustics disabled, treat dielectrics as opaque shadow
    // blockers; >0 = max dielectric bounces a single shadow ray may
    // chain through). Capped to 16 in the engine for safety.
    int kMaxRefract = int(w2j_row1.w);
    float3 t = float3(1.0);
    float3 o = ro;
    float3 d = rd;
    for (int i = 0; i <= kMaxRefract; ++i) {
        HitInfo h = traceScene(o, d);
        if (!h.hit || h.t >= t_max) return t * cloud_t;     // unoccluded
        if (h.mat != MAT_DIELECTRIC) return float3(0.0);    // opaque blocker
        if (i == kMaxRefract)        return float3(0.0);    // budget hit

        // Refract through the dielectric face. Same Snell + Fresnel
        // logic as the path-tracer's dielectric BRDF, but we keep
        // only the transmitted lobe (the reflected lobe doesn't reach
        // the light along this path). Fresnel is folded into the
        // transmittance so glancing angles correctly attenuate.
        float ndi   = dot(h.normal, d);
        float3 nf   = (ndi < 0.0) ? h.normal : -h.normal;
        float ior_a = (ndi < 0.0) ? 1.0   : h.ior;
        float ior_b = (ndi < 0.0) ? h.ior : 1.0;
        float eta   = ior_a / ior_b;
        float cos_i = abs(dot(nf, -d));
        float r0    = (ior_a - ior_b) / (ior_a + ior_b);
        r0 = r0 * r0;
        float fr    = fresnelSchlick(cos_i, r0);   // reflected fraction
        float3 refracted = refract(d, nf, eta);
        if (length(refracted) < 1e-3) return float3(0.0);   // TIR
        // Transmitted fraction times Beer-ish tint.
        t *= (1.0 - fr) * h.albedo;
        // Step past the interface, continue in the refracted direction.
        float advance = h.t + 1e-3;
        o = o + d * advance;
        d = normalize(refracted);
        t_max = max(t_max - advance, 0.0);
    }
    return float3(0.0);
}

// Convenience predicate around transmittance() for the metal /
// dielectric branches that just want the legacy boolean answer.
bool occluded(float3 ro, float3 rd, float t_max) {
    return all(transmittance(ro, rd, t_max) <= float3(0.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    uint2 dim;
    output.GetDimensions(dim.x, dim.y);
    if (any(tid >= dim)) return;

    uint seed = pcgHash(tid.x * 1973u + tid.y * 9277u + frame_index * 26699u);

    // P10 G-buffer pass (no shading): unjittered primary ray writes
    // depth + motion vectors for MetalFX TemporalDenoisedScaler.
    // Stable per-pixel data is what the denoiser wants; we don't
    // include the pixel jitter here.
    if (denoiser_enabled != 0u) {
        float2 uv0 = ((float2(tid) + 0.5) / float2(dim)) * 2.0 - 1.0;
        uv0.y = -uv0.y;
        float3 ro0 = pos_fovtan.xyz;
        float3 rd0 = normalize(fwd_aspect.xyz
                             + right_xyz.xyz * (uv0.x * fwd_aspect.w * pos_fovtan.w)
                             + up_xyz.xyz    * (uv0.y * pos_fovtan.w));
        HitInfo h0 = traceScene(ro0, rd0);

        float3 hit_world;
        float3 hit_normal;
        float3 hit_albedo;   // linear RGB at primary hit; sky writes float3(0)
        float  depth_view;   // linear camera-space Z (world units along forward)
        if (h0.hit) {
            hit_world  = ro0 + rd0 * h0.t;
            hit_normal = h0.normal;
            hit_albedo = h0.albedo;
            depth_view = h0.t * dot(rd0, normalize(fwd_aspect.xyz));
        } else {
            hit_world  = ro0 + rd0 * 1000.0;   // far point so sky still gets motion
            hit_normal = float3(0.0, 0.0, 0.0); // sentinel: sky / no surface
            hit_albedo = float3(0.0, 0.0, 0.0); // sky has no surface albedo
            depth_view = 1000.0;
        }

        // Pixel-space motion via reprojection. We still need the
        // perspective matrices for this even though depth itself goes
        // to MetalFX in linear units.
        float4 pc = mul(curr_view_proj, float4(hit_world, 1.0));
        float4 pp = mul(prev_view_proj, float4(hit_world, 1.0));
        float2 cur_pix = (float2( pc.x, -pc.y) / max(pc.w, 1e-6) * 0.5 + 0.5) * float2(dim);
        float2 pre_pix = (float2( pp.x, -pp.y) / max(pp.w, 1e-6) * 0.5 + 0.5) * float2(dim);
        float2 motion  = pre_pix - cur_pix;

        depth_tex[tid]  = depth_view;
        motion_tex[tid] = motion;
        // Normal G-buffer: only written when the active denoiser
        // actually consumes it (SVGF + OptiX AOV variants). The engine
        // doesn't allocate or bind normal_tex_id_ for metalfx / plain
        // optix_hdr / optix_temporal_hdr, so an unconditional write
        // here would hit an unbound texture on Metal and rely on
        // robust-buffer-access for a no-op on Vulkan -- the explicit
        // gate is cleaner on both and saves one RGBA16F write per
        // pixel per frame on non-normal-consuming modes.
        if (write_normal_gbuffer != 0u) {
            normal_tex[tid] = float4(hit_normal, 0.0);
        }
#ifdef PT_TARGET_SPIRV
        // Albedo G-buffer: only allocated/bound by the engine when
        // r_denoiser is optix_hdr_aov / optix_temporal_hdr_aov. Gated
        // because the descriptor layout uses
        // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT -- binding 17 is
        // simply absent from the dispatch's descriptor view in other
        // modes (it used to be a null descriptor + robust-buffer-
        // access no-op, but MoltenVK on Apple Silicon doesn't
        // implement nullDescriptor, hence the move to partially-
        // bound). Metal SVGF doesn't use albedo so the declaration
        // stays SPIR-V-only -- adding a new texture slot on the
        // Metal side has a real cost (MSL texture-index assignment
        // cascades) and SVGF without an albedo guide matches the
        // Vulkan svgf_basic / svgf_atrous behaviour exactly.
        if (write_albedo_gbuffer != 0u) {
            albedo_tex[tid] = float4(hit_albedo, 0.0);
        }
#endif
    }

    // Multi-sample loop: average `spp` independent paths into a single
    // radiance estimate. spp>1 buys cleaner frames during motion at a
    // proportional GPU cost; spp=1 is the classic 1-spp interactive mode.
    uint sample_count = (spp == 0u) ? 1u : spp;
    float3 frame_radiance = float3(0.0);
    // Star-split accumulator for issue #46. When star_split_enabled
    // AND denoiser_enabled, the primary-miss sky term is split into
    // (sky_minus_stars, stars_only). frame_radiance gets the former
    // (denoised normally); this gets the latter (composited back in
    // post-denoise so the a-trous filter can't smudge it). Stays at 0
    // when star-split is off; the texture write below is gated, so the
    // unused slot stays harmless.
    float3 frame_stars    = float3(0.0);
    // Star-split gate. star_split_enabled is in the Frame UBO on
    // Vulkan and the Metal Push cbuffer on Metal, but the field name
    // and offset are identical from the shader's perspective so this
    // single expression works on both targets.
    const bool star_split_active =
        (denoiser_enabled != 0u) && (write_hdr_aux != 0u) &&
        (star_split_enabled != 0u);
    for (uint s = 0u; s < sample_count; ++s) {
        // When MetalFX is on, all pixels of the frame share the same
        // sub-pixel offset (Halton sequence picked on host) so the
        // denoiser can reconstruct the supersampled result. With the
        // denoiser off, per-pixel random jitter gives free AA via
        // accumulation. spp>1 always gets per-sample random offsets so
        // multi-sampling actually reduces variance.
        float2 jitter;
        if (denoiser_enabled != 0u && s == 0u) {
            jitter = halton_jitter;
        } else {
            jitter = float2(randf(seed), randf(seed)) - 0.5;
        }
        float2 uv = ((float2(tid) + 0.5 + jitter) / float2(dim)) * 2.0 - 1.0;
        uv.y = -uv.y;

        float3 ro = pos_fovtan.xyz;
        float3 rd = normalize(fwd_aspect.xyz
                           + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                           + up_xyz.xyz    * (uv.y * pos_fovtan.w));

        // Thin-lens DOF. Sample a point on the aperture, recompute
        // the ray to pass through the focal-plane intersection of
        // the original pinhole ray. Aperture sampling shape =
        // round disk for circular bokeh, regular polygon for
        // polygonal-iris bokeh (matching real lens aperture blades).
        // dof_params.x = aperture radius (0 = skip, pinhole).
        if (dof_params.x > 0.0) {
            float aperture = dof_params.x;
            float focal_d  = dof_params.y;
            int   blades   = int(dof_params.z + 0.5);
            float u1 = randf(seed), u2 = randf(seed);
            float2 ap;
            if (blades < 3) {
                // Concentric-disk mapping (Shirley-Chiu): square
                // (u1,u2) -> uniformly distributed unit disk.
                float a = 2.0 * u1 - 1.0;
                float b = 2.0 * u2 - 1.0;
                float r, phi;
                if (abs(a) > abs(b)) {
                    r = a;
                    phi = (3.14159265358979 / 4.0) * (b / max(abs(a), 1e-6));
                } else {
                    r = b;
                    phi = (3.14159265358979 / 2.0) -
                          (3.14159265358979 / 4.0) * (a / max(abs(b), 1e-6));
                }
                ap = r * float2(cos(phi), sin(phi));
            } else {
                // Polygon sampling: pick a triangular wedge of the
                // regular n-gon, sample uniformly inside it. Bokeh
                // gets that classic 5-/6-/8-sided look.
                float n = float(blades);
                float wedge = 6.28318530717958 / n;
                int   i = int(floor(u1 * n));
                float r1 = frac(u1 * n);
                float r2 = u2;
                if (r1 + r2 > 1.0) { r1 = 1.0 - r1; r2 = 1.0 - r2; }
                float a0 = float(i) * wedge;
                float a1 = a0 + wedge;
                float2 p0 = float2(cos(a0), sin(a0));
                float2 p1 = float2(cos(a1), sin(a1));
                ap = r1 * p0 + r2 * p1;
            }
            float3 lens_off = right_xyz.xyz * (ap.x * aperture)
                            + up_xyz.xyz    * (ap.y * aperture);
            float3 focus_p  = ro + rd * focal_d;
            ro = ro + lens_off;
            rd = normalize(focus_p - ro);
        }

        float3 throughput = float3(1.0);
        float3 radiance   = float3(0.0);
        // After a Lambert bounce we did NEE for direct env-map lighting,
        // so the next BRDF-bounce ray's miss contribution is already
        // counted -- skip on miss to avoid double counting. With MIS
        // (mis_enabled != 0 && env_map present) we instead KEEP the
        // sky-miss contribution and weight both NEE and BRDF samples
        // by the balance heuristic; tracked by `prev_was_lambert_nee`
        // + `prev_lambert_brdf_pdf` below.
        bool   skip_sky_on_miss   = false;
        bool   prev_was_lambert_nee = false;   // previous bounce was Lambert with env-map NEE
        float  prev_lambert_brdf_pdf = 0.0;    // cosine-hemisphere PDF of the bounce dir sampled there

        // Save primary-ray params for the volumetric march (single
        // scatter along the camera ray, evaluated after the bounce
        // loop). primary_t is the depth at which the camera ray hits
        // something; sky-miss defaults to 1000m so we still get
        // atmospheric haze along the full visible distance.
        float3 primary_ro = ro;
        float3 primary_rd = rd;
        // Sky-pixel default extends to ~500km so cirrus near the
        // horizon (8000-9500m altitude, needs ~500km horizontal travel
        // for 1 deg elevation rays) is reachable. The cloud march
        // separately caps its span at 30km, so this just unlocks the
        // ray-slab intersection to find the layer in the first place;
        // it doesn't multiply the actual march cost.
        float  primary_t  = 500000.0;
        bool   primary_t_recorded = false;
        bool   primary_hit = false;

        for (uint b = 0u; b < max_bounces; ++b) {
        HitInfo h = traceScene(ro, rd);
        if (b == 0u) {
            // Sky pixels (no hit) keep the 50km default so cloud
            // marches reach the layer; object hits override.
            primary_t = h.hit ? h.t : primary_t;
            primary_hit = h.hit;
            primary_t_recorded = true;
        }
        if (!h.hit) {
            if (mis_enabled != 0u && prev_was_lambert_nee) {
                // MIS path: previous bounce was a Lambert hit that did
                // env-map NEE. The NEE side already contributed its
                // weighted estimator; here we add the BRDF-sampled
                // miss contribution, weighted by the balance heuristic.
                // pdf_other = pdf the env-map sampler would have picked
                // this same direction; pdf_self = cosine-hemisphere PDF
                // recorded at the Lambert bounce site.
                //
                // Cluster directions: env-map CDF masks them to 0 AND
                // extracted-lights NEE samples them every frame on its
                // own sampler. Adding a BRDF-side contribution would
                // double-count, so skip. The check happens once here;
                // pdfEnvMapDir does NOT re-check internally (caller
                // contract documented above its definition). The
                // proper long-term fix is a 3-way MIS across
                // env-CDF / extracted-lights / BRDF.
                if (!isExtractedClusterDir(rd)) {
                    // transmittance() matches the NEE-side estimator's
                    // visibility term (cloud Beer's-law + dielectric
                    // tint along the bounce direction). Without it the
                    // BRDF strategy and the NEE strategy estimate
                    // different integrands, biasing cloudy bounce
                    // directions bright. ro is already the post-Lambert
                    // origin (offset by nf*1e-3 from the surface).
                    float pdf_env = pdfEnvMapDir(rd);
                    float w_brdf  = misBalance(prev_lambert_brdf_pdf, pdf_env);
                    float3 trans  = transmittance(ro, rd, 1e30);
                    radiance += clampIndirect(throughput * skyColor(rd) * trans * w_brdf);
                }
            } else if (!skip_sky_on_miss) {
                // b == 0 is the camera ray hitting the sky directly --
                // unbounded so the HDRI sun renders at full intensity.
                // b > 0 is a bounce ray escaping; clamp those to kill
                // fireflies from BSDF-sampled bounces hitting the sun.
                if (b == 0u) {
                    // Star-split (issue #46): when the denoiser is on
                    // and the user opted in (r_star_split 1, default
                    // on), peel sub-pixel star energy out of the
                    // primary-miss sky term so the SVGF a-trous kernel
                    // doesn't smudge it. The peeled value lands in
                    // frame_stars; the post-denoise finalize stage
                    // adds it back pre-tonemap. Procedural / BSC star
                    // paths only -- HDRI mode is a no-op (stars are
                    // baked into env_map and starsAtPrimaryMiss
                    // returns 0). Bounce rays (b > 0) keep their
                    // existing path: clampIndirect already kills
                    // firefly contributions from sub-pixel stars on
                    // secondary bounces, and the indirect star
                    // contribution is small enough that denoiser
                    // smudging is invisible there.
                    //
                    // Gate the starsAtPrimaryMiss() call on
                    // star_split_active so the legacy branch (cvar
                    // off, denoiser off, etc.) doesn't pay for the
                    // starmap sample + per-cell hash work whose
                    // result it would just discard. The split branch
                    // computes stars_only once and subtracts it from
                    // sky_full BEFORE the throughput multiply, so the
                    // active and legacy paths differ by exactly
                    // `throughput * stars_only` instead of relying on
                    // the algebraic identity `T*(A+S) - T*S == T*A`
                    // (which is not bit-identical in fp16 storage
                    // paths). Also keeps a single source of truth for
                    // the star sampling logic on the active path -- a
                    // future change inside procSky's star contribution
                    // would propagate through both sides via
                    // starsAtPrimaryMiss without manual re-syncing.
                    float3 sky_full = skyColor(rd);
                    if (star_split_active) {
                        float3 stars_only = starsAtPrimaryMiss(rd);
                        frame_stars += throughput * stars_only;
                        radiance    += throughput * (sky_full - stars_only);
                    } else {
                        radiance    += throughput * sky_full;
                    }
                } else {
                    radiance += clampIndirect(throughput * skyColor(rd));
                }
            }
            break;
        }

        float3 hit_pt = ro + rd * h.t;
        float  ndi = dot(h.normal, rd);
        float3 nf  = (ndi < 0.0) ? h.normal : -h.normal;

        if (h.mat == MAT_LAMBERT) {
            if (env_map_present != 0u) {
                // NEE: importance-sampled env-map direct light. With
                // MIS, this contribution is weighted by the balance
                // heuristic vs. the cosine-hemisphere (BRDF) sampler,
                // and the BRDF-sampled bounce's sky-miss contribution
                // is kept (weighted) instead of being skipped. The
                // two strategies together give a lower-variance
                // estimate than NEE alone on Lambert surfaces under
                // HDRIs with concentrated bright regions.
                float2 xi = float2(randf(seed), randf(seed));
                EnvSample env = sampleEnvMap(xi);
                float n_dot_l = dot(nf, env.dir);
                if (env.pdf > 0.0 && n_dot_l > 0.0) {
                    // RGB transmittance through any dielectrics
                    // between this hit and the light -- 1.0 free path,
                    // 0.0 fully blocked, in-between for refraction
                    // through glass / diamond meshes (caustics).
                    float3 trans = transmittance(hit_pt + nf * 1e-3, env.dir, 1e30);
                    if (any(trans > float3(0.0))) {
                        float3 brdf = h.albedo / 3.14159265358979;
                        float pdf_brdf_at_env = n_dot_l * (1.0 / 3.14159265358979);
                        // Only apply the MIS down-weight when the BRDF
                        // complement will actually be evaluated (next
                        // bounce in the loop). On the last allowed
                        // bounce (b + 1 >= max_bounces; includes the
                        // r_max_bounces=1 edge case) the BRDF strategy
                        // never fires, so the unbiased estimator is
                        // full-weight NEE alone.
                        bool brdf_complement_will_fire =
                            (mis_enabled != 0u) && ((b + 1u) < max_bounces);
                        float w_light = brdf_complement_will_fire
                            ? misBalance(env.pdf, pdf_brdf_at_env)
                            : 1.0;
                        radiance += clampIndirect(throughput * env.color * trans * brdf * n_dot_l * w_light / env.pdf);
                    }
                }
                // HDRI extracted-lights NEE: env-map NEE samples the
                // CDF that has all extracted-cluster pixels zeroed
                // (so those clusters are missing from that estimator).
                // Add them back via stochastic single-pick over the
                // extracted lights, weighted by cluster luminance.
                // One shadow ray per sample regardless of how many
                // lights were extracted -- sun-only HDRIs converge
                // identically to the prior single-light path; multi-
                // lamp interiors and night HDRIs (moon as a cluster)
                // light the scene without any time-of-day input.
                if (hdri_lights_count > 0u) {
                    HdriLightSample hl = sampleHdriLight(seed);
                    float n_dot_l = dot(nf, hl.dir);
                    if (n_dot_l > 0.0 && hl.pdf > 0.0) {
                        float3 t_s = transmittance(hit_pt + nf * 1e-3, hl.dir, 1e30);
                        if (any(t_s > float3(0.0))) {
                            float3 brdf = h.albedo / 3.14159265358979;
                            radiance += clampIndirect(
                                throughput * hl.color * t_s * brdf * n_dot_l / hl.pdf);
                        }
                    }
                }
                // No procedural moon NEE in HDRI mode: the HDRI is the
                // canonical sky source, and adding moon_dir_phase-driven
                // light here would re-introduce time-of-day dependence
                // that contradicts the HDRI's own moonlight (or lack of
                // it). If the chosen HDRI doesn't include a moon at
                // night, the scene reads dim -- that's the HDRI being
                // incomplete, not a missing engine feature. Pick an
                // HDRI with the moon baked in or switch r_sky_mode to
                // procedural.
                // Diffuse skylight ambient via cosine-weighted hemisphere
                // sample. Previous version sampled skyColor() at the
                // surface normal exactly -- a single-direction shortcut
                // that causes severe per-pixel fireflies on HDRI scenes:
                // every pixel whose surface normal happens to align with
                // the HDRI sun direction (a 1e6-nit pixel) gets a huge
                // single-sample contribution, producing the classic
                // sphere-speckle pattern visible on Lambert surfaces
                // under sunset/noon HDRIs. Cosine-weighted hemisphere
                // sampling integrates properly: the per-frame value is
                // a random hemisphere tap, and spp + temporal averaging
                // converges to the true diffuse irradiance. Bypasses
                // cloud occlusion intentionally (skyColor reads env_map
                // directly), so cloud-shadowed surfaces still receive
                // ambient light -- the same approximation the original
                // hack made, just without the firefly artifact.
                {
                    float xi_a = randf(seed), xi_b = randf(seed);
                    float r    = sqrt(xi_a);
                    float phi  = 6.28318530717958 * xi_b;
                    float3 t1  = abs(nf.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
                    float3 tx  = normalize(cross(t1, nf));
                    float3 ty  = cross(nf, tx);
                    float3 amb_dir = tx * (r * cos(phi))
                                   + ty * (r * sin(phi))
                                   + nf * sqrt(max(0.0, 1.0 - xi_a));
                    radiance += clampIndirect(throughput * h.albedo * skyColor(amb_dir));
                }
                // With MIS we KEEP the bounce-ray sky-miss contribution
                // and balance-heuristic-weight both sides. Without MIS
                // we suppress the miss to avoid double-counting NEE.
                skip_sky_on_miss = (mis_enabled == 0u);
            } else {
                // Procedural sun NEE + skylight ambient. Drives the
                // procedural / gradient sky modes -- the scene's
                // directional light follows the actual sun position
                // (so it goes dark at night) and a small diffuse term
                // emulates skylight so unlit faces aren't pure black.
                float3 sun_dir  = sun_and_mode.xyz;
                float  sun_elev = sun_dir.y;            // sin(elevation)

                if (sun_elev > 0.0) {
                    float n_dot_l = max(0.0, dot(nf, sun_dir));
                    float3 sun_jit = sampleSunDisc(sun_dir, seed);
                    float3 trans  = (n_dot_l > 0.0)
                        ? transmittance(hit_pt + nf * 1e-3, sun_jit, 1e30)
                        : float3(0.0);
                    if (n_dot_l > 0.0 && any(trans > float3(0.0))) {
                        // Atmospheric attenuation: optical depth times
                        // air mass. Air mass = 1/sin(elev) (thin-slab
                        // approximation, accurate down to ~5 deg). The
                        // exp(-tau*am) curve matches real sunset
                        // behaviour: brightness drops off a cliff once
                        // elev < 5 deg, but stays bright through midday.
                        float am  = 1.0 / max(sun_elev, 0.04);
                        float tau = 0.30;             // tunable haze
                        float trans_atm = exp(-tau * am);
                        // Wavelength-dependent attenuation: blue
                        // scatters more, so colour reddens with air
                        // mass. Wavelength weights mimic Rayleigh
                        // (~1/lambda^4) without doing a full spectral
                        // walk -- enough to read as "warm at sunset".
                        float warmth = saturate((am - 1.0) * 0.18);
                        float3 sun_col = lerp(float3(1.00, 0.97, 0.92),
                                              float3(1.55, 0.55, 0.20),
                                              warmth);
                        // I0 in tonemap-relative units. 80 is tuned so
                        // a noon sun on a Lambert white wall reads
                        // ~1.0 after exposure*ACES at typical exposure.
                        float strength = 80.0 * trans_atm;
                        float3 sun_rad = sun_col * strength;
                        float3 brdf    = h.albedo / 3.14159265358979;
                        // Multiply by per-channel transmittance through
                        // any dielectrics on the shadow ray, so glass
                        // tint / diamond focal lensing modulate the
                        // direct light correctly.
                        radiance += throughput * sun_rad * trans * brdf * n_dot_l;
                    }
                }
                // Moon NEE for the procedural-sky path (see env_map
                // branch above for the rationale).
                if (moon_dir_phase.y > 0.0) {
                    float phase_b = 0.5 * (1.0 - cos(moon_dir_phase.w));
                    if (phase_b > 0.01) {
                        float3 moon_jit = sampleSunDisc(moon_dir_phase.xyz, seed);
                        float ndl = max(0.0, dot(nf, moon_jit));
                        if (ndl > 0.0) {
                            float3 t_m = transmittance(hit_pt + nf * 1e-3, moon_jit, 1e30);
                            if (any(t_m > float3(0.0))) {
                                // Atmospheric attenuation + warmth at low
                                // moon altitude, mirroring sun radiance.
                                float m_elev = max(moon_dir_phase.y, 0.04);
                                float m_am = 1.0 / m_elev;
                                float m_atm = exp(-0.30 * m_am);
                                float m_warmth = saturate((m_am - 1.0) * 0.18);
                                float3 m_col = lerp(float3(0.92, 0.94, 1.00),
                                                    float3(1.20, 0.65, 0.40),
                                                    m_warmth);
                                // 0.3x of sun (was 0.8) -- night feels
                                // genuinely dim. Auto-exposure still
                                // amplifies for visibility, but the
                                // ground / clouds stay readably night
                                // (~1/3 of daylight luminance) instead
                                // of looking like a slightly cool day.
                                float3 moon_rad = m_col * (0.3 * phase_b * m_atm);
                                float3 brdf = h.albedo / 3.14159265358979;
                                radiance += throughput * moon_rad * t_m * brdf * ndl;
                            }
                        }
                    }
                }
                // Calculated diffuse skylight ambient (matches the
                // env_map branch above). Sample the procedural sky
                // model at the surface normal -- single-direction
                // Lambert ambient approximation, real and derived
                // from the sky model rather than hardcoded RGB.
                radiance += throughput * h.albedo * skyColor(nf);
                skip_sky_on_miss = true;
            }
            throughput *= h.albedo;
            ro = hit_pt + nf * 1e-3;
            rd = sampleCosineHemisphere(nf, seed);
            // For MIS: record the cosine-hemisphere PDF of the bounce
            // direction we just sampled, so the next iteration's
            // sky-miss branch can balance-heuristic-weight it against
            // the env-map PDF that would have produced the same dir.
            // Only meaningful when the env map was the NEE source.
            if (mis_enabled != 0u && env_map_present != 0u) {
                prev_lambert_brdf_pdf = max(0.0, dot(nf, rd)) * (1.0 / 3.14159265358979);
                prev_was_lambert_nee  = true;
            } else {
                prev_was_lambert_nee  = false;
            }
        }
        else if (h.mat == MAT_METAL) {
            float3 r = reflect(rd, nf);
            r = normalize(r + h.roughness * (sampleCosineHemisphere(nf, seed) - nf));
            throughput *= h.albedo;
            ro = hit_pt + nf * 1e-3;
            rd = r;
            skip_sky_on_miss = false;   // metal didn't NEE; sky on miss is real
            prev_was_lambert_nee = false;   // specular bounce: MIS doesn't apply here
        }
        else {  // dielectric
            float ior_a = (ndi < 0.0) ? 1.0   : h.ior;
            float ior_b = (ndi < 0.0) ? h.ior : 1.0;
            float eta = ior_a / ior_b;
            float cos_i = abs(dot(nf, -rd));
            float r0 = (ior_a - ior_b) / (ior_a + ior_b);
            r0 = r0 * r0;
            float fr = fresnelSchlick(cos_i, r0);
            float3 reflected = reflect(rd, nf);
            float3 refracted = refract(rd, nf, eta);
            bool tir = (length(refracted) < 1e-3);
            if (tir || randf(seed) < fr) {
                rd = reflected;
                ro = hit_pt + nf * 1e-3;
            } else {
                rd = normalize(refracted);
                ro = hit_pt - nf * 1e-3;
            }
            skip_sky_on_miss = false;   // dielectric didn't NEE either
            prev_was_lambert_nee = false;   // specular/refraction: MIS doesn't apply
        }

            if (b >= 2u) {
                float p = max(throughput.r, max(throughput.g, throughput.b));
                p = clamp(p, 0.05, 1.0);
                if (randf(seed) > p) break;
                throughput /= p;
            }
        }
        // Surface aerial perspective. All radiance accumulated above
        // (direct surface NEE + multi-bounce indirect + diffuse sky
        // ambient at the surface point) reaches the camera through the
        // air mass between camera and primary hit, so attenuate by the
        // analytical Mie + Rayleigh transmittance over [0, primary_t].
        // Gated on primary_hit: sky-on-miss radiance is the sky's
        // already-baked appearance at infinity (HDRI / procSky already
        // include atmospheric absorption); attenuating again would
        // double-count and dim the horizon.
        //
        // Order matters: this multiplication runs BEFORE the atmospheric
        // in-scatter march and cloud march below. Those marches add
        // their own pre-attenuated contributions (each in-scatter event
        // weighted by trans_eye-up-to-that-depth, each cloud sample
        // multiplied by atmosphericTransmittance(0, sample_t) inside
        // the loop) so they compose correctly with the surface's
        // aerial-perspective term.
        if (primary_hit) {
            float3 trans_atm_surf = atmosphericTransmittance(
                primary_ro, primary_rd, primary_t);
            radiance *= trans_atm_surf;
            // HDRI-sourced haze additive. The atmospheric in-scatter
            // march below is gated off in HDRI mode to avoid double-
            // counting the HDRI's already-baked Rayleigh tint, but
            // that means surfaces get attenuated WITHOUT the additive
            // "absorbed light replaced by sky color" term that gives
            // distant features their hazy lift toward the horizon.
            // Without this, the ground plane near the horizon reads
            // as a dark band where it should be fading into the sky.
            //
            // Approximation: treat (1 - trans_atm_surf) as the
            // fraction of the path's radiance that the air absorbed,
            // and replace it with the sky color in the camera's view
            // direction. skyColor(primary_rd) samples env_map for
            // HDRI mode (the actual sunset / clear-sky / etc. tint
            // visible behind the surface). Clamp-wrapped to bound
            // the magnitude when the view ray points toward a bright
            // HDRI sun-cluster pixel.
            //
            // Procedural / gradient modes get this additive from the
            // explicit Mie + Rayleigh in-scatter march below, which
            // is more physically correct (uses sun_rad and the HG
            // phase function), so we skip the simpler haze additive
            // there to avoid double-counting.
            if (env_map_present != 0u) {
                radiance += clampIndirect(
                    skyColor(primary_rd) * (float3(1.0) - trans_atm_surf));
            }
        }
        // Volumetric single-scatter along the primary ray. Two
        // independent marches because the natural sampling rates
        // differ:
        //   - Haze: dense near the camera, attenuates exp(-density*t)
        //     so beyond ~5km it contributes nothing. Cap the march
        //     range at 5km to keep dt small.
        //   - Clouds: live in a thin altitude band (e.g. 200-500m).
        //     Clip the march to the actual ray-slab intersection so
        //     all N samples land inside the cloud, not in empty air.
        // The two contributions add into `radiance`. Both share the
        // sun radiance and HG phase setup below.
        if ((vol_params.x > 0.0 || clouds_p1.w > 0.0 || clouds_p3.w > 0.0)
            && primary_t_recorded) {
            float  g         = vol_params.y;
            float  vol_int   = vol_params.z;
            int    N         = max(1, int(vol_params.w));
            float3 sun_dir   = sun_and_mode.xyz;
            float  sun_elev  = sun_dir.y;
            // Sun radiance at this elevation (matches Lambert sun
            // colour so beams + lit surfaces agree on tint). Warmth
            // saturates more aggressively (0.30 instead of 0.18) and
            // the warm endpoint is a deeper red, so sunset clouds
            // actually read as "blood orange" rather than "tinted
            // beige". am stored separately so the cloud march can
            // re-scale per-sample altitude (high clouds catch less-
            // attenuated, brighter / less-red sunlight than ground).
            float3 sun_rad = float3(0.0);
            float  sun_am  = 0.0;
            if (sun_elev > 0.0) {
                sun_am = 1.0 / max(sun_elev, 0.04);
                float tau = 0.30;
                float t_atm = exp(-tau * sun_am);
                float warmth = saturate((sun_am - 1.0) * 0.30);
                sun_rad = lerp(float3(1.00, 0.97, 0.92),
                               float3(1.55, 0.40, 0.12), warmth)
                        * (80.0 * t_atm);
            }
            float cos_theta = dot(primary_rd, -sun_dir);
            float g2 = g * g;
            float phase = (1.0 - g2)
                        / (4.0 * 3.14159265358979
                                * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));

            // --- Atmospheric (Mie + Rayleigh) march -------------------------
            // Real atmosphere has two scattering populations:
            //   Mie (haze, dust, water droplets): grey, forward-peaked HG
            //     phase, scale height ~1.5km, sigma controlled by
            //     vol_params.x. Gives god rays and sunset warmth.
            //   Rayleigh (gas molecules): wavelength-dependent, blue is
            //     scattered ~6x more than red. Phase is 3/(16π)*(1+cos²θ).
            //     Scale height ~8km. Real per-channel sea-level sigma:
            //       R 5.8e-6  G 13.5e-6  B 33.1e-6  m^-1.
            //     Rayleigh is what makes the daytime sky blue and the
            //     sunset red (long path attenuates blue out, leaving
            //     red); without it, the sky bleaches white at noon.
            //
            // Per channel transmittance through the column is the product
            // of Mie and Rayleigh exp(-sigma*dt), so trans_eye is a
            // float3 (Mie alone is grey; Rayleigh tints the column).
            //
            // Bounded at 60km because Rayleigh's 8km scale height means
            // most of its contribution lives in the bottom ~25km, and
            // 60km is plenty of headroom while keeping dt manageable.
            const float3 kSigmaR = float3(5.8e-6, 13.5e-6, 33.1e-6);
            const float kHmie = 1500.0;
            const float kHrayleigh = 8000.0;
            const float kPi = 3.14159265358979;
            float rayleigh_int = clouds_p3.w;   // 0 disables, 1 = real
            // Atmospheric in-scatter march (Mie + Rayleigh). Runs in
            // procedural and gradient sky modes -- the engine sun is
            // the in-scatter source, and the cloud march below also
            // uses analytical atmospheric transmittance to attenuate
            // its own samples regardless of sky mode (aerial
            // perspective). HDRI mode skips this in-scatter step
            // because the HDRI has the atmospheric scattering baked
            // into its sky pixels; adding procedural in-scatter on
            // top double-counts and would follow sun_and_mode.xyz
            // (engine time-of-day) instead of the HDRI's actual sun.
            // The cloud march still gets aerial-perspective fade in
            // HDRI mode via the analytical transmittance helper.
            bool march_mie = (vol_params.x > 0.0) && (env_map_present == 0u);
            bool march_ray = (rayleigh_int > 0.0) && (env_map_present == 0u);
            if (march_mie || march_ray) {
                float  sigma_0 = vol_params.x;
                float  t_far   = min(primary_t, 60000.0);
                float  dt      = t_far / float(N);
                float  jitter  = randf(seed) * dt;
                float3 trans_eye = float3(1.0);
                float3 v_color = float3(0.0);
                float phase_mie_v = phase;   // HG, computed earlier
                float phase_ray_v = (3.0 / (16.0 * kPi)) * (1.0 + cos_theta * cos_theta);
                for (int vi = 0; vi < N; ++vi) {
                    float t = (float(vi) + 0.5) * dt + jitter;
                    if (t >= t_far) break;
                    float3 sp = primary_ro + primary_rd * t;
                    float h = max(sp.y, 0.0);
                    float  sigma_mie = march_mie ? sigma_0 * exp(-h / kHmie) : 0.0;
                    float3 sigma_ray = march_ray
                        ? kSigmaR * rayleigh_int * exp(-h / kHrayleigh)
                        : float3(0.0);
                    float3 sigma_total = sigma_ray + float3(sigma_mie);
                    if (sun_elev > 0.0) {
                        float3 sun_jit = sampleSunDisc(sun_dir, seed);
                        float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                        if (any(trans_sun > float3(0.0))) {
                            float3 in_scatter =
                                  float3(sigma_mie) * phase_mie_v
                                + sigma_ray         * phase_ray_v;
                            v_color += sun_rad * trans_sun * in_scatter
                                     * dt * trans_eye;
                        }
                    }
                    trans_eye *= exp(-sigma_total * dt);
                }
                radiance += v_color * vol_int;
            }

            // --- Cloud march ------------------------------------------------
            // Layer-clip [t_in, t_out] to the ray-slab intersection of the
            // cloud altitude band. For object-hit primary rays we also
            // clip at primary_t so clouds don't shine through walls. For
            // sky pixels primary_t = 50km, so the layer is fully visible.
            if (clouds_p1.w > 0.0) {
                float base_y = clouds_p1.x;
                float top_y  = clouds_p1.y;
                float ry = primary_rd.y;
                float t_in, t_out;
                if (abs(ry) > 1e-6) {
                    float t_b = (base_y - primary_ro.y) / ry;
                    float t_t = (top_y  - primary_ro.y) / ry;
                    t_in  = max(0.0, min(t_b, t_t));
                    t_out = max(t_b, t_t);
                } else if (primary_ro.y >= base_y && primary_ro.y <= top_y) {
                    t_in = 0.0; t_out = primary_t;
                } else {
                    t_in = 1.0; t_out = 0.0;   // miss
                }
                t_out = min(t_out, primary_t);
                // Cap the cloud march span at 30km. For low-elevation
                // rays the layer can span hundreds of km; uniform
                // sampling that range produces samples spaced wider
                // than cloud features and looks worse than just
                // marching the closer 30km. Beyond 30km the sky
                // gradient takes over visually.
                if (t_out - t_in > 30000.0) t_in = t_out - 30000.0;
                if (t_out > t_in) {
                    int   Nc = max(N, 12);    // user slider can ask for more
                    float dtc = (t_out - t_in) / float(Nc);
                    float jitc = randf(seed) * dtc;
                    float trans_eye_c = 1.0;
                    float3 v_color_c  = float3(0.0);
                    // In HDRI mode the env map is the canonical sky and the
                    // engine's procedural sun + moon are NOT contributors --
                    // otherwise clouds get double-lit by both the HDRI's
                    // baked sun and the engine sun, which is wrong. The
                    // HDRI path uses the extracted-sun direction for crisp
                    // shadows + env-map MIS NEE for diffuse sky.
                    bool hdri_lit = (env_map_present != 0u);
                    // Closed-form skylight ambient (procedural sky only).
                    // Integrating the sky hemisphere around a cloud sample
                    // gives ~15% of sun irradiance at the surface (real-
                    // Earth ratio of skylight/sun irradiance is ~0.15-0.20).
                    // The sky is also slightly blue from Rayleigh, so the
                    // ambient tint is a touch bluer than sun_rad. Without
                    // this term cloud cores go pitch-black under sun self-
                    // shadow -- correct single-scatter direct-sun-only
                    // physics, but real clouds get filled in by the
                    // surrounding sky. 1/π folds the hemisphere integral in
                    // (same factor that turns Lambert irradiance into radiance).
                    float3 sky_ambient = float3(0.0);
                    if (!hdri_lit) {
                        if (sun_elev > 0.0) {
                            sky_ambient = sun_rad * float3(0.7, 0.85, 1.0) * 0.15
                                        * (1.0 / 3.14159265358979);
                        } else if (moon_dir_phase.y > 0.0) {
                            // Moonlight skylight on the cloud at night.
                            // Without this term, clouds at night go pitch
                            // black -- they're correctly receiving zero
                            // sun in shadow but missing the diffuse moon-
                            // hemisphere contribution that real moonlit
                            // clouds actually receive.
                            float phb = 0.5 * (1.0 - cos(moon_dir_phase.w));
                            float ma  = 1.0 / max(moon_dir_phase.y, 0.04);
                            float mat = exp(-0.30 * ma);
                            sky_ambient = float3(0.55, 0.70, 1.00) * (0.04 * phb * mat);
                        }
                    }
                    for (int vi = 0; vi < Nc; ++vi) {
                        float t = t_in + (float(vi) + 0.5) * dtc + jitc;
                        if (t > t_out) break;
                        float3 sp = primary_ro + primary_rd * t;
                        float dens = cloud_density_at(sp, clouds_p1,
                                                      clouds_p2, clouds_p3, 6);
                        if (dens <= 0.0) continue;
                        // Aerial perspective: per-channel atmospheric
                        // transmittance from camera to this sample,
                        // analytical Mie + Rayleigh integral. Each
                        // contribution below gets multiplied by
                        // trans_atm so distant clouds fade toward the
                        // sky color naturally (Rayleigh tints, Mie
                        // dims). Active in all sky modes -- the HDRI
                        // doesn't model in-front-of-cloud air mass so
                        // distant clouds need this regardless. Wrapped
                        // in clampIndirect to bound per-sample fireflies
                        // (HG forward-scatter peak * env-map NEE * long
                        // horizon ray was producing white halos).
                        float3 trans_atm = atmosphericTransmittance(
                            primary_ro, primary_rd, t);
                        float  step_w = dens * dtc * trans_eye_c;
                        if (hdri_lit) {
                            // HDRI extracted-lights NEE: stochastic
                            // single-pick weighted by cluster luminance.
                            if (hdri_lights_count > 0u) {
                                HdriLightSample hl = sampleHdriLight(seed);
                                if (hl.pdf > 0.0) {
                                    float c_t_h = dot(primary_rd, -hl.dir);
                                    float phase_h = (1.0 - g2) / (4.0 * 3.14159265358979
                                        * pow(max(1.0 + g2 - 2.0 * g * c_t_h, 1e-4), 1.5));
                                    float3 t_s = transmittance(sp, hl.dir, 1e30);
                                    if (any(t_s > float3(0.0))) {
                                        v_color_c += clampIndirect(
                                            hl.color * t_s * phase_h * step_w * trans_atm
                                            / hl.pdf);
                                    }
                                }
                            }
                            // HDRI ambient via env-map MIS NEE (CDF
                            // excludes the extracted clusters; this
                            // samples only the diffuse-sky portion).
                            {
                                float2 xi_e = float2(randf(seed), randf(seed));
                                EnvSample env = sampleEnvMap(xi_e);
                                if (env.pdf > 0.0) {
                                    float3 t_e = transmittance(sp, env.dir, 1e30);
                                    if (any(t_e > float3(0.0))) {
                                        float c_t_e = dot(primary_rd, -env.dir);
                                        float phase_e = (1.0 - g2) / (4.0 * 3.14159265358979
                                            * pow(max(1.0 + g2 - 2.0 * g * c_t_e, 1e-4), 1.5));
                                        v_color_c += clampIndirect(
                                            env.color * t_e * phase_e * step_w * trans_atm
                                            / env.pdf);
                                    }
                                }
                            }
                        } else {
                            // Direct sun (single-scatter NEE). Altitude-
                            // boosted sun_rad: high clouds (cirrus at 8km)
                            // sit ABOVE most of the absorbing atmosphere
                            // and catch brighter / less-reddened light
                            // than the ground does.
                            if (sun_elev > 0.0) {
                                float h_alt   = max(sp.y, 0.0);
                                float h_norm  = saturate(h_alt / 8000.0);
                                float boost   = exp(0.30 * sun_am * h_norm);
                                float3 sun_at = sun_rad * boost;
                                float3 sun_jit = sampleSunDisc(sun_dir, seed);
                                float3 trans_sun = transmittance(sp, sun_jit, 1e30);
                                if (any(trans_sun > float3(0.0))) {
                                    v_color_c += clampIndirect(
                                        sun_at * trans_sun * phase * step_w * trans_atm);
                                }
                            }
                            // Direct moon (single-scatter NEE). Moon is the
                            // night sky's key light; without this clouds go
                            // pitch-black after sunset. Phase shading via
                            // 0.5*(1-cos(phase)) so new moon contributes
                            // nothing, full moon is full-strength.
                            if (moon_dir_phase.y > 0.0) {
                                float phase_b = 0.5 * (1.0 - cos(moon_dir_phase.w));
                                if (phase_b > 0.01) {
                                    float3 moon_jit = sampleSunDisc(moon_dir_phase.xyz, seed);
                                    float3 trans_moon = transmittance(sp, moon_jit, 1e30);
                                    if (any(trans_moon > float3(0.0))) {
                                        float c_t_m = dot(primary_rd, -moon_dir_phase.xyz);
                                        float phase_m = (1.0 - g2) / (4.0 * 3.14159265358979
                                            * pow(max(1.0 + g2 - 2.0 * g * c_t_m, 1e-4), 1.5));
                                        // Atmospheric attenuation + warmth at
                                        // low moon altitude, mirroring sun.
                                        float m_elev   = max(moon_dir_phase.y, 0.04);
                                        float m_am     = 1.0 / m_elev;
                                        float m_atm    = exp(-0.30 * m_am);
                                        float m_warmth = saturate((m_am - 1.0) * 0.18);
                                        float3 m_col   = lerp(float3(0.92, 0.94, 1.00),
                                                              float3(1.20, 0.65, 0.40),
                                                              m_warmth);
                                        // 0.3x of sun -- night reads as ~1/3
                                        // daylight luminance after auto-exposure.
                                        float3 moon_rad = m_col * (0.3 * phase_b * m_atm);
                                        v_color_c += clampIndirect(
                                            moon_rad * trans_moon * phase_m * step_w * trans_atm);
                                    }
                                }
                            }
                            // Diffuse skylight (omnidirectional, no phase
                            // function).
                            v_color_c += clampIndirect(sky_ambient * step_w * trans_atm);
                        }
                        trans_eye_c *= exp(-dens * dtc);
                    }
                    radiance += v_color_c * vol_int;
                }
            }
        }

        frame_radiance += radiance;
    }
    frame_radiance /= float(sample_count);
    frame_stars    /= float(sample_count);

    // Sanitize: replace NaN / Inf / negative channels with 0 before the
    // running-mean update below. Without this, a single rare bad sample
    // (e.g. transmittance edge case, env.pdf underflowing to ~0 between
    // the > 0 check and the divide, sqrt of slightly-negative numerical
    // residues) injects NaN into accum_hdr[tid]. The running-mean update
    // is `avg = (prev * n_old + new) / n_new`, so NaN propagates through
    // prev forever -- that pixel becomes a permanent black speck. Over
    // many minutes of static-camera accumulation, the contaminated pixel
    // set grows monotonically. Replacing non-finite samples with 0 lets
    // the running mean simply skip the bad sample and keep accumulating
    // the good ones for that pixel.
    {
        bool3 bad = isnan(frame_radiance) || isinf(frame_radiance);
        frame_radiance = select(bad, float3(0.0), max(frame_radiance, float3(0.0)));
    }
    // Same sanitisation for frame_stars: starsAtPrimaryMiss reads a
    // host-uploaded RGBA16F starmap which we trust to be clean, but a
    // NaN in throughput (from an upstream BRDF edge case) would
    // propagate identically into accum_stars. Snap to 0 so the texture
    // never carries non-finite state.
    {
        bool3 bad = isnan(frame_stars) || isinf(frame_stars);
        frame_stars = select(bad, float3(0.0), max(frame_stars, float3(0.0)));
    }

    // Online running mean. reset_accum=1 starts a fresh average. The
    // `n_new` we record is the count of samples averaged into this
    // pixel -- with spp>1 we could store sample_count there, but the
    // shader only reads it as a weight, so 1-per-frame is fine.
    float4 prev = (reset_accum != 0u) ? float4(0,0,0,0) : accum_hdr[tid];
    // Sanitize prev: if the previous-frame sample was already NaN/Inf
    // (e.g. from a build before sanitization landed, or imported state),
    // restart that pixel rather than re-averaging in the contamination.
    {
        bool3 prev_bad = isnan(prev.rgb) || isinf(prev.rgb);
        if (any(prev_bad) || isnan(prev.a) || isinf(prev.a)) {
            prev = float4(0, 0, 0, 0);
        }
    }
    // accum_params.x = EMA history retention. 0 -> legacy online running
    // mean (unbounded convergence). >0 -> blend at this retention factor
    // so old samples decay; lets the accumulator track wind-drifting
    // clouds / animated lights without the wet-paint smear that pure
    // running mean produces on non-stationary input. Trade-off: the
    // running mean's variance falls as 1/N, EMA's floors at ~variance
    // of the per-frame estimator divided by the effective window
    // 1/(1-alpha). With a denoiser on, this is moot -- the denoiser's
    // own motion-vector temporal reuse is strictly stronger and the
    // engine still writes accum_hdr only as a fallback path for
    // toggling the denoiser off mid-session.
    float ema_alpha = clamp(accum_params.x, 0.0, 0.999);
    float n_new;
    float3 avg;
    if (prev.a < 0.5 || ema_alpha <= 0.0) {
        // Legacy / first-frame-of-epoch path. prev.a==0 case (after
        // reset_accum or a NaN-driven prev sanitization) seeds the
        // mean cleanly; ema_alpha==0 keeps unbounded online averaging.
        n_new = prev.a + 1.0;
        avg   = (prev.rgb * prev.a + frame_radiance) / n_new;
    } else {
        avg   = prev.rgb * ema_alpha + frame_radiance * (1.0 - ema_alpha);
        // Count is informational only (tonemap stats); cap so it
        // doesn't grow without bound and lose float precision over
        // long runs.
        n_new = min(prev.a + 1.0, 1.0 / max(1.0 - ema_alpha, 1e-3));
    }
    accum_hdr[tid] = float4(avg, n_new);

    // Per-frame radiance feeds the denoiser. With the HDR pipeline
    // (w2j_row0.w = 1) we hand MetalFX raw linear HDR and the
    // post-pass `tonemap` kernel applies exposure + ACES at the very
    // end -- this preserves highlight detail through temporal reuse.
    // With the legacy LDR pipeline (= 0) we pre-tonemap here and the
    // post-pass kernel runs as a passthrough copy. The toggle exists
    // so r_hdr_pipeline can be flipped for A/B comparison without a
    // rebuild.
    // Pull the live exposure scalar from the GPU-side state buffer.
    // AutoExposure.slang updates this each frame (auto mode); the
    // engine WriteBuffer's it once when manual mode toggles. Reading
    // here avoids a CPU readback round-trip that on dGPU stalls
    // the queue once per N frames.
    float exposure = exposure_state[0];

    if (write_hdr_aux != 0u) {
        // denoise_color is an RGBA16F linear-HDR intermediate read by
        // the post-pass Tonemap kernel later -- stays linear, no OETF.
        // Gated on write_hdr_aux (not denoiser_enabled) so the engine
        // can request this write for the bloom-without-denoiser path
        // too, without also forcing the depth/motion G-buffer block
        // above (which is denoiser-only).
        //
        // Source-radiance choice depends on whether a denoiser is
        // chaining after us:
        //   - denoiser on  -> frame_radiance (the 1-spp noisy estimate);
        //     the denoiser does its own temporal accumulation, so pre-
        //     accumulating here would double-accumulate / lag history.
        //   - denoiser off -> avg (the accumulated linear HDR `accum`
        //     average); matches the value the inline swapchain write
        //     below uses, so the engine Tonemap chain reads the same
        //     accumulated image and just adds bloom on top.
        //
        // Note (issue #46 star-split): when star_split_active was set
        // above, the primary-miss code already subtracted the star
        // contribution from `radiance`, so frame_radiance carries
        // sky-without-stars. The bypassed stars sit in frame_stars
        // and are written separately to accum_stars below. With
        // star-split off, frame_radiance is identical to the
        // pre-fix value (the subtraction branch never fires) and
        // frame_stars stays at 0.
        float3 hdr_src = (denoiser_enabled != 0u) ? frame_radiance : avg;
        if (w2j_row0.w > 0.5) {
            denoise_color[tid] = float4(hdr_src, 1.0);
        } else {
            float3 mapped = acesTonemap(hdr_src * exposure);
            denoise_color[tid] = float4(mapped, 1.0);
        }
    }

    // Star-split accumulator (issue #46). Same EMA / running-mean
    // update shape as accum_hdr above, so a static-camera night
    // scene converges identically and we don't introduce a fresh
    // noise source on the bypass path. Gate kept tight: only writes
    // when star_split_active was the gating condition for the
    // subtraction above -- a write under a slack condition would
    // leak stars into the legacy / off path's accum_stars (currently
    // zero) and the finalize stage would composite them twice.
    if (star_split_active) {
        float4 prev_s = (reset_accum != 0u) ? float4(0,0,0,0) : accum_stars[tid];
        {
            // Sanitize prev: mirrors accum_hdr's NaN/Inf guard above.
            bool3 prev_bad = isnan(prev_s.rgb) || isinf(prev_s.rgb);
            if (any(prev_bad) || isnan(prev_s.a) || isinf(prev_s.a)) {
                prev_s = float4(0, 0, 0, 0);
            }
        }
        float n_new_s;
        float3 avg_s;
        if (prev_s.a < 0.5 || ema_alpha <= 0.0) {
            n_new_s = prev_s.a + 1.0;
            avg_s   = (prev_s.rgb * prev_s.a + frame_stars) / n_new_s;
        } else {
            avg_s   = prev_s.rgb * ema_alpha + frame_stars * (1.0 - ema_alpha);
            n_new_s = min(prev_s.a + 1.0, 1.0 / max(1.0 - ema_alpha, 1e-3));
        }
        accum_stars[tid] = float4(avg_s, n_new_s);
    }

    // Swapchain write. Apply the OETF on the SPIR-V path so the OS's
    // EOTF-on-display gives back the original linear values. On Metal
    // the swapchain's BGRA8Unorm_sRGB pixel format encodes implicitly
    // on Apple silicon storage writes -- nothing for the shader to do.
    float3 ldr = acesTonemap(avg * exposure);
#if defined(PT_TARGET_SPIRV)
    ldr = srgb_oetf(ldr);
#endif
    output[tid] = float4(ldr, 1.0);
}
